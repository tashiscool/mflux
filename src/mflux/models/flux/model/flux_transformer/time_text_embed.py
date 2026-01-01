import math

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.flux.model.flux_transformer.guidance_embedder import GuidanceEmbedder
from mflux.models.flux.model.flux_transformer.text_embedder import TextEmbedder
from mflux.models.flux.model.flux_transformer.timestep_embedder import TimestepEmbedder


# Compiled time projection for kernel fusion - called every diffusion step
@mx.compile
def _time_proj_compiled(time_steps: mx.array, exponent_base: mx.array) -> mx.array:
    """Compiled time projection with fused sin/cos operations."""
    emb = time_steps[:, None].astype(mx.float32) * exponent_base[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    half_dim = emb.shape[-1] // 2
    emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    return emb


class TimeTextEmbed(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.text_embedder = TextEmbedder()
        self.guidance_embedder = GuidanceEmbedder() if model_config.supports_guidance else None
        self.timestep_embedder = TimestepEmbedder()
        # Pre-compute exponent base for time projection (constant across all calls)
        max_period = 10000
        half_dim = 128
        exponent = -math.log(max_period) * mx.arange(start=0, stop=half_dim, step=None, dtype=mx.float32)
        self._exponent_base = mx.exp(exponent / half_dim)

    def __call__(
        self,
        time_step: mx.array,
        pooled_projection: mx.array,
        guidance: mx.array,
    ) -> mx.array:
        time_steps_proj = _time_proj_compiled(time_step, self._exponent_base)
        time_steps_emb = self.timestep_embedder(time_steps_proj)
        if self.guidance_embedder is not None:
            time_steps_emb += self.guidance_embedder(_time_proj_compiled(guidance, self._exponent_base))
        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = time_steps_emb + pooled_projections
        return conditioning.astype(ModelConfig.precision)
