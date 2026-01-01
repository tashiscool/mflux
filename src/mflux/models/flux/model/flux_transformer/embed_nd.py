import mlx.core as mx
from mlx import nn


# Compiled RoPE core computation for kernel fusion
@mx.compile
def _rope_core(pos_expanded: mx.array, omega_expanded: mx.array, batch_size: int, seq_length: int, half_dim: int) -> mx.array:
    """Compiled RoPE with fused sin/cos operations."""
    out = pos_expanded * omega_expanded
    cos_out = mx.cos(out)
    sin_out = mx.sin(out)
    stacked_out = mx.stack([cos_out, -sin_out, sin_out, cos_out], axis=-1)
    return mx.reshape(stacked_out, (batch_size, seq_length, half_dim, 2, 2))


class EmbedND(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 3072
        self.theta = 10000
        self.axes_dim = [16, 56, 56]
        # Pre-compute omega values for each axis dimension
        self._omegas = [
            1.0 / (self.theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
            for dim in self.axes_dim
        ]

    def __call__(self, ids: mx.array) -> mx.array:
        emb = mx.concatenate(
            [self._rope_with_precomputed_omega(ids[..., i], i) for i in range(3)],
            axis=-3,
        )
        return mx.expand_dims(emb, axis=1)

    def _rope_with_precomputed_omega(self, pos: mx.array, axis_idx: int) -> mx.array:
        """RoPE using pre-computed omega and compiled core."""
        batch_size, seq_length = pos.shape
        pos_expanded = mx.expand_dims(pos, axis=-1)
        omega_expanded = mx.expand_dims(self._omegas[axis_idx], axis=0)
        half_dim = self.axes_dim[axis_idx] // 2
        return _rope_core(pos_expanded, omega_expanded, batch_size, seq_length, half_dim)
