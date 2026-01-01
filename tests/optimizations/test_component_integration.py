"""Integration tests for optimized MLX components.

These tests instantiate the actual model components and verify they produce
correct outputs with the expected shapes and value ranges.
"""
import mlx.core as mx
import pytest
from mlx import nn


class TestTimeTextEmbedIntegration:
    """Integration tests for optimized TimeTextEmbed."""

    def test_time_text_embed_instantiation(self):
        """Verify TimeTextEmbed can be instantiated with model config."""
        from mflux.models.flux.model.flux_transformer.time_text_embed import TimeTextEmbed
        from mflux.models.common.config.model_config import ModelConfig

        # Test with dev config (supports guidance)
        config = ModelConfig.dev()
        embed = TimeTextEmbed(model_config=config)
        assert embed is not None
        assert embed.guidance_embedder is not None
        assert hasattr(embed, '_exponent_base')

    def test_time_text_embed_schnell_instantiation(self):
        """Verify TimeTextEmbed works with schnell config (no guidance)."""
        from mflux.models.flux.model.flux_transformer.time_text_embed import TimeTextEmbed
        from mflux.models.common.config.model_config import ModelConfig

        config = ModelConfig.schnell()
        embed = TimeTextEmbed(model_config=config)
        assert embed is not None
        assert embed.guidance_embedder is None

    def test_time_text_embed_forward_pass(self):
        """Verify TimeTextEmbed forward pass produces correct shapes."""
        from mflux.models.flux.model.flux_transformer.time_text_embed import TimeTextEmbed
        from mflux.models.common.config.model_config import ModelConfig

        config = ModelConfig.dev()
        embed = TimeTextEmbed(model_config=config)

        # Create test inputs matching expected shapes
        time_step = mx.array([500.0])
        pooled_projection = mx.random.normal((1, 768))  # CLIP pooled output
        guidance = mx.array([3.5])

        # Forward pass
        output = embed(time_step, pooled_projection, guidance)
        mx.eval(output)

        # Verify output shape (should be conditioning vector)
        assert output.shape[0] == 1
        assert output.shape[1] == 3072  # Inner dim for flux

    def test_time_text_embed_different_timesteps(self):
        """Verify different timesteps produce different outputs."""
        from mflux.models.flux.model.flux_transformer.time_text_embed import TimeTextEmbed
        from mflux.models.common.config.model_config import ModelConfig

        config = ModelConfig.dev()
        embed = TimeTextEmbed(model_config=config)

        pooled_projection = mx.random.normal((1, 768))
        guidance = mx.array([3.5])

        # Different timesteps should produce different embeddings
        output1 = embed(mx.array([0.0]), pooled_projection, guidance)
        output2 = embed(mx.array([500.0]), pooled_projection, guidance)
        output3 = embed(mx.array([999.0]), pooled_projection, guidance)

        mx.eval(output1, output2, output3)

        # They should all have the same shape
        assert output1.shape == output2.shape == output3.shape

        # But different values
        diff12 = mx.abs(output1 - output2).max().item()
        diff23 = mx.abs(output2 - output3).max().item()
        assert diff12 > 0.01, "Different timesteps should produce different embeddings"
        assert diff23 > 0.01, "Different timesteps should produce different embeddings"


class TestEmbedNDIntegration:
    """Integration tests for optimized EmbedND (RoPE)."""

    def test_embed_nd_instantiation(self):
        """Verify EmbedND can be instantiated."""
        from mflux.models.flux.model.flux_transformer.embed_nd import EmbedND

        embed = EmbedND()
        assert embed is not None
        assert embed.dim == 3072
        assert embed.theta == 10000
        assert embed.axes_dim == [16, 56, 56]
        assert hasattr(embed, '_omegas')
        assert len(embed._omegas) == 3

    def test_embed_nd_forward_pass(self):
        """Verify EmbedND forward pass produces correct shapes."""
        from mflux.models.flux.model.flux_transformer.embed_nd import EmbedND

        embed = EmbedND()

        # Create test IDs (batch=1, seq_len, 3 axes)
        # This simulates typical latent image IDs
        seq_len = 256
        ids = mx.zeros((1, seq_len, 3))

        # Forward pass
        output = embed(ids)
        mx.eval(output)

        # Expected output shape: (batch, 1, ...)
        assert output.shape[0] == 1
        assert output.shape[1] == 1

    def test_embed_nd_different_positions(self):
        """Verify different positions produce different embeddings."""
        from mflux.models.flux.model.flux_transformer.embed_nd import EmbedND

        embed = EmbedND()

        # Two different position sets
        ids1 = mx.zeros((1, 64, 3))
        ids2 = mx.ones((1, 64, 3)) * 10  # Different positions

        output1 = embed(ids1)
        output2 = embed(ids2)
        mx.eval(output1, output2)

        # Should have same shape but different values
        assert output1.shape == output2.shape
        diff = mx.abs(output1 - output2).max().item()
        assert diff > 0.01, "Different positions should produce different embeddings"


class TestVisionAttentionIntegration:
    """Integration tests for optimized VisionAttention."""

    def test_vision_attention_instantiation(self):
        """Verify VisionAttention can be instantiated."""
        from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_attention import VisionAttention

        attn = VisionAttention(embed_dim=1280, num_heads=16)
        assert attn is not None
        assert attn.embed_dim == 1280
        assert attn.num_heads == 16
        assert attn.head_dim == 80
        assert hasattr(attn, 'scale')

    def test_vision_attention_forward_pass(self):
        """Verify VisionAttention forward pass produces correct shapes."""
        from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_attention import VisionAttention

        attn = VisionAttention(embed_dim=1280, num_heads=16)

        # Input: (seq_len, embed_dim) - no batch dimension
        seq_len = 64
        x = mx.random.normal((seq_len, 1280))

        output = attn(x)
        mx.eval(output)

        # Output should have same shape as input
        assert output.shape == x.shape

    def test_vision_attention_with_position_embeddings(self):
        """Verify VisionAttention works with position embeddings."""
        from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_attention import VisionAttention

        attn = VisionAttention(embed_dim=1280, num_heads=16)

        seq_len = 64
        x = mx.random.normal((seq_len, 1280))

        # Create position embeddings (cos, sin)
        cos_emb = mx.random.normal((seq_len, attn.head_dim))
        sin_emb = mx.random.normal((seq_len, attn.head_dim))

        output = attn(x, position_embeddings=(cos_emb, sin_emb))
        mx.eval(output)

        assert output.shape == x.shape

    def test_vision_attention_with_chunked_attention(self):
        """Verify VisionAttention works with cu_seqlens (chunked attention)."""
        from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_attention import VisionAttention

        attn = VisionAttention(embed_dim=1280, num_heads=16)

        # Input with multiple chunks
        total_len = 128
        x = mx.random.normal((total_len, 1280))

        # Cumulative sequence lengths for 4 chunks of 32 each
        cu_seqlens = mx.array([0, 32, 64, 96, 128])

        output = attn(x, cu_seqlens=cu_seqlens)
        mx.eval(output)

        assert output.shape == x.shape


class TestQwenImageAttentionBlock3DIntegration:
    """Integration tests for optimized QwenImageAttentionBlock3D."""

    def test_attention_block_instantiation(self):
        """Verify QwenImageAttentionBlock3D can be instantiated."""
        from mflux.models.qwen.model.qwen_vae.qwen_image_attention_block_3d import QwenImageAttentionBlock3D

        block = QwenImageAttentionBlock3D(dim=512)
        assert block is not None
        assert block.dim == 512
        assert hasattr(block, 'scale')

    def test_attention_block_forward_pass(self):
        """Verify QwenImageAttentionBlock3D forward pass produces correct shapes."""
        from mflux.models.qwen.model.qwen_vae.qwen_image_attention_block_3d import QwenImageAttentionBlock3D

        dim = 512
        block = QwenImageAttentionBlock3D(dim=dim)

        # Input: (batch, channels, time, height, width)
        batch_size = 1
        time = 4
        height = 16
        width = 16
        x = mx.random.normal((batch_size, dim, time, height, width))

        output = block(x)
        mx.eval(output)

        # Output should have same shape as input (residual connection)
        assert output.shape == x.shape

    def test_attention_block_residual_connection(self):
        """Verify the residual connection is working."""
        from mflux.models.qwen.model.qwen_vae.qwen_image_attention_block_3d import QwenImageAttentionBlock3D

        dim = 128
        block = QwenImageAttentionBlock3D(dim=dim)

        x = mx.random.normal((1, dim, 2, 8, 8))
        output = block(x)
        mx.eval(output)

        # With residual, output should be different from just attention
        # (input + attention_output)
        assert output.shape == x.shape

        # Output shouldn't be exactly equal to input (attention adds something)
        diff = mx.abs(output - x).mean().item()
        assert diff > 0.001, "Attention should modify the input"


class TestLoRALayerIntegration:
    """Integration tests for optimized LoRA layers."""

    def test_lora_linear_instantiation(self):
        """Verify LoRALinear can be instantiated."""
        from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear

        lora = LoRALinear(input_dims=512, output_dims=512, r=16, scale=1.0)
        assert lora is not None
        assert lora.scale == 1.0
        assert lora.lora_A.shape == (512, 16)
        assert lora.lora_B.shape == (16, 512)

    def test_lora_linear_forward_pass(self):
        """Verify LoRALinear forward pass produces correct shapes."""
        from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear

        lora = LoRALinear(input_dims=512, output_dims=512, r=16, scale=1.0)

        x = mx.random.normal((4, 64, 512))
        output = lora(x)
        mx.eval(output)

        assert output.shape == (4, 64, 512)

    def test_lora_linear_from_linear(self):
        """Verify LoRALinear.from_linear works correctly."""
        from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear

        base_linear = nn.Linear(512, 512)
        lora = LoRALinear.from_linear(base_linear, r=16, scale=0.5)

        assert lora is not None
        assert lora.linear is base_linear
        assert lora.scale == 0.5

        x = mx.random.normal((2, 32, 512))
        output = lora(x)
        mx.eval(output)

        assert output.shape == (2, 32, 512)

    def test_fused_lora_linear_instantiation(self):
        """Verify FusedLoRALinear can be instantiated."""
        from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
        from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear

        base_linear = nn.Linear(512, 512)
        lora1 = LoRALinear.from_linear(base_linear, r=16, scale=0.5)
        lora2 = LoRALinear.from_linear(base_linear, r=8, scale=0.3)

        fused = FusedLoRALinear(base_linear, [lora1, lora2])
        assert fused is not None
        assert len(fused.loras) == 2

    def test_fused_lora_linear_forward_pass(self):
        """Verify FusedLoRALinear forward pass works with multiple LoRAs."""
        from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
        from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear

        base_linear = nn.Linear(512, 512)
        lora1 = LoRALinear.from_linear(base_linear, r=16, scale=0.5)
        lora2 = LoRALinear.from_linear(base_linear, r=8, scale=0.3)

        fused = FusedLoRALinear(base_linear, [lora1, lora2])

        x = mx.random.normal((2, 32, 512))
        output = fused(x)
        mx.eval(output)

        assert output.shape == (2, 32, 512)


class TestEndToEndComponentChain:
    """Test that optimized components work together in typical usage patterns."""

    def test_transformer_initialization_chain(self):
        """Verify the optimized components can be initialized as part of a transformer."""
        from mflux.models.flux.model.flux_transformer.embed_nd import EmbedND
        from mflux.models.flux.model.flux_transformer.time_text_embed import TimeTextEmbed
        from mflux.models.common.config.model_config import ModelConfig

        config = ModelConfig.dev()

        # These are instantiated together in Transformer.__init__
        pos_embed = EmbedND()
        time_text_embed = TimeTextEmbed(model_config=config)

        # Verify they work
        ids = mx.zeros((1, 256, 3))
        rope_output = pos_embed(ids)
        mx.eval(rope_output)

        time_output = time_text_embed(
            mx.array([500.0]),
            mx.random.normal((1, 768)),
            mx.array([3.5])
        )
        mx.eval(time_output)

        assert rope_output is not None
        assert time_output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
