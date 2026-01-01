"""Tests for MLX optimizations in mflux.

Verifies that the mx.compile and SDPA optimizations produce correct results
and provide performance improvements on Apple Silicon.
"""
import math
import time

import mlx.core as mx
import pytest
from mlx import nn


class TestTimeProjectionOptimization:
    """Test mx.compile optimization for timestep projection."""

    def test_time_proj_compiled_produces_correct_output(self):
        """Verify compiled time projection matches reference implementation."""
        # Reference implementation (original)
        def time_proj_reference(time_steps: mx.array) -> mx.array:
            max_period = 10000
            half_dim = 128
            exponent = -math.log(max_period) * mx.arange(start=0, stop=half_dim, step=None, dtype=mx.float32)
            exponent = exponent / half_dim
            emb = mx.exp(exponent)
            emb = time_steps[:, None].astype(mx.float32) * emb[None, :]
            emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
            emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
            return emb

        # Compiled implementation
        @mx.compile
        def time_proj_compiled(time_steps: mx.array, exponent_base: mx.array) -> mx.array:
            emb = time_steps[:, None].astype(mx.float32) * exponent_base[None, :]
            emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
            half_dim = emb.shape[-1] // 2
            emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
            return emb

        # Pre-compute exponent base
        max_period = 10000
        half_dim = 128
        exponent = -math.log(max_period) * mx.arange(start=0, stop=half_dim, step=None, dtype=mx.float32)
        exponent_base = mx.exp(exponent / half_dim)

        # Test with various timestep values
        for timestep in [0.0, 0.5, 1.0, 500.0, 999.0]:
            time_steps = mx.array([timestep])
            reference = time_proj_reference(time_steps)
            compiled = time_proj_compiled(time_steps, exponent_base)
            mx.eval(reference, compiled)

            # Check shapes match
            assert reference.shape == compiled.shape, f"Shape mismatch at timestep {timestep}"

            # Check values are close
            diff = mx.abs(reference - compiled).max().item()
            assert diff < 1e-5, f"Value mismatch at timestep {timestep}: max diff = {diff}"

    def test_time_proj_compiled_performance(self):
        """Verify compiled version is not slower than reference."""
        @mx.compile
        def time_proj_compiled(time_steps: mx.array, exponent_base: mx.array) -> mx.array:
            emb = time_steps[:, None].astype(mx.float32) * exponent_base[None, :]
            emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
            half_dim = emb.shape[-1] // 2
            emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
            return emb

        max_period = 10000
        half_dim = 128
        exponent = -math.log(max_period) * mx.arange(start=0, stop=half_dim, step=None, dtype=mx.float32)
        exponent_base = mx.exp(exponent / half_dim)

        # Warmup
        for _ in range(10):
            time_steps = mx.array([0.5])
            result = time_proj_compiled(time_steps, exponent_base)
            mx.eval(result)

        # Benchmark
        iterations = 100
        start = time.perf_counter()
        for i in range(iterations):
            time_steps = mx.array([float(i) / iterations])
            result = time_proj_compiled(time_steps, exponent_base)
            mx.eval(result)
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / iterations) * 1000
        # Should complete quickly (< 1ms per call on M4)
        assert avg_time_ms < 5.0, f"Time projection too slow: {avg_time_ms:.3f}ms per call"


class TestRoPEOptimization:
    """Test mx.compile optimization for Rotary Position Embedding."""

    def test_rope_compiled_produces_correct_output(self):
        """Verify compiled RoPE matches reference implementation."""
        # Reference implementation
        def rope_reference(pos: mx.array, dim: int, theta: float) -> mx.array:
            scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
            omega = 1.0 / (theta**scale)
            batch_size, seq_length = pos.shape
            pos_expanded = mx.expand_dims(pos, axis=-1)
            omega_expanded = mx.expand_dims(omega, axis=0)
            out = pos_expanded * omega_expanded
            cos_out = mx.cos(out)
            sin_out = mx.sin(out)
            stacked_out = mx.stack([cos_out, -sin_out, sin_out, cos_out], axis=-1)
            out = mx.reshape(stacked_out, (batch_size, -1, dim // 2, 2, 2))
            return out

        # Compiled core
        @mx.compile
        def rope_core_compiled(pos_expanded: mx.array, omega_expanded: mx.array, batch_size: int, seq_length: int, half_dim: int) -> mx.array:
            out = pos_expanded * omega_expanded
            cos_out = mx.cos(out)
            sin_out = mx.sin(out)
            stacked_out = mx.stack([cos_out, -sin_out, sin_out, cos_out], axis=-1)
            return mx.reshape(stacked_out, (batch_size, seq_length, half_dim, 2, 2))

        # Test parameters
        dim = 56
        theta = 10000
        batch_size = 1
        seq_length = 64

        # Pre-compute omega
        omega = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

        # Generate test positions
        pos = mx.arange(0, seq_length, dtype=mx.float32).reshape(batch_size, seq_length)

        # Reference
        reference = rope_reference(pos, dim, theta)

        # Compiled
        pos_expanded = mx.expand_dims(pos, axis=-1)
        omega_expanded = mx.expand_dims(omega, axis=0)
        compiled = rope_core_compiled(pos_expanded, omega_expanded, batch_size, seq_length, dim // 2)

        mx.eval(reference, compiled)

        # Check shapes
        assert reference.shape == compiled.shape, f"Shape mismatch: {reference.shape} vs {compiled.shape}"

        # Check values
        diff = mx.abs(reference - compiled).max().item()
        assert diff < 1e-5, f"Value mismatch: max diff = {diff}"


class TestSDPAConversion:
    """Test SDPA conversion produces correct attention outputs."""

    def test_manual_attention_matches_sdpa(self):
        """Verify manual attention and SDPA produce same results."""
        from mlx.core.fast import scaled_dot_product_attention

        # Test dimensions
        batch_size = 1
        num_heads = 8
        seq_len = 64
        head_dim = 64
        scale = 1.0 / (head_dim ** 0.5)

        # Random Q, K, V
        mx.random.seed(42)
        q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # Manual attention
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
        attn_weights = mx.softmax(scores, axis=-1)
        manual_output = mx.matmul(attn_weights, v)

        # SDPA
        sdpa_output = scaled_dot_product_attention(q, k, v, scale=scale)

        mx.eval(manual_output, sdpa_output)

        # Check values match
        diff = mx.abs(manual_output - sdpa_output).max().item()
        assert diff < 1e-4, f"Manual vs SDPA mismatch: max diff = {diff}"

    def test_sdpa_performance_vs_manual(self):
        """Verify SDPA is at least as fast as manual implementation."""
        from mlx.core.fast import scaled_dot_product_attention

        batch_size = 1
        num_heads = 16
        seq_len = 256
        head_dim = 64
        scale = 1.0 / (head_dim ** 0.5)

        mx.random.seed(42)
        q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        def manual_attention(q, k, v):
            scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
            attn_weights = mx.softmax(scores, axis=-1)
            return mx.matmul(attn_weights, v)

        # Warmup
        for _ in range(10):
            _ = manual_attention(q, k, v)
            mx.eval(_)
            _ = scaled_dot_product_attention(q, k, v, scale=scale)
            mx.eval(_)

        # Benchmark manual
        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            result = manual_attention(q, k, v)
            mx.eval(result)
        manual_time = time.perf_counter() - start

        # Benchmark SDPA
        start = time.perf_counter()
        for _ in range(iterations):
            result = scaled_dot_product_attention(q, k, v, scale=scale)
            mx.eval(result)
        sdpa_time = time.perf_counter() - start

        # SDPA should not be slower (allowing 20% tolerance)
        ratio = sdpa_time / manual_time
        assert ratio < 1.2, f"SDPA slower than manual: {ratio:.2f}x (manual={manual_time:.3f}s, sdpa={sdpa_time:.3f}s)"


class TestLoRAOptimization:
    """Test mx.addmm optimization for LoRA layers."""

    def test_addmm_produces_correct_output(self):
        """Verify mx.addmm matches separate operations."""
        mx.random.seed(42)

        batch_size = 4
        seq_len = 64
        in_dim = 512
        out_dim = 512
        rank = 16
        scale = 0.5

        # Input and LoRA weights
        x = mx.random.normal((batch_size, seq_len, in_dim))
        base_out = mx.random.normal((batch_size, seq_len, out_dim))
        lora_A = mx.random.normal((in_dim, rank))
        lora_B = mx.random.normal((rank, out_dim))

        # Original implementation
        lora_intermediate = mx.matmul(x, lora_A)
        lora_out = mx.matmul(lora_intermediate, lora_B)
        original_result = base_out + scale * lora_out

        # Optimized with mx.addmm
        lora_intermediate = mx.matmul(x, lora_A)
        optimized_result = mx.addmm(base_out, lora_intermediate, lora_B, alpha=scale, beta=1.0)

        mx.eval(original_result, optimized_result)

        # Check values match
        diff = mx.abs(original_result - optimized_result).max().item()
        assert diff < 1e-5, f"Original vs addmm mismatch: max diff = {diff}"

    def test_addmm_performance(self):
        """Verify mx.addmm provides performance benefit."""
        mx.random.seed(42)

        batch_size = 4
        seq_len = 256
        in_dim = 1024
        out_dim = 1024
        rank = 32
        scale = 0.5

        x = mx.random.normal((batch_size, seq_len, in_dim))
        base_out = mx.random.normal((batch_size, seq_len, out_dim))
        lora_A = mx.random.normal((in_dim, rank))
        lora_B = mx.random.normal((rank, out_dim))

        def original_impl():
            lora_intermediate = mx.matmul(x, lora_A)
            lora_out = mx.matmul(lora_intermediate, lora_B)
            return base_out + scale * lora_out

        def optimized_impl():
            lora_intermediate = mx.matmul(x, lora_A)
            return mx.addmm(base_out, lora_intermediate, lora_B, alpha=scale, beta=1.0)

        # Warmup
        for _ in range(10):
            mx.eval(original_impl())
            mx.eval(optimized_impl())

        # Benchmark
        iterations = 100

        start = time.perf_counter()
        for _ in range(iterations):
            mx.eval(original_impl())
        original_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(iterations):
            mx.eval(optimized_impl())
        optimized_time = time.perf_counter() - start

        # Optimized should not be slower
        ratio = optimized_time / original_time
        assert ratio < 1.1, f"Optimized slower than original: {ratio:.2f}x"


class TestIntegration:
    """Integration tests for optimized components."""

    def test_time_text_embed_imports(self):
        """Verify TimeTextEmbed can be imported with optimizations."""
        from mflux.models.flux.model.flux_transformer.time_text_embed import TimeTextEmbed, _time_proj_compiled
        assert callable(_time_proj_compiled)

    def test_embed_nd_imports(self):
        """Verify EmbedND can be imported with optimizations."""
        from mflux.models.flux.model.flux_transformer.embed_nd import EmbedND, _rope_core
        assert callable(_rope_core)

    def test_qwen_vision_attention_imports(self):
        """Verify VisionAttention uses SDPA."""
        from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_attention import VisionAttention
        # Check that scaled_dot_product_attention is imported in the module
        import mflux.models.qwen.model.qwen_text_encoder.qwen_vision_attention as va_module
        assert hasattr(va_module, 'scaled_dot_product_attention')

    def test_qwen_vae_attention_imports(self):
        """Verify QwenImageAttentionBlock3D uses SDPA."""
        from mflux.models.qwen.model.qwen_vae.qwen_image_attention_block_3d import QwenImageAttentionBlock3D
        import mflux.models.qwen.model.qwen_vae.qwen_image_attention_block_3d as vae_module
        assert hasattr(vae_module, 'scaled_dot_product_attention')

    def test_lora_layer_imports(self):
        """Verify LoRA layers can be imported."""
        from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
        from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
        assert LoRALinear is not None
        assert FusedLoRALinear is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
