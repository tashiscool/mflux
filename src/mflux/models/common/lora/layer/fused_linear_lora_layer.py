import mlx.core as mx
from mlx import nn

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear


class FusedLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear | nn.QuantizedLinear, loras: list[LoRALinear]):
        super().__init__()
        self.base_linear = base_linear
        self.loras = loras

    def __call__(self, x):
        base_out = self.base_linear(x)

        # Use fused multiply-add (mx.addmm) for each LoRA adapter
        # Accumulates directly into base_out for better memory efficiency
        result = base_out
        for lora in self.loras:
            lora_intermediate = mx.matmul(x, lora.lora_A)
            result = mx.addmm(result, lora_intermediate, lora.lora_B, alpha=lora.scale, beta=1.0)

        return result
