"""ClimateAdapt V3 GroupNorm Compact U-Net.

V3 intentionally retains the proven V2 spatial architecture so performance
changes can be attributed primarily to broader data, geographic splitting and
the stable feature representation. The default input count changes to 11.
"""

from .unet_v2 import CompactUNetV2


class CompactUNetV3(CompactUNetV2):
    """V3 model identity with an 11-channel default input contract."""

    def __init__(
        self,
        input_channels=11,
        base_channels=16,
        dropout_probability=0.20,
    ):
        super().__init__(
            input_channels=input_channels,
            base_channels=base_channels,
            dropout_probability=dropout_probability,
        )


if __name__ == "__main__":
    import torch

    model = CompactUNetV3()
    example = torch.zeros(1, 11, 128, 128)
    output = model(example)
    print("Output shape:", tuple(output.shape))
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
