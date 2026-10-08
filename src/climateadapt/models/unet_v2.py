import torch
import torch.nn as nn
import torch.nn.functional as F


def make_group_norm(channels, maximum_groups=8):
    """Return GroupNorm with a group count that divides the channels."""
    groups = min(maximum_groups, channels)

    while channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(groups, channels)


class DoubleConvV2(nn.Module):
    """Two convolutions using batch-size-independent GroupNorm."""

    def __init__(
        self,
        input_channels,
        output_channels,
        dropout_probability=0.0,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_group_norm(output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_group_norm(output_channels),
            nn.SiLU(inplace=True),
        ]

        if dropout_probability > 0:
            layers.append(
                nn.Dropout2d(
                    p=dropout_probability
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.block(inputs)


class DownBlockV2(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        dropout_probability,
    ):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.convolution = DoubleConvV2(
            input_channels,
            output_channels,
            dropout_probability,
        )

    def forward(self, inputs):
        return self.convolution(
            self.pool(inputs)
        )


class UpBlockV2(nn.Module):
    def __init__(
        self,
        input_channels,
        skip_channels,
        output_channels,
        dropout_probability,
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=2,
            stride=2,
        )

        self.convolution = DoubleConvV2(
            output_channels + skip_channels,
            output_channels,
            dropout_probability,
        )

    def forward(self, inputs, skip_connection):
        inputs = self.upsample(inputs)

        difference_y = (
            skip_connection.size(2)
            - inputs.size(2)
        )
        difference_x = (
            skip_connection.size(3)
            - inputs.size(3)
        )

        inputs = F.pad(
            inputs,
            [
                difference_x // 2,
                difference_x - difference_x // 2,
                difference_y // 2,
                difference_y - difference_y // 2,
            ],
        )

        return self.convolution(
            torch.cat(
                [skip_connection, inputs],
                dim=1,
            )
        )


class CompactUNetV2(nn.Module):
    """
    Compact binary-segmentation U-Net for Experiment 2.

    GroupNorm is stable with the four-sample batches required by a
    4 GB GPU. Dropout remains in the network for later Monte Carlo
    dropout uncertainty estimation.
    """

    def __init__(
        self,
        input_channels=25,
        base_channels=16,
        dropout_probability=0.20,
    ):
        super().__init__()

        self.initial = DoubleConvV2(
            input_channels,
            base_channels,
        )
        self.encoder_1 = DownBlockV2(
            base_channels,
            base_channels * 2,
            dropout_probability,
        )
        self.encoder_2 = DownBlockV2(
            base_channels * 2,
            base_channels * 4,
            dropout_probability,
        )
        self.encoder_3 = DownBlockV2(
            base_channels * 4,
            base_channels * 8,
            dropout_probability,
        )
        self.bottleneck = DownBlockV2(
            base_channels * 8,
            base_channels * 16,
            dropout_probability,
        )

        self.decoder_1 = UpBlockV2(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
            dropout_probability,
        )
        self.decoder_2 = UpBlockV2(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            dropout_probability,
        )
        self.decoder_3 = UpBlockV2(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            dropout_probability,
        )
        self.decoder_4 = UpBlockV2(
            base_channels * 2,
            base_channels,
            base_channels,
            dropout_probability=0.0,
        )

        self.output_layer = nn.Conv2d(
            base_channels,
            1,
            kernel_size=1,
        )

    def forward(self, inputs):
        level_1 = self.initial(inputs)
        level_2 = self.encoder_1(level_1)
        level_3 = self.encoder_2(level_2)
        level_4 = self.encoder_3(level_3)

        bottleneck = self.bottleneck(level_4)

        decoded = self.decoder_1(
            bottleneck,
            level_4,
        )
        decoded = self.decoder_2(
            decoded,
            level_3,
        )
        decoded = self.decoder_3(
            decoded,
            level_2,
        )
        decoded = self.decoder_4(
            decoded,
            level_1,
        )

        return self.output_layer(decoded)


if __name__ == "__main__":
    model = CompactUNetV2(input_channels=25)
    example = torch.zeros(1, 25, 128, 128)
    output = model(example)

    print("Output shape:", tuple(output.shape))
    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )
