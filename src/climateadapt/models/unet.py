import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
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
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
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


class DownBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        dropout_probability,
    ):
        super().__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.convolution = DoubleConv(
            input_channels,
            output_channels,
            dropout_probability,
        )

    def forward(self, inputs):
        inputs = self.pool(inputs)
        return self.convolution(inputs)


class UpBlock(nn.Module):
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

        self.convolution = DoubleConv(
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

        combined = torch.cat(
            [
                skip_connection,
                inputs,
            ],
            dim=1,
        )

        return self.convolution(combined)


class CompactUNet(nn.Module):
    """
    Compact U-Net for binary coastal-flood susceptibility.

    Dropout is included in the encoder and bottleneck so it
    can later remain active during repeated inference passes
    for Monte Carlo dropout uncertainty estimation.
    """

    def __init__(
        self,
        input_channels=24,
        base_channels=16,
        dropout_probability=0.20,
    ):
        super().__init__()

        self.initial = DoubleConv(
            input_channels,
            base_channels,
            dropout_probability=0.0,
        )

        self.encoder_1 = DownBlock(
            base_channels,
            base_channels * 2,
            dropout_probability,
        )

        self.encoder_2 = DownBlock(
            base_channels * 2,
            base_channels * 4,
            dropout_probability,
        )

        self.encoder_3 = DownBlock(
            base_channels * 4,
            base_channels * 8,
            dropout_probability,
        )

        self.bottleneck = DownBlock(
            base_channels * 8,
            base_channels * 16,
            dropout_probability,
        )

        self.decoder_1 = UpBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
            dropout_probability,
        )

        self.decoder_2 = UpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            dropout_probability,
        )

        self.decoder_3 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            dropout_probability,
        )

        self.decoder_4 = UpBlock(
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