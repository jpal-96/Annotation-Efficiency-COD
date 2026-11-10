
# ============================================================
# model.py
# PVTv2-Based Encoder-Decoder for Camouflaged Object Detection
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ------------------------------------------------------------
# 1. Convolution Block
# ------------------------------------------------------------

class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------
# 2. Decoder Block
# ------------------------------------------------------------

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        self.refinement = nn.Sequential(
            ConvBNReLU(out_channels, out_channels),
            ConvBNReLU(out_channels, out_channels)
        )

    def forward(self, x):

        x = self.projection(x)
        x = self.refinement(x)

        return x


# ------------------------------------------------------------
# 3. PVTv2 COD Model
# ------------------------------------------------------------

class PVTv2COD(nn.Module):

    def __init__(
        self,
        backbone_name="pvt_v2_b2",
        pretrained=True,
        decoder_channels=128,
        in_channels=3
    ):

        super().__init__()

        # PVTv2 Encoder
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3)
        )

        encoder_channels = self.encoder.feature_info.channels()

        # Multi-scale Decoder Blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(ch, decoder_channels)
            for ch in encoder_channels
        ])

        # Four Side-output Prediction Heads
        self.side_heads = nn.ModuleList([
            nn.Conv2d(
                decoder_channels,
                1,
                kernel_size=1
            )
            for _ in range(4)
        ])

    # --------------------------------------------------------
    # Forward Propagation
    # --------------------------------------------------------

    def forward(self, x):

        input_size = x.shape[2:]

        # Extract hierarchical encoder features
        features = self.encoder(x)

        # Project multi-scale features
        projected = [
            block(feature)
            for block, feature in zip(
                self.decoder_blocks,
                features
            )
        ]

        # Progressive top-down feature aggregation
        decoded = [None] * 4

        decoded[3] = projected[3]

        for i in range(2, -1, -1):

            upsampled = F.interpolate(
                decoded[i + 1],
                size=projected[i].shape[2:],
                mode="bilinear",
                align_corners=False
            )

            decoded[i] = projected[i] + upsampled

        # Generate four side-output predictions
        predictions = []

        for decoder_feature, side_head in zip(
            decoded,
            self.side_heads
        ):

            logits = side_head(decoder_feature)

            logits = F.interpolate(
                logits,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )

            probability_map = torch.sigmoid(logits)

            predictions.append(probability_map)

        return predictions


# ------------------------------------------------------------
# 4. Model Testing
# ------------------------------------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = PVTv2COD(
        backbone_name="pvt_v2_b2",
        pretrained=False
    ).to(device)

    # Dummy input: Batch size 2, RGB, 384 x 384
    dummy_input = torch.randn(
        2, 3, 384, 384
    ).to(device)

    # Forward pass
    outputs = model(dummy_input)

    print("\nPVTv2 COD Model Verification")
    print("-" * 45)

    print("Device:", device)
    print("Number of side outputs:", len(outputs))

    for i, output in enumerate(outputs):

        print(
            f"Side output {i + 1}: "
            f"{tuple(output.shape)}"
        )

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("-" * 45)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print("\nModel verification completed.")
