"""
El Mehdi Ziate
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    One convolutional block:  Conv → BatchNorm → ReLU → MaxPool

    This is the fundamental building unit. Every block:
      - Learns features at the current spatial scale (Conv)
      - Stabilises the activation distribution (BatchNorm)
      - Introduces non-linearity (ReLU)
      - Halves the spatial dimensions (MaxPool 2×2)

    Parameters
    ----------
    in_channels  : number of input feature maps
    out_channels : number of output feature maps (filters to learn)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            # 3×3 conv, padding=1 keeps spatial size the same before pooling
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            # bias=False because BatchNorm has its own learnable bias (beta)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # halves H and W: 224→112→56→28→14 across 4 blocks
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNModel(nn.Module):
    """
    Input:  (B, 3, 224, 224)

    Block 1: Conv(3→32)   + BN + ReLU + MaxPool  → (B, 32,  112, 112)
    Block 2: Conv(32→64)  + BN + ReLU + MaxPool  → (B, 64,   56,  56)
    Block 3: Conv(64→128) + BN + ReLU + MaxPool  → (B, 128,  28,  28)
    Block 4: Conv(128→256)+ BN + ReLU + MaxPool  → (B, 256,  14,  14)

    GAP:     AdaptiveAvgPool2d(1)                → (B, 256,   1,   1)
    Flatten:                                     → (B, 256)
    Dropout: p=0.5
    FC:      Linear(256 → 128) + ReLU
    Dropout: p=0.3
    Output:  Linear(128 → num_classes)           → (B, 28)

    Parameters
    ----------
    num_classes : number of output classes (default 28 for WaRP-C)
    dropout_p   : dropout probability before first FC (default 0.5)
    """

    def __init__(self, num_classes: int = 28, dropout_p: float = 0.5):
        super().__init__()

        self.num_classes = num_classes

        # Channel progression 3 → 32 → 64 → 128 → 256
        # Doubles channels each block — standard VGG-style progression
        self.features = nn.Sequential(
            ConvBlock(3,   32),    # block 1: RGB → 32 feature maps
            ConvBlock(32,  64),    # block 2: 32  → 64 feature maps
            ConvBlock(64,  128),   # block 3: 64  → 128 feature maps
            ConvBlock(128, 256),   # block 4: 128 → 256 feature maps
        )

        # Collapses each 14×14 feature map to a single number.
        # Output: (B, 256, 1, 1) → flatten to (B, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),        # dropout before first FC
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),              # lighter dropout before output
            nn.Linear(128, num_classes),    # final logits — NO softmax here
        )

        # Kaiming He initialisation for ReLU networks 
        # Without this, deep networks train much more slowly
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming He initialisation for all Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)   # gamma = 1
                nn.init.constant_(m.bias,   0)   # beta  = 0
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, 3, 224, 224) normalised image tensor

        Returns
        -------
        logits : (B, num_classes) — raw scores before softmax
        """
        x = self.features(x)   # (B, 256, 14, 14)
        x = self.gap(x)        # (B, 256,  1,  1)
        x = x.flatten(1)       # (B, 256)
        x = self.classifier(x) # (B, 28)
        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)