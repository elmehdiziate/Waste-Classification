"""
Author: Sayed Omar Aabid

UNet for semantic segmentation on WaRP-S.

Architecture:
  Encoder  : ResNet-50 (ImageNet-1k pretrained) — replaces the original
              UNet contracting path with a stronger pretrained backbone.
              Skip connections are taken from four intermediate feature maps:
                layer1 → 256 ch   (stride 4  → 56×56  for 224px input)
                layer2 → 512 ch   (stride 8  → 28×28)
                layer3 → 1024 ch  (stride 16 → 14×14)
                layer4 → 2048 ch  (stride 32 →  7×7)

  Bottleneck: 2048-channel feature map from layer4.

  Decoder  : Four progressive upsample blocks (bilinear ×2 + conv).
             Each block concatenates the encoder skip and applies
             two 3×3 Conv-BN-ReLU operations (standard UNet pattern).

  Head     : 1×1 Conv → NUM_CLASSES logits (no softmax — use with
             CrossEntropyLoss which applies log-softmax internally).

Fine-tuning strategy (two-phase, mirrors ViT_B convention):
  Phase 1 — freeze encoder, train decoder + head only.
  Phase 2 — unfreeze encoder, end-to-end fine-tuning with small LR.

Design decisions:
  - ResNet-50 encoder instead of VGG: 4× fewer parameters, BatchNorm
    built-in, stronger ImageNet features for transfer to WaRP-S.
  - Bilinear upsampling + conv instead of transposed conv: avoids
    checkerboard artefacts (Odena et al., 2016).
  - Deep supervision disabled by default: WaRP-S masks are relatively
    clean, and deep supervision adds implementation complexity without
    a guaranteed accuracy gain on small datasets.
  - Dropout=0.1 applied before the final head to match ViT_B convention
    and provide light regularisation.

References:
  Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical
  Image Segmentation." MICCAI 2015.

  He et al. (2016) "Deep Residual Learning for Image Recognition." CVPR 2016.

  Odena et al. (2016) "Deconvolution and Checkerboard Artifacts."
  Distill. https://distill.pub/2016/deconv-checkerboard/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── Decoder block ──────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    One decoder stage: upsample → concat skip → Conv-BN-ReLU × 2.

    Parameters
    ----------
    in_ch   : channels coming UP from the previous decoder stage
    skip_ch : channels from the encoder skip connection
    out_ch  : output channels after this block
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()

        # Two 3×3 Conv-BN-ReLU layers — same as the original UNet contracting block
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Bilinear ×2 upsample — avoids checkerboard artefacts vs transposed conv
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Pad if spatial dims differ (rounding in strided convolutions)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                          0, skip.shape[-2] - x.shape[-2]])

        x = torch.cat([x, skip], dim=1)   # channel concat
        return self.conv(x)


# ── UNet ───────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    ResNet-50 encoder + UNet decoder for WaRP-S semantic segmentation.

    Parameters
    ----------
    num_classes : number of output classes (including background)
                  WaRP-S labelmap has 29 entries (background + 28 waste classes)
    dropout     : dropout probability applied before the 1×1 head
    freeze      : if True, freeze the encoder on init (Phase 1)
    pretrained  : load ImageNet-1k weights for the encoder
    """

    def __init__(
        self,
        num_classes: int   = 29,      # background + 28 WaRP-S classes
        dropout:     float = 0.1,
        freeze:      bool  = True,
        pretrained:  bool  = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.dropout_p   = dropout
        self.model_name  = "UNet-ResNet50"

        # ── Encoder (ResNet-50 backbone) ──────────────────────────────────────
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = models.resnet50(weights=weights)

        # Stem: conv1 + bn1 + relu → output stride 2
        self.encoder_stem  = nn.Sequential(
            resnet.conv1,    # 3 → 64, stride 2
            resnet.bn1,
            resnet.relu,
        )
        self.encoder_pool  = resnet.maxpool   # stride 2  → total stride 4

        # Four residual stages — skip connections taken here
        self.encoder1 = resnet.layer1    # 64 → 256,  stride 4  total
        self.encoder2 = resnet.layer2    # 256 → 512, stride 8  total
        self.encoder3 = resnet.layer3    # 512 → 1024, stride 16 total
        self.encoder4 = resnet.layer4    # 1024 → 2048, stride 32 total

        # ── Phase 1 freeze ────────────────────────────────────────────────────
        if freeze:
            self._set_encoder_grad(False)

        # ── Decoder ───────────────────────────────────────────────────────────
        # Each block: (channels_from_below, skip_channels, output_channels)
        self.decoder4 = DecoderBlock(2048, 1024, 512)
        self.decoder3 = DecoderBlock(512,  512,  256)
        self.decoder2 = DecoderBlock(256,  256,  128)
        self.decoder1 = DecoderBlock(128,  64,   64)

        # Final upsample from stride-2 back to full resolution
        self.final_up = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── Segmentation head ─────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(64, num_classes, kernel_size=1),   # 64 → num_classes
        )

    # ── Encoder gradient helpers ──────────────────────────────────────────────

    def _set_encoder_grad(self, requires_grad: bool) -> None:
        encoder_modules = [
            self.encoder_stem, self.encoder_pool,
            self.encoder1, self.encoder2,
            self.encoder3, self.encoder4,
        ]
        for module in encoder_modules:
            for param in module.parameters():
                param.requires_grad = requires_grad

    def freeze(self) -> None:
        """Freeze encoder — use for Phase 1 warm-up."""
        self._set_encoder_grad(False)

    def unfreeze(self) -> None:
        """Unfreeze encoder for Phase 2 end-to-end fine-tuning.
        Use a small backbone LR (e.g. 1e-5) to avoid catastrophic forgetting."""
        self._set_encoder_grad(True)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)  — normalised input image

        Returns
        -------
        logits : (B, num_classes, H, W)  — raw logits at full resolution
        """
        H, W = x.shape[2], x.shape[3]

        # ── Encode ────────────────────────────────────────────────────────────
        s0 = self.encoder_stem(x)         # (B, 64,   H/2,  W/2)
        s0p = self.encoder_pool(s0)       # (B, 64,   H/4,  W/4)
        s1 = self.encoder1(s0p)           # (B, 256,  H/4,  W/4)
        s2 = self.encoder2(s1)            # (B, 512,  H/8,  W/8)
        s3 = self.encoder3(s2)            # (B, 1024, H/16, W/16)
        s4 = self.encoder4(s3)            # (B, 2048, H/32, W/32)

        # ── Decode ────────────────────────────────────────────────────────────
        d4 = self.decoder4(s4, s3)        # (B, 512,  H/16, W/16)
        d3 = self.decoder3(d4, s2)        # (B, 256,  H/8,  W/8)
        d2 = self.decoder2(d3, s1)        # (B, 128,  H/4,  W/4)
        d1 = self.decoder1(d2, s0)        # (B, 64,   H/2,  W/2)

        # Final upsample to full resolution
        out = F.interpolate(d1, size=(H, W), mode="bilinear", align_corners=False)
        out = self.final_up(out)          # (B, 64,   H,    W)

        logits = self.head(out)           # (B, num_classes, H, W)
        return logits

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_parameter_counts(self) -> dict:
        """Return total and trainable parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model"           : self.model_name,
            "total_params"    : total,
            "trainable_params": trainable,
            "total_M"         : round(total / 1e6, 2),
            "trainable_M"     : round(trainable / 1e6, 2),
        }

    def __repr__(self) -> str:
        counts = self.get_parameter_counts()
        return (
            f"{self.model_name}\n"
            f"  num_classes     : {self.num_classes}\n"
            f"  dropout         : {self.dropout_p}\n"
            f"  total params    : {counts['total_M']}M\n"
            f"  trainable params: {counts['trainable_M']}M\n"
            f"  encoder         : ResNet-50 (ImageNet-1k V1)\n"
            f"  decoder         : 4× DecoderBlock (bilinear upsample + skip concat)\n"
            f"  head            : Dropout2d({self.dropout_p}) → Conv2d(64→{self.num_classes})"
        )