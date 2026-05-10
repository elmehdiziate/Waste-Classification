"""
El Mehdi Ziate
-----------------
Swin Transformer fine-tuned for WaRP-C 28-class classification.)
Reference: Liu et al. (2021) Swin Transformer paper, fine-tuning recipe.
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required. Install with: pip install timm"
    )


class SwinTransformerWaRP(nn.Module):
    """
    Swin-Tiny fine-tuned for WaRP-C 28-class classification.
    """

    # timm model name: Swin-Tiny pretrained on ImageNet-1K
    MODEL_NAME = "swin_tiny_patch4_window7_224"

    def __init__(
        self,
        num_classes:    int   = 28,
        pretrained:     bool  = True,
        drop_rate:      float = 0.0,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()

        self.num_classes = num_classes

        #  Load pretrained Swin-Tiny backbone via timm 

        self.backbone = timm.create_model(
            self.MODEL_NAME,
            pretrained      = pretrained,
            num_classes     = 0,        # remove 1000-class ImageNet head
            drop_rate       = drop_rate,
            drop_path_rate  = drop_path_rate,
        )

        #Find the backbone's output feature dimension 
        num_features = self.backbone.num_features   # 768 for Swin-Tiny

        # Our new classification head 
        self.head = nn.Linear(num_features, num_classes)

        # Initialise the new head properly 
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="relu")
        nn.init.zeros_(self.head.bias)

        if pretrained:
            print(f"[SwinTransformerWaRP] Loaded pretrained {self.MODEL_NAME}")
            print(f"  Backbone features : {num_features}")
            print(f"  Head              : Linear({num_features} -> {num_classes})")
            print(f"  Parameters        : {self.count_parameters():,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        # backbone does: patch partition -> 4 stages -> global avg pool
        # output shape: (B, 768)
        features = self.backbone(x)

        # head maps to class scores
        # output shape: (B, 28)
        return self.head(features)

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters: only head will be trained.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[SwinTransformerWaRP] Backbone FROZEN — training head only")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all parameters: train backbone + head together.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[SwinTransformerWaRP] Backbone UNFROZEN — full fine-tuning")

    def get_param_groups(
        self,
        head_lr:     float = 1e-4,
        backbone_lr: float = 1e-5,
    ) -> list[dict]:
        """
        Return two parameter groups with DIFFERENT learning rates.

        Why differential LR?
        The backbone already has good features from ImageNet.
        It only needs a small nudge to adapt to WaRP-C -> low LR (1e-5).
        The head is brand new and needs to learn from scratch -> higher LR (1e-4).

        If you use the same LR for both:
          - High LR on backbone -> destroys pretrained features
          - Low LR on head -> head learns too slowly
        """
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(),     "lr": head_lr},
        ]

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_trainable(self) -> dict:
        """Return counts of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}