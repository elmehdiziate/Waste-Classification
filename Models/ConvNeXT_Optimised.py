# ======================== Models/ConvNeXT_OptimisedV2.py ============================
"""
Date: 21/04/2026
Author: Scott Lewis

ConvNeXt-Base fine-tuned for 28-class WaRP-C classification.

Key points:
- Backbone: torchvision ConvNeXt-Base (IMAGENET1K_V1), unchanged
- Head: LayerNorm + MLP classifier with configurable depth
- Default: freeze_backbone=False (full fine-tuning)
- Added: get_param_groups() for optional differential LR
"""

import torch
import torch.nn as nn
from torchvision import models


class ConvNeXT_OptimisedV2(nn.Module):

    def __init__(
        self,
        num_classes: int = 28,
        dropout: float = 0.4,
        freeze_backbone: bool = False,
        head_depth: int = 2,
    ):
        super().__init__()

        self.num_classes     = num_classes
        self.dropout         = dropout
        self.head_depth      = head_depth
        self.model_name      = "ConvNeXt-Base (Optimised V2)"

        # ---------------------------------------------------------
        # 1. Load pretrained ConvNeXt-Base backbone
        # ---------------------------------------------------------
        backbone = models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        )

        # ConvNeXt backbone: features + global avgpool
        self.CN_backbone = nn.Sequential(
            backbone.features,
            backbone.avgpool,   # (B, 1024, 1, 1)
        )

        self.in_features = backbone.classifier[2].in_features  # 1024

        # Optional freezing
        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

        # ---------------------------------------------------------
        # 2. Classification head (MLP with configurable depth)
        # ---------------------------------------------------------
        # Optionally scale dropout with head depth (mild regularisation)
        effective_dropout = self.dropout + 0.1 * max(0, self.head_depth - 1)

        layers = [
            nn.Flatten(),                      # (B,1024,1,1) → (B,1024)
            nn.LayerNorm(self.in_features),    # stabilises features
            nn.Dropout(p=effective_dropout),
        ]

        # Hidden layers: (Linear → GELU → Dropout) × (head_depth - 1)
        for _ in range(self.head_depth - 1):
            layers.append(nn.Linear(self.in_features, self.in_features))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=effective_dropout))

        # Final normalisation + projection to logits
        layers.append(nn.LayerNorm(self.in_features))
        layers.append(nn.Linear(self.in_features, self.num_classes))

        self.classifier = nn.Sequential(*layers)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.CN_backbone(x)   # (B, 1024, 1, 1)
        logits   = self.classifier(features)
        return logits

    # ---------------------------------------------------------
    # Backbone freeze / unfreeze
    # ---------------------------------------------------------
    def freeze_backbone(self):
        for p in self.CN_backbone.parameters():
            p.requires_grad = False
        self.CN_backbone.eval()

    def unfreeze_backbone(self):
        for p in self.CN_backbone.parameters():
            p.requires_grad = True
        self.CN_backbone.train()

    # ---------------------------------------------------------
    # Differential LR support (optional)
    # ---------------------------------------------------------
    def get_param_groups(self, head_lr: float, backbone_lr: float):
        """
        Returns parameter groups for optimisers like AdamW:
        - backbone: lower LR
        - head: higher LR
        """
        return [
            {"params": self.CN_backbone.parameters(), "lr": backbone_lr},
            {"params": self.classifier.parameters(),  "lr": head_lr},
        ]

    # ---------------------------------------------------------
    # Parameter counts (for reporting)
    # ---------------------------------------------------------
    def get_parameter_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model": self.model_name,
            "total_params": total,
            "trainable_params": trainable,
            "total_M": round(total / 1e6, 2),
            "trainable_M": round(trainable / 1e6, 2),
        }

    def __repr__(self) -> str:
        c = self.get_parameter_counts()
        return (
            f"{self.model_name}\n"
            f"  num_classes     : {self.num_classes}\n"
            f"  dropout         : {self.dropout}\n"
            f"  head_depth      : {self.head_depth}\n"
            f"  total params    : {c['total_M']}M\n"
            f"  trainable params: {c['trainable_M']}M\n"
            f"  backbone        : ConvNeXt-Base (ImageNet-1k V1)\n"
            f"  head            : LayerNorm → Dropout → "
            f"{'(Linear→GELU→Dropout)×' + str(self.head_depth - 1) + ' → ' if self.head_depth > 1 else ''}"
            f"LayerNorm → Linear({self.in_features}→{self.num_classes})"
        )
