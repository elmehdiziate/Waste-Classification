# ======================== Models/ConvNeXT.py ============================
'''
Date: 28/03/2026
Author: Scott Lewis

ConvNeXt-Base fine-tuned for 28-class Warp-C classification.

Updated:
- Added head_depth parameter for deeper MLP classifier heads
- Default behaviour (head_depth=1) matches original implementation

Original paper: https://arxiv.org/abs/2201.03545
'''

import torch
import torch.nn as nn
from torchvision import models


class ConvNeXT(nn.Module):

    def __init__(
        self,
        num_classes: int = 28,
        dropout: float = 0.4,
        freeze: bool = True,
        head_depth: int = 2
    ):
        super(ConvNeXT, self).__init__()

        self.num_classes = num_classes
        self.dropout     = dropout
        self.head_depth  = head_depth
        self.model_name  = "ConvNeXt-Base"

        # Load pretrained backbone
        CN_backbone      = models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        )
        self.in_features = CN_backbone.classifier[2].in_features   # 1024 for ConvNeXt-Base

        # Keep features + avgpool; remove original classifier
        self.CN_backbone = nn.Sequential(
            CN_backbone.features,
            CN_backbone.avgpool
        )

        # Optional freezing (Phase 1 behaviour)
        if freeze:
            for parameter in self.CN_backbone.parameters():
                parameter.requires_grad = False

        # ConvNeXt-Base canonical head — appropriate for WaRP-C
        # Dropout rate can be tuned upward (e.g. 0.5) if overfitting observed in Phase 2
        layers = [
            nn.Flatten(),                      # (B,1024,1,1) → (B,1024)
            nn.LayerNorm(self.in_features),    # stabilises features
            nn.Dropout(p=dropout),
        ]

        # Add (head_depth - 1) hidden layers
        for _ in range(head_depth - 1):
            layers.append(nn.Linear(self.in_features, self.in_features))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout))

        # Final projection layer
        layers.append(nn.Linear(self.in_features, num_classes))

        self.classifier = nn.Sequential(*layers)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.CN_backbone(x)
        logits   = self.classifier(features)
        return logits

    # ---------------------------------------------------------
    # Utility functions
    # ---------------------------------------------------------
    def unfreeze(self):
        for parameter in self.CN_backbone.parameters():
            parameter.requires_grad = True

    def freeze(self):
        for parameter in self.CN_backbone.parameters():
            parameter.requires_grad = False

    def get_parameter_counts(self) -> dict:
        total_parameters     = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)
        return {
            "model"           : self.model_name,
            "total_params"    : total_parameters,
            "trainable_params": trainable_parameters,
            "total_M"         : round(total_parameters / 1e6, 2),
            "trainable_M"     : round(trainable_parameters / 1e6, 2),
        }

    def __repr__(self) -> str:
        counts = self.get_parameter_counts()
        return (
            f"{self.model_name}\n"
            f"  num_classes     : {self.num_classes}\n"
            f"  dropout         : {self.dropout}\n"
            f"  head_depth      : {self.head_depth}\n"
            f"  total params    : {counts['total_M']}M\n"
            f"  trainable params: {counts['trainable_M']}M\n"
            f"  backbone        : ConvNeXt-Base (ImageNet-1k V1)\n"
            f"  head            : LayerNorm → Dropout → "
            f"{'(Linear→GELU→Dropout)×' + str(self.head_depth - 1) + ' → ' if self.head_depth > 1 else ''}"
            f"Linear({self.in_features}→{self.num_classes})"
        )
