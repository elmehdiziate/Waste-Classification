# ======================== Models/ConvNeXTv2.py ============================
'''
Date: 28/03/2026
Author: Scott Lewis

ConvNeXt-V2-Base fine-tuned for 28-class Warp-C classification.

Note: Requires timm — pip install timm
ConvNeXt V2 is not available in torchvision; loaded via timm.
timm's num_classes=0 removes the original head and returns pooled
features as (B, 1024) directly — no Flatten() needed.

Original paper: https://arxiv.org/abs/2301.00808
'''

import torch
import torch.nn as nn
import timm


class ConvNeXTv2(nn.Module):

    def __init__(self, num_classes: int = 28, dropout: float = 0.4, freeze: bool = True):
        super(ConvNeXTv2, self).__init__()

        self.num_classes = num_classes
        self.dropout     = dropout
        self.model_name  = "ConvNeXt-V2-Base"

        # Load pretrained backbone via timm
        # num_classes=0 removes the original head
        # backbone returns pooled (B, 1024) features directly — no Flatten needed
        self.CN2_backbone = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k",
            pretrained=True,
            num_classes=0
        )
        self.in_features = self.CN2_backbone.num_features   # 1024 for ConvNeXt-V2-Base

        # Freeze backbone if Phase 1
        if freeze:
            for parameter in self.CN2_backbone.parameters():
                parameter.requires_grad = False

        # ConvNeXt-V2 canonical head — consistent with paper (Woo et al., 2023)
        # No Flatten — timm returns pooled (B, 1024) directly
        # Dropout rate can be tuned upward (e.g. 0.5) if overfitting observed in Phase 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.in_features),            # normalises over last dim; consistent with paper
            nn.Dropout(p=dropout),                     # regularisation before projection
            nn.Linear(self.in_features, num_classes)   # direct projection to class scores
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.CN2_backbone(x)       # pooled visual features; timm handles pooling
        logits   = self.classifier(features)  # mapping features to class predictions
        return logits

    def unfreeze(self):
        for parameter in self.CN2_backbone.parameters():
            parameter.requires_grad = True

    def freeze(self):
        for parameter in self.CN2_backbone.parameters():
            parameter.requires_grad = False

    def get_parameter_counts(self) -> dict:     # for efficiency comparison table
        total_parameters     = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in self.parameters()
                                   if parameter.requires_grad)
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
            f"  total params    : {counts['total_M']}M\n"
            f"  trainable params: {counts['trainable_M']}M\n"
            f"  backbone        : ConvNeXt-V2-Base (FCMAE → IN-22k → IN-1k)\n"
            f"  head            : LayerNorm → Dropout → Linear({self.in_features}→{self.num_classes})"
        )