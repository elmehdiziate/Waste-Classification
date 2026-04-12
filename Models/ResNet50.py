# ======================== Models/ResNet50.py ============================
'''
Date: 10/03/2026
Author: Scott Lewis

ResNet50 fine-tuned for 28-class Warp-C classification.

Design rationale:
- Bottleneck (→512) gives the head capacity to bridge ImageNet features
  to industrial waste domain; justified by small dataset size (8,823 images)
- ReLU after Linear (correct placement — not before)
- Single Dropout after activation (standard placement)

Original paper: https://arxiv.org/abs/1512.03385
'''

import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):

    def __init__(self, num_classes: int = 28, dropout: float = 0.4, freeze: bool = True):
        super(ResNet50, self).__init__()

        self.num_classes = num_classes
        self.dropout     = dropout
        self.model_name  = "ResNet-50 (WaRP-C)"

        # Load pretrained backbone
        RN_backbone      = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.in_features = RN_backbone.fc.in_features   # 2048 for ResNet50

        # Remove fc; retain avgpool → guarantees (B,2048,1,1) output
        self.RN_backbone = nn.Sequential(*list(RN_backbone.children())[:-1])

        # Freeze backbone if Phase 1
        if freeze:
            for parameter in self.RN_backbone.parameters():
                parameter.requires_grad = False

        # WaRP-C optimised head: bottleneck with domain-shift stabilisation
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # (B,2048,1,1) → (B,2048)
            nn.Linear(self.in_features, 512),  # bottleneck
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)        # final logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.RN_backbone(x)        # visual features extracted
        logits   = self.classifier(features)  # mapping features to class predictions
        return logits

    def unfreeze(self):
        for parameter in self.RN_backbone.parameters():
            parameter.requires_grad = True

    def freeze(self):
        for parameter in self.RN_backbone.parameters():
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
            f"  backbone        : ResNet-50 (ImageNet-1k V2)\n"
            f"  head            : Linear({self.in_features}→512) → ReLU → Dropout → Linear(512→{self.num_classes})"
        )