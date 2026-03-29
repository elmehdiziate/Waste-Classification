# ======================== vit_b_model.py ============================
'''
Date: 29/03/2026
Author: [Your Name]

Vision Transformer Base (ViT-B/16) for industrial waste classification.
Pretrained on ImageNet-1k (google/vit-base-patch16-224 via torchvision).

Architecture:
  - Patch size  : 16 × 16
  - Resolution  : 224 × 224  →  196 patches + 1 [CLS] token
  - Hidden dim  : 768
  - Depth       : 12 transformer encoder blocks
  - Heads       : 12 attention heads
  - Parameters  : ~86 M total

Fine-tuning strategy (two-phase):
  Phase 1 — freeze backbone, train classification head only  (fast warm-up)
  Phase 2 — unfreeze all layers, end-to-end fine-tuning     (full adaptation)
'''

import torch
import torch.nn as nn
from torchvision import models


class ViT_B(nn.Module):

    def __init__(self, num_classes: int = 28, dropout: float = 0.3, freeze: bool = True):
        super(ViT_B, self).__init__()

        self.num_classes = num_classes
        self.dropout     = dropout
        self.model_name  = "ViT-B/16"

        # Load pretrained ViT-B/16 backbone
        # IMAGENET1K_V1 weights give strong general visual features for transfer learning.
        # ViT learns global context via self-attention rather than local convolution,
        # making it well-suited to recognising shape and texture cues in waste images.
        vit_backbone = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1
        )

        # Freeze backbone if Phase 1
        # Freezing prevents catastrophic forgetting of ImageNet features while the
        # new classification head stabilises during the first training phase.
        if freeze:
            for parameter in vit_backbone.parameters():
                parameter.requires_grad = False

        # Remove the original classification head (768 → 1000 classes)
        # We keep all encoder blocks and only replace the final projection.
        in_features = vit_backbone.heads.head.in_features   # 768
        vit_backbone.heads = nn.Identity()                  # strip original head

        self.vit_backbone = vit_backbone

        # New classification head tailored to the WaRP-C dataset
        # Dropout on the [CLS] token embedding reduces reliance on any single feature
        # dimension, which is important given the visual similarity between waste classes.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),                     # regularisation on [CLS] embedding
            nn.Linear(in_features, 512),               # compress 768-dim representation
            nn.GELU(),                                 # GELU matches ViT's internal activation
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)                # final class prediction layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit_backbone(x)    # [CLS] token embedding — shape (B, 768)
        logits   = self.classifier(features)
        return logits

    def unfreeze(self):
        # Unfreeze entire backbone for Phase 2 full fine-tuning.
        # Use a small learning rate (e.g. 1e-5) to avoid disrupting pretrained weights.
        for parameter in self.vit_backbone.parameters():
            parameter.requires_grad = True

    def freeze(self):
        for parameter in self.vit_backbone.parameters():
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
            f"  backbone        : ViT-B/16 (ImageNet-1k V1)\n"
            f"  head            : Linear(768→512) → Linear(512→{self.num_classes})"
        )
