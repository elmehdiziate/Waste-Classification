'''
Author: Sayed Omar Aabid

Vision Transformer Base (ViT-B/16) for industrial waste classification.
Pretrained on ImageNet-1k (google/vit-base-patch16-224 via torchvision).
'''

import torch
import torch.nn as nn
from torchvision import models


class ViT_B(nn.Module):

    def __init__(self, num_classes: int = 28, dropout: float = 0.1, freeze: bool = True):
        super(ViT_B, self).__init__()

        self.num_classes = num_classes
        self.dropout     = dropout
        self.model_name  = "ViT-B/16"

        # Backbone 
        vit_backbone = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1
        )

        # phase 1 freeze 
        if freeze:
            for parameter in vit_backbone.parameters():
                parameter.requires_grad = False

        # Replace head 
        in_features = vit_backbone.heads.head.in_features   # 768
        vit_backbone.heads = nn.Identity()                  # strip original head

        self.vit_backbone = vit_backbone

        # Classification head 
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),          # light regularisation on [CLS] embedding
            nn.Linear(in_features, num_classes),   # 768 -> 28  (single linear layer)
        )

    # Forward pass 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit_backbone(x)     # [CLS] token embedding: shape (B, 768)
        logits   = self.classifier(features)
        return logits

    # Phase control helpers
    def unfreeze(self):
        '''Unfreeze entire backbone for Phase 2 end-to-end fine-tuning.
        Use a small learning rate (e.g. 1e-5) to avoid disrupting pretrained
        weights '''
        for parameter in self.vit_backbone.parameters():
            parameter.requires_grad = True

    def freeze(self):
        '''Re-freeze backbone (e.g. to revert to Phase 1 for ablations).'''
        for parameter in self.vit_backbone.parameters():
            parameter.requires_grad = False

    # Utilities 
    def get_parameter_counts(self) -> dict:
        '''Return total and trainable parameter counts (for comparison tables).'''
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
            f"  total params    : {counts['total_M']}M\n"
            f"  trainable params: {counts['trainable_M']}M\n"
            f"  backbone        : ViT-B/16 (ImageNet-1k V1)\n"
            f"  head            : Dropout({self.dropout}) -> Linear(768->{self.num_classes})"
        )
