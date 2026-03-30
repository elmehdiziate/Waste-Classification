"""
Author: Mohamed Fahmi Ahmed
"""

import torch
import torch.nn as nn
from torchvision import models

from config.efficientnet_config import cfg


class EfficientNetV2S(nn.Module):

    def __init__(self):
        super().__init__()

        num_classes = cfg["num_classes"]  # 28
        dropout     = cfg["dropout"]      # 0.3

        # Load backbone pretrained (ImageNet)
        backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )

        # backbone.features : convolution layers
        self.features = backbone.features

        # backbone.avgpool : Global Average Pooling
        self.avgpool  = backbone.avgpool

        # Read the backbone's output size (1280) from its last layer so the code adapts
        # Replace the original ImageNet head (1000 classes) with ours for WaRP-C:
        #   Flatten  : (B, 1280, 1, 1) to (B, 1280)
        #   Linear   : 1280 features to 28 class scores 
        in_features = backbone.classifier[-1].in_features  

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)   
        x = self.avgpool(x)    
        x = self.head(x)       
        return x

    #count parameters for the summary
    def get_parameter_counts(self) -> dict:
        total    = sum(p.numel() for p in self.parameters())
        backbone = sum(p.numel() for p in self.features.parameters())
        head     = sum(p.numel() for p in self.head.parameters())
        return {
            "total":    total,    
            "backbone": backbone, 
            "head":     head,     
        }

    def summary(self) -> None:
        c   = self.get_parameter_counts()
        sep = "=" * 48
        print(f"\n{sep}")
        print(f"  EfficientNet-V2-S  —  WaRP-C ({cfg['num_classes']} classes)")
        print(sep)
        print(f"  Total params    : {c['total']:,}")
        print(f"  Backbone params : {c['backbone']:,}")
        print(f"  Head params     : {c['head']:,}")
        print(sep)

