"""
El Mehdi Ziate
Waste Classification with EfficientViT-B1
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError('timm is required. Install with: pip install timm')


class EfficientViT_WaRP(nn.Module):
    """
    EfficientViT-B1 fine-tuned for WaRP-C 28-class waste classification.

    EfficientViT uses cascaded group attention (CGA): 
    each attention head processes a different split of the feature channels,
    reducing redundancy and improving diversity at linear computational cost.

    Architecture: EfficientViT-B1 backbone (timm) → GlobalAvgPool → Linear(1600→28)

    Two-phase fine-tuning:
      Phase 1: freeze backbone, train head only
      Phase 2: unfreeze all, differential LR: backbone_lr = head_lr / 10
    """

    MODEL_NAME = 'efficientvit_b1'

    def __init__(self, num_classes: int = 28, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            self.MODEL_NAME,
            pretrained  = pretrained,
            num_classes = 0,
        )

        # Detect true output dim empirically 
        # EfficientViT-B is 1600, but let's confirm with a dummy forward pass
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, 224, 224)
            num_features = self.backbone(_dummy).shape[-1]

        self.head = nn.Linear(num_features, num_classes)

        # trunc_normal_(std=0.02) : that s what I have foound as a standard for transformer heads
        # Reference: Dosovitskiy et al. (2021) ViT paper; timm default: you can check this reference
        # Kaiming init assumes ReLU activation
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        if pretrained:
            print(f'[EfficientViT_WaRP] Loaded pretrained {self.MODEL_NAME}')
            print(f'  Backbone features : {num_features}')
            print(f'  Head              : Linear({num_features} → {num_classes})')
            print(f'  Total parameters  : {self.count_all()["total"]:,}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        print('[EfficientViT_WaRP] Backbone FROZEN — training head only')

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        print('[EfficientViT_WaRP] Backbone UNFROZEN — full fine-tuning')

    def get_param_groups(self, head_lr: float = 1e-4, backbone_lr: float = 1e-5):
        """Differential LR parameter groups."""
        return [
            {'params': self.backbone.parameters(), 'lr': backbone_lr},
            {'params': self.head.parameters(),     'lr': head_lr},
        ]

    def count_all(self) -> dict:
        """Total, trainable, and frozen parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {'trainable': trainable, 'frozen': frozen, 'total': trainable + frozen}