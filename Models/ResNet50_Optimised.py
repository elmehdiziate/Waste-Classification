"""
ResNet50_Optimised (V8)
-----------------------
This version preserves the EXACT architecture and forward pass of the original

ResNet50 model (your baseline), but adds:
1. freeze_backbone() / unfreeze_backbone() for two-phase training
2. differential LR support via get_param_groups()
3. dropout in the classifier head
4. identical backbone module tree (no architectural drift)
5. identical pooling and forward pass
6. safe BatchNorm behaviour when freezing/unfreezing

This ensures performance matches the original (~80%) whilst also enabling advanced training.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet50_Optimised(nn.Module):

    def __init__(self, num_classes: int = 28, dropout: float = 0.4,
                 freeze_backbone: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.dropout     = dropout
        self.model_name  = "ResNet-50 (Optimised V8)"

        # load the ImageNet-pretrained ResNet-50 backbone.
        # note: IMAGENET1K_V2 gives improved transfer performance vs IMAGENET1K_V1)
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # extract the backbone excatly as defined by torchvision:
        # conv1 -> bn1 -> relu -> maxpool -> layer1–4 -> avgpool
        # output shape remains (B, 2048, 1, 1).
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # number of features produced by avgpool (2048 for ResNet-50)
        self.in_features = backbone.fc.in_features  # 2048

        # freeze or unfreeze backbone parameters
        # phase 1: freeze backbone -> train classifier only
        # phase 2: unfreeze backbone -> full fine-tuning
        # this is because batchnorm layers require special handling:
        # .eval() stops BN from updating running stats when frozen
        # .train() re-enables BN adaptation when unfreezing        
        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

        # classification head (same as baseline)
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # flatten: (B, 2048, 1, 1) -> (B, 2048)
            nn.Linear(self.in_features, 512),   # linear(2048->512): bottleneck to reduce overfitting
            nn.ReLU(inplace=True),              # ReLU for non-linearity
            nn.Dropout(p=dropout),              # dropout for regularisation
            nn.Linear(512, num_classes)         # linear(512->num_classes(28)) == final logits
        )

    # forward pass: extract features using the backbone & Pass through classifier head (same as baseline)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)             # visual features extracted (B, 2048, 1, 1)
        logits   = self.classifier(features)    # mapping features to class predictions (B, num_classes)
        return logits

    # freeze backbone parameters & set BN layers to eval mode.
    # prevents batchnorm from updating running mean/var during Phase 1.
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()  # freeze BN stats

    # unfreeze backbone parameters & set BN layers to train mode.
    # allows BN to adapt to warp-c during Phase 2 fine-tuning.
    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()  # BN adapts to WaRP-C

    # support for differential learning rate:
    # backbone_lr: small LR for pretrained backbone
    # head_lr: larger LR for randomly initialised classifier
    # (used by optimisers like AdamW)
    def get_param_groups(self, head_lr: float, backbone_lr: float):
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.classifier.parameters(), "lr": head_lr},
        ]

    # return parameter counts for reporting
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

    # print the model summary (for debugging/logging).
    def __repr__(self) -> str:
        c = self.get_parameter_counts()
        return (
            f"{self.model_name}\n"
            f"  num_classes     : {self.num_classes}\n"
            f"  dropout         : {self.dropout}\n"
            f"  total params    : {c['total_M']}M\n"
            f"  trainable params: {c['trainable_M']}M\n"
            f"  backbone        : ResNet-50 (ImageNet-1k V2)\n"
            f"  head            : Linear({self.in_features}->512) -> ReLU -> Dropout -> Linear(512->{self.num_classes})"
        )
