"""
Mohamed Fahmi Ahmed

EfficientNet-V2-M — WaRP-C Waste Classification

"""

import torch
import torch.nn as nn
from torchvision import models

class EfficientNetV2M(nn.Module):

    def __init__(self, num_classes: int = 28, dropout: float = 0.3):
        super().__init__()

        n_classes = num_classes
        drop_rate = dropout
        self.num_classes = num_classes

        # EfficientNet-V2-S pretrained on ImageNet-1K (83.9% top-1).
        backbone = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
        )

        # Extract convolutional stages
        # backbone.features : all MBConv / Fused-MBConv stages
        #                     input  (B, 3, 224, 224)
        #                     output (B, 1280, 7, 7)
        #
        # backbone.avgpool  : Global Average Pooling
        #                     collapses (B, 1280, 7, 7) -> (B, 1280, 1, 1)
        #                     stops us from flattening 62 720 values into the classifier,
        #                     otherwise we'd definitely overfit on ~9000 images
        #
        # backbone.classifier : original head for ImageNet (1000 classes) toss it
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        # new classification head 
        # we grab in_features from the backbone instead of hardcoding 1280
        # that way it won't break if we swap the model later
        #
        # flatten : (B, 1280, 1, 1) -> (B, 1280)
        # dropout : kills 30% of neurons so the model doesn't get lazy
        # linear  : 28 classes (logits). no softmax bc CrossEntropyLoss does it
        in_features = backbone.classifier[-1].in_features

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(in_features, n_classes),
        )

        # kaiming normal init on our new linear layer
        # the backbone weights are already solid, but the head is random
        # this kaiming trick stops the gradients from exploding at the start
        nn.init.kaiming_normal_(self.head[-1].weight, nonlinearity="relu")
        nn.init.zeros_(self.head[-1].bias)

    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, 3, 224, 224)  normalised image batch
        return : (B, 28)           raw logits, no softmax
        """
        x = self.features(x)     # (B, 3, 224, 224) -> (B, 1280, 7, 7)
        x = self.avgpool(x)      # (B, 1280, 7, 7)  -> (B, 1280, 1, 1)
        x = self.head(x)         # (B, 1280, 1, 1)  -> (B, 28)
        return x

    # Two-phase training  
    def freeze(self) -> None:
        #phase 1 — freeze the backbone and only train the head.
        #the head has random weights, so we freeze the rest to stop bad gradients from ruining the ImageNet features at the start.
        #it drops trainable params from ~21M down to like ~35K

        for p in self.features.parameters():
            p.requires_grad = False
        print("[EfficientNetV2S] backbone frozen  —  Phase 1 (head only)")

    def unfreeze(self) -> None:
        #phase 2 — unfreeze everything for full fine-tuning.
        #run this only after phase 1 when the head isn't randomly initialized anymore.
        #need to use different learning rates for each part (check param_groups)

        for p in self.features.parameters():
            p.requires_grad = True
        print("[EfficientNetV2S] backbone unfrozen  —  Phase 2 (full fine-tuning)")

    def param_groups(self, lr_head: float, lr_backbone: float) -> list:

        #two AdamW groups with different learning rates for phase 2.
        #lr_backbone << lr_head cuz the backbone already knows from ImageNet
        #sually: lr_head=1e-3, lr_backbone=1e-4 

        return [
            {"params": self.features.parameters(), "lr": lr_backbone},
            {"params": self.head.parameters(),     "lr": lr_head},
        ]

    def parameter_counts(self) -> dict:
        #for efficiency comparison table in the report
            
        total    = sum(p.numel() for p in self.parameters())
        backbone = sum(p.numel() for p in self.features.parameters())
        head     = sum(p.numel() for p in self.head.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total":     total,
            "backbone":  backbone,
            "head":      head,
            "trainable": trainable,
            "frozen":    total - trainable,
        }

    def summary(self) -> None:
        c     = self.parameter_counts()
        phase = "Phase 1 (frozen)" if c["frozen"] > 0 else "Phase 2 (unfrozen)"
        sep   = "=" * 50
        print(f"\n{sep}")
        print(f"  EfficientNet-V2-S  —  WaRP-C ({self.num_classes} classes)")
        print(f"  {phase}")
        print(sep)
        print(f"  Total params     : {c['total']:,}")
        print(f"  Backbone params  : {c['backbone']:,}")
        print(f"  Head params      : {c['head']:,}")
        print(f"  Trainable params : {c['trainable']:,}")
        print(f"  Frozen params    : {c['frozen']:,}")
        print(sep)


