# ======================== Models/ConvNeXT.py ============================
'''
Date: 28/03/2026
Author: Scott Lewis

ConvNeXt-Base fine-tuned for 28-class WaRP-C classification.

Key design notes:
1. ConvNeXt is a modernised CNN incorporating transformer-style components
(depthwise convs, inverted bottlenecks, GELU, LayerNorm).
2. head_depth controls the depth of the MLP classifier head.
3. Default head_depth=1 matches the canonical ConvNeXt classifier.
4. Deeper heads (head_depth>1) help adapt to domain shift on small datasets.

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

        # load pretrained convnext-Base backbone. 
        # (IMAGENET1K_V1 weights provide strong transfer performance).
        CN_backbone  = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # ConvNeXt backbone: features + global avgpool
        # extract the number of features output by the ConvNeXt classifier.
        # note: convnext-base outputs a 1024-dim feature convnext after avgpool).
        self.in_features = CN_backbone.classifier[2].in_features   # 1024 for convnext-Base

        # extract the backbone. only keep the feature extractor & avgpool.
        # entirely removes the original classifier head.
        # CN_backbone.features == hierarchical convnext blocks
        # CN_backbone.avgpool =- global average pooling
        # output shape == (B, 1024, 1, 1).
        self.CN_backbone = nn.Sequential(CN_backbone.features, CN_backbone.avgpool)

        # freeze backbone parameters for Phase 1 training.
        # note: training only the classifier head initially prevents 'catastrophic forgetting' with small datasets.
        if freeze:
            for parameter in self.CN_backbone.parameters():
                parameter.requires_grad = False

        # builds the classifier head.
        # canonical convnext head == Flatten -> LayerNorm -> Dropout -> Linear(1024→num_classes)
        # this head generalises canonical with head_depth:
        # head_depth = 1 == canonical head 
        # head_depth > 1 == deeper MLP with GELU activations
        # only head_depth = 1 is used for tests
        # note: layernorm stabilises convnext features (convnext uses LN internally).
        layers = [
            nn.Flatten(),                   # (B,1024,1,1) -> (B,1024)
            nn.LayerNorm(self.in_features), # stabilises feature distribution
            nn.Dropout(p=dropout),          # regularisation
        ]

        # adds (head_depth - 1) hidden layers.
        # each hidden layer ==  Linear -> GELU -> dropout
        # increases model capacity for domain adaptation.
        for _ in range(head_depth - 1):
            layers.append(nn.Linear(self.in_features, self.in_features))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout))

        # final projection to 28 classes.
        layers.append(nn.Linear(self.in_features, num_classes))

        # wrap all layers into a single nn.Sequential
        self.classifier = nn.Sequential(*layers)

    # forward pass: 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.CN_backbone(x)          # extract convnext features
        logits   = self.classifier(features)    # pass through mlp classifier head
        return logits

     # unfreeze backbone for Phase 2 fine-tuning.
    def unfreeze(self):
        for parameter in self.CN_backbone.parameters():
            parameter.requires_grad = True

    # freeze backbone for Phase 1 (and ablations).
    def freeze(self):
        for parameter in self.CN_backbone.parameters():
            parameter.requires_grad = False

    # return parameter counts for reporting
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

    # print the model summary (for debugging/logging).
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
