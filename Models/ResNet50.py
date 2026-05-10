# ======================== Models/ResNet50.py ============================
'''
Date: 10/03/2026
Author: Scott Lewis

ResNet50 fine-tuned for 28-class Warp-C classification.

Design rationale:
1. Use ImageNet-pretrained ResNet-50 as backbone (strong transfer learning baseline)
2. Remove the final FC layer and replace with a lightweight bottleneck head
3. Bottleneck (2048->512) stabilises transfer to small dataset (8,823 images)
4. ReLU after Linear (correct placement)
5. Dropout after activation (standard for regularisation)

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

        # load the ImageNet-pretrained ResNet-50 backbone.
        # note: IMAGENET1K_V2 gives improved transfer performance vs IMAGENET1K_V1)
        RN_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # extract the number of features output by the final pooling layer.
        # note: ResNet-50 outputs a 2048-dim feature vector after avgpool
        self.in_features = RN_backbone.fc.in_features   # 2048 for ResNet50

        # remove the final fully-connected classification layer.
        # conv1 -> bn1 -> relu -> maxpool -> layer1–4 -> avgpool
        # output shape == (B, 2048, 1, 1).
        self.RN_backbone = nn.Sequential(*list(RN_backbone.children())[:-1])

        # freeze backbone parameters for Phase 1 training.
        # note: training only the classifier head initially prevents 'catastrophic forgetting' with small datasets.
        if freeze:
            for parameter in self.RN_backbone.parameters():
                parameter.requires_grad = False

        # classification head: intentionally lightweight to avoid overfitting.
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # flatten: (B, 2048, 1, 1) -> (B, 2048)
            nn.Linear(self.in_features, 512),   # linear(2048->512): bottleneck to reduce overfitting
            nn.ReLU(inplace=True),              # ReLU for non-linearity
            nn.Dropout(p=dropout),              # dropout for regularisation
            nn.Linear(512, num_classes)         # linear(512->num_classes(28)) == final logits
        )

    # forward pass: Extract features using the backbone & Pass through classifier head
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.RN_backbone(x)        # visual features extracted (B, 2048, 1, 1)
        logits   = self.classifier(features)  # mapping features to class predictions (B, num_classes)
        return logits

     # unfreeze backbone for Phase 2 fine-tuning.
    def unfreeze(self):
        for parameter in self.RN_backbone.parameters():
            parameter.requires_grad = True

    # freeze backbone for Phase 1 (and ablations).
    def freeze(self):
        for parameter in self.RN_backbone.parameters():
            parameter.requires_grad = False

    # return parameter counts for reporting
    def get_parameter_counts(self) -> dict:
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

    # print the model summary (for debugging/logging).
    def __repr__(self) -> str:
        counts = self.get_parameter_counts()
        return (
            f"{self.model_name}\n"
            f"  num_classes     : {self.num_classes}\n"
            f"  dropout         : {self.dropout}\n"
            f"  total params    : {counts['total_M']}M\n"
            f"  trainable params: {counts['trainable_M']}M\n"
            f"  backbone        : ResNet-50 (ImageNet-1k V2)\n"
            f"  head            : Linear({self.in_features}->512) -> ReLU -> Dropout -> Linear(512->{self.num_classes})"
        )