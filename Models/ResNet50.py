# ======================== resnet50_model.py ============================
'''
Date: 10/03/2026
Author: Umme-Yusrah Sumtally

Date: 25/03/2026
added forward_features function and updated forward function
'''

import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):

    def __init__(self, num_classes: int = 28,dropout: float = 0.4,freeze: bool = True):
        super(ResNet50, self).__init__()

        self.num_classes = num_classes
        self.dropout     = dropout
        self.model_name  = "ResNet-50"

        #Load pretrained backbone
        RN_backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        #Freeze backbone if Phase 1 
        if freeze:
            for parameter in RN_backbone.parameters():
                parameter.requires_grad = False

        #remove original classification head that output 1000 classes scores
        self.RN_backbone = nn.Sequential(*list(RN_backbone.children())[:-1])

	#input features from backbone
        in_features = RN_backbone.fc.in_features 

        # adding a new classification head that outputs only 28 classes scores for Warp-C dataset
        self.classifier = nn.Sequential(
            nn.Flatten(),                          #convert 3D tensor to flat vector
            nn.Dropout(p=dropout),		   #regularization to prevent overfitting and reliance on certain features only
            nn.Linear(in_features, 512),	   #main compression layer that learns combinations of features for the classification task
            nn.ReLU(inplace=True),		   #add non linearity and converts all negative values to zero.
            nn.Dropout(p=dropout / 2),			
            nn.Linear(512, num_classes)            #final prediction layer
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor: #extract features before classification
        return self.RN_backbone(x)  							# a separate function is used to ensure reusability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)   
        logits   = self.classifier(features)  # mapping features to class predictions
        return logits

    def unfreeze(self):
        for parameter in self.RN_backbone.parameters():
            parameter.requires_grad = True

    def freeze(self):
        for parameter in self.RN_backbone.parameters():
            parameter.requires_grad = False

    def get_parameter_counts(self) -> dict:     #for efficiency comparison table
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
            f"  head            : Linear(2048→512) → Linear(512→{self.num_classes})"
        )
