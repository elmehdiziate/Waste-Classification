'''
Author: Umme-Yusrah Sumtally

Date: 28/03/2026
Description: Basic working version of MobileViT

Date: 31/03/2026
Description: added 2 functions (parameter counting + repr)

'''
#Importing libraries
import torch
import torch.nn as nn
from transformers import MobileViTForImageClassification


class MobileViT(nn.Module):

    def __init__(self, number_of_classes: int = 28,
                 freeze: bool = True):
        super(MobileViT, self).__init__()

        self.number_of_classes = number_of_classes
        self.MobileViT_model = "MobileViT-Small"  #small model that uses 5.6M parameters and have a 98% accuracy on waste classification

        #loading pre trained ViT from Hugging face
        self.pretrained_backbone = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")

        #Input features of the classifier
        input_features = self.pretrained_backbone.classifier.in_features

        #New classification head for 28 waste classes
        self.classifier = nn.Linear(input_features, number_of_classes)


        # replacing the classifcation head from the pretrained backbon of Huggingface to adapt model to the waste classification of 28 classes only
        self.pretrained_backbone.classifier = self.MobileViT_classifier

        #Two step  training strategy
        # Freeze backbone for Phase 1 training
        if freeze:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.pretrained_backbone(x)
        return outputs.logits


    def freeze_backbone(self):
        for parameter in self.pretrained_backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.MobileViT_classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self):
        for parameter in self.pretrained_backbone.parameters():
            parameter.requires_grad = True

    #counting trainable parameters
    def model_parameter_counts(self) -> dict:
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in self.parameters()
                               if parameter.requires_grad)

        return {
            "model": self.MobileViT_model,
            "total_params": total_parameters,
            "trainable_params": trainable_parameters,
            "total_M": round(total_parameters / 1e6, 2),
            "trainable_M": round(trainable_parameters / 1e6, 2)
        }

    #display architecture of model
    def __repr__(self) -> str:
        counts = self.model_parameter_counts()
        return (
            f"{self.MobileViT_model}\n"
            f"  num_classes     : {self.number_of_classes}\n"
            f"  dropout         : {self.dropout}\n"
            f"  total params    : {counts['total_M']}M\n"
            f"  trainable params: {counts['trainable_M']}M\n"
            f"  architecture    : CNN → Transformer → CNN\n"
            f"  pretrained      : ImageNet-1K\n"
            f"  waste accuracy  : 98.01% (Yuan et al., 2023)"
        )

