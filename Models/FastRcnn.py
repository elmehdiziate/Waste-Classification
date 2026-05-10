"""
El Mehdi Ziate
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNWaRP(nn.Module):
    """
    Faster R-CNN fine-tuned for WaRP-D 28-class object detection.

    Uses torchvision's pretrained Faster R-CNN (COCO weights) and
    replaces only the classification head for 28 WaRP classes.

    Parameters: 
    num_classes : int
        Number of WaRP-D classes + 1 background. Default 29.
    pretrained  : bool
        Load COCO pretrained weights. Always True unless debugging.
    score_thresh : float
        Minimum confidence score to keep a detection at inference.
    nms_thresh   : float
        IoU threshold for non-maximum suppression.

    Usage: 
    model = FasterRCNNWaRP(num_classes=29)
    model.train()
    loss_dict = model(images, targets)   # training
    model.eval()
    predictions = model(images)          # inference
    """

    NUM_CLASSES = 29   # 28 WaRP classes + 1 background

    def __init__(
        self,
        num_classes:  int   = NUM_CLASSES,
        pretrained:   bool  = True,
        score_thresh: float = 0.05,
        nms_thresh:   float = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Load COCO pretrained Faster R-CNN with ResNet-50 + FPN backbone.
        self.model = fasterrcnn_resnet50_fpn(
            weights              = "DEFAULT" if pretrained else None,
            box_score_thresh     = score_thresh,
            box_nms_thresh       = nms_thresh,
        )

        # we replace the classification head.
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        if pretrained:
            print(f"[FasterRCNN] ResNet-50 FPN : COCO pretrained")
            print(f"  Classes     : {num_classes} (28 WaRP + 1 background)")
            print(f"  Parameters  : {self.count_parameters():,}")

    def forward(self, images, targets=None):
        """
        Forward pass.

        Training:   pass both images and targets -> returns loss dict
        Inference:  pass images only              -> returns predictions

        Parameters
        ----------
        images  : list of (C, H, W) tensors — variable size is fine
        targets : list of dicts with keys:
                    "boxes"  : (N, 4) float tensor  [x1, y1, x2, y2]
                    "labels" : (N,)   int64 tensor   class indices 1..28

        Returns
        -------
        Training : dict {"loss_classifier", "loss_box_reg",
                         "loss_objectness", "loss_rpn_box_reg"}
        Inference: list of dicts {"boxes", "labels", "scores"}
        """
        return self.model(images, targets)

    def freeze_backbone(self):
        """Freeze ResNet-50 + FPN. Train RPN and RoI heads only."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        print("[FasterRCNN] Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze everything for full fine-tuning."""
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        print("[FasterRCNN] Backbone unfrozen")

    def get_param_groups(self, backbone_lr=1e-5, head_lr=1e-4):
        """
        Differential learning rates.
        Backbone already has good features -> small LR.
        Heads are newly adapted to WaRP -> larger LR.
        """
        backbone_params = list(self.model.backbone.parameters())
        head_params = (
            list(self.model.rpn.parameters()) +
            list(self.model.roi_heads.parameters())
        )
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params,     "lr": head_lr},
        ]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)