import torch
import torch.nn as nn
import torch.nn.functional as F

#--- based on Lin et al. (2017) Focal Loss for Dense Object Detection
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha      # scalar or tensor
        self.weight = weight    # class weights
        self.reduction = reduction

    def forward(self, logits, targets):
        # Standard CE loss per sample (no reduction)
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none"
        )

        # pt = probability of the correct class
        pt = torch.exp(-ce)

        # focal scaling
        focal = (1 - pt) ** self.gamma * ce

        # apply alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # per-class alpha
                alpha_factor = self.alpha[targets]
            else:
                # scalar alpha
                alpha_factor = self.alpha

            focal = alpha_factor * focal

        # reduction
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal
