'''
Author: Sayed Omar Aabid

Vision Transformer Base (ViT-B/16) for industrial waste classification.
Pretrained on ImageNet-1k (google/vit-base-patch16-224 via torchvision).

Architecture:
  - Patch size  : 16 × 16
  - Resolution  : 224 × 224  →  196 patches + 1 [CLS] token
  - Hidden dim  : 768
  - Depth       : 12 transformer encoder blocks
  - Heads       : 12 attention heads
  - Parameters  : ~86 M total

Fine-tuning strategy (two-phase):
  Phase 1 — freeze backbone, train classification head only  (fast warm-up)
  Phase 2 — unfreeze all layers, end-to-end fine-tuning     (full adaptation)

Head design rationale (aligned with Dosovitskiy et al., 2021 & Steiner et al., 2022):
  The original ViT paper specifies a *single linear layer* at fine-tuning time.
  Steiner et al. ("How to Train Your ViT?") further show that adding a hidden
  layer in the head empirically does not improve accuracy and often causes
  optimisation instabilities — so we follow the single-layer recommendation.
  A lightweight MLP head is retained as an optional ablation variant below.
'''

import torch
import torch.nn as nn
from torchvision import models


class ViT_B(nn.Module):

    def __init__(self, num_classes: int = 28, dropout: float = 0.1, freeze: bool = True):
        super(ViT_B, self).__init__()

        self.num_classes = num_classes
        self.dropout     = dropout
        self.model_name  = "ViT-B/16"

        # ── Backbone ──────────────────────────────────────────────────────────
        # Load pretrained ViT-B/16 with ImageNet-1k weights.
        # ViT learns global context via self-attention rather than local
        # convolution, making it well-suited to recognising shape and texture
        # cues in waste images.
        #
        # Note: ImageNet-21k pretrained weights (e.g. via HuggingFace
        # google/vit-base-patch16-224-in21k) offer stronger transfer for
        # out-of-domain datasets such as WaRP-C and are worth considering
        # if compute allows.
        vit_backbone = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1
        )

        # ── Phase 1 freeze ────────────────────────────────────────────────────
        # Freezing prevents catastrophic forgetting of ImageNet features while
        # the new classification head stabilises during warm-up.
        if freeze:
            for parameter in vit_backbone.parameters():
                parameter.requires_grad = False

        # ── Replace head ──────────────────────────────────────────────────────
        # Remove the original projection (768 → 1000 classes) and expose the
        # raw [CLS] token embedding for our custom head.
        in_features = vit_backbone.heads.head.in_features   # 768
        vit_backbone.heads = nn.Identity()                  # strip original head

        self.vit_backbone = vit_backbone

        # ── Classification head ───────────────────────────────────────────────
        # Following Dosovitskiy et al. (2021) and Steiner et al. (2022), we use
        # a *single linear layer* at fine-tuning time.  A hidden MLP layer is
        # not recommended: it does not consistently improve accuracy and can
        # cause optimisation instabilities (Steiner et al., 2022).
        #
        # Dropout is set to 0.1 (down from 0.3) — aggressive dropout on the
        # [CLS] embedding risks underfitting during Phase 1 when only the head
        # is trained.  0.1 provides light regularisation without stifling
        # learning.
        #
        # ── Ablation variant (MLP head) ───────────────────────────────────────
        # If you want to compare a two-layer MLP head against this baseline,
        # uncomment the block below and comment out the single-layer head.
        #
        #   self.classifier = nn.Sequential(
        #       nn.Dropout(p=dropout),
        #       nn.Linear(in_features, 512),
        #       nn.GELU(),
        #       nn.Dropout(p=dropout / 2),
        #       nn.Linear(512, num_classes),
        #   )
        #
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),          # light regularisation on [CLS] embedding
            nn.Linear(in_features, num_classes),   # 768 → 28  (single linear layer)
        )

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit_backbone(x)     # [CLS] token embedding — shape (B, 768)
        logits   = self.classifier(features)
        return logits

    # ── Phase control helpers ──────────────────────────────────────────────────
    def unfreeze(self):
        '''Unfreeze entire backbone for Phase 2 end-to-end fine-tuning.
        Use a small learning rate (e.g. 1e-5) to avoid disrupting pretrained
        weights — catastrophic forgetting is a real risk without lr decay.'''
        for parameter in self.vit_backbone.parameters():
            parameter.requires_grad = True

    def freeze(self):
        '''Re-freeze backbone (e.g. to revert to Phase 1 for ablations).'''
        for parameter in self.vit_backbone.parameters():
            parameter.requires_grad = False

    # ── Utilities ──────────────────────────────────────────────────────────────
    def get_parameter_counts(self) -> dict:
        '''Return total and trainable parameter counts (for comparison tables).'''
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

    def __repr__(self) -> str:
        counts = self.get_parameter_counts()
        return (
            f"{self.model_name}\n"
            f"  num_classes     : {self.num_classes}\n"
            f"  dropout         : {self.dropout}\n"
            f"  total params    : {counts['total_M']}M\n"
            f"  trainable params: {counts['trainable_M']}M\n"
            f"  backbone        : ViT-B/16 (ImageNet-1k V1)\n"
            f"  head            : Dropout({self.dropout}) → Linear(768→{self.num_classes})"
        )
