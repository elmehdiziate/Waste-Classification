"""
El Mehdi Ziate
-----------------
WHAT IS SWIN TRANSFORMER:
CNN slides a 3×3 filter across the image, looking at tiny patches.
ViT (plain Vision Transformer) makes every patch attend to every other
patch across the whole image which means that it is very powerful but extremely expensive.

Swin does something in between:
  1. Divide the image into non-overlapping 7×7 WINDOWS of patches.
  2. Each patch only attends to other patches INSIDE its own window.
     → cost goes from quadratic to linear with image size.
  3. Every other layer, SHIFT the windows by half their size.
     → patches that were in different windows can now communicate.
  4. After each stage, MERGE adjacent patches (like downsampling in CNN).
     → builds a hierarchy: fine details → coarse structure, just like ResNet.

The name SWIN = Shifted WINdow.

WHY SWIN FOR WARP-C
---------------------
WaRP-C has 28 visually similar classes (17 bottle sub-types).
A CNN's fixed 3×3 kernel struggles to capture the global shape of
"bottle-transp-full vs bottle-transp" whihc are two classes that differ mainly
in whether the bottle is compressed or inflated.

Swin's shifted-window attention can relate distant parts of the image
to each other within a few layers. It can compare "the cap region" with
"the body region" and "the base region" simultaneously — which is exactly
what you need to distinguish subtle bottle sub-types.

ARCHITECTURE — SWIN-TINY
--------------------------
  Input: (B, 3, 224, 224)

  Stage 0 — Patch Partition + Linear Embedding
    Split image into 4×4 pixel patches → 56×56 = 3136 tokens
    Each token = 48-dim vector (4×4×3 pixels flattened)
    Linear projection to 96-dim  →  (B, 56×56, 96)

  Stage 1 — 2 Swin Transformer Blocks
    Window attention (7×7 windows) + Shifted window attention
    Output: (B, 56×56, 96)

  Patch Merging 1 — halve spatial, double channels
    Output: (B, 28×28, 192)

  Stage 2 — 2 Swin Transformer Blocks
    Output: (B, 28×28, 192)

  Patch Merging 2
    Output: (B, 14×14, 384)

  Stage 3 — 6 Swin Transformer Blocks  ← most of the learning happens here
    Output: (B, 14×14, 384)

  Patch Merging 3
    Output: (B, 7×7, 768)

  Stage 4 — 2 Swin Transformer Blocks
    Output: (B, 7×7, 768)

  Global Average Pooling  →  (B, 768)
  Head: Linear(768 → 28)  →  (B, 28) logits

  Total parameters: ~28 million
  Compare: CNN baseline ~2.4M, ResNet-50 ~25M

FINE-TUNING STRATEGY — TWO PHASES
------------------------------------
Swin is pretrained on ImageNet-1K (80.9% top-1) or ImageNet-22K (83.5%).
We use `swin_tiny_patch4_window7_224` pretrained on ImageNet-1K.

Phase 1 — HEAD ONLY:
  Freeze the entire backbone. Only train the new 768→28 classifier head.
  Why: the pretrained weights are tuned for ImageNet. The head weights
  are random. If you update both at once, large gradients from the random
  head damage the carefully pretrained backbone weights in early epochs.

Phase 2 — FULL FINE-TUNING:
  Unfreeze everything. Train backbone + head together with a low LR.
  Why: now the head is stable, so smaller gradients from the backbone
  can safely adapt the pretrained features to WaRP-C's domain.

  Backbone LR = head LR / 10 (differential learning rates)
  Why: backbone features are already good → small nudge.
  Head features are new → larger update.

Reference: Liu et al. (2021) Swin Transformer paper, fine-tuning recipe.
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required. Install with: pip install timm"
    )


class SwinTransformerWaRP(nn.Module):
    """
    Swin-Tiny fine-tuned for WaRP-C 28-class classification.

    Uses timm's pretrained weights — no need to implement Swin from scratch.
    timm (PyTorch Image Models) is the standard library for pretrained
    vision models. It contains 600+ architectures including all Swin variants.

    Parameters
    ----------
    num_classes  : number of output classes (28 for WaRP-C)
    pretrained   : load ImageNet-1K pretrained weights (strongly recommended)
    drop_rate    : dropout on the classification head (0.0 = no dropout)
    drop_path_rate: stochastic depth rate — randomly drops entire transformer
                    blocks during training (regularisation). Default 0.2
                    follows the Swin paper's recommendation for Swin-Tiny.

    Attributes
    ----------
    backbone : the full Swin-Tiny feature extractor (768-dim output)
    head     : Linear(768 → num_classes) — our new classification layer

    Usage
    -----
    model = SwinTransformerWaRP(num_classes=28, pretrained=True)
    logits = model(images)   # (B, 28) — raw scores before softmax
    """

    # timm model name — Swin-Tiny pretrained on ImageNet-1K
    # swin_tiny = 28M params, window_size=7, patch_size=4, input=224
    # "patch4" = each patch covers 4×4 pixels
    # "window7" = attention windows are 7×7 patches = 28×28 pixels
    MODEL_NAME = "swin_tiny_patch4_window7_224"

    def __init__(
        self,
        num_classes:    int   = 28,
        pretrained:     bool  = True,
        drop_rate:      float = 0.0,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()

        self.num_classes = num_classes

        # ── Load pretrained Swin-Tiny backbone via timm ──────────────────
        # num_classes=0 removes the original ImageNet head (1000 classes).
        # We will attach our own 28-class head below.
        # drop_path_rate=0.2 is the stochastic depth from the Swin paper.
        self.backbone = timm.create_model(
            self.MODEL_NAME,
            pretrained      = pretrained,
            num_classes     = 0,        # remove 1000-class ImageNet head
            drop_rate       = drop_rate,
            drop_path_rate  = drop_path_rate,
        )

        # ── Find the backbone's output feature dimension ─────────────────
        # Swin-Tiny outputs 768-dim features after Global Average Pooling.
        # We retrieve this automatically rather than hardcoding it.
        num_features = self.backbone.num_features   # 768 for Swin-Tiny

        # ── Our new classification head ───────────────────────────────────
        # Simple Linear layer: 768 features → 28 class logits.
        # No softmax here — CrossEntropyLoss applies it internally.
        self.head = nn.Linear(num_features, num_classes)

        # ── Initialise the new head properly ──────────────────────────────
        # The backbone has pretrained weights. Our head is brand new.
        # Kaiming init ensures gradients flow well in the first epochs.
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="relu")
        nn.init.zeros_(self.head.bias)

        if pretrained:
            print(f"[SwinTransformerWaRP] Loaded pretrained {self.MODEL_NAME}")
            print(f"  Backbone features : {num_features}")
            print(f"  Head              : Linear({num_features} → {num_classes})")
            print(f"  Parameters        : {self.count_parameters():,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, 3, 224, 224) normalised tensor

        Returns
        -------
        logits : (B, num_classes) — raw scores, no softmax
        """
        # backbone does: patch partition → 4 stages → global avg pool
        # output shape: (B, 768)
        features = self.backbone(x)

        # head maps to class scores
        # output shape: (B, 28)
        return self.head(features)

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters — only head will be trained.

        Call this at the START of training (Phase 1).
        This protects the pretrained ImageNet features from being
        damaged by large gradients from the randomly initialised head.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[SwinTransformerWaRP] Backbone FROZEN — training head only")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all parameters — train backbone + head together.

        Call this after Phase 1 warm-up (Phase 2 full fine-tuning).
        Use a LOWER learning rate for the backbone than the head.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[SwinTransformerWaRP] Backbone UNFROZEN — full fine-tuning")

    def get_param_groups(
        self,
        head_lr:     float = 1e-4,
        backbone_lr: float = 1e-5,
    ) -> list[dict]:
        """
        Return two parameter groups with DIFFERENT learning rates.

        Why differential LR?
        ---------------------
        The backbone already has good features from ImageNet.
        It only needs a small nudge to adapt to WaRP-C → low LR (1e-5).
        The head is brand new and needs to learn from scratch → higher LR (1e-4).

        If you use the same LR for both:
          - High LR on backbone → destroys pretrained features
          - Low LR on head → head learns too slowly

        Usage:
            param_groups = model.get_param_groups(head_lr=1e-4, backbone_lr=1e-5)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)

        Parameters
        ----------
        head_lr     : learning rate for the new classification head
        backbone_lr : learning rate for the pretrained Swin backbone
                      typically head_lr / 10
        """
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(),     "lr": head_lr},
        ]

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_trainable(self) -> dict:
        """Return counts of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}