"""
El Mehdi Ziate, 2026-03-17
"""

import json
import shutil
import csv
from pathlib import Path
from collections import defaultdict
from typing import Optional, Callable
 
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
 
 
# ── Constants derived from EDA ─────────────────────────────────────────────
 
IMG_SIZE = 224
 
# WaRP-C pixel statistics — EDA computed, mean_diff from ImageNet = 0.105 > 0.05
# We use WaRP-C stats, NOT ImageNet stats
WARP_MEAN = [0.337, 0.344, 0.350]
WARP_STD  = [0.216, 0.209, 0.218]
 
# ImageNet stats — kept for reference only
# All pretrained models (ResNet, ViT, Swin, EfficientNet) were originally
# trained on ImageNet using these exact values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
 
# minority threshold: class with < 30% of global mean (315) = minority
MINORITY_THRESHOLD = 0.30
 
VALID_EXTS = {".jpg", ".jpeg", ".png"}
 
 
 
class PadToSquare:
    """
    Pad a PIL image to a square BEFORE resizing.
 
    WHY: EDA showed aspect ratios 0.2-5.2. Naive Resize(224,224) squashes
    non-square images — a tall narrow bottle becomes a distorted square,
    destroying the shape information the model needs.
 
    WHY REFLECT (not zero): Zero-padding creates black borders. A CNN can
    learn "black border = rare class" as a shortcut. Reflect padding mirrors
    edge pixels so there is no artificial border signal.
 
    Example:
        Original:   80 × 200 px  (portrait, ratio = 0.4)
        After pad:  200 × 200 px (reflect-padded on left and right)
        After Resize: 224 × 224 px (no distortion)
 
    Applied to ALL models — this is preprocessing, not augmentation.
    """
 
    def __init__(self, padding_mode: str = "reflect"):
        self.padding_mode = padding_mode
 
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h  = img.size
        diff  = abs(w - h)
        pad_a = diff // 2
        pad_b = diff - pad_a
 
        if w < h:
            padding = (pad_a, 0, pad_b, 0)
        elif h < w:
            padding = (0, pad_a, 0, pad_b)
        else:
            return img
 
        return TF.pad(img, padding, padding_mode=self.padding_mode)
 
    def __repr__(self):
        return f"PadToSquare(mode={self.padding_mode!r})"
 
 
 
class WaRPDataset(Dataset):
    """
    PyTorch Dataset for WaRP-C.
 
    Automatically routes each sample to the correct transform pipeline:
      - Minority classes (bottle-oil-full 24, detergent-box 66,
        bottle-blue5l-full 89) → minority_transform (stronger aug)
      - All other classes → transform (standard aug for that model)
 
    This routing is invisible to the training loop — it happens inside
    __getitem__() automatically based on the class name.
    """
 
    def __init__(
        self,
        root:               str | Path,
        transform:          Optional[Callable] = None,
        minority_transform: Optional[Callable] = None,
        stats_file:         str | Path = "Dataset/dataset_stats.json",
    ):
        self.root               = Path(root)
        self.transform          = transform
        self.minority_transform = minority_transform
        self._minority_classes: set[str]             = set()
        self._class_weights:    Optional[torch.Tensor] = None
 
        if not self.root.exists():
            raise FileNotFoundError(
                f"Processed split not found: {self.root}\n"
                "→ Run WaRPPreprocessor().prepare() first."
            )
 
        # sorted alphabetically → deterministic label integers across all machines
        self.classes:      list[str]      = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}
 
        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            label = self.class_to_idx[cls]
            for p in sorted((self.root / cls).iterdir()):
                if p.suffix.lower() in VALID_EXTS:
                    self.samples.append((p, label))
 
        self._load_stats(Path(stats_file))
 
    def _load_stats(self, stats_file: Path) -> None:
        if not stats_file.exists():
            return
        with open(stats_file) as f:
            stats = json.load(f)
        names  = stats["class_distribution"]["class_names"]
        counts = stats["class_distribution"]["train_counts"]
        mean   = sum(counts) / len(counts)
        for cls, cnt in zip(names, counts):
            if cnt < MINORITY_THRESHOLD * mean:
                self._minority_classes.add(cls)
        count_map = dict(zip(names, counts))
        raw_w     = [1.0 / count_map.get(c, 1) for c in self.classes]
        norm      = len(raw_w) / sum(raw_w)
        self._class_weights = torch.tensor(
            [w * norm for w in raw_w], dtype=torch.float32
        )
 
    @property
    def class_weights(self) -> Optional[torch.Tensor]:
        """Inverse-frequency weights for CrossEntropyLoss(weight=...)."""
        return self._class_weights
 
    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        if self._class_weights is None:
            return torch.ones(len(self.samples))
        return torch.tensor(
            [self._class_weights[label].item() for _, label in self.samples],
            dtype=torch.float32,
        )
 
    def __len__(self) -> int:
        return len(self.samples)
 
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img             = Image.open(img_path).convert("RGB")
        cls_name        = self.classes[label]
 
        if cls_name in self._minority_classes and self.minority_transform is not None:
            tfm = self.minority_transform
        else:
            tfm = self.transform
 
        if tfm is not None:
            img = tfm(img)
        return img, label
 
 
 
class WaRPPreprocessor:
    """
    All-in-one preprocessing and data pipeline for WaRP-C.
 
    Usage (same 8 lines for every teammate, just change model_type):
    -----------------------------------------------------------------
    pp = WaRPPreprocessor()
    pp.prepare()                                    # run ONCE
    train_loader, test_loader = pp.get_loaders(model_type='swin')
    criterion = nn.CrossEntropyLoss(
                    weight=pp.get_class_weights(device='cuda'))
    """
 
    def __init__(
        self,
        raw_root:       str | Path = "Dataset/raw/Warp-C",
        processed_root: str | Path = "Dataset/processed",
        stats_file:     str | Path = "Dataset/dataset_stats.json",
        img_size:       int  = IMG_SIZE,
        mean:           list = WARP_MEAN,
        std:            list = WARP_STD,
        batch_size:     int  = 32,
        num_workers:    int  = 4,
        seed:           int  = 42,
    ):
        self.raw_root       = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.stats_file     = Path(stats_file)
        self.img_size       = img_size
        self.mean           = mean
        self.std            = std
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.seed           = seed
        self._use_mixup     = False
 
        self.train_dir = self.raw_root / "train_crops"
        self.test_dir  = self.raw_root / "test_crops"
 
  
 
    def get_val_transforms(self) -> T.Compose:
        """
        Deterministic pipeline for validation / test / inference.
        Applied to ALL models at test time — no randomness whatsoever.
 
        Steps: PadToSquare → Resize(224) → ToTensor → Normalize
        """
        return T.Compose([
            PadToSquare(padding_mode="reflect"),
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])
 
 
    def _get_cnn_transforms(self) -> T.Compose:
        """
        LIGHT augmentation for CNNs training FROM SCRATCH.
 
        WHY LIGHT?
        ----------
        A CNN from scratch has never seen any image. It needs to first learn
        basic visual features — edges, shapes, colours. Heavy augmentation
        (MixUp, strong ColorJitter, RandomErasing) prevents this by showing
        the model distorted images before it has built any representation.
 
        What papers do for CNN baselines:
        - Ogrezeanu et al. (2024): resize + horizontal flip only
        - Chhabra et al. (2024): rotation + horizontal flip only
        - Standard waste classification literature: flip + small rotation
 
        We add mild ColorJitter because WaRP-C has lighting variation,
        but keep everything else minimal.
 
        Steps:
          PadToSquare → Resize(224) → HFlip → SmallRotation
          → MildColorJitter → ToTensor → Normalize
        """
        return T.Compose([
            PadToSquare(padding_mode="reflect"),
            T.Resize((self.img_size, self.img_size)),
            T.RandomHorizontalFlip(p=0.5),
            # small rotation only — large rotations confuse a model still
            # learning what objects look like
            T.RandomRotation(degrees=15),
            # mild jitter — enough to handle WaRP-C lighting variation
            # but not so strong it makes images unrecognisable
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])
 
 
    def _get_pretrained_cnn_transforms(self, crop_scale_min: float = 0.6) -> T.Compose:
        """
        STANDARD augmentation for pretrained CNN backbones.
 
        WHY STRONGER THAN CNN FROM SCRATCH?
        -------------------------------------
        ResNet50 and EfficientNet are pretrained on 1.2M ImageNet images.
        They already know what objects look like. Stronger augmentation
        teaches them to be robust to WaRP-C-specific challenges without
        preventing learning.
 
        Based on:
        - He et al. (2016) ResNet: RandomResizedCrop + HFlip standard
        - Tan & Le (2019) EfficientNet: same baseline + ColorJitter
        - Sayem et al. (2024) on WaRP-C: color variation + perspective + fill
 
        Steps:
          PadToSquare → RandomResizedCrop → Flips → Rotation
          → ColorJitter → GaussianBlur → ToTensor → Normalize → Erasing
        """
        return T.Compose([
            PadToSquare(padding_mode="reflect"),
            T.RandomResizedCrop(self.img_size, scale=(crop_scale_min, 1.0),
                                ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomChoice([
                T.RandomRotation(degrees=30),
                T.RandomRotation(degrees=(90,  90)),
                T.RandomRotation(degrees=(180, 180)),
                T.RandomRotation(degrees=(270, 270)),
            ]),
            T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
            T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
        ])
 
 
    def _get_transformer_transforms(self) -> T.Compose:
        """
        STRONG augmentation for transformer-based pretrained models.
 
        WHY STRONGER THAN PRETRAINED CNNs?
        ------------------------------------
        Transformers (Swin, ViT, ConvNeXt) use self-attention over patches.
        They benefit more from strong augmentation because:
          1. They have more parameters → more prone to overfitting without it
          2. Their attention mechanism handles MixUp soft labels well
          3. Liu et al. (2021) Swin paper uses: RandomResizedCrop, HFlip,
             RandAugment, Mixup, CutMix, label smoothing
 
        We use a slightly simplified version appropriate for our dataset size.
 
        Reference: Liu et al. (2021) Swin Transformer paper, training recipe.
 
        Steps:
          PadToSquare → RandomResizedCrop(0.5-1.0) → Flips → Rotation
          → StrongColorJitter → GaussianBlur → ToTensor → Normalize → Erasing
        NOTE: MixUp is applied in the training loop, not here.
        """
        return T.Compose([
            PadToSquare(padding_mode="reflect"),
            # wider crop range than CNN — transformers handle scale better
            T.RandomResizedCrop(self.img_size, scale=(0.5, 1.0),
                                ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomChoice([
                T.RandomRotation(degrees=30),
                T.RandomRotation(degrees=(90,  90)),
                T.RandomRotation(degrees=(180, 180)),
                T.RandomRotation(degrees=(270, 270)),
            ]),
            # slightly stronger than pretrained CNN
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.25),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
            T.RandomErasing(p=0.35, scale=(0.02, 0.25), ratio=(0.3, 3.3), value="random"),
        ])
 
 
    def _get_minority_transforms(self) -> T.Compose:
        """
        EXTRA-STRONG augmentation for minority classes only.
 
        Which classes: bottle-oil-full (24), detergent-box (66),
                       bottle-blue5l-full (89) — all < 30% of global mean.
 
        WHY EXTRA STRONG?
        -----------------
        The WeightedRandomSampler makes these 24 images appear ~13x per epoch.
        Without augmentation diversity, the model memorises exactly those
        24 images — classic overfitting. Stronger augmentation makes each
        repetition look visually different, effectively creating synthetic
        variants within the image manifold.
 
        Reference: Buda et al. (2018) — for rho > 50, oversampling +
        strong augmentation is optimal. Applied to pretrained models only
        (CNN from scratch gets val pipeline for minority classes because it
        needs clean examples to learn from).
        """
        return T.Compose([
            PadToSquare(padding_mode="reflect"),
            T.RandomResizedCrop(self.img_size, scale=(0.45, 1.0),
                                ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomChoice([
                T.RandomRotation(degrees=45),
                T.RandomRotation(degrees=(90,  90)),
                T.RandomRotation(degrees=(180, 180)),
                T.RandomRotation(degrees=(270, 270)),
            ]),
            T.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.4, hue=0.08),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
            T.RandomErasing(p=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.3), value="random"),
        ])
 
    def get_train_transforms(self) -> T.Compose:
        """
        Alias for the standard pretrained CNN training pipeline.
        Kept for backward compatibility.
        """
        return self._get_pretrained_cnn_transforms()
 
    def get_minority_transforms(self) -> T.Compose:
        """Public alias for minority transform pipeline."""
        return self._get_minority_transforms()
 
    def get_tta_transforms(self) -> list[T.Compose]:
        """
        Test-Time Augmentation — 5 deterministic variants.
        Run the same image through all 5, average softmax outputs.
        Typical gain: +1-3% accuracy with zero training cost.
        """
        base = [PadToSquare("reflect"), T.Resize((self.img_size, self.img_size))]
        norm = [T.ToTensor(), T.Normalize(mean=self.mean, std=self.std)]
        return [
            T.Compose(base + norm),
            T.Compose(base + [T.RandomHorizontalFlip(p=1.0)] + norm),
            T.Compose(base + [T.RandomVerticalFlip(p=1.0)]   + norm),
            T.Compose(base + [T.RandomRotation((90, 90))]    + norm),
            T.Compose(base + [T.ColorJitter(brightness=0.1)] + norm),
        ]
 
    @staticmethod
    def mixup_batch(
        images: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int = 28,
        alpha: float = 0.4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        MixUp batch augmentation — used in training loop for transformers.
 
        Example:
          λ = 0.7  →  mixed = 0.7×bottle-blue + 0.3×bottle-transp
          label   = [0.7, 0.3, 0, 0, ...]  (soft label)
 
        Use ONLY for: swin, vit, convnext, efficientnet.
        Do NOT use for: CNN from scratch (resnet50 profile), llava, gnn.
 
        The _use_mixup flag is set automatically by get_loaders(model_type=...).
        """
        lam   = float(np.random.beta(alpha, alpha))
        idx   = torch.randperm(images.size(0), device=images.device)  # same device
        mixed = lam * images + (1 - lam) * images[idx]
        # create one_hot on the SAME device as labels
        one_hot   = torch.zeros(images.size(0), num_classes, device=images.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        mixed_lbl = lam * one_hot + (1 - lam) * one_hot[idx]
        return mixed, mixed_lbl
 
    _MODEL_PROFILES: dict = {
        #                  sampler  min_aug  mixup  transform
        "cnn":           (False,   False,   False, "cnn"),
 
        "resnet50":      (True,    True,    False, "pretrained_cnn"),
        # Pretrained ResNet: sampler ON (rho=59.67 needs it), minority aug ON,
        # no mixup (not standard in original ResNet training recipe)
 
        "efficientnet":  (True,    True,    True,  "pretrained_cnn"),
        # EfficientNet benefits from MixUp — Tan & Le (2019) used it
 
        "swin":          (True,    True,    True,  "transformer"),
        # Swin: Liu et al. (2021) training recipe uses MixUp + CutMix
 
        "vit":           (True,    True,    True,  "transformer"),
        # ViT: same transformer recipe as Swin
 
        "convnext":      (True,    True,    True,  "transformer"),
        # ConvNeXt: Liu et al. (2022) uses same training recipe as Swin
 
        "edgevit":       (True,    True,    False, "pretrained_cnn_gentle"),
        # EdgeViT: lighter model, no mixup, slightly gentler crop
 
        "mobilevit":     (True,    True,    False, "pretrained_cnn_gentle"),
        # MobileViT: same as EdgeViT
 
        "llava":         (False,   False,   False, "val"),
        # LLaVA: inference only, no training
 
        "gnn":           (False,   False,   False, "val"),
        # GNN: feature extraction only, val pipeline
 
        "default":       (True,    True,    False, "pretrained_cnn"),
    }
 
    def _get_transform_for_key(self, key: str) -> T.Compose:
        """Map profile transform key to actual transform pipeline."""
        if key == "cnn":
            return self._get_cnn_transforms()
        elif key in ("pretrained_cnn",):
            return self._get_pretrained_cnn_transforms(crop_scale_min=0.6)
        elif key == "pretrained_cnn_gentle":
            return self._get_pretrained_cnn_transforms(crop_scale_min=0.7)
        elif key == "transformer":
            return self._get_transformer_transforms()
        else:  # "val" or unknown
            return self.get_val_transforms()
 

 
    def get_loaders(
        self,
        model_type:  str  = "default",
    ) -> tuple[DataLoader, DataLoader]:
        """
        Build train and test DataLoaders.
 
        Just pass model_type — everything else is automatic.
 
        Model types and what they do
        ----------------------------
        "cnn"          → light aug (flip + small rotation), NO sampler
                         Use for your baseline CNN from scratch
        "resnet50"     → standard aug, sampler ON, mixup OFF
        "efficientnet" → standard aug, sampler ON, mixup ON
        "swin"         → strong aug, sampler ON, mixup ON
        "vit"          → strong aug, sampler ON, mixup ON
        "convnext"     → strong aug, sampler ON, mixup ON
        "edgevit"      → gentle aug, sampler ON, mixup OFF
        "mobilevit"    → gentle aug, sampler ON, mixup OFF
        "llava"        → val pipeline only (no training)
        "gnn"          → val pipeline only (feature extraction)
 
        KEY POINT: Dataset/processed/ stores ORIGINAL-SIZE cleaned images.
        The 224×224 normalised tensor is created in RAM per batch by the
        DataLoader. Nothing augmented is ever saved to disk.
 
        Imbalance strategy (rho=59.67, Buda et al. 2018 Tier 3):
          Layer 1 — WeightedRandomSampler (minority seen ~13x per epoch)
          Layer 2 — CrossEntropyLoss(weight=class_weights) via get_class_weights()
          Layer 3 — Stronger augmentation on minority classes
          NOTE: CNN profile disables all 3 layers — see _MODEL_PROFILES above
        """
        profile = self._MODEL_PROFILES.get(
            model_type.lower(), self._MODEL_PROFILES["default"]
        )
        use_sampler, use_minority_aug, self._use_mixup, tfm_key = profile
 
        print(f"[get_loaders] model='{model_type}'")
        print(f"  sampler={use_sampler}  minority_aug={use_minority_aug}  "
              f"mixup={self._use_mixup}  pipeline='{tfm_key}'")
 
        train_tfm   = self._get_transform_for_key(tfm_key)
        minority_tfm = (self._get_minority_transforms()
                        if use_minority_aug else None)
 
        train_ds = WaRPDataset(
            root               = self.processed_root / "train",
            transform          = train_tfm,
            minority_transform = minority_tfm,
            stats_file         = self.stats_file,
        )
        test_ds = WaRPDataset(
            root       = self.processed_root / "test",
            transform  = self.get_val_transforms(),
            stats_file = self.stats_file,
        )
 
        sampler = None
        if use_sampler:
            g = torch.Generator()
            g.manual_seed(self.seed)
            sampler = WeightedRandomSampler(
                weights     = train_ds.get_sample_weights(),
                num_samples = len(train_ds),
                replacement = True,
                generator   = g,
            )
 
        train_loader = DataLoader(
            train_ds,
            batch_size         = self.batch_size,
            sampler            = sampler,
            shuffle            = sampler is None,
            num_workers        = self.num_workers,
            pin_memory         = True,
            persistent_workers = self.num_workers > 0,
            drop_last          = True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = True,
            persistent_workers = self.num_workers > 0,
        )
 
        print(f"  train={len(train_loader)} batches  "
              f"test={len(test_loader)} batches  "
              f"sampler={'WeightedRandom' if use_sampler else 'Uniform (shuffle)'}")
 
        self._train_ds = train_ds
        return train_loader, test_loader
 
    def get_class_weights(self, device: str = "cpu") -> Optional[torch.Tensor]:
        """
        Inverse-frequency class weights for CrossEntropyLoss.
 
        Formula: w_c = (1/count_c) × (N/num_classes)
 
        Usage:
            weights   = pp.get_class_weights(device='cuda')
            criterion = nn.CrossEntropyLoss(weight=weights)
        """
        if not self.stats_file.exists():
            return None
        with open(self.stats_file) as f:
            stats = json.load(f)
        names  = stats["class_distribution"]["class_names"]
        counts = stats["class_distribution"]["train_counts"]
        paired = sorted(zip(names, counts))
        raw_w  = [1.0 / c for _, c in paired]
        norm   = len(raw_w) / sum(raw_w)
        return torch.tensor(
            [w * norm for w in raw_w], dtype=torch.float32
        ).to(device)
 

 
    def prepare(self, force: bool = False) -> None:
        """
        One-time cleaning step: remove data leakage, copy to processed/.
 
        Steps
        -----
        1. Find 18 filenames in BOTH train and test (data leakage)
        2. Remove from TRAIN only (keep test → benchmark comparability)
        3. Copy to Dataset/processed/train/<class>/ and .../test/<class>/
        4. Write manifest.csv (one row per image)
        5. Write split_stats.json
 
        Parameters
        ----------
        force : if True, delete and recreate even if processed/ exists
        """
        print("=" * 60)
        print("  WaRPPreprocessor.prepare()")
        print("=" * 60)
 
        if self.processed_root.exists() and not force:
            print(f"  {self.processed_root}/ already exists. "
                  f"Pass force=True to recreate.")
            return
 
        if not self.train_dir.exists():
            raise FileNotFoundError(
                f"Raw data not found at {self.train_dir}\n"
                "→ Run download_data.py first."
            )
 
        duplicates   = self._find_duplicates()
        minority_set = self._get_minority_set()
        print(f"\n  Leakage filenames found : {len(duplicates)}")
        print(f"  Minority classes        : {sorted(minority_set)}")
 
        if self.processed_root.exists():
            shutil.rmtree(self.processed_root)
 
        manifest: list[dict] = []
        train_counts = self._copy_split(
            self.train_dir, "train", duplicates, minority_set, manifest
        )
        test_counts = self._copy_split(
            self.test_dir, "test", set(), minority_set, manifest
        )
 
        manifest_path = self.processed_root / "manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
            writer.writeheader()
            writer.writerows(manifest)
 
        all_cls = sorted(set(train_counts) | set(test_counts))
        tr_vals = [train_counts.get(c, 0) for c in all_cls]
        rho     = round(max(tr_vals) / max(1, min(tr_vals)), 2)
        stats   = {
            "total_train":        sum(train_counts.values()),
            "total_test":         sum(test_counts.values()),
            "duplicates_removed": len(duplicates),
            "imbalance_ratio":    rho,
            "minority_classes":   sorted(minority_set),
            "normalisation":      {"mean": self.mean, "std": self.std},
            "img_size":           self.img_size,
            "per_class": [
                {"class": c,
                 "train": train_counts.get(c, 0),
                 "test":  test_counts.get(c, 0),
                 "is_minority": c in minority_set}
                for c in all_cls
            ],
        }
        with open(self.processed_root / "split_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
 
        print(f"\n  ✓ Train images    : {stats['total_train']}")
        print(f"  ✓ Test  images    : {stats['total_test']}")
        print(f"  ✓ Leakage removed : {len(duplicates)}")
        print(f"  ✓ Output          : {self.processed_root}/")
 
    
 
    def summary(self) -> None:
        """Print full pipeline summary."""
        sep = "=" * 62
        print(f"\n{sep}")
        print("  WaRPPreprocessor — Pipeline Summary")
        print(sep)
        print(f"  Input size   : {self.img_size}×{self.img_size} px")
        print(f"  Mean (R,G,B) : {self.mean}  [WaRP-C EDA stats]")
        print(f"  Std  (R,G,B) : {self.std}")
        print(f"  Batch size   : {self.batch_size}")
        print()
        print("  AUGMENTATION PER MODEL TYPE:")
        print("  ─────────────────────────────────────────────────────")
        print("  cnn          → flip + rotation(15°) + mild jitter")
        print("                 NO sampler, NO minority aug, NO mixup")
        print("                 (model trains from scratch — keep simple)")
        print("  resnet50     → ResizedCrop + flips + rotation + jitter")
        print("                 + blur + erasing  |  sampler ON")
        print("  efficientnet → same as resnet50  |  mixup ON")
        print("  swin         → stronger crop(0.5) + flips + rotation")
        print("                 + jitter + blur + erasing  |  mixup ON")
        print("  vit          → same as swin  |  mixup ON")
        print("  convnext     → same as swin  |  mixup ON")
        print("  edgevit      → pretrained_cnn with gentler crop(0.7)")
        print("  mobilevit    → same as edgevit")
        print("  llava / gnn  → val pipeline only (inference)")
        print()
        print("  Imbalance strategy (rho=59.67, Buda et al. 2018):")
        print("    Layer 1 — WeightedRandomSampler (for pretrained models)")
        print("    Layer 2 — CrossEntropyLoss(weight=class_weights)")
        print("    Layer 3 — Minority-specific augmentation")
        print(f"\n  Leakage fix  : 18 duplicate filenames removed from train")
        print(sep)
 
    def check_upscale_risk(self, threshold: float = 3.0) -> dict:
        """
        Flag images where resizing to img_size requires upscale > threshold.
        threshold=3.0 means shorter side < 75px would be flagged.
        """
        limit_px = self.img_size / threshold
        by_class: dict[str, int]                    = defaultdict(int)
        examples: list[tuple[str, int, int, float]] = []
        total = risky = 0
 
        for split_dir in [self.train_dir, self.test_dir]:
            if not split_dir.exists():
                continue
            for parent in sorted(split_dir.iterdir()):
                for sub in sorted(parent.iterdir()):
                    if not sub.is_dir():
                        continue
                    for p in sorted(sub.iterdir()):
                        if p.suffix.lower() not in VALID_EXTS:
                            continue
                        total += 1
                        try:
                            with Image.open(p) as img:
                                w, h   = img.size
                                short  = min(w, h)
                                factor = self.img_size / short
                                if factor > threshold:
                                    risky += 1
                                    by_class[sub.name] += 1
                                    if len(examples) < 10:
                                        examples.append(
                                            (str(p), w, h, round(factor, 2))
                                        )
                        except Exception:
                            pass
 
        result = {
            "threshold": threshold, "risky_limit_px": round(limit_px, 1),
            "total_scanned": total, "risky_count": risky,
            "risky_pct": round(risky / total * 100, 2) if total else 0.0,
            "by_class": dict(sorted(by_class.items(), key=lambda x: -x[1])),
            "examples": examples,
        }
 
        print(f"\n  Upscale Risk Check (target={self.img_size}px, "
              f"threshold={threshold}x)")
        print(f"  Scanned: {total:,}  |  Risky: {risky:,} "
              f"({result['risky_pct']}%)")
        if risky:
            for cls, cnt in list(result["by_class"].items())[:5]:
                print(f"    {cls:<28}  {cnt:>4}")
        return result
 
 
    def _find_duplicates(self) -> set[str]:
        train_names: set[str] = set()
        test_names:  set[str] = set()
        for split_dir, name_set in [
            (self.train_dir, train_names), (self.test_dir, test_names)
        ]:
            for parent in split_dir.iterdir():
                for sub in parent.iterdir():
                    if sub.is_dir():
                        for p in sub.iterdir():
                            if p.suffix.lower() in VALID_EXTS:
                                name_set.add(p.name)
        return train_names & test_names
 
    def _get_minority_set(self) -> set[str]:
        if not self.stats_file.exists():
            return set()
        with open(self.stats_file) as f:
            stats = json.load(f)
        names  = stats["class_distribution"]["class_names"]
        counts = stats["class_distribution"]["train_counts"]
        mean   = sum(counts) / len(counts)
        return {cls for cls, cnt in zip(names, counts)
                if cnt < MINORITY_THRESHOLD * mean}
 
    def _copy_split(
        self,
        raw_split:    Path,
        split_name:   str,
        duplicates:   set[str],
        minority_set: set[str],
        manifest:     list[dict],
    ) -> dict[str, int]:
        dest_root = self.processed_root / split_name
        counts: dict[str, int] = {}
        all_classes: dict[str, list[Path]] = defaultdict(list)
 
        for parent in sorted(raw_split.iterdir()):
            for sub in sorted(parent.iterdir()):
                if not sub.is_dir():
                    continue
                for p in sorted(sub.iterdir()):
                    if p.suffix.lower() not in VALID_EXTS:
                        continue
                    if split_name == "train" and p.name in duplicates:
                        continue
                    all_classes[sub.name].append(p)
 
        for cls_name in tqdm(sorted(all_classes),
                             desc=f"  Copying {split_name}"):
            paths    = all_classes[cls_name]
            dest_cls = dest_root / cls_name
            dest_cls.mkdir(parents=True, exist_ok=True)
            for src in paths:
                shutil.copy2(src, dest_cls / src.name)
                manifest.append({
                    "split":       split_name,
                    "class":       cls_name,
                    "filename":    src.name,
                    "is_minority": cls_name in minority_set,
                })
            counts[cls_name] = len(paths)
        return counts