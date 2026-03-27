"""
Pipeline_/preprocessor.py
==========================
OOP wrapper that encapsulates ALL preprocessing and augmentation decisions
for the WaRP-C dataset in one single class.
EEEM068: Applied Machine Learning — University of Surrey, Spring 2026

Usage:
    from Pipeline_.preprocessor import WaRPPreprocessor
    pp = WaRPPreprocessor()
    pp.summary()
    train_loader, test_loader = pp.get_loaders()
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


# ─────────────────────────────────────────────────────────────────────────────
# Constants derived from EDA results (dataset_stats.json)
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE = 224          # required by ALL pretrained backbones (ResNet, ViT, YOLO…)
                        #       ImageNet, the dataset all pretrained models were trained on,
                        #       uses 224×224 as the standard input size.

WARP_MEAN = [0.337, 0.344, 0.350]   # computed by EDA from 1500 WaRP-C training images
WARP_STD  = [0.216, 0.209, 0.218]   # mean_diff from ImageNet = 0.105 > threshold 0.05

IMAGENET_MEAN = [0.485, 0.456, 0.406]   # kept for reference — pretrained models were
IMAGENET_STD  = [0.229, 0.224, 0.225]   # normalised with these values during pretraining

MINORITY_THRESHOLD = 0.30   # classes with < 30% of global mean = minority
                             # global mean = 8823/28 ≈ 315 → minority < 94 images

VALID_EXTS = {".jpg", ".jpeg", ".png"}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PadToSquare  (custom torchvision-compatible transform)
# ─────────────────────────────────────────────────────────────────────────────

class PadToSquare:
    """
    Pad a PIL image to a square BEFORE resizing.

    WHY:  EDA showed aspect ratios 0.2–5.2.  A plain Resize(224,224) distorts
          non-square images — squashing a tall narrow bottle into a square
          destroys the shape information the model relies on.

    WHY REFLECT (not zero):  Zero-padding creates black borders.  A CNN can
          learn "images with black borders → rare class" as a shortcut.
          Reflect padding mirrors the edge pixels so there is no artificial
          border signal.

    Example:
          Original: 80 × 200 px  (portrait, ratio = 0.4)
          After pad: 200 × 200 px (reflect-padded on left and right)
          After Resize: 224 × 224 px (no distortion)
    """

    def __init__(self, padding_mode: str = "reflect"):
        self.padding_mode = padding_mode

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h  = img.size
        diff  = abs(w - h)
        pad_a = diff // 2
        pad_b = diff - pad_a

        if w < h:
            padding = (pad_a, 0, pad_b, 0)   # pad left and right
        elif h < w:
            padding = (0, pad_a, 0, pad_b)   # pad top and bottom
        else:
            return img                         # already square

        return TF.pad(img, padding, padding_mode=self.padding_mode)

    def __repr__(self):
        return f"PadToSquare(mode={self.padding_mode!r})"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  WaRPDataset  (PyTorch Dataset)
# ─────────────────────────────────────────────────────────────────────────────

class WaRPDataset(Dataset):
    """
    PyTorch Dataset for WaRP-C.

    Automatically routes each sample to the correct transform pipeline:
      - Minority classes (< 94 images)  → minority_transform  (stronger aug)
      - All other classes               → transform            (standard aug)

    Parameters
    ----------
    root               : path to Dataset/processed/train or .../test
    transform          : standard pipeline (train aug or val/test pipeline)
    minority_transform : stronger pipeline for severely underrepresented classes
    stats_file         : path to dataset_stats.json written by EDAModule
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

        # class list = sorted subfolder names → deterministic label integers
        self.classes:      list[str]      = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        # scan all image paths
        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            label = self.class_to_idx[cls]
            for p in sorted((self.root / cls).iterdir()):
                if p.suffix.lower() in VALID_EXTS:
                    self.samples.append((p, label))

        self._load_stats(Path(stats_file))

    # ── private ──────────────────────────────────────────────────────────────

    def _load_stats(self, stats_file: Path) -> None:
        """
        Load EDA JSON to identify minority classes and compute class weights.

        Class weight formula:  w_c = (1 / count_c) * (N / num_classes)
        This normalises weights so their expected value is 1.0.
        Reference: King & Zeng (2001); standard PyTorch documentation.
        """
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
        self._class_weights = torch.tensor([w * norm for w in raw_w], dtype=torch.float32)

    # ── public ───────────────────────────────────────────────────────────────

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

        # route to the right pipeline transparently
        if cls_name in self._minority_classes and self.minority_transform is not None:
            tfm = self.minority_transform
        else:
            tfm = self.transform

        if tfm is not None:
            img = tfm(img)

        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# 3.  WaRPPreprocessor  (main OOP interface)
# ─────────────────────────────────────────────────────────────────────────────

class WaRPPreprocessor:
    """
    All-in-one preprocessing and data pipeline for WaRP-C.

    This class owns:
      - All transform pipelines (train / val / minority / TTA)
      - Dataset construction
      - DataLoader construction with WeightedRandomSampler
      - The one-time dataset cleaning step (leakage removal + copy)

    Typical usage
    -------------
    pp = WaRPPreprocessor()
    pp.prepare()                          # run once — cleans & copies dataset
    train_loader, test_loader = pp.get_loaders()
    weights = pp.get_class_weights(device="cuda")
    """

    def __init__(
        self,
        raw_root:       str | Path = "Dataset/raw/WaRP-C",
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

        self.train_dir = self.raw_root / "train_crops"
        self.test_dir  = self.raw_root / "test_crops"

    # ── A. Transform pipelines ────────────────────────────────────────────────

    def get_train_transforms(self) -> T.Compose:
        """
        Standard training augmentation pipeline.

        Step-by-step with concrete examples
        ------------------------------------
        1. PadToSquare(reflect)
           Input:  80×200 px tall bottle crop
           Output: 200×200 px  (padded left+right by 60px each, mirrored)
           Why:    aspect ratios 0.2–5.2 in dataset; naive resize squashes shapes

        2. RandomResizedCrop(224, scale=0.6–1.0)
           Randomly crops 60%–100% of the padded image then resizes to 224px
           Example: crops the top 70% of a bottle image, zooming in on the cap
           Why:    WaRP crops have varying zoom levels from the detector

        3. RandomHorizontalFlip(p=0.5)
           50% chance: mirror the image left↔right
           Example: bottle facing right becomes bottle facing left
           Why:    items arrive from both directions on the belt

        4. RandomVerticalFlip(p=0.3)
           30% chance: mirror top↔bottom
           Example: upright bottle becomes upside-down bottle
           Why:    items can be flipped on conveyor; lower p = less common

        5. RandomRotation(30° or 90° or 180° or 270°)
           Randomly picks one discrete rotation per image
           Example: glass bottle rotated 90° still = glass bottle
           Why:    items rotate freely on belt

        6. ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.05)
           brightness: multiplies pixel values by random factor in [0.6, 1.4]
           Example: glass-dark (mean brightness 0.213) made brighter/darker
           Why:    EDA range = 0.268 between darkest/brightest class
           hue=0.05: tiny shift — larger would turn blue bottles green (wrong class)

        7. GaussianBlur(p=0.2)
           20% chance: apply slight motion blur (kernel=3px)
           Example: bottle moving fast on belt appears slightly blurred
           Why:    industrial cameras capture moving objects

        8. ToTensor + Normalize(WaRP-C stats)
           Converts [0,255] pixels to [0,1] float, then zero-centres each channel
           Example: red channel mean 0.337 → subtract 0.337, divide by 0.216
           Why:    mean_diff from ImageNet = 0.105 > 0.05 threshold → use WaRP stats
                   (ImageNet stats are used by all pretrained models like ResNet/ViT
                    since they were trained on ImageNet, but WaRP-C is too different)

        9. RandomErasing(p=0.3, scale=2%–20%)
           30% chance: erase a random rectangle and fill with noise
           Example: a 40×60px patch on a bottle image is replaced with random pixels
           Why:    simulates one item partially occluding another on the belt
        """
        return T.Compose([
            PadToSquare(padding_mode="reflect"),
            T.RandomResizedCrop(self.img_size, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
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

    def get_val_transforms(self) -> T.Compose:
        """
        Deterministic pipeline for validation / test (NO augmentation).

        Why no augmentation on test?
        ----------------------------
        Test transforms must be identical every run so accuracy numbers are
        comparable across experiments and across teammates' machines.

        Steps:
          PadToSquare → Resize(224) → ToTensor → Normalize
        """
        return T.Compose([
            PadToSquare(padding_mode="reflect"),
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])

    def get_minority_transforms(self) -> T.Compose:
        """
        Extra-strong augmentation for minority classes only.

        Which classes?  bottle-oil-full (24), detergent-box (66),
                        bottle-blue5l-full (89) — all < 30% of global mean (315)

        Why stronger?
        -------------
        These 24 images are seen ~13× per epoch by the sampler.
        Without diversity, the model memorises exactly those 24 images.
        Stronger augmentation makes each repetition look different —
        effectively generating synthetic variants within the image space.

        Differences vs standard train pipeline:
          - RandomResizedCrop scale: 0.5–1.0  (vs 0.6–1.0)
          - ColorJitter brightness:  0.5       (vs 0.4)
          - RandomErasing p:         0.4       (vs 0.3)
          - RandomErasing scale:     up to 25% (vs 20%)
        """
        return T.Compose([
            PadToSquare(padding_mode="reflect"),
            T.RandomResizedCrop(self.img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
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

    def get_tta_transforms(self) -> list[T.Compose]:
        """
        Test-Time Augmentation — 5 deterministic variants for inference.

        Usage: run the same image through all 5, average the softmax outputs.
        Typical gain: +1–3% accuracy with zero additional training.
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
        MixUp augmentation — blend two images and their labels.

        Example:
          λ = 0.7 (sampled from Beta(0.4, 0.4))
          mixed_image = 0.7 × bottle-blue + 0.3 × bottle-transp
          mixed_label = [0.7 for bottle-blue, 0.3 for bottle-transp, 0 otherwise]

        Why for WaRP-C?
          bottle-blue and bottle-transp look very similar → MixUp forces the
          model to handle ambiguous in-between examples, improving calibration.

        Call inside training loop:
          imgs_mix, lbls_mix = WaRPPreprocessor.mixup_batch(imgs, lbls, 28)
          loss = criterion(model(imgs_mix), lbls_mix)
        """
        lam  = float(np.random.beta(alpha, alpha))
        idx  = torch.randperm(images.size(0))
        mixed = lam * images + (1 - lam) * images[idx]

        one_hot  = torch.zeros(images.size(0), num_classes).scatter_(1, labels.view(-1, 1), 1.0)
        mixed_lbl = lam * one_hot + (1 - lam) * one_hot[idx]
        return mixed, mixed_lbl

    # ── B. Dataset cleaning (one-time) ────────────────────────────────────────

    def prepare(self, force: bool = False) -> None:
        """
        One-time cleaning step: remove data leakage, copy to processed/.

        Steps
        -----
        1. Find the 18 filenames that appear in BOTH train and test  (leakage)
        2. Remove those 18 filenames from the TRAIN copy only
           (keep in test → preserves official benchmark comparability)
        3. Copy cleaned images to Dataset/processed/train/<class>/
           and Dataset/processed/test/<class>/
        4. Write manifest.csv  (one row per image)
        5. Write split_stats.json (per-class counts + key parameters)

        Parameters
        ----------
        force : if True, delete and recreate processed/ even if it exists
        """
        print("=" * 60)
        print("  WaRPPreprocessor.prepare()")
        print("=" * 60)

        if self.processed_root.exists() and not force:
            print(f"  {self.processed_root}/ already exists. Pass force=True to recreate.")
            return

        if not self.train_dir.exists():
            raise FileNotFoundError(
                f"Raw data not found at {self.train_dir}\n"
                "→ Run download_data.py first."
            )

        # step 1: find all duplicate filenames
        duplicates = self._find_duplicates()
        print(f"\n  Leakage filenames found: {len(duplicates)}")

        # step 2: identify minority classes
        minority_set = self._get_minority_set()
        print(f"  Minority classes: {sorted(minority_set)}")

        # step 3 & 4: scan + copy
        if self.processed_root.exists():
            shutil.rmtree(self.processed_root)

        manifest:      list[dict] = []
        train_counts = self._copy_split(self.train_dir,  "train", duplicates,    minority_set, manifest)
        test_counts  = self._copy_split(self.test_dir,   "test",  set(),          minority_set, manifest)

        # write manifest.csv
        manifest_path = self.processed_root / "manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
            writer.writeheader()
            writer.writerows(manifest)

        # write split_stats.json
        all_cls  = sorted(set(train_counts) | set(test_counts))
        tr_vals  = [train_counts.get(c, 0) for c in all_cls]
        rho      = round(max(tr_vals) / max(1, min(tr_vals)), 2)
        stats    = {
            "total_train":        sum(train_counts.values()),
            "total_test":         sum(test_counts.values()),
            "duplicates_removed": len(duplicates),
            "imbalance_ratio":    rho,
            "minority_classes":   sorted(minority_set),
            "normalisation":      {"mean": self.mean, "std": self.std},
            "img_size":           self.img_size,
            "per_class": [
                {"class": c, "train": train_counts.get(c, 0),
                 "test": test_counts.get(c, 0), "is_minority": c in minority_set}
                for c in all_cls
            ],
        }
        with open(self.processed_root / "split_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n  ✓ Train images : {stats['total_train']}")
        print(f"  ✓ Test  images : {stats['total_test']}")
        print(f"  ✓ Leakage removed : {len(duplicates)}")
        print(f"  ✓ Output → {self.processed_root}/")

    # ── C. DataLoader factory ─────────────────────────────────────────────────

    def get_loaders(
        self,
        use_sampler:      bool = True,
        use_minority_aug: bool = True,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Build train and test DataLoaders.

        Imbalance strategy (ρ = 59.67 → Buda et al. 2018 Tier 3):
          Layer 1 — WeightedRandomSampler   : minority classes sampled more often
          Layer 2 — CrossEntropyLoss weight : via get_class_weights()
          Layer 3 — Minority transforms     : stronger augmentation on rare classes

        Parameters
        ----------
        use_sampler      : enable WeightedRandomSampler (recommended)
        use_minority_aug : route minority classes to stronger transform pipeline
        """
        train_ds = WaRPDataset(
            root               = self.processed_root / "train",
            transform          = self.get_train_transforms(),
            minority_transform = self.get_minority_transforms() if use_minority_aug else None,
            stats_file         = self.stats_file,
        )
        test_ds = WaRPDataset(
            root       = self.processed_root / "test",
            transform  = self.get_val_transforms(),
            stats_file = self.stats_file,
        )

        # WeightedRandomSampler
        sampler = None
        if use_sampler:
            g = torch.Generator()
            g.manual_seed(self.seed)
            sampler = WeightedRandomSampler(
                weights     = train_ds.get_sample_weights(),
                num_samples = len(train_ds),
                replacement = True,   # MUST be True — minority needs repeated sampling
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

        print(f"[DataLoaders]  train={len(train_loader)} batches  "
              f"test={len(test_loader)} batches  "
              f"sampler={'WeightedRandom' if use_sampler else 'Uniform'}")

        self._train_ds = train_ds   # store for class_weights access
        return train_loader, test_loader

    def get_class_weights(self, device: str = "cpu") -> Optional[torch.Tensor]:
        """
        Inverse-frequency class weights for CrossEntropyLoss.

        Usage:
            weights   = pp.get_class_weights(device="cuda")
            criterion = nn.CrossEntropyLoss(weight=weights)

        Formula: w_c = (1/count_c) * (N/num_classes)
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
        return torch.tensor([w * norm for w in raw_w], dtype=torch.float32).to(device)

    # ── D. Summary ────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a full pipeline summary — useful at the top of any notebook."""
        sep = "=" * 60
        print(f"\n{sep}")
        print("  WaRPPreprocessor — Pipeline Summary")
        print(f"{sep}")
        print(f"  Input size      : {self.img_size}×{self.img_size} px")
        print(f"  Mean (R,G,B)    : {self.mean}  [WaRP-C stats]")
        print(f"  Std  (R,G,B)    : {self.std}")
        print(f"  Batch size      : {self.batch_size}")
        print(f"  Minority thresh : < {MINORITY_THRESHOLD*100:.0f}% of class mean")
        print(f"\n  Train pipeline  :")
        print(f"    PadToSquare(reflect) → RandomResizedCrop(224)")
        print(f"    → Flips → Rotation → ColorJitter(b=0.4) → Blur → Normalize → Erase")
        print(f"\n  Val/Test pipeline:")
        print(f"    PadToSquare(reflect) → Resize(224) → Normalize")
        print(f"\n  Imbalance strategy (ρ=59.67, Buda et al. 2018):")
        print(f"    Layer 1 — WeightedRandomSampler")
        print(f"    Layer 2 — CrossEntropyLoss(weight=class_weights)")
        print(f"    Layer 3 — Stronger augmentation for minority classes")
        print(f"\n  Leakage fix     : 18 duplicate filenames removed from train")
        print(sep)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_duplicates(self) -> set[str]:
        """Live scan: filenames that appear in both train and test splits."""
        train_names: set[str] = set()
        test_names:  set[str] = set()

        for split_dir, name_set in [(self.train_dir, train_names), (self.test_dir, test_names)]:
            for parent in split_dir.iterdir():
                for sub in parent.iterdir():
                    if sub.is_dir():
                        for p in sub.iterdir():
                            if p.suffix.lower() in VALID_EXTS:
                                name_set.add(p.name)

        return train_names & test_names

    def _get_minority_set(self) -> set[str]:
        """Identify minority classes from stats JSON."""
        if not self.stats_file.exists():
            return set()
        with open(self.stats_file) as f:
            stats = json.load(f)
        names  = stats["class_distribution"]["class_names"]
        counts = stats["class_distribution"]["train_counts"]
        mean   = sum(counts) / len(counts)
        return {cls for cls, cnt in zip(names, counts) if cnt < MINORITY_THRESHOLD * mean}

    def _copy_split(
        self,
        raw_split:    Path,
        split_name:   str,
        duplicates:   set[str],
        minority_set: set[str],
        manifest:     list[dict],
    ) -> dict[str, int]:
        """Scan raw split, skip duplicates (train only), copy to processed/."""
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

        for cls_name in tqdm(sorted(all_classes), desc=f"  Copying {split_name}"):
            paths    = all_classes[cls_name]
            dest_cls = dest_root / cls_name
            dest_cls.mkdir(parents=True, exist_ok=True)
            for src in paths:
                dest = dest_cls / src.name
                shutil.copy2(src, dest)
                manifest.append({
                    "split":      split_name,
                    "class":      cls_name,
                    "filename":   src.name,
                    "is_minority": cls_name in minority_set,
                })
            counts[cls_name] = len(paths)

        return counts

    def check_upscale_risk(self, threshold: float = 3.0) -> dict:
        """
        Scan all raw images and flag those where resizing to img_size
        would require an upscale factor above `threshold`.

        An upscale factor = img_size / shorter_side.
        Factor > 3.0 means we are more than tripling the resolution —
        the model will be learning from interpolation artefacts, not real
        pixel information.

        The threshold is intentionally a parameter because there is no
        universal rule — 3.0 is the commonly cited safe limit, but you
        might accept 4.0 for a small number of images, or set 2.0 if you
        want to be conservative.

        Parameters
        ----------
        threshold : upscale factor above which an image is considered risky
                    (default 3.0 — images whose shorter side < img_size/3)

        Returns
        -------
        dict with:
          'threshold'       : the factor used
          'risky_limit_px'  : shorter side below this px is flagged
          'total_scanned'   : total images scanned across both splits
          'risky_count'     : number of images above threshold
          'risky_pct'       : percentage of total
          'by_class'        : {class_name: count_of_risky_images}
          'examples'        : up to 10 (path, w, h, factor) tuples
        """
        limit_px = self.img_size / threshold   # shorter side below this = risky

        by_class:  dict[str, int]                   = defaultdict(int)
        examples:  list[tuple[str, int, int, float]] = []
        total      = 0
        risky      = 0

        for split_dir in [self.train_dir, self.test_dir]:
            if not split_dir.exists():
                continue
            for parent in sorted(split_dir.iterdir()):
                for sub in sorted(parent.iterdir()):
                    if not sub.is_dir():
                        continue
                    cls_name = sub.name
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
                                    by_class[cls_name] += 1
                                    if len(examples) < 10:
                                        examples.append((str(p), w, h, round(factor, 2)))
                        except Exception:
                            pass

        result = {
            "threshold":      threshold,
            "risky_limit_px": round(limit_px, 1),
            "target_size_px": self.img_size,
            "total_scanned":  total,
            "risky_count":    risky,
            "risky_pct":      round(risky / total * 100, 2) if total else 0.0,
            "by_class":       dict(sorted(by_class.items(), key=lambda x: -x[1])),
            "examples":       examples,
        }

        # ── print summary ────────────────────────────────────────────────
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  Upscale Risk Check  (target={self.img_size}px, threshold={threshold}×)")
        print(sep)
        print(f"  Images scanned   : {total:,}")
        print(f"  Risky limit      : shorter side < {limit_px:.1f}px")
        print(f"  Risky images     : {risky:,}  ({result['risky_pct']}% of dataset)")
        print()

        if risky == 0:
            print("  ✓ No upscale risk detected.")
        else:
            print("  By class:")
            for cls, cnt in result["by_class"].items():
                bar = "█" * min(cnt, 30)
                print(f"    {cls:<28}  {cnt:>4}  {bar}")
            print()
            print("  Examples (path, w×h px, upscale factor):")
            for path, w, h, fac in examples:
                print(f"    {fac:.2f}×  {w}×{h}px  {Path(path).name}")

        print()
        if risky / total < 0.01 if total else False:
            print("  → < 1% of data. Acceptable — treat as a documented limitation.")
        elif risky / total < 0.05 if total else False:
            print("  → 1–5% of data. Moderate. Consider filtering in prepare().")
        else:
            print("  → > 5% of data. Significant. Recommend filtering before training.")
        print(sep)

        return result