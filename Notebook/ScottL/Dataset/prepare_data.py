"""
EEEM068: Industrial Waste Classification
prepare_data.py — One-time data preparation script for WaRP-C

USAGE:
    python prepare_data.py

    # Custom paths if needed:
    python Dataset/prepare_data.py --warp_c_root Dataset/Warp-C --output_root data --val_split 0.2 --seed 42

WHAT THIS SCRIPT DOES:
    1. Walks the nested WaRP-C folder structure to find all 28 leaf class folders
    2. Creates a flat data/train/, data/val/, data/test/ structure (symlinks — no copying)
    3. Splits train_crops 80/20 into train/val (stratified, respects class balance)
    4. Generates dataset/train.csv, dataset/val.csv, dataset/test.csv
    5. Prints a class distribution report so you can see imbalance immediately

RUN THIS ONCE before running train.py.
Do NOT commit the data/ folder to Git.

OUTPUT STRUCTURE:
    data/
    ├── train/
    │   ├── bottle-blue/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── bottle-dark/
    │   └── ... (28 classes)
    ├── val/
    │   └── ... (same 28 classes, 20% of train)
    └── test/
        └── ... (same 28 classes)

    dataset/
    ├── train.csv    (filepath, label, class_name)
    ├── val.csv
    └── test.csv
"""

import os
import csv
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


# =============================================================================
# CONFIGURATION
# =============================================================================

# Expected 28 WaRP-C class names — used for validation
WARP_C_CLASSES = [
    "bottle-blue", "bottle-blue-full", "bottle-blue5l", "bottle-blue5l-full",
    "bottle-dark", "bottle-dark-full", "bottle-green", "bottle-green-full",
    "bottle-milk", "bottle-milk-full", "bottle-multicolor", "bottle-multicolor-full",
    "bottle-oil", "bottle-oil-full", "bottle-transp", "bottle-transp-full",
    "bottle-yogurt", "canister", "cans", "detergent-box",
    "detergent-color", "detergent-transparent", "detergent-white",
    "glass-dark", "glass-green", "glass-transp",
    "juice-cardboard", "milk-cardboard"
]


# =============================================================================
# HELPERS
# =============================================================================

def find_leaf_class_folders(root: Path) -> dict:
    """
    Walk the nested WaRP-C structure and find all leaf-level class folders.
    Returns a dict of {class_name: Path} for all 28 classes.

    WaRP-C structure:
        root/
        ├── bottle/
        │   ├── bottle-blue/     ← leaf class folder we want
        │   ├── bottle-dark/
        │   └── ...
        ├── detergent/
        │   ├── detergent-box/
        │   └── ...
        └── ...
    """
    class_folders = {}

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        for class_dir in sorted(category_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            class_folders[class_name] = class_dir

    return class_folders


def get_images(class_dir: Path) -> list:
    """Return all image file paths in a class directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([
        f for f in class_dir.iterdir()
        if f.suffix.lower() in extensions
    ])


def copy_images(image_paths: list, dest_dir: Path, class_name: str):
    """
    Copy images to destination directory.
    Uses copy2 to preserve metadata.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in image_paths:
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)


def write_csv(records: list, csv_path: Path):
    """
    Write a CSV file with columns: filepath, label, class_name.
    filepath is relative to the project root.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "class_name"])
        writer.writerows(records)


def print_distribution(split_name: str, records: list, class_to_label: dict):
    """Print a class distribution table for a split."""
    counts = defaultdict(int)
    for _, label, class_name in records:
        counts[class_name] += 1

    print(f"\n  {split_name} distribution ({len(records)} images):")
    print(f"  {'Class':<30} {'Count':>6}  {'Bar'}")
    print(f"  {'-'*60}")

    max_count = max(counts.values()) if counts else 1
    for class_name in sorted(counts.keys()):
        count = counts[class_name]
        bar = "█" * int(20 * count / max_count)
        print(f"  {class_name:<30} {count:>6}  {bar}")


# =============================================================================
# MAIN
# =============================================================================

def prepare(warp_c_root: str, output_root: str, val_split: float, seed: int):

    warp_c_root = Path(warp_c_root)
    output_root = Path(output_root)
    dataset_dir = Path("dataset")

    random.seed(seed)

    print("=" * 60)
    print(" EEEM068 — WaRP-C Data Preparation")
    print("=" * 60)
    print(f"  Source:     {warp_c_root}")
    print(f"  Output:     {output_root}")
    print(f"  Val split:  {val_split:.0%}")
    print(f"  Seed:       {seed}")

    # ── Locate train and test folders ────────────────────────────────────────
    train_crops = warp_c_root / "train_crops"
    test_crops  = warp_c_root / "test_crops"

    if not train_crops.exists():
        raise FileNotFoundError(
            f"Could not find train_crops at: {train_crops}\n"
            f"Check your --warp_c_root path."
        )
    if not test_crops.exists():
        raise FileNotFoundError(
            f"Could not find test_crops at: {test_crops}\n"
            f"Check your --warp_c_root path."
        )

    # ── Find all leaf class folders ───────────────────────────────────────────
    print(f"\n[1/5] Scanning class folders...")
    train_classes = find_leaf_class_folders(train_crops)
    test_classes  = find_leaf_class_folders(test_crops)

    print(f"  Found {len(train_classes)} classes in train_crops")
    print(f"  Found {len(test_classes)} classes in test_crops")

    # validate against expected 28 classes
    missing_train = set(WARP_C_CLASSES) - set(train_classes.keys())
    missing_test  = set(WARP_C_CLASSES) - set(test_classes.keys())
    extra_train   = set(train_classes.keys()) - set(WARP_C_CLASSES)

    if missing_train:
        print(f"\n  WARNING: Missing from train_crops: {sorted(missing_train)}")
    if missing_test:
        print(f"\n  WARNING: Missing from test_crops:  {sorted(missing_test)}")
    if extra_train:
        print(f"\n  NOTE: Extra folders found (not in expected 28): {sorted(extra_train)}")

    # build class → label mapping (sorted alphabetically for consistency)
    all_classes    = sorted(set(list(train_classes.keys()) + list(test_classes.keys())))
    class_to_label = {cls: idx for idx, cls in enumerate(all_classes)}

    print(f"\n  Class → label mapping:")
    for cls, lbl in class_to_label.items():
        print(f"    [{lbl:>2}] {cls}")

    # ── Split train into train/val ────────────────────────────────────────────
    print(f"\n[2/5] Splitting train_crops into train ({1-val_split:.0%}) / val ({val_split:.0%})...")

    train_images, val_images = [], []

    for class_name, class_dir in sorted(train_classes.items()):
        images = get_images(class_dir)
        random.shuffle(images)
        n_val   = max(1, int(len(images) * val_split))
        val_images.extend(images[:n_val])
        train_images.extend(images[n_val:])

    print(f"  Train images: {len(train_images)}")
    print(f"  Val images:   {len(val_images)}")

    # ── Copy images to flat output structure ──────────────────────────────────
    print(f"\n[3/5] Copying images to flat output structure...")
    print(f"  (This may take a minute — copying {len(train_images) + len(val_images)} train/val images)")

    # train
    for img_path in train_images:
        class_name = img_path.parent.name
        copy_images([img_path], output_root / "train" / class_name, class_name)

    # val
    for img_path in val_images:
        class_name = img_path.parent.name
        copy_images([img_path], output_root / "val" / class_name, class_name)

    # test — copy all test images
    test_images = []
    for class_name, class_dir in sorted(test_classes.items()):
        images = get_images(class_dir)
        copy_images(images, output_root / "test" / class_name, class_name)
        test_images.extend(images)

    print(f"  Test images:  {len(test_images)}")
    print(f"  Output structure created at: {output_root}/")

    # ── Generate CSV files ────────────────────────────────────────────────────
    print(f"\n[4/5] Generating CSV files...")

    def make_records(images, split):
        records = []
        for img_path in sorted(images):
            class_name = img_path.parent.name
            label      = class_to_label[class_name]
            # filepath relative to project root
            rel_path   = str(output_root / split / class_name / img_path.name)
            records.append((rel_path, label, class_name))
        return records

    train_records = make_records(train_images, "train")
    val_records   = make_records(val_images,   "val")
    test_records  = make_records(test_images,  "test")

    write_csv(train_records, dataset_dir / "train.csv")
    write_csv(val_records,   dataset_dir / "val.csv")
    write_csv(test_records,  dataset_dir / "test.csv")

    print(f"  dataset/train.csv  ({len(train_records)} rows)")
    print(f"  dataset/val.csv    ({len(val_records)} rows)")
    print(f"  dataset/test.csv   ({len(test_records)} rows)")

    # ── Save class mapping to JSON ────────────────────────────────────────────
    mapping = {
        "class_to_label": class_to_label,
        "label_to_class": {str(v): k for k, v in class_to_label.items()},
        "n_classes": len(class_to_label)
    }
    mapping_path = dataset_dir / "class_mapping.json"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  dataset/class_mapping.json")

    # ── Distribution report ───────────────────────────────────────────────────
    print(f"\n[5/5] Class distribution report:")
    print_distribution("Train", train_records, class_to_label)
    print_distribution("Val",   val_records,   class_to_label)
    print_distribution("Test",  test_records,  class_to_label)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Train: {len(train_records):>5} images across {len(class_to_label)} classes")
    print(f"  Val:   {len(val_records):>5} images across {len(class_to_label)} classes")
    print(f"  Test:  {len(test_records):>5} images across {len(class_to_label)} classes")
    print(f"  Total: {len(train_records)+len(val_records)+len(test_records):>5} images")
    print(f"\n  Next step: python train.py --config configs/experiments/smoke_test.yaml")
    print(f"{'='*60}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare WaRP-C dataset for training"
    )
    parser.add_argument(
        "--warp_c_root",
        type=str,
        default="Dataset/Warp-C",
        help="Path to the Warp-C folder (default: Dataset/Warp-C)"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data",
        help="Where to write the flat train/val/test structure (default: data)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of training data to use for validation (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    args = parser.parse_args()

    prepare(
        warp_c_root=args.warp_c_root,
        output_root=args.output_root,
        val_split=args.val_split,
        seed=args.seed
    )
