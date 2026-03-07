# this is a class that will explore the data that we have initialy so we can detect trends, class imbalance ...
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm



class EDAModule:
    """
    Encapsulates all EDA steps for the WaRP-C waste classification dataset.
    """
    # here I am just defining where the stats and figures will be saved and also the the data location
    # you can pass them as where you want to save them or you can use the default implementation
    def __init__(
        self,
        data_root:   str | Path = "Dataset/raw/WaRP-C",
        figures_dir: str | Path = "Dataset/figures",
        stats_file:  str | Path = "Dataset/dataset_stats.json",
        seed: int = 42,
    ):
        self.data_root   = Path(data_root)
        self.train_dir   = self.data_root / "train_crops"
        self.test_dir    = self.data_root / "test_crops"
        self.figures_dir = Path(figures_dir)
        self.stats_file  = Path(stats_file)
        self.seed        = seed
        self.parent_classes = defaultdict(set)

        random.seed(seed)
        np.random.seed(seed)

        # This is done because we want to check on data balanace between the test and train also
        self._train_data: dict[str, list[Path]] = {}
        self._test_data:  dict[str, list[Path]] = {}
        # This will be the dictionary that will collect all the stats from different methods and also will be saved as a json file to reuse it
        self._stats: dict = {}        

        # this is to v alidate dataset exists, then load
        self._validate_dataset()
        self._load_paths()
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        print(f"[EDAModule] Dataset loaded from: {self.data_root}")
        print(f"[EDAModule] Train classes : {len(self._train_data)}")
        print(f"[EDAModule] Test  classes : {len(self._test_data)}")
        print(f"[EDAModule] Figures → {self.figures_dir}")


    def _validate_dataset(self) -> None:
        """This is just to check if the paths given actually exists"""
        if not self.train_dir.exists() or not self.test_dir.exists():
            raise FileNotFoundError(
                f"\n[EDAModule] Dataset not found.\n"
                f"  Expected train dir : {self.train_dir}\n"
                f"  Expected test  dir : {self.test_dir}\n"
                f"  Please check data_root or re-run download_data.py"
            )

    def _load_paths(self) -> None:
        def _scan(root: Path) -> dict[str, list[Path]]:
            data = defaultdict(list)
            for cls_dir in sorted(root.iterdir()):
                for cls_dir1 in sorted(cls_dir.iterdir()):
                    # store the parent classes too
                    self.parent_classes[cls_dir.name].add(cls_dir1.name)
                    if cls_dir1.is_dir():
                        # we will have something like this: ['bottle-blue':[list of images]), ...]
                        data[cls_dir1.name] = sorted(
                            p for p in cls_dir1.iterdir()
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                        )

            return dict(data)

        self._train_data = _scan(self.train_dir)
        self._test_data  = _scan(self.test_dir)

    def _save_fig(self, fig: plt.Figure, filename: str) -> Path:
        # thit is a general function used to save figures 
        out = self.figures_dir / filename
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved → {out}")
        return out

    @property
    def classes(self) -> list[str]:
        # this returns the classes that we have in our dataset (the labels)
        return sorted(self._train_data.keys())


    def plot_class_distribution(self) -> dict:
        """
        Bar chart: image counts per class for train and test, plus an
        imbalance-ratio chart.

        WHY we are doing this: because based on the literature review and what we have studied in the lectures
        Class imbalance biases models toward majority classes by ignoring the minority do the accuracy matrix actually
        does not give us a clear definition if the model is performing well or not
        We compute the imbalance ratio = max_count / min_count.

        Decision rules: it was based on this papers https://doi.org/10.1016/j.neunet.2018.07.011

        We produce ONE figure per parent group so the 28 classes are never
        crammed into a single unreadable chart.  Each figure shows:
          top subplot   : raw train / test counts for that group's sub-classes
          bottom subplot: per-sub-class imbalance ratio relative to the
                          GLOBAL training mean (so ratios are comparable
                          across groups)
        """
        classes      = self.classes
        train_counts = [len(self._train_data.get(c, [])) for c in classes]
        test_counts  = [len(self._test_data.get(c,  [])) for c in classes]

        # we need the global mean so every group's ratio bar uses the same baseline
        # this lets us compare imbalance severity across groups fairly
        global_train_mean = np.mean(train_counts)
        global_test_mean  = np.mean(test_counts)

        # ── one figure per parent group ───────────────────────────────────────
        # parent_classes is a dict built in _load_paths:
        #   { 'bottle': {'bottle-blue', 'bottle-dark', ...},
        #     'glass':  {'glass-dark', ...}, ... }
        for parent, children in sorted(self.parent_classes.items()):

            # keep only children that actually appear in our loaded data
            # and sort them so the bars are always in alphabetical order
            group_classes = sorted(c for c in children if c in self._train_data)
            if not group_classes:
                continue

            g_train = [len(self._train_data.get(c, [])) for c in group_classes]
            g_test  = [len(self._test_data.get(c,  [])) for c in group_classes]

            x     = np.arange(len(group_classes))
            width = 0.4

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(group_classes) * 1.6), 12))
            fig.suptitle(
                f"WaRP-C — Class Distribution: '{parent}' group",
                fontsize=14, fontweight="bold",
            )

            # ── top: raw counts ──────────────────────────────────────────────
            b1 = ax1.bar(x - width/2, g_train, width,
                         label="Train", color="#2196F3", alpha=0.85)
            ax1.bar(x + width/2, g_test, width,
                    label="Test",  color="#FF5722", alpha=0.85)

            # draw the GLOBAL mean lines so every group chart uses the same reference
            ax1.axhline(global_train_mean, color="#2196F3", linestyle="--", alpha=0.6,
                        label=f"Global train mean ({global_train_mean:.0f})")
            ax1.axhline(global_test_mean,  color="#FF5722", linestyle="--", alpha=0.6,
                        label=f"Global test mean  ({global_test_mean:.0f})")

            # annotate each train bar with its exact count
            for bar in b1:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=8, rotation=0,
                )

            ax1.set_xticks(x)
            ax1.set_xticklabels(group_classes, rotation=30, ha="right", fontsize=9)
            ax1.set_ylabel("Number of images")
            ax1.legend()

            # ── bottom: imbalance ratio vs GLOBAL mean ───────────────────────
            # ratio > 1 → this sub-class is above the dataset average (green)
            # ratio < 1 → this sub-class is below the dataset average (red)
            imbalance = [c / global_train_mean for c in g_train]
            colors    = ["#4CAF50" if r >= 1.0 else "#F44336" for r in imbalance]
            ax2.bar(x, imbalance, color=colors, alpha=0.85, edgecolor="white")
            ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.5,
                        label="Global mean (ratio = 1)")
            ax2.set_xticks(x)
            ax2.set_xticklabels(group_classes, rotation=30, ha="right", fontsize=9)
            ax2.set_ylabel("Ratio to global mean count")
            ax2.set_title(
                "Per-class imbalance ratio  —  green ≥ global mean  |  red < global mean",
                fontsize=10,
            )
            ax2.legend()

            plt.tight_layout()
            # save as  01_class_distribution_bottle.png,  01_class_distribution_glass.png  etc.
            self._save_fig(fig, f"01_class_distribution_{parent}.png")

        # ── compute global imbalance stats (unchanged from before) ───────────
        max_c = max(train_counts)
        min_c = min(train_counts)
        ratio = round(max_c / min_c, 2)

        result = {
            "class_names":     classes,
            "train_counts":    train_counts,
            "test_counts":     test_counts,
            "imbalance_ratio": ratio,
            "total_train":     sum(train_counts),
            "total_test":      sum(test_counts),
            "majority_class":  classes[train_counts.index(max_c)],
            "minority_class":  classes[train_counts.index(min_c)],
        }
        self._stats["class_distribution"] = result

        print(f"\n  Imbalance ratio : {ratio}:1")
        print(f"  Majority class  : {result['majority_class']} ({max_c} images)")
        print(f"  Minority class  : {result['minority_class']} ({min_c} images)")
        print(f"  Total train     : {result['total_train']}")
        print(f"  Total test      : {result['total_test']}")

        # this is based on the paper https://doi.org/10.1016/j.neunet.2018.07.011
        if ratio < 5:
            print("  → Mild imbalance. Weighted loss function should suffice.")
        elif ratio < 50:
            print("  → Moderate imbalance. Use WeightedRandomSampler + weighted loss.")
        else:
            print("  → Severe imbalance. Consider oversampling minority classes too.")

        return result


    def plot_sample_grid(self, n_samples: int = 5) -> None:
        """
        Show ``n_samples`` random images per class in a grid.

        WHY THIS MATTERS
        ----------------
        Statistics alone never show you what the model actually sees.
        A quick visual scan reveals:
          • Classes that look nearly identical → expect confusion-matrix errors
          • Images that are mostly background with a tiny cropped object
          • Corrupted / extremely dark / blurry images
          • Inconsistent crop quality across classes

        These observations feed directly into augmentation design and help
        you predict which class pairs will dominate your confusion matrix.

        Parameters
        ----------
        n_samples : int
            Number of random images to show per class (default 5).
        """
        # one grid per parent group — same readability fix as plot_class_distribution
        for parent, children in sorted(self.parent_classes.items()):
            group_classes = sorted(c for c in children if c in self._train_data)
            if not group_classes:
                continue

            n_classes = len(group_classes)
            fig, axes = plt.subplots(
                n_classes, n_samples,
                figsize=(n_samples * 2.2, n_classes * 2.0),
                squeeze=False,   # always return 2-D array even for 1 class
            )
            fig.suptitle(
                f"WaRP-C — Random samples: '{parent}' group",
                fontsize=13, fontweight="bold", y=1.002,
            )

            for row, cls in enumerate(group_classes):
                paths   = self._train_data[cls]
                samples = random.sample(paths, min(n_samples, len(paths)))

                for col, img_path in enumerate(samples):
                    ax = axes[row, col]
                    try:
                        ax.imshow(Image.open(img_path).convert("RGB"))
                    except Exception:
                        ax.text(0.5, 0.5, "ERR", ha="center",
                                va="center", color="red", transform=ax.transAxes)
                    ax.axis("off")
                    if col == 0:
                        ax.set_ylabel(
                            cls, rotation=0, labelpad=70,
                            fontsize=8, fontweight="bold", va="center",
                        )

            plt.tight_layout()
            self._save_fig(fig, f"02_sample_grid_{parent}.png")


    def analyze_image_sizes(self) -> dict:
        """
        Histograms of image width, height and aspect ratio across the full
        dataset (train + test).

        WHY THIS MATTERS
        ----------------
        WaRP-C images are bounding-box crops, so they are NOT all the same
        size.  Before choosing a resize target we need to know:

        1. Typical size → informs target resolution (224 vs 256 vs 320).
           Rule of thumb: use the 75th-percentile of the shorter side as
           the resize target.  Upscaling by more than 3× introduces artefacts.

        2. Aspect ratio spread → if images are highly non-square, a plain
           Resize(224, 224) distorts shapes.  Alternatives:
             - Pad to square first, then resize  (preserves shape exactly)
             - RandomResizedCrop  (standard for classification, handles it
               naturally and adds implicit augmentation)

        Returns
        -------
        dict with per-axis (width / height / aspect_ratio) stats:
        mean, median, min, max, p25, p75
        """
        all_paths = [
            p
            for data in (self._train_data, self._test_data)
            for paths in data.values()
            for p in paths
        ]

        widths, heights, ratios = [], [], []
        for img_path in tqdm(all_paths, desc="  Scanning image sizes"):
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    ratios.append(w / h)
            except Exception:
                pass

        widths  = np.array(widths,  dtype=float)
        heights = np.array(heights, dtype=float)
        ratios  = np.array(ratios,  dtype=float)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("WaRP-C — Image size analysis", fontsize=14, fontweight="bold")

        def _hist(ax, data, color, xlabel, title):
            ax.hist(data, bins=60, color=color, alpha=0.85, edgecolor="white")
            ax.axvline(np.median(data), color="red",    linestyle="--",
                       label=f"Median {np.median(data):.0f}")
            ax.axvline(np.mean(data),   color="orange", linestyle="--",
                       label=f"Mean   {np.mean(data):.0f}")
            ax.set_xlabel(xlabel); ax.set_title(title); ax.legend()

        _hist(axes[0], widths,  "#2196F3", "Width (px)",          "Image widths")
        _hist(axes[1], heights, "#4CAF50", "Height (px)",         "Image heights")
        _hist(axes[2], ratios,  "#FF5722", "Width / Height",      "Aspect ratios")
        axes[2].axvline(1.0, color="black", linewidth=2, label="Square (1:1)")
        axes[2].legend()

        plt.tight_layout()
        self._save_fig(fig, "03_image_sizes.png")

        def _stat(arr):
            return {
                "mean":   round(float(np.mean(arr)),   1),
                "median": round(float(np.median(arr)), 1),
                "min":    round(float(np.min(arr)),    1),
                "max":    round(float(np.max(arr)),    1),
                "p25":    round(float(np.percentile(arr, 25)), 1),
                "p75":    round(float(np.percentile(arr, 75)), 1),
            }

        result = {
            "width":        _stat(widths),
            "height":       _stat(heights),
            "aspect_ratio": _stat(ratios),
        }
        self._stats["image_sizes"] = result

        print(f"\n  Width  median={result['width']['median']}px   "
              f"range=[{result['width']['min']:.0f}, {result['width']['max']:.0f}]")
        print(f"  Height median={result['height']['median']}px   "
              f"range=[{result['height']['min']:.0f}, {result['height']['max']:.0f}]")
        print(f"  Aspect median={result['aspect_ratio']['median']}   "
              f"range=[{result['aspect_ratio']['min']:.2f}, {result['aspect_ratio']['max']:.2f}]")

        # Recommendation
        rec_size = int(np.percentile(np.minimum(widths, heights), 75))
        # Round up to nearest multiple of 32 (GPU-friendly)
        rec_size = int(np.ceil(rec_size / 32) * 32)
        print(f"\n  Recommended resize target : {rec_size}px  "
              f"(75th-pct of shorter side, rounded to nearest 32)")
        result["recommended_resize"] = rec_size

        return result


    def compute_pixel_stats(self, sample_size: int = 1500) -> dict:
        """
        Compute per-channel (R, G, B) mean and standard deviation and compare
        with standard ImageNet statistics.

        WHY THIS MATTERS
        ----------------
        Every pretrained backbone was trained on ImageNet-normalised inputs.
        Fine-tuning works best when you normalise the same way.

        Option A  Use ImageNet stats (safe default for pretrained models):
                    mean=[0.485, 0.456, 0.406]  std=[0.229, 0.224, 0.225]

        Option B  Compute WaRP-C specific stats (better if the domain
                  diverges significantly from natural photos — industrial
                  waste on a conveyor belt is quite different).

        Decision rule used here:
          If mean absolute deviation from ImageNet < 0.05 → ImageNet stats OK.
          Otherwise → use WaRP-C stats in Normalize().

        We sample ``sample_size`` training images (fast estimate).  For the
        final transforms.py you can increase this to all 8 823 images.

        Parameters
        ----------
        sample_size : int
            Number of training images to sample (default 1 500).

        Returns
        -------
        dict with warp_mean, warp_std, imagenet_mean, imagenet_std,
        mean_diff, recommendation
        """
        all_train = [p for paths in self._train_data.values() for p in paths]
        sampled   = random.sample(all_train, min(sample_size, len(all_train)))

        chan_sum    = np.zeros(3)
        chan_sq_sum = np.zeros(3)
        n = 0

        for img_path in tqdm(sampled, desc="  Computing pixel stats"):
            try:
                arr = (
                    np.array(
                        Image.open(img_path).convert("RGB").resize((224, 224)),
                        dtype=np.float32,
                    ) / 255.0
                )                              # shape (224, 224, 3), range [0,1]
                chan_sum    += arr.mean(axis=(0, 1))
                chan_sq_sum += (arr ** 2).mean(axis=(0, 1))
                n += 1
            except Exception:
                pass

        mean = chan_sum    / n
        std  = np.sqrt(chan_sq_sum / n - mean ** 2)

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std  = np.array([0.229, 0.224, 0.225])
        diff = float(np.abs(mean - imagenet_mean).mean())

        # ── visualisation ────────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("WaRP-C vs ImageNet normalisation statistics",
                     fontsize=13, fontweight="bold")

        ch_labels = ["Red", "Green", "Blue"]
        x         = np.arange(3)
        w         = 0.35
        ch_colors = ["#F44336", "#4CAF50", "#2196F3"]

        ax1.bar(x - w/2, mean,          w, label="WaRP-C",   color=ch_colors, alpha=0.9)
        ax1.bar(x + w/2, imagenet_mean, w, label="ImageNet", color=ch_colors, alpha=0.4)
        ax1.set_xticks(x); ax1.set_xticklabels(ch_labels)
        ax1.set_ylim(0, 0.7); ax1.set_ylabel("Mean pixel value (0–1)")
        ax1.set_title("Channel means"); ax1.legend()

        ax2.bar(x - w/2, std,          w, label="WaRP-C",   color=ch_colors, alpha=0.9)
        ax2.bar(x + w/2, imagenet_std, w, label="ImageNet", color=ch_colors, alpha=0.4)
        ax2.set_xticks(x); ax2.set_xticklabels(ch_labels)
        ax2.set_ylim(0, 0.4); ax2.set_ylabel("Std of pixel values (0–1)")
        ax2.set_title("Channel standard deviations"); ax2.legend()

        plt.tight_layout()
        self._save_fig(fig, "04_pixel_stats.png")

        # ── recommendation ───────────────────────────────────────────────────
        rec = (
            "Use ImageNet stats — WaRP-C is close enough for pretrained models."
            if diff < 0.05 else
            "Use WaRP-C stats — notable difference from ImageNet domain."
        )

        result = {
            "warp_mean":      mean.tolist(),
            "warp_std":       std.tolist(),
            "imagenet_mean":  imagenet_mean.tolist(),
            "imagenet_std":   imagenet_std.tolist(),
            "mean_diff":      round(diff, 5),
            "recommendation": rec,
        }
        self._stats["pixel_stats"] = result

        print(f"\n  WaRP-C mean  : {[round(v, 3) for v in mean]}")
        print(f"  WaRP-C std   : {[round(v, 3) for v in std]}")
        print(f"  ImageNet mean: {imagenet_mean.tolist()}")
        print(f"  ImageNet std : {imagenet_std.tolist()}")
        print(f"  Mean deviation from ImageNet: {diff:.4f}")
        print(f"  → {rec}")

        return result


    def plot_train_test_comparison(self) -> dict:
        """
        Compare per-class proportion (%) between the train and test splits.

        WHY THIS MATTERS
        ----------------
        A well-constructed dataset has matching class proportions in both
        splits (stratified split).  If they diverge:
          • A class rare in test but common in train → model over-trains on it
            but evaluation barely tests it.
          • A class common in test but rare in train → model never learned it
            properly, dragging down test metrics.

        We flag any class whose proportion deviates by more than 2 percentage
        points — worth mentioning in the report.

        Returns
        -------
        dict with per-class percentages and the max deviation found.
        """
        classes     = self.classes
        total_train = sum(len(v) for v in self._train_data.values())
        total_test  = sum(len(v) for v in self._test_data.values())

        train_pct = [len(self._train_data.get(c, [])) / total_train * 100
                     for c in classes]
        test_pct  = [len(self._test_data.get(c,  [])) / total_test  * 100
                     for c in classes]

        # one figure per parent group — same readability fix
        for parent, children in sorted(self.parent_classes.items()):
            group_classes = sorted(c for c in children if c in self._train_data)
            if not group_classes:
                continue

            idx         = [classes.index(c) for c in group_classes]
            g_train_pct = [train_pct[i] for i in idx]
            g_test_pct  = [test_pct[i]  for i in idx]

            x = np.arange(len(group_classes))
            w = 0.4

            fig, ax = plt.subplots(figsize=(max(8, len(group_classes) * 1.6), 5))
            ax.bar(x - w/2, g_train_pct, w, label="Train %", color="#2196F3", alpha=0.85)
            ax.bar(x + w/2, g_test_pct,  w, label="Test %",  color="#FF5722", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(group_classes, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Share of split (%)")
            ax.set_title(
                f"Class proportion Train vs Test — '{parent}' group\n"
                "(identical bars = perfectly stratified split)",
                fontsize=11, fontweight="bold",
            )
            ax.legend()
            plt.tight_layout()
            self._save_fig(fig, f"05_train_test_split_{parent}.png")

        deviations    = [abs(a - b) for a, b in zip(train_pct, test_pct)]
        max_dev       = max(deviations)
        max_dev_class = classes[deviations.index(max_dev)]

        result = {
            "train_pct":      train_pct,
            "test_pct":       test_pct,
            "max_deviation":  round(max_dev, 3),
            "max_dev_class":  max_dev_class,
        }
        self._stats["train_test_split"] = result

        print(f"\n  Max proportion deviation : {max_dev:.2f}%  "
              f"(class '{max_dev_class}')")
        print(
            "  → Split looks well-stratified." if max_dev < 2.0
            else "  → Notable deviation — mention in the report."
        )

        return result


    def check_duplicates(self) -> dict:
        """
        Filename-level duplicate check between the train and test sets.

        WHY THIS MATTERS
        ----------------
        If the same image appears in both splits (data leakage), test accuracy
        is artificially inflated — the model has already seen the test images.
        Filename matching is O(n) and catches obvious leakage; a deeper check
        would use MD5 or perceptual hashing (see imagehash library).

        Returns
        -------
        dict with counts and up to 10 example duplicate filenames.
        """
        train_names = {p.name for paths in self._train_data.values() for p in paths}
        test_names  = {p.name for paths in self._test_data.values()  for p in paths}
        overlap     = train_names & test_names

        result = {
            "train_unique":      len(train_names),
            "test_unique":       len(test_names),
            "overlap_count":     len(overlap),
            "overlap_examples":  sorted(overlap)[:10],
        }
        self._stats["duplicates"] = result

        print(f"\n  Train unique filenames : {len(train_names)}")
        print(f"  Test  unique filenames : {len(test_names)}")
        print(f"  Overlap count          : {len(overlap)}")

        if len(overlap) == 0:
            print("  → No duplicates found between train and test.  ✓")
        else:
            print(f"  ⚠  WARNING: {len(overlap)} duplicate filenames — "
                  f"potential data leakage!")
            print(f"  Examples: {sorted(overlap)[:5]}")

        return result


    def plot_brightness_per_class(self, n_per_class: int = 60) -> dict:
        """
        Mean luminance (brightness) per class, sampled from training images.

        WHY THIS MATTERS
        ----------------
        WaRP-C is filmed under industrial conveyor-belt lighting that varies
        significantly.  Some categories are inherently dark (dark glass,
        dark bottles); others are bright (white detergent, full bottles).

        This matters because:
          1. If two visually similar classes differ mainly in brightness,
             the model may learn brightness as a shortcut — which won't
             generalise to new lighting conditions.
          2. The range of brightness values tells us how strong to make the
             ``brightness`` parameter in ``ColorJitter``.  A wide range means
             we need aggressive augmentation to make the model lighting-invariant.

        Parameters
        ----------
        n_per_class : int
            Number of images sampled per class for efficiency (default 60).

        Returns
        -------
        dict with per-class brightness values plus darkest/brightest class.
        """
        classes    = self.classes
        brightness = {}

        for cls in tqdm(classes, desc="  Brightness per class"):
            paths   = self._train_data[cls]
            sampled = random.sample(paths, min(n_per_class, len(paths)))
            vals = []
            for p in sampled:
                try:
                    arr = np.array(
                        Image.open(p).convert("L"), dtype=np.float32
                    )
                    vals.append(arr.mean() / 255.0)
                except Exception:
                    pass
            brightness[cls] = round(float(np.mean(vals)), 4) if vals else 0.0

        # one brightness chart per parent group
        for parent, children in sorted(self.parent_classes.items()):
            group_classes = sorted(c for c in children if c in brightness)
            if not group_classes:
                continue

            mean_vals = [brightness[c] for c in group_classes]
            colors    = plt.cm.RdYlGn(np.array(mean_vals))  # red=dark, green=bright

            fig, ax = plt.subplots(figsize=(max(8, len(group_classes) * 1.6), 5))
            ax.bar(group_classes, mean_vals, color=colors, alpha=0.9, edgecolor="white")
            ax.axhline(
                np.mean(mean_vals), color="black", linestyle="--",
                label=f"Group mean: {np.mean(mean_vals):.3f}",
            )
            ax.set_xticklabels(group_classes, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Mean brightness  (0 = black, 1 = white)")
            ax.set_title(
                f"Per-class brightness — '{parent}' group\n"
                "red = dark  |  green = bright",
                fontsize=11, fontweight="bold",
            )
            ax.legend()
            plt.tight_layout()
            self._save_fig(fig, f"06_brightness_{parent}.png")

        darkest   = min(brightness, key=brightness.get)
        brightest = max(brightness, key=brightness.get)
        all_vals  = list(brightness.values())

        result = {
            "brightness_per_class": brightness,
            "darkest_class":        darkest,
            "brightest_class":      brightest,
            "dataset_mean":         round(float(np.mean(all_vals)), 4),
            "dataset_std":          round(float(np.std(all_vals)),  4),
        }
        self._stats["brightness"] = result

        print(f"\n  Darkest class  : '{darkest}'  ({brightness[darkest]:.3f})")
        print(f"  Brightest class: '{brightest}' ({brightness[brightest]:.3f})")
        print(f"  Brightness std across classes: {result['dataset_std']:.3f}")
        if result["dataset_std"] > 0.08:
            print("  → High brightness variance — use strong ColorJitter augmentation.")
        else:
            print("  → Moderate brightness variance — standard ColorJitter is fine.")

        return result


    def save_stats(self) -> Path:
        """
        Write all accumulated stats to ``self.stats_file`` as JSON.

        This file is loaded by ``Pipeline_/dataset.py`` so that class weights,
        normalization values, and other derived numbers are computed only once
        (here) rather than recomputed on every training run.

        Returns
        -------
        Path to the written JSON file.
        """
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_file, "w") as f:
            json.dump(self._stats, f, indent=2)
        print(f"\n  ✓ Stats saved → {self.stats_file}")
        return self.stats_file


    def summary(self) -> None:
        """
        Print a concise summary table of all findings.
        Call this after running all analysis methods.
        """
        sep = "=" * 62
        print(f"\n{sep}")
        print("  EDA COMPLETE — WaRP-C Summary")
        print(sep)

        dist  = self._stats.get("class_distribution", {})
        sizes = self._stats.get("image_sizes", {})
        px    = self._stats.get("pixel_stats", {})
        dup   = self._stats.get("duplicates", {})

        if dist:
            print(f"  Total training images  : {dist.get('total_train', '?')}")
            print(f"  Total test images      : {dist.get('total_test', '?')}")
            print(f"  Number of classes      : {len(dist.get('class_names', []))}")
            print(f"  Class imbalance ratio  : {dist.get('imbalance_ratio', '?')}:1")

        if sizes:
            w = sizes.get("width",  {})
            h = sizes.get("height", {})
            r = sizes.get("aspect_ratio", {})
            print(f"  Median image size      : "
                  f"{w.get('median', '?')} × {h.get('median', '?')} px")
            print(f"  Aspect ratio (median)  : {r.get('median', '?')}")
            print(f"  Recommended resize     : "
                  f"{sizes.get('recommended_resize', '?')} px")

        if px:
            print(f"  WaRP-C mean (R,G,B)    : "
                  f"{[round(v,3) for v in px.get('warp_mean', [])]}")
            print(f"  WaRP-C std  (R,G,B)    : "
                  f"{[round(v,3) for v in px.get('warp_std', [])]}")
            print(f"  Normalisation advice   : {px.get('recommendation', '')}")

        if dup:
            status = "✓ Clean" if dup.get("overlap_count", 1) == 0 else "⚠ Leakage detected"
            print(f"  Train/test leakage     : {status}")

        print(f"\n  Figures saved to : {self.figures_dir}/")
        print(f"  Stats saved to   : {self.stats_file}")
        print(f"\n  Next step → Pipeline_/transforms.py")
        print(sep)