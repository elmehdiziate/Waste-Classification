# Weekly Training Log — EEEM068
**Week:** 7 — 20 Mar – 26 Mar 2026
**Member:** Scott Lewis

---

## 1. Work Completed This Week

- [x] Cloned repository and set up local development environment (Windows, VS Code)
- [x] Created and configured Python virtual environment with CUDA-enabled PyTorch (cu128)
- [x] Verified GPU availability — NVIDIA RTX 3060m confirmed working with PyTorch
- [x] Investigated and resolved WaRP-C dataset structure mismatch
- [x] Wrote `Dataset/prepare_data.py` to flatten nested WaRP-C structure and generate CSV splits
- [x] Designed and implemented YAML-based experiment config system (`base.yaml` + experiment overrides)
- [x] Implemented `train.py` — full training pipeline
- [x] Implemented `test.py` — evaluation pipeline
- [x] Generated 52 experiment configs across 4 models (Phase 1 grid + Phase 2 fine-tuning)
- [x] Generated 10 runner scripts (.ps1 for Windows, .sh for Linux/Mac)
- [x] Ran and verified smoke test end-to-end on GPU
- [x] Updated `README.md` with full project setup and usage instructions
- [x] Configured `.gitignore` to exclude data, model weights, and experiment results

---

## 2. Experiments Run

### Experiment A — Smoke Test (Pipeline Verification)

| Setting | Value |
|---------|-------|
| Model | ResNet50 |
| Backbone | resnet50 |
| Input resolution | 224 × 224 |
| Batch size | 8 |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| LR schedule | CosineAnnealingLR |
| Epochs | 2 |
| Loss function | Weighted CrossEntropyLoss |
| Augmentations | RandomResizedCrop, RandomHorizontalFlip, ColorJitter |
| Class imbalance strategy | Weighted CrossEntropyLoss (inverse class frequency) |
| Hardware | NVIDIA RTX 3060m (6GB VRAM), CUDA 12.8 |
| Training time | ~95 seconds per epoch (GPU) vs ~22 minutes per epoch (CPU) |
| Pretrained | False (ImageNet weights disabled for speed) |

**Training results:**

| Metric | Value |
|--------|-------|
| Best val F1 (macro) | 0.1465 |
| Best epoch | 2 |

**Test results (test.py):**

| Metric | Value |
|--------|-------|
| Accuracy | 0.2186 |
| F1 (macro) | 0.1704 |
| F1 (weighted) | 0.1896 |
| Precision (macro) | 0.2558 |
| Recall (macro) | 0.2183 |
| AUC | 0.8155 |
| mAP | 0.2126 |

**Observations:**

- Pipeline verified end-to-end: data loading → training → checkpoint saving → evaluation → plots
- Metrics are expected for 2 epochs with no pretrained weights on a 28-class problem
- Random chance baseline = ~3.6% accuracy — model at 21.9% shows meaningful learning
- AUC of 0.8155 is particularly encouraging given the minimal training
- GPU speedup of ~14× over CPU confirms local training is viable for Phase 1 grid
- At ~95s per epoch, full 30-epoch Phase 1 run ≈ 47 mins per config on local GPU

---

## 3. Issues Encountered

| Issue | Status | Notes |
|-------|--------|-------|
| WaRP-C nested folder structure incompatible with ImageFolder | Resolved | Wrote prepare_data.py to flatten structure |
| `pin_memory=True` warning on CPU | Resolved | Made conditional on `torch.cuda.is_available()` |
| `num_workers=4` issues on Windows | Resolved | Set to 0 on Windows, 4 on Linux/Mac automatically |
| `KeyError: save_loss_curves` | Resolved | Missing keys added to base.yaml output block |
| `TypeError: object of type int has no len()` in test.py | Resolved | Passed `class_names` list instead of `n_classes` int to confusion matrix |
| `load_class_mapping` not defined in test.py | Resolved | Added to import from train.py |
| PyTorch installed without CUDA support | Resolved | Reinstalled with `--index-url https://download.pytorch.org/whl/cu128` |
| Dataset/ folder excluded from Git by overly broad .gitignore | Resolved | Excluded specific subfolders (Warp-C/, raw/) instead of parent |
| 10,000+ files staged before .gitignore fix | Resolved | Ran `git rm -r --cached .` then re-staged |

---

## 4. Key Observations & Analysis

**GPU vs CPU performance:**
Local training on the RTX 3060m is ~14× faster than CPU. At ~95s per epoch, the full Phase 1 grid (48 runs × 30 epochs) is approximately 38 hours locally — feasible spread across multiple overnight sessions, with Google Colab as a backup for parallel runs.

**Dataset structure:**
WaRP-C uses a two-level hierarchy (`category/class/images`) rather than the flat `class/images` structure expected by PyTorch's ImageFolder. `prepare_data.py` resolves this by copying images into a flat structure and generating CSV splits. The val split (20% of train_crops, stratified per class, seed=42) is committed to Git via `dataset/val.csv` to ensure all team members use identical splits.

**Config system:**
The YAML config system separates locked constants (`base.yaml`) from per-experiment variables (experiment configs). This ensures fair cross-model comparison during Phase 1 and provides full reproducibility — a copy of the merged config is saved alongside every run's outputs.

**Class imbalance:**
Preliminary inspection of the dataset suggests significant class imbalance across the 28 categories. Weighted CrossEntropyLoss (inverse class frequency) is implemented as the default strategy. This will be investigated further in the EDA notebook.

---

## 5. Plan for Next Week

- [ ] Present infrastructure to team and agree on final model selection
- [ ] Assign model ownership (one backbone per member)
- [ ] Team members update `author` field in their model's configs
- [ ] Begin EDA notebook — class distribution, image statistics, sample visualisations
- [ ] Run Phase 1 reference run (R8: lr=1e-4, bs=64) for ResNet50 with pretrained=True
- [ ] Set up Google Drive output folder for sharing results across team

---

## 6. Commits Made This Week

| Commit | Description |
|--------|-------------|
| TBD | Project infrastructure setup — configs, scripts, data pipeline, README |

---

*Log submitted by: Scott Lewis on 21 Mar 2026*
