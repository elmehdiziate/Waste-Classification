"""
EEEM068: Industrial Waste Classification
test.py — Evaluation pipeline

USAGE:
    python test.py --run experiments/results/smoke_test/ # smoke test!
    python test.py --run experiments/results/resnet50_phase1_lr1e-4_bs64/
    python test.py --run experiments/results/resnet50_phase1_lr1e-4_bs64/ --gradcam

Loads the config saved during training, restores the best checkpoint,
and runs full evaluation on the test set. Saves all metrics and plots
back into the same run folder.
"""

import os
import json
import argparse

from sklearn import metrics
import yaml
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# reuse model builder from train.py
from train import get_model, get_transforms, save_confusion_matrix, load_class_mapping


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    """
    Run full evaluation on the test set.
    Returns predictions, labels, and softmax probabilities.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )


def compute_metrics(labels, preds, probs, n_classes, class_names=None):
    """Compute all required evaluation metrics."""
    metrics = {
        "accuracy":         float(np.mean(labels == preds)),
        "f1_macro":         float(f1_score(labels, preds, average="macro",     zero_division=0)),
        "f1_weighted":      float(f1_score(labels, preds, average="weighted",  zero_division=0)),
        "precision_macro":  float(precision_score(labels, preds, average="macro",    zero_division=0)),
        "recall_macro":     float(recall_score(labels, preds, average="macro",       zero_division=0)),
    }

    # AUC — one-vs-rest, requires probability scores
    try:
        one_hot = np.eye(n_classes)[labels]
        metrics["auc"] = float(roc_auc_score(one_hot, probs, average="macro", multi_class="ovr"))
    except Exception as e:
        metrics["auc"] = None
        print(f"[Warning] AUC could not be computed: {e}")

    # mAP — mean average precision per class
    try:
        one_hot = np.eye(n_classes)[labels]
        metrics["mAP"] = float(average_precision_score(one_hot, probs, average="macro"))
    except Exception as e:
        metrics["mAP"] = None
        print(f"[Warning] mAP could not be computed: {e}")

    # per-class report
    metrics["per_class_report"] = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return metrics


def save_test_metrics(metrics: dict, output_dir: str):
    """Save test metrics to test_metrics.json."""
    path = os.path.join(output_dir, "test_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Test results saved to {path}")

    # print summary to console
    print(f"\n{'='*40}")
    print(f"  TEST RESULTS")
    print(f"{'='*40}")
    for k, v in metrics.items():
        if k == "per_class_report":
            continue
        print(f"  {k:<20} {v:.4f}" if v is not None else f"  {k:<20} N/A")
    print(f"{'='*40}\n")


# =============================================================================
# GRADCAM (extra credit)
# =============================================================================

def run_gradcam(model, loader, device, output_dir, n_samples=8):
    """
    Generate GradCAM heatmaps for a sample of test images.
    Requires: pip install grad-cam

    TODO: Implement full GradCAM visualisation using pytorch-grad-cam library.
    This is the extra credit item — enable via --gradcam flag or save_gradcam: true in config.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # TODO: select the correct target layer per backbone:
        #   resnet50:       model.layer4[-1]
        #   efficientnet:   model.features[-1]
        #   swin_t:         model.layers[-1].blocks[-1]
        #   convnext_t:     model.features[-1]

        print("[GradCAM] TODO: implement GradCAM visualisation here")
        print("[GradCAM] See pytorch-grad-cam docs: https://github.com/jacobgil/pytorch-grad-cam")

    except ImportError:
        print("[GradCAM] Install pytorch-grad-cam: pip install grad-cam")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="EEEM068 Waste Classification Evaluation")
    parser.add_argument("--run", type=str, required=True,
                        help="Path to run output folder (contains config.yaml and best_model.pth)")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate GradCAM heatmaps (extra credit)")
    args = parser.parse_args()

    output_dir = args.run

    # ── Load saved config from training run ───────────────────────────────────
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f" Evaluating: {config['run']['name']}")
    print(f" Model:      {config['run']['model']}")
    print(f"{'='*60}\n")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # ── Model — load best checkpoint ──────────────────────────────────────────
    model = get_model(config).to(device)
    ckpt_path = os.path.join(output_dir, "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[Model] Loaded checkpoint from {ckpt_path}")

    # ── Test dataloader ───────────────────────────────────────────────────────
    test_transforms = get_transforms(config, train=False)
    test_dataset = datasets.ImageFolder(
        root=os.path.join(config["dataset"]["data_root"], "test"),
        transform=test_transforms
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    class_names = test_dataset.classes
    n_classes  = config["dataset"]["n_classes"]
    mapping    = load_class_mapping(config)
    class_names = [mapping["label_to_class"][str(i)] for i in range(n_classes)]

    # ── Evaluate ─────────────────────────────────────────────────────────────
    preds, labels, probs = evaluate(model, test_loader, device, n_classes)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(labels, preds, probs, n_classes, class_names)
    save_test_metrics(metrics, output_dir)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    if config["output"]["save_confusion_matrix"]:
        save_confusion_matrix(labels, preds, class_names, output_dir)

    # ── GradCAM (extra credit) ────────────────────────────────────────────────
    if args.gradcam or config["output"].get("save_gradcam", False):
        run_gradcam(model, test_loader, device, output_dir)

    print(f"[Done] All test outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
