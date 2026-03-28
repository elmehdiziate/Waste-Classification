"""
EEEM068: Industrial Waste Classification
train.py — Training pipeline

USAGE:
    python train.py --config configs/experiments/smoke_test.yaml # SMoke test
    python train.py --config configs/experiments/resnet50_phase1_lr1e-4_bs64.yaml
    python train.py --config configs/experiments/resnet50_phase1_lr1e-4_bs64.yaml --learning_rate 0.0003

FIXES APPLIED:
    - pin_memory now conditional on CUDA availability (no warning on CPU)
    - num_workers conditional on OS (0 on Windows, 4 on Linux/Mac)
    - output_dir now includes model subfolder: experiments/results/{model}/{run_name}/
    - save_loss_curves and save_confusion_matrix read from config
    - class_mapping.json loaded for label <-> class name resolution
"""

import os
import json
import copy
import argparse
import random
import time
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# CONFIG LOADING
# =============================================================================

def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base.
    Override values take precedence over base values.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "base":
            continue
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(experiment_path: str, cli_overrides: dict = None) -> dict:
    """
    Load experiment config, merge with base.yaml, apply CLI overrides.

    Priority (highest to lowest):
        1. CLI arguments
        2. Experiment YAML values
        3. base.yaml values
    """
    with open(experiment_path) as f:
        experiment = yaml.safe_load(f)

    if "base" in experiment:
        with open(experiment["base"]) as f:
            base = yaml.safe_load(f)
        config = deep_merge(base, experiment)
    else:
        config = experiment

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                config["training"][key] = value

    return config


def save_config(config: dict, output_dir: str):
    """Save a copy of the merged config alongside results."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[Config] Saved to {path}")


def load_class_mapping(config: dict) -> dict:
    """Load label <-> class name mapping from class_mapping.json."""
    mapping_path = config["dataset"]["class_mapping"]
    with open(mapping_path) as f:
        mapping = json.load(f)
    return mapping


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Set to {seed}")


# =============================================================================
# DATASET
# =============================================================================

def get_transforms(config: dict, train: bool = True) -> transforms.Compose:
    """
    Build train or validation transforms from config.
    Validation transforms never include augmentation.
    """
    aug = config["augmentation"]
    img_size = (config["dataset"]["img_height"], config["dataset"]["img_width"])

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip() if aug["random_horizontal_flip"] else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(
                brightness=aug["brightness"],
                contrast=aug["contrast"],
                saturation=aug["saturation"],
                hue=aug["hue"]
            ),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])


def get_dataloaders(config: dict):
    """
    Build train, validation, and test DataLoaders.

    pin_memory: enabled only when CUDA is available
    num_workers: 0 on Windows (avoids multiprocessing issues), 4 on Linux/Mac
    """
    train_transforms = get_transforms(config, train=True)
    val_transforms   = get_transforms(config, train=False)

    data_root  = config["dataset"]["data_root"]
    batch_size = config["training"]["batch_size"]
    seed       = config["dataset"]["seed"]

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_root, "train"),
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_root, "val"),
        transform=val_transforms
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_root, "test"),
        transform=val_transforms
    )

    # class weights for imbalance handling
    class_counts  = np.array([train_dataset.targets.count(i)
                               for i in range(config["dataset"]["n_classes"])])
    class_weights = torch.FloatTensor(1.0 / (class_counts + 1e-6))

    # device-aware DataLoader settings
    pin    = torch.cuda.is_available()
    workers = 0 if os.name == "nt" else 4   # 0 on Windows, 4 on Linux/Mac

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=pin, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["evaluation"]["batch_size"],
        shuffle=False, num_workers=workers, pin_memory=pin
    )

    print(f"[Data] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"[Data] pin_memory={pin} | num_workers={workers}")
    return train_loader, val_loader, test_loader, class_weights


# =============================================================================
# MODEL
# =============================================================================

def get_model(config: dict) -> nn.Module:
    """
    Load pretrained backbone and replace classification head.
    Supports: resnet50, efficientnet_b3, swin_t, convnext_t
    """
    backbone  = config["run"]["model"]
    n_classes = config["dataset"]["n_classes"]
    pretrained = config["training"]["pretrained"]
    weights   = "IMAGENET1K_V1" if pretrained else None

    print(f"[Model] Loading {backbone} (pretrained={pretrained})")

    if backbone == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    elif backbone == "efficientnet_b3":
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    elif backbone == "swin_t":
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, n_classes)

    elif backbone == "convnext_t":
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, n_classes)

    else:
        raise ValueError(
            f"Unknown model: {backbone}. "
            f"Choose from: resnet50, efficientnet_b3, swin_t, convnext_t"
        )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params — total: {total:,} | trainable: {trainable:,}")
    return model


def get_optimizer(model: nn.Module, config: dict):
    """
    Build optimizer with optional staged (differential) learning rates.

    staged_lr=True: backbone trains at lr * base_lr_mult, head at full lr.
    staged_lr=False: all parameters train at the same lr.
    """
    opt_cfg = config["optimizer"]
    ft_cfg  = config["fine_tuning"]
    lr      = config["training"]["learning_rate"]

    if ft_cfg["staged_lr"]:
        new_layer_names = ft_cfg.get("new_layers", ["fc", "classifier", "head"])
        head_params, backbone_params = [], []

        for name, param in model.named_parameters():
            if any(layer in name for layer in new_layer_names):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr * ft_cfg["base_lr_mult"]},
            {"params": head_params,     "lr": lr}
        ]
        print(f"[Optimizer] Staged LR — backbone: {lr * ft_cfg['base_lr_mult']:.2e} | head: {lr:.2e}")
    else:
        param_groups = model.parameters()
        print(f"[Optimizer] Uniform LR — {lr:.2e}")

    if opt_cfg["type"] == "adam":
        return torch.optim.Adam(
            param_groups, lr=lr,
            betas=(opt_cfg["adam_beta1"], opt_cfg["adam_beta2"]),
            weight_decay=opt_cfg["weight_decay"]
        )
    elif opt_cfg["type"] == "sgd":
        return torch.optim.SGD(
            param_groups, lr=lr,
            momentum=opt_cfg["momentum"],
            dampening=opt_cfg["sgd_dampening"],
            nesterov=opt_cfg["sgd_nesterov"],
            weight_decay=opt_cfg["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")


def get_scheduler(optimizer, config: dict):
    """Build LR scheduler from config."""
    sched_type = config["scheduler"]["type"]
    ft_cfg     = config["fine_tuning"]

    if sched_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ft_cfg["T_max"]
        )
    elif sched_type == "multi_step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=ft_cfg["stepsize"], gamma=ft_cfg["gamma"]
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, log_interval):
    """Run one full training epoch. Returns average loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += images.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [Batch {batch_idx+1}/{len(loader)}] loss: {loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation. Returns loss, accuracy, F1, and all predictions."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, f1, all_preds, all_labels


# =============================================================================
# SAVING RESULTS
# =============================================================================

def save_metrics(metrics: dict, output_dir: str):
    """Save all metrics to metrics.json for later chart generation."""
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved to {path}")


def save_loss_curves(metrics: dict, output_dir: str):
    """Plot and save training/validation loss and accuracy curves."""
    epochs = range(1, len(metrics["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, metrics["train_loss"], label="Train")
    ax1.plot(epochs, metrics["val_loss"],   label="Val")
    ax1.set_title(f"Loss — {metrics['model']}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, metrics["train_acc"], label="Train")
    ax2.plot(epochs, metrics["val_acc"],   label="Val")
    ax2.set_title(f"Accuracy — {metrics['model']}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Loss curves saved to {path}")


def save_confusion_matrix(labels, preds, class_names: list, output_dir: str):
    """Plot and save confusion matrix with class names."""
    cm  = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues", ax=ax,
        xticklabels=class_names, yticklabels=class_names
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved to {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="EEEM068 Waste Classification Training")
    parser.add_argument("--config",        type=str,   required=True)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--epochs",        type=int,   default=None)
    args = parser.parse_args()

    cli_overrides = {
        "learning_rate": args.learning_rate,
        "batch_size":    args.batch_size,
        "epochs":        args.epochs
    }

    # ── Load config ───────────────────────────────────────────────────────────
    config = load_config(args.config, cli_overrides)

    run_name   = config["run"]["name"]
    model_name = config["run"]["model"]

    # resolve {model} placeholder in base_dir
    base_dir   = config["output"]["base_dir"].replace("{model}", model_name)
    output_dir = os.path.join(base_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Run:    {run_name}")
    print(f" Model:  {model_name}")
    print(f" Phase:  {config['run']['phase']}")
    print(f" Author: {config['run']['author']}")
    print(f" LR:     {config['training']['learning_rate']}")
    print(f" BS:     {config['training']['batch_size']}")
    print(f" Epochs: {config['training']['epochs']}")
    print(f" Output: {output_dir}")
    print(f"{'='*60}\n")

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_seed(config["dataset"]["seed"])

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    if device.type == "cuda":
        print(f"[Device] {torch.cuda.get_device_name(0)}")
        print(f"[Device] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Save config copy ──────────────────────────────────────────────────────
    save_config(config, output_dir)

    # ── Class mapping ─────────────────────────────────────────────────────────
    mapping      = load_class_mapping(config)
    class_names  = [mapping["label_to_class"][str(i)]
                    for i in range(config["dataset"]["n_classes"])]

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(config)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(config).to(device)

    # ── Loss — weighted CrossEntropy handles class imbalance ───────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # ── Training loop ──────────────────────────────────────────────────────────
    epochs       = config["training"]["epochs"]
    log_interval = config["output"]["log_interval"]
    best_val_f1  = 0.0
    best_epoch   = 0

    metrics = {
        "run_name":      run_name,
        "model":         model_name,
        "learning_rate": config["training"]["learning_rate"],
        "batch_size":    config["training"]["batch_size"],
        "epoch":         [],
        "train_loss":    [], "train_acc": [],
        "val_loss":      [], "val_acc":   [], "val_f1": []
    }

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, log_interval
        )
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  train loss: {train_loss:.4f} | train acc: {train_acc:.4f}")
        print(f"  val loss:   {val_loss:.4f}   | val acc:   {val_acc:.4f} | val f1: {val_f1:.4f}")
        print(f"  epoch time: {elapsed:.1f}s")

        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(round(train_loss, 6))
        metrics["train_acc"].append(round(train_acc, 6))
        metrics["val_loss"].append(round(val_loss, 6))
        metrics["val_acc"].append(round(val_acc, 6))
        metrics["val_f1"].append(round(val_f1, 6))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            ckpt_path   = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [Checkpoint] New best saved (val_f1={best_val_f1:.4f})")

    print(f"\n[Training complete] Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if config["output"]["save_loss_curves"]:
        save_loss_curves(metrics, output_dir)

    if config["output"]["save_confusion_matrix"]:
        save_confusion_matrix(val_labels, val_preds, class_names, output_dir)

    # ── Metrics ───────────────────────────────────────────────────────────────
    save_metrics(metrics, output_dir)

    print(f"\n[Done] All outputs saved to: {output_dir}")
    print(f"       Next step: python test.py --run {output_dir}")


if __name__ == "__main__":
    main()
