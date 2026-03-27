import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Save training/validation metrics as JSON
def save_metrics(metrics: dict, output_dir: str):
    # Persisting metrics allows full reproducibility and enables later comparison
    # across experiments, hyperparameter sweeps, or ablation studies.
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved to {path}")


# Plot loss and accuracy curves over training
def save_loss_curves(metrics: dict, output_dir: str):
    # Visualising training dynamics helps diagnose underfitting, overfitting,
    # learning‑rate issues, and optimisation instability.
    epochs = range(1, len(metrics["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves reveal optimisation behaviour and whether the model is converging.
    ax1.plot(epochs, metrics["train_loss"], label="Train")
    ax1.plot(epochs, metrics["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy curves provide an intuitive measure of performance progression.
    ax2.plot(epochs, metrics["train_acc"], label="Train")
    ax2.plot(epochs, metrics["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Loss curves saved to {path}")


# Save confusion matrix heatmap
def save_confusion_matrix(labels, preds, class_names: list, output_dir: str):
    # Confusion matrices expose systematic class‑pair confusions and are essential
    # for diagnosing where the model struggles, especially in imbalanced datasets.
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(20, 18))

    sns.heatmap(
        cm,
        annot=False,          # heatmap is cleaner without dense text for many classes
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names
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