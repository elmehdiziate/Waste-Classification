import torch.nn as nn
from torchvision import models

# Build a classification model based on the selected backbone
def get_model(config: dict) -> nn.Module:
    backbone   = config["run"]["model"]
    n_classes  = config["dataset"]["n_classes"]
    pretrained = config["training"]["pretrained"]
    weights    = "IMAGENET1K_V1" if pretrained else None

    print(f"[Model] Loading {backbone} (pretrained={pretrained})")

    # For all backbones, we replace only the classification head.
    # This preserves the pretrained feature extractor while adapting the final layer
    # to the number of classes in our dataset — a standard and efficient fine‑tuning strategy.
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
            "Choose from: resnet50, efficientnet_b3, swin_t, convnext_t"
        )

    # Parameter summary helps verify fine‑tuning strategy (e.g., staged LR, frozen layers).
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params — total: {total_params:,} | trainable: {trainable_params:,}")

    return model

