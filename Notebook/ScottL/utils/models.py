import torch.nn as nn
from torchvision import models

# Build a classification model based on the selected backbone
def get_model(config: dict) -> nn.Module:
    backbone   = config["run"]["model"]
    n_classes  = config["dataset"]["n_classes"]
    pretrained = config["training"]["pretrained"]

    # Using pretrained ImageNet weights accelerates convergence and improves feature quality
    # when the target dataset is smaller or visually similar to ImageNet.
    weights = "IMAGENET1K_V1" if pretrained else None

    print(f"[Model] Loading {backbone} (pretrained={pretrained})")

    # Replacing only the classification head preserves the pretrained feature extractor.
    # This is the core principle of transfer learning: reuse general visual features
    # while adapting the final layer to the specific number of classes.
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

    # Parameter counts help verify the fine‑tuning strategy.
    # A large gap between total and trainable parameters indicates frozen layers,
    # which is useful when applying staged learning rates or partial fine‑tuning.
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params — total: {total_params:,} | trainable: {trainable_params:,}")

    return model