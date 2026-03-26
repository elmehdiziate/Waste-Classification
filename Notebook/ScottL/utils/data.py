import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# Load label ↔ class name mapping
def load_class_mapping(config: dict) -> dict:
    mapping_path = config["dataset"]["class_mapping"]
    with open(mapping_path) as f:
        return json.load(f)


# Build training or evaluation transforms
def get_transforms(config: dict, train: bool = True):
    aug = config["augmentation"]
    img_size = (config["dataset"]["img_height"], config["dataset"]["img_width"])

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if train:
        tf_list = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=aug["brightness"],
                contrast=aug["contrast"],
                saturation=aug["saturation"],
                hue=aug["hue"]
            ),
            transforms.ToTensor(),
            normalize
        ]

        if aug.get("random_erasing", False):
            tf_list.append(
                transforms.RandomErasing(
                    p=0.25,
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3),
                    value='random'
                )
            )

        return transforms.Compose(tf_list)

    # Validation / test transforms
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])


# Build train/val/test dataloaders
def get_dataloaders(config: dict):
    train_tf = get_transforms(config, train=True)
    val_tf   = get_transforms(config, train=False)

    data_root   = config["dataset"]["data_root"]
    batch_size  = config["training"]["batch_size"]
    seed        = config["dataset"]["seed"]
    use_sampler = config["training"].get("use_weighted_sampler", False)

    train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
    val_dataset   = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=val_tf)
    test_dataset  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=val_tf)

    sampler = None
    if use_sampler:
        targets = train_dataset.targets
        class_counts = np.bincount(targets, minlength=len(train_dataset.classes))
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[targets]

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"[Imbalance Fix] WeightedRandomSampler ENABLED ({len(train_dataset.classes)} classes)")
    else:
        print("[Imbalance Fix] Sampler DISABLED")

    pin_memory = torch.cuda.is_available()
    workers = 0 if os.name == "nt" else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not use_sampler,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )

    print(f"[Data] Train={len(train_dataset)} | Val={len(val_dataset)} | Test={len(test_dataset)}")
    return train_loader, val_loader, test_loader
