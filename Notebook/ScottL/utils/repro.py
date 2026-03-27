import random
import numpy as np
import torch

# Fix all random seeds for reproducible experiments
def set_seed(seed: int):
    # Seeding all major RNGs ensures that data shuffling, weight initialisation,
    # and augmentation behave identically across repeated runs.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforcing deterministic cuDNN behaviour removes nondeterministic kernel choices,
    # trading a small amount of speed for fully reproducible results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed] Set to {seed}")