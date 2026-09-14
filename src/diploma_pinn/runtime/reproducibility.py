"""Explicit reproducibility policy."""

import random

import numpy as np
import torch

def seed_everything(seed: int, deterministic: bool) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = not deterministic
