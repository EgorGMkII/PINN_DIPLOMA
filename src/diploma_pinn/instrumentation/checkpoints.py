"""Checkpoint schema and round-trip contract."""

from pathlib import Path

from torch import nn
from torch.optim import Optimizer


def save_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer, step: int) -> None:
    raise NotImplementedError("save atomically with schema version and RNG state")


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer) -> int:
    raise NotImplementedError("validate schema and return restored optimizer step")
