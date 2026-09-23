"""Atomic, versioned checkpoints for reproducible training runs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


SCHEMA_VERSION = 2


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    *,
    scheduler: object | None = None,
    config_fingerprint: str = "",
    auxiliary: dict[str, object] | None = None,
) -> None:
    """Save a complete final state without leaving a partially written file."""
    if step < 0:
        raise ValueError("step must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config_fingerprint": config_fingerprint,
        "auxiliary": auxiliary or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    scheduler: object | None = None,
    expected_config_fingerprint: str = "",
) -> int:
    """Restore state and return the last completed zero-based optimizer step."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") not in {1, SCHEMA_VERSION}:
        raise ValueError("unsupported checkpoint schema")
    if expected_config_fingerprint and payload.get("config_fingerprint") != expected_config_fingerprint:
        raise ValueError("checkpoint configuration fingerprint does not match")
    if (payload.get("scheduler") is None) != (scheduler is None):
        raise ValueError("checkpoint scheduler state does not match the current experiment")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["rng_cpu"])
    if torch.cuda.is_available() and payload.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
    return int(payload["step"])
