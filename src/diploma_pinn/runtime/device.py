"""Explicit device and precision resolution."""

import torch


def resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    allowed = {"float32": torch.float32, "float64": torch.float64}
    try:
        return allowed[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error
