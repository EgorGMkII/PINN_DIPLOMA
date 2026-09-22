"""Field metrics, including pressure gauge handling."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FieldMetrics:
    relative_l2: dict[str, float]
    mse: dict[str, float]
    correlation: dict[str, float]


def compute_scalar_metrics(predicted: Tensor, target: Tensor) -> dict[str, float]:
    """Return scalar-field error statistics without a pressure gauge operation."""
    if predicted.shape != target.shape:
        raise ValueError("predicted and target tensors must have matching shapes")
    error = predicted - target
    denominator = torch.linalg.vector_norm(target).clamp_min(torch.finfo(target.dtype).eps)
    centered_prediction = predicted - predicted.mean()
    centered_target = target - target.mean()
    correlation = (centered_prediction * centered_target).sum() / (
        torch.linalg.vector_norm(centered_prediction)
        * torch.linalg.vector_norm(centered_target)
    ).clamp_min(torch.finfo(target.dtype).eps)
    return {
        "mse": float(error.square().mean()),
        "relative_l2": float(torch.linalg.vector_norm(error) / denominator),
        "correlation": float(correlation),
    }


def align_pressure_gauge(predicted: Tensor, target: Tensor, times: Tensor) -> Tensor:
    """Align predicted pressure to target independently at every evaluation time."""
    if predicted.shape != target.shape or predicted.ndim != 1:
        raise ValueError("predicted and target pressure must be matching [N] tensors")
    if times.shape != predicted.shape:
        raise ValueError("times must have shape [N]")
    aligned = predicted.clone()
    for time in torch.unique(times):
        mask = times == time
        aligned[mask] -= (predicted[mask] - target[mask]).mean()
    return aligned


def compute_field_metrics(predicted: Tensor, target: Tensor, times: Tensor) -> FieldMetrics:
    if (
        predicted.ndim != 2
        or predicted.shape[1] not in (4, 5)
        or target.shape != (predicted.shape[0], 5)
    ):
        raise ValueError("predicted fields must be [N,4] or [N,5]; target fields must be [N,5]")
    if times.shape != (predicted.shape[0],):
        raise ValueError("times must have shape [N]")

    aligned = predicted.clone()
    names = ("u", "v", "w", "T")
    if predicted.shape[1] == 5:
        aligned[:, 4] = align_pressure_gauge(predicted[:, 4], target[:, 4], times)
        names = (*names, "p")
    mse: dict[str, float] = {}
    relative_l2: dict[str, float] = {}
    correlation: dict[str, float] = {}
    for index, name in enumerate(names):
        metrics = compute_scalar_metrics(aligned[:, index], target[:, index])
        mse[name] = metrics["mse"]
        relative_l2[name] = metrics["relative_l2"]
        correlation[name] = metrics["correlation"]
    return FieldMetrics(relative_l2=relative_l2, mse=mse, correlation=correlation)
