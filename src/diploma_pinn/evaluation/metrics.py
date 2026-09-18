"""Field metrics, including pressure gauge handling."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FieldMetrics:
    relative_l2: dict[str, float]
    mse: dict[str, float]
    correlation: dict[str, float]


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
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 5:
        raise ValueError("predicted and target fields must have shape [N,5]")
    if times.shape != (predicted.shape[0],):
        raise ValueError("times must have shape [N]")

    aligned = predicted.clone()
    aligned[:, 4] = align_pressure_gauge(predicted[:, 4], target[:, 4], times)
    names = ("u", "v", "w", "T", "p")
    mse: dict[str, float] = {}
    relative_l2: dict[str, float] = {}
    correlation: dict[str, float] = {}
    for index, name in enumerate(names):
        error = aligned[:, index] - target[:, index]
        mse[name] = float(error.square().mean())
        denominator = torch.linalg.vector_norm(target[:, index])
        relative_l2[name] = float(torch.linalg.vector_norm(error) / denominator.clamp_min(torch.finfo(target.dtype).eps))
        centered_prediction = aligned[:, index] - aligned[:, index].mean()
        centered_target = target[:, index] - target[:, index].mean()
        correlation[name] = float(
            (centered_prediction * centered_target).sum()
            / (torch.linalg.vector_norm(centered_prediction) * torch.linalg.vector_norm(centered_target)).clamp_min(torch.finfo(target.dtype).eps)
        )
    return FieldMetrics(relative_l2=relative_l2, mse=mse, correlation=correlation)
