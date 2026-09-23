"""Coordinate-derivative contract for the strong VP formulation."""

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FieldDerivatives:
    """Network fields and named first/diagonal-second coordinate derivatives."""

    fields: Mapping[str, Tensor]
    first: Mapping[str, Tensor]
    second: Mapping[str, Tensor]


def compute_vp_derivatives(model: nn.Module, points: Tensor) -> FieldDerivatives:
    """Build one connected graph containing exactly the derivatives VP requires."""
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"expected points [N,4], got {tuple(points.shape)}")
    if not points.is_floating_point():
        raise TypeError("points must have a floating dtype")

    coordinates = points.detach().requires_grad_(True)
    outputs = model(coordinates)
    if outputs.ndim != 2 or outputs.shape != (points.shape[0], 5):
        raise ValueError(f"expected model output [N,5], got {tuple(outputs.shape)}")

    field_names = ("u", "v", "w", "T", "p")
    fields = dict(zip(field_names, outputs.unbind(dim=1)))
    first: dict[str, Tensor] = {}
    second: dict[str, Tensor] = {}
    derivative_fields = ("u", "v", "w", "T")

    for name, value in fields.items():
        gradient = torch.autograd.grad(
            value,
            coordinates,
            grad_outputs=torch.ones_like(value),
            create_graph=True,
            retain_graph=True,
        )[0]
        axes = ("t", "x", "y", "z") if name in derivative_fields else ("x", "y", "z")
        axis_indices = {"t": 0, "x": 1, "y": 2, "z": 3}
        for axis in axes:
            first[f"{name}_{axis}"] = gradient[:, axis_indices[axis]]

        if name in derivative_fields:
            for axis in ("x", "y", "z"):
                first_derivative = first[f"{name}_{axis}"]
                if not first_derivative.requires_grad:
                    second[f"{name}_{axis}{axis}"] = torch.zeros_like(first_derivative)
                else:
                    second_gradient = torch.autograd.grad(
                        first_derivative,
                        coordinates,
                        grad_outputs=torch.ones_like(first_derivative),
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    second[f"{name}_{axis}{axis}"] = (
                        torch.zeros_like(first_derivative)
                        if second_gradient is None
                        else second_gradient[:, axis_indices[axis]]
                    )

    return FieldDerivatives(fields=fields, first=first, second=second)
