"""First derivatives for the mixed first-order formulation."""

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FOFieldDerivatives:
    fields: Mapping[str, Tensor]
    first: Mapping[str, Tensor]


def compute_fo_derivatives(model: nn.Module, points: Tensor) -> FOFieldDerivatives:
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"expected points [N,4], got {tuple(points.shape)}")
    coordinates = points.detach().requires_grad_(True)
    outputs = model(coordinates)
    if outputs.shape != (points.shape[0], 17):
        raise ValueError(f"expected FO model output [N,17], got {tuple(outputs.shape)}")
    names = (
        "u", "v", "w", "T", "p",
        "G_ux", "G_uy", "G_uz", "G_vx", "G_vy", "G_vz",
        "G_wx", "G_wy", "G_wz", "q_x", "q_y", "q_z",
    )
    fields = dict(zip(names, outputs.unbind(dim=1)))
    first: dict[str, Tensor] = {}
    for name, value in fields.items():
        gradient = torch.autograd.grad(
            value, coordinates, torch.ones_like(value), create_graph=True, retain_graph=True
        )[0]
        for index, axis in enumerate(("t", "x", "y", "z")):
            first[f"{name}_{axis}"] = gradient[:, index]
    return FOFieldDerivatives(fields=fields, first=first)
