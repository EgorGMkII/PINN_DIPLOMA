"""Coordinate derivatives required by the direct strong VV formulation."""

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class VVFieldDerivatives:
    """Velocity/temperature fields, their derivatives, and vorticity derivatives."""

    fields: Mapping[str, Tensor]
    first: Mapping[str, Tensor]
    second: Mapping[str, Tensor]
    vorticity: Mapping[str, Tensor]
    vorticity_first: Mapping[str, Tensor]
    vorticity_second: Mapping[str, Tensor]


def _gradient(value: Tensor, coordinates: Tensor) -> Tensor:
    if not value.requires_grad:
        return torch.zeros_like(coordinates)
    gradient = torch.autograd.grad(
        value,
        coordinates,
        grad_outputs=torch.ones_like(value),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(coordinates) if gradient is None else gradient


def compute_vv_derivatives(model: nn.Module, points: Tensor) -> VVFieldDerivatives:
    """Build the connected direct-VV graph, including third derivatives of velocity."""
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"expected points [N,4], got {tuple(points.shape)}")
    if not points.is_floating_point():
        raise TypeError("points must have a floating dtype")
    coordinates = points.detach().requires_grad_(True)
    outputs = model(coordinates)
    if outputs.ndim != 2 or outputs.shape != (points.shape[0], 4):
        raise ValueError(f"expected VV model output [N,4], got {tuple(outputs.shape)}")

    fields = dict(zip(("u", "v", "w", "T"), outputs.unbind(dim=1)))
    axis_names = ("t", "x", "y", "z")
    axis_indices = {axis: index for index, axis in enumerate(axis_names)}
    first: dict[str, Tensor] = {}
    second: dict[str, Tensor] = {}
    for name, value in fields.items():
        gradient = _gradient(value, coordinates)
        for axis in axis_names:
            first[f"{name}_{axis}"] = gradient[:, axis_indices[axis]]
        for axis in ("x", "y", "z"):
            second_gradient = _gradient(first[f"{name}_{axis}"], coordinates)
            second[f"{name}_{axis}{axis}"] = second_gradient[:, axis_indices[axis]]

    vorticity = {
        "omega_x": first["w_y"] - first["v_z"],
        "omega_y": first["u_z"] - first["w_x"],
        "omega_z": first["v_x"] - first["u_y"],
    }
    vorticity_first: dict[str, Tensor] = {}
    vorticity_second: dict[str, Tensor] = {}
    for name, value in vorticity.items():
        gradient = _gradient(value, coordinates)
        for axis in axis_names:
            vorticity_first[f"{name}_{axis}"] = gradient[:, axis_indices[axis]]
        for axis in ("x", "y", "z"):
            second_gradient = _gradient(vorticity_first[f"{name}_{axis}"], coordinates)
            vorticity_second[f"{name}_{axis}{axis}"] = second_gradient[:, axis_indices[axis]]
    return VVFieldDerivatives(
        fields=fields,
        first=first,
        second=second,
        vorticity=vorticity,
        vorticity_first=vorticity_first,
        vorticity_second=vorticity_second,
    )
