"""Residual sensitivity of VP and VV to temperature basis modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class ModeSensitivity:
    name: str
    vp_momentum_rms: float
    vv_buoyancy_curl_rms: float
    energy_rms: float
    boundary_max: float


def analyze_temperature_modes(
    points: Tensor, *, diffusivity: float, vertical_modes: int = 3,
    horizontal_modes: int = 3,
) -> list[ModeSensitivity]:
    coordinates = points.detach().requires_grad_(True)
    _, x, y, z = coordinates.unbind(dim=1)
    definitions: list[tuple[str, Tensor, Tensor]] = []
    for n in range(1, vertical_modes + 1):
        temperature = torch.sin(n * math.pi * z)
        pressure = -torch.cos(n * math.pi * z) / (n * math.pi)
        definitions.append((f"vertical_{n}", temperature, pressure))
    for axis_name, axis in (("x", x), ("y", y)):
        for k in range(1, horizontal_modes + 1):
            horizontal = torch.cos(2 * math.pi * k * axis)
            temperature = torch.sin(math.pi * z) * horizontal
            pressure = -torch.cos(math.pi * z) * horizontal / math.pi
            definitions.append((f"{axis_name}_fourier_{k}", temperature, pressure))
    return [_sensitivity(name, temperature, pressure, coordinates, diffusivity)
            for name, temperature, pressure in definitions]


def write_mode_report(report: list[ModeSensitivity], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in report], indent=2), encoding="utf-8")


def _gradient(value: Tensor, coordinates: Tensor, *, create_graph: bool = True) -> Tensor:
    return torch.autograd.grad(
        value, coordinates, torch.ones_like(value), create_graph=create_graph, retain_graph=True
    )[0]


def _sensitivity(name: str, temperature: Tensor, pressure: Tensor,
                 coordinates: Tensor, diffusivity: float) -> ModeSensitivity:
    grad_t = _gradient(temperature, coordinates)
    grad_p = _gradient(pressure, coordinates)
    buoyancy = torch.stack(
        (torch.zeros_like(temperature), torch.zeros_like(temperature), temperature), dim=1
    )
    vp = grad_p[:, 1:] - buoyancy
    vv = torch.stack((grad_t[:, 2], -grad_t[:, 1], torch.zeros_like(temperature)), dim=1)
    laplacian = torch.zeros_like(temperature)
    for axis in (1, 2, 3):
        laplacian = laplacian + _gradient(grad_t[:, axis], coordinates)[:, axis]
    energy = -diffusivity * laplacian
    boundary_mask = (coordinates[:, 3] == coordinates[:, 3].min()) | (
        coordinates[:, 3] == coordinates[:, 3].max()
    )
    boundary = temperature[boundary_mask].abs().max() if bool(boundary_mask.any()) else temperature.new_tensor(0)
    return ModeSensitivity(
        name=name,
        vp_momentum_rms=float(vp.square().mean().sqrt().detach()),
        vv_buoyancy_curl_rms=float(vv.square().mean().sqrt().detach()),
        energy_rms=float(energy.square().mean().sqrt().detach()),
        boundary_max=float(boundary.detach()),
    )
