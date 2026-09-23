"""Pressure recovery from a frozen VV velocity-temperature model."""

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.optim import Adam

from diploma_pinn.config.schema import PressureRecoveryConfig
from diploma_pinn.data import RBCDNSDataset
from diploma_pinn.formulations import VPParameters
from diploma_pinn.integrations import Tracker
from diploma_pinn.models import SineMLP
from diploma_pinn.sampling import StratifiedTimeSampler


def _gradient(value: Tensor, coordinates: Tensor) -> Tensor:
    if not value.requires_grad:
        return torch.zeros_like(coordinates)
    result = torch.autograd.grad(
        value, coordinates, torch.ones_like(value), create_graph=True,
        retain_graph=True, allow_unused=True,
    )[0]
    return torch.zeros_like(coordinates) if result is None else result


@dataclass(frozen=True)
class PressureRecoveryResult:
    model: SineMLP
    optimizer: Adam
    steps: int
    final_loss: float


def momentum_pressure_gradient(model: nn.Module, points: Tensor, parameters: VPParameters) -> Tensor:
    """Return detached momentum-implied grad(p) without DNS pressure labels."""
    coordinates = points.detach().requires_grad_(True)
    outputs = model(coordinates)
    if outputs.shape != (points.shape[0], 4):
        raise ValueError("pressure recovery requires a VV model with output [N,4]")
    u, v, w, temperature = outputs.unbind(dim=1)
    velocity = (u, v, w)
    first: list[Tensor] = []
    laplacians: list[Tensor] = []
    for value in velocity:
        gradient = _gradient(value, coordinates)
        first.append(gradient)
        diagonal = []
        for axis in (1, 2, 3):
            second = _gradient(gradient[:, axis], coordinates)[:, axis]
            diagonal.append(second)
        laplacians.append(sum(diagonal))
    nu = math.sqrt(parameters.prandtl / parameters.rayleigh)
    components = []
    for index in range(3):
        convection = sum(velocity[j] * first[index][:, j + 1] for j in range(3))
        buoyancy = temperature if index == 2 else torch.zeros_like(temperature)
        components.append(-first[index][:, 0] - convection + nu * laplacians[index] + buoyancy)
    return torch.stack(components, dim=1).detach()


def pressure_loss(model: nn.Module, points: Tensor, target_gradient: Tensor,
                  gauge_weight: float) -> tuple[Tensor, Mapping[str, Tensor]]:
    coordinates = points.detach().requires_grad_(True)
    pressure = model(coordinates).squeeze(1)
    gradient = torch.autograd.grad(
        pressure, coordinates, torch.ones_like(pressure), create_graph=True
    )[0][:, 1:]
    gradient_loss = (gradient - target_gradient).square().mean()
    gauge_terms = [
        pressure[coordinates[:, 0] == time].mean().square()
        for time in torch.unique(coordinates[:, 0])
    ]
    gauge_loss = torch.stack(gauge_terms).mean()
    total = gradient_loss + gauge_weight * gauge_loss
    return total, {"pressure/gradient": gradient_loss, "pressure/gauge": gauge_loss}


def train_pressure_recovery(
    vv_model: nn.Module,
    dataset: RBCDNSDataset,
    parameters: VPParameters,
    config: PressureRecoveryConfig,
    tracker: Tracker,
    *,
    seed: int,
    step_offset: int = 0,
) -> PressureRecoveryResult:
    device = next(vv_model.parameters()).device
    dtype = next(vv_model.parameters()).dtype
    pressure_model = SineMLP(config.widths).to(device=device, dtype=dtype)
    optimizer = Adam(pressure_model.parameters(), lr=config.learning_rate, eps=1e-7)
    sampler = StratifiedTimeSampler(dataset, config.batch_size, seed)
    old_requires_grad = [parameter.requires_grad for parameter in vv_model.parameters()]
    for parameter in vv_model.parameters():
        parameter.requires_grad_(False)
    vv_model.eval()
    rows = dataset.points.shape[0]
    steps_per_epoch = math.ceil(rows / config.batch_size)
    final = float("nan")
    try:
        step = 0
        for _epoch in range(config.epochs):
            for _ in range(steps_per_epoch):
                batch = sampler.sample().points.to(device=device, dtype=dtype)
                target = momentum_pressure_gradient(vv_model, batch, parameters)
                optimizer.zero_grad(set_to_none=True)
                loss, components = pressure_loss(pressure_model, batch, target, config.gauge_weight)
                loss.backward()
                optimizer.step()
                final = float(loss.detach())
                if step % config.log_interval_steps == 0 or step + 1 == config.epochs * steps_per_epoch:
                    tracker.log(
                        {"pressure/loss": final, **{k: float(v.detach()) for k, v in components.items()}},
                        step=step_offset + step,
                    )
                step += 1
    finally:
        for parameter, state in zip(vv_model.parameters(), old_requires_grad):
            parameter.requires_grad_(state)
        vv_model.train()
    return PressureRecoveryResult(pressure_model, optimizer, config.epochs * steps_per_epoch, final)


def append_recovered_pressure(vv_fields: Tensor, pressure: Tensor) -> Tensor:
    if vv_fields.ndim != 2 or vv_fields.shape[1] != 4 or pressure.shape != (vv_fields.shape[0], 1):
        raise ValueError("expected VV fields [N,4] and pressure [N,1]")
    return torch.cat((vv_fields, pressure), dim=1)
