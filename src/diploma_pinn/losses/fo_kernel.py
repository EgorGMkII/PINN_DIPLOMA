"""Loss assembly for the mixed first-order formulation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from diploma_pinn.boundaries import RBCBoundaryLoss
from diploma_pinn.contracts import LossReport, PointBatch
from diploma_pinn.formulations.fo import FOResiduals
from diploma_pinn.losses.assembler import LossAssembler
from diploma_pinn.observations import velocity_mse


class FOLossKernel:
    def __init__(self, formulation: FOResiduals, loss_assembler: LossAssembler,
                 boundary_loss: RBCBoundaryLoss | None = None) -> None:
        self._formulation = formulation
        self._loss_assembler = loss_assembler
        self._boundary_loss = boundary_loss

    def __call__(self, model: nn.Module, batch: PointBatch) -> LossReport:
        residuals = self._formulation.residuals(model, batch.domain)
        components: dict[str, Tensor] = {
            "data": velocity_mse(model, batch.observations),
            "momentum": torch.stack((residuals["r_u"], residuals["r_v"], residuals["r_w"])).square().mean(),
            "energy": residuals["r_T"].square().mean(),
            "continuity": residuals["r_div"].square().mean(),
            "velocity_gradient": residuals["r_G"].square().mean(),
            "temperature_gradient": residuals["r_q"].square().mean(),
        }
        if self._boundary_loss is not None:
            boundary = self._boundary_loss(model, batch.boundaries)
            components["boundary"] = torch.stack(tuple(boundary.values())).sum()
        return self._loss_assembler(components)
