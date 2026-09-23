"""Concrete EXP-001 loss-kernel assembly point."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from diploma_pinn.boundaries import RBCBoundaryLoss
from diploma_pinn.contracts import LossReport, PointBatch
from diploma_pinn.formulations import VPResiduals
from diploma_pinn.losses.assembler import LossAssembler
from diploma_pinn.observations import velocity_mse


class VPReferenceLossKernel:
    """Fuse active VP, observation, and boundary calculations into one graph."""

    def __init__(
        self,
        formulation: VPResiduals,
        loss_assembler: LossAssembler,
        boundary_loss: RBCBoundaryLoss | None = None,
    ) -> None:
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
        }
        if self._boundary_loss is not None:
            boundary_components = self._boundary_loss(model, batch.boundaries)
            components["boundary"] = torch.stack(tuple(boundary_components.values())).sum()
        return self._loss_assembler(components)
