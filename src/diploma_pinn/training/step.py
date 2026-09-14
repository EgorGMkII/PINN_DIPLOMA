"""Specialized single-update training step assembled before training."""

from typing import Callable

from torch import nn
from torch.optim import Optimizer

from diploma_pinn.contracts import LossReport, PointBatch, StepMetrics


LossKernel = Callable[[nn.Module, PointBatch], LossReport]


class TrainStep:
    """Run exactly one total backward pass and one optimizer update."""

    def __init__(self, model: nn.Module, optimizer: Optimizer, loss_kernel: LossKernel) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_kernel = loss_kernel

    def __call__(self, batch: PointBatch, step: int) -> StepMetrics:
        self.optimizer.zero_grad(set_to_none=True)
        report = self.loss_kernel(self.model, batch)
        report.total.backward()
        self.optimizer.step()
        return StepMetrics(
            step=step,
            total=report.total.detach().item(),
            components={name: value.detach().item() for name, value in report.components.items()},
        )
