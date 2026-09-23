"""Specialized single-update training step assembled before training."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer

from diploma_pinn.contracts import LossReport, PointBatch, StepMetrics
from diploma_pinn.instrumentation.profiler import StepProfiler


LossKernel = Callable[[nn.Module, PointBatch], LossReport]


class TrainStep:
    """Run exactly one total backward pass and one optimizer update."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_kernel: LossKernel,
        *,
        validate_finite: bool = False,
        profiler: StepProfiler | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_kernel = loss_kernel
        self.validate_finite = validate_finite
        self.profiler = profiler or StepProfiler(enabled=False)

    def __call__(self, batch: PointBatch, step: int) -> StepMetrics:
        self.optimizer.zero_grad(set_to_none=True)
        with self.profiler.phase("residual_derivatives"):
            report = self.loss_kernel(self.model, batch)
        if self.validate_finite and not bool(torch.isfinite(report.total)):
            raise FloatingPointError("non-finite total loss")
        with self.profiler.phase("backward"):
            report.total.backward()
        if self.validate_finite:
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"non-finite gradient: {name}")
        with self.profiler.phase("optimizer"):
            self.optimizer.step()
        return StepMetrics(
            step=step,
            total=report.total.detach().item(),
            components={name: value.detach().item() for name, value in report.components.items()},
        )
