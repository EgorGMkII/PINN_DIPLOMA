"""Training lifecycle orchestration outside the numerical loss kernel."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import time
from typing import Protocol

import torch

from diploma_pinn.contracts import PointBatch, StepMetrics
from diploma_pinn.integrations import NullTracker, Tracker
from diploma_pinn.instrumentation.profiler import StepProfiler
from diploma_pinn.training.step import TrainStep


class BatchSource(Protocol):
    def next_batch(self) -> PointBatch: ...


class Trainer:
    """Coordinate steps, evaluation, logging, checkpointing, and shutdown."""

    def __init__(
        self,
        step: TrainStep,
        batches: BatchSource,
        *,
        max_steps: int,
        tracker: Tracker | None = None,
        scheduler: object | None = None,
        scheduler_epoch_steps: int = 0,
        profiler: StepProfiler | None = None,
        epoch_callback: Callable[[int], Mapping[str, float]] | None = None,
        log_interval_steps: int = 1,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if scheduler is None and scheduler_epoch_steps:
            raise ValueError("scheduler_epoch_steps requires a scheduler")
        if scheduler is not None and scheduler_epoch_steps <= 0:
            raise ValueError("a scheduler requires positive scheduler_epoch_steps")
        self._step = step
        self._batches = batches
        self._max_steps = max_steps
        self._tracker = tracker or NullTracker()
        self._scheduler = scheduler
        self._scheduler_epoch_steps = scheduler_epoch_steps
        self._profiler = profiler or StepProfiler(enabled=False)
        self._epoch_callback = epoch_callback
        self._log_interval_steps = log_interval_steps

    def set_epoch_callback(self, callback: Callable[[int], Mapping[str, float]] | None) -> None:
        self._epoch_callback = callback

    def run(self) -> StepMetrics:
        last: StepMetrics | None = None
        epoch_loss = 0.0
        steps_in_epoch = 0
        epoch_started = time.perf_counter()
        for index in range(self._max_steps):
            with self._profiler.phase("sampling"):
                batch = self._batches.next_batch()
            last = self._step(batch, index)
            epoch_loss += last.total
            steps_in_epoch += 1
            if index % self._log_interval_steps == 0 or index + 1 == self._max_steps:
                self._tracker.log(
                    {
                        "loss/total": last.total,
                        "optimizer/learning_rate": self._step.optimizer.param_groups[0]["lr"],
                        **last.components,
                    },
                    step=index,
                )
            if self._scheduler is not None and steps_in_epoch == self._scheduler_epoch_steps:
                self._scheduler.step(epoch_loss / steps_in_epoch)
                epoch = (index + 1) // self._scheduler_epoch_steps
                epoch_values: dict[str, float] = {
                    "epoch/index": float(epoch),
                    "epoch/loss_mean": epoch_loss / steps_in_epoch,
                    "epoch/seconds": time.perf_counter() - epoch_started,
                }
                if torch.cuda.is_available():
                    epoch_values.update({
                        "gpu/max_allocated_bytes": float(torch.cuda.max_memory_allocated()),
                        "gpu/max_reserved_bytes": float(torch.cuda.max_memory_reserved()),
                    })
                if self._epoch_callback is not None:
                    with self._profiler.phase("evaluation"):
                        epoch_values.update(self._epoch_callback(epoch))
                self._tracker.log(epoch_values, step=index)
                epoch_loss = 0.0
                steps_in_epoch = 0
                epoch_started = time.perf_counter()
        assert last is not None
        return last
