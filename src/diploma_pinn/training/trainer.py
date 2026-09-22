"""Training lifecycle orchestration outside the numerical loss kernel."""

from typing import Protocol

from diploma_pinn.contracts import PointBatch, StepMetrics
from diploma_pinn.integrations import NullTracker, Tracker
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

    def run(self) -> StepMetrics:
        last: StepMetrics | None = None
        epoch_loss = 0.0
        steps_in_epoch = 0
        for index in range(self._max_steps):
            last = self._step(self._batches.next_batch(), index)
            epoch_loss += last.total
            steps_in_epoch += 1
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
                epoch_loss = 0.0
                steps_in_epoch = 0
        assert last is not None
        self._tracker.finish({"loss/total": last.total, **last.components})
        return last
