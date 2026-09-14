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
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._step = step
        self._batches = batches
        self._max_steps = max_steps
        self._tracker = tracker or NullTracker()

    def run(self) -> StepMetrics:
        last: StepMetrics | None = None
        for index in range(self._max_steps):
            last = self._step(self._batches.next_batch(), index)
            self._tracker.log({"loss/total": last.total, **last.components}, step=index)
        assert last is not None
        self._tracker.finish({"loss/total": last.total, **last.components})
        return last
