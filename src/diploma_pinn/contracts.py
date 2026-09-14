"""Shared tensor contracts used outside the performance-critical train step."""

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor


@dataclass(frozen=True)
class ObservationBatch:
    """Velocity observations at points ordered as ``(t, x, y, z)``."""

    points: Tensor
    velocity: Tensor


@dataclass(frozen=True)
class PointBatch:
    """All independently sampled point groups needed by one optimizer step."""

    observations: ObservationBatch
    domain: Tensor
    boundaries: Mapping[str, Tensor]


@dataclass(frozen=True)
class LossReport:
    """Differentiable total loss and named differentiable components."""

    total: Tensor
    components: Mapping[str, Tensor]


@dataclass(frozen=True)
class StepMetrics:
    """Detached scalar values emitted after an optimizer update."""

    step: int
    total: float
    components: Mapping[str, float]
