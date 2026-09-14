"""Structured numerical parity comparisons."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from torch import Tensor


@dataclass(frozen=True)
class TensorComparison:
    name: str
    max_absolute_error: float
    relative_l2_error: float
    passed: bool


@dataclass(frozen=True)
class ParityReport:
    comparisons: tuple[TensorComparison, ...]

    def write_json(self, path: Path) -> None:
        raise NotImplementedError("write a stable machine-readable report")


def compare_tensors(
    actual: Mapping[str, Tensor],
    expected: Mapping[str, Tensor],
    tolerances: Mapping[str, tuple[float, float]],
) -> ParityReport:
    raise NotImplementedError("report every named tensor without early exit")
