"""Warm-up-aware microbenchmark runner."""

from dataclasses import dataclass
from pathlib import Path

from diploma_pinn.training import TrainStep
from diploma_pinn.training.trainer import BatchSource


@dataclass(frozen=True)
class BenchmarkSummary:
    warmup_steps: int
    measured_steps: int
    phase_ms: dict[str, float]
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def write_json(self, path: Path) -> None:
        raise NotImplementedError("serialize aggregate and repeat-level measurements")


def run_benchmark(
    step: TrainStep,
    batches: BatchSource,
    *,
    warmup_steps: int,
    measured_steps: int,
) -> BenchmarkSummary:
    raise NotImplementedError("synchronize only at declared timing boundaries")
