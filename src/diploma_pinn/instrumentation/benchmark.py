"""Warm-up-aware training-step benchmark."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING

from diploma_pinn.instrumentation.profiler import StepProfiler

if TYPE_CHECKING:
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


def run_benchmark(
    step: "TrainStep",
    batches: "BatchSource",
    *,
    warmup_steps: int,
    measured_steps: int,
) -> BenchmarkSummary:
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and measured_steps positive")
    disabled = StepProfiler(enabled=False)
    original = step.profiler
    step.profiler = disabled
    for index in range(warmup_steps):
        step(batches.next_batch(), index)
    profiler = StepProfiler(enabled=True)
    step.profiler = profiler
    for offset in range(measured_steps):
        with profiler.phase("sampling"):
            batch = batches.next_batch()
        step(batch, warmup_steps + offset)
    step.profiler = original
    summary = profiler.summary()
    return BenchmarkSummary(
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        phase_ms={name: values["mean_ms"] for name, values in summary["phases"].items()},
        peak_allocated_bytes=int(summary["peak_allocated_bytes"]),
        peak_reserved_bytes=int(summary["peak_reserved_bytes"]),
    )
