"""Low-overhead phase timing with explicit CUDA synchronization."""

from collections import defaultdict
from contextlib import contextmanager
import time
from typing import Iterator

import torch


class StepProfiler:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._elapsed: dict[str, list[float]] = defaultdict(list)
        if enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        _synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            _synchronize()
            self._elapsed[name].append((time.perf_counter() - started) * 1000.0)

    def summary(self) -> dict[str, object]:
        phases = {
            name: {
                "count": len(samples),
                "total_ms": sum(samples),
                "mean_ms": sum(samples) / len(samples),
            }
            for name, samples in self._elapsed.items() if samples
        }
        return {
            "phases": phases,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
        }


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
