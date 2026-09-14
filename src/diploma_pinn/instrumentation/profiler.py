"""Synchronized phase timing and device-memory metrics."""

from contextlib import AbstractContextManager


class StepProfiler:
    def phase(self, name: str) -> AbstractContextManager[None]:
        raise NotImplementedError("implement CPU/CUDA synchronized profile mode")

    def summary(self) -> dict[str, float]:
        raise NotImplementedError
