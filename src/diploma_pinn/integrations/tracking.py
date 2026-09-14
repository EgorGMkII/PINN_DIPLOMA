"""Optional experiment tracking isolated from numerical code."""

from typing import Mapping, Protocol


class Tracker(Protocol):
    def log(self, values: Mapping[str, float], step: int) -> None: ...
    def finish(self, summary: Mapping[str, float]) -> None: ...


class NullTracker:
    def log(self, values: Mapping[str, float], step: int) -> None:
        return None

    def finish(self, summary: Mapping[str, float]) -> None:
        return None


def build_wandb_tracker(*args: object, **kwargs: object) -> Tracker:
    raise NotImplementedError("import wandb lazily and support disabled/offline modes")
