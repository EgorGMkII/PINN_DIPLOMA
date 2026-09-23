"""Local and optional W&B experiment tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Protocol


class Tracker(Protocol):
    def log(self, values: Mapping[str, float], step: int) -> None: ...
    def finish(self, summary: Mapping[str, object]) -> None: ...


class NullTracker:
    def log(self, values: Mapping[str, float], step: int) -> None:
        return None

    def finish(self, summary: Mapping[str, object]) -> None:
        return None


class JsonlTracker:
    """Append records immediately so interrupted jobs retain their history."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8", newline="\n")

    def log(self, values: Mapping[str, float], step: int) -> None:
        self._handle.write(json.dumps({"step": step, **values}, sort_keys=True) + "\n")
        self._handle.flush()

    def finish(self, summary: Mapping[str, object]) -> None:
        self._handle.flush()
        self._handle.close()


class WandbTracker:
    def __init__(self, run: object) -> None:
        self._run = run

    def log(self, values: Mapping[str, float], step: int) -> None:
        self._run.log(dict(values), step=step)

    def finish(self, summary: Mapping[str, object]) -> None:
        for key, value in _flatten(summary).items():
            self._run.summary[key] = value
        self._run.finish()


class CompositeTracker:
    def __init__(self, trackers: tuple[Tracker, ...]) -> None:
        self._trackers = trackers

    def log(self, values: Mapping[str, float], step: int) -> None:
        for tracker in self._trackers:
            tracker.log(values, step)

    def finish(self, summary: Mapping[str, object]) -> None:
        for tracker in self._trackers:
            tracker.finish(summary)


def build_tracker(
    metrics_path: Path,
    *,
    enabled: bool,
    mode: str,
    project: str,
    group: str,
    run_name: str,
    config: Mapping[str, object],
) -> Tracker:
    trackers: list[Tracker] = [JsonlTracker(metrics_path)]
    if enabled and mode != "disabled":
        try:
            trackers.append(build_wandb_tracker(
                mode=mode, project=project, group=group, run_name=run_name, config=config
            ))
        except Exception:
            if mode != "online":
                raise
            trackers.append(build_wandb_tracker(
                mode="offline", project=project, group=group, run_name=run_name, config=config
            ))
    return CompositeTracker(tuple(trackers))


def build_wandb_tracker(
    *, mode: str, project: str, group: str, run_name: str,
    config: Mapping[str, object],
) -> Tracker:
    if mode not in {"online", "offline"}:
        raise ValueError("W&B mode must be online, offline, or disabled")
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B tracking requires the optional 'wandb' dependency") from error
    run = wandb.init(
        project=project,
        group=group or None,
        name=run_name or None,
        mode=mode,
        config=dict(config),
    )
    return WandbTracker(run)


def _flatten(values: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[name] = value
    return flattened
