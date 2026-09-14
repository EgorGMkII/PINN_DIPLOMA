"""Machine-readable reproducibility manifest."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RunManifest:
    experiment_id: str
    run_id: str
    git_commit: str
    git_dirty: bool
    dataset_sha256: str
    seed: int
    environment: Mapping[str, Any]
    resolved_config: Mapping[str, Any]

    def write_json(self, path: Path) -> None:
        raise NotImplementedError("write atomically and validate required fields")


def collect_environment() -> Mapping[str, Any]:
    raise NotImplementedError("collect Python, PyTorch, CUDA, GPU and platform versions")
