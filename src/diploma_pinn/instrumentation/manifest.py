"""Machine-readable reproducibility manifest."""

from dataclasses import dataclass
from dataclasses import asdict
import json
import platform
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


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
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)


def collect_environment() -> Mapping[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    }
