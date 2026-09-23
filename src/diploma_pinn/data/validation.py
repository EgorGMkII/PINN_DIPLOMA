"""Dataset fingerprint and schema validation before allocation on an accelerator."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetReport:
    path: Path
    sha256: str
    rows: int
    unique_times: tuple[float, ...]
    finite: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the file at once."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rbc_dns(path: Path, expected_sha256: str) -> DatasetReport:
    """Validate the combined article NPZ before a training process starts."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"dataset fingerprint mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"inputs", "outputs"}:
            raise ValueError(f"expected NPZ keys inputs, outputs; got {archive.files}")
        inputs = archive["inputs"]
        outputs = archive["outputs"]

    if inputs.ndim != 2 or inputs.shape[1] != 4:
        raise ValueError(f"expected inputs [N,4], got {inputs.shape}")
    if outputs.shape != (inputs.shape[0], 5):
        raise ValueError(f"expected outputs [N,5], got {outputs.shape}")
    if not np.issubdtype(inputs.dtype, np.number) or not np.issubdtype(outputs.dtype, np.number):
        raise ValueError("inputs and outputs must be numeric")

    finite = bool(np.isfinite(inputs).all() and np.isfinite(outputs).all())
    if not finite:
        raise ValueError("dataset contains NaN or Inf")
    coordinates = inputs[:, 1:]
    if np.any(coordinates < 0.0) or np.any(coordinates > 1.0):
        raise ValueError("RBC coordinates must lie in [0, 1]")

    times = tuple(float(value) for value in np.unique(inputs[:, 0]))
    if not times:
        raise ValueError("dataset contains no rows")
    return DatasetReport(
        path=path,
        sha256=actual_sha256,
        rows=int(inputs.shape[0]),
        unique_times=times,
        finite=finite,
    )


def validate_ptv_csv(path: Path, expected_sha256: str) -> DatasetReport:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(f"dataset fingerprint mismatch: expected {expected_sha256}, got {actual_sha256}")
    values = np.loadtxt(path, delimiter="\t", skiprows=1, usecols=range(7))
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("invalid PTV table")
    coordinates = values[:, 1:4]
    if np.any(coordinates < 0.0) or np.any(coordinates > 1.0):
        raise ValueError("PTV coordinates must lie in [0,1]")
    return DatasetReport(
        path=path, sha256=actual_sha256, rows=int(values.shape[0]),
        unique_times=tuple(float(value) for value in np.unique(values[:, 0])), finite=True,
    )
