"""Create a small time-stratified smoke NPZ from article RBC snapshots."""

import argparse
import hashlib
from pathlib import Path

import numpy as np


def build_smoke_dataset(raw_dir: Path, output: Path, rows_per_time: int, seed: int) -> str:
    files = sorted(raw_dir.glob("Points_1E6_07_*.npz"))
    if not files:
        raise FileNotFoundError(f"no RBC snapshots found in {raw_dir}")
    if rows_per_time <= 0:
        raise ValueError("rows_per_time must be positive")

    generator = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            data = archive["data"]
        if data.ndim != 2 or data.shape[1] != 9:
            raise ValueError(f"expected {path.name} data [N,9], got {data.shape}")
        if rows_per_time >= data.shape[0]:
            samples.append(data)
        else:
            indices = generator.choice(data.shape[0], size=rows_per_time, replace=False)
            samples.append(data[indices])

    combined = np.concatenate(samples, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, inputs=combined[:, :4], outputs=combined[:, 4:])
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"created {output} rows={combined.shape[0]} sha256={digest}")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-time", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2204)
    arguments = parser.parse_args()
    build_smoke_dataset(
        arguments.raw_dir, arguments.output, arguments.rows_per_time, arguments.seed
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
