"""Loader for the article's combined RBC DNS NPZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class RBCDNSDataset:
    """Read DNS fields while exposing only velocity through the training API."""

    def __init__(self, path: str | Path, *, dtype: torch.dtype = torch.float32) -> None:
        with np.load(Path(path), allow_pickle=False) as archive:
            inputs = archive["inputs"]
            outputs = archive["outputs"]
        if inputs.ndim != 2 or inputs.shape[1] != 4:
            raise ValueError(f"expected inputs [N,4], got {inputs.shape}")
        if outputs.shape != (inputs.shape[0], 5):
            raise ValueError(f"expected outputs [N,5], got {outputs.shape}")
        self.points: Tensor = torch.as_tensor(inputs, dtype=dtype)
        self._fields: Tensor = torch.as_tensor(outputs, dtype=dtype)

    @property
    def velocity(self) -> Tensor:
        return self._fields[:, :3]

    def evaluation_fields(self) -> Tensor:
        """Return ``(u,v,w,T,p)`` only to evaluation code."""
        return self._fields

    @property
    def times(self) -> Tensor:
        return torch.unique(self.points[:, 0], sorted=True)


class VelocityDatasetView:
    """A deterministic velocity-only subset, optionally with synthetic noise."""

    def __init__(self, dataset: object, *, fraction: float, per_time: int,
                 noise_std: float, seed: int, holdout_fraction: float = 0.0,
                 partition: str = "train") -> None:
        if partition not in {"train", "holdout"}:
            raise ValueError("partition must be train or holdout")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selections = []
        for time in dataset.times:
            candidates = torch.nonzero(dataset.points[:, 0] == time, as_tuple=False).flatten()
            count = min(candidates.numel(), per_time if per_time else max(1, round(candidates.numel() * fraction)))
            selected = candidates[torch.randperm(candidates.numel(), generator=generator)[:count]]
            holdout = round(count * holdout_fraction)
            selections.append(selected[:-holdout] if holdout and partition == "train"
                              else selected[-holdout:] if holdout else selected[:0] if partition == "holdout"
                              else selected)
        indices = torch.cat(selections)
        self.points = dataset.points.index_select(0, indices)
        self._velocity = dataset.velocity.index_select(0, indices).clone()
        if noise_std:
            noise = torch.randn(self._velocity.shape, generator=generator, dtype=self._velocity.dtype)
            self._velocity.add_(noise, alpha=noise_std)

    @property
    def velocity(self) -> Tensor:
        return self._velocity

    @property
    def times(self) -> Tensor:
        return torch.unique(self.points[:, 0], sorted=True)
