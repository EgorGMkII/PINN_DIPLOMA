"""Velocity-only adapter for the tab-separated PTV exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class PTVVelocityDataset:
    def __init__(self, path: str | Path, *, dtype: torch.dtype = torch.float32) -> None:
        values = np.loadtxt(Path(path), delimiter="\t", skiprows=1, usecols=range(7))
        if values.ndim != 2 or values.shape[1] != 7:
            raise ValueError("expected PTV columns t,X,Y,Z,VX,VY,VZ")
        if not np.isfinite(values).all():
            raise ValueError("PTV data contains NaN or Inf")
        self.points: Tensor = torch.as_tensor(values[:, :4], dtype=dtype)
        self._velocity: Tensor = torch.as_tensor(values[:, 4:7], dtype=dtype)

    @property
    def velocity(self) -> Tensor:
        return self._velocity

    @property
    def times(self) -> Tensor:
        return torch.unique(self.points[:, 0], sorted=True)
