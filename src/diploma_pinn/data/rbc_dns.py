"""Loader for the article's combined RBC DNS NPZ."""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class RBCDNSDataset:
    """Read DNS fields while exposing only velocity through the training API."""

    def __init__(self, path: str | Path, *, dtype: torch.dtype = torch.float32) -> None:
        archive = np.load(Path(path), allow_pickle=False)
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
