"""Rayleigh-Benard cube boundary point sampling."""

from typing import Mapping

import torch
from torch import Tensor


class RBCBoundarySampler:
    def __init__(self, *, device: torch.device, dtype: torch.dtype, seed: int) -> None:
        self._device = device
        self._dtype = dtype
        self._generator = torch.Generator(device=device.type).manual_seed(seed)

    def sample(self, times: Tensor) -> Mapping[str, Tensor]:
        if times.ndim not in (1, 2) or (times.ndim == 2 and times.shape[1] != 1):
            raise ValueError("times must have shape [N] or [N,1]")
        times = times.reshape(-1, 1).to(device=self._device, dtype=self._dtype)
        count = times.shape[0]
        uniform = lambda columns: torch.rand(
            (count, columns), device=self._device, dtype=self._dtype, generator=self._generator
        )
        binary = lambda: torch.randint(
            0,
            2,
            (count, 1),
            device=self._device,
            dtype=self._dtype,
            generator=self._generator,
        )
        x_face = torch.cat((times, binary(), uniform(2)), dim=1)
        y_face = torch.cat((times, uniform(1), binary(), uniform(1)), dim=1)
        z_xy = uniform(2)
        z0_face = torch.cat((times, z_xy, torch.zeros_like(times)), dim=1)
        z1_face = torch.cat((times, uniform(2), torch.ones_like(times)), dim=1)
        return {"x": x_face, "y": y_face, "z0": z0_face, "z1": z1_face}
