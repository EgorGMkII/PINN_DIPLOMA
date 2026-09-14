"""Unlabelled interior collocation sampling."""

import torch
from torch import Tensor


class UniformDomainSampler:
    def __init__(self, *, device: torch.device, dtype: torch.dtype, seed: int) -> None:
        self._device = device
        self._dtype = dtype
        self._generator = torch.Generator(device=device.type).manual_seed(seed)

    def sample(self, times: Tensor) -> Tensor:
        """Reuse times and draw x,y,z independently from Uniform(0,1)."""
        if times.ndim not in (1, 2) or (times.ndim == 2 and times.shape[1] != 1):
            raise ValueError("times must have shape [N] or [N,1]")
        times = times.reshape(-1, 1).to(device=self._device, dtype=self._dtype)
        spatial = torch.rand(
            (times.shape[0], 3),
            device=self._device,
            dtype=self._dtype,
            generator=self._generator,
        )
        return torch.cat((times, spatial), dim=1)
