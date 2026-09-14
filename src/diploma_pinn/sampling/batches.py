"""Assembly of independently sampled point populations into one train batch."""

import torch

from diploma_pinn.contracts import ObservationBatch, PointBatch
from diploma_pinn.sampling.protocols import BoundarySampler, DomainSampler, ObservationSampler


class RBCBatchSource:
    """Move observations and build matching PDE/boundary points for one update."""

    def __init__(
        self,
        observations: ObservationSampler,
        domain: DomainSampler,
        boundaries: BoundarySampler,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._observations = observations
        self._domain = domain
        self._boundaries = boundaries
        self._device = device
        self._dtype = dtype

    def next_batch(self) -> PointBatch:
        observed = self._observations.sample()
        observations = ObservationBatch(
            points=observed.points.to(device=self._device, dtype=self._dtype),
            velocity=observed.velocity.to(device=self._device, dtype=self._dtype),
        )
        times = observations.points[:, :1]
        return PointBatch(
            observations=observations,
            domain=self._domain.sample(times),
            boundaries=self._boundaries.sample(times),
        )
