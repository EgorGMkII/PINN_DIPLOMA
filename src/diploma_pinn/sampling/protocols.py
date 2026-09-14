"""Sampling interfaces; policies are selected before building the train step."""

from typing import Mapping, Protocol

from torch import Tensor

from diploma_pinn.contracts import ObservationBatch


class ObservationSampler(Protocol):
    def sample(self) -> ObservationBatch: ...


class DomainSampler(Protocol):
    def sample(self, times: Tensor) -> Tensor: ...


class BoundarySampler(Protocol):
    def sample(self, times: Tensor) -> Mapping[str, Tensor]: ...
