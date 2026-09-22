"""Named loss aggregation independent of sampling and PDE formulation."""

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor

from diploma_pinn.contracts import LossReport


@dataclass(frozen=True)
class LossWeights:
    data: float = 1.0
    momentum: float = 1e-1
    vorticity: float = 1e-1
    energy: float = 1e-2
    continuity: float = 1e-3
    boundary: float = 1e-4


class LossAssembler:
    """Combine already-reduced active components into one backward scalar."""

    def __init__(self, weights: LossWeights) -> None:
        self.weights = weights

    def __call__(self, components: Mapping[str, Tensor]) -> LossReport:
        weighted: dict[str, Tensor] = {}
        for name, value in components.items():
            weight = getattr(self.weights, name)
            if weight != 0:
                weighted[name] = weight * value
        if not weighted:
            raise ValueError("at least one active loss component is required")
        return LossReport(total=sum(weighted.values()), components=dict(components))
