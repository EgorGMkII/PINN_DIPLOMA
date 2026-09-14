"""Active Rayleigh-Benard cube boundary constraints."""

from typing import Mapping

from torch import Tensor, nn


class RBCBoundaryLoss:
    """Return named losses for boundary terms active in the selected profile."""

    def __call__(self, model: nn.Module, points: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        required = {"z0", "z1"}
        missing = required.difference(points)
        if missing:
            raise ValueError(f"missing required boundary points: {sorted(missing)}")
        z0_temperature = model(points["z0"])[:, 3]
        z1_temperature = model(points["z1"])[:, 3]
        return {
            "temperature_z0": (z0_temperature - 0.5).square().mean(),
            "temperature_z1": (z1_temperature + 0.5).square().mean(),
        }
