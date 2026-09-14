"""Strong velocity-pressure formulation contract for Rayleigh-Benard flow."""

from dataclasses import dataclass
from typing import Mapping, Protocol

import math
from torch import Tensor, nn

from diploma_pinn.operators import compute_vp_derivatives


class ResidualFormulation(Protocol):
    def residuals(self, model: nn.Module, points: Tensor) -> Mapping[str, Tensor]: ...


@dataclass(frozen=True)
class VPParameters:
    rayleigh: float = 1e6
    prandtl: float = 0.7


class VPResiduals:
    """Compute ``momentum``, ``energy`` and ``continuity`` residual tensors."""

    def __init__(self, parameters: VPParameters) -> None:
        self.parameters = parameters

    def residuals(self, model: nn.Module, points: Tensor) -> Mapping[str, Tensor]:
        values = compute_vp_derivatives(model, points)
        fields, first, second = values.fields, values.first, values.second
        viscosity = math.sqrt(self.parameters.prandtl / self.parameters.rayleigh)
        diffusivity = math.sqrt(1.0 / (self.parameters.prandtl * self.parameters.rayleigh))

        def laplacian(name: str) -> Tensor:
            return second[f"{name}_xx"] + second[f"{name}_yy"] + second[f"{name}_zz"]

        u, v, w, temperature = (fields[name] for name in ("u", "v", "w", "T"))
        convection_u = u * first["u_x"] + v * first["u_y"] + w * first["u_z"]
        convection_v = u * first["v_x"] + v * first["v_y"] + w * first["v_z"]
        convection_w = u * first["w_x"] + v * first["w_y"] + w * first["w_z"]
        convection_T = u * first["T_x"] + v * first["T_y"] + w * first["T_z"]
        return {
            "r_u": first["u_t"] + convection_u + first["p_x"] - viscosity * laplacian("u"),
            "r_v": first["v_t"] + convection_v + first["p_y"] - viscosity * laplacian("v"),
            "r_w": first["w_t"] + convection_w + first["p_z"] - viscosity * laplacian("w") - temperature,
            "r_T": first["T_t"] + convection_T - diffusivity * laplacian("T"),
            "r_div": first["u_x"] + first["v_y"] + first["w_z"],
        }
