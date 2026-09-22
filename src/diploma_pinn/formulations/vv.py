"""Direct strong velocity--vorticity formulation for Rayleigh--Benard flow."""

import math
from typing import Mapping

from torch import Tensor, nn

from diploma_pinn.formulations.vp import VPParameters
from diploma_pinn.operators.vv_derivatives import compute_vv_derivatives


class VVResiduals:
    """Compute vorticity transport, energy, and continuity residuals."""

    def __init__(self, parameters: VPParameters) -> None:
        self.parameters = parameters

    def residuals(self, model: nn.Module, points: Tensor) -> Mapping[str, Tensor]:
        values = compute_vv_derivatives(model, points)
        fields, first, second = values.fields, values.first, values.second
        omega, omega_first, omega_second = (
            values.vorticity,
            values.vorticity_first,
            values.vorticity_second,
        )
        viscosity = math.sqrt(self.parameters.prandtl / self.parameters.rayleigh)
        diffusivity = math.sqrt(1.0 / (self.parameters.prandtl * self.parameters.rayleigh))

        def laplacian(name: str) -> Tensor:
            return omega_second[f"{name}_xx"] + omega_second[f"{name}_yy"] + omega_second[f"{name}_zz"]

        u, v, w, temperature = (fields[name] for name in ("u", "v", "w", "T"))
        omega_x, omega_y, omega_z = (omega[name] for name in ("omega_x", "omega_y", "omega_z"))
        velocity = (u, v, w)
        vorticity = (omega_x, omega_y, omega_z)

        def transport(name: str, velocity_name: str) -> Tensor:
            return (
                omega_first[f"{name}_t"]
                + sum(component * omega_first[f"{name}_{axis}"] for component, axis in zip(velocity, ("x", "y", "z"), strict=True))
                - sum(component * first[f"{velocity_name}_{axis}"] for component, axis in zip(vorticity, ("x", "y", "z"), strict=True))
                - viscosity * laplacian(name)
            )

        energy = (
            first["T_t"]
            + u * first["T_x"]
            + v * first["T_y"]
            + w * first["T_z"]
            - diffusivity * (second["T_xx"] + second["T_yy"] + second["T_zz"])
        )
        return {
            "r_omega_x": transport("omega_x", "u") - first["T_y"],
            "r_omega_y": transport("omega_y", "v") + first["T_x"],
            "r_omega_z": transport("omega_z", "w"),
            "r_T": energy,
            "r_div": first["u_x"] + first["v_y"] + first["w_z"],
        }
