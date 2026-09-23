"""Mixed first-order Boussinesq formulation."""

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from diploma_pinn.formulations.vp import VPParameters
from diploma_pinn.operators.fo_derivatives import compute_fo_derivatives


class FOResiduals:
    def __init__(self, parameters: VPParameters) -> None:
        self.parameters = parameters

    def residuals(self, model: nn.Module, points: Tensor) -> Mapping[str, Tensor]:
        values = compute_fo_derivatives(model, points)
        f, d = values.fields, values.first
        nu = math.sqrt(self.parameters.prandtl / self.parameters.rayleigh)
        kappa = math.sqrt(1.0 / (self.parameters.prandtl * self.parameters.rayleigh))
        u, v, w = f["u"], f["v"], f["w"]

        def advection(prefix: str) -> Tensor:
            return u * f[f"G_{prefix}x"] + v * f[f"G_{prefix}y"] + w * f[f"G_{prefix}z"]

        def div_gradient(prefix: str) -> Tensor:
            return d[f"G_{prefix}x_x"] + d[f"G_{prefix}y_y"] + d[f"G_{prefix}z_z"]

        velocity_compat = []
        for field in ("u", "v", "w"):
            for axis in ("x", "y", "z"):
                velocity_compat.append(f[f"G_{field}{axis}"] - d[f"{field}_{axis}"])
        temperature_compat = [f[f"q_{axis}"] - d[f"T_{axis}"] for axis in ("x", "y", "z")]
        return {
            "r_u": d["u_t"] + advection("u") + d["p_x"] - nu * div_gradient("u"),
            "r_v": d["v_t"] + advection("v") + d["p_y"] - nu * div_gradient("v"),
            "r_w": d["w_t"] + advection("w") + d["p_z"] - nu * div_gradient("w") - f["T"],
            "r_T": d["T_t"] + u * f["q_x"] + v * f["q_y"] + w * f["q_z"]
            - kappa * (d["q_x_x"] + d["q_y_y"] + d["q_z_z"]),
            "r_div": f["G_ux"] + f["G_vy"] + f["G_wz"],
            "r_G": torch.stack(velocity_compat, dim=1),
            "r_q": torch.stack(temperature_compat, dim=1),
        }
