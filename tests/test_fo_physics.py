import torch
from torch import nn

from diploma_pinn.formulations import FOResiduals, VPParameters


class HydrostaticFO(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        z = points[:, 3]
        zero = z * 0
        fields = [zero, zero, zero, z, 0.5 * z.square()]
        gradients = [zero] * 9
        temperature_gradient = [zero, zero, torch.ones_like(z)]
        return torch.stack((*fields, *gradients, *temperature_gradient), dim=1)


def test_fo_hydrostatic_solution_has_zero_residuals() -> None:
    residuals = FOResiduals(VPParameters()).residuals(HydrostaticFO(), torch.rand(7, 4))
    for residual in residuals.values():
        torch.testing.assert_close(residual, torch.zeros_like(residual))
