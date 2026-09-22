import torch
from torch import nn

from diploma_pinn.formulations import VPParameters, VVResiduals
from diploma_pinn.models import SineMLP
from diploma_pinn.operators import compute_vv_derivatives


class ZeroVVFields(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return points[:, :4] * 0


class SolidBodyRotation(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        _, x, y, _ = points.unbind(dim=1)
        return torch.stack((-y, x, x * 0, x * 0), dim=1)


class LinearTemperature(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        _, x, _, _ = points.unbind(dim=1)
        return torch.stack((x * 0, x * 0, x * 0, x), dim=1)


def test_vv_zero_field_has_zero_residuals() -> None:
    residuals = VVResiduals(VPParameters()).residuals(ZeroVVFields(), torch.rand(4, 4))
    for residual in residuals.values():
        torch.testing.assert_close(residual, torch.zeros_like(residual))


def test_vv_curl_of_solid_body_rotation_is_constant_and_residual_is_zero() -> None:
    points = torch.rand(5, 4)
    derivatives = compute_vv_derivatives(SolidBodyRotation(), points)
    torch.testing.assert_close(derivatives.vorticity["omega_x"], torch.zeros(5))
    torch.testing.assert_close(derivatives.vorticity["omega_y"], torch.zeros(5))
    torch.testing.assert_close(derivatives.vorticity["omega_z"], 2 * torch.ones(5))
    residuals = VVResiduals(VPParameters()).residuals(SolidBodyRotation(), points)
    for residual in residuals.values():
        torch.testing.assert_close(residual, torch.zeros_like(residual))


def test_vv_buoyancy_curl_uses_temperature_gradient_with_expected_sign() -> None:
    residuals = VVResiduals(VPParameters()).residuals(LinearTemperature(), torch.rand(3, 4))
    torch.testing.assert_close(residuals["r_omega_x"], torch.zeros(3))
    torch.testing.assert_close(residuals["r_omega_y"], torch.ones(3))
    torch.testing.assert_close(residuals["r_omega_z"], torch.zeros(3))
    torch.testing.assert_close(residuals["r_T"], torch.zeros(3))
    torch.testing.assert_close(residuals["r_div"], torch.zeros(3))


def test_vv_residuals_remain_connected_to_model_parameters() -> None:
    model = SineMLP((4, 8, 4))
    residuals = VVResiduals(VPParameters()).residuals(model, torch.rand(4, 4))
    loss = sum(residual.square().mean() for residual in residuals.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
