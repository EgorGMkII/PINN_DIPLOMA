import torch
from torch import nn

from diploma_pinn.boundaries import RBCBoundaryLoss
from diploma_pinn.contracts import ObservationBatch
from diploma_pinn.formulations import VPParameters, VPResiduals
from diploma_pinn.observations import velocity_mse
from diploma_pinn.operators import compute_vp_derivatives


class PolynomialFields(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        t, x, y, z = points.unbind(dim=1)
        return torch.stack((t + x.square(), y.square(), z.square(), x + y + z, x - y + z), dim=1)


class ConstantTemperature(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        output = torch.zeros((points.shape[0], 5), dtype=points.dtype, device=points.device)
        output[:, 3] = 0.5
        return output


def test_derivative_contract_for_polynomial_fields() -> None:
    points = torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.7, 0.8, 0.1, 0.9]])
    derivatives = compute_vp_derivatives(PolynomialFields(), points)
    torch.testing.assert_close(derivatives.first["u_t"], torch.ones(2))
    torch.testing.assert_close(derivatives.first["u_x"], 2 * points[:, 1])
    torch.testing.assert_close(derivatives.first["p_x"], torch.ones(2))
    assert "p_t" not in derivatives.first
    torch.testing.assert_close(derivatives.second["u_xx"], 2 * torch.ones(2))
    torch.testing.assert_close(derivatives.second["v_yy"], 2 * torch.ones(2))
    torch.testing.assert_close(derivatives.second["w_zz"], 2 * torch.ones(2))


def test_vp_zero_field_has_zero_residuals() -> None:
    model = nn.Linear(4, 5, bias=False)
    nn.init.zeros_(model.weight)
    residuals = VPResiduals(VPParameters()).residuals(model, torch.rand(4, 4))
    for residual in residuals.values():
        torch.testing.assert_close(residual, torch.zeros_like(residual))


def test_velocity_loss_uses_only_velocity_outputs() -> None:
    model = PolynomialFields()
    points = torch.tensor([[0.0, 0.2, 0.3, 0.4]])
    labels = model(points)[:, :3].detach()
    torch.testing.assert_close(velocity_mse(model, ObservationBatch(points, labels)), torch.tensor(0.0))


def test_active_temperature_boundary_targets_match_article() -> None:
    model = ConstantTemperature()
    boundary = {"z0": torch.rand(3, 4), "z1": torch.rand(3, 4)}
    losses = RBCBoundaryLoss()(model, boundary)
    torch.testing.assert_close(losses["temperature_z0"], torch.tensor(0.0))
    torch.testing.assert_close(losses["temperature_z1"], torch.tensor(1.0))
