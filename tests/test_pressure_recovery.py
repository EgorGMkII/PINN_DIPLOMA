import torch
from torch import nn

from diploma_pinn.formulations import VPParameters
from diploma_pinn.pressure import momentum_pressure_gradient, pressure_loss


class BuoyancyOnlyVV(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        z = points[:, 3] * self.scale
        zero = z * 0
        return torch.stack((zero, zero, zero, z), dim=1)


class QuadraticPressure(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return 0.5 * points[:, 3:4].square()


def test_momentum_gradient_matches_hydrostatic_pressure() -> None:
    model = BuoyancyOnlyVV()
    points = torch.rand(8, 4)
    target = momentum_pressure_gradient(model, points, VPParameters())
    expected = torch.stack((points[:, 3] * 0, points[:, 3] * 0, points[:, 3]), dim=1)
    torch.testing.assert_close(target, expected)
    assert not target.requires_grad

    loss, components = pressure_loss(QuadraticPressure(), points, target, gauge_weight=0.0)
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    assert model.scale.grad is None
    assert components["pressure/gradient"].item() == 0.0


def test_pressure_gauge_is_computed_independently_per_time() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0, 2.0], [1.0, 0.0, 0.0, 2.0]])
    target = torch.zeros(4, 3)
    _, components = pressure_loss(QuadraticPressure(), points, target, gauge_weight=1.0)
    expected = torch.tensor((0.5 ** 2 + 2.0 ** 2) / 2)
    torch.testing.assert_close(components["pressure/gauge"], expected)
