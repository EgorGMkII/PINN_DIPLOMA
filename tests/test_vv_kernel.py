import torch
from torch import nn

from diploma_pinn.boundaries import RBCBoundaryLoss
from diploma_pinn.contracts import ObservationBatch, PointBatch
from diploma_pinn.formulations import VPParameters, VVResiduals
from diploma_pinn.losses import LossAssembler, LossWeights, VVLossKernel


def test_vv_kernel_assembles_pressure_free_components() -> None:
    model = nn.Linear(4, 4, bias=False)
    nn.init.zeros_(model.weight)
    points = torch.rand(3, 4)
    batch = PointBatch(
        observations=ObservationBatch(points, torch.zeros(3, 3)),
        domain=points,
        boundaries={"z0": points, "z1": points},
    )
    kernel = VVLossKernel(
        VVResiduals(VPParameters()),
        LossAssembler(LossWeights()),
        RBCBoundaryLoss(),
    )
    report = kernel(model, batch)
    assert set(report.components) == {"data", "vorticity", "energy", "continuity", "boundary"}
    torch.testing.assert_close(report.components["boundary"], torch.tensor(0.5))
    torch.testing.assert_close(report.total, torch.tensor(0.5e-4))
    report.total.backward()
    assert model.weight.grad is not None
