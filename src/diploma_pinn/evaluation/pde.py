"""Independent residual evaluation outside the training point stream."""

from typing import Mapping

from torch import Tensor, nn


def compute_pde_residual_metrics(
    model: nn.Module, formulation: object, points: Tensor
) -> Mapping[str, float]:
    was_training = model.training
    model.eval()
    residuals = formulation.residuals(model, points)
    result = {
        name: float(value.detach().square().mean().sqrt())
        for name, value in residuals.items()
    }
    model.train(was_training)
    return result
