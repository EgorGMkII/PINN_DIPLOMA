"""Velocity-only observation operator for hidden T,p reconstruction."""

from torch import Tensor, nn

from diploma_pinn.contracts import ObservationBatch


def velocity_mse(model: nn.Module, observations: ObservationBatch) -> Tensor:
    """Return article-compatible joint MSE over the u,v,w components."""
    predicted = model(observations.points)
    if predicted.ndim != 2 or predicted.shape != (observations.points.shape[0], 5):
        raise ValueError(f"expected model output [N,5], got {tuple(predicted.shape)}")
    if observations.velocity.shape != (observations.points.shape[0], 3):
        raise ValueError("velocity labels must have shape [N,3]")
    return (predicted[:, :3] - observations.velocity).square().mean()
