"""Mode-resolved diagnostics for hidden temperature and pressure fields."""

from dataclasses import dataclass

import torch
from torch import Tensor

from diploma_pinn.evaluation.metrics import align_pressure_gauge, compute_scalar_metrics


@dataclass(frozen=True)
class FieldDiagnostics:
    temperature_profile: dict[str, object]
    temperature_fluctuations: dict[str, float]
    pressure_hydrostatic: dict[str, object] | None
    pressure_dynamic: dict[str, float] | None
    convective_heat_flux_profile: dict[str, object]


def compute_field_diagnostics(predicted: Tensor, target: Tensor, points: Tensor) -> FieldDiagnostics:
    """Decompose fields by horizontal planes for the RBC DNS grid."""
    if (
        predicted.ndim != 2
        or predicted.shape[1] not in (4, 5)
        or target.shape != (predicted.shape[0], 5)
    ):
        raise ValueError("predicted fields must be [N,4] or [N,5]; target fields must be [N,5]")
    if points.shape != (predicted.shape[0], 4):
        raise ValueError("points must have shape [N,4]")
    times, time_index = torch.unique(points[:, 0], sorted=True, return_inverse=True)
    heights, height_index = torch.unique(points[:, 3], sorted=True, return_inverse=True)
    groups = time_index * heights.numel() + height_index
    count = torch.bincount(groups, minlength=times.numel() * heights.numel()).to(dtype=predicted.dtype)
    valid_planes = count > 0

    def plane_mean(values: Tensor) -> Tensor:
        sums = torch.zeros_like(count)
        sums.scatter_add_(0, groups, values)
        return (sums / count.clamp_min(1)).reshape(times.numel(), heights.numel())

    temperature_predicted = plane_mean(predicted[:, 3])
    temperature_target = plane_mean(target[:, 3])
    temperature_fluctuation_predicted = predicted[:, 3] - temperature_predicted[time_index, height_index]
    temperature_fluctuation_target = target[:, 3] - temperature_target[time_index, height_index]
    flux_predicted = plane_mean(predicted[:, 2] * predicted[:, 3])
    flux_target = plane_mean(target[:, 2] * target[:, 3])

    def profile(prediction: Tensor, truth: Tensor) -> dict[str, object]:
        valid = valid_planes.reshape(times.numel(), heights.numel())
        return {
            "metrics": compute_scalar_metrics(prediction[valid], truth[valid]),
            "times": times.tolist(),
            "z": heights.tolist(),
            "predicted": prediction.tolist(),
            "target": truth.tolist(),
            "valid": valid.tolist(),
        }

    pressure_hydrostatic: dict[str, object] | None = None
    pressure_dynamic: dict[str, float] | None = None
    if predicted.shape[1] == 5:
        predicted_pressure = align_pressure_gauge(predicted[:, 4], target[:, 4], points[:, 0])
        pressure_predicted = plane_mean(predicted_pressure)
        pressure_target = plane_mean(target[:, 4])
        pressure_dynamic_predicted = predicted_pressure - pressure_predicted[time_index, height_index]
        pressure_dynamic_target = target[:, 4] - pressure_target[time_index, height_index]
        pressure_hydrostatic = profile(pressure_predicted, pressure_target)
        pressure_dynamic = compute_scalar_metrics(pressure_dynamic_predicted, pressure_dynamic_target)

    return FieldDiagnostics(
        temperature_profile=profile(temperature_predicted, temperature_target),
        temperature_fluctuations=compute_scalar_metrics(temperature_fluctuation_predicted, temperature_fluctuation_target),
        pressure_hydrostatic=pressure_hydrostatic,
        pressure_dynamic=pressure_dynamic,
        convective_heat_flux_profile=profile(flux_predicted, flux_target),
    )
