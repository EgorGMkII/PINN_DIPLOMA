import torch
from torch import nn

from diploma_pinn.config.schema import OptimizerConfig
from diploma_pinn.evaluation.metrics import align_pressure_gauge, compute_field_metrics
from diploma_pinn.runtime import resolve_device, resolve_dtype
from diploma_pinn.training.optimizers import build_optimizer, build_scheduler


def test_pressure_alignment_removes_independent_time_gauges() -> None:
    target = torch.tensor([1.0, 3.0, -2.0, 4.0])
    predicted = torch.tensor([11.0, 13.0, -7.0, -1.0])
    times = torch.tensor([0.0, 0.0, 0.5, 0.5])
    torch.testing.assert_close(align_pressure_gauge(predicted, target, times), target)


def test_metrics_use_pressure_gauge_alignment() -> None:
    target = torch.ones(4, 5)
    predicted = target.clone()
    predicted[:2, 4] += 10
    predicted[2:, 4] -= 4
    metrics = compute_field_metrics(predicted, target, torch.tensor([0.0, 0.0, 0.5, 0.5]))
    assert metrics.mse["p"] == 0.0
    assert metrics.relative_l2["u"] == 0.0


def test_reference_adam_factory_and_runtime_resolution() -> None:
    optimizer = build_optimizer(nn.Linear(1, 1), OptimizerConfig())
    assert optimizer.defaults["eps"] == 1e-7
    assert optimizer.defaults["weight_decay"] == 0.0
    assert build_scheduler(optimizer, OptimizerConfig()) is not None
    assert resolve_device("cpu").type == "cpu"
    assert resolve_dtype("float32") is torch.float32
