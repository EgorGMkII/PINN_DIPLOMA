import torch
from torch import nn

from diploma_pinn.config.schema import OptimizerConfig
from diploma_pinn.evaluation.diagnostics import compute_field_diagnostics
from diploma_pinn.evaluation.metrics import align_pressure_gauge, compute_field_metrics
from diploma_pinn.instrumentation.checkpoints import load_checkpoint, save_checkpoint
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


def test_diagnostics_separate_horizontal_mean_from_fluctuations() -> None:
    points = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
    )
    target = torch.zeros(4, 5)
    target[:, 3] = torch.tensor([1.0, 3.0, 2.0, 4.0])
    target[:, 4] = torch.tensor([5.0, 7.0, 6.0, 8.0])
    predicted = target.clone()
    predicted[:, 3] += torch.tensor([1.0, 1.0, -2.0, -2.0])
    diagnostics = compute_field_diagnostics(predicted, target, points)
    assert diagnostics.temperature_profile["metrics"]["mse"] == 2.5
    assert diagnostics.temperature_fluctuations["mse"] == 0.0
    assert len(diagnostics.pressure_hydrostatic["z"]) == 2


def test_diagnostics_handle_incomplete_time_height_coverage() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 1.0]])
    target = torch.zeros(2, 5)
    predicted = target.clone()
    diagnostics = compute_field_diagnostics(predicted, target, points)
    assert diagnostics.temperature_profile["metrics"]["mse"] == 0.0
    assert diagnostics.convective_heat_flux_profile["metrics"]["correlation"] == 0.0


def test_checkpoint_round_trip(tmp_path) -> None:
    model = nn.Linear(2, 1)
    optimizer = build_optimizer(model, OptimizerConfig())
    path = tmp_path / "final.pt"
    save_checkpoint(path, model, optimizer, step=7, config_fingerprint="test-config")
    restored = nn.Linear(2, 1)
    restored_optimizer = build_optimizer(restored, OptimizerConfig())
    step = load_checkpoint(
        path, restored, restored_optimizer, expected_config_fingerprint="test-config"
    )
    assert step == 7
    for actual, expected in zip(restored.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected)
