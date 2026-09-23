"""Command-line entry points for validation, parity, training, and benchmarking."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import math

from diploma_pinn.analysis import analyze_temperature_modes, write_mode_report
from diploma_pinn.config import config_to_dict, load_config, write_resolved_config
from diploma_pinn.data import validate_ptv_csv, validate_rbc_dns
from diploma_pinn.evaluation import (
    Evaluator, compute_field_metrics, compute_pde_residual_metrics, compute_scalar_metrics,
)
from diploma_pinn.instrumentation import StepProfiler, run_benchmark
from diploma_pinn.instrumentation.checkpoints import save_checkpoint
from diploma_pinn.instrumentation.manifest import RunManifest, collect_environment
from diploma_pinn.integrations import build_tracker
from diploma_pinn.formulations import FOResiduals, VPParameters, VPResiduals, VVResiduals
from diploma_pinn.sampling import UniformDomainSampler
from diploma_pinn.pressure import append_recovered_pressure, train_pressure_recovery
from diploma_pinn.runtime import seed_everything
from diploma_pinn.training.factory import build_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diploma-pinn")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-data", "parity", "train", "benchmark", "evaluate", "observability"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--local-overlay", type=Path)
        if command == "train":
            child.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_config(arguments.config, local_overlay=arguments.local_overlay)
    if arguments.command == "validate-data":
        report = _validate_data(config)
        print(json.dumps({**report.__dict__, "path": str(report.path)}, indent=2, default=list))
        return 0
    if arguments.command == "train":
        return _run_training(config, require_smoke=arguments.smoke)
    if arguments.command == "benchmark":
        return _run_benchmark(config)
    if arguments.command == "observability":
        return _run_observability(config)
    raise NotImplementedError(f"command is not implemented yet: {arguments.command}")


def _run_training(config, *, require_smoke: bool) -> int:
    if require_smoke and config.runtime.execution_profile != "smoke":
        raise ValueError("--smoke requires runtime.execution_profile: smoke")
    report = _validate_data(config)
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = config.runtime.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    write_resolved_config(config, output_dir / "resolved_config.yaml")
    tracker = build_tracker(
        output_dir / "metrics.jsonl",
        enabled=config.tracking.enabled,
        mode=config.tracking.mode,
        project=config.tracking.project,
        group=config.tracking.group,
        run_name=config.tracking.run_name or f"{config.experiment_id}-{run_id}",
        config=config_to_dict(config),
    )
    profiler = StepProfiler(enabled=config.runtime.profile_steps)
    experiment = build_experiment(config, tracker=tracker, profiler=profiler)
    if config.runtime.evaluation_interval_epochs:
        if config.data.kind != "rbc_dns":
            raise ValueError("intermediate hidden-field evaluation requires rbc_dns data")
        experiment.trainer.set_epoch_callback(
            _sample_evaluation_callback(config, experiment, config.runtime.evaluation_interval_epochs)
        )
    before = [parameter.detach().clone() for parameter in experiment.model.parameters()]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    metrics = experiment.trainer.run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    parameters_changed = any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, experiment.model.parameters())
    )
    if not parameters_changed:
        raise RuntimeError("optimizer completed but no model parameter changed")

    summary = {
        "status": "ok",
        "experiment_id": config.experiment_id,
        "run_id": run_id,
        "steps": config.runtime.max_steps,
        "elapsed_seconds": elapsed,
        "device": str(next(experiment.model.parameters()).device),
        "parameters_changed": parameters_changed,
        "loss_total": metrics.total,
        "loss_components": dict(metrics.components),
        "dataset_rows": report.rows,
        "dataset_sha256": report.sha256,
    }
    auxiliary: dict[str, object] = {}
    pressure_result = None
    if config.pressure_recovery.enabled:
        pressure_result = train_pressure_recovery(
            experiment.model,
            experiment.dataset,
            VPParameters(config.physics.rayleigh, config.physics.prandtl),
            config.pressure_recovery,
            tracker,
            seed=config.runtime.seed + 100,
            step_offset=config.runtime.max_steps,
        )
        summary["pressure_recovery"] = {
            "steps": pressure_result.steps,
            "final_loss": pressure_result.final_loss,
        }
        auxiliary["pressure_recovery"] = {
            "model": pressure_result.model.state_dict(),
            "optimizer": pressure_result.optimizer.state_dict(),
            "steps": pressure_result.steps,
        }

    config_fingerprint = _config_fingerprint(config)
    if config.runtime.save_final_checkpoint:
        save_checkpoint(
            output_dir / "checkpoints" / "final.pt",
            experiment.model,
            experiment.optimizer,
            config.runtime.max_steps - 1,
            scheduler=experiment.scheduler,
            config_fingerprint=config_fingerprint,
            auxiliary=auxiliary,
        )

    if experiment.holdout is not None and experiment.holdout.points.shape[0]:
        holdout_evaluator = Evaluator(experiment.dataset, config.runtime.evaluation_batch_size)
        holdout_prediction = holdout_evaluator.predict_points(
            experiment.model, experiment.holdout.points
        )[:, :3]
        summary["holdout_velocity"] = {
            name: compute_scalar_metrics(holdout_prediction[:, index], experiment.holdout.velocity[:, index])
            for index, name in enumerate(("u", "v", "w"))
        }

    if config.runtime.evaluate_after_run:
        if config.data.kind != "rbc_dns":
            raise ValueError("full hidden-field evaluation requires rbc_dns data")
        evaluator = Evaluator(experiment.dataset, config.runtime.evaluation_batch_size)
        predicted = evaluator.predict_full(experiment.model)
        if pressure_result is not None:
            recovered = evaluator.predict_points(pressure_result.model, experiment.dataset.points)
            predicted = append_recovered_pressure(predicted, recovered)
        evaluation = compute_field_metrics(
            predicted, experiment.dataset.evaluation_fields(), experiment.dataset.points[:, 0]
        )
        summary["evaluation"] = {
            "mse": evaluation.mse,
            "relative_l2": evaluation.relative_l2,
            "correlation": evaluation.correlation,
        }
        if config.runtime.write_diagnostics:
            diffusivity = math.sqrt(1.0 / (config.physics.prandtl * config.physics.rayleigh))
            diagnostics = evaluator.diagnose(predicted, diffusivity=diffusivity)
            (output_dir / "diagnostics.json").write_text(
                json.dumps(diagnostics.__dict__, indent=2, sort_keys=True), encoding="utf-8"
            )
        if config.runtime.pde_evaluation_size:
            device = next(experiment.model.parameters()).device
            dtype = next(experiment.model.parameters()).dtype
            generator = torch.Generator(device="cpu").manual_seed(config.runtime.seed + 20_000)
            time_indices = torch.randint(
                experiment.dataset.points.shape[0],
                (config.runtime.pde_evaluation_size,), generator=generator,
            )
            times = experiment.dataset.points[time_indices, :1].to(device=device, dtype=dtype)
            points = UniformDomainSampler(
                device=device, dtype=dtype, seed=config.runtime.seed + 20_001
            ).sample(times)
            parameters = VPParameters(config.physics.rayleigh, config.physics.prandtl)
            formulations = {"vp": VPResiduals, "vv": VVResiduals, "fo": FOResiduals}
            summary["pde_residual_rms"] = dict(compute_pde_residual_metrics(
                experiment.model, formulations[config.physics.formulation](parameters), points
            ))
    (output_dir / "profiler.json").write_text(
        json.dumps(profiler.summary(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    commit, dirty = _git_state()
    RunManifest(
        experiment_id=config.experiment_id,
        run_id=run_id,
        git_commit=commit,
        git_dirty=dirty,
        dataset_sha256=report.sha256,
        seed=config.runtime.seed,
        environment=collect_environment(),
        resolved_config=config_to_dict(config),
    ).write_json(output_dir / "manifest.json")
    tracker.finish(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _sample_evaluation_callback(config, experiment, interval: int):
    indices = _stratified_indices(
        experiment.dataset.points[:, 0],
        min(config.runtime.evaluation_sample_size, experiment.dataset.points.shape[0]),
        config.runtime.seed + 10_000,
    )
    evaluator = Evaluator(experiment.dataset, config.runtime.evaluation_batch_size)

    def evaluate(epoch: int) -> dict[str, float]:
        if epoch % interval:
            return {}
        predicted = evaluator.predict_points(experiment.model, experiment.dataset.points[indices])
        target = experiment.dataset.evaluation_fields()[indices]
        metrics = compute_field_metrics(predicted, target, experiment.dataset.points[indices, 0])
        values: dict[str, float] = {}
        for name in ("T", "p"):
            if name in metrics.correlation:
                values[f"evaluation/correlation/{name}"] = metrics.correlation[name]
                values[f"evaluation/mse/{name}"] = metrics.mse[name]
        return values

    return evaluate


def _stratified_indices(times: torch.Tensor, size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    unique = torch.unique(times, sorted=True)
    base, remainder = divmod(size, unique.numel())
    selections = []
    for index, time_value in enumerate(unique):
        candidates = torch.nonzero(times == time_value, as_tuple=False).flatten()
        count = min(candidates.numel(), base + (1 if index < remainder else 0))
        order = torch.randperm(candidates.numel(), generator=generator)[:count]
        selections.append(candidates[order])
    return torch.cat(selections)


def _run_benchmark(config) -> int:
    _validate_data(config)
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    experiment = build_experiment(config)
    summary = run_benchmark(
        experiment.train_step,
        experiment.batches,
        warmup_steps=config.runtime.benchmark_warmup_steps,
        measured_steps=config.runtime.benchmark_steps,
    )
    output = config.runtime.output_dir / "benchmark.json"
    summary.write_json(output)
    print(output.read_text(encoding="utf-8"))
    return 0


def _run_observability(config) -> int:
    if config.data.kind != "rbc_dns":
        raise ValueError("observability mode analysis requires the regular DNS data")
    validate_rbc_dns(config.data.path, config.data.expected_sha256)
    from diploma_pinn.data import RBCDNSDataset

    dataset = RBCDNSDataset(config.data.path)
    indices = _stratified_indices(
        dataset.points[:, 0], min(config.runtime.evaluation_sample_size, dataset.points.shape[0]),
        config.runtime.seed,
    )
    diffusivity = math.sqrt(1.0 / (config.physics.prandtl * config.physics.rayleigh))
    report = analyze_temperature_modes(dataset.points[indices], diffusivity=diffusivity)
    output = config.runtime.output_dir / "observability.json"
    write_mode_report(report, output)
    print(output.read_text(encoding="utf-8"))
    return 0


def _validate_data(config):
    validator = validate_rbc_dns if config.data.kind == "rbc_dns" else validate_ptv_csv
    return validator(config.data.path, config.data.expected_sha256)


def _git_state() -> tuple[str, bool]:
    injected = os.environ.get("DIPLOMA_PINN_GIT_COMMIT")
    if injected:
        return injected, False
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown", False


def _config_fingerprint(config: object) -> str:
    serialized = json.dumps(config_to_dict(config), sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
