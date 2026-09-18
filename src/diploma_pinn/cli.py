"""Command-line entry points for validation, parity, training, and benchmarking."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from diploma_pinn.config import config_to_dict, load_config, write_resolved_config
from diploma_pinn.data import validate_rbc_dns
from diploma_pinn.evaluation import Evaluator
from diploma_pinn.instrumentation.manifest import RunManifest, collect_environment
from diploma_pinn.runtime import seed_everything
from diploma_pinn.training.factory import build_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diploma-pinn")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-data", "parity", "train", "benchmark", "evaluate"):
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
        report = validate_rbc_dns(config.data.path, config.data.expected_sha256)
        print(json.dumps({**report.__dict__, "path": str(report.path)}, indent=2, default=list))
        return 0
    if arguments.command == "train":
        return _run_training(config, require_smoke=arguments.smoke)
    raise NotImplementedError(f"command is not implemented yet: {arguments.command}")


def _run_training(config, *, require_smoke: bool) -> int:
    if require_smoke and config.runtime.execution_profile != "smoke":
        raise ValueError("--smoke requires runtime.execution_profile: smoke")
    report = validate_rbc_dns(config.data.path, config.data.expected_sha256)
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = config.runtime.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    write_resolved_config(config, output_dir / "resolved_config.yaml")

    experiment = build_experiment(config)
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
        for old, new in zip(before, experiment.model.parameters(), strict=True)
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
    if config.runtime.evaluate_after_run:
        evaluation = Evaluator(experiment.dataset, config.runtime.evaluation_batch_size).evaluate(experiment.model)
        summary["evaluation"] = {
            "mse": evaluation.mse,
            "relative_l2": evaluation.relative_l2,
            "correlation": evaluation.correlation,
        }
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
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
