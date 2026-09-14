"""Strict YAML loading and resolved-configuration serialization."""

from pathlib import Path

from diploma_pinn.config.schema import ExperimentConfig


def load_config(path: Path, *, local_overlay: Path | None = None) -> ExperimentConfig:
    raise NotImplementedError("reject unknown keys and validate profile combinations")


def write_resolved_config(config: ExperimentConfig, path: Path) -> None:
    raise NotImplementedError("serialize deterministically without secrets")
