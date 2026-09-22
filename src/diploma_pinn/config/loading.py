"""Strict YAML loading and resolved-configuration serialization."""

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from diploma_pinn.config.schema import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    PhysicsConfig,
    RuntimeConfig,
    TrackingConfig,
)


T = TypeVar("T")


def _strict_dataclass(cls: type[T], values: Mapping[str, Any]) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = set(values).difference(allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return content


def load_config(path: Path, *, local_overlay: Path | None = None) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = _read_yaml(path)
    if local_overlay is not None:
        raw = _deep_merge(raw, _read_yaml(Path(local_overlay).resolve()))

    allowed = {"experiment_id", "data", "model", "physics", "optimizer", "runtime", "tracking"}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"unknown ExperimentConfig keys: {sorted(unknown)}")
    if "experiment_id" not in raw or "data" not in raw:
        raise ValueError("experiment_id and data sections are required")

    data_values = dict(raw["data"])
    data_path = Path(data_values["path"])
    if not data_path.is_absolute():
        data_path = (path.parent / data_path).resolve()
    data_values["path"] = data_path

    runtime_values = dict(raw.get("runtime", {}))
    output_dir = Path(runtime_values.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = (path.parent / output_dir).resolve()
    runtime_values["output_dir"] = output_dir

    model_values = dict(raw.get("model", {}))
    if "widths" in model_values:
        model_values["widths"] = tuple(int(value) for value in model_values["widths"])

    config = ExperimentConfig(
        experiment_id=str(raw["experiment_id"]),
        data=_strict_dataclass(DataConfig, data_values),
        model=_strict_dataclass(ModelConfig, model_values),
        physics=_strict_dataclass(PhysicsConfig, raw.get("physics", {})),
        optimizer=_strict_dataclass(OptimizerConfig, raw.get("optimizer", {})),
        runtime=_strict_dataclass(RuntimeConfig, runtime_values),
        tracking=_strict_dataclass(TrackingConfig, raw.get("tracking", {})),
    )
    expected_outputs = {"vp": 5, "vv": 4}
    try:
        output_width = expected_outputs[config.physics.formulation]
    except KeyError as error:
        raise ValueError(f"unsupported physics formulation: {config.physics.formulation}") from error
    if config.model.widths[0] != 4 or config.model.widths[-1] != output_width:
        raise ValueError(
            f"{config.physics.formulation} model widths must start with 4 and end with {output_width}"
        )
    if config.runtime.execution_profile == "smoke" and config.runtime.max_steps > 10:
        raise ValueError("smoke profile is limited to 10 optimizer steps")
    return config


def write_resolved_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=True)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    values = asdict(config)

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return convert(values)
