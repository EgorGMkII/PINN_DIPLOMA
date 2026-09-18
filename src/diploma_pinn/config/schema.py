"""Typed, serializable configuration contract for EXP-001."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    path: Path
    expected_sha256: str = ""
    batch_size: int = 4096
    observation_sampling: str = "global_shuffle"


@dataclass(frozen=True)
class ModelConfig:
    widths: tuple[int, ...] = (4, *(256,) * 10, 5)
    dtype: str = "float32"


@dataclass(frozen=True)
class PhysicsConfig:
    formulation: str = "vp"
    rayleigh: float = 1e6
    prandtl: float = 0.7


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-7
    fused: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 2204
    deterministic: bool = False
    device: str = "auto"
    execution_profile: str = "parity"
    max_steps: int = 1
    output_dir: Path = Path("outputs")
    evaluate_after_run: bool = False
    evaluation_batch_size: int = 65536


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    mode: str = "disabled"
    project: str = "diploma-pinn"
    group: str = "EXP-001"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
