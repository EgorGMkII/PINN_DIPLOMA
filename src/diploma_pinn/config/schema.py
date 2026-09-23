"""Typed, serializable configuration contract for EXP-001."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    path: Path
    kind: str = "rbc_dns"
    expected_sha256: str = ""
    batch_size: int = 4096
    observation_sampling: str = "global_shuffle"
    observation_fraction: float = 1.0
    observations_per_time: int = 0
    velocity_noise_std: float = 0.0
    holdout_fraction: float = 0.0


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
    scheduler_epoch_steps: int = 0
    evaluate_after_run: bool = False
    evaluation_batch_size: int = 65536
    save_final_checkpoint: bool = True
    write_diagnostics: bool = True
    evaluation_interval_epochs: int = 0
    evaluation_sample_size: int = 65536
    profile_steps: bool = False
    benchmark_warmup_steps: int = 20
    benchmark_steps: int = 50
    pde_evaluation_size: int = 0
    log_interval_steps: int = 1


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    mode: str = "disabled"
    project: str = "diploma-pinn"
    group: str = "EXP-001"
    run_name: str = ""


@dataclass(frozen=True)
class PressureRecoveryConfig:
    enabled: bool = False
    widths: tuple[int, ...] = (4, *(256,) * 10, 1)
    epochs: int = 100
    batch_size: int = 4096
    learning_rate: float = 1e-3
    gauge_weight: float = 1.0
    log_interval_steps: int = 10


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pressure_recovery: PressureRecoveryConfig = field(default_factory=PressureRecoveryConfig)
