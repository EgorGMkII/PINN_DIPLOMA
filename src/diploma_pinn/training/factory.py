"""Resolve configuration once and construct a specialized experiment graph."""

from __future__ import annotations

from dataclasses import dataclass

from torch.optim import Optimizer

from diploma_pinn.config import ExperimentConfig
from diploma_pinn.data import PTVVelocityDataset, RBCDNSDataset, VelocityDatasetView
from diploma_pinn.formulations import FOResiduals, VPParameters, VPResiduals, VVResiduals
from diploma_pinn.losses import FOLossKernel, LossAssembler, LossWeights, VPReferenceLossKernel, VVLossKernel
from diploma_pinn.models import SineMLP
from diploma_pinn.runtime import resolve_device, resolve_dtype
from diploma_pinn.sampling import (
    GlobalShuffleSampler,
    RBCBatchSource,
    RBCBoundarySampler,
    StratifiedTimeSampler,
    UniformDomainSampler,
)
from diploma_pinn.training.optimizers import build_optimizer, build_scheduler
from diploma_pinn.training.step import TrainStep
from diploma_pinn.training.trainer import Trainer
from diploma_pinn.boundaries import RBCBoundaryLoss
from diploma_pinn.integrations import Tracker
from diploma_pinn.instrumentation import StepProfiler


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    model: SineMLP
    dataset: object
    optimizer: Optimizer
    scheduler: object | None
    trainer: Trainer
    train_step: TrainStep
    batches: RBCBatchSource
    holdout: VelocityDatasetView | None


def build_experiment(
    config: ExperimentConfig,
    *,
    tracker: Tracker | None = None,
    profiler: StepProfiler | None = None,
) -> Experiment:
    device = resolve_device(config.runtime.device)
    dtype = resolve_dtype(config.model.dtype)
    dataset_types = {"rbc_dns": RBCDNSDataset, "ptv_csv": PTVVelocityDataset}
    dataset = dataset_types[config.data.kind](config.data.path, dtype=dtype)
    observations = VelocityDatasetView(
        dataset,
        fraction=config.data.observation_fraction,
        per_time=config.data.observations_per_time,
        noise_std=config.data.velocity_noise_std,
        seed=config.runtime.seed,
        holdout_fraction=config.data.holdout_fraction,
    )
    holdout = None
    if config.data.holdout_fraction:
        holdout = VelocityDatasetView(
            dataset,
            fraction=config.data.observation_fraction,
            per_time=config.data.observations_per_time,
            noise_std=0.0,
            seed=config.runtime.seed,
            holdout_fraction=config.data.holdout_fraction,
            partition="holdout",
        )
    sampler_types = {
        "global_shuffle": GlobalShuffleSampler,
        "stratified_time": StratifiedTimeSampler,
    }
    try:
        observation_sampler = sampler_types[config.data.observation_sampling](
            observations, config.data.batch_size, config.runtime.seed
        )
    except KeyError as error:
        raise ValueError(f"unsupported observation sampling: {config.data.observation_sampling}") from error
    domain_sampler = UniformDomainSampler(device=device, dtype=dtype, seed=config.runtime.seed + 1)
    boundary_sampler = RBCBoundarySampler(device=device, dtype=dtype, seed=config.runtime.seed + 2)
    batches = RBCBatchSource(
        observation_sampler,
        domain_sampler,
        boundary_sampler,
        device=device,
        dtype=dtype,
    )
    model = SineMLP(config.model.widths).to(device=device, dtype=dtype)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = (
        build_scheduler(optimizer, config.optimizer)
        if config.runtime.scheduler_epoch_steps > 0
        else None
    )
    parameters = VPParameters(config.physics.rayleigh, config.physics.prandtl)
    kernels = {
        "vp": lambda: VPReferenceLossKernel(
            VPResiduals(parameters), LossAssembler(LossWeights()), RBCBoundaryLoss()
        ),
        "vv": lambda: VVLossKernel(
            VVResiduals(parameters), LossAssembler(LossWeights()), RBCBoundaryLoss()
        ),
        "fo": lambda: FOLossKernel(
            FOResiduals(parameters), LossAssembler(LossWeights()), RBCBoundaryLoss()
        ),
    }
    try:
        kernel = kernels[config.physics.formulation]()
    except KeyError as error:
        raise ValueError(f"unsupported physics formulation: {config.physics.formulation}") from error
    profiler = profiler or StepProfiler(enabled=False)
    train_step = TrainStep(
            model,
            optimizer,
            kernel,
            validate_finite=config.runtime.execution_profile in {"smoke", "pilot"},
            profiler=profiler,
        )
    trainer = Trainer(
        train_step,
        batches,
        max_steps=config.runtime.max_steps,
        tracker=tracker,
        scheduler=scheduler,
        scheduler_epoch_steps=config.runtime.scheduler_epoch_steps,
        profiler=profiler,
        log_interval_steps=config.runtime.log_interval_steps,
    )
    return Experiment(
        config=config,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer=trainer,
        train_step=train_step,
        batches=batches,
        holdout=holdout,
    )
