"""Resolve configuration once and construct a specialized experiment graph."""

from dataclasses import dataclass

from diploma_pinn.config import ExperimentConfig
from diploma_pinn.data import RBCDNSDataset
from diploma_pinn.formulations import VPParameters, VPResiduals
from diploma_pinn.losses import LossAssembler, LossWeights, VPReferenceLossKernel
from diploma_pinn.models import SineMLP
from diploma_pinn.runtime import resolve_device, resolve_dtype
from diploma_pinn.sampling import (
    GlobalShuffleSampler,
    RBCBatchSource,
    RBCBoundarySampler,
    StratifiedTimeSampler,
    UniformDomainSampler,
)
from diploma_pinn.training.optimizers import build_optimizer
from diploma_pinn.training.step import TrainStep
from diploma_pinn.training.trainer import Trainer
from diploma_pinn.boundaries import RBCBoundaryLoss


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    model: SineMLP
    trainer: Trainer


def build_experiment(config: ExperimentConfig) -> Experiment:
    device = resolve_device(config.runtime.device)
    dtype = resolve_dtype(config.model.dtype)
    dataset = RBCDNSDataset(config.data.path, dtype=dtype)
    sampler_types = {
        "global_shuffle": GlobalShuffleSampler,
        "stratified_time": StratifiedTimeSampler,
    }
    try:
        observation_sampler = sampler_types[config.data.observation_sampling](
            dataset, config.data.batch_size, config.runtime.seed
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
    formulation = VPResiduals(VPParameters(config.physics.rayleigh, config.physics.prandtl))
    kernel = VPReferenceLossKernel(formulation, LossAssembler(LossWeights()), RBCBoundaryLoss())
    trainer = Trainer(
        TrainStep(
            model,
            optimizer,
            kernel,
            validate_finite=config.runtime.execution_profile == "smoke",
        ),
        batches,
        max_steps=config.runtime.max_steps,
    )
    return Experiment(config=config, model=model, trainer=trainer)
