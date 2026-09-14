"""Optimizer and scheduler factories with no implicit backend defaults."""

from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from diploma_pinn.config.schema import OptimizerConfig


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    if config.name.lower() != "adam":
        raise ValueError(f"unsupported optimizer: {config.name}")
    if config.learning_rate <= 0 or config.epsilon <= 0:
        raise ValueError("learning_rate and epsilon must be positive")
    kwargs = {
        "lr": config.learning_rate,
        "betas": (config.beta1, config.beta2),
        "eps": config.epsilon,
        "weight_decay": 0.0,
    }
    if config.fused:
        kwargs["fused"] = True
    return Adam(model.parameters(), **kwargs)


def build_scheduler(optimizer: Optimizer, config: OptimizerConfig) -> object:
    """Match the article's plateau parameters; trainer defines observation cadence."""
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=100,
        threshold=5e-6,
        threshold_mode="abs",
        min_lr=1e-4,
    )
