"""Ordered EXP-001 TensorFlow-to-PyTorch parity workflow."""

from pathlib import Path

from diploma_pinn.config import ExperimentConfig
from diploma_pinn.parity.report import ParityReport


def run_parity(config: ExperimentConfig, fixture: Path) -> ParityReport:
    raise NotImplementedError("compare outputs through one Adam update in specified order")
