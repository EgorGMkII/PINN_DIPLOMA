"""Chunked full-field evaluation kept outside the hot training step."""

from torch import nn

from diploma_pinn.data import RBCDNSDataset
from diploma_pinn.evaluation.metrics import FieldMetrics


class Evaluator:
    def __init__(self, dataset: RBCDNSDataset, batch_size: int) -> None:
        raise NotImplementedError("implement no-grad chunked prediction")

    def evaluate(self, model: nn.Module) -> FieldMetrics:
        raise NotImplementedError
