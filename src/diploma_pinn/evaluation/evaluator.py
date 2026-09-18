"""Chunked full-field evaluation kept outside the hot training step."""

import torch
from torch import nn

from diploma_pinn.data import RBCDNSDataset
from diploma_pinn.evaluation.metrics import FieldMetrics, compute_field_metrics


class Evaluator:
    def __init__(self, dataset: RBCDNSDataset, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._dataset = dataset
        self._batch_size = batch_size

    def evaluate(self, model: nn.Module) -> FieldMetrics:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        was_training = model.training
        predictions: list[torch.Tensor] = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, self._dataset.points.shape[0], self._batch_size):
                points = self._dataset.points[start : start + self._batch_size].to(device=device, dtype=dtype)
                predictions.append(model(points).cpu())
        model.train(was_training)
        return compute_field_metrics(
            torch.cat(predictions), self._dataset.evaluation_fields(), self._dataset.points[:, 0]
        )
