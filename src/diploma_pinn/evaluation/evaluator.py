"""Chunked full-field evaluation kept outside the hot training step."""

from __future__ import annotations

import torch
from torch import nn

from diploma_pinn.data import RBCDNSDataset
from diploma_pinn.evaluation.diagnostics import FieldDiagnostics, compute_field_diagnostics
from diploma_pinn.evaluation.metrics import FieldMetrics, compute_field_metrics


class Evaluator:
    def __init__(self, dataset: RBCDNSDataset, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._dataset = dataset
        self._batch_size = batch_size

    def predict_full(self, model: nn.Module) -> torch.Tensor:
        return self.predict_points(model, self._dataset.points)

    def predict_points(self, model: nn.Module, source_points: torch.Tensor) -> torch.Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        was_training = model.training
        predictions: list[torch.Tensor] = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, source_points.shape[0], self._batch_size):
                points = source_points[start : start + self._batch_size].to(device=device, dtype=dtype)
                values = model(points)
                predictions.append(values[:, :5].cpu() if values.shape[1] == 17 else values.cpu())
        model.train(was_training)
        return torch.cat(predictions)

    def evaluate(self, model: nn.Module) -> FieldMetrics:
        return compute_field_metrics(
            self.predict_full(model), self._dataset.evaluation_fields(), self._dataset.points[:, 0]
        )

    def diagnose(self, predicted: torch.Tensor, *, diffusivity: float | None = None) -> FieldDiagnostics:
        return compute_field_diagnostics(
            predicted, self._dataset.evaluation_fields(), self._dataset.points,
            diffusivity=diffusivity,
        )
