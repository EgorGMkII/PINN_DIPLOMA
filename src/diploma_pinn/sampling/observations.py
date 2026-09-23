"""Deterministic observation sampling policies."""

import torch
from torch import Tensor

from diploma_pinn.contracts import ObservationBatch
from diploma_pinn.data import RBCDNSDataset


class GlobalShuffleSampler:
    """Cycle through fresh seeded global permutations of observation rows."""

    def __init__(self, dataset: RBCDNSDataset, batch_size: int, seed: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if dataset.points.shape[0] == 0:
            raise ValueError("dataset must contain at least one row")
        self._dataset = dataset
        self._batch_size = batch_size
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._order = torch.empty(0, dtype=torch.long)
        self._cursor = 0

    def sample(self) -> ObservationBatch:
        indices = _draw_from_permutation(
            self._dataset.points.shape[0],
            self._batch_size,
            self._generator,
            self._order,
            self._cursor,
        )
        self._order, self._cursor = indices.order, indices.cursor
        return _observation_batch(self._dataset, indices.values)


class StratifiedTimeSampler:
    """Balance every batch across stored time levels without label leakage."""

    def __init__(self, dataset: RBCDNSDataset, batch_size: int, seed: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        groups = tuple(
            torch.nonzero(dataset.points[:, 0] == time, as_tuple=False).flatten()
            for time in dataset.times
        )
        if not groups or any(group.numel() == 0 for group in groups):
            raise ValueError("every time stratum must contain at least one row")
        self._dataset = dataset
        self._batch_size = batch_size
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._groups = groups
        self._orders = tuple(torch.empty(0, dtype=torch.long) for _ in groups)
        self._cursors = [0 for _ in groups]
        self._remainder_start = 0

    def sample(self) -> ObservationBatch:
        count = len(self._groups)
        base, remainder = divmod(self._batch_size, count)
        quotas = [base] * count
        for offset in range(remainder):
            quotas[(self._remainder_start + offset) % count] += 1
        self._remainder_start = (self._remainder_start + remainder) % count

        selections: list[Tensor] = []
        updated_orders: list[Tensor] = []
        for group, order, cursor, quota in zip(
            self._groups, self._orders, self._cursors, quotas
        ):
            draw = _draw_from_permutation(group.numel(), quota, self._generator, order, cursor)
            selections.append(group.index_select(0, draw.values))
            updated_orders.append(draw.order)
            self._cursors[len(updated_orders) - 1] = draw.cursor
        self._orders = tuple(updated_orders)
        indices = torch.cat(selections)
        return _observation_batch(self._dataset, indices)


class _Draw:
    def __init__(self, values: Tensor, order: Tensor, cursor: int) -> None:
        self.values = values
        self.order = order
        self.cursor = cursor


def _draw_from_permutation(
    size: int,
    count: int,
    generator: torch.Generator,
    order: Tensor,
    cursor: int,
) -> _Draw:
    """Draw with replacement only across permutation epochs, never within one."""
    pieces: list[Tensor] = []
    remaining = count
    while remaining:
        if cursor == order.numel():
            order = torch.randperm(size, generator=generator)
            cursor = 0
        available = order.numel() - cursor
        take = min(remaining, available)
        pieces.append(order[cursor : cursor + take])
        cursor += take
        remaining -= take
    return _Draw(torch.cat(pieces), order, cursor)


def _observation_batch(dataset: RBCDNSDataset, indices: Tensor) -> ObservationBatch:
    return ObservationBatch(
        points=dataset.points.index_select(0, indices),
        velocity=dataset.velocity.index_select(0, indices),
    )
