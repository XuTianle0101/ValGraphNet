"""Sampling utilities for transient graph trajectories."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch.utils.data import Sampler


class PerTrajectoryStepSampler(Sampler[int]):
    """Sample a bounded number of time steps from every trajectory per epoch."""

    def __init__(
        self,
        groups: Sequence[Iterable[int]],
        steps_per_trajectory: int,
        *,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        self.groups = [tuple(int(index) for index in group) for group in groups]
        self.steps_per_trajectory = int(steps_per_trajectory)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.steps_per_trajectory <= 0:
            raise ValueError("steps_per_trajectory must be positive")
        if any(not group for group in self.groups):
            raise ValueError("trajectory groups must not be empty")

    def set_epoch(self, epoch: int) -> None:
        """Select the reproducible sample set for an epoch."""

        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.shuffle:
            group_order = torch.randperm(len(self.groups), generator=generator).tolist()
        else:
            group_order = range(len(self.groups))

        for group_index in group_order:
            group = self.groups[group_index]
            count = min(self.steps_per_trajectory, len(group))
            if self.shuffle:
                positions = torch.randperm(len(group), generator=generator)[:count].tolist()
            elif count == 1:
                positions = [(len(group) - 1) // 2]
            else:
                positions = (
                    torch.linspace(0, len(group) - 1, steps=count).round().long().tolist()
                )
            for position in positions:
                yield group[position]

    def __len__(self) -> int:
        return sum(min(self.steps_per_trajectory, len(group)) for group in self.groups)
