"""Dataset balancing used by InstructBLIP-style instruction tuning."""

import math
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class SquareRootMixtureSampler(Sampler[int]):
    """Sample dataset d with probability proportional to sqrt(dataset_size[d])."""

    def __init__(
        self,
        dataset_lengths: Sequence[int],
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        if not dataset_lengths or any(length <= 0 for length in dataset_lengths):
            raise ValueError("All mixture datasets must be non-empty")
        self.lengths = torch.tensor(dataset_lengths, dtype=torch.long)
        self.offsets = torch.tensor(
            [0, *torch.cumsum(self.lengths, dim=0).tolist()[:-1]], dtype=torch.long
        )
        weights = torch.tensor([math.sqrt(length) for length in dataset_lengths])
        self.probabilities = weights / weights.sum()
        self.num_samples = num_samples or int(self.lengths.sum().item())
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        choices = torch.multinomial(
            self.probabilities, self.num_samples, replacement=True, generator=generator
        )
        random_values = torch.rand(self.num_samples, generator=generator)
        local_indices = (random_values * self.lengths[choices]).long()
        indices = self.offsets[choices] + local_indices
        return iter(indices.tolist())
