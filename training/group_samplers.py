"""Deterministic group-balanced sampling for Protocol v3."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Iterable, Iterator

import torch
from torch.utils.data import Sampler


class DeterministicGroupBalancedSampler(Sampler[int]):
    """Draw an equal replacement quota from every (generator, label) group."""

    def __init__(
        self,
        generators: Iterable[str],
        labels: Iterable[int],
        samples_per_group: int,
        seed: int,
    ):
        self.samples_per_group = int(samples_per_group)
        self.seed = int(seed)
        self.epoch = 0
        if self.samples_per_group <= 0:
            raise ValueError("samples_per_group must be positive")

        groups = defaultdict(list)
        generator_list = list(generators)
        label_list = [int(label) for label in labels]
        if len(generator_list) != len(label_list) or not generator_list:
            raise ValueError("generators and labels must have the same non-zero length")
        for index, (generator, label) in enumerate(zip(generator_list, label_list)):
            if label not in (0, 1):
                raise ValueError("labels must be 0 or 1")
            groups[(str(generator), label)].append(index)
        self.groups = {key: tuple(value) for key, value in sorted(groups.items())}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.groups) * self.samples_per_group

    def __iter__(self) -> Iterator[int]:
        payload = f"{self.seed}|{self.epoch}|group-balanced".encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        rng = torch.Generator(device="cpu").manual_seed(derived_seed)
        draws = []
        for indices in self.groups.values():
            selected = torch.randint(
                len(indices), (self.samples_per_group,), generator=rng
            ).tolist()
            draws.extend(indices[position] for position in selected)
        order = torch.randperm(len(draws), generator=rng).tolist()
        return iter(draws[position] for position in order)


def validate_protocol_groups(
    generators: Iterable[str], labels: Iterable[int], expected_groups: int = 4
) -> None:
    groups = set(zip(map(str, generators), map(int, labels)))
    if len(groups) != expected_groups:
        raise ValueError(f"Expected {expected_groups} generator-label groups, got {len(groups)}")
