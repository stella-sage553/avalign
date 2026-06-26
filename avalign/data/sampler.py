"""Index batching and negative sampling for contrastive training."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

__all__ = ["BatchSampler", "sample_negatives"]


class BatchSampler:
    """Yield arrays of item indices, one per training batch.

    With ``drop_last=True`` (the default for contrastive training, where a
    constant batch size keeps the number of in-batch negatives fixed) any
    trailing partial batch is discarded.
    """

    def __init__(
        self,
        n_items: int,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.n_items = n_items
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n_items // self.batch_size
        return math.ceil(self.n_items / self.batch_size)

    def __iter__(self) -> Iterator[np.ndarray]:
        order = np.arange(self.n_items)
        if self.shuffle:
            self._rng.shuffle(order)
        for start in range(0, self.n_items, self.batch_size):
            batch = order[start : start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield batch


def sample_negatives(
    n_items: int,
    anchor: int,
    n_negatives: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample ``n_negatives`` indices in ``[0, n_items)`` excluding ``anchor``."""
    if n_items <= 1:
        raise ValueError("need at least 2 items to sample a negative")
    gen = rng if rng is not None else np.random.default_rng()
    candidates = np.delete(np.arange(n_items), anchor)
    replace = n_negatives > len(candidates)
    return gen.choice(candidates, size=n_negatives, replace=replace)
