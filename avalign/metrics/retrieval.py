"""Cross-modal retrieval metrics (recall@k, rank statistics).

A retrieval problem here is a square similarity matrix ``sim`` of shape
``(N, N)`` whose diagonal holds the correct query/candidate pairs. Row
``i`` ranks all candidates for query ``i``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

__all__ = ["recall_at_k", "median_rank", "mean_rank", "mrr"]


def _rank_of(sim_row: np.ndarray, i: int) -> int:
    """1-indexed rank of candidate ``i`` within a single query row."""
    order = np.argsort(-sim_row)
    return int(np.where(order == i)[0][0]) + 1


def recall_at_k(sim: np.ndarray, ks: Iterable[int] = (1, 5, 10)) -> dict[int, float]:
    """Recall@k for matched-pair retrieval, returned as ``{k: recall}``."""
    sim = np.asarray(sim, dtype=np.float64)
    n = sim.shape[0]
    ks = list(ks)
    hits = dict.fromkeys(ks, 0)
    for i in range(n):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        for k in ks:
            if rank <= k:
                hits[k] += 1
    return {k: hits[k] / n for k in ks}


def median_rank(sim: np.ndarray) -> float:
    """Median 1-indexed rank of the correct match across queries."""
    sim = np.asarray(sim, dtype=np.float64)
    ranks = [_rank_of(sim[i], i) for i in range(sim.shape[0])]
    return float(np.median(ranks))


def mean_rank(sim: np.ndarray) -> float:
    """Mean 1-indexed rank of the correct match across queries."""
    sim = np.asarray(sim, dtype=np.float64)
    ranks = [_rank_of(sim[i], i) for i in range(sim.shape[0])]
    return float(np.mean(ranks))


def mrr(sim: np.ndarray) -> float:
    """Mean reciprocal rank of the correct match across queries."""
    sim = np.asarray(sim, dtype=np.float64)
    ranks = [_rank_of(sim[i], i) for i in range(sim.shape[0])]
    return float(np.mean([1.0 / r for r in ranks]))
