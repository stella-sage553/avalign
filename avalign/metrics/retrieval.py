"""Cross-modal retrieval metrics (recall@k, rank statistics).

A retrieval problem here is a square similarity matrix ``sim`` of shape
``(N, N)`` whose diagonal holds the correct query/candidate pairs. Row
``i`` ranks all candidates for query ``i``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

__all__ = ["recall_at_k"]


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
