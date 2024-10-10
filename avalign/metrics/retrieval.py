"""Cross-modal retrieval metrics (recall@k, rank statistics).

A retrieval problem here is a square similarity matrix ``sim`` of shape
``(N, N)`` whose diagonal holds the correct query/candidate pairs. Row
``i`` ranks all candidates for query ``i``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from avalign.losses.functional import cosine_similarity_matrix

__all__ = [
    "recall_at_k",
    "median_rank",
    "mean_rank",
    "mrr",
    "retrieval_report",
]


def _ranks(sim: np.ndarray) -> np.ndarray:
    """1-indexed rank of each query's correct (diagonal) match."""
    sim = np.asarray(sim, dtype=np.float64)
    n = sim.shape[0]
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        order = np.argsort(-sim[i])
        ranks[i] = int(np.where(order == i)[0][0]) + 1
    return ranks


def recall_at_k(sim: np.ndarray, ks: Iterable[int] = (1, 5, 10)) -> dict[int, float]:
    """Recall@k for matched-pair retrieval, returned as ``{k: recall}``."""
    ranks = _ranks(sim)
    n = len(ranks)
    return {k: float(np.mean(ranks <= k)) for k in ks} if n else {k: 0.0 for k in ks}


def median_rank(sim: np.ndarray) -> float:
    """Median 1-indexed rank of the correct match across queries."""
    return float(np.median(_ranks(sim)))


def mean_rank(sim: np.ndarray) -> float:
    """Mean 1-indexed rank of the correct match across queries."""
    return float(np.mean(_ranks(sim)))


def mrr(sim: np.ndarray) -> float:
    """Mean reciprocal rank of the correct match across queries."""
    return float(np.mean(1.0 / _ranks(sim)))


def _direction_report(sim: np.ndarray, ks: Iterable[int]) -> dict[str, object]:
    return {
        "recall": recall_at_k(sim, ks),
        "median_rank": median_rank(sim),
        "mrr": mrr(sim),
    }


def retrieval_report(
    audio: np.ndarray, video: np.ndarray, ks: Iterable[int] = (1, 5, 10)
) -> dict[str, dict[str, object]]:
    """Bidirectional retrieval report from paired embeddings.

    Returns metrics for both ``audio_to_video`` (audio queries video) and
    ``video_to_audio`` directions.
    """
    ks = list(ks)
    sim = cosine_similarity_matrix(audio, video)
    return {
        "audio_to_video": _direction_report(sim, ks),
        "video_to_audio": _direction_report(sim.T, ks),
    }
