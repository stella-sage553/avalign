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
