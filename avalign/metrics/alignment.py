"""Alignment diagnostics for audio-visual similarity matrices."""

from __future__ import annotations

import numpy as np

__all__ = ["alignment_accuracy", "diagonal_dominance"]


def alignment_accuracy(sim: np.ndarray) -> float:
    """Fraction of queries whose top-1 match is the correct (diagonal) item."""
    sim = np.asarray(sim, dtype=np.float64)
    preds = np.argmax(sim, axis=1)
    return float(np.mean(preds == np.arange(sim.shape[0])))


def diagonal_dominance(sim: np.ndarray) -> float:
    """Mean matched-pair similarity minus mean in-batch-negative similarity.

    A quick scalar for how well matched pairs separate from negatives;
    larger is better and values near zero signal a collapsed embedding.
    """
    sim = np.asarray(sim, dtype=np.float64)
    n = sim.shape[0]
    diag = np.diag(sim)
    off = sim[~np.eye(n, dtype=bool)]
    return float(diag.mean() - off.mean())
