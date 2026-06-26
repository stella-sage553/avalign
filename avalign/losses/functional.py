"""Vectorised math helpers shared by the loss functions.

These are deliberately small, dependency-free NumPy routines so the loss
math can be unit-tested without a deep-learning framework installed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["softmax", "log_softmax", "l2_normalize", "cosine_similarity_matrix"]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax along ``axis``."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Scale entries along ``axis`` to unit L2 norm.

    ``eps`` floors the denominator so all-zero rows map to zeros instead of
    producing NaNs.
    """
    x = np.asarray(x, dtype=np.float64)
    norm = np.sqrt(np.sum(x * x, axis=axis, keepdims=True))
    return x / np.maximum(norm, eps)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine-similarity matrix between two row sets.

    Args:
        a: Array of shape ``(n, d)``.
        b: Array of shape ``(m, d)``.

    Returns:
        Array of shape ``(n, m)`` where entry ``(i, j)`` is the cosine
        similarity between ``a[i]`` and ``b[j]``.
    """
    a = l2_normalize(np.asarray(a, dtype=np.float64), axis=-1)
    b = l2_normalize(np.asarray(b, dtype=np.float64), axis=-1)
    return a @ b.T
