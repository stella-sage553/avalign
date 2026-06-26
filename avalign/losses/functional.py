"""Vectorised math helpers shared by the loss functions.

These are deliberately small, dependency-free NumPy routines so the loss
math can be unit-tested without a deep-learning framework installed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["softmax", "log_softmax", "l2_normalize"]


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


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Scale entries along ``axis`` to unit L2 norm."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.sqrt(np.sum(x * x, axis=axis, keepdims=True))
    return x / norm
