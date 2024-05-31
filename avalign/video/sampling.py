"""Frame-index sampling strategies for video clips.

These return integer index arrays into a clip of ``n_frames`` frames, so
they stay decoder-agnostic: the caller gathers the actual frames.
"""

from __future__ import annotations

import numpy as np

__all__ = ["uniform_indices"]


def uniform_indices(n_frames: int, n_samples: int) -> np.ndarray:
    """Evenly spaced frame indices covering ``[0, n_frames)``."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    return np.linspace(0, n_frames - 1, n_samples).round().astype(int)
