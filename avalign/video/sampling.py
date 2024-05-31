"""Frame-index sampling strategies for video clips.

These return integer index arrays into a clip of ``n_frames`` frames, so
they stay decoder-agnostic: the caller gathers the actual frames.
"""

from __future__ import annotations

import numpy as np

__all__ = ["uniform_indices", "center_indices", "random_indices", "sample_indices"]


def uniform_indices(n_frames: int, n_samples: int) -> np.ndarray:
    """Evenly spaced frame indices covering ``[0, n_frames)``."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    return np.linspace(0, n_frames - 1, n_samples).round().astype(int)


def center_indices(n_frames: int, n_samples: int) -> np.ndarray:
    """A contiguous window of ``n_samples`` frames centred in the clip."""
    if n_samples <= 0 or n_frames <= 0:
        raise ValueError("n_frames and n_samples must be positive")
    start = max(0, (n_frames - n_samples) // 2)
    idx = np.arange(start, start + n_samples)
    return np.clip(idx, 0, n_frames - 1)


def random_indices(
    n_frames: int, n_samples: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """A sorted random subset of frame indices (with replacement if needed)."""
    if n_samples <= 0 or n_frames <= 0:
        raise ValueError("n_frames and n_samples must be positive")
    gen = rng if rng is not None else np.random.default_rng()
    replace = n_samples > n_frames
    idx = gen.choice(n_frames, size=n_samples, replace=replace)
    return np.sort(idx)


def sample_indices(
    n_frames: int,
    n_samples: int,
    strategy: str = "uniform",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Dispatch to a named sampling strategy."""
    if strategy == "uniform":
        return uniform_indices(n_frames, n_samples)
    if strategy == "center":
        return center_indices(n_frames, n_samples)
    if strategy == "random":
        return random_indices(n_frames, n_samples, rng)
    raise ValueError(f"unknown strategy: {strategy!r}")
