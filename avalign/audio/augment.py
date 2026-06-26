"""SpecAugment-style masking for log-mel spectrograms.

Operates on arrays shaped ``(n_mels, n_frames)`` as produced by
:func:`avalign.audio.log_mel_spectrogram`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["time_mask", "freq_mask"]


def time_mask(spec: np.ndarray, max_width: int, n_masks: int = 1) -> np.ndarray:
    """Zero out up to ``n_masks`` random time bands of width <= ``max_width``."""
    spec = np.array(spec, dtype=np.float64)
    n_frames = spec.shape[1]
    for _ in range(n_masks):
        width = np.random.randint(0, max_width + 1)
        if width == 0:
            continue
        start = np.random.randint(0, max(1, n_frames - width))
        spec[:, start : start + width] = 0.0
    return spec


def freq_mask(spec: np.ndarray, max_width: int, n_masks: int = 1) -> np.ndarray:
    """Zero out up to ``n_masks`` random mel bands of width <= ``max_width``."""
    spec = np.array(spec, dtype=np.float64)
    n_mels = spec.shape[0]
    for _ in range(n_masks):
        width = np.random.randint(0, max_width + 1)
        if width == 0:
            continue
        start = np.random.randint(0, max(1, n_mels - width))
        spec[start : start + width, :] = 0.0
    return spec
