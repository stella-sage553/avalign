"""SpecAugment-style masking for log-mel spectrograms.

Operates on arrays shaped ``(n_mels, n_frames)`` as produced by
:func:`avalign.audio.log_mel_spectrogram`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["time_mask", "freq_mask", "SpecAugment"]


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def time_mask(
    spec: np.ndarray,
    max_width: int,
    n_masks: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Zero out up to ``n_masks`` random time bands of width <= ``max_width``."""
    spec = np.array(spec, dtype=np.float64)
    gen = _rng(rng)
    n_frames = spec.shape[1]
    for _ in range(n_masks):
        width = int(gen.integers(0, max_width + 1))
        if width == 0:
            continue
        start = int(gen.integers(0, max(1, n_frames - width)))
        spec[:, start : start + width] = 0.0
    return spec


def freq_mask(
    spec: np.ndarray,
    max_width: int,
    n_masks: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Zero out up to ``n_masks`` random mel bands of width <= ``max_width``."""
    spec = np.array(spec, dtype=np.float64)
    gen = _rng(rng)
    n_mels = spec.shape[0]
    for _ in range(n_masks):
        width = int(gen.integers(0, max_width + 1))
        if width == 0:
            continue
        start = int(gen.integers(0, max(1, n_mels - width)))
        spec[start : start + width, :] = 0.0
    return spec


@dataclass
class SpecAugment:
    """Apply frequency then time masking with fixed widths.

    A seed makes the policy reproducible, which matters for ablations where
    the same augmentation stream should be replayed across runs.
    """

    freq_width: int = 8
    time_width: int = 16
    n_freq_masks: int = 1
    n_time_masks: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        spec = freq_mask(spec, self.freq_width, self.n_freq_masks, self._rng)
        spec = time_mask(spec, self.time_width, self.n_time_masks, self._rng)
        return spec
