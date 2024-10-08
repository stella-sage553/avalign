"""Mel-scale conversions and triangular filterbank construction.

Uses the HTK mel formula. Filterbank rows are triangular windows whose
centres are equally spaced on the mel scale between ``f_min`` and ``f_max``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["hz_to_mel", "mel_to_hz", "mel_filterbank"]


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray:
    """Convert a frequency in Hz to the mel scale (HTK convention)."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    """Convert a value on the mel scale back to Hz (HTK convention)."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Build an ``(n_mels, n_fft // 2 + 1)`` triangular mel filterbank."""
    if f_max is None:
        f_max = sample_rate / 2.0

    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)
        fb[m - 1] = np.clip(np.minimum(rising, falling), 0.0, None)

    # Slaney-style normalisation: scale each filter by its bandwidth so the
    # rows are roughly constant-energy rather than constant-peak.
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    fb *= enorm[:, np.newaxis]
    return fb
