"""Log-mel spectrogram extraction in pure NumPy.

The pipeline mirrors the usual torchaudio/librosa stack: frame the
waveform, apply a window, take the power spectrum via the real FFT,
project onto a mel filterbank, then convert to decibels.
"""

from __future__ import annotations

import numpy as np

__all__ = ["frame_signal"]


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Split a 1-D waveform into overlapping frames.

    Returns an array of shape ``(n_frames, frame_length)``. Signals shorter
    than one frame are zero-padded up to a single frame.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("expected a 1-D waveform")
    if len(signal) < frame_length:
        signal = np.pad(signal, (0, frame_length - len(signal)))

    n_frames = 1 + (len(signal) - frame_length) // hop_length
    indices = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return signal[indices]
