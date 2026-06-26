"""Log-mel spectrogram extraction in pure NumPy.

The pipeline mirrors the usual torchaudio/librosa stack: frame the
waveform, apply a window, take the power spectrum via the real FFT,
project onto a mel filterbank, then convert to decibels.
"""

from __future__ import annotations

import numpy as np

__all__ = ["frame_signal", "stft_power"]


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


def stft_power(
    signal: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """Short-time power spectrum of a waveform.

    Returns an array of shape ``(n_frames, n_fft // 2 + 1)``. A Hann window
    is applied by default.
    """
    frames = frame_signal(signal, n_fft, hop_length)
    if window is None:
        window = np.hanning(n_fft)
    spectrum = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)
    return np.abs(spectrum) ** 2
