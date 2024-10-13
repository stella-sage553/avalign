"""Log-mel spectrogram extraction in pure NumPy.

The pipeline mirrors the usual torchaudio/librosa stack: frame the
waveform, apply a window, take the power spectrum via the real FFT,
project onto a mel filterbank, then convert to decibels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avalign.audio.melscale import mel_filterbank

__all__ = ["frame_signal", "stft_power", "log_mel_spectrogram", "LogMelSpectrogram"]


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


def log_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 64,
    f_min: float = 0.0,
    f_max: float | None = None,
    amin: float = 1e-10,
) -> np.ndarray:
    """Compute a log-mel spectrogram of shape ``(n_mels, n_frames)``."""
    power = stft_power(signal, n_fft, hop_length)  # (T, F)
    fb = mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max)  # (M, F)
    mel = power @ fb.T  # (T, M)
    log_mel = 10.0 * np.log10(np.maximum(mel, amin))  # floor silence to avoid -inf
    return log_mel.T  # (M, T)


@dataclass
class LogMelSpectrogram:
    """Callable wrapper that pins log-mel parameters for reuse in a pipeline."""

    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float | None = None

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return log_mel_spectrogram(
            signal,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )
