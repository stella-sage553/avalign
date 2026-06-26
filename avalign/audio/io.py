"""Minimal WAV loading with no third-party audio dependencies.

Just enough to read PCM WAV files in tests and examples; for real
pipelines you'll usually decode with ffmpeg/torchaudio upstream.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

__all__ = ["load_wav"]


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a PCM WAV file as a mono float64 waveform in ``[-1, 1]``.

    Returns ``(waveform, sample_rate)``. Multi-channel audio is downmixed
    to mono by averaging channels.
    """
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sample_width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sample_width} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sample_rate
