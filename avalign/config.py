"""Configuration dataclasses for avalign pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Parameters for log-mel spectrogram extraction."""

    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 64
