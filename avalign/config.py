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
    f_min: float = 0.0
    f_max: float | None = None

    def __post_init__(self) -> None:
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if self.f_max is not None and self.f_max <= self.f_min:
            raise ValueError("f_max must be greater than f_min")
