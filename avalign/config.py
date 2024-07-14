"""Configuration dataclasses for avalign pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class ModelConfig:
    """Encoder and projection dimensions for the contrastive model."""

    audio_hidden: int = 256
    video_hidden: int = 256
    embed_dim: int = 128
    temperature: float = 0.07

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")


@dataclass
class TrainConfig:
    """Optimisation and schedule settings."""

    batch_size: int = 64
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class Config:
    """Top-level configuration bundling the three sub-configs."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
