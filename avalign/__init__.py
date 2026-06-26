"""avalign: a toolkit for contrastive audio-visual alignment.

The package is split into a dependency-light NumPy core (preprocessing,
losses, metrics, data handling) and an optional PyTorch layer under
:mod:`avalign.torchext` for training neural encoders.
"""

from avalign import audio, data, losses, metrics, video
from avalign.audio import LogMelSpectrogram, log_mel_spectrogram
from avalign.audio.augment import SpecAugment
from avalign.config import AudioConfig, Config, ModelConfig, TrainConfig
from avalign.losses import (
    cosine_similarity_matrix,
    info_nce,
    l2_normalize,
    symmetric_info_nce,
)
from avalign.metrics import alignment_accuracy, recall_at_k, retrieval_report

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "audio",
    "video",
    "losses",
    "metrics",
    "data",
    "log_mel_spectrogram",
    "LogMelSpectrogram",
    "SpecAugment",
    "info_nce",
    "symmetric_info_nce",
    "cosine_similarity_matrix",
    "l2_normalize",
    "recall_at_k",
    "retrieval_report",
    "alignment_accuracy",
    "AudioConfig",
    "ModelConfig",
    "TrainConfig",
    "Config",
]
