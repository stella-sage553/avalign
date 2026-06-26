"""Audio preprocessing: log-mel spectrograms and SpecAugment."""

from avalign.audio.melscale import hz_to_mel, mel_filterbank, mel_to_hz
from avalign.audio.melspec import (
    LogMelSpectrogram,
    frame_signal,
    log_mel_spectrogram,
    stft_power,
)

__all__ = [
    "hz_to_mel",
    "mel_to_hz",
    "mel_filterbank",
    "frame_signal",
    "stft_power",
    "log_mel_spectrogram",
    "LogMelSpectrogram",
]
