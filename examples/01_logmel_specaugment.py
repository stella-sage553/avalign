"""Turn a synthetic tone into a log-mel spectrogram and apply SpecAugment.

Run with:  python examples/01_logmel_specaugment.py
"""

import numpy as np

from avalign.audio import log_mel_spectrogram
from avalign.audio.augment import SpecAugment


def main() -> None:
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    # a 440 Hz tone plus a little noise
    wav = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav += 0.01 * np.random.default_rng(0).standard_normal(wav.shape)

    mel = log_mel_spectrogram(wav, sample_rate=sample_rate, n_mels=64)
    print(f"log-mel shape: {mel.shape}  (n_mels, n_frames)")
    print(f"value range:   [{mel.min():.1f}, {mel.max():.1f}] dB")

    augmented = SpecAugment(freq_width=8, time_width=16, seed=0)(mel)
    masked = int((augmented == 0).sum())
    print(f"masked bins after SpecAugment: {masked}")


if __name__ == "__main__":
    main()
