import numpy as np

from avalign.audio import frame_signal, log_mel_spectrogram
from avalign.audio.melscale import hz_to_mel, mel_filterbank, mel_to_hz


def test_mel_scale_roundtrip():
    hz = np.array([0.0, 100.0, 1000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(hz)), hz, atol=1e-6)


def test_filterbank_shape_and_nonnegative():
    fb = mel_filterbank(40, 400, 16000)
    assert fb.shape == (40, 201)
    assert (fb >= 0).all()


def test_log_mel_shape_and_finite():
    sr = 16000
    wav = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
    mel = log_mel_spectrogram(wav, sample_rate=sr, n_mels=64, n_fft=400, hop_length=160)
    assert mel.shape[0] == 64
    assert np.isfinite(mel).all()


def test_frame_signal_pads_short_input():
    frames = frame_signal(np.ones(10), frame_length=400, hop_length=160)
    assert frames.shape == (1, 400)
