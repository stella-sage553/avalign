import pytest

from avalign.config import AudioConfig, Config, ModelConfig


def test_config_json_roundtrip(tmp_path):
    cfg = Config()
    cfg.model.embed_dim = 64
    cfg.audio.n_mels = 80
    path = tmp_path / "config.json"
    cfg.save(path)
    loaded = Config.load(path)
    assert loaded == cfg
    assert loaded.model.embed_dim == 64
    assert loaded.audio.n_mels == 80


def test_invalid_temperature_rejected():
    with pytest.raises(ValueError):
        ModelConfig(temperature=0.0)


def test_invalid_audio_rejected():
    with pytest.raises(ValueError):
        AudioConfig(n_mels=0)
