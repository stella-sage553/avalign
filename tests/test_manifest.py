import pytest

from avalign.data.manifest import (
    AVPair,
    read_manifest,
    validate_manifest,
    write_manifest,
)


def test_manifest_roundtrip(tmp_path):
    pairs = [
        AVPair("clip0", "a0.wav", "v0.mp4"),
        AVPair("clip1", "a1.wav", "v1.mp4", label="dog"),
    ]
    path = tmp_path / "manifest.jsonl"
    write_manifest(pairs, path)
    assert read_manifest(path) == pairs


def test_validate_rejects_duplicate_ids():
    pairs = [AVPair("a", "a.wav", "a.mp4"), AVPair("a", "b.wav", "b.mp4")]
    with pytest.raises(ValueError):
        validate_manifest(pairs)


def test_validate_rejects_empty():
    with pytest.raises(ValueError):
        validate_manifest([])
