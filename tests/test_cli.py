import numpy as np

from avalign.cli.main import main


def test_help_returns_nonzero_and_prints(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 1
    assert "avalign" in out


def test_prep_then_eval(tmp_path, capsys):
    audio_dir = tmp_path / "audio"
    video_dir = tmp_path / "video"
    audio_dir.mkdir()
    video_dir.mkdir()
    for name in ["c1", "c2", "c3"]:
        (audio_dir / f"{name}.wav").write_bytes(b"x")
        (video_dir / f"{name}.mp4").write_bytes(b"x")

    manifest = tmp_path / "m.jsonl"
    rc = main(
        [
            "prep",
            "--audio-dir",
            str(audio_dir),
            "--video-dir",
            str(video_dir),
            "--out",
            str(manifest),
        ]
    )
    assert rc == 0
    assert manifest.exists()

    audio_emb = tmp_path / "a.npy"
    video_emb = tmp_path / "v.npy"
    np.save(audio_emb, np.eye(4))
    np.save(video_emb, np.eye(4))
    rc = main(
        ["eval", "--audio-emb", str(audio_emb), "--video-emb", str(video_emb), "--ks", "1,2"]
    )
    assert rc == 0
    assert "audio_to_video" in capsys.readouterr().out


def test_eval_missing_file_returns_friendly_error(capsys):
    rc = main(["eval", "--audio-emb", "/no/such/a.npy", "--video-emb", "/no/such/v.npy"])
    assert rc == 2
    assert "not found" in capsys.readouterr().err
