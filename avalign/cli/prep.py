"""``avalign prep`` — build a JSONL manifest by pairing audio/video files."""

from __future__ import annotations

import argparse
from pathlib import Path

from avalign.data import AVPair, validate_manifest, write_manifest


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prep", help="pair audio/video files by filename stem into a manifest"
    )
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out", required=True, help="output .jsonl path")
    parser.add_argument("--audio-ext", default=".wav")
    parser.add_argument("--video-ext", default=".mp4")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    audio_dir = Path(args.audio_dir)
    video_dir = Path(args.video_dir)

    pairs: list[AVPair] = []
    for audio_path in sorted(audio_dir.glob(f"*{args.audio_ext}")):
        video_path = video_dir / f"{audio_path.stem}{args.video_ext}"
        if video_path.exists():
            pairs.append(AVPair(id=audio_path.stem, audio=str(audio_path), video=str(video_path)))

    validate_manifest(pairs)
    write_manifest(pairs, args.out)
    print(f"wrote {len(pairs)} pairs to {args.out}")
    return 0
