"""``avalign eval`` — retrieval metrics from saved embedding arrays."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from avalign.metrics import retrieval_report


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "eval", help="compute bidirectional retrieval metrics from two .npy files"
    )
    parser.add_argument("--audio-emb", required=True, help="(N, D) audio embeddings .npy")
    parser.add_argument("--video-emb", required=True, help="(N, D) video embeddings .npy")
    parser.add_argument("--ks", default="1,5,10", help="comma-separated cutoffs")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        audio = np.load(args.audio_emb)
        video = np.load(args.video_emb)
    except FileNotFoundError as exc:
        print(f"error: embedding file not found: {exc.filename}", file=sys.stderr)
        return 2
    ks = tuple(int(x) for x in args.ks.split(","))
    report = retrieval_report(audio, video, ks=ks)
    print(json.dumps(report, indent=2))
    return 0
