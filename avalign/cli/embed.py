"""``avalign embed`` — baseline time-pooled log-mel descriptors for a manifest.

This is the dependency-light baseline embedding: it needs no trained model,
just the audio pipeline, so it is handy for smoke-testing retrieval plumbing
before wiring up a real encoder from :mod:`avalign.torchext`.
"""

from __future__ import annotations

import argparse

import numpy as np

from avalign.audio import log_mel_spectrogram
from avalign.audio.io import load_wav
from avalign.data import read_manifest


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "embed", help="time-pooled log-mel descriptors for a manifest's audio"
    )
    parser.add_argument("--manifest", required=True, help="JSONL manifest path")
    parser.add_argument("--out", required=True, help="output .npy of shape (N, n_mels)")
    parser.add_argument("--n-mels", type=int, default=64)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    pairs = read_manifest(args.manifest)
    descriptors = []
    for pair in pairs:
        waveform, sample_rate = load_wav(pair.audio)
        mel = log_mel_spectrogram(waveform, sample_rate=sample_rate, n_mels=args.n_mels)
        descriptors.append(mel.mean(axis=1))  # pool over time
    out = np.stack(descriptors)
    np.save(args.out, out)
    print(f"wrote {out.shape} descriptors to {args.out}")
    return 0
