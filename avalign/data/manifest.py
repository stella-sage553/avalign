"""Audio-visual dataset manifests stored as JSON Lines.

Each line is a JSON object describing one aligned pair, e.g.::

    {"id": "clip0001", "audio": "audio/clip0001.wav", "video": "video/clip0001.mp4"}

Keeping the manifest decoder-agnostic lets the same files drive both the
NumPy reference pipeline and the optional torch trainer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["AVPair", "read_manifest", "write_manifest"]


@dataclass
class AVPair:
    """A single aligned audio-visual example."""

    id: str
    audio: str
    video: str
    label: str | None = None


def read_manifest(path: str | Path) -> list[AVPair]:
    """Read a JSON Lines manifest into a list of :class:`AVPair`."""
    pairs: list[AVPair] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pairs.append(AVPair(**json.loads(line)))
    return pairs


def write_manifest(pairs: list[AVPair], path: str | Path) -> None:
    """Write :class:`AVPair` records to a JSON Lines manifest."""
    with open(path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            obj = {k: v for k, v in asdict(pair).items() if v is not None}
            fh.write(json.dumps(obj) + "\n")
