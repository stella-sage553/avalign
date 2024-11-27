"""Optional PyTorch layer: neural encoders, the contrastive model, and a trainer.

This subpackage requires the ``torch`` extra::

    pip install "avalign[torch]"

Importing :mod:`avalign.torchext` never imports torch; the heavy modules are
loaded lazily on first attribute access so the NumPy core stays usable on
machines without a deep-learning stack.
"""

from __future__ import annotations

from typing import Any

try:
    import torch as _torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch
    _HAS_TORCH = False

__all__ = [
    "AudioEncoder",
    "VideoEncoder",
    "ProjectionHead",
    "AVContrastiveModel",
    "InfoNCELoss",
    "Trainer",
]


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "avalign.torchext requires PyTorch. Install it with: pip install 'avalign[torch]'"
        )


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _require_torch()
    from avalign.torchext import encoders, loss, model, trainer

    table = {
        "AudioEncoder": encoders.AudioEncoder,
        "VideoEncoder": encoders.VideoEncoder,
        "ProjectionHead": encoders.ProjectionHead,
        "AVContrastiveModel": model.AVContrastiveModel,
        "InfoNCELoss": loss.InfoNCELoss,
        "Trainer": trainer.Trainer,
    }
    return table[name]
