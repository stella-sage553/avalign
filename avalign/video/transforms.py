"""NumPy frame transforms for video tensors shaped ``(..., H, W, C)``."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["to_float", "normalize", "center_crop"]

#: ImageNet RGB statistics, a common default for pretrained vision backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_float(frames: np.ndarray) -> np.ndarray:
    """Convert ``uint8`` frames in ``[0, 255]`` to float in ``[0, 1]``."""
    return np.asarray(frames, dtype=np.float64) / 255.0


def normalize(
    frames: np.ndarray,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    """Channel-wise normalisation; ``frames`` are expected in ``[0, 1]``."""
    frames = np.asarray(frames, dtype=np.float64)
    mean_arr = np.asarray(mean, dtype=np.float64)
    std_arr = np.asarray(std, dtype=np.float64)
    return (frames - mean_arr) / std_arr


def center_crop(frames: np.ndarray, size: int) -> np.ndarray:
    """Crop a centred ``size x size`` square from the spatial dimensions."""
    h, w = frames.shape[-3], frames.shape[-2]
    if size > h or size > w:
        raise ValueError(f"crop size {size} exceeds frame ({h}x{w})")
    top = (h - size) // 2
    left = (w - size) // 2
    return frames[..., top : top + size, left : left + size, :]
