"""Deterministic seeding across NumPy, the stdlib, and (optionally) torch."""

from __future__ import annotations

import os
import random


def set_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """Seed all RNGs avalign may touch.

    Args:
        seed: The seed value.
        deterministic_torch: If True and torch is installed, also configure
            torch for deterministic behaviour (slower, but reproducible).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
