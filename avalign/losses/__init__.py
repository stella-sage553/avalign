"""Contrastive loss functions (NumPy reference implementations)."""

from avalign.losses.functional import (
    cosine_similarity_matrix,
    l2_normalize,
    log_softmax,
    softmax,
)
from avalign.losses.info_nce import info_nce, symmetric_info_nce

__all__ = [
    "softmax",
    "log_softmax",
    "l2_normalize",
    "cosine_similarity_matrix",
    "info_nce",
    "symmetric_info_nce",
]
