"""Retrieval and alignment evaluation metrics."""

from avalign.metrics.alignment import alignment_accuracy, diagonal_dominance
from avalign.metrics.retrieval import (
    mean_rank,
    median_rank,
    mrr,
    recall_at_k,
    retrieval_report,
)

__all__ = [
    "recall_at_k",
    "median_rank",
    "mean_rank",
    "mrr",
    "retrieval_report",
    "alignment_accuracy",
    "diagonal_dominance",
]
