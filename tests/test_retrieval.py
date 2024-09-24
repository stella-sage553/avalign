import numpy as np

from avalign.metrics.retrieval import (
    mean_rank,
    median_rank,
    mrr,
    recall_at_k,
    retrieval_report,
)


def test_recall_perfect_identity():
    sim = np.eye(10)
    recall = recall_at_k(sim, [1, 5])
    assert recall[1] == 1.0
    assert recall[5] == 1.0


def test_recall_worst_case():
    # diagonal is the smallest value in each row -> correct item ranked last.
    sim = np.ones((4, 4)) - np.eye(4)
    assert recall_at_k(sim, [1])[1] == 0.0
    assert median_rank(sim) == 4.0
    assert mean_rank(sim) == 4.0


def test_mrr_within_bounds():
    rng = np.random.default_rng(0)
    sim = np.eye(6) + 0.1 * rng.standard_normal((6, 6))
    assert 0.0 < mrr(sim) <= 1.0


def test_report_structure():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((8, 16))
    v = rng.standard_normal((8, 16))
    report = retrieval_report(a, v, ks=[1, 2])
    assert set(report) == {"audio_to_video", "video_to_audio"}
    assert set(report["audio_to_video"]) == {"recall", "median_rank", "mrr"}
