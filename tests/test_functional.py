import numpy as np

from avalign.losses.functional import (
    cosine_similarity_matrix,
    l2_normalize,
    log_softmax,
    softmax,
)


def test_softmax_sums_to_one():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = softmax(x, axis=1)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


def test_log_softmax_matches_log_of_softmax():
    x = np.random.default_rng(0).standard_normal((4, 5))
    assert np.allclose(log_softmax(x, axis=1), np.log(softmax(x, axis=1)))


def test_l2_normalize_unit_norm():
    x = np.random.default_rng(1).standard_normal((3, 7))
    normed = l2_normalize(x, axis=1)
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)


def test_cosine_similarity_matrix_diagonal_is_one():
    x = np.random.default_rng(2).standard_normal((5, 8))
    sim = cosine_similarity_matrix(x, x)
    assert sim.shape == (5, 5)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-6)
