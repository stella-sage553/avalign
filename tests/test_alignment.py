import numpy as np

from avalign.losses.functional import cosine_similarity_matrix
from avalign.metrics.alignment import alignment_accuracy, diagonal_dominance


def test_alignment_accuracy_identity():
    assert alignment_accuracy(np.eye(5)) == 1.0


def test_alignment_accuracy_shuffled():
    sim = np.eye(4)[::-1]  # argmax never on the diagonal for n>1
    assert alignment_accuracy(sim) < 1.0


def test_diagonal_dominance_positive_when_aligned():
    z = np.random.default_rng(0).standard_normal((10, 8))
    sim = cosine_similarity_matrix(z, z)
    assert diagonal_dominance(sim) > 0
