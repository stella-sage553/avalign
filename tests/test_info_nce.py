import numpy as np
import pytest

from avalign.losses import info_nce, symmetric_info_nce


def test_perfect_alignment_has_low_loss():
    z = np.random.default_rng(0).standard_normal((16, 32))
    assert symmetric_info_nce(z, z, temperature=0.05) < 0.01


def test_symmetric_is_average_of_both_directions():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((8, 16))
    v = rng.standard_normal((8, 16))
    expected = 0.5 * (info_nce(a, v, 0.1) + info_nce(v, a, 0.1))
    assert np.isclose(symmetric_info_nce(a, v, 0.1), expected)


def test_invalid_temperature_raises():
    z = np.zeros((4, 4))
    with pytest.raises(ValueError):
        info_nce(z, z, temperature=0.0)


def test_batch_size_mismatch_raises():
    with pytest.raises(ValueError):
        info_nce(np.zeros((4, 8)), np.zeros((5, 8)))
