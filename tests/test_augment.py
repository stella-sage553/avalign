import numpy as np

from avalign.audio.augment import SpecAugment, time_mask


def test_time_mask_zeros_some_entries():
    spec = np.ones((10, 60))
    rng = np.random.default_rng(0)
    out = time_mask(spec, max_width=12, n_masks=3, rng=rng)
    assert out.shape == spec.shape
    assert (out == 0).any()


def test_specaugment_is_deterministic_with_seed():
    spec = np.ones((20, 40))
    a = SpecAugment(seed=7)(spec.copy())
    b = SpecAugment(seed=7)(spec.copy())
    assert np.array_equal(a, b)


def test_specaugment_does_not_mutate_input():
    spec = np.ones((8, 8))
    SpecAugment(seed=1)(spec)
    assert (spec == 1.0).all()
