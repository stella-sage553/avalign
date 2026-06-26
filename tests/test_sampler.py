import numpy as np

from avalign.data.sampler import BatchSampler, sample_negatives


def test_batch_sampler_drop_last():
    sampler = BatchSampler(10, batch_size=4, shuffle=False, drop_last=True)
    batches = list(sampler)
    assert len(batches) == 2
    assert all(len(b) == 4 for b in batches)
    assert len(sampler) == 2


def test_batch_sampler_keeps_partial_batch():
    sampler = BatchSampler(10, batch_size=4, shuffle=False, drop_last=False)
    assert len(list(sampler)) == 3
    assert len(sampler) == 3


def test_sample_negatives_excludes_anchor():
    rng = np.random.default_rng(0)
    negatives = sample_negatives(20, anchor=3, n_negatives=5, rng=rng)
    assert 3 not in negatives
    assert len(negatives) == 5
