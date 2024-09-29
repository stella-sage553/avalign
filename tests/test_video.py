import numpy as np
import pytest

from avalign.video.sampling import (
    center_indices,
    sample_indices,
    uniform_indices,
)
from avalign.video.transforms import center_crop, normalize, to_float


def test_uniform_indices_cover_endpoints():
    idx = uniform_indices(10, 4)
    assert idx[0] == 0
    assert idx[-1] == 9
    assert len(idx) == 4


def test_center_indices_are_contiguous():
    idx = center_indices(10, 4)
    assert len(idx) == 4
    assert (np.diff(idx) == 1).all()


def test_sample_indices_dispatch_and_error():
    assert len(sample_indices(20, 5, "uniform")) == 5
    with pytest.raises(ValueError):
        sample_indices(20, 5, "nope")


def test_frame_transforms():
    frames = (np.ones((2, 8, 8, 3)) * 255).astype(np.uint8)
    floated = to_float(frames)
    assert floated.max() <= 1.0
    assert normalize(floated).shape == floated.shape
    assert center_crop(floated, 4).shape == (2, 4, 4, 3)
