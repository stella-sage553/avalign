import pytest

torch = pytest.importorskip("torch")

from avalign.torchext import InfoNCELoss  # noqa: E402


def test_infonce_aligned_is_low():
    z = torch.randn(16, 32)
    loss = InfoNCELoss(temperature=0.05, learnable=False)
    assert loss(z, z).item() < 0.5


def test_infonce_is_positive_for_random_pairs():
    a = torch.randn(8, 16)
    v = torch.randn(8, 16)
    assert InfoNCELoss()(a, v).item() > 0.0
