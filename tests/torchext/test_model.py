import pytest

torch = pytest.importorskip("torch")

from avalign.torchext import AVContrastiveModel  # noqa: E402


def test_forward_shapes_and_normalisation():
    model = AVContrastiveModel(embed_dim=64)
    audio = torch.randn(2, 1, 64, 64)
    video = torch.randn(2, 3, 4, 32, 32)
    emb_a, emb_v = model(audio, video)
    assert emb_a.shape == (2, 64)
    assert emb_v.shape == (2, 64)
    assert torch.allclose(emb_a.norm(dim=-1), torch.ones(2), atol=1e-5)
