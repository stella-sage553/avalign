import pytest

torch = pytest.importorskip("torch")

from avalign.torchext import AVContrastiveModel, Trainer  # noqa: E402


def test_training_overfits_a_fixed_batch():
    torch.manual_seed(0)
    model = AVContrastiveModel(embed_dim=32)
    trainer = Trainer(model, lr=1e-3)
    audio = torch.randn(4, 1, 64, 64)
    video = torch.randn(4, 3, 4, 16, 16)

    first = trainer.step(audio, video)
    last = first
    for _ in range(8):
        last = trainer.step(audio, video)

    assert isinstance(first, float)
    assert last < first  # the loop should reduce loss on a repeated batch
