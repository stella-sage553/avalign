"""End-to-end audio-visual contrastive model."""

from __future__ import annotations

import torch
from torch import nn

from avalign.torchext.encoders import AudioEncoder, ProjectionHead, VideoEncoder

__all__ = ["AVContrastiveModel"]


class AVContrastiveModel(nn.Module):
    """Two encoders + projection heads producing aligned embeddings.

    ``forward`` returns ``(audio_embedding, video_embedding)``, each L2
    normalised and ready for :class:`avalign.torchext.loss.InfoNCELoss`.
    """

    def __init__(
        self,
        audio_hidden: int = 256,
        video_hidden: int = 256,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.audio_encoder = AudioEncoder(audio_hidden)
        self.video_encoder = VideoEncoder(video_hidden)
        self.audio_proj = ProjectionHead(audio_hidden, embed_dim)
        self.video_proj = ProjectionHead(video_hidden, embed_dim)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.audio_proj(self.audio_encoder(audio))

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return self.video_proj(self.video_encoder(video))

    def forward(
        self, audio: torch.Tensor, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_audio(audio), self.encode_video(video)
