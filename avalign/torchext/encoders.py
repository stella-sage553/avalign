"""PyTorch encoders for audio spectrograms and video frame stacks."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["AudioEncoder", "VideoEncoder", "ProjectionHead"]


class AudioEncoder(nn.Module):
    """A small 2-D CNN over log-mel spectrograms.

    Input shape ``(B, 1, n_mels, n_frames)`` (a missing channel dimension is
    added automatically); output shape ``(B, hidden_dim)``.
    """

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.features(x).flatten(1)
        return self.fc(h)


class VideoEncoder(nn.Module):
    """A 3-D CNN over a stack of frames.

    Input shape ``(B, C, T, H, W)``; output shape ``(B, hidden_dim)``.
    """

    def __init__(self, hidden_dim: int = 256, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(64, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.fc(h)


class ProjectionHead(nn.Module):
    """MLP head mapping encoder features into the shared, L2-normalised space."""

    def __init__(self, in_dim: int, embed_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)
