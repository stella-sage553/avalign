"""A minimal training loop for the audio-visual contrastive model."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from avalign.torchext.loss import InfoNCELoss

__all__ = ["Trainer"]

Batch = tuple[torch.Tensor, torch.Tensor]


class Trainer:
    """Wire a model, an InfoNCE criterion, and an AdamW optimiser together.

    The temperature parameter inside :class:`InfoNCELoss` is registered with
    the optimiser so it is learned alongside the encoders.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        temperature: float = 0.07,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.criterion = InfoNCELoss(temperature).to(device)
        params = list(self.model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def step(self, audio: torch.Tensor, video: torch.Tensor) -> float:
        """Run one optimisation step and return the scalar loss."""
        self.model.train()
        audio = audio.to(self.device)
        video = video.to(self.device)
        emb_a, emb_v = self.model(audio, video)
        loss = self.criterion(emb_a, emb_v)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def fit_epoch(self, batches: Iterable[Batch]) -> float:
        """Train over an iterable of ``(audio, video)`` batches; return mean loss."""
        losses = [self.step(audio, video) for audio, video in batches]
        return sum(losses) / max(len(losses), 1)
