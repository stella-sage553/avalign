"""Symmetric InfoNCE as a torch Module with a learnable temperature."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["InfoNCELoss"]


class InfoNCELoss(nn.Module):
    """CLIP-style symmetric contrastive loss.

    The temperature is parameterised in log-space and, by default, learned
    jointly with the encoders (as in CLIP). The matched pairs are the
    diagonal of the audio-video similarity matrix.
    """

    def __init__(self, temperature: float = 0.07, learnable: bool = True) -> None:
        super().__init__()
        log_temp = torch.tensor(math.log(temperature))
        if learnable:
            self.log_temp = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temp", log_temp)

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        audio = F.normalize(audio, dim=-1)
        video = F.normalize(video, dim=-1)
        logits = audio @ video.t() / self.log_temp.exp()
        targets = torch.arange(audio.size(0), device=audio.device)
        loss_a = F.cross_entropy(logits, targets)
        loss_v = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_a + loss_v)
