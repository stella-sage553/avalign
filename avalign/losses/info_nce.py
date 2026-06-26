"""InfoNCE / NT-Xent contrastive losses for paired embeddings.

The canonical audio-visual setup is a batch of ``N`` aligned
``(audio, video)`` pairs. We build the ``N x N`` cosine-similarity matrix,
scale it by a temperature, and treat the diagonal as the positive pairs.
The symmetric loss averages the audio->video and video->audio
cross-entropy terms, exactly as in CLIP.
"""

from __future__ import annotations

import numpy as np

from avalign.losses.functional import cosine_similarity_matrix, log_softmax

__all__ = ["info_nce", "symmetric_info_nce"]


def _nce_from_logits(logits: np.ndarray, axis: int) -> float:
    """Cross-entropy of the diagonal positives over ``axis`` of ``logits``."""
    n = logits.shape[0]
    idx = np.arange(n)
    return float(-np.mean(log_softmax(logits, axis=axis)[idx, idx]))


def _validate(audio: np.ndarray, video: np.ndarray, temperature: float) -> None:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if audio.ndim != 2 or video.ndim != 2:
        raise ValueError("audio and video embeddings must be 2-D (n, d)")
    if audio.shape[0] != video.shape[0]:
        raise ValueError(
            f"batch size mismatch: audio has {audio.shape[0]} rows, "
            f"video has {video.shape[0]}"
        )


def info_nce(audio: np.ndarray, video: np.ndarray, temperature: float = 0.07) -> float:
    """One-directional InfoNCE matching each audio row to its paired video row.

    Args:
        audio: Audio embeddings, shape ``(n, d)``.
        video: Video embeddings, shape ``(n, d)``. Row ``i`` is the positive
            for ``audio[i]``; all other rows are in-batch negatives.
        temperature: Softmax temperature applied to the similarities.

    Returns:
        The mean negative log-likelihood of the positive pairs.
    """
    audio = np.asarray(audio)
    video = np.asarray(video)
    _validate(audio, video, temperature)
    logits = cosine_similarity_matrix(audio, video) / temperature
    return _nce_from_logits(logits, axis=1)


def symmetric_info_nce(
    audio: np.ndarray, video: np.ndarray, temperature: float = 0.07
) -> float:
    """Symmetric (CLIP-style) InfoNCE averaging both retrieval directions."""
    audio = np.asarray(audio)
    video = np.asarray(video)
    _validate(audio, video, temperature)
    logits = cosine_similarity_matrix(audio, video) / temperature
    return 0.5 * (_nce_from_logits(logits, axis=1) + _nce_from_logits(logits, axis=0))
