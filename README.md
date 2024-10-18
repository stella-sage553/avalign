# avalign

**Contrastive audio-visual alignment, with a dependency-light core.**

`avalign` is a toolkit for learning and evaluating aligned audio and video
representations with a CLIP-style symmetric InfoNCE objective. The numerical
core — log-mel preprocessing, SpecAugment, the contrastive loss, retrieval
metrics, and dataset plumbing — is pure NumPy, so it installs and tests in
seconds and runs anywhere. Neural encoders and a training loop live behind an
optional `torch` extra.

## Why

Most audio-visual self-supervised code ships as a research dump: one training
script, hard-coded paths, no tests. `avalign` factors the reusable pieces into
a small, typed, tested library you can drop into your own pipeline.

## Features

- Log-mel spectrograms and a triangular mel filterbank in NumPy
- SpecAugment time/frequency masking with reproducible seeding
- Symmetric InfoNCE (audio↔video) over a cosine-similarity matrix
- Retrieval metrics: recall@k, median/mean rank, MRR, both directions
- Alignment diagnostics (top-1 accuracy, diagonal dominance)
- JSON-Lines dataset manifests and in-batch negative sampling
- Optional PyTorch encoders, an `AVContrastiveModel`, and a `Trainer`
- A small CLI: `prep`, `embed`, `eval`, and (with torch) `train`

## Install

```bash
pip install avalign            # NumPy core
pip install "avalign[torch]"   # adds the PyTorch encoders + trainer
```

## Quickstart

```python
import numpy as np
from avalign.audio import log_mel_spectrogram
from avalign.losses import symmetric_info_nce
from avalign.metrics import retrieval_report

# 1) turn a waveform into a log-mel spectrogram
wav = np.random.default_rng(0).standard_normal(16000)   # 1s @ 16 kHz
mel = log_mel_spectrogram(wav, sample_rate=16000, n_mels=64)

# 2) train against the contrastive loss (audio/video are (B, D) embeddings)
audio = np.random.default_rng(1).standard_normal((32, 128))
video = np.random.default_rng(2).standard_normal((32, 128))
loss = symmetric_info_nce(audio, video, temperature=0.07)

# 3) evaluate retrieval in both directions
report = retrieval_report(audio, video, ks=(1, 5, 10))
print(report["audio_to_video"]["recall"])
```

## Command line

```bash
# pair audio/video files into a manifest
avalign prep --audio-dir audio/ --video-dir video/ --out pairs.jsonl

# baseline (no-model) time-pooled log-mel descriptors
avalign embed --manifest pairs.jsonl --out audio_emb.npy

# retrieval metrics from two embedding arrays
avalign eval --audio-emb audio_emb.npy --video-emb video_emb.npy --ks 1,5,10
```

## Package layout

```
avalign/
├── audio/      log-mel spectrograms, SpecAugment, wav I/O
├── video/      frame sampling and transforms
├── losses/     functional helpers + InfoNCE
├── metrics/    retrieval and alignment metrics
├── data/       manifests and batch sampling
├── torchext/   optional PyTorch encoders, model, loss, trainer
└── cli/        the avalign command
```

## License

MIT — see [LICENSE](LICENSE).
