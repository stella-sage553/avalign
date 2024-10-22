# Architecture

`avalign` is organised as a NumPy core with an optional PyTorch layer bolted on
top. Nothing in the core imports `torch`, and importing `avalign.torchext` does
not import `torch` until a class is actually used.

## Data flow

```
raw media ──▶ audio: log-mel + SpecAugment ─┐
              video: frame sampling ─────────┤
                                             ▼
                              encoders (torchext, optional)
                                             ▼
                              projection heads ─▶ embeddings
                                             ▼
              losses.symmetric_info_nce  /  torchext.InfoNCELoss
                                             ▼
              metrics: recall@k, rank stats, alignment
```

## Modules

| Package            | Responsibility                                            |
| ------------------ | --------------------------------------------------------- |
| `avalign.audio`    | `log_mel_spectrogram`, mel filterbank, SpecAugment, WAV I/O |
| `avalign.video`    | frame-index sampling strategies and frame transforms      |
| `avalign.losses`   | softmax/normalize helpers and the InfoNCE losses          |
| `avalign.metrics`  | retrieval (recall@k, ranks, MRR) and alignment diagnostics |
| `avalign.data`     | JSONL manifests and batch / negative sampling             |
| `avalign.torchext` | encoders, `AVContrastiveModel`, `InfoNCELoss`, `Trainer`  |
| `avalign.cli`      | the `avalign` command and its subcommands                 |

## The contrastive objective

A batch is `N` aligned `(audio, video)` pairs. We L2-normalise both sides,
form the `N×N` cosine-similarity matrix, scale by a temperature, and treat the
diagonal as the positives. The loss is the average of the audio→video and
video→audio cross-entropies — the same construction as CLIP, specialised to the
two modalities here.

## Why a NumPy core

The math (loss, mel filterbank, metrics) is small and exact, so it is written
in NumPy and tested directly. That keeps the dependency surface tiny, makes CI
fast, and lets the library be used for evaluation or preprocessing without a
deep-learning stack. The torch layer reuses the same definitions in spirit, but
as differentiable `nn.Module`s.
