# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-27

### Added

- Log-mel spectrogram pipeline: framing, Hann-windowed STFT power, and a
  Slaney-normalised triangular mel filterbank (`avalign.audio`).
- SpecAugment time/frequency masking with reproducible seeding.
- Symmetric InfoNCE loss over a cosine-similarity matrix (`avalign.losses`).
- Retrieval metrics (recall@k, median/mean rank, MRR, bidirectional report)
  and alignment diagnostics (`avalign.metrics`).
- JSON-Lines dataset manifests, batch sampling, and in-batch negatives
  (`avalign.data`).
- Optional PyTorch layer: audio/video encoders, `AVContrastiveModel`,
  `InfoNCELoss`, and a `Trainer` (`avalign.torchext`).
- `avalign` CLI with `prep`, `embed`, `eval`, and `train` subcommands.
- Configuration dataclasses with JSON save/load.

[Unreleased]: https://github.com/stella-sage553/avalign/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/stella-sage553/avalign/releases/tag/v0.1.0
