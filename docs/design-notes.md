# Design notes

A few decisions worth recording, mostly so I don't relitigate them later.

## Temperature

InfoNCE is sensitive to temperature. The NumPy `info_nce`/`symmetric_info_nce`
take it as a plain argument (default `0.07`, the CLIP value). The torch
`InfoNCELoss` parameterises it in log-space and, by default, *learns* it — also
following CLIP, where a learnable logit scale trained better than a fixed one.
The log-space parameterisation keeps the temperature positive without a clamp.

## HTK vs Slaney filterbanks

The filterbank uses HTK mel spacing (the `2595·log10` formula) with Slaney-style
area normalisation, so each filter integrates to roughly the same energy
instead of peaking at 1. This matches the common `librosa` default closely
enough for transfer, while staying a few lines of NumPy.

## Ranks under ties

`metrics._ranks` defines a match's rank as *one plus the number of candidates
scoring strictly higher*. Under ties this is the optimistic rank. It is
vectorised as a single broadcasted comparison rather than a per-row `argsort`,
which also sidesteps argsort's arbitrary tie-breaking.

## Decoder-agnostic data

Manifests store *paths*, and the video helpers return *frame indices* rather
than decoded frames. Decoding (ffmpeg, torchaudio, decord, …) is intentionally
left to the caller, so the toolkit doesn't pin a heavy media stack.

## Open questions

- A momentum encoder / queue (MoCo-style) would decouple negatives from batch
  size. Not done yet — in-batch negatives are enough for the current scope.
- The video encoder is a small 3-D CNN; swapping in a pretrained backbone is
  future work. See the TODO in `torchext/encoders.py`.
