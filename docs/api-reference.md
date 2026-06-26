# API reference

Stable, public entry points. Anything prefixed with `_` is internal.

## `avalign.audio`

- `log_mel_spectrogram(signal, sample_rate=16000, n_fft=400, hop_length=160, n_mels=64, f_min=0.0, f_max=None, amin=1e-10) -> np.ndarray`
  — log-mel spectrogram of shape `(n_mels, n_frames)`.
- `LogMelSpectrogram(...)` — a callable dataclass pinning the parameters above.
- `frame_signal(signal, frame_length, hop_length)` — overlapping frames.
- `stft_power(signal, n_fft, hop_length, window=None)` — short-time power spectrum.
- `mel_filterbank(n_mels, n_fft, sample_rate, f_min=0.0, f_max=None)` — triangular filters.
- `hz_to_mel(freq)`, `mel_to_hz(mel)` — HTK mel-scale conversions.

### `avalign.audio.augment`

- `time_mask(spec, max_width, n_masks=1, rng=None)` / `freq_mask(...)`
- `SpecAugment(freq_width=8, time_width=16, n_freq_masks=1, n_time_masks=1, seed=None)`

## `avalign.losses`

- `info_nce(audio, video, temperature=0.07) -> float`
- `symmetric_info_nce(audio, video, temperature=0.07) -> float`
- `cosine_similarity_matrix(a, b)`, `l2_normalize(x, axis=-1, eps=1e-12)`
- `softmax(x, axis=-1)`, `log_softmax(x, axis=-1)`

## `avalign.metrics`

- `recall_at_k(sim, ks=(1, 5, 10)) -> dict[int, float]`
- `median_rank(sim)`, `mean_rank(sim)`, `mrr(sim)`
- `retrieval_report(audio, video, ks=(1, 5, 10))` — both directions
- `alignment_accuracy(sim)`, `diagonal_dominance(sim)`

## `avalign.data`

- `AVPair(id, audio, video, label=None)`
- `read_manifest(path)`, `write_manifest(pairs, path)`, `validate_manifest(pairs)`
- `BatchSampler(n_items, batch_size, shuffle=True, drop_last=True, seed=None)`
- `sample_negatives(n_items, anchor, n_negatives, rng=None)`

## `avalign.video`

- `uniform_indices`, `center_indices`, `random_indices`, `sample_indices`
- `to_float`, `normalize`, `center_crop`

## `avalign.torchext` (requires the `torch` extra)

- `AudioEncoder`, `VideoEncoder`, `ProjectionHead`
- `AVContrastiveModel(audio_hidden=256, video_hidden=256, embed_dim=128)`
- `InfoNCELoss(temperature=0.07, learnable=True)`
- `Trainer(model, lr=3e-4, weight_decay=1e-4, temperature=0.07, device="cpu")`

## `avalign.config`

- `AudioConfig`, `ModelConfig`, `TrainConfig`, `Config`
- `Config.save(path)` / `Config.load(path)` — JSON round-trip.
