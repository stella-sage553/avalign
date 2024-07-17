# Usage

> Draft — expands as the API stabilises.

## Install

```bash
pip install avalign           # NumPy core
pip install "avalign[torch]"  # adds the PyTorch encoders + trainer
```

## Audio preprocessing

```python
import numpy as np
from avalign.audio import log_mel_spectrogram

wav = np.random.default_rng(0).standard_normal(16000)  # 1s @ 16 kHz
mel = log_mel_spectrogram(wav, sample_rate=16000, n_mels=64)
print(mel.shape)  # (64, n_frames)
```

## Contrastive loss

```python
from avalign.losses import symmetric_info_nce

# audio/video are (batch, embed_dim) arrays of matched pairs
loss = symmetric_info_nce(audio, video, temperature=0.07)
```

## Retrieval evaluation

```python
from avalign.metrics import retrieval_report

report = retrieval_report(audio, video, ks=(1, 5, 10))
print(report["audio_to_video"]["recall"])
```
