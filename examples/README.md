# Examples

Small, runnable scripts. They use only the NumPy core, so:

```bash
pip install avalign
python examples/01_logmel_specaugment.py
python examples/02_retrieval_eval.py
```

- `01_logmel_specaugment.py` — waveform → log-mel spectrogram → SpecAugment.
- `02_retrieval_eval.py` — build embeddings and compute a retrieval report,
  comparing aligned vs. random pairs.
