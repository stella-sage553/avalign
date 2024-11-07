"""Compare retrieval metrics for aligned vs. random audio-visual embeddings.

Run with:  python examples/02_retrieval_eval.py
"""

import numpy as np

from avalign.metrics import retrieval_report


def main() -> None:
    rng = np.random.default_rng(0)
    n, dim = 64, 128

    # Aligned: video is a noisy copy of the audio embedding -> easy retrieval.
    audio = rng.standard_normal((n, dim))
    video_aligned = audio + 0.05 * rng.standard_normal((n, dim))
    video_random = rng.standard_normal((n, dim))

    for name, video in [("aligned", video_aligned), ("random", video_random)]:
        report = retrieval_report(audio, video, ks=(1, 5, 10))
        a2v = report["audio_to_video"]
        recall = {k: round(v, 3) for k, v in a2v["recall"].items()}
        print(f"{name:8s} | recall@k {recall} | median_rank {a2v['median_rank']:.1f}")


if __name__ == "__main__":
    main()
