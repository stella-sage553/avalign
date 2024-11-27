"""``avalign train`` — train the AV contrastive model (requires the torch extra).

torch is imported lazily inside :func:`run` so registering this subcommand
never pulls in a deep-learning stack. Without precomputed features the
command runs on synthetic tensors, which is enough to smoke-test the loop.
"""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train", help="train the contrastive model on synthetic data (needs torch)"
    )
    parser.add_argument("--config", default=None, help="optional Config JSON path")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    import torch

    from avalign.config import Config
    from avalign.torchext import AVContrastiveModel, Trainer
    from avalign.utils.seed import set_seed

    cfg = Config.load(args.config) if args.config else Config()
    set_seed(args.seed)

    model = AVContrastiveModel(
        audio_hidden=cfg.model.audio_hidden,
        video_hidden=cfg.model.video_hidden,
        embed_dim=cfg.model.embed_dim,
    )
    trainer = Trainer(
        model,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        temperature=cfg.model.temperature,
    )

    generator = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        audio = torch.randn(args.batch_size, 1, args.n_mels, args.frames, generator=generator)
        video = torch.randn(args.batch_size, 3, 4, 16, 16, generator=generator)
        loss = trainer.step(audio, video)
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:4d} | loss {loss:.4f}")
    return 0
