"""Command-line entry point for avalign.

Subcommands are registered lazily so that ``avalign --help`` stays fast and
the optional torch-backed ``train`` command does not import torch unless it
is actually invoked.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from avalign import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avalign",
        description="Contrastive audio-visual alignment toolkit.",
    )
    parser.add_argument(
        "--version", action="version", version=f"avalign {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    from avalign.cli import embed, eval, prep

    prep.add_parser(subparsers)
    embed.add_parser(subparsers)
    eval.add_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
