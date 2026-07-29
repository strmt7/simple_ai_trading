"""Installed CLI entry point with independently registered command extensions."""

from __future__ import annotations

import argparse
import sys

from . import cli
from .polymarket_live_cli import register_polymarket_live_command


def _build_parser() -> argparse.ArgumentParser:
    """Extend the established CLI parser without duplicating legacy commands."""

    parser = cli._build_parser()  # noqa: SLF001 - compatibility boundary
    subparsers = next(
        (
            action
            for action in parser._actions  # noqa: SLF001
            if isinstance(action, argparse._SubParsersAction)  # noqa: SLF001
        ),
        None,
    )
    if subparsers is None:
        raise RuntimeError("CLI parser has no command registry")
    register_polymarket_live_command(subparsers)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return cli.command_menu(argparse.Namespace())
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - exercised by process smoke
    raise SystemExit(main())
