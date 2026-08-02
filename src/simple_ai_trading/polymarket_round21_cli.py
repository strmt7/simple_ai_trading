"""CLI registration for terminal Polymarket Round 21 corpus publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import duckdb

from .polymarket_round21_corpus_store import publish_round21_core_corpus
from .polymarket_round21_terminal import load_round21_terminal_transport_manifest


def register_polymarket_round21_commands(
    subparsers: argparse._SubParsersAction,  # noqa: SLF001 - argparse has no public type
) -> None:
    parser = subparsers.add_parser(
        "polymarket-round21-corpus",
        help="publish the terminal target-blind Round 21 core corpus",
        description=(
            "After the independent Polymarket capture is terminal, reconcile its "
            "exact receipts and atomically publish physically separate development "
            "and sealed-test core feature stores. This command reads no outcomes, "
            "models, Binance data, credentials, accounts, or orders."
        ),
    )
    parser.add_argument(
        "--source-database",
        required=True,
        help="closed Polymarket Round 21 evidence DuckDB",
    )
    parser.add_argument(
        "--terminal-transport-manifest",
        required=True,
        help="hash-valid terminal Round 21 transport manifest JSON",
    )
    parser.add_argument(
        "--publication-directory",
        required=True,
        help="new directory for one atomic development/sealed-test publication",
    )
    parser.add_argument(
        "--repository",
        default=".",
        help="repository root containing the frozen Round 21 design",
    )
    parser.add_argument(
        "--observed-at-ms",
        type=int,
        default=None,
        help="optional fixed audit timestamp for reproducible controlled runs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the complete canonical publication manifest",
    )
    parser.set_defaults(func=command_polymarket_round21_corpus)


def command_polymarket_round21_corpus(args: argparse.Namespace) -> int:
    try:
        transport = load_round21_terminal_transport_manifest(
            Path(args.terminal_transport_manifest)
        )
        manifest = publish_round21_core_corpus(
            repository=Path(args.repository),
            source_database=Path(args.source_database),
            terminal_transport_manifest=transport,
            publication_directory=Path(args.publication_directory),
            observed_at_ms=args.observed_at_ms,
        )
    except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
        print(
            "polymarket-round21-corpus failed: "
            f"{exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            "Round 21 core corpus published: "
            f"manifest={manifest['manifest_sha256']} "
            "authority=false"
        )
    return 0


__all__ = [
    "command_polymarket_round21_corpus",
    "register_polymarket_round21_commands",
]
