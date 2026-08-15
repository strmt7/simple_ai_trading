#!/usr/bin/env python3
"""Materialize the target-blind Round 28 Binance BBO overlay."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
import json
from pathlib import Path
import sys
import threading
import time

from simple_ai_trading.polymarket_round21_sidecar_replay import (
    replay_round21_optional_binance_features,
)
from simple_ai_trading.polymarket_round27_feature_store import Round27FeatureStore
from simple_ai_trading.polymarket_round28_book_ticker import (
    materialize_round28_book_ticker_overlay,
    write_round28_book_ticker_overlay,
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 28 input JSON has duplicate keys")
        output[key] = value
    return output


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Round 28 sidecar terminal manifest is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 28 sidecar terminal manifest must be an object")
    return value


class _Heartbeat(AbstractContextManager["_Heartbeat"]):
    def __init__(
        self, *, phase: str, interval_seconds: int, detail: Mapping[str, object]
    ):
        self.phase = phase
        self.interval_seconds = interval_seconds
        self.detail = dict(detail)
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_Heartbeat":
        self._emit()
        self.thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop.set()
        self.thread.join(timeout=self.interval_seconds + 1.0)

    def _emit(self) -> None:
        payload = {
            "phase": self.phase,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            "observed_at_ms": time.time_ns() // 1_000_000,
            **self.detail,
        }
        print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)

    def _run(self) -> None:
        while not self.stop.wait(self.interval_seconds):
            self._emit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a terminal public Binance sidecar into the incremental, "
            "target-blind Round 28 BBO overlay."
        )
    )
    parser.add_argument("--round27-feature-store", type=Path, required=True)
    parser.add_argument("--sidecar-database", type=Path, required=True)
    parser.add_argument("--sidecar-terminal-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--progress-interval-seconds",
        type=int,
        default=30,
        choices=range(5, 301),
        metavar="[5-300]",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() or Path(f"{output}.wal").exists():
        raise ValueError("Round 28 output must be a fresh terminal path")
    with Round27FeatureStore(args.round27_feature_store, read_only=True) as store:
        base_rows = tuple(
            sorted(store.load_rows(), key=lambda row: row.decision_time_ms)
        )
        feature_audit = store.audit()
    decisions = tuple(row.decision_time_ms for row in base_rows)
    if not base_rows or len(decisions) != len(set(decisions)):
        raise ValueError("Round 28 base decision population differs")
    terminal_manifest = _read_json(args.sidecar_terminal_manifest.resolve())
    with _Heartbeat(
        phase="terminal-sidecar-replay",
        interval_seconds=args.progress_interval_seconds,
        detail={
            "base_decision_count": len(decisions),
            "round27_feature_store_audit_sha256": feature_audit["audit_sha256"],
        },
    ):
        replay = replay_round21_optional_binance_features(
            source_database=args.sidecar_database.resolve(),
            terminal_manifest=terminal_manifest,
            decision_times_ms=decisions,
        )
    overlay, report = materialize_round28_book_ticker_overlay(
        base_rows=base_rows,
        sidecar_replay=replay,
    )
    with _Heartbeat(
        phase="overlay-store-write",
        interval_seconds=args.progress_interval_seconds,
        detail={
            "accepted_decision_count": len(overlay),
            "rejected_decision_count": len(base_rows) - len(overlay),
        },
    ):
        write_round28_book_ticker_overlay(output, rows=overlay, report=report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
