"""Materialize the preregistered Round 25 forensic feature store once."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from simple_ai_trading.polymarket_round25_forensic_materialization import (
    materialize_round25_forensic_joint_feature_store,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-v2-transport-failure-forensic-audit-2026-08-14.json"
)
DEFAULT_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-v2-condition-salvage-contract-v1.json"
)


def _mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _write_once(path: Path, value: dict[str, object]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    if path.is_symlink() or path.exists() or partial.exists() or partial.is_symlink():
        raise ValueError("forensic scan receipt destination is not empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with partial.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one target-blind compact feature store from the qualified "
            "Round 25 transport-failure capture."
        )
    )
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--destination-db", type=Path, required=True)
    parser.add_argument("--scan-receipt", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--progress-interval", type=int, default=100_000)
    parser.add_argument("--observed-at-ms", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.monotonic()

    def progress(processed: int, total: int, elapsed: float) -> None:
        rate = processed / elapsed if elapsed > 0 else 0.0
        percent = processed * 100.0 / total if total > 0 else 100.0
        print(
            json.dumps(
                {
                    "elapsed_seconds": round(elapsed, 3),
                    "percent": round(percent, 3),
                    "processed_receipts": processed,
                    "receipts_per_second": round(rate, 1),
                    "total_receipts": total,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    manifest, scan = materialize_round25_forensic_joint_feature_store(
        source_database=args.source_db,
        destination_database=args.destination_db,
        forensic_audit=_mapping(args.audit),
        salvage_contract=_mapping(args.contract),
        observed_at_ms=args.observed_at_ms,
        progress=progress,
        progress_interval=args.progress_interval,
    )
    _write_once(args.scan_receipt, scan)
    print(
        json.dumps(
            {
                "admitted_condition_count": manifest["admitted_condition_count"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "feature_row_count": manifest["feature_row_count"],
                "manifest_sha256": manifest["manifest_sha256"],
                "rejected_condition_count": manifest["rejected_condition_count"],
                "scan_sha256": scan["scan_sha256"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
