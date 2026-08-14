"""Publish the target-blind Round 25 forensic diagnostic partition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from simple_ai_trading.polymarket_round25_forensic_partition import (
    build_round25_forensic_partition_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "polymarket"
DEFAULT_AUDIT = (
    RESEARCH / "round-025-v2-transport-failure-forensic-audit-2026-08-14.json"
)
DEFAULT_CONTRACT = RESEARCH / "round-025-v2-condition-salvage-contract-v1.json"
DEFAULT_VENUE_PARAMETERS = (
    RESEARCH / "round-025-v2-forensic-venue-parameters-2026-08-14.json"
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
        raise ValueError("forensic partition destination is not empty")
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
            "Deep-audit a completed forensic feature store and freeze its "
            "chronological diagnostic condition roles."
        )
    )
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--venue-parameters", type=Path, default=DEFAULT_VENUE_PARAMETERS
    )
    parser.add_argument("--observed-at-ms", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_round25_forensic_partition_manifest(
        feature_store=args.feature_store,
        forensic_audit=_mapping(args.audit),
        salvage_contract=_mapping(args.contract),
        venue_parameter_audit=_mapping(args.venue_parameters),
        observed_at_ms=args.observed_at_ms,
    )
    _write_once(args.output, manifest)
    print(
        json.dumps(
            {
                "condition_count": manifest["condition_count"],
                "partition_sha256": manifest["partition_sha256"],
                "role_counts": manifest["role_counts"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
