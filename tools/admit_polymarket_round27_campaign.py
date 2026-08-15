#!/usr/bin/env python3
"""Freeze the target-free Round 27 campaign admission gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

from simple_ai_trading.polymarket_round27_campaign_admission import (
    build_round27_campaign_admission,
    load_round27_campaign_admission,
)
from simple_ai_trading.polymarket_round27_feature_store import Round27FeatureStore
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.storage import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _progress(phase: str, detail: Mapping[str, object]) -> None:
    print(
        json.dumps(
            {"at_ms": time.time_ns() // 1_000_000, "phase": phase, **detail},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve(strict=True)
    feature_database = _resolve(repository, arguments.feature_store).resolve(
        strict=True
    )
    output = _resolve(repository, arguments.output).resolve()
    if feature_database.is_symlink() or Path(f"{feature_database}.wal").exists():
        raise ValueError("Round 27 admission requires a terminal feature store")
    contract = load_round27_model_contract(repository)
    with Round27FeatureStore(feature_database, read_only=True) as store:
        feature_audit = store.audit()
        feature_rows = store.load_rows()
    if output.exists():
        persisted = load_round27_campaign_admission(
            output,
            contract=contract,
            feature_store_audit_sha256=str(feature_audit["audit_sha256"]),
        )
        admission = build_round27_campaign_admission(
            contract=contract,
            feature_store_audit=feature_audit,
            feature_rows=feature_rows,
            admitted_at_ms=int(persisted["admitted_at_ms"]),
        )
        if persisted != admission:
            raise ValueError("Round 27 campaign was already admitted differently")
        created = False
    else:
        admission = build_round27_campaign_admission(
            contract=contract,
            feature_store_audit=feature_audit,
            feature_rows=feature_rows,
            admitted_at_ms=time.time_ns() // 1_000_000,
        )
        write_json_atomic(output, admission, indent=2, sort_keys=True)
        persisted = load_round27_campaign_admission(
            output,
            contract=contract,
            feature_store_audit_sha256=str(feature_audit["audit_sha256"]),
        )
        if persisted != admission:
            raise ValueError("Round 27 persisted campaign admission differs")
        created = True
    _progress(
        "campaign-admitted",
        {
            "created": created,
            "admission_sha256": admission["admission_sha256"],
            "eligible_condition_count": admission["eligible_condition_count"],
            "role_condition_counts": admission["role_condition_counts"],
            "target_access_authorized": True,
            "orders_submitted": False,
            "trading_authority": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
