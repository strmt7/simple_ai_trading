"""Monitored target-blind Round 27 feature replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Mapping

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_round27_features import (
    load_round27_public_source_series,
    materialize_round27_target_blind_features,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--maximum-conditions", type=int, default=0)
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--threads", type=int, default=2)
    return parser


def _progress(phase: str, detail: Mapping[str, object]) -> None:
    print(
        json.dumps(
            {
                "at_ms": time.time_ns() // 1_000_000,
                "phase": phase,
                **detail,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _load_audit(path: Path, maximum: int) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError("Round 27 feature audit must be an object")
    if maximum:
        if not 1 <= maximum <= 10_000:
            raise ValueError("maximum conditions must lie in [1, 10000]")
        conditions = value.get("conditions")
        if not isinstance(conditions, list):
            raise ValueError("Round 27 feature audit conditions differ")
        eligible = [
            item
            for item in conditions
            if isinstance(item, Mapping) and item.get("eligible") is True
        ][:maximum]
        value["conditions"] = eligible
        value["condition_count"] = len(eligible)
        value["eligible_condition_count"] = len(eligible)
        value["failed_condition_count"] = 0
        value["eligible_condition_ids"] = [item["condition_id"] for item in eligible]
        value["failed_condition_ids"] = []
    return value


def main() -> int:
    args = _parser().parse_args()
    database = args.database.resolve(strict=True)
    if (
        database.is_symlink()
        or not database.is_file()
        or Path(f"{database}.wal").exists()
    ):
        raise ValueError(
            "Round 27 feature replay requires a terminal WAL-free database"
        )
    with PolymarketEvidenceStore(
        database,
        read_only=True,
        memory_limit=args.memory_limit,
        threads=args.threads,
    ) as store:
        row = (
            store.connect()
            .execute(
                "SELECT run_id FROM polymarket_recorder_run "
                "WHERE status IN ('complete', 'degraded') "
                "ORDER BY ended_at_ms DESC, run_id DESC LIMIT 1"
            )
            .fetchone()
        )
        if row is None:
            raise ValueError("Round 27 feature replay has no terminal run")
        run_id = str(row[0])
        if args.audit is None:
            source = load_round27_public_source_series(
                store,
                run_id=run_id,
                progress=_progress,
            )
            compact_bytes = sum(
                value.nbytes
                for series in (source.spot, source.usdm)
                for value in (
                    series.received_wall_ms,
                    series.received_monotonic_ns,
                    series.price,
                    series.quantity,
                    series.buyer_is_maker,
                )
            )
            _progress(
                "source-smoke-complete",
                {
                    "run_id": run_id,
                    "spot_trade_count": source.spot.count,
                    "usdm_trade_count": source.usdm.count,
                    "twap_tick_count": len(source.twap),
                    "compact_trade_bytes": compact_bytes,
                    "source_chain_sha256": source.source_chain_sha256,
                },
            )
            return 0
        audit = _load_audit(args.audit.resolve(strict=True), args.maximum_conditions)
        rows, report = materialize_round27_target_blind_features(
            store,
            run_id=run_id,
            condition_audit=audit,
            progress=_progress,
        )
        _progress(
            "feature-smoke-complete",
            {
                "run_id": run_id,
                "feature_row_count": len(rows),
                "report_sha256": report["report_sha256"],
                "admitted_condition_count": report["admitted_condition_count"],
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
