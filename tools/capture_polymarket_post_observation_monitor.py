"""Capture a bounded public BTC/ETH/SOL oracle-to-CLOB close monitor."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Mapping

from simple_ai_trading.polymarket_recorder import (
    POLYMARKET_RTDS_CHAINLINK_TWAP_60_TOPIC,
    PolymarketPublicRecorder,
)
from simple_ai_trading.storage import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "data/polymarket-post-observation-monitor-v2.duckdb"
DEFAULT_REPORT = ROOT / "data/polymarket-post-observation-monitor-v2-report.json"


def _progress(phase: str, payload: Mapping[str, object]) -> None:
    selected = {
        key: payload.get(key)
        for key in (
            "run_id",
            "elapsed_seconds",
            "duration_seconds",
            "written_message_count",
            "written_gap_count",
            "received_stream_counts",
            "queue_size",
            "error_count",
            "status",
        )
        if key in payload
    }
    print(json.dumps({"phase": phase, **selected}, sort_keys=True), flush=True)


async def _capture(database: Path, report_path: Path, duration: int) -> None:
    if database.exists():
        raise FileExistsError(f"evidence database already exists: {database}")
    if report_path.exists():
        raise FileExistsError(f"report already exists: {report_path}")

    recorder = PolymarketPublicRecorder(
        database,
        assets=("BTC", "ETH", "SOL"),
        include_binance_futures=False,
        include_binance_spot=True,
        include_rtds_binance=True,
        chainlink_price_mode="twap_60s",
        clob_lane_ids=("clob",),
        include_polymarket_core=True,
        memory_limit="1GB",
        database_threads=2,
        queue_capacity=100_000,
        discovery_interval_seconds=30,
        market_subscription_grace_seconds=120,
    )

    def manifest_factory(run_id: str, started_at_ms: int) -> Mapping[str, object]:
        manifest: dict[str, object] = {
            "schema_version": "polymarket-post-observation-monitor-manifest-v1",
            "run_id": run_id,
            "created_at_ms": started_at_ms,
            "capture_duration_seconds": duration,
            "required_assets": ["BTC", "ETH", "SOL"],
            "required_streams": [
                "binance_spot",
                "clob_market",
                "polymarket_rtds",
            ],
            "required_clob_lanes": ["clob"],
            "required_chainlink_topic": POLYMARKET_RTDS_CHAINLINK_TWAP_60_TOPIC,
            "purpose": (
                "Prospectively test cross-asset persistence of public CLOB bid "
                "growth and winner-side fills after local receipt of the exact "
                "closing 60-second Chainlink TWAP."
            ),
            "authority": {
                "credentials_used": False,
                "funds_used": False,
                "orders_submitted": False,
                "paper_trading_authority": False,
                "live_trading_authority": False,
                "edge_claim": False,
                "profitability_claim": False,
            },
        }
        canonical = json.dumps(
            manifest,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
        manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
        return manifest

    report = await recorder.run(
        duration_seconds=duration,
        progress=_progress,
        progress_interval_seconds=30,
        preregistration_manifest_factory=manifest_factory,
    )
    write_json_atomic(report_path, report.asdict(), indent=2, sort_keys=True)
    print(json.dumps(report.asdict(), sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=int, default=660)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    asyncio.run(
        _capture(
            args.database.resolve(),
            args.report.resolve(),
            int(args.duration_seconds),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
