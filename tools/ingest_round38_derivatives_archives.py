"""Ingest and certify the frozen Round 38 premium-index and funding archives."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.derivatives_archive import (  # noqa: E402
    DerivativesArchiveIngestResult,
    ingest_derivatives_archive_range,
    monthly_periods,
)
from simple_ai_trading.market_store import MarketDataStore  # noqa: E402
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
START_PERIOD = "2021-12"
END_PERIOD = "2025-06"
EXPECTED_PREMIUM_ROWS = 1_883_520
EXPECTED_PREMIUM_QUALITY = {
    "BTCUSDT": (1_880_574, 9, 2_946, 1_440),
    "ETHUSDT": (1_880_577, 9, 2_943, 1_440),
    "SOLUSDT": (1_882_018, 7, 1_502, 1_440),
}
DESIGN_PATH = (
    ROOT
    / "docs/model-research/action-value/round-038-derivatives-hurdle-ai-ablation-design.json"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Round 38 source-ingestion Git check failed") from exc


def _design_sha256() -> str:
    value = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Round 38 design root is not an object")
    canonical = dict(value)
    claimed = str(canonical.pop("design_sha256", ""))
    if (
        value.get("schema_version")
        != "derivatives-hurdle-ai-ablation-design-v2"
        or value.get("round") != 38
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 38 design identity is invalid")
    return claimed


def _archive_evidence(store: MarketDataStore) -> list[dict[str, object]]:
    periods = monthly_periods(START_PERIOD, END_PERIOD)
    output: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for data_type in ("premiumIndexKlines", "fundingRate"):
            records = store.derivatives_archive_files(
                symbol=symbol, data_type=data_type
            )
            selected = [item for item in records if item.period in periods]
            observed_periods = [item.period for item in selected]
            if (
                observed_periods != periods
                or len(selected) != 43
                or any(item.status != "complete" for item in selected)
                or any(item.checksum_status != "verified" for item in selected)
                or any(len(item.sha256) != 64 for item in selected)
                or any(len(item.row_stream_sha256) != 64 for item in selected)
            ):
                raise ValueError(
                    f"incomplete derivatives archive evidence: {symbol} {data_type}"
                )
            output.append(
                {
                    "symbol": symbol,
                    "data_type": data_type,
                    "period_count": len(selected),
                    "first_period": selected[0].period,
                    "last_period": selected[-1].period,
                    "rows_read": sum(item.rows_read for item in selected),
                    "bytes_downloaded": sum(
                        item.bytes_downloaded for item in selected
                    ),
                    "archive_identity_sha256": _canonical_sha256(
                        [
                            {
                                "url": item.url,
                                "period": item.period,
                                "rows_read": item.rows_read,
                                "sha256": item.sha256,
                                "checksum_sha256": item.checksum_sha256,
                                "row_stream_sha256": item.row_stream_sha256,
                            }
                            for item in selected
                        ]
                    ),
                }
            )
    return output


def _series_evidence(store: MarketDataStore) -> list[dict[str, object]]:
    conn = store.connect()
    output: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        premium = conn.execute(
            """
            SELECT COUNT(*) AS rows, MIN(open_time) AS first_time,
                   MAX(open_time) AS last_time
            FROM futures_reference_bars
            WHERE symbol=? AND market_type='futures'
              AND kind='premium_index' AND interval='1m'
              AND open_time BETWEEN 1638316800000 AND 1751327940000
            """,
            (symbol,),
        ).fetchone()
        gaps = conn.execute(
            """
            WITH ordered AS (
                SELECT open_time,
                       LAG(open_time) OVER (ORDER BY open_time) AS previous_time
                FROM futures_reference_bars
                WHERE symbol=? AND market_type='futures'
                  AND kind='premium_index' AND interval='1m'
                  AND open_time BETWEEN 1638316800000 AND 1751327940000
            )
            SELECT COUNT(*) AS gap_count,
                   COALESCE(SUM((open_time - previous_time) / 60000 - 1), 0)
                       AS missing_minutes,
                   COALESCE(MAX((open_time - previous_time) / 60000 - 1), 0)
                       AS maximum_gap_minutes
            FROM ordered
            WHERE previous_time IS NOT NULL AND open_time - previous_time != 60000
            """,
            (symbol,),
        ).fetchone()
        funding = conn.execute(
            """
            SELECT COUNT(*) AS rows, MIN(calc_time) AS first_time,
                   MAX(calc_time) AS last_time,
                   MIN(funding_interval_hours) AS minimum_interval_hours,
                   MAX(funding_interval_hours) AS maximum_interval_hours,
                   SUM(CASE WHEN funding_rate IS NULL THEN 1 ELSE 0 END) AS null_rates
            FROM funding_rates
            WHERE symbol=? AND market_type='futures'
              AND calc_time BETWEEN 1638316800000 AND 1751327999999
            """,
            (symbol,),
        ).fetchone()
        expected_quality = EXPECTED_PREMIUM_QUALITY[symbol]
        observed_quality = (
            int(premium["rows"]),
            int(gaps["gap_count"]),
            int(gaps["missing_minutes"]),
            int(gaps["maximum_gap_minutes"]),
        )
        if (
            observed_quality != expected_quality
            or int(premium["first_time"]) != 1_638_316_800_000
            or int(premium["last_time"]) != 1_751_327_940_000
            or int(funding["rows"]) <= 0
            or int(funding["null_rates"]) != 0
            or int(funding["minimum_interval_hours"]) < 1
            or int(funding["maximum_interval_hours"]) > 8
        ):
            raise ValueError(f"derivatives series audit failed: {symbol}")
        output.append(
            {
                "symbol": symbol,
                "premium_rows": int(premium["rows"]),
                "premium_expected_grid_rows": EXPECTED_PREMIUM_ROWS,
                "premium_first_open_time_ms": int(premium["first_time"]),
                "premium_last_open_time_ms": int(premium["last_time"]),
                "premium_gap_count": int(gaps["gap_count"]),
                "premium_missing_minutes": int(gaps["missing_minutes"]),
                "premium_missing_fraction": (
                    int(gaps["missing_minutes"]) / EXPECTED_PREMIUM_ROWS
                ),
                "premium_maximum_gap_minutes": int(
                    gaps["maximum_gap_minutes"]
                ),
                "funding_rows": int(funding["rows"]),
                "funding_first_calc_time_ms": int(funding["first_time"]),
                "funding_last_calc_time_ms": int(funding["last_time"]),
                "funding_minimum_interval_hours": int(
                    funding["minimum_interval_hours"]
                ),
                "funding_maximum_interval_hours": int(
                    funding["maximum_interval_hours"]
                ),
            }
        )
    return output


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if _git("status", "--porcelain"):
        raise ValueError("Round 38 source ingestion requires a clean worktree")
    implementation_commit = _git("rev-parse", "HEAD")
    design_sha256 = _design_sha256()
    started = time.perf_counter()
    database = arguments.database.resolve()
    database_size_before = database.stat().st_size if database.exists() else 0
    completed = 0

    def progress(result: DerivativesArchiveIngestResult) -> None:
        nonlocal completed
        completed += 1
        print(
            json.dumps(
                {
                    "completed": completed,
                    "total": len(SYMBOLS) * 2 * 43,
                    "symbol": result.symbol,
                    "data_type": result.data_type,
                    "period": result.period,
                    "status": result.status,
                    "rows_read": result.rows_read,
                    "bytes_downloaded": result.bytes_downloaded,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )

    results = ingest_derivatives_archive_range(
        db_path=database,
        symbols=SYMBOLS,
        start_period=START_PERIOD,
        end_period=END_PERIOD,
        timeout=arguments.timeout,
        force=arguments.force,
        progress=progress,
    )
    if len(results) != len(SYMBOLS) * 2 * 43:
        raise ValueError("Round 38 derivatives result count is incomplete")
    with MarketDataStore(database) as store:
        archives = _archive_evidence(store)
        series = _series_evidence(store)
    payload: dict[str, object] = {
        "schema_version": "round-038-derivatives-source-certificate-v1",
        "round": 38,
        "design_sha256": design_sha256,
        "implementation_commit": implementation_commit,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "database_path": str(database),
        "database_bytes_before": database_size_before,
        "database_bytes_after": database.stat().st_size,
        "persistent_zip_archive_created": False,
        "symbols": list(SYMBOLS),
        "start_period": START_PERIOD,
        "end_period": END_PERIOD,
        "archive_evidence": archives,
        "series_evidence": series,
        "elapsed_seconds": time.perf_counter() - started,
        "source_certificate_sha256": "PENDING",
    }
    canonical = dict(payload)
    canonical.pop("source_certificate_sha256")
    payload["source_certificate_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(
        arguments.output.resolve(), payload, indent=2, sort_keys=True
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(_parser().parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
