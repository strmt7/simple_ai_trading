"""Ingest and certify Round 60's frozen funding-only archive ranges."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.derivatives_archive import (  # noqa: E402
    derivatives_archive_file_url,
    ingest_derivatives_archive_url,
    monthly_periods,
)
from simple_ai_trading.market_store import MarketDataStore  # noqa: E402
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_round59_funding_persistence_feasibility import (  # noqa: E402
    _archive_identity,
    _canonical_sha256,
    _period_bounds_ms,
    _row_stream_sha256,
)


ROUND = 60
DESIGN_SCHEMA = "round-060-full-history-funding-replication-design-v1"
CERTIFICATE_SCHEMA = "round-060-full-history-funding-source-certificate-v1"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_design(path: Path) -> dict[str, object]:
    design = _read_object(path, "Round 60 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    source = design.get("source_contract", {})
    ranges = source.get("ranges_by_symbol", {})
    expected = {
        "BTCUSDT": ("2020-01", "2026-06", 78),
        "ETHUSDT": ("2020-01", "2026-06", 78),
        "SOLUSDT": ("2020-09", "2026-06", 70),
    }
    observed = {
        symbol: (
            ranges.get(symbol, {}).get("start_period"),
            ranges.get(symbol, {}).get("end_period"),
            ranges.get(symbol, {}).get("period_count"),
        )
        for symbol in SYMBOLS
    }
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or claimed != _canonical_sha256(canonical)
        or tuple(source.get("symbols", ())) != SYMBOLS
        or source.get("data_type") != "fundingRate"
        or source.get("funding_only_ingestion") is not True
        or source.get("persistent_zip_archive_permitted") is not False
        or any(
            source.get(key) is not False
            for key in (
                "price_rows_permitted",
                "premium_index_rows_permitted",
                "spot_rows_permitted",
            )
        )
        or observed != expected
    ):
        raise ValueError("Round 60 frozen source design drifted")
    return design


def _archive_evidence(
    store: MarketDataStore, design: Mapping[str, object]
) -> list[dict[str, object]]:
    connection = store.connect()
    ranges = design["source_contract"]["ranges_by_symbol"]
    evidence: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        contract = ranges[symbol]
        periods = monthly_periods(contract["start_period"], contract["end_period"])
        metadata = connection.execute(
            """
            SELECT url, period, status, rows_read, bytes_downloaded, sha256,
                   checksum_sha256, checksum_status, row_stream_sha256
            FROM derivatives_archive_files
            WHERE symbol=? AND market_type='futures' AND data_type='fundingRate'
              AND interval='' AND period BETWEEN ? AND ?
            ORDER BY period
            """,
            (symbol, periods[0], periods[-1]),
        ).fetchall()
        if (
            [row["period"] for row in metadata] != periods
            or len(metadata) != int(contract["period_count"])
            or any(row["status"] != "complete" for row in metadata)
            or any(row["checksum_status"] != "verified" for row in metadata)
            or any(row["sha256"] != row["checksum_sha256"] for row in metadata)
            or any(len(str(row["sha256"])) != 64 for row in metadata)
            or any(len(str(row["row_stream_sha256"])) != 64 for row in metadata)
        ):
            raise ValueError(f"{symbol} funding archive metadata is incomplete")

        all_rows = []
        period_evidence: list[dict[str, object]] = []
        for item in metadata:
            start_ms, end_ms = _period_bounds_ms(str(item["period"]))
            rows = connection.execute(
                """
                SELECT calc_time, funding_interval_hours, funding_rate
                FROM funding_rates
                WHERE symbol=? AND market_type='futures'
                  AND calc_time>=? AND calc_time<?
                ORDER BY calc_time
                """,
                (symbol, start_ms, end_ms),
            ).fetchall()
            digest = _row_stream_sha256(rows)
            if (
                not rows
                or len(rows) != int(item["rows_read"])
                or digest != item["row_stream_sha256"]
                or any(
                    not 1 <= int(row["funding_interval_hours"]) <= 8
                    or not math.isfinite(float(row["funding_rate"]))
                    or abs(float(row["funding_rate"])) > 0.1
                    for row in rows
                )
            ):
                raise ValueError(f"{symbol} {item['period']} funding rows drifted")
            all_rows.extend(rows)
            period_evidence.append(
                {
                    "period": item["period"],
                    "rows": len(rows),
                    "row_stream_sha256": digest,
                }
            )
        times = [int(row["calc_time"]) for row in all_rows]
        if any(current <= previous for previous, current in zip(times, times[1:])):
            raise ValueError(f"{symbol} funding timestamps are not strictly increasing")
        evidence.append(
            {
                "symbol": symbol,
                "data_type": "fundingRate",
                "period_count": len(metadata),
                "first_period": periods[0],
                "last_period": periods[-1],
                "rows_read": len(all_rows),
                "bytes_downloaded": sum(
                    int(row["bytes_downloaded"]) for row in metadata
                ),
                "first_calc_time_ms": times[0],
                "last_calc_time_ms": times[-1],
                "minimum_interval_hours": min(
                    int(row["funding_interval_hours"]) for row in all_rows
                ),
                "maximum_interval_hours": max(
                    int(row["funding_interval_hours"]) for row in all_rows
                ),
                "archive_identity_sha256": _archive_identity(metadata),
                "funding_row_stream_sha256": _row_stream_sha256(all_rows),
                "period_evidence_sha256": _canonical_sha256(period_evidence),
            }
        )
    return evidence


def run(arguments: argparse.Namespace) -> dict[str, object]:
    design = _validate_design(arguments.design.resolve())
    if _git("status", "--porcelain"):
        raise ValueError("Round 60 source ingestion requires a clean worktree")
    implementation_commit = _git("rev-parse", "HEAD")
    ranges = design["source_contract"]["ranges_by_symbol"]
    total = sum(int(ranges[symbol]["period_count"]) for symbol in SYMBOLS)
    completed = 0
    with MarketDataStore(arguments.database.resolve()) as store:
        for symbol in SYMBOLS:
            contract = ranges[symbol]
            periods = monthly_periods(contract["start_period"], contract["end_period"])
            for period in periods:
                url = derivatives_archive_file_url(
                    symbol=symbol,
                    data_type="fundingRate",
                    period=period,
                    interval="",
                )
                if (
                    not arguments.force
                    and store.derivatives_archive_file_status(url) == "complete"
                ):
                    status = "skipped"
                    rows_read = 0
                    bytes_downloaded = 0
                else:
                    result = ingest_derivatives_archive_url(
                        store,
                        url=url,
                        symbol=symbol,
                        data_type="fundingRate",
                        period=period,
                        interval="",
                        timeout=arguments.timeout,
                        force=arguments.force,
                    )
                    if result.status != "complete":
                        raise ValueError(
                            f"funding archive ingestion failed: {symbol} {period}: {result.error}"
                        )
                    status = result.status
                    rows_read = result.rows_read
                    bytes_downloaded = result.bytes_downloaded
                completed += 1
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": total,
                            "symbol": symbol,
                            "period": period,
                            "status": status,
                            "rows_read": rows_read,
                            "bytes_downloaded": bytes_downloaded,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    flush=True,
                )
        archives = _archive_evidence(store, design)
    payload: dict[str, object] = {
        "schema_version": CERTIFICATE_SCHEMA,
        "round": ROUND,
        "design_sha256": design["design_sha256"],
        "implementation_commit": implementation_commit,
        "database_file": arguments.database.resolve().name,
        "persistent_zip_archive_created": False,
        "symbols": list(SYMBOLS),
        "archive_evidence": archives,
        "source_certificate_sha256": "PENDING",
    }
    canonical = dict(payload)
    canonical.pop("source_certificate_sha256")
    payload["source_certificate_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(arguments.output.resolve(), payload, indent=2, sort_keys=True)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT
        / "docs/model-research/action-value/round-060-full-history-funding-replication-design.json",
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "round": payload["round"],
                "source_certificate_sha256": payload["source_certificate_sha256"],
                "rows": {
                    row["symbol"]: row["rows_read"]
                    for row in payload["archive_evidence"]
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
