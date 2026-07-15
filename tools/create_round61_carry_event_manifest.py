"""Create Round 61's price-blind carry event and archive manifest."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.binance_archive import archive_file_url  # noqa: E402
from simple_ai_trading.derivatives_archive import (  # noqa: E402
    derivatives_archive_file_url,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_round59_funding_persistence_feasibility import (  # noqa: E402
    SYMBOLS,
    _canonical_sha256,
    _connect_read_only,
    _episodes,
    _file_sha256,
    _load_verified_funding,
    _read_object,
)
from tools.run_round60_full_history_funding_replication import (  # noqa: E402
    REPORT_SCHEMA as ROUND60_REPORT_SCHEMA,
    _validate_certificate,
    _validate_design,
)


ROUND60_REPORT_FILE_SHA256 = (
    "0bf9dc26b6bc53a9bfebedba9f6ae43cca879f14ced85bc7b4b64de70f388b5d"
)
ROUND60_REPORT_CANONICAL_SHA256 = (
    "a020bd2f26280b82705ffa4bda83b37d439dfa09377eda64ffdfcbf17c9e9ba4"
)
MANIFEST_SCHEMA = "round-061-carry-event-manifest-v1"


def _period(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime("%Y-%m")


def _validate_report(path: Path) -> dict[str, object]:
    if _file_sha256(path) != ROUND60_REPORT_FILE_SHA256:
        raise ValueError("Round 60 report file hash drifted")
    report = _read_object(path, "Round 60 report")
    canonical = dict(report)
    claimed = str(canonical.pop("report_sha256", ""))
    if (
        report.get("schema_version") != ROUND60_REPORT_SCHEMA
        or report.get("status")
        != "full_history_support_passed_spot_ingestion_authorized"
        or report.get("spot_history_ingestion_authorized") is not True
        or claimed != ROUND60_REPORT_CANONICAL_SHA256
        or _canonical_sha256(canonical) != claimed
    ):
        raise ValueError("Round 60 report did not authorize the event manifest")
    return report


def create(arguments: argparse.Namespace) -> dict[str, object]:
    design, protocol = _validate_design(arguments.design.resolve())
    certificate, certificate_file_sha, certificate_sha = _validate_certificate(
        arguments.certificate.resolve(), design
    )
    report = _validate_report(arguments.report.resolve())
    ranges = design["source_contract"]["ranges_by_symbol"]
    archive_by_symbol = {row["symbol"]: row for row in certificate["archive_evidence"]}
    symbol_manifests: list[dict[str, object]] = []
    with _connect_read_only(arguments.database.resolve()) as connection:
        for symbol in SYMBOLS:
            contract = ranges[symbol]
            source_view = {
                "source_contract": {
                    "start_period": contract["start_period"],
                    "end_period": contract["end_period"],
                    "periods_per_symbol": contract["period_count"],
                    "expected_rows": {symbol: archive_by_symbol[symbol]["rows_read"]},
                }
            }
            rows, source_evidence = _load_verified_funding(
                connection,
                symbol=symbol,
                design=source_view,
                certificate=certificate,
            )
            episodes = _episodes(
                rows,
                trigger={"operator": "greater_or_equal", "value": 2.0},
                horizon_hours=168,
            )
            times = [int(row["calc_time"]) for row in rows]
            output_episodes: list[dict[str, object]] = []
            spot_times: set[int] = set()
            mark_times: set[int] = set()
            for episode in episodes:
                decision_ms = int(episode["decision_time_ms"])
                end_ms = int(episode["end_time_ms"])
                future_start = bisect_right(times, decision_ms)
                future_end = bisect_right(times, end_ms)
                funding_times = times[future_start:future_end]
                episode_id = _canonical_sha256(
                    {
                        "symbol": symbol,
                        "decision_time_ms": decision_ms,
                        "end_time_ms": end_ms,
                    }
                )[:20]
                output_episodes.append(
                    {
                        "episode_id": episode_id,
                        "decision_time_ms": decision_ms,
                        "end_time_ms": end_ms,
                        "current_funding_bps": episode["current_funding_bps"],
                        "future_funding_calc_times_ms": funding_times,
                    }
                )
                spot_times.update((decision_ms, end_ms))
                mark_times.update(funding_times)
            spot_months = sorted({_period(value) for value in spot_times})
            mark_months = sorted({_period(value) for value in mark_times})
            symbol_manifest: dict[str, object] = {
                "symbol": symbol,
                "source_funding_rows": source_evidence["rows"],
                "episodes": output_episodes,
                "episode_count": len(output_episodes),
                "required_spot_open_times_ms": sorted(spot_times),
                "required_futures_execution_open_times_ms": sorted(spot_times),
                "required_mark_open_times_ms": sorted(mark_times),
                "spot_archive_months": spot_months,
                "mark_archive_months": mark_months,
                "spot_archive_urls": [
                    archive_file_url(
                        symbol=symbol,
                        interval="1m",
                        period=period,
                        market_type="spot",
                        cadence="monthly",
                        data_type="klines",
                    )
                    for period in spot_months
                ],
                "futures_execution_archive_urls": [
                    archive_file_url(
                        symbol=symbol,
                        interval="1m",
                        period=period,
                        market_type="futures",
                        cadence="monthly",
                        data_type="klines",
                    )
                    for period in spot_months
                ],
                "mark_archive_urls": [
                    derivatives_archive_file_url(
                        symbol=symbol,
                        data_type="markPriceKlines",
                        period=period,
                        interval="1m",
                    )
                    for period in mark_months
                ],
            }
            symbol_manifest["symbol_manifest_sha256"] = _canonical_sha256(
                symbol_manifest
            )
            symbol_manifests.append(symbol_manifest)
    expected_counts = {
        result["symbol"]: next(
            cell["episodes"]
            for cell in result["cells"]
            if cell["trigger_id"] == "at_least_2bp"
            and int(cell["horizon_hours"]) == 168
        )
        for result in report["symbol_results"]
    }
    if {
        row["symbol"]: row["episode_count"] for row in symbol_manifests
    } != expected_counts:
        raise ValueError("Round 61 episode inventory differs from Round 60")
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "round": 61,
        "source_round": 60,
        "source_report_file_sha256": ROUND60_REPORT_FILE_SHA256,
        "source_report_canonical_sha256": ROUND60_REPORT_CANONICAL_SHA256,
        "source_design_sha256": design["design_sha256"],
        "source_certificate_file_sha256": certificate_file_sha,
        "source_certificate_canonical_sha256": certificate_sha,
        "price_values_read": False,
        "trigger": {
            "operator": "greater_or_equal",
            "value_bps": 2.0,
            "decision_time": "immediately after the settled funding observation",
        },
        "horizon_hours": 168,
        "symbols": list(SYMBOLS),
        "symbol_manifests": symbol_manifests,
        "manifest_sha256": "PENDING",
    }
    canonical = dict(manifest)
    canonical.pop("manifest_sha256")
    manifest["manifest_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(arguments.output.resolve(), manifest, indent=2)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT
        / "docs/model-research/action-value/round-060-full-history-funding-replication-design.json",
    )
    parser.add_argument(
        "--certificate",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round60-full-funding-source-20260715-v1\certificate.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\evidence\round60-full-history-funding-replication-20260715-v1.json"
        ),
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "docs/model-research/action-value/round-061-carry-event-manifest.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    manifest = create(_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "round": manifest["round"],
                "manifest_sha256": manifest["manifest_sha256"],
                "episode_counts": {
                    row["symbol"]: row["episode_count"]
                    for row in manifest["symbol_manifests"]
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
