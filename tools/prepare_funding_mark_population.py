"""Freeze archive-bound native mark requests before opening any new responses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3

from simple_ai_trading.derivatives_archive import (
    _canonical_row_digest_update,
    monthly_periods,
)
from simple_ai_trading.funding_cash import (
    MAX_HISTORY_ROWS,
    USDM_SYMBOLS,
    parse_binance_usdm_funding_history,
)
from tools.capture_public_source_contract import _canonical_hash, _root_path


ROOT = Path(__file__).resolve().parents[1]


def _write_new(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _month_bounds(period: str) -> tuple[int, int]:
    start = datetime.fromisoformat(period + "-01").replace(tzinfo=timezone.utc)
    year, month = start.year + start.month // 12, start.month % 12 + 1
    end = datetime(year, month, 1, tzinfo=timezone.utc)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000) - 1


def _archive_rows(
    database: Path, periods: list[str], symbol: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Verify each complete monthly warehouse stream against registered archive lineage."""
    rows: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=2) as db:
        db.execute("PRAGMA query_only=ON")
        for period in periods:
            metadata = db.execute(
                """SELECT url,status,rows_read,sha256,checksum_sha256,checksum_status,row_stream_sha256
                FROM derivatives_archive_files WHERE symbol=? AND market_type='futures'
                AND data_type='fundingRate' AND period=?""",
                (symbol, period),
            ).fetchall()
            if len(metadata) != 1:
                raise ValueError("monthly archive metadata is absent or ambiguous")
            url, status, count, sha, checksum, checksum_status, stream_sha = metadata[0]
            if (
                status != "complete"
                or checksum_status != "verified"
                or sha != checksum
                or len(sha) != 64
            ):
                raise ValueError("monthly archive lineage is unqualified")
            start, end = _month_bounds(period)
            selected = db.execute(
                """SELECT calc_time,funding_interval_hours,funding_rate FROM funding_rates
                WHERE symbol=? AND market_type='futures' AND calc_time BETWEEN ? AND ?
                ORDER BY calc_time""",
                (symbol, start, end),
            ).fetchall()
            digest = hashlib.sha256()
            previous_time: int | None = None
            for row in selected:
                timestamp, interval, rate = row
                if (
                    type(timestamp) is not int
                    or (previous_time is not None and timestamp <= previous_time)
                    or type(interval) is not int
                    or not 1 <= interval <= 8
                    or not isinstance(rate, float)
                    or not math.isfinite(rate)
                ):
                    raise ValueError("monthly warehouse rows are invalid")
                previous_time = timestamp
                _canonical_row_digest_update(digest, row)
            if (
                not selected
                or len(selected) != count
                or digest.hexdigest() != stream_sha
            ):
                raise ValueError(
                    "monthly warehouse stream does not match archive metadata"
                )
            rows.extend(
                {
                    "calc_time": t,
                    "interval_hours": interval,
                    "funding_rate_float_repr": repr(rate),
                }
                for t, interval, rate in selected
            )
            evidence.append(
                {
                    "period": period,
                    "url": url,
                    "rows": count,
                    "archive_sha256": sha,
                    "row_stream_sha256": stream_sha,
                }
            )
    return rows, evidence


def prepare(
    database: Path,
    output: Path,
    start_period: str,
    end_period: str,
    retained: Path,
    max_requests: int,
) -> dict[str, object]:
    """Freeze complete request/reference files; do not capture or select financial outcomes."""
    output, retained = _root_path(output.as_posix()), _root_path(retained.as_posix())
    database = database.resolve()
    database.relative_to(ROOT)
    if output.exists() or max_requests <= 0:
        raise ValueError("preparation directory exists or request budget is invalid")
    periods = monthly_periods(start_period, end_period)
    old_contract = json.loads((retained / "contract.json").read_bytes())
    old_source = json.loads((retained / "native-result.json").read_bytes())
    old_comparison = json.loads((retained / "comparison.json").read_bytes())
    for payload, field in (
        (old_contract, "contract_sha256"),
        (old_source, "result_sha256"),
        (old_comparison, "result_sha256"),
    ):
        if _canonical_hash(payload, field) != payload[field]:
            raise ValueError("retained source self hash differs")
    if (
        not old_source["source_gate"]["passed"]
        or not old_comparison["comparison"]["exact_timestamp_and_rate_join"]
    ):
        raise ValueError("retained native clock comparison is unqualified")
    if (
        old_source["contract"]["sha256"] != old_contract["contract_sha256"]
        or old_comparison["native_result_sha256"] != old_source["result_sha256"]
    ):
        raise ValueError("retained source bindings differ")
    old_raw = _root_path(old_contract["outputs"]["raw_path"]).read_bytes()
    old_events = parse_binance_usdm_funding_history(
        old_raw,
        expected_sha256=old_source["capture"]["receipt"]["response_sha256"],
        expected_symbol="BTCUSDT",
    )
    retained_keys = {(event.funding_time_ms, event.rate) for event in old_events}
    full: dict[str, list[dict[str, object]]] = {}
    archive_evidence: dict[str, list[dict[str, object]]] = {}
    groups: list[tuple[str, list[dict[str, object]]]] = []
    for symbol in sorted(USDM_SYMBOLS):
        rows, lineage = _archive_rows(database, periods, symbol)
        full[symbol], archive_evidence[symbol] = rows, lineage
        if symbol == "BTCUSDT":
            observed_keys = {
                (row["calc_time"], Fraction(row["funding_rate_float_repr"]))
                for row in rows
            }
            if not retained_keys <= observed_keys:
                raise ValueError(
                    "retained native population lies outside the selected prefix"
                )
            rows = [
                row
                for row in rows
                if (row["calc_time"], Fraction(row["funding_rate_float_repr"]))
                not in retained_keys
            ]
        groups.extend(
            (symbol, rows[i : i + MAX_HISTORY_ROWS])
            for i in range(0, len(rows), MAX_HISTORY_ROWS)
        )
    if len(groups) > max_requests:
        raise ValueError("complete population exceeds the frozen request budget")
    # Complete validation precedes all output creation and all future network access.
    output.mkdir(parents=True, exist_ok=False)
    frozen = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    inventory = {
        "schema_version": "funding-mark-archive-population-v1",
        "start_period": start_period,
        "end_period": end_period,
        "rows": full,
        "archive_evidence": archive_evidence,
        "retained_source_contract_sha256": old_contract["contract_sha256"],
        "retained_source_result_sha256": old_source["result_sha256"],
        "retained_comparison_result_sha256": old_comparison["result_sha256"],
        "native_decimal_mark_or_entitlement_qualification": False,
    }
    inventory["population_sha256"] = _canonical_hash(inventory, "population_sha256")
    _write_new(output / "archive-population.json", inventory)
    contracts: list[str] = []
    for index, (symbol, rows) in enumerate(groups, 1):
        stem = f"{index:02d}-{symbol.lower()}"
        relative = output.relative_to(ROOT).as_posix()
        reference = {
            "schema_version": "funding-archive-clock-reference-v1",
            "symbol": symbol,
            "start_time_ms": rows[0]["calc_time"],
            "end_time_ms": rows[-1]["calc_time"],
            "rows": rows,
            "complete_population_sha256": inventory["population_sha256"],
            "snapshot_only_not_native_decimal_or_mark_proof": True,
        }
        reference["reference_sha256"] = _canonical_hash(reference, "reference_sha256")
        _write_new(output / (stem + "-reference.json"), reference)
        plan = {
            k: v
            for k, v in old_contract.items()
            if k
            not in (
                "contract_sha256",
                "archive_reference",
                "adjudication_output",
                "adjudication_journal",
                "selection",
                "parser_preflight",
            )
        }
        plan.update(
            {
                "frozen_at_utc": frozen,
                "contract_path": relative + "/" + stem + "-contract.json",
                "request_name": "native_funding_mark_population_" + stem,
                "request": {
                    "method": "GET",
                    "url": f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime={rows[0]['calc_time']}&endTime={rows[-1]['calc_time']}&limit=1000",
                    "count": 1,
                    "body_sha256": old_contract["request"]["body_sha256"],
                },
                "required_utf8_phrases": [
                    "fundingTime",
                    "fundingRate",
                    "markPrice",
                    symbol,
                ],
                "outputs": {
                    "raw_path": relative + "/" + stem + ".raw",
                    "journal_path": relative + "/" + stem + "-requests.jsonl",
                    "result_path": relative + "/" + stem + "-result.json",
                },
                "archive_reference": {
                    "path": relative + "/" + stem + "-reference.json",
                    "reference_sha256": reference["reference_sha256"],
                },
                "adjudication_output": relative + "/" + stem + "-comparison.json",
                "adjudication_journal": relative
                + "/"
                + stem
                + "-comparison-journal.jsonl",
                "question": "Require exact complete archive/native timestamp-rate joins and positive native settlement marks before cash-label integration; not old outcome validation or profit selection",
                "not_authorized": [
                    "old acceptance rerun",
                    "model fit",
                    "prediction or edge promotion",
                    "fees/books/account requests",
                    "credentials",
                    "orders",
                    "protected capture access",
                    "history outside the frozen population",
                ],
                "population_sha256": inventory["population_sha256"],
                "batch_stop": "Stop all remaining requests on any capture/schema/exact-join failure. No rate filtering, timestamp tolerance, missing-mark repair, refetch, alias, adaptive page or model fit.",
                "retained_population_excluded_from_new_requests": symbol == "BTCUSDT",
                "preparation_implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )
        plan["contract_sha256"] = _canonical_hash(plan, "contract_sha256")
        _write_new(output / (stem + "-contract.json"), plan)
        contracts.append(plan["contract_path"])
    batch = {
        "schema_version": "funding-mark-batch-contract-v1",
        "frozen_at_utc": frozen,
        "population_sha256": inventory["population_sha256"],
        "contracts_in_fixed_order": contracts,
        "new_request_count": len(groups),
        "max_new_requests": max_requests,
        "maximum_new_response_bytes": len(groups)
        * old_contract["response_byte_ceiling"],
        "retained_native_events_reused": len(old_events),
        "expected_events_by_symbol": {
            symbol: len(rows) for symbol, rows in full.items()
        },
        "all_or_nothing_population_admission": True,
        "stop_on_first_failure": True,
        "label_admission_qualified": False,
        "accepted_edge": False,
        "profitability_claim": False,
    }
    batch["batch_sha256"] = _canonical_hash(batch, "batch_sha256")
    _write_new(output / "batch-contract.json", batch)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-period", required=True)
    parser.add_argument("--end-period", required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--max-new-requests", type=int, required=True)
    args = parser.parse_args()
    batch = prepare(
        args.database,
        args.output,
        args.start_period,
        args.end_period,
        args.retained,
        args.max_new_requests,
    )
    print(
        json.dumps(
            {
                "new_requests_frozen": batch["new_request_count"],
                "expected_events_by_symbol": batch["expected_events_by_symbol"],
                "retained_events_reused": batch["retained_native_events_reused"],
                "requests_performed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
