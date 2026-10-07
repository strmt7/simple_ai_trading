"""Compare archived calculation times to native funding events without time repair."""

from __future__ import annotations

import argparse
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

from simple_ai_trading.funding_cash import (
    USDM_SYMBOLS,
    LinearFundingSettlement,
    parse_binance_usdm_funding_history,
)
from tools.capture_public_source_contract import (
    _canonical_hash,
    _root_path,
    _validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def compare_clock(
    reference: dict[str, object], events: tuple[LinearFundingSettlement, ...]
) -> dict[str, object]:
    """Separate exact joins from same-order clock diagnostics; admit no near matches."""
    rows = reference["rows"]
    symbol = reference["symbol"]
    if symbol not in USDM_SYMBOLS:
        raise ValueError("unsupported archive reference symbol")
    if not isinstance(rows, list) or not rows or not events:
        raise ValueError("nonempty archive and native populations required")
    start, end = reference["start_time_ms"], reference["end_time_ms"]
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        raise ValueError("reference UTC bounds are invalid")
    archive: list[tuple[int, Fraction]] = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"calc_time", "funding_rate_float_repr", "interval_hours"}
            or type(row["calc_time"]) is not int
            or not start <= row["calc_time"] <= end
            or type(row["interval_hours"]) is not int
            or not 1 <= row["interval_hours"] <= 8
            or not isinstance(row["funding_rate_float_repr"], str)
        ):
            raise ValueError("archive reference row is invalid")
        timestamp = row["calc_time"]
        if archive and timestamp <= archive[-1][0]:
            raise ValueError("archive reference times are not strictly increasing")
        archive.append((timestamp, Fraction(row["funding_rate_float_repr"])))
    if any(
        event.symbol != symbol or not start <= event.funding_time_ms <= end
        for event in events
    ):
        raise ValueError("native event symbol or UTC bounds differ")
    native = [(event.funding_time_ms, event.rate) for event in events]
    if any(right[0] <= left[0] for left, right in zip(native, native[1:])):
        raise ValueError("native event times are not strictly increasing")
    same_count = len(archive) == len(native)
    exact_join = archive == native
    ordinal_rate_match = same_count and all(
        left[1] == right[1] for left, right in zip(archive, native, strict=True)
    )
    diagnostic = []
    if ordinal_rate_match:
        diagnostic = [
            {
                "archive_calc_time_ms": old[0],
                "native_funding_time_ms": new[0],
                "archive_minus_native_ms": old[0] - new[0],
                "exact_time_and_rate_match": old == new,
                "diagnostic_only_not_an_admitted_mapping": not exact_join,
            }
            for old, new in zip(archive, native, strict=True)
        ]
    return {
        "archive_events": len(archive),
        "native_events": len(native),
        "same_population_count": same_count,
        "exact_timestamp_and_rate_join": exact_join,
        "same_order_rate_equality_diagnostic": ordinal_rate_match,
        "clock_diagnostic": diagnostic,
        "warehouse_float_rates_are_native_decimal_proof": False,
        "complete_historical_mark_coverage": False,
        "own_payment_entitlement_qualified": False,
        "time_rounding_interpolation_or_candle_fallback": False,
    }


def adjudicate(contract_path: Path) -> dict[str, object]:
    """Adjudicate one consumed native response from its frozen archive reference."""
    contract_path = contract_path.resolve()
    plan = json.loads(contract_path.read_bytes())
    _validate_contract(plan, contract_path)
    reference_path = _root_path(plan["archive_reference"]["path"])
    reference = json.loads(reference_path.read_bytes())
    if (
        _canonical_hash(reference, "reference_sha256") != reference["reference_sha256"]
        or reference["reference_sha256"]
        != plan["archive_reference"]["reference_sha256"]
    ):
        raise ValueError("frozen archive reference differs")
    expected_url = (
        f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={reference['symbol']}"
        f"&startTime={reference['start_time_ms']}&endTime={reference['end_time_ms']}&limit=1000"
    )
    if plan["request"]["url"] != expected_url:
        raise ValueError("native funding request differs from the frozen UTC reference")
    source = json.loads(_root_path(plan["outputs"]["result_path"]).read_bytes())
    if (
        _canonical_hash(source, "result_sha256") != source["result_sha256"]
        or source["contract"]["sha256"] != plan["contract_sha256"]
        or not source["source_gate"]["passed"]
    ):
        raise ValueError("native source receipt is not qualified")
    receipt = source["capture"]["receipt"]
    journal = [
        json.loads(line)
        for line in _root_path(plan["outputs"]["journal_path"]).read_text().splitlines()
    ]
    frozen_ms = int(
        datetime.fromisoformat(plan["frozen_at_utc"].replace("Z", "+00:00")).timestamp()
        * 1000
    )
    if (
        len(journal) != 2
        or journal[0]["phase"] != "intent"
        or journal[0]["request"] != plan["request"]
        or journal[0]["contract_sha256"] != plan["contract_sha256"]
        or journal[1] != receipt
        or receipt["status_code"] != 200
        or receipt["error_type"] is not None
        or not frozen_ms
        <= journal[0]["requested_at_ms"]
        == receipt["requested_at_ms"]
        <= receipt["completed_at_ms"]
    ):
        raise ValueError("native funding request journal differs")
    raw = _root_path(plan["outputs"]["raw_path"]).read_bytes()
    if (
        len(raw) != receipt["response_bytes"]
        or len(raw) > plan["response_byte_ceiling"]
    ):
        raise ValueError("native source byte count differs")
    events = parse_binance_usdm_funding_history(
        raw,
        expected_sha256=receipt["response_sha256"],
        expected_symbol=reference["symbol"],
    )
    comparison = compare_clock(reference, events)
    result = {
        "schema_version": "funding-archive-clock-comparison-v1",
        "contract_sha256": plan["contract_sha256"],
        "reference_sha256": reference["reference_sha256"],
        "native_result_sha256": source["result_sha256"],
        "native_body_sha256": hashlib.sha256(raw).hexdigest(),
        "comparison": comparison,
        "status": "exact_small_population_join_confirmed"
        if comparison["exact_timestamp_and_rate_join"]
        else "exact_join_rejected_no_time_repair",
        "label_admission_qualified": False,
        "accepted_edge": False,
        "profitability_claim": False,
        "new_network_requests": 0,
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    return result


def write_once(contract_path: Path) -> dict[str, object]:
    """Persist one adjudication intent and terminal result without retrying failures."""
    plan = json.loads(contract_path.read_bytes())
    _validate_contract(plan, contract_path.resolve())
    output = _root_path(plan["adjudication_output"])
    journal_path = _root_path(plan["adjudication_journal"])
    if output == journal_path or output.exists() or journal_path.exists():
        raise FileExistsError("one-use adjudication output or journal already exists")
    with journal_path.open("x", encoding="ascii", newline="\n") as journal:
        journal.write(
            json.dumps({"phase": "intent", "contract_sha256": plan["contract_sha256"]})
            + "\n"
        )
        journal.flush()
        os.fsync(journal.fileno())
        try:
            result = adjudicate(contract_path)
        except (ValueError, OSError, KeyError, TypeError) as error:
            result = {
                "schema_version": "funding-archive-clock-comparison-v1",
                "status": "adjudication_failed_closed",
                "error_type": type(error).__name__,
                "contract_sha256": plan["contract_sha256"],
                "label_admission_qualified": False,
                "accepted_edge": False,
                "profitability_claim": False,
                "new_network_requests": 0,
            }
            result["result_sha256"] = _canonical_hash(result, "result_sha256")
        with output.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        journal.write(
            json.dumps({"phase": "completed", "result_sha256": result["result_sha256"]})
            + "\n"
        )
        journal.flush()
        os.fsync(journal.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    result = write_once(args.contract)
    print(json.dumps({"status": result["status"], "label_admission_qualified": False}))


if __name__ == "__main__":
    main()
