"""Test fixed-orientation funding persistence for prefiltered TradFi perps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
import json
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlencode

import requests

from simple_ai_trading.storage import write_bytes_atomic
from tools.screen_polymarket_binance_tradfi_perps_funding_prefilter import (
    EMPTY_BODY_SHA256,
    RawJournal,
    _canonical_json,
    _canonical_payload_hash,
    _decimal,
    _get_json,
    _integer,
    _list,
    _mapping,
    _parse_utc,
    _sha256,
)


HOUR_MS = 3_600_000
EIGHT_HOURS_MS = 8 * HOUR_MS
YEAR_HOURS = Decimal("8760")


@dataclass(frozen=True)
class FundingObservation:
    timestamp_ms: int
    carry: Decimal
    return_8h: Decimal


def _nearest_hour(timestamp_ms: int, *, maximum_skew_ms: int) -> tuple[int, int]:
    nominal = ((timestamp_ms + HOUR_MS // 2) // HOUR_MS) * HOUR_MS
    skew = abs(timestamp_ms - nominal)
    if skew > maximum_skew_ms:
        raise ValueError(
            f"timestamp {timestamp_ms} is {skew} ms from its nearest UTC hour"
        )
    return nominal, skew


def parse_polymarket_funding(payload: object) -> list[tuple[int, Decimal]]:
    page = _mapping(payload, name="Polymarket funding page")
    if page.get("more") is not False:
        raise ValueError("Polymarket funding page is not complete within frozen range")
    rows: list[tuple[int, Decimal]] = []
    for raw in _list(page.get("data"), name="Polymarket funding data"):
        row = _mapping(raw, name="Polymarket funding row")
        rows.append(
            (
                _integer(row.get("timestamp"), name="Polymarket funding timestamp"),
                _decimal(row.get("funding_rate"), name="Polymarket funding rate"),
            )
        )
    if len({timestamp for timestamp, _ in rows}) != len(rows):
        raise ValueError("duplicate Polymarket funding timestamp")
    return sorted(rows)


def parse_binance_funding(
    payload: object,
) -> tuple[list[tuple[int, Decimal]], dict[str, int]]:
    by_timestamp: dict[int, Decimal] = {}
    rate_type_counts: dict[str, int] = {}
    for raw in _list(payload, name="Binance funding data"):
        row = _mapping(raw, name="Binance funding row")
        timestamp = _integer(row.get("fundingTime"), name="Binance funding timestamp")
        rate = _decimal(row.get("fundingRate"), name="Binance funding rate")
        by_timestamp[timestamp] = by_timestamp.get(timestamp, Decimal("0")) + rate
        rate_type = str(row.get("rateType") or "Unknown")
        rate_type_counts[rate_type] = rate_type_counts.get(rate_type, 0) + 1
    return sorted(by_timestamp.items()), dict(sorted(rate_type_counts.items()))


def parse_binance_klines(payload: object) -> list[tuple[int, Decimal, Decimal]]:
    rows: list[tuple[int, Decimal, Decimal]] = []
    for raw in _list(payload, name="Binance klines"):
        row = _list(raw, name="Binance kline")
        if len(row) < 5:
            raise ValueError("Binance kline is too short")
        rows.append(
            (
                _integer(row[0], name="Binance kline open time"),
                _decimal(row[1], name="Binance kline open"),
                _decimal(row[4], name="Binance kline close"),
            )
        )
    if len({timestamp for timestamp, _, _ in rows}) != len(rows):
        raise ValueError("duplicate Binance kline timestamp")
    return sorted(rows)


def align_funding(
    polymarket_rows: Sequence[tuple[int, Decimal]],
    binance_rows: Sequence[tuple[int, Decimal]],
    kline_rows: Sequence[tuple[int, Decimal, Decimal]],
    *,
    orientation: str,
    maximum_timestamp_phase_skew_ms: int,
) -> tuple[tuple[FundingObservation, ...], dict[str, int]]:
    if orientation not in {
        "short_polymarket_long_binance",
        "long_polymarket_short_binance",
    }:
        raise ValueError("orientation is invalid")
    factor = (
        Decimal("1")
        if orientation == "short_polymarket_long_binance"
        else Decimal("-1")
    )
    observed_skews: list[int] = []
    polymarket: dict[int, Decimal] = {}
    for timestamp, rate in polymarket_rows:
        nominal, skew = _nearest_hour(
            timestamp, maximum_skew_ms=maximum_timestamp_phase_skew_ms
        )
        observed_skews.append(skew)
        if nominal in polymarket:
            raise ValueError("duplicate nominal Polymarket funding hour")
        polymarket[nominal] = rate
    binance: dict[int, Decimal] = {}
    for timestamp, rate in binance_rows:
        nominal, skew = _nearest_hour(
            timestamp, maximum_skew_ms=maximum_timestamp_phase_skew_ms
        )
        observed_skews.append(skew)
        binance[nominal] = binance.get(nominal, Decimal("0")) + rate
    klines: dict[int, tuple[Decimal, Decimal]] = {}
    for timestamp, open_price, close_price in kline_rows:
        nominal, skew = _nearest_hour(
            timestamp, maximum_skew_ms=maximum_timestamp_phase_skew_ms
        )
        observed_skews.append(skew)
        if nominal in klines:
            raise ValueError("duplicate nominal Binance kline hour")
        klines[nominal] = (open_price, close_price)
    result: list[FundingObservation] = []
    for timestamp, binance_rate in sorted(binance.items()):
        hourly_timestamps = tuple(timestamp - offset * HOUR_MS for offset in range(8))
        if any(value not in polymarket for value in hourly_timestamps):
            continue
        price_hours = tuple(
            timestamp - EIGHT_HOURS_MS + offset * HOUR_MS for offset in range(8)
        )
        if any(value not in klines for value in price_hours):
            continue
        opening_price = klines[price_hours[0]][0]
        closing_price = klines[price_hours[-1]][1]
        if opening_price <= 0:
            continue
        polymarket_rate = sum(
            (polymarket[value] for value in hourly_timestamps), Decimal("0")
        )
        result.append(
            FundingObservation(
                timestamp_ms=timestamp,
                carry=factor * (polymarket_rate - binance_rate),
                return_8h=closing_price / opening_price - Decimal("1"),
            )
        )
    return tuple(result), {
        "maximum_observed_timestamp_phase_skew_ms": max(observed_skews, default=0),
        "normalized_binance_funding_hours": len(binance),
        "normalized_binance_kline_hours": len(klines),
        "normalized_polymarket_funding_hours": len(polymarket),
    }


def _slice_stats(rows: Sequence[FundingObservation]) -> dict[str, object]:
    carry_bips = [row.carry * Decimal("10000") for row in rows]
    return {
        "count": len(rows),
        "positive_count": sum(value > 0 for value in carry_bips),
        "sum_gross_carry_bips": str(sum(carry_bips, Decimal("0"))),
    }


def evaluate_history(
    rows: Sequence[FundingObservation],
    *,
    execution_hurdle_bips: Decimal,
    annual_opportunity_hurdle_bips_per_leg: Decimal,
    minimum_aligned_rows: int,
    minimum_regime_rows: int,
) -> dict[str, object]:
    if len(rows) < minimum_aligned_rows:
        return {
            "aligned_count": len(rows),
            "cross_regime_pass": False,
            "economic_role_pass": False,
            "status": "insufficient_aligned_history",
        }
    first_end = len(rows) // 2
    second_end = first_end + (len(rows) - first_end) // 2
    roles = {
        "training": rows[:first_end],
        "validation": rows[first_end:second_end],
        "test": rows[second_end:],
    }
    role_results: dict[str, object] = {}
    for name, selected in roles.items():
        gross = sum((row.carry for row in selected), Decimal("0")) * Decimal("10000")
        opportunity = (
            annual_opportunity_hurdle_bips_per_leg
            * Decimal("2")
            * Decimal(len(selected) * 8)
            / YEAR_HOURS
        )
        net = gross - execution_hurdle_bips - opportunity
        role_results[name] = {
            "count": len(selected),
            "gross_carry_bips": str(gross),
            "net_after_execution_and_capital_bips": str(net),
            "opportunity_hurdle_bips": str(opportunity),
            "passes": net > 0,
            "positive_count": sum(row.carry > 0 for row in selected),
        }
    economic_role_pass = all(
        bool(_mapping(value, name="role result")["passes"])
        for value in role_results.values()
    )

    absolute_returns = sorted(abs(row.return_8h) for row in rows)
    median_abs_return = absolute_returns[len(absolute_returns) // 2]
    regimes = {
        "down": [row for row in rows if row.return_8h < Decimal("-0.0025")],
        "high_volatility": [
            row for row in rows if abs(row.return_8h) >= median_abs_return
        ],
        "low_volatility": [
            row for row in rows if abs(row.return_8h) < median_abs_return
        ],
        "sideways": [row for row in rows if abs(row.return_8h) <= Decimal("0.0025")],
        "up": [row for row in rows if row.return_8h > Decimal("0.0025")],
    }
    regime_results = {
        name: _slice_stats(selected) for name, selected in regimes.items()
    }
    cross_regime_pass = all(
        int(value["count"]) >= minimum_regime_rows
        and Decimal(str(value["sum_gross_carry_bips"])) > 0
        for value in regime_results.values()
    )

    total_gross = sum((row.carry for row in rows), Decimal("0")) * Decimal("10000")
    total_opportunity = (
        annual_opportunity_hurdle_bips_per_leg
        * Decimal("2")
        * Decimal(len(rows) * 8)
        / YEAR_HOURS
    )
    total_net = total_gross - execution_hurdle_bips - total_opportunity
    leave_one_out = []
    for index in range(len(rows)):
        selected = tuple(row for offset, row in enumerate(rows) if offset != index)
        gross = sum((row.carry for row in selected), Decimal("0")) * Decimal("10000")
        opportunity = (
            annual_opportunity_hurdle_bips_per_leg
            * Decimal("2")
            * Decimal(len(selected) * 8)
            / YEAR_HOURS
        )
        leave_one_out.append(gross - execution_hurdle_bips - opportunity)
    worst_leave_one_out = min(leave_one_out)
    sign_consistency = Decimal(sum(row.carry > 0 for row in rows)) / Decimal(len(rows))
    persistence_candidate = (
        economic_role_pass
        and cross_regime_pass
        and worst_leave_one_out > 0
        and sign_consistency >= Decimal("0.75")
    )
    return {
        "aligned_count": len(rows),
        "cross_regime_pass": cross_regime_pass,
        "economic_role_pass": economic_role_pass,
        "first_timestamp_ms": rows[0].timestamp_ms,
        "historical_persistence_candidate": persistence_candidate,
        "last_timestamp_ms": rows[-1].timestamp_ms,
        "regimes": regime_results,
        "roles": role_results,
        "sign_consistency": str(sign_consistency),
        "status": "evaluated",
        "total_gross_carry_bips": str(total_gross),
        "total_net_after_execution_and_capital_bips": str(total_net),
        "total_opportunity_hurdle_bips": str(total_opportunity),
        "worst_leave_one_settlement_out_net_bips": str(worst_leave_one_out),
    }


def _verify_prefilter(path: Path, expected_hash: str) -> dict[str, object]:
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), name="prefilter")
    observed = _canonical_payload_hash(payload, "result_sha256")
    if observed != expected_hash:
        raise ValueError("prefilter hash differs from frozen contract")
    return payload


def _expected_request_plan(
    selected: Sequence[str],
    selected_rows: Mapping[str, Mapping[str, object]],
    window: Mapping[str, object],
) -> list[dict[str, str]]:
    polymarket_start = int(window["polymarket_start_timestamp_ms"])
    binance_funding_start = int(window["binance_funding_start_timestamp_ms"])
    binance_kline_start = int(window["binance_kline_start_timestamp_ms"])
    end = int(window["end_timestamp_ms"])
    binance_kline_end = int(window["binance_kline_end_timestamp_ms"])
    if not (
        polymarket_start < binance_funding_start <= end
        and binance_kline_start < binance_funding_start
        and binance_kline_end == end - 1
    ):
        raise ValueError("history window ordering is invalid")
    plan: list[dict[str, str]] = []
    for symbol in selected:
        row = selected_rows[symbol]
        instrument_id = int(row["polymarket_instrument_id"])
        binance_symbol = str(row["binance_symbol"])
        label = symbol.lower()
        requests_for_symbol = (
            (
                f"polymarket-{label}-funding",
                "https://api.perpetuals.polymarket.com/v1/info/funding?"
                + urlencode(
                    {
                        "instrument_id": instrument_id,
                        "start_timestamp": polymarket_start,
                        "end_timestamp": end,
                    }
                ),
            ),
            (
                f"binance-{label}-funding",
                "https://fapi.binance.com/fapi/v1/fundingRate?"
                + urlencode(
                    {
                        "symbol": binance_symbol,
                        "startTime": binance_funding_start,
                        "endTime": end,
                        "limit": 1000,
                    }
                ),
            ),
            (
                f"binance-{label}-klines",
                "https://fapi.binance.com/fapi/v1/klines?"
                + urlencode(
                    {
                        "symbol": binance_symbol,
                        "interval": "1h",
                        "startTime": binance_kline_start,
                        "endTime": binance_kline_end,
                        "limit": 1000,
                    }
                ),
            ),
        )
        plan.extend(
            {
                "body_sha256": EMPTY_BODY_SHA256,
                "label": request_label,
                "method": "GET",
                "url": url,
            }
            for request_label, url in requests_for_symbol
        )
    return plan


def run(contract_path: Path, output_path: Path, raw_root: Path) -> dict[str, object]:
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="utf-8")), name="contract"
    )
    contract_hash = _canonical_payload_hash(contract, "result_sha256")
    frozen_at = _parse_utc(contract.get("frozen_at_utc"), name="frozen_at_utc")
    if frozen_at > datetime.now(UTC):
        raise ValueError("frozen_at_utc is in the future")
    if str(contract.get("implementation_sha256") or "") != _sha256(
        Path(__file__).read_bytes()
    ):
        raise ValueError("implementation SHA-256 does not match frozen contract")
    dependency = _mapping(contract.get("dependency"), name="dependency")
    dependency_path = Path(str(dependency["path"]))
    if _sha256(dependency_path.read_bytes()) != str(dependency["sha256"]):
        raise ValueError("prefilter tool dependency hash mismatch")

    prefilter_lineage = _mapping(
        contract.get("prefilter_lineage"), name="prefilter lineage"
    )
    prefilter = _verify_prefilter(
        Path(str(prefilter_lineage["path"])),
        str(prefilter_lineage["result_sha256"]),
    )
    selected = [
        str(value)
        for value in _list(contract.get("selected_symbols"), name="selected symbols")
    ]
    prefilter_selected = _list(
        _mapping(prefilter["analysis"], name="prefilter analysis").get(
            "selected_for_separate_history_contract"
        ),
        name="prefilter selected symbols",
    )
    if selected != prefilter_selected:
        raise ValueError("selected symbols differ from source-bound prefilter")
    ranked_rows = _list(
        _mapping(prefilter["analysis"], name="prefilter analysis").get("ranked_rows"),
        name="prefilter ranked rows",
    )
    selected_rows = {
        str(_mapping(row, name="prefilter row")["base_asset"]): _mapping(
            row, name="prefilter row"
        )
        for row in ranked_rows
        if isinstance(row, Mapping) and str(row.get("base_asset") or "") in selected
    }
    if set(selected_rows) != set(selected):
        raise ValueError("one or more selected prefilter rows are missing")

    request_contract = _mapping(
        contract.get("request_contract"), name="request contract"
    )
    expected_raw_root = Path(str(request_contract["raw_directory"]))
    if raw_root != expected_raw_root:
        raise ValueError("raw directory differs from frozen contract")
    plan = [
        _mapping(raw, name="request plan row")
        for raw in _list(request_contract.get("requests"), name="request plan")
    ]
    if len(plan) != len(selected) * 3:
        raise ValueError("request plan must contain exactly three GETs per symbol")
    if any(
        row.get("method") != "GET" or row.get("body_sha256") != EMPTY_BODY_SHA256
        for row in plan
    ):
        raise ValueError("request plan contains a non-GET or nonempty body")
    history_window = _mapping(
        request_contract.get("history_window"), name="history window"
    )
    if plan != _expected_request_plan(selected, selected_rows, history_window):
        raise ValueError("request plan differs from the exact frozen history window")

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-tradfi-perps-funding-history/1.0",
        }
    )
    journal = RawJournal(
        raw_root,
        contract_hash=contract_hash,
        request_plan=plan,
    )
    payloads = {
        str(row["label"]): _get_json(
            session,
            journal,
            label=str(row["label"]),
            url=str(row["url"]),
        )
        for row in plan
    }

    economics = _mapping(contract.get("economics"), name="economics")
    symbol_results: dict[str, object] = {}
    for symbol in selected:
        polymarket_rows = parse_polymarket_funding(
            payloads[f"polymarket-{symbol.lower()}-funding"]
        )
        binance_rows, rate_type_counts = parse_binance_funding(
            payloads[f"binance-{symbol.lower()}-funding"]
        )
        kline_rows = parse_binance_klines(payloads[f"binance-{symbol.lower()}-klines"])
        prefilter_row = selected_rows[symbol]
        aligned, alignment_diagnostics = align_funding(
            polymarket_rows,
            binance_rows,
            kline_rows,
            orientation=str(prefilter_row["current_orientation"]),
            maximum_timestamp_phase_skew_ms=int(
                economics["maximum_timestamp_phase_skew_ms"]
            ),
        )
        symbol_results[symbol] = {
            "alignment_diagnostics": alignment_diagnostics,
            "binance_rate_type_counts": rate_type_counts,
            "fixed_orientation": prefilter_row["current_orientation"],
            "prefilter_spread_bips_per_8h": prefilter_row["funding_spread_bips_per_8h"],
            "result": evaluate_history(
                aligned,
                execution_hurdle_bips=Decimal(str(economics["execution_hurdle_bips"])),
                annual_opportunity_hurdle_bips_per_leg=Decimal(
                    str(economics["annual_opportunity_hurdle_bips_per_leg"])
                ),
                minimum_aligned_rows=int(economics["minimum_aligned_rows"]),
                minimum_regime_rows=int(economics["minimum_regime_rows"]),
            ),
            "source_counts": {
                "binance_funding_settlements": len(binance_rows),
                "binance_klines": len(kline_rows),
                "polymarket_hourly_funding": len(polymarket_rows),
            },
        }

    candidates = [
        symbol
        for symbol in selected
        if _mapping(
            _mapping(symbol_results[symbol], name="symbol result")["result"],
            name="evaluation result",
        ).get("historical_persistence_candidate")
        is True
    ]
    role_only_survivors = [
        symbol
        for symbol in selected
        if _mapping(
            _mapping(symbol_results[symbol], name="symbol result")["result"],
            name="evaluation result",
        ).get("economic_role_pass")
        is True
    ]
    result: dict[str, object] = {
        "schema_version": ("polymarket-binance-tradfi-perps-funding-history-v1"),
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract_path.as_posix(),
            "result_sha256": contract_hash,
        },
        "authority": {
            "account_state_accessed": False,
            "authenticated_requests": 0,
            "credentials_used": False,
            "http_get_requests": len(plan),
            "orders_transfers_or_account_mutations": 0,
            "paper_or_live_trading_authority": False,
        },
        "economics": economics,
        "selected_symbols": selected,
        "symbol_results": symbol_results,
        "role_only_survivors": role_only_survivors,
        "historical_persistence_candidates": candidates,
        "adjudication": {
            "accepted_edge": False,
            "candidate_for_books": False,
            "deployment_ready": False,
            "status": (
                "bounded_history_has_persistence_candidates"
                if candidates
                else "bounded_history_rejected_before_books"
            ),
        },
        "next_action": (
            "require_a_separate_conversion_fee_and_synchronized_depth_contract_"
            "for_only_the_persistent_candidates"
            if candidates
            else "do_not_request_books_for_this_snapshot_selected_population_"
            "without_a_material_funding_fee_session_or_instrument_change"
        ),
    }
    result_hash = _sha256(_canonical_json(result).encode("ascii"))
    result["result_sha256"] = result_hash
    write_bytes_atomic(
        output_path,
        (json.dumps(result, ensure_ascii=True, indent=2) + "\n").encode("ascii"),
    )
    journal.complete(result_hash)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run(args.contract, args.output, args.raw_root)
    print(
        _canonical_json(
            {
                "accepted_edge": result["adjudication"]["accepted_edge"],
                "historical_persistence_candidates": result[
                    "historical_persistence_candidates"
                ],
                "result_sha256": result["result_sha256"],
                "role_only_survivors": result["role_only_survivors"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
