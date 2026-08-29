"""Stress retained Binance option/perpetual conversion candidates offline."""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-perpetual-conversion-retained-stress-contract-v1.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-perpetual-conversion-retained-stress-v1-2026-08-29.json"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: Mapping[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _load_hash_bound(source: Mapping[str, object]) -> object:
    path = ROOT / str(source["path"])
    payload = path.read_bytes()
    if _sha256(payload) != source["sha256"]:
        raise ValueError(f"source hash mismatch: {path}")
    return json.loads(payload)


def _filter(symbol: Mapping[str, object], filter_type: str) -> Mapping[str, object]:
    filters = symbol.get("filters")
    if not isinstance(filters, list):
        raise ValueError("symbol filters must be a list")
    return next(
        row
        for row in filters
        if isinstance(row, Mapping) and row.get("filterType") == filter_type
    )


def _common_quantity(steps: list[Decimal], minima: list[Decimal]) -> Decimal:
    scale = max(max(0, -step.as_tuple().exponent) for step in steps)
    factor = 10**scale
    integer_steps = [int(step * factor) for step in steps]
    common_integer = math.lcm(*integer_steps)
    common_step = Decimal(common_integer) / Decimal(factor)
    minimum = max(minima)
    units = (minimum / common_step).to_integral_value(rounding=ROUND_CEILING)
    return units * common_step


def _worst_adverse_funding(
    rows: list[Mapping[str, object]], *, direction: str, event_count: int
) -> dict[str, object]:
    rates = [_decimal(row["fundingRate"]) for row in rows]
    adverse = [
        max(rate, Decimal("0"))
        if direction == "reversal_long_perpetual"
        else max(-rate, Decimal("0"))
        for rate in rates
    ]
    times = [int(row["fundingTime"]) for row in rows]
    interval_ms = int(median(b - a for a, b in zip(times, times[1:], strict=False)))
    if event_count <= 0:
        rate_sum = Decimal("0")
        method = "no_future_funding_event"
    elif event_count <= len(adverse):
        rate_sum = max(
            sum(adverse[start : start + event_count], Decimal("0"))
            for start in range(len(adverse) - event_count + 1)
        )
        method = "worst_exact_length_rolling_sum"
    else:
        extrapolation_window = min(90, len(adverse))
        worst_window_sum = max(
            sum(adverse[start : start + extrapolation_window], Decimal("0"))
            for start in range(len(adverse) - extrapolation_window + 1)
        )
        rate_sum = (
            worst_window_sum / Decimal(extrapolation_window) * Decimal(event_count)
        )
        method = "worst_90_event_average_extrapolated"
    return {
        "adverse_rate_sum": format(rate_sum, "f"),
        "event_count_to_expiry": event_count,
        "history_event_count": len(rows),
        "history_first_time_ms": times[0],
        "history_last_time_ms": times[-1],
        "median_interval_ms": interval_ms,
        "method": method,
    }


def main() -> int:
    if RESULT_PATH.exists():
        raise FileExistsError("one-use retained stress result already exists")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise ValueError("contract hash mismatch")
    implementation = ROOT / contract["implementation"]["path"]
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise ValueError("implementation hash mismatch")

    parent = _load_hash_bound(contract["retained_sources"]["parent_v2_result"])
    option_exchange = _load_hash_bound(
        contract["retained_sources"]["options_exchange_info"]
    )
    futures_exchange = _load_hash_bound(
        contract["retained_sources"]["futures_exchange_info"]
    )
    futures_books = {
        str(row["symbol"]): row
        for row in _load_hash_bound(
            contract["retained_sources"]["futures_all_book_tickers"]
        )
    }
    funding = {
        symbol: _load_hash_bound(source)
        for symbol, source in contract["retained_sources"]["funding_histories"].items()
    }
    options = {str(row["symbol"]): row for row in option_exchange["optionSymbols"]}
    futures = {str(row["symbol"]): row for row in futures_exchange["symbols"]}

    stress = contract["stress"]
    fixed_bps = sum(
        _decimal(stress[key])
        for key in (
            "two_option_taker_fees_bps",
            "one_option_settlement_fee_bps",
            "futures_round_trip_fee_bps",
            "perpetual_expiry_basis_bps",
        )
    )
    rows: list[dict[str, object]] = []
    for candidate in parent["synchronized_gross_positive_rows"]:
        option_symbols = [
            symbol
            for symbol in candidate["symbols"]
            if symbol != candidate["underlying"]
        ]
        call = options[
            next(symbol for symbol in option_symbols if symbol.endswith("-C"))
        ]
        put = options[
            next(symbol for symbol in option_symbols if symbol.endswith("-P"))
        ]
        future = futures[str(candidate["underlying"])]
        future_book = futures_books[str(candidate["underlying"])]
        option_steps = [
            _decimal(_filter(call, "LOT_SIZE")["stepSize"]),
            _decimal(_filter(put, "LOT_SIZE")["stepSize"]),
        ]
        future_step = _decimal(_filter(future, "LOT_SIZE")["stepSize"])
        exact_quantity = _common_quantity(
            [*option_steps, future_step],
            [
                max(
                    _decimal(call["minQty"]),
                    _decimal(_filter(call, "LOT_SIZE")["minQty"]),
                ),
                max(
                    _decimal(put["minQty"]),
                    _decimal(_filter(put, "LOT_SIZE")["minQty"]),
                ),
                _decimal(_filter(future, "LOT_SIZE")["minQty"]),
            ],
        )
        direction = str(candidate["direction"])
        future_price = _decimal(
            future_book[
                "askPrice" if direction == "reversal_long_perpetual" else "bidPrice"
            ]
        )
        notional = future_price
        gross = _decimal(candidate["gross_profit_per_unit_USDT"])
        expiry_ms = int(candidate["expiry_date_ms"])
        capture_ms = int(stress["capture_completed_at_ms"])
        history = funding[str(candidate["underlying"])]
        history_times = [int(item["fundingTime"]) for item in history]
        interval_ms = int(
            median(
                b - a for a, b in zip(history_times, history_times[1:], strict=False)
            )
        )
        event_count = max(0, math.ceil((expiry_ms - capture_ms) / interval_ms))
        funding_stress = _worst_adverse_funding(
            history, direction=direction, event_count=event_count
        )
        funding_cost = (
            notional
            * _decimal(funding_stress["adverse_rate_sum"])
            * _decimal(stress["funding_notional_multiplier"])
        )
        fixed_cost = notional * fixed_bps / Decimal("10000")
        tick_cost = _decimal(stress["adverse_ticks_per_leg"]) * (
            _decimal(_filter(call, "PRICE_FILTER")["tickSize"])
            + _decimal(_filter(put, "PRICE_FILTER")["tickSize"])
            + _decimal(_filter(future, "PRICE_FILTER")["tickSize"])
        )
        horizon_years = Decimal(max(0, expiry_ms - capture_ms)) / Decimal("31557600000")
        capital_cost = (
            notional
            * _decimal(stress["capital_notional_multiplier"])
            * _decimal(stress["annual_capital_hurdle"])
            * horizon_years
        )
        after_stress = gross - fixed_cost - tick_cost - funding_cost - capital_cost
        rows.append(
            {
                "underlying": candidate["underlying"],
                "expiry_date_ms": expiry_ms,
                "strike_USDT": candidate["strike_USDT"],
                "direction": direction,
                "symbols": candidate["symbols"],
                "exact_common_quantity": format(exact_quantity, "f"),
                "gross_profit_per_unit_USDT": format(gross, "f"),
                "gross_profit_at_exact_quantity_USDT": format(
                    gross * exact_quantity, "f"
                ),
                "underlying_notional_per_unit_USDT": format(notional, "f"),
                "fixed_fee_and_basis_bps": format(fixed_bps, "f"),
                "fixed_fee_and_basis_cost_per_unit_USDT": format(fixed_cost, "f"),
                "adverse_tick_cost_per_unit_USDT": format(tick_cost, "f"),
                "funding": funding_stress,
                "funding_cost_per_unit_USDT": format(funding_cost, "f"),
                "capital_cost_per_unit_USDT": format(capital_cost, "f"),
                "after_all_retained_stress_per_unit_USDT": format(after_stress, "f"),
                "after_all_retained_stress_at_exact_quantity_USDT": format(
                    after_stress * exact_quantity, "f"
                ),
                "after_all_retained_stress_bps": format(
                    after_stress / notional * Decimal("10000"), "f"
                ),
                "eligible_for_separate_current_depth_confirmation": after_stress > 0,
                "option_depth_quantity_verified": False,
                "accepted_edge": False,
                "deployment_ready": False,
            }
        )

    survivors = [
        row for row in rows if row["eligible_for_separate_current_depth_confirmation"]
    ]
    survivors.sort(
        key=lambda row: _decimal(row["after_all_retained_stress_bps"]), reverse=True
    )
    result: dict[str, object] = {
        "schema_version": "binance-options-perpetual-conversion-retained-stress-v1",
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "retained_sources": contract["retained_sources"],
        "stress": stress,
        "population": {
            "synchronized_gross_positive_input_count": len(rows),
            "exact_quantity_grid_count": len(rows),
            "after_all_retained_stress_positive_count": len(survivors),
        },
        "top_current_depth_eligible_rows": survivors[
            : contract["reporting"]["maximum_current_depth_candidates"]
        ],
        "all_rows": rows,
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "current_market_requests": 0,
            "status": (
                "retained_stress_survivors_require_current_depth_confirmation"
                if survivors
                else "no_candidate_survives_complete_retained_stress"
            ),
            "next_action": (
                "freeze_one_top_candidate_current_two_option_depth_plus_matching_perpetual_book_confirmation"
                if survivors
                else "stop_without_current_market_requests"
            ),
        },
        "limitations": contract["limitations"],
        "authority": contract["authority"],
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        _canonical_json(
            {
                "after_all_retained_stress_positive_count": len(survivors),
                "input_count": len(rows),
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
