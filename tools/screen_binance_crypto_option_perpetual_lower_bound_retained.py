"""Screen retained Binance crypto options against opposite perpetual hedges."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
from statistics import median
from typing import Mapping

from simple_ai_trading.storage import write_bytes_atomic
from tools.stress_binance_options_perpetual_conversion_retained import (
    _canonical_hash,
    _canonical_json,
    _common_quantity,
    _decimal,
    _filter,
    _sha256,
    _worst_adverse_funding,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "binance-crypto-option-perpetual-lower-bound-retained-contract-v1"
RESULT_SCHEMA_VERSION = "binance-crypto-option-perpetual-lower-bound-retained-result-v1"
UNDERLYINGS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _load_hash_bound(source: Mapping[str, object]) -> object:
    path = ROOT / str(source["path"])
    payload = path.read_bytes()
    if _sha256(payload) != source["sha256"]:
        raise ValueError(f"source hash mismatch: {path}")
    return json.loads(payload)


def _validate_contract(contract: Mapping[str, object], *, preflight_only: bool) -> None:
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected contract schema")
    expected_status = (
        "preflight_only_unconsumed"
        if preflight_only
        else "frozen_before_zero_network_retained_economic_screen"
    )
    if contract.get("status") != expected_status:
        raise ValueError("contract status does not match invocation mode")
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise ValueError("contract hash mismatch")
    for key in ("implementation", "shared_dependency"):
        binding = _mapping(contract[key], name=key)
        path = ROOT / str(binding["path"])
        if _sha256(path.read_bytes()) != binding["sha256"]:
            raise ValueError(f"{key} hash mismatch")
    population = _mapping(contract["population"], name="population")
    if population != {
        "allowed_underlyings": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "expected_eligible_option_count": 1410,
        "futures_contract_type": "PERPETUAL",
        "futures_margin_asset": "USDT",
        "futures_status": "TRADING",
        "option_contract_type": "CRYPTO_OPTIONS",
        "option_quote_asset": "USDT",
        "option_status": "TRADING",
        "option_underlying_type": "CRYPTO",
        "option_unit": "1",
    }:
        raise ValueError("unexpected population contract")


def _load_inputs(contract: Mapping[str, object]) -> dict[str, object]:
    retained = _mapping(contract["retained_sources"], name="retained sources")
    option_exchange = _mapping(
        _load_hash_bound(
            _mapping(retained["options_exchange_info"], name="options exchange source")
        ),
        name="options exchangeInfo",
    )
    option_tickers_raw = _list(
        _load_hash_bound(
            _mapping(retained["options_all_tickers"], name="options ticker source")
        ),
        name="option tickers",
    )
    futures_exchange = _mapping(
        _load_hash_bound(
            _mapping(retained["futures_exchange_info"], name="futures exchange source")
        ),
        name="futures exchangeInfo",
    )
    futures_books_raw = _list(
        _load_hash_bound(
            _mapping(retained["futures_all_book_tickers"], name="futures book source")
        ),
        name="futures books",
    )
    funding_sources = _mapping(retained["funding_histories"], name="funding sources")
    funding = {
        symbol: [
            _mapping(row, name=f"{symbol} funding row")
            for row in _list(
                _load_hash_bound(_mapping(source, name=f"{symbol} funding source")),
                name=f"{symbol} funding history",
            )
        ]
        for symbol, source in funding_sources.items()
    }
    if set(funding) != UNDERLYINGS:
        raise ValueError("funding population differs from contract")
    option_rows = [
        _mapping(row, name="option symbol")
        for row in _list(option_exchange.get("optionSymbols"), name="option symbols")
    ]
    options = [
        row
        for row in option_rows
        if row.get("status") == "TRADING"
        and row.get("contractType") == "CRYPTO_OPTIONS"
        and row.get("underlyingType") == "CRYPTO"
        and row.get("underlying") in UNDERLYINGS
        and row.get("quoteAsset") == "USDT"
        and _decimal(row.get("unit")) == Decimal("1")
    ]
    options.sort(key=lambda row: str(row["symbol"]))
    expected_count = int(
        _mapping(contract["population"], name="population")[
            "expected_eligible_option_count"
        ]
    )
    if len(options) != expected_count:
        raise ValueError("eligible option population count changed")
    option_tickers = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="option ticker") for value in option_tickers_raw
        )
    }
    futures = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="futures symbol")
            for value in _list(futures_exchange.get("symbols"), name="futures symbols")
        )
    }
    futures_books = {
        str(row["symbol"]): row
        for row in (_mapping(value, name="futures book") for value in futures_books_raw)
    }
    if len(option_tickers) != len(option_tickers_raw):
        raise ValueError("option ticker symbols are not unique")
    if len(futures_books) != len(futures_books_raw):
        raise ValueError("futures book symbols are not unique")
    for underlying in sorted(UNDERLYINGS):
        future = futures.get(underlying)
        book = futures_books.get(underlying)
        if future is None or book is None:
            raise ValueError(f"missing matching future or book for {underlying}")
        if (
            future.get("status") != "TRADING"
            or future.get("contractType") != "PERPETUAL"
            or future.get("marginAsset") != "USDT"
        ):
            raise ValueError(f"futures identity changed for {underlying}")
        if len(funding[underlying]) != 500:
            raise ValueError(f"funding history is incomplete for {underlying}")
        times = [int(row["fundingTime"]) for row in funding[underlying]]
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError(f"funding history ordering changed for {underlying}")
    observation = _mapping(contract["observation"], name="observation")
    option_interval = _mapping(
        observation["options_ticker_http_interval_ms"], name="option interval"
    )
    futures_interval = _mapping(
        observation["futures_book_http_interval_ms"], name="futures interval"
    )
    gap_ms = max(
        0,
        int(futures_interval["requested_at_ms"])
        - int(option_interval["completed_at_ms"]),
    )
    if gap_ms > int(observation["maximum_observation_window_skew_ms"]):
        raise ValueError("retained HTTP observation windows are not synchronized")
    capture_ms = int(futures_interval["completed_at_ms"])
    for underlying in UNDERLYINGS:
        book_time = int(futures_books[underlying]["time"])
        if capture_ms - book_time > int(observation["maximum_futures_book_age_ms"]):
            raise ValueError(f"retained futures book is too old for {underlying}")
    return {
        "options": options,
        "option_tickers": option_tickers,
        "futures": futures,
        "futures_books": futures_books,
        "funding": funding,
        "capture_ms": capture_ms,
        "observation_gap_ms": gap_ms,
    }


def _screen(
    contract: Mapping[str, object], inputs: Mapping[str, object]
) -> dict[str, object]:
    options = _list(inputs["options"], name="eligible options")
    option_tickers = _mapping(inputs["option_tickers"], name="option tickers")
    futures = _mapping(inputs["futures"], name="futures")
    futures_books = _mapping(inputs["futures_books"], name="futures books")
    funding = _mapping(inputs["funding"], name="funding")
    capture_ms = int(inputs["capture_ms"])
    stress = _mapping(contract["stress"], name="stress")
    fixed_bps = sum(
        _decimal(stress[key])
        for key in (
            "one_option_taker_fee_bps",
            "one_option_settlement_fee_bps",
            "futures_round_trip_fee_bps",
            "perpetual_expiry_basis_bps",
        )
    )
    rows: list[dict[str, object]] = []
    for option_value in options:
        option = _mapping(option_value, name="eligible option")
        symbol = str(option["symbol"])
        underlying = str(option["underlying"])
        ticker = _mapping(option_tickers.get(symbol), name=f"{symbol} ticker")
        future = _mapping(futures.get(underlying), name=f"{underlying} future")
        future_book = _mapping(futures_books.get(underlying), name=f"{underlying} book")
        option_side = str(option["side"])
        if option_side not in {"CALL", "PUT"}:
            raise ValueError(f"unsupported option side for {symbol}")
        option_ask = _decimal(ticker["askPrice"])
        perpetual_entry = _decimal(
            future_book["bidPrice" if option_side == "CALL" else "askPrice"]
        )
        perpetual_quantity = _decimal(
            future_book["bidQty" if option_side == "CALL" else "askQty"]
        )
        strike = _decimal(option["strikePrice"])
        gross = (
            perpetual_entry - strike - option_ask
            if option_side == "CALL"
            else strike - perpetual_entry - option_ask
        )
        positive_entry = min(option_ask, perpetual_entry, perpetual_quantity) > 0
        row: dict[str, object] = {
            "symbol": symbol,
            "underlying": underlying,
            "expiry_date_ms": int(option["expiryDate"]),
            "option_side": option_side,
            "strike_USDT": format(strike, "f"),
            "option_ask_USDT": format(option_ask, "f"),
            "perpetual_entry_side": "bid" if option_side == "CALL" else "ask",
            "perpetual_entry_USDT": format(perpetual_entry, "f"),
            "positive_entry_sides": positive_entry,
            "gross_terminal_floor_per_unit_USDT": format(gross, "f"),
            "gross_terminal_floor_positive": positive_entry and gross > 0,
            "option_depth_quantity_verified": False,
            "accepted_edge": False,
            "deployment_ready": False,
        }
        if not positive_entry or gross <= 0:
            rows.append(row)
            continue
        option_step = _decimal(_filter(option, "LOT_SIZE")["stepSize"])
        future_lot = _filter(future, "LOT_SIZE")
        future_step = _decimal(future_lot["stepSize"])
        quantity = _common_quantity(
            [option_step, future_step],
            [
                max(
                    _decimal(option["minQty"]),
                    _decimal(_filter(option, "LOT_SIZE")["minQty"]),
                ),
                max(
                    _decimal(future_lot["minQty"]),
                    _decimal(_filter(future, "MIN_NOTIONAL")["notional"])
                    / perpetual_entry,
                ),
            ],
        )
        expiry_ms = int(option["expiryDate"])
        history = [
            _mapping(value, name=f"{underlying} funding row")
            for value in _list(funding[underlying], name=f"{underlying} funding")
        ]
        history_times = [int(item["fundingTime"]) for item in history]
        interval_ms = int(
            median(
                b - a for a, b in zip(history_times, history_times[1:], strict=False)
            )
        )
        event_count = max(0, math.ceil((expiry_ms - capture_ms) / interval_ms))
        direction = (
            "conversion_short_perpetual"
            if option_side == "CALL"
            else "reversal_long_perpetual"
        )
        funding_stress = _worst_adverse_funding(
            history, direction=direction, event_count=event_count
        )
        funding_cost = (
            perpetual_entry
            * _decimal(funding_stress["adverse_rate_sum"])
            * _decimal(stress["funding_notional_multiplier"])
        )
        fixed_cost = perpetual_entry * fixed_bps / Decimal("10000")
        tick_cost = _decimal(stress["adverse_ticks_per_leg"]) * (
            _decimal(_filter(option, "PRICE_FILTER")["tickSize"])
            + _decimal(_filter(future, "PRICE_FILTER")["tickSize"])
        )
        horizon_years = Decimal(max(0, expiry_ms - capture_ms)) / Decimal("31557600000")
        capital_cost = (
            perpetual_entry
            * _decimal(stress["capital_notional_multiplier"])
            * _decimal(stress["annual_capital_hurdle"])
            * horizon_years
        )
        after_stress = gross - fixed_cost - tick_cost - funding_cost - capital_cost
        future_capacity = quantity <= perpetual_quantity
        row.update(
            {
                "exact_common_quantity": format(quantity, "f"),
                "futures_top_level_capacity_passes": future_capacity,
                "fixed_fee_and_basis_bps": format(fixed_bps, "f"),
                "fixed_fee_and_basis_cost_per_unit_USDT": format(fixed_cost, "f"),
                "adverse_tick_cost_per_unit_USDT": format(tick_cost, "f"),
                "funding": funding_stress,
                "funding_cost_per_unit_USDT": format(funding_cost, "f"),
                "capital_cost_per_unit_USDT": format(capital_cost, "f"),
                "after_all_retained_stress_per_unit_USDT": format(after_stress, "f"),
                "after_all_retained_stress_bps": format(
                    after_stress / perpetual_entry * Decimal("10000"), "f"
                ),
                "eligible_for_one_exact_option_depth_request": (
                    future_capacity and after_stress > 0
                ),
            }
        )
        rows.append(row)
    gross_positive = [row for row in rows if row["gross_terminal_floor_positive"]]
    survivors = [
        row
        for row in gross_positive
        if row.get("eligible_for_one_exact_option_depth_request") is True
    ]
    gross_positive.sort(
        key=lambda row: (
            -_decimal(row["gross_terminal_floor_per_unit_USDT"]),
            str(row["symbol"]),
        )
    )
    survivors.sort(
        key=lambda row: (
            -_decimal(row["after_all_retained_stress_bps"]),
            str(row["symbol"]),
        )
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "contract": {
            "sha256": contract["contract_sha256"],
        },
        "retained_sources": contract["retained_sources"],
        "observation": {
            **_mapping(contract["observation"], name="observation"),
            "observed_http_gap_ms": inputs["observation_gap_ms"],
        },
        "stress": stress,
        "population": {
            "eligible_option_count": len(rows),
            "positive_entry_side_count": sum(
                bool(row["positive_entry_sides"]) for row in rows
            ),
            "gross_positive_count": len(gross_positive),
            "after_all_retained_stress_positive_count": len(survivors),
        },
        "best_gross_rows": gross_positive[:10],
        "depth_request_candidates": survivors[
            : int(_mapping(contract["reporting"], name="reporting")["maximum_rows"])
        ],
        "all_rows": rows,
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "public_requests": 0,
            "status": (
                "retained_stress_survivors_require_one_separately_frozen_exact_option_depth_request"
                if survivors
                else "no_candidate_survives_complete_retained_stress"
            ),
            "next_action": (
                "freeze_one_exact_option_depth_request_for_the_deterministically_ranked_top_survivor"
                if survivors
                else "stop_without_current_market_requests"
            ),
        },
        "authority": contract["authority"],
        "implementation": contract["implementation"],
        "shared_dependency": contract["shared_dependency"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and args.output is None:
        raise ValueError("--output is required outside preflight mode")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    contract = _mapping(
        json.loads(args.contract.read_text(encoding="ascii")), name="contract"
    )
    _validate_contract(contract, preflight_only=args.preflight_only)
    inputs = _load_inputs(contract)
    if args.preflight_only:
        print(
            _canonical_json(
                {
                    "eligible_option_count": len(inputs["options"]),
                    "funding_symbols": sorted(inputs["funding"]),
                    "observation_gap_ms": inputs["observation_gap_ms"],
                    "status": "preflight_passed_no_economics_evaluated",
                }
            )
        )
        return 0
    result = _screen(contract, inputs)
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    assert args.output is not None
    write_bytes_atomic(
        args.output,
        (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
            "ascii"
        ),
    )
    print(
        _canonical_json(
            {
                **_mapping(result["population"], name="result population"),
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
