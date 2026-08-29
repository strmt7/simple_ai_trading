"""Screen retained Binance option/perpetual conversion and reversal parity."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-perpetual-conversion-retained-prefilter-contract-v1.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-perpetual-conversion-retained-prefilter-v1-2026-08-29.json"
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


def _load_source(source: Mapping[str, object]) -> object:
    path = ROOT / str(source["path"])
    payload = path.read_bytes()
    if _sha256(payload) != source["sha256"]:
        raise ValueError(f"source hash mismatch: {path}")
    return json.loads(payload)


def _option_minimum_quantity(symbol: Mapping[str, object]) -> Decimal:
    filters = symbol.get("filters")
    if not isinstance(filters, list):
        raise ValueError("option filters must be a list")
    lot = next(
        row
        for row in filters
        if isinstance(row, Mapping) and row.get("filterType") == "LOT_SIZE"
    )
    return max(_decimal(symbol["minQty"]), _decimal(lot["minQty"]))


def _futures_minimum_quantity(symbol: Mapping[str, object]) -> Decimal:
    filters = symbol.get("filters")
    if not isinstance(filters, list):
        raise ValueError("futures filters must be a list")
    lot = next(
        row
        for row in filters
        if isinstance(row, Mapping) and row.get("filterType") == "LOT_SIZE"
    )
    return _decimal(lot["minQty"])


def _option_quote(
    ticker: Mapping[str, object], *, capture_completed_at_ms: int
) -> dict[str, object]:
    raw_close = ticker.get("closeTime")
    close_time_ms = None if raw_close is None else int(str(raw_close))
    return {
        "bid": _decimal(ticker["bidPrice"]),
        "ask": _decimal(ticker["askPrice"]),
        "close_time_ms": close_time_ms,
        "age_ms": (
            None if close_time_ms is None else capture_completed_at_ms - close_time_ms
        ),
    }


def _evaluate(
    *,
    direction: str,
    option_call: Mapping[str, object],
    option_put: Mapping[str, object],
    call_ticker: Mapping[str, object],
    put_ticker: Mapping[str, object],
    futures_symbol: Mapping[str, object],
    futures_book: Mapping[str, object],
    options_completed_at_ms: int,
    futures_completed_at_ms: int,
    maximum_option_quote_age_ms: int,
    maximum_cross_source_skew_ms: int,
) -> dict[str, object] | None:
    call = _option_quote(call_ticker, capture_completed_at_ms=options_completed_at_ms)
    put = _option_quote(put_ticker, capture_completed_at_ms=options_completed_at_ms)
    future_bid = _decimal(futures_book["bidPrice"])
    future_ask = _decimal(futures_book["askPrice"])
    future_time_ms = int(futures_book["time"])
    strike = _decimal(option_call["strikePrice"])
    if direction == "conversion_short_perpetual":
        required = [call["ask"], put["bid"], future_bid]
        gross_profit = future_bid - strike - call["ask"] + put["bid"]
        roles = {
            str(option_call["symbol"]): "buy_call_at_ask",
            str(option_put["symbol"]): "sell_put_at_bid",
            str(futures_symbol["symbol"]): "sell_perpetual_at_bid",
        }
    elif direction == "reversal_long_perpetual":
        required = [call["bid"], put["ask"], future_ask]
        gross_profit = strike - future_ask + call["bid"] - put["ask"]
        roles = {
            str(option_call["symbol"]): "sell_call_at_bid",
            str(option_put["symbol"]): "buy_put_at_ask",
            str(futures_symbol["symbol"]): "buy_perpetual_at_ask",
        }
    else:
        raise ValueError(f"unsupported direction: {direction}")
    if any(value <= 0 for value in required):
        return None

    option_times = [call["close_time_ms"], put["close_time_ms"]]
    option_ages = [call["age_ms"], put["age_ms"]]
    has_option_times = all(value is not None for value in option_times)
    option_age_gate = has_option_times and all(
        0 <= int(value) <= maximum_option_quote_age_ms for value in option_ages
    )
    futures_age_ms = futures_completed_at_ms - future_time_ms
    cross_source_skew_ms = (
        max(abs(future_time_ms - int(value)) for value in option_times)
        if has_option_times
        else None
    )
    synchronization_gate = (
        option_age_gate
        and 0 <= futures_age_ms <= maximum_option_quote_age_ms
        and cross_source_skew_ms is not None
        and cross_source_skew_ms <= maximum_cross_source_skew_ms
    )
    common_minimum_quantity = max(
        _option_minimum_quantity(option_call),
        _option_minimum_quantity(option_put),
        _futures_minimum_quantity(futures_symbol),
    )
    return {
        "direction": direction,
        "underlying": option_call["underlying"],
        "expiry_date_ms": int(option_call["expiryDate"]),
        "strike_USDT": format(strike, "f"),
        "symbols": [
            option_call["symbol"],
            option_put["symbol"],
            futures_symbol["symbol"],
        ],
        "roles": roles,
        "gross_profit_per_unit_USDT": format(gross_profit, "f"),
        "common_minimum_quantity": format(common_minimum_quantity, "f"),
        "gross_profit_at_common_minimum_USDT": format(
            gross_profit * common_minimum_quantity, "f"
        ),
        "call_close_time_ms": call["close_time_ms"],
        "put_close_time_ms": put["close_time_ms"],
        "futures_book_time_ms": future_time_ms,
        "maximum_option_quote_age_ms": (
            max(int(value) for value in option_ages) if has_option_times else None
        ),
        "futures_book_age_ms": futures_age_ms,
        "maximum_cross_source_skew_ms": cross_source_skew_ms,
        "passes_frozen_synchronization_gate": synchronization_gate,
        "optimistic_only": True,
    }


def main() -> int:
    if RESULT_PATH.exists():
        raise FileExistsError("one-use retained prefilter result already exists")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise ValueError("contract hash mismatch")
    implementation = ROOT / contract["implementation"]["path"]
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise ValueError("implementation hash mismatch")

    sources = contract["retained_sources"]
    option_exchange = _load_source(sources["options_exchange_info"])
    option_ticker_rows = _load_source(sources["options_all_tickers"])
    futures_exchange = _load_source(sources["futures_exchange_info"])
    futures_book_rows = _load_source(sources["futures_all_book_tickers"])
    option_tickers = {str(row["symbol"]): row for row in option_ticker_rows}
    futures_books = {str(row["symbol"]): row for row in futures_book_rows}

    allowed = set(contract["population"]["allowed_underlyings"])
    option_symbols = [
        row
        for row in option_exchange["optionSymbols"]
        if row.get("status") == "TRADING"
        and row.get("contractType") == "CRYPTO_OPTIONS"
        and row.get("underlyingType") == "CRYPTO"
        and row.get("underlying") in allowed
        and _decimal(row.get("unit")) == Decimal("1")
    ]
    futures_symbols = {
        str(row["symbol"]): row
        for row in futures_exchange["symbols"]
        if row.get("status") == "TRADING"
        and row.get("contractType") == "PERPETUAL"
        and row.get("symbol") in allowed
    }
    pairs: dict[tuple[str, int, Decimal], dict[str, Mapping[str, object]]] = {}
    for symbol in option_symbols:
        key = (
            str(symbol["underlying"]),
            int(symbol["expiryDate"]),
            _decimal(symbol["strikePrice"]),
        )
        pairs.setdefault(key, {})[str(symbol["side"])] = symbol

    rows: list[dict[str, object]] = []
    evaluated_direction_count = 0
    for (underlying, _expiry, _strike), legs in sorted(pairs.items()):
        if set(legs) != {"CALL", "PUT"}:
            continue
        if underlying not in futures_symbols or underlying not in futures_books:
            continue
        call_symbol = legs["CALL"]
        put_symbol = legs["PUT"]
        call_ticker = option_tickers.get(str(call_symbol["symbol"]))
        put_ticker = option_tickers.get(str(put_symbol["symbol"]))
        if call_ticker is None or put_ticker is None:
            continue
        for direction in (
            "conversion_short_perpetual",
            "reversal_long_perpetual",
        ):
            evaluated_direction_count += 1
            row = _evaluate(
                direction=direction,
                option_call=call_symbol,
                option_put=put_symbol,
                call_ticker=call_ticker,
                put_ticker=put_ticker,
                futures_symbol=futures_symbols[underlying],
                futures_book=futures_books[underlying],
                options_completed_at_ms=sources["options_all_tickers"][
                    "completed_at_ms"
                ],
                futures_completed_at_ms=sources["futures_all_book_tickers"][
                    "completed_at_ms"
                ],
                maximum_option_quote_age_ms=contract["gates"][
                    "maximum_option_quote_age_ms"
                ],
                maximum_cross_source_skew_ms=contract["gates"][
                    "maximum_cross_source_skew_ms"
                ],
            )
            if row is not None:
                rows.append(row)

    gross_positive = [
        row
        for row in rows
        if _decimal(row["gross_profit_per_unit_USDT"]) > Decimal("0")
    ]
    synchronized = [row for row in rows if row["passes_frozen_synchronization_gate"]]
    synchronized_positive = [
        row for row in gross_positive if row["passes_frozen_synchronization_gate"]
    ]
    top = sorted(
        gross_positive,
        key=lambda row: _decimal(row["gross_profit_per_unit_USDT"]),
        reverse=True,
    )[: contract["reporting"]["maximum_retained_rows"]]
    result: dict[str, object] = {
        "schema_version": (
            "binance-options-perpetual-conversion-retained-prefilter-v1"
        ),
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "retained_sources": sources,
        "population": {
            "eligible_option_symbol_count": len(option_symbols),
            "complete_call_put_strike_count": sum(
                1 for legs in pairs.values() if set(legs) == {"CALL", "PUT"}
            ),
            "evaluated_direction_count": evaluated_direction_count,
            "executable_side_direction_count": len(rows),
            "gross_positive_count": len(gross_positive),
            "synchronized_count": len(synchronized),
            "synchronized_gross_positive_count": len(synchronized_positive),
        },
        "identity": contract["identity"],
        "gates": contract["gates"],
        "top_optimistic_gross_positive_rows": top,
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "current_market_requests": 0,
            "status": (
                "retained_synchronized_gross_positive_candidate_only"
                if synchronized_positive
                else "no_retained_synchronized_gross_positive_candidate"
            ),
            "next_action": (
                "freeze_candidate_specific_retained_funding_and_expiry_basis_stress"
                if synchronized_positive
                else "stop_without_current_market_requests"
            ),
        },
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
                "evaluated_direction_count": evaluated_direction_count,
                "gross_positive_count": len(gross_positive),
                "synchronized_gross_positive_count": len(synchronized_positive),
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
