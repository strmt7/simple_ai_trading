from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-clob-box-retained-prefilter-contract-v1.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-clob-box-retained-prefilter-v1-2026-08-29.json"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256(encoded)


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _minimum_quantity(symbol: dict[str, Any]) -> Decimal:
    lot = next(
        row for row in symbol["filters"] if row.get("filterType") == "LOT_SIZE"
    )
    return max(_decimal(symbol["minQty"]), _decimal(lot["minQty"]))


def _quote(
    ticker: dict[str, Any], *, capture_completed_at_ms: int
) -> dict[str, Any]:
    bid = _decimal(ticker["bidPrice"])
    ask = _decimal(ticker["askPrice"])
    close_time_ms = int(ticker["closeTime"])
    return {
        "bid": bid,
        "ask": ask,
        "close_time_ms": close_time_ms,
        "age_ms": capture_completed_at_ms - close_time_ms,
    }


def _row(
    *,
    direction: str,
    underlying: str,
    expiry_date_ms: int,
    low_strike: Decimal,
    high_strike: Decimal,
    low_call: dict[str, Any],
    low_put: dict[str, Any],
    high_call: dict[str, Any],
    high_put: dict[str, Any],
    tickers: dict[str, dict[str, Any]],
    capture_completed_at_ms: int,
    maximum_quote_age_ms: int,
) -> dict[str, Any] | None:
    symbols = [low_call, low_put, high_call, high_put]
    if any(symbol["symbol"] not in tickers for symbol in symbols):
        return None
    quotes = {
        symbol["symbol"]: _quote(
            tickers[symbol["symbol"]],
            capture_completed_at_ms=capture_completed_at_ms,
        )
        for symbol in symbols
    }
    if any(
        quote["bid"] <= 0 or quote["ask"] <= 0 or quote["ask"] < quote["bid"]
        for quote in quotes.values()
    ):
        return None
    width = high_strike - low_strike
    low_call_quote = quotes[low_call["symbol"]]
    low_put_quote = quotes[low_put["symbol"]]
    high_call_quote = quotes[high_call["symbol"]]
    high_put_quote = quotes[high_put["symbol"]]
    if direction == "long_box":
        entry_cashflow = -(
            low_call_quote["ask"]
            - high_call_quote["bid"]
            + high_put_quote["ask"]
            - low_put_quote["bid"]
        )
        terminal_cashflow = width
        leg_roles = {
            low_call["symbol"]: "buy_at_ask",
            high_call["symbol"]: "sell_at_bid",
            high_put["symbol"]: "buy_at_ask",
            low_put["symbol"]: "sell_at_bid",
        }
    elif direction == "reverse_box":
        entry_cashflow = (
            low_call_quote["bid"]
            - high_call_quote["ask"]
            + high_put_quote["bid"]
            - low_put_quote["ask"]
        )
        terminal_cashflow = -width
        leg_roles = {
            low_call["symbol"]: "sell_at_bid",
            high_call["symbol"]: "buy_at_ask",
            high_put["symbol"]: "sell_at_bid",
            low_put["symbol"]: "buy_at_ask",
        }
    else:
        raise ValueError(f"unsupported direction {direction}")
    gross_profit = entry_cashflow + terminal_cashflow
    minimum_quantity = max(_minimum_quantity(symbol) for symbol in symbols)
    ages = [int(quote["age_ms"]) for quote in quotes.values()]
    synchronization_gate = all(0 <= age <= maximum_quote_age_ms for age in ages)
    return {
        "direction": direction,
        "underlying": underlying,
        "expiry_date_ms": expiry_date_ms,
        "low_strike": str(low_strike),
        "high_strike": str(high_strike),
        "strike_width_USDT": str(width),
        "symbols": [symbol["symbol"] for symbol in symbols],
        "leg_roles": leg_roles,
        "entry_cashflow_per_unit_USDT": str(entry_cashflow),
        "terminal_cashflow_per_unit_USDT": str(terminal_cashflow),
        "gross_profit_per_unit_USDT": str(gross_profit),
        "minimum_common_quantity": str(minimum_quantity),
        "gross_profit_at_minimum_quantity_USDT": str(
            gross_profit * minimum_quantity
        ),
        "minimum_quote_age_ms": min(ages),
        "maximum_quote_age_ms": max(ages),
        "passes_frozen_quote_synchronization_gate": synchronization_gate,
        "optimistic_only": True,
    }


def main() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    implementation = ROOT / contract["implementation"]["path"]
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    if RESULT_PATH.exists():
        raise RuntimeError("one-use result already exists")

    exchange_path = ROOT / contract["retained_sources"]["exchange_info"]["path"]
    ticker_path = ROOT / contract["retained_sources"]["all_tickers"]["path"]
    if _sha256(exchange_path.read_bytes()) != contract["retained_sources"][
        "exchange_info"
    ]["sha256"]:
        raise RuntimeError("exchange-info source hash mismatch")
    if _sha256(ticker_path.read_bytes()) != contract["retained_sources"][
        "all_tickers"
    ]["sha256"]:
        raise RuntimeError("all-tickers source hash mismatch")
    exchange = json.loads(exchange_path.read_bytes())
    ticker_rows = json.loads(ticker_path.read_bytes())
    tickers = {str(row["symbol"]): row for row in ticker_rows}

    allowed_underlyings = set(contract["population"]["allowed_underlyings"])
    symbols = [
        row
        for row in exchange["optionSymbols"]
        if row.get("status") == "TRADING"
        and row.get("contractType") == "CRYPTO_OPTIONS"
        and row.get("underlyingType") == "CRYPTO"
        and row.get("underlying") in allowed_underlyings
        and _decimal(row.get("unit")) == Decimal("1")
    ]
    grouped: dict[tuple[str, int], dict[Decimal, dict[str, dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for symbol in symbols:
        grouped[(str(symbol["underlying"]), int(symbol["expiryDate"]))][
            _decimal(symbol["strikePrice"])
        ][str(symbol["side"])] = symbol

    rows: list[dict[str, Any]] = []
    for (underlying, expiry_date_ms), by_strike in sorted(grouped.items()):
        complete = {
            strike: legs
            for strike, legs in by_strike.items()
            if set(legs) == {"CALL", "PUT"}
        }
        strikes = sorted(complete)
        for low_index, low_strike in enumerate(strikes):
            for high_strike in strikes[low_index + 1 :]:
                for direction in ("long_box", "reverse_box"):
                    row = _row(
                        direction=direction,
                        underlying=underlying,
                        expiry_date_ms=expiry_date_ms,
                        low_strike=low_strike,
                        high_strike=high_strike,
                        low_call=complete[low_strike]["CALL"],
                        low_put=complete[low_strike]["PUT"],
                        high_call=complete[high_strike]["CALL"],
                        high_put=complete[high_strike]["PUT"],
                        tickers=tickers,
                        capture_completed_at_ms=contract["retained_sources"][
                            "all_tickers"
                        ]["completed_at_ms"],
                        maximum_quote_age_ms=contract["gates"][
                            "maximum_quote_age_ms"
                        ],
                    )
                    if row is not None:
                        rows.append(row)

    gross_positive = [
        row
        for row in rows
        if _decimal(row["gross_profit_per_unit_USDT"]) > Decimal("0")
    ]
    synchronized = [
        row for row in rows if row["passes_frozen_quote_synchronization_gate"]
    ]
    synchronized_positive = [
        row
        for row in gross_positive
        if row["passes_frozen_quote_synchronization_gate"]
    ]
    top = sorted(
        gross_positive,
        key=lambda row: (
            _decimal(row["gross_profit_per_unit_USDT"]),
            -int(row["maximum_quote_age_ms"]),
        ),
        reverse=True,
    )[: contract["reporting"]["maximum_retained_rows"]]
    result: dict[str, Any] = {
        "schema_version": "binance-options-clob-box-retained-prefilter-v1",
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "implementation": contract["implementation"],
        "retained_sources": contract["retained_sources"],
        "population": {
            "eligible_option_symbol_count": len(symbols),
            "underlying_expiry_group_count": len(grouped),
            "evaluated_box_direction_count": len(rows),
            "gross_positive_count": len(gross_positive),
            "synchronized_count": len(synchronized),
            "synchronized_gross_positive_count": len(synchronized_positive),
        },
        "payoff_identity": contract["payoff_identity"],
        "gates": contract["gates"],
        "top_optimistic_gross_positive_rows": top,
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "current_market_requests": 0,
            "next_action": (
                "freeze_one_current_exact_depth_fee_and_leg_risk_confirmation"
                if synchronized_positive
                else "stop_without_current_market_requests"
            ),
            "status": (
                "retained_synchronized_gross_positive_candidate_only"
                if synchronized_positive
                else "no_retained_synchronized_gross_positive_candidate"
            ),
        },
        "authority": contract["authority"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "evaluated_box_direction_count": len(rows),
                "gross_positive_count": len(gross_positive),
                "synchronized_gross_positive_count": len(synchronized_positive),
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
