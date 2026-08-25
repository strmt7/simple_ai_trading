"""Screen displayed Binance quarterly cash-and-carry basis without trading."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

import simple_ai_trading.quarterly_carry as carry_module
from simple_ai_trading.paper_execution import BookLevel
from simple_ai_trading.quarterly_carry import screen_quarterly_cash_and_carry
from simple_ai_trading.storage import write_bytes_atomic


SCHEMA_VERSION = "binance-quarterly-cash-and-carry-screen-v1"
SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"
UNDERLYINGS = ("BTC", "ETH")
QUANTITIES = {
    "BTC": (Decimal("0.001"), Decimal("0.01"), Decimal("0.1")),
    "ETH": (Decimal("0.01"), Decimal("0.1"), Decimal("1")),
}
ALL_IN_COST_HURDLE_BIPS = Decimal("35")
DEPTH_LIMIT = 100
MAX_RECEIPT_SPAN_MS = 2_000
MAX_FUTURE_EVENT_AGE_MS = 5_000


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


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _get(
    session: requests.Session,
    url: str,
    *,
    params: Mapping[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    before_ms = time.time_ns() // 1_000_000
    response = session.get(url, params=params, timeout=30)
    after_ms = time.time_ns() // 1_000_000
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        raise RuntimeError(
            "Binance rate limit reached; stopped without retry"
            + (f"; Retry-After={retry_after}" if retry_after else "")
        )
    response.raise_for_status()
    try:
        decoded = response.json()
    except requests.JSONDecodeError as exc:
        raise ValueError(f"source {url} did not return JSON") from exc
    return decoded, {
        "url": response.url,
        "payload_sha256": _sha256(response.content),
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "request_elapsed_ms": after_ms - before_ms,
    }


def _filter(contract: Mapping[str, object], filter_type: str) -> dict[str, object]:
    matches = [
        _mapping(item, name="futures symbol filter")
        for item in _list(contract.get("filters"), name="futures symbol filters")
        if isinstance(item, Mapping) and item.get("filterType") == filter_type
    ]
    if len(matches) != 1:
        raise ValueError(f"futures symbol requires one {filter_type} filter")
    return matches[0]


def _contracts(raw: object) -> tuple[dict[str, object], ...]:
    payload = _mapping(raw, name="futures exchange information")
    selected: list[dict[str, object]] = []
    for item in _list(payload.get("symbols"), name="futures symbols"):
        contract = _mapping(item, name="futures symbol")
        pair = str(contract.get("pair") or "")
        if not (
            pair in {f"{asset}USDT" for asset in UNDERLYINGS}
            and contract.get("status") == "TRADING"
            and contract.get("contractType") in {"CURRENT_QUARTER", "NEXT_QUARTER"}
            and contract.get("quoteAsset") == "USDT"
            and contract.get("marginAsset") == "USDT"
        ):
            continue
        symbol = str(contract.get("symbol") or "")
        base_asset = str(contract.get("baseAsset") or "")
        delivery_ms = contract.get("deliveryDate")
        if (
            not symbol
            or base_asset not in UNDERLYINGS
            or pair != f"{base_asset}USDT"
            or not isinstance(delivery_ms, int)
            or isinstance(delivery_ms, bool)
            or delivery_ms <= 0
        ):
            raise ValueError("selected quarterly contract identity is invalid")
        lot = _filter(contract, "LOT_SIZE")
        selected.append(
            {
                "symbol": symbol,
                "pair": pair,
                "base_asset": base_asset,
                "contract_type": contract["contractType"],
                "delivery_time_ms": delivery_ms,
                "minimum_quantity": str(lot.get("minQty") or ""),
                "quantity_step": str(lot.get("stepSize") or ""),
            }
        )
    selected.sort(key=lambda row: (int(row["delivery_time_ms"]), str(row["symbol"])))
    if len(selected) != 4 or {
        (str(row["base_asset"]), str(row["contract_type"])) for row in selected
    } != {
        (asset, contract_type)
        for asset in UNDERLYINGS
        for contract_type in ("CURRENT_QUARTER", "NEXT_QUARTER")
    }:
        raise ValueError(
            "expected exactly current and next BTC/ETH quarterly contracts"
        )
    return tuple(selected)


def _levels(raw: object, *, side: str, descending: bool) -> tuple[BookLevel, ...]:
    payload = _mapping(raw, name="depth payload")
    result: list[BookLevel] = []
    for item in _list(payload.get(side), name=f"depth {side}"):
        level = _list(item, name="depth level")
        if len(level) < 2:
            raise ValueError("depth level has fewer than two fields")
        result.append(
            BookLevel(price=Decimal(str(level[0])), quantity=Decimal(str(level[1])))
        )
    normalized = tuple(result)
    if (
        not normalized
        or tuple(sorted(normalized, key=lambda level: level.price, reverse=descending))
        != normalized
    ):
        raise ValueError(f"depth {side} is empty or incorrectly sorted")
    return normalized


def _book_payload(raw: object) -> dict[str, object]:
    payload = _mapping(raw, name="depth payload")
    result: dict[str, object] = {
        "last_update_id": payload.get("lastUpdateId"),
        "bids": _list(payload.get("bids"), name="depth bids"),
        "asks": _list(payload.get("asks"), name="depth asks"),
    }
    if "E" in payload:
        result["event_time_ms"] = payload["E"]
    if "T" in payload:
        result["transaction_time_ms"] = payload["T"]
    return result


def _result_payload(result: carry_module.QuarterlyCarryResult) -> dict[str, object]:
    return {
        "quantity": str(result.quantity),
        "capture_time_ms": result.capture_time_ms,
        "delivery_time_ms": result.delivery_time_ms,
        "spot_ask_vwap": str(result.spot_buy.price),
        "spot_cost_quote": str(result.spot_buy.quote_value),
        "future_bid_vwap": str(result.future_sale.price),
        "future_sale_quote": str(result.future_sale.quote_value),
        "gross_profit_quote": str(result.gross_profit_quote),
        "gross_basis_bips": str(result.gross_basis_bips),
        "all_in_cost_hurdle_bips": str(result.all_in_cost_hurdle_bips),
        "after_hurdle_profit_quote": str(result.after_hurdle_profit_quote),
        "after_hurdle_basis_bips": str(result.after_hurdle_basis_bips),
        "gross_simple_annualized_bips": str(result.gross_simple_annualized_bips),
        "after_hurdle_simple_annualized_bips": str(
            result.after_hurdle_simple_annualized_bips
        ),
        "gross_positive": result.gross_positive,
        "after_hurdle_positive": result.after_hurdle_positive,
    }


def run(*, session: requests.Session | None = None) -> dict[str, object]:
    """Run one bounded public-data screen and return its source-bound result."""

    started_ms = time.time_ns() // 1_000_000
    http = session or requests.Session()
    exchange_raw, exchange_source = _get(
        http, f"{FUTURES_BASE_URL}/fapi/v1/exchangeInfo"
    )
    contracts = _contracts(exchange_raw)
    screens: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for contract in contracts:
        symbol = str(contract["symbol"])
        pair = str(contract["pair"])
        base_asset = str(contract["base_asset"])
        spot_raw, spot_source = _get(
            http,
            f"{SPOT_BASE_URL}/api/v3/depth",
            params={"symbol": pair, "limit": DEPTH_LIMIT},
        )
        future_raw, future_source = _get(
            http,
            f"{FUTURES_BASE_URL}/fapi/v1/depth",
            params={"symbol": symbol, "limit": DEPTH_LIMIT},
        )
        capture_time_ms = int(future_source["received_after_ms"])
        receipt_span_ms = capture_time_ms - int(spot_source["requested_before_ms"])
        future_payload = _mapping(future_raw, name="futures depth payload")
        event_time_ms = future_payload.get("E")
        if not isinstance(event_time_ms, int) or isinstance(event_time_ms, bool):
            raise ValueError("futures depth event time is invalid")
        future_event_age_ms = capture_time_ms - event_time_ms
        freshness_passed = (
            0 <= future_event_age_ms <= MAX_FUTURE_EVENT_AGE_MS
            and receipt_span_ms <= MAX_RECEIPT_SPAN_MS
        )
        spot_asks = _levels(spot_raw, side="asks", descending=False)
        future_bids = _levels(future_raw, side="bids", descending=True)
        minimum = Decimal(str(contract["minimum_quantity"]))
        step = Decimal(str(contract["quantity_step"]))
        rows: list[dict[str, object]] = []
        for quantity in QUANTITIES[base_asset]:
            if quantity < minimum or quantity % step != 0:
                raise ValueError("configured carry quantity violates futures LOT_SIZE")
            result = screen_quarterly_cash_and_carry(
                spot_asks=spot_asks,
                future_bids=future_bids,
                quantity=quantity,
                capture_time_ms=capture_time_ms,
                delivery_time_ms=int(contract["delivery_time_ms"]),
                all_in_cost_hurdle_bips=ALL_IN_COST_HURDLE_BIPS,
            )
            rows.append(
                {"quantity": str(quantity), "depth_available": result is not None}
                if result is None
                else _result_payload(result)
            )
        screens.append(
            {
                **contract,
                "receipt_span_ms": receipt_span_ms,
                "future_event_age_ms": future_event_age_ms,
                "freshness_passed": freshness_passed,
                "quantity_results": rows,
            }
        )
        sources.append(
            {
                "symbol": symbol,
                "spot_request": spot_source,
                "future_request": future_source,
                "spot_book": _book_payload(spot_raw),
                "future_book": _book_payload(future_raw),
            }
        )

    result_rows = [
        row
        for screen in screens
        for row in _list(screen["quantity_results"], name="quantity results")
        if row.get("depth_available") is not False
    ]
    gross_positive = sum(row.get("gross_positive") is True for row in result_rows)
    after_hurdle_positive = sum(
        row.get("after_hurdle_positive") is True for row in result_rows
    )
    fresh_positive = sum(
        screen["freshness_passed"] is True
        and any(
            row.get("after_hurdle_positive") is True
            for row in _list(screen["quantity_results"], name="quantity results")
        )
        for screen in screens
    )
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "direction_neutral_quarterly_cash_and_carry_depth_screen",
        "started_at_ms": started_ms,
        "completed_at_ms": time.time_ns() // 1_000_000,
        "scope": {
            "underlyings": list(UNDERLYINGS),
            "contract_types": ["CURRENT_QUARTER", "NEXT_QUARTER"],
            "depth_limit": DEPTH_LIMIT,
            "quantities": {
                key: [str(value) for value in values]
                for key, values in QUANTITIES.items()
            },
            "all_in_cost_hurdle_bips": str(ALL_IN_COST_HURDLE_BIPS),
            "maximum_receipt_span_ms": MAX_RECEIPT_SPAN_MS,
            "maximum_future_event_age_ms": MAX_FUTURE_EVENT_AGE_MS,
        },
        "source_contract": {
            "exchange_information": exchange_source,
            "selected_contracts": list(contracts),
            "book_sources": sources,
            "implementation": {
                "tool_path": Path(__file__).name,
                "tool_sha256": _sha256(Path(__file__).read_bytes()),
                "module_path": Path(carry_module.__file__).name,
                "module_sha256": _sha256(Path(carry_module.__file__).read_bytes()),
            },
        },
        "screens": screens,
        "verdict": {
            "status": (
                "unqualified_positive_basis_requires_exact_account_and_carry_costs"
                if after_hurdle_positive
                else "rejected_no_positive_basis_after_configured_hurdle"
            ),
            "contract_count": len(screens),
            "quantity_screen_count": len(result_rows),
            "gross_positive_count": gross_positive,
            "after_hurdle_positive_count": after_hurdle_positive,
            "fresh_contract_with_after_hurdle_positive_count": fresh_positive,
            "accepted_edge": False,
            "trading_authority": False,
        },
        "safety": {
            "public_market_data_only": True,
            "credentials_used": False,
            "orders_placed": False,
            "account_commission_inferred": False,
            "spot_index_convergence_assumed": False,
            "liquidation_impossibility_assumed": False,
        },
        "limitations": [
            "The 35 bps all-in hurdle is a sensitivity input, not authenticated account commission evidence.",
            "The spot depth response has no exchange event timestamp, so receipt proximity is not atomic cross-venue synchronization.",
            "Realized carry still depends on delivery-index versus spot exit basis, exact settlement charges, and spot disposal depth.",
            "Futures margin, liquidation buffer, capital opportunity cost, collateral haircuts, outages, and taxes remain unmeasured.",
            "A current positive basis is post-selection discovery and cannot establish persistence or profitability.",
        ],
    }
    artifact["result_sha256"] = _sha256(_canonical_json(artifact).encode("ascii"))
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    write_bytes_atomic(
        args.output,
        (_canonical_json(result) + "\n").encode("ascii"),
    )
    print(json.dumps(result["verdict"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
