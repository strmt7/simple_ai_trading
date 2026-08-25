"""Screen public Binance USDT/USDC perpetual funding differentials once."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
from math import gcd, lcm
from pathlib import Path
import time
from typing import Mapping, Sequence

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-cross-stablecoin-funding-screen-contract-v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-cross-stablecoin-funding-snapshot-v1-2026-08-25.json"
)
BASE_URL = "https://fapi.binance.com"
ASSETS = ("BTC", "ETH", "SOL")
QUOTES = ("USDT", "USDC")
HISTORY_LIMIT = 1000
MINIMUM_ALIGNED_ROWS = 90
MAXIMUM_REQUESTS = 9
SCHEMA_VERSION = "binance-cross-stablecoin-funding-screen-v1"


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


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError(f"{name} must be a finite decimal")
    return parsed


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _load_contract() -> dict[str, object]:
    contract = _mapping(
        json.loads(CONTRACT_PATH.read_text(encoding="ascii")),
        name="screen contract",
    )
    claimed = str(contract.get("result_sha256") or "")
    body = dict(contract)
    body.pop("result_sha256", None)
    observed = _sha256(_canonical_json(body).encode("ascii"))
    if claimed != observed:
        raise ValueError("screen contract canonical hash mismatch")
    if contract.get("status") != "frozen_before_any_binance_market_request":
        raise ValueError("screen contract was not frozen before market access")
    return contract


def _get(
    session: requests.Session,
    path: str,
    *,
    params: Mapping[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    before_ms = time.time_ns() // 1_000_000
    response = session.get(f"{BASE_URL}{path}", params=params, timeout=30)
    after_ms = time.time_ns() // 1_000_000
    receipt = {
        "url": response.url,
        "status_code": response.status_code,
        "payload_bytes": len(response.content),
        "payload_sha256": _sha256(response.content),
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "request_elapsed_ms": after_ms - before_ms,
    }
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
        raise ValueError(f"source {response.url} did not return JSON") from exc
    return decoded, receipt


def _filter(contract: Mapping[str, object], filter_type: str) -> dict[str, object]:
    matches = [
        _mapping(item, name="contract filter")
        for item in _list(contract.get("filters"), name="contract filters")
        if isinstance(item, Mapping) and item.get("filterType") == filter_type
    ]
    if len(matches) != 1:
        raise ValueError(f"contract requires exactly one {filter_type} filter")
    return matches[0]


def _selected_contracts(raw: object) -> dict[str, dict[str, dict[str, object]]]:
    payload = _mapping(raw, name="exchange information")
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for raw_item in _list(payload.get("symbols"), name="exchange symbols"):
        contract = _mapping(raw_item, name="exchange symbol")
        base = str(contract.get("baseAsset") or "")
        quote = str(contract.get("quoteAsset") or "")
        margin = str(contract.get("marginAsset") or "")
        if not (
            base in ASSETS
            and quote in QUOTES
            and margin == quote
            and contract.get("contractType") == "PERPETUAL"
            and contract.get("status") == "TRADING"
        ):
            continue
        symbol = str(contract.get("symbol") or "")
        pair = str(contract.get("pair") or "")
        if not symbol or pair != symbol or symbol != f"{base}{quote}":
            raise ValueError("selected perpetual identity is invalid")
        if quote in grouped[base]:
            raise ValueError("selected base and quote contract is duplicated")
        lot = _filter(contract, "LOT_SIZE")
        market_lot = _filter(contract, "MARKET_LOT_SIZE")
        minimum_notional = _filter(contract, "MIN_NOTIONAL")
        grouped[base][quote] = {
            "symbol": symbol,
            "pair": pair,
            "base_asset": base,
            "quote_asset": quote,
            "margin_asset": margin,
            "contract_type": contract["contractType"],
            "status": contract["status"],
            "price_precision": contract.get("pricePrecision"),
            "quantity_precision": contract.get("quantityPrecision"),
            "lot_size": lot,
            "market_lot_size": market_lot,
            "minimum_notional": minimum_notional,
        }
    result = {
        base: by_quote
        for base, by_quote in grouped.items()
        if set(by_quote) == set(QUOTES)
    }
    if not result:
        raise ValueError("no eligible USDT/USDC perpetual pair exists")
    return dict(sorted(result.items()))


def _rows_by_symbol(raw: object, *, name: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for item in _list(raw, name=name):
        row = _mapping(item, name=f"{name} row")
        symbol = str(row.get("symbol") or "")
        if not symbol or symbol in result:
            raise ValueError(f"{name} symbols must be nonempty and unique")
        result[symbol] = row
    return result


def _fraction_lcm(values: Sequence[Fraction]) -> Fraction:
    return Fraction(
        lcm(*(value.numerator for value in values)),
        gcd(*(value.denominator for value in values)),
    )


def _common_quantity(contracts: Sequence[Mapping[str, object]]) -> Decimal:
    minimums: list[Decimal] = []
    steps: list[Decimal] = []
    for contract in contracts:
        lot = _mapping(contract.get("lot_size"), name="LOT_SIZE")
        minimums.append(
            _decimal(lot.get("minQty"), name="minimum quantity", positive=True)
        )
        steps.append(_decimal(lot.get("stepSize"), name="quantity step", positive=True))
    common_step = _fraction_lcm(tuple(Fraction(step) for step in steps))
    largest_minimum = Fraction(max(minimums))
    multiple = (largest_minimum / common_step).__ceil__()
    quantity = common_step * multiple
    result = Decimal(quantity.numerator) / Decimal(quantity.denominator)
    if any(
        result % step != 0 or result < minimum
        for step, minimum in zip(steps, minimums, strict=True)
    ):
        raise ValueError("failed to derive exact common quantity")
    return result


def _funding_history(raw: object, *, symbol: str) -> dict[int, Decimal]:
    rows: dict[int, Decimal] = {}
    previous_time = 0
    for item in _list(raw, name=f"{symbol} funding history"):
        row = _mapping(item, name="funding row")
        if row.get("symbol") != symbol:
            raise ValueError("funding row symbol mismatch")
        funding_time = _integer(
            row.get("fundingTime"), name="funding time", positive=True
        )
        if funding_time <= previous_time or funding_time in rows:
            raise ValueError("funding times must be unique and strictly increasing")
        rows[funding_time] = _decimal(row.get("fundingRate"), name="funding rate")
        previous_time = funding_time
    return rows


def _statistics(values: Sequence[Decimal]) -> dict[str, object]:
    if not values:
        raise ValueError("statistics require at least one value")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    total = sum(values, Decimal("0"))
    return {
        "count": len(values),
        "mean": str(total / len(values)),
        "median": str(median),
        "minimum": str(min(values)),
        "maximum": str(max(values)),
        "sum": str(total),
        "positive_count": sum(value > 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
    }


def _drawdown_bips(values: Sequence[Decimal]) -> Decimal:
    cumulative = Decimal("0")
    peak = Decimal("0")
    maximum = Decimal("0")
    for value in values:
        cumulative += value * Decimal("10000")
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)
    return maximum


def _month_key(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).strftime("%Y-%m")


def _book(
    raw: Mapping[str, object],
    *,
    expected_symbol: str,
) -> dict[str, Decimal]:
    if raw.get("symbol") != expected_symbol:
        raise ValueError("book symbol mismatch")
    bid = _decimal(raw.get("bidPrice"), name="book bid", positive=True)
    ask = _decimal(raw.get("askPrice"), name="book ask", positive=True)
    bid_quantity = _decimal(raw.get("bidQty"), name="book bid quantity", positive=True)
    ask_quantity = _decimal(raw.get("askQty"), name="book ask quantity", positive=True)
    if bid > ask:
        raise ValueError("book is crossed")
    return {
        "bid": bid,
        "ask": ask,
        "bid_quantity": bid_quantity,
        "ask_quantity": ask_quantity,
    }


def _evaluate_pair(
    *,
    base: str,
    contracts: Mapping[str, Mapping[str, object]],
    histories: Mapping[str, object],
    premium_by_symbol: Mapping[str, Mapping[str, object]],
    book_by_symbol: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    usdt_symbol = str(contracts["USDT"]["symbol"])
    usdc_symbol = str(contracts["USDC"]["symbol"])
    usdt_history = _funding_history(histories[usdt_symbol], symbol=usdt_symbol)
    usdc_history = _funding_history(histories[usdc_symbol], symbol=usdc_symbol)
    aligned_times = sorted(set(usdt_history) & set(usdc_history))
    if len(aligned_times) < MINIMUM_ALIGNED_ROWS:
        return {
            "base_asset": base,
            "symbols": [usdt_symbol, usdc_symbol],
            "aligned_row_count": len(aligned_times),
            "qualified_public_screen": False,
            "failure_reasons": ["fewer_than_90_exactly_aligned_funding_rows"],
        }
    split = len(aligned_times) // 2
    selection_times = aligned_times[:split]
    evaluation_times = aligned_times[split:]
    usdt_selection_mean = sum(
        (usdt_history[item] for item in selection_times), Decimal("0")
    ) / len(selection_times)
    usdc_selection_mean = sum(
        (usdc_history[item] for item in selection_times), Decimal("0")
    ) / len(selection_times)
    if usdt_selection_mean == usdc_selection_mean:
        return {
            "base_asset": base,
            "symbols": [usdt_symbol, usdc_symbol],
            "aligned_row_count": len(aligned_times),
            "qualified_public_screen": False,
            "failure_reasons": ["selection_mean_funding_tie"],
        }
    short_quote = "USDT" if usdt_selection_mean > usdc_selection_mean else "USDC"
    long_quote = "USDC" if short_quote == "USDT" else "USDT"
    short_symbol = str(contracts[short_quote]["symbol"])
    long_symbol = str(contracts[long_quote]["symbol"])
    history_by_symbol = {usdt_symbol: usdt_history, usdc_symbol: usdc_history}
    differences = [
        history_by_symbol[short_symbol][item] - history_by_symbol[long_symbol][item]
        for item in evaluation_times
    ]
    evaluation_split = len(differences) // 2
    chronological_halves = [
        differences[:evaluation_split],
        differences[evaluation_split:],
    ]
    monthly: dict[str, list[Decimal]] = defaultdict(list)
    for funding_time, difference in zip(evaluation_times, differences, strict=True):
        monthly[_month_key(funding_time)].append(difference)
    month_keys = sorted(monthly)
    complete_month_keys = month_keys[1:-1]
    monthly_rows = [
        {
            "month": month,
            "complete_inner_month": month in complete_month_keys,
            **_statistics(monthly[month]),
        }
        for month in month_keys
    ]
    common_quantity = _common_quantity((contracts["USDT"], contracts["USDC"]))
    short_book = _book(book_by_symbol[short_symbol], expected_symbol=short_symbol)
    long_book = _book(book_by_symbol[long_symbol], expected_symbol=long_symbol)
    depth_passed = all(
        quantity >= common_quantity
        for quantity in (
            short_book["bid_quantity"],
            short_book["ask_quantity"],
            long_book["bid_quantity"],
            long_book["ask_quantity"],
        )
    )
    short_mid = (short_book["bid"] + short_book["ask"]) / 2
    long_mid = (long_book["bid"] + long_book["ask"]) / 2
    spread_hurdle_bips = (
        (short_book["ask"] - short_book["bid"]) / short_mid
        + (long_book["ask"] - long_book["bid"]) / long_mid
    ) * Decimal("10000")
    evaluation_sum_bips = sum(differences, Decimal("0")) * Decimal("10000")
    equal_fee_break_even = (evaluation_sum_bips - spread_hurdle_bips) / 4
    short_premium = _mapping(premium_by_symbol[short_symbol], name="short premium row")
    long_premium = _mapping(premium_by_symbol[long_symbol], name="long premium row")
    short_mark = _decimal(
        short_premium.get("markPrice"), name="short mark", positive=True
    )
    long_mark = _decimal(long_premium.get("markPrice"), name="long mark", positive=True)
    mark_divergence_bips = (
        abs(short_mark - long_mark) / ((short_mark + long_mark) / 2) * Decimal("10000")
    )
    failure_reasons: list[str] = []
    if any(sum(part, Decimal("0")) <= 0 for part in chronological_halves):
        failure_reasons.append("evaluation_chronological_half_nonpositive")
    if not complete_month_keys:
        failure_reasons.append("no_complete_inner_evaluation_month")
    elif any(sum(monthly[key], Decimal("0")) < 0 for key in complete_month_keys):
        failure_reasons.append("complete_evaluation_month_negative")
    if evaluation_sum_bips <= 0:
        failure_reasons.append("evaluation_total_nonpositive")
    if equal_fee_break_even <= 0:
        failure_reasons.append("no_positive_fee_break_even_after_current_spreads")
    if not depth_passed:
        failure_reasons.append("current_common_quantity_depth_missing")
    return {
        "base_asset": base,
        "symbols": {"short": short_symbol, "long": long_symbol},
        "quote_orientation": {"short": short_quote, "long": long_quote},
        "aligned_row_count": len(aligned_times),
        "aligned_start_time_ms": aligned_times[0],
        "aligned_end_time_ms": aligned_times[-1],
        "selection": {
            "row_count": len(selection_times),
            "start_time_ms": selection_times[0],
            "end_time_ms": selection_times[-1],
            "usdt_mean_rate": str(usdt_selection_mean),
            "usdc_mean_rate": str(usdc_selection_mean),
        },
        "evaluation": {
            "row_count": len(evaluation_times),
            "start_time_ms": evaluation_times[0],
            "end_time_ms": evaluation_times[-1],
            "statistics": _statistics(differences),
            "first_chronological_half": _statistics(chronological_halves[0]),
            "second_chronological_half": _statistics(chronological_halves[1]),
            "calendar_months": monthly_rows,
            "worst_cumulative_rate_drawdown_bips": str(_drawdown_bips(differences)),
            "cumulative_oriented_funding_bips": str(evaluation_sum_bips),
        },
        "current_execution_diagnostic": {
            "common_minimum_quantity": str(common_quantity),
            "all_four_top_sides_cover_common_quantity": depth_passed,
            "two_book_round_trip_spread_hurdle_bips": str(spread_hurdle_bips),
            "equal_per_leg_fee_break_even_after_spread_bips": str(equal_fee_break_even),
            "short_book": {key: str(value) for key, value in short_book.items()},
            "long_book": {key: str(value) for key, value in long_book.items()},
            "short_mark_price": str(short_mark),
            "long_mark_price": str(long_mark),
            "mark_price_divergence_bips": str(mark_divergence_bips),
            "stablecoin_fx_basis_assumed_zero": False,
        },
        "failure_reasons": failure_reasons,
        "qualified_public_screen": not failure_reasons,
    }


def run(*, session: requests.Session | None = None) -> dict[str, object]:
    """Run the frozen public screen without authentication or orders."""

    contract = _load_contract()
    started_ms = time.time_ns() // 1_000_000
    http = session or requests.Session()
    request_ledger: list[dict[str, object]] = []
    exchange_raw, exchange_receipt = _get(http, "/fapi/v1/exchangeInfo")
    request_ledger.append(exchange_receipt)
    contracts = _selected_contracts(exchange_raw)
    premium_raw, premium_receipt = _get(http, "/fapi/v1/premiumIndex")
    request_ledger.append(premium_receipt)
    book_raw, book_receipt = _get(http, "/fapi/v1/ticker/bookTicker")
    request_ledger.append(book_receipt)
    premium_by_symbol = _rows_by_symbol(premium_raw, name="premium index")
    book_by_symbol = _rows_by_symbol(book_raw, name="book ticker")
    histories: dict[str, object] = {}
    for base in sorted(contracts):
        for quote in QUOTES:
            symbol = str(contracts[base][quote]["symbol"])
            history_raw, history_receipt = _get(
                http,
                "/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": HISTORY_LIMIT},
            )
            request_ledger.append(history_receipt)
            histories[symbol] = history_raw
    if len(request_ledger) > MAXIMUM_REQUESTS:
        raise ValueError("frozen request limit was exceeded")
    pair_results = [
        _evaluate_pair(
            base=base,
            contracts=contracts[base],
            histories=histories,
            premium_by_symbol=premium_by_symbol,
            book_by_symbol=book_by_symbol,
        )
        for base in sorted(contracts)
    ]
    qualified = [row for row in pair_results if row["qualified_public_screen"]]
    finished_ms = time.time_ns() // 1_000_000
    selected_symbols = {
        str(contract["symbol"])
        for by_quote in contracts.values()
        for contract in by_quote.values()
    }
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.fromtimestamp(
            finished_ms / 1000, tz=UTC
        ).isoformat(),
        "started_at_ms": started_ms,
        "finished_at_ms": finished_ms,
        "request_count": len(request_ledger),
        "source_contract": {
            "contract_path": str(CONTRACT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "contract_result_sha256": contract["result_sha256"],
            "implementation_path": str(
                Path(__file__).resolve().relative_to(ROOT)
            ).replace("\\", "/"),
            "implementation_sha256": _sha256(Path(__file__).read_bytes()),
            "request_ledger": request_ledger,
            "exchange_info_payload": exchange_raw,
            "premium_index_selected_payload": [
                row
                for row in _list(premium_raw, name="premium index")
                if isinstance(row, Mapping) and row.get("symbol") in selected_symbols
            ],
            "book_ticker_selected_payload": [
                row
                for row in _list(book_raw, name="book ticker")
                if isinstance(row, Mapping) and row.get("symbol") in selected_symbols
            ],
            "funding_history_payloads": histories,
        },
        "eligible_contracts": contracts,
        "pair_results": pair_results,
        "verdict": {
            "status": (
                "public_persistence_candidate_requires_authenticated_cost_and_collateral_evidence"
                if qualified
                else "rejected_public_cross_stablecoin_funding_differential"
            ),
            "eligible_pair_count": len(pair_results),
            "qualified_public_screen_count": len(qualified),
            "accepted_edge": False,
            "profitability_claim": False,
            "trading_authority": False,
            "credentials_used": False,
            "orders_placed": False,
        },
    }
    artifact["result_sha256"] = _sha256(_canonical_json(artifact).encode("ascii"))
    return artifact


def _failure(error: Exception, *, started_ms: int) -> dict[str, object]:
    finished_ms = time.time_ns() // 1_000_000
    result: dict[str, object] = {
        "schema_version": f"{SCHEMA_VERSION}-terminal-failure-v1",
        "created_at_utc": datetime.fromtimestamp(
            finished_ms / 1000, tz=UTC
        ).isoformat(),
        "started_at_ms": started_ms,
        "finished_at_ms": finished_ms,
        "error_type": type(error).__name__,
        "error": str(error),
        "accepted_edge": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started_ms = time.time_ns() // 1_000_000
    try:
        result = run()
    except Exception as exc:
        result = _failure(exc, started_ms=started_ms)
        payload = (_canonical_json(result) + "\n").encode("ascii")
        write_bytes_atomic(args.output, payload)
        print(f"terminal_failure={type(exc).__name__}: {exc}")
        print(f"output={args.output}")
        return 1
    payload = (_canonical_json(result) + "\n").encode("ascii")
    write_bytes_atomic(args.output, payload)
    print(f"request_count={result['request_count']}")
    print(f"eligible_pair_count={result['verdict']['eligible_pair_count']}")
    print(
        "qualified_public_screen_count="
        f"{result['verdict']['qualified_public_screen_count']}"
    )
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
