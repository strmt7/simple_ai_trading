"""Screen Binance BTC/ETH/SOL options for vertical and convexity parity."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

import simple_ai_trading.option_parity as option_parity_module
from simple_ai_trading.option_parity import (
    OptionBookLevel,
    OptionContractQuote,
    OptionDepthQuote,
    OptionParityCandidate,
    confirm_option_candidate,
    discover_option_parity,
)
from simple_ai_trading.storage import write_bytes_atomic


SCHEMA_VERSION = "binance-option-parity-screen-v1"
BINANCE_OPTION_BASE_URL = "https://eapi.binance.com"
ALLOWED_UNDERLYINGS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT"})
DEPTH_LIMIT = 100
MAX_TICKER_POSITIVE_CANDIDATES = 25
MAX_UNIQUE_DEPTH_SYMBOLS = 75
MAX_BOOK_AGE_MS = 5_000
MAX_BOOK_SKEW_MS = 1_000
DEFAULT_CONFIRMATION_SWEEPS = 3
DEFAULT_CONFIRMATION_DELAY_SECONDS = 2.0


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


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _positive_decimal_or_none(value: object) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed > 0 else None


def _get(
    session: requests.Session,
    url: str,
) -> tuple[object, dict[str, object]]:
    before_ms = time.time_ns() // 1_000_000
    response = session.get(url, timeout=30)
    after_ms = time.time_ns() // 1_000_000
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        raise RuntimeError(
            "Binance rate limit reached; stopped without retry"
            + (f"; Retry-After={retry_after}" if retry_after else "")
        )
    response.raise_for_status()
    payload = response.content
    try:
        decoded = response.json()
    except requests.JSONDecodeError as exc:
        raise ValueError(f"source {url} did not return JSON") from exc
    return decoded, {
        "url": url,
        "payload_sha256": _sha256(payload),
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "request_elapsed_ms": after_ms - before_ms,
        "x_mbx_used_weight_1m": response.headers.get("X-MBX-USED-WEIGHT-1M"),
    }


def _filter_map(raw: object, *, symbol: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for item in _list(raw, name=f"Binance {symbol} filters"):
        parsed = _mapping(item, name=f"Binance {symbol} filter")
        filter_type = str(parsed.get("filterType") or "")
        if not filter_type or filter_type in result:
            raise ValueError(f"Binance {symbol} filters are ambiguous")
        result[filter_type] = parsed
    return result


def _contract_payload(contract: OptionContractQuote) -> dict[str, object]:
    return {
        "symbol": contract.symbol,
        "underlying": contract.underlying,
        "expiry_date_ms": contract.expiry_date_ms,
        "side": contract.side,
        "strike": _decimal_text(contract.strike),
        "unit": _decimal_text(contract.unit),
        "minimum_quantity": _decimal_text(contract.minimum_quantity),
        "maximum_quantity": _decimal_text(contract.maximum_quantity),
        "step_size": _decimal_text(contract.step_size),
        "bid_price": _decimal_text(contract.bid_price),
        "ask_price": _decimal_text(contract.ask_price),
    }


def _candidate_payload(candidate: OptionParityCandidate) -> dict[str, object]:
    return {
        "mechanism": candidate.mechanism,
        "symbols": list(candidate.symbols),
        "roles": list(candidate.roles),
        "strikes": [_decimal_text(value) for value in candidate.strikes],
        "integer_weights": list(candidate.integer_weights),
        "minimum_quantities": [_decimal_text(value) for value in candidate.quantities],
        "ticker_gross_credit_quote": _decimal_text(candidate.gross_credit_quote),
        "expiry_payoff_floor_quote": "0",
    }


def _depth_quote(symbol: str, raw: object) -> OptionDepthQuote:
    payload = _mapping(raw, name=f"Binance {symbol} depth")

    def levels(name: str) -> tuple[OptionBookLevel, ...]:
        parsed: list[OptionBookLevel] = []
        for raw_level in _list(payload.get(name), name=f"Binance {symbol} {name}"):
            if not isinstance(raw_level, list) or len(raw_level) != 2:
                raise ValueError(f"Binance {symbol} {name} level is invalid")
            parsed.append(
                OptionBookLevel(
                    price=Decimal(str(raw_level[0])),
                    quantity=Decimal(str(raw_level[1])),
                )
            )
        return tuple(parsed)

    event_time = payload.get("T")
    if isinstance(event_time, bool) or not isinstance(event_time, int):
        raise ValueError(f"Binance {symbol} depth timestamp is invalid")
    return OptionDepthQuote(
        symbol=symbol,
        event_time_ms=event_time,
        bids=levels("bids"),
        asks=levels("asks"),
    ).validated()


def _depth_payload(book: OptionDepthQuote) -> dict[str, object]:
    return {
        "symbol": book.symbol,
        "event_time_ms": book.event_time_ms,
        "bids": [
            [_decimal_text(level.price), _decimal_text(level.quantity)]
            for level in book.bids
        ],
        "asks": [
            [_decimal_text(level.price), _decimal_text(level.quantity)]
            for level in book.asks
        ],
    }


def _fetch_depth_sweep(
    session: requests.Session,
    *,
    candidates: tuple[OptionParityCandidate, ...],
    sweep_index: int,
) -> tuple[dict[str, object], tuple[OptionParityCandidate, ...]]:
    symbols = tuple(sorted({symbol for item in candidates for symbol in item.symbols}))
    if len(symbols) > MAX_UNIQUE_DEPTH_SYMBOLS:
        raise ValueError(
            "ticker-positive option candidates exceed the depth request budget"
        )
    books: dict[str, OptionDepthQuote] = {}
    request_evidence: list[dict[str, object]] = []
    for symbol in symbols:
        url = (
            f"{BINANCE_OPTION_BASE_URL}/eapi/v1/depth"
            f"?symbol={symbol}&limit={DEPTH_LIMIT}"
        )
        raw, source = _get(session, url)
        books[symbol] = _depth_quote(symbol, raw)
        request_evidence.append(source)
    server_raw, server_source = _get(session, f"{BINANCE_OPTION_BASE_URL}/eapi/v1/time")
    server = _mapping(server_raw, name="Binance option server time")
    server_time = server.get("serverTime")
    if isinstance(server_time, bool) or not isinstance(server_time, int):
        raise ValueError("Binance option server time is invalid")

    rows: list[dict[str, object]] = []
    passing: list[OptionParityCandidate] = []
    for candidate in candidates:
        confirmation = confirm_option_candidate(candidate, books)
        times = confirmation.book_event_times_ms
        skew = max(times) - min(times) if times else None
        maximum_age = max(server_time - value for value in times) if times else None
        timestamps_not_future = bool(times) and max(times) <= server_time
        fresh = (
            confirmation.executable
            and skew is not None
            and maximum_age is not None
            and timestamps_not_future
            and skew <= MAX_BOOK_SKEW_MS
            and maximum_age <= MAX_BOOK_AGE_MS
        )
        gross_positive = bool(
            confirmation.gross_credit_quote is not None
            and confirmation.gross_credit_quote > 0
        )
        passed = fresh and gross_positive
        if passed:
            passing.append(candidate)
        rows.append(
            {
                "symbols": list(candidate.symbols),
                "executable": confirmation.executable,
                "gross_credit_quote": _decimal_text(confirmation.gross_credit_quote),
                "book_event_times_ms": list(times),
                "book_skew_ms": skew,
                "maximum_book_age_ms": maximum_age,
                "timestamps_not_future": timestamps_not_future,
                "freshness_gate_passed": fresh,
                "gross_positive": gross_positive,
                "sweep_passed": passed,
            }
        )
    return (
        {
            "sweep_index": sweep_index,
            "server_time_ms": server_time,
            "server_time_request": server_source,
            "depth_requests": request_evidence,
            "books": [_depth_payload(books[symbol]) for symbol in symbols],
            "candidate_results": rows,
            "passing_candidate_count": len(passing),
        },
        tuple(passing),
    )


def run(
    *,
    confirmation_sweeps: int = DEFAULT_CONFIRMATION_SWEEPS,
    confirmation_delay_seconds: float = DEFAULT_CONFIRMATION_DELAY_SECONDS,
) -> dict[str, object]:
    if (
        isinstance(confirmation_sweeps, bool)
        or not isinstance(confirmation_sweeps, int)
        or confirmation_sweeps < 1
    ):
        raise ValueError("confirmation sweeps must be a positive integer")
    if confirmation_sweeps > 5:
        raise ValueError("confirmation sweeps exceed the request budget")
    if isinstance(confirmation_delay_seconds, bool) or not (
        0 <= confirmation_delay_seconds <= 30
    ):
        raise ValueError("confirmation delay is outside [0, 30] seconds")
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-option-parity-research/1.0",
        }
    )
    exchange_url = f"{BINANCE_OPTION_BASE_URL}/eapi/v1/exchangeInfo"
    exchange_raw, exchange_source = _get(session, exchange_url)
    exchange = _mapping(exchange_raw, name="Binance option exchangeInfo")
    ticker_url = f"{BINANCE_OPTION_BASE_URL}/eapi/v1/ticker"
    ticker_raw, ticker_source = _get(session, ticker_url)
    ticker_rows = _list(ticker_raw, name="Binance option ticker")
    tickers: dict[str, dict[str, object]] = {}
    for raw_ticker in ticker_rows:
        ticker = _mapping(raw_ticker, name="Binance option ticker row")
        symbol = str(ticker.get("symbol") or "").strip().upper()
        if not symbol or symbol in tickers:
            raise ValueError("Binance option ticker identity is ambiguous")
        tickers[symbol] = ticker

    contract_terms: dict[str, dict[str, object]] = {}
    for raw_term in _list(
        exchange.get("optionContracts"),
        name="Binance option contract terms",
    ):
        term = _mapping(raw_term, name="Binance option contract term")
        underlying = str(term.get("underlying") or "").strip().upper()
        if underlying in ALLOWED_UNDERLYINGS:
            if underlying in contract_terms:
                raise ValueError("Binance option contract term is duplicated")
            if not isinstance(term.get("nakedSell"), bool):
                raise ValueError("Binance option nakedSell term is invalid")
            contract_terms[underlying] = term
    if set(contract_terms) != set(ALLOWED_UNDERLYINGS):
        raise ValueError("Binance option contract terms omit a scoped underlying")

    contracts: list[OptionContractQuote] = []
    excluded_non_unit: list[str] = []
    missing_ticker: list[str] = []
    for raw_symbol in _list(
        exchange.get("optionSymbols"),
        name="Binance option symbols",
    ):
        symbol = _mapping(raw_symbol, name="Binance option symbol")
        name = str(symbol.get("symbol") or "").strip().upper()
        underlying = str(symbol.get("underlying") or "").strip().upper()
        if not name:
            raise ValueError("Binance option symbol identity is absent")
        if (
            underlying not in ALLOWED_UNDERLYINGS
            or symbol.get("status") != "TRADING"
            or symbol.get("contractType") != "CRYPTO_OPTIONS"
        ):
            continue
        filters = _filter_map(symbol.get("filters"), symbol=name)
        lot = filters.get("LOT_SIZE")
        if lot is None:
            raise ValueError(f"Binance {name} LOT_SIZE filter is absent")
        unit = _positive_decimal_or_none(symbol.get("unit"))
        if unit != Decimal("1"):
            excluded_non_unit.append(name)
            continue
        ticker = tickers.get(name)
        if ticker is None:
            missing_ticker.append(name)
            bid = None
            ask = None
        else:
            bid = _positive_decimal_or_none(ticker.get("bidPrice"))
            ask = _positive_decimal_or_none(ticker.get("askPrice"))
        contracts.append(
            OptionContractQuote(
                symbol=name,
                underlying=underlying,
                expiry_date_ms=symbol.get("expiryDate"),  # type: ignore[arg-type]
                side=str(symbol.get("side") or ""),
                strike=Decimal(str(symbol.get("strikePrice"))),
                unit=unit,
                minimum_quantity=Decimal(str(lot.get("minQty"))),
                maximum_quantity=Decimal(str(lot.get("maxQty"))),
                step_size=Decimal(str(lot.get("stepSize"))),
                bid_price=bid,
                ask_price=ask,
            ).validated()
        )
    if not contracts:
        raise ValueError("Binance returned no scoped tradable unit-one options")
    discovery = discover_option_parity(contracts)
    candidates = discovery.gross_positive_candidates
    request_budget_exceeded = len(candidates) > MAX_TICKER_POSITIVE_CANDIDATES

    sweeps: list[dict[str, object]] = []
    surviving: tuple[OptionParityCandidate, ...] = ()
    if candidates and not request_budget_exceeded:
        first, surviving = _fetch_depth_sweep(
            session,
            candidates=candidates,
            sweep_index=1,
        )
        sweeps.append(first)
        for sweep_index in range(2, confirmation_sweeps + 1):
            if not surviving:
                break
            if confirmation_delay_seconds:
                time.sleep(confirmation_delay_seconds)
            current, surviving = _fetch_depth_sweep(
                session,
                candidates=surviving,
                sweep_index=sweep_index,
            )
            sweeps.append(current)

    initial_positive = int(sweeps[0]["passing_candidate_count"]) if sweeps else 0
    persistent_positive = len(surviving) if len(sweeps) == confirmation_sweeps else 0
    if request_budget_exceeded:
        status = "rejected_ticker_candidates_exceed_request_budget"
    elif not candidates:
        status = "rejected_no_gross_positive_ticker_candidate"
    elif not sweeps or initial_positive == 0:
        status = "rejected_no_fresh_gross_positive_depth_candidate"
    elif len(sweeps) < confirmation_sweeps or persistent_positive == 0:
        status = "rejected_not_persistent_across_confirmation_sweeps"
    else:
        status = "unqualified_gross_positive_requires_fees_margin_atomicity"

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "target_free_market_direction_independent_option_payoff_parity",
        "mechanics_contract": {
            "identities": [
                "call_vertical_buy_lower_sell_higher_has_nonnegative_expiry_payoff",
                "put_vertical_buy_higher_sell_lower_has_nonnegative_expiry_payoff",
                "strike_convexity_weighted_wings_minus_middle_has_nonnegative_expiry_payoff",
            ],
            "official_references": [
                "https://developers.binance.com/en/docs/products/derivatives-trading-options/Introduction",
                "https://developers.binance.com/en/docs/products/derivatives-trading-options/common-definition",
                "https://academy.binance.com/en/articles/what-is-options-trading",
            ],
            "settlement_mechanics_used": "European-style cash-settled plain-vanilla call and put payoff",
        },
        "scope": {
            "underlyings": sorted(ALLOWED_UNDERLYINGS),
            "allowed_status": "TRADING",
            "allowed_contract_type": "CRYPTO_OPTIONS",
            "required_unit": "1",
            "depth_limit": DEPTH_LIMIT,
            "maximum_ticker_positive_candidates": MAX_TICKER_POSITIVE_CANDIDATES,
            "maximum_unique_depth_symbols": MAX_UNIQUE_DEPTH_SYMBOLS,
            "maximum_book_age_ms": MAX_BOOK_AGE_MS,
            "maximum_book_skew_ms": MAX_BOOK_SKEW_MS,
            "confirmation_sweeps_requested": confirmation_sweeps,
            "confirmation_delay_seconds": confirmation_delay_seconds,
        },
        "source_contract": {
            "exchange_info": exchange_source,
            "ticker": ticker_source,
            "exchange_info_server_time_ms": exchange.get("serverTime"),
            "catalog_fetched_once": True,
            "all_symbol_ticker_fetched_once": True,
            "depth_requested_only_for_ticker_positive_candidates": True,
            "stopped_repeating_after_no_passing_candidate": True,
            "implementation": {
                "tool_path": Path(__file__).name,
                "tool_sha256": _sha256(Path(__file__).read_bytes()),
                "module_path": Path(option_parity_module.__file__).name,
                "module_sha256": _sha256(
                    Path(option_parity_module.__file__).read_bytes()
                ),
            },
        },
        "contract_terms": {
            underlying: {
                "base_asset": term.get("baseAsset"),
                "quote_asset": term.get("quoteAsset"),
                "settle_asset": term.get("settleAsset"),
                "naked_sell": term.get("nakedSell"),
            }
            for underlying, term in sorted(contract_terms.items())
        },
        "contracts": [_contract_payload(contract) for contract in contracts],
        "exclusions": {
            "non_unit_one_symbols": sorted(excluded_non_unit),
            "missing_ticker_symbols": sorted(missing_ticker),
        },
        "discovery": {
            "scoped_contract_count": len(contracts),
            "chain_count": len(
                {
                    (item.underlying, item.expiry_date_ms, item.side)
                    for item in contracts
                }
            ),
            "evaluated_vertical_count": discovery.evaluated_vertical_count,
            "executable_vertical_count": discovery.executable_vertical_count,
            "evaluated_convexity_count": discovery.evaluated_convexity_count,
            "executable_convexity_count": discovery.executable_convexity_count,
            "ticker_gross_positive_candidate_count": len(candidates),
            "request_budget_exceeded": request_budget_exceeded,
            "ticker_gross_positive_candidates": [
                _candidate_payload(candidate) for candidate in candidates
            ],
        },
        "depth_confirmation_sweeps": sweeps,
        "verdict": {
            "status": status,
            "initial_fresh_gross_positive_candidate_count": initial_positive,
            "persistent_gross_positive_candidate_count": persistent_positive,
            "exact_account_commission_verified": False,
            "account_margin_and_short_inventory_verified": False,
            "atomic_multi_leg_execution_verified": False,
            "accepted_edge": False,
            "trading_authority": False,
        },
        "safety": {
            "public_market_data_only": True,
            "credentials_used": False,
            "orders_placed": False,
            "ticker_is_discovery_not_execution_evidence": True,
            "public_books_prove_fills": False,
        },
        "limitations": [
            "Ticker prices have no displayed quantities and only select candidates for depth confirmation.",
            "REST depth legs are not an atomic multi-leg snapshot or execution guarantee.",
            "Exact account commission requires authenticated USER_DATA evidence and was not available to this public-only tool.",
            "Short-option margin, inventory, capital, partial-fill risk, and exercise handling remain unverified.",
            "A gross-positive public snapshot cannot establish an accepted after-cost edge.",
        ],
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirmation-sweeps",
        type=int,
        default=DEFAULT_CONFIRMATION_SWEEPS,
    )
    parser.add_argument(
        "--confirmation-delay-seconds",
        type=float,
        default=DEFAULT_CONFIRMATION_DELAY_SECONDS,
    )
    args = parser.parse_args()
    result = run(
        confirmation_sweeps=args.confirmation_sweeps,
        confirmation_delay_seconds=args.confirmation_delay_seconds,
    )
    write_bytes_atomic(
        args.output,
        (_canonical_json(result) + "\n").encode("ascii"),
    )
    print(json.dumps(result["verdict"], indent=2))
    print(
        "ticker_gross_positive_candidate_count="
        f"{result['discovery']['ticker_gross_positive_candidate_count']}"
    )
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
