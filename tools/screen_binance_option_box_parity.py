"""Confirm fixed-payoff Binance option boxes from a source-bound snapshot."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

import simple_ai_trading.option_box_parity as option_box_module
from simple_ai_trading.option_box_parity import (
    OptionBoxCandidate,
    confirm_option_box,
    discover_option_boxes,
)
from simple_ai_trading.option_parity import (
    OptionBookLevel,
    OptionContractQuote,
    OptionDepthQuote,
)
from simple_ai_trading.storage import write_bytes_atomic


SCHEMA_VERSION = "binance-option-box-parity-screen-v1"
SOURCE_SCHEMA_VERSION = "binance-option-parity-screen-v1"
BINANCE_OPTION_BASE_URL = "https://eapi.binance.com"
DEPTH_LIMIT = 100
MAX_CANDIDATES = 25
MAX_UNIQUE_DEPTH_SYMBOLS = 100
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


def _load_source(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("source option snapshot is not JSON") from exc
    source = _mapping(report, name="source option snapshot")
    if source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("source option snapshot schema is unsupported")
    claimed = source.pop("result_sha256", None)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("source option snapshot result hash is invalid")
    reconstructed = _sha256(_canonical_json(source).encode("ascii"))
    if reconstructed != claimed:
        raise ValueError("source option snapshot result hash does not reconstruct")
    source["result_sha256"] = claimed
    return source, _sha256(raw)


def _contract(row: object) -> OptionContractQuote:
    item = _mapping(row, name="source option contract")
    return OptionContractQuote(
        symbol=str(item.get("symbol") or ""),
        underlying=str(item.get("underlying") or ""),
        expiry_date_ms=item.get("expiry_date_ms"),  # type: ignore[arg-type]
        side=str(item.get("side") or ""),
        strike=Decimal(str(item.get("strike"))),
        unit=Decimal(str(item.get("unit"))),
        minimum_quantity=Decimal(str(item.get("minimum_quantity"))),
        maximum_quantity=Decimal(str(item.get("maximum_quantity"))),
        step_size=Decimal(str(item.get("step_size"))),
        bid_price=(
            None
            if item.get("bid_price") is None
            else Decimal(str(item.get("bid_price")))
        ),
        ask_price=(
            None
            if item.get("ask_price") is None
            else Decimal(str(item.get("ask_price")))
        ),
    ).validated()


def _candidate_payload(candidate: OptionBoxCandidate) -> dict[str, object]:
    return {
        "kind": candidate.kind,
        "underlying": candidate.underlying,
        "expiry_date_ms": candidate.expiry_date_ms,
        "lower_strike": _decimal_text(candidate.lower_strike),
        "upper_strike": _decimal_text(candidate.upper_strike),
        "symbols": list(candidate.symbols),
        "roles": list(candidate.roles),
        "minimum_quantity": _decimal_text(candidate.quantity),
        "fixed_expiry_cashflow_quote": _decimal_text(
            candidate.fixed_expiry_cashflow_quote
        ),
        "ticker_initial_credit_quote": _decimal_text(candidate.initial_credit_quote),
        "ticker_gross_expiry_profit_quote": _decimal_text(
            candidate.gross_expiry_profit_quote
        ),
        "annualized_simple_return_before_costs": _decimal_text(
            candidate.annualized_simple_return
        ),
    }


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
    try:
        decoded = response.json()
    except requests.JSONDecodeError as exc:
        raise ValueError(f"source {url} did not return JSON") from exc
    return decoded, {
        "url": url,
        "payload_sha256": _sha256(response.content),
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "request_elapsed_ms": after_ms - before_ms,
        "x_mbx_used_weight_1m": response.headers.get("X-MBX-USED-WEIGHT-1M"),
    }


def _depth_quote(symbol: str, raw: object) -> OptionDepthQuote:
    payload = _mapping(raw, name=f"Binance {symbol} depth")

    def levels(name: str) -> tuple[OptionBookLevel, ...]:
        result: list[OptionBookLevel] = []
        for raw_level in _list(payload.get(name), name=f"Binance {symbol} {name}"):
            if not isinstance(raw_level, list) or len(raw_level) != 2:
                raise ValueError(f"Binance {symbol} {name} level is invalid")
            result.append(
                OptionBookLevel(
                    price=Decimal(str(raw_level[0])),
                    quantity=Decimal(str(raw_level[1])),
                )
            )
        return tuple(result)

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


def _depth_sweep(
    session: requests.Session,
    *,
    candidates: tuple[OptionBoxCandidate, ...],
    sweep_index: int,
) -> tuple[dict[str, object], tuple[OptionBoxCandidate, ...]]:
    symbols = tuple(sorted({symbol for item in candidates for symbol in item.symbols}))
    if len(symbols) > MAX_UNIQUE_DEPTH_SYMBOLS:
        raise ValueError("option box candidates exceed the depth request budget")
    books: dict[str, OptionDepthQuote] = {}
    requests_evidence: list[dict[str, object]] = []
    for symbol in symbols:
        url = (
            f"{BINANCE_OPTION_BASE_URL}/eapi/v1/depth"
            f"?symbol={symbol}&limit={DEPTH_LIMIT}"
        )
        raw, source = _get(session, url)
        books[symbol] = _depth_quote(symbol, raw)
        requests_evidence.append(source)
    server_raw, server_source = _get(session, f"{BINANCE_OPTION_BASE_URL}/eapi/v1/time")
    server = _mapping(server_raw, name="Binance option server time")
    server_time = server.get("serverTime")
    if isinstance(server_time, bool) or not isinstance(server_time, int):
        raise ValueError("Binance option server time is invalid")

    rows: list[dict[str, object]] = []
    passing: list[OptionBoxCandidate] = []
    for candidate in candidates:
        confirmation = confirm_option_box(candidate, books)
        times = tuple(books[symbol].event_time_ms for symbol in candidate.symbols)
        skew = max(times) - min(times)
        maximum_age = max(server_time - value for value in times)
        fresh = (
            max(times) <= server_time
            and skew <= MAX_BOOK_SKEW_MS
            and maximum_age <= MAX_BOOK_AGE_MS
        )
        gross_positive = bool(
            confirmation.gross_expiry_profit_quote is not None
            and confirmation.gross_expiry_profit_quote > 0
        )
        passed = confirmation.executable and fresh and gross_positive
        if passed:
            passing.append(candidate)
        rows.append(
            {
                "kind": candidate.kind,
                "symbols": list(candidate.symbols),
                "executable": confirmation.executable,
                "initial_credit_quote": _decimal_text(
                    confirmation.initial_credit_quote
                ),
                "gross_expiry_profit_quote": _decimal_text(
                    confirmation.gross_expiry_profit_quote
                ),
                "book_event_times_ms": list(times),
                "book_skew_ms": skew,
                "maximum_book_age_ms": maximum_age,
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
            "depth_requests": requests_evidence,
            "books": [_depth_payload(books[symbol]) for symbol in symbols],
            "candidate_results": rows,
            "passing_candidate_count": len(passing),
        },
        tuple(passing),
    )


def run(
    *,
    source_snapshot: Path,
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

    source, source_file_hash = _load_source(source_snapshot)
    source_contract = _mapping(
        source.get("source_contract"), name="source option contract"
    )
    as_of_ms = source_contract.get("exchange_info_server_time_ms")
    if isinstance(as_of_ms, bool) or not isinstance(as_of_ms, int):
        raise ValueError("source option snapshot as-of time is invalid")
    contracts = tuple(
        _contract(row)
        for row in _list(source.get("contracts"), name="source option contracts")
    )
    discovery = discover_option_boxes(contracts, as_of_ms=as_of_ms)
    candidates = discovery.candidates
    request_budget_exceeded = len(candidates) > MAX_CANDIDATES

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-option-box-research/1.0",
        }
    )
    sweeps: list[dict[str, object]] = []
    surviving: tuple[OptionBoxCandidate, ...] = ()
    if candidates and not request_budget_exceeded:
        first, surviving = _depth_sweep(
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
            current, surviving = _depth_sweep(
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
        status = "rejected_no_ticker_positive_box_candidate"
    elif not sweeps or initial_positive == 0:
        status = "rejected_no_fresh_executable_depth_positive_box"
    elif len(sweeps) < confirmation_sweeps or persistent_positive == 0:
        status = "rejected_not_persistent_across_confirmation_sweeps"
    else:
        status = "unqualified_gross_positive_requires_costs_margin_atomicity"

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "target_free_market_direction_independent_fixed_payoff_option_boxes",
        "source_contract": {
            "source_snapshot_path": source_snapshot.name,
            "source_snapshot_file_sha256": source_file_hash,
            "source_snapshot_result_sha256": source["result_sha256"],
            "source_snapshot_as_of_ms": as_of_ms,
            "full_option_catalog_refetched": False,
            "all_symbol_ticker_refetched": False,
            "depth_requested_only_for_source_ticker_candidates": True,
            "implementation": {
                "tool_path": Path(__file__).name,
                "tool_sha256": _sha256(Path(__file__).read_bytes()),
                "module_path": Path(option_box_module.__file__).name,
                "module_sha256": _sha256(Path(option_box_module.__file__).read_bytes()),
            },
        },
        "payoff_contract": {
            "long_box": "buy_lower_call_sell_upper_call_buy_upper_put_sell_lower_put",
            "short_box": "sell_lower_call_buy_upper_call_sell_upper_put_buy_lower_put",
            "fixed_cashflow_absolute_quote": "(upper_strike-lower_strike)*unit*quantity",
            "long_box_cashflow_sign": "positive_at_expiry",
            "short_box_cashflow_sign": "negative_at_expiry",
            "unit_required": "1",
        },
        "scope": {
            "depth_limit": DEPTH_LIMIT,
            "maximum_candidates": MAX_CANDIDATES,
            "maximum_unique_depth_symbols": MAX_UNIQUE_DEPTH_SYMBOLS,
            "maximum_book_age_ms": MAX_BOOK_AGE_MS,
            "maximum_book_skew_ms": MAX_BOOK_SKEW_MS,
            "confirmation_sweeps_requested": confirmation_sweeps,
            "confirmation_delay_seconds": confirmation_delay_seconds,
        },
        "discovery": {
            "chain_count": discovery.chain_count,
            "evaluated_strike_pair_count": discovery.evaluated_strike_pair_count,
            "executable_long_box_count": discovery.executable_long_box_count,
            "executable_short_box_count": discovery.executable_short_box_count,
            "nominal_positive_long_box_count": len(
                discovery.nominal_positive_long_boxes
            ),
            "strict_positive_short_box_count": len(
                discovery.strict_positive_short_boxes
            ),
            "ticker_candidate_count": len(candidates),
            "request_budget_exceeded": request_budget_exceeded,
            "ticker_candidates": [_candidate_payload(item) for item in candidates],
        },
        "depth_confirmation_sweeps": sweeps,
        "verdict": {
            "status": status,
            "initial_fresh_executable_positive_count": initial_positive,
            "persistent_positive_count": persistent_positive,
            "exact_account_commission_verified": False,
            "financing_and_opportunity_cost_verified": False,
            "account_margin_and_short_inventory_verified": False,
            "atomic_multi_leg_execution_verified": False,
            "accepted_edge": False,
            "trading_authority": False,
        },
        "safety": {
            "public_market_data_only": True,
            "credentials_used": False,
            "orders_placed": False,
            "source_ticker_is_discovery_not_execution_evidence": True,
            "public_books_prove_fills": False,
        },
        "limitations": [
            "A nominally positive long box is a financing return, not an arbitrage claim without opportunity-cost evidence.",
            "Ticker prices have no displayed quantities and only select candidates for depth confirmation.",
            "REST depth legs are not an atomic multi-leg snapshot or execution guarantee.",
            "Exact commission, margin, inventory, capital lockup, exercise handling, and partial-fill risk remain unverified.",
            "A gross-positive public snapshot cannot establish an accepted after-cost edge.",
        ],
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, required=True)
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
        source_snapshot=args.source_snapshot,
        confirmation_sweeps=args.confirmation_sweeps,
        confirmation_delay_seconds=args.confirmation_delay_seconds,
    )
    write_bytes_atomic(
        args.output,
        (_canonical_json(result) + "\n").encode("ascii"),
    )
    print(json.dumps(result["verdict"], indent=2))
    print(f"ticker_candidate_count={result['discovery']['ticker_candidate_count']}")
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
