"""Backfill and evaluate Binance USDT/USDC perpetual funding candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import importlib.util
import json
from math import ceil
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
V2_TOOL_PATH = ROOT / "tools" / "adjudicate_binance_cross_stablecoin_funding.py"
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-cross-stablecoin-funding-full-history-contract-v3.json"
)
V2_RESULT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-cross-stablecoin-funding-adjudication-v2-2026-08-25.json"
)
V1_RESULT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-cross-stablecoin-funding-snapshot-v1-2026-08-25.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-cross-stablecoin-funding-full-history-v3-2026-08-25.json"
)
FUTURES_BASE_URL = "https://fapi.binance.com"
SPOT_BASE_URL = "https://api.binance.com"
RECENT_START_MS = 1773273600000
FIRST_BACKFILL_END_MS = RECENT_START_MS - 1
FX_PARAMETERS = {
    "symbol": "USDCUSDT",
    "interval": "1d",
    "startTime": 1704240000000,
    "endTime": 1787702399999,
    "limit": 1000,
}
HISTORY_LIMIT = 1000
MAX_REQUESTS_PER_SYMBOL = 8
MAXIMUM_REQUESTS = 33
SIDEWAYS_BIPS = Decimal("10")
MINIMUM_REGIME_ROWS = 30
SCHEMA_VERSION = "binance-cross-stablecoin-funding-full-history-v3"


def _load_module(path: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{name} cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = _load_module(V2_TOOL_PATH, "cross_stablecoin_v2")


def _canonical_json(value: object) -> str:
    return V2._canonical_json(value)  # noqa: SLF001


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, object]:
    return V2._mapping(value, name=name)  # noqa: SLF001


def _list(value: object, *, name: str) -> list[object]:
    return V2._list(value, name=name)  # noqa: SLF001


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    return V2._decimal(value, name=name, positive=positive)  # noqa: SLF001


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    return V2._integer(value, name=name, positive=positive)  # noqa: SLF001


def _load_self_hashed(path: Path, *, name: str) -> dict[str, object]:
    payload = _mapping(json.loads(path.read_text(encoding="ascii")), name=name)
    claimed = str(payload.get("result_sha256") or "")
    body = dict(payload)
    body.pop("result_sha256", None)
    if claimed != _sha256(_canonical_json(body).encode("ascii")):
        raise ValueError(f"{name} canonical hash mismatch")
    return payload


def _load_sources() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    contract = _load_self_hashed(CONTRACT_PATH, name="full-history contract")
    if contract.get("status") != "frozen_before_any_full_history_backfill_request":
        raise ValueError("full-history contract was not frozen before market access")
    binding = _mapping(contract.get("source_binding"), name="source binding")
    expected_files = {
        V2_RESULT_PATH: str(binding["adjudication_result_file_sha256"]),
        V2_TOOL_PATH: str(binding["adjudication_tool_file_sha256"]),
        ROOT / str(binding["adjudication_contract_path"]): str(
            binding["adjudication_contract_file_sha256"]
        ),
    }
    for path, expected in expected_files.items():
        if _sha256(path.read_bytes()) != expected:
            raise ValueError(f"source file hash mismatch: {path.name}")
    v2_result = _load_self_hashed(V2_RESULT_PATH, name="adjudication result")
    if v2_result["result_sha256"] != binding["adjudication_result_sha256"]:
        raise ValueError("adjudication result identity mismatch")
    v2_source = _mapping(v2_result["source_contract"], name="adjudication source")
    if (
        v2_source.get("implementation_sha256")
        != binding["adjudication_tool_file_sha256"]
    ):
        raise ValueError("adjudication implementation identity mismatch")
    if (
        v2_source.get("adjudication_contract_result_sha256")
        != binding["adjudication_contract_result_sha256"]
    ):
        raise ValueError("adjudication contract identity mismatch")
    v1_result = _load_self_hashed(V1_RESULT_PATH, name="attempt1 result")
    _verified_v2_contract, verified_v1_result = V2._load_sources()  # noqa: SLF001
    if verified_v1_result["result_sha256"] != v1_result["result_sha256"]:
        raise ValueError("attempt1 transitive source identity mismatch")
    return contract, v2_result, v1_result


def _get(
    session: requests.Session,
    base_url: str,
    path: str,
    *,
    params: Mapping[str, object],
) -> tuple[object, dict[str, object]]:
    before_ms = time.time_ns() // 1_000_000
    response = session.get(f"{base_url}{path}", params=params, timeout=30)
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
        return response.json(), receipt
    except requests.JSONDecodeError as exc:
        raise ValueError(f"source {response.url} did not return JSON") from exc


def _validated_page(raw: object, *, symbol: str, end_time: int) -> list[object]:
    rows = _list(raw, name=f"{symbol} backfill page")
    previous = 0
    for raw_row in rows:
        row = _mapping(raw_row, name="funding backfill row")
        if row.get("symbol") != symbol:
            raise ValueError("backfill funding symbol mismatch")
        funding_time = _integer(
            row.get("fundingTime"), name="funding time", positive=True
        )
        _decimal(row.get("fundingRate"), name="funding rate")
        _decimal(row.get("markPrice"), name="funding mark", positive=True)
        if funding_time <= previous or funding_time > end_time:
            raise ValueError("backfill funding page is not bounded and ascending")
        previous = funding_time
    return rows


def _backfill_symbol(
    session: requests.Session,
    *,
    symbol: str,
    onboard_time_ms: int,
) -> tuple[list[list[object]], list[dict[str, object]]]:
    pages: list[list[object]] = []
    receipts: list[dict[str, object]] = []
    end_time = FIRST_BACKFILL_END_MS
    complete = False
    for _ in range(MAX_REQUESTS_PER_SYMBOL):
        raw, receipt = _get(
            session,
            FUTURES_BASE_URL,
            "/fapi/v1/fundingRate",
            params={"symbol": symbol, "endTime": end_time, "limit": HISTORY_LIMIT},
        )
        rows = _validated_page(raw, symbol=symbol, end_time=end_time)
        pages.append(rows)
        receipts.append(receipt)
        if not rows:
            complete = True
            break
        earliest = _integer(
            _mapping(rows[0], name="earliest funding row")["fundingTime"],
            name="earliest funding time",
            positive=True,
        )
        if earliest <= onboard_time_ms:
            complete = True
            break
        end_time = earliest - 1
    if not complete:
        raise RuntimeError(f"{symbol} backfill hit the frozen request cap")
    return pages, receipts


def _merge_history(
    *,
    symbol: str,
    pages: list[list[object]],
    recent: object,
) -> list[object]:
    by_time: dict[int, dict[str, object]] = {}
    for raw_row in [row for page in pages for row in page] + _list(
        recent, name=f"{symbol} recent history"
    ):
        row = _mapping(raw_row, name="merged funding row")
        if row.get("symbol") != symbol:
            raise ValueError("merged funding symbol mismatch")
        funding_time = _integer(
            row.get("fundingTime"), name="funding time", positive=True
        )
        existing = by_time.get(funding_time)
        if existing is not None and existing != row:
            raise ValueError("overlapping funding rows conflict")
        by_time[funding_time] = row
    result = [by_time[key] for key in sorted(by_time)]
    V2._funding_rows(result, symbol=symbol)  # noqa: SLF001
    return result


def _nearest_rank_75(values: list[Decimal]) -> Decimal:
    if not values:
        raise ValueError("volatility threshold requires selection returns")
    ordered = sorted(values)
    return ordered[ceil(Decimal("0.75") * len(ordered)) - 1]


def _quarter_key(epoch_ms: int) -> str:
    value = datetime.fromtimestamp(epoch_ms / 1000, tz=UTC)
    return f"{value.year}-Q{(value.month - 1) // 3 + 1}"


def _slice_statistics(values: list[Decimal]) -> dict[str, object]:
    if not values:
        return {"count": 0, "sum": "0", "eligible": False}
    result = V2._statistics(values)  # noqa: SLF001
    result["eligible"] = len(values) >= MINIMUM_REGIME_ROWS
    return result


def _partition_report(
    *,
    name: str,
    times: list[int],
    cashflows: list[Decimal],
) -> dict[str, object]:
    midpoint = len(cashflows) // 2
    halves = [cashflows[:midpoint], cashflows[midpoint:]]
    by_quarter: dict[str, list[Decimal]] = defaultdict(list)
    for timestamp, cash in zip(times, cashflows, strict=True):
        by_quarter[_quarter_key(timestamp)].append(cash)
    keys = sorted(by_quarter)
    complete_inner = keys[1:-1]
    return {
        "name": name,
        "row_count": len(times),
        "start_time_ms": times[0],
        "end_time_ms": times[-1],
        "statistics": V2._statistics(cashflows),  # noqa: SLF001
        "first_chronological_half": V2._statistics(halves[0]),  # noqa: SLF001
        "second_chronological_half": V2._statistics(halves[1]),  # noqa: SLF001
        "calendar_quarters": [
            {
                "quarter": key,
                "complete_inner_quarter": key in complete_inner,
                **V2._statistics(by_quarter[key]),  # noqa: SLF001
            }
            for key in keys
        ],
        "complete_inner_quarters": complete_inner,
    }


def _regime_report(
    *,
    evaluation_times: list[int],
    evaluation_cashflows: list[Decimal],
    returns_bips: Mapping[int, Decimal],
    volatility_threshold_bips: Decimal,
    initial_previous_nonzero_sign: int,
) -> dict[str, object]:
    buckets: dict[str, list[Decimal]] = defaultdict(list)
    previous_nonzero_sign = initial_previous_nonzero_sign
    for timestamp, cash in zip(evaluation_times, evaluation_cashflows, strict=True):
        value = returns_bips[timestamp]
        if value > SIDEWAYS_BIPS:
            buckets["up"].append(cash)
        elif value < -SIDEWAYS_BIPS:
            buckets["down"].append(cash)
        else:
            buckets["sideways"].append(cash)
        buckets[
            "high_volatility"
            if abs(value) >= volatility_threshold_bips
            else "regular_volatility"
        ].append(cash)
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if sign == 0:
            buckets["zero_return"].append(cash)
        elif previous_nonzero_sign and sign != previous_nonzero_sign:
            buckets["reversal"].append(cash)
        else:
            buckets["continuation"].append(cash)
        if sign:
            previous_nonzero_sign = sign
    names = (
        "up",
        "down",
        "sideways",
        "high_volatility",
        "regular_volatility",
        "reversal",
        "continuation",
        "zero_return",
    )
    return {
        "selection_only_high_volatility_threshold_bips": str(volatility_threshold_bips),
        "slices": {name: _slice_statistics(buckets[name]) for name in names},
    }


def _evaluate_candidate(
    *,
    candidate: Mapping[str, object],
    contracts: Mapping[str, Mapping[str, object]],
    histories_raw: Mapping[str, object],
    books_raw: Mapping[str, Mapping[str, object]],
    fx_days: Mapping[int, Mapping[str, Decimal]],
) -> dict[str, object]:
    base = str(candidate["base_asset"])
    usdt_symbol = str(candidate["usdt_symbol"])
    usdc_symbol = str(candidate["usdc_symbol"])
    symbols = (usdt_symbol, usdc_symbol)
    quotes = {usdt_symbol: "USDT", usdc_symbol: "USDC"}
    histories = {
        symbol: V2._funding_rows(histories_raw[symbol], symbol=symbol)  # noqa: SLF001
        for symbol in symbols
    }
    common_start = max(min(rows) for rows in histories.values())
    common_end = min(max(rows) for rows in histories.values())
    times = sorted(
        {
            timestamp
            for rows in histories.values()
            for timestamp in rows
            if common_start <= timestamp <= common_end
        }
    )
    third = len(times) // 3
    if third < 90:
        raise ValueError(f"{base} full history is too short for the frozen split")
    selection_times = times[:third]
    validation_times = times[third : 2 * third]
    test_times = times[2 * third :]
    orientation_rows: list[dict[str, object]] = []
    for short_symbol, long_symbol in (
        (usdt_symbol, usdc_symbol),
        (usdc_symbol, usdt_symbol),
    ):
        cashflows = V2._orientation_cashflows(  # noqa: SLF001
            times=selection_times,
            short_symbol=short_symbol,
            long_symbol=long_symbol,
            histories=histories,
            quotes=quotes,
            fx_days=fx_days,
        )
        orientation_rows.append(
            {
                "short_symbol": short_symbol,
                "long_symbol": long_symbol,
                "cashflows": cashflows,
                "sum": sum(cashflows, Decimal("0")),
            }
        )
    orientation_rows.sort(key=lambda row: row["sum"], reverse=True)
    chosen = orientation_rows[0]
    short_symbol = str(chosen["short_symbol"])
    long_symbol = str(chosen["long_symbol"])
    validation_cashflows = V2._orientation_cashflows(  # noqa: SLF001
        times=validation_times,
        short_symbol=short_symbol,
        long_symbol=long_symbol,
        histories=histories,
        quotes=quotes,
        fx_days=fx_days,
    )
    test_cashflows = V2._orientation_cashflows(  # noqa: SLF001
        times=test_times,
        short_symbol=short_symbol,
        long_symbol=long_symbol,
        histories=histories,
        quotes=quotes,
        fx_days=fx_days,
    )
    usdt_marks = histories[usdt_symbol]
    returns_bips: dict[int, Decimal] = {}
    previous_mark: Decimal | None = None
    for timestamp in times:
        row = usdt_marks.get(timestamp)
        if row is None:
            raise ValueError("USDT funding mark is absent at a union settlement")
        mark = row["mark"]
        if previous_mark is not None:
            returns_bips[timestamp] = ((mark / previous_mark) - Decimal("1")) * Decimal(
                "10000"
            )
        previous_mark = mark
    selection_abs_returns = [
        abs(returns_bips[timestamp])
        for timestamp in selection_times
        if timestamp in returns_bips
    ]
    volatility_threshold = _nearest_rank_75(selection_abs_returns)
    initial_previous_nonzero_sign = 0
    for timestamp in selection_times:
        value = returns_bips.get(timestamp)
        if value is not None and value != 0:
            initial_previous_nonzero_sign = 1 if value > 0 else -1
    combined_evaluation_times = validation_times + test_times
    combined_evaluation_cashflows = validation_cashflows + test_cashflows
    regime_report = _regime_report(
        evaluation_times=combined_evaluation_times,
        evaluation_cashflows=combined_evaluation_cashflows,
        returns_bips=returns_bips,
        volatility_threshold_bips=volatility_threshold,
        initial_previous_nonzero_sign=initial_previous_nonzero_sign,
    )
    validation = _partition_report(
        name="validation", times=validation_times, cashflows=validation_cashflows
    )
    test = _partition_report(name="test", times=test_times, cashflows=test_cashflows)
    common_quantity = V2._common_quantity(  # noqa: SLF001
        [contracts["USDT"], contracts["USDC"]]
    )
    books = {
        symbol: V2._book(books_raw[symbol], expected_symbol=symbol)  # noqa: SLF001
        for symbol in symbols
    }
    depth_passed = all(
        quantity >= common_quantity
        for book in books.values()
        for quantity in (book["bid_quantity"], book["ask_quantity"])
    )
    current_fx_high = fx_days[max(fx_days)]["high"]
    spread_cash_per_base = Decimal("0")
    mid_notional_usdt_per_base = Decimal("0")
    for symbol, book in books.items():
        native_spread = book["bid"] - book["ask"]
        native_mid = (book["bid"] + book["ask"]) / 2
        multiplier = Decimal("1") if quotes[symbol] == "USDT" else current_fx_high
        spread_cash_per_base += native_spread * multiplier
        mid_notional_usdt_per_base += native_mid * multiplier
    spread_cost_per_base = -spread_cash_per_base
    test_cash_per_base = sum(test_cashflows, Decimal("0"))
    test_break_even_bips = (
        (test_cash_per_base - spread_cost_per_base)
        / (Decimal("2") * mid_notional_usdt_per_base)
        * Decimal("10000")
    )
    failures: list[str] = []
    if Decimal(str(chosen["sum"])) <= 0:
        failures.append("selection_cashflow_nonpositive")
    for report in (validation, test):
        if Decimal(str(report["statistics"]["sum"])) <= 0:
            failures.append(f"{report['name']}_total_nonpositive")
        if Decimal(str(report["first_chronological_half"]["sum"])) <= 0:
            failures.append(f"{report['name']}_first_half_nonpositive")
        if Decimal(str(report["second_chronological_half"]["sum"])) <= 0:
            failures.append(f"{report['name']}_second_half_nonpositive")
        complete = set(report["complete_inner_quarters"])
        if not complete:
            failures.append(f"{report['name']}_has_no_complete_inner_quarter")
        elif any(
            Decimal(str(row["sum"])) < 0
            for row in report["calendar_quarters"]
            if row["quarter"] in complete
        ):
            failures.append(f"{report['name']}_complete_quarter_negative")
    slices = regime_report["slices"]
    for name in (
        "up",
        "down",
        "sideways",
        "high_volatility",
        "regular_volatility",
        "reversal",
        "continuation",
    ):
        row = slices[name]
        if not row["eligible"]:
            failures.append(f"{name}_has_fewer_than_30_rows")
        elif Decimal(str(row["sum"])) <= 0:
            failures.append(f"{name}_cashflow_nonpositive")
    if test_break_even_bips <= 0:
        failures.append("test_fee_break_even_after_current_spread_nonpositive")
    if not depth_passed:
        failures.append("current_common_quantity_depth_missing")
    return {
        "base_asset": base,
        "symbols": {"short": short_symbol, "long": long_symbol},
        "quote_orientation": {
            "short": quotes[short_symbol],
            "long": quotes[long_symbol],
        },
        "history": {
            "common_start_time_ms": common_start,
            "common_end_time_ms": common_end,
            "union_settlement_count": len(times),
            "per_symbol_row_counts": {
                symbol: len(rows) for symbol, rows in histories.items()
            },
            "all_symbol_timestamp_sets_equal": len(
                {tuple(rows) for rows in histories.values()}
            )
            == 1,
        },
        "selection": {
            "row_count": len(selection_times),
            "start_time_ms": selection_times[0],
            "end_time_ms": selection_times[-1],
            "chosen_conservative_usdt_cash_per_base": str(chosen["sum"]),
            "alternative_conservative_usdt_cash_per_base": str(
                orientation_rows[1]["sum"]
            ),
        },
        "validation": validation,
        "test": test,
        "regime_evaluation": regime_report,
        "current_execution_diagnostic": {
            "common_minimum_quantity": str(common_quantity),
            "all_four_top_sides_cover_common_quantity": depth_passed,
            "current_round_trip_spread_cost_usdt_per_base": str(spread_cost_per_base),
            "test_conservative_usdt_cash_per_base": str(test_cash_per_base),
            "test_equal_per_leg_fee_break_even_after_current_spread_bips": str(
                test_break_even_bips
            ),
            "future_exit_prices_assumed_current_for_diagnostic": True,
            "account_fee_evidence_present": False,
        },
        "failure_reasons": failures,
        "qualified_public_full_history_screen": not failures,
    }


def run(*, session: requests.Session | None = None) -> dict[str, object]:
    """Run the frozen bounded backfill and three-way evaluation."""

    contract, v2_result, v1_result = _load_sources()
    started_ms = time.time_ns() // 1_000_000
    http = session or requests.Session()
    candidate_rows = [
        _mapping(row, name="candidate")
        for row in _list(contract["candidate_scope"]["pairs"], name="candidates")
    ]
    onboard_by_symbol = {
        str(candidate["usdt_symbol"]): _integer(
            candidate["usdc_onboard_time_ms"], name="onboard time", positive=True
        )
        for candidate in candidate_rows
    }
    onboard_by_symbol.update(
        {
            str(candidate["usdc_symbol"]): _integer(
                candidate["usdc_onboard_time_ms"], name="onboard time", positive=True
            )
            for candidate in candidate_rows
        }
    )
    backfill_pages: dict[str, list[list[object]]] = {}
    request_ledger: list[dict[str, object]] = []
    for symbol in sorted(onboard_by_symbol):
        pages, receipts = _backfill_symbol(
            http, symbol=symbol, onboard_time_ms=onboard_by_symbol[symbol]
        )
        backfill_pages[symbol] = pages
        request_ledger.extend(receipts)
    fx_raw, fx_receipt = _get(
        http,
        SPOT_BASE_URL,
        "/api/v3/klines",
        params=FX_PARAMETERS,
    )
    request_ledger.append(fx_receipt)
    if len(request_ledger) > MAXIMUM_REQUESTS:
        raise RuntimeError("full-history screen exceeded the frozen request cap")
    fx_days = V2._fx_days(fx_raw)  # noqa: SLF001
    v1_source = _mapping(v1_result["source_contract"], name="attempt1 source")
    recent_histories = _mapping(
        v1_source["funding_history_payloads"], name="recent funding histories"
    )
    merged_histories = {
        symbol: _merge_history(
            symbol=symbol,
            pages=backfill_pages[symbol],
            recent=recent_histories[symbol],
        )
        for symbol in sorted(onboard_by_symbol)
    }
    books_raw = {
        str(row["symbol"]): _mapping(row, name="book row")
        for row in _list(
            v1_source["book_ticker_selected_payload"], name="selected books"
        )
        if isinstance(row, Mapping)
    }
    all_contracts = _mapping(v1_result["eligible_contracts"], name="eligible contracts")
    candidate_results = []
    for candidate in candidate_rows:
        by_quote = _mapping(
            all_contracts[str(candidate["base_asset"])], name="candidate contracts"
        )
        candidate_results.append(
            _evaluate_candidate(
                candidate=candidate,
                contracts={
                    quote: _mapping(row, name="candidate contract")
                    for quote, row in by_quote.items()
                },
                histories_raw=merged_histories,
                books_raw=books_raw,
                fx_days=fx_days,
            )
        )
    qualified = [
        row for row in candidate_results if row["qualified_public_full_history_screen"]
    ]
    finished_ms = time.time_ns() // 1_000_000
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.fromtimestamp(
            finished_ms / 1000, tz=UTC
        ).isoformat(),
        "started_at_ms": started_ms,
        "finished_at_ms": finished_ms,
        "new_request_count": len(request_ledger),
        "source_contract": {
            "full_history_contract_path": str(CONTRACT_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "full_history_contract_result_sha256": contract["result_sha256"],
            "adjudication_result_path": str(V2_RESULT_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "adjudication_result_sha256": v2_result["result_sha256"],
            "implementation_path": str(
                Path(__file__).resolve().relative_to(ROOT)
            ).replace("\\", "/"),
            "implementation_sha256": _sha256(Path(__file__).read_bytes()),
            "request_ledger": request_ledger,
            "backfill_pages": backfill_pages,
            "full_fx_payload": fx_raw,
            "merged_funding_histories": merged_histories,
        },
        "candidate_results": candidate_results,
        "verdict": {
            "status": (
                "full_history_public_candidate_requires_authenticated_cost_collateral_and_prospective_evidence"
                if qualified
                else "rejected_full_history_cross_stablecoin_funding_differential"
            ),
            "candidate_count": len(candidate_results),
            "qualified_public_full_history_count": len(qualified),
            "accepted_edge": False,
            "profitability_claim": False,
            "trading_authority": False,
            "credentials_used": False,
            "orders_placed": False,
        },
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


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
        write_bytes_atomic(
            args.output, (_canonical_json(result) + "\n").encode("ascii")
        )
        print(f"terminal_failure={type(exc).__name__}: {exc}")
        print(f"output={args.output}")
        return 1
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(f"new_request_count={result['new_request_count']}")
    print(f"candidate_count={result['verdict']['candidate_count']}")
    print(
        "qualified_public_full_history_count="
        f"{result['verdict']['qualified_public_full_history_count']}"
    )
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
