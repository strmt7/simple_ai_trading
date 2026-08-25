"""Recover the Binance funding backfill with durable native-cash evidence."""

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
V3_TOOL_PATH = ROOT / "tools" / "backfill_binance_cross_stablecoin_funding.py"
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-cross-stablecoin-funding-recovery-contract-v4.json"
)
JOURNAL_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-cross-stablecoin-funding-recovery-journal-v4.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-cross-stablecoin-funding-recovery-v4-2026-08-25.json"
)
FUTURES_BASE_URL = "https://fapi.binance.com"
FIRST_BACKFILL_END_MS = 1773273599999
HISTORY_LIMIT = 1000
MAX_REQUESTS_PER_SYMBOL = 8
MAXIMUM_REQUESTS = 32
SIDEWAYS_BIPS = Decimal("10")
MINIMUM_REGIME_ROWS = 30
FX_STRESSES = (Decimal("0.98"), Decimal("1.02"))
SCHEMA_VERSION = "binance-cross-stablecoin-funding-recovery-v4"
JOURNAL_SCHEMA_VERSION = "binance-cross-stablecoin-funding-recovery-journal-v4"


def _load_module(path: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{name} cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V3 = _load_module(V3_TOOL_PATH, "cross_stablecoin_v3")


def _canonical_json(value: object) -> str:
    return V3._canonical_json(value)  # noqa: SLF001


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, object]:
    return V3._mapping(value, name=name)  # noqa: SLF001


def _list(value: object, *, name: str) -> list[object]:
    return V3._list(value, name=name)  # noqa: SLF001


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    return V3._decimal(value, name=name, positive=positive)  # noqa: SLF001


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    return V3._integer(value, name=name, positive=positive)  # noqa: SLF001


def _load_self_hashed(path: Path, *, name: str) -> dict[str, object]:
    payload = _mapping(json.loads(path.read_text(encoding="ascii")), name=name)
    claimed = str(payload.get("result_sha256") or "")
    body = dict(payload)
    body.pop("result_sha256", None)
    if claimed != _sha256(_canonical_json(body).encode("ascii")):
        raise ValueError(f"{name} canonical hash mismatch")
    return payload


def _load_sources() -> tuple[dict[str, object], dict[str, object]]:
    contract = _load_self_hashed(CONTRACT_PATH, name="recovery contract")
    if (
        contract.get("status")
        != "frozen_before_the_one_deliberate_funding_backfill_recovery"
    ):
        raise ValueError("recovery contract was not frozen before market access")
    boundary = _mapping(contract["recovery_boundary"], name="recovery boundary")
    expected_files = {
        ROOT / str(boundary["attempt3_failure_path"]): str(
            boundary["attempt3_failure_file_sha256"]
        ),
        ROOT / str(boundary["attempt3_contract_path"]): str(
            boundary["attempt3_contract_file_sha256"]
        ),
        ROOT / str(boundary["attempt3_tool_path"]): str(
            boundary["attempt3_tool_file_sha256"]
        ),
    }
    for path, expected in expected_files.items():
        if _sha256(path.read_bytes()) != expected:
            raise ValueError(f"recovery predecessor file hash mismatch: {path.name}")
    failure = _load_self_hashed(
        ROOT / str(boundary["attempt3_failure_path"]), name="attempt3 failure"
    )
    if failure["result_sha256"] != boundary["attempt3_failure_result_sha256"]:
        raise ValueError("attempt3 failure identity mismatch")
    v3_contract, _v2_result, recent_result = V3._load_sources()  # noqa: SLF001
    if v3_contract["result_sha256"] != boundary["attempt3_contract_result_sha256"]:
        raise ValueError("attempt3 contract identity mismatch")
    if (
        recent_result["result_sha256"]
        != contract["source_binding"]["recent_result_sha256"]
    ):
        raise ValueError("recent source identity mismatch")
    return contract, recent_result


def _journal_hash(payload: Mapping[str, object]) -> str:
    body = dict(payload)
    body.pop("journal_sha256", None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _write_journal(payload: dict[str, object]) -> None:
    payload["journal_sha256"] = _journal_hash(payload)
    write_bytes_atomic(JOURNAL_PATH, (_canonical_json(payload) + "\n").encode("ascii"))


def _load_or_create_journal(
    *, contract: Mapping[str, object], candidates: list[dict[str, object]]
) -> dict[str, object]:
    if JOURNAL_PATH.exists():
        journal = _mapping(
            json.loads(JOURNAL_PATH.read_text(encoding="ascii")),
            name="recovery journal",
        )
        if journal.get("journal_sha256") != _journal_hash(journal):
            raise ValueError("recovery journal canonical hash mismatch")
        if journal.get("contract_result_sha256") != contract["result_sha256"]:
            raise ValueError("recovery journal contract identity mismatch")
        if journal.get("status") not in {"active", "data_complete"}:
            raise RuntimeError("recovery journal is terminal and cannot be regenerated")
        return journal
    symbol_states: dict[str, object] = {}
    for candidate in candidates:
        onboard = _integer(
            candidate["usdc_onboard_time_ms"], name="onboard time", positive=True
        )
        for key in ("usdt_symbol", "usdc_symbol"):
            symbol = str(candidate[key])
            symbol_states[symbol] = {
                "onboard_time_ms": onboard,
                "next_end_time_ms": FIRST_BACKFILL_END_MS,
                "complete": False,
                "responses": [],
            }
    journal = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "contract_result_sha256": contract["result_sha256"],
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "status": "active",
        "next_request": None,
        "completed_request_count": 0,
        "symbol_states": dict(sorted(symbol_states.items())),
    }
    _write_journal(journal)
    return journal


def _request_fingerprint(*, symbol: str, end_time: int) -> dict[str, object]:
    return {
        "method": "GET",
        "url": f"{FUTURES_BASE_URL}/fapi/v1/fundingRate",
        "parameters": {
            "symbol": symbol,
            "endTime": end_time,
            "limit": HISTORY_LIMIT,
        },
    }


def _request_page(
    session: requests.Session,
    *,
    fingerprint: Mapping[str, object],
) -> tuple[object, dict[str, object]]:
    parameters = _mapping(fingerprint["parameters"], name="request parameters")
    before_ms = time.time_ns() // 1_000_000
    response = session.get(
        str(fingerprint["url"]),
        params=parameters,
        timeout=30,
    )
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
        raise ValueError("funding backfill response did not return JSON") from exc


def _capture_backfill(
    session: requests.Session,
    *,
    journal: dict[str, object],
) -> dict[str, object]:
    if journal["status"] == "data_complete":
        return journal
    states = _mapping(journal["symbol_states"], name="journal symbol states")
    for symbol in sorted(states):
        state = _mapping(states[symbol], name="journal symbol state")
        while not bool(state["complete"]):
            responses = _list(state["responses"], name="journal responses")
            if len(responses) >= MAX_REQUESTS_PER_SYMBOL:
                raise RuntimeError(f"{symbol} recovery hit the frozen request cap")
            if int(journal["completed_request_count"]) >= MAXIMUM_REQUESTS:
                raise RuntimeError("recovery exceeded the frozen total request cap")
            end_time = _integer(
                state["next_end_time_ms"], name="next end time", positive=True
            )
            fingerprint = _request_fingerprint(symbol=symbol, end_time=end_time)
            journal["next_request"] = fingerprint
            states[symbol] = state
            journal["symbol_states"] = states
            _write_journal(journal)
            raw, receipt = _request_page(session, fingerprint=fingerprint)
            rows = V3._validated_page(  # noqa: SLF001
                raw, symbol=symbol, end_time=end_time
            )
            responses.append(
                {
                    "request": fingerprint,
                    "receipt": receipt,
                    "payload": rows,
                }
            )
            state["responses"] = responses
            if not rows:
                state["complete"] = True
            else:
                earliest = _integer(
                    _mapping(rows[0], name="earliest recovery row")["fundingTime"],
                    name="earliest recovery time",
                    positive=True,
                )
                if earliest <= int(state["onboard_time_ms"]):
                    state["complete"] = True
                else:
                    state["next_end_time_ms"] = earliest - 1
            journal["completed_request_count"] = (
                int(journal["completed_request_count"]) + 1
            )
            journal["next_request"] = None
            states[symbol] = state
            journal["symbol_states"] = states
            _write_journal(journal)
    journal["status"] = "data_complete"
    journal["next_request"] = None
    _write_journal(journal)
    return journal


def _journal_failure(journal: dict[str, object], error: Exception) -> None:
    journal["status"] = "terminal_failure"
    journal["next_request"] = None
    journal["failure"] = {"type": type(error).__name__, "message": str(error)}
    _write_journal(journal)


def _merged_histories(
    *,
    journal: Mapping[str, object],
    recent_histories: Mapping[str, object],
) -> dict[str, list[object]]:
    states = _mapping(journal["symbol_states"], name="journal states")
    result: dict[str, list[object]] = {}
    for symbol, raw_state in states.items():
        state = _mapping(raw_state, name="journal state")
        if not state.get("complete"):
            raise ValueError("journal contains an incomplete funding symbol")
        pages = [
            _list(_mapping(item, name="journal response")["payload"], name="page")
            for item in _list(state["responses"], name="journal responses")
        ]
        result[symbol] = V3._merge_history(  # noqa: SLF001
            symbol=symbol,
            pages=pages,
            recent=recent_histories[symbol],
        )
    return result


def _native_cashflows(
    *,
    times: list[int],
    short_symbol: str,
    long_symbol: str,
    histories: Mapping[str, Mapping[int, Mapping[str, Decimal]]],
    quotes: Mapping[str, str],
) -> list[tuple[Decimal, Decimal]]:
    result: list[tuple[Decimal, Decimal]] = []
    for timestamp in times:
        cash = {"USDT": Decimal("0"), "USDC": Decimal("0")}
        for symbol, sign in (
            (short_symbol, Decimal("1")),
            (long_symbol, Decimal("-1")),
        ):
            row = histories[symbol].get(timestamp)
            if row is not None:
                cash[quotes[symbol]] += sign * row["mark"] * row["rate"]
        result.append((cash["USDT"], cash["USDC"]))
    return result


def _sum_vector(values: list[tuple[Decimal, Decimal]]) -> tuple[Decimal, Decimal]:
    return (
        sum((value[0] for value in values), Decimal("0")),
        sum((value[1] for value in values), Decimal("0")),
    )


def _stress_cash(vector: tuple[Decimal, Decimal], fx: Decimal) -> Decimal:
    return vector[0] + fx * vector[1]


def _break_even_fx(vector: tuple[Decimal, Decimal]) -> dict[str, object]:
    usdt, usdc = vector
    if usdc == 0:
        return {
            "threshold": None,
            "profitable_when": "all_positive_fx" if usdt > 0 else "no_positive_fx",
        }
    threshold = -usdt / usdc
    return {
        "threshold": str(threshold),
        "profitable_when": "fx_strictly_above_threshold"
        if usdc > 0
        else "fx_strictly_below_threshold",
    }


def _vector_report(values: list[tuple[Decimal, Decimal]]) -> dict[str, object]:
    vector = _sum_vector(values)
    stress = {str(fx): str(_stress_cash(vector, fx)) for fx in FX_STRESSES}
    worst = min(Decimal(value) for value in stress.values())
    return {
        "count": len(values),
        "cumulative_usdt": str(vector[0]),
        "cumulative_usdc": str(vector[1]),
        "stress_usdt_values": stress,
        "worst_stress_usdt": str(worst),
        "stress_passed": worst > 0,
        "break_even_usdcusdt": _break_even_fx(vector),
    }


def _partition_report(
    *,
    name: str,
    times: list[int],
    cashflows: list[tuple[Decimal, Decimal]],
) -> dict[str, object]:
    midpoint = len(cashflows) // 2
    halves = (cashflows[:midpoint], cashflows[midpoint:])
    quarters: dict[str, list[tuple[Decimal, Decimal]]] = defaultdict(list)
    for timestamp, cash in zip(times, cashflows, strict=True):
        quarters[V3._quarter_key(timestamp)].append(cash)  # noqa: SLF001
    keys = sorted(quarters)
    complete_inner = keys[1:-1]
    return {
        "name": name,
        "row_count": len(times),
        "start_time_ms": times[0],
        "end_time_ms": times[-1],
        "total": _vector_report(cashflows),
        "first_chronological_half": _vector_report(halves[0]),
        "second_chronological_half": _vector_report(halves[1]),
        "calendar_quarters": [
            {
                "quarter": key,
                "complete_inner_quarter": key in complete_inner,
                **_vector_report(quarters[key]),
            }
            for key in keys
        ],
        "complete_inner_quarters": complete_inner,
    }


def _nearest_rank_75(values: list[Decimal]) -> Decimal:
    if not values:
        raise ValueError("volatility threshold requires selection returns")
    ordered = sorted(values)
    return ordered[ceil(Decimal("0.75") * len(ordered)) - 1]


def _regime_report(
    *,
    times: list[int],
    cashflows: list[tuple[Decimal, Decimal]],
    returns_bips: Mapping[int, Decimal],
    volatility_threshold: Decimal,
    initial_sign: int,
) -> dict[str, object]:
    buckets: dict[str, list[tuple[Decimal, Decimal]]] = defaultdict(list)
    previous_sign = initial_sign
    for timestamp, cash in zip(times, cashflows, strict=True):
        value = returns_bips[timestamp]
        if value > SIDEWAYS_BIPS:
            buckets["up"].append(cash)
        elif value < -SIDEWAYS_BIPS:
            buckets["down"].append(cash)
        else:
            buckets["sideways"].append(cash)
        buckets[
            "high_volatility"
            if abs(value) >= volatility_threshold
            else "regular_volatility"
        ].append(cash)
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if sign == 0:
            buckets["zero_return"].append(cash)
        elif previous_sign and sign != previous_sign:
            buckets["reversal"].append(cash)
        else:
            buckets["continuation"].append(cash)
        if sign:
            previous_sign = sign
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
        "selection_only_high_volatility_threshold_bips": str(volatility_threshold),
        "slices": {
            name: {
                **_vector_report(buckets[name]),
                "minimum_row_count_passed": len(buckets[name]) >= MINIMUM_REGIME_ROWS,
            }
            for name in names
        },
    }


def _evaluate_candidate(
    *,
    candidate: Mapping[str, object],
    contracts: Mapping[str, Mapping[str, object]],
    histories_raw: Mapping[str, object],
    books_raw: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    base = str(candidate["base_asset"])
    usdt_symbol = str(candidate["usdt_symbol"])
    usdc_symbol = str(candidate["usdc_symbol"])
    symbols = (usdt_symbol, usdc_symbol)
    quotes = {usdt_symbol: "USDT", usdc_symbol: "USDC"}
    histories = {
        symbol: V3.V2._funding_rows(  # noqa: SLF001
            histories_raw[symbol], symbol=symbol
        )
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
    orientations: list[dict[str, object]] = []
    for short_symbol, long_symbol in (
        (usdt_symbol, usdc_symbol),
        (usdc_symbol, usdt_symbol),
    ):
        cashflows = _native_cashflows(
            times=selection_times,
            short_symbol=short_symbol,
            long_symbol=long_symbol,
            histories=histories,
            quotes=quotes,
        )
        vector = _sum_vector(cashflows)
        orientations.append(
            {
                "short_symbol": short_symbol,
                "long_symbol": long_symbol,
                "cashflows": cashflows,
                "fx_one_value": _stress_cash(vector, Decimal("1")),
            }
        )
    orientations.sort(key=lambda row: row["fx_one_value"], reverse=True)
    chosen = orientations[0]
    short_symbol = str(chosen["short_symbol"])
    long_symbol = str(chosen["long_symbol"])
    validation_cashflows = _native_cashflows(
        times=validation_times,
        short_symbol=short_symbol,
        long_symbol=long_symbol,
        histories=histories,
        quotes=quotes,
    )
    test_cashflows = _native_cashflows(
        times=test_times,
        short_symbol=short_symbol,
        long_symbol=long_symbol,
        histories=histories,
        quotes=quotes,
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
            returns_bips[timestamp] = (mark / previous_mark - Decimal("1")) * Decimal(
                "10000"
            )
        previous_mark = mark
    selection_abs_returns = [
        abs(returns_bips[timestamp])
        for timestamp in selection_times
        if timestamp in returns_bips
    ]
    volatility_threshold = _nearest_rank_75(selection_abs_returns)
    initial_sign = 0
    for timestamp in selection_times:
        value = returns_bips.get(timestamp)
        if value is not None and value != 0:
            initial_sign = 1 if value > 0 else -1
    combined_times = validation_times + test_times
    combined_cashflows = validation_cashflows + test_cashflows
    regimes = _regime_report(
        times=combined_times,
        cashflows=combined_cashflows,
        returns_bips=returns_bips,
        volatility_threshold=volatility_threshold,
        initial_sign=initial_sign,
    )
    selection = _vector_report(chosen["cashflows"])
    validation = _partition_report(
        name="validation", times=validation_times, cashflows=validation_cashflows
    )
    test = _partition_report(name="test", times=test_times, cashflows=test_cashflows)
    common_quantity = V3.V2._common_quantity(  # noqa: SLF001
        [contracts["USDT"], contracts["USDC"]]
    )
    books = {
        symbol: V3.V2._book(books_raw[symbol], expected_symbol=symbol)  # noqa: SLF001
        for symbol in symbols
    }
    depth_passed = all(
        quantity >= common_quantity
        for book in books.values()
        for quantity in (book["bid_quantity"], book["ask_quantity"])
    )
    spread_cash_usdt_per_base = Decimal("0")
    mid_notional_usdt_per_base = Decimal("0")
    for symbol, book in books.items():
        multiplier = Decimal("1") if quotes[symbol] == "USDT" else Decimal("1.02")
        spread_cash_usdt_per_base += (book["bid"] - book["ask"]) * multiplier
        mid_notional_usdt_per_base += (book["bid"] + book["ask"]) / 2 * multiplier
    spread_cost = -spread_cash_usdt_per_base
    test_vector = _sum_vector(test_cashflows)
    worst_test_cash = min(_stress_cash(test_vector, fx) for fx in FX_STRESSES)
    fee_break_even_bips = (
        (worst_test_cash - spread_cost)
        / (Decimal("2") * mid_notional_usdt_per_base)
        * Decimal("10000")
    )
    failures: list[str] = []
    if not selection["stress_passed"]:
        failures.append("selection_fx_stress_failed")
    for report in (validation, test):
        if not report["total"]["stress_passed"]:
            failures.append(f"{report['name']}_total_fx_stress_failed")
        if not report["first_chronological_half"]["stress_passed"]:
            failures.append(f"{report['name']}_first_half_fx_stress_failed")
        if not report["second_chronological_half"]["stress_passed"]:
            failures.append(f"{report['name']}_second_half_fx_stress_failed")
        complete = set(report["complete_inner_quarters"])
        if not complete:
            failures.append(f"{report['name']}_has_no_complete_inner_quarter")
        elif any(
            not row["stress_passed"]
            for row in report["calendar_quarters"]
            if row["quarter"] in complete
        ):
            failures.append(f"{report['name']}_complete_quarter_fx_stress_failed")
    for name in (
        "up",
        "down",
        "sideways",
        "high_volatility",
        "regular_volatility",
        "reversal",
        "continuation",
    ):
        row = regimes["slices"][name]
        if not row["minimum_row_count_passed"]:
            failures.append(f"{name}_has_fewer_than_30_rows")
        elif not row["stress_passed"]:
            failures.append(f"{name}_fx_stress_failed")
    if fee_break_even_bips <= 0:
        failures.append("test_fee_break_even_after_stressed_spread_nonpositive")
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
            **selection,
            "alternative_fx_one_value": str(orientations[1]["fx_one_value"]),
        },
        "validation": validation,
        "test": test,
        "regime_evaluation": regimes,
        "current_execution_diagnostic": {
            "common_minimum_quantity": str(common_quantity),
            "all_four_top_sides_cover_common_quantity": depth_passed,
            "current_round_trip_spread_cost_usdt_per_base_at_fx_1_02": str(spread_cost),
            "worst_test_fx_stress_cash_usdt_per_base": str(worst_test_cash),
            "test_equal_per_leg_fee_break_even_after_stressed_spread_bips": str(
                fee_break_even_bips
            ),
            "future_exit_prices_assumed_current_for_diagnostic": True,
            "account_fee_evidence_present": False,
        },
        "failure_reasons": failures,
        "qualified_public_recovery_screen": not failures,
    }


def run(*, session: requests.Session | None = None) -> dict[str, object]:
    """Capture or resume the recovery, then evaluate the frozen candidates."""

    contract, recent_result = _load_sources()
    started_ms = time.time_ns() // 1_000_000
    candidates = [
        _mapping(row, name="candidate")
        for row in _list(contract["candidate_scope"], name="candidate scope")
    ]
    journal = _load_or_create_journal(contract=contract, candidates=candidates)
    http = session or requests.Session()
    try:
        journal = _capture_backfill(http, journal=journal)
    except Exception as exc:
        _journal_failure(journal, exc)
        raise
    recent_source = _mapping(recent_result["source_contract"], name="recent source")
    recent_histories = _mapping(
        recent_source["funding_history_payloads"], name="recent histories"
    )
    merged_histories = _merged_histories(
        journal=journal, recent_histories=recent_histories
    )
    books_raw = {
        str(row["symbol"]): _mapping(row, name="book row")
        for row in _list(
            recent_source["book_ticker_selected_payload"], name="selected books"
        )
        if isinstance(row, Mapping)
    }
    all_contracts = _mapping(
        recent_result["eligible_contracts"], name="eligible contracts"
    )
    candidate_results = []
    for candidate in candidates:
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
            )
        )
    qualified = [
        row for row in candidate_results if row["qualified_public_recovery_screen"]
    ]
    finished_ms = time.time_ns() // 1_000_000
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.fromtimestamp(
            finished_ms / 1000, tz=UTC
        ).isoformat(),
        "started_at_ms": started_ms,
        "finished_at_ms": finished_ms,
        "new_request_count": journal["completed_request_count"],
        "source_contract": {
            "recovery_contract_path": str(CONTRACT_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "recovery_contract_result_sha256": contract["result_sha256"],
            "recent_result_sha256": recent_result["result_sha256"],
            "implementation_path": str(
                Path(__file__).resolve().relative_to(ROOT)
            ).replace("\\", "/"),
            "implementation_sha256": _sha256(Path(__file__).read_bytes()),
            "journal_path": str(JOURNAL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "journal_file_sha256": _sha256(JOURNAL_PATH.read_bytes()),
            "journal_sha256": journal["journal_sha256"],
        },
        "candidate_results": candidate_results,
        "verdict": {
            "status": (
                "full_history_fx_stressed_candidate_requires_authenticated_cost_collateral_and_prospective_evidence"
                if qualified
                else "rejected_full_history_fx_stressed_cross_stablecoin_funding_differential"
            ),
            "candidate_count": len(candidate_results),
            "qualified_public_recovery_count": len(qualified),
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
        "qualified_public_recovery_count="
        f"{result['verdict']['qualified_public_recovery_count']}"
    )
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
