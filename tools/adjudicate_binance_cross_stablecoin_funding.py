"""Correct the Binance cross-stablecoin funding screen with cash accounting."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
V1_TOOL_PATH = ROOT / "tools" / "screen_binance_cross_stablecoin_funding.py"
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-cross-stablecoin-funding-adjudication-contract-v2.json"
)
SOURCE_PATH = (
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
    / "binance-cross-stablecoin-funding-adjudication-v2-2026-08-25.json"
)
SPOT_BASE_URL = "https://api.binance.com"
FX_PARAMETERS = {
    "symbol": "USDCUSDT",
    "interval": "1d",
    "startTime": 1773273600000,
    "endTime": 1787702399999,
    "limit": 1000,
}
DAY_MS = 86_400_000
SCHEMA_VERSION = "binance-cross-stablecoin-funding-adjudication-v2"


def _load_v1_module() -> object:
    spec = importlib.util.spec_from_file_location("cross_stablecoin_v1", V1_TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("attempt1 tool cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = _load_v1_module()


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


def _load_self_hashed(path: Path, *, name: str) -> dict[str, object]:
    payload = _mapping(json.loads(path.read_text(encoding="ascii")), name=name)
    claimed = str(payload.get("result_sha256") or "")
    body = dict(payload)
    body.pop("result_sha256", None)
    if claimed != _sha256(_canonical_json(body).encode("ascii")):
        raise ValueError(f"{name} canonical hash mismatch")
    return payload


def _load_sources() -> tuple[dict[str, object], dict[str, object]]:
    contract = _load_self_hashed(CONTRACT_PATH, name="adjudication contract")
    if contract.get("status") != "frozen_before_the_single_usdcusdt_fx_request":
        raise ValueError("adjudication contract was not frozen before FX access")
    binding = _mapping(contract.get("source_binding"), name="source binding")
    checks = {
        CONTRACT_PATH: None,
        SOURCE_PATH: str(binding["attempt1_result_file_sha256"]),
        V1_TOOL_PATH: str(binding["attempt1_tool_file_sha256"]),
        ROOT / str(binding["attempt1_contract_path"]): str(
            binding["attempt1_contract_file_sha256"]
        ),
    }
    for path, expected in checks.items():
        if expected is not None and _sha256(path.read_bytes()) != expected:
            raise ValueError(f"source file hash mismatch: {path.name}")
    source = _load_self_hashed(SOURCE_PATH, name="attempt1 result")
    if source["result_sha256"] != binding["attempt1_result_sha256"]:
        raise ValueError("attempt1 result identity mismatch")
    source_contract = _mapping(source.get("source_contract"), name="attempt1 source")
    if (
        source_contract.get("implementation_sha256")
        != binding["attempt1_tool_file_sha256"]
    ):
        raise ValueError("attempt1 implementation identity mismatch")
    if (
        source_contract.get("contract_result_sha256")
        != binding["attempt1_contract_result_sha256"]
    ):
        raise ValueError("attempt1 contract identity mismatch")
    return contract, source


def _get_fx(
    session: requests.Session,
) -> tuple[object, dict[str, object]]:
    before_ms = time.time_ns() // 1_000_000
    response = session.get(
        f"{SPOT_BASE_URL}/api/v3/klines",
        params=FX_PARAMETERS,
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
        raise ValueError("USDCUSDT daily klines did not return JSON") from exc


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    return V1._decimal(value, name=name, positive=positive)  # noqa: SLF001


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    return V1._integer(value, name=name, positive=positive)  # noqa: SLF001


def _fx_days(raw: object) -> dict[int, dict[str, Decimal]]:
    days: dict[int, dict[str, Decimal]] = {}
    for raw_row in _list(raw, name="USDCUSDT daily klines"):
        row = _list(raw_row, name="USDCUSDT daily kline")
        if len(row) < 7:
            raise ValueError("USDCUSDT daily kline has fewer than seven fields")
        open_time = _integer(row[0], name="FX open time")
        close_time = _integer(row[6], name="FX close time")
        if (
            open_time % DAY_MS != 0
            or close_time != open_time + DAY_MS - 1
            or open_time in days
        ):
            raise ValueError("USDCUSDT daily kline time boundary is invalid")
        high = _decimal(row[2], name="FX high", positive=True)
        low = _decimal(row[3], name="FX low", positive=True)
        if low > high:
            raise ValueError("USDCUSDT daily low exceeds high")
        days[open_time] = {"low": low, "high": high}
    if not days:
        raise ValueError("USDCUSDT daily kline payload is empty")
    return days


def _funding_rows(raw: object, *, symbol: str) -> dict[int, dict[str, Decimal]]:
    result: dict[int, dict[str, Decimal]] = {}
    previous = 0
    for raw_row in _list(raw, name=f"{symbol} funding history"):
        row = _mapping(raw_row, name="funding row")
        if row.get("symbol") != symbol:
            raise ValueError("funding history symbol mismatch")
        funding_time = _integer(
            row.get("fundingTime"), name="funding time", positive=True
        )
        if funding_time <= previous or funding_time in result:
            raise ValueError("funding times must be unique and strictly increasing")
        result[funding_time] = {
            "rate": _decimal(row.get("fundingRate"), name="funding rate"),
            "mark": _decimal(row.get("markPrice"), name="funding mark", positive=True),
        }
        previous = funding_time
    if not result:
        raise ValueError("funding history is empty")
    return result


def _convert_native_cash(
    cash: Decimal,
    *,
    quote: str,
    funding_time: int,
    fx_days: Mapping[int, Mapping[str, Decimal]],
) -> Decimal:
    if quote == "USDT" or cash == 0:
        return cash
    if quote != "USDC":
        raise ValueError("unsupported funding quote asset")
    day = funding_time - funding_time % DAY_MS
    fx = fx_days.get(day)
    if fx is None:
        raise ValueError("same-day USDCUSDT conversion bar is absent")
    return cash * fx["low" if cash > 0 else "high"]


def _orientation_cashflows(
    *,
    times: list[int],
    short_symbol: str,
    long_symbol: str,
    histories: Mapping[str, Mapping[int, Mapping[str, Decimal]]],
    quotes: Mapping[str, str],
    fx_days: Mapping[int, Mapping[str, Decimal]],
) -> list[Decimal]:
    result: list[Decimal] = []
    for funding_time in times:
        cash = Decimal("0")
        for symbol, sign in (
            (short_symbol, Decimal("1")),
            (long_symbol, Decimal("-1")),
        ):
            row = histories[symbol].get(funding_time)
            if row is None:
                continue
            native_cash = sign * row["mark"] * row["rate"]
            cash += _convert_native_cash(
                native_cash,
                quote=quotes[symbol],
                funding_time=funding_time,
                fx_days=fx_days,
            )
        result.append(cash)
    return result


def _statistics(values: list[Decimal]) -> dict[str, object]:
    return V1._statistics(values)  # noqa: SLF001


def _month_key(epoch_ms: int) -> str:
    return V1._month_key(epoch_ms)  # noqa: SLF001


def _book(raw: Mapping[str, object], *, expected_symbol: str) -> dict[str, Decimal]:
    return V1._book(raw, expected_symbol=expected_symbol)  # noqa: SLF001


def _common_quantity(contracts: list[Mapping[str, object]]) -> Decimal:
    return V1._common_quantity(contracts)  # noqa: SLF001


def _evaluate_pair(
    *,
    base: str,
    contracts: Mapping[str, Mapping[str, object]],
    raw_histories: Mapping[str, object],
    raw_books: Mapping[str, Mapping[str, object]],
    fx_days: Mapping[int, Mapping[str, Decimal]],
) -> dict[str, object]:
    symbols = {quote: str(contracts[quote]["symbol"]) for quote in ("USDT", "USDC")}
    quotes = {symbol: quote for quote, symbol in symbols.items()}
    histories = {
        symbol: _funding_rows(raw_histories[symbol], symbol=symbol)
        for symbol in symbols.values()
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
    if len(times) < 90:
        raise ValueError("fewer than 90 funding settlements exist in the common window")
    split = len(times) // 2
    selection_times = times[:split]
    evaluation_times = times[split:]
    orientations = [
        (symbols["USDT"], symbols["USDC"]),
        (symbols["USDC"], symbols["USDT"]),
    ]
    selection_results = []
    for short_symbol, long_symbol in orientations:
        cashflows = _orientation_cashflows(
            times=selection_times,
            short_symbol=short_symbol,
            long_symbol=long_symbol,
            histories=histories,
            quotes=quotes,
            fx_days=fx_days,
        )
        selection_results.append(
            {
                "short_symbol": short_symbol,
                "long_symbol": long_symbol,
                "cashflows": cashflows,
                "sum": sum(cashflows, Decimal("0")),
            }
        )
    selection_results.sort(key=lambda row: row["sum"], reverse=True)
    chosen = selection_results[0]
    short_symbol = str(chosen["short_symbol"])
    long_symbol = str(chosen["long_symbol"])
    evaluation_cashflows = _orientation_cashflows(
        times=evaluation_times,
        short_symbol=short_symbol,
        long_symbol=long_symbol,
        histories=histories,
        quotes=quotes,
        fx_days=fx_days,
    )
    evaluation_split = len(evaluation_cashflows) // 2
    halves = [
        evaluation_cashflows[:evaluation_split],
        evaluation_cashflows[evaluation_split:],
    ]
    monthly: dict[str, list[Decimal]] = defaultdict(list)
    for funding_time, cash in zip(evaluation_times, evaluation_cashflows, strict=True):
        monthly[_month_key(funding_time)].append(cash)
    month_keys = sorted(monthly)
    complete_months = month_keys[1:-1]
    monthly_rows = [
        {
            "month": month,
            "complete_inner_month": month in complete_months,
            **_statistics(monthly[month]),
        }
        for month in month_keys
    ]
    common_quantity = _common_quantity([contracts["USDT"], contracts["USDC"]])
    books = {
        symbol: _book(raw_books[symbol], expected_symbol=symbol)
        for symbol in symbols.values()
    }
    depth_passed = all(
        quantity >= common_quantity
        for book in books.values()
        for quantity in (book["bid_quantity"], book["ask_quantity"])
    )
    current_day = max(fx_days)
    current_fx_high = fx_days[current_day]["high"]
    spread_cash_per_base = Decimal("0")
    current_mid_usdt_per_base = Decimal("0")
    for symbol, book in books.items():
        native_spread_cash = book["bid"] - book["ask"]
        native_mid = (book["bid"] + book["ask"]) / 2
        quote = quotes[symbol]
        spread_cash_per_base += (
            native_spread_cash
            if quote == "USDT"
            else native_spread_cash * current_fx_high
        )
        current_mid_usdt_per_base += (
            native_mid if quote == "USDT" else native_mid * current_fx_high
        )
    evaluation_cash_per_base = sum(evaluation_cashflows, Decimal("0"))
    spread_cost_per_base = -spread_cash_per_base
    break_even_bips = (
        (evaluation_cash_per_base - spread_cost_per_base)
        / (Decimal("2") * current_mid_usdt_per_base)
        * Decimal("10000")
    )
    failures: list[str] = []
    if Decimal(str(chosen["sum"])) <= 0:
        failures.append("selection_cashflow_nonpositive")
    if any(sum(half, Decimal("0")) <= 0 for half in halves):
        failures.append("evaluation_chronological_half_nonpositive")
    if not complete_months:
        failures.append("no_complete_inner_evaluation_month")
    elif any(sum(monthly[month], Decimal("0")) < 0 for month in complete_months):
        failures.append("complete_evaluation_month_negative")
    if evaluation_cash_per_base <= 0:
        failures.append("evaluation_total_cashflow_nonpositive")
    if break_even_bips <= 0:
        failures.append("no_positive_fee_break_even_after_current_spread")
    if not depth_passed:
        failures.append("current_common_quantity_depth_missing")
    return {
        "base_asset": base,
        "symbols": {"short": short_symbol, "long": long_symbol},
        "quote_orientation": {
            "short": quotes[short_symbol],
            "long": quotes[long_symbol],
        },
        "common_window": {
            "start_time_ms": common_start,
            "end_time_ms": common_end,
            "union_settlement_count": len(times),
            "per_symbol_settlement_counts": {
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
                selection_results[1]["sum"]
            ),
        },
        "evaluation": {
            "row_count": len(evaluation_times),
            "start_time_ms": evaluation_times[0],
            "end_time_ms": evaluation_times[-1],
            "conservative_usdt_cash_per_base_statistics": _statistics(
                evaluation_cashflows
            ),
            "first_chronological_half": _statistics(halves[0]),
            "second_chronological_half": _statistics(halves[1]),
            "calendar_months": monthly_rows,
            "conservative_usdt_cash_per_base_total": str(evaluation_cash_per_base),
            "conservative_usdt_cash_at_common_minimum_quantity": str(
                evaluation_cash_per_base * common_quantity
            ),
        },
        "current_execution_diagnostic": {
            "common_minimum_quantity": str(common_quantity),
            "all_four_top_sides_cover_common_quantity": depth_passed,
            "current_round_trip_spread_cost_usdt_per_base": str(spread_cost_per_base),
            "current_round_trip_spread_cost_at_common_quantity": str(
                spread_cost_per_base * common_quantity
            ),
            "equal_per_leg_fee_break_even_after_current_spread_bips": str(
                break_even_bips
            ),
            "future_exit_prices_assumed_current_for_diagnostic": True,
            "account_fee_evidence_present": False,
        },
        "failure_reasons": failures,
        "qualified_public_screen": not failures,
    }


def run(*, session: requests.Session | None = None) -> dict[str, object]:
    """Run the one-request cashflow adjudication without account access."""

    contract, source = _load_sources()
    started_ms = time.time_ns() // 1_000_000
    http = session or requests.Session()
    fx_raw, fx_receipt = _get_fx(http)
    fx_days = _fx_days(fx_raw)
    source_contract = _mapping(source["source_contract"], name="attempt1 source")
    raw_histories = _mapping(
        source_contract["funding_history_payloads"], name="funding histories"
    )
    contracts = _mapping(source["eligible_contracts"], name="eligible contracts")
    raw_books = {
        str(row["symbol"]): _mapping(row, name="book row")
        for row in _list(
            source_contract["book_ticker_selected_payload"], name="selected books"
        )
        if isinstance(row, Mapping)
    }
    pair_results = [
        _evaluate_pair(
            base=base,
            contracts={
                quote: _mapping(contract, name="eligible contract")
                for quote, contract in _mapping(
                    contracts[base], name="eligible quote contracts"
                ).items()
            },
            raw_histories=raw_histories,
            raw_books=raw_books,
            fx_days=fx_days,
        )
        for base in sorted(contracts)
    ]
    qualified = [row for row in pair_results if row["qualified_public_screen"]]
    finished_ms = time.time_ns() // 1_000_000
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.fromtimestamp(
            finished_ms / 1000, tz=UTC
        ).isoformat(),
        "started_at_ms": started_ms,
        "finished_at_ms": finished_ms,
        "new_request_count": 1,
        "source_contract": {
            "adjudication_contract_path": str(CONTRACT_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "adjudication_contract_result_sha256": contract["result_sha256"],
            "attempt1_result_path": str(SOURCE_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "attempt1_result_sha256": source["result_sha256"],
            "implementation_path": str(
                Path(__file__).resolve().relative_to(ROOT)
            ).replace("\\", "/"),
            "implementation_sha256": _sha256(Path(__file__).read_bytes()),
            "fx_request": fx_receipt,
            "fx_payload": fx_raw,
        },
        "attempt1_adjudication": {
            "attempt1_rate_only_qualified_pair_count": source["verdict"][
                "qualified_public_screen_count"
            ],
            "attempt1_timestamp_omission_observed": False,
            "attempt1_rate_only_economics_superseded": True,
        },
        "pair_results": pair_results,
        "verdict": {
            "status": (
                "cashflow_corrected_public_candidate_requires_authenticated_and_prospective_evidence"
                if qualified
                else "rejected_cashflow_corrected_cross_stablecoin_funding_differential"
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
