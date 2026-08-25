"""Run the frozen public screen for a delta-hedged BNB fee inventory."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
from math import ceil, gcd, lcm
from pathlib import Path
import time
from typing import Mapping, Sequence

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs" / "model-research" / "action-value"
CONTRACT_PATH = ACTION_VALUE / "binance-bnb-fee-discount-hedge-contract-v1.json"
JOURNAL_PATH = ACTION_VALUE / "binance-bnb-fee-discount-hedge-journal-v1.json"
RAW_ROOT = ACTION_VALUE / "raw" / "binance-bnb-fee-discount-hedge-v1"
DEFAULT_OUTPUT = (
    ACTION_VALUE / "binance-bnb-fee-discount-hedge-screen-v1-2026-08-25.json"
)
SYMBOL = "BNBUSDT"
HISTORY_LIMIT = 1000
MINIMUM_HISTORY_ROWS = 270
MAXIMUM_REQUESTS = 6
SCHEMA_VERSION = "binance-bnb-fee-discount-hedge-screen-v1"
JOURNAL_SCHEMA_VERSION = "binance-bnb-fee-discount-hedge-journal-v1"

REQUEST_SPECS: tuple[dict[str, object], ...] = (
    {
        "name": "spot_exchange_info",
        "method": "GET",
        "url": "https://api.binance.com/api/v3/exchangeInfo",
        "parameters": {"symbol": SYMBOL},
        "raw_filename": "01-spot-exchange-info.json",
    },
    {
        "name": "spot_book_ticker",
        "method": "GET",
        "url": "https://api.binance.com/api/v3/ticker/bookTicker",
        "parameters": {"symbol": SYMBOL},
        "raw_filename": "02-spot-book-ticker.json",
    },
    {
        "name": "futures_exchange_info",
        "method": "GET",
        "url": "https://fapi.binance.com/fapi/v1/exchangeInfo",
        "parameters": {},
        "raw_filename": "03-futures-exchange-info.json",
    },
    {
        "name": "futures_book_ticker",
        "method": "GET",
        "url": "https://fapi.binance.com/fapi/v1/ticker/bookTicker",
        "parameters": {"symbol": SYMBOL},
        "raw_filename": "04-futures-book-ticker.json",
    },
    {
        "name": "futures_premium_index",
        "method": "GET",
        "url": "https://fapi.binance.com/fapi/v1/premiumIndex",
        "parameters": {"symbol": SYMBOL},
        "raw_filename": "05-futures-premium-index.json",
    },
    {
        "name": "futures_funding_history",
        "method": "GET",
        "url": "https://fapi.binance.com/fapi/v1/fundingRate",
        "parameters": {"limit": HISTORY_LIMIT, "symbol": SYMBOL},
        "raw_filename": "06-futures-funding-history.json",
    },
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


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _embedded_hash(payload: Mapping[str, object], *, field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _write_hashed_json(path: Path, payload: dict[str, object], *, field: str) -> None:
    payload[field] = _embedded_hash(payload, field=field)
    write_bytes_atomic(path, (_canonical_json(payload) + "\n").encode("ascii"))


def _request_contract_rows() -> list[dict[str, object]]:
    return [
        {
            "name": str(spec["name"]),
            "method": "GET",
            "url": str(spec["url"]),
            "parameters": dict(_mapping(spec["parameters"], name="request parameters")),
            "raw_filename": str(spec["raw_filename"]),
        }
        for spec in REQUEST_SPECS
    ]


def _load_contract(path: Path = CONTRACT_PATH) -> dict[str, object]:
    contract = _mapping(json.loads(path.read_bytes()), name="screen contract")
    if contract.get("result_sha256") != _embedded_hash(contract, field="result_sha256"):
        raise ValueError("screen contract canonical hash mismatch")
    if contract.get("status") != "frozen_before_any_binance_market_request":
        raise ValueError("screen contract is not frozen before market access")
    source_binding = _mapping(contract.get("source_binding"), name="source binding")
    implementation_path = ROOT / str(source_binding.get("implementation_path") or "")
    if implementation_path.resolve() != Path(__file__).resolve():
        raise ValueError("screen contract implementation path mismatch")
    if source_binding.get("implementation_sha256") != _sha256(
        implementation_path.read_bytes()
    ):
        raise ValueError("screen contract implementation hash mismatch")
    public_plan = _mapping(
        contract.get("frozen_public_request_plan"), name="public request plan"
    )
    if public_plan.get("maximum_requests") != MAXIMUM_REQUESTS:
        raise ValueError("screen contract request cap mismatch")
    if public_plan.get("requests") != _request_contract_rows():
        raise ValueError("screen contract request sequence mismatch")
    return contract


def _new_journal(contract: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "contract_result_sha256": contract["result_sha256"],
        "implementation_sha256": _sha256(Path(__file__).read_bytes()),
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "status": "active",
        "next_request": None,
        "completed_request_count": 0,
        "responses": [],
    }


def _write_journal(path: Path, journal: dict[str, object]) -> None:
    _write_hashed_json(path, journal, field="journal_sha256")


def _create_journal(path: Path, contract: Mapping[str, object]) -> dict[str, object]:
    if path.exists():
        existing = _mapping(json.loads(path.read_bytes()), name="existing journal")
        if existing.get("journal_sha256") != _embedded_hash(
            existing, field="journal_sha256"
        ):
            raise ValueError("existing journal canonical hash mismatch")
        raise RuntimeError("one-use journal already exists; rerun is prohibited")
    journal = _new_journal(contract)
    _write_journal(path, journal)
    return journal


def _request_fingerprint(spec: Mapping[str, object]) -> dict[str, object]:
    return {
        "name": str(spec["name"]),
        "method": "GET",
        "url": str(spec["url"]),
        "parameters": dict(_mapping(spec["parameters"], name="request parameters")),
        "raw_filename": str(spec["raw_filename"]),
    }


def _capture_one(
    session: requests.Session,
    *,
    spec: Mapping[str, object],
    journal: dict[str, object],
    journal_path: Path,
    raw_root: Path,
) -> object:
    if int(journal["completed_request_count"]) >= MAXIMUM_REQUESTS:
        raise RuntimeError("frozen request cap reached")
    fingerprint = _request_fingerprint(spec)
    journal["next_request"] = fingerprint
    _write_journal(journal_path, journal)

    before_ms = time.time_ns() // 1_000_000
    response = session.get(
        str(fingerprint["url"]),
        params=_mapping(fingerprint["parameters"], name="request parameters"),
        timeout=30,
    )
    after_ms = time.time_ns() // 1_000_000
    raw = bytes(response.content)
    raw_path = raw_root / str(fingerprint["raw_filename"])
    write_bytes_atomic(raw_path, raw)

    headers = getattr(response, "headers", {})
    receipt = {
        "status_code": int(response.status_code),
        "payload_bytes": len(raw),
        "payload_sha256": _sha256(raw),
        "raw_path": _display_path(raw_path),
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "request_elapsed_ms": after_ms - before_ms,
        "content_type": str(headers.get("Content-Type", "")),
        "retry_after": str(headers.get("Retry-After", "")),
        "response_url": str(response.url),
    }
    responses = _list(journal["responses"], name="journal responses")
    responses.append({"request": fingerprint, "receipt": receipt})
    journal["responses"] = responses
    journal["completed_request_count"] = int(journal["completed_request_count"]) + 1
    journal["next_request"] = None
    _write_journal(journal_path, journal)

    if response.status_code == 429:
        raise RuntimeError("Binance rate limit reached; stopped without retry")
    if response.status_code != 200:
        raise RuntimeError(f"Binance returned HTTP {response.status_code}; no retry")
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{fingerprint['name']} raw body is not valid JSON; raw body retained"
        ) from exc


def _filter(contract: Mapping[str, object], filter_type: str) -> dict[str, object]:
    matches = [
        _mapping(item, name=f"{filter_type} filter")
        for item in _list(contract.get("filters"), name="contract filters")
        if isinstance(item, Mapping) and item.get("filterType") == filter_type
    ]
    if len(matches) != 1:
        raise ValueError(f"contract requires exactly one {filter_type} filter")
    return matches[0]


def _minimum_notional(contract: Mapping[str, object]) -> Decimal:
    filters = [
        _mapping(item, name="notional filter")
        for item in _list(contract.get("filters"), name="contract filters")
        if isinstance(item, Mapping)
        and item.get("filterType") in {"MIN_NOTIONAL", "NOTIONAL"}
    ]
    if len(filters) != 1:
        raise ValueError("contract requires exactly one notional filter")
    value = filters[0].get("minNotional", filters[0].get("notional"))
    return _decimal(value, name="minimum notional", positive=True)


def _spot_contract(raw: object) -> dict[str, object]:
    payload = _mapping(raw, name="spot exchange information")
    symbols = [
        _mapping(row, name="spot symbol")
        for row in _list(payload.get("symbols"), name="spot symbols")
        if isinstance(row, Mapping) and row.get("symbol") == SYMBOL
    ]
    if len(symbols) != 1:
        raise ValueError("spot exchange information lacks one BNBUSDT symbol")
    contract = symbols[0]
    if (
        contract.get("status") != "TRADING"
        or contract.get("baseAsset") != "BNB"
        or contract.get("quoteAsset") != "USDT"
    ):
        raise ValueError("spot BNBUSDT identity is not eligible")
    return contract


def _futures_contract(raw: object) -> dict[str, object]:
    payload = _mapping(raw, name="futures exchange information")
    symbols = [
        _mapping(row, name="futures symbol")
        for row in _list(payload.get("symbols"), name="futures symbols")
        if isinstance(row, Mapping) and row.get("symbol") == SYMBOL
    ]
    if len(symbols) != 1:
        raise ValueError("futures exchange information lacks one BNBUSDT symbol")
    contract = symbols[0]
    if (
        contract.get("status") != "TRADING"
        or contract.get("contractType") != "PERPETUAL"
        or contract.get("baseAsset") != "BNB"
        or contract.get("quoteAsset") != "USDT"
        or contract.get("marginAsset") != "USDT"
    ):
        raise ValueError("futures BNBUSDT identity is not eligible")
    return contract


def _selected_contract(contract: Mapping[str, object]) -> dict[str, object]:
    return {
        "symbol": contract["symbol"],
        "status": contract["status"],
        "base_asset": contract["baseAsset"],
        "quote_asset": contract["quoteAsset"],
        "contract_type": contract.get("contractType", "SPOT"),
        "margin_asset": contract.get("marginAsset"),
        "lot_size": _filter(contract, "LOT_SIZE"),
        "minimum_notional": str(_minimum_notional(contract)),
    }


def _book(raw: object, *, name: str) -> dict[str, Decimal]:
    row = _mapping(raw, name=name)
    if row.get("symbol") != SYMBOL:
        raise ValueError(f"{name} symbol mismatch")
    result = {
        "bid": _decimal(row.get("bidPrice"), name=f"{name} bid", positive=True),
        "ask": _decimal(row.get("askPrice"), name=f"{name} ask", positive=True),
        "bid_quantity": _decimal(
            row.get("bidQty"), name=f"{name} bid quantity", positive=True
        ),
        "ask_quantity": _decimal(
            row.get("askQty"), name=f"{name} ask quantity", positive=True
        ),
    }
    if result["bid"] >= result["ask"]:
        raise ValueError(f"{name} book must have a positive spread")
    return result


def _premium(raw: object) -> dict[str, object]:
    row = _mapping(raw, name="premium index")
    if row.get("symbol") != SYMBOL:
        raise ValueError("premium index symbol mismatch")
    return {
        "symbol": SYMBOL,
        "mark_price": str(
            _decimal(row.get("markPrice"), name="mark price", positive=True)
        ),
        "index_price": str(
            _decimal(row.get("indexPrice"), name="index price", positive=True)
        ),
        "last_funding_rate": str(
            _decimal(row.get("lastFundingRate"), name="last funding rate")
        ),
        "next_funding_time": _integer(
            row.get("nextFundingTime"), name="next funding time", positive=True
        ),
        "time": _integer(row.get("time"), name="premium time", positive=True),
    }


def _funding_rows(raw: object) -> list[dict[str, object]]:
    source = _list(raw, name="funding history")
    if not MINIMUM_HISTORY_ROWS <= len(source) <= HISTORY_LIMIT:
        raise ValueError(
            f"funding history requires {MINIMUM_HISTORY_ROWS}..{HISTORY_LIMIT} rows"
        )
    rows: list[dict[str, object]] = []
    previous_time = 0
    for item in source:
        row = _mapping(item, name="funding row")
        if row.get("symbol") != SYMBOL:
            raise ValueError("funding row symbol mismatch")
        funding_time = _integer(
            row.get("fundingTime"), name="funding time", positive=True
        )
        if funding_time <= previous_time:
            raise ValueError("funding times must be unique and strictly increasing")
        rate = _decimal(row.get("fundingRate"), name="funding rate")
        mark = _decimal(row.get("markPrice"), name="funding mark", positive=True)
        rows.append(
            {
                "symbol": SYMBOL,
                "fundingTime": funding_time,
                "fundingRate": str(rate),
                "markPrice": str(mark),
            }
        )
        previous_time = funding_time
    return rows


def _fraction_lcm(values: Sequence[Fraction]) -> Fraction:
    return Fraction(
        lcm(*(value.numerator for value in values)),
        gcd(*(value.denominator for value in values)),
    )


def _common_quantity(
    *,
    spot_contract: Mapping[str, object],
    futures_contract: Mapping[str, object],
    spot_ask: Decimal,
    futures_bid: Decimal,
) -> Decimal:
    contracts = (spot_contract, futures_contract)
    lot_filters = tuple(_filter(contract, "LOT_SIZE") for contract in contracts)
    minimums = [
        _decimal(row.get("minQty"), name="minimum quantity", positive=True)
        for row in lot_filters
    ]
    steps = [
        _decimal(row.get("stepSize"), name="quantity step", positive=True)
        for row in lot_filters
    ]
    minimums.extend(
        (
            _minimum_notional(spot_contract) / spot_ask,
            _minimum_notional(futures_contract) / futures_bid,
        )
    )
    common_step = _fraction_lcm(tuple(Fraction(step) for step in steps))
    multiple = ceil(Fraction(max(minimums)) / common_step)
    quantity_fraction = common_step * multiple
    quantity = Decimal(quantity_fraction.numerator) / Decimal(
        quantity_fraction.denominator
    )
    if any(quantity % step != 0 for step in steps):
        raise ValueError("common quantity does not satisfy both step sizes")
    for contract, lot in zip(contracts, lot_filters, strict=True):
        maximum = _decimal(lot.get("maxQty"), name="maximum quantity", positive=True)
        if quantity > maximum:
            raise ValueError(f"{contract['symbol']} common quantity exceeds maximum")
    return quantity


def _statistics(values: Sequence[Decimal]) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "sum": "0",
            "mean": None,
            "minimum": None,
            "maximum": None,
            "positive_count": 0,
            "zero_count": 0,
            "negative_count": 0,
        }
    return {
        "count": len(values),
        "sum": str(sum(values, Decimal("0"))),
        "mean": str(sum(values, Decimal("0")) / len(values)),
        "minimum": str(min(values)),
        "maximum": str(max(values)),
        "positive_count": sum(value > 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
    }


def _drawdown_bips(values: Sequence[Decimal]) -> Decimal:
    cumulative = Decimal("0")
    peak = Decimal("0")
    worst = Decimal("0")
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        worst = max(worst, peak - cumulative)
    return worst * Decimal("10000")


def _nearest_rank_75(values: Sequence[Decimal]) -> Decimal:
    if not values:
        raise ValueError("volatility threshold requires returns")
    ordered = sorted(values)
    return ordered[ceil(Decimal("0.75") * len(ordered)) - 1]


def _funding_evaluation(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rates = [_decimal(row["fundingRate"], name="funding rate") for row in rows]
    marks = [
        _decimal(row["markPrice"], name="funding mark", positive=True) for row in rows
    ]
    midpoint = len(rows) // 2
    oldest_rates = rates[:midpoint]
    newest_rates = rates[midpoint:]
    returns_bips = [
        (marks[index] / marks[index - 1] - 1) * Decimal("10000")
        for index in range(1, len(marks))
    ]
    selection_abs_returns = [abs(value) for value in returns_bips[: midpoint - 1]]
    volatility_threshold = _nearest_rank_75(selection_abs_returns)

    regime_values: dict[str, list[Decimal]] = defaultdict(list)
    for index in range(midpoint, len(rows)):
        move = returns_bips[index - 1]
        direction = (
            "down"
            if move < Decimal("-10")
            else "up"
            if move > Decimal("10")
            else "sideways"
        )
        volatility = "high" if abs(move) >= volatility_threshold else "regular"
        regime_values[f"direction_{direction}"].append(rates[index])
        regime_values[f"volatility_{volatility}"].append(rates[index])

    monthly: dict[str, list[Decimal]] = defaultdict(list)
    for row, rate in zip(rows, rates, strict=True):
        stamp = datetime.fromtimestamp(int(row["fundingTime"]) / 1000, tz=UTC)
        monthly[stamp.strftime("%Y-%m")].append(rate)
    month_keys = sorted(monthly)
    complete_month_keys = month_keys[1:-1]
    month_rows = [
        {
            "month": key,
            "complete_inner_month": key in complete_month_keys,
            **_statistics(monthly[key]),
            "cumulative_short_funding_bips": str(
                sum(monthly[key], Decimal("0")) * Decimal("10000")
            ),
        }
        for key in month_keys
    ]
    if not complete_month_keys:
        raise ValueError("funding history lacks a complete inner calendar month")
    complete_month_bips = [
        sum(monthly[key], Decimal("0")) * Decimal("10000")
        for key in complete_month_keys
    ]
    return {
        "row_count": len(rows),
        "start_time_ms": rows[0]["fundingTime"],
        "end_time_ms": rows[-1]["fundingTime"],
        "short_cashflow_sign": "positive_funding_rate_is_cash_received_by_short",
        "all_rows": {
            **_statistics(rates),
            "cumulative_short_funding_bips": str(
                sum(rates, Decimal("0")) * Decimal("10000")
            ),
            "worst_cumulative_drawdown_bips": str(_drawdown_bips(rates)),
        },
        "oldest_half": {
            **_statistics(oldest_rates),
            "cumulative_short_funding_bips": str(
                sum(oldest_rates, Decimal("0")) * Decimal("10000")
            ),
        },
        "newest_half": {
            **_statistics(newest_rates),
            "cumulative_short_funding_bips": str(
                sum(newest_rates, Decimal("0")) * Decimal("10000")
            ),
            "worst_cumulative_drawdown_bips": str(_drawdown_bips(newest_rates)),
        },
        "selection_only_absolute_return_75th_percentile_bips": str(
            volatility_threshold
        ),
        "newest_half_regimes": {
            key: {
                **_statistics(values),
                "cumulative_short_funding_bips": str(
                    sum(values, Decimal("0")) * Decimal("10000")
                ),
            }
            for key, values in sorted(regime_values.items())
        },
        "calendar_months": month_rows,
        "complete_inner_month_count": len(complete_month_keys),
        "worst_complete_inner_month_short_funding_bips": str(min(complete_month_bips)),
    }


def _execution(
    *,
    spot_contract: Mapping[str, object],
    futures_contract: Mapping[str, object],
    spot_book: Mapping[str, Decimal],
    futures_book: Mapping[str, Decimal],
) -> dict[str, object]:
    quantity = _common_quantity(
        spot_contract=spot_contract,
        futures_contract=futures_contract,
        spot_ask=spot_book["ask"],
        futures_bid=futures_book["bid"],
    )
    spot_mid = (spot_book["bid"] + spot_book["ask"]) / 2
    futures_mid = (futures_book["bid"] + futures_book["ask"]) / 2
    round_trip_spread_bips = (
        (spot_book["ask"] - spot_book["bid"]) / spot_mid
        + (futures_book["ask"] - futures_book["bid"]) / futures_mid
    ) * Decimal("10000")
    return {
        "common_minimum_base_quantity": str(quantity),
        "minimum_spot_inventory_usdt": str(quantity * spot_book["ask"]),
        "minimum_futures_short_notional_usdt": str(quantity * futures_book["bid"]),
        "top_book_depth_covers_common_quantity": all(
            row >= quantity
            for row in (
                spot_book["bid_quantity"],
                spot_book["ask_quantity"],
                futures_book["bid_quantity"],
                futures_book["ask_quantity"],
            )
        ),
        "spot_round_trip_spread_bips": str(
            (spot_book["ask"] - spot_book["bid"]) / spot_mid * Decimal("10000")
        ),
        "futures_round_trip_spread_bips": str(
            (futures_book["ask"] - futures_book["bid"]) / futures_mid * Decimal("10000")
        ),
        "combined_round_trip_spread_bips": str(round_trip_spread_bips),
        "current_futures_minus_spot_mid_basis_bips": str(
            (futures_mid / spot_mid - 1) * Decimal("10000")
        ),
        "spot_book": {key: str(value) for key, value in spot_book.items()},
        "futures_book": {key: str(value) for key, value in futures_book.items()},
        "exit_basis_locked": False,
    }


def _turnover_scenarios(
    *, evaluation: Mapping[str, object], execution: Mapping[str, object]
) -> list[dict[str, object]]:
    newest = _mapping(evaluation["newest_half"], name="newest half")
    spread = _decimal(execution["combined_round_trip_spread_bips"], name="spread bips")
    drawdown = _decimal(
        newest["worst_cumulative_drawdown_bips"], name="funding drawdown bips"
    )
    worst_month = _decimal(
        evaluation["worst_complete_inner_month_short_funding_bips"],
        name="worst monthly funding bips",
    )
    scenarios = (
        ("ten_bips_standard_25_percent_discount", Decimal("10"), Decimal("0.25")),
        ("five_bips_standard_25_percent_discount", Decimal("5"), Decimal("0.25")),
        ("ten_bips_standard_10_percent_discount", Decimal("10"), Decimal("0.10")),
    )
    rows: list[dict[str, object]] = []
    for name, standard_bips, discount_fraction in scenarios:
        savings_bips = standard_bips * discount_fraction
        rows.append(
            {
                "scenario": name,
                "non_authoritative_account_example": True,
                "standard_commission_bips": str(standard_bips),
                "discount_fraction": str(discount_fraction),
                "eligible_fee_savings_bips_per_spot_notional": str(savings_bips),
                "one_time_turnover_multiple_to_cover_top_spreads": str(
                    spread / savings_bips
                ),
                "worst_monthly_turnover_multiple_to_cover_negative_funding": str(
                    max(Decimal("0"), -worst_month) / savings_bips
                ),
                "newest_half_turnover_multiple_to_cover_funding_drawdown": str(
                    drawdown / savings_bips
                ),
                "combined_spread_and_funding_drawdown_turnover_multiple": str(
                    (spread + drawdown) / savings_bips
                ),
            }
        )
    return rows


def _terminalize_journal(
    path: Path, journal: dict[str, object], error: Exception
) -> None:
    journal["status"] = "terminal_failure"
    journal["failure"] = {"type": type(error).__name__, "message": str(error)}
    _write_journal(path, journal)


def run(
    *,
    session: requests.Session | None = None,
    contract_path: Path = CONTRACT_PATH,
    journal_path: Path = JOURNAL_PATH,
    raw_root: Path = RAW_ROOT,
) -> dict[str, object]:
    """Capture and evaluate the source-bound one-use public experiment."""

    contract = _load_contract(contract_path)
    journal = _create_journal(journal_path, contract)
    http = session or requests.Session()
    payloads: dict[str, object] = {}
    started_ms = time.time_ns() // 1_000_000
    try:
        for spec in REQUEST_SPECS:
            payloads[str(spec["name"])] = _capture_one(
                http,
                spec=spec,
                journal=journal,
                journal_path=journal_path,
                raw_root=raw_root,
            )
        journal["status"] = "data_complete"
        _write_journal(journal_path, journal)

        spot_contract = _spot_contract(payloads["spot_exchange_info"])
        futures_contract = _futures_contract(payloads["futures_exchange_info"])
        spot_book = _book(payloads["spot_book_ticker"], name="spot book")
        futures_book = _book(payloads["futures_book_ticker"], name="futures book")
        premium = _premium(payloads["futures_premium_index"])
        funding_rows = _funding_rows(payloads["futures_funding_history"])
        evaluation = _funding_evaluation(funding_rows)
        execution = _execution(
            spot_contract=spot_contract,
            futures_contract=futures_contract,
            spot_book=spot_book,
            futures_book=futures_book,
        )
        scenarios = _turnover_scenarios(evaluation=evaluation, execution=execution)
        gate = _mapping(contract["decision_gate"], name="decision gate")
        primary = next(
            row
            for row in scenarios
            if row["scenario"] == gate["primary_non_authoritative_scenario"]
        )
        regimes = _mapping(
            evaluation["newest_half_regimes"], name="newest half regimes"
        )
        required_regimes = {
            "direction_down",
            "direction_sideways",
            "direction_up",
            "volatility_high",
            "volatility_regular",
        }
        failure_reasons: list[str] = []
        if not bool(execution["top_book_depth_covers_common_quantity"]):
            failure_reasons.append("current_minimum_common_quantity_lacks_top_depth")
        if int(evaluation["complete_inner_month_count"]) < int(
            gate["minimum_complete_inner_months"]
        ):
            failure_reasons.append("insufficient_complete_inner_months")
        if set(regimes) != required_regimes or any(
            int(_mapping(regimes[key], name=key)["count"])
            < int(gate["minimum_rows_per_newest_half_regime"])
            for key in required_regimes
        ):
            failure_reasons.append("insufficient_newest_half_cross_regime_coverage")
        if _decimal(
            primary["worst_monthly_turnover_multiple_to_cover_negative_funding"],
            name="monthly turnover multiple",
        ) > _decimal(
            gate["maximum_primary_worst_monthly_turnover_multiple"],
            name="monthly turnover gate",
        ):
            failure_reasons.append("primary_worst_monthly_turnover_gate_failed")
        if _decimal(
            primary["combined_spread_and_funding_drawdown_turnover_multiple"],
            name="combined turnover multiple",
        ) > _decimal(
            gate["maximum_primary_combined_turnover_multiple"],
            name="combined turnover gate",
        ):
            failure_reasons.append("primary_combined_turnover_gate_failed")
    except Exception as exc:
        _terminalize_journal(journal_path, journal, exc)
        raise

    finished_ms = time.time_ns() // 1_000_000
    journal_bytes = journal_path.read_bytes()
    qualified = not failure_reasons
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.fromtimestamp(
            finished_ms / 1000, tz=UTC
        ).isoformat(),
        "started_at_ms": started_ms,
        "finished_at_ms": finished_ms,
        "request_count": journal["completed_request_count"],
        "source_binding": {
            "contract_path": _display_path(contract_path),
            "contract_result_sha256": contract["result_sha256"],
            "implementation_path": _display_path(Path(__file__)),
            "implementation_sha256": _sha256(Path(__file__).read_bytes()),
            "journal_path": _display_path(journal_path),
            "journal_file_sha256": _sha256(journal_bytes),
            "journal_sha256": journal["journal_sha256"],
            "raw_response_count": journal["completed_request_count"],
        },
        "mechanism": {
            "inventory_leg": "long_spot_BNB_reserved_only_for_eligible_fee_payment",
            "hedge_leg": "short_equal_base_BNBUSDT_perpetual",
            "fee_savings_identity": "eligible_spot_notional_times_account_standard_commission_rate_times_account_discount_fraction",
            "first_order_bnb_direction_exposure": "neutral_only_while_spot_inventory_and_short_base_quantity_remain_equal",
            "inventory_consumption_requires_rehedging": True,
            "standalone_profit_strategy": False,
        },
        "selected_contracts": {
            "spot": _selected_contract(spot_contract),
            "futures": _selected_contract(futures_contract),
        },
        "current_execution": execution,
        "current_premium_index": premium,
        "funding_evaluation": evaluation,
        "turnover_break_even_scenarios": scenarios,
        "funding_history_payload": funding_rows,
        "failure_reasons": failure_reasons,
        "verdict": {
            "status": (
                "public_cost_edge_candidate_requires_authenticated_account_commission_turnover_and_realized_fee_evidence"
                if qualified
                else "rejected_public_bnb_fee_discount_hedge_prequalification"
            ),
            "qualified_public_prequalification": qualified,
            "accepted_edge": False,
            "profitability_claim": False,
            "credentials_used": False,
            "signed_requests_made": 0,
            "orders_placed": False,
            "trading_authority": False,
        },
    }
    result["result_sha256"] = _embedded_hash(result, field="result_sha256")
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
        "credentials_used": False,
        "signed_requests_made": 0,
        "orders_placed": False,
        "trading_authority": False,
    }
    result["result_sha256"] = _embedded_hash(result, field="result_sha256")
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
    print(f"request_count={result['request_count']}")
    print(
        "qualified_public_prequalification="
        f"{result['verdict']['qualified_public_prequalification']}"
    )
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
