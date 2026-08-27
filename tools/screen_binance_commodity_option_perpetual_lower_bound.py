"""Run one frozen Binance commodity-option/perpetual lower-bound screen."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


OPTIONS_BASE_URL = "https://eapi.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"
SCHEMA_VERSION = "binance-commodity-option-perpetual-lower-bound-v1"
UNDERLYINGS = ("XAGUSDT", "XAUUSDT")
FIXED_STRESS_BPS = Decimal("33.5")
FUNDING_WINDOW_EVENTS = 8


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


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _canonical_hash(payload: Mapping[str, object], *, field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return _sha256(_canonical_json(body).encode("ascii"))


def _filter(symbol: Mapping[str, object], filter_type: str) -> dict[str, object]:
    for value in _list(symbol.get("filters"), name="symbol filters"):
        row = _mapping(value, name="symbol filter")
        if row.get("filterType") == filter_type:
            return row
    raise ValueError(f"{symbol.get('symbol')} lacks {filter_type}")


def _minimum_notional(symbol: Mapping[str, object]) -> Decimal:
    for filter_name in ("MIN_NOTIONAL", "NOTIONAL"):
        try:
            row = _filter(symbol, filter_name)
        except ValueError:
            continue
        value = row.get("minNotional") or row.get("notional")
        if value is not None:
            return Decimal(str(value))
    raise ValueError(f"{symbol.get('symbol')} lacks minimum notional")


def _option_step(symbol: Mapping[str, object]) -> Decimal:
    scale = int(symbol["quantityScale"])
    if scale < 0 or scale > 18:
        raise ValueError(f"{symbol.get('symbol')} has invalid quantityScale")
    return Decimal(1).scaleb(-scale)


def _common_step(left: Decimal, right: Decimal) -> Decimal:
    step = max(left, right)
    smaller = min(left, right)
    if step <= 0 or smaller <= 0 or step % smaller != 0:
        raise ValueError("quantity steps do not have an exact common coarser step")
    return step


def _round_up(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


def _active_commodity_options(exchange: Mapping[str, object]) -> list[dict[str, object]]:
    rows = [
        _mapping(value, name="option symbol")
        for value in _list(exchange.get("optionSymbols"), name="option symbols")
    ]
    selected = [
        row
        for row in rows
        if row.get("status") == "TRADING"
        and row.get("contractType") == "TRADFI_OPTIONS"
        and row.get("underlyingType") == "COMMODITY"
        and row.get("underlying") in UNDERLYINGS
    ]
    selected.sort(key=lambda row: str(row["symbol"]))
    return selected


def _population_hash(rows: list[Mapping[str, object]]) -> str:
    symbols = [str(row["symbol"]) for row in rows]
    return _sha256("\n".join(symbols).encode("ascii"))


def _load_retained_exchange_info(
    raw_dir: Path, journal_path: Path
) -> tuple[object, dict[str, object]]:
    if not journal_path.is_file():
        raise FileNotFoundError("retained one-request journal is absent")
    journal_lines = journal_path.read_text(encoding="ascii").splitlines()
    if len(journal_lines) != 1:
        raise ValueError("resume requires exactly one retained request receipt")
    receipt = _mapping(json.loads(journal_lines[0]), name="retained request receipt")
    expected_url = f"{OPTIONS_BASE_URL}/eapi/v1/exchangeInfo"
    if (
        receipt.get("name") != "options-exchange-info"
        or receipt.get("method") != "GET"
        or receipt.get("url") != expected_url
        or receipt.get("status_code") != 200
    ):
        raise ValueError("retained request receipt is not the expected successful GET")
    raw_path = raw_dir / "options-exchange-info.raw"
    payload = raw_path.read_bytes()
    if (
        len(payload) != int(receipt["response_bytes"])
        or _sha256(payload) != receipt["response_sha256"]
    ):
        raise ValueError("retained exchange-info response fails receipt integrity")
    return json.loads(payload), receipt


def _funding_stress(
    rows: list[object], *, symbol: str, hedge_direction: str
) -> tuple[Decimal, dict[str, object]]:
    parsed: list[tuple[int, Decimal]] = []
    for value in rows:
        row = _mapping(value, name=f"{symbol} funding row")
        if row.get("symbol") != symbol:
            raise ValueError(f"{symbol} funding history contains another symbol")
        event_time = int(row["fundingTime"])
        rate = Decimal(str(row["fundingRate"]))
        parsed.append((event_time, rate))
    parsed.sort(key=lambda value: value[0])
    if len(parsed) < FUNDING_WINDOW_EVENTS:
        raise ValueError(f"{symbol} has fewer than eight retained funding events")
    if len({event_time for event_time, _ in parsed}) != len(parsed):
        raise ValueError(f"{symbol} funding history contains duplicate event times")
    adverse = [
        max(-rate, Decimal(0))
        if hedge_direction == "short"
        else max(rate, Decimal(0))
        for _, rate in parsed
    ]
    windows = [
        sum(adverse[index : index + FUNDING_WINDOW_EVENTS], Decimal(0))
        for index in range(len(adverse) - FUNDING_WINDOW_EVENTS + 1)
    ]
    worst = max(windows)
    worst_index = windows.index(worst)
    return worst, {
        "retained_event_count": len(parsed),
        "first_funding_time_ms": parsed[0][0],
        "last_funding_time_ms": parsed[-1][0],
        "worst_window_first_funding_time_ms": parsed[worst_index][0],
        "worst_window_last_funding_time_ms": parsed[
            worst_index + FUNDING_WINDOW_EVENTS - 1
        ][0],
        "worst_adverse_eight_event_rate_sum": _decimal_text(worst),
    }


class _Client:
    def __init__(self, raw_dir: Path, journal_path: Path, *, replay: bool = False) -> None:
        self.raw_dir = raw_dir
        self.journal_path = journal_path
        self.replay = replay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept-Encoding": "identity",
                "User-Agent": "simple-ai-trading-public-edge-research/1.0",
            }
        )
        self.replay_receipts: dict[str, dict[str, object]] = {}
        if replay:
            lines = journal_path.read_text(encoding="ascii").splitlines()
            if len(lines) != 7:
                raise ValueError("offline replay requires exactly seven receipts")
            receipts = [
                _mapping(json.loads(line), name="retained request receipt")
                for line in lines
            ]
            self.replay_receipts = {str(row["name"]): row for row in receipts}
            if len(self.replay_receipts) != 7:
                raise ValueError("offline replay receipts contain duplicate names")

    def get(
        self,
        url: str,
        *,
        name: str,
        params: Mapping[str, object] | None = None,
    ) -> tuple[object, dict[str, object]]:
        if self.replay:
            if name not in self.replay_receipts:
                raise ValueError(f"offline replay lacks receipt {name}")
            receipt = self.replay_receipts[name]
            prepared = requests.Request("GET", url, params=params).prepare()
            if (
                receipt.get("method") != "GET"
                or receipt.get("url") != prepared.url
                or receipt.get("status_code") != 200
            ):
                raise ValueError(f"offline replay receipt identity differs for {name}")
            raw_path = self.raw_dir / f"{name}.raw"
            payload = raw_path.read_bytes()
            if (
                len(payload) != int(receipt["response_bytes"])
                or _sha256(payload) != receipt["response_sha256"]
            ):
                raise ValueError(f"offline replay response fails integrity for {name}")
            return json.loads(payload), receipt
        started_ms = time.time_ns() // 1_000_000
        response = self.session.get(url, params=params, timeout=30)
        finished_ms = time.time_ns() // 1_000_000
        payload = response.content
        raw_path = self.raw_dir / f"{name}.raw"
        if raw_path.exists():
            raise FileExistsError(
                f"refusing to overwrite retained response: {raw_path}"
            )
        write_bytes_atomic(raw_path, payload)
        receipt = {
            "name": name,
            "method": "GET",
            "url": response.url,
            "status_code": response.status_code,
            "requested_at_ms": started_ms,
            "completed_at_ms": finished_ms,
            "response_bytes": len(payload),
            "response_sha256": _sha256(payload),
            "raw_path": str(raw_path.as_posix()),
        }
        line = (_canonical_json(receipt) + "\n").encode("ascii")
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        response.raise_for_status()
        try:
            return response.json(), receipt
        except requests.JSONDecodeError as exc:
            raise ValueError(f"{name} did not return JSON") from exc


def run(
    *,
    contract_path: Path,
    raw_dir: Path,
    journal_path: Path,
    resume_after_retained_exchange_info: bool = False,
    replay_retained_capture: bool = False,
) -> dict[str, object]:
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="utf-8")), name="contract"
    )
    contract_hash = _canonical_hash(contract, field="contract_sha256")
    if contract_hash != contract.get("contract_sha256"):
        raise ValueError("contract hash does not match canonical contents")
    frozen = _mapping(
        contract.get("discovery_boundary_already_observed"),
        name="frozen discovery boundary",
    )

    if resume_after_retained_exchange_info and replay_retained_capture:
        raise ValueError("resume and full offline replay are mutually exclusive")
    client = _Client(raw_dir, journal_path, replay=replay_retained_capture)
    if resume_after_retained_exchange_info:
        options_exchange_raw, options_exchange_source = _load_retained_exchange_info(
            raw_dir, journal_path
        )
    else:
        options_exchange_raw, options_exchange_source = client.get(
            f"{OPTIONS_BASE_URL}/eapi/v1/exchangeInfo",
            name="options-exchange-info",
        )
    options_exchange = _mapping(options_exchange_raw, name="options exchange info")
    options = _active_commodity_options(options_exchange)
    observed_population_hash = _population_hash(options)
    if len(options) != int(frozen["active_TRADFI_OPTIONS_COMMODITY_XAU_XAG_row_count"]):
        raise ValueError("active commodity option population count changed")
    if observed_population_hash != frozen["sorted_symbol_population_sha256"]:
        raise ValueError("active commodity option population hash changed")

    options_tickers_raw, options_tickers_source = client.get(
        f"{OPTIONS_BASE_URL}/eapi/v1/ticker",
        name="options-all-tickers",
    )
    futures_exchange_raw, futures_exchange_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/exchangeInfo",
        name="futures-exchange-info",
    )
    futures_books_raw, futures_books_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/ticker/bookTicker",
        name="futures-all-book-tickers",
    )
    premium_raw, premium_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/premiumIndex",
        name="futures-all-premium-index",
    )
    funding_raw: dict[str, list[object]] = {}
    funding_sources: list[dict[str, object]] = []
    for underlying in UNDERLYINGS:
        response, source = client.get(
            f"{FUTURES_BASE_URL}/fapi/v1/fundingRate",
            name=f"futures-funding-{underlying.lower()}",
            params={"symbol": underlying, "limit": 1000},
        )
        funding_raw[underlying] = _list(
            response, name=f"{underlying} funding history"
        )
        funding_sources.append(source)

    option_tickers = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="option ticker")
            for value in _list(options_tickers_raw, name="option tickers")
        )
    }
    futures_exchange = _mapping(futures_exchange_raw, name="futures exchange info")
    futures_symbols = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="futures symbol")
            for value in _list(futures_exchange.get("symbols"), name="futures symbols")
        )
    }
    futures_books = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="futures book")
            for value in _list(futures_books_raw, name="futures books")
        )
    }
    premiums = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="premium index row")
            for value in _list(premium_raw, name="premium indexes")
        )
    }
    if len(option_tickers) != len(
        _list(options_tickers_raw, name="option tickers")
    ):
        raise ValueError("option tickers contain duplicate symbols")

    funding_stress: dict[tuple[str, str], tuple[Decimal, dict[str, object]]] = {}
    for underlying in UNDERLYINGS:
        if (
            underlying not in futures_symbols
            or underlying not in futures_books
            or underlying not in premiums
        ):
            raise ValueError(f"{underlying} is absent from futures public data")
        info = futures_symbols[underlying]
        if (
            info.get("status") != "TRADING"
            or info.get("contractType") != "TRADIFI_PERPETUAL"
            or info.get("underlyingType") != "COMMODITY"
            or info.get("marginAsset") != "USDT"
        ):
            raise ValueError(f"{underlying} identity or trading state changed")
        funding_stress[(underlying, "short")] = _funding_stress(
            funding_raw[underlying], symbol=underlying, hedge_direction="short"
        )
        funding_stress[(underlying, "long")] = _funding_stress(
            funding_raw[underlying], symbol=underlying, hedge_direction="long"
        )

    rows: list[dict[str, object]] = []
    for option in options:
        symbol = str(option["symbol"])
        underlying = str(option["underlying"])
        if symbol not in option_tickers:
            raise ValueError(f"{symbol} is absent from the all-options ticker")
        ticker = option_tickers[symbol]
        futures_info = futures_symbols[underlying]
        futures_book = futures_books[underlying]
        option_side = str(option["side"])
        if option_side not in {"CALL", "PUT"}:
            raise ValueError(f"{symbol} has unsupported option side")
        if Decimal(str(option["unit"])) != Decimal(1):
            raise ValueError(f"{symbol} contract unit changed")
        if option.get("quoteAsset") != "USDT":
            raise ValueError(f"{symbol} quote asset changed")

        option_ask = Decimal(str(ticker["askPrice"]))
        option_ask_qty_value = ticker.get("askQty")
        option_ask_qty = (
            Decimal(str(option_ask_qty_value))
            if option_ask_qty_value is not None
            else None
        )
        futures_entry = Decimal(
            str(
                futures_book["bidPrice"]
                if option_side == "CALL"
                else futures_book["askPrice"]
            )
        )
        futures_entry_qty = Decimal(
            str(
                futures_book["bidQty"]
                if option_side == "CALL"
                else futures_book["askQty"]
            )
        )
        row: dict[str, object] = {
            "symbol": symbol,
            "underlying": underlying,
            "expiry_date_ms": int(option["expiryDate"]),
            "option_side": option_side,
            "strike_price": _decimal_text(Decimal(str(option["strikePrice"]))),
            "option_best_ask": _decimal_text(option_ask),
            "option_best_ask_quantity": (
                None if option_ask_qty is None else _decimal_text(option_ask_qty)
            ),
            "option_ask_quantity_available": option_ask_qty is not None,
            "perpetual_entry_side": "bid" if option_side == "CALL" else "ask",
            "perpetual_entry_price": _decimal_text(futures_entry),
            "perpetual_entry_quantity": _decimal_text(futures_entry_qty),
            "option_close_time_ms": ticker.get("closeTime"),
            "perpetual_book_time_ms": futures_book.get("time"),
        }
        if min(option_ask, futures_entry, futures_entry_qty) <= 0:
            row.update(
                {
                    "quoted_positive_ask": False,
                    "top_level_capacity_passes": False,
                    "failure_reason": "positive_option_ask_and_perpetual_entry_book_required",
                }
            )
            rows.append(row)
            continue

        futures_lot = _filter(futures_info, "LOT_SIZE")
        step = _common_step(
            _option_step(option), Decimal(str(futures_lot["stepSize"]))
        )
        minimum_quantity = max(
            Decimal(str(option["minQty"])),
            Decimal(str(futures_lot["minQty"])),
            _minimum_notional(futures_info) / futures_entry,
        )
        quantity = _round_up(minimum_quantity, step)
        known_maximum_quantity = min(
            Decimal(str(option["maxQty"])),
            Decimal(str(futures_lot["maxQty"])),
            futures_entry_qty,
        )
        maximum_quantity = (
            min(known_maximum_quantity, option_ask_qty)
            if option_ask_qty is not None
            else None
        )
        capacity_ok = maximum_quantity is not None and quantity <= maximum_quantity
        strike = Decimal(str(option["strikePrice"]))
        hedge_direction = "short" if option_side == "CALL" else "long"
        funding_rate, funding_metadata = funding_stress[
            (underlying, hedge_direction)
        ]
        indicative_underlying_notional = futures_entry * quantity
        indicative_gross = (
            (
                futures_entry - strike - option_ask
                if option_side == "CALL"
                else strike - futures_entry - option_ask
            )
            * quantity
        )
        indicative_gross_bps = (
            indicative_gross / indicative_underlying_notional * Decimal(10_000)
        )
        indicative_fixed_stress = (
            indicative_underlying_notional * FIXED_STRESS_BPS / Decimal(10_000)
        )
        indicative_adverse_funding_stress = (
            indicative_underlying_notional * funding_rate
        )
        indicative_after_stress = (
            indicative_gross
            - indicative_fixed_stress
            - indicative_adverse_funding_stress
        )
        underlying_notional = (
            indicative_underlying_notional if capacity_ok else Decimal(0)
        )
        gross = indicative_gross if capacity_ok else Decimal(0)
        gross_bps = indicative_gross_bps if capacity_ok else Decimal(0)
        fixed_stress = indicative_fixed_stress if capacity_ok else Decimal(0)
        adverse_funding_stress = (
            indicative_adverse_funding_stress if capacity_ok else Decimal(0)
        )
        after_stress = indicative_after_stress if capacity_ok else Decimal(0)
        row.update(
            {
                "quoted_positive_ask": True,
                "common_quantity_step": _decimal_text(step),
                "minimum_common_quantity": _decimal_text(quantity),
                "maximum_known_quantity_without_option_book_quantity": (
                    _decimal_text(known_maximum_quantity)
                ),
                "maximum_top_level_quantity": (
                    None if maximum_quantity is None else _decimal_text(maximum_quantity)
                ),
                "top_level_capacity_passes": capacity_ok,
                "failure_reason": (
                    None
                    if capacity_ok
                    else (
                        "all_options_ticker_omits_option_ask_quantity"
                        if option_ask_qty is None
                        else "minimum_common_quantity_exceeds_top_level_capacity"
                    )
                ),
                "hedge_direction": hedge_direction,
                "indicative_underlying_notional_usdt_at_minimum_quantity": (
                    _decimal_text(indicative_underlying_notional)
                ),
                "indicative_gross_terminal_payoff_floor_usdt": (
                    _decimal_text(indicative_gross)
                ),
                "indicative_gross_terminal_payoff_floor_bps": (
                    _decimal_text(indicative_gross_bps)
                ),
                "indicative_after_all_frozen_stress_usdt": (
                    _decimal_text(indicative_after_stress)
                ),
                "indicative_after_all_frozen_stress_positive": (
                    indicative_after_stress > 0
                ),
                "underlying_notional_usdt": _decimal_text(underlying_notional),
                "gross_terminal_payoff_floor_usdt": _decimal_text(gross),
                "gross_terminal_payoff_floor_bps": _decimal_text(gross_bps),
                "fixed_nonfunding_stress_bps": _decimal_text(FIXED_STRESS_BPS),
                "fixed_nonfunding_stress_usdt": _decimal_text(fixed_stress),
                "adverse_funding_stress_rate": _decimal_text(funding_rate),
                "adverse_funding_stress_usdt": _decimal_text(adverse_funding_stress),
                "after_all_frozen_stress_usdt": _decimal_text(after_stress),
                "after_all_frozen_stress_positive": capacity_ok and after_stress > 0,
                "funding_stress_metadata": funding_metadata,
            }
        )
        rows.append(row)

    sources = [
        options_exchange_source,
        options_tickers_source,
        futures_exchange_source,
        futures_books_source,
        premium_source,
        *funding_sources,
    ]
    if len(sources) != 7:
        raise AssertionError("frozen request plan must contain exactly seven requests")
    started_ms = min(int(source["requested_at_ms"]) for source in sources)
    completed_ms = max(int(source["completed_at_ms"]) for source in sources)
    executable_rows = [row for row in rows if row.get("top_level_capacity_passes")]
    positive_gross = [
        row
        for row in executable_rows
        if Decimal(str(row["gross_terminal_payoff_floor_usdt"])) > 0
    ]
    positive_stressed = [
        row for row in executable_rows if row["after_all_frozen_stress_positive"]
    ]
    indicative_rows = [row for row in rows if row.get("quoted_positive_ask")]
    indicative_positive_gross = [
        row
        for row in indicative_rows
        if Decimal(str(row["indicative_gross_terminal_payoff_floor_usdt"])) > 0
    ]
    indicative_positive_stressed = [
        row
        for row in indicative_rows
        if row["indicative_after_all_frozen_stress_positive"]
    ]
    best = (
        max(
            executable_rows,
            key=lambda row: Decimal(str(row["gross_terminal_payoff_floor_bps"])),
        )
        if executable_rows
        else None
    )
    best_indicative = (
        max(
            indicative_rows,
            key=lambda row: Decimal(
                str(row["indicative_gross_terminal_payoff_floor_bps"])
            ),
        )
        if indicative_rows
        else None
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": {
            "path": str(contract_path.as_posix()),
            "sha256": contract_hash,
        },
        "authority": {
            "public_unauthenticated_GET_requests": len(sources),
            "authenticated_requests": 0,
            "account_state_accessed": False,
            "orders_quotes_transfers_or_wallet_actions": 0,
            "paper_or_live_trading_authority": False,
        },
        "capture": {
            "started_at_ms": started_ms,
            "completed_at_ms": completed_ms,
            "window_ms": completed_ms - started_ms,
            "request_count": len(sources),
            "all_status_codes_200": all(
                source["status_code"] == 200 for source in sources
            ),
            "raw_response_bytes": sum(
                int(source["response_bytes"]) for source in sources
            ),
            "journal_path": str(journal_path.as_posix()),
            "sources": sources,
        },
        "population": {
            "active_commodity_option_count": len(options),
            "sorted_symbol_population_sha256": observed_population_hash,
            "underlyings": list(UNDERLYINGS),
        },
        "funding_stress": {
            underlying: {
                direction: metadata
                for direction in ("short", "long")
                for _, metadata in [funding_stress[(underlying, direction)]]
            }
            for underlying in UNDERLYINGS
        },
        "economics": {
            "fixed_nonfunding_stress_bps": _decimal_text(FIXED_STRESS_BPS),
            "funding_window_events": FUNDING_WINDOW_EVENTS,
            "row_count": len(rows),
            "positive_ask_count": sum(row["quoted_positive_ask"] is True for row in rows),
            "option_ask_quantity_available_count": sum(
                row["option_ask_quantity_available"] is True for row in rows
            ),
            "top_level_capacity_pass_count": len(executable_rows),
            "gross_positive_count": len(positive_gross),
            "after_all_frozen_stress_positive_count": len(positive_stressed),
            "indicative_gross_positive_count_without_option_quantity": len(
                indicative_positive_gross
            ),
            "indicative_after_stress_positive_count_without_option_quantity": len(
                indicative_positive_stressed
            ),
            "best_symbol": None if best is None else best["symbol"],
            "best_gross_terminal_payoff_floor_bps": (
                None if best is None else best["gross_terminal_payoff_floor_bps"]
            ),
            "best_indicative_symbol_without_option_quantity": (
                None if best_indicative is None else best_indicative["symbol"]
            ),
            "best_indicative_gross_terminal_payoff_floor_bps": (
                None
                if best_indicative is None
                else best_indicative["indicative_gross_terminal_payoff_floor_bps"]
            ),
            "rows": rows,
        },
        "adjudication": {
            "status": (
                "unaccepted_public_lower_bound_candidate_survives_frozen_stress"
                if positive_stressed
                else "current_commodity_option_population_has_no_executable_positive_lower_bound_after_frozen_stress"
            ),
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "trading_authority": False,
            "public_stressed_candidate_count": len(positive_stressed),
            "retry_trigger": (
                "exact_account_commissions_and_access_then_frozen_synchronized_recurrence_capture"
                if positive_stressed
                else "new_listed_expiry_or_material_fee_funding_basis_book_or_product_access_change"
            ),
        },
        "limitations": [
            "option_and_perpetual_entries_are_non_atomic_and_captured_sequentially",
            "displayed_top_level_quantity_is_not_an_owned_fill",
            "perpetual_exit_basis_is_stressed_not_locked",
            "exact_account_access_commissions_margin_and_settlement_are_unbound",
            "one_public_snapshot_cannot_establish_recurrence_or_deployability",
            "the_all_options_ticker_endpoint_does_not_publish_option_ask_quantity",
        ],
        "implementation": {
            "path": "tools/screen_binance_commodity_option_perpetual_lower_bound.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume-after-retained-exchange-info",
        action="store_true",
        help="reuse one receipt-bound exchangeInfo response and issue only requests 2-7",
    )
    parser.add_argument(
        "--replay-retained-capture",
        action="store_true",
        help="perform no HTTP and replay exactly seven receipt-bound raw responses",
    )
    args = parser.parse_args()
    if (
        args.journal.exists()
        and not args.resume_after_retained_exchange_info
        and not args.replay_retained_capture
    ):
        raise FileExistsError(f"refusing to append to prior journal: {args.journal}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prior result: {args.output}")
    if (
        args.raw_dir.exists()
        and any(args.raw_dir.iterdir())
        and not args.resume_after_retained_exchange_info
        and not args.replay_retained_capture
    ):
        raise FileExistsError(f"refusing to reuse non-empty raw directory: {args.raw_dir}")
    result = run(
        contract_path=args.contract,
        raw_dir=args.raw_dir,
        journal_path=args.journal,
        resume_after_retained_exchange_info=args.resume_after_retained_exchange_info,
        replay_retained_capture=args.replay_retained_capture,
    )
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["adjudication"], indent=2))
    print(
        json.dumps(
            {key: value for key, value in result["economics"].items() if key != "rows"},
            indent=2,
        )
    )
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
