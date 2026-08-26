"""Capture one public Binance bStock-to-reference conversion-parity screen."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"
BSTOCK_LIST_URL = (
    "https://www.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/"
    "market/token/rwa/stock/detail/list/ai"
)
SCHEMA_VERSION = "binance-bstock-reference-conversion-parity-v1"
SIZES_USDT = (Decimal("1000"), Decimal("5000"))
SPOT_COST_SENSITIVITY_BIPS = Decimal("10")
STOCK_EXIT_COST_SENSITIVITY_BIPS = Decimal("10")
FUNDING_CARRY_TICKERS = ("DRAM", "MU", "MRVL", "SNDK")
ROUND_TRIP_TAKER_COST_SENSITIVITY_BIPS = Decimal("30")


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


class _Client:
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept-Encoding": "identity",
                "User-Agent": "simple-ai-trading-public-edge-research/1.0",
            }
        )

    def get(
        self,
        url: str,
        *,
        name: str,
        params: Mapping[str, object] | None = None,
    ) -> tuple[object, dict[str, object]]:
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.content
        raw_path = self.raw_dir / f"{name}.raw"
        write_bytes_atomic(raw_path, payload)
        receipt = {
            "method": "GET",
            "url": response.url,
            "status_code": response.status_code,
            "response_bytes": len(payload),
            "response_sha256": _sha256(payload),
            "raw_path": str(raw_path.as_posix()),
        }
        try:
            return response.json(), receipt
        except requests.JSONDecodeError as exc:
            raise ValueError(f"{name} did not return JSON") from exc


def _fill_asks(raw_asks: object, *, target_usdt: Decimal) -> dict[str, object]:
    remaining = target_usdt
    shares = Decimal(0)
    cost = Decimal(0)
    levels_used = 0
    for raw_level in _list(raw_asks, name="asks"):
        level = _list(raw_level, name="ask level")
        if len(level) != 2:
            raise ValueError("ask level must contain price and quantity")
        price = Decimal(str(level[0]))
        available = Decimal(str(level[1]))
        if price <= 0 or available <= 0:
            raise ValueError("ask price and quantity must be positive")
        quantity = min(available, remaining / price)
        shares += quantity
        cost += quantity * price
        remaining -= quantity * price
        levels_used += 1
        if remaining <= Decimal("0.00000001"):
            remaining = Decimal(0)
            break
    if remaining > 0 or shares <= 0:
        raise ValueError(f"depth cannot fill {target_usdt} USDT")
    return {
        "target_usdt": _decimal_text(target_usdt),
        "shares": _decimal_text(shares),
        "spot_cost_usdt": _decimal_text(cost),
        "spot_vwap": _decimal_text(cost / shares),
        "levels_used": levels_used,
    }


def _historical_diagnostic(
    spot_klines: object, index_klines: object
) -> dict[str, object]:
    spot_rows = _list(spot_klines, name="spot klines")
    index_rows = _list(index_klines, name="index klines")
    if len(spot_rows) >= 1000 or len(index_rows) >= 1000:
        raise ValueError("historical diagnostic reached its one-page limit")
    index_close: dict[int, Decimal] = {}
    for raw_row in index_rows:
        row = _list(raw_row, name="index kline")
        index_close[int(row[0])] = Decimal(str(row[4]))
    discounts: list[Decimal] = []
    timestamps: list[int] = []
    quote_volume = Decimal(0)
    for raw_row in spot_rows:
        row = _list(raw_row, name="spot kline")
        timestamp = int(row[0])
        if timestamp not in index_close or Decimal(str(row[7])) <= 0:
            continue
        spot_close = Decimal(str(row[4]))
        discounts.append(
            Decimal(10_000) * (index_close[timestamp] / spot_close - Decimal(1))
        )
        timestamps.append(timestamp)
        quote_volume += Decimal(str(row[7]))
    if not discounts:
        raise ValueError("historical diagnostic has no matched nonzero-volume hours")
    ordered = sorted(discounts)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)
    )
    return {
        "role": "non_executable_persistence_diagnostic_only",
        "matched_nonzero_volume_bar_count": len(discounts),
        "first_open_time_ms": min(timestamps),
        "last_open_time_ms": max(timestamps),
        "mean_reference_discount_bips": _decimal_text(
            sum(discounts, Decimal(0)) / Decimal(len(discounts))
        ),
        "median_reference_discount_bips": _decimal_text(median),
        "minimum_reference_discount_bips": _decimal_text(min(discounts)),
        "maximum_reference_discount_bips": _decimal_text(max(discounts)),
        "above_20_bips_count": sum(value > 20 for value in discounts),
        "below_zero_count": sum(value < 0 for value in discounts),
        "total_spot_quote_volume_usdt": _decimal_text(quote_volume),
        "limitation": "aligned four-hour closes are not simultaneous executable books or stock-sale quotes",
    }


def _funding_diagnostic(
    raw_history: object,
    *,
    spot_ask: Decimal,
    future_bid: Decimal,
) -> dict[str, object]:
    history = [
        _mapping(value, name="funding row")
        for value in _list(raw_history, name="funding history")
    ]
    if not history or len(history) >= 1000:
        raise ValueError("funding history is empty or reached its page limit")
    monthly: dict[str, list[Decimal]] = defaultdict(list)
    rates: list[Decimal] = []
    for row in history:
        rate_bips = Decimal(str(row["fundingRate"])) * Decimal(10_000)
        month = datetime.fromtimestamp(
            int(row["fundingTime"]) / 1000, tz=timezone.utc
        ).strftime("%Y-%m")
        monthly[month].append(rate_bips)
        rates.append(rate_bips)
    ordered_months = sorted(monthly)
    complete_inner = ordered_months[1:-1]
    complete_rows = [
        {
            "month": month,
            "settlement_count": len(monthly[month]),
            "short_funding_received_bips": _decimal_text(
                sum(monthly[month], Decimal(0))
            ),
            "clears_labeled_round_trip_cost_sensitivity": (
                sum(monthly[month], Decimal(0)) > ROUND_TRIP_TAKER_COST_SENSITIVITY_BIPS
            ),
        }
        for month in complete_inner
    ]
    return {
        "settlement_count": len(rates),
        "first_funding_time_ms": int(history[0]["fundingTime"]),
        "last_funding_time_ms": int(history[-1]["fundingTime"]),
        "total_short_funding_received_bips": _decimal_text(sum(rates, Decimal(0))),
        "negative_settlement_count": sum(rate < 0 for rate in rates),
        "worst_settlement_bips": _decimal_text(min(rates)),
        "best_settlement_bips": _decimal_text(max(rates)),
        "current_long_spot_short_future_entry_basis_bips": _decimal_text(
            Decimal(10_000) * (future_bid / spot_ask - Decimal(1))
        ),
        "labeled_round_trip_taker_cost_sensitivity_bips": _decimal_text(
            ROUND_TRIP_TAKER_COST_SENSITIVITY_BIPS
        ),
        "complete_inner_months": complete_rows,
        "all_complete_inner_months_clear_labeled_cost_sensitivity": (
            len(complete_rows) >= 2
            and all(
                row["clears_labeled_round_trip_cost_sensitivity"] is True
                for row in complete_rows
            )
        ),
    }


def run(*, raw_dir: Path) -> dict[str, object]:
    client = _Client(raw_dir)
    started_ms = time.time_ns() // 1_000_000
    exchange_raw, exchange_source = client.get(
        f"{SPOT_BASE_URL}/api/v3/exchangeInfo", name="spot-exchange-info"
    )
    books_raw, books_source = client.get(
        f"{SPOT_BASE_URL}/api/v3/ticker/bookTicker", name="spot-book-tickers"
    )
    futures_exchange_raw, futures_exchange_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/exchangeInfo", name="futures-exchange-info"
    )
    futures_books_raw, futures_books_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/ticker/bookTicker",
        name="futures-book-tickers",
    )
    bstocks_raw, bstocks_source = client.get(
        BSTOCK_LIST_URL,
        name="bstock-token-list",
        params={"type": 3},
    )
    exchange = _mapping(exchange_raw, name="exchange info")
    trading_symbols = {
        str(_mapping(value, name="exchange symbol")["symbol"])
        for value in _list(exchange.get("symbols"), name="exchange symbols")
        if _mapping(value, name="exchange symbol").get("status") == "TRADING"
    }
    books = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="book ticker")
            for value in _list(books_raw, name="book tickers")
        )
    }
    futures_exchange = _mapping(futures_exchange_raw, name="futures exchange info")
    trading_tradifi = {
        str(row["symbol"])
        for row in (
            _mapping(value, name="futures symbol")
            for value in _list(futures_exchange.get("symbols"), name="futures symbols")
        )
        if row.get("status") == "TRADING"
        and row.get("contractType") == "TRADIFI_PERPETUAL"
    }
    futures_books = {
        str(row["symbol"]): row
        for row in (
            _mapping(value, name="futures book ticker")
            for value in _list(futures_books_raw, name="futures book tickers")
        )
    }
    bstock_payload = (
        _mapping(bstocks_raw, name="bStock envelope").get("data")
        if isinstance(bstocks_raw, Mapping)
        else bstocks_raw
    )
    bstocks = [
        _mapping(value, name="bStock")
        for value in _list(bstock_payload, name="bStocks")
        if _mapping(value, name="bStock").get("type") == 3
    ]
    screened: list[dict[str, object]] = []
    reference_sources: list[dict[str, object]] = []
    for token in sorted(bstocks, key=lambda value: str(value.get("cs"))):
        symbol = str(token.get("cs") or "")
        if symbol not in trading_symbols or symbol not in books:
            continue
        reference_raw, reference_source = client.get(
            f"{SPOT_BASE_URL}/api/v3/referencePrice",
            name=f"reference-{symbol.lower()}",
            params={"symbol": symbol},
        )
        reference = _mapping(reference_raw, name=f"{symbol} reference price")
        if reference.get("symbol") != symbol:
            raise ValueError(f"{symbol} reference identity differs")
        price = Decimal(str(reference["referencePrice"]))
        book = books[symbol]
        bid = Decimal(str(book["bidPrice"]))
        ask = Decimal(str(book["askPrice"]))
        if min(price, bid, ask) <= 0:
            raise ValueError(f"{symbol} contains a nonpositive price")
        screened.append(
            {
                "symbol": symbol,
                "ticker": str(token["ticker"]),
                "contract_address": str(token["contractAddress"]).lower(),
                "chain_id": str(token["chainId"]),
                "multiplier": str(token["multiplier"]),
                "reference_price": _decimal_text(price),
                "reference_timestamp_ms": int(reference["timestamp"]),
                "best_bid": _decimal_text(bid),
                "best_ask": _decimal_text(ask),
                "ask_reference_discount_bips": _decimal_text(
                    Decimal(10_000) * (price / ask - Decimal(1))
                ),
                "bid_reference_premium_bips": _decimal_text(
                    Decimal(10_000) * (bid / price - Decimal(1))
                ),
            }
        )
        reference_sources.append(reference_source)
    if not screened:
        raise ValueError("no trading bStocks with reference prices were screened")
    selected = max(
        screened, key=lambda row: Decimal(str(row["ask_reference_discount_bips"]))
    )
    selected_symbol = str(selected["symbol"])
    depth_started_ms = time.time_ns() // 1_000_000
    depth_raw, depth_source = client.get(
        f"{SPOT_BASE_URL}/api/v3/depth",
        name=f"selected-depth-{selected_symbol.lower()}",
        params={"symbol": selected_symbol, "limit": 1000},
    )
    depth_finished_ms = time.time_ns() // 1_000_000
    reference_after_raw, reference_after_source = client.get(
        f"{SPOT_BASE_URL}/api/v3/referencePrice",
        name=f"selected-reference-after-{selected_symbol.lower()}",
        params={"symbol": selected_symbol},
    )
    calculation_raw, calculation_source = client.get(
        f"{SPOT_BASE_URL}/api/v3/referencePrice/calculation",
        name=f"selected-reference-calculation-{selected_symbol.lower()}",
        params={"symbol": selected_symbol},
    )
    reference_after = _mapping(reference_after_raw, name="selected reference after")
    calculation = _mapping(calculation_raw, name="selected reference calculation")
    if (
        reference_after.get("symbol") != selected_symbol
        or calculation.get("symbol") != selected_symbol
    ):
        raise ValueError("selected reference identity differs")
    conservative_reference = min(
        Decimal(str(selected["reference_price"])),
        Decimal(str(reference_after["referencePrice"])),
    )
    depth = _mapping(depth_raw, name="selected depth")
    size_results: list[dict[str, object]] = []
    total_cost_bips = SPOT_COST_SENSITIVITY_BIPS + STOCK_EXIT_COST_SENSITIVITY_BIPS
    for target in SIZES_USDT:
        fill = _fill_asks(depth.get("asks"), target_usdt=target)
        shares = Decimal(str(fill["shares"]))
        cost = Decimal(str(fill["spot_cost_usdt"]))
        gross = shares * conservative_reference - cost
        sensitivity_cost = cost * total_cost_bips / Decimal(10_000)
        size_results.append(
            {
                **fill,
                "conservative_reference_value_usdt": _decimal_text(
                    shares * conservative_reference
                ),
                "gross_reference_surplus_usdt": _decimal_text(gross),
                "gross_reference_surplus_bips": _decimal_text(
                    Decimal(10_000) * gross / cost
                ),
                "labeled_20_bips_cost_sensitivity_usdt": _decimal_text(
                    sensitivity_cost
                ),
                "after_labeled_cost_sensitivity_usdt": _decimal_text(
                    gross - sensitivity_cost
                ),
                "after_labeled_cost_sensitivity_positive": (gross > sensitivity_cost),
            }
        )
    ticker = str(selected["ticker"])
    spot_klines_raw, spot_klines_source = client.get(
        f"{SPOT_BASE_URL}/api/v3/klines",
        name=f"selected-spot-four-hour-{selected_symbol.lower()}",
        params={"symbol": selected_symbol, "interval": "4h", "limit": 1000},
    )
    index_klines_raw, index_klines_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/indexPriceKlines",
        name=f"selected-index-four-hour-{ticker.lower()}usdt",
        params={"pair": f"{ticker}USDT", "interval": "4h", "limit": 1000},
    )
    historical = _historical_diagnostic(spot_klines_raw, index_klines_raw)
    token_by_ticker = {str(token["ticker"]): token for token in bstocks}
    funding_candidates: list[dict[str, object]] = []
    funding_sources: list[dict[str, object]] = []
    for funding_ticker in FUNDING_CARRY_TICKERS:
        spot_symbol = f"{funding_ticker}BUSDT"
        future_symbol = f"{funding_ticker}USDT"
        if (
            funding_ticker not in token_by_ticker
            or spot_symbol not in books
            or future_symbol not in futures_books
            or future_symbol not in trading_tradifi
        ):
            raise ValueError(f"{funding_ticker} funding-carry identity is unavailable")
        funding_raw, funding_source = client.get(
            f"{FUTURES_BASE_URL}/fapi/v1/fundingRate",
            name=f"funding-{future_symbol.lower()}",
            params={"symbol": future_symbol, "limit": 1000},
        )
        token = token_by_ticker[funding_ticker]
        spot_ask = Decimal(str(books[spot_symbol]["askPrice"]))
        future_bid = Decimal(str(futures_books[future_symbol]["bidPrice"]))
        if min(spot_ask, future_bid) <= 0:
            raise ValueError(f"{funding_ticker} funding-carry book is invalid")
        funding_candidates.append(
            {
                "ticker": funding_ticker,
                "spot_symbol": spot_symbol,
                "future_symbol": future_symbol,
                "bstock_multiplier": str(token["multiplier"]),
                "spot_best_ask": _decimal_text(spot_ask),
                "future_best_bid": _decimal_text(future_bid),
                **_funding_diagnostic(
                    funding_raw,
                    spot_ask=spot_ask,
                    future_bid=future_bid,
                ),
            }
        )
        funding_sources.append(funding_source)
    positive_sizes = sum(
        row["after_labeled_cost_sensitivity_positive"] is True for row in size_results
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_ms": time.time_ns() // 1_000_000,
        "capture_started_at_ms": started_ms,
        "purpose": "public_market_direction_independent_bstock_reference_conversion_parity_screen",
        "authority": {
            "credentials_used": False,
            "funds_used": False,
            "orders_placed": False,
            "conversions_requested": False,
            "trading_authority": False,
        },
        "scope": {
            "research_only_outside_current_btc_eth_sol_execution_scope": True,
            "screened_trading_bstock_count": len(screened),
            "sizes_usdt": [_decimal_text(value) for value in SIZES_USDT],
        },
        "source_contract": {
            "official_bstock_terms_url": "https://academy.binance.com/en/articles/what-are-bstocks-a-guide-to-tokenized-stocks-on-binance",
            "official_reference_price_api_url": "https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#query-reference-price",
            "official_reference_calculation_api_url": "https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#query-reference-price-calculation",
            "conversion_identity": "eligible users may convert stocks and corresponding bStocks 1:1 with no conversion fee",
            "cost_sensitivity": {
                "spot_bips": _decimal_text(SPOT_COST_SENSITIVITY_BIPS),
                "stock_exit_bips": _decimal_text(STOCK_EXIT_COST_SENSITIVITY_BIPS),
                "role": "labeled sensitivity only not account fee or executable stock quote",
            },
        },
        "universe": {
            "rows": screened,
            "sources": {
                "exchange_info": exchange_source,
                "book_tickers": books_source,
                "bstock_token_list": bstocks_source,
                "reference_prices": reference_sources,
            },
        },
        "selected_candidate": {
            **selected,
            "reference_price_after": str(reference_after["referencePrice"]),
            "reference_timestamp_after_ms": int(reference_after["timestamp"]),
            "conservative_reference_price": _decimal_text(conservative_reference),
            "reference_calculation_type": str(calculation["calculationType"]),
            "external_calculation_id": calculation.get("externalCalculationId"),
            "depth_capture_window_ms": depth_finished_ms - depth_started_ms,
            "depth_last_update_id": int(depth["lastUpdateId"]),
            "size_results": size_results,
            "historical_diagnostic": historical,
            "sources": {
                "depth": depth_source,
                "reference_after": reference_after_source,
                "reference_calculation": calculation_source,
                "spot_four_hour_klines": spot_klines_source,
                "index_four_hour_klines": index_klines_source,
            },
        },
        "delta_neutral_funding_carry_diagnostic": {
            "mechanism": "long_one_bstock_share_short_one_same_underlying_tradifi_perpetual_unit",
            "market_direction_forecast_required": False,
            "candidate_count": len(funding_candidates),
            "all_candidates_clear_every_complete_inner_month": all(
                row["all_complete_inner_months_clear_labeled_cost_sensitivity"] is True
                for row in funding_candidates
            ),
            "candidates": funding_candidates,
            "sources": {
                "futures_exchange_info": futures_exchange_source,
                "futures_book_tickers": futures_books_source,
                "funding_histories": funding_sources,
            },
            "adjudication": "promising_persistent_gross_candidate_not_after_cost_or_in_scope",
        },
        "verdict": {
            "status": "gross_positive_public_reference_parity_candidate_account_and_stock_exit_quote_gated",
            "gross_positive_size_count_after_labeled_cost_sensitivity": positive_sizes,
            "accepted_edge": False,
            "deployment_ready": False,
            "trading_authority": False,
        },
        "blocking_evidence": [
            "same-account bStock and Binance Stocks eligibility for the selected symbol",
            "synchronous executable Binance Stocks sale quote rather than an external reference price",
            "exact account spot and stock execution costs taxes and rounding",
            "conversion availability limit cadence completion and balance reconciliation",
            "prospective trigger frequency fill persistence and capacity",
            "explicit scope expansion beyond BTC ETH and SOL before any execution work",
        ],
        "prohibited_shortcuts": [
            "reference price as an executable stock sale quote",
            "public 1:1 terms as same-account conversion eligibility",
            "aligned four-hour closes as fills",
            "a positive labeled fee sensitivity as after-cost profit",
            "campaign rewards or self-trading as persistent edge",
        ],
        "implementation": {
            "path": "tools/screen_binance_bstock_reference_parity.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(raw_dir=args.raw_dir)
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["verdict"], indent=2))
    print(f"selected={result['selected_candidate']['symbol']}")
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
