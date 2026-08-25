"""Screen current Binance BTC/ETH/SOL stablecoin spot conversion triangles."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

import simple_ai_trading.structural_parity as structural_parity_module
from simple_ai_trading.storage import write_bytes_atomic
from simple_ai_trading.structural_parity import (
    SpotPairQuote,
    SpotTrianglePath,
    screen_spot_triangles,
)


SCHEMA_VERSION = "binance-spot-structural-parity-screen-v1"
BINANCE_SPOT_BASE_URL = "https://api.binance.com"
ALLOWED_ASSETS = frozenset({"BTC", "ETH", "SOL", "USDC", "USDT"})
START_ASSETS = ("USDC", "USDT")
FEE_SCENARIOS_BIPS = (Decimal("0"), Decimal("7.5"), Decimal("10"))


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


def _get(session: requests.Session, url: str) -> tuple[object, str, int, int]:
    before_ms = time.time_ns() // 1_000_000
    response = session.get(url, timeout=30)
    after_ms = time.time_ns() // 1_000_000
    response.raise_for_status()
    payload = response.content
    try:
        decoded = response.json()
    except requests.JSONDecodeError as exc:
        raise ValueError(f"source {url} did not return JSON") from exc
    return decoded, _sha256(payload), before_ms, after_ms


def _path_payload(path: SpotTrianglePath | None) -> dict[str, object] | None:
    if path is None:
        return None
    return {
        "assets": list(path.assets),
        "symbols": list(path.symbols),
        "gross_multiplier": _decimal_text(path.gross_multiplier),
        "gross_net_bips": _decimal_text(path.gross_net_bips),
        "after_fee_multiplier": _decimal_text(path.after_fee_multiplier),
        "after_fee_net_bips": _decimal_text(path.after_fee_net_bips),
        "optimistic_top_book_capacity_start": _decimal_text(
            path.optimistic_capacity_start
        ),
        "break_even_fee_bips_per_leg": _decimal_text(path.break_even_fee_bips_per_leg),
    }


def run() -> dict[str, object]:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "simple-ai-trading-structural-research/1.0",
        }
    )
    exchange_url = f"{BINANCE_SPOT_BASE_URL}/api/v3/exchangeInfo"
    exchange_payload, exchange_hash, exchange_before, exchange_after = _get(
        session, exchange_url
    )
    exchange = _mapping(exchange_payload, name="Binance exchangeInfo")
    raw_symbols = _list(exchange.get("symbols"), name="Binance exchangeInfo symbols")
    symbols: dict[str, dict[str, object]] = {}
    for raw_symbol in raw_symbols:
        symbol = _mapping(raw_symbol, name="Binance exchangeInfo symbol")
        name = str(symbol.get("symbol") or "").upper()
        base = str(symbol.get("baseAsset") or "").upper()
        quote = str(symbol.get("quoteAsset") or "").upper()
        if (
            name
            and base in ALLOWED_ASSETS
            and quote in ALLOWED_ASSETS
            and symbol.get("status") == "TRADING"
            and symbol.get("isSpotTradingAllowed") is True
        ):
            if name in symbols:
                raise ValueError("Binance exchangeInfo symbol is duplicated")
            symbols[name] = symbol
    if len(symbols) < 3:
        raise ValueError("fewer than three scoped Binance spot symbols are tradable")

    ticker_url = f"{BINANCE_SPOT_BASE_URL}/api/v3/ticker/bookTicker"
    ticker_payload, ticker_hash, ticker_before, ticker_after = _get(session, ticker_url)
    raw_tickers = _list(ticker_payload, name="Binance bookTicker response")
    tickers: dict[str, dict[str, object]] = {}
    for raw_ticker in raw_tickers:
        ticker = _mapping(raw_ticker, name="Binance bookTicker")
        name = str(ticker.get("symbol") or "").upper()
        if name in tickers:
            raise ValueError("Binance bookTicker symbol is duplicated")
        tickers[name] = ticker
    if not set(symbols).issubset(tickers):
        raise ValueError("Binance bookTicker omitted a scoped tradable symbol")

    quotes: list[SpotPairQuote] = []
    symbol_evidence: list[dict[str, object]] = []
    for name in sorted(symbols):
        symbol = symbols[name]
        ticker = tickers[name]
        quote = SpotPairQuote(
            symbol=name,
            base_asset=str(symbol["baseAsset"]),
            quote_asset=str(symbol["quoteAsset"]),
            bid_price=Decimal(str(ticker.get("bidPrice"))),
            bid_quantity=Decimal(str(ticker.get("bidQty"))),
            ask_price=Decimal(str(ticker.get("askPrice"))),
            ask_quantity=Decimal(str(ticker.get("askQty"))),
        ).validated()
        filters = _list(symbol.get("filters"), name=f"Binance {name} filters")
        filter_types = [
            str(_mapping(item, name=f"Binance {name} filter").get("filterType"))
            for item in filters
        ]
        if "LOT_SIZE" not in filter_types:
            raise ValueError(f"Binance {name} LOT_SIZE filter is absent")
        quotes.append(quote)
        symbol_evidence.append(
            {
                "symbol": name,
                "base_asset": quote.base_asset,
                "quote_asset": quote.quote_asset,
                "bid_price": _decimal_text(quote.bid_price),
                "bid_quantity": _decimal_text(quote.bid_quantity),
                "ask_price": _decimal_text(quote.ask_price),
                "ask_quantity": _decimal_text(quote.ask_quantity),
                "filter_types": filter_types,
            }
        )

    scenarios: list[dict[str, object]] = []
    for fee_bips in FEE_SCENARIOS_BIPS:
        screen = screen_spot_triangles(
            quotes,
            start_assets=START_ASSETS,
            taker_fee_bips_per_leg=fee_bips,
        )
        scenarios.append(
            {
                "taker_fee_bips_per_leg": _decimal_text(fee_bips),
                "evaluated_path_count": screen.evaluated_path_count,
                "gross_positive_path_count": screen.gross_positive_path_count,
                "after_fee_positive_path_count": screen.after_fee_positive_path_count,
                "best_gross_path": _path_payload(screen.best_gross_path),
                "best_after_fee_path": _path_payload(screen.best_after_fee_path),
            }
        )
    zero_fee = scenarios[0]
    reference_fee = scenarios[-1]
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "target_free_market_direction_independent_spot_triangle_screen",
        "scope": {
            "assets": sorted(ALLOWED_ASSETS),
            "start_assets": list(START_ASSETS),
            "fee_scenarios_bips_per_leg": [
                _decimal_text(value) for value in FEE_SCENARIOS_BIPS
            ],
        },
        "source_contract": {
            "exchange_info_url": exchange_url,
            "exchange_info_payload_sha256": exchange_hash,
            "exchange_info_requested_before_ms": exchange_before,
            "exchange_info_received_after_ms": exchange_after,
            "exchange_info_server_time_ms": exchange.get("serverTime"),
            "book_ticker_url": ticker_url,
            "book_ticker_payload_sha256": ticker_hash,
            "book_ticker_requested_before_ms": ticker_before,
            "book_ticker_received_after_ms": ticker_after,
            "book_ticker_request_elapsed_ms": ticker_after - ticker_before,
            "book_ticker_timestamp_limitation": "the all-symbol REST response has no per-symbol event timestamp",
            "implementation": {
                "tool_path": Path(__file__).name,
                "tool_sha256": _sha256(Path(__file__).read_bytes()),
                "module_path": Path(structural_parity_module.__file__).name,
                "module_sha256": _sha256(
                    Path(structural_parity_module.__file__).read_bytes()
                ),
            },
        },
        "symbols": symbol_evidence,
        "scenarios": scenarios,
        "verdict": {
            "status": (
                "diagnostic_positive_requires_exact_account_fee_and_atomic_execution"
                if int(reference_fee["after_fee_positive_path_count"]) > 0
                else "rejected_current_snapshot_at_10_bps_reference_fee"
            ),
            "gross_positive_path_count": zero_fee["gross_positive_path_count"],
            "reference_fee_positive_path_count": reference_fee[
                "after_fee_positive_path_count"
            ],
            "edge_claim": False,
            "trading_authority": False,
        },
        "limitations": [
            "The exact taker commission is account, pair, tier, and discount dependent and requires authenticated account evidence.",
            "The public all-symbol BBO response is not an atomic multi-symbol execution snapshot.",
            "Positive paths would still require step-size, notional, inventory, latency, partial-fill, and order-acknowledgement validation.",
            "This tool reads public market data and never places orders.",
        ],
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["verdict"], indent=2))
    zero_fee = result["scenarios"][0]
    print(json.dumps(zero_fee["best_gross_path"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
