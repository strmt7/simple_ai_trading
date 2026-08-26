"""Screen BFUSD spot depth against its 1:1 Binance redemption identity."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import requests

from simple_ai_trading.storage import write_bytes_atomic


SCHEMA_VERSION = "binance-bfusd-spot-redemption-parity-screen-v1"
SYMBOL = "BFUSDUSDT"
ONE = Decimal("1")
BIPS = Decimal("10000")


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


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


class _Client:
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "simple-ai-trading-public-edge-research/1.0"}
        )

    def get(
        self,
        url: str,
        *,
        name: str,
        params: Mapping[str, object] | None = None,
        expect_json: bool = True,
    ) -> tuple[object | bytes, dict[str, object]]:
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
        if not expect_json:
            return payload, receipt
        try:
            return response.json(), receipt
        except requests.JSONDecodeError as exc:
            raise ValueError(f"{name} did not return JSON") from exc


def _levels(
    value: object, *, name: str, descending: bool
) -> list[tuple[Decimal, Decimal]]:
    levels: list[tuple[Decimal, Decimal]] = []
    for raw in _list(value, name=name):
        row = _list(raw, name=f"{name} level")
        if len(row) < 2:
            raise ValueError(f"{name} level is incomplete")
        price = Decimal(str(row[0]))
        quantity = Decimal(str(row[1]))
        if price <= 0 or quantity <= 0:
            raise ValueError(f"{name} level is nonpositive")
        levels.append((price, quantity))
    expected = sorted(levels, key=lambda level: level[0], reverse=descending)
    if levels != expected or len({price for price, _ in levels}) != len(levels):
        raise ValueError(f"{name} is not uniquely price ordered")
    return levels


def _fill(
    levels: Sequence[tuple[Decimal, Decimal]], quantity: Decimal
) -> tuple[Decimal, Decimal] | None:
    remaining = quantity
    quote = Decimal(0)
    for price, available in levels:
        taken = min(remaining, available)
        quote += taken * price
        remaining -= taken
        if remaining == 0:
            return quote, quote / quantity
    return None


def _screen_quantity(
    *,
    quantity: Decimal,
    asks: Sequence[tuple[Decimal, Decimal]],
    bids: Sequence[tuple[Decimal, Decimal]],
    spot_fee_bips: Decimal,
    subscription_fee_bips: Decimal,
    redemption_fee_bips: Decimal,
) -> dict[str, object]:
    buy = _fill(asks, quantity)
    sell = _fill(bids, quantity)
    if buy is None or sell is None:
        return {"quantity_bfusd": _decimal_text(quantity), "executable": False}
    buy_quote, buy_vwap = buy
    sell_quote, sell_vwap = sell
    spot_fee = spot_fee_bips / BIPS
    subscription_fee = subscription_fee_bips / BIPS
    redemption_fee = redemption_fee_bips / BIPS
    buy_then_redeem_net = quantity * (ONE - redemption_fee) - buy_quote * (
        ONE + spot_fee
    )
    subscribe_then_sell_net = sell_quote * (ONE - spot_fee) - quantity * (
        ONE + subscription_fee
    )
    round_trip_net = sell_quote * (ONE - spot_fee) - buy_quote * (ONE + spot_fee)
    return {
        "quantity_bfusd": _decimal_text(quantity),
        "executable": True,
        "buy_vwap_usdt": _decimal_text(buy_vwap),
        "sell_vwap_usdt": _decimal_text(sell_vwap),
        "buy_spot_then_redeem_net_usdt": _decimal_text(buy_then_redeem_net),
        "subscribe_then_sell_spot_net_usdt": _decimal_text(subscribe_then_sell_net),
        "spot_round_trip_net_before_yield_usdt": _decimal_text(round_trip_net),
        "buy_spot_then_redeem_positive": buy_then_redeem_net > 0,
        "subscribe_then_sell_spot_positive": subscribe_then_sell_net > 0,
    }


def run(
    *,
    raw_dir: Path,
    quantities: Sequence[Decimal],
    spot_fee_bips: Decimal,
    subscription_fee_bips: Decimal,
    redemption_fee_bips: Decimal,
) -> dict[str, object]:
    if (
        not quantities
        or any(quantity <= 0 for quantity in quantities)
        or any(
            fee < 0
            for fee in (spot_fee_bips, subscription_fee_bips, redemption_fee_bips)
        )
    ):
        raise ValueError("quantities and fee sensitivities must be nonnegative")
    client = _Client(raw_dir)
    started_ms = time.time_ns() // 1_000_000
    exchange_raw, exchange_source = client.get(
        "https://api.binance.com/api/v3/exchangeInfo",
        name="exchange-info",
        params={"symbol": SYMBOL},
    )
    exchange = _mapping(exchange_raw, name="exchange info")
    symbols = _list(exchange.get("symbols"), name="exchange symbols")
    if len(symbols) != 1:
        raise ValueError("BFUSDUSDT did not resolve exactly once")
    symbol = _mapping(symbols[0], name="BFUSDUSDT symbol")
    if (
        symbol.get("symbol") != SYMBOL
        or symbol.get("status") != "TRADING"
        or symbol.get("baseAsset") != "BFUSD"
        or symbol.get("quoteAsset") != "USDT"
        or symbol.get("isSpotTradingAllowed") is not True
    ):
        raise ValueError("BFUSDUSDT identity or trading status differs")

    depth_raw, depth_source = client.get(
        "https://api.binance.com/api/v3/depth",
        name="depth",
        params={"symbol": SYMBOL, "limit": 1000},
    )
    depth = _mapping(depth_raw, name="depth")
    bids = _levels(depth.get("bids"), name="bids", descending=True)
    asks = _levels(depth.get("asks"), name="asks", descending=False)

    ticker_raw, ticker_source = client.get(
        "https://api.binance.com/api/v3/ticker/24hr",
        name="ticker-24hr",
        params={"symbol": SYMBOL},
    )
    ticker = _mapping(ticker_raw, name="24-hour ticker")
    if ticker.get("symbol") != SYMBOL:
        raise ValueError("24-hour ticker symbol differs")

    klines_raw, klines_source = client.get(
        "https://api.binance.com/api/v3/klines",
        name="daily-klines",
        params={"symbol": SYMBOL, "interval": "1d", "startTime": 0, "limit": 1000},
    )
    klines = [
        _list(row, name="daily kline") for row in _list(klines_raw, name="klines")
    ]
    if not klines or len(klines) >= 1000 or any(len(row) < 8 for row in klines):
        raise ValueError("daily history is empty, incomplete, or requires pagination")

    screens = [
        _screen_quantity(
            quantity=quantity,
            asks=asks,
            bids=bids,
            spot_fee_bips=spot_fee_bips,
            subscription_fee_bips=subscription_fee_bips,
            redemption_fee_bips=redemption_fee_bips,
        )
        for quantity in quantities
    ]
    direct_positive = any(
        screen.get("buy_spot_then_redeem_positive") is True
        or screen.get("subscribe_then_sell_spot_positive") is True
        for screen in screens
    )
    lows = [Decimal(str(row[3])) for row in klines]
    highs = [Decimal(str(row[2])) for row in klines]
    quote_volumes = [Decimal(str(row[7])) for row in klines]
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_ms": time.time_ns() // 1_000_000,
        "capture_started_at_ms": started_ms,
        "purpose": "public_direction_neutral_bfusd_spot_subscription_redemption_parity_screen",
        "authority": {
            "credentials_used": False,
            "orders_placed": False,
            "funds_used": False,
            "trading_authority": False,
        },
        "mechanism": {
            "buy_spot_then_redeem": "buy BFUSD below its 1:1 redemption value after exact spot and redemption costs",
            "subscribe_then_sell_spot": "subscribe BFUSD at 1:1 then sell above that cost after exact subscription and spot costs",
            "holding_route": "buy BFUSD on spot, earn account-eligible daily rewards, then sell on spot; exact reward and account fees remain signed evidence",
            "market_direction_forecast_required": False,
        },
        "sensitivity": {
            "spot_fee_bips_per_trade": _decimal_text(spot_fee_bips),
            "subscription_fee_bips": _decimal_text(subscription_fee_bips),
            "redemption_fee_bips": _decimal_text(redemption_fee_bips),
            "classification": "official_guide_typical_subscription_and_redemption_fee_plus_non_authoritative_spot_fee_sensitivity_not_account_cost_evidence",
        },
        "current_book": {
            "last_update_id": depth.get("lastUpdateId"),
            "best_bid": _decimal_text(bids[0][0]),
            "best_ask": _decimal_text(asks[0][0]),
            "spread_bips": _decimal_text((asks[0][0] / bids[0][0] - ONE) * BIPS),
            "screens": screens,
        },
        "public_history": {
            "daily_row_count": len(klines),
            "first_open_time_ms": int(klines[0][0]),
            "last_open_time_ms": int(klines[-1][0]),
            "minimum_daily_low": _decimal_text(min(lows)),
            "maximum_daily_high": _decimal_text(max(highs)),
            "total_quote_volume_usdt": _decimal_text(sum(quote_volumes, Decimal(0))),
            "interpretation": "trade-price history justifies a prospective executable-depth monitor but never proves historical fillable parity",
        },
        "ticker_24h": {
            key: ticker.get(key)
            for key in (
                "openTime",
                "closeTime",
                "bidPrice",
                "askPrice",
                "lowPrice",
                "highPrice",
                "volume",
                "quoteVolume",
            )
        },
        "sources": {
            "exchange_info": exchange_source,
            "depth": depth_source,
            "ticker_24h": ticker_source,
            "daily_klines": klines_source,
            "official_bfusd_guide_url": "https://academy.binance.com/en/articles/what-is-bfusd",
            "account_evidence_gate_path": "docs/model-research/action-value/binance-stable-yield-allocation-evidence-gate-v1.json",
            "account_evidence_gate_result_sha256": "3096867474c4b5a0b3f893645bac68081ceb3783ad14393261e6d88793b64a8a",
        },
        "verdict": {
            "status": (
                "public_direct_parity_candidate_requires_exact_account_cost_and_execution_evidence"
                if direct_positive
                else "no_current_positive_direct_subscription_or_redemption_parity_at_reference_costs"
            ),
            "current_direct_positive_path": direct_positive,
            "holding_yield_edge_accepted": False,
            "next_trigger": "exact_signed_account_reward_rate_quota_subscription_redemption_and_BFUSDUSDT_commission_evidence_then_public_depth_monitoring",
            "profitability_claim": False,
            "trading_authority": False,
        },
        "limitations": [
            "The 0.1% subscription and redemption fees are typical guide values, not exact account quota fields.",
            "The spot fee sensitivity is not the account's signed BFUSDUSDT commission.",
            "Daily kline extremes do not prove historical displayed depth or fills.",
            "The holding route requires signed account reward, eligibility, quota, and alternative-yield evidence.",
            "BFUSD has venue, redemption-delay, reward-variability, regional, custody, and tax risks.",
        ],
        "implementation": {
            "path": "tools/screen_binance_bfusd_redemption_parity.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantity", action="append", default=["100", "1000", "10000"])
    parser.add_argument("--spot-fee-bips", default="10")
    parser.add_argument("--subscription-fee-bips", default="10")
    parser.add_argument("--redemption-fee-bips", default="10")
    args = parser.parse_args()
    result = run(
        raw_dir=args.raw_dir,
        quantities=[Decimal(value) for value in args.quantity],
        spot_fee_bips=Decimal(args.spot_fee_bips),
        subscription_fee_bips=Decimal(args.subscription_fee_bips),
        redemption_fee_bips=Decimal(args.redemption_fee_bips),
    )
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["verdict"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
