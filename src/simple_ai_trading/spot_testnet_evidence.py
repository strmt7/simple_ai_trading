"""Exact, quantity-based accounting for isolated Spot-testnet execution probes."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any


def decimal(value: Any) -> Decimal:
    """Reject missing and nonfinite exchange quantities before accounting."""
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("invalid nonnegative exchange quantity")
    return result


def grid(value: Decimal, step: Decimal, *, up: bool = False) -> Decimal:
    """Round by multiples rather than decimal places, including non-power steps."""
    if not value.is_finite() or value < 0 or not step.is_finite() or step <= 0:
        raise ValueError("invalid grid input")
    return (value / step).to_integral_value(
        rounding=ROUND_UP if up else ROUND_DOWN
    ) * step


@dataclass(frozen=True)
class Rules:
    symbol: str
    base: str
    step: Decimal
    tick: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_notional: Decimal
    max_notional: Decimal

    @classmethod
    def parse(cls, item: dict[str, Any]) -> Rules:
        """Require the exact scoped trading symbol and limit-order filters."""
        symbol = item["symbol"]
        if (
            symbol not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
            or item["status"] != "TRADING"
        ):
            raise ValueError("unsupported or inactive symbol")
        if item["baseAsset"] != symbol[:-4] or item["quoteAsset"] != "USDT":
            raise ValueError("asset identity differs")
        if not {"LIMIT", "LIMIT_MAKER"}.issubset(item["orderTypes"]):
            raise ValueError("required order type unavailable")
        if "EXPIRE_TAKER" not in item.get("allowedSelfTradePreventionModes", []):
            raise ValueError("self-trade prevention unavailable")
        filters = {x["filterType"]: x for x in item["filters"]}
        lot, price = filters["LOT_SIZE"], filters["PRICE_FILTER"]
        notional = filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))
        rule = cls(
            symbol,
            item["baseAsset"],
            decimal(lot["stepSize"]),
            decimal(price["tickSize"]),
            decimal(lot["minQty"]),
            decimal(lot["maxQty"]),
            decimal(notional["minNotional"]),
            decimal(notional.get("maxNotional", 0)),
        )
        if (
            rule.step <= 0
            or rule.tick <= 0
            or rule.min_qty <= 0
            or rule.max_qty < rule.min_qty
        ):
            raise ValueError("invalid symbol grid")
        return rule

    def validate(self, qty: Decimal, price: Decimal) -> None:
        """Enforce published limit-order grid and notional constraints locally."""
        if qty != grid(qty, self.step) or price != grid(price, self.tick) or price <= 0:
            raise ValueError("off-grid order")
        if not self.min_qty <= qty <= self.max_qty or qty * price < self.min_notional:
            raise ValueError("order below size/notional floor")
        if self.max_notional and qty * price > self.max_notional:
            raise ValueError("order above notional ceiling")


def owned_trade_cash(
    order: dict[str, Any], trades: list[dict[str, Any]], base: str
) -> dict[str, Any]:
    """Reconcile exact owned fills, fee assets and net base; never assume full fill."""
    if order["status"] not in {
        "FILLED",
        "CANCELED",
        "EXPIRED",
        "EXPIRED_IN_MATCH",
        "REJECTED",
    }:
        raise ValueError("nonterminal order")
    if order["side"] not in {"BUY", "SELL"}:
        raise ValueError("invalid side")
    seen: set[int] = set()
    qty = quote = base_fee = quote_fee = Decimal(0)
    third_fees: dict[str, str] = {}
    for row in trades:
        if row["symbol"] != order["symbol"] or row["orderId"] != order["orderId"]:
            raise ValueError("foreign trade")
        if type(row["id"]) is not int or row["id"] in seen:
            raise ValueError("ambiguous trade identity")
        seen.add(row["id"])
        if type(row["isBuyer"]) is not bool or row["isBuyer"] != (
            order["side"] == "BUY"
        ):
            raise ValueError("trade side differs")
        qty += decimal(row["qty"])
        quote += decimal(row["quoteQty"])
        fee = decimal(row["commission"])
        asset = row["commissionAsset"]
        if asset == base:
            base_fee += fee
        elif asset == "USDT":
            quote_fee += fee
        elif fee:
            third_fees[asset] = str(decimal(third_fees.get(asset, 0)) + fee)
    if qty != decimal(order["executedQty"]) or quote != decimal(
        order["cummulativeQuoteQty"]
    ):
        raise ValueError("trade ledger is incomplete or inconsistent")
    sign = Decimal(1) if order["side"] == "BUY" else Decimal(-1)
    return {
        "base_delta": str(sign * qty - base_fee),
        "quote_delta": str(-sign * quote - quote_fee),
        "third_asset_fees": third_fees,
        "trade_count": len(trades),
        "partial_fill": 0 < qty < decimal(order["origQty"]),
    }
