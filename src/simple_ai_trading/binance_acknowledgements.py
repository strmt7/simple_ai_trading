"""Reconcile first-response fill quantities before legacy float projections."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Literal

from .binance_open_intents import OpenIntentError
from .binance_terminal_fills import _decimal, _text, execution_id


@dataclass(frozen=True)
class AcknowledgedFill:
    quantity: str
    price: str
    quote_quantity: str | None
    price_basis: Literal["cumulative_quote", "fills", "average_only"]


def acknowledged_fill(
    order: Mapping[str, object], *, market_type: str
) -> AcknowledgedFill | None:
    """Reject conflicting cumulative and fill-row evidence, never guess a fill."""
    if not isinstance(order, Mapping) or market_type not in {"spot", "futures"}:
        raise OpenIntentError("acknowledgement product or payload is invalid")
    if "orderId" in order:
        execution_id(order["orderId"])
    quote_key = "cummulativeQuoteQty" if market_type == "spot" else "cumQuote"
    foreign_quote = "cumQuote" if market_type == "spot" else "cummulativeQuoteQty"
    if foreign_quote in order or "cumBase" in order:
        raise OpenIntentError("acknowledgement cash units differ from the product")
    if "executedQty" not in order:
        if order.get("fills"):
            raise OpenIntentError("fill rows lack cumulative executed quantity")
        return None
    quantity = _decimal(order["executedQty"])
    if "origQty" in order:
        original = _decimal(order["origQty"])
        if (
            original <= 0
            or quantity > original
            or (order.get("status") == "FILLED" and quantity != original)
        ):
            raise OpenIntentError(
                "acknowledgement original and executed quantities conflict"
            )
    quote = _decimal(order[quote_key]) if quote_key in order else None
    average = _decimal(order["avgPrice"]) if "avgPrice" in order else None
    with localcontext() as context:
        # At most 1000 products of two <=60-digit decimal inputs require at
        # most 123 significant digits; cash sums must not round silently.
        context.prec = 128
        fill_quantity, fill_quote = Decimal(0), Decimal(0)
        has_fills = "fills" in order
        if has_fills:
            fills = order["fills"]
            if not isinstance(fills, list) or len(fills) > 1000:
                raise OpenIntentError("acknowledgement fill rows exceed their contract")
            seen: set[str] = set()
            for fill in fills:
                if not isinstance(fill, Mapping):
                    raise OpenIntentError("acknowledgement fill row is invalid")
                qty, price = _decimal(fill.get("qty")), _decimal(fill.get("price"))
                if qty <= 0 or price <= 0:
                    raise OpenIntentError(
                        "acknowledgement fill quantity or price is invalid"
                    )
                if "quoteQty" in fill and _decimal(fill["quoteQty"]) != qty * price:
                    raise OpenIntentError(
                        "acknowledgement fill cash does not reconcile"
                    )
                if "tradeId" in fill:
                    trade_id = execution_id(fill["tradeId"])
                    if trade_id in seen:
                        raise OpenIntentError(
                            "acknowledgement fill identity is duplicated"
                        )
                    seen.add(trade_id)
                if "commission" in fill or "commissionAsset" in fill:
                    _decimal(fill.get("commission"), signed=True)
                    asset = fill.get("commissionAsset")
                    if not isinstance(asset, str) or not re.fullmatch(
                        r"[A-Z0-9]{1,32}", asset
                    ):
                        raise OpenIntentError("acknowledgement fee asset is invalid")
                fill_quantity += qty
                fill_quote += qty * price
            if fill_quantity != quantity:
                raise OpenIntentError(
                    "acknowledgement cumulative and fill quantities conflict"
                )
            if quote is not None and fill_quote != quote:
                raise OpenIntentError(
                    "acknowledgement cumulative and fill cash conflict"
                )
        if quantity == 0:
            if quote not in (None, Decimal(0)):
                raise OpenIntentError("zero execution has nonzero acknowledgement cash")
            return None
        if quote is not None:
            if quote <= 0:
                raise OpenIntentError("executed acknowledgement has no positive cash")
            price, basis = quote / quantity, "cumulative_quote"
        elif has_fills:
            quote = fill_quote
            price, basis = quote / quantity, "fills"
        elif average is not None and average > 0:
            # An average alone supplies a price projection, not native cash or
            # fee evidence. Exact cash/fill evidence above takes precedence.
            price, basis = average, "average_only"
        else:
            # The order's limit price is not an executed-price observation.
            return None
        return AcknowledgedFill(
            _text(quantity),
            _text(price),
            None if quote is None else _text(quote),
            basis,
        )
