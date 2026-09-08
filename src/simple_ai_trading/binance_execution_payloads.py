"""Product-specific retention fields shared by opening and closing recovery."""

from __future__ import annotations

import json
from collections.abc import Mapping

from .binance_execution_scope import BinanceExecutionScope

_ORDER_FIELDS = (
    "symbol",
    "orderId",
    "clientOrderId",
    "origClientOrderId",
    "side",
    "type",
    "status",
    "origQty",
    "executedQty",
    "time",
    "updateTime",
)
_TRADE_FIELDS = (
    "symbol",
    "orderId",
    "id",
    "price",
    "qty",
    "quoteQty",
    "commission",
    "commissionAsset",
    "time",
)


def encode_execution_payloads(
    scope: BinanceExecutionScope,
    order: Mapping[str, object],
    trades: list[Mapping[str, object]],
) -> tuple[str, str]:
    """Project already-validated evidence; this allowlist is not a validator."""
    order_fields = _ORDER_FIELDS + (
        ("cummulativeQuoteQty",)
        if scope.market_type == "spot"
        else ("cumQuote", "positionSide", "reduceOnly")
    )
    trade_fields = _TRADE_FIELDS + (
        ("isBuyer", "isMaker")
        if scope.market_type == "spot"
        else ("side", "positionSide", "buyer", "maker", "realizedPnl")
    )
    return (
        json.dumps(
            {key: order[key] for key in order_fields if key in order},
            sort_keys=True,
            allow_nan=False,
        ),
        json.dumps(
            [
                {key: trade[key] for key in trade_fields if key in trade}
                for trade in trades
            ],
            sort_keys=True,
            allow_nan=False,
        ),
    )
