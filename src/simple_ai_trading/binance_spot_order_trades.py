"""Complete an exact Spot order observation with one bound native trade page."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import TYPE_CHECKING

from .binance_acknowledgements import acknowledged_fill
from .binance_execution_scope import BinanceExecutionScope, parse_execution_id
from .binance_open_intents import OpenIntentError
from .binance_order_responses import MarketOrderBinding, SpotQuoteOrderBinding
from .binance_terminal_fills import _decimal

if TYPE_CHECKING:
    from .api import BinanceClient


def observe_spot_order_trades(
    client: BinanceClient,
    order: Mapping[str, object],
    *,
    binding: MarketOrderBinding | SpotQuoteOrderBinding,
    scope: BinanceExecutionScope,
) -> dict[str, object]:
    """Attach one exact-order page; never submit, persist or release intent.

    The caller owns the existing admission/recovery guard and durable obligation.
    A page that fails cumulative cash/quantity reconciliation remains incomplete,
    including when fills advance between the status and trade observations.
    """
    if (
        not isinstance(scope, BinanceExecutionScope)
        or scope.market_type != "spot"
        or not isinstance(binding, (MarketOrderBinding, SpotQuoteOrderBinding))
        or binding.market_type != "spot"
        or not isinstance(order, Mapping)
    ):
        raise OpenIntentError("Spot trade observation scope or binding is invalid")
    try:
        binding.validate(order)
    except ValueError as exc:
        raise OpenIntentError(
            "Spot trade observation differs from its request"
        ) from exc
    if "fills" in order:
        raise OpenIntentError("Spot order query unexpectedly embeds fill evidence")
    if client.execution_scope() != scope:
        raise OpenIntentError("Spot trade observation execution scope changed")
    observed = dict(order)
    side = "BUY" if isinstance(binding, SpotQuoteOrderBinding) else binding.side
    executed = _decimal(observed.get("executedQty"))
    fills = []
    if executed > 0:
        trades = client.get_order_trades(
            binding.symbol,
            order_id=observed["orderId"],
            expected_scope=scope,
        )
        if not isinstance(trades, list) or not 1 <= len(trades) <= 1_000:
            raise OpenIntentError("Spot exact-order trades are missing or unbounded")
        for trade in trades:
            if (
                not isinstance(trade, Mapping)
                or trade.get("symbol") != binding.symbol
                or parse_execution_id(trade.get("orderId")) is None
                or parse_execution_id(trade.get("orderId"))
                != parse_execution_id(observed["orderId"])
                or parse_execution_id(trade.get("id")) is None
                or trade.get("isBuyer") is not (side == "BUY")
                or not isinstance(trade.get("commissionAsset"), str)
                or re.fullmatch(r"[A-Z0-9]{1,32}", trade["commissionAsset"]) is None
            ):
                raise OpenIntentError(
                    "Spot trade identity, side or fee asset is invalid"
                )
            _decimal(trade.get("commission"))
            fills.append(
                {
                    "tradeId": trade["id"],
                    "qty": trade.get("qty"),
                    "price": trade.get("price"),
                    "quoteQty": trade.get("quoteQty"),
                    "commission": trade["commission"],
                    "commissionAsset": trade["commissionAsset"],
                }
            )
    observed["fills"] = fills
    fill = acknowledged_fill(observed, market_type="spot")
    if executed > 0 and fill is None:
        raise OpenIntentError("Spot queried execution lacks complete fill evidence")
    return observed
