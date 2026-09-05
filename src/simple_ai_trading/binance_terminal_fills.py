"""Exact terminal execution evidence; not inventory reconciliation or rearm authority."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext

from .binance_execution_scope import BinanceExecutionScope, parse_execution_id
from .binance_open_intents import BinanceOpenIntentJournal, OpenIntentError
from .positions import OpenPosition


def _decimal(value: object, *, signed: bool = False) -> Decimal:
    # Venue decimal strings stay exact; accepting floats would hide prior loss.
    if not isinstance(value, str) or not re.fullmatch(
        r"-?\d{1,30}(?:\.\d{1,30})?", value, flags=re.ASCII
    ):
        raise OpenIntentError("execution decimal evidence is invalid")
    number = Decimal(value)
    if not signed and number < 0:
        raise OpenIntentError("execution quantity must not be negative")
    return number


def _text(number: Decimal) -> str:
    if not number:
        return "0"
    return (
        format(number, "f").rstrip("0").rstrip(".")
        if number.as_tuple().exponent < 0
        else str(number)
    )


def execution_id(value: object) -> str:
    """Reject coercion and truncation when binding venue execution identities."""
    parsed = parse_execution_id(value)
    if parsed is not None:
        return parsed
    raise OpenIntentError("execution identity is invalid")


def _time(value: object) -> int:
    if type(value) is not int or not 0 < value < 10**16:
        raise OpenIntentError("execution timestamp is invalid")
    return value


@dataclass(frozen=True)
class TerminalOrder:
    order_id: str
    status: str
    original_quantity: str
    executed_quantity: str
    quote_quantity: str
    created_at_ms: int
    updated_at_ms: int


@dataclass(frozen=True)
class TerminalFill:
    trade_id: str
    price: str
    quantity: str
    quote_quantity: str
    commission: str
    commission_asset: str
    time_ms: int


@dataclass(frozen=True)
class TerminalFillEvidence:
    order: TerminalOrder
    fills: tuple[TerminalFill, ...]
    executed_quantity: str
    quote_quantity: str
    commissions: tuple[tuple[str, str], ...]


def validate_terminal_order(
    requested: OpenPosition, scope: BinanceExecutionScope, order: Mapping[str, object]
) -> TerminalOrder:
    if not isinstance(scope, BinanceExecutionScope):
        raise OpenIntentError("terminal recovery requires an execution scope")
    BinanceOpenIntentJournal._request(requested, scope)
    side = "BUY" if requested.side == "LONG" else "SELL"
    if (
        not isinstance(order, Mapping)
        or requested.market_type == "spot"
        and requested.side != "LONG"
        or order.get("symbol") != requested.symbol
        or order.get("clientOrderId") != requested.open_client_order_id
        or "origClientOrderId" in order
        and order["origClientOrderId"] != requested.open_client_order_id
        or order.get("side") != side
        or order.get("type") != "MARKET"
        or order.get("status")
        not in ("FILLED", "CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "REJECTED")
    ):
        raise OpenIntentError("terminal order identity or state is unresolved")
    order_id = execution_id(order.get("orderId"))
    if (
        requested.open_exchange_order_id
        and requested.open_exchange_order_id != order_id
    ):
        raise OpenIntentError("terminal order differs from recorded exchange identity")
    if requested.market_type == "futures" and (
        order.get("positionSide") != "BOTH" or order.get("reduceOnly") is not False
    ):
        raise OpenIntentError("terminal order does not match one-way opening semantics")
    original = _decimal(order.get("origQty"))
    executed = _decimal(order.get("executedQty"))
    quote = _decimal(
        order.get(
            "cummulativeQuoteQty" if requested.market_type == "spot" else "cumQuote"
        )
    )
    if (
        original <= 0
        or original != Decimal(f"{requested.qty:.8f}")
        or executed > original
        or order["status"] == "FILLED"
        and executed != original
        or order["status"] == "REJECTED"
        and executed != 0
        or (executed == 0) != (quote == 0)
    ):
        raise OpenIntentError("terminal cumulative quantities do not reconcile")
    created, updated = _time(order.get("time")), _time(order.get("updateTime"))
    if created > updated:
        raise OpenIntentError("terminal order chronology is inconsistent")
    return TerminalOrder(
        order_id,
        str(order["status"]),
        _text(original),
        _text(executed),
        _text(quote),
        created,
        updated,
    )


def validate_terminal_fills(
    requested: OpenPosition,
    scope: BinanceExecutionScope,
    order: Mapping[str, object],
    trades: object,
) -> TerminalFillEvidence:
    """Require exact totals and native commissions, never a guessed fee conversion."""
    terminal = validate_terminal_order(requested, scope, order)
    if not isinstance(trades, list) or len(trades) > 1000:
        raise OpenIntentError(
            "terminal trade evidence exceeds its single-page contract"
        )
    fills: list[TerminalFill] = []
    seen: set[str] = set()
    commissions: dict[str, Decimal] = {}
    with localcontext() as context:
        context.prec = 100  # Bounds above plus <=1000 rows keep every sum exact.
        quantity, quote_quantity = Decimal(0), Decimal(0)
        for trade in trades:
            if not isinstance(trade, Mapping):
                raise OpenIntentError("terminal trade row is invalid")
            trade_id = execution_id(trade.get("id"))
            buyer_key, maker_key = (
                ("isBuyer", "isMaker")
                if scope.market_type == "spot"
                else ("buyer", "maker")
            )
            if (
                trade_id in seen
                or execution_id(trade.get("orderId")) != terminal.order_id
                or trade.get("symbol") != requested.symbol
                or trade.get(buyer_key) is not (requested.side == "LONG")
                or trade.get(maker_key) is not False
            ):
                raise OpenIntentError("terminal trade identity or side is inconsistent")
            if scope.market_type == "futures" and (
                trade.get("side") != order["side"]
                or trade.get("positionSide") != "BOTH"
                or _decimal(trade.get("realizedPnl"), signed=True) != 0
            ):
                raise OpenIntentError(
                    "trade cannot establish an opening-only execution"
                )
            price, qty, quote = (
                _decimal(trade.get(key)) for key in ("price", "qty", "quoteQty")
            )
            commission = _decimal(trade.get("commission"), signed=True)
            asset = trade.get("commissionAsset")
            timestamp = _time(trade.get("time"))
            if (
                min(price, qty, quote) <= 0
                or not isinstance(asset, str)
                or not re.fullmatch(r"[A-Z0-9]{1,32}", asset)
                or not terminal.created_at_ms <= timestamp <= terminal.updated_at_ms
            ):
                raise OpenIntentError("trade price, fee asset or chronology is invalid")
            if price * qty != quote:
                # Do not invent a tolerance without the instrument's precision
                # contract. A rounded discrepancy needs explicit adjudication.
                raise OpenIntentError(
                    "trade price and quote require precision adjudication"
                )
            seen.add(trade_id)
            quantity += qty
            quote_quantity += quote
            commissions[asset] = commissions.get(asset, Decimal(0)) + commission
            fills.append(
                TerminalFill(
                    trade_id,
                    _text(price),
                    _text(qty),
                    _text(quote),
                    _text(commission),
                    asset,
                    timestamp,
                )
            )
        if quantity != Decimal(terminal.executed_quantity) or quote_quantity != Decimal(
            terminal.quote_quantity
        ):
            raise OpenIntentError(
                "trade page does not reconcile terminal cumulative totals"
            )
        return TerminalFillEvidence(
            terminal,
            tuple(sorted(fills, key=lambda fill: int(fill.trade_id))),
            _text(quantity),
            _text(quote_quantity),
            tuple(
                (asset, _text(amount)) for asset, amount in sorted(commissions.items())
            ),
        )
