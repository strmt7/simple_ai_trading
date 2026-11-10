"""Bind order responses to outgoing requests before callers may account for them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
import re
from typing import ClassVar

from .binance_execution_scope import parse_execution_id


def _validate_market_request_fields(
    symbol: str,
    side: str,
    amount: str,
    client_order_id: str,
    market_type: str,
    reduce_only: bool,
) -> None:
    if (
        not isinstance(symbol, str)
        or re.fullmatch(r"[A-Z0-9]{1,32}", symbol) is None
        or side not in ("BUY", "SELL")
        or market_type not in ("spot", "futures")
        or not isinstance(amount, str)
        or re.fullmatch(r"[0-9]{1,30}(?:\.[0-9]{1,30})?", amount) is None
        or Decimal(amount) <= 0
        or not isinstance(client_order_id, str)
        or not 1 <= len(client_order_id) <= 36
        or client_order_id != client_order_id.strip()
        or type(reduce_only) is not bool
    ):
        raise ValueError("submitted order binding is invalid")


def _nonnegative_response_amount(value: object) -> Decimal:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9]{1,30}(?:\.[0-9]{1,30})?", value) is None
    ):
        raise ValueError("order response amount is not an exact nonnegative decimal")
    return Decimal(value)


def validate_queried_order(
    response: Mapping[str, object],
    *,
    symbol: str,
    order_id: str | None,
    client_order_id: str | None,
) -> None:
    """Require every supplied selector, not only whichever the venue prioritizes."""
    if not isinstance(response, Mapping) or response.get("symbol") != symbol:
        raise ValueError("order response symbol differs from the request")
    observed_id = parse_execution_id(response.get("orderId"))
    if observed_id is None or order_id is not None and observed_id != order_id:
        raise ValueError("order response exchange identity differs from the request")
    if client_order_id is not None and (
        response.get("clientOrderId") != client_order_id
        or "origClientOrderId" in response
        and response["origClientOrderId"] != client_order_id
    ):
        raise ValueError("order response client identity differs from the request")


@dataclass(frozen=True)
class MarketOrderBinding:
    symbol: str
    side: str
    quantity: str
    client_order_id: str
    market_type: str
    reduce_only: bool

    def __post_init__(self) -> None:
        _validate_market_request_fields(
            self.symbol,
            self.side,
            self.quantity,
            self.client_order_id,
            self.market_type,
            self.reduce_only,
        )

    def validate(self, response: Mapping[str, object]) -> None:
        """Check submitted MARKET semantics; this does not prove terminal fills."""
        validate_queried_order(
            response,
            symbol=self.symbol,
            order_id=None,
            client_order_id=self.client_order_id,
        )
        original = response.get("origQty")
        if (
            response.get("side") != self.side
            or response.get("type") != "MARKET"
            or not isinstance(original, str)
            or re.fullmatch(r"[0-9]{1,30}(?:\.[0-9]{1,30})?", original) is None
            or Decimal(original) != Decimal(self.quantity)
        ):
            raise ValueError(
                "order response side, type or quantity differs from the request"
            )
        if self.market_type == "futures" and (
            response.get("positionSide") != "BOTH"
            or response.get("reduceOnly") is not self.reduce_only
            or "closePosition" in response
            and response["closePosition"] is not False
            or "origType" in response
            and response["origType"] != "MARKET"
        ):
            raise ValueError("order response differs from one-way MARKET semantics")


@dataclass(frozen=True)
class SpotQuoteOrderBinding:
    """Bind a quote-denominated BUY principal; fees and terminal fills are separate.

    Base quantity is chosen by the venue. It must never be compared with the
    requested quote amount or used to substitute a base-denominated request.
    """

    symbol: str
    quote_quantity: str
    client_order_id: str
    market_type: ClassVar[str] = "spot"

    def __post_init__(self) -> None:
        _validate_market_request_fields(
            self.symbol, "BUY", self.quote_quantity, self.client_order_id, "spot", False
        )

    def validate(self, response: Mapping[str, object]) -> None:
        validate_queried_order(
            response,
            symbol=self.symbol,
            order_id=None,
            client_order_id=self.client_order_id,
        )
        if response.get("side") != "BUY" or response.get("type") != "MARKET":
            raise ValueError("quote BUY response side or type differs from the request")
        original = _nonnegative_response_amount(response.get("origQuoteOrderQty"))
        spent = _nonnegative_response_amount(response.get("cummulativeQuoteQty"))
        requested = Decimal(self.quote_quantity)
        if original != requested or spent > requested:
            raise ValueError("quote BUY response differs from or exceeds its principal")
