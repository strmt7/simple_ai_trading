"""Bind order responses to outgoing requests before callers may account for them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
import re

from .binance_execution_scope import parse_execution_id


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
        if (
            not isinstance(self.symbol, str)
            or re.fullmatch(r"[A-Z0-9]{1,32}", self.symbol) is None
            or self.side not in ("BUY", "SELL")
            or self.market_type not in ("spot", "futures")
            or not isinstance(self.quantity, str)
            or re.fullmatch(r"[0-9]{1,30}(?:\.[0-9]{1,30})?", self.quantity) is None
            or Decimal(self.quantity) <= 0
            or not isinstance(self.client_order_id, str)
            or not 1 <= len(self.client_order_id) <= 36
            or self.client_order_id != self.client_order_id.strip()
            or type(self.reduce_only) is not bool
        ):
            raise ValueError("submitted order binding is invalid")

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
