"""Receipt-bound spot resale limits; wallet balance is not purchase ownership."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, localcontext

from .assets import SUPPORTED_MAJOR_BASE_ASSETS, SUPPORTED_MAJOR_QUOTE_ASSETS
from .binance_acknowledgements import acknowledged_fill
from .binance_open_intents import OpenIntentError
from .binance_terminal_fills import _decimal


def spot_buy_received_quantity(
    order: Mapping[str, object] | None,
    *,
    symbol: str,
    base_asset: str,
    quote_asset: str,
) -> Decimal:
    """Subtract complete native base commissions from a terminal owned BUY receipt.

    The caller must first bind the receipt to its submitted order. This function
    proves neither account availability nor fees on a subsequent SELL. Negative
    commissions are not admitted as additional sale inventory under this contract.
    """
    if (
        base_asset not in SUPPORTED_MAJOR_BASE_ASSETS
        or quote_asset not in SUPPORTED_MAJOR_QUOTE_ASSETS
        or symbol != base_asset + quote_asset
        or not isinstance(order, Mapping)
        or order.get("symbol") != symbol
        or order.get("side") != "BUY"
        or order.get("type") != "MARKET"
        or order.get("status")
        not in {"FILLED", "CANCELED", "EXPIRED", "EXPIRED_IN_MATCH"}
    ):
        raise OpenIntentError("spot resale requires an exact terminal major-asset BUY")
    fill = acknowledged_fill(order, market_type="spot")
    rows = order.get("fills")
    if fill is None or not isinstance(rows, list) or not rows:
        raise OpenIntentError("spot resale requires complete native commission rows")
    with localcontext() as context:
        context.prec = 128
        base_fee = Decimal(0)
        for row in rows:
            if "commission" not in row or "commissionAsset" not in row:
                raise OpenIntentError("spot resale commission evidence is incomplete")
            # Exact acknowledgement validation already checked row totals/assets.
            commission = _decimal(row["commission"])
            if row["commissionAsset"] == base_asset:
                base_fee += commission
        received = Decimal(fill.quantity) - base_fee
        if received <= 0:
            raise OpenIntentError("spot BUY supplied no positive net resale inventory")
        return received
