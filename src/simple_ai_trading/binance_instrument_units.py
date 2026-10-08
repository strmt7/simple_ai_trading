"""Shared explicit units for native opening and closing observations."""

from __future__ import annotations

from collections.abc import Mapping

from .binance_execution_scope import BinanceExecutionScope
from .binance_open_intents import OpenIntentError


def instrument_asset_units(
    instrument: Mapping[str, object], symbol: str, scope: BinanceExecutionScope
) -> dict[str, str]:
    """Validate supplied units without inferring an asset or proving freshness."""
    if not isinstance(instrument, Mapping):
        raise OpenIntentError("native inventory requires explicit instrument metadata")
    base, quote = instrument.get("baseAsset"), instrument.get("quoteAsset")
    if (
        base not in ("BTC", "ETH", "SOL")
        or quote not in ("USDT", "USDC")
        or instrument.get("symbol") != symbol
        or symbol != base + quote
    ):
        raise OpenIntentError("instrument asset identity is inconsistent")
    result = {"symbol": symbol, "baseAsset": base, "quoteAsset": quote}
    if scope.market_type == "futures":
        if (
            instrument.get("contractType") != "PERPETUAL"
            or instrument.get("marginAsset") != quote
            or instrument.get("underlyingType") != "COIN"
        ):
            raise OpenIntentError("native inventory requires linear crypto units")
        result.update(
            contractType="PERPETUAL", marginAsset=quote, underlyingType="COIN"
        )
    return result
