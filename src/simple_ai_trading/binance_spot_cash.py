"""Receipt-derived Spot realized cash PnL, without guessed fee conversions.

Exact rational lot allocation avoids cumulative partial-close rounding. The
existing float fields are checked projections, not an account balance ledger.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING

from .assets import SUPPORTED_MAJOR_BASE_ASSETS, symbol_base_for_supported_quote
from .binance_acknowledgements import acknowledged_fill
from .binance_open_intents import OpenIntentError
from .binance_spot_receipts import native_spot_entry_net
from .binance_terminal_fills import _decimal, execution_id

if TYPE_CHECKING:
    from .positions import ClosedTrade, OpenPosition

_ORDER_FIELDS = (
    "symbol",
    "side",
    "type",
    "status",
    "orderId",
    "clientOrderId",
    "executedQty",
    "origQty",
    "cummulativeQuoteQty",
    "fills",
)
_FILL_FIELDS = ("qty", "price", "commission", "commissionAsset", "tradeId", "quoteQty")


@dataclass(frozen=True)
class SpotCashReceipt:
    status: str
    quantity: Fraction
    cash: Fraction
    base: str
    quote: str
    commissions: tuple[tuple[str, Fraction], ...]

    def fee(self, asset: str) -> Fraction:
        return dict(self.commissions).get(asset, Fraction(0))


def _receipt(
    order: Mapping[str, object],
    *,
    symbol: str,
    side: str,
    client_id: str,
    order_id: str,
) -> SpotCashReceipt:
    base = symbol_base_for_supported_quote(symbol)
    if (
        base not in SUPPORTED_MAJOR_BASE_ASSETS
        or symbol[len(base) :] not in {"USDT", "USDC"}
        or order.get("symbol") != symbol
        or order.get("side") != side
        or side not in {"BUY", "SELL"}
        or order.get("type") != "MARKET"
        or order.get("status")
        not in ({"FILLED"} if side == "BUY" else {"FILLED", "PARTIALLY_FILLED"})
        or order.get("clientOrderId") != client_id
        or not isinstance(client_id, str)
        or not re.fullmatch(r"sait-[oc]-[A-Za-z0-9_-]{1,29}", client_id)
        or not client_id.startswith("sait-o-" if side == "BUY" else "sait-c-")
        or execution_id(order.get("orderId")) != order_id
    ):
        raise OpenIntentError("native cash receipt identity is inconsistent")
    fill = acknowledged_fill(order, market_type="spot")
    rows = order.get("fills")
    if (
        fill is None
        or fill.quote_quantity is None
        or not isinstance(rows, list)
        or not rows
    ):
        raise OpenIntentError("native cash requires complete cash and commission rows")
    fees: dict[str, Fraction] = {}
    for row in rows:
        if "commission" not in row or "commissionAsset" not in row:
            raise OpenIntentError("native cash commission evidence is incomplete")
        asset = row["commissionAsset"]
        fees[asset] = fees.get(asset, Fraction(0)) + Fraction(
            _decimal(row["commission"])
        )
    return SpotCashReceipt(
        order["status"],
        Fraction(fill.quantity),
        Fraction(fill.quote_quantity),
        base,
        symbol[len(base) :],
        tuple(sorted(fees.items())),
    )


def encode_spot_cash_receipt(
    order: Mapping[str, object],
    *,
    symbol: str,
    side: str,
    client_id: str,
    order_id: str,
) -> str:
    """Retain only validated execution fields; never serialize arbitrary response data."""
    _receipt(order, symbol=symbol, side=side, client_id=client_id, order_id=order_id)
    return _encode_selected(order)


def _encode_selected(order: Mapping[str, object]) -> str:
    selected = {key: order[key] for key in _ORDER_FIELDS if key in order}
    selected["fills"] = [
        {key: row[key] for key in _FILL_FIELDS if key in row} for row in order["fills"]
    ]
    encoded = json.dumps(
        selected, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if len(encoded) > 1_000_000:
        raise OpenIntentError("native cash receipt exceeds its storage bound")
    return encoded


def _load(
    text: object, *, symbol: str, side: str, client_id: str, order_id: str
) -> SpotCashReceipt:
    if not isinstance(text, str) or not 0 < len(text) <= 1_000_000:
        raise OpenIntentError("native cash receipt encoding is invalid")
    try:
        order = json.loads(text)
    except (ValueError, RecursionError):
        raise OpenIntentError("native cash receipt is unreadable") from None
    if not isinstance(order, dict):
        raise OpenIntentError("native cash receipt is not canonical")
    receipt = _receipt(
        order, symbol=symbol, side=side, client_id=client_id, order_id=order_id
    )
    if _encode_selected(order) != text:
        raise OpenIntentError("native cash receipt is not canonical")
    return receipt


def entry_cash_receipt(record: OpenPosition | ClosedTrade) -> SpotCashReceipt | None:
    text = getattr(record, "spot_entry_cash_receipt", "")
    if text == "":
        if getattr(record, "spot_close_cash_receipt", "") != "":
            raise OpenIntentError("native close cash lacks original entry evidence")
        return None
    net = native_spot_entry_net(record)
    if net is None:
        raise OpenIntentError("native cash lacks received inventory evidence")
    entry = _load(
        text,
        symbol=record.symbol,
        side="BUY",
        client_id=record.open_client_order_id,
        order_id=record.open_exchange_order_id,
    )
    if (
        entry.quantity != Fraction(record.spot_gross_entry_quantity)
        or entry.fee(entry.base) != Fraction(record.spot_entry_base_commission)
        or entry.quantity - entry.fee(entry.base) != Fraction(net)
        or record.entry_price != float(entry.cash / entry.quantity)
    ):
        raise OpenIntentError("native cash differs from original inventory or price")
    return entry


def native_spot_cash_projection(record: ClosedTrade) -> tuple[float, float, float]:
    """Return PnL, allocated native fees and return on allocated quote cash."""
    entry = entry_cash_receipt(record)
    if entry is None:
        raise OpenIntentError("native cash projection requires entry evidence")
    close = _load(
        record.spot_close_cash_receipt,
        symbol=record.symbol,
        side="SELL",
        client_id=record.close_client_order_id,
        order_id=record.close_exchange_order_id,
    )
    if (
        close.status != record.exchange_status
        or close.quantity != Fraction(Decimal(str(record.qty)))
        or record.exit_price != float(close.cash / close.quantity)
    ):
        raise OpenIntentError(
            "native close quantity or price differs from cash receipt"
        )
    for leg in (entry, close):
        if any(
            fee and asset not in {leg.base, leg.quote} for asset, fee in leg.commissions
        ):
            raise OpenIntentError("third-asset commission valuation remains unresolved")
    if close.fee(close.base):
        raise OpenIntentError("SELL base commission requires inventory reconciliation")
    net = entry.quantity - entry.fee(entry.base)
    allocation = close.quantity / net
    if not 0 < allocation <= 1:
        raise OpenIntentError("native close exceeds original received inventory")
    cost = (entry.cash + entry.fee(entry.quote)) * allocation
    proceeds = close.cash - close.fee(close.quote)
    pnl = proceeds - cost
    fees = (
        entry.fee(entry.quote) + entry.fee(entry.base) * entry.cash / entry.quantity
    ) * allocation + close.fee(close.quote)
    return float(pnl), float(fees), float(pnl / cost)


def validate_spot_cash_record(record: OpenPosition | ClosedTrade) -> None:
    """Reconstruct persisted projections on load and before paired publication."""
    if entry_cash_receipt(record) is None or not hasattr(record, "realized_pnl"):
        return
    pnl, fees, pct = native_spot_cash_projection(record)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (record.realized_pnl, record.fees, record.realized_pnl_pct)
    ):
        raise OpenIntentError("native cash projections must be numeric")
    if (record.realized_pnl, record.fees, record.realized_pnl_pct) != (pnl, fees, pct):
        raise OpenIntentError("native cash projections differ from retained receipts")
