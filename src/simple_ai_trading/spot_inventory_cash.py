"""Exact supplied spot cash deltas, without native-fill or cost qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
import hashlib
import json
import re

from .funding_cash import USDM_SYMBOLS


def _digest(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _amount(value: Fraction, *, positive: bool = False) -> bool:
    return isinstance(value, Fraction) and (value > 0 if positive else value >= 0)


def cash_input_digest(value: object) -> str:
    """Bind exact rational supplied inputs without float conversion."""

    def encode(item: object) -> str:
        if not isinstance(item, Fraction):
            raise TypeError("cash binding contains an unsupported value")
        return str(item)

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=encode,
            allow_nan=False,
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SpotCashTrade:
    """One incremental fill; gross cash excludes the explicitly supplied fees.

    This is not a cumulative order acknowledgement. An execution ID and source
    digest bind the supplied declaration, not its venue origin or completeness.
    Only base/quote commissions are supported; other assets require prior
    independent valuation. SELL base fees remain unsupported, as in native cash.
    """

    execution_id: str
    boundary_index: int
    side: str
    gross_base_quantity: Fraction
    gross_quote_quantity: Fraction
    base_commission: Fraction
    quote_commission: Fraction
    source_body_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.execution_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", self.execution_id) is None
            or type(self.boundary_index) is not int
            or self.boundary_index < 0
            or self.side not in ("BUY", "SELL")
            or not _amount(self.gross_base_quantity, positive=True)
            or not _amount(self.gross_quote_quantity, positive=True)
            or not _amount(self.base_commission)
            or not _amount(self.quote_commission)
            or not _digest(self.source_body_sha256)
            or self.base_commission >= self.gross_base_quantity
            or (self.side == "SELL" and self.base_commission != 0)
        ):
            raise ValueError("supplied incremental spot trade is invalid")

    @property
    def base_delta(self) -> Fraction:
        return (
            self.gross_base_quantity - self.base_commission
            if self.side == "BUY"
            else -self.gross_base_quantity
        )

    @property
    def quote_delta(self) -> Fraction:
        gross = (
            self.gross_quote_quantity
            if self.side == "SELL"
            else -self.gross_quote_quantity
        )
        return gross - self.quote_commission


@dataclass(frozen=True)
class SpotInventoryCashPath:
    symbol: str
    boundary_time_ms: tuple[int, ...]
    boundary_price: tuple[Fraction, ...]
    initial_base_quantity: Fraction
    trades: tuple[SpotCashTrade, ...]
    price_source_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.symbol, str)
            or self.symbol not in USDM_SYMBOLS
            or not isinstance(self.boundary_time_ms, tuple)
            or len(self.boundary_time_ms) < 2
            or any(
                type(t) is not int or not 0 <= t < 2**63 for t in self.boundary_time_ms
            )
            or tuple(sorted(set(self.boundary_time_ms))) != self.boundary_time_ms
            or not isinstance(self.boundary_price, tuple)
            or len(self.boundary_price) != len(self.boundary_time_ms)
            or any(not _amount(p, positive=True) for p in self.boundary_price)
            or not _amount(self.initial_base_quantity)
            or not isinstance(self.trades, tuple)
            or any(not isinstance(row, SpotCashTrade) for row in self.trades)
            or not _digest(self.price_source_sha256)
        ):
            raise ValueError("supplied spot inventory path is invalid")
        indices = tuple(row.boundary_index for row in self.trades)
        if (
            indices != tuple(sorted(indices))
            or any(i >= len(self.boundary_price) for i in indices)
            or len({row.execution_id for row in self.trades}) != len(self.trades)
        ):
            raise ValueError(
                "spot trades require unique identities and ordered boundaries"
            )


@dataclass(frozen=True)
class SpotInventoryCashReplay:
    symbol: str
    quantity_by_interval: tuple[Fraction, ...]
    quantity_range_by_boundary: tuple[tuple[Fraction, Fraction], ...]
    price_cash_by_interval: tuple[Fraction, ...]
    execution_adjustment_by_boundary: tuple[Fraction, ...]
    recorded_quote_cash_change: Fraction
    initial_marked_inventory: Fraction
    incremental_wealth_change: Fraction
    input_sha256: str
    financially_qualified: bool = field(default=False, init=False)


def replay_spot_inventory_cash(path: SpotInventoryCashPath) -> SpotInventoryCashReplay:
    """Conserve cash plus net base inventory through explicit terminal liquidation."""
    if not isinstance(path, SpotInventoryCashPath):
        raise ValueError("validated supplied spot path required")
    grouped: list[list[SpotCashTrade]] = [[] for _ in path.boundary_time_ms]
    for trade in path.trades:
        grouped[trade.boundary_index].append(trade)
    quantity = path.initial_base_quantity
    cash = Fraction(0)
    quantities, ranges, adjustments = [], [], []
    for price, trades in zip(path.boundary_price, grouped, strict=True):
        low = high = quantity
        adjustment = Fraction(0)
        for trade in trades:
            quantity += trade.base_delta
            if quantity < 0:
                raise ValueError("spot disposal exceeds supplied net received quantity")
            cash += trade.quote_delta
            adjustment += trade.quote_delta + trade.base_delta * price
            low, high = min(low, quantity), max(high, quantity)
        quantities.append(quantity)
        ranges.append((low, high))
        adjustments.append(adjustment)
    if quantity != 0:
        raise ValueError("terminal spot inventory requires explicit unwind cash")
    price_cash = tuple(
        q * (right - left)
        for q, left, right in zip(
            quantities[:-1],
            path.boundary_price[:-1],
            path.boundary_price[1:],
            strict=True,
        )
    )
    initial_mark = path.initial_base_quantity * path.boundary_price[0]
    wealth = sum(price_cash, Fraction(0)) + sum(adjustments, Fraction(0))
    if wealth != cash - initial_mark:
        raise ValueError("spot cash and inventory conservation failed")
    return SpotInventoryCashReplay(
        path.symbol,
        tuple(quantities[:-1]),
        tuple(ranges),
        price_cash,
        tuple(adjustments),
        cash,
        initial_mark,
        wealth,
        cash_input_digest(asdict(path)),
    )
