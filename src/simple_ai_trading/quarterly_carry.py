"""Exact displayed-depth arithmetic for fully funded quarterly cash-and-carry."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Sequence

from .paper_execution import BookLevel


_BIPS = Decimal("10000")
_MILLISECONDS_PER_YEAR = Decimal("31557600000")


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        qualifier = "positive " if positive else "finite "
        raise ValueError(f"{name} must be a {qualifier}decimal")
    return parsed


@dataclass(frozen=True, slots=True)
class DepthVwap:
    """An exact quantity filled by walking already sorted displayed levels."""

    quantity: Decimal
    quote_value: Decimal

    @property
    def price(self) -> Decimal:
        return self.quote_value / self.quantity


@dataclass(frozen=True, slots=True)
class QuarterlyCarryResult:
    """One direction-neutral basis diagnostic before unresolved account risks."""

    quantity: Decimal
    capture_time_ms: int
    delivery_time_ms: int
    spot_buy: DepthVwap
    future_sale: DepthVwap
    gross_profit_quote: Decimal
    gross_basis_bips: Decimal
    all_in_cost_hurdle_bips: Decimal
    after_hurdle_profit_quote: Decimal
    after_hurdle_basis_bips: Decimal
    gross_simple_annualized_bips: Decimal
    after_hurdle_simple_annualized_bips: Decimal

    @property
    def gross_positive(self) -> bool:
        return self.gross_profit_quote > 0

    @property
    def after_hurdle_positive(self) -> bool:
        return self.after_hurdle_profit_quote > 0


def walk_depth(
    levels: Sequence[BookLevel],
    *,
    quantity: Decimal,
    descending: bool,
) -> DepthVwap | None:
    """Walk best-to-worst displayed depth without extrapolating missing size."""

    requested = _decimal(quantity, name="carry quantity", positive=True)
    normalized = tuple(level.validated() for level in levels)
    if (
        tuple(sorted(normalized, key=lambda level: level.price, reverse=descending))
        != normalized
    ):
        raise ValueError("carry book levels are not best-to-worst sorted")
    if len({level.price for level in normalized}) != len(normalized):
        raise ValueError("carry book prices are duplicated")
    remaining = requested
    quote_value = Decimal("0")
    for level in normalized:
        consumed = min(remaining, level.quantity)
        quote_value += consumed * level.price
        remaining -= consumed
        if remaining == 0:
            return DepthVwap(quantity=requested, quote_value=quote_value)
    return None


def screen_quarterly_cash_and_carry(
    *,
    spot_asks: Sequence[BookLevel],
    future_bids: Sequence[BookLevel],
    quantity: Decimal,
    capture_time_ms: int,
    delivery_time_ms: int,
    all_in_cost_hurdle_bips: Decimal,
) -> QuarterlyCarryResult | None:
    """Price long spot plus short dated future at exact displayed depth.

    The cost hurdle is an explicit sensitivity input, not an inferred account
    commission. A positive result remains unqualified until exact commissions,
    collateral opportunity cost, liquidation protection, and spot/index exit
    basis are independently established.
    """

    if (
        not isinstance(capture_time_ms, int)
        or isinstance(capture_time_ms, bool)
        or not isinstance(delivery_time_ms, int)
        or isinstance(delivery_time_ms, bool)
    ):
        raise ValueError("carry timestamps must be integer milliseconds")
    captured = capture_time_ms
    delivery = delivery_time_ms
    if captured <= 0 or delivery <= captured:
        raise ValueError("carry delivery must be after a positive capture time")
    hurdle = _decimal(
        all_in_cost_hurdle_bips,
        name="all-in cost hurdle bips",
    )
    if hurdle < 0 or hurdle >= _BIPS:
        raise ValueError("all-in cost hurdle bips is outside [0, 10000)")

    spot = walk_depth(spot_asks, quantity=quantity, descending=False)
    future = walk_depth(future_bids, quantity=quantity, descending=True)
    if spot is None or future is None:
        return None
    gross_profit = future.quote_value - spot.quote_value
    gross_bips = gross_profit / spot.quote_value * _BIPS
    hurdle_quote = spot.quote_value * hurdle / _BIPS
    after_hurdle_profit = gross_profit - hurdle_quote
    after_hurdle_bips = gross_bips - hurdle
    annualizer = _MILLISECONDS_PER_YEAR / Decimal(delivery - captured)
    return QuarterlyCarryResult(
        quantity=spot.quantity,
        capture_time_ms=captured,
        delivery_time_ms=delivery,
        spot_buy=spot,
        future_sale=future,
        gross_profit_quote=gross_profit,
        gross_basis_bips=gross_bips,
        all_in_cost_hurdle_bips=hurdle,
        after_hurdle_profit_quote=after_hurdle_profit,
        after_hurdle_basis_bips=after_hurdle_bips,
        gross_simple_annualized_bips=gross_bips * annualizer,
        after_hurdle_simple_annualized_bips=after_hurdle_bips * annualizer,
    )


__all__ = [
    "DepthVwap",
    "QuarterlyCarryResult",
    "screen_quarterly_cash_and_carry",
    "walk_depth",
]
