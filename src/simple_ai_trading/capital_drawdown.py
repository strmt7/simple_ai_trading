"""Exact settled-PnL peak retracement in one caller-qualified quote unit."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .positions import ClosedTrade


def _cash(value: object) -> Fraction:
    if type(value) not in {int, float, Decimal, Fraction}:
        raise ValueError("drawdown cash must be a finite number")
    try:
        return Fraction(str(value))
    except (ValueError, OverflowError, ZeroDivisionError):
        raise ValueError("drawdown cash must be a finite number") from None


@dataclass(frozen=True)
class SettledCashDrawdown:
    settled_peak_pnl_quote: Fraction
    current_pnl_quote: Fraction
    drawdown_quote: Fraction
    reference_fraction: Fraction


def settled_cash_drawdown(
    closed: Sequence[ClosedTrade],
    *,
    unrealized_pnl_quote: object,
    reference_cash_quote: object,
    observed_at_ms: int,
) -> SettledCashDrawdown:
    """Reconstruct retained settled-PnL prefixes plus the current supplied mark.

    Equal-time settlements form one group; file order cannot fabricate a peak.
    The denominator is the supplied risk-capital reference, not peak equity.
    This is neither a complete marked-NAV high-water nor account/FX evidence.
    It cannot reconstruct omitted intermediate marks, fills or external flows.
    """
    reference = _cash(reference_cash_quote)
    if reference <= 0 or type(observed_at_ms) is not int or observed_at_ms < 0:
        raise ValueError("drawdown reference or observation time is invalid")
    grouped: dict[int, Fraction] = {}
    for trade in closed:
        stamp = trade.closed_at_ms
        if (
            type(stamp) is not int
            or type(trade.opened_at_ms) is not int
            or not 0 <= trade.opened_at_ms <= stamp <= observed_at_ms
        ):
            raise ValueError("settled cash chronology is invalid")
        grouped[stamp] = grouped.get(stamp, Fraction(0)) + _cash(trade.realized_pnl)
    cumulative = peak = Fraction(0)
    for stamp in sorted(grouped):
        cumulative += grouped[stamp]
        peak = max(peak, cumulative)
    current = cumulative + _cash(unrealized_pnl_quote)
    drawdown = max(Fraction(0), peak - current)
    return SettledCashDrawdown(peak, current, drawdown, drawdown / reference)
