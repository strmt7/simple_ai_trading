"""Exact conditional payout floors for two fully paid long European options.

This is supplied-law research, not a venue adapter or profitability test. Both
options must use the same nonnegative settlement scalar, linear base quantity
and quote-cash asset. Exercise, ownership, premium payment, native settlement,
fee rounding and account treatment require independent source qualification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction


def _rational(value: Decimal, label: str) -> Fraction:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{label} must be a finite Decimal")
    if abs(value.as_tuple().exponent) > 100 or len(value.as_tuple().digits) > 100:
        raise ValueError(f"{label} exceeds the bounded arithmetic contract")
    return Fraction(value)


@dataclass(frozen=True, slots=True)
class CappedExerciseFee:
    """Supplied per-base fee: min(index_rate*S, cap*positive_intrinsic)."""

    index_rate: Decimal
    intrinsic_cap_fraction: Decimal

    def coefficients(self) -> tuple[Fraction, Fraction]:
        rate = _rational(self.index_rate, "exercise index rate")
        cap = _rational(self.intrinsic_cap_fraction, "exercise intrinsic cap")
        if not 0 <= rate <= 1 or not 0 <= cap <= 1:
            raise ValueError("exercise rate and intrinsic cap must be in [0, 1]")
        return rate, cap


@dataclass(frozen=True, slots=True)
class LongOptionFloorPoint:
    settlement_price: Fraction
    call_gross_payout_quote: Fraction
    put_gross_payout_quote: Fraction
    call_exercise_fee_quote: Fraction
    put_exercise_fee_quote: Fraction
    terminal_net_payout_quote: Fraction


@dataclass(frozen=True, slots=True)
class LongOptionFloor:
    gross_payout_floor_quote: Fraction
    terminal_net_payout_floor_quote: Fraction
    minimizing_breakpoints: tuple[Fraction, ...]
    critical_points: tuple[LongOptionFloorPoint, ...]
    # Supplied arithmetic cannot establish acquisition economics or venue terms.
    qualified_edge: bool = field(default=False, init=False)


def evaluate_fully_paid_long_option_floor(
    *,
    lower_call_strike: Decimal,
    upper_put_strike: Decimal,
    matched_base_quantity: Decimal,
    call_exercise_fee: CappedExerciseFee,
    put_exercise_fee: CappedExerciseFee,
    terminal_cost_reserve_quote: Decimal,
) -> LongOptionFloor:
    """Find the exact all-state payout floor, without sampling settlement prices.

    The call pays q*max(S-K_low,0); the put pays q*max(K_high-S,0).
    Premiums, entry fees and opportunity costs are NOT subtracted here. Compare
    this payout with complete initial cash and a feasible equal-capital alternative
    separately. A positive payout floor alone is not a positive profit floor.

    Every kink is a strike or a valid fee-switch point. Above the upper strike
    only the call pays; with both fee coefficients in [0,1], each terminal affine
    slope is nonnegative. Thus no minimum is lost at infinity. Exact fractions
    retain nonterminating switch points independently of Decimal precision.

    The nonnegative reserve must bound additional terminal costs, including
    rounding, for this complete quantity. Its coverage is supplied, not proved.
    Different expiry clocks, settlement rules/assets or unmatched quantities do
    not satisfy the common-scalar law; this helper does not certify those inputs.
    """
    lower = _rational(lower_call_strike, "lower call strike")
    upper = _rational(upper_put_strike, "upper put strike")
    quantity = _rational(matched_base_quantity, "matched base quantity")
    reserve = _rational(terminal_cost_reserve_quote, "terminal cost reserve")
    if not 0 < lower < upper or quantity <= 0 or reserve < 0:
        raise ValueError(
            "positive ordered strikes/quantity and nonnegative reserve required"
        )
    if not isinstance(call_exercise_fee, CappedExerciseFee) or not isinstance(
        put_exercise_fee, CappedExerciseFee
    ):
        raise ValueError("both supplied exercise fee laws are required")
    call_rate, call_cap = call_exercise_fee.coefficients()
    put_rate, put_cap = put_exercise_fee.coefficients()

    prices = {Fraction(0), lower, upper}
    if call_cap != call_rate:
        switch = call_cap * lower / (call_cap - call_rate)
        if switch >= lower:
            prices.add(switch)
    if put_cap + put_rate:
        switch = put_cap * upper / (put_cap + put_rate)
        if 0 <= switch <= upper:
            prices.add(switch)

    points = []
    for price in sorted(prices):
        call = max(price - lower, Fraction(0))
        put = max(upper - price, Fraction(0))
        call_fee = min(call_rate * price, call_cap * call)
        put_fee = min(put_rate * price, put_cap * put)
        points.append(
            LongOptionFloorPoint(
                settlement_price=price,
                call_gross_payout_quote=quantity * call,
                put_gross_payout_quote=quantity * put,
                call_exercise_fee_quote=quantity * call_fee,
                put_exercise_fee_quote=quantity * put_fee,
                terminal_net_payout_quote=quantity * (call + put - call_fee - put_fee)
                - reserve,
            )
        )
    minimum = min(point.terminal_net_payout_quote for point in points)
    return LongOptionFloor(
        gross_payout_floor_quote=quantity * (upper - lower),
        terminal_net_payout_floor_quote=minimum,
        minimizing_breakpoints=tuple(
            point.settlement_price
            for point in points
            if point.terminal_net_payout_quote == minimum
        ),
        critical_points=tuple(points),
    )
