"""Conditional linear carry settlement replication, not an execution policy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction


def _finite(value: Decimal, label: str, *, positive: bool = False) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{label} must be a finite Decimal")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    if abs(value.as_tuple().exponent) > 100 or len(value.as_tuple().digits) > 100:
        raise ValueError(f"{label} exceeds the bounded arithmetic contract")


@dataclass(frozen=True, slots=True)
class SettlementObservation:
    timestamp_ms: int
    index_price: Decimal
    spot_reference_price: Decimal
    spot_execution_price: Decimal | None
    settlement_weight_units: Decimal
    spot_exit_weight: Decimal


@dataclass(frozen=True, slots=True)
class SettlementTrackingResult:
    settlement_price: Decimal
    spot_exit_price: Decimal
    entry_basis_quote: Decimal
    index_tracking_quote: Decimal
    timing_tracking_quote: Decimal
    execution_tracking_quote: Decimal
    total_cost_quote: Decimal
    net_cashflow_quote: Decimal
    direct_cashflow_quote: Decimal
    timing_weight_distance: Decimal
    conditional_timing_loss_bound_quote: Decimal
    qualified_edge: bool = False


def evaluate_linear_settlement_tracking(
    *,
    quantity: Decimal,
    spot_entry_price: Decimal,
    futures_entry_price: Decimal,
    total_cost_quote: Decimal,
    declared_timestamps_ms: tuple[int, ...],
    observations: tuple[SettlementObservation, ...],
    spot_price_lower_bound: Decimal,
    spot_price_upper_bound: Decimal,
) -> SettlementTrackingResult:
    """Decompose linear carry into basis, index tracking and exit-timing error.

    All prices/costs must share a quote unit and quantities a linear base unit.
    Settlement weight units are normalized as exact rational weights; exit
    weights are actual fractions of total base quantity and must sum to one.
    Price slippage is accounted from executions, not charged again in costs.
    The caller must source-prove the declared complete settlement schedule,
    execution feasibility, cost coverage and any forward price envelope. This
    calculation cannot turn realized sample extrema into a prospective bound.
    It does not apply to inverse/quanto contracts or certify interim solvency.
    """
    for label, value in (
        ("quantity", quantity),
        ("spot entry", spot_entry_price),
        ("future entry", futures_entry_price),
        ("spot lower bound", spot_price_lower_bound),
        ("spot upper bound", spot_price_upper_bound),
    ):
        _finite(value, label, positive=True)
    _finite(total_cost_quote, "total cost")
    if total_cost_quote < 0 or spot_price_lower_bound > spot_price_upper_bound:
        raise ValueError("cost or price envelope is invalid")
    if (
        not isinstance(declared_timestamps_ms, tuple)
        or not declared_timestamps_ms
        or any(
            type(timestamp) is not int or timestamp <= 0
            for timestamp in declared_timestamps_ms
        )
        or tuple(sorted(set(declared_timestamps_ms))) != declared_timestamps_ms
        or not isinstance(observations, tuple)
        or any(not isinstance(row, SettlementObservation) for row in observations)
        or tuple(row.timestamp_ms for row in observations) != declared_timestamps_ms
        or any(type(row.timestamp_ms) is not int for row in observations)
    ):
        raise ValueError("observations do not match the declared complete schedule")
    for row in observations:
        _finite(row.index_price, "index price", positive=True)
        _finite(row.spot_reference_price, "spot reference", positive=True)
        _finite(row.settlement_weight_units, "settlement weight", positive=True)
        _finite(row.spot_exit_weight, "spot exit weight")
        if not 0 <= row.spot_exit_weight <= 1:
            raise ValueError("exit weights must describe full long-spot exit")
        if row.spot_exit_weight:
            _finite(row.spot_execution_price, "spot execution", positive=True)
        elif row.spot_execution_price is not None:
            raise ValueError("a zero-exit slot must not contain a fictitious execution")
        if (
            not spot_price_lower_bound
            <= row.spot_reference_price
            <= spot_price_upper_bound
        ):
            raise ValueError("spot reference is outside the supplied price envelope")
    zero = Fraction(0)
    total_units = sum(
        (Fraction(row.settlement_weight_units) for row in observations), zero
    )
    if sum((Fraction(row.spot_exit_weight) for row in observations), zero) != 1:
        raise ValueError("exit weights must sum exactly to one")
    settlement = spot_exit = zero
    index_tracking = timing_tracking = execution_tracking = zero
    distance = zero
    q = Fraction(quantity)
    for row in observations:
        weight = Fraction(row.settlement_weight_units) / total_units
        exit_weight = Fraction(row.spot_exit_weight)
        reference = Fraction(row.spot_reference_price)
        index = Fraction(row.index_price)
        settlement += weight * index
        index_tracking += q * weight * (reference - index)
        timing_tracking += q * (exit_weight - weight) * reference
        distance += abs(exit_weight - weight) / 2
        if row.spot_execution_price is not None:
            execution = Fraction(row.spot_execution_price)
            spot_exit += exit_weight * execution
            execution_tracking += q * exit_weight * (execution - reference)
    basis = q * (Fraction(futures_entry_price) - Fraction(spot_entry_price))
    timing_bound = (
        q
        * distance
        * (Fraction(spot_price_upper_bound) - Fraction(spot_price_lower_bound))
    )
    net = (
        basis
        + index_tracking
        + timing_tracking
        + execution_tracking
        - Fraction(total_cost_quote)
    )
    direct = q * (
        spot_exit
        - Fraction(spot_entry_price)
        + Fraction(futures_entry_price)
        - settlement
    ) - Fraction(total_cost_quote)
    values = {
        "settlement_price": settlement,
        "spot_exit_price": spot_exit,
        "entry_basis_quote": basis,
        "index_tracking_quote": index_tracking,
        "timing_tracking_quote": timing_tracking,
        "execution_tracking_quote": execution_tracking,
        "total_cost_quote": Fraction(total_cost_quote),
        "net_cashflow_quote": net,
        "direct_cashflow_quote": direct,
        "timing_weight_distance": distance,
        "conditional_timing_loss_bound_quote": timing_bound,
    }
    # Rational arithmetic preserves equal weights such as 1/1800. Only the
    # presentation boundary rounds recurring outputs, consistently to 60 digits.
    with localcontext() as context:
        context.prec = 60
        return SettlementTrackingResult(
            **{
                name: Decimal(value.numerator) / Decimal(value.denominator)
                for name, value in values.items()
            }
        )
