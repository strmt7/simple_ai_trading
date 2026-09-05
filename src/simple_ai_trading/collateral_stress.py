"""Conditional linear-hedge margin stress, not an exchange margin engine.

All inputs must share a quote unit and scenario instant. The caller must supply
joint spot/futures marks, actual credit semantics and total margin requirements;
the function cannot infer any of them from a delta hedge or nominal leverage.
Fully credited nonborrowed quote cash and full short-PnL recognition are explicit
model assumptions. No interest accrual, auto-exchange or liquidation is simulated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


def _nonnegative(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError("finite nonnegative Decimal required")
    return value


@dataclass(frozen=True, slots=True)
class CollateralState:
    label: str
    spot_mark: Decimal
    future_mark: Decimal
    asset_credit_ratio: Decimal
    total_margin_requirement: Decimal
    cumulative_cost_debits: Decimal


@dataclass(frozen=True, slots=True)
class CollateralStateValue:
    label: str
    short_unrealized_pnl: Decimal
    economic_equity: Decimal
    credited_equity: Decimal
    collateral_discount: Decimal
    headroom_after_requirement_and_buffer: Decimal


@dataclass(frozen=True, slots=True)
class CollateralStress:
    states: tuple[CollateralStateValue, ...]
    minimum_headroom: Decimal
    extra_fully_credited_cash_to_meet_supplied_states: Decimal
    strictly_positive_in_every_supplied_state: bool
    # Neither a finite scenario set nor arithmetic proves venue margin adequacy.
    venue_margin_qualified: bool = field(default=False, init=False)
    qualified_edge: bool = field(default=False, init=False)


def stress_linear_hedge(
    *,
    owned_asset_quantity: Decimal,
    short_base_quantity: Decimal,
    future_entry_price: Decimal,
    quote_cash: Decimal,
    required_quote_buffer: Decimal,
    states: tuple[CollateralState, ...],
) -> CollateralStress:
    """Measure equity and cash shortfalls for an unchanged hedge in supplied states.

    The owned asset is assumed admitted into the same collateral pool. Cash is
    fully credited; costs are cumulative quote debits, deducted exactly once.
    Requirements include all caller-supplied margin/add-ons, not an inferred
    maintenance-only rule. The extra-cash result reaches equality, not strict
    safety, and covers only these states before any forced exchange or close.
    """
    for value in (
        owned_asset_quantity,
        short_base_quantity,
        future_entry_price,
        quote_cash,
        required_quote_buffer,
    ):
        _nonnegative(value)
    if future_entry_price == 0 or not isinstance(states, tuple) or not states:
        raise ValueError("positive entry price and nonempty state tuple required")
    labels: set[str] = set()
    values = []
    for state in states:
        if not isinstance(state, CollateralState):
            raise ValueError("CollateralState required")
        if (
            not isinstance(state.label, str)
            or not state.label.strip()
            or state.label in labels
        ):
            raise ValueError("unique nonempty state labels required")
        labels.add(state.label)
        for value in (
            state.spot_mark,
            state.future_mark,
            state.asset_credit_ratio,
            state.total_margin_requirement,
            state.cumulative_cost_debits,
        ):
            _nonnegative(value)
        if (
            state.spot_mark == 0
            or state.future_mark == 0
            or state.asset_credit_ratio > 1
        ):
            raise ValueError("positive marks and credit ratio at most one required")
        asset_value = owned_asset_quantity * state.spot_mark
        short_pnl = short_base_quantity * (future_entry_price - state.future_mark)
        equity = quote_cash + asset_value + short_pnl - state.cumulative_cost_debits
        discount = asset_value * (1 - state.asset_credit_ratio)
        credited = equity - discount
        headroom = credited - state.total_margin_requirement - required_quote_buffer
        values.append(
            CollateralStateValue(
                state.label, short_pnl, equity, credited, discount, headroom
            )
        )
    minimum = min(value.headroom_after_requirement_and_buffer for value in values)
    return CollateralStress(
        states=tuple(values),
        minimum_headroom=minimum,
        extra_fully_credited_cash_to_meet_supplied_states=max(Decimal(0), -minimum),
        strictly_positive_in_every_supplied_state=minimum > 0,
    )
