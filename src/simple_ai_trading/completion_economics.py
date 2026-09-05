"""Conditional inventory-completion economics, never venue or edge qualification.

Inputs must already share a payoff identity, quote unit and valuation horizon.
Prices are net executable proceeds for the specified residual sizes, not marks.
Probabilities are supplied scenario weights, not inferred fill probabilities.
This research helper cannot establish ownership, depth, causality or execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


def _finite(value: Decimal, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    return value


def _nonnegative(value: Decimal, name: str) -> Decimal:
    if _finite(value, name) < 0:
        raise ValueError(f"{name} must be a finite nonnegative Decimal")
    return value


@dataclass(frozen=True, slots=True)
class CompletionState:
    """One joint fill/exit scenario; all quantities are net of share-denominated fees.

    Acquisition cash includes quote fees; additional cost excludes those fees
    and covers other incremental costs once. Pair value and residual net bids
    already reflect their realization costs and timing at the common horizon.
    Cash outlays and the comparator must be valued at that same horizon.
    Opposite overfills remain explicit residual inventory, never silently clipped.
    """

    label: str
    probability: Decimal
    opposite_net_shares: Decimal
    acquisition_cash: Decimal
    additional_cost: Decimal
    pair_net_realization: Decimal | None
    original_residual_net_bid: Decimal | None
    opposite_residual_net_bid: Decimal | None


@dataclass(frozen=True, slots=True)
class CompletionStateValue:
    label: str
    probability: Decimal
    matched_shares: Decimal
    original_residual_shares: Decimal
    opposite_residual_shares: Decimal
    completion_net_value: Decimal
    liquidation_net_value: Decimal
    incremental_value: Decimal
    historical_completion_pnl: Decimal | None
    historical_liquidation_pnl: Decimal | None


@dataclass(frozen=True, slots=True)
class CompletionComparison:
    states: tuple[CompletionStateValue, ...]
    expected_incremental_value: Decimal
    worst_state_incremental_value: Decimal
    probability_underperforming_liquidation: Decimal
    expected_completion_net_value: Decimal
    positive_in_every_supplied_state: bool
    # Always false: arithmetic cannot qualify an edge or a probability model.
    qualified_edge: bool = field(default=False, init=False)


def _value(quantity: Decimal, price: Decimal | None, name: str) -> Decimal:
    if price is None:
        if quantity != 0:
            raise ValueError(f"missing {name} for nonzero quantity")
        return Decimal(0)
    return quantity * _finite(price, name)


def compare_completion(
    *,
    original_net_shares: Decimal,
    liquidation_net_proceeds: Decimal | None,
    states: tuple[CompletionState, ...],
    historical_acquisition_cost: Decimal | None = None,
) -> CompletionComparison:
    """Compare joint completion outcomes with a common feasible sale comparator.

    The caller must supply an already horizon-adjusted, full-quantity executable
    sale comparator. No current market data is fetched or invented. Missing
    valuation for a nonzero leg rejects the calculation rather than pricing it
    at zero. History is optional accounting context and cannot change ranking.
    """
    quantity = _nonnegative(original_net_shares, "original_net_shares")
    if quantity == 0:
        raise ValueError("preexisting inventory must be positive")
    if liquidation_net_proceeds is None:
        raise ValueError("missing executable liquidation comparator")
    baseline = _finite(liquidation_net_proceeds, "liquidation_net_proceeds")
    if historical_acquisition_cost is not None:
        _nonnegative(historical_acquisition_cost, "historical_acquisition_cost")
    if not states or len({x.label for x in states}) != len(states):
        raise ValueError("nonempty uniquely labeled joint states required")
    if any(not x.label.strip() for x in states):
        raise ValueError("state label must be nonblank")
    probabilities = [_nonnegative(x.probability, "probability") for x in states]
    if any(p == 0 for p in probabilities) or sum(probabilities, Decimal(0)) != 1:
        raise ValueError(
            "strictly positive joint probabilities must sum exactly to one"
        )
    values = []
    for state in states:
        opposite = _nonnegative(state.opposite_net_shares, "opposite_net_shares")
        acquisition = _nonnegative(state.acquisition_cash, "acquisition_cash")
        cost = _nonnegative(state.additional_cost, "additional_cost")
        matched = min(quantity, opposite)
        original_residual = quantity - matched
        opposite_residual = opposite - matched
        value = (
            _value(matched, state.pair_net_realization, "pair_net_realization")
            + _value(
                original_residual,
                state.original_residual_net_bid,
                "original_residual_net_bid",
            )
            + _value(
                opposite_residual,
                state.opposite_residual_net_bid,
                "opposite_residual_net_bid",
            )
            - acquisition
            - cost
        )
        values.append(
            CompletionStateValue(
                label=state.label,
                probability=state.probability,
                matched_shares=matched,
                original_residual_shares=original_residual,
                opposite_residual_shares=opposite_residual,
                completion_net_value=value,
                liquidation_net_value=baseline,
                incremental_value=value - baseline,
                historical_completion_pnl=None
                if historical_acquisition_cost is None
                else value - historical_acquisition_cost,
                historical_liquidation_pnl=None
                if historical_acquisition_cost is None
                else baseline - historical_acquisition_cost,
            )
        )
    return CompletionComparison(
        states=tuple(values),
        expected_incremental_value=sum(
            (x.probability * x.incremental_value for x in values), Decimal(0)
        ),
        worst_state_incremental_value=min(x.incremental_value for x in values),
        probability_underperforming_liquidation=sum(
            (x.probability for x in values if x.incremental_value < 0), Decimal(0)
        ),
        expected_completion_net_value=sum(
            (x.probability * x.completion_net_value for x in values), Decimal(0)
        ),
        positive_in_every_supplied_state=all(x.incremental_value > 0 for x in values),
    )
