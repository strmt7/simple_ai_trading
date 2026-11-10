"""Exact separation of long-only covers of an exhaustive one-winner partition.

Costs are supplied nonnegative linear acquisition costs in the payout unit.
State weights are mathematical certificates, never estimated probabilities.
None means an unavailable instrument, not a free leg. Venue rules, quotes,
depth, nonlinear fees, execution and actual profitability remain caller gates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

_ZERO = Fraction(0)
_ONE = Fraction(1)
MAX_PARTITION_STATES = 4096


def _costs(
    yes: Sequence[Fraction | None], no: Sequence[Fraction | None]
) -> tuple[tuple[Fraction | None, ...], tuple[Fraction | None, ...]]:
    if not 2 <= len(yes) <= MAX_PARTITION_STATES or len(no) != len(yes):
        raise ValueError("partition requires matching costs for 2..4096 states")
    result = tuple(yes), tuple(no)
    if any(
        value is not None and (not isinstance(value, Fraction) or value < 0)
        for side in result
        for value in side
    ):
        raise ValueError("partition costs must be nonnegative Fractions or None")
    return result


def validate_partition_cover_certificate(
    yes_costs: Sequence[Fraction | None],
    no_costs: Sequence[Fraction | None],
    state_weights: Sequence[Fraction],
) -> None:
    """Prove every available leg costs at least its weighted supplied payoff.

    Then any nonnegative portfolio costs at least its weighted payout, which is
    at least its minimum state payout. This rules out a strictly positive nominal
    floor surplus under these costs; it does not rule out expected-value trades.
    """
    yes, no = _costs(yes_costs, no_costs)
    weights = tuple(state_weights)
    if (
        len(weights) != len(yes)
        or any(not isinstance(value, Fraction) or value < 0 for value in weights)
        or sum(weights, _ZERO) != _ONE
        or any(
            (a is not None and weight > a) or (b is not None and _ONE - weight > b)
            for a, b, weight in zip(yes, no, weights)
        )
    ):
        raise ValueError("partition state-price certificate is invalid")


@dataclass(frozen=True, slots=True)
class PartitionCover:
    yes_quantities: tuple[Fraction, ...]
    no_quantities: tuple[Fraction, ...]
    acquisition_cost: Fraction
    minimum_payout: Fraction

    @property
    def floor_surplus(self) -> Fraction:
        return self.minimum_payout - self.acquisition_cost


def evaluate_partition_cover(
    yes_costs: Sequence[Fraction | None],
    no_costs: Sequence[Fraction | None],
    yes_quantities: Sequence[Fraction],
    no_quantities: Sequence[Fraction],
) -> PartitionCover:
    """Evaluate arbitrary unequal quantities in O(n), without a payoff matrix."""
    yes, no = _costs(yes_costs, no_costs)
    quantities = tuple(yes_quantities), tuple(no_quantities)
    if any(
        len(side) != len(yes)
        or any(not isinstance(value, Fraction) or value < 0 for value in side)
        for side in quantities
    ):
        raise ValueError("partition quantities must be matching nonnegative Fractions")
    cost = _ZERO
    for prices, amounts in zip((yes, no), quantities):
        for price, amount in zip(prices, amounts):
            if price is None and amount:
                raise ValueError("partition portfolio uses an unavailable leg")
            if price is not None:
                cost += price * amount
    yes_amounts, no_amounts = quantities
    no_total = sum(no_amounts, _ZERO)
    floor = min(y + no_total - n for y, n in zip(yes_amounts, no_amounts))
    return PartitionCover(yes_amounts, no_amounts, cost, floor)


@dataclass(frozen=True, slots=True)
class PartitionCoverSeparation:
    state_weights: tuple[Fraction, ...] | None
    strict_cover: PartitionCover | None


def separate_partition_cover(
    yes_costs: Sequence[Fraction | None], no_costs: Sequence[Fraction | None]
) -> PartitionCoverSeparation:
    """Return an exact no-strict-floor-profit certificate or a strict witness.

    Feasible certificate weights satisfy max(0, 1-NO_i) <= w_i <= min(1, YES_i)
    and sum(w)=1. Missing sides impose no constraint. The only infeasibility
    cases yield a binary straddle, all-YES, or a NO-subset cover respectively.
    This proves completeness for every nonnegative mixed/unequal portfolio at
    supplied linear costs, without subset enumeration, fitting or numerical LP.
    """
    yes, no = _costs(yes_costs, no_costs)
    lower = tuple(max(_ZERO, _ONE - b) if b is not None else _ZERO for b in no)
    upper = tuple(min(_ONE, a) if a is not None else _ONE for a in yes)
    yes_amounts = [_ZERO] * len(yes)
    no_amounts = [_ZERO] * len(yes)
    conflict = next(
        (i for i, (lo, hi) in enumerate(zip(lower, upper)) if lo > hi), None
    )
    if conflict is not None:
        yes_amounts[conflict] = no_amounts[conflict] = _ONE
    elif sum(upper, _ZERO) < _ONE:
        yes_amounts = [_ONE] * len(yes)
    elif sum(lower, _ZERO) > _ONE:
        no_amounts = [_ONE if lo > 0 else _ZERO for lo in lower]
    else:
        weights = list(lower)
        remaining = _ONE - sum(weights, _ZERO)
        for index, (lo, hi) in enumerate(zip(lower, upper)):
            extra = min(remaining, hi - lo)
            weights[index] += extra
            remaining -= extra
        validate_partition_cover_certificate(yes, no, weights)
        return PartitionCoverSeparation(tuple(weights), None)
    cover = evaluate_partition_cover(yes, no, yes_amounts, no_amounts)
    if cover.floor_surplus <= 0:
        raise RuntimeError("partition separation failed to construct a strict witness")
    return PartitionCoverSeparation(None, cover)
