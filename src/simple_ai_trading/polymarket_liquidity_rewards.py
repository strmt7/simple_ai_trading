"""Exact arithmetic primitives for conditional Polymarket reward diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


_ONE = Decimal("1")


def _decimal(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    if positive and parsed <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


@dataclass(frozen=True, slots=True)
class PairedBuyEconomics:
    """Settlement bounds for buying equal YES and NO share quantities."""

    quantity: Decimal
    yes_price: Decimal
    no_price: Decimal
    combined_price: Decimal
    both_fill_gross_profit: Decimal
    yes_only_maximum_loss: Decimal
    no_only_maximum_loss: Decimal
    maximum_orphan_loss: Decimal


def reward_order_score(
    *,
    maximum_spread: Decimal,
    distance: Decimal,
    size: Decimal,
    multiplier: Decimal = _ONE,
) -> Decimal:
    """Apply the documented quadratic order score in consistent price units."""

    maximum = _decimal(maximum_spread, name="maximum spread", positive=True)
    spread = _decimal(distance, name="order distance", nonnegative=True)
    quantity = _decimal(size, name="order size", positive=True)
    factor = _decimal(multiplier, name="score multiplier", nonnegative=True)
    if spread >= maximum:
        return Decimal("0")
    return ((maximum - spread) / maximum) ** 2 * factor * quantity


def maker_minimum_score(
    *,
    q_one: Decimal,
    q_two: Decimal,
    midpoint: Decimal,
    single_side_divisor: Decimal = Decimal("3"),
) -> Decimal:
    """Apply Polymarket's documented two-sided/single-sided score adjustment."""

    first = _decimal(q_one, name="Q one", nonnegative=True)
    second = _decimal(q_two, name="Q two", nonnegative=True)
    middle = _decimal(midpoint, name="market midpoint")
    divisor = _decimal(
        single_side_divisor,
        name="single-side divisor",
        positive=True,
    )
    if middle < 0 or middle > _ONE:
        raise ValueError("market midpoint must be inside [0, 1]")
    paired = min(first, second)
    if Decimal("0.10") <= middle <= Decimal("0.90"):
        return max(paired, first / divisor, second / divisor)
    return paired


def conservative_instantaneous_share(
    *,
    own_minimum_score: Decimal,
    old_aggregate_q_one: Decimal,
    old_aggregate_q_two: Decimal,
) -> Decimal:
    """Bound one sample's share using sum(Qmin_old) <= sum(Q1_old + Q2_old).

    This is conservative only when the supplied old Q aggregates are complete
    upper bounds under the same scoring midpoint. It is not a payout forecast:
    epoch persistence and the second normalization remain separate evidence.
    """

    own = _decimal(
        own_minimum_score,
        name="own minimum score",
        nonnegative=True,
    )
    old_one = _decimal(
        old_aggregate_q_one,
        name="old aggregate Q one",
        nonnegative=True,
    )
    old_two = _decimal(
        old_aggregate_q_two,
        name="old aggregate Q two",
        nonnegative=True,
    )
    if own == 0:
        return Decimal("0")
    return own / (own + old_one + old_two)


def paired_buy_economics(
    *,
    yes_price: Decimal,
    no_price: Decimal,
    quantity: Decimal,
) -> PairedBuyEconomics:
    """Price an equal-size YES+NO buy and worst settlement loss if orphaned."""

    yes = _decimal(yes_price, name="YES price", positive=True)
    no = _decimal(no_price, name="NO price", positive=True)
    size = _decimal(quantity, name="paired quantity", positive=True)
    if yes >= _ONE or no >= _ONE:
        raise ValueError("paired prices must be inside (0, 1)")
    combined = yes + no
    yes_loss = yes * size
    no_loss = no * size
    return PairedBuyEconomics(
        quantity=size,
        yes_price=yes,
        no_price=no,
        combined_price=combined,
        both_fill_gross_profit=(_ONE - combined) * size,
        yes_only_maximum_loss=yes_loss,
        no_only_maximum_loss=no_loss,
        maximum_orphan_loss=max(yes_loss, no_loss),
    )


def minimum_reward_days_to_cover(
    *,
    maximum_orphan_loss: Decimal,
    daily_reward_bound: Decimal,
) -> Decimal | None:
    """Return idealized reward days needed to cover a stated orphan-loss bound."""

    loss = _decimal(
        maximum_orphan_loss,
        name="maximum orphan loss",
        nonnegative=True,
    )
    reward = _decimal(
        daily_reward_bound,
        name="daily reward bound",
        nonnegative=True,
    )
    if loss == 0:
        return Decimal("0")
    if reward == 0:
        return None
    return loss / reward


__all__ = [
    "PairedBuyEconomics",
    "conservative_instantaneous_share",
    "maker_minimum_score",
    "minimum_reward_days_to_cover",
    "paired_buy_economics",
    "reward_order_score",
]
