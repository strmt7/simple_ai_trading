"""Exact arithmetic primitives for conditional Polymarket reward diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Sequence

from .paper_execution import BookLevel


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


@dataclass(frozen=True, slots=True)
class PairedMakerBidDiagnostic:
    """Conditional score and settlement diagnostic for two physical BUY orders."""

    economics: PairedBuyEconomics
    post_quote_yes_ask: Decimal
    post_quote_no_ask: Decimal
    conditional_yes_midpoint: Decimal
    conditional_no_midpoint: Decimal
    conditional_yes_order_score: Decimal
    conditional_no_order_score: Decimal
    conditional_own_minimum_score: Decimal
    conservative_old_aggregate_q_one: Decimal
    conservative_old_aggregate_q_two: Decimal
    conditional_instantaneous_share_lower_bound: Decimal
    conditional_daily_rate_equivalent_lower_bound: Decimal
    conditional_reward_days_to_cover_maximum_orphan_loss: Decimal | None


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


def _book(
    levels: Sequence[BookLevel],
    *,
    side: str,
) -> tuple[BookLevel, ...]:
    normalized = tuple(level.validated() for level in levels)
    reverse = side == "asks"
    if (
        not normalized
        or tuple(sorted(normalized, key=lambda level: level.price, reverse=reverse))
        != normalized
    ):
        raise ValueError(f"reward {side} are empty or not in CLOB response order")
    if len({level.price for level in normalized}) != len(normalized):
        raise ValueError(f"reward {side} contain duplicate prices")
    return normalized


def _score_book(
    levels: Sequence[BookLevel],
    *,
    midpoint: Decimal,
    maximum_spread: Decimal,
) -> Decimal:
    return sum(
        (
            reward_order_score(
                maximum_spread=maximum_spread,
                distance=abs(level.price - midpoint),
                size=level.quantity,
            )
            for level in levels
        ),
        Decimal("0"),
    )


def paired_maker_bid_diagnostic(
    *,
    yes_bids: Sequence[BookLevel],
    yes_asks: Sequence[BookLevel],
    no_bids: Sequence[BookLevel],
    no_asks: Sequence[BookLevel],
    tick_size: Decimal,
    reward_size: Decimal,
    maximum_spread: Decimal,
    daily_reward_rate: Decimal,
) -> PairedMakerBidDiagnostic:
    """Evaluate one-tick-improved paired bids with their mirrored own asks.

    The midpoint and reward values remain conditional because Polymarket does
    not publicly define the size-cutoff-adjusted midpoint algorithm. Public
    books also cannot establish maker grouping, persistence, queue, or payout.
    """

    yes_bid_levels = _book(yes_bids, side="bids")
    yes_ask_levels = _book(yes_asks, side="asks")
    no_bid_levels = _book(no_bids, side="bids")
    no_ask_levels = _book(no_asks, side="asks")
    tick = _decimal(tick_size, name="reward tick size", positive=True)
    size = _decimal(reward_size, name="reward quote size", positive=True)
    maximum = _decimal(
        maximum_spread,
        name="reward maximum spread",
        positive=True,
    )
    daily_rate = _decimal(
        daily_reward_rate,
        name="daily reward rate",
        positive=True,
    )
    yes_quote = yes_bid_levels[-1].price + tick
    no_quote = no_bid_levels[-1].price + tick
    if yes_quote % tick != 0 or no_quote % tick != 0:
        raise ValueError("paired reward quotes are not tick aligned")
    economics = paired_buy_economics(
        yes_price=yes_quote,
        no_price=no_quote,
        quantity=size,
    )
    if economics.combined_price >= _ONE:
        raise ValueError("paired reward bids would self-cross or lack gross surplus")

    # Each physical bid is also a complementary ask. Include those asks when
    # constructing the hypothetical post-quote books, while scoring each own
    # physical order only once.
    post_yes_ask = min(yes_ask_levels[-1].price, _ONE - no_quote)
    post_no_ask = min(no_ask_levels[-1].price, _ONE - yes_quote)
    if yes_quote >= post_yes_ask or no_quote >= post_no_ask:
        raise ValueError("paired reward bids would cross the post-quote book")
    yes_midpoint = (yes_quote + post_yes_ask) / 2
    no_midpoint = (no_quote + post_no_ask) / 2
    yes_score = reward_order_score(
        maximum_spread=maximum,
        distance=abs(yes_quote - yes_midpoint),
        size=size,
    )
    no_score = reward_order_score(
        maximum_spread=maximum,
        distance=abs(no_quote - no_midpoint),
        size=size,
    )
    own_minimum = maker_minimum_score(
        q_one=yes_score,
        q_two=no_score,
        midpoint=yes_midpoint,
    )
    old_q_one = _score_book(
        yes_bid_levels,
        midpoint=yes_midpoint,
        maximum_spread=maximum,
    ) + _score_book(
        no_ask_levels,
        midpoint=no_midpoint,
        maximum_spread=maximum,
    )
    old_q_two = _score_book(
        yes_ask_levels,
        midpoint=yes_midpoint,
        maximum_spread=maximum,
    ) + _score_book(
        no_bid_levels,
        midpoint=no_midpoint,
        maximum_spread=maximum,
    )
    share = conservative_instantaneous_share(
        own_minimum_score=own_minimum,
        old_aggregate_q_one=old_q_one,
        old_aggregate_q_two=old_q_two,
    )
    daily_equivalent = daily_rate * share
    return PairedMakerBidDiagnostic(
        economics=economics,
        post_quote_yes_ask=post_yes_ask,
        post_quote_no_ask=post_no_ask,
        conditional_yes_midpoint=yes_midpoint,
        conditional_no_midpoint=no_midpoint,
        conditional_yes_order_score=yes_score,
        conditional_no_order_score=no_score,
        conditional_own_minimum_score=own_minimum,
        conservative_old_aggregate_q_one=old_q_one,
        conservative_old_aggregate_q_two=old_q_two,
        conditional_instantaneous_share_lower_bound=share,
        conditional_daily_rate_equivalent_lower_bound=daily_equivalent,
        conditional_reward_days_to_cover_maximum_orphan_loss=(
            minimum_reward_days_to_cover(
                maximum_orphan_loss=economics.maximum_orphan_loss,
                daily_reward_bound=daily_equivalent,
            )
        ),
    )


__all__ = [
    "PairedBuyEconomics",
    "PairedMakerBidDiagnostic",
    "conservative_instantaneous_share",
    "maker_minimum_score",
    "minimum_reward_days_to_cover",
    "paired_buy_economics",
    "paired_maker_bid_diagnostic",
    "reward_order_score",
]
