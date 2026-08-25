from __future__ import annotations

from decimal import Decimal

import pytest

from simple_ai_trading.polymarket_liquidity_rewards import (
    conservative_instantaneous_share,
    maker_minimum_score,
    minimum_reward_days_to_cover,
    paired_buy_economics,
    reward_order_score,
)


def test_reward_order_score_matches_documented_quadratic() -> None:
    assert (
        reward_order_score(
            maximum_spread=Decimal("0.03"),
            distance=Decimal("0.01"),
            size=Decimal("100"),
        )
        == ((Decimal("0.03") - Decimal("0.01")) / Decimal("0.03")) ** 2 * 100
    )
    assert (
        reward_order_score(
            maximum_spread=Decimal("0.03"),
            distance=Decimal("0.03"),
            size=Decimal("100"),
        )
        == 0
    )
    assert (
        reward_order_score(
            maximum_spread=Decimal("0.03"),
            distance=Decimal("0.04"),
            size=Decimal("100"),
        )
        == 0
    )


def test_minimum_score_handles_middle_and_extreme_probabilities() -> None:
    assert maker_minimum_score(
        q_one=Decimal("9"), q_two=Decimal("0"), midpoint=Decimal("0.5")
    ) == Decimal("3")
    assert maker_minimum_score(
        q_one=Decimal("9"), q_two=Decimal("6"), midpoint=Decimal("0.5")
    ) == Decimal("6")
    assert maker_minimum_score(
        q_one=Decimal("9"), q_two=Decimal("0"), midpoint=Decimal("0.09")
    ) == Decimal("0")
    assert maker_minimum_score(
        q_one=Decimal("9"), q_two=Decimal("6"), midpoint=Decimal("0.91")
    ) == Decimal("6")


def test_conservative_share_uses_sum_not_minimum_aggregate() -> None:
    assert conservative_instantaneous_share(
        own_minimum_score=Decimal("10"),
        old_aggregate_q_one=Decimal("30"),
        old_aggregate_q_two=Decimal("50"),
    ) == Decimal("1") / Decimal("9")
    assert (
        conservative_instantaneous_share(
            own_minimum_score=Decimal("0"),
            old_aggregate_q_one=Decimal("0"),
            old_aggregate_q_two=Decimal("0"),
        )
        == 0
    )


def test_paired_buy_economics_reports_both_and_orphan_bounds() -> None:
    result = paired_buy_economics(
        yes_price=Decimal("0.469"),
        no_price=Decimal("0.466"),
        quantity=Decimal("20"),
    )
    assert result.combined_price == Decimal("0.935")
    assert result.both_fill_gross_profit == Decimal("1.300")
    assert result.yes_only_maximum_loss == Decimal("9.380")
    assert result.no_only_maximum_loss == Decimal("9.320")
    assert result.maximum_orphan_loss == Decimal("9.380")


def test_reward_payback_handles_zero_loss_and_zero_reward() -> None:
    assert minimum_reward_days_to_cover(
        maximum_orphan_loss=Decimal("9"), daily_reward_bound=Decimal("3")
    ) == Decimal("3")
    assert (
        minimum_reward_days_to_cover(
            maximum_orphan_loss=Decimal("9"), daily_reward_bound=Decimal("0")
        )
        is None
    )
    assert (
        minimum_reward_days_to_cover(
            maximum_orphan_loss=Decimal("0"), daily_reward_bound=Decimal("0")
        )
        == 0
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: reward_order_score(
                maximum_spread=Decimal("0"),
                distance=Decimal("0"),
                size=Decimal("1"),
            ),
            "maximum spread",
        ),
        (
            lambda: reward_order_score(
                maximum_spread=Decimal("1"),
                distance=Decimal("-1"),
                size=Decimal("1"),
            ),
            "order distance",
        ),
        (
            lambda: maker_minimum_score(
                q_one=Decimal("0"),
                q_two=Decimal("0"),
                midpoint=Decimal("1.1"),
            ),
            "inside",
        ),
        (
            lambda: paired_buy_economics(
                yes_price=Decimal("1"),
                no_price=Decimal("0.5"),
                quantity=Decimal("1"),
            ),
            "inside",
        ),
        (
            lambda: minimum_reward_days_to_cover(
                maximum_orphan_loss=Decimal("-1"),
                daily_reward_bound=Decimal("1"),
            ),
            "orphan loss",
        ),
        (
            lambda: reward_order_score(
                maximum_spread=True,  # type: ignore[arg-type]
                distance=Decimal("0"),
                size=Decimal("1"),
            ),
            "finite decimal",
        ),
        (
            lambda: reward_order_score(
                maximum_spread=None,  # type: ignore[arg-type]
                distance=Decimal("0"),
                size=Decimal("1"),
            ),
            "finite decimal",
        ),
        (
            lambda: reward_order_score(
                maximum_spread=Decimal("NaN"),
                distance=Decimal("0"),
                size=Decimal("1"),
            ),
            "finite decimal",
        ),
    ],
)
def test_invalid_reward_inputs_fail_closed(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]
