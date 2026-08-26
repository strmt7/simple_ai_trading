from decimal import Decimal

import pytest

from tools.screen_polymarket_binance_perps_oi_carry import (
    EIGHT_HOURS_MS,
    HOUR_MS,
    AlignedFunding,
    _parse_polymarket_funding_page,
    aggregate_aligned_funding,
    evaluate_aligned_funding,
)


def test_polymarket_funding_page_requires_newest_first() -> None:
    with pytest.raises(ValueError, match="newest-first"):
        _parse_polymarket_funding_page(
            {
                "data": [
                    {"funding_rate": "0.0001", "timestamp": 1},
                    {"funding_rate": "0.0002", "timestamp": 2},
                ],
                "more": False,
            }
        )


def test_alignment_requires_all_eight_hourly_settlements_and_matching_kline() -> None:
    timestamp = 20 * EIGHT_HOURS_MS
    polymarket = [
        (timestamp - offset * HOUR_MS, Decimal("0.00001")) for offset in range(8)
    ]
    aligned = aggregate_aligned_funding(
        polymarket,
        [(timestamp, Decimal("0.00002"))],
        [(timestamp - EIGHT_HOURS_MS, Decimal("100"), Decimal("101"))],
    )

    assert len(aligned) == 1
    assert aligned[0].polymarket_rate_8h == Decimal("0.00008")
    assert aligned[0].difference == Decimal("0.00006")
    assert aligned[0].return_8h == Decimal("0.01")
    assert (
        aggregate_aligned_funding(
            polymarket[:-1],
            [(timestamp, Decimal("0.00002"))],
            [(timestamp - EIGHT_HOURS_MS, Decimal("100"), Decimal("101"))],
        )
        == ()
    )


def test_evaluation_freezes_training_orientation_and_checks_regime_slices() -> None:
    returns = [Decimal("0.01")] * 10 + [Decimal("-0.01")] * 10 + [Decimal("0")] * 10
    rows = tuple(
        AlignedFunding(
            timestamp_ms=index * EIGHT_HOURS_MS,
            polymarket_rate_8h=Decimal("0.001"),
            binance_rate_8h=Decimal("0"),
            return_8h=market_return,
        )
        for index, market_return in enumerate(returns)
    )

    result = evaluate_aligned_funding(
        rows,
        execution_hurdle_bips=Decimal("0"),
        annual_oi_reward_bips=Decimal("0"),
        annual_opportunity_hurdle_bips_per_leg=Decimal("0"),
    )

    assert result["orientation"] == "short_polymarket_long_binance"
    assert result["public_persistence_candidate"] is True
    assert result["roles"]["validation"]["funding"]["sum_bips"] == "70.000"
