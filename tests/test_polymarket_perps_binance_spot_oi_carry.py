from decimal import Decimal

import pytest

from tools.adjudicate_polymarket_perps_binance_spot_oi_carry import (
    HOUR_MS,
    _normalize_hour,
    role_economics,
)


def test_normalize_hour_enforces_frozen_tolerance() -> None:
    assert _normalize_hour(10 * HOUR_MS + 172, 500) == (10 * HOUR_MS, 172)
    with pytest.raises(ValueError, match="exceeds"):
        _normalize_hour(10 * HOUR_MS + 501, 500)


def test_missing_funding_hours_are_zero_not_dropped_from_capital_time() -> None:
    result = role_economics(
        {HOUR_MS: Decimal("0.001")},
        start_ms=0,
        end_ms=2 * HOUR_MS,
        annual_reward_bips=Decimal("0"),
        annual_opportunity_bips_per_leg=Decimal("0"),
    )

    assert result["expected_funding_hours"] == 2
    assert result["observed_funding_hours"] == 1
    assert result["missing_funding_hours_valued_at_zero"] == 1
    assert result["funding_bips"] == "10.000"
