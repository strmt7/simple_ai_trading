from dataclasses import replace
from fractions import Fraction

import pytest

from simple_ai_trading.funding_cash import LinearFundingSettlement
from tools.adjudicate_funding_archive_clock import compare_clock


def _reference(times=(10, 20)):
    return {
        "symbol": "BTCUSDT",
        "start_time_ms": 0,
        "end_time_ms": 100,
        "rows": [
            {"calc_time": t, "funding_rate_float_repr": "0.0001", "interval_hours": 8}
            for t in times
        ],
    }


def _events(times=(10, 20)):
    return tuple(
        LinearFundingSettlement(
            "BTCUSDT", "USDT", t, Fraction("0.0001"), Fraction(100), "a" * 64
        )
        for t in times
    )


def test_exact_small_population_join_does_not_qualify_historical_cash_or_entitlement():
    result = compare_clock(_reference(), _events())
    assert result["exact_timestamp_and_rate_join"]
    assert result["same_order_rate_equality_diagnostic"]
    assert all(x["archive_minus_native_ms"] == 0 for x in result["clock_diagnostic"])
    assert not result["complete_historical_mark_coverage"]
    assert not result["own_payment_entitlement_qualified"]
    assert not result["warehouse_float_rates_are_native_decimal_proof"]


def test_two_millisecond_difference_rejects_join_without_rounding_or_alias():
    result = compare_clock(_reference((12, 22)), _events())
    assert not result["exact_timestamp_and_rate_join"]
    assert result["same_order_rate_equality_diagnostic"]
    assert [x["archive_minus_native_ms"] for x in result["clock_diagnostic"]] == [2, 2]
    assert all(
        x["diagnostic_only_not_an_admitted_mapping"] for x in result["clock_diagnostic"]
    )
    assert not result["time_rounding_interpolation_or_candle_fallback"]


def test_missing_events_and_rate_disagreement_do_not_create_ordinal_mapping():
    mismatch = compare_clock(_reference(), _events((10,)))
    rate_mismatch = compare_clock(
        _reference(),
        (_events()[0], replace(_events()[1], rate=Fraction("0.0001000000000001"))),
    )
    for result in (mismatch, rate_mismatch):
        assert not result["exact_timestamp_and_rate_join"]
        assert not result["same_order_rate_equality_diagnostic"]
        assert result["clock_diagnostic"] == []


@pytest.mark.parametrize("times", [(), (20, 10), (10, 10), (101,)])
def test_reference_or_native_invalid_time_population_fails(times):
    with pytest.raises(ValueError):
        compare_clock(_reference(times), _events())
    with pytest.raises(ValueError):
        compare_clock(_reference(), _events(times))


def test_wrong_product_invalid_reference_schema_and_bounds_fail():
    with pytest.raises(ValueError):
        compare_clock(_reference(), (replace(_events()[0], symbol="ETHUSDT"),))
    for field, value in (
        ("calc_time", True),
        ("interval_hours", 9),
        ("funding_rate_float_repr", 0.0001),
    ):
        reference = _reference()
        reference["rows"][0][field] = value
        with pytest.raises(ValueError):
            compare_clock(reference, _events())
    reference = _reference()
    reference["start_time_ms"] = True
    with pytest.raises(ValueError):
        compare_clock(reference, _events())
