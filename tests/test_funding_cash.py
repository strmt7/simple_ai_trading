from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import json

import pytest

from simple_ai_trading.funding_cash import (
    MAX_HISTORY_BYTES,
    MAX_HISTORY_ROWS,
    FundingEntitlement,
    LinearFundingSettlement,
    funding_cash_bounds_for_events,
    parse_binance_usdm_funding_history,
    reconcile_funding_event_population,
)


def _row(**overrides: object) -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "fundingTime": 10,
        "fundingRate": "0.005",
        "markPrice": "102",
        **overrides,
    }


def _parse(raw: bytes) -> tuple[LinearFundingSettlement, ...]:
    return parse_binance_usdm_funding_history(
        raw, expected_sha256=hashlib.sha256(raw).hexdigest(), expected_symbol="BTCUSDT"
    )


def _events(*rows: dict[str, object]) -> tuple[LinearFundingSettlement, ...]:
    return _parse(json.dumps(list(rows)).encode())


def _cash(events, entitlements, *, quantity=Fraction(1), price=Fraction(100)):
    return funding_cash_bounds_for_events(
        events,
        entitlements,
        expected_symbol="BTCUSDT",
        signed_base_quantity=quantity,
        entry_price=price,
    )


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("rate", ["0.005", "-0.005", "0"])
def test_exact_fixed_base_cash_uses_settlement_mark_and_signed_quantity(side, rate):
    events = _events(_row(fundingRate=rate))
    result = _cash(events, [FundingEntitlement.HELD], quantity=Fraction(side, 3))
    reference_cash = -Fraction(side, 3) * Fraction(102) * Fraction(rate)
    assert result.cash_lower == result.cash_upper == reference_cash
    assert (
        result.entry_relative_lower_bps
        == result.entry_relative_upper_bps
        == (-side * Fraction(102, 100) * Fraction(rate) * 10_000)
    )
    assert result.cash_amount_exact and result.uncertain_entitlement_events == 0
    assert result.supplied_events == 1 and result.payment_asset == "USDT"
    assert result.source_body_sha256 == (events[0].source_body_sha256,)


def test_mark_scaling_changes_the_thin_margin_sign_for_both_sides():
    events = _events(_row())
    gross = (Fraction("100.505") / 100 - 1) * 10_000
    long_cash = _cash(events, [FundingEntitlement.HELD])
    short_cash = _cash(events, [FundingEntitlement.HELD], quantity=Fraction(-1))
    assert gross - 50 > 0 > gross + long_cash.entry_relative_lower_bps
    assert -gross + 50 < 0 < -gross + short_cash.entry_relative_lower_bps


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("rate", ["0.005", "-0.005", "0"])
def test_unknown_entitlement_never_credits_income_as_certain(side, rate):
    events = _events(_row(fundingRate=rate))
    result = _cash(events, [FundingEntitlement.UNKNOWN], quantity=Fraction(side))
    exact_payment = -side * Fraction(102) * Fraction(rate)
    assert result.cash_lower == min(0, exact_payment)
    assert result.cash_upper == max(0, exact_payment)
    assert result.uncertain_entitlement_events == 1
    assert result.cash_amount_exact == (rate == "0")


def test_known_nonholding_is_zero_and_opposite_rates_do_not_hide_uncertainty():
    events = _events(_row(), _row(fundingTime=20, fundingRate="-0.005"))
    outside = _cash(events, [FundingEntitlement.NOT_HELD] * 2)
    exact = _cash(events, [FundingEntitlement.HELD] * 2)
    uncertain = _cash(events, [FundingEntitlement.UNKNOWN] * 2)
    assert outside.cash_amount_exact and outside.cash_lower == 0
    assert exact.cash_amount_exact and exact.cash_lower == 0
    assert uncertain.cash_lower == Fraction("-0.51")
    assert uncertain.cash_upper == Fraction("0.51")
    assert not uncertain.cash_amount_exact


@pytest.mark.parametrize("side", [1, -1])
def test_opposite_rates_at_different_marks_do_not_cancel_fixed_base_cash(side):
    events = _events(
        _row(), _row(fundingTime=20, fundingRate="-0.005", markPrice="100")
    )
    result = _cash(events, [FundingEntitlement.HELD] * 2, quantity=Fraction(side * 3))
    assert sum(event.rate for event in events) == 0
    assert result.cash_lower == result.cash_upper == -side * Fraction("0.03")
    assert result.entry_relative_lower_bps == -side


def test_decimal_precision_and_missing_rate_type_are_preserved():
    events = _events(_row(markPrice="123456.123456789", fundingRate="0.00000001"))
    assert events[0].rate_type is None
    assert events[0].settlement_mark == Fraction("123456.123456789")
    regular = _events(_row(rateType="Regular"))
    assert regular[0].rate_type == "Regular"
    assert _parse(b"[]") == ()  # An empty page is not a completeness certificate.
    assert _cash((), ()).supplied_events == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"symbol": "ETHUSDT"},
        {"symbol": "BTCUSD_PERP"},
        {"fundingTime": True},
        {"fundingTime": 10.0},
        {"fundingTime": -1},
        {"fundingRate": 0.005},
        {"fundingRate": "NaN"},
        {"fundingRate": "1e1000000"},
        {"fundingRate": "0" * 129},
        {"markPrice": "0"},
        {"markPrice": "-1"},
        {"markPrice": "Infinity"},
        {"rateType": "Special"},
        {"rateType": "Unknown"},
        {"unexpected_field": "not admitted"},
    ],
)
def test_parser_rejects_unqualified_schema_numbers_product_and_rate_type(overrides):
    with pytest.raises(ValueError):
        _events(_row(**overrides))


@pytest.mark.parametrize(
    "raw",
    [
        b'[{"symbol":"BTCUSDT","symbol":"ETHUSDT"}]',
        b'[{"value":NaN}]',
        b'[{"value":Infinity}]',
        b"{}",
        b"[null]",
        b"[{}]",
        b"[",
        b"\xff",
    ],
)
def test_parser_rejects_ambiguous_and_malformed_responses(raw):
    with pytest.raises(ValueError):
        _parse(raw)


@pytest.mark.parametrize("second_time", [10, 9])
def test_parser_rejects_duplicate_or_reversed_events(second_time):
    with pytest.raises(ValueError, match="duplicated or not strictly increasing"):
        _events(_row(), _row(fundingTime=second_time))


def test_parser_rejects_changed_body_bad_hash_oversize_and_unsupported_symbol():
    raw = json.dumps([_row()]).encode()
    for body, sha, symbol in (
        (raw, "a" * 64, "BTCUSDT"),
        (raw, "invalid", "BTCUSDT"),
        (raw, hashlib.sha256(raw).hexdigest(), "BTCUSDC"),
        (b" " * (MAX_HISTORY_BYTES + 1), "a" * 64, "BTCUSDT"),
        ("not bytes", "a" * 64, "BTCUSDT"),
    ):
        with pytest.raises(ValueError, match="source binding"):
            parse_binance_usdm_funding_history(
                body, expected_sha256=sha, expected_symbol=symbol
            )
    with pytest.raises(ValueError, match="bounded row array"):
        _events(*[_row(fundingTime=i) for i in range(MAX_HISTORY_ROWS + 1)])


@pytest.mark.parametrize(
    "overrides",
    [
        {"payment_asset": "USDC"},
        {"rate": 0.005},
        {"settlement_mark": 102.0},
        {"source_body_sha256": "invalid"},
    ],
)
def test_direct_event_construction_enforces_cash_units(overrides):
    with pytest.raises(ValueError, match="settlement contract"):
        replace(_events(_row())[0], **overrides)


@pytest.mark.parametrize("quantity,price", [(0, 100), (1.0, 100), (1, 0), (1, -1)])
def test_cash_rejects_invalid_or_inexact_position_inputs(quantity, price):
    q = quantity if isinstance(quantity, float) else Fraction(quantity)
    with pytest.raises(ValueError, match="position contract"):
        _cash(
            _events(_row()),
            [FundingEntitlement.HELD],
            quantity=q,
            price=Fraction(price),
        )


def test_cash_rejects_population_mismatch_duplicate_product_or_fake_entitlement():
    event = _events(_row())[0]
    with pytest.raises(ValueError, match="position contract"):
        _cash([event], [])
    for events, entitlements in (
        ([event, event], [FundingEntitlement.HELD] * 2),
        ([replace(event, symbol="ETHUSDT")], [FundingEntitlement.HELD]),
        ([event], ["held"]),
        ([None], [FundingEntitlement.HELD]),
    ):
        with pytest.raises(ValueError, match="population or entitlement"):
            _cash(events, entitlements)


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
def test_each_supported_linear_product_preserves_explicit_currency_and_symbol(symbol):
    raw = json.dumps([_row(symbol=symbol)]).encode()
    events = parse_binance_usdm_funding_history(
        raw, expected_sha256=hashlib.sha256(raw).hexdigest(), expected_symbol=symbol
    )
    assert events[0].symbol == symbol and events[0].payment_asset == "USDT"
    result = funding_cash_bounds_for_events(
        events,
        [FundingEntitlement.HELD],
        expected_symbol=symbol,
        signed_base_quantity=Fraction(1),
        entry_price=Fraction(100),
    )
    assert result.entry_relative_lower_bps == -51


def test_population_reconciliation_matches_exact_ordered_time_and_rate():
    events = _events(_row(), _row(fundingTime=20, fundingRate="-0.001"))
    expected = [(10, Fraction("0.005")), (20, Fraction("-0.001"))]
    assert (
        reconcile_funding_event_population(events, expected, expected_symbol="BTCUSDT")
        == events
    )
    assert reconcile_funding_event_population((), (), expected_symbol="BTCUSDT") == ()


@pytest.mark.parametrize(
    "expected",
    [
        [],
        [(10, Fraction("0.005")), (20, Fraction("0.005"))],
        [(11, Fraction("0.005"))],
        [(10, Fraction("0.00500000000000000001"))],
        [(10, 0.005)],
        [(True, Fraction("0.005"))],
        [(10,)],
        [[10, Fraction("0.005")]],
        [None],
    ],
)
def test_population_reconciliation_rejects_missing_extra_shifted_or_inexact_events(
    expected,
):
    with pytest.raises(ValueError, match="population"):
        reconcile_funding_event_population(
            _events(_row()), expected, expected_symbol="BTCUSDT"
        )


def test_population_reconciliation_rejects_duplicate_product_and_wrong_input_type():
    event = _events(_row())[0]
    for events, expected, symbol in (
        ([event, event], [(10, event.rate)] * 2, "BTCUSDT"),
        ([event], [(10, event.rate)], "ETHUSDT"),
        ([event], [(10, event.rate)], "BTCUSDC"),
        ([None], [(10, event.rate)], "BTCUSDT"),
    ):
        with pytest.raises(ValueError, match="population"):
            reconcile_funding_event_population(events, expected, expected_symbol=symbol)
    with pytest.raises(ValueError, match="position contract"):
        funding_cash_bounds_for_events(
            (),
            (),
            expected_symbol="BTCUSDC",
            signed_base_quantity=Fraction(1),
            entry_price=Fraction(100),
        )
    with pytest.raises(ValueError, match="position contract"):
        _cash((), (), price=100.0)
