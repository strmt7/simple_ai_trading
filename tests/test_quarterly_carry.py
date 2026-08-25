from __future__ import annotations

from decimal import Decimal

import pytest

from simple_ai_trading.paper_execution import BookLevel
from simple_ai_trading.quarterly_carry import (
    screen_quarterly_cash_and_carry,
    walk_depth,
)


def _level(price: str, quantity: str) -> BookLevel:
    return BookLevel(price=Decimal(price), quantity=Decimal(quantity))


def test_walk_depth_prices_exact_quantity_without_extrapolation() -> None:
    fill = walk_depth(
        (_level("100", "1"), _level("101", "2")),
        quantity=Decimal("2"),
        descending=False,
    )
    assert fill is not None
    assert fill.quote_value == Decimal("201")
    assert fill.price == Decimal("100.5")
    assert (
        walk_depth(
            (_level("105", "0.5"),),
            quantity=Decimal("1"),
            descending=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("levels", "descending", "message"),
    [
        ((_level("101", "1"), _level("100", "1")), False, "best-to-worst"),
        ((_level("100", "1"), _level("100", "2")), False, "duplicated"),
    ],
)
def test_walk_depth_rejects_invalid_book_ordering(
    levels: tuple[BookLevel, ...], descending: bool, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        walk_depth(levels, quantity=Decimal("1"), descending=descending)


def test_screen_quarterly_cash_and_carry_reports_exact_hurdle() -> None:
    result = screen_quarterly_cash_and_carry(
        spot_asks=(_level("100", "1"), _level("101", "2")),
        future_bids=(_level("105", "1.5"), _level("104", "2")),
        quantity=Decimal("2"),
        capture_time_ms=1_000,
        delivery_time_ms=31_557_601_000,
        all_in_cost_hurdle_bips=Decimal("30"),
    )
    assert result is not None
    assert result.spot_buy.quote_value == Decimal("201")
    assert result.future_sale.quote_value == Decimal("209.5")
    assert result.gross_profit_quote == Decimal("8.5")
    assert result.gross_basis_bips == Decimal("8.5") / Decimal("201") * 10_000
    assert result.after_hurdle_profit_quote == Decimal("7.897")
    assert result.after_hurdle_basis_bips == result.gross_basis_bips - 30
    assert result.gross_simple_annualized_bips == result.gross_basis_bips
    assert result.after_hurdle_simple_annualized_bips == result.after_hurdle_basis_bips
    assert result.gross_positive is True
    assert result.after_hurdle_positive is True


def test_screen_returns_none_when_either_leg_lacks_depth() -> None:
    assert (
        screen_quarterly_cash_and_carry(
            spot_asks=(_level("100", "1"),),
            future_bids=(_level("101", "2"),),
            quantity=Decimal("2"),
            capture_time_ms=1,
            delivery_time_ms=2,
            all_in_cost_hurdle_bips=Decimal("0"),
        )
        is None
    )
    assert (
        screen_quarterly_cash_and_carry(
            spot_asks=(_level("100", "2"),),
            future_bids=(_level("101", "1"),),
            quantity=Decimal("2"),
            capture_time_ms=1,
            delivery_time_ms=2,
            all_in_cost_hurdle_bips=Decimal("0"),
        )
        is None
    )


@pytest.mark.parametrize(
    ("capture", "delivery", "hurdle", "message"),
    [
        (True, 2, Decimal("0"), "integer milliseconds"),
        (1.5, 2, Decimal("0"), "integer milliseconds"),
        (1.0, 2, Decimal("0"), "integer milliseconds"),
        (0, 2, Decimal("0"), "after a positive capture"),
        (2, 2, Decimal("0"), "after a positive capture"),
        (1, 2, Decimal("-1"), "outside"),
        (1, 2, Decimal("10000"), "outside"),
    ],
)
def test_screen_rejects_invalid_contract(
    capture: object, delivery: object, hurdle: Decimal, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        screen_quarterly_cash_and_carry(
            spot_asks=(_level("100", "1"),),
            future_bids=(_level("101", "1"),),
            quantity=Decimal("1"),
            capture_time_ms=capture,  # type: ignore[arg-type]
            delivery_time_ms=delivery,  # type: ignore[arg-type]
            all_in_cost_hurdle_bips=hurdle,
        )


@pytest.mark.parametrize("quantity", [True, None, "nan", "0"])
def test_walk_depth_rejects_invalid_quantity(quantity: object) -> None:
    with pytest.raises(ValueError, match="carry quantity"):
        walk_depth(
            (_level("100", "1"),),
            quantity=quantity,  # type: ignore[arg-type]
            descending=False,
        )
