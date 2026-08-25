from __future__ import annotations

from decimal import Decimal

import pytest

from simple_ai_trading.quarterly_delivery_basis import (
    quarterly_delivery_basis_observation,
    stressed_after_hurdle_basis_bips,
)


DELIVERY_MS = 1_700_000_040_000


def _bars() -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            DELIVERY_MS + index * 60_000,
            "100",
            "101",
            str(Decimal("99.9") - Decimal(index) / 100),
            str(Decimal("100.1") + Decimal(index) / 100),
            "1",
            DELIVERY_MS + (index + 1) * 60_000 - 1,
        )
        for index in range(5)
    )


def test_delivery_basis_observation_uses_minimum_low_and_fifth_close() -> None:
    result = quarterly_delivery_basis_observation(
        delivery_time_ms=DELIVERY_MS,
        delivery_price=Decimal("100"),
        spot_klines=_bars(),
    )

    assert result.post_delivery_minimum_low == Decimal("99.86")
    assert result.post_delivery_fifth_close == Decimal("100.14")
    assert result.minimum_low_mismatch_bips == Decimal("-14.0000")
    assert result.fifth_close_mismatch_bips == Decimal("14.0000")


def test_delivery_stress_never_credits_favorable_mismatch() -> None:
    assert stressed_after_hurdle_basis_bips(
        after_hurdle_basis_bips=Decimal("50"),
        worst_observed_mismatch_bips=Decimal("-7"),
    ) == Decimal("43")
    assert stressed_after_hurdle_basis_bips(
        after_hurdle_basis_bips=Decimal("50"),
        worst_observed_mismatch_bips=Decimal("3"),
    ) == Decimal("50")


@pytest.mark.parametrize("value", [True, object(), Decimal("NaN")])
def test_delivery_stress_rejects_non_decimal_inputs(value: object) -> None:
    with pytest.raises(ValueError, match="finite decimal"):
        stressed_after_hurdle_basis_bips(
            after_hurdle_basis_bips=value,  # type: ignore[arg-type]
            worst_observed_mismatch_bips=Decimal("0"),
        )


@pytest.mark.parametrize(
    ("delivery_time", "price", "bars", "message"),
    [
        (0, Decimal("100"), _bars(), "minute-aligned"),
        (DELIVERY_MS + 1, Decimal("100"), _bars(), "minute-aligned"),
        (DELIVERY_MS, Decimal("0"), _bars(), "positive"),
        (DELIVERY_MS, Decimal("100"), _bars()[:-1], "exactly five"),
        (
            DELIVERY_MS,
            Decimal("100"),
            ((DELIVERY_MS, "100"),) + _bars()[1:],
            "seven fields",
        ),
        (
            DELIVERY_MS,
            Decimal("100"),
            ((DELIVERY_MS + 60_000,) + _bars()[0][1:],) + _bars()[1:],
            "exact consecutive",
        ),
        (
            DELIVERY_MS,
            Decimal("100"),
            ((_bars()[0][0], "100", "99", "100", "101", "1", _bars()[0][6]),)
            + _bars()[1:],
            "OHLC",
        ),
    ],
)
def test_delivery_basis_observation_fails_closed(
    delivery_time: int,
    price: Decimal,
    bars: tuple[tuple[object, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        quarterly_delivery_basis_observation(
            delivery_time_ms=delivery_time,
            delivery_price=price,
            spot_klines=bars,
        )
