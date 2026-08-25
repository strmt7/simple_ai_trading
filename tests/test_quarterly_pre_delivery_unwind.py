from __future__ import annotations

from decimal import Decimal

import pytest

from simple_ai_trading.quarterly_pre_delivery_unwind import (
    pre_delivery_unwind_observation,
    stressed_pre_delivery_basis_bips,
)


DELIVERY_MS = 1_800_000_000_000
START_MS = DELIVERY_MS - 60 * 60_000


def _bars(*, count: int, basis: Decimal = Decimal("1")) -> list[list[object]]:
    rows: list[list[object]] = []
    for index in range(count):
        mid = Decimal("100") + Decimal(index) / 10 + basis
        rows.append(
            [
                START_MS + index * 60_000,
                str(mid),
                str(mid + Decimal("0.2")),
                str(mid - Decimal("0.2")),
                str(mid + Decimal("0.1")),
                "1",
                START_MS + (index + 1) * 60_000 - 1,
            ]
        )
    return rows


def test_pre_delivery_unwind_uses_exact_primary_bar_and_adverse_extremes() -> None:
    spot = _bars(count=70, basis=Decimal("0"))
    futures = _bars(count=60, basis=Decimal("1"))

    result = pre_delivery_unwind_observation(
        scheduled_delivery_ms=DELIVERY_MS,
        spot_klines=spot,
        futures_klines=futures,
        horizons_minutes=(60, 10, 1),
    )

    primary = result.horizon_observations[1]
    assert primary.horizon_minutes == 10
    assert primary.bar_open_time_ms == DELIVERY_MS - 10 * 60_000
    assert primary.spot_low == Decimal("104.8")
    assert primary.future_high == Decimal("106.2")
    assert primary.adverse_exit_basis_bips == (
        Decimal("1.4") / Decimal("104.8") * Decimal("10000")
    )
    assert primary.close_basis_bips == (
        Decimal("1") / Decimal("105.1") * Decimal("10000")
    )
    assert result.futures_last_bar_close_ms == DELIVERY_MS - 1


def test_favorable_extremes_are_not_credited_as_negative_stress() -> None:
    spot = _bars(count=70, basis=Decimal("2"))
    futures = _bars(count=60, basis=Decimal("0"))

    result = pre_delivery_unwind_observation(
        scheduled_delivery_ms=DELIVERY_MS,
        spot_klines=spot,
        futures_klines=futures,
        horizons_minutes=(10,),
    )

    assert result.horizon_observations[0].adverse_exit_basis_bips == Decimal("0")
    assert stressed_pre_delivery_basis_bips(
        after_hurdle_basis_bips=Decimal("50"),
        adverse_exit_basis_bips=Decimal("7"),
    ) == Decimal("43")


@pytest.mark.parametrize(
    ("delivery", "spot", "futures", "horizons", "message"),
    [
        (0, _bars(count=70), _bars(count=60), (10,), "minute-aligned"),
        (DELIVERY_MS, _bars(count=70), _bars(count=60), (), "unique integers"),
        (DELIVERY_MS, _bars(count=70), _bars(count=60), (10, 10), "unique"),
        (DELIVERY_MS, _bars(count=69), _bars(count=60), (10,), "70 bars"),
        (DELIVERY_MS, _bars(count=70), _bars(count=59), (10,), "60 bars"),
        (
            DELIVERY_MS,
            [[START_MS, "1"]] + _bars(count=70)[1:],
            _bars(count=60),
            (10,),
            "seven fields",
        ),
        (
            DELIVERY_MS,
            [[START_MS + 1, *_bars(count=70)[0][1:]]] + _bars(count=70)[1:],
            _bars(count=60),
            (10,),
            "exact consecutive",
        ),
        (
            DELIVERY_MS,
            [[START_MS, "100", "99", "100", "101", "1", START_MS + 59_999]]
            + _bars(count=70)[1:],
            _bars(count=60),
            (10,),
            "OHLC",
        ),
        (
            DELIVERY_MS,
            _bars(count=70),
            _bars(count=59) + [_bars(count=60)[-1][:-1] + [DELIVERY_MS - 2]],
            (10,),
            "exact consecutive",
        ),
    ],
)
def test_pre_delivery_unwind_fails_closed(
    delivery: int,
    spot: list[list[object]],
    futures: list[list[object]],
    horizons: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pre_delivery_unwind_observation(
            scheduled_delivery_ms=delivery,
            spot_klines=spot,
            futures_klines=futures,
            horizons_minutes=horizons,
        )


@pytest.mark.parametrize("value", [True, object(), Decimal("NaN")])
def test_stressed_basis_rejects_invalid_decimals(value: object) -> None:
    with pytest.raises(ValueError, match="finite decimal"):
        stressed_pre_delivery_basis_bips(
            after_hurdle_basis_bips=value,  # type: ignore[arg-type]
            adverse_exit_basis_bips=Decimal("0"),
        )


def test_stressed_basis_rejects_negative_stress() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        stressed_pre_delivery_basis_bips(
            after_hurdle_basis_bips=Decimal("10"),
            adverse_exit_basis_bips=Decimal("-1"),
        )
