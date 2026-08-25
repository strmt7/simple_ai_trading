"""Exact adverse historical basis arithmetic for scheduled quarterly unwinds."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Sequence


_BIPS = Decimal("10000")
_MINUTE_MS = 60_000
_PRE_DELIVERY_MINUTES = 60
_SPOT_WINDOW_MINUTES = 70


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError(f"{name} must be a positive finite decimal")
    return parsed


@dataclass(frozen=True, slots=True)
class ValidatedKline:
    """One validated one-minute trade bar."""

    open_time_ms: int
    opened: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    close_time_ms: int


@dataclass(frozen=True, slots=True)
class PreDeliveryHorizonObservation:
    """One paired spot/future residual basis diagnostic at a fixed horizon."""

    horizon_minutes: int
    bar_open_time_ms: int
    spot_low: Decimal
    spot_close: Decimal
    future_high: Decimal
    future_close: Decimal
    adverse_exit_basis_bips: Decimal
    close_basis_bips: Decimal


@dataclass(frozen=True, slots=True)
class PreDeliveryUnwindObservation:
    """Validated cutoff and fixed-horizon diagnostics for one dated future."""

    scheduled_delivery_ms: int
    futures_last_bar_open_ms: int
    futures_last_bar_close_ms: int
    horizon_observations: tuple[PreDeliveryHorizonObservation, ...]


def _validated_window(
    rows: Sequence[Sequence[object]],
    *,
    name: str,
    expected_start_ms: int,
    expected_count: int,
) -> tuple[ValidatedKline, ...]:
    values = tuple(tuple(row) for row in rows)
    if len(values) != expected_count:
        raise ValueError(f"{name} requires exactly {expected_count} bars")
    parsed: list[ValidatedKline] = []
    for index, row in enumerate(values):
        if len(row) < 7:
            raise ValueError(f"{name} kline has fewer than seven fields")
        open_time = row[0]
        close_time = row[6]
        expected_open = expected_start_ms + index * _MINUTE_MS
        if (
            not isinstance(open_time, int)
            or isinstance(open_time, bool)
            or not isinstance(close_time, int)
            or isinstance(close_time, bool)
            or open_time != expected_open
            or close_time != expected_open + _MINUTE_MS - 1
        ):
            raise ValueError(f"{name} klines are not the exact consecutive window")
        opened = _decimal(row[1], name=f"{name} open", positive=True)
        high = _decimal(row[2], name=f"{name} high", positive=True)
        low = _decimal(row[3], name=f"{name} low", positive=True)
        close = _decimal(row[4], name=f"{name} close", positive=True)
        if high < max(opened, close) or low > min(opened, close) or low > high:
            raise ValueError(f"{name} kline OHLC ordering is invalid")
        parsed.append(
            ValidatedKline(
                open_time_ms=open_time,
                opened=opened,
                high=high,
                low=low,
                close=close,
                close_time_ms=close_time,
            )
        )
    return tuple(parsed)


def pre_delivery_unwind_observation(
    *,
    scheduled_delivery_ms: int,
    spot_klines: Sequence[Sequence[object]],
    futures_klines: Sequence[Sequence[object]],
    horizons_minutes: Sequence[int],
) -> PreDeliveryUnwindObservation:
    """Validate one normal cutoff and calculate fixed-horizon adverse basis."""

    if (
        not isinstance(scheduled_delivery_ms, int)
        or isinstance(scheduled_delivery_ms, bool)
        or scheduled_delivery_ms <= 0
        or scheduled_delivery_ms % _MINUTE_MS != 0
    ):
        raise ValueError("scheduled delivery must be a positive minute-aligned epoch")
    horizons = tuple(horizons_minutes)
    if (
        not horizons
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            or value > _PRE_DELIVERY_MINUTES
            for value in horizons
        )
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("unwind horizons must be unique integers from one to sixty")
    start_ms = scheduled_delivery_ms - _PRE_DELIVERY_MINUTES * _MINUTE_MS
    spot = _validated_window(
        spot_klines,
        name="spot",
        expected_start_ms=start_ms,
        expected_count=_SPOT_WINDOW_MINUTES,
    )
    futures = _validated_window(
        futures_klines,
        name="futures",
        expected_start_ms=start_ms,
        expected_count=_PRE_DELIVERY_MINUTES,
    )

    observations: list[PreDeliveryHorizonObservation] = []
    for horizon in horizons:
        index = _PRE_DELIVERY_MINUTES - horizon
        spot_bar = spot[index]
        future_bar = futures[index]
        adverse = max(
            (future_bar.high - spot_bar.low) / spot_bar.low * _BIPS,
            Decimal("0"),
        )
        observations.append(
            PreDeliveryHorizonObservation(
                horizon_minutes=horizon,
                bar_open_time_ms=future_bar.open_time_ms,
                spot_low=spot_bar.low,
                spot_close=spot_bar.close,
                future_high=future_bar.high,
                future_close=future_bar.close,
                adverse_exit_basis_bips=adverse,
                close_basis_bips=(
                    (future_bar.close - spot_bar.close) / spot_bar.close * _BIPS
                ),
            )
        )
    return PreDeliveryUnwindObservation(
        scheduled_delivery_ms=scheduled_delivery_ms,
        futures_last_bar_open_ms=futures[-1].open_time_ms,
        futures_last_bar_close_ms=futures[-1].close_time_ms,
        horizon_observations=tuple(observations),
    )


def stressed_pre_delivery_basis_bips(
    *,
    after_hurdle_basis_bips: Decimal,
    adverse_exit_basis_bips: Decimal,
) -> Decimal:
    """Subtract a nonnegative pre-delivery residual-basis stress."""

    basis = _decimal(after_hurdle_basis_bips, name="after-hurdle basis")
    stress = _decimal(adverse_exit_basis_bips, name="adverse exit basis")
    if stress < 0:
        raise ValueError("adverse exit basis must be nonnegative")
    return basis - stress


__all__ = [
    "PreDeliveryHorizonObservation",
    "PreDeliveryUnwindObservation",
    "ValidatedKline",
    "pre_delivery_unwind_observation",
    "stressed_pre_delivery_basis_bips",
]
