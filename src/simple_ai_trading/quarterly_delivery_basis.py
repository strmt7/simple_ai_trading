"""Exact historical delivery/spot mismatch arithmetic for quarterly carry."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Sequence


_BIPS = Decimal("10000")
_MINUTE_MS = 60_000


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
class QuarterlyDeliveryBasisObservation:
    """One settlement price compared with five exact post-delivery spot bars."""

    delivery_time_ms: int
    delivery_price: Decimal
    post_delivery_minimum_low: Decimal
    post_delivery_fifth_close: Decimal
    minimum_low_mismatch_bips: Decimal
    fifth_close_mismatch_bips: Decimal


def quarterly_delivery_basis_observation(
    *,
    delivery_time_ms: int,
    delivery_price: Decimal,
    spot_klines: Sequence[Sequence[object]],
) -> QuarterlyDeliveryBasisObservation:
    """Validate five consecutive one-minute bars and compute mismatch diagnostics."""

    if (
        not isinstance(delivery_time_ms, int)
        or isinstance(delivery_time_ms, bool)
        or delivery_time_ms <= 0
        or delivery_time_ms % _MINUTE_MS != 0
    ):
        raise ValueError("delivery time must be a positive minute-aligned epoch")
    delivery = _decimal(delivery_price, name="delivery price", positive=True)
    rows = tuple(tuple(row) for row in spot_klines)
    if len(rows) != 5:
        raise ValueError("delivery basis requires exactly five spot bars")
    lows: list[Decimal] = []
    closes: list[Decimal] = []
    for index, row in enumerate(rows):
        if len(row) < 7:
            raise ValueError("spot kline has fewer than seven fields")
        open_time = row[0]
        close_time = row[6]
        if (
            not isinstance(open_time, int)
            or isinstance(open_time, bool)
            or not isinstance(close_time, int)
            or isinstance(close_time, bool)
            or open_time != delivery_time_ms + index * _MINUTE_MS
            or close_time != open_time + _MINUTE_MS - 1
        ):
            raise ValueError(
                "spot klines are not the exact consecutive delivery window"
            )
        opened = _decimal(row[1], name="spot open", positive=True)
        high = _decimal(row[2], name="spot high", positive=True)
        low = _decimal(row[3], name="spot low", positive=True)
        close = _decimal(row[4], name="spot close", positive=True)
        if high < max(opened, close) or low > min(opened, close) or low > high:
            raise ValueError("spot kline OHLC ordering is invalid")
        lows.append(low)
        closes.append(close)
    minimum_low = min(lows)
    fifth_close = closes[-1]
    return QuarterlyDeliveryBasisObservation(
        delivery_time_ms=delivery_time_ms,
        delivery_price=delivery,
        post_delivery_minimum_low=minimum_low,
        post_delivery_fifth_close=fifth_close,
        minimum_low_mismatch_bips=(minimum_low - delivery) / delivery * _BIPS,
        fifth_close_mismatch_bips=(fifth_close - delivery) / delivery * _BIPS,
    )


def stressed_after_hurdle_basis_bips(
    *,
    after_hurdle_basis_bips: Decimal,
    worst_observed_mismatch_bips: Decimal,
) -> Decimal:
    """Apply only adverse historical delivery/spot mismatch to a carry result."""

    basis = _decimal(after_hurdle_basis_bips, name="after-hurdle basis")
    mismatch = _decimal(worst_observed_mismatch_bips, name="delivery mismatch")
    return basis + min(mismatch, Decimal("0"))


__all__ = [
    "QuarterlyDeliveryBasisObservation",
    "quarterly_delivery_basis_observation",
    "stressed_after_hurdle_basis_bips",
]
