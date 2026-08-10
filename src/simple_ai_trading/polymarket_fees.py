"""Exact Polymarket fee primitives shared by research and live execution."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING


def _finite_decimal(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        qualifier = "positive " if positive else "finite "
        raise ValueError(f"{name} must be a {qualifier}decimal")
    return parsed


@dataclass(frozen=True)
class PolymarketFeeModel:
    """Recorded Polymarket V2 fee curve with conservative precision."""

    enabled: bool
    rate: Decimal
    exponent: int
    taker_only: bool

    def __call__(self, price: Decimal, quantity: Decimal, role: str) -> Decimal:
        if not self.enabled or (role == "maker" and self.taker_only):
            return Decimal("0")
        rate = _finite_decimal(self.rate, name="Polymarket fee rate")
        if rate < 0 or rate > 1:
            raise ValueError("Polymarket fee rate is outside [0, 1]")
        exponent_value = _finite_decimal(
            self.exponent,
            name="Polymarket fee exponent",
            positive=True,
        )
        if exponent_value != exponent_value.to_integral_value():
            raise ValueError("Polymarket fee exponent must be a positive integer")
        exponent = int(exponent_value)
        if price <= 0 or price >= 1:
            raise ValueError("Polymarket match price must lie strictly between 0 and 1")
        curve = (price * (Decimal("1") - price)) ** exponent
        raw = quantity * rate * curve
        if raw < Decimal("0.00001"):
            return Decimal("0")
        return raw.quantize(Decimal("0.00001"), rounding=ROUND_CEILING)


__all__ = ["PolymarketFeeModel"]
