"""Decision boundaries for locally observed snapshot availability."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math


@dataclass(frozen=True)
class SnapshotAvailability:
    received_at_ns: int
    clock_offset_ms: float

    @property
    def decision_not_before_ms(self) -> int:
        """Use the first whole millisecond strictly after the exchange-clock estimate."""
        if type(self.received_at_ns) is not int or self.received_at_ns <= 0:
            raise ValueError("snapshot receipt must be a positive native integer")
        offset = self.clock_offset_ms
        if (
            type(offset) not in (int, float)
            or not math.isfinite(offset)
            or abs(offset) > 60_000
        ):
            raise ValueError("snapshot clock offset is invalid")
        instant = Fraction(self.received_at_ns, 1_000_000) + Fraction.from_float(
            float(offset)
        )
        boundary = instant.numerator // instant.denominator + 1
        if boundary <= 0:
            raise ValueError("snapshot decision boundary must be positive")
        return boundary
