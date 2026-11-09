"""Shared native identity and level validation for capture and normalization."""

from __future__ import annotations

import math
from typing import Mapping


def positive_native_integer(value: object, field: str) -> int:
    """Keep native clock and sequence identities exact instead of coercing them."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive native integer")
    return value


def nonnegative_native_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative native integer")
    return value


def depth_update_ids(data: Mapping[str, object]) -> tuple[int, int, int]:
    first = positive_native_integer(data.get("U"), "U")
    final = positive_native_integer(data.get("u"), "u")
    previous = positive_native_integer(data.get("pu"), "pu")
    if final < first:
        raise ValueError("invalid depth update identifiers")
    return first, final, previous


def validate_price_levels(levels: object) -> int:
    if not isinstance(levels, list):
        raise ValueError("depth levels must be a list")
    for level in levels:
        if not isinstance(level, list) or len(level) < 2:
            raise ValueError("invalid depth level")
        if isinstance(level[0], bool) or isinstance(level[1], bool):
            raise ValueError("depth economics must not be booleans")
        price = float(level[0])
        quantity = float(level[1])
        if (
            not math.isfinite(price)
            or price <= 0.0
            or not math.isfinite(quantity)
            or quantity < 0.0
        ):
            raise ValueError("non-finite or negative depth level")
    return len(levels)
