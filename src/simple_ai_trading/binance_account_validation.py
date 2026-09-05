"""Validate account quantity evidence before it can establish reconciled exposure."""

from __future__ import annotations

import math
from collections.abc import Mapping


def _quantity(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def account_quantity_rejection(account: object, market_type: str) -> str | None:
    """Reject ambiguous identity or quantity rows, including duplicates and hedge conflicts."""
    if not isinstance(account, Mapping):
        return "account_payload_not_mapping"
    if market_type not in {"spot", "futures"}:
        return "account_market_type_invalid"
    futures = market_type == "futures"
    field = "positions" if futures else "balances"
    rows = account.get(field)
    if not isinstance(rows, list):
        return (
            "futures_positions_missing_or_not_list"
            if futures
            else "spot_balances_missing_or_not_list"
        )
    identities: set[tuple[str, str]] = set()
    modes: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return "account_quantity_row_not_mapping"
        identity = row.get("symbol" if futures else "asset")
        if (
            not isinstance(identity, str)
            or not identity
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in identity)
        ):
            return "account_quantity_identity_invalid"
        identity = identity.upper()
        side = row.get("positionSide", "BOTH") if futures else "SPOT"
        if not isinstance(side, str) or side not in (
            {"BOTH", "LONG", "SHORT"} if futures else {"SPOT"}
        ):
            return "futures_position_side_invalid"
        key = (identity, side)
        if key in identities:
            return "account_quantity_identity_duplicate"
        identities.add(key)
        modes.setdefault(identity, set()).add(side)
        if "BOTH" in modes[identity] and len(modes[identity]) != 1:
            return "futures_position_modes_conflict"
        if futures:
            amount = _quantity(row.get("positionAmt"))
            if amount is None:
                return "futures_position_quantity_invalid"
            if (side == "LONG" and amount < 0) or (side == "SHORT" and amount > 0):
                return "futures_position_side_quantity_conflict"
        else:
            free = _quantity(row.get("free"))
            locked = _quantity(row.get("locked"))
            if (
                free is None
                or locked is None
                or free < 0
                or locked < 0
                or not math.isfinite(free + locked)
            ):
                return "spot_balance_quantity_invalid"
    return None
