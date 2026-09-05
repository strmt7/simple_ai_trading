"""Lifecycle coverage tied to immutable exchange order IDs, not cancel-ID styling."""

from __future__ import annotations

from decimal import Decimal
from typing import Any


def lifecycle_coverage(
    rows: list[dict[str, Any]],
    cash: dict[str, dict[str, Any]],
    symbols: tuple[str, ...],
) -> dict[str, dict[str, bool]]:
    """Require owned terminal state and acknowledged cancellation for each asset."""
    intents = {
        row["params"]["newClientOrderId"]: row["params"]
        for row in rows
        if row["kind"] == "order_intent"
    }
    ids: dict[str, int] = {}
    successful_cancels: set[tuple[str, int]] = set()
    pending = None
    for row in rows:
        if row["kind"] == "order_identity":
            cid = row["original_client_id"]
            if cid not in intents or intents[cid]["symbol"] != row["symbol"]:
                raise ValueError("unowned order identity")
            if cid in ids and ids[cid] != row["orderId"]:
                raise ValueError("ambiguous order identity")
            ids[cid] = row["orderId"]
        elif row["kind"] == "http_intent":
            if pending is not None:
                raise ValueError("overlapping requests")
            pending = row
        elif row["kind"] in {"http_completed", "http_failure"}:
            if pending is None or (pending["method"], pending["path"]) != (
                row["method"],
                row["path"],
            ):
                raise ValueError("request receipt mismatch")
            if (
                row["kind"] == "http_completed"
                and row.get("http_status") == 200
                and pending["method"] == "DELETE"
                and pending["path"] == "/api/v3/order"
            ):
                successful_cancels.add(
                    (pending["params"]["symbol"], pending["params"]["orderId"])
                )
            pending = None
    if pending is not None:
        raise ValueError("request still unresolved")
    result = {}
    for symbol in symbols:
        owned = {cid: x for cid, x in intents.items() if x["symbol"] == symbol}
        maker_ids = {
            ids[cid]
            for cid, x in owned.items()
            if x["type"] == "LIMIT_MAKER" and cid in ids
        }
        observations = [
            x["order"]
            for x in rows
            if x["kind"] == "order_observation" and x["order"]["symbol"] == symbol
        ]
        final_checks = [
            x for x in rows if x["kind"] == "open_order_check" and x["symbol"] == symbol
        ]
        result[symbol] = {
            "resting_order_observed": any(
                x["orderId"] in maker_ids and x["status"] == "NEW" for x in observations
            ),
            "cancellation_observed": any(
                x["orderId"] in maker_ids
                and x["status"] == "CANCELED"
                and (symbol, x["orderId"]) in successful_cancels
                for x in observations
            ),
            "buy_fill_observed": any(
                cid in cash
                and x["type"] == "LIMIT"
                and x["side"] == "BUY"
                and Decimal(cash[cid]["base_delta"]) > 0
                for cid, x in owned.items()
            ),
            "sell_fill_observed": any(
                cid in cash
                and x["side"] == "SELL"
                and Decimal(cash[cid]["base_delta"]) < 0
                for cid, x in owned.items()
            ),
            "owned_flat": bool(owned)
            and all(cid in cash for cid in owned)
            and sum(
                (Decimal(cash[cid]["base_delta"]) for cid in owned if cid in cash),
                Decimal(0),
            )
            == 0,
            "no_open_orders": bool(final_checks) and final_checks[-1]["total"] == 0,
        }
    return result
