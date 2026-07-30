"""Deterministic Stop service for bot-owned Polymarket exposure only."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from decimal import Decimal
import time
from typing import Callable

from .polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveCoordinator,
    PolymarketLiveOrderLedger,
)
from .polymarket_runtime_control import PolymarketRuntimeControl

Clock = Callable[[], float]
Sleeper = Callable[[float], None]


def _reconciliation_payload(result: object) -> dict[str, object]:
    payload = asdict(result)
    for key in (
        "foreign_order_ids",
        "foreign_position_token_ids",
        "missing_position_token_ids",
        "blocking_intent_ids",
        "errors",
    ):
        payload[key] = list(payload[key])
    return payload


def _owned_inventory_payload(
    ledger: PolymarketLiveOrderLedger,
) -> list[dict[str, object]]:
    return [
        {
            "market_id": item.market_id,
            "token_id": item.token_id,
            "quantity": format(item.quantity, "f"),
            "provisional": item.provisional,
        }
        for item in ledger.owned_inventory()
    ]


def _stop_owned_polymarket_exposure_unlocked(
    *,
    coordinator: PolymarketLiveCoordinator,
    ledger: PolymarketLiveOrderLedger,
    started: float,
    deadline: float,
    monotonic: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> tuple[dict[str, object], int]:
    initial_inventory = ledger.owned_inventory()
    initial_quantity = sum(
        (item.quantity for item in initial_inventory),
        Decimal("0"),
    )
    cancellation = coordinator.cancel_owned_open_orders()
    submitted_intent_ids: list[str] = []
    reason = ""
    final = coordinator.reconcile()
    while True:
        inventory = ledger.owned_inventory()
        open_order_ids = ledger.open_owned_order_ids()
        if not inventory and not open_order_ids:
            break
        if monotonic() >= deadline:
            reason = "stop_timeout"
            break
        if open_order_ids:
            sleep(min(0.25, max(0.0, deadline - monotonic())))
            final = coordinator.reconcile()
            continue
        try:
            closes = coordinator.submit_owned_close_orders(
                maximum_book_age_ms=1_500,
            )
        except PolymarketLiveBlocked as exc:
            reason = str(exc)
            break
        submitted_intent_ids.extend(record.intent.intent_id for record in closes)
        if not closes:
            reason = "owned_inventory_has_no_executable_close"
            break
        sleep(min(0.25, max(0.0, deadline - monotonic())))
        final = coordinator.reconcile()
    final = coordinator.reconcile()
    remaining = ledger.owned_inventory()
    remaining_quantity = sum(
        (item.quantity for item in remaining),
        Decimal("0"),
    )
    completed = not remaining and not ledger.open_owned_order_ids()
    if not completed and not reason:
        reason = "owned_exposure_remains"
    return (
        {
            "schema_version": "polymarket-live-stop-result-v1",
            "action": "stop",
            "venue": "polymarket",
            "symbol": "BTC",
            "completed": completed,
            "cancelled_order_ids": list(cancellation.cancelled_order_ids),
            "failed_order_ids": list(cancellation.failed_order_ids),
            "submitted_close_intent_ids": submitted_intent_ids,
            "initial_owned_quantity": format(initial_quantity, "f"),
            "remaining_owned_quantity": format(remaining_quantity, "f"),
            "remaining_owned_inventory": _owned_inventory_payload(ledger),
            "reason": reason,
            "foreign_state_untouched": True,
            "elapsed_seconds": round(monotonic() - started, 3),
            "reconciliation": _reconciliation_payload(final),
        },
        0 if completed else 2,
    )


def stop_owned_polymarket_exposure(
    *,
    coordinator: PolymarketLiveCoordinator,
    ledger: PolymarketLiveOrderLedger,
    timeout_seconds: float,
    monotonic: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> tuple[dict[str, object], int]:
    """Serialize, cancel, and close only exact bot-owned Polymarket exposure."""

    timeout = float(timeout_seconds)
    if not 1 <= timeout <= 300:
        raise ValueError("stop-timeout-seconds must lie in [1, 300]")
    started = monotonic()
    path = getattr(ledger, "path", None)
    guard = (
        PolymarketRuntimeControl(path).close_guard(timeout_seconds=timeout)
        if path is not None
        else nullcontext()
    )
    with guard:
        return _stop_owned_polymarket_exposure_unlocked(
            coordinator=coordinator,
            ledger=ledger,
            started=started,
            deadline=started + timeout,
            monotonic=monotonic,
            sleep=sleep,
        )


__all__ = ["stop_owned_polymarket_exposure"]
