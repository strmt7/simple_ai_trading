"""CLI controls for the independent BTC Polymarket live execution boundary."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
import sys
import time
from typing import Mapping

from .polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveCoordinator,
    PolymarketLiveOrderLedger,
    PolymarketLiveRiskLimits,
)
from .polymarket_live_runtime import (
    PolymarketAuthenticatedUserStream,
    PolymarketLiveRuntimeGuard,
    PolymarketReconciliationService,
    PolymarketUserStreamConsumer,
)
from .polymarket_live_settlement import (
    OfficialPolymarketUnifiedRedemptionVenue,
    PolymarketGaslessCredentials,
    PolymarketRedemptionCoordinator,
    PolymarketSettlementService,
)
from .polymarket_live_v2 import (
    OfficialPolymarketV2Venue,
    PolymarketLiveCredentials,
)


DEFAULT_POLYMARKET_LIVE_LEDGER = Path("data/polymarket/live-ownership.sqlite3")
_ACTIONS = (
    "status",
    "preflight",
    "reconcile",
    "supervise",
    "cancel-owned",
    "recover-redemptions",
    "redeem",
)
_RISK_LIMITS = {
    "conservative": (Decimal("10"), Decimal("20")),
    "regular": (Decimal("25"), Decimal("50")),
    "aggressive": (Decimal("50"), Decimal("100")),
}


def register_polymarket_live_command(
    subparsers: argparse._SubParsersAction,  # noqa: SLF001 - argparse has no public type
) -> None:
    parser = subparsers.add_parser(
        "polymarket-live",
        help="operate the independent BTC Polymarket live safety boundary",
        description=(
            "Inspect or supervise the independent BTC Polymarket CLOB V2 boundary. "
            "It never shares Binance orders, balances, positions, or risk state. "
            "No action opens exposure because no Polymarket model has live authority."
        ),
    )
    parser.add_argument("--action", choices=_ACTIONS, default="status")
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_POLYMARKET_LIVE_LEDGER),
        help="durable hash-bound Polymarket ownership ledger",
    )
    parser.add_argument(
        "--risk-level",
        choices=tuple(_RISK_LIMITS),
        default="conservative",
        help="hard Polymarket execution ceiling profile; conservative is default",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="supervision duration; zero runs until interrupted",
    )
    parser.add_argument(
        "--reconciliation-seconds",
        type=float,
        default=5.0,
        help="authenticated reconciliation interval",
    )
    parser.add_argument(
        "--condition-id",
        default=None,
        help="optional exact condition ID for one redemption",
    )
    parser.add_argument(
        "--confirm-redemption",
        action="store_true",
        help="confirm one settlement transaction after a fresh read-only preflight",
    )
    parser.add_argument(
        "--automatic-redemption",
        action="store_true",
        help="allow supervision to redeem proven resolved inventory one condition at a time",
    )
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=command_polymarket_live)


def _risk_limits(profile: str) -> PolymarketLiveRiskLimits:
    maximum_quote, maximum_tokens = _RISK_LIMITS[str(profile)]
    return PolymarketLiveRiskLimits(
        maximum_order_quote=maximum_quote,
        maximum_token_quantity=maximum_tokens,
    )


def _local_status(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "schema_version": "polymarket-live-operator-status-v1",
            "venue": "polymarket",
            "symbol": "BTC",
            "ledger": str(path),
            "ledger_exists": False,
            "open_owned_order_count": 0,
            "owned_position_count": 0,
            "unresolved_redemption_count": 0,
            "can_open": False,
            "reason": "authenticated_preflight_required",
        }
    ledger = PolymarketLiveOrderLedger(path)
    records = ledger.records()
    inventory = ledger.owned_inventory()
    redemptions = ledger.redemption_records()
    unresolved = tuple(
        record
        for record in redemptions
        if record.state in {"prepared", "submitting", "submitted", "unknown"}
    )
    return {
        "schema_version": "polymarket-live-operator-status-v1",
        "venue": "polymarket",
        "symbol": "BTC",
        "ledger": str(path),
        "ledger_exists": True,
        "order_count": len(records),
        "open_owned_order_count": len(ledger.open_owned_order_ids()),
        "owned_position_count": len(inventory),
        "owned_position_quantity": format(
            sum((item.quantity for item in inventory), Decimal("0")),
            "f",
        ),
        "unresolved_redemption_count": len(unresolved),
        "can_open": False,
        "reason": "authenticated_preflight_and_promoted_model_required",
    }


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


def _credentials_and_venue() -> tuple[
    PolymarketLiveCredentials,
    OfficialPolymarketV2Venue,
]:
    credentials = PolymarketLiveCredentials.from_environment()
    return credentials, OfficialPolymarketV2Venue(credentials)


def _gasless_credentials(
    credentials: PolymarketLiveCredentials,
) -> PolymarketGaslessCredentials | None:
    if credentials.signature_type == 0:
        return None
    return PolymarketGaslessCredentials.from_environment()


def _settlement_coordinator(
    *,
    credentials: PolymarketLiveCredentials,
    account: OfficialPolymarketV2Venue,
    ledger: PolymarketLiveOrderLedger,
) -> tuple[
    PolymarketRedemptionCoordinator,
    OfficialPolymarketUnifiedRedemptionVenue,
]:
    settlement_venue = OfficialPolymarketUnifiedRedemptionVenue(
        credentials,
        gasless_credentials=_gasless_credentials(credentials),
    )
    return (
        PolymarketRedemptionCoordinator(account, settlement_venue, ledger),
        settlement_venue,
    )


async def _supervise(
    *,
    credentials: PolymarketLiveCredentials,
    venue: OfficialPolymarketV2Venue,
    ledger: PolymarketLiveOrderLedger,
    risk_limits: PolymarketLiveRiskLimits,
    duration_seconds: float,
    reconciliation_seconds: float,
    automatic_redemption: bool,
) -> dict[str, object]:
    duration = float(duration_seconds)
    if not 0 <= duration <= 31_536_000:
        raise ValueError("duration-seconds must lie in [0, 31536000]")
    interval = float(reconciliation_seconds)
    if not 1 <= interval <= 60:
        raise ValueError("reconciliation-seconds must lie in [1, 60]")
    guard = PolymarketLiveRuntimeGuard(
        maximum_reconciliation_age_ms=max(5_000, int(interval * 3_000)),
    )
    coordinator = PolymarketLiveCoordinator(
        venue,
        ledger,
        risk_limits=risk_limits,
        runtime_authority=guard,
    )
    initial = await asyncio.to_thread(coordinator.preflight)
    if not initial.can_close:
        raise PolymarketLiveBlocked(
            f"supervision preflight failed: {initial.errors}"
        )
    consumer = PolymarketUserStreamConsumer(ledger, guard)
    stream = PolymarketAuthenticatedUserStream(
        credentials,
        consumer,
        markets=tuple(
            sorted({record.intent.market_id for record in ledger.records()})
        ),
    )
    reconciliation = PolymarketReconciliationService(
        coordinator,
        guard,
        interval_seconds=interval,
    )
    redemption_coordinator, settlement_venue = _settlement_coordinator(
        credentials=credentials,
        account=venue,
        ledger=ledger,
    )
    settlement = PolymarketSettlementService(
        redemption_coordinator,
        guard,
        automatic_redemption_enabled=automatic_redemption,
        interval_seconds=max(5.0, interval),
    )
    stop = asyncio.Event()
    tasks = (
        asyncio.create_task(stream.run(stop)),
        asyncio.create_task(reconciliation.run(stop)),
        asyncio.create_task(settlement.run(stop)),
    )
    started = time.monotonic()
    try:
        if duration:
            await asyncio.sleep(duration)
        else:
            await asyncio.Future()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        guard.mark_stopped()
        settlement_venue.close()
    final = await asyncio.to_thread(coordinator.reconcile)
    snapshot = guard.snapshot()
    return {
        "schema_version": "polymarket-live-supervision-v1",
        "venue": "polymarket",
        "symbol": "BTC",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "automatic_redemption": automatic_redemption,
        "reconciliation": _reconciliation_payload(final),
        "runtime": asdict(snapshot),
        "opened_exposure": False,
    }


def _render(payload: Mapping[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(dict(payload), indent=2, sort_keys=True, default=str))
        return
    action = str(payload.get("action") or "status")
    detail = " ".join(
        f"{key}={value}"
        for key, value in payload.items()
        if key not in {"schema_version", "action", "reconciliation", "runtime"}
    )
    print(f"polymarket-live: action={action}" + (f" {detail}" if detail else ""))
    reconciliation = payload.get("reconciliation")
    if isinstance(reconciliation, Mapping):
        print(
            "reconciliation: "
            f"ok={reconciliation.get('ok')} "
            f"can_open={reconciliation.get('can_open')} "
            f"can_close={reconciliation.get('can_close')} "
            f"errors={reconciliation.get('errors')}"
        )


def command_polymarket_live(args: argparse.Namespace) -> int:
    action = str(args.action)
    ledger_path = Path(args.ledger)
    try:
        if action == "status":
            payload = {"action": action, **_local_status(ledger_path)}
            _render(payload, as_json=bool(args.json))
            return 0

        credentials, venue = _credentials_and_venue()
        ledger = PolymarketLiveOrderLedger(ledger_path)
        risk_limits = _risk_limits(args.risk_level)
        coordinator = PolymarketLiveCoordinator(
            venue,
            ledger,
            risk_limits=risk_limits,
        )
        payload: dict[str, object]
        try:
            if action == "preflight":
                result = coordinator.preflight()
                payload = {
                    "schema_version": "polymarket-live-operator-result-v1",
                    "action": action,
                    "venue": "polymarket",
                    "symbol": "BTC",
                    "reconciliation": _reconciliation_payload(result),
                }
                status = 0 if result.ok else 2
            elif action == "reconcile":
                result = coordinator.reconcile()
                payload = {
                    "schema_version": "polymarket-live-operator-result-v1",
                    "action": action,
                    "venue": "polymarket",
                    "symbol": "BTC",
                    "reconciliation": _reconciliation_payload(result),
                }
                status = 0 if result.ok else 2
            elif action == "cancel-owned":
                gate = coordinator.preflight()
                if not gate.can_close:
                    raise PolymarketLiveBlocked(
                        f"owned cancellation blocked by reconciliation: {gate.errors}"
                    )
                result = coordinator.cancel_owned_open_orders()
                final = coordinator.reconcile()
                payload = {
                    "schema_version": "polymarket-live-operator-result-v1",
                    "action": action,
                    "venue": "polymarket",
                    "symbol": "BTC",
                    "cancelled_order_ids": list(result.cancelled_order_ids),
                    "failed_order_ids": list(result.failed_order_ids),
                    "reconciliation": _reconciliation_payload(final),
                }
                status = 0 if final.ok else 2
            elif action in {"recover-redemptions", "redeem"}:
                if action == "redeem" and not bool(args.confirm_redemption):
                    raise PolymarketLiveBlocked(
                        "redemption requires --confirm-redemption"
                    )
                redemption, settlement_venue = _settlement_coordinator(
                    credentials=credentials,
                    account=venue,
                    ledger=ledger,
                )
                try:
                    if action == "recover-redemptions":
                        records = redemption.recover_incomplete()
                        payload = {
                            "schema_version": "polymarket-live-operator-result-v1",
                            "action": action,
                            "venue": "polymarket",
                            "symbol": "BTC",
                            "redemptions": [asdict(record) for record in records],
                        }
                        status = (
                            0
                            if all(
                                record.state not in {"submitted", "unknown"}
                                for record in records
                            )
                            else 2
                        )
                    else:
                        record = redemption.redeem_next_ready(
                            condition_id=args.condition_id,
                        )
                        payload = {
                            "schema_version": "polymarket-live-operator-result-v1",
                            "action": action,
                            "venue": "polymarket",
                            "symbol": "BTC",
                            "redemption": (
                                None if record is None else asdict(record)
                            ),
                        }
                        status = 0
                finally:
                    settlement_venue.close()
            elif action == "supervise":
                payload = asyncio.run(
                    _supervise(
                        credentials=credentials,
                        venue=venue,
                        ledger=ledger,
                        risk_limits=risk_limits,
                        duration_seconds=args.duration_seconds,
                        reconciliation_seconds=args.reconciliation_seconds,
                        automatic_redemption=bool(args.automatic_redemption),
                    )
                )
                payload["action"] = action
                status = 0
            else:  # pragma: no cover - argparse owns the finite action set
                raise ValueError("unsupported Polymarket live action")
        finally:
            venue.close()
    except Exception as exc:
        print(
            f"polymarket-live failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    _render(payload, as_json=bool(args.json))
    return status


__all__ = [
    "DEFAULT_POLYMARKET_LIVE_LEDGER",
    "command_polymarket_live",
    "register_polymarket_live_command",
]
