"""CLI controls for the independent BTC Polymarket live execution boundary."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Mapping

from .polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveCoordinator,
    PolymarketLiveOrderLedger,
    PolymarketLiveRiskLimits,
)
from .polymarket_live_activation import (
    DEFAULT_POLYMARKET_LIVE_ACTIVATION,
    load_polymarket_live_activation,
    write_polymarket_live_activation,
)
from .polymarket_live_runtime import (
    PolymarketAuthenticatedUserStream,
    PolymarketLiveRuntimeGuard,
    PolymarketReconciliationService,
    PolymarketUserStreamConsumer,
)
from .polymarket_live_stop import stop_owned_polymarket_exposure
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
from .polymarket_runtime_control import (
    PolymarketRuntimeControl,
    PolymarketRuntimeControlService,
    PolymarketRuntimeLeaseInterlock,
)


DEFAULT_POLYMARKET_LIVE_LEDGER = Path("data/polymarket/live-ownership.sqlite3")
_ACTIONS = (
    "status",
    "preflight",
    "reconcile",
    "supervise",
    "prepare-autonomous",
    "autonomous",
    "pause",
    "resume",
    "cancel-owned",
    "stop",
    "recover-redemptions",
    "redeem",
)
_RISK_LIMITS = {
    "conservative": (Decimal("10"), Decimal("20"), Decimal("10"), 1),
    "regular": (Decimal("25"), Decimal("50"), Decimal("50"), 2),
    "aggressive": (Decimal("50"), Decimal("100"), Decimal("100"), 3),
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
            "Autonomous opening remains blocked unless an unexpired, hash-bound "
            "promotion and exact authenticated lifecycle qualification both pass."
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
        "--stop-timeout-seconds",
        type=float,
        default=30.0,
        help="bounded time for exact owned-order cancellation and position close",
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
    parser.add_argument(
        "--activation",
        default=None,
        help="portable hash-bound activation bundle used by autonomous mode",
    )
    parser.add_argument(
        "--activation-output",
        default=str(DEFAULT_POLYMARKET_LIVE_ACTIVATION),
        help="non-secret activation bundle written by prepare-autonomous",
    )
    parser.add_argument(
        "--replace-activation",
        action="store_true",
        help="explicitly replace an existing activation bundle after revalidation",
    )
    parser.add_argument(
        "--promotion",
        default=None,
        help="hash-bound live-promotion JSON required by autonomous mode",
    )
    parser.add_argument(
        "--evidence-root",
        default=None,
        help="root containing the exact promotion-bound evidence files",
    )
    parser.add_argument(
        "--lifecycle-qualification",
        default=None,
        help=(
            "canonical authenticated lifecycle qualification required by "
            "autonomous mode"
        ),
    )
    parser.add_argument(
        "--lifecycle-qualification-sha256",
        default=None,
        help="exact file SHA-256 of the authenticated lifecycle qualification",
    )
    parser.add_argument(
        "--round16-contract",
        default=(
            "docs/model-research/polymarket/"
            "round-016-btc-15m-horizon-comparison-v2.json"
        ),
        help="frozen contract required only by a fifteen-minute promotion",
    )
    parser.add_argument(
        "--pretest-envelope-sha256",
        default=None,
        help="fifteen-minute-only promoted pretest-envelope SHA-256 pin",
    )
    parser.add_argument(
        "--evaluation-envelope-sha256",
        default=None,
        help="fifteen-minute-only promoted evaluation-envelope SHA-256 pin",
    )
    parser.add_argument(
        "--requested-quantity",
        type=Decimal,
        default="5",
        help="maximum requested outcome-token quantity before deterministic risk gates",
    )
    parser.add_argument(
        "--risk-capital-quote",
        type=Decimal,
        default=None,
        help=(
            "explicit dedicated-wallet capital basis required by autonomous "
            "daily-loss and drawdown gates"
        ),
    )
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=command_polymarket_live)


def _risk_limits(profile: str) -> PolymarketLiveRiskLimits:
    maximum_quote, maximum_tokens, maximum_at_risk, maximum_markets = _RISK_LIMITS[
        str(profile)
    ]
    return PolymarketLiveRiskLimits(
        maximum_order_quote=maximum_quote,
        maximum_token_quantity=maximum_tokens,
        maximum_total_at_risk_quote=maximum_at_risk,
        maximum_active_markets=maximum_markets,
    )


_POLYMARKET_LIVE_STATUS_BOUNDARY: dict[str, object] = {
    "schema_version": "polymarket-live-operator-status-v2",
    "venue": "polymarket",
    "execution_mode": "authenticated_live_clob_v2",
    "execution_venue": "polymarket_clob_v2",
    "settlement_venue": "polymarket_polygon",
    "symbol": "BTC",
    "paper_execution": False,
    "paper_fallback_allowed": False,
    "binance_required_for_core_model": False,
    "binance_public_predictor_optional": True,
    "binance_public_predictor_data_allowed": True,
    "binance_public_predictor_trading_authority": False,
    "binance_credentials_allowed": False,
    "binance_account_authority": False,
    "binance_execution_authority": False,
    "binance_position_authority": False,
    "binance_risk_or_stop_authority": False,
}


def _local_status(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            **_POLYMARKET_LIVE_STATUS_BOUNDARY,
            "ledger": str(path),
            "ledger_exists": False,
            "open_owned_order_count": 0,
            "owned_position_count": 0,
            "unresolved_redemption_count": 0,
            "runtime_state": "unconfigured",
            "can_open": False,
            "reason": "authenticated_preflight_required",
        }
    ledger = PolymarketLiveOrderLedger(path)
    runtime_control = PolymarketRuntimeControl(path).snapshot()
    records = ledger.records()
    inventory = ledger.owned_inventory()
    redemptions = ledger.redemption_records()
    unresolved = tuple(
        record
        for record in redemptions
        if record.state in {"prepared", "submitting", "submitted", "unknown"}
    )
    return {
        **_POLYMARKET_LIVE_STATUS_BOUNDARY,
        "ledger": str(path),
        "ledger_exists": True,
        "order_count": len(records),
        "open_owned_order_count": len(ledger.open_owned_order_ids()),
        "owned_position_count": len(inventory),
        "owned_position_quantity": format(
            sum((item.quantity for item in inventory), Decimal("0")),
            "f",
        ),
        "unverified_fill_accounting_count": (ledger.unverified_fill_accounting_count()),
        "unverified_redemption_accounting_count": (
            ledger.unverified_redemption_accounting_count()
        ),
        "unresolved_redemption_count": len(unresolved),
        "runtime_state": runtime_control.state,
        "runtime_control": runtime_control.asdict(),
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


def _stop_owned_exposure(
    *,
    coordinator: PolymarketLiveCoordinator,
    ledger: PolymarketLiveOrderLedger,
    timeout_seconds: float,
) -> tuple[dict[str, object], int]:
    return stop_owned_polymarket_exposure(
        coordinator=coordinator,
        ledger=ledger,
        timeout_seconds=timeout_seconds,
        monotonic=time.monotonic,
        sleep=time.sleep,
    )


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
        raise PolymarketLiveBlocked(f"supervision preflight failed: {initial.errors}")
    consumer = PolymarketUserStreamConsumer(ledger, guard)
    stream = PolymarketAuthenticatedUserStream(
        credentials,
        consumer,
        markets=tuple(sorted({record.intent.market_id for record in ledger.records()})),
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


def _required_autonomous_argument(value: object, *, name: str) -> str:
    selected = str(value or "").strip()
    if not selected:
        raise PolymarketLiveBlocked(f"autonomous Polymarket mode requires --{name}")
    return selected


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _positive_risk_capital(value: object) -> Decimal:
    if value is None:
        raise PolymarketLiveBlocked(
            "autonomous Polymarket mode requires positive --risk-capital-quote"
        )
    selected = Decimal(str(value))
    if not selected.is_finite() or selected <= 0:
        raise PolymarketLiveBlocked(
            "autonomous Polymarket mode requires positive --risk-capital-quote"
        )
    return selected


def _prepare_autonomous_activation(
    args: argparse.Namespace,
) -> dict[str, object]:
    if args.activation:
        raise PolymarketLiveBlocked(
            "prepare-autonomous accepts source flags, not --activation"
        )
    promotion_path = _required_autonomous_argument(
        args.promotion,
        name="promotion",
    )
    evidence_root = _required_autonomous_argument(
        args.evidence_root,
        name="evidence-root",
    )
    lifecycle_path = _required_autonomous_argument(
        args.lifecycle_qualification,
        name="lifecycle-qualification",
    )
    risk_capital = _positive_risk_capital(args.risk_capital_quote)
    components = _load_autonomous_components()
    observed_at_ms = int(time.time() * 1_000)
    promotion_file_sha256 = _sha256_file(promotion_path)
    promotion = components.load_polymarket_live_promotion(
        promotion_path,
        evidence_root=evidence_root,
        require_live_authority=True,
        observed_at_ms=observed_at_ms,
        expected_file_sha256=promotion_file_sha256,
    )
    credentials = PolymarketLiveCredentials.from_environment()
    lifecycle_file_sha256 = _sha256_file(lifecycle_path)
    lifecycle = components.load_polymarket_lifecycle_qualification(
        lifecycle_path,
        expected_file_sha256=lifecycle_file_sha256,
        observed_at_ms=observed_at_ms,
    )
    lifecycle.assert_runtime_binding(
        credentials=credentials,
        promotion=promotion,
        observed_at_ms=observed_at_ms,
    )
    if _sha256_file(promotion_path) != promotion_file_sha256:
        raise PolymarketLiveBlocked(
            "Polymarket promotion changed during activation preparation"
        )
    if _sha256_file(lifecycle_path) != lifecycle_file_sha256:
        raise PolymarketLiveBlocked(
            "Polymarket lifecycle qualification changed during activation preparation"
        )
    variant = promotion.promotion.market_variant
    round16_path: str | None = None
    pretest_sha256 = ""
    evaluation_sha256 = ""
    if variant == "fifteenminute":
        round16_path = _required_autonomous_argument(
            args.round16_contract,
            name="round16-contract",
        )
        pretest_sha256 = _required_autonomous_argument(
            args.pretest_envelope_sha256,
            name="pretest-envelope-sha256",
        )
        evaluation_sha256 = _required_autonomous_argument(
            args.evaluation_envelope_sha256,
            name="evaluation-envelope-sha256",
        )
    elif args.pretest_envelope_sha256 or args.evaluation_envelope_sha256:
        raise PolymarketLiveBlocked(
            "five-minute activation cannot contain Round 16 envelope pins"
        )
    verified = write_polymarket_live_activation(
        args.activation_output,
        market_variant=variant,
        risk_level=args.risk_level,
        risk_capital_quote=risk_capital,
        requested_quantity=args.requested_quantity,
        promotion_path=promotion_path,
        evidence_root=evidence_root,
        lifecycle_qualification_path=lifecycle_path,
        round16_contract_path=round16_path,
        pretest_envelope_sha256=pretest_sha256,
        evaluation_envelope_sha256=evaluation_sha256,
        created_at_ms=observed_at_ms,
        replace_existing=bool(args.replace_activation),
    )
    activation = verified.activation
    return {
        "schema_version": "polymarket-live-operator-result-v1",
        "action": "prepare-autonomous",
        "venue": "polymarket",
        "symbol": "BTC",
        "activation_path": str(verified.path),
        "activation_sha256": activation.activation_sha256,
        "activation_file_sha256": verified.file_sha256,
        "market_variant": activation.market_variant,
        "risk_level": activation.risk_level,
        "risk_capital_quote": str(activation.risk_capital_quote),
        "requested_quantity": str(activation.requested_quantity),
        "credentials_stored": False,
        "orders_submitted": False,
        "trading_authority_granted": False,
    }


def _activation_autonomous_arguments(
    args: argparse.Namespace,
) -> dict[str, object]:
    if not args.activation:
        return {
            "promotion_path": args.promotion,
            "promotion_file_sha256": "",
            "evidence_root": args.evidence_root,
            "lifecycle_qualification_path": args.lifecycle_qualification,
            "lifecycle_qualification_sha256": (args.lifecycle_qualification_sha256),
            "round16_contract": args.round16_contract,
            "round16_contract_file_sha256": "",
            "pretest_envelope_sha256": args.pretest_envelope_sha256,
            "evaluation_envelope_sha256": args.evaluation_envelope_sha256,
            "requested_quantity": args.requested_quantity,
            "risk_level": args.risk_level,
            "risk_capital_quote": args.risk_capital_quote,
            "activation_sha256": "",
            "activation_file_sha256": "",
        }
    mixed = {
        "promotion": args.promotion,
        "evidence-root": args.evidence_root,
        "lifecycle-qualification": args.lifecycle_qualification,
        "lifecycle-qualification-sha256": args.lifecycle_qualification_sha256,
        "risk-capital-quote": args.risk_capital_quote,
        "pretest-envelope-sha256": args.pretest_envelope_sha256,
        "evaluation-envelope-sha256": args.evaluation_envelope_sha256,
    }
    supplied = sorted(name for name, value in mixed.items() if value is not None)
    if supplied:
        raise PolymarketLiveBlocked(
            "--activation cannot be mixed with source flags: " + ", ".join(supplied)
        )
    verified = load_polymarket_live_activation(args.activation)
    activation = verified.activation
    return {
        "promotion_path": str(verified.promotion_path),
        "promotion_file_sha256": activation.promotion.file_sha256,
        "evidence_root": str(verified.evidence_root),
        "lifecycle_qualification_path": str(verified.lifecycle_qualification_path),
        "lifecycle_qualification_sha256": (
            activation.lifecycle_qualification.file_sha256
        ),
        "round16_contract": (
            None
            if verified.round16_contract_path is None
            else str(verified.round16_contract_path)
        ),
        "round16_contract_file_sha256": (
            ""
            if activation.round16_contract is None
            else activation.round16_contract.file_sha256
        ),
        "pretest_envelope_sha256": activation.pretest_envelope_sha256,
        "evaluation_envelope_sha256": activation.evaluation_envelope_sha256,
        "requested_quantity": activation.requested_quantity,
        "risk_level": activation.risk_level,
        "risk_capital_quote": activation.risk_capital_quote,
        "activation_sha256": activation.activation_sha256,
        "activation_file_sha256": verified.file_sha256,
    }


def _load_autonomous_components() -> SimpleNamespace:
    """Load the Polymarket-only stack used for autonomous operation."""
    from .polymarket import PolymarketPublicClient
    from .polymarket_autonomous_runtime import PolymarketAutonomousSupervisor
    from .polymarket_historical_shadow import PolymarketBtcFlowBuffer
    from .polymarket_historical_shadow_feed import PolymarketHistoricalShadowFeed
    from .polymarket_live_promotion import load_polymarket_live_promotion
    from .polymarket_live_qualification import (
        load_polymarket_lifecycle_qualification,
    )
    from .polymarket_round21_runtime import (
        build_polymarket_round21_runtime_stack,
    )
    from .polymarket_round16_decision import (
        PolymarketRound16PromotedDecisionProvider,
    )
    from .polymarket_round16_shadow import (
        PolymarketRound16LiveFeatureBuilder,
        PolymarketRound16ShadowScorer,
        load_verified_round16_shadow_predictor,
    )

    return SimpleNamespace(
        PolymarketAutonomousSupervisor=PolymarketAutonomousSupervisor,
        PolymarketBtcFlowBuffer=PolymarketBtcFlowBuffer,
        PolymarketHistoricalShadowFeed=PolymarketHistoricalShadowFeed,
        PolymarketPublicClient=PolymarketPublicClient,
        PolymarketRound16LiveFeatureBuilder=PolymarketRound16LiveFeatureBuilder,
        PolymarketRound16PromotedDecisionProvider=(
            PolymarketRound16PromotedDecisionProvider
        ),
        PolymarketRound16ShadowScorer=PolymarketRound16ShadowScorer,
        build_polymarket_round21_runtime_stack=(build_polymarket_round21_runtime_stack),
        load_polymarket_live_promotion=load_polymarket_live_promotion,
        load_polymarket_lifecycle_qualification=(
            load_polymarket_lifecycle_qualification
        ),
        load_verified_round16_shadow_predictor=(load_verified_round16_shadow_predictor),
    )


def _build_autonomous_decision_stack(
    *,
    components: SimpleNamespace,
    public_client: object,
    promotion: object,
    round16_contract: object,
    pretest_envelope_sha256: object,
    evaluation_envelope_sha256: object,
    requested_quantity: Decimal,
    risk_level: str,
    round16_contract_file_sha256: str = "",
) -> tuple[object, object]:
    """Select one promotion-bound predictor without sharing venue state."""

    policy = promotion.promotion
    if policy.market_variant == "fiveminute":
        stack = components.build_polymarket_round21_runtime_stack(
            public_client=public_client,
            promotion=promotion,
            requested_quantity=requested_quantity,
            risk_level=risk_level,
        )
        return stack.data_service, stack.decision_provider
    if policy.market_variant != "fifteenminute":
        raise PolymarketLiveBlocked(
            "autonomous Polymarket promotion market variant is unsupported"
        )
    selected_contract = _required_autonomous_argument(
        round16_contract,
        name="round16-contract",
    )
    pretest_sha = _required_autonomous_argument(
        pretest_envelope_sha256,
        name="pretest-envelope-sha256",
    )
    evaluation_sha = _required_autonomous_argument(
        evaluation_envelope_sha256,
        name="evaluation-envelope-sha256",
    )
    predictor_arguments: dict[str, object] = {
        "contract_path": selected_contract,
        "pretest_path": promotion.model_artifact_path,
        "evaluation_path": promotion.evaluation_report_path,
        "expected_pretest_envelope_sha256": pretest_sha,
        "expected_evaluation_envelope_sha256": evaluation_sha,
    }
    if round16_contract_file_sha256:
        predictor_arguments["expected_contract_file_sha256"] = (
            round16_contract_file_sha256
        )
    predictor = components.load_verified_round16_shadow_predictor(
        **predictor_arguments,
    )
    flow = components.PolymarketBtcFlowBuffer(retention_seconds=1_200)
    predictor_feed = components.PolymarketHistoricalShadowFeed(flow=flow)
    scorer = components.PolymarketRound16ShadowScorer(
        predictor=predictor,
        feature_builder=components.PolymarketRound16LiveFeatureBuilder(flow),
    )
    decision_provider = components.PolymarketRound16PromotedDecisionProvider(
        public_client=public_client,
        scorer=scorer,
        promotion=promotion,
        requested_quantity=requested_quantity,
    )
    return predictor_feed, decision_provider


def _finalize_autonomous_resources(
    *,
    runtime_control: PolymarketRuntimeControl,
    runtime_lease: str,
    runtime_released: bool,
    ledger: PolymarketLiveOrderLedger,
    settlement_venue: OfficialPolymarketUnifiedRedemptionVenue | None,
    public_client: object | None,
) -> None:
    errors: list[BaseException] = []
    if not runtime_released:
        inventory_known = True
        try:
            has_owned_inventory = bool(ledger.owned_inventory())
        except BaseException as exc:
            inventory_known = False
            has_owned_inventory = True
            errors.append(exc)
        try:
            if has_owned_inventory:
                runtime_control.request_stop(
                    reason=(
                        "autonomous_exit_with_owned_exposure"
                        if inventory_known
                        else "autonomous_exit_inventory_unknown"
                    )
                )
            else:
                runtime_control.release(
                    runtime_lease,
                    reason="autonomous_process_exit",
                )
        except BaseException as exc:
            errors.append(exc)
    if settlement_venue is not None:
        try:
            settlement_venue.close()
        except BaseException as exc:
            errors.append(exc)
    if public_client is not None:
        try:
            public_client.session.close()
        except BaseException as exc:
            errors.append(exc)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("Polymarket autonomous cleanup failed", errors)


async def _autonomous(
    *,
    credentials: PolymarketLiveCredentials,
    venue: OfficialPolymarketV2Venue,
    ledger: PolymarketLiveOrderLedger,
    risk_limits: PolymarketLiveRiskLimits,
    duration_seconds: float,
    reconciliation_seconds: float,
    stop_timeout_seconds: float,
    automatic_redemption: bool,
    promotion_path: object,
    evidence_root: object,
    lifecycle_qualification_path: object,
    lifecycle_qualification_sha256: object,
    round16_contract: object,
    pretest_envelope_sha256: object,
    evaluation_envelope_sha256: object,
    requested_quantity: Decimal,
    risk_level: str,
    risk_capital_quote: Decimal | None,
    activation_sha256: str = "",
    activation_file_sha256: str = "",
    promotion_file_sha256: str = "",
    round16_contract_file_sha256: str = "",
) -> dict[str, object]:
    duration = float(duration_seconds)
    interval = float(reconciliation_seconds)
    stop_timeout = float(stop_timeout_seconds)
    if not 0 <= duration <= 31_536_000:
        raise ValueError("duration-seconds must lie in [0, 31536000]")
    if not 1 <= interval <= 60:
        raise ValueError("reconciliation-seconds must lie in [1, 60]")
    if not 1 <= stop_timeout <= 300:
        raise ValueError("stop-timeout-seconds must lie in [1, 300]")
    selected_risk_capital = _positive_risk_capital(risk_capital_quote)
    selected_promotion = _required_autonomous_argument(
        promotion_path,
        name="promotion",
    )
    selected_root = _required_autonomous_argument(
        evidence_root,
        name="evidence-root",
    )
    components = _load_autonomous_components()
    observed_at_ms = int(time.time() * 1_000)
    promotion_arguments: dict[str, object] = {
        "evidence_root": selected_root,
        "require_live_authority": True,
        "observed_at_ms": observed_at_ms,
    }
    if promotion_file_sha256:
        promotion_arguments["expected_file_sha256"] = promotion_file_sha256
    promotion = components.load_polymarket_live_promotion(
        selected_promotion,
        **promotion_arguments,
    )
    selected_lifecycle = _required_autonomous_argument(
        lifecycle_qualification_path,
        name="lifecycle-qualification",
    )
    lifecycle_sha256 = _required_autonomous_argument(
        lifecycle_qualification_sha256,
        name="lifecycle-qualification-sha256",
    )
    lifecycle_qualification = components.load_polymarket_lifecycle_qualification(
        selected_lifecycle,
        expected_file_sha256=lifecycle_sha256,
        observed_at_ms=observed_at_ms,
    )
    lifecycle_qualification.assert_runtime_binding(
        credentials=credentials,
        promotion=promotion,
        observed_at_ms=observed_at_ms,
    )
    runtime_control = PolymarketRuntimeControl(ledger.path)
    runtime_lease = runtime_control.acquire()
    runtime_released = False
    public_client: object | None = None
    settlement_venue: OfficialPolymarketUnifiedRedemptionVenue | None = None
    started = time.monotonic()
    try:
        public_client = components.PolymarketPublicClient()
        predictor_feed, decision_provider = _build_autonomous_decision_stack(
            components=components,
            public_client=public_client,
            promotion=promotion,
            round16_contract=round16_contract,
            pretest_envelope_sha256=pretest_envelope_sha256,
            evaluation_envelope_sha256=evaluation_envelope_sha256,
            requested_quantity=requested_quantity,
            risk_level=risk_level,
            round16_contract_file_sha256=round16_contract_file_sha256,
        )
        guard = PolymarketLiveRuntimeGuard(
            maximum_reconciliation_age_ms=max(5_000, int(interval * 3_000)),
            opening_interlock=PolymarketRuntimeLeaseInterlock(
                control=runtime_control,
                lease_id=runtime_lease,
            ),
        )
        coordinator = PolymarketLiveCoordinator(
            venue,
            ledger,
            risk_limits=risk_limits,
            runtime_authority=guard,
        )
        consumer = PolymarketUserStreamConsumer(ledger, guard)
        user_stream = PolymarketAuthenticatedUserStream(
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
        redemption, settlement_venue = _settlement_coordinator(
            credentials=credentials,
            account=venue,
            ledger=ledger,
        )
        settlement = PolymarketSettlementService(
            redemption,
            guard,
            automatic_redemption_enabled=automatic_redemption,
            interval_seconds=max(5.0, interval),
        )
        supervisor = components.PolymarketAutonomousSupervisor(
            public_client=public_client,
            coordinator=coordinator,
            ledger=ledger,
            runtime_guard=guard,
            user_stream=user_stream,
            reconciliation=reconciliation,
            promotion=promotion,
            lifecycle_qualification=lifecycle_qualification,
            decision_provider=decision_provider,
            risk_capital_quote=selected_risk_capital,
            risk_level=risk_level,
            settlement=settlement,
            decision_data_service=predictor_feed,
            durable_control_service=PolymarketRuntimeControlService(
                runtime_control,
                lease_id=runtime_lease,
            ),
            stop_timeout_seconds=stop_timeout,
        )
        await supervisor.run(duration_seconds=duration)
        final = await asyncio.to_thread(coordinator.reconcile)
        snapshot = supervisor.snapshot()
        if ledger.owned_inventory():
            runtime_control.request_stop(reason="autonomous_exit_with_owned_exposure")
            raise PolymarketLiveBlocked(
                "Polymarket autonomous runtime retained owned exposure at exit"
            )
        runtime_control.release(
            runtime_lease,
            reason="autonomous_stop_completed",
        )
        runtime_released = True
        feed_health = predictor_feed.health()
        return {
            "schema_version": "polymarket-live-autonomous-v1",
            "venue": "polymarket",
            "symbol": "BTC",
            "market_variant": snapshot.market_variant,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "promotion_sha256": promotion.promotion.promotion_sha256,
            "activation_sha256": activation_sha256,
            "activation_file_sha256": activation_file_sha256,
            "lifecycle_qualification_sha256": (
                lifecycle_qualification.qualification.qualification_sha256
            ),
            "model_artifact_sha256": promotion.promotion.model_artifact.sha256,
            "automatic_redemption": automatic_redemption,
            "risk_level": risk_level,
            "risk_capital_quote": format(selected_risk_capital, "f"),
            "reconciliation": _reconciliation_payload(final),
            "runtime": asdict(snapshot),
            "predictor_feed": asdict(feed_health),
            "binance_credentials_used": False,
            "binance_execution_connected": False,
            "runtime_control": runtime_control.snapshot().asdict(),
            "opened_exposure": snapshot.submitted_opens > 0,
        }
    finally:
        _finalize_autonomous_resources(
            runtime_control=runtime_control,
            runtime_lease=runtime_lease,
            runtime_released=runtime_released,
            ledger=ledger,
            settlement_venue=settlement_venue,
            public_client=public_client,
        )


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
            if args.activation:
                activation = load_polymarket_live_activation(args.activation)
                payload.update(
                    {
                        "activation_path": str(activation.path),
                        "activation_sha256": (activation.activation.activation_sha256),
                        "activation_file_sha256": activation.file_sha256,
                        "activation_market_variant": (
                            activation.activation.market_variant
                        ),
                        "activation_risk_level": activation.activation.risk_level,
                    }
                )
            _render(payload, as_json=bool(args.json))
            return 0

        if action == "prepare-autonomous":
            payload = _prepare_autonomous_activation(args)
            _render(payload, as_json=bool(args.json))
            return 0

        runtime_control = PolymarketRuntimeControl(ledger_path)
        if action in {"pause", "resume"}:
            snapshot = runtime_control.set_paused(
                action == "pause",
                reason=f"operator_{action}",
            )
            payload = {
                "schema_version": "polymarket-live-operator-result-v1",
                "action": action,
                "venue": "polymarket",
                "symbol": "BTC",
                "runtime_control": snapshot.asdict(),
            }
            _render(payload, as_json=bool(args.json))
            return 0
        if action == "stop":
            runtime_control.request_stop(reason="operator_stop")
        autonomous_arguments = (
            _activation_autonomous_arguments(args) if action == "autonomous" else None
        )
        credentials, venue = _credentials_and_venue()
        ledger = PolymarketLiveOrderLedger(ledger_path)
        selected_risk_level = (
            args.risk_level
            if autonomous_arguments is None
            else str(autonomous_arguments["risk_level"])
        )
        risk_limits = _risk_limits(selected_risk_level)
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
                # Cancellation cannot increase exposure and the coordinator
                # independently proves every target by exact local order hash.
                # Do not let unrelated foreign positions or a geoblock strand a
                # known bot-owned resting order.
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
            elif action == "stop":
                payload, status = _stop_owned_exposure(
                    coordinator=coordinator,
                    ledger=ledger,
                    timeout_seconds=args.stop_timeout_seconds,
                )
                if bool(payload.get("completed")):
                    runtime_control.complete_stale_stop(
                        exposure_closed=not bool(ledger.owned_inventory()),
                    )
                    if runtime_control.snapshot().state != "stopped":
                        runtime_control.wait_until_stopped(
                            timeout_seconds=min(
                                3.0,
                                float(args.stop_timeout_seconds),
                            ),
                        )
                        runtime_control.complete_stale_stop(
                            exposure_closed=not bool(ledger.owned_inventory()),
                        )
                control_snapshot = runtime_control.snapshot()
                payload["runtime_control"] = control_snapshot.asdict()
                if control_snapshot.state != "stopped":
                    payload["completed"] = False
                    payload["reason"] = "autonomous_runtime_stop_pending"
                    status = 2
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
                            "redemption": (None if record is None else asdict(record)),
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
            elif action == "autonomous":
                if autonomous_arguments is None:  # pragma: no cover - branch invariant
                    raise RuntimeError("Polymarket autonomous inputs are missing")
                payload = asyncio.run(
                    _autonomous(
                        credentials=credentials,
                        venue=venue,
                        ledger=ledger,
                        risk_limits=risk_limits,
                        duration_seconds=args.duration_seconds,
                        reconciliation_seconds=args.reconciliation_seconds,
                        stop_timeout_seconds=args.stop_timeout_seconds,
                        automatic_redemption=bool(args.automatic_redemption),
                        promotion_path=autonomous_arguments["promotion_path"],
                        evidence_root=autonomous_arguments["evidence_root"],
                        lifecycle_qualification_path=(
                            autonomous_arguments["lifecycle_qualification_path"]
                        ),
                        lifecycle_qualification_sha256=(
                            autonomous_arguments["lifecycle_qualification_sha256"]
                        ),
                        round16_contract=autonomous_arguments["round16_contract"],
                        pretest_envelope_sha256=(
                            autonomous_arguments["pretest_envelope_sha256"]
                        ),
                        evaluation_envelope_sha256=(
                            autonomous_arguments["evaluation_envelope_sha256"]
                        ),
                        requested_quantity=autonomous_arguments["requested_quantity"],
                        risk_level=str(autonomous_arguments["risk_level"]),
                        risk_capital_quote=autonomous_arguments["risk_capital_quote"],
                        activation_sha256=str(
                            autonomous_arguments["activation_sha256"]
                        ),
                        activation_file_sha256=str(
                            autonomous_arguments["activation_file_sha256"]
                        ),
                        promotion_file_sha256=str(
                            autonomous_arguments["promotion_file_sha256"]
                        ),
                        round16_contract_file_sha256=str(
                            autonomous_arguments["round16_contract_file_sha256"]
                        ),
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
