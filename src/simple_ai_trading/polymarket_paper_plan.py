"""Promotion-gated model-to-paper replay for Polymarket evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any

from .autonomous import STATE_PAUSED, STATE_RUNNING, STATE_STOPPING
from .paper_execution import paper_intent_id
from .polymarket_paper import PolymarketPaperBroker, PolymarketPaperCoordinator
from .polymarket_publication import validate_polymarket_model_artifact
from .polymarket_source_verification import (
    validate_polymarket_source_verification,
)


POLYMARKET_PAPER_PLAN_SCHEMA_VERSION = "polymarket-paper-plan-v1"
POLYMARKET_PAPER_RUN_SCHEMA_VERSION = "polymarket-paper-model-run-v1"
_MODEL_GATES = (
    "validation_probability_improved",
    "untouched_test_probability_improved",
    "minimum_confirmatory_test_time_groups_met",
    "after_cost_execution_improved",
    "after_cost_model_improved_at_every_stress_latency",
    "all_positions_officially_settled",
    "all_order_outcomes_terminal",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return parsed


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


@dataclass(frozen=True)
class PolymarketPaperPlan:
    schema_version: str
    artifact_sha256: str
    source_verification_sha256: str
    recorder_report_sha256: str
    run_id: str
    policy: str
    primary_network_latency_ms: int
    confirmed_for_paper_run: bool
    research_override: bool
    blocking_reasons: tuple[str, ...]
    execution_report_sha256: str
    execution_config: Mapping[str, object]
    trades: tuple[Mapping[str, object], ...]
    plan_sha256: str
    trading_authority: bool = False
    live_order_authority: bool = False
    profitability_claim: bool = False
    leverage_applied: bool = False

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_sha256": self.artifact_sha256,
            "source_verification_sha256": self.source_verification_sha256,
            "recorder_report_sha256": self.recorder_report_sha256,
            "run_id": self.run_id,
            "policy": self.policy,
            "primary_network_latency_ms": self.primary_network_latency_ms,
            "confirmed_for_paper_run": self.confirmed_for_paper_run,
            "research_override": self.research_override,
            "blocking_reasons": list(self.blocking_reasons),
            "execution_report_sha256": self.execution_report_sha256,
            "execution_config": dict(self.execution_config),
            "trade_count": len(self.trades),
            "trades": [dict(item) for item in self.trades],
            "plan_sha256": self.plan_sha256,
            "trading_authority": self.trading_authority,
            "live_order_authority": self.live_order_authority,
            "profitability_claim": self.profitability_claim,
            "leverage_applied": self.leverage_applied,
        }


@dataclass(frozen=True)
class PolymarketPaperModelRun:
    schema_version: str
    status: str
    plan_sha256: str
    artifact_sha256: str
    source_verification_sha256: str
    recorder_report_sha256: str
    run_id: str
    policy: str
    planned_trade_count: int
    skipped_expired_count: int
    attempted_order_count: int
    matched_execution_count: int
    filled_order_count: int
    settled_position_count: int
    realized_pnl_quote: Decimal
    final_control_state: str
    final_reconciliation: Mapping[str, object]
    errors: tuple[str, ...]
    report_sha256: str
    trading_authority: bool = False
    live_order_authority: bool = False
    profitability_claim: bool = False
    leverage_applied: bool = False

    @property
    def successful(self) -> bool:
        return self.status in {"COMPLETED", "PAUSED"} and not self.errors

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "plan_sha256": self.plan_sha256,
            "artifact_sha256": self.artifact_sha256,
            "source_verification_sha256": self.source_verification_sha256,
            "recorder_report_sha256": self.recorder_report_sha256,
            "run_id": self.run_id,
            "policy": self.policy,
            "planned_trade_count": self.planned_trade_count,
            "skipped_expired_count": self.skipped_expired_count,
            "attempted_order_count": self.attempted_order_count,
            "matched_execution_count": self.matched_execution_count,
            "filled_order_count": self.filled_order_count,
            "settled_position_count": self.settled_position_count,
            "realized_pnl_quote": format(self.realized_pnl_quote, "f"),
            "final_control_state": self.final_control_state,
            "final_reconciliation": dict(self.final_reconciliation),
            "errors": list(self.errors),
            "report_sha256": self.report_sha256,
            "trading_authority": self.trading_authority,
            "live_order_authority": self.live_order_authority,
            "profitability_claim": self.profitability_claim,
            "leverage_applied": self.leverage_applied,
        }


def _plan_identity(plan: PolymarketPaperPlan) -> dict[str, object]:
    payload = plan.asdict()
    payload.pop("plan_sha256")
    return payload


def _run_identity(report: PolymarketPaperModelRun) -> dict[str, object]:
    payload = report.asdict()
    payload.pop("report_sha256")
    return payload


def _source_verification_payload(path: str | Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Polymarket source-verification file is invalid") from exc
    return _mapping(payload, "Polymarket source verification")


def build_polymarket_paper_plan(
    artifact_path: str | Path,
    source_verification_path: str | Path,
    *,
    policy: str = "auto",
    allow_unconfirmed_research: bool = False,
) -> PolymarketPaperPlan:
    """Build an immutable paper-only plan from verified held-out execution evidence."""

    artifact = validate_polymarket_model_artifact(artifact_path)
    source = validate_polymarket_source_verification(
        _source_verification_payload(source_verification_path),
        artifact_sha256=artifact.artifact_sha256,
        run_id=str(artifact.payload["run_id"]),
    )
    requested = str(policy or "auto").strip().lower()
    if requested not in {"auto", "baseline", "model", "ai"}:
        raise ValueError("Polymarket paper policy must be auto, baseline, model, or ai")
    gates = _mapping(artifact.payload["evidence_gates"], "evidence gates")
    model_reasons = tuple(name for name in _MODEL_GATES if gates.get(name) is not True)
    ai_reasons = (
        *model_reasons,
        *(
            ()
            if gates.get("ai_enabled") is True
            else ("ai_not_enabled",)
        ),
        *(
            ()
            if gates.get("ai_uplift_accepted") is True
            else ("ai_uplift_not_accepted",)
        ),
    )

    sensitivity = _mapping(
        artifact.payload["execution_latency_sensitivity"],
        "execution latency sensitivity",
    )
    primary_latency = int(sensitivity["primary_network_latency_ms"])
    latency_policies = _mapping(sensitivity["policies"], "latency policies")

    def economic_reasons(candidate: str) -> tuple[str, ...]:
        raw_reports = _mapping(
            latency_policies.get(candidate),
            f"{candidate} latency reports",
        )
        reasons: list[str] = []
        for latency, raw_report in raw_reports.items():
            report = _mapping(raw_report, f"{candidate} {latency}ms execution")
            if _decimal(report["net_realized_pnl_quote"], "execution PnL") <= 0:
                reasons.append(f"nonpositive_after_cost_pnl_at_{latency}ms")
        return tuple(sorted(reasons))

    reasons_by_policy = {
        "baseline": ("baseline_is_not_a_promoted_model",),
        "model": (*model_reasons, *economic_reasons("model")),
    }
    if "ai" in artifact.executions:
        reasons_by_policy["ai"] = (*ai_reasons, *economic_reasons("ai"))

    if requested == "auto":
        selected = next(
            (
                candidate
                for candidate in ("ai", "model")
                if candidate in reasons_by_policy and not reasons_by_policy[candidate]
            ),
            None,
        )
        if selected is None and allow_unconfirmed_research:
            selected = "ai" if "ai" in artifact.executions else "model"
        if selected is None:
            combined = sorted(
                set(
                    reason
                    for candidate in ("model", "ai")
                    for reason in reasons_by_policy.get(candidate, ())
                )
            )
            raise ValueError(
                "no confirmed Polymarket paper policy: " + ",".join(combined)
            )
    else:
        selected = requested
    if selected not in artifact.executions:
        raise ValueError(f"Polymarket artifact has no {selected} execution policy")
    blocking_reasons = tuple(sorted(set(reasons_by_policy[selected])))
    if blocking_reasons and not allow_unconfirmed_research:
        raise ValueError(
            f"Polymarket {selected} policy is not confirmed: "
            + ",".join(blocking_reasons)
        )

    execution = artifact.executions[selected]
    source_hashes = _mapping(
        source["execution_report_sha256_by_policy_and_latency"],
        "source-verified execution reports",
    )
    policy_hashes = _mapping(
        source_hashes.get(selected),
        f"source-verified {selected} execution reports",
    )
    if policy_hashes.get(str(primary_latency)) != execution["report_sha256"]:
        raise ValueError("primary paper policy is not source-verified")
    trades = tuple(
        sorted(
            (_mapping(item, "paper plan trade") for item in execution["trades"]),
            key=lambda item: (
                int(item["event_start_ms"]),
                int(item["decision_received_monotonic_ns"]),
                str(item["asset"]),
                str(item["trade_id"]),
            ),
        )
    )
    provisional = PolymarketPaperPlan(
        schema_version=POLYMARKET_PAPER_PLAN_SCHEMA_VERSION,
        artifact_sha256=artifact.artifact_sha256,
        source_verification_sha256=str(source["report_sha256"]),
        recorder_report_sha256=str(source["recorder_report_sha256"]),
        run_id=str(artifact.payload["run_id"]),
        policy=selected,
        primary_network_latency_ms=primary_latency,
        confirmed_for_paper_run=not blocking_reasons,
        research_override=bool(blocking_reasons and allow_unconfirmed_research),
        blocking_reasons=blocking_reasons,
        execution_report_sha256=str(execution["report_sha256"]),
        execution_config=dict(_mapping(execution["config"], "execution config")),
        trades=trades,
        plan_sha256="",
    )
    return replace(
        provisional,
        plan_sha256=_canonical_sha256(_plan_identity(provisional)),
    )


def _execution_matches(
    expected: Mapping[str, object],
    actual: object,
    context: tuple[object, ...] | None,
) -> bool:
    if context is None:
        return False
    return (
        str(getattr(actual, "state")) == str(expected["execution_state"])
        and getattr(actual, "filled_quantity")
        == _decimal(expected["filled_quantity"], "expected filled quantity")
        and getattr(actual, "average_fill_price")
        == _decimal(expected["average_fill_price"], "expected fill price")
        and getattr(actual, "fee_quote")
        == _decimal(expected["fee_quote"], "expected fee")
        and str(getattr(actual, "source_payload_sha256"))
        == str(expected["source_payload_sha256"])
        and tuple(context)
        == (
            str(expected["decision_book_event_id"]),
            str(expected["execution_book_event_id"]),
            int(expected["effective_latency_ms"]),
        )
    )


def run_polymarket_paper_plan(
    broker: PolymarketPaperBroker,
    coordinator: PolymarketPaperCoordinator,
    plan: PolymarketPaperPlan,
) -> PolymarketPaperModelRun:
    """Execute one verified historical plan through the owned paper lifecycle."""

    if plan.schema_version != POLYMARKET_PAPER_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Polymarket paper plan schema")
    if plan.plan_sha256 != _canonical_sha256(_plan_identity(plan)):
        raise ValueError("Polymarket paper plan identity is invalid")
    if not all(
        _is_sha256(value)
        for value in (
            plan.artifact_sha256,
            plan.source_verification_sha256,
            plan.recorder_report_sha256,
            plan.execution_report_sha256,
            plan.plan_sha256,
        )
    ):
        raise ValueError("Polymarket paper plan contains an invalid evidence hash")
    if plan.policy not in {"baseline", "model", "ai"}:
        raise ValueError("Polymarket paper plan policy is invalid")
    if plan.confirmed_for_paper_run != (not plan.blocking_reasons):
        raise ValueError("Polymarket paper plan promotion gates are incoherent")
    if plan.research_override != bool(plan.blocking_reasons):
        raise ValueError("Polymarket paper plan research override is incoherent")
    if any(
        (
            plan.trading_authority,
            plan.live_order_authority,
            plan.profitability_claim,
            plan.leverage_applied,
        )
    ):
        raise ValueError("Polymarket paper plan claims unsupported authority")
    if broker.replay.run_id != plan.run_id:
        raise ValueError("Polymarket paper plan belongs to another recorder run")
    config = plan.execution_config
    if int(config["submission_latency_ms"]) != plan.primary_network_latency_ms:
        raise ValueError("Polymarket paper plan primary latency drifted from execution")
    expected_observation_delay = int(
        config["maximum_execution_observation_delay_ms"]
    )
    if (
        broker.maximum_execution_observation_delay_ms
        != expected_observation_delay
        or broker.maximum_book_age_ms != int(config["maximum_book_age_ms"])
        or broker.order_ttl_ms != int(config["order_ttl_ms"])
    ):
        raise ValueError("Polymarket paper broker configuration drifted from the plan")
    recorder = broker.store.connect().execute(
        """
        SELECT report_sha256 FROM polymarket_recorder_run WHERE run_id = ?
        """,
        [plan.run_id],
    ).fetchone()
    if recorder is None:
        raise ValueError("Polymarket paper plan recorder report is missing")
    if str(recorder[0]) != plan.recorder_report_sha256:
        raise ValueError("Polymarket paper plan recorder report identity drifted")
    existing_intents = int(
        broker.store.connect().execute(
            "SELECT count(*) FROM paper_order_intent WHERE venue = 'polymarket'"
        ).fetchone()[0]
    )
    if existing_intents:
        raise ValueError(
            "model paper run requires a clean Polymarket journal to prevent replay mixing"
        )
    reconciliation = broker.reconcile()
    if not reconciliation.can_open or any(
        item.remaining_quantity > 0 for item in reconciliation.journal.inventory
    ):
        raise ValueError("model paper run requires clean, flat reconciliation")
    coordinator.resume()

    skipped_expired = 0
    attempted = 0
    matched = 0
    filled = 0
    settled = 0
    realized = Decimal("0")
    errors: list[str] = []
    status = "COMPLETED"
    resolutions = {item.event_id: item for item in broker.replay.resolutions}
    books = {
        (item.event_id, item.outcome): item for item in broker.replay.books
    }
    groups: dict[int, list[Mapping[str, object]]] = {}
    for trade in plan.trades:
        groups.setdefault(int(trade["event_start_ms"]), []).append(trade)

    try:
        for _group_start, group_trades in sorted(groups.items()):
            opened: list[tuple[str, Mapping[str, object]]] = []
            for trade in group_trades:
                state = coordinator.control.state()
                if state == STATE_PAUSED:
                    status = "PAUSED"
                    break
                if state == STATE_STOPPING:
                    status = "STOPPING"
                    errors.append("operator_stop_requested")
                    break
                if state != STATE_RUNNING:
                    status = "FAILED"
                    errors.append(f"control_state:{state}")
                    break
                coordinator.require_open_allowed()
                key = (
                    str(trade["decision_book_event_id"]),
                    str(trade["outcome"]),
                )
                decision = books.get(key)
                if decision is None:
                    raise ValueError("paper plan decision book is absent from replay")
                order_created_at_ms = int(trade["decision_received_wall_ms"]) + int(
                    trade["decision_delay_ms"]
                )
                if order_created_at_ms >= int(trade["end_ms"]):
                    if not (
                        str(trade["execution_state"]) == "EXPIRED"
                        and str(trade["execution_book_event_id"]) == ""
                        and _decimal(trade["filled_quantity"], "expired quantity") == 0
                        and _decimal(trade["average_fill_price"], "expired price") == 0
                        and _decimal(trade["fee_quote"], "expired fee") == 0
                        and str(trade["source_payload_sha256"])
                        == decision.snapshot.source_payload_sha256
                    ):
                        raise ValueError(
                            "unsubmittable paper order is not valid expired evidence"
                        )
                    skipped_expired += 1
                    matched += 1
                    continue
                position, execution = broker.open_position(
                    position_id=str(trade["sample_id"]),
                    decision=decision,
                    outcome=str(trade["outcome"]),
                    quantity=trade["quantity"],
                    maximum_price=trade["limit_price"],
                    submission_latency_ms=int(trade["submission_latency_ms"]),
                    decision_delay_ms=int(trade["decision_delay_ms"]),
                    order_type="FOK",
                )
                attempted += 1
                intent_id = paper_intent_id(
                    "polymarket",
                    str(trade["sample_id"]),
                    "open",
                )
                context = broker.store.connect().execute(
                    """
                    SELECT decision_event_id, execution_event_id,
                           effective_latency_ms
                    FROM polymarket_paper_order_context WHERE intent_id = ?
                    """,
                    [intent_id],
                ).fetchone()
                if not _execution_matches(trade, execution, context):
                    errors.append(f"execution_mismatch:{trade['trade_id']}")
                    status = "FAILED"
                    if position is not None:
                        opened.append((position.opening_intent_id, trade))
                    break
                matched += 1
                if position is not None:
                    filled += 1
                    opened.append((position.opening_intent_id, trade))
            for opening_intent_id, trade in opened:
                resolution = resolutions.get(
                    str(trade["official_resolution_event_id"])
                )
                if resolution is None:
                    errors.append(f"missing_resolution:{trade['trade_id']}")
                    status = "FAILED"
                    continue
                settlement = broker.settle_position(
                    opening_intent_id=opening_intent_id,
                    resolution=resolution,
                )
                expected_pnl = _decimal(
                    trade["realized_pnl_quote"],
                    "expected realized PnL",
                )
                if (
                    settlement.gross_payout_quote
                    != _decimal(trade["gross_payout_quote"], "expected payout")
                    or settlement.realized_pnl_quote != expected_pnl
                ):
                    errors.append(f"settlement_mismatch:{trade['trade_id']}")
                    status = "FAILED"
                else:
                    settled += 1
                    realized += settlement.realized_pnl_quote
            if status != "COMPLETED":
                break
    except Exception as exc:  # The stop path below must preserve every failure.
        status = "FAILED"
        errors.append(f"{exc.__class__.__name__}:{exc}")

    try:
        final_reconciliation = broker.reconcile()
        flat = not any(
            item.remaining_quantity > 0
            for item in final_reconciliation.journal.inventory
        )
        if status == "COMPLETED" and final_reconciliation.can_open and flat:
            coordinator.pause()
        elif status == "PAUSED" and final_reconciliation.can_close and flat:
            if coordinator.control.state() == STATE_RUNNING:
                coordinator.pause()
        else:
            stop = coordinator.stop_all_positions(
                submission_latency_ms=plan.primary_network_latency_ms,
            )
            if not stop.stopped:
                status = "STOPPING"
                errors.extend(stop.errors)
                errors.extend(
                    f"blocking_intent:{item}" for item in stop.blocking_intent_ids
                )
                errors.extend(
                    f"remaining_inventory:{item}"
                    for item in stop.remaining_opening_intent_ids
                )
        final_reconciliation = broker.reconcile()
        final_state = coordinator.control.state()
        final_flat = not any(
            item.remaining_quantity > 0
            for item in final_reconciliation.journal.inventory
        )
        if status in {"COMPLETED", "PAUSED"} and (
            final_state != STATE_PAUSED
            or not final_reconciliation.can_open
            or not final_flat
        ):
            status = "FAILED"
            errors.append("final_flat_reconciliation_failed")
    except Exception as finalization_exc:
        try:
            coordinator.stop_all_positions(
                submission_latency_ms=plan.primary_network_latency_ms,
            )
        except Exception as stop_exc:  # Stop writes STOPPING before broker work.
            finalization_exc.add_note(
                f"fail-closed Stop also failed: {stop_exc.__class__.__name__}: {stop_exc}"
            )
        raise

    provisional = PolymarketPaperModelRun(
        schema_version=POLYMARKET_PAPER_RUN_SCHEMA_VERSION,
        status=status,
        plan_sha256=plan.plan_sha256,
        artifact_sha256=plan.artifact_sha256,
        source_verification_sha256=plan.source_verification_sha256,
        recorder_report_sha256=plan.recorder_report_sha256,
        run_id=plan.run_id,
        policy=plan.policy,
        planned_trade_count=len(plan.trades),
        skipped_expired_count=skipped_expired,
        attempted_order_count=attempted,
        matched_execution_count=matched,
        filled_order_count=filled,
        settled_position_count=settled,
        realized_pnl_quote=realized,
        final_control_state=final_state,
        final_reconciliation=final_reconciliation.asdict(),
        errors=tuple(sorted(set(errors))),
        report_sha256="",
    )
    return replace(
        provisional,
        report_sha256=_canonical_sha256(_run_identity(provisional)),
    )


__all__ = [
    "POLYMARKET_PAPER_PLAN_SCHEMA_VERSION",
    "POLYMARKET_PAPER_RUN_SCHEMA_VERSION",
    "PolymarketPaperModelRun",
    "PolymarketPaperPlan",
    "build_polymarket_paper_plan",
    "run_polymarket_paper_plan",
]
