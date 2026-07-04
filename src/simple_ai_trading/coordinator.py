"""Independent loop heartbeat coordinator for autonomous trading safety."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from .ai_runtime import detect_ai_capabilities
from .positions import PositionsStore, load_learning_feedback
from .risk_controls import build_risk_policy_report
from .types import RuntimeConfig, StrategyConfig


@dataclass(frozen=True)
class LoopContract:
    """Static safety contract for one independently reported loop."""

    name: str
    stale_after_ms: int
    required: bool
    blocks_execution: bool
    blocks_new_entries: bool
    description: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LoopHeartbeat:
    """Latest observed state for one independent loop."""

    name: str
    ok: bool
    status: str
    updated_at_ms: int
    detail: str = ""
    metrics: dict[str, object] = field(default_factory=dict)

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CoordinatorDecision:
    """Combined gate decision from independent loop heartbeats."""

    state: str
    allow_execution: bool
    allow_new_entries: bool
    blocking_loops: tuple[str, ...]
    stale_loops: tuple[str, ...]
    advisory_loops: tuple[str, ...]
    actions: tuple[str, ...]
    heartbeats: dict[str, dict[str, object]]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


class LoopCoordinator:
    """Deterministically combine independent loop heartbeats."""

    def __init__(self, contracts: tuple[LoopContract, ...]) -> None:
        self.contracts = {contract.name: contract for contract in contracts}
        self.heartbeats: dict[str, LoopHeartbeat] = {}

    def ingest(self, heartbeat: LoopHeartbeat) -> None:
        if heartbeat.name not in self.contracts:
            raise ValueError(f"unknown loop heartbeat: {heartbeat.name}")
        self.heartbeats[heartbeat.name] = heartbeat

    def decide(self, *, now_ms: int) -> CoordinatorDecision:
        blocking: list[str] = []
        stale: list[str] = []
        advisory: list[str] = []
        actions: list[str] = []
        allow_execution = True
        allow_entries = True
        for name, contract in self.contracts.items():
            heartbeat = self.heartbeats.get(name)
            is_missing = heartbeat is None
            is_stale = False
            is_ok = False
            if heartbeat is not None:
                age = max(0, int(now_ms) - int(heartbeat.updated_at_ms))
                is_stale = contract.required and contract.stale_after_ms > 0 and age > contract.stale_after_ms
                is_ok = bool(heartbeat.ok)
            if is_missing and contract.required:
                stale.append(name)
                actions.append(f"publish_{name}_heartbeat")
            elif is_stale:
                stale.append(name)
                actions.append(f"refresh_{name}")
            elif heartbeat is not None and not is_ok:
                blocking.append(name)
                actions.append(f"resolve_{name}:{heartbeat.status}")
            elif heartbeat is not None and not contract.required and not is_ok:
                advisory.append(name)

            failed = (is_missing and contract.required) or is_stale or (heartbeat is not None and not is_ok)
            if failed and contract.blocks_execution:
                allow_execution = False
            if failed and contract.blocks_new_entries:
                allow_entries = False

        if not allow_execution:
            state = "blocked_execution"
        elif not allow_entries:
            state = "waiting"
        elif blocking or stale:
            state = "review_required"
        else:
            state = "ready"
        return CoordinatorDecision(
            state=state,
            allow_execution=allow_execution,
            allow_new_entries=allow_entries,
            blocking_loops=tuple(dict.fromkeys(blocking)),
            stale_loops=tuple(dict.fromkeys(stale)),
            advisory_loops=tuple(dict.fromkeys(advisory)),
            actions=tuple(dict.fromkeys(actions)),
            heartbeats={name: heartbeat.asdict() for name, heartbeat in sorted(self.heartbeats.items())},
        )


def default_loop_contracts(
    *,
    ai_enabled: bool,
    live_mode: bool,
) -> tuple[LoopContract, ...]:
    """Return the independent loop contracts in coordinator order."""

    return (
        LoopContract("risk", 30_000, True, True, True, "hard capital and pre-trade risk controls"),
        LoopContract("execution", 15_000, True, True, True, "order-path and account safety"),
        LoopContract("reconciliation", 60_000, live_mode, True, True, "exchange exposure versus local ledger"),
        LoopContract("market_data", 60_000, True, False, True, "fresh prices, spread, and liquidity evidence"),
        LoopContract("machine_learning", 3_600_000, True, False, True, "promoted non-AI model artifact readiness"),
        LoopContract("ai", 3_600_000, ai_enabled, False, bool(ai_enabled), "local multibillion AI capability and review"),
        LoopContract("learning", 86_400_000, False, False, False, "post-trade mistake feedback for retraining review"),
    )


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


def _load_json(path: Path) -> Mapping[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def build_runtime_coordinator(
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    *,
    model_path: Path = Path("data/model.json"),
    positions_root: Path = Path("data/autonomous"),
    now_ms: int | None = None,
) -> CoordinatorDecision:
    """Build a coordinator decision from current local artifacts."""

    now = int(now_ms if now_ms is not None else _now_ms())
    live_mode = bool(not runtime.dry_run)
    coordinator = LoopCoordinator(
        default_loop_contracts(ai_enabled=bool(runtime.ai_enabled), live_mode=live_mode)
    )

    risk_report = build_risk_policy_report(runtime, strategy, effective_dry_run=runtime.dry_run)
    coordinator.ingest(LoopHeartbeat(
        "risk",
        risk_report.allowed,
        "ok" if risk_report.allowed else "blocked",
        now,
        f"blocks={risk_report.block_count} warnings={risk_report.warning_count}",
        {"block_count": risk_report.block_count, "warning_count": risk_report.warning_count},
    ))

    execution_ok = bool(runtime.dry_run or runtime.testnet or runtime.demo)
    if not runtime.dry_run:
        execution_ok = execution_ok and bool(runtime.api_key and runtime.api_secret)
    coordinator.ingest(LoopHeartbeat(
        "execution",
        execution_ok,
        "ok" if execution_ok else "non_mainnet_or_missing_credentials",
        now,
        "paper/testnet/demo execution only" if execution_ok else "live mainnet or missing credentials blocked",
    ))

    reconciliation_path = positions_root / "reconciliation.json"
    reconciliation_payload = _load_json(reconciliation_path)
    reconciliation_ok = not live_mode
    reconciliation_detail = "paper mode"
    if live_mode:
        reconciliation_ok = bool(reconciliation_payload and reconciliation_payload.get("ok") is True)
        reconciliation_detail = "accepted" if reconciliation_ok else "missing_or_failed_reconciliation"
    coordinator.ingest(LoopHeartbeat(
        "reconciliation",
        reconciliation_ok,
        "ok" if reconciliation_ok else "blocked",
        now,
        reconciliation_detail,
    ))

    market_ok = bool(runtime.symbol and runtime.interval and runtime.symbols)
    coordinator.ingest(LoopHeartbeat(
        "market_data",
        market_ok,
        "ok" if market_ok else "missing_symbol_or_interval",
        now,
        f"symbol={runtime.symbol} interval={runtime.interval}",
    ))

    model_ok = model_path.exists()
    coordinator.ingest(LoopHeartbeat(
        "machine_learning",
        model_ok,
        "ok" if model_ok else "missing_model",
        now,
        str(model_path),
    ))

    ai_report = detect_ai_capabilities(runtime.ai_runtime_config())
    coordinator.ingest(LoopHeartbeat(
        "ai",
        ai_report.ok or not runtime.ai_enabled,
        "ok" if ai_report.ok else ("disabled" if not runtime.ai_enabled else "blocked"),
        now,
        "; ".join(ai_report.messages or ai_report.warnings) or ai_report.model,
        {"model_parameters_b": ai_report.model_parameters_b, "gpu_vendor": ai_report.gpu_vendor},
    ))

    feedback = load_learning_feedback(PositionsStore(root=positions_root))
    coordinator.ingest(LoopHeartbeat(
        "learning",
        bool(feedback.promotion_safe or feedback.closed_trades == 0),
        "ok" if feedback.promotion_safe else "observe",
        now,
        ",".join(feedback.recommendations[:3]),
        {"closed_trades": feedback.closed_trades, "max_loss_streak": feedback.max_consecutive_losses},
    ))
    return coordinator.decide(now_ms=now)


def render_coordinator_decision(decision: CoordinatorDecision) -> str:
    """Render the coordinator state as a simple operator summary."""

    lines = [
        "Coordinator",
        (
            f"state={decision.state} allow_execution={decision.allow_execution} "
            f"allow_new_entries={decision.allow_new_entries}"
        ),
    ]
    if decision.blocking_loops:
        lines.append("blocking_loops=" + ",".join(decision.blocking_loops))
    if decision.stale_loops:
        lines.append("stale_loops=" + ",".join(decision.stale_loops))
    for name, payload in sorted(decision.heartbeats.items()):
        lines.append(f"[{payload['status']}] {name}: {payload.get('detail', '')}")
    for action in decision.actions:
        lines.append(f"- {action}")
    return "\n".join(lines)


__all__ = [
    "CoordinatorDecision",
    "LoopContract",
    "LoopCoordinator",
    "LoopHeartbeat",
    "build_runtime_coordinator",
    "default_loop_contracts",
    "render_coordinator_decision",
]
