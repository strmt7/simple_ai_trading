"""Execution lifecycle preflight for autonomous order safety.

This module deliberately sits above strategy entry risk.  Entry risk answers
"should a new trade be opened now?"  Execution lifecycle answers the harder
operational question: "is it safe for this process to place or close signed
orders at all?"  Keeping the two separate prevents a normal risk block from
blocking emergency closes, while reconciliation, ownership, and ledger
integrity failures still fail closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .api_budget import ApiBudgetReport, api_budget_startup_block_reason
from .positions import OpenPosition, PositionsStore, bot_ownership_rejection_reason
from .reconciliation import ReconciliationReport
from .risk_controls import RiskPolicyReport, build_risk_policy_report
from .types import RuntimeConfig, StrategyConfig


@dataclass(frozen=True)
class LifecycleCheck:
    """One execution lifecycle check and the capabilities it blocks."""

    status: str
    label: str
    detail: str
    blocks_open: bool = False
    blocks_close: bool = False

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionLifecyclePlan:
    """A deterministic capability plan for autonomous execution."""

    action: str
    effective_dry_run: bool
    local_open_count: int
    local_live_open_count: int
    local_paper_open_count: int
    bot_owned_live_open_count: int
    unverified_live_open_count: int
    exchange_exposure_count: int
    external_exchange_exposure_count: int
    stale_local_position_count: int
    close_requires_reduce_only: bool
    checks: tuple[LifecycleCheck, ...]

    @property
    def can_open(self) -> bool:
        return not any(check.status == "block" and check.blocks_open for check in self.checks)

    @property
    def can_close(self) -> bool:
        return not any(check.status == "block" and check.blocks_close for check in self.checks)

    @property
    def fail_closed(self) -> bool:
        return not (self.can_open and self.can_close)

    @property
    def open_block_reasons(self) -> tuple[str, ...]:
        return tuple(
            f"{check.label}:{check.detail}"
            for check in self.checks
            if check.status == "block" and check.blocks_open
        )

    @property
    def close_block_reasons(self) -> tuple[str, ...]:
        return tuple(
            f"{check.label}:{check.detail}"
            for check in self.checks
            if check.status == "block" and check.blocks_close
        )

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["can_open"] = self.can_open
        payload["can_close"] = self.can_close
        payload["fail_closed"] = self.fail_closed
        payload["open_block_reasons"] = list(self.open_block_reasons)
        payload["close_block_reasons"] = list(self.close_block_reasons)
        return payload


def _check(
    status: str,
    label: str,
    detail: str,
    *,
    blocks_open: bool = False,
    blocks_close: bool = False,
) -> LifecycleCheck:
    return LifecycleCheck(
        status=status,
        label=label,
        detail=detail,
        blocks_open=blocks_open,
        blocks_close=blocks_close,
    )


def _risk_block_detail(report: RiskPolicyReport) -> str:
    blocked = [f"{check.label}={check.detail}" for check in report.checks if check.status == "block"]
    return "; ".join(blocked) if blocked else "passed"


def _position_ownership_rejections(positions: list[OpenPosition]) -> list[tuple[OpenPosition, str]]:
    rejections: list[tuple[OpenPosition, str]] = []
    for position in positions:
        if position.dry_run:
            continue
        reason = bot_ownership_rejection_reason(position)
        if reason is not None:
            rejections.append((position, reason))
    return rejections


def _api_budget_reason(report: ApiBudgetReport | Mapping[str, object] | None) -> str | None:
    return api_budget_startup_block_reason(report, max_used_ratio=0.80)


def build_execution_lifecycle_plan(
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    store: PositionsStore,
    *,
    action: str,
    effective_dry_run: bool | None = None,
    reconciliation: ReconciliationReport | None = None,
    risk_report: RiskPolicyReport | None = None,
    api_budget_report: ApiBudgetReport | Mapping[str, object] | None = None,
    require_reconciliation: bool = True,
    require_api_budget_headroom: bool = True,
) -> ExecutionLifecyclePlan:
    """Return the fail-closed order lifecycle plan for the current state."""

    dry_run = bool(runtime.dry_run if effective_dry_run is None else effective_dry_run)
    normalized_action = str(action or "unknown").strip().lower() or "unknown"
    checks: list[LifecycleCheck] = []

    ledger_errors = store.open_integrity_errors()
    open_positions = [] if ledger_errors else store.load_open()
    live_positions = [position for position in open_positions if not position.dry_run]
    paper_positions = [position for position in open_positions if position.dry_run]
    ownership_rejections = _position_ownership_rejections(open_positions)
    bot_owned_live_count = len(live_positions) - len(ownership_rejections)

    if ledger_errors:
        checks.append(
            _check(
                "block",
                "ledger integrity",
                "; ".join(ledger_errors),
                blocks_open=True,
                blocks_close=True,
            )
        )
    else:
        checks.append(_check("ok", "ledger integrity", f"open_positions={len(open_positions)}"))

    if dry_run:
        checks.append(_check("ok", "order mode", "paper/dry-run"))
    else:
        safe_endpoint = bool(runtime.testnet or getattr(runtime, "demo", False))
        checks.append(
            _check(
                "ok" if safe_endpoint else "block",
                "signed endpoint",
                "non-mainnet" if safe_endpoint else "mainnet is not allowed",
                blocks_open=not safe_endpoint,
                blocks_close=not safe_endpoint,
            )
        )
        has_credentials = bool(str(runtime.api_key or "").strip() and str(runtime.api_secret or "").strip())
        checks.append(
            _check(
                "ok" if has_credentials else "block",
                "signed credentials",
                "configured" if has_credentials else "missing API key/secret",
                blocks_open=not has_credentials,
                blocks_close=not has_credentials,
            )
        )

    report = risk_report or build_risk_policy_report(
        runtime,
        strategy,
        effective_dry_run=dry_run,
    )
    if report.allowed:
        checks.append(
            _check(
                "ok",
                "risk policy",
                (
                    f"notional_cap={report.notional_cap_pct:.2%} "
                    f"loss_at_stop={report.max_loss_per_trade_pct:.2%}"
                ),
            )
        )
    else:
        checks.append(
            _check(
                "block",
                "risk policy",
                _risk_block_detail(report),
                blocks_open=True,
                blocks_close=False,
            )
        )

    if not dry_run and require_api_budget_headroom:
        budget_reason = _api_budget_reason(api_budget_report)
        if budget_reason is None:
            status = "ok" if api_budget_report is not None else "warn"
            checks.append(
                _check(
                    status,
                    "api budget",
                    "headroom ok" if api_budget_report is not None else "no current sample",
                )
            )
        else:
            checks.append(
                _check(
                    "block",
                    "api budget",
                    budget_reason,
                    blocks_open=True,
                    blocks_close=False,
                )
            )

    if not dry_run and require_reconciliation:
        if reconciliation is None:
            checks.append(
                _check(
                    "block",
                    "reconciliation",
                    "missing signed account reconciliation",
                    blocks_open=True,
                    blocks_close=True,
                )
            )
        elif reconciliation.ok:
            checks.append(
                _check(
                    "ok",
                    "reconciliation",
                    (
                        f"exchange={reconciliation.exchange_exposure_count} "
                        f"local_live={reconciliation.local_live_open_count}"
                    ),
                )
            )
        else:
            reasons = ",".join(mismatch.reason for mismatch in reconciliation.mismatches) or "mismatch"
            checks.append(
                _check(
                    "block",
                    "reconciliation",
                    reasons,
                    blocks_open=True,
                    blocks_close=True,
                )
            )

    if ownership_rejections:
        detail = ",".join(f"{position.id}:{reason}" for position, reason in ownership_rejections)
        checks.append(
            _check(
                "block",
                "bot ownership",
                detail,
                blocks_open=True,
                blocks_close=True,
            )
        )
    elif live_positions:
        checks.append(_check("ok", "bot ownership", f"verified_live_positions={len(live_positions)}"))

    max_open = max(0, int(strategy.max_open_positions))
    if len(open_positions) >= max_open and max_open >= 0:
        checks.append(
            _check(
                "block",
                "open capacity",
                f"{len(open_positions)}/{max_open}",
                blocks_open=True,
                blocks_close=False,
            )
        )
    else:
        checks.append(_check("ok", "open capacity", f"{len(open_positions)}/{max_open}"))

    if normalized_action in {"stop", "close", "risk-close", "operator-stop"} and not open_positions:
        checks.append(_check("ok", "close intent", "no local open positions"))

    return ExecutionLifecyclePlan(
        action=normalized_action,
        effective_dry_run=dry_run,
        local_open_count=len(open_positions),
        local_live_open_count=len(live_positions),
        local_paper_open_count=len(paper_positions),
        bot_owned_live_open_count=bot_owned_live_count,
        unverified_live_open_count=len(ownership_rejections),
        exchange_exposure_count=0 if reconciliation is None else reconciliation.exchange_exposure_count,
        external_exchange_exposure_count=0
        if reconciliation is None
        else reconciliation.external_exchange_exposure_count,
        stale_local_position_count=0
        if reconciliation is None
        else reconciliation.stale_local_position_count,
        close_requires_reduce_only=runtime.market_type == "futures" and bool(strategy.reduce_only_on_close),
        checks=tuple(checks),
    )


def render_execution_lifecycle_plan(plan: ExecutionLifecyclePlan) -> str:
    """Render a concise operator summary for CLI/app surfaces."""

    lines = [
        "Execution lifecycle",
        (
            f"action={plan.action} dry_run={plan.effective_dry_run} "
            f"can_open={plan.can_open} can_close={plan.can_close}"
        ),
        (
            f"local_open={plan.local_open_count} live={plan.local_live_open_count} "
            f"paper={plan.local_paper_open_count} bot_owned_live={plan.bot_owned_live_open_count}"
        ),
    ]
    for check in plan.checks:
        lines.append(f"[{check.status}] {check.label}: {check.detail}")
    return "\n".join(lines)


__all__ = [
    "ExecutionLifecyclePlan",
    "LifecycleCheck",
    "build_execution_lifecycle_plan",
    "render_execution_lifecycle_plan",
]
