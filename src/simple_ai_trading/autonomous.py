"""Autonomous testnet live-trading loop with operator-controlled pause / resume / stop.

The autonomous loop is deliberately conservative:

* Requires ``testnet=True`` or ``demo=True`` on the runtime — it refuses to start otherwise.
* Honors every existing risk gate (daily trade cap, drawdown limit, cooldown,
  max open positions) through the same strategy config the live CLI uses.
* Writes a heartbeat artifact after every iteration so operators can see
  liveness from another shell or from the TUI.
* Reads a small control file each iteration.  The file contains one of
  ``RUNNING``, ``PAUSED``, ``STOPPING``.  A separate command
  (``autonomous pause/resume/stop``) just rewrites that file.
* Uses an objective-tagged model artifact so the user can flip between
  Conservative, Regular, and Aggressive without rebooting.

Safety design principles:
* No real-money execution.  The client must point at testnet; we re-verify this
  before every entry by reading ``client.base_url``.
* No credential leakage — every artifact writes through ``RuntimeConfig.public_dict``.
* No infinite fast loop — the poll interval is clamped to at least 1 second
  even if the strategy advertises ``--sleep 0``.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field, replace
from numbers import Real
from pathlib import Path
from typing import Callable, Mapping

from .api_budget import (
    ApiBudgetReport,
    api_budget_startup_block_reason,
    build_api_budget_report,
    summarize_api_budget,
)
from .api import BinanceAPIError, BinanceClient
from .binance_owned_execution import (
    close_trade_template as _close_to_trade,
    _position_order_side as _position_order_side,
    _order_text as _order_text,
    _order_float as _order_float,
    _order_fill_details as _order_fill_details,
    _order_has_fill_evidence as _order_has_fill_evidence,
    _order_exchange_status as _order_exchange_status,
    _apply_open_order as _apply_open_order,
    _apply_close_order as _apply_close_order,
    _submit_open_position as _submit_open_position,
    _submit_durable_open_position as _submit_durable_open_position,
    _submit_close_position as _submit_close_position,
    _submit_durable_close_position as _submit_durable_close_position,
    _next_close_client_order_id as _next_close_client_order_id,
)
from .binance_open_intents import OpenIntentError
from .binance_paper import BinancePaperBroker
from .execution_lifecycle import ExecutionLifecyclePlan, build_execution_lifecycle_plan
from .intervals import interval_milliseconds
from .logging_ext import configure as configure_logging
from .objective import ObjectiveSpec, get_objective
from .positions import (
    ClosedTrade,
    OpenPosition,
    PositionsStore,
    bot_client_order_id,
    bot_ownership_rejection_reason,
    compute_stats,
    compute_snapshot_stats,
    new_position_id,
    now_ms,
)
from .position_lifecycle import evaluate_position_exit
from .reconciliation import ReconciliationReport, reconcile_account_positions
from .risk_controls import market_regime_unpredictability, stop_loss_sized_notional_pct
from .storage import write_json_atomic
from .types import RuntimeConfig, StrategyConfig

STATE_RUNNING = "RUNNING"
STATE_PAUSED = "PAUSED"
STATE_STOPPING = "STOPPING"
STATE_STOPPED = "STOPPED"
_VALID_STATES = {STATE_RUNNING, STATE_PAUSED, STATE_STOPPING, STATE_STOPPED}
_VALID_EXECUTION_MODES = {"paper", "live"}
_MIN_INTERVAL_SECONDS = 1.0
_DEFAULT_AUTONOMOUS_DIR = Path("data/autonomous")
_API_BUDGET_LIVE_START_MAX_USED_RATIO = 0.80


def default_paper_journal_path(
    positions_root: Path = _DEFAULT_AUTONOMOUS_DIR,
) -> Path:
    """Return the one journal path shared by worker and operator controls."""

    return Path(positions_root) / "paper_execution.duckdb"


@dataclass
class AutonomousControl:
    """Thin filesystem-backed state machine used to pause / resume / stop."""

    path: Path = field(default_factory=lambda: _DEFAULT_AUTONOMOUS_DIR / "state.json")

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def write(
        self,
        state: str,
        *,
        note: str = "",
        execution: str | None = None,
    ) -> None:
        if state not in _VALID_STATES:
            raise ValueError(f"Invalid state {state!r}")
        mode = execution
        if mode is None:
            current_mode = self.read().get("execution")
            mode = str(current_mode) if current_mode in _VALID_EXECUTION_MODES else None
        if mode is not None and mode not in _VALID_EXECUTION_MODES:
            raise ValueError(f"Invalid execution mode {mode!r}")
        payload = {"state": state, "note": note, "ts_ms": now_ms()}
        if mode is not None:
            payload["execution"] = mode
        write_json_atomic(self.path, payload, indent=2, sort_keys=True)

    def read(self) -> dict[str, object]:
        if not self.path.exists():
            return {"state": STATE_STOPPED, "note": "", "ts_ms": 0}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"state": STATE_STOPPED, "note": "read-error", "ts_ms": 0}
        if (
            not isinstance(payload, dict)
            or payload.get("state") not in _VALID_STATES
            or (
                payload.get("execution") is not None
                and payload.get("execution") not in _VALID_EXECUTION_MODES
            )
        ):
            return {"state": STATE_STOPPED, "note": "malformed", "ts_ms": 0}
        return payload

    def state(self) -> str:
        return str(self.read().get("state") or STATE_STOPPED)


@dataclass
class Heartbeat:
    """Serializable snapshot of the autonomous loop's current status."""

    iteration: int
    state: str
    last_signal: float
    last_side: str
    last_price: float
    open_positions: int
    realized_pnl: float
    unrealized_pnl: float
    objective: str
    updated_at_ms: int
    message: str = ""
    ai_assist_status: str = ""
    ai_assist_case_id: str = ""

    def write(self, path: Path) -> None:
        write_json_atomic(path, asdict(self), indent=2, sort_keys=True)


@dataclass
class AutonomousConfig:
    """All knobs the autonomous loop exposes.

    Every field has a safe testnet-first default.  ``poll_seconds`` is clamped
    at runtime to ``_MIN_INTERVAL_SECONDS`` so even a misconfigured deployment
    never busy-waits.
    """

    objective: str = "conservative"
    poll_seconds: float = 30.0
    min_unrealized_close_pct: float | None = None  # auto-close at this pnl%
    max_unrealized_close_pct: float | None = None  # auto-close above this pnl%
    stop_after_iterations: int | None = None  # None = infinite
    heartbeat_every: int = 1  # write heartbeat every N iterations
    dry_run: bool = True  # paper by default — operator must explicitly live
    control_path: Path = field(
        default_factory=lambda: _DEFAULT_AUTONOMOUS_DIR / "state.json"
    )
    heartbeat_path: Path = field(
        default_factory=lambda: _DEFAULT_AUTONOMOUS_DIR / "heartbeat.json"
    )
    positions_root: Path = field(default_factory=lambda: _DEFAULT_AUTONOMOUS_DIR)
    log_path: Path = field(
        default_factory=lambda: _DEFAULT_AUTONOMOUS_DIR / "autonomous.log"
    )
    paper_journal_path: Path | None = None
    starting_reference_cash: float = 1000.0


def ensure_testnet(runtime: RuntimeConfig) -> None:
    """Raise if the runtime is not configured for a non-mainnet exchange."""

    if not (runtime.testnet or getattr(runtime, "demo", False)):
        raise RuntimeError(
            "Autonomous mode refuses to start unless runtime.testnet=True or runtime.demo=True. "
            "This phase blocks real-money execution."
        )


def ensure_credentials(runtime: RuntimeConfig, cfg: AutonomousConfig) -> None:
    """Live mode requires API credentials.  Paper mode does not."""

    if cfg.dry_run:
        return
    if not runtime.api_key or not runtime.api_secret:
        raise RuntimeError(
            "Autonomous live mode requires runtime.api_key and runtime.api_secret."
        )


def ensure_api_budget_headroom(
    runtime: RuntimeConfig,
    client: BinanceClient,
    *,
    max_used_ratio: float = _API_BUDGET_LIVE_START_MAX_USED_RATIO,
) -> ApiBudgetReport:
    """Fail closed before autonomous live startup when Binance rate budget is tight."""

    fetch_exchange_info = getattr(client, "get_exchange_info", None)
    exchange_info = fetch_exchange_info() if callable(fetch_exchange_info) else None
    request_info = dict(getattr(client, "last_request_info", {}) or {})
    report = build_api_budget_report(
        market_type=runtime.market_type,
        exchange_info=exchange_info if isinstance(exchange_info, Mapping) else None,
        request_info=request_info,
    )
    reason = api_budget_startup_block_reason(report, max_used_ratio=max_used_ratio)
    if reason is not None:
        raise RuntimeError(reason)
    return report


@dataclass
class Decision:
    """A single iteration's decision — injected so the loop stays testable."""

    side: str  # "LONG" / "SHORT" / "FLAT"
    confidence: float
    mark_price: float
    size_multiplier: float = 1.0
    meta_label_enabled: bool = False
    meta_label_evidence_schema_version: str = ""
    meta_label_action: str = ""
    meta_label_reason: str = ""
    meta_label_signal_strength: float = 0.0
    meta_label_validation_minimum_sample_count: int = 0
    meta_label_validation_minimum_precision: float = 0.0
    meta_label_validation_sample_count: int = 0
    meta_label_validation_precision: float = 0.0
    meta_label_expected_after_cost_return: float = 0.0
    meta_label_expected_after_cost_pnl: float = 0.0
    meta_label_validation_bootstrap_samples: int = 0
    meta_label_validation_bootstrap_confidence: float = 0.0
    meta_label_validation_bootstrap_block_length: int = 0
    meta_label_validation_bootstrap_lower_after_cost_return: float = 0.0
    regime: str = ""
    regime_confidence: float = 0.0
    regime_notes: tuple[str, ...] = ()
    regime_unpredictability_score: float | None = None
    observed_at_ms: int = 0
    model_features: tuple[float, ...] = ()
    ai_evidence: Mapping[str, object] = field(default_factory=dict)
    ai_assist_mode: str = ""
    ai_assist_status: str = ""
    ai_assist_case_id: str = ""
    ai_assist_action: str = ""
    ai_assist_risk_multiplier: float = 0.0
    ai_assist_reason: str = ""
    ai_assist_entry_ready: bool = True
    model_entry_ready: bool = True
    model_entry_reason: str = ""


DecisionFn = Callable[
    [BinanceClient, RuntimeConfig, StrategyConfig, ObjectiveSpec], Decision
]
ReconcileFn = Callable[
    [BinanceClient, RuntimeConfig, PositionsStore], ReconciliationReport
]


@dataclass(frozen=True)
class CapitalGuard:
    """Hard local capital-at-risk decision for autonomous execution."""

    allowed: bool
    reason: str
    daily_loss: float
    session_loss: float
    consecutive_losses: int
    force_close: bool = False
    realized_peak_drawdown: float | None = None
    capital_policy_sha256: str | None = None


@dataclass(frozen=True)
class CloseAllReport:
    """Result of closing locally tracked autonomous positions."""

    closed: int
    skipped: int
    failed: int
    partial: int = 0
    failures: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.partial == 0


def _is_partial_close(position: OpenPosition, trade: ClosedTrade) -> bool:
    open_qty = max(0.0, float(position.qty))
    close_qty = max(0.0, float(trade.qty))
    tolerance = max(1e-12, open_qty * 1e-8)
    return open_qty > 0.0 and close_qty < open_qty - tolerance


def _default_reconcile(
    client: BinanceClient,
    runtime: RuntimeConfig,
    store: PositionsStore,
) -> ReconciliationReport:
    return reconcile_account_positions(client.get_account(), runtime, store)


def _lifecycle_reasons(plan: ExecutionLifecyclePlan, *, capability: str) -> str:
    reasons = (
        plan.open_block_reasons if capability == "open" else plan.close_block_reasons
    )
    return "; ".join(reasons) or "execution lifecycle blocked"


def _loop_lifecycle_plan(
    client: BinanceClient,
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    cfg: AutonomousConfig,
    store: PositionsStore,
    *,
    action: str,
    reconcile: ReconcileFn,
    api_budget_report: ApiBudgetReport | None = None,
    require_api_budget_headroom: bool = True,
) -> ExecutionLifecyclePlan:
    reconciliation = None
    if not cfg.dry_run:
        reconciliation = reconcile(client, runtime, store)
    return build_execution_lifecycle_plan(
        runtime,
        strategy,
        store,
        action=action,
        effective_dry_run=cfg.dry_run,
        reconciliation=reconciliation,
        api_budget_report=api_budget_report,
        require_api_budget_headroom=require_api_budget_headroom,
    )


def _default_decision(
    client: BinanceClient,
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    objective: ObjectiveSpec,
) -> Decision:
    """Placeholder decision that only reads the ticker.  Real inference is wired by callers.

    Keeping the default free of model calls lets the module be imported in
    environments without training data.  The autonomous CLI command supplies a
    real decision function that runs features + model prediction.
    """

    price, _ts = client.get_symbol_price(runtime.symbol)
    del strategy, objective
    return Decision(side="FLAT", confidence=0.0, mark_price=float(price))


def _evaluate_auto_close(
    position: OpenPosition,
    mark_price: float,
    cfg: AutonomousConfig,
    strategy: StrategyConfig,
    *,
    decision_side: str = "FLAT",
    observed_at_ms: int = 0,
    interval_ms: int = 1_000,
) -> tuple[bool, str]:
    """Return (should_close, reason) for an open position at the given mark."""

    from .binance_spot_cash import native_spot_cash_mark

    cash_mark = native_spot_cash_mark(position, mark_price)
    pnl_pct = (
        float(cash_mark.return_fraction)
        if cash_mark is not None
        else position.unrealized_pnl_pct(mark_price)
    )
    if (
        cfg.max_unrealized_close_pct is not None
        and pnl_pct >= cfg.max_unrealized_close_pct
    ):
        return True, f"auto-take-profit@{cfg.max_unrealized_close_pct:+.2%}"
    if (
        cfg.min_unrealized_close_pct is not None
        and pnl_pct <= cfg.min_unrealized_close_pct
    ):
        return True, f"auto-stop-loss@{cfg.min_unrealized_close_pct:+.2%}"
    signal_direction = {"LONG": 1, "SHORT": -1}.get(str(decision_side).upper(), 0)
    elapsed_ms = (
        max(0, int(observed_at_ms) - int(position.opened_at_ms))
        if observed_at_ms > 0
        else 0
    )
    bars_held = elapsed_ms // max(1, int(interval_ms))
    lifecycle_exit = evaluate_position_exit(
        position_side=1 if str(position.side).upper() == "LONG" else -1,
        signal_direction=signal_direction,
        current_pnl_pct=pnl_pct,
        bars_held=bars_held,
        flat_signal_streak=0,
        stop_loss_pct=strategy.stop_loss_pct,
        take_profit_pct=strategy.take_profit_pct,
        min_position_hold_bars=strategy.min_position_hold_bars,
        flat_signal_exit_grace_bars=strategy.flat_signal_exit_grace_bars,
        max_position_hold_bars=strategy.max_position_hold_bars,
        allow_flat_signal_exit=False,
    )
    reason = lifecycle_exit.reason
    if reason == "take_profit_close":
        reason = f"take-profit@{strategy.take_profit_pct:+.2%}"
    elif reason == "stop_loss_close":
        reason = f"stop-loss@{strategy.stop_loss_pct:+.2%}"
    elif reason == "time_limit":
        reason = f"time-limit@{bars_held}bars"
    elif reason == "signal_reverse":
        reason = "signal-reverse"
    return lifecycle_exit.should_close, reason


def _open_position_from_decision(
    decision: Decision,
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    objective: ObjectiveSpec,
    cfg: AutonomousConfig,
    *,
    clock=time.time,
) -> OpenPosition:
    """Build a position record from a decision + runtime state."""

    if decision.model_entry_ready is not True:
        raise ValueError("position entry requires qualified model-opening evidence")
    if decision.ai_assist_entry_ready is not True:
        raise ValueError("position entry requires a completed AI pre-entry review")
    numeric_rejection = _opening_decision_numeric_rejection(decision)
    if decision.side not in {"LONG", "SHORT"} or numeric_rejection:
        raise ValueError(
            numeric_rejection or "position entry requires a directional decision"
        )
    price = float(decision.mark_price)
    notional_pct = stop_loss_sized_notional_pct(
        strategy, runtime.market_type, leverage=strategy.leverage
    )
    size_multiplier = _decision_size_multiplier(decision)
    if size_multiplier <= 0.0:
        raise ValueError("position entry requires a positive qualified size")
    reference = _decision_number(cfg.starting_reference_cash)
    if reference is None or reference <= 0.0:
        raise ValueError("position entry requires finite positive reference cash")
    target_notional = max(0.0, reference * notional_pct * size_multiplier)
    qty = max(0.0, target_notional / price)
    notional = qty * price
    if not all(math.isfinite(value) and value > 0.0 for value in (qty, notional)):
        raise ValueError(
            "position entry requires finite positive quantity and notional"
        )
    position_id = new_position_id()
    return OpenPosition(
        id=position_id,
        symbol=runtime.symbol,
        market_type=runtime.market_type,
        side=decision.side,
        qty=qty,
        entry_price=price,
        leverage=float(strategy.leverage),
        opened_at_ms=int(clock() * 1000),
        notional=notional,
        strategy_profile="autonomous",
        objective=objective.name,
        dry_run=cfg.dry_run,
        stop_loss_pct=strategy.stop_loss_pct,
        take_profit_pct=strategy.take_profit_pct,
        open_client_order_id=bot_client_order_id(position_id, "open"),
        exchange_status="paper" if cfg.dry_run else "pending_open",
        entry_fees=notional * max(0.0, float(strategy.taker_fee_bps)) / 10_000.0,
        ai_review_mode=decision.ai_assist_mode,
        ai_review_case_id=decision.ai_assist_case_id,
        ai_review_status=decision.ai_assist_status,
    )


def close_all_open_positions(
    store: PositionsStore,
    mark_price: float | None,
    reason: str,
    *,
    clock=time.time,
    client: BinanceClient | None = None,
    reduce_only: bool = True,
    taker_fee_bps: float = 0.0,
    paper_broker: BinancePaperBroker | None = None,
) -> int:
    """Close verified positions through the configured live or paper broker."""

    return close_tracked_open_positions(
        store,
        mark_price,
        reason,
        clock=clock,
        client=client,
        reduce_only=reduce_only,
        taker_fee_bps=taker_fee_bps,
        paper_broker=paper_broker,
    ).closed


def close_tracked_open_positions(
    store: PositionsStore,
    mark_price: float | None,
    reason: str,
    *,
    clock=time.time,
    client: BinanceClient | None = None,
    reduce_only: bool = True,
    taker_fee_bps: float = 0.0,
    paper_broker: BinancePaperBroker | None = None,
) -> CloseAllReport:
    """Close bot-owned local ledger positions and preserve anything uncertain."""

    positions = list(store.load_open())
    paper_positions = [position for position in positions if position.dry_run]
    if paper_positions and paper_broker is not None:
        paper_reconciliation = paper_broker.reconcile_positions(store)
        if not paper_reconciliation.can_close:
            reason_text = (
                f"paper-reconciliation-blocked:{paper_reconciliation.asdict()}"
            )
            return CloseAllReport(
                closed=0,
                skipped=0,
                failed=len(paper_positions),
                partial=0,
                failures=tuple(
                    f"{position.id}:{reason_text}" for position in paper_positions
                ),
            )

    closed = 0
    skipped = 0
    failed = 0
    partial = 0
    failures: list[str] = []
    for position in positions:
        ownership_rejection = bot_ownership_rejection_reason(position)
        if ownership_rejection is not None:
            skipped += 1
            failures.append(f"{position.id}:{ownership_rejection}")
            continue
        close_price = (
            float(mark_price)
            if mark_price and mark_price > 0
            else float(position.entry_price)
        )
        trade = _close_to_trade(
            position,
            close_price,
            reason,
            clock=clock,
            fees=max(0.0, float(position.entry_fees)),
        )
        try:
            if not position.dry_run:
                if client is None:
                    raise BinanceAPIError("live close requires Binance client")
                trade = _submit_durable_close_position(
                    client,
                    position,
                    trade,
                    store,
                    reduce_only=reduce_only,
                    taker_fee_bps=taker_fee_bps,
                )
            else:
                if paper_broker is None:
                    raise ValueError("paper close requires shared execution broker")
                paper_trade, paper_result = paper_broker.close_position(
                    position,
                    reason=reason,
                )
                if paper_trade is None:
                    raise ValueError(
                        f"paper close unresolved:{paper_result.state}:{paper_result.reason}"
                    )
                trade = paper_trade
        except (BinanceAPIError, ValueError) as exc:
            failed += 1
            failures.append(f"{position.id}:{exc}")
            continue
        if position.dry_run:
            store.record_close_result(position, trade)
        closed += 1
        if _is_partial_close(position, trade):
            partial += 1
            failures.append(
                f"{position.id}:partial-close {trade.qty:.12g}/{position.qty:.12g}"
            )
    return CloseAllReport(
        closed=closed,
        skipped=skipped,
        failed=failed,
        partial=partial,
        failures=tuple(failures),
    )


def _stop_close_mark_price(
    client: BinanceClient,
    runtime: RuntimeConfig,
    store: PositionsStore,
    last_mark_price: float | None,
    logger: logging.Logger,
) -> float | None:
    if last_mark_price is not None and last_mark_price > 0:
        return last_mark_price
    try:
        price, _ts = client.get_symbol_price(runtime.symbol)
        parsed = float(price)
        if parsed > 0:
            return parsed
    except Exception as exc:  # noqa: BLE001 - stop must not be blocked by a quote failure
        logger.warning("autonomous stop quote unavailable: %s", exc)
    opens = store.load_open()
    return opens[0].entry_price if opens else None


@dataclass
class LoopResult:
    """What ``run_loop`` returns once it exits."""

    iterations: int
    final_state: str
    heartbeats_written: int
    closed_trades: int
    opened_trades: int
    exit_reason: str
    skipped_entries: int = 0


@dataclass(frozen=True)
class EntryGate:
    """Pre-entry risk decision used by the autonomous loop."""

    allowed: bool
    reason: str
    open_positions: int
    daily_entries: int
    cooldown_remaining_ms: int
    drawdown: float
    directional_confidence: float


def _safe_day_ms(timestamp_ms: int) -> int:
    return int(timestamp_ms // (24 * 60 * 60 * 1000))


def _decision_number(value: object) -> float | None:
    """Accept finite real decision values without boolean or string coercion."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _decision_unit_fraction(value: object) -> float | None:
    number = _decision_number(value)
    return number if number is not None and 0.0 <= number <= 1.0 else None


def _opening_decision_numeric_rejection(decision: Decision) -> str | None:
    if _decision_unit_fraction(decision.confidence) is None:
        return "decision-confidence-unqualified"
    if _decision_unit_fraction(decision.size_multiplier) is None:
        return "decision-size-unqualified"
    price = _decision_number(decision.mark_price)
    if price is None or price <= 0.0:
        return "decision-mark-unqualified"
    return None


def _directional_confidence(decision: Decision) -> float:
    """Convert only a qualified probability into directional confidence."""
    confidence = _decision_unit_fraction(decision.confidence)
    if confidence is None:
        return 0.0
    if decision.side == "SHORT":
        return 1.0 - confidence
    return confidence


def _decision_size_multiplier(decision: Decision) -> float:
    multiplier = _decision_unit_fraction(decision.size_multiplier)
    return multiplier if multiplier is not None else 0.0


def _daily_entry_count(store: PositionsStore, day: int) -> int:
    opened = sum(
        1
        for position in store.load_open()
        if _safe_day_ms(position.opened_at_ms) == day
    )
    closed = sum(
        1 for trade in store.load_ledger() if _safe_day_ms(trade.opened_at_ms) == day
    )
    return opened + closed


def _loss_budget_guard(
    store: PositionsStore,
    mark_price: float | None,
    strategy: StrategyConfig,
    cfg: AutonomousConfig,
    *,
    now_ms_value: int,
    mark_symbol: str = "",
    mark_market_type: str = "",
) -> CapitalGuard:
    """Apply cash-aware native loss limits; unqualified metrics never admit risk."""
    limits = (
        strategy.max_daily_loss_pct,
        strategy.max_session_loss_pct,
        strategy.max_drawdown_limit,
    )
    qualified_limits = tuple(_decision_number(value) for value in limits)
    if (
        any(value is None or value < 0 for value in qualified_limits)
        or type(strategy.max_consecutive_losses) is not int
        or strategy.max_consecutive_losses < 0
    ):
        return CapitalGuard(False, "risk-policy-loss-limits-unqualified", 0.0, 0.0, 0)
    if type(cfg.dry_run) is not bool:
        return CapitalGuard(
            False, "risk-policy-execution-mode-unqualified", 0.0, 0.0, 0
        )
    policy_digest = None
    policy_blocked = False
    if not cfg.dry_run:
        from .binance_capital_policy import BinanceCapitalPolicyStore
        from .binance_runtime_control import RuntimeControlError

        try:
            policy_digest = BinanceCapitalPolicyStore(store).assert_effective(
                strategy,
                cfg.starting_reference_cash,
                symbol=mark_symbol,
                market_type=mark_market_type,
            )
        except (
            OSError,
            sqlite3.Error,
            ValueError,
            TypeError,
            KeyError,
            RuntimeControlError,
        ):
            policy_blocked = True
    mark = _decision_number(mark_price)
    if mark is not None and mark <= 0.0:
        mark = None
    try:
        snapshot = store.load_snapshot(strict=True)
    except (OSError, sqlite3.Error, ValueError, TypeError, OverflowError):
        return CapitalGuard(False, "cash-valuation-ledger-unqualified", 0.0, 0.0, 0)
    opens, closed = snapshot
    consecutive_losses = 0
    for trade in reversed(closed):
        if trade.realized_pnl >= 0.0:
            break
        consecutive_losses += 1
    try:
        reference = float(cfg.starting_reference_cash)
    except (TypeError, ValueError, OverflowError):
        reference = float("nan")
    if (
        isinstance(cfg.starting_reference_cash, bool)
        or not math.isfinite(reference)
        or reference <= 0
    ):
        return CapitalGuard(
            False, "cash-valuation-reference-unqualified", 0.0, 0.0, consecutive_losses
        )
    # A scalar mark can value only its exact instrument. Closing inventory does
    # not erase the denomination of realized cash, including legacy records.
    if opens or closed:
        if (
            not mark_symbol
            or mark_market_type not in {"spot", "futures"}
            or opens
            and (
                mark is None
                or any(
                    (position.symbol, position.market_type)
                    != (mark_symbol, mark_market_type)
                    for position in opens
                )
            )
        ):
            return CapitalGuard(
                False,
                "cash-valuation-instrument-unqualified",
                0.0,
                0.0,
                consecutive_losses,
            )
        from .assets import symbol_base_for_supported_quote

        base = symbol_base_for_supported_quote(mark_symbol)
        quote = mark_symbol[len(base) :] if base else ""
        if not quote or any(not trade.symbol.endswith(quote) for trade in closed):
            return CapitalGuard(
                False, "cash-valuation-quote-unqualified", 0.0, 0.0, consecutive_losses
            )
    try:
        stats = compute_snapshot_stats(
            snapshot,
            mark_price=mark,
            starting_reference_cash=cfg.starting_reference_cash,
            include_native_entry_costs=True,
        )
    except OpenIntentError:
        return CapitalGuard(
            False, "cash-valuation-fees-unqualified", 0.0, 0.0, consecutive_losses
        )
    day = _safe_day_ms(now_ms_value)
    realized_today = sum(
        trade.realized_pnl
        for trade in closed
        if _safe_day_ms(trade.closed_at_ms) == day
    )
    unrealized = float(stats.unrealized_pnl) if mark is not None else 0.0
    daily_cash = realized_today + unrealized
    session_cash = float(stats.realized_pnl) + unrealized
    # Finite records can overflow when aggregated. NaN must never become zero
    # loss through max(), nor can infinite profit offset real losses.
    if not all(math.isfinite(value) for value in (daily_cash, session_cash)):
        return CapitalGuard(
            False, "cash-valuation-nonfinite", 0.0, 0.0, consecutive_losses
        )
    daily_loss = max(0.0, -daily_cash / reference)
    session_loss = max(0.0, -session_cash / reference)
    if not math.isfinite(daily_loss) or not math.isfinite(session_loss):
        return CapitalGuard(
            False, "cash-valuation-nonfinite", 0.0, 0.0, consecutive_losses
        )
    from .capital_drawdown import settled_cash_drawdown

    try:
        cash_peak = settled_cash_drawdown(
            closed,
            unrealized_pnl_quote=unrealized,
            reference_cash_quote=reference,
            observed_at_ms=now_ms_value,
        )
        drawdown = float(cash_peak.reference_fraction)
    except (ValueError, TypeError):
        return CapitalGuard(
            False, "cash-valuation-history-unqualified", 0.0, 0.0, consecutive_losses
        )
    except OverflowError:
        return CapitalGuard(
            False, "cash-valuation-nonfinite", 0.0, 0.0, consecutive_losses
        )
    if not math.isfinite(drawdown):
        return CapitalGuard(
            False, "cash-valuation-nonfinite", 0.0, 0.0, consecutive_losses
        )
    if strategy.max_daily_loss_pct > 0.0 and daily_loss >= strategy.max_daily_loss_pct:
        return CapitalGuard(
            False,
            f"daily-loss-lockout:{daily_loss:.2%}",
            daily_loss,
            session_loss,
            consecutive_losses,
            force_close=True,
            realized_peak_drawdown=drawdown,
            capital_policy_sha256=policy_digest,
        )
    if (
        strategy.max_session_loss_pct > 0.0
        and session_loss >= strategy.max_session_loss_pct
    ):
        return CapitalGuard(
            False,
            f"session-loss-lockout:{session_loss:.2%}",
            daily_loss,
            session_loss,
            consecutive_losses,
            force_close=True,
            realized_peak_drawdown=drawdown,
            capital_policy_sha256=policy_digest,
        )
    if strategy.max_drawdown_limit > 0.0 and drawdown >= strategy.max_drawdown_limit:
        return CapitalGuard(
            False,
            f"drawdown-lockout:{drawdown:.2%}",
            daily_loss,
            session_loss,
            consecutive_losses,
            force_close=True,
            realized_peak_drawdown=drawdown,
            capital_policy_sha256=policy_digest,
        )
    if (
        strategy.max_consecutive_losses > 0
        and consecutive_losses >= strategy.max_consecutive_losses
    ):
        return CapitalGuard(
            False,
            f"loss-streak-lockout:{consecutive_losses}",
            daily_loss,
            session_loss,
            consecutive_losses,
            force_close=False,
            realized_peak_drawdown=drawdown,
            capital_policy_sha256=policy_digest,
        )
    return CapitalGuard(
        not policy_blocked,
        "risk-policy-activation-unqualified" if policy_blocked else "allowed",
        daily_loss,
        session_loss,
        consecutive_losses,
        realized_peak_drawdown=drawdown,
        capital_policy_sha256=policy_digest,
    )


def _capital_blocks_owned_management(
    guard: CapitalGuard, store: PositionsStore
) -> bool:
    """Keep qualified owned exits running when only new-entry policy is unavailable."""
    if guard.allowed:
        return False
    if guard.force_close:
        return True
    return guard.reason != "risk-policy-activation-unqualified" or not store.load_open(
        strict=True
    )


def _last_activity_ms(store: PositionsStore) -> int:
    candidates = [position.opened_at_ms for position in store.load_open()]
    candidates.extend(trade.closed_at_ms for trade in store.load_ledger())
    return max(candidates, default=0)


def _entry_gate(
    store: PositionsStore,
    decision: Decision,
    strategy: StrategyConfig,
    cfg: AutonomousConfig,
    objective: ObjectiveSpec,
    *,
    now_ms_value: int,
    symbol: str = "",
    market_type: str = "",
) -> EntryGate:
    """Return the deterministic pre-entry gate decision for a signal."""

    opens = store.load_open()
    max_open = int(strategy.max_open_positions)
    day = _safe_day_ms(now_ms_value)
    daily_entries = _daily_entry_count(store, day)
    capital_guard = _loss_budget_guard(
        store,
        decision.mark_price,
        strategy,
        cfg,
        now_ms_value=now_ms_value,
        mark_symbol=symbol,
        mark_market_type=market_type,
    )
    drawdown = (
        capital_guard.realized_peak_drawdown
        if capital_guard.realized_peak_drawdown is not None
        else capital_guard.session_loss
    )
    confidence = _directional_confidence(decision)
    min_confidence = (
        objective.training.signal_threshold
        if objective.training is not None
        else strategy.signal_threshold
    )

    cooldown_ms = max(0, int(strategy.cooldown_minutes)) * 60 * 1000
    last_activity = _last_activity_ms(store)
    cooldown_remaining = 0
    if cooldown_ms > 0 and last_activity > 0:
        cooldown_remaining = max(0, cooldown_ms - max(0, now_ms_value - last_activity))
    if decision.regime_unpredictability_score is None:
        regime_score = (
            market_regime_unpredictability(
                decision.regime,
                decision.regime_confidence,
                decision.regime_notes,
            )
            if decision.regime
            else 0.0
        )
    else:
        try:
            regime_score = float(decision.regime_unpredictability_score)
        except (TypeError, ValueError, OverflowError):
            regime_score = 1.0
        if not math.isfinite(regime_score):
            regime_score = 1.0
    regime_score = max(0.0, min(1.0, regime_score))
    regime_limit = max(0.0, min(1.0, float(strategy.max_regime_unpredictability)))
    normalized_symbol = str(symbol).strip().upper()
    normalized_market_type = str(market_type).strip().lower()
    same_instrument_open = bool(
        normalized_symbol
        and any(
            position.symbol.upper() == normalized_symbol
            and (
                not normalized_market_type
                or position.market_type.lower() == normalized_market_type
            )
            for position in opens
        )
    )

    if decision.model_entry_ready is not True:
        reason = decision.model_entry_reason or "model-opening-evidence-unavailable"
    elif decision.side not in {"LONG", "SHORT"}:
        reason = "flat-signal"
    elif numeric_rejection := _opening_decision_numeric_rejection(decision):
        reason = numeric_rejection
    elif same_instrument_open:
        reason = f"same-instrument-open:{normalized_symbol}:{normalized_market_type or 'any'}"
    elif decision.ai_assist_entry_ready is not True:
        reason = decision.ai_assist_reason or "ai-pre-entry-review-unavailable"
    elif _decision_size_multiplier(decision) <= 0.0:
        reason = decision.meta_label_reason or "meta-label-skip"
    elif confidence < min_confidence:
        reason = f"low-confidence<{min_confidence:.3f}"
    elif max_open <= 0:
        reason = "max-open-disabled"
    elif len(opens) >= max_open:
        reason = f"max-open-reached:{len(opens)}/{max_open}"
    elif int(strategy.max_trades_per_day) > 0 and daily_entries >= int(
        strategy.max_trades_per_day
    ):
        reason = f"daily-cap-reached:{daily_entries}/{int(strategy.max_trades_per_day)}"
    elif cooldown_remaining > 0:
        reason = f"cooldown-active:{cooldown_remaining}ms"
    elif regime_score > regime_limit:
        regime_name = decision.regime or "unknown"
        reason = (
            f"regime-unpredictable:{regime_name}:{regime_score:.2f}>{regime_limit:.2f}"
        )
    elif not capital_guard.allowed:
        reason = capital_guard.reason
    else:
        reason = "allowed"

    return EntryGate(
        allowed=reason == "allowed",
        reason=reason,
        open_positions=len(opens),
        daily_entries=daily_entries,
        cooldown_remaining_ms=cooldown_remaining,
        drawdown=drawdown,
        directional_confidence=confidence,
    )


def run_loop(
    client: BinanceClient,
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    cfg: AutonomousConfig,
    *,
    decision_fn: DecisionFn = _default_decision,
    sleep=time.sleep,
    clock=time.time,
    logger: logging.Logger | None = None,
    reconcile_fn: ReconcileFn | None = None,
) -> LoopResult:
    """Run the autonomous loop until the control file requests a stop."""

    ensure_testnet(runtime)
    ensure_credentials(runtime, cfg)
    objective = get_objective(cfg.objective)
    logger = logger or configure_logging(path=cfg.log_path)
    api_budget_report: ApiBudgetReport | None = None
    if not cfg.dry_run:
        api_budget_report = ensure_api_budget_headroom(runtime, client)
        logger.info("autonomous api-budget %s", summarize_api_budget(api_budget_report))
    store = PositionsStore(root=cfg.positions_root)
    paper_broker: BinancePaperBroker | None = None
    if cfg.dry_run:
        journal_path = (
            Path(cfg.paper_journal_path)
            if cfg.paper_journal_path is not None
            else default_paper_journal_path(cfg.positions_root)
        )
        paper_broker = BinancePaperBroker(
            journal_path,
            client,
            market_type=runtime.market_type,
            taker_fee_bps=strategy.taker_fee_bps,
        )
        try:
            paper_startup = paper_broker.reconcile_positions(store)
            needs_new_entry_permission = not store.load_open()
            unsafe_startup = not paper_startup.can_close or (
                needs_new_entry_permission and not paper_startup.can_open
            )
        except Exception:
            paper_broker.close()
            raise
        if unsafe_startup:
            paper_broker.close()
            raise RuntimeError(
                "Autonomous paper mode refuses to start with unsafe execution lifecycle: "
                f"{paper_startup.asdict()}"
            )
    control = AutonomousControl(path=cfg.control_path)
    reconcile = reconcile_fn or _default_reconcile
    if not cfg.dry_run:
        startup_lifecycle = _loop_lifecycle_plan(
            client,
            runtime,
            strategy,
            cfg,
            store,
            action="start",
            reconcile=reconcile,
            api_budget_report=api_budget_report,
        )
        needs_new_entry_permission = startup_lifecycle.local_open_count == 0
        if not startup_lifecycle.can_close or (
            needs_new_entry_permission and not startup_lifecycle.can_open
        ):
            raise RuntimeError(
                "Autonomous live mode refuses to start with unsafe execution lifecycle: "
                f"{startup_lifecycle.asdict()}"
            )
    control.write(
        STATE_RUNNING,
        note=f"objective={objective.name}",
        execution="paper" if cfg.dry_run else "live",
    )
    poll = max(_MIN_INTERVAL_SECONDS, float(cfg.poll_seconds))

    iteration = 0
    heartbeats = 0
    closed = 0
    opened = 0
    skipped = 0
    exit_reason = "requested-stop"
    final_state = STATE_STOPPING
    last_mark_price: float | None = None
    network_errors = 0
    recovery_pending = False
    try:
        while True:
            iteration += 1
            state = control.state()
            if state == STATE_STOPPING:
                mark_price = _stop_close_mark_price(
                    client, runtime, store, last_mark_price, logger
                )
                if not cfg.dry_run:
                    try:
                        stop_lifecycle = _loop_lifecycle_plan(
                            client,
                            runtime,
                            strategy,
                            cfg,
                            store,
                            action="stop",
                            reconcile=reconcile,
                            api_budget_report=api_budget_report,
                            require_api_budget_headroom=False,
                        )
                    except BinanceAPIError as exc:
                        logger.error(
                            "autonomous iter=%d operator-stop reconciliation failed: %s",
                            iteration,
                            exc,
                        )
                        exit_reason = "operator-stop-reconciliation-failed"
                        break
                    if not stop_lifecycle.can_close:
                        logger.error(
                            "autonomous iter=%d operator-stop lifecycle blocked: %s",
                            iteration,
                            _lifecycle_reasons(stop_lifecycle, capability="close"),
                        )
                        exit_reason = "operator-stop-lifecycle-blocked"
                        break
                close_report = close_tracked_open_positions(
                    store,
                    mark_price,
                    "operator-stop",
                    clock=clock,
                    client=None if cfg.dry_run else client,
                    reduce_only=runtime.market_type == "futures"
                    and strategy.reduce_only_on_close,
                    taker_fee_bps=strategy.taker_fee_bps,
                    paper_broker=paper_broker,
                )
                closed += close_report.closed
                if close_report.closed:
                    logger.info(
                        "autonomous iter=%d force-close-open=%d reason=operator-stop",
                        iteration,
                        close_report.closed,
                    )
                if not close_report.ok or close_report.skipped:
                    logger.error(
                        "autonomous iter=%d operator-stop incomplete close report=%s",
                        iteration,
                        close_report,
                    )
                    exit_reason = "operator-stop-close-incomplete"
                else:
                    exit_reason = "operator-stop"
                break
            if state == STATE_PAUSED:
                logger.info("autonomous iter=%d paused", iteration)
                sleep(poll)
                continue
            try:
                set_entry_review_required = getattr(
                    decision_fn,
                    "set_entry_review_required",
                    None,
                )
                if callable(set_entry_review_required):
                    open_positions = store.load_open()
                    same_instrument_open = any(
                        position.symbol.upper() == runtime.symbol.upper()
                        and position.market_type.lower() == runtime.market_type.lower()
                        for position in open_positions
                    )
                    set_entry_review_required(
                        not recovery_pending
                        and int(strategy.max_open_positions) > len(open_positions)
                        and not same_instrument_open
                    )
                decision = decision_fn(client, runtime, strategy, objective)
            except BinanceAPIError as err:
                network_errors += 1
                recovery_pending = True
                logger.warning(
                    "autonomous iter=%d binance-error=%s network_errors=%d/%d",
                    iteration,
                    err,
                    network_errors,
                    strategy.max_network_errors,
                )
                if iteration % max(1, cfg.heartbeat_every) == 0:
                    stats = compute_stats(
                        store,
                        mark_price=last_mark_price,
                        starting_reference_cash=cfg.starting_reference_cash,
                    )
                    Heartbeat(
                        iteration=iteration,
                        state=STATE_RUNNING,
                        last_signal=0.0,
                        last_side="NETWORK_ERROR",
                        last_price=float(last_mark_price or 0.0),
                        open_positions=stats.open_positions,
                        realized_pnl=stats.realized_pnl,
                        unrealized_pnl=stats.unrealized_pnl,
                        objective=objective.name,
                        updated_at_ms=int(clock() * 1000),
                        message=(
                            f"network-interruption:{network_errors}/{strategy.max_network_errors}; "
                            "reconcile-before-resume"
                        ),
                    ).write(cfg.heartbeat_path)
                    heartbeats += 1
                sleep(poll)
                continue
            except Exception as err:  # noqa: BLE001 - loop-wide guard
                logger.error("autonomous iter=%d decision-error=%s", iteration, err)
                exit_reason = "decision-exception"
                break

            decision_mark = _decision_number(decision.mark_price)
            if decision_mark is None or decision_mark <= 0.0:
                logger.error("autonomous iter=%d decision-mark-unqualified", iteration)
                exit_reason = "decision-mark-unqualified"
                break
            decision = replace(decision, mark_price=decision_mark)
            decision_confidence = _decision_unit_fraction(decision.confidence)
            logger.info(
                "autonomous iter=%d side=%s conf=%s mark=%.2f ai=%s",
                iteration,
                decision.side,
                decision_confidence
                if decision_confidence is not None
                else "unqualified",
                decision.mark_price,
                decision.ai_assist_status or "inactive",
            )
            last_mark_price = (
                float(decision.mark_price)
                if decision.mark_price > 0
                else last_mark_price
            )
            if recovery_pending:
                if not cfg.dry_run:
                    recovery_report = reconcile(client, runtime, store)
                    if not recovery_report.ok:
                        logger.error(
                            "autonomous iter=%d recovery-reconciliation-failed report=%s",
                            iteration,
                            recovery_report.asdict(),
                        )
                        exit_reason = "reconciliation-mismatch"
                        break
                capital_guard = _loss_budget_guard(
                    store,
                    last_mark_price,
                    strategy,
                    cfg,
                    now_ms_value=int(clock() * 1000),
                    mark_symbol=runtime.symbol,
                    mark_market_type=runtime.market_type,
                )
                if _capital_blocks_owned_management(capital_guard, store):
                    if capital_guard.force_close:
                        if not cfg.dry_run:
                            try:
                                close_lifecycle = _loop_lifecycle_plan(
                                    client,
                                    runtime,
                                    strategy,
                                    cfg,
                                    store,
                                    action="risk-close",
                                    reconcile=reconcile,
                                    api_budget_report=api_budget_report,
                                    require_api_budget_headroom=False,
                                )
                            except BinanceAPIError as exc:
                                logger.error(
                                    "autonomous iter=%d recovery close reconciliation failed: %s",
                                    iteration,
                                    exc,
                                )
                                exit_reason = "recovery-close-reconciliation-failed"
                                break
                            if not close_lifecycle.can_close:
                                logger.error(
                                    "autonomous iter=%d recovery close lifecycle blocked: %s",
                                    iteration,
                                    _lifecycle_reasons(
                                        close_lifecycle, capability="close"
                                    ),
                                )
                                exit_reason = "recovery-close-lifecycle-blocked"
                                break
                        close_report = close_tracked_open_positions(
                            store,
                            last_mark_price,
                            capital_guard.reason,
                            clock=clock,
                            client=None if cfg.dry_run else client,
                            reduce_only=runtime.market_type == "futures"
                            and strategy.reduce_only_on_close,
                            taker_fee_bps=strategy.taker_fee_bps,
                            paper_broker=paper_broker,
                        )
                        closed += close_report.closed
                        if close_report.closed:
                            logger.warning(
                                "autonomous iter=%d recovery force-close-open=%d reason=%s",
                                iteration,
                                close_report.closed,
                                capital_guard.reason,
                            )
                        if not close_report.ok or close_report.skipped:
                            exit_reason = f"{capital_guard.reason}:close-incomplete"
                            break
                    exit_reason = capital_guard.reason
                    break
                recovery_pending = False
                network_errors = 0
                cooldown = max(0, int(strategy.recovery_cooldown_seconds))
                stats = compute_stats(
                    store,
                    mark_price=last_mark_price,
                    starting_reference_cash=cfg.starting_reference_cash,
                )
                Heartbeat(
                    iteration=iteration,
                    state=STATE_RUNNING,
                    last_signal=decision_confidence
                    if decision_confidence is not None
                    else 0.0,
                    last_side="RECOVERY_OBSERVE",
                    last_price=decision.mark_price,
                    open_positions=stats.open_positions,
                    realized_pnl=stats.realized_pnl,
                    unrealized_pnl=stats.unrealized_pnl,
                    objective=objective.name,
                    updated_at_ms=int(clock() * 1000),
                    message=f"recovery-clean; cooldown={cooldown}s; no-entry-observation",
                    ai_assist_status=decision.ai_assist_status,
                    ai_assist_case_id=decision.ai_assist_case_id,
                ).write(cfg.heartbeat_path)
                heartbeats += 1
                sleep(max(poll, float(cooldown)))
                if (
                    cfg.stop_after_iterations is not None
                    and iteration >= cfg.stop_after_iterations
                ):
                    exit_reason = "iteration-cap"
                    break
                continue

            capital_guard = _loss_budget_guard(
                store,
                last_mark_price,
                strategy,
                cfg,
                now_ms_value=int(clock() * 1000),
                mark_symbol=runtime.symbol,
                mark_market_type=runtime.market_type,
            )
            if _capital_blocks_owned_management(capital_guard, store):
                if capital_guard.force_close:
                    if not cfg.dry_run:
                        try:
                            close_lifecycle = _loop_lifecycle_plan(
                                client,
                                runtime,
                                strategy,
                                cfg,
                                store,
                                action="risk-close",
                                reconcile=reconcile,
                                api_budget_report=api_budget_report,
                                require_api_budget_headroom=False,
                            )
                        except BinanceAPIError as exc:
                            logger.error(
                                "autonomous iter=%d risk close reconciliation failed: %s",
                                iteration,
                                exc,
                            )
                            exit_reason = "risk-close-reconciliation-failed"
                            break
                        if not close_lifecycle.can_close:
                            logger.error(
                                "autonomous iter=%d risk close lifecycle blocked: %s",
                                iteration,
                                _lifecycle_reasons(close_lifecycle, capability="close"),
                            )
                            exit_reason = "risk-close-lifecycle-blocked"
                            break
                    close_report = close_tracked_open_positions(
                        store,
                        last_mark_price,
                        capital_guard.reason,
                        clock=clock,
                        client=None if cfg.dry_run else client,
                        reduce_only=runtime.market_type == "futures"
                        and strategy.reduce_only_on_close,
                        taker_fee_bps=strategy.taker_fee_bps,
                        paper_broker=paper_broker,
                    )
                    closed += close_report.closed
                    if close_report.closed:
                        logger.warning(
                            "autonomous iter=%d force-close-open=%d reason=%s",
                            iteration,
                            close_report.closed,
                            capital_guard.reason,
                        )
                    if not close_report.ok or close_report.skipped:
                        exit_reason = f"{capital_guard.reason}:close-incomplete"
                        break
                exit_reason = capital_guard.reason
                break

            # Close any open position that meets auto-close thresholds first
            for position in store.load_open():
                observed_at_ms = (
                    int(decision.observed_at_ms)
                    if int(decision.observed_at_ms) > 0
                    else int(clock() * 1000)
                )
                should_close, reason = _evaluate_auto_close(
                    position,
                    decision.mark_price,
                    cfg,
                    strategy,
                    decision_side=decision.side,
                    observed_at_ms=observed_at_ms,
                    interval_ms=interval_milliseconds(runtime.interval),
                )
                if should_close:
                    if cfg.dry_run:
                        if paper_broker is None:
                            logger.error(
                                "autonomous iter=%d paper close lifecycle has no broker id=%s",
                                iteration,
                                position.id,
                            )
                            exit_reason = "paper-close-lifecycle-blocked"
                            break
                        paper_close_lifecycle = paper_broker.reconcile_positions(store)
                        if not paper_close_lifecycle.can_close:
                            logger.error(
                                "autonomous iter=%d paper close lifecycle blocked id=%s report=%s",
                                iteration,
                                position.id,
                                paper_close_lifecycle.asdict(),
                            )
                            exit_reason = "paper-close-lifecycle-blocked"
                            break
                    else:
                        try:
                            close_lifecycle = _loop_lifecycle_plan(
                                client,
                                runtime,
                                strategy,
                                cfg,
                                store,
                                action="close",
                                reconcile=reconcile,
                                api_budget_report=api_budget_report,
                                require_api_budget_headroom=False,
                            )
                        except BinanceAPIError as exc:
                            logger.error(
                                "autonomous iter=%d close reconciliation failed id=%s error=%s",
                                iteration,
                                position.id,
                                exc,
                            )
                            exit_reason = "close-reconciliation-failed"
                            break
                        if not close_lifecycle.can_close:
                            logger.error(
                                "autonomous iter=%d close lifecycle blocked id=%s reason=%s",
                                iteration,
                                position.id,
                                _lifecycle_reasons(close_lifecycle, capability="close"),
                            )
                            exit_reason = "close-lifecycle-blocked"
                            break
                    trade = _close_to_trade(
                        position,
                        decision.mark_price,
                        reason,
                        clock=clock,
                        fees=max(0.0, float(position.entry_fees)),
                    )
                    try:
                        if not position.dry_run:
                            trade = _submit_durable_close_position(
                                client,
                                position,
                                trade,
                                store,
                                reduce_only=runtime.market_type == "futures"
                                and strategy.reduce_only_on_close,
                                taker_fee_bps=strategy.taker_fee_bps,
                            )
                        else:
                            if paper_broker is None:
                                raise ValueError(
                                    "paper close requires shared execution broker"
                                )
                            paper_trade, paper_result = paper_broker.close_position(
                                position,
                                reason=reason,
                            )
                            if paper_trade is None:
                                logger.error(
                                    "autonomous iter=%d paper-close-unresolved id=%s state=%s reason=%s",
                                    iteration,
                                    position.id,
                                    paper_result.state,
                                    paper_result.reason,
                                )
                                exit_reason = f"{reason}:close-incomplete"
                                break
                            trade = paper_trade
                    except (BinanceAPIError, ValueError) as exc:
                        logger.error(
                            "autonomous iter=%d close-order-failed id=%s error=%s",
                            iteration,
                            position.id,
                            exc,
                        )
                        exit_reason = "close-order-failed"
                        break
                    if position.dry_run:
                        store.record_close_result(position, trade)
                    closed += 1
                    logger.info(
                        "autonomous iter=%d close id=%s reason=%s pnl=%+.2f (%+.2f%%)",
                        iteration,
                        trade.id,
                        reason,
                        trade.realized_pnl,
                        trade.realized_pnl_pct,
                    )
                    if _is_partial_close(position, trade):
                        logger.error(
                            "autonomous iter=%d close-incomplete id=%s qty=%s/%s",
                            iteration,
                            position.id,
                            trade.qty,
                            position.qty,
                        )
                        exit_reason = f"{reason}:close-incomplete"
                        break
            if exit_reason == "close-order-failed" or exit_reason.endswith(
                ":close-incomplete"
            ):
                break
            if exit_reason in {
                "close-reconciliation-failed",
                "close-lifecycle-blocked",
                "paper-close-lifecycle-blocked",
            }:
                break

            # Open new position only after the same risk gates used in operator
            # readiness checks approve it.
            gate = _entry_gate(
                store,
                decision,
                strategy,
                cfg,
                objective,
                now_ms_value=int(clock() * 1000),
                symbol=runtime.symbol,
                market_type=runtime.market_type,
            )
            if decision.side in {"LONG", "SHORT"} and gate.allowed:
                if cfg.dry_run:
                    if paper_broker is None:
                        logger.error(
                            "autonomous iter=%d paper open lifecycle has no broker",
                            iteration,
                        )
                        exit_reason = "paper-open-lifecycle-blocked"
                        break
                    paper_open_lifecycle = paper_broker.reconcile_positions(store)
                    if not paper_open_lifecycle.can_open:
                        logger.error(
                            "autonomous iter=%d paper open lifecycle blocked: %s",
                            iteration,
                            paper_open_lifecycle.asdict(),
                        )
                        exit_reason = "paper-open-lifecycle-blocked"
                        break
                else:
                    try:
                        open_lifecycle = _loop_lifecycle_plan(
                            client,
                            runtime,
                            strategy,
                            cfg,
                            store,
                            action="open",
                            reconcile=reconcile,
                            api_budget_report=api_budget_report,
                        )
                    except BinanceAPIError as exc:
                        logger.error(
                            "autonomous iter=%d open reconciliation failed: %s",
                            iteration,
                            exc,
                        )
                        exit_reason = "open-reconciliation-failed"
                        break
                    if not open_lifecycle.can_open:
                        logger.error(
                            "autonomous iter=%d open lifecycle blocked: %s",
                            iteration,
                            _lifecycle_reasons(open_lifecycle, capability="open"),
                        )
                        exit_reason = "open-lifecycle-blocked"
                        break
                position = _open_position_from_decision(
                    decision,
                    runtime,
                    strategy,
                    objective,
                    cfg,
                    clock=clock,
                )
                try:
                    paper_result = None
                    if position.dry_run:
                        if paper_broker is None:
                            raise ValueError(
                                "paper open requires shared execution broker"
                            )
                        position, paper_result = paper_broker.open_position(position)
                    else:
                        position = _submit_durable_open_position(
                            client, position, store
                        )
                except (BinanceAPIError, ValueError, OSError) as exc:
                    logger.error(
                        "autonomous iter=%d open-order-failed id=%s error=%s",
                        iteration,
                        position.id,
                        exc,
                    )
                    exit_reason = "open-order-failed"
                    break
                if position is None:
                    skipped += 1
                    logger.info(
                        "autonomous iter=%d paper-open-unfilled state=%s reason=%s",
                        iteration,
                        paper_result.state,
                        paper_result.reason,
                    )
                    if paper_result.state == "UNKNOWN":
                        exit_reason = "paper-open-unknown"
                        break
                else:
                    if position.dry_run:
                        store.record_open(position)
                    opened += 1
                    logger.info(
                        "autonomous iter=%d open id=%s side=%s qty=%.6f entry=%.2f",
                        iteration,
                        position.id,
                        position.side,
                        position.qty,
                        position.entry_price,
                    )
            elif decision.side in {"LONG", "SHORT"}:
                skipped += 1
                logger.info(
                    "autonomous iter=%d skip-entry reason=%s open=%d daily=%d conf=%.4f drawdown=%.4f",
                    iteration,
                    gate.reason,
                    gate.open_positions,
                    gate.daily_entries,
                    gate.directional_confidence,
                    gate.drawdown,
                )

            if iteration % max(1, cfg.heartbeat_every) == 0:
                stats = compute_stats(
                    store,
                    mark_price=decision.mark_price,
                    starting_reference_cash=cfg.starting_reference_cash,
                )
                heartbeat = Heartbeat(
                    iteration=iteration,
                    state=STATE_RUNNING,
                    last_signal=decision_confidence
                    if decision_confidence is not None
                    else 0.0,
                    last_side=decision.side,
                    last_price=decision.mark_price,
                    open_positions=stats.open_positions,
                    realized_pnl=stats.realized_pnl,
                    unrealized_pnl=stats.unrealized_pnl,
                    objective=objective.name,
                    updated_at_ms=int(clock() * 1000),
                    message=(
                        gate.reason
                        if decision.model_entry_ready is not True
                        else (
                            f"ai:{decision.ai_assist_status}:{decision.ai_assist_reason}"
                            if decision.ai_assist_status
                            and decision.ai_assist_status
                            not in {"shadow_idle", "shadow_approve"}
                            else (
                                ""
                                if gate.allowed or decision.side == "FLAT"
                                else gate.reason
                            )
                        )
                    ),
                    ai_assist_status=decision.ai_assist_status,
                    ai_assist_case_id=decision.ai_assist_case_id,
                )
                heartbeat.write(cfg.heartbeat_path)
                heartbeats += 1

            if (
                cfg.stop_after_iterations is not None
                and iteration >= cfg.stop_after_iterations
            ):
                exit_reason = "iteration-cap"
                break
            sleep(poll)
    finally:
        terminal_reasons: list[str] = []
        close_decision_fn = getattr(decision_fn, "close", None)
        if callable(close_decision_fn):
            try:
                ai_worker_stopped = bool(close_decision_fn(0.25))
            except Exception as exc:  # noqa: BLE001 - shadow AI cannot block shutdown
                logger.warning("autonomous AI shadow reviewer shutdown failed: %s", exc)
            else:
                if not ai_worker_stopped:
                    logger.warning(
                        "autonomous AI shadow reviewer exceeded bounded shutdown; "
                        "daemon worker detached"
                    )
        ledger_errors = store.open_integrity_errors()
        if ledger_errors:
            terminal_reasons.append(
                f"open-ledger-integrity-errors={len(ledger_errors)}"
            )
        else:
            local_open_count = len(store.load_open())
            if local_open_count:
                terminal_reasons.append(f"tracked-open-positions={local_open_count}")
        if paper_broker is not None:
            try:
                paper_terminal = paper_broker.reconcile_positions(store)
            except Exception as exc:  # noqa: BLE001 - terminal state must fail closed
                terminal_reasons.append(
                    f"paper-terminal-reconciliation={exc.__class__.__name__}"
                )
            else:
                remaining_inventory = sum(
                    item.remaining_quantity > 0
                    for item in paper_terminal.journal.inventory
                )
                if not paper_terminal.ok or remaining_inventory:
                    terminal_reasons.append(
                        "paper-terminal-exposure="
                        f"inventory:{remaining_inventory},"
                        f"errors:{len(paper_terminal.position_errors)},"
                        f"blocking:{len(paper_terminal.journal.blocking_intent_ids)}"
                    )
        elif not cfg.dry_run:
            try:
                live_terminal = reconcile(client, runtime, store)
            except Exception as exc:  # noqa: BLE001 - terminal state must fail closed
                terminal_reasons.append(
                    f"live-terminal-reconciliation={exc.__class__.__name__}"
                )
            else:
                if not live_terminal.ok or live_terminal.exchange_exposure_count:
                    terminal_reasons.append(
                        "live-terminal-exposure="
                        f"exchange:{live_terminal.exchange_exposure_count},"
                        f"mismatches:{len(live_terminal.mismatches)}"
                    )
        final_state = STATE_STOPPING if terminal_reasons else STATE_STOPPED
        note = exit_reason
        if terminal_reasons:
            note += "; " + "; ".join(terminal_reasons)
            logger.error(
                "autonomous exit remains STOPPING until exposure is reconciled: %s",
                "; ".join(terminal_reasons),
            )
        control.write(final_state, note=note)
        if paper_broker is not None:
            paper_broker.close()

    return LoopResult(
        iterations=iteration,
        final_state=final_state,
        heartbeats_written=heartbeats,
        closed_trades=closed,
        opened_trades=opened,
        exit_reason=exit_reason,
        skipped_entries=skipped,
    )
