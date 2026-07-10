"""Adaptive model-zoo experts layered on top of the trained base model.

The experts in this module are original implementations inspired by common
free/community day-trading model families: Lorentzian nearest-neighbor voting,
rational-quadratic kernel smoothing, and technical confluence controllers
similar in spirit to SuperTrend/MACD/Bollinger-style dashboards.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import math
import os
from statistics import median
from typing import Callable, Mapping, Sequence

from .backtest import BacktestResult, calibrate_threshold_for_backtest, closed_trades_per_day, run_backtest
from .compute import BackendInfo, resolve_backend
from .execution_simulation import (
    EXECUTION_ACTIVITY_ESTIMATOR,
    EXECUTION_MODEL_VERSION,
    SymbolExecutionProfile,
    market_row_execution_assumptions,
    market_row_quote_volume_notional,
    market_row_reported_quote_volume_notional,
    market_row_trailing_quote_volume_24h_estimate,
    simulate_market_fill,
)
from .features import ModelRow
from .model import HybridExpert, HybridPrototype, TrainedModel
from .objective import get_objective
from .risk_controls import stop_loss_sized_notional_pct
from .types import StrategyConfig


UTILITY_RANK_GROUP_DURATION_MS = 4 * 60 * 60 * 1000
MAX_UTILITY_RANK_GROUP_ROWS = 8192


@dataclass(frozen=True)
class HybridAblationResult:
    removed_expert_kind: str
    removed_expert_count: int
    remaining_expert_count: int
    accepted: bool
    score: float
    delta_vs_best: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HybridProfileResult:
    profile: str
    search_score: float
    accepted: bool
    realized_pnl: float | None
    closed_trades: int
    trades_per_day: float | None
    max_drawdown: float | None
    profit_factor: float | None
    expectancy: float | None
    reject_reason: str
    threshold: float | None = None
    long_threshold: float | None = None
    short_threshold: float | None = None
    expert_kinds: tuple[str, ...] = ()
    selected_search_best: bool = False
    promotion_eligible: bool = True
    win_rate: float | None = None
    total_fees: float | None = None
    gross_profit: float | None = None
    gross_loss: float | None = None
    average_trade_return: float | None = None
    max_consecutive_losses: int = 0
    exit_reason_counts: dict[str, int] = field(default_factory=dict)
    side_counts: dict[str, int] = field(default_factory=dict)
    average_bars_held: float | None = None
    mean_round_trip_execution_cost_bps: float | None = None
    gross_profitable_trades: int = 0
    net_profitable_trades: int = 0

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HybridOptimizationReport:
    accepted: bool
    model: TrainedModel
    base_score: float
    best_score: float
    best_profile: str
    evaluated_profiles: int
    base_result: BacktestResult | None
    best_result: BacktestResult | None
    ablation_results: tuple[HybridAblationResult, ...] = ()
    neural_expert_params: dict[str, object] | None = None
    payoff_expert_params: tuple[dict[str, object], ...] = ()
    payoff_expert_failures: tuple[dict[str, object], ...] = ()
    profile_results: tuple[HybridProfileResult, ...] = ()
    selection_search_rows: int = 0
    selection_full_rows: int = 0

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["base_result"] = asdict(self.base_result) if self.base_result is not None else None
        payload["best_result"] = asdict(self.best_result) if self.best_result is not None else None
        payload["ablation_results"] = [item.asdict() for item in self.ablation_results]
        payload["profile_results"] = [item.asdict() for item in self.profile_results]
        return payload


@dataclass(frozen=True)
class _WeightProfile:
    name: str
    base: float
    lorentzian: float
    kernel: float
    technical: float
    neural: float = 0.0
    payoff: float = 0.0
    invert_probability: bool = False
    selection_eligible: bool = True


@dataclass(frozen=True)
class _PayoffTrainingExamples:
    rows: list[ModelRow]
    targets: list[float]
    source_indexes: list[int]
    meta: dict[str, object]
    long_action_targets: list[float] | None = None
    short_action_targets: list[float] | None = None


def _normalized_features(model: TrainedModel, row: ModelRow) -> list[float]:
    return list(model._normalize(row.features))  # noqa: SLF001 - internal model-zoo attachment point


def _even_sample(rows: Sequence[ModelRow], limit: int) -> list[ModelRow]:
    values = list(rows)
    if limit <= 0 or len(values) <= limit:
        return values
    if limit == 1:
        return [values[-1]]
    step = (len(values) - 1) / float(limit - 1)
    indexes = sorted({round(index * step) for index in range(limit)})
    return [values[int(index)] for index in indexes[:limit]]


def _even_sample_indexes(count: int, limit: int) -> list[int]:
    count = max(0, int(count))
    if count <= 0:
        return []
    if limit <= 0 or count <= limit:
        return list(range(count))
    if limit == 1:
        return [count - 1]
    step = (count - 1) / float(limit - 1)
    return [int(index) for index in sorted({round(i * step) for i in range(limit)})[:limit]]


def _prototype_limit(objective_name: str) -> int:
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    if objective_name == "conservative":
        return 220
    if objective_name == "aggressive":
        return 520
    return 360


def _hybrid_selection_search_limit(objective_name: str) -> int:
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    if objective_name == "conservative":
        return 200_000
    if objective_name == "aggressive":
        return 300_000
    return 250_000


def _build_prototypes(rows: Sequence[ModelRow], model: TrainedModel, objective_name: str) -> list[HybridPrototype]:
    limit = _prototype_limit(objective_name)
    positives = [row for row in rows if int(row.label) == 1]
    negatives = [row for row in rows if int(row.label) == 0]
    half = max(1, limit // 2)
    selected = [*_even_sample(positives, half), *_even_sample(negatives, half)]
    if len(selected) < min(limit, len(rows)):
        already = {row.timestamp for row in selected}
        selected.extend(row for row in _even_sample(rows, limit) if row.timestamp not in already)
    selected = sorted(selected[:limit], key=lambda row: row.timestamp)
    return [
        HybridPrototype(
            features=_normalized_features(model, row),
            label=int(row.label),
            timestamp=int(row.timestamp),
            close=float(row.close),
        )
        for row in selected
    ]


def _estimate_bandwidth(prototypes: Sequence[HybridPrototype], feature_dim: int) -> float:
    if len(prototypes) < 2:
        return 1.0
    distances: list[float] = []
    sample = list(prototypes)
    stride = max(1, len(sample) // 80)
    for index in range(0, len(sample) - stride, stride):
        left = sample[index].features
        right = sample[index + stride].features
        squared = 0.0
        for a, b in zip(left, right, strict=True):
            delta = a - b
            squared += delta * delta
        distances.append(math.sqrt(squared / max(1, feature_dim)))
    if not distances:
        return 1.0
    return max(0.05, min(5.0, float(median(distances))))


def _profiles_for(objective_name: str) -> tuple[_WeightProfile, ...]:
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    if objective_name == "conservative":
        return (
            _WeightProfile("base_only", 1.00, 0.00, 0.00, 0.00),
            _WeightProfile("signed_payoff_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
            _WeightProfile("signed_payoff_mlp_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
            _WeightProfile("signed_payoff_tree_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
            _WeightProfile("signed_payoff_inverse_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, True, False),
            _WeightProfile("signed_payoff_mlp_inverse_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, True, False),
            _WeightProfile("signed_payoff_neural_gate", 0.00, 0.00, 0.00, 0.00, 0.25, 0.75),
            _WeightProfile("signed_payoff_mlp_neural_gate", 0.00, 0.00, 0.00, 0.00, 0.25, 0.75),
            _WeightProfile("guarded_neighbors", 0.68, 0.16, 0.11, 0.05),
            _WeightProfile("kernel_confirmation", 0.62, 0.12, 0.20, 0.06),
            _WeightProfile("technical_tiebreaker", 0.72, 0.10, 0.08, 0.10),
            _WeightProfile("technical_rescue_core", 0.10, 0.06, 0.09, 0.75),
            _WeightProfile("neighbor_kernel_rescue", 0.15, 0.44, 0.31, 0.10),
            _WeightProfile("balanced_rescue_committee", 0.25, 0.25, 0.25, 0.25),
            _WeightProfile("neural_guarded_committee", 0.45, 0.12, 0.12, 0.06, 0.25),
            _WeightProfile("neural_confirmed_rescue", 0.18, 0.16, 0.16, 0.12, 0.38),
            _WeightProfile("signed_payoff_guarded", 0.32, 0.06, 0.06, 0.06, 0.00, 0.50),
            _WeightProfile("signed_payoff_neural_committee", 0.24, 0.06, 0.06, 0.04, 0.20, 0.40),
        )
    if objective_name == "aggressive":
        return (
            _WeightProfile("base_only", 1.00, 0.00, 0.00, 0.00),
            _WeightProfile("signed_payoff_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
            _WeightProfile("signed_payoff_mlp_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
            _WeightProfile("signed_payoff_tree_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
            _WeightProfile("signed_payoff_inverse_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, True, False),
            _WeightProfile("signed_payoff_mlp_inverse_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, True, False),
            _WeightProfile("signed_payoff_neural_gate", 0.00, 0.00, 0.00, 0.00, 0.30, 0.70),
            _WeightProfile("signed_payoff_mlp_neural_gate", 0.00, 0.00, 0.00, 0.00, 0.30, 0.70),
            _WeightProfile("neighbor_momentum", 0.38, 0.34, 0.16, 0.12),
            _WeightProfile("kernel_regime", 0.34, 0.20, 0.30, 0.16),
            _WeightProfile("technical_breakout", 0.30, 0.24, 0.18, 0.28),
            _WeightProfile("balanced_aggressive", 0.36, 0.26, 0.22, 0.16),
            _WeightProfile("neural_momentum_committee", 0.24, 0.18, 0.14, 0.12, 0.32),
            _WeightProfile("neural_dominant_rescue", 0.10, 0.16, 0.14, 0.10, 0.50),
            _WeightProfile("signed_payoff_scalper", 0.16, 0.08, 0.06, 0.04, 0.16, 0.50),
            _WeightProfile("signed_payoff_dominant", 0.08, 0.06, 0.04, 0.02, 0.20, 0.60),
        )
    return (
        _WeightProfile("base_only", 1.00, 0.00, 0.00, 0.00),
        _WeightProfile("signed_payoff_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
        _WeightProfile("signed_payoff_mlp_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
        _WeightProfile("signed_payoff_tree_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
        _WeightProfile("signed_payoff_inverse_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, True, False),
        _WeightProfile("signed_payoff_mlp_inverse_direct", 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, True, False),
        _WeightProfile("signed_payoff_neural_gate", 0.00, 0.00, 0.00, 0.00, 0.25, 0.75),
        _WeightProfile("signed_payoff_mlp_neural_gate", 0.00, 0.00, 0.00, 0.00, 0.25, 0.75),
        _WeightProfile("balanced_neighbors", 0.50, 0.24, 0.17, 0.09),
        _WeightProfile("smooth_kernel", 0.48, 0.16, 0.26, 0.10),
        _WeightProfile("technical_blend", 0.46, 0.20, 0.18, 0.16),
        _WeightProfile("neural_balanced_committee", 0.34, 0.18, 0.18, 0.10, 0.20),
        _WeightProfile("neural_regime_rescue", 0.16, 0.18, 0.16, 0.10, 0.40),
        _WeightProfile("signed_payoff_balanced", 0.22, 0.08, 0.08, 0.04, 0.18, 0.40),
        _WeightProfile("signed_payoff_confirmation", 0.34, 0.04, 0.06, 0.06, 0.00, 0.50),
    )


def _experts_for_profile(
    profile: _WeightProfile,
    prototypes: Sequence[HybridPrototype],
    *,
    feature_dim: int,
    feature_count: int,
    bandwidth: float,
    objective_name: str,
    neural_expert: HybridExpert | None = None,
    payoff_experts: Sequence[HybridExpert] = (),
) -> list[HybridExpert]:
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    k = 13 if objective_name == "conservative" else (31 if objective_name == "aggressive" else 21)
    experts = [
        HybridExpert(
            name="lorentzian_neighbor_vote",
            kind="lorentzian_knn",
            weight=profile.lorentzian,
            prototypes=list(prototypes),
            k=k,
            feature_count=feature_count,
            notes="Lorentzian-distance neighbor vote inspired by public TradingView ML indicator patterns.",
        ),
        HybridExpert(
            name="rational_quadratic_kernel",
            kind="rational_quadratic_kernel",
            weight=profile.kernel,
            prototypes=list(prototypes),
            bandwidth=bandwidth,
            alpha=1.25 if objective_name == "conservative" else 0.85,
            feature_count=feature_count,
            notes="Rational-quadratic kernel smoother for regime-aware probability confirmation.",
        ),
        HybridExpert(
            name="technical_confluence_controller",
            kind="technical_confluence",
            weight=profile.technical,
            prototypes=[],
            feature_count=feature_count,
            notes="Original technical controller using trend, mean-reversion, volatility, and volume features.",
        ),
    ]
    if neural_expert is not None and profile.neural > 0.0:
        expert = copy.deepcopy(neural_expert)
        expert.weight = max(0.0, float(profile.neural))
        experts.append(expert)
    if payoff_experts and profile.payoff > 0.0:
        selected_payoff_experts = list(payoff_experts)
        profile_name = str(profile.name)
        if "payoff_mlp" in profile_name:
            selected_payoff_experts = [
                expert for expert in payoff_experts if str(expert.kind) == "signed_payoff_mlp_ranker"
            ]
        elif "payoff_tree" in profile_name:
            selected_payoff_experts = [
                expert for expert in payoff_experts if str(expert.kind) == "signed_payoff_lightgbm_ranker"
            ]
        elif "payoff_linear" in profile_name:
            selected_payoff_experts = [
                expert for expert in payoff_experts if str(expert.kind) == "signed_payoff_ranker"
            ]
        if not selected_payoff_experts and not _required_payoff_expert_kind(profile):
            selected_payoff_experts = list(payoff_experts)
        if not selected_payoff_experts:
            return experts
        payoff_weight = max(0.0, float(profile.payoff)) / float(len(selected_payoff_experts))
        for payoff_expert in selected_payoff_experts:
            expert = copy.deepcopy(payoff_expert)
            expert.weight = payoff_weight
            experts.append(expert)
    return experts


def _evaluate_model(
    model: TrainedModel,
    rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    objective_name: str,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None,
    score_batch_size: int,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> tuple[float, BacktestResult | None, TrainedModel]:
    if not rows:
        return float("-inf"), None, model
    objective = get_objective(objective_name)
    row_list = list(rows)
    result = run_backtest(
        row_list,
        model,
        strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
        symbol_profile=symbol_profile,
    )
    if not objective.accepts(result):
        return float("-inf"), result, model
    return float(objective.score(result)), result, model


def _evaluate_model_with_threshold_calibration(
    model: TrainedModel,
    rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    objective_name: str,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None,
    score_batch_size: int,
    symbol_profile: SymbolExecutionProfile | None = None,
    status_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> tuple[float, BacktestResult | None, TrainedModel]:
    """Evaluate a hybrid after a small, profit-gated threshold search."""

    if not rows:
        return float("-inf"), None, model
    objective = get_objective(objective_name)
    row_list = list(rows)
    try:
        threshold_report = calibrate_threshold_for_backtest(
            row_list,
            model,
            strategy,
            starting_cash=starting_cash,
            market_type=market_type,
            baseline_threshold=getattr(model, "decision_threshold", None),
            start=0.50 if str(market_type).lower() == "futures" else 0.05,
            end=0.72 if objective.name == "conservative" else 0.78,
            steps=7,
            min_closed_trades=objective.min_closed_trades,
            min_trades_per_day=objective.min_trades_per_day,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
            symbol_profile=symbol_profile,
            adaptive_probability_thresholds=True,
            max_adaptive_thresholds=12,
            allowed_sides="both",
            status_callback=status_callback,
        )
    except Exception:
        threshold_report = None
    if threshold_report is not None and bool(getattr(threshold_report, "accepted", False)):
        calibrated = copy.deepcopy(model)
        calibrated.decision_threshold = float(threshold_report.threshold)
        if str(market_type).lower() == "futures":
            calibrated.long_decision_threshold = threshold_report.long_threshold
            calibrated.short_decision_threshold = threshold_report.short_threshold
        else:
            calibrated.long_decision_threshold = None
            calibrated.short_decision_threshold = None
        calibrated.threshold_source = "hybrid_profit_backtest"
        score, result, _model = _evaluate_model(
            calibrated,
            row_list,
            strategy,
            objective_name=objective_name,
            market_type=market_type,
            starting_cash=starting_cash,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
            symbol_profile=symbol_profile,
        )
        return score, result, calibrated
    if threshold_report is not None and (
        float(getattr(threshold_report, "best_score", float("-inf"))) > float(getattr(threshold_report, "baseline_score", float("-inf"))) + 1e-12
        or float(getattr(threshold_report, "best_realized_pnl", float("-inf"))) > float(getattr(threshold_report, "baseline_realized_pnl", float("-inf"))) + 1e-12
    ):
        diagnostic = copy.deepcopy(model)
        diagnostic.decision_threshold = float(threshold_report.best_threshold)
        if str(market_type).lower() == "futures":
            diagnostic.long_decision_threshold = threshold_report.best_long_threshold
            diagnostic.short_decision_threshold = threshold_report.best_short_threshold
        else:
            diagnostic.long_decision_threshold = None
            diagnostic.short_decision_threshold = None
        diagnostic.threshold_source = "hybrid_diagnostic_threshold_search"
        _score, result, _model = _evaluate_model(
            diagnostic,
            row_list,
            strategy,
            objective_name=objective_name,
            market_type=market_type,
            starting_cash=starting_cash,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
            symbol_profile=symbol_profile,
        )
        return float("-inf"), result, diagnostic
    return _evaluate_model(
        model,
        row_list,
        strategy,
        objective_name=objective_name,
        market_type=market_type,
        starting_cash=starting_cash,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
        symbol_profile=symbol_profile,
    )


def _model_has_payoff_expert(model: TrainedModel) -> bool:
    return any(
        str(expert.kind).startswith("signed_payoff_")
        for expert in getattr(model, "hybrid_experts", []) or []
    )


def _finite_status_number(value: float, default: float = -1.0e308) -> float:
    return float(value) if math.isfinite(float(value)) else float(default)


def _profile_uses_cpu_prototype_scan(profile: _WeightProfile) -> bool:
    return float(profile.lorentzian) > 0.0 or float(profile.kernel) > 0.0


def _required_payoff_expert_kind(profile: _WeightProfile) -> str:
    name = str(profile.name)
    if "payoff_mlp" in name:
        return "signed_payoff_mlp_ranker"
    if "payoff_tree" in name:
        return "signed_payoff_lightgbm_ranker"
    if "payoff_linear" in name:
        return "signed_payoff_ranker"
    return ""


def _skip_profile_for_large_payoff_search(
    profile: _WeightProfile,
    *,
    selection_rows: Sequence[ModelRow],
    payoff_experts: Sequence[HybridExpert],
    neural_expert: HybridExpert | None = None,
) -> str:
    if float(profile.neural) > 0.0 and neural_expert is None:
        return "required_neural_expert_unavailable"
    required_kind = _required_payoff_expert_kind(profile)
    if required_kind and not any(str(expert.kind) == required_kind for expert in payoff_experts):
        return f"required_payoff_expert_unavailable:{required_kind}"
    if (
        len(selection_rows) >= 100_000
        and payoff_experts
        and _profile_uses_cpu_prototype_scan(profile)
    ):
        return "skipped_large_second_level_cpu_prototype_profile"
    return ""


def _profile_result(
    *,
    profile_name: str,
    score: float,
    result: BacktestResult | None,
    model: TrainedModel,
    objective_name: str,
    selected_search_best: bool = False,
    promotion_eligible: bool = True,
) -> HybridProfileResult:
    objective = get_objective(objective_name)
    if result is None:
        return HybridProfileResult(
            profile=profile_name,
            search_score=_finite_status_number(float(score)),
            accepted=False,
            realized_pnl=None,
            closed_trades=0,
            trades_per_day=None,
            max_drawdown=None,
            profit_factor=None,
            expectancy=None,
            reject_reason="no_backtest_result",
            threshold=getattr(model, "decision_threshold", None),
            long_threshold=getattr(model, "long_decision_threshold", None),
            short_threshold=getattr(model, "short_decision_threshold", None),
            expert_kinds=tuple(sorted({str(expert.kind) for expert in getattr(model, "hybrid_experts", []) or []})),
            selected_search_best=bool(selected_search_best),
            promotion_eligible=bool(promotion_eligible),
        )
    accepted = bool(objective.accepts(result))
    reject_reason = "" if accepted else str(objective.reject_reason(result) or "selection_gate_failed")
    trade_log = [item for item in (getattr(result, "trade_log", ()) or ()) if isinstance(item, dict)]
    exit_reason_counts: dict[str, int] = {}
    side_counts: dict[str, int] = {}
    bars_held: list[float] = []
    round_trip_costs: list[float] = []
    gross_profitable_trades = 0
    net_profitable_trades = 0
    for trade in trade_log:
        exit_reason = str(trade.get("exit_reason", "unknown") or "unknown")
        exit_reason_counts[exit_reason] = exit_reason_counts.get(exit_reason, 0) + 1
        side = str(trade.get("side", "unknown") or "unknown")
        side_counts[side] = side_counts.get(side, 0) + 1
        try:
            held = float(trade.get("bars_held", 0.0) or 0.0)
            entry_cost = float(trade.get("entry_execution_cost_bps", 0.0) or 0.0)
            exit_cost = float(trade.get("exit_execution_cost_bps", 0.0) or 0.0)
            gross_pnl = float(trade.get("realized_pnl", 0.0) or 0.0)
            net_pnl = float(trade.get("net_pnl", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(held):
            bars_held.append(max(0.0, held))
        if math.isfinite(entry_cost) and math.isfinite(exit_cost):
            round_trip_costs.append(max(0.0, entry_cost) + max(0.0, exit_cost))
        gross_profitable_trades += int(math.isfinite(gross_pnl) and gross_pnl > 0.0)
        net_profitable_trades += int(math.isfinite(net_pnl) and net_pnl > 0.0)
    return HybridProfileResult(
        profile=profile_name,
        search_score=_finite_status_number(float(score)),
        accepted=accepted,
        realized_pnl=float(result.realized_pnl),
        closed_trades=int(result.closed_trades),
        trades_per_day=float(closed_trades_per_day(result)),
        max_drawdown=float(result.max_drawdown),
        profit_factor=float(getattr(result, "profit_factor", 0.0) or 0.0),
        expectancy=float(getattr(result, "expectancy", 0.0) or 0.0),
        reject_reason=reject_reason,
        threshold=getattr(model, "decision_threshold", None),
        long_threshold=getattr(model, "long_decision_threshold", None),
        short_threshold=getattr(model, "short_decision_threshold", None),
        expert_kinds=tuple(sorted({str(expert.kind) for expert in getattr(model, "hybrid_experts", []) or []})),
        selected_search_best=bool(selected_search_best),
        promotion_eligible=bool(promotion_eligible),
        win_rate=float(getattr(result, "win_rate", 0.0) or 0.0),
        total_fees=float(getattr(result, "total_fees", 0.0) or 0.0),
        gross_profit=float(getattr(result, "gross_profit", 0.0) or 0.0),
        gross_loss=float(getattr(result, "gross_loss", 0.0) or 0.0),
        average_trade_return=float(getattr(result, "average_trade_return", 0.0) or 0.0),
        max_consecutive_losses=int(getattr(result, "max_consecutive_losses", 0) or 0),
        exit_reason_counts=exit_reason_counts,
        side_counts=side_counts,
        average_bars_held=(sum(bars_held) / len(bars_held) if bars_held else 0.0),
        mean_round_trip_execution_cost_bps=(
            sum(round_trip_costs) / len(round_trip_costs) if round_trip_costs else 0.0
        ),
        gross_profitable_trades=int(gross_profitable_trades),
        net_profitable_trades=int(net_profitable_trades),
    )


def _score_delta(score: float, best_score: float) -> float:
    if not math.isfinite(score) or not math.isfinite(best_score):
        return float("-inf")
    return float(score - best_score)


def _hybrid_ablation_results(
    *,
    base_model: TrainedModel,
    best_model: TrainedModel,
    base_score: float,
    best_score: float,
    rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    objective_name: str,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None,
    score_batch_size: int,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> tuple[HybridAblationResult, ...]:
    """Replay the selected hybrid with individual expert families removed."""

    if not best_model.hybrid_experts:
        return ()
    results: list[HybridAblationResult] = [
        HybridAblationResult(
            removed_expert_kind="all_hybrid_experts",
            removed_expert_count=len(best_model.hybrid_experts),
            remaining_expert_count=0,
            accepted=math.isfinite(base_score),
            score=float(base_score),
            delta_vs_best=_score_delta(float(base_score), best_score),
        )
    ]
    for kind in sorted({expert.kind for expert in best_model.hybrid_experts}):
        candidate = copy.deepcopy(best_model)
        original_count = len(candidate.hybrid_experts)
        candidate.hybrid_experts = [expert for expert in candidate.hybrid_experts if expert.kind != kind]
        removed = original_count - len(candidate.hybrid_experts)
        if removed <= 0:
            continue
        score, _result, _evaluated_model = _evaluate_model(
            candidate if candidate.hybrid_experts else base_model,
            rows,
            strategy,
            objective_name=objective_name,
            market_type=market_type,
            starting_cash=starting_cash,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
            symbol_profile=symbol_profile,
        )
        results.append(HybridAblationResult(
            removed_expert_kind=kind,
            removed_expert_count=removed,
            remaining_expert_count=len(candidate.hybrid_experts),
            accepted=math.isfinite(score),
            score=float(score),
            delta_vs_best=_score_delta(float(score), best_score),
        ))
    return tuple(results)


def _torch_device_for_backend(backend: BackendInfo):
    if backend.kind == "directml":
        import torch_directml  # type: ignore

        return torch_directml.device()
    return backend.device


def _neural_training_limit(objective_name: str) -> int:
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    if objective_name == "aggressive":
        return 120_000
    if objective_name == "conservative":
        return 80_000
    return 100_000


def _neural_hidden_dims(objective_name: str, feature_dim: int) -> tuple[int, int]:
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    width_cap = 96 if objective_name == "aggressive" else (64 if objective_name == "conservative" else 80)
    first = max(12, min(width_cap, max(16, int(feature_dim))))
    second = max(8, min(32, max(8, first // 2)))
    return int(first), int(second)


def _neural_epochs(objective_name: str) -> int:
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    if objective_name == "aggressive":
        return 18
    if objective_name == "conservative":
        return 12
    return 15


def _neural_class_weights(rows: Sequence[ModelRow]) -> tuple[float, float]:
    positives = sum(1 for row in rows if int(row.label) == 1)
    negatives = len(rows) - positives
    if positives <= 0 or negatives <= 0:
        return 1.0, 1.0
    total = float(len(rows))
    return total / (2.0 * positives), total / (2.0 * negatives)


def _train_dense_mlp_expert(
    rows: Sequence[ModelRow],
    model: TrainedModel,
    *,
    objective_name: str,
    compute_backend: str | None,
    batch_size: int,
    seed: int = 7,
) -> HybridExpert | None:
    """Fit a compact dense neural expert and serialize it into a HybridExpert."""

    sampled_rows = _even_sample(rows, _neural_training_limit(objective_name))
    if len(sampled_rows) < 64:
        return None
    positives = sum(1 for row in sampled_rows if int(row.label) == 1)
    negatives = len(sampled_rows) - positives
    if positives <= 0 or negatives <= 0:
        return None
    try:
        import torch  # type: ignore
    except Exception:
        return None

    backend = resolve_backend(compute_backend or "auto")
    try:
        device = _torch_device_for_backend(backend)
    except Exception:
        backend = resolve_backend("cpu")
        device = "cpu"

    validation_count = max(16, min(len(sampled_rows) // 5, 10_000))
    if len(sampled_rows) - validation_count < 48:
        validation_count = max(8, len(sampled_rows) // 4)
    train_rows = sampled_rows[:-validation_count]
    validation_rows = sampled_rows[-validation_count:]
    if not train_rows or not validation_rows:
        return None

    input_dim = int(model.feature_dim)
    hidden_1, hidden_2 = _neural_hidden_dims(objective_name, input_dim)
    try:
        torch.manual_seed(int(seed))
        x_train = torch.tensor(
            [model._normalize(row.features) for row in train_rows],  # noqa: SLF001 - serialized expert parity
            dtype=torch.float32,
            device=device,
        )
        y_train = torch.tensor([float(row.label) for row in train_rows], dtype=torch.float32, device=device)
        x_validation = torch.tensor(
            [model._normalize(row.features) for row in validation_rows],  # noqa: SLF001
            dtype=torch.float32,
            device=device,
        )
        y_validation = torch.tensor([float(row.label) for row in validation_rows], dtype=torch.float32, device=device)
        scale_1 = math.sqrt(2.0 / max(1, input_dim))
        scale_2 = math.sqrt(2.0 / max(1, hidden_1))
        scale_3 = math.sqrt(2.0 / max(1, hidden_2))
        w1 = (torch.randn((input_dim, hidden_1), dtype=torch.float32, device=device) * scale_1).requires_grad_()
        b1 = torch.zeros((hidden_1,), dtype=torch.float32, device=device, requires_grad=True)
        w2 = (torch.randn((hidden_1, hidden_2), dtype=torch.float32, device=device) * scale_2).requires_grad_()
        b2 = torch.zeros((hidden_2,), dtype=torch.float32, device=device, requires_grad=True)
        w3 = (torch.randn((hidden_2, 1), dtype=torch.float32, device=device) * scale_3).requires_grad_()
        b3 = torch.zeros((1,), dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.SGD([w1, b1, w2, b2, w3, b3], lr=0.015, momentum=0.9, weight_decay=1e-4)
        positive_weight, negative_weight = _neural_class_weights(train_rows)
        pos_weight_t = torch.tensor(float(positive_weight), dtype=torch.float32, device=device)
        neg_weight_t = torch.tensor(float(negative_weight), dtype=torch.float32, device=device)
        focal_gamma = 1.5 if objective_name == "conservative" else 2.0
        epochs = _neural_epochs(objective_name)
        batch = max(256, min(max(1, int(batch_size or 4096)), len(train_rows)))
        best_state: tuple[object, ...] | None = None
        best_validation_loss = float("inf")
        stagnant_epochs = 0

        def forward(values):
            hidden = torch.relu(torch.matmul(values, w1) + b1)
            hidden = torch.relu(torch.matmul(hidden, w2) + b2)
            return torch.clamp(torch.matmul(hidden, w3).reshape(-1) + b3.reshape(()), min=-50.0, max=50.0)

        def weighted_focal_loss(logits, labels):
            per_row = torch.clamp(logits, min=0.0) - logits * labels + torch.log1p(torch.exp(-torch.abs(logits)))
            probabilities = torch.sigmoid(logits)
            pt = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
            focal = torch.pow(torch.clamp(1.0 - pt, min=0.0, max=1.0), float(focal_gamma))
            sample_weights = torch.where(labels > 0.5, pos_weight_t, neg_weight_t)
            return (per_row * focal * sample_weights).mean()

        for _epoch in range(epochs):
            for start in range(0, len(train_rows), batch):
                end = min(start + batch, len(train_rows))
                optimizer.zero_grad()
                loss = weighted_focal_loss(forward(x_train[start:end]), y_train[start:end])
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                validation_loss = float(weighted_focal_loss(forward(x_validation), y_validation).detach().cpu().item())
            if validation_loss < best_validation_loss - 1e-5:
                best_validation_loss = validation_loss
                best_state = (
                    w1.detach().clone(),
                    b1.detach().clone(),
                    w2.detach().clone(),
                    b2.detach().clone(),
                    w3.detach().clone(),
                    b3.detach().clone(),
                )
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1
                if stagnant_epochs >= 4:
                    break
        if best_state is not None:
            w1, b1, w2, b2, w3, b3 = best_state  # type: ignore[assignment]
        with torch.no_grad():
            training_loss = float(weighted_focal_loss(forward(x_train), y_train).detach().cpu().item())
            validation_loss = float(weighted_focal_loss(forward(x_validation), y_validation).detach().cpu().item())
        layers = [
            {
                "weights": [[float(value) for value in row] for row in w1.detach().cpu().tolist()],
                "bias": [float(value) for value in b1.detach().cpu().tolist()],
                "activation": "relu",
            },
            {
                "weights": [[float(value) for value in row] for row in w2.detach().cpu().tolist()],
                "bias": [float(value) for value in b2.detach().cpu().tolist()],
                "activation": "relu",
            },
            {
                "weights": [[float(value) for value in row] for row in w3.detach().cpu().tolist()],
                "bias": [float(value) for value in b3.detach().cpu().tolist()],
                "activation": "sigmoid",
            },
        ]
    except Exception:
        return None

    return HybridExpert(
        name="dense_mlp_neural_edge",
        kind="dense_mlp",
        weight=0.0,
        prototypes=[],
        feature_count=input_dim,
        notes="Compact GPU-trained dense neural expert; selected only through real backtest gates.",
        params={
            "input_dim": int(input_dim),
            "layers": layers,
            "output_activation": "sigmoid",
            "training_rows": int(len(train_rows)),
            "validation_rows": int(len(validation_rows)),
            "source_rows": int(len(rows)),
            "sampled_rows": int(len(sampled_rows)),
            "positive_rows": int(positives),
            "negative_rows": int(negatives),
            "training_loss": float(training_loss),
            "validation_loss": float(validation_loss),
            "training_backend_requested": str(backend.requested),
            "training_backend_kind": str(backend.kind),
            "training_backend_device": str(backend.device),
            "training_backend_reason": str(backend.reason),
        },
    )


def _payoff_training_limit(objective_name: str) -> int:
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    if objective_name == "aggressive":
        return 300_000
    if objective_name == "conservative":
        return 220_000
    return 260_000


def _payoff_epochs(objective_name: str) -> int:
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    if objective_name == "aggressive":
        return 20
    if objective_name == "conservative":
        return 14
    return 17


def _payoff_mlp_hidden_dims(objective_name: str, feature_dim: int) -> tuple[int, int]:
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    width_cap = 112 if objective_name == "aggressive" else (80 if objective_name == "conservative" else 96)
    first = max(16, min(width_cap, max(24, int(feature_dim) // 2)))
    second = max(8, min(48, max(12, first // 2)))
    return int(first), int(second)


def _payoff_target_weight_values(meta: Mapping[str, object]) -> tuple[float, float, float]:
    positive_long = max(0, int(meta["positive_long_rows"]))
    positive_short = max(0, int(meta["positive_short_rows"]))
    neutral = max(0, int(meta["neutral_rows"]))
    positive_total = max(1, positive_long + positive_short)
    neutral_weight_value = max(0.10, min(1.0, (positive_total / max(1, neutral)) * 8.0))
    long_weight_value = max(1.0, min(40.0, (neutral / max(1, positive_long)) * neutral_weight_value))
    short_weight_value = max(1.0, min(40.0, (neutral / max(1, positive_short)) * neutral_weight_value))
    return float(long_weight_value), float(short_weight_value), float(neutral_weight_value)


def _median_row_interval_ms(rows: Sequence[ModelRow]) -> int:
    if len(rows) < 2:
        return 1000
    diffs: list[int] = []
    max_checks = min(10_000, len(rows) - 1)
    stride = max(1, (len(rows) - 1) // max_checks)
    for index in range(0, len(rows) - 1, stride):
        current = int(rows[index].timestamp)
        following = int(rows[index + 1].timestamp)
        if following > current:
            diffs.append(following - current)
        if len(diffs) >= max_checks:
            break
    return max(1, int(median(diffs))) if diffs else 1000


def _intraday_utility_rank_groups(
    rows: Sequence[ModelRow],
    *,
    duration_ms: int = UTILITY_RANK_GROUP_DURATION_MS,
    max_group_rows: int = MAX_UTILITY_RANK_GROUP_ROWS,
) -> list[int]:
    """Build chronological ranking queries that stay below LightGBM's row cap."""

    session_ms = max(1, int(duration_ms))
    row_cap = max(1, min(9999, int(max_group_rows)))
    groups: list[int] = []
    current_session: int | None = None
    current_count = 0
    for row in rows:
        session = int(row.timestamp) // session_ms
        if current_count > 0 and (session != current_session or current_count >= row_cap):
            groups.append(current_count)
            current_count = 0
        current_session = session
        current_count += 1
    if current_count > 0:
        groups.append(current_count)
    return groups


def _payoff_horizon_bars(
    rows: Sequence[ModelRow],
    objective_name: str,
    *,
    max_position_hold_bars: int = 0,
) -> tuple[int, ...]:
    lifecycle_horizon = max(0, int(max_position_hold_bars or 0))
    if lifecycle_horizon > 0:
        return (lifecycle_horizon,)
    interval_ms = _median_row_interval_ms(rows)
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    seconds = (15, 60, 180) if objective_name == "conservative" else ((10, 45, 150) if objective_name == "aggressive" else (15, 60, 240))
    horizons = {
        max(2, min(900, int(round(float(value) * 1000.0 / float(interval_ms)))))
        for value in seconds
    }
    return tuple(sorted(horizons))


def _reference_notional(strategy: StrategyConfig, *, market_type: str) -> float:
    leverage = 1.0 if str(market_type).lower() == "spot" else max(1.0, float(strategy.leverage))
    return max(
        0.0,
        1000.0 * stop_loss_sized_notional_pct(strategy, market_type, leverage=leverage),
    )


def _path_exit_reference_price(
    rows: Sequence[ModelRow],
    *,
    entry_index: int,
    entry_price: float,
    horizon_bars: int,
    side: int,
    strategy: StrategyConfig,
) -> tuple[float, int]:
    entry = float(entry_price)
    if entry <= 0.0 or not math.isfinite(entry):
        return 0.0, entry_index
    stop_pct = max(0.0, float(strategy.stop_loss_pct))
    take_pct = max(0.0, float(strategy.take_profit_pct))
    end_index = min(len(rows) - 1, entry_index + max(1, int(horizon_bars)))
    for index in range(entry_index + 1, end_index + 1):
        row = rows[index]
        close = float(row.close)
        high = float(row.high) if row.high is not None else close
        low = float(row.low) if row.low is not None else close
        if not all(math.isfinite(value) and value > 0.0 for value in (close, high, low)):
            continue
        high = max(high, close, low)
        low = min(low, close, high)
        if side > 0:
            stop_price = entry * (1.0 - stop_pct)
            take_price = entry * (1.0 + take_pct)
            if stop_pct > 0.0 and low <= stop_price:
                return stop_price, index
            if take_pct > 0.0 and high >= take_price:
                return take_price, index
        else:
            stop_price = entry * (1.0 + stop_pct)
            take_price = max(0.0, entry * (1.0 - take_pct))
            if stop_pct > 0.0 and high >= stop_price:
                return stop_price, index
            if take_pct > 0.0 and low <= take_price:
                return take_price, index
    return float(rows[end_index].close), end_index


def path_net_edge_bps(
    rows: Sequence[ModelRow],
    *,
    signal_index: int,
    horizon_bars: int,
    side: int,
    strategy: StrategyConfig,
    market_type: str,
    symbol_profile: SymbolExecutionProfile | None,
) -> float | None:
    entry_index = signal_index + 1
    if entry_index >= len(rows):
        return None
    entry_reference = float(rows[entry_index].close)
    notional = _reference_notional(strategy, market_type=market_type)
    if entry_reference <= 0.0 or notional <= 0.0 or not math.isfinite(entry_reference):
        return None
    entry_assumptions = market_row_execution_assumptions(
        rows[entry_index],
        strategy,
        symbol_profile=symbol_profile,
        include_range=False,
    )
    entry_fill = simulate_market_fill(
        entry_reference,
        side,
        notional,
        strategy,
        bar_volume_notional=market_row_quote_volume_notional(rows[entry_index], entry_reference),
        daily_volume_notional=market_row_trailing_quote_volume_24h_estimate(rows[entry_index]),
        assumptions=entry_assumptions,
        symbol_profile=symbol_profile,
    )
    if entry_fill.fill_price <= 0.0 or not math.isfinite(entry_fill.fill_price):
        return None
    exit_reference, exit_index = _path_exit_reference_price(
        rows,
        entry_index=entry_index,
        entry_price=entry_fill.fill_price,
        horizon_bars=horizon_bars,
        side=side,
        strategy=strategy,
    )
    if exit_reference <= 0.0 or not math.isfinite(exit_reference):
        return None
    exit_assumptions = market_row_execution_assumptions(
        rows[exit_index],
        strategy,
        symbol_profile=symbol_profile,
        include_range=True,
    )
    reported_exit_volume = market_row_reported_quote_volume_notional(rows[exit_index])
    exit_fill = simulate_market_fill(
        exit_reference,
        -side,
        notional,
        strategy,
        bar_volume_notional=(reported_exit_volume if reported_exit_volume > 0.0 else notional * 20.0),
        daily_volume_notional=market_row_trailing_quote_volume_24h_estimate(rows[exit_index]),
        assumptions=exit_assumptions,
        symbol_profile=symbol_profile,
    )
    if exit_fill.fill_price <= 0.0 or not math.isfinite(exit_fill.fill_price):
        return None
    quantity = notional / entry_fill.fill_price
    realized = float(side) * (exit_fill.fill_price - entry_fill.fill_price) * quantity
    fee_rate = max(0.0, float(strategy.taker_fee_bps)) / 10_000.0
    entry_fee = notional * fee_rate
    exit_fee = abs(exit_fill.fill_price * quantity) * fee_rate
    net_pnl = realized - entry_fee - exit_fee
    return float(net_pnl / notional * 10_000.0)


def _directional_payoff_targets_bps(
    rows: Sequence[ModelRow],
    *,
    signal_index: int,
    horizon_bars: int,
    strategy: StrategyConfig,
    market_type: str,
    symbol_profile: SymbolExecutionProfile | None,
) -> tuple[float, float | None] | None:
    long_edge = path_net_edge_bps(
        rows,
        signal_index=signal_index,
        horizon_bars=horizon_bars,
        side=1,
        strategy=strategy,
        market_type=market_type,
        symbol_profile=symbol_profile,
    )
    if long_edge is None:
        return None
    if str(market_type).lower() != "futures":
        return float(long_edge), None
    short_edge = path_net_edge_bps(
        rows,
        signal_index=signal_index,
        horizon_bars=horizon_bars,
        side=-1,
        strategy=strategy,
        market_type=market_type,
        symbol_profile=symbol_profile,
    )
    if short_edge is None:
        return None
    return float(long_edge), float(short_edge)


def _signed_payoff_target_bps(
    rows: Sequence[ModelRow],
    *,
    signal_index: int,
    horizon_bars: int,
    strategy: StrategyConfig,
    market_type: str,
    symbol_profile: SymbolExecutionProfile | None,
) -> float | None:
    action_targets = _directional_payoff_targets_bps(
        rows,
        signal_index=signal_index,
        horizon_bars=horizon_bars,
        strategy=strategy,
        market_type=market_type,
        symbol_profile=symbol_profile,
    )
    if action_targets is None:
        return None
    long_edge, short_edge = action_targets
    if short_edge is None:
        return float(long_edge if long_edge > 0.0 else 0.0)
    best_edge = max(float(long_edge), float(short_edge))
    if best_edge <= 0.0:
        return 0.0
    return float(long_edge if long_edge >= short_edge else -short_edge)


def _payoff_clip_bps(targets: Sequence[float], strategy: StrategyConfig) -> float:
    magnitudes = sorted(abs(float(value)) for value in targets if math.isfinite(float(value)) and abs(float(value)) > 1e-9)
    if not magnitudes:
        return max(5.0, min(80.0, float(strategy.take_profit_pct) * 10_000.0))
    index = max(0, min(len(magnitudes) - 1, int(round((len(magnitudes) - 1) * 0.95))))
    structural_floor = max(float(strategy.stop_loss_pct), float(strategy.take_profit_pct), 0.0005) * 10_000.0
    return max(5.0, min(120.0, max(float(magnitudes[index]), structural_floor * 0.35)))


def _payoff_training_examples(
    rows: Sequence[ModelRow],
    model: TrainedModel,
    *,
    horizon_bars: int,
    strategy: StrategyConfig,
    objective_name: str,
    market_type: str,
    symbol_profile: SymbolExecutionProfile | None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> _PayoffTrainingExamples | None:
    available = max(0, len(rows) - max(1, int(horizon_bars)) - 2)
    if available < 128:
        return None
    sampled_indexes = _even_sample_indexes(available, _payoff_training_limit(objective_name))
    raw_targets: list[float] = []
    raw_long_action_targets: list[float] = []
    raw_short_action_targets: list[float] = []
    valid_rows: list[ModelRow] = []
    valid_source_indexes: list[int] = []
    progress_interval = max(1_000, min(10_000, len(sampled_indexes) // 20 or 1_000))
    for processed, source_index in enumerate(sampled_indexes, start=1):
        row = rows[source_index]
        action_targets = _directional_payoff_targets_bps(
            rows,
            signal_index=source_index,
            horizon_bars=horizon_bars,
            strategy=strategy,
            market_type=market_type,
            symbol_profile=symbol_profile,
        )
        if action_targets is None:
            continue
        long_target, short_target = action_targets
        if not math.isfinite(long_target) or (short_target is not None and not math.isfinite(short_target)):
            continue
        if short_target is None:
            target = float(long_target if long_target > 0.0 else 0.0)
        else:
            best_edge = max(float(long_target), float(short_target))
            target = (
                0.0
                if best_edge <= 0.0
                else float(long_target if long_target >= short_target else -short_target)
            )
        raw_targets.append(float(target))
        raw_long_action_targets.append(float(long_target))
        if short_target is not None:
            raw_short_action_targets.append(float(short_target))
        valid_rows.append(row)
        valid_source_indexes.append(int(source_index))
        if progress_callback is not None and (
            processed == len(sampled_indexes) or processed % progress_interval == 0
        ):
            progress_callback(
                {
                    "processed_samples": int(processed),
                    "sample_count": int(len(sampled_indexes)),
                    "valid_targets": int(len(raw_targets)),
                    "positive_long_targets": int(sum(1 for value in raw_targets if value > 0.0)),
                    "positive_short_targets": int(sum(1 for value in raw_targets if value < 0.0)),
                }
            )
    if len(valid_rows) < 128:
        return None
    positive_long = sum(1 for value in raw_targets if value > 0.0)
    positive_short = sum(1 for value in raw_targets if value < 0.0)
    neutral = len(raw_targets) - positive_long - positive_short
    if positive_long + positive_short < max(16, len(raw_targets) // 500):
        return None
    clip_bps = _payoff_clip_bps(raw_targets, strategy)
    targets = [max(-1.0, min(1.0, float(value) / clip_bps)) for value in raw_targets]
    action_targets_available = len(raw_short_action_targets) == len(valid_rows)
    action_clip_bps = (
        _payoff_clip_bps([*raw_long_action_targets, *raw_short_action_targets], strategy)
        if action_targets_available
        else clip_bps
    )
    long_action_targets = (
        [max(-1.0, min(1.0, float(value) / action_clip_bps)) for value in raw_long_action_targets]
        if action_targets_available
        else None
    )
    short_action_targets = (
        [max(-1.0, min(1.0, float(value) / action_clip_bps)) for value in raw_short_action_targets]
        if action_targets_available
        else None
    )
    return _PayoffTrainingExamples(
        rows=valid_rows,
        targets=targets,
        source_indexes=valid_source_indexes,
        meta={
            "horizon_bars": int(horizon_bars),
            "source_rows": int(len(rows)),
            "sampled_rows": int(len(sampled_indexes)),
            "training_examples": int(len(valid_rows)),
            "positive_long_rows": int(positive_long),
            "positive_short_rows": int(positive_short),
            "neutral_rows": int(neutral),
            "clip_bps": float(clip_bps),
            "action_clip_bps": float(action_clip_bps),
            "long_action_positive_rows": int(sum(1 for value in raw_long_action_targets if value > 0.0)),
            "short_action_positive_rows": int(sum(1 for value in raw_short_action_targets if value > 0.0)),
            "long_action_mean_bps": float(sum(raw_long_action_targets) / len(raw_long_action_targets)),
            "short_action_mean_bps": (
                float(sum(raw_short_action_targets) / len(raw_short_action_targets))
                if raw_short_action_targets
                else 0.0
            ),
            "median_interval_ms": int(_median_row_interval_ms(rows)),
        },
        long_action_targets=long_action_targets,
        short_action_targets=short_action_targets,
    )


def _purged_payoff_train_validation_split(
    examples: _PayoffTrainingExamples,
    *,
    horizon_bars: int,
) -> tuple[list[ModelRow], list[float], list[ModelRow], list[float], dict[str, int]] | None:
    count = len(examples.rows)
    validation_count = max(64, min(count // 5, 35_000))
    if count - validation_count < 160:
        validation_count = max(32, count // 4)
    validation_start = count - validation_count
    if validation_start <= 0 or validation_start >= len(examples.source_indexes):
        return None
    validation_source_start = int(examples.source_indexes[validation_start])
    latest_safe_training_source = validation_source_start - max(1, int(horizon_bars)) - 2
    training_end = 0
    while (
        training_end < validation_start
        and int(examples.source_indexes[training_end]) <= latest_safe_training_source
    ):
        training_end += 1
    if training_end < 160:
        return None
    purged_examples = max(0, validation_start - training_end)
    return (
        examples.rows[:training_end],
        examples.targets[:training_end],
        examples.rows[validation_start:],
        examples.targets[validation_start:],
        {
            "internal_validation_purge_bars": max(1, int(horizon_bars)) + 1,
            "internal_validation_purged_examples": int(purged_examples),
            "training_source_end": int(examples.source_indexes[training_end - 1]),
            "validation_source_start": validation_source_start,
        },
    )


def _purged_action_payoff_train_validation_split(
    examples: _PayoffTrainingExamples,
    *,
    horizon_bars: int,
) -> tuple[
    list[ModelRow],
    list[float],
    list[float],
    list[ModelRow],
    list[float],
    list[float],
    dict[str, int],
] | None:
    long_targets = examples.long_action_targets
    short_targets = examples.short_action_targets
    if (
        long_targets is None
        or short_targets is None
        or len(long_targets) != len(examples.rows)
        or len(short_targets) != len(examples.rows)
    ):
        return None
    signed_split = _purged_payoff_train_validation_split(examples, horizon_bars=horizon_bars)
    if signed_split is None:
        return None
    train_rows, _train_targets, validation_rows, _validation_targets, purge_meta = signed_split
    training_end = len(train_rows)
    validation_start = len(examples.rows) - len(validation_rows)
    return (
        train_rows,
        long_targets[:training_end],
        short_targets[:training_end],
        validation_rows,
        long_targets[validation_start:],
        short_targets[validation_start:],
        purge_meta,
    )


def _train_signed_payoff_ranker_expert(
    rows: Sequence[ModelRow],
    model: TrainedModel,
    *,
    horizon_bars: int,
    objective_name: str,
    market_type: str,
    strategy: StrategyConfig,
    compute_backend: str | None,
    batch_size: int,
    symbol_profile: SymbolExecutionProfile | None = None,
    seed: int = 7,
    examples: _PayoffTrainingExamples | None = None,
) -> HybridExpert | None:
    examples = examples or _payoff_training_examples(
        rows,
        model,
        horizon_bars=horizon_bars,
        strategy=strategy,
        objective_name=objective_name,
        market_type=market_type,
        symbol_profile=symbol_profile,
    )
    if examples is None:
        return None
    example_rows = examples.rows
    meta = examples.meta
    try:
        import torch  # type: ignore
    except Exception:
        return None

    backend = resolve_backend(compute_backend or "auto")
    try:
        device = _torch_device_for_backend(backend)
    except Exception:
        backend = resolve_backend("cpu")
        device = "cpu"
    if len(example_rows) < 128:
        return None
    split = _purged_payoff_train_validation_split(examples, horizon_bars=horizon_bars)
    if split is None:
        return None
    train_rows, train_targets, validation_rows, validation_targets, purge_meta = split
    input_dim = int(model.feature_dim)
    try:
        torch.manual_seed(int(seed) + int(horizon_bars))
        def rows_to_tensor(values: Sequence[ModelRow]):
            tensor = torch.empty((len(values), input_dim), dtype=torch.float32, device=device)
            chunk_size = max(256, min(4096, max(1, int(batch_size or 4096))))
            for start in range(0, len(values), chunk_size):
                end = min(start + chunk_size, len(values))
                tensor[start:end] = torch.tensor(
                    [model._normalize(row.features) for row in values[start:end]],  # noqa: SLF001
                    dtype=torch.float32,
                    device=device,
                )
            return tensor

        x_train = rows_to_tensor(train_rows)
        y_train = torch.tensor(train_targets, dtype=torch.float32, device=device)
        x_validation = rows_to_tensor(validation_rows)
        y_validation = torch.tensor(validation_targets, dtype=torch.float32, device=device)
        long_weight_value, short_weight_value, neutral_weight_value = _payoff_target_weight_values(meta)

        def target_weights(target):
            long_weight = torch.tensor(float(long_weight_value), dtype=torch.float32, device=device)
            short_weight = torch.tensor(float(short_weight_value), dtype=torch.float32, device=device)
            neutral_weight = torch.tensor(float(neutral_weight_value), dtype=torch.float32, device=device)
            return torch.where(
                target > 1e-8,
                long_weight,
                torch.where(target < -1e-8, short_weight, neutral_weight),
            )

        train_target_weights = target_weights(y_train)
        validation_target_weights = target_weights(y_validation)
        weights = (torch.randn((input_dim,), dtype=torch.float32, device=device) * math.sqrt(1.0 / max(1, input_dim))).requires_grad_()
        bias = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
        learning_rate = 0.035
        weight_decay = 4e-4
        batch = max(512, min(max(1, int(batch_size or 8192)), len(train_rows)))
        epochs = _payoff_epochs(objective_name)
        best_state: tuple[object, object] | None = None
        best_validation_loss = float("inf")
        stagnant_epochs = 0

        def payoff_loss(prediction, target, sample_weight):
            delta = prediction - target
            abs_delta = torch.abs(delta)
            huber = torch.where(abs_delta <= 0.10, 0.5 * delta * delta / 0.10, abs_delta - 0.05)
            signal_weight = 1.0 + 4.0 * torch.clamp(torch.abs(target), min=0.0, max=1.0)
            wrong_side = torch.relu(-(prediction * target)) * torch.clamp(torch.abs(target), min=0.0, max=1.0)
            weighted = sample_weight * signal_weight * (huber + 0.35 * wrong_side)
            return weighted.sum() / torch.clamp(sample_weight.sum(), min=1e-6)

        def linear_projection(values):
            projected = torch.mm(values, weights.reshape(input_dim, 1)).reshape(-1)
            return projected + bias

        for _epoch in range(epochs):
            for start in range(0, len(train_rows), batch):
                end = min(start + batch, len(train_rows))
                if weights.grad is not None:
                    weights.grad = None
                if bias.grad is not None:
                    bias.grad = None
                prediction = torch.clamp(linear_projection(x_train[start:end]), min=-1.25, max=1.25)
                loss = payoff_loss(prediction, y_train[start:end], train_target_weights[start:end])
                loss.backward()
                with torch.no_grad():
                    if weight_decay > 0.0:
                        weights.mul_(1.0 - learning_rate * weight_decay)
                    if weights.grad is not None:
                        weights.add_(weights.grad, alpha=-learning_rate)
                    if bias.grad is not None:
                        bias.add_(bias.grad, alpha=-learning_rate)
                    weights.clamp_(min=-4.0, max=4.0)
                    bias.clamp_(min=-4.0, max=4.0)
            with torch.no_grad():
                validation_prediction = torch.clamp(linear_projection(x_validation), min=-1.25, max=1.25)
                validation_loss = float(payoff_loss(validation_prediction, y_validation, validation_target_weights).detach().cpu().item())
            if validation_loss < best_validation_loss - 1e-5:
                best_validation_loss = validation_loss
                best_state = (weights.detach().clone(), bias.detach().clone())
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1
                if stagnant_epochs >= 5:
                    break
        if best_state is not None:
            weights, bias = best_state  # type: ignore[assignment]
        with torch.no_grad():
            training_prediction = torch.clamp(linear_projection(x_train), min=-1.25, max=1.25)
            validation_prediction = torch.clamp(linear_projection(x_validation), min=-1.25, max=1.25)
            training_loss = float(payoff_loss(training_prediction, y_train, train_target_weights).detach().cpu().item())
            validation_loss = float(payoff_loss(validation_prediction, y_validation, validation_target_weights).detach().cpu().item())
            prediction_abs_mean = float(torch.mean(torch.abs(validation_prediction)).detach().cpu().item())
        serialized_weights = [float(value) for value in weights.detach().cpu().tolist()]
        serialized_bias = float(bias.detach().cpu().item())
    except Exception:
        return None

    objective_name = "aggressive" if objective_name == "risky" else objective_name
    sensitivity = 7.0 if objective_name == "aggressive" else (5.5 if objective_name == "conservative" else 6.25)
    interval_ms = int(meta["median_interval_ms"])
    approx_seconds = float(horizon_bars) * float(interval_ms) / 1000.0
    clip_bps = float(meta["clip_bps"])
    return HybridExpert(
        name=f"signed_payoff_ranker_{int(round(approx_seconds))}s",
        kind="signed_payoff_ranker",
        weight=0.0,
        prototypes=[],
        feature_count=input_dim,
        notes="GPU-trained signed after-cost payoff ranker; positive scores favor long, negative scores favor short.",
        params={
            "input_dim": int(input_dim),
            "weights": serialized_weights,
            "bias": serialized_bias,
            "horizon_bars": int(horizon_bars),
            "approx_horizon_seconds": float(approx_seconds),
            "clip_bps": float(clip_bps),
            "deadband_bps": float(max(0.25, min(2.5, clip_bps * 0.06))),
            "sensitivity": float(sensitivity),
            "probability_bias": 0.0,
            "training_rows": int(len(train_rows)),
            "validation_rows": int(len(validation_rows)),
            **purge_meta,
            "source_rows": int(meta["source_rows"]),
            "sampled_rows": int(meta["sampled_rows"]),
            "training_examples": int(meta["training_examples"]),
            "positive_long_rows": int(meta["positive_long_rows"]),
            "positive_short_rows": int(meta["positive_short_rows"]),
            "neutral_rows": int(meta["neutral_rows"]),
            "target_long_weight": float(long_weight_value),
            "target_short_weight": float(short_weight_value),
            "target_neutral_weight": float(neutral_weight_value),
            "optimizer": "manual_sgd_directml",
            "seed": int(seed),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "training_loss": float(training_loss),
            "validation_loss": float(validation_loss),
            "validation_prediction_abs_mean": float(prediction_abs_mean),
            "training_backend_requested": str(backend.requested),
            "training_backend_kind": str(backend.kind),
            "training_backend_device": str(backend.device),
            "training_backend_reason": str(backend.reason),
            "target": "next-entry lifecycle-horizon signed net path payoff after fees/spread/slippage/live-buffer",
            "execution_model_version": EXECUTION_MODEL_VERSION,
            "execution_activity_estimator": EXECUTION_ACTIVITY_ESTIMATOR,
        },
    )


def _train_signed_payoff_mlp_ranker_expert(
    rows: Sequence[ModelRow],
    model: TrainedModel,
    *,
    horizon_bars: int,
    objective_name: str,
    market_type: str,
    strategy: StrategyConfig,
    compute_backend: str | None,
    batch_size: int,
    symbol_profile: SymbolExecutionProfile | None = None,
    seed: int = 17,
    examples: _PayoffTrainingExamples | None = None,
) -> HybridExpert | None:
    examples = examples or _payoff_training_examples(
        rows,
        model,
        horizon_bars=horizon_bars,
        strategy=strategy,
        objective_name=objective_name,
        market_type=market_type,
        symbol_profile=symbol_profile,
    )
    if examples is None:
        return None
    example_rows = examples.rows
    meta = examples.meta
    try:
        import torch  # type: ignore
    except Exception:
        return None

    backend = resolve_backend(compute_backend or "auto")
    try:
        device = _torch_device_for_backend(backend)
    except Exception:
        backend = resolve_backend("cpu")
        device = "cpu"
    if len(example_rows) < 256:
        return None
    split = _purged_payoff_train_validation_split(examples, horizon_bars=horizon_bars)
    if split is None:
        return None
    train_rows, train_targets, validation_rows, validation_targets, purge_meta = split
    input_dim = int(model.feature_dim)
    hidden_1, hidden_2 = _payoff_mlp_hidden_dims(objective_name, input_dim)
    try:
        torch.manual_seed(int(seed) + int(horizon_bars))

        def rows_to_tensor(values: Sequence[ModelRow]):
            tensor = torch.empty((len(values), input_dim), dtype=torch.float32, device=device)
            chunk_size = max(256, min(4096, max(1, int(batch_size or 4096))))
            for start in range(0, len(values), chunk_size):
                end = min(start + chunk_size, len(values))
                tensor[start:end] = torch.tensor(
                    [model._normalize(row.features) for row in values[start:end]],  # noqa: SLF001
                    dtype=torch.float32,
                    device=device,
                )
            return tensor

        x_train = rows_to_tensor(train_rows)
        y_train = torch.tensor(train_targets, dtype=torch.float32, device=device)
        x_validation = rows_to_tensor(validation_rows)
        y_validation = torch.tensor(validation_targets, dtype=torch.float32, device=device)
        long_weight_value, short_weight_value, neutral_weight_value = _payoff_target_weight_values(meta)

        def target_weights(target):
            long_weight = torch.tensor(float(long_weight_value), dtype=torch.float32, device=device)
            short_weight = torch.tensor(float(short_weight_value), dtype=torch.float32, device=device)
            neutral_weight = torch.tensor(float(neutral_weight_value), dtype=torch.float32, device=device)
            return torch.where(
                target > 1e-8,
                long_weight,
                torch.where(target < -1e-8, short_weight, neutral_weight),
            )

        train_target_weights = target_weights(y_train)
        validation_target_weights = target_weights(y_validation)
        scale_1 = math.sqrt(2.0 / max(1, input_dim))
        scale_2 = math.sqrt(2.0 / max(1, hidden_1))
        scale_3 = math.sqrt(1.0 / max(1, hidden_2))
        w1 = (torch.randn((input_dim, hidden_1), dtype=torch.float32, device=device) * scale_1).requires_grad_()
        b1 = torch.zeros((hidden_1,), dtype=torch.float32, device=device, requires_grad=True)
        w2 = (torch.randn((hidden_1, hidden_2), dtype=torch.float32, device=device) * scale_2).requires_grad_()
        b2 = torch.zeros((hidden_2,), dtype=torch.float32, device=device, requires_grad=True)
        w3 = (torch.randn((hidden_2, 1), dtype=torch.float32, device=device) * scale_3).requires_grad_()
        b3 = torch.zeros((1,), dtype=torch.float32, device=device, requires_grad=True)
        parameters = (w1, b1, w2, b2, w3, b3)
        learning_rate = 0.018 if objective_name == "conservative" else 0.022
        weight_decay = 3e-4
        batch = max(512, min(max(1, int(batch_size or 8192)), len(train_rows)))
        epochs = max(8, _payoff_epochs(objective_name) + 4)
        best_state: tuple[object, ...] | None = None
        best_validation_loss = float("inf")
        stagnant_epochs = 0

        def forward(values):
            hidden = torch.relu(torch.mm(values, w1) + b1)
            hidden = torch.relu(torch.mm(hidden, w2) + b2)
            return torch.tanh(torch.mm(hidden, w3).reshape(-1) + b3.reshape(()))

        def payoff_loss(prediction, target, sample_weight):
            delta = prediction - target
            abs_delta = torch.abs(delta)
            huber = torch.where(abs_delta <= 0.08, 0.5 * delta * delta / 0.08, abs_delta - 0.04)
            signal_weight = 1.0 + 5.0 * torch.clamp(torch.abs(target), min=0.0, max=1.0)
            wrong_side = torch.relu(-(prediction * target)) * torch.clamp(torch.abs(target), min=0.0, max=1.0)
            overtrade_neutral = torch.where(
                torch.abs(target) <= 1e-8,
                0.15 * torch.abs(prediction),
                torch.zeros_like(prediction),
            )
            weighted = sample_weight * signal_weight * (huber + 0.45 * wrong_side + overtrade_neutral)
            return weighted.sum() / torch.clamp(sample_weight.sum(), min=1e-6)

        for _epoch in range(epochs):
            for start in range(0, len(train_rows), batch):
                end = min(start + batch, len(train_rows))
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad = None
                loss = payoff_loss(forward(x_train[start:end]), y_train[start:end], train_target_weights[start:end])
                loss.backward()
                with torch.no_grad():
                    for parameter in (w1, w2, w3):
                        parameter.mul_(1.0 - learning_rate * weight_decay)
                    for parameter in parameters:
                        if parameter.grad is None:
                            continue
                        parameter.add_(torch.clamp(parameter.grad, min=-2.5, max=2.5), alpha=-learning_rate)
                        parameter.clamp_(min=-5.0, max=5.0)
            with torch.no_grad():
                validation_prediction = forward(x_validation)
                validation_loss = float(payoff_loss(validation_prediction, y_validation, validation_target_weights).detach().cpu().item())
            if validation_loss < best_validation_loss - 1e-5:
                best_validation_loss = validation_loss
                best_state = tuple(parameter.detach().clone() for parameter in parameters)
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1
                if stagnant_epochs >= 5:
                    break
        if best_state is not None:
            w1, b1, w2, b2, w3, b3 = best_state  # type: ignore[assignment]
        with torch.no_grad():
            training_prediction = forward(x_train)
            validation_prediction = forward(x_validation)
            training_loss = float(payoff_loss(training_prediction, y_train, train_target_weights).detach().cpu().item())
            validation_loss = float(payoff_loss(validation_prediction, y_validation, validation_target_weights).detach().cpu().item())
            prediction_abs_mean = float(torch.mean(torch.abs(validation_prediction)).detach().cpu().item())
        layers = [
            {
                "weights": [[float(value) for value in row] for row in w1.detach().cpu().tolist()],
                "bias": [float(value) for value in b1.detach().cpu().tolist()],
                "activation": "relu",
            },
            {
                "weights": [[float(value) for value in row] for row in w2.detach().cpu().tolist()],
                "bias": [float(value) for value in b2.detach().cpu().tolist()],
                "activation": "relu",
            },
            {
                "weights": [[float(value) for value in row] for row in w3.detach().cpu().tolist()],
                "bias": [float(value) for value in b3.detach().cpu().tolist()],
                "activation": "tanh",
            },
        ]
    except Exception:
        return None

    objective_name = "aggressive" if objective_name == "risky" else objective_name
    sensitivity = 8.0 if objective_name == "aggressive" else (6.25 if objective_name == "conservative" else 7.0)
    interval_ms = int(meta["median_interval_ms"])
    approx_seconds = float(horizon_bars) * float(interval_ms) / 1000.0
    clip_bps = float(meta["clip_bps"])
    return HybridExpert(
        name=f"signed_payoff_mlp_ranker_{int(round(approx_seconds))}s_seed{int(seed)}",
        kind="signed_payoff_mlp_ranker",
        weight=0.0,
        prototypes=[],
        feature_count=input_dim,
        notes="GPU-trained nonlinear signed after-cost payoff network; positive scores favor long, negative scores favor short.",
        params={
            "input_dim": int(input_dim),
            "layers": layers,
            "output_activation": "signed_payoff",
            "hidden_1": int(hidden_1),
            "hidden_2": int(hidden_2),
            "horizon_bars": int(horizon_bars),
            "approx_horizon_seconds": float(approx_seconds),
            "clip_bps": float(clip_bps),
            "deadband_bps": float(max(0.20, min(2.0, clip_bps * 0.045))),
            "sensitivity": float(sensitivity),
            "probability_bias": 0.0,
            "training_rows": int(len(train_rows)),
            "validation_rows": int(len(validation_rows)),
            **purge_meta,
            "source_rows": int(meta["source_rows"]),
            "sampled_rows": int(meta["sampled_rows"]),
            "training_examples": int(meta["training_examples"]),
            "positive_long_rows": int(meta["positive_long_rows"]),
            "positive_short_rows": int(meta["positive_short_rows"]),
            "neutral_rows": int(meta["neutral_rows"]),
            "target_long_weight": float(long_weight_value),
            "target_short_weight": float(short_weight_value),
            "target_neutral_weight": float(neutral_weight_value),
            "optimizer": "manual_sgd_directml",
            "seed": int(seed),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "training_loss": float(training_loss),
            "validation_loss": float(validation_loss),
            "validation_prediction_abs_mean": float(prediction_abs_mean),
            "training_backend_requested": str(backend.requested),
            "training_backend_kind": str(backend.kind),
            "training_backend_device": str(backend.device),
            "training_backend_reason": str(backend.reason),
            "target": "nonlinear next-entry lifecycle-horizon signed net path payoff after fees/spread/slippage/live-buffer",
            "execution_model_version": EXECUTION_MODEL_VERSION,
            "execution_activity_estimator": EXECUTION_ACTIVITY_ESTIMATOR,
        },
    )


def _train_signed_payoff_lightgbm_ranker_expert(
    rows: Sequence[ModelRow],
    model: TrainedModel,
    *,
    horizon_bars: int,
    objective_name: str,
    market_type: str,
    strategy: StrategyConfig,
    compute_backend: str | None,
    batch_size: int,
    symbol_profile: SymbolExecutionProfile | None = None,
    seed: int = 71,
    training_mode: str = "binary_hurdle",
    examples: _PayoffTrainingExamples | None = None,
    failure_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> HybridExpert | None:
    """Train calibrated action hurdles on causal, after-cost lifecycle payoffs."""

    def fail(
        stage: str,
        reason: str,
        exc: Exception | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        if failure_callback is not None:
            payload: dict[str, object] = {
                "expert_kind": "signed_payoff_lightgbm_ranker",
                "seed": int(seed),
                "training_mode": str(training_mode),
                "stage": str(stage),
                "reason": str(reason),
            }
            if exc is not None:
                payload["exception_type"] = type(exc).__name__
                payload["exception"] = str(exc)[:500]
            if details is not None:
                payload.update(dict(details))
            failure_callback(payload)
        return None

    del batch_size
    training_mode = str(training_mode or "binary_hurdle").strip().lower()
    if training_mode not in {"binary_hurdle", "daily_utility_rank"}:
        return fail("configuration", "unsupported_action_payoff_training_mode")
    if str(market_type).lower() != "futures":
        return fail("market", "action_payoff_tree_requires_futures_long_short_targets")
    examples = examples or _payoff_training_examples(
        rows,
        model,
        horizon_bars=horizon_bars,
        strategy=strategy,
        objective_name=objective_name,
        market_type=market_type,
        symbol_profile=symbol_profile,
    )
    if examples is None:
        return fail("examples", "payoff_training_examples_unavailable")
    split = _purged_action_payoff_train_validation_split(examples, horizon_bars=horizon_bars)
    if split is None:
        return fail("split", "purged_action_payoff_training_validation_split_unavailable")
    (
        train_rows,
        train_long_targets,
        train_short_targets,
        validation_rows,
        validation_long_targets,
        validation_short_targets,
        purge_meta,
    ) = split

    # Keep early stopping, probability calibration, and edge gating chronologically
    # distinct. Each boundary is purged by the target horizon so no lifecycle path
    # can contribute labels to both adjacent stages.
    validation_start = len(examples.rows) - len(validation_rows)
    validation_source_indexes = examples.source_indexes[validation_start:]
    segment = len(validation_rows) // 3
    calibration_start = segment
    gate_start = segment * 2
    if calibration_start < 32 or gate_start - calibration_start < 32 or len(validation_rows) - gate_start < 32:
        return fail("split", "nested_hurdle_validation_segments_too_small")

    def purged_segment_end(start: int, end: int, next_start: int) -> int:
        latest_safe_source = (
            int(validation_source_indexes[next_start]) - max(1, int(horizon_bars)) - 2
        )
        cursor = int(start)
        while cursor < int(end) and int(validation_source_indexes[cursor]) <= latest_safe_source:
            cursor += 1
        return cursor

    tuning_end = purged_segment_end(0, calibration_start, calibration_start)
    calibration_end = purged_segment_end(calibration_start, gate_start, gate_start)
    tuning_rows = validation_rows[:tuning_end]
    tuning_long_targets = validation_long_targets[:tuning_end]
    tuning_short_targets = validation_short_targets[:tuning_end]
    calibration_rows = validation_rows[calibration_start:calibration_end]
    calibration_long_targets = validation_long_targets[calibration_start:calibration_end]
    calibration_short_targets = validation_short_targets[calibration_start:calibration_end]
    gate_rows = validation_rows[gate_start:]
    gate_long_targets = validation_long_targets[gate_start:]
    gate_short_targets = validation_short_targets[gate_start:]
    if min(len(tuning_rows), len(calibration_rows), len(gate_rows)) < 32:
        return fail("split", "purged_nested_hurdle_validation_segments_too_small")
    nested_split_meta = {
        "internal_tuning_rows": int(len(tuning_rows)),
        "internal_calibration_rows": int(len(calibration_rows)),
        "internal_edge_gate_rows": int(len(gate_rows)),
        "internal_tuning_calibration_purged_examples": int(calibration_start - tuning_end),
        "internal_calibration_gate_purged_examples": int(gate_start - calibration_end),
        "internal_calibration_source_start": int(validation_source_indexes[calibration_start]),
        "internal_edge_gate_source_start": int(validation_source_indexes[gate_start]),
    }

    try:
        import lightgbm as lgb  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        return fail("import", "lightgbm_or_numpy_unavailable", exc)

    backend = resolve_backend(compute_backend or "auto")
    use_gpu = backend.kind != "cpu"
    try:
        platform_id = int(os.getenv("SIMPLE_AI_TRADING_OPENCL_PLATFORM_ID", "0"))
        device_id = int(os.getenv("SIMPLE_AI_TRADING_OPENCL_DEVICE_ID", "0"))
    except ValueError as exc:
        return fail("backend", "invalid_opencl_platform_or_device_id", exc)
    input_dim = int(model.feature_dim)
    objective_name_normalized = "aggressive" if objective_name == "risky" else objective_name
    try:
        def feature_matrix(values: Sequence[ModelRow]):
            return np.asarray(
                [model._normalize(row.features) for row in values],  # noqa: SLF001
                dtype=np.float32,
            )

        x_train = feature_matrix(train_rows)
        x_tuning = feature_matrix(tuning_rows)
        x_calibration = feature_matrix(calibration_rows)
        x_gate = feature_matrix(gate_rows)
        y_train_long = np.asarray(train_long_targets, dtype=np.float32)
        y_train_short = np.asarray(train_short_targets, dtype=np.float32)
        y_tuning_long = np.asarray(tuning_long_targets, dtype=np.float32)
        y_tuning_short = np.asarray(tuning_short_targets, dtype=np.float32)
        y_calibration_long = np.asarray(calibration_long_targets, dtype=np.float32)
        y_calibration_short = np.asarray(calibration_short_targets, dtype=np.float32)
        y_gate_long = np.asarray(gate_long_targets, dtype=np.float32)
        y_gate_short = np.asarray(gate_short_targets, dtype=np.float32)
        matrices = (x_train, x_tuning, x_calibration, x_gate)
        if any(values.ndim != 2 for values in matrices):
            return fail("arrays", "training_arrays_are_not_matrices")
        if any(values.shape[1] != input_dim for values in matrices):
            return fail("arrays", "training_feature_dimension_mismatch")
        if not all(
            len(long_targets) == len(short_targets) == len(features)
            for long_targets, short_targets, features in (
                (y_train_long, y_train_short, x_train),
                (y_tuning_long, y_tuning_short, x_tuning),
                (y_calibration_long, y_calibration_short, x_calibration),
                (y_gate_long, y_gate_short, x_gate),
            )
        ):
            return fail("arrays", "action_payoff_target_length_mismatch")

        minimum_class_rows = max(8, min(64, len(gate_rows) // 100))
        for stage_name, long_targets, short_targets in (
            ("training", y_train_long, y_train_short),
            ("tuning", y_tuning_long, y_tuning_short),
            ("calibration", y_calibration_long, y_calibration_short),
            ("edge_gate", y_gate_long, y_gate_short),
        ):
            for side, targets in (("long", long_targets), ("short", short_targets)):
                positives = int(np.sum(targets > 0.0))
                negatives = int(len(targets) - positives)
                if min(positives, negatives) < minimum_class_rows:
                    return fail(
                        "classes",
                        "insufficient_profitable_and_nonprofitable_action_paths",
                        details={
                            "class_stage": stage_name,
                            "class_side": side,
                            "positive_rows": positives,
                            "nonpositive_rows": negatives,
                            "minimum_class_rows": int(minimum_class_rows),
                        },
                    )

        leaves = (
            63
            if objective_name_normalized == "aggressive"
            else (31 if objective_name_normalized == "conservative" else 47)
        )
        parameters: dict[str, object] = {
            "objective": "lambdarank" if training_mode == "daily_utility_rank" else "binary",
            "metric": "ndcg" if training_mode == "daily_utility_rank" else "binary_logloss",
            "learning_rate": 0.035,
            "num_leaves": leaves,
            "max_depth": 7,
            "min_data_in_leaf": max(24, min(128, len(train_rows) // 300)),
            "feature_fraction": 0.82,
            "bagging_fraction": 0.82,
            "bagging_freq": 1,
            "lambda_l1": 1e-4,
            "lambda_l2": 2e-3,
            "max_bin": 63,
            "seed": int(seed),
            "feature_fraction_seed": int(seed) + 1,
            "bagging_seed": int(seed) + 2,
            "data_random_seed": int(seed) + 3,
            "verbosity": -1,
            "num_threads": max(1, min(16, os.cpu_count() or 1)),
            "device_type": "gpu" if use_gpu else "cpu",
        }
        if training_mode == "daily_utility_rank":
            parameters.update(
                {
                    "ndcg_eval_at": [10, 30],
                    "label_gain": [0, 1, 3, 7, 15],
                    "lambdarank_truncation_level": 31,
                    "lambdarank_norm": True,
                }
            )
        if use_gpu:
            parameters.update(
                {
                    "gpu_platform_id": platform_id,
                    "gpu_device_id": device_id,
                    "gpu_use_dp": False,
                }
            )

        def binary_labels(targets):
            return (targets > 0.0).astype(np.float32)

        def economic_weights(targets):
            # Magnitude weighting changes ranking emphasis, not the probability
            # mapping: the latter is refit without weights on a separate period.
            return (1.0 + 2.0 * np.clip(np.abs(targets), 0.0, 1.0)).astype(np.float32)

        def relevance_thresholds(targets) -> tuple[float, float, float]:
            positive = targets[targets > 0.0]
            if len(positive) < minimum_class_rows:
                raise RuntimeError("insufficient_positive_paths_for_utility_relevance_labels")
            return tuple(float(value) for value in np.quantile(positive, [0.25, 0.50, 0.75]))

        def relevance_labels(targets, thresholds):
            labels = np.zeros(len(targets), dtype=np.int32)
            labels[targets > 0.0] = 1
            for relevance, threshold in enumerate(thresholds, start=2):
                labels[targets > float(threshold)] = relevance
            return labels

        def train_action_model(
            target_name,
            train_targets,
            tuning_targets,
            calibration_features,
            gate_features,
            seed_offset,
        ):
            action_parameters = dict(parameters)
            action_parameters.update(
                {
                    "seed": int(seed) + int(seed_offset),
                    "feature_fraction_seed": int(seed) + int(seed_offset) + 1,
                    "bagging_seed": int(seed) + int(seed_offset) + 2,
                    "data_random_seed": int(seed) + int(seed_offset) + 3,
                }
            )
            relevance_cutoffs = (
                relevance_thresholds(train_targets)
                if training_mode == "daily_utility_rank"
                else None
            )
            train_labels = (
                relevance_labels(train_targets, relevance_cutoffs)
                if relevance_cutoffs is not None
                else binary_labels(train_targets)
            )
            tuning_labels = (
                relevance_labels(tuning_targets, relevance_cutoffs)
                if relevance_cutoffs is not None
                else binary_labels(tuning_targets)
            )
            train_dataset = lgb.Dataset(
                x_train,
                label=train_labels,
                weight=economic_weights(train_targets),
                group=_intraday_utility_rank_groups(train_rows) if relevance_cutoffs is not None else None,
                free_raw_data=False,
            )
            tuning_dataset = lgb.Dataset(
                x_tuning,
                label=tuning_labels,
                weight=economic_weights(tuning_targets),
                group=_intraday_utility_rank_groups(tuning_rows) if relevance_cutoffs is not None else None,
                reference=train_dataset,
                free_raw_data=False,
            )
            booster = lgb.train(
                action_parameters,
                train_dataset,
                num_boost_round=320,
                valid_sets=[tuning_dataset],
                valid_names=["tuning"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, first_metric_only=True, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            best_iteration = max(1, int(booster.best_iteration or booster.current_iteration()))
            tuning_margin = booster.predict(x_tuning, num_iteration=best_iteration, raw_score=True)
            calibration_margin = booster.predict(
                calibration_features,
                num_iteration=best_iteration,
                raw_score=True,
            )
            gate_margin = booster.predict(gate_features, num_iteration=best_iteration, raw_score=True)
            model_dump = booster.dump_model(num_iteration=best_iteration)
            tree_info = model_dump.get("tree_info")
            if not isinstance(tree_info, list) or not tree_info:
                raise RuntimeError(f"trained_{target_name}_booster_has_no_serializable_trees")
            if not all(
                bool(np.all(np.isfinite(values)))
                for values in (tuning_margin, calibration_margin, gate_margin)
            ):
                raise RuntimeError(f"trained_{target_name}_booster_produced_non_finite_margins")
            return {
                "best_iteration": best_iteration,
                "model_dump": model_dump,
                "tree_info": tree_info,
                "tuning_margin": tuning_margin,
                "calibration_margin": calibration_margin,
                "gate_margin": gate_margin,
                "relevance_cutoffs": relevance_cutoffs,
            }

        long_fit = train_action_model(
            "long",
            y_train_long,
            y_tuning_long,
            x_calibration,
            x_gate,
            0,
        )
        short_fit = train_action_model(
            "short",
            y_train_short,
            y_tuning_short,
            x_calibration,
            x_gate,
            10_000,
        )
    except Exception as exc:
        return fail("training", "lightgbm_training_failed", exc)

    clip_bps = float(examples.meta["action_clip_bps"])
    interval_ms = int(examples.meta["median_interval_ms"])
    approx_seconds = float(horizon_bars) * float(interval_ms) / 1000.0
    sensitivity = 8.0 if objective_name_normalized == "aggressive" else (6.25 if objective_name_normalized == "conservative" else 7.0)
    deadband_bps = float(max(0.50, min(2.0, clip_bps * 0.025)))
    deadband = min(0.95, deadband_bps / max(1e-9, clip_bps))

    def sigmoid_array(values):
        clipped = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def binary_log_loss(probabilities, labels) -> float:
        bounded = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
        return float(np.mean(-(labels * np.log(bounded) + (1.0 - labels) * np.log(1.0 - bounded))))

    def fit_platt(raw_margin, labels) -> tuple[float, float]:
        margin = np.asarray(raw_margin, dtype=np.float64)
        target = np.asarray(labels, dtype=np.float64)
        slope = 1.0
        intercept = 0.0
        regularization = 1e-3

        def objective(candidate_slope: float, candidate_intercept: float) -> float:
            probability = sigmoid_array(candidate_slope * margin + candidate_intercept)
            return binary_log_loss(probability, target) + 0.5 * regularization * (candidate_slope - 1.0) ** 2

        current = objective(slope, intercept)
        for _iteration in range(60):
            probability = sigmoid_array(slope * margin + intercept)
            residual = probability - target
            curvature = np.maximum(probability * (1.0 - probability), 1e-8)
            gradient = np.asarray(
                [
                    float(np.mean(residual * margin)) + regularization * (slope - 1.0),
                    float(np.mean(residual)),
                ],
                dtype=np.float64,
            )
            hessian = np.asarray(
                [
                    [float(np.mean(curvature * margin * margin)) + regularization, float(np.mean(curvature * margin))],
                    [float(np.mean(curvature * margin)), float(np.mean(curvature)) + 1e-9],
                ],
                dtype=np.float64,
            )
            step = np.linalg.solve(hessian, gradient)
            if float(np.max(np.abs(step))) < 1e-8:
                break
            accepted = False
            scale = 1.0
            for _line_search in range(20):
                candidate_slope = max(0.0, min(100.0, slope - scale * float(step[0])))
                candidate_intercept = max(-100.0, min(100.0, intercept - scale * float(step[1])))
                candidate = objective(candidate_slope, candidate_intercept)
                if math.isfinite(candidate) and candidate <= current + 1e-12:
                    slope = candidate_slope
                    intercept = candidate_intercept
                    current = candidate
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                break
        return float(slope), float(intercept)

    def rank_auc(labels, scores) -> float:
        target = np.asarray(labels, dtype=np.int8)
        values = np.asarray(scores, dtype=np.float64)
        positive_count = int(np.sum(target == 1))
        negative_count = int(np.sum(target == 0))
        if positive_count <= 0 or negative_count <= 0:
            return 0.5
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        cursor = 0
        while cursor < len(order):
            end = cursor + 1
            while end < len(order) and values[order[end]] == values[order[cursor]]:
                end += 1
            average_rank = (cursor + 1 + end) / 2.0
            ranks[order[cursor:end]] = average_rank
            cursor = end
        positive_rank_sum = float(np.sum(ranks[target == 1]))
        return float(
            (positive_rank_sum - positive_count * (positive_count + 1) / 2.0)
            / (positive_count * negative_count)
        )

    def expected_calibration_error(probabilities, labels, bins: int = 10) -> float:
        probability = np.asarray(probabilities, dtype=np.float64)
        target = np.asarray(labels, dtype=np.float64)
        total = max(1, len(target))
        error = 0.0
        for bin_index in range(max(1, int(bins))):
            low = bin_index / float(bins)
            high = (bin_index + 1) / float(bins)
            mask = (probability >= low) & (probability < high if bin_index + 1 < bins else probability <= high)
            count = int(np.sum(mask))
            if count <= 0:
                continue
            error += count / total * abs(float(np.mean(probability[mask])) - float(np.mean(target[mask])))
        return float(error)

    minimum_auc = 0.520 if objective_name_normalized == "conservative" else (0.510 if objective_name_normalized == "aggressive" else 0.515)

    def calibrate_and_gate_action(side: str, fit, calibration_targets, gate_targets) -> dict[str, object]:
        calibration_labels = (calibration_targets > 0.0).astype(np.float64)
        gate_labels = (gate_targets > 0.0).astype(np.float64)
        raw_calibration_probability = sigmoid_array(fit["calibration_margin"])
        slope, intercept = fit_platt(fit["calibration_margin"], calibration_labels)
        calibrated_probability = sigmoid_array(slope * fit["calibration_margin"] + intercept)
        positive_mean = float(np.mean(calibration_targets[calibration_targets > 0.0]))
        nonpositive_mean = float(np.mean(calibration_targets[calibration_targets <= 0.0]))
        gate_probability = sigmoid_array(slope * fit["gate_margin"] + intercept)
        gate_expected = gate_probability * positive_mean + (1.0 - gate_probability) * nonpositive_mean
        auc = rank_auc(gate_labels, fit["gate_margin"])
        top_count = max(16, min(len(gate_targets), int(math.ceil(len(gate_targets) * 0.10))))
        top_indexes = np.argsort(gate_expected, kind="mergesort")[-top_count:]
        top_mean_edge_bps = float(np.mean(gate_targets[top_indexes]) * clip_bps)
        top_hit_rate = float(np.mean(gate_targets[top_indexes] > 0.0))
        tail_diagnostics: dict[str, dict[str, float | int]] = {}
        for fraction in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
            tail_count = max(16, min(len(gate_targets), int(math.ceil(len(gate_targets) * fraction))))
            tail_indexes = np.argsort(gate_expected, kind="mergesort")[-tail_count:]
            tail_values = gate_targets[tail_indexes]
            tail_diagnostics[f"{fraction:.3f}"] = {
                "rows": int(tail_count),
                "mean_edge_bps": float(np.mean(tail_values) * clip_bps),
                "hit_rate": float(np.mean(tail_values > 0.0)),
            }
        actionable_mask = gate_expected > deadband
        actionable_count = int(np.sum(actionable_mask))
        actionable_mean_edge_bps = (
            float(np.mean(gate_targets[actionable_mask]) * clip_bps) if actionable_count > 0 else 0.0
        )
        actionable_hit_rate = (
            float(np.mean(gate_targets[actionable_mask] > 0.0)) if actionable_count > 0 else 0.0
        )
        minimum_actionable = max(16, len(gate_targets) // 1000)
        reject_reasons: list[str] = []
        if slope <= 1e-9:
            reject_reasons.append("nonpositive_calibration_slope")
        if auc < minimum_auc:
            reject_reasons.append("insufficient_gate_auc")
        if top_mean_edge_bps <= deadband_bps:
            reject_reasons.append("nonpositive_top_decile_edge_after_hurdle")
        if actionable_count < minimum_actionable:
            reject_reasons.append("insufficient_actionable_gate_rows")
        if actionable_mean_edge_bps <= 0.0:
            reject_reasons.append("nonpositive_actionable_gate_edge")
        return {
            "side": side,
            "enabled": not reject_reasons,
            "reject_reasons": reject_reasons,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "positive_mean": positive_mean,
            "nonpositive_mean": nonpositive_mean,
            "gate_expected": gate_expected,
            "gate_auc": auc,
            "calibration_brier_raw": float(np.mean((raw_calibration_probability - calibration_labels) ** 2)),
            "calibration_brier_calibrated": float(np.mean((calibrated_probability - calibration_labels) ** 2)),
            "calibration_log_loss_raw": binary_log_loss(raw_calibration_probability, calibration_labels),
            "calibration_log_loss_calibrated": binary_log_loss(calibrated_probability, calibration_labels),
            "calibration_ece_raw": expected_calibration_error(raw_calibration_probability, calibration_labels),
            "calibration_ece_calibrated": expected_calibration_error(calibrated_probability, calibration_labels),
            "gate_positive_rate": float(np.mean(gate_labels)),
            "gate_expected_quantiles": [
                float(value) for value in np.quantile(gate_expected, [0.01, 0.10, 0.50, 0.90, 0.99])
            ],
            "gate_top_decile_rows": int(top_count),
            "gate_top_decile_mean_edge_bps": top_mean_edge_bps,
            "gate_top_decile_hit_rate": top_hit_rate,
            "gate_tail_diagnostics": tail_diagnostics,
            "gate_actionable_rows": actionable_count,
            "gate_actionable_mean_edge_bps": actionable_mean_edge_bps,
            "gate_actionable_hit_rate": actionable_hit_rate,
        }

    try:
        long_diagnostic = calibrate_and_gate_action(
            "long",
            long_fit,
            y_calibration_long,
            y_gate_long,
        )
        short_diagnostic = calibrate_and_gate_action(
            "short",
            short_fit,
            y_calibration_short,
            y_gate_short,
        )
    except Exception as exc:
        return fail("calibration", "action_hurdle_calibration_failed", exc)

    enabled_sides = [
        side
        for side, diagnostic in (("long", long_diagnostic), ("short", short_diagnostic))
        if bool(diagnostic["enabled"])
    ]
    diagnostic_payload = {
        "minimum_gate_auc": float(minimum_auc),
        "long_gate_diagnostics": {key: value for key, value in long_diagnostic.items() if key != "gate_expected"},
        "short_gate_diagnostics": {key: value for key, value in short_diagnostic.items() if key != "gate_expected"},
    }
    if not enabled_sides:
        return fail("edge_gate", "no_action_side_passed_hurdle_edge_gate", details=diagnostic_payload)

    long_gate_expected = (
        long_diagnostic["gate_expected"]
        if bool(long_diagnostic["enabled"])
        else np.full(len(gate_rows), -1.0, dtype=np.float64)
    )
    short_gate_expected = (
        short_diagnostic["gate_expected"]
        if bool(short_diagnostic["enabled"])
        else np.full(len(gate_rows), -1.0, dtype=np.float64)
    )
    predicted_long = long_gate_expected >= short_gate_expected
    predicted_best = np.where(predicted_long, long_gate_expected, short_gate_expected)
    realized_selected = np.where(predicted_long, y_gate_long, y_gate_short)
    actionable_mask = predicted_best > deadband
    actionable_count = int(np.sum(actionable_mask))
    actionable_realized = realized_selected[actionable_mask]
    actionable_mean_edge_bps = (
        float(np.mean(actionable_realized) * clip_bps) if actionable_count > 0 else 0.0
    )
    actionable_hit_rate = (
        float(np.mean(actionable_realized > 0.0)) if actionable_count > 0 else 0.0
    )
    if actionable_count < max(16, len(gate_rows) // 1000) or actionable_mean_edge_bps <= 0.0:
        return fail(
            "edge_gate",
            "combined_action_hurdle_failed_after_cost_edge_gate",
            details={
                **diagnostic_payload,
                "validation_actionable_rows": actionable_count,
                "validation_actionable_realized_mean_edge_bps": actionable_mean_edge_bps,
                "validation_actionable_hit_rate": actionable_hit_rate,
            },
        )

    expert_name_prefix = (
        "action_payoff_daily_utility_rank_lightgbm"
        if training_mode == "daily_utility_rank"
        else "action_payoff_hurdle_lightgbm"
    )
    optimizer_name = (
        "lightgbm_intraday_grouped_lambdarank_platt_hurdle_action_value"
        if training_mode == "daily_utility_rank"
        else "lightgbm_economic_weighted_binary_platt_hurdle_action_value"
    )
    expert = HybridExpert(
        name=f"{expert_name_prefix}_{int(round(approx_seconds))}s_seed{int(seed)}",
        kind="signed_payoff_lightgbm_ranker",
        weight=0.0,
        prototypes=[],
        feature_count=input_dim,
        notes="OpenCL GPU-trained calibrated long/short profitability hurdles with explicit abstention and dependency-free inference.",
        params={
            "input_dim": input_dim,
            "payoff_tree_schema": "action_value_hurdle_v1",
            "payoff_training_mode": training_mode,
            "utility_rank_group_duration_ms": (
                UTILITY_RANK_GROUP_DURATION_MS if training_mode == "daily_utility_rank" else 0
            ),
            "utility_rank_max_group_rows": (
                MAX_UTILITY_RANK_GROUP_ROWS if training_mode == "daily_utility_rank" else 0
            ),
            "long_classifier_tree_info": long_fit["tree_info"],
            "short_classifier_tree_info": short_fit["tree_info"],
            "long_classifier_average_output": bool(long_fit["model_dump"].get("average_output", False)),
            "short_classifier_average_output": bool(short_fit["model_dump"].get("average_output", False)),
            "long_enabled": bool(long_diagnostic["enabled"]),
            "short_enabled": bool(short_diagnostic["enabled"]),
            "long_calibration_slope": float(long_diagnostic["calibration_slope"]),
            "long_calibration_intercept": float(long_diagnostic["calibration_intercept"]),
            "short_calibration_slope": float(short_diagnostic["calibration_slope"]),
            "short_calibration_intercept": float(short_diagnostic["calibration_intercept"]),
            "long_positive_mean": float(long_diagnostic["positive_mean"]),
            "long_nonpositive_mean": float(long_diagnostic["nonpositive_mean"]),
            "short_positive_mean": float(short_diagnostic["positive_mean"]),
            "short_nonpositive_mean": float(short_diagnostic["nonpositive_mean"]),
            "horizon_bars": int(horizon_bars),
            "approx_horizon_seconds": float(approx_seconds),
            "clip_bps": clip_bps,
            "deadband_bps": deadband_bps,
            "sensitivity": sensitivity,
            "probability_bias": 0.0,
            "training_rows": int(len(train_rows)),
            "validation_rows": int(len(gate_rows)),
            **purge_meta,
            **nested_split_meta,
            "source_rows": int(examples.meta["source_rows"]),
            "sampled_rows": int(examples.meta["sampled_rows"]),
            "training_examples": int(examples.meta["training_examples"]),
            "positive_long_rows": int(examples.meta["positive_long_rows"]),
            "positive_short_rows": int(examples.meta["positive_short_rows"]),
            "neutral_rows": int(examples.meta["neutral_rows"]),
            "long_action_positive_rows": int(examples.meta["long_action_positive_rows"]),
            "short_action_positive_rows": int(examples.meta["short_action_positive_rows"]),
            "long_action_mean_bps": float(examples.meta["long_action_mean_bps"]),
            "short_action_mean_bps": float(examples.meta["short_action_mean_bps"]),
            "optimizer": optimizer_name,
            "long_relevance_cutoffs": (
                list(long_fit["relevance_cutoffs"])
                if long_fit["relevance_cutoffs"] is not None
                else []
            ),
            "short_relevance_cutoffs": (
                list(short_fit["relevance_cutoffs"])
                if short_fit["relevance_cutoffs"] is not None
                else []
            ),
            "lightgbm_version": str(lgb.__version__),
            "best_iteration": int(max(int(long_fit["best_iteration"]), int(short_fit["best_iteration"]))),
            "long_best_iteration": int(long_fit["best_iteration"]),
            "short_best_iteration": int(short_fit["best_iteration"]),
            "seed": int(seed),
            "minimum_gate_auc": float(minimum_auc),
            "long_gate_diagnostics": diagnostic_payload["long_gate_diagnostics"],
            "short_gate_diagnostics": diagnostic_payload["short_gate_diagnostics"],
            "validation_actionable_rows": actionable_count,
            "validation_actionable_rate": float(actionable_count / max(1, len(gate_rows))),
            "validation_actionable_realized_mean_edge_bps": actionable_mean_edge_bps,
            "validation_actionable_hit_rate": actionable_hit_rate,
            "validation_prediction_abs_mean": float(np.mean(np.abs(predicted_best))),
            "training_backend_requested": str(backend.requested),
            "training_backend_kind": "opencl" if use_gpu else "cpu",
            "training_backend_device": f"opencl:{platform_id}:{device_id}" if use_gpu else "cpu",
            "training_backend_reason": "",
            "target": (
                "intraday-session-ranked after-cost utility with calibrated positive-path probability and conditional payoff"
                if training_mode == "daily_utility_rank"
                else "calibrated probability and conditional payoff of positive action-conditioned next-entry lifecycle net paths after fees/spread/slippage/live-buffer"
            ),
            "execution_model_version": EXECUTION_MODEL_VERSION,
            "execution_activity_estimator": EXECUTION_ACTIVITY_ESTIMATOR,
        },
    )

    # Fail closed unless the stdlib evaluator reproduces both serialized
    # classifier margins, calibration maps, and action-value decisions.
    for index in range(min(64, len(gate_rows))):
        actual_probability = model._signed_payoff_lightgbm_ranker_probability(  # noqa: SLF001
            expert,
            gate_rows[index].features,
        )
        long_score = float(long_gate_expected[index])
        short_score = float(short_gate_expected[index])
        if max(long_score, short_score) <= deadband or abs(long_score - short_score) <= 1e-12:
            expected_probability = 0.5
        else:
            best_score = max(long_score, short_score)
            adjusted = (best_score - deadband) / max(1e-9, 1.0 - deadband)
            action_confidence = 1.0 / (1.0 + math.exp(-(adjusted * sensitivity)))
            expected_probability = action_confidence if long_score > short_score else 1.0 - action_confidence
        if actual_probability is None or abs(actual_probability - expected_probability) > 1e-8:
            return fail("parity", "serialized_tree_inference_parity_failed")
    return expert


def _train_signed_payoff_ranker_experts(
    rows: Sequence[ModelRow],
    model: TrainedModel,
    *,
    objective_name: str,
    market_type: str,
    strategy: StrategyConfig,
    compute_backend: str | None,
    batch_size: int,
    symbol_profile: SymbolExecutionProfile | None = None,
    status_callback: Callable[[str, Mapping[str, object]], None] | None = None,
    failure_diagnostics: list[dict[str, object]] | None = None,
) -> tuple[HybridExpert, ...]:
    experts: list[HybridExpert] = []
    horizons = _payoff_horizon_bars(
        rows,
        objective_name,
        max_position_hold_bars=int(getattr(strategy, "max_position_hold_bars", 0) or 0),
    )
    if status_callback is not None:
        status_callback(
            "hybrid_payoff_ranker_started",
            {
                "horizons": list(horizons),
                "training_rows": len(rows),
                "market_type": str(market_type),
            },
        )
    for horizon in horizons:
        def record_failure(payload: Mapping[str, object]) -> None:
            diagnostic = {"horizon_bars": int(horizon), **dict(payload)}
            if failure_diagnostics is not None:
                failure_diagnostics.append(diagnostic)
            if status_callback is not None:
                status_callback("hybrid_payoff_ranker_expert_failed", diagnostic)

        examples = _payoff_training_examples(
            rows,
            model,
            horizon_bars=horizon,
            strategy=strategy,
            objective_name=objective_name,
            market_type=market_type,
            symbol_profile=symbol_profile,
            progress_callback=(
                (
                    lambda payload, _horizon=horizon: status_callback(
                        "hybrid_payoff_target_generation_progress",
                        {"horizon_bars": int(_horizon), **dict(payload)},
                    )
                )
                if status_callback is not None
                else None
            ),
        )
        if examples is None:
            if status_callback is not None:
                status_callback(
                    "hybrid_payoff_ranker_horizon_skipped",
                    {
                        "horizon_bars": int(horizon),
                        "reason": "insufficient_after_cost_positive_examples",
                    },
                )
            continue
        expert = _train_signed_payoff_ranker_expert(
            rows,
            model,
            horizon_bars=horizon,
            objective_name=objective_name,
            market_type=market_type,
            strategy=strategy,
            compute_backend=compute_backend,
            batch_size=batch_size,
            symbol_profile=symbol_profile,
            examples=examples,
        )
        if expert is None:
            record_failure(
                {
                    "expert_kind": "signed_payoff_ranker",
                    "stage": "training",
                    "reason": "trainer_returned_no_artifact",
                }
            )
        mlp_experts: list[HybridExpert] = []
        for seed in (17, 43, 89):
            trained = _train_signed_payoff_mlp_ranker_expert(
                rows,
                model,
                horizon_bars=horizon,
                objective_name=objective_name,
                market_type=market_type,
                strategy=strategy,
                compute_backend=compute_backend,
                batch_size=batch_size,
                symbol_profile=symbol_profile,
                seed=seed,
                examples=examples,
            )
            if trained is None:
                record_failure(
                    {
                        "expert_kind": "signed_payoff_mlp_ranker",
                        "seed": int(seed),
                        "stage": "training",
                        "reason": "trainer_returned_no_artifact",
                    }
                )
            else:
                mlp_experts.append(trained)
        lightgbm_expert = _train_signed_payoff_lightgbm_ranker_expert(
            rows,
            model,
            horizon_bars=horizon,
            objective_name=objective_name,
            market_type=market_type,
            strategy=strategy,
            compute_backend=compute_backend,
            batch_size=batch_size,
            symbol_profile=symbol_profile,
            examples=examples,
            failure_callback=record_failure,
        )
        trained_for_horizon = [item for item in (expert, *mlp_experts, lightgbm_expert) if item is not None]
        if trained_for_horizon:
            experts.extend(trained_for_horizon)
            if status_callback is not None:
                for trained_expert in trained_for_horizon:
                    params = trained_expert.params if isinstance(trained_expert.params, dict) else {}
                    status_callback(
                        "hybrid_payoff_ranker_horizon_complete",
                        {
                            "horizon_bars": int(horizon),
                            "expert_kind": str(trained_expert.kind),
                            "approx_horizon_seconds": float(params.get("approx_horizon_seconds", 0.0) or 0.0),
                            "training_examples": int(params.get("training_examples", 0) or 0),
                            "positive_long_rows": int(params.get("positive_long_rows", 0) or 0),
                            "positive_short_rows": int(params.get("positive_short_rows", 0) or 0),
                            "validation_loss": float(params.get("validation_loss", 0.0) or 0.0),
                            "backend_kind": str(params.get("training_backend_kind", "") or ""),
                        },
                    )
        elif status_callback is not None:
            status_callback(
                "hybrid_payoff_ranker_horizon_skipped",
                {
                    "horizon_bars": int(horizon),
                    "reason": "insufficient_after_cost_positive_examples_or_backend_failure",
                },
            )
    if status_callback is not None:
        status_callback(
            "hybrid_payoff_ranker_complete",
            {
                "trained_experts": len(experts),
                "expert_kinds": sorted({str(expert.kind) for expert in experts}),
                "requested_horizons": len(horizons),
            },
        )
    return tuple(experts)


def optimize_hybrid_model_zoo(
    model: TrainedModel,
    training_rows: Sequence[ModelRow],
    selection_rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    objective_name: str,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
    feature_count: int = 13,
    symbol_profile: SymbolExecutionProfile | None = None,
    status_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> HybridOptimizationReport:
    """Attach the best profitable hybrid expert pack, or keep the base model.

    The selection set is chronological and separate from the final holdout used
    by the training suite.  Hybrid profiles that cannot pass the objective's
    profitability, trade-count, edge, and drawdown gates are rejected.
    """

    base_model = copy.deepcopy(model)
    base_model.model_family = "advanced_logistic"
    base_model.hybrid_base_weight = 1.0
    base_model.hybrid_experts = []
    selection_search_rows = _even_sample(selection_rows, _hybrid_selection_search_limit(objective_name))
    base_search_score, _base_search_result, _base_search_model = _evaluate_model(
        base_model,
        selection_search_rows,
        strategy,
        objective_name=objective_name,
        market_type=market_type,
        starting_cash=starting_cash,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
        symbol_profile=symbol_profile,
    )
    prototypes = _build_prototypes(training_rows, base_model, objective_name)
    bandwidth = _estimate_bandwidth(prototypes, base_model.feature_dim)
    neural_expert = _train_dense_mlp_expert(
        training_rows,
        base_model,
        objective_name=objective_name,
        compute_backend=compute_backend,
        batch_size=score_batch_size,
    )
    payoff_expert_failures: list[dict[str, object]] = []
    payoff_experts = _train_signed_payoff_ranker_experts(
        training_rows,
        base_model,
        objective_name=objective_name,
        market_type=market_type,
        strategy=strategy,
        compute_backend=compute_backend,
        batch_size=score_batch_size,
        symbol_profile=symbol_profile,
        status_callback=status_callback,
        failure_diagnostics=payoff_expert_failures,
    )
    best_model = base_model
    best_search_score = base_search_score
    best_profile = "base_only"
    evaluated = 0
    profile_results: list[HybridProfileResult] = []
    profiles = _profiles_for(objective_name)
    for profile in profiles:
        evaluated += 1
        if status_callback is not None:
            status_callback(
                "hybrid_model_zoo_profile_started",
                {
                    "profile": profile.name,
                    "profile_index": int(evaluated),
                    "profile_count": int(len(profiles)),
                    "payoff_expert_count": int(len(payoff_experts)),
                    "neural_expert_present": bool(neural_expert is not None),
                },
            )
        skip_reason = _skip_profile_for_large_payoff_search(
            profile,
            selection_rows=selection_search_rows,
            payoff_experts=payoff_experts,
            neural_expert=neural_expert,
        )
        if skip_reason:
            skipped_score = -1.0e308
            profile_results.append(
                HybridProfileResult(
                    profile=profile.name,
                    search_score=skipped_score,
                    accepted=False,
                    realized_pnl=None,
                    closed_trades=0,
                    trades_per_day=None,
                    max_drawdown=None,
                    profit_factor=None,
                    expectancy=None,
                    reject_reason=skip_reason,
                    threshold=getattr(base_model, "decision_threshold", None),
                    long_threshold=getattr(base_model, "long_decision_threshold", None),
                    short_threshold=getattr(base_model, "short_decision_threshold", None),
                    expert_kinds=(),
                    selected_search_best=False,
                    promotion_eligible=bool(profile.selection_eligible),
                )
            )
            if status_callback is not None:
                status_callback(
                    "hybrid_model_zoo_profile_complete",
                    {
                        "profile": profile.name,
                        "profile_index": int(evaluated),
                        "profile_count": int(len(profiles)),
                        "score": skipped_score,
                        "best_score": _finite_status_number(float(best_search_score)),
                        "closed_trades": 0,
                        "realized_pnl": None,
                        "reject_reason": skip_reason,
                    },
                )
            continue
        candidate = copy.deepcopy(base_model)
        candidate.model_family = "adaptive_hybrid_model_zoo"
        candidate.probability_inverted = bool(getattr(base_model, "probability_inverted", False)) ^ bool(profile.invert_probability)
        candidate.hybrid_base_weight = max(0.0, float(profile.base))
        candidate.hybrid_experts = _experts_for_profile(
            profile,
            prototypes,
            feature_dim=base_model.feature_dim,
            feature_count=feature_count,
            bandwidth=bandwidth,
            objective_name=objective_name,
            neural_expert=neural_expert,
            payoff_experts=payoff_experts,
        )
        evaluator = _evaluate_model_with_threshold_calibration if profile.payoff > 0.0 else _evaluate_model
        if profile.payoff > 0.0:
            nested_status_callback = None
            if status_callback is not None:
                def nested_status_callback(phase: str, payload: Mapping[str, object]) -> None:
                    status_callback(
                        phase,
                        {
                            "profile": profile.name,
                            "profile_index": int(evaluated),
                            "profile_count": int(len(profiles)),
                            **dict(payload),
                        },
                    )
            score, result, evaluated_candidate = evaluator(
                candidate,
                selection_search_rows,
                strategy,
                objective_name=objective_name,
                market_type=market_type,
                starting_cash=starting_cash,
                compute_backend=compute_backend,
                score_batch_size=score_batch_size,
                symbol_profile=symbol_profile,
                status_callback=nested_status_callback,
            )
        else:
            score, result, evaluated_candidate = evaluator(
                candidate,
                selection_search_rows,
                strategy,
                objective_name=objective_name,
                market_type=market_type,
                starting_cash=starting_cash,
                compute_backend=compute_backend,
                score_batch_size=score_batch_size,
                symbol_profile=symbol_profile,
            )
        selected_search_best = bool(profile.selection_eligible and score > best_search_score + 1e-12)
        profile_results.append(
            _profile_result(
                profile_name=profile.name,
                score=score,
                result=result,
                model=evaluated_candidate,
                objective_name=objective_name,
                selected_search_best=selected_search_best,
                promotion_eligible=bool(profile.selection_eligible),
            )
        )
        if status_callback is not None:
            status_callback(
                "hybrid_model_zoo_profile_complete",
                {
                    "profile": profile.name,
                    "profile_index": int(evaluated),
                    "profile_count": int(len(profiles)),
                    "score": _finite_status_number(float(score)),
                    "best_score": _finite_status_number(float(best_search_score)),
                    "closed_trades": int(result.closed_trades) if result is not None else 0,
                    "realized_pnl": float(result.realized_pnl) if result is not None else None,
                    "reject_reason": (
                        get_objective(objective_name).reject_reason(result)
                        if result is not None and not get_objective(objective_name).accepts(result)
                        else ""
                    ),
                },
            )
        if selected_search_best:
            best_model = evaluated_candidate
            best_search_score = score
            best_profile = profile.name
    base_score, base_result, _base_full_model = _evaluate_model(
        base_model,
        selection_rows,
        strategy,
        objective_name=objective_name,
        market_type=market_type,
        starting_cash=starting_cash,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
        symbol_profile=symbol_profile,
    )
    best_score = base_score
    best_result = base_result
    if best_model.hybrid_experts and math.isfinite(best_search_score):
        full_evaluator = _evaluate_model_with_threshold_calibration if _model_has_payoff_expert(best_model) else _evaluate_model
        full_score, full_result, evaluated_full_model = full_evaluator(
            best_model,
            selection_rows,
            strategy,
            objective_name=objective_name,
            market_type=market_type,
            starting_cash=starting_cash,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
            symbol_profile=symbol_profile,
        )
        if full_score > base_score + 1e-12:
            best_score = full_score
            best_result = full_result
            best_model = evaluated_full_model
        else:
            best_model = base_model
            best_profile = "base_only"
    else:
        best_model = base_model
        best_profile = "base_only"
    accepted = bool(best_model.hybrid_experts and math.isfinite(best_score))
    ablation_results = (
        _hybrid_ablation_results(
            base_model=base_model,
            best_model=best_model,
            base_score=float(base_score),
            best_score=float(best_score),
            rows=selection_rows,
            strategy=strategy,
            objective_name=objective_name,
            market_type=market_type,
            starting_cash=starting_cash,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
            symbol_profile=symbol_profile,
        )
        if accepted
        else ()
    )
    return HybridOptimizationReport(
        accepted=accepted,
        model=best_model,
        base_score=float(base_score),
        best_score=float(best_score),
        best_profile=best_profile,
        evaluated_profiles=evaluated,
        base_result=base_result,
        best_result=best_result,
        ablation_results=ablation_results,
        neural_expert_params=dict(neural_expert.params) if neural_expert is not None else {},
        payoff_expert_params=tuple(
            {**dict(expert.params), "expert_kind": str(expert.kind), "expert_name": str(expert.name)}
            for expert in payoff_experts
            if isinstance(getattr(expert, "params", None), dict)
        ),
        payoff_expert_failures=tuple(payoff_expert_failures),
        profile_results=tuple(profile_results),
        selection_search_rows=len(selection_search_rows),
        selection_full_rows=len(selection_rows),
    )
