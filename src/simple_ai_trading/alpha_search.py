"""Interpretable intraday alpha-template search.

This module adds a Freqtrade/vectorbt-style research layer before ML model
promotion.  It searches a bounded set of original, interpretable day-trading
templates, then validates each candidate through the same execution/risk
backtest used by trained models.  A candidate is promoted only when the target
objective accepts the full selection backtest.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

from .advanced_model import AdvancedFeatureConfig, advanced_feature_group_spans
from .backtest import (
    BacktestResult,
    precompute_backtest_liquidity_adjustments,
    precompute_backtest_regime_scores,
    run_backtest,
)
from .execution_simulation import (
    SymbolExecutionProfile,
    execution_assumptions_for_symbol,
    execution_assumptions_from_strategy,
)
from .features import ModelRow
from .model import (
    HybridExpert,
    TrainedModel,
    _rule_alpha_score_from_values,
    market_direction_from_probability,
)
from .objective import get_objective
from .strategy_overrides import strategy_overrides_from_config
from .types import StrategyConfig


DEFAULT_RULE_ALPHA_MAX_CANDIDATES = 225
DEFAULT_EMPIRICAL_RULE_ALPHA_MAX_CANDIDATES = 18
DEFAULT_EMPIRICAL_RULE_ALPHA_INTERACTION_MAX_CANDIDATES = 18
DEFAULT_RULE_ALPHA_EVENT_RANK_POOL_MULTIPLIER = 3
RULE_ALPHA_LARGE_WINDOW_ROW_COUNT = 20_000
RULE_ALPHA_EXECUTION_ENTRY_OFFSET = 1
_RULE_ALPHA_COST_FLOOR_PARTICIPATION = 0.05
_RULE_ALPHA_MIN_MAJOR_COST_FLOOR_PARTICIPATION = 0.001
_RULE_ALPHA_MIN_PROFILED_COST_FLOOR_PARTICIPATION = 0.005
_RULE_ALPHA_MIN_PROFIT_BUFFER_BPS = 2.0
_RULE_ALPHA_MIN_STOP_BUFFER_BPS = 1.0
_EMPIRICAL_RULE_ALPHA_FAMILY = "empirical_feature_edge"


def _price_move_scale_for_market(strategy: StrategyConfig, market_type: str) -> float:
    del strategy, market_type
    return 1.0


def _median_row_interval_ms(rows: Sequence[ModelRow]) -> int:
    if len(rows) < 2:
        return 1000
    diffs: list[int] = []
    max_checks = min(10_000, len(rows) - 1)
    stride = max(1, (len(rows) - 1) // max_checks)
    for index in range(0, len(rows) - 1, stride):
        delta = int(rows[index + 1].timestamp) - int(rows[index].timestamp)
        if delta > 0:
            diffs.append(delta)
        if len(diffs) >= max_checks:
            break
    if not diffs:
        return 1000
    diffs.sort()
    return max(1, int(diffs[len(diffs) // 2]))


def _seconds_to_bars(seconds: int, bar_interval_ms: int, *, allow_zero: bool = False) -> int:
    value = max(0, int(seconds))
    if allow_zero and value == 0:
        return 0
    return max(1, int(math.ceil(float(value * 1000) / float(max(1, bar_interval_ms)))))


def _rule_alpha_execution_profiles(
    bar_interval_ms: int,
) -> tuple[tuple[str, float, float, float, int, int, int], ...]:
    interval_ms = max(1, int(bar_interval_ms))
    if interval_ms <= 1000:
        raw = (
            ("scalp_3s", 0.06, 0.05, 0.0, 1, 0, 3),
            ("scalp_8s", 0.08, 0.07, 0.0, 2, 0, 8),
            ("scalp_20s", 0.10, 0.09, 0.0, 4, 1, 20),
            ("micro", 0.14, 0.10, 0.0, 2, 0, 2),
            ("balanced", 0.18, 0.14, 0.0, 3, 1, 3),
            ("guarded", 0.22, 0.18, 0.25, 4, 1, 4),
            ("held_30s", 0.18, 0.16, 0.0, 30, 10, 30),
            ("held_90s", 0.24, 0.22, 0.0, 90, 20, 90),
            ("held_180s", 0.30, 0.28, 0.0, 180, 30, 180),
        )
    else:
        raw = (
            ("held_180s", 0.30, 0.28, 0.0, 60, 30, 180),
            ("held_15m", 0.35, 0.32, 0.0, 300, 60, 900),
            ("held_60m", 0.45, 0.42, 0.10, 600, 120, 3600),
            ("held_120m", 0.55, 0.52, 0.20, 900, 180, 7200),
        )
    return tuple(
        profile
        for profile in raw
        if profile[6] * 1000 >= interval_ms
        and (profile[6] * 1000) % interval_ms == 0
        and (profile[6] * 1000) // interval_ms <= 300
    )


@dataclass(frozen=True)
class _EmpiricalCondition:
    feature_index: int
    threshold_value: float
    feature_scale: float
    tail_direction: float
    tail_name: str
    train_stats: dict[str, object]


@dataclass(frozen=True)
class RuleAlphaCandidate:
    name: str
    family: str
    threshold: float
    sensitivity: float
    deadband: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    cooldown_multiplier: float
    min_position_hold_bars: int
    flat_signal_exit_grace_bars: int
    max_position_hold_bars: int = 0
    params: dict[str, object] = field(default_factory=dict)

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RuleAlphaCandidateResult:
    candidate: RuleAlphaCandidate
    probability_inverted: bool
    accepted: bool
    score: float
    raw_score: float
    reject_reason: str | None
    result: BacktestResult
    event_study: dict[str, object] = field(default_factory=dict)

    def asdict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.asdict(),
            "probability_inverted": bool(self.probability_inverted),
            "accepted": bool(self.accepted),
            "score": float(self.score),
            "raw_score": float(self.raw_score),
            "reject_reason": self.reject_reason,
            "result": asdict(self.result),
            "event_study": dict(self.event_study),
        }


@dataclass(frozen=True)
class RuleAlphaOptimizationReport:
    accepted: bool
    model: TrainedModel | None
    best_score: float
    best_candidate: RuleAlphaCandidate | None
    best_probability_inverted: bool
    best_reject_reason: str | None
    evaluated_candidates: int
    best_result: BacktestResult | None
    candidate_results: tuple[RuleAlphaCandidateResult, ...]
    candidate_summary: dict[str, object] = field(default_factory=dict)

    def asdict(self) -> dict[str, object]:
        return {
            "accepted": bool(self.accepted),
            "model": asdict(self.model) if self.model is not None else None,
            "best_score": float(self.best_score),
            "best_candidate": self.best_candidate.asdict() if self.best_candidate is not None else None,
            "best_probability_inverted": bool(self.best_probability_inverted),
            "best_reject_reason": self.best_reject_reason,
            "evaluated_candidates": int(self.evaluated_candidates),
            "best_result": asdict(self.best_result) if self.best_result is not None else None,
            "candidate_results": [item.asdict() for item in self.candidate_results],
            "candidate_summary": dict(self.candidate_summary),
        }


def summarize_rule_alpha_trade_path(result: BacktestResult) -> dict[str, Any]:
    exit_reasons: Counter[str] = Counter()
    side_counts: Counter[str] = Counter()
    bars_held: list[int] = []
    for trade in getattr(result, "trade_log", ()) or ():
        if not isinstance(trade, dict):
            continue
        exit_reasons[str(trade.get("exit_reason") or "unknown")] += 1
        side = int(trade.get("side") or 0)
        side_counts["long" if side > 0 else ("short" if side < 0 else "flat")] += 1
        if isinstance(trade.get("bars_held"), (int, float)):
            bars_held.append(max(0, int(trade["bars_held"])))
    return {
        "win_rate": float(getattr(result, "win_rate", 0.0) or 0.0),
        "profit_factor": float(getattr(result, "profit_factor", 0.0) or 0.0),
        "max_drawdown": float(getattr(result, "max_drawdown", 0.0) or 0.0),
        "average_trade_return": float(getattr(result, "average_trade_return", 0.0) or 0.0),
        "max_consecutive_losses": int(getattr(result, "max_consecutive_losses", 0) or 0),
        "exit_reason_counts": dict(sorted(exit_reasons.items())),
        "side_counts": dict(sorted(side_counts.items())),
        "average_bars_held": (sum(bars_held) / len(bars_held)) if bars_held else 0.0,
    }


def summarize_rule_alpha_candidate_distribution(
    results: Sequence[RuleAlphaCandidateResult],
) -> dict[str, object]:
    """Return compact audit telemetry for the full rule-alpha search surface."""

    result_list = list(results)
    if not result_list:
        return {
            "evaluated_candidates": 0,
            "accepted_candidates": 0,
            "active_candidates": 0,
            "profitable_candidates": 0,
            "event_candidates_with_signals": 0,
            "event_positive_candidates": 0,
            "event_best_candidate": "",
            "event_best_net_edge_bps": 0.0,
            "event_best_signal_count": 0,
            "event_best_hit_rate": 0.0,
            "event_best_horizon_bars": 0,
            "event_best_probability_inverted": False,
            "max_closed_trades": 0,
            "most_active_candidate": "",
            "most_active_pnl": 0.0,
            "most_active_reject_reason": "",
            "best_pnl_candidate": "",
            "best_pnl": 0.0,
            "best_pnl_closed_trades": 0,
            "families_with_trades": "",
            "profiles_with_trades": "",
        }
    active = [item for item in result_list if int(getattr(item.result, "closed_trades", 0) or 0) > 0]
    profitable = [
        item
        for item in result_list
        if int(getattr(item.result, "closed_trades", 0) or 0) > 0
        and float(getattr(item.result, "realized_pnl", 0.0) or 0.0) > 0.0
    ]
    event_active = [
        item
        for item in result_list
        if int(item.event_study.get("signal_count", 0) or 0) > 0
    ]
    event_positive = [
        item
        for item in event_active
        if float(item.event_study.get("net_mean_edge_bps", 0.0) or 0.0) > 0.0
    ]
    best_event = (
        max(
            event_active,
            key=lambda item: (
                float(item.event_study.get("net_mean_edge_bps", float("-inf"))),
                int(item.event_study.get("signal_count", 0) or 0),
                float(item.event_study.get("hit_rate", 0.0) or 0.0),
            ),
        )
        if event_active
        else None
    )
    most_active = max(
        result_list,
        key=lambda item: (
            int(getattr(item.result, "closed_trades", 0) or 0),
            float(getattr(item.result, "realized_pnl", 0.0) or 0.0),
            float(getattr(item.result, "profit_factor", 0.0) or 0.0),
        ),
    )
    best_pnl = max(
        result_list,
        key=lambda item: (
            float(getattr(item.result, "realized_pnl", 0.0) or 0.0),
            int(getattr(item.result, "closed_trades", 0) or 0),
            -float(getattr(item.result, "max_drawdown", 0.0) or 0.0),
        ),
    )
    active_families = sorted({item.candidate.family for item in active})
    active_profiles = sorted({item.candidate.name.split(":")[1] for item in active if ":" in item.candidate.name})
    return {
        "evaluated_candidates": int(len(result_list)),
        "accepted_candidates": int(sum(1 for item in result_list if item.accepted)),
        "active_candidates": int(len(active)),
        "profitable_candidates": int(len(profitable)),
        "event_candidates_with_signals": int(len(event_active)),
        "event_positive_candidates": int(len(event_positive)),
        "event_best_candidate": str(best_event.candidate.name) if best_event is not None else "",
        "event_best_net_edge_bps": float(best_event.event_study.get("net_mean_edge_bps", 0.0) or 0.0) if best_event is not None else 0.0,
        "event_best_signal_count": int(best_event.event_study.get("signal_count", 0) or 0) if best_event is not None else 0,
        "event_best_hit_rate": float(best_event.event_study.get("hit_rate", 0.0) or 0.0) if best_event is not None else 0.0,
        "event_best_horizon_bars": int(best_event.event_study.get("horizon_bars", 0) or 0) if best_event is not None else 0,
        "event_best_probability_inverted": bool(best_event.probability_inverted) if best_event is not None else False,
        "max_closed_trades": int(getattr(most_active.result, "closed_trades", 0) or 0),
        "most_active_candidate": str(most_active.candidate.name),
        "most_active_pnl": float(getattr(most_active.result, "realized_pnl", 0.0) or 0.0),
        "most_active_profit_factor": float(getattr(most_active.result, "profit_factor", 0.0) or 0.0),
        "most_active_reject_reason": str(most_active.reject_reason or ""),
        "best_pnl_candidate": str(best_pnl.candidate.name),
        "best_pnl": float(getattr(best_pnl.result, "realized_pnl", 0.0) or 0.0),
        "best_pnl_closed_trades": int(getattr(best_pnl.result, "closed_trades", 0) or 0),
        "families_with_trades": ",".join(active_families),
        "profiles_with_trades": ",".join(active_profiles),
    }


def rule_alpha_feature_params(feature_cfg: AdvancedFeatureConfig | None) -> dict[str, object]:
    if feature_cfg is None:
        return {}
    params: dict[str, object] = {}
    for span in advanced_feature_group_spans(feature_cfg):
        if span.name == "order_flow_microstructure" and span.size > 0:
            window_count = max(0, len(tuple(feature_cfg.order_flow_windows)))
            params["order_flow_start"] = int(span.start)
            params["order_flow_width"] = int(span.size) // window_count if window_count > 0 else int(span.size)
            params["order_flow_window_count"] = window_count
        elif span.name == "trade_tape_microstructure" and span.size > 0:
            window_count = max(0, len(tuple(feature_cfg.trade_tape_windows)))
            params["trade_tape_start"] = int(span.start)
            params["trade_tape_width"] = int(span.size) // window_count if window_count > 0 else int(span.size)
            params["trade_tape_window_count"] = window_count
        elif span.name == "higher_timeframe_context" and span.size > 0:
            window_count = max(0, len(tuple(feature_cfg.higher_timeframe_windows)))
            params["higher_timeframe_start"] = int(span.start)
            params["higher_timeframe_width"] = int(span.size) // window_count if window_count > 0 else int(span.size)
            params["higher_timeframe_window_count"] = window_count
    return params


def _empirical_feature_scan_indices(
    *,
    feature_dim: int,
    max_feature_count: int | None,
    feature_cfg: AdvancedFeatureConfig | None,
) -> tuple[int, ...]:
    if feature_dim <= 0:
        return ()
    limit = min(feature_dim, 512 if max_feature_count is None else max(1, int(max_feature_count)))
    if feature_cfg is None:
        return tuple(range(limit))
    priority = (
        "trade_tape_microstructure",
        "order_flow_microstructure",
        "market_quality_regime",
        "higher_timeframe_context",
        "technical_confluence",
        "base_features",
    )
    spans = advanced_feature_group_spans(feature_cfg)
    ordered: list[int] = []
    seen: set[int] = set()
    for name in priority:
        for span in spans:
            if span.name != name:
                continue
            for index in range(max(0, int(span.start)), min(feature_dim, int(span.end))):
                if index not in seen:
                    ordered.append(index)
                    seen.add(index)
                    if len(ordered) >= limit:
                        return tuple(ordered)
    for index in range(feature_dim):
        if index not in seen:
            ordered.append(index)
            seen.add(index)
            if len(ordered) >= limit:
                break
    return tuple(ordered)


def rule_alpha_candidates(
    objective_name: str,
    *,
    max_candidates: int | None = None,
    bar_interval_ms: int = 1000,
) -> tuple[RuleAlphaCandidate, ...]:
    """Return a bounded, diversified set of intraday alpha templates."""

    name = "aggressive" if objective_name == "risky" else str(objective_name or "conservative").lower()
    thresholds = {
        "conservative": (0.54, 0.58, 0.62, 0.66, 0.70),
        "regular": (0.52, 0.56, 0.60, 0.64, 0.68),
        "aggressive": (0.50, 0.54, 0.58, 0.62, 0.66),
    }.get(name, (0.54, 0.58, 0.62, 0.66, 0.70))
    sensitivities = {
        "conservative": (6.0, 8.0),
        "regular": (5.0, 7.0, 9.0),
        "aggressive": (4.5, 6.5, 8.5),
    }.get(name, (6.0, 8.0))
    deadbands = (0.02, 0.05, 0.08)
    families = (
        "momentum_breakout",
        "mean_reversion_vwap",
        "trend_pullback",
        "volatility_breakout",
        "volume_flow_proxy",
        "order_flow_momentum",
        "flow_reversion",
        "flow_consensus_breakout",
        "liquidity_absorption_reversal",
        "micro_flow_scalp",
        "vwap_snapback_scalp",
        "liquidity_sweep_reversal",
        "compression_breakout_scalp",
        "volume_synchronized_flow",
        "adaptive_tape_regime",
        "higher_timeframe_alignment",
        "directional_regime_rider",
    )
    interval_ms = max(1, int(bar_interval_ms))
    execution_profiles = _rule_alpha_execution_profiles(interval_ms)
    if not execution_profiles:
        return ()
    limit = max(1, int(max_candidates)) if max_candidates is not None else DEFAULT_RULE_ALPHA_MAX_CANDIDATES
    ranked: list[tuple[tuple[int, int, int, int, int, int], str, str, RuleAlphaCandidate]] = []
    base_by_family_profile: dict[tuple[str, str], RuleAlphaCandidate] = {}
    for threshold_index, threshold in enumerate(thresholds):
        for sensitivity_index, sensitivity in enumerate(sensitivities):
            for deadband_index, deadband in enumerate(deadbands):
                for family_index, family in enumerate(families):
                    for profile_index, (
                        profile,
                        stop_mult,
                        take_mult,
                        cooldown_mult,
                        min_hold_seconds,
                        grace_seconds,
                        max_hold_seconds,
                    ) in enumerate(execution_profiles):
                        min_hold_bars = _seconds_to_bars(min_hold_seconds, interval_ms)
                        grace_bars = _seconds_to_bars(grace_seconds, interval_ms, allow_zero=True)
                        max_hold_bars = _seconds_to_bars(max_hold_seconds, interval_ms)
                        candidate = RuleAlphaCandidate(
                            name=f"{family}:{profile}:t{threshold:.2f}:s{sensitivity:.1f}:d{deadband:.2f}",
                            family=family,
                            threshold=float(threshold),
                            sensitivity=float(sensitivity),
                            deadband=float(deadband),
                            stop_loss_multiplier=float(stop_mult),
                            take_profit_multiplier=float(take_mult),
                            cooldown_multiplier=float(cooldown_mult),
                            min_position_hold_bars=int(min_hold_bars),
                            flat_signal_exit_grace_bars=int(grace_bars),
                            max_position_hold_bars=int(max_hold_bars),
                            params={
                                "duration_contract": "wall_clock_seconds_v1",
                                "bar_interval_ms": int(interval_ms),
                                "intended_min_hold_seconds": int(min_hold_seconds),
                                "intended_grace_seconds": int(grace_seconds),
                                "intended_max_hold_seconds": int(max_hold_seconds),
                                "effective_min_hold_seconds": float(min_hold_bars * interval_ms) / 1000.0,
                                "effective_grace_seconds": float(grace_bars * interval_ms) / 1000.0,
                                "effective_max_hold_seconds": float(max_hold_bars * interval_ms) / 1000.0,
                            },
                        )
                        ranked.append((
                            (
                                max(threshold_index, sensitivity_index, deadband_index, profile_index),
                                threshold_index,
                                family_index,
                                profile_index,
                                sensitivity_index,
                                deadband_index,
                            ),
                            family,
                            profile,
                            candidate,
                        ))
                        if threshold_index == 0 and sensitivity_index == 0 and deadband_index == 0:
                            base_by_family_profile.setdefault((family, profile), candidate)
    ranked.sort(key=lambda item: item[0])
    output: list[RuleAlphaCandidate] = []
    seen: set[str] = set()

    def add(candidate: RuleAlphaCandidate | None) -> None:
        if candidate is None or candidate.name in seen or len(output) >= limit:
            return
        output.append(candidate)
        seen.add(candidate.name)

    profile_names = tuple(profile for profile, *_rest in execution_profiles)
    # First guarantee coverage of every alpha family and every execution profile.
    for family_index, family in enumerate(families):
        add(base_by_family_profile.get((family, profile_names[family_index % len(profile_names)])))
    for profile_index, profile in enumerate(profile_names):
        add(base_by_family_profile.get((families[profile_index % len(families)], profile)))
    # Then cover the full base family/profile matrix before spending slots on nearby parameter variants.
    for profile in profile_names:
        for family in families:
            add(base_by_family_profile.get((family, profile)))
    for _rank, _family, _profile, candidate in ranked:
        add(candidate)
    return tuple(output)


def rule_alpha_event_study(
    rows: Sequence[ModelRow],
    model: TrainedModel | None,
    strategy: StrategyConfig,
    candidate: RuleAlphaCandidate,
    *,
    market_type: str,
    symbol_profile: SymbolExecutionProfile | None = None,
    feature_params: Mapping[str, object] | None = None,
    probability_inverted: bool = False,
) -> dict[str, object]:
    """Measure path-aware directional edge before full trade-lifecycle replay.

    This is telemetry only.  Promotion still depends on `run_backtest` and the
    objective gates, because stop/take/cooldown behavior can differ from a raw
    forward-return event study.  The prefilter still models stop/take paths so
    one future close cannot hide an adverse excursion that would have stopped
    the trade first.
    """

    row_list = list(rows)
    horizon = max(
        1,
        min(300, int(candidate.max_position_hold_bars or candidate.min_position_hold_bars or 1)),
    )
    if len(row_list) <= horizon:
        return {
            "horizon_bars": int(horizon),
            "signal_count": 0,
            "long_signal_count": 0,
            "short_signal_count": 0,
            "mean_edge_bps": 0.0,
            "net_mean_edge_bps": 0.0,
            "hit_rate": 0.0,
            "cost_floor_bps": float(rule_alpha_take_profit_floor_pct(strategy, symbol_profile=symbol_profile) * 10_000.0),
            "path_take_count": 0,
            "path_stop_count": 0,
            "path_timeout_count": 0,
            "path_average_bars_held": 0.0,
        }
    if model is None:
        long_threshold = float(candidate.threshold)
        short_threshold = float(1.0 - candidate.threshold) if market_type == "futures" else None
        direct_params = _rule_alpha_direct_params(candidate, feature_params)
    else:
        long_threshold = (
            float(model.long_decision_threshold)
            if model.long_decision_threshold is not None
            else float(model.decision_threshold if model.decision_threshold is not None else candidate.threshold)
        )
        short_threshold = (
            float(model.short_decision_threshold)
            if market_type == "futures" and model.short_decision_threshold is not None
            else None
        )
        direct_params = None
    cost_floor_bps = float(rule_alpha_take_profit_floor_pct(strategy, symbol_profile=symbol_profile) * 10_000.0)
    edges: list[float] = []
    bars_held_values: list[int] = []
    exit_reasons: Counter[str] = Counter()
    long_count = 0
    short_count = 0
    cooldown_ms = max(0, int(round(float(getattr(strategy, "cooldown_minutes", 0) or 0.0) * 60_000.0)))
    index = 0
    while index < len(row_list) - horizon:
        current = row_list[index]
        if current.close <= 0.0:
            index += 1
            continue
        if direct_params is not None:
            probability = _rule_alpha_direct_probability(
                current.features,
                direct_params,
                probability_inverted=probability_inverted,
            )
        else:
            probability = float(model.predict_proba(current.features)) if model is not None else 0.5
        side = market_direction_from_probability(
            probability,
            long_threshold,
            market_type=market_type,
            short_threshold=short_threshold,
        )
        if side > 0:
            long_count += 1
        elif side < 0:
            short_count += 1
        if side == 0:
            index += 1
            continue
        edge_bps, bars_held, exit_reason = _rule_alpha_path_edge_bps(
            row_list,
            index,
            horizon,
            side,
            strategy,
        )
        if math.isfinite(edge_bps):
            edges.append(edge_bps)
            bars_held_values.append(int(bars_held))
            exit_reasons[str(exit_reason)] += 1
        next_index = index + max(1, int(bars_held))
        if cooldown_ms > 0:
            exit_timestamp = int(row_list[min(len(row_list) - 1, index + max(1, int(bars_held)))].timestamp)
            resume_timestamp = exit_timestamp + cooldown_ms
            while next_index < len(row_list) and int(row_list[next_index].timestamp) < resume_timestamp:
                next_index += 1
        index = max(index + 1, next_index)
    if not edges:
        return {
            "horizon_bars": int(horizon),
            "signal_count": 0,
            "long_signal_count": 0,
            "short_signal_count": 0,
            "mean_edge_bps": 0.0,
            "net_mean_edge_bps": -cost_floor_bps,
            "hit_rate": 0.0,
            "cost_floor_bps": cost_floor_bps,
            "path_take_count": 0,
            "path_stop_count": 0,
            "path_timeout_count": 0,
            "path_average_bars_held": 0.0,
        }
    mean_edge = sum(edges) / len(edges)
    wins = sum(1 for value in edges if value > cost_floor_bps)
    return {
        "horizon_bars": int(horizon),
        "signal_count": int(len(edges)),
        "long_signal_count": int(long_count),
        "short_signal_count": int(short_count),
        "mean_edge_bps": float(mean_edge),
        "net_mean_edge_bps": float(mean_edge - cost_floor_bps),
        "hit_rate": float(wins / len(edges)),
        "cost_floor_bps": cost_floor_bps,
        "path_take_count": int(exit_reasons["take"]),
        "path_stop_count": int(exit_reasons["stop"]),
        "path_timeout_count": int(exit_reasons["timeout"]),
        "path_average_bars_held": float(sum(bars_held_values) / len(bars_held_values)) if bars_held_values else 0.0,
    }


def _bounded_param_float(params: Mapping[str, object], key: str, default: float, *, low: float, high: float) -> float:
    try:
        value = float(params.get(key, default))
    except (TypeError, ValueError, OverflowError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return _clamp(value, low, high)


def _rule_alpha_direct_params(
    candidate: RuleAlphaCandidate,
    feature_params: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "family": candidate.family,
        "sensitivity": candidate.sensitivity,
        "deadband": candidate.deadband,
        **dict(feature_params or {}),
        **dict(candidate.params or {}),
    }


def _rule_alpha_direct_probability(
    features: Sequence[float],
    params: Mapping[str, object],
    *,
    probability_inverted: bool,
) -> float:
    if not features:
        probability = 0.5
    else:
        values = list(features)
        while len(values) < 13:
            values.append(0.0)
        score = _rule_alpha_score_from_values(values, dict(params))
        sensitivity = _bounded_param_float(params, "sensitivity", 7.0, low=0.1, high=30.0)
        bias = _bounded_param_float(params, "bias", 0.0, low=-5.0, high=5.0)
        probability = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, score * sensitivity + bias))))
        probability = _clamp(probability, 0.0, 1.0)
    return float(_clamp(1.0 - probability if probability_inverted else probability, 0.0, 1.0))


def _rule_alpha_path_edge_bps(
    rows: Sequence[ModelRow],
    start_index: int,
    horizon: int,
    side: int,
    strategy: StrategyConfig,
    *,
    entry_offset: int = RULE_ALPHA_EXECUTION_ENTRY_OFFSET,
) -> tuple[float, int, str]:
    """Return gross path edge using stop-first intrabar semantics.

    The full backtester remains authoritative.  This helper exists only for
    ranking candidate alpha templates cheaply and avoids accepting a signal
    whose future close looks good after a stop would already have fired.
    """

    row_count = len(rows)
    if row_count <= 0:
        return 0.0, 0, "timeout"
    signal_start = max(0, min(row_count - 1, int(start_index)))
    offset = max(0, int(entry_offset))
    start = signal_start + offset
    if start >= row_count:
        return 0.0, offset, "timeout"
    current = rows[start]
    entry = float(current.close)
    if entry <= 0.0 or side == 0:
        return 0.0, offset, "timeout"
    stop_pct = max(0.0, float(getattr(strategy, "stop_loss_pct", 0.0) or 0.0))
    take_pct = max(0.0, float(getattr(strategy, "take_profit_pct", 0.0) or 0.0))
    edge_bps, bars_held, exit_reason = _rule_alpha_path_edge_bps_with_barriers(
        rows,
        start,
        horizon,
        side,
        stop_loss_pct=stop_pct,
        take_profit_pct=take_pct,
    )
    return edge_bps, offset + int(bars_held), exit_reason


def _rule_alpha_path_edge_bps_with_barriers(
    rows: Sequence[ModelRow],
    start_index: int,
    horizon: int,
    side: int,
    *,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> tuple[float, int, str]:
    row_count = len(rows)
    if row_count <= 0:
        return 0.0, 0, "timeout"
    start = max(0, min(row_count - 1, int(start_index)))
    if start >= row_count - 1:
        return 0.0, 0, "timeout"
    current = rows[start]
    entry = float(current.close)
    if entry <= 0.0 or side == 0:
        return 0.0, 0, "timeout"
    stop_pct = max(0.0, float(stop_loss_pct))
    take_pct = max(0.0, float(take_profit_pct))
    end = max(start + 1, min(row_count - 1, start + max(1, int(horizon))))
    for index in range(start + 1, end + 1):
        row = rows[index]
        close = max(0.0, float(row.close))
        high = _coerce_row_high(row, close)
        low = _coerce_row_low(row, close)
        if side > 0:
            stop_price = entry * (1.0 - stop_pct)
            take_price = entry * (1.0 + take_pct)
            if stop_pct > 0.0 and low <= stop_price:
                return ((stop_price - entry) / entry) * 10_000.0, index - start, "stop"
            if take_pct > 0.0 and high >= take_price:
                return ((take_price - entry) / entry) * 10_000.0, index - start, "take"
        else:
            stop_price = entry * (1.0 + stop_pct)
            take_price = max(0.0, entry * (1.0 - take_pct))
            if stop_pct > 0.0 and high >= stop_price:
                return ((entry - stop_price) / entry) * 10_000.0, index - start, "stop"
            if take_pct > 0.0 and low <= take_price:
                return ((entry - take_price) / entry) * 10_000.0, index - start, "take"
    final = max(0.0, float(rows[end].close))
    if final <= 0.0:
        return 0.0, end - start, "timeout"
    edge = float(side) * ((final - entry) / entry) * 10_000.0
    return edge, end - start, "timeout"


def _coerce_row_high(row: ModelRow, close: float) -> float:
    try:
        high = float(row.high if row.high is not None else close)
    except (TypeError, ValueError, OverflowError):
        high = close
    return max(close, high) if math.isfinite(high) else close


def _coerce_row_low(row: ModelRow, close: float) -> float:
    try:
        low = float(row.low if row.low is not None else close)
    except (TypeError, ValueError, OverflowError):
        low = close
    return min(close, low) if math.isfinite(low) else close


def _event_rank_slices(rows: Sequence[ModelRow]) -> tuple[list[ModelRow], list[ModelRow]]:
    row_list = list(rows)
    if len(row_list) < 240:
        return row_list, row_list
    split = max(120, min(len(row_list) - 80, int(len(row_list) * 0.60)))
    training = row_list[:split]
    validation = row_list[split:]
    if len(training) < 120 or len(validation) < 80:
        return row_list, row_list
    return training, validation


def _event_rank_rule_alpha_candidates(
    rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    candidates: Sequence[RuleAlphaCandidate],
    *,
    market_type: str,
    replay_limit: int,
    feature_params: dict[str, object],
    symbol_profile: SymbolExecutionProfile | None = None,
    status_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> tuple[tuple[RuleAlphaCandidate, ...], dict[str, object]]:
    """Rank a larger static template pool by cheap after-cost event evidence.

    This ranking decides which candidates deserve the bounded full lifecycle
    replay.  It does not approve any candidate; only `run_backtest` and the
    objective gates can do that.
    """

    limit = max(1, int(replay_limit))
    row_list = list(rows)
    training_rows, validation_rows = _event_rank_slices(row_list)
    split_mode = "full_sample" if len(training_rows) == len(row_list) and len(validation_rows) == len(row_list) else "chronological"
    scored: list[
        tuple[
            tuple[float, ...],
            RuleAlphaCandidate,
            dict[str, object],
            dict[str, object],
            bool,
        ]
    ] = []
    candidate_count = len(candidates)
    best_progress_key: tuple[float, ...] | None = None
    best_progress_candidate = ""
    best_progress_edge = 0.0
    density_floor = max(
        3,
        min(30, int(max(1, len(validation_rows)) * 0.001)),
    )
    for index, candidate in enumerate(candidates):
        if status_callback is not None:
            status_callback(
                "rule_alpha_event_rank_candidate_started",
                {
                    "candidate_index": index + 1,
                    "candidate_count": candidate_count,
                    "candidate": candidate.name,
                    "candidate_family": candidate.family,
                    "selected_limit": limit,
                    "training_rows": len(training_rows),
                    "validation_rows": len(validation_rows),
                },
            )
        candidate_strategy = _candidate_strategy(
            strategy,
            candidate,
            market_type=market_type,
            symbol_profile=symbol_profile,
        )
        best_training_event: dict[str, object] | None = None
        best_validation_event: dict[str, object] | None = None
        best_inverted = False
        best_key: tuple[float, ...] | None = None
        for inverted in (False, True):
            training_event = rule_alpha_event_study(
                training_rows,
                None,
                candidate_strategy,
                candidate,
                market_type=market_type,
                symbol_profile=symbol_profile,
                feature_params=feature_params,
                probability_inverted=inverted,
            )
            validation_event = rule_alpha_event_study(
                validation_rows,
                None,
                candidate_strategy,
                candidate,
                market_type=market_type,
                symbol_profile=symbol_profile,
                feature_params=feature_params,
                probability_inverted=inverted,
            )
            training_signal_count = int(training_event.get("signal_count", 0) or 0)
            validation_signal_count = int(validation_event.get("signal_count", 0) or 0)
            training_net_edge = float(training_event.get("net_mean_edge_bps", 0.0) or 0.0)
            validation_net_edge = float(validation_event.get("net_mean_edge_bps", 0.0) or 0.0)
            training_hit_rate = float(training_event.get("hit_rate", 0.0) or 0.0)
            validation_hit_rate = float(validation_event.get("hit_rate", 0.0) or 0.0)
            min_signal_count = min(training_signal_count, validation_signal_count)
            min_net_edge = min(training_net_edge, validation_net_edge)
            min_hit_rate = min(training_hit_rate, validation_hit_rate)
            repeatable_edge_score = (
                min_net_edge * math.log1p(float(min_signal_count))
                if min_net_edge > 0.0 and min_signal_count > 0
                else min_net_edge
            )
            key = (
                1.0 if min_signal_count > 0 else 0.0,
                1.0 if min_net_edge > 0.0 else 0.0,
                1.0 if min_signal_count >= density_floor else 0.0,
                repeatable_edge_score,
                min_net_edge,
                validation_net_edge,
                training_net_edge,
                min_hit_rate,
                float(min_signal_count),
                -float(index),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_training_event = training_event
                best_validation_event = validation_event
                best_inverted = bool(inverted)
        if best_key is not None and best_training_event is not None and best_validation_event is not None:
            scored.append((best_key, candidate, best_training_event, best_validation_event, best_inverted))
            candidate_min_edge = min(
                float(best_training_event.get("net_mean_edge_bps", 0.0) or 0.0),
                float(best_validation_event.get("net_mean_edge_bps", 0.0) or 0.0),
            )
            if best_progress_key is None or best_key > best_progress_key:
                best_progress_key = best_key
                best_progress_candidate = candidate.name
                best_progress_edge = candidate_min_edge
        if status_callback is not None:
            status_callback(
                "rule_alpha_event_rank_candidate_scored",
                {
                    "candidate_index": index + 1,
                    "candidate_count": candidate_count,
                    "candidate": candidate.name,
                    "candidate_family": candidate.family,
                    "scored_candidates": len(scored),
                    "selected_limit": limit,
                    "density_floor": int(density_floor),
                    "best_candidate": best_progress_candidate,
                    "best_net_edge_bps": float(best_progress_edge),
                },
            )
    scored.sort(key=lambda item: item[0], reverse=True)
    positive_scored = [
        item
        for item in scored
        if min(
            float(item[2].get("net_mean_edge_bps", 0.0) or 0.0),
            float(item[3].get("net_mean_edge_bps", 0.0) or 0.0),
        ) > 0.0
    ]
    replay_source = positive_scored if positive_scored else []
    selected = tuple(item[1] for item in replay_source[:limit])
    best = scored[0] if scored else None
    summary = {
        "event_rank_split_mode": split_mode,
        "event_rank_training_rows": int(len(training_rows)),
        "event_rank_validation_rows": int(len(validation_rows)),
        "event_rank_density_floor": int(density_floor),
        "event_rank_pool_candidates": int(len(scored)),
        "event_rank_selected_template_candidates": int(len(selected)),
        "event_rank_positive_pool_candidates": int(
            sum(
                1
                for _key, _candidate, training_event, validation_event, _inverted in scored
                if min(
                    float(training_event.get("net_mean_edge_bps", 0.0) or 0.0),
                    float(validation_event.get("net_mean_edge_bps", 0.0) or 0.0),
                ) > 0.0
            )
        ),
        "event_rank_signal_pool_candidates": int(
            sum(
                1
                for _key, _candidate, training_event, validation_event, _inverted in scored
                if min(
                    int(training_event.get("signal_count", 0) or 0),
                    int(validation_event.get("signal_count", 0) or 0),
                ) > 0
            )
        ),
        "event_rank_best_candidate": str(best[1].name) if best is not None else "",
        "event_rank_best_net_edge_bps": (
            min(
                float(best[2].get("net_mean_edge_bps", 0.0) or 0.0),
                float(best[3].get("net_mean_edge_bps", 0.0) or 0.0),
            ) if best is not None else 0.0
        ),
        "event_rank_best_training_net_edge_bps": (
            float(best[2].get("net_mean_edge_bps", 0.0) or 0.0) if best is not None else 0.0
        ),
        "event_rank_best_validation_net_edge_bps": (
            float(best[3].get("net_mean_edge_bps", 0.0) or 0.0) if best is not None else 0.0
        ),
        "event_rank_best_signal_count": (
            min(
                int(best[2].get("signal_count", 0) or 0),
                int(best[3].get("signal_count", 0) or 0),
            ) if best is not None else 0
        ),
        "event_rank_best_training_signal_count": (
            int(best[2].get("signal_count", 0) or 0) if best is not None else 0
        ),
        "event_rank_best_validation_signal_count": (
            int(best[3].get("signal_count", 0) or 0) if best is not None else 0
        ),
        "event_rank_best_hit_rate": (
            min(
                float(best[2].get("hit_rate", 0.0) or 0.0),
                float(best[3].get("hit_rate", 0.0) or 0.0),
            ) if best is not None else 0.0
        ),
        "event_rank_best_training_hit_rate": (
            float(best[2].get("hit_rate", 0.0) or 0.0) if best is not None else 0.0
        ),
        "event_rank_best_validation_hit_rate": (
            float(best[3].get("hit_rate", 0.0) or 0.0) if best is not None else 0.0
        ),
        "event_rank_best_probability_inverted": bool(best[4]) if best is not None else False,
    }
    return selected, summary


def mine_empirical_rule_alpha_candidates(
    rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    objective_name: str,
    market_type: str,
    max_candidates: int = DEFAULT_EMPIRICAL_RULE_ALPHA_MAX_CANDIDATES,
    max_feature_count: int | None = None,
    feature_cfg: AdvancedFeatureConfig | None = None,
    symbol_profile: SymbolExecutionProfile | None = None,
    bar_interval_ms: int | None = None,
    status_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> tuple[RuleAlphaCandidate, ...]:
    """Mine simple one-feature edge rules with chronological validation.

    This is a feature-screening layer, not a promotion shortcut.  A mined rule
    must show positive net forward edge after the modeled cost floor on both the
    earlier mining slice and later validation slice before it is even allowed
    into the normal rule-alpha replay/objective gates.
    """

    row_list = list(rows)
    limit = max(0, int(max_candidates))
    if limit <= 0 or len(row_list) < 240:
        return ()
    feature_dim = len(row_list[0].features)
    feature_indices = _empirical_feature_scan_indices(
        feature_dim=feature_dim,
        max_feature_count=max_feature_count,
        feature_cfg=feature_cfg,
    )
    feature_count = len(feature_indices)
    if feature_count <= 0:
        return ()
    split = max(120, min(len(row_list) - 80, int(len(row_list) * 0.60)))
    training = row_list[:split]
    validation = row_list[split:]
    if len(training) < 120 or len(validation) < 80:
        return ()

    name = "aggressive" if objective_name == "risky" else str(objective_name or "conservative").lower()
    threshold_by_risk = {
        "conservative": 0.56,
        "regular": 0.54,
        "aggressive": 0.52,
    }
    probability_threshold = float(threshold_by_risk.get(name, 0.56))
    interval_ms = max(1, int(bar_interval_ms or _median_row_interval_ms(row_list)))
    raw_profiles = (
        (
            ("empirical_8s", 0.10, 0.09, 8, 2),
            ("empirical_30s", 0.18, 0.16, 30, 8),
            ("empirical_90s", 0.26, 0.24, 90, 20),
        )
        if interval_ms <= 1000
        else (
            ("empirical_15m", 0.35, 0.32, 900, 60),
            ("empirical_60m", 0.45, 0.42, 3600, 120),
            ("empirical_120m", 0.55, 0.52, 7200, 180),
        )
    )
    profiles = tuple(
        (
            profile,
            stop_mult,
            take_mult,
            _seconds_to_bars(horizon_seconds, interval_ms),
            _seconds_to_bars(grace_seconds, interval_ms, allow_zero=True),
            int(horizon_seconds),
            int(grace_seconds),
        )
        for profile, stop_mult, take_mult, horizon_seconds, grace_seconds in raw_profiles
        if horizon_seconds * 1000 >= interval_ms
        and (horizon_seconds * 1000) % interval_ms == 0
        and (horizon_seconds * 1000) // interval_ms <= 300
    )
    quantiles = (
        (0.08, -1.0, "low08"),
        (0.12, -1.0, "low12"),
        (0.20, -1.0, "low20"),
        (0.80, 1.0, "high80"),
        (0.88, 1.0, "high88"),
        (0.92, 1.0, "high92"),
    )
    cost_floor_bps = float(rule_alpha_take_profit_floor_pct(strategy, symbol_profile=symbol_profile) * 10_000.0)
    candidates: list[tuple[tuple[float, float, int, float], RuleAlphaCandidate]] = []
    condition_pools: dict[tuple[str, int, int, float, float, float], list[_EmpiricalCondition]] = {}
    progress_stride = max(1, feature_count // 16)

    for scan_index, feature_index in enumerate(feature_indices):
        if status_callback is not None and (
            scan_index == 0
            or (scan_index + 1) % progress_stride == 0
            or scan_index + 1 == feature_count
        ):
            status_callback(
                "rule_alpha_empirical_feature_scan_progress",
                {
                    "feature_index": int(scan_index + 1),
                    "feature_column_index": int(feature_index),
                    "feature_scan_index": int(scan_index + 1),
                    "feature_count": int(feature_count),
                    "feature_dim": int(feature_dim),
                    "candidate_count": int(len(candidates)),
                    "condition_pool_count": int(sum(len(pool) for pool in condition_pools.values())),
                },
            )
        feature_values = [
            float(row.features[feature_index])
            for row in training
            if len(row.features) > feature_index and math.isfinite(float(row.features[feature_index]))
        ]
        if len(feature_values) < 120:
            continue
        feature_values.sort()
        distinct_values = sorted(set(feature_values))
        q10 = _quantile(feature_values, 0.10)
        q90 = _quantile(feature_values, 0.90)
        feature_scale = max(abs(q90 - q10), 1e-9)
        if not math.isfinite(feature_scale) or feature_scale <= 1e-9:
            continue
        for (
            profile,
            stop_mult,
            take_mult,
            horizon,
            grace_bars,
            horizon_seconds,
            grace_seconds,
        ) in profiles:
            if len(training) <= horizon or len(validation) <= horizon:
                continue
            stop_loss_pct, take_profit_pct = _rule_alpha_profile_barriers(
                strategy,
                market_type=market_type,
                stop_loss_multiplier=float(stop_mult),
                take_profit_multiplier=float(take_mult),
                symbol_profile=symbol_profile,
            )
            min_train_signals = max(40, int((len(training) - horizon) * 0.003))
            min_validation_signals = max(25, int((len(validation) - horizon) * 0.003))
            for quantile, tail_direction, tail_name in quantiles:
                threshold_value = _quantile(feature_values, quantile)
                if not math.isfinite(threshold_value):
                    continue
                if len(distinct_values) > 1:
                    if tail_direction > 0.0 and threshold_value >= distinct_values[-1]:
                        threshold_value = (distinct_values[-1] + distinct_values[-2]) / 2.0
                    elif tail_direction < 0.0 and threshold_value <= distinct_values[0]:
                        threshold_value = (distinct_values[0] + distinct_values[1]) / 2.0
                for trade_side in ((1.0, -1.0) if market_type == "futures" else (1.0,)):
                    train_stats = _empirical_edge_stats(
                        training,
                        feature_index=feature_index,
                        threshold_value=threshold_value,
                        feature_scale=feature_scale,
                        tail_direction=tail_direction,
                        trade_side=trade_side,
                        horizon=horizon,
                        cost_floor_bps=cost_floor_bps,
                    )
                    if (
                        int(train_stats["signal_count"]) >= min_train_signals
                        and float(train_stats["net_mean_edge_bps"]) > -max(2.0, cost_floor_bps * 0.25)
                        and float(train_stats["hit_rate"]) >= 0.45
                    ):
                        train_stats = _empirical_edge_stats(
                            training,
                            feature_index=feature_index,
                            threshold_value=threshold_value,
                            feature_scale=feature_scale,
                            tail_direction=tail_direction,
                            trade_side=trade_side,
                            horizon=horizon,
                            cost_floor_bps=cost_floor_bps,
                            stop_loss_pct=stop_loss_pct,
                            take_profit_pct=take_profit_pct,
                        )
                    if (
                        int(train_stats["signal_count"]) >= min_train_signals
                        and float(train_stats["net_mean_edge_bps"]) > -max(2.0, cost_floor_bps * 0.25)
                        and float(train_stats["hit_rate"]) >= 0.45
                    ):
                        key = (
                            profile,
                            int(horizon),
                            int(grace_bars),
                            float(trade_side),
                            float(stop_loss_pct),
                            float(take_profit_pct),
                        )
                        condition_pools.setdefault(key, []).append(_EmpiricalCondition(
                            feature_index=int(feature_index),
                            threshold_value=float(threshold_value),
                            feature_scale=float(feature_scale),
                            tail_direction=float(tail_direction),
                            tail_name=str(tail_name),
                            train_stats=dict(train_stats),
                        ))
                    if (
                        int(train_stats["signal_count"]) < min_train_signals
                        or float(train_stats["net_mean_edge_bps"]) <= 0.0
                        or float(train_stats["hit_rate"]) < 0.52
                    ):
                        continue
                    validation_stats = _empirical_edge_stats(
                        validation,
                        feature_index=feature_index,
                        threshold_value=threshold_value,
                        feature_scale=feature_scale,
                        tail_direction=tail_direction,
                        trade_side=trade_side,
                        horizon=horizon,
                        cost_floor_bps=cost_floor_bps,
                    )
                    if (
                        int(validation_stats["signal_count"]) < min_validation_signals
                        or float(validation_stats["net_mean_edge_bps"]) <= 0.0
                        or float(validation_stats["hit_rate"]) < 0.52
                    ):
                        continue
                    validation_stats = _empirical_edge_stats(
                        validation,
                        feature_index=feature_index,
                        threshold_value=threshold_value,
                        feature_scale=feature_scale,
                        tail_direction=tail_direction,
                        trade_side=trade_side,
                        horizon=horizon,
                        cost_floor_bps=cost_floor_bps,
                        stop_loss_pct=stop_loss_pct,
                        take_profit_pct=take_profit_pct,
                    )
                    if (
                        int(validation_stats["signal_count"]) < min_validation_signals
                        or float(validation_stats["net_mean_edge_bps"]) <= 0.0
                        or float(validation_stats["hit_rate"]) < 0.52
                    ):
                        continue
                    trade_name = "long" if trade_side > 0.0 else "short"
                    confidence = _clamp(
                        min(float(train_stats["net_mean_edge_bps"]), float(validation_stats["net_mean_edge_bps"])) / max(cost_floor_bps, 1.0),
                        0.15,
                        1.0,
                    )
                    candidate = RuleAlphaCandidate(
                        name=f"{_EMPIRICAL_RULE_ALPHA_FAMILY}:{profile}:f{feature_index}:{tail_name}:{trade_name}",
                        family=_EMPIRICAL_RULE_ALPHA_FAMILY,
                        threshold=probability_threshold,
                        sensitivity=8.0,
                        deadband=0.0,
                        stop_loss_multiplier=float(stop_mult),
                        take_profit_multiplier=float(take_mult),
                        cooldown_multiplier=0.0,
                        min_position_hold_bars=int(horizon),
                        flat_signal_exit_grace_bars=int(grace_bars),
                        max_position_hold_bars=int(horizon),
                        params={
                            "feature_index": int(feature_index),
                            "feature_threshold": float(threshold_value),
                            "feature_scale": float(feature_scale),
                            "tail_direction": float(tail_direction),
                            "trade_side": float(trade_side),
                            "edge_confidence": float(confidence),
                            "edge_slope": 1.0,
                            "training_signal_count": int(train_stats["signal_count"]),
                            "validation_signal_count": int(validation_stats["signal_count"]),
                            "training_net_edge_bps": float(train_stats["net_mean_edge_bps"]),
                            "validation_net_edge_bps": float(validation_stats["net_mean_edge_bps"]),
                            "training_hit_rate": float(train_stats["hit_rate"]),
                            "validation_hit_rate": float(validation_stats["hit_rate"]),
                            "horizon_bars": int(horizon),
                            "duration_contract": "wall_clock_seconds_v1",
                            "bar_interval_ms": int(interval_ms),
                            "intended_max_hold_seconds": int(horizon_seconds),
                            "intended_grace_seconds": int(grace_seconds),
                            "effective_max_hold_seconds": float(horizon * interval_ms) / 1000.0,
                            "effective_grace_seconds": float(grace_bars * interval_ms) / 1000.0,
                        },
                    )
                    score = (
                        min(float(train_stats["net_mean_edge_bps"]), float(validation_stats["net_mean_edge_bps"])),
                        min(float(train_stats["hit_rate"]), float(validation_stats["hit_rate"])),
                        min(int(train_stats["signal_count"]), int(validation_stats["signal_count"])),
                        -float(feature_index),
                    )
                    candidates.append((score, candidate))

    _mine_empirical_interaction_candidates(
        candidates,
        condition_pools,
        training=training,
        validation=validation,
        cost_floor_bps=cost_floor_bps,
        probability_threshold=probability_threshold,
        market_type=market_type,
        bar_interval_ms=interval_ms,
        status_callback=status_callback,
    )

    candidates.sort(key=lambda item: item[0], reverse=True)
    output: list[RuleAlphaCandidate] = []
    seen: set[tuple[int, float, float, int, int, float]] = set()

    def add(candidate: RuleAlphaCandidate) -> None:
        key = (
            int(candidate.params.get("feature_index", -1)),
            float(candidate.params.get("tail_direction", 0.0)),
            float(candidate.params.get("trade_side", 0.0)),
            int(candidate.params.get("horizon_bars", 0)),
            int(candidate.params.get("second_feature_index", -1)),
            float(candidate.params.get("second_tail_direction", 0.0)),
        )
        if key in seen or len(output) >= limit:
            return
        seen.add(key)
        output.append(candidate)

    best_single = next(
        (candidate for _score, candidate in candidates if "second_feature_index" not in candidate.params),
        None,
    )
    best_interaction = next(
        (candidate for _score, candidate in candidates if "second_feature_index" in candidate.params),
        None,
    )
    if best_single is not None:
        add(best_single)
    if best_interaction is not None and limit > 1:
        add(best_interaction)
    for _score, candidate in candidates:
        add(candidate)
    return tuple(output)


def _mine_empirical_interaction_candidates(
    candidates: list[tuple[tuple[float, float, int, float], RuleAlphaCandidate]],
    condition_pools: Mapping[tuple[str, int, int, float, float, float], list[_EmpiricalCondition]],
    *,
    training: Sequence[ModelRow],
    validation: Sequence[ModelRow],
    cost_floor_bps: float,
    probability_threshold: float,
    market_type: str,
    bar_interval_ms: int,
    status_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> None:
    """Append validated two-condition empirical rules to ``candidates``."""

    del market_type  # side handling is already encoded in the condition-pool key.
    emitted = 0
    pool_items = list(condition_pools.items())
    pool_count = len(pool_items)
    for pool_index, ((profile, horizon, grace_bars, trade_side, stop_loss_pct, take_profit_pct), pool) in enumerate(pool_items, start=1):
        if status_callback is not None:
            status_callback(
                "rule_alpha_empirical_interaction_scan_progress",
                {
                    "pool_index": int(pool_index),
                    "pool_count": int(pool_count),
                    "pool_size": int(len(pool)),
                    "emitted_interactions": int(emitted),
                    "candidate_count": int(len(candidates)),
                },
            )
        if emitted >= DEFAULT_EMPIRICAL_RULE_ALPHA_INTERACTION_MAX_CANDIDATES:
            break
        if len(pool) < 2:
            continue
        pool.sort(
            key=lambda item: (
                float(item.train_stats.get("net_mean_edge_bps", 0.0) or 0.0),
                float(item.train_stats.get("hit_rate", 0.0) or 0.0),
                int(item.train_stats.get("signal_count", 0) or 0),
                -int(item.feature_index),
            ),
            reverse=True,
        )
        pool = pool[:36]
        min_train_signals = max(35, int((len(training) - horizon) * 0.002))
        min_validation_signals = max(20, int((len(validation) - horizon) * 0.002))
        for left_index, left in enumerate(pool):
            for right in pool[left_index + 1:]:
                if left.feature_index == right.feature_index:
                    continue
                train_stats = _empirical_edge_stats(
                    training,
                    feature_index=left.feature_index,
                    threshold_value=left.threshold_value,
                    feature_scale=left.feature_scale,
                    tail_direction=left.tail_direction,
                    trade_side=trade_side,
                    horizon=horizon,
                    cost_floor_bps=cost_floor_bps,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    second_feature_index=right.feature_index,
                    second_threshold_value=right.threshold_value,
                    second_feature_scale=right.feature_scale,
                    second_tail_direction=right.tail_direction,
                )
                if (
                    int(train_stats["signal_count"]) < min_train_signals
                    or float(train_stats["net_mean_edge_bps"]) <= 0.0
                    or float(train_stats["hit_rate"]) < 0.53
                ):
                    continue
                validation_stats = _empirical_edge_stats(
                    validation,
                    feature_index=left.feature_index,
                    threshold_value=left.threshold_value,
                    feature_scale=left.feature_scale,
                    tail_direction=left.tail_direction,
                    trade_side=trade_side,
                    horizon=horizon,
                    cost_floor_bps=cost_floor_bps,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    second_feature_index=right.feature_index,
                    second_threshold_value=right.threshold_value,
                    second_feature_scale=right.feature_scale,
                    second_tail_direction=right.tail_direction,
                )
                if (
                    int(validation_stats["signal_count"]) < min_validation_signals
                    or float(validation_stats["net_mean_edge_bps"]) <= 0.0
                    or float(validation_stats["hit_rate"]) < 0.53
                ):
                    continue
                trade_name = "long" if trade_side > 0.0 else "short"
                first, second = sorted((left, right), key=lambda item: item.feature_index)
                confidence = _clamp(
                    min(float(train_stats["net_mean_edge_bps"]), float(validation_stats["net_mean_edge_bps"])) / max(cost_floor_bps, 1.0),
                    0.15,
                    1.0,
                )
                candidate = RuleAlphaCandidate(
                    name=(
                        f"{_EMPIRICAL_RULE_ALPHA_FAMILY}:{profile}:"
                        f"f{first.feature_index}{first.tail_name}+f{second.feature_index}{second.tail_name}:{trade_name}"
                    ),
                    family=_EMPIRICAL_RULE_ALPHA_FAMILY,
                    threshold=float(probability_threshold),
                    sensitivity=8.0,
                    deadband=0.0,
                    stop_loss_multiplier=0.12 if horizon <= 8 else (0.20 if horizon <= 30 else 0.28),
                    take_profit_multiplier=0.10 if horizon <= 8 else (0.18 if horizon <= 30 else 0.26),
                    cooldown_multiplier=0.0,
                    min_position_hold_bars=int(horizon),
                    flat_signal_exit_grace_bars=int(grace_bars),
                    max_position_hold_bars=int(horizon),
                    params={
                        "condition_count": 2,
                        "feature_index": int(first.feature_index),
                        "feature_threshold": float(first.threshold_value),
                        "feature_scale": float(first.feature_scale),
                        "tail_direction": float(first.tail_direction),
                        "second_feature_index": int(second.feature_index),
                        "second_feature_threshold": float(second.threshold_value),
                        "second_feature_scale": float(second.feature_scale),
                        "second_tail_direction": float(second.tail_direction),
                        "trade_side": float(trade_side),
                        "edge_confidence": float(confidence),
                        "edge_slope": 1.0,
                        "training_signal_count": int(train_stats["signal_count"]),
                        "validation_signal_count": int(validation_stats["signal_count"]),
                        "training_net_edge_bps": float(train_stats["net_mean_edge_bps"]),
                        "validation_net_edge_bps": float(validation_stats["net_mean_edge_bps"]),
                        "training_hit_rate": float(train_stats["hit_rate"]),
                        "validation_hit_rate": float(validation_stats["hit_rate"]),
                        "horizon_bars": int(horizon),
                        "duration_contract": "wall_clock_seconds_v1",
                        "bar_interval_ms": int(bar_interval_ms),
                        "effective_max_hold_seconds": float(horizon * bar_interval_ms) / 1000.0,
                        "effective_grace_seconds": float(grace_bars * bar_interval_ms) / 1000.0,
                    },
                )
                score = (
                    min(float(train_stats["net_mean_edge_bps"]), float(validation_stats["net_mean_edge_bps"])),
                    min(float(train_stats["hit_rate"]), float(validation_stats["hit_rate"])),
                    min(int(train_stats["signal_count"]), int(validation_stats["signal_count"])),
                    -float(first.feature_index + second.feature_index),
                )
                candidates.append((score, candidate))
                emitted += 1
                if emitted >= DEFAULT_EMPIRICAL_RULE_ALPHA_INTERACTION_MAX_CANDIDATES:
                    return


def _empirical_edge_stats(
    rows: Sequence[ModelRow],
    *,
    feature_index: int,
    threshold_value: float,
    feature_scale: float,
    tail_direction: float,
    trade_side: float,
    horizon: int,
    cost_floor_bps: float,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    second_feature_index: int | None = None,
    second_threshold_value: float | None = None,
    second_feature_scale: float | None = None,
    second_tail_direction: float | None = None,
    entry_offset: int = RULE_ALPHA_EXECUTION_ENTRY_OFFSET,
) -> dict[str, object]:
    edges: list[float] = []
    side = 1.0 if trade_side >= 0.0 else -1.0
    tail = 1.0 if tail_direction >= 0.0 else -1.0
    scale = max(abs(float(feature_scale)), 1e-9)
    offset = max(0, int(entry_offset))
    available = max(0, len(rows) - horizon - offset)
    for index in range(0, available):
        current = rows[index]
        entry = rows[index + offset]
        future = rows[index + offset + horizon]
        if len(current.features) <= feature_index or entry.close <= 0.0 or future.close <= 0.0:
            continue
        value = float(current.features[feature_index])
        if not math.isfinite(value):
            continue
        if tail * (value - threshold_value) <= 0.0:
            continue
        if second_feature_index is not None:
            if len(current.features) <= second_feature_index:
                continue
            second_value = float(current.features[second_feature_index])
            if not math.isfinite(second_value):
                continue
            second_tail = 1.0 if float(second_tail_direction if second_tail_direction is not None else 1.0) >= 0.0 else -1.0
            second_threshold = float(second_threshold_value if second_threshold_value is not None else 0.0)
            if second_tail * (second_value - second_threshold) <= 0.0:
                continue
        strength = math.tanh(max(0.0, tail * (value - threshold_value) / scale))
        if second_feature_index is not None:
            second_scale = max(abs(float(second_feature_scale if second_feature_scale is not None else 1.0)), 1e-9)
            second_tail = 1.0 if float(second_tail_direction if second_tail_direction is not None else 1.0) >= 0.0 else -1.0
            second_threshold = float(second_threshold_value if second_threshold_value is not None else 0.0)
            second_value = float(current.features[second_feature_index])
            second_strength = math.tanh(max(0.0, second_tail * (second_value - second_threshold) / second_scale))
            strength = min(strength, second_strength)
        if strength <= 0.0:
            continue
        if stop_loss_pct is not None and take_profit_pct is not None:
            edge_bps, _bars_held, _exit_reason = _rule_alpha_path_edge_bps_with_barriers(
                rows,
                index + offset,
                horizon,
                int(side),
                stop_loss_pct=float(stop_loss_pct),
                take_profit_pct=float(take_profit_pct),
            )
        else:
            edge_bps = side * ((float(future.close) - float(entry.close)) / float(entry.close)) * 10_000.0
        if math.isfinite(edge_bps):
            edges.append(edge_bps)
    if not edges:
        return {
            "signal_count": 0,
            "mean_edge_bps": 0.0,
            "net_mean_edge_bps": -float(cost_floor_bps),
            "hit_rate": 0.0,
            "cost_floor_bps": float(cost_floor_bps),
        }
    mean_edge = sum(edges) / len(edges)
    wins = sum(1 for edge in edges if edge > cost_floor_bps)
    return {
        "signal_count": int(len(edges)),
        "mean_edge_bps": float(mean_edge),
        "net_mean_edge_bps": float(mean_edge - cost_floor_bps),
        "hit_rate": float(wins / len(edges)),
        "cost_floor_bps": float(cost_floor_bps),
    }


def _quantile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    q = _clamp(float(quantile), 0.0, 1.0)
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def model_for_rule_alpha(
    rows: Sequence[ModelRow],
    candidate: RuleAlphaCandidate,
    strategy: StrategyConfig,
    *,
    market_type: str,
    probability_inverted: bool = False,
    feature_params: dict[str, object] | None = None,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> TrainedModel:
    """Build a serializable model that emits only the selected rule alpha."""

    if not rows:
        raise ValueError("rule alpha model requires at least one feature row")
    feature_dim = len(rows[0].features)
    threshold = _clamp(candidate.threshold, 0.50 if market_type == "futures" else 0.05, 0.95)
    short_threshold = 1.0 - threshold if market_type == "futures" else None
    expert_params = {
        "family": candidate.family,
        "sensitivity": candidate.sensitivity,
        "deadband": candidate.deadband,
        **dict(feature_params or {}),
        **dict(candidate.params or {}),
    }
    model = TrainedModel(
        weights=[0.0] * feature_dim,
        bias=0.0,
        feature_dim=feature_dim,
        epochs=0,
        feature_means=[0.0] * feature_dim,
        feature_stds=[1.0] * feature_dim,
        decision_threshold=threshold,
        long_decision_threshold=threshold if market_type == "futures" else None,
        short_decision_threshold=short_threshold,
        threshold_source="rule_alpha_model_zoo",
        model_family="rule_alpha_model_zoo",
        probability_inverted=bool(probability_inverted),
        hybrid_base_weight=0.0,
        hybrid_profile="rule_alpha_only",
        rule_alpha_profile=f"{candidate.name}:inverted" if probability_inverted else candidate.name,
        rule_alpha_family=candidate.family,
        hybrid_experts=[
            HybridExpert(
                name="rule_alpha_intraday_template",
                kind="rule_alpha",
                weight=1.0,
                feature_count=feature_dim,
                params=expert_params,
                notes="Interpretable intraday alpha template selected by bounded real-backtest search.",
            )
        ],
    )
    model.strategy_overrides = strategy_overrides_from_config(
        _candidate_strategy(
            strategy,
            candidate,
            market_type=market_type,
            symbol_profile=symbol_profile,
        )
    )
    if probability_inverted:
        model.quality_warnings = [
            *list(getattr(model, "quality_warnings", []) or []),
            "rule_alpha_probability_inversion_variant",
        ]
    return model


def _rule_alpha_one_side_cost_bps(
    strategy: StrategyConfig,
    *,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> float:
    assumptions = (
        execution_assumptions_for_symbol(strategy, symbol_profile)
        if symbol_profile is not None
        else execution_assumptions_from_strategy(strategy)
    )
    latency_seconds = min(10.0, max(0.0, float(assumptions.latency_ms)) / 1000.0)
    participation = _rule_alpha_cost_floor_participation(symbol_profile)
    impact_cost = max(0.0, float(assumptions.impact_coefficient)) * math.sqrt(participation)
    return max(
        0.0,
        max(0.0, float(assumptions.spread_bps)) / 2.0
        + max(0.0, float(assumptions.volatility_buffer_bps)) * latency_seconds
        + impact_cost
        + max(0.0, float(assumptions.testnet_to_live_buffer_bps)),
    )


def _rule_alpha_cost_floor_participation(symbol_profile: SymbolExecutionProfile | None = None) -> float:
    """Return a conservative participation proxy for label/event cost floors.

    The final backtest uses per-row quote volume and actual order notional. This
    preselection floor only prevents obviously sub-cost labels from dominating
    training, so it should not assume a fixed 5% share of the one-second book for
    BTC/ETH/SOL-sized autonomous positions.
    """

    if symbol_profile is None:
        return _RULE_ALPHA_COST_FLOOR_PARTICIPATION
    liquidity_score = min(1.0, max(0.0, float(symbol_profile.liquidity_score)))
    quote_volume = max(0.0, float(symbol_profile.quote_volume))
    trade_count = max(0, int(symbol_profile.trade_count))
    profiled_floor = _RULE_ALPHA_MIN_PROFILED_COST_FLOOR_PARTICIPATION
    if quote_volume >= 1_000_000_000.0 and trade_count >= 1_000_000:
        profiled_floor = _RULE_ALPHA_MIN_MAJOR_COST_FLOOR_PARTICIPATION
    liquidity_scaled = _RULE_ALPHA_COST_FLOOR_PARTICIPATION * ((1.0 - liquidity_score) ** 2)
    return max(profiled_floor, min(_RULE_ALPHA_COST_FLOOR_PARTICIPATION, liquidity_scaled))


def rule_alpha_stop_loss_floor_pct(
    strategy: StrategyConfig,
    *,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> float:
    """Return a minimum stop distance above one adverse modeled fill."""

    fee_bps = max(0.0, float(strategy.taker_fee_bps))
    floor_bps = (
        _rule_alpha_one_side_cost_bps(strategy, symbol_profile=symbol_profile)
        + fee_bps
        + _RULE_ALPHA_MIN_STOP_BUFFER_BPS
    )
    return max(0.0005, floor_bps / 10_000.0)


def rule_alpha_take_profit_floor_pct(
    strategy: StrategyConfig,
    *,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> float:
    """Return the minimum target move needed to clear round-trip modeled costs."""

    fee_bps = max(0.0, float(strategy.taker_fee_bps))
    floor_bps = (
        (2.0 * _rule_alpha_one_side_cost_bps(strategy, symbol_profile=symbol_profile))
        + (2.0 * fee_bps)
        + _RULE_ALPHA_MIN_PROFIT_BUFFER_BPS
    )
    return max(0.0005, floor_bps / 10_000.0)


def _rule_alpha_event_rank_pool_limit(row_count: int, template_replay_limit: int) -> int:
    replay_limit = max(1, int(template_replay_limit))
    rows = max(0, int(row_count))
    if rows >= RULE_ALPHA_LARGE_WINDOW_ROW_COUNT:
        return replay_limit
    return max(
        replay_limit,
        replay_limit * DEFAULT_RULE_ALPHA_EVENT_RANK_POOL_MULTIPLIER,
    )


def optimize_rule_alpha_model_zoo(
    selection_rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    objective_name: str,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
    max_candidates: int = DEFAULT_RULE_ALPHA_MAX_CANDIDATES,
    empirical_max_feature_count: int | None = None,
    feature_cfg: AdvancedFeatureConfig | None = None,
    symbol_profile: SymbolExecutionProfile | None = None,
    ranking_rows: Sequence[ModelRow] | None = None,
    status_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> RuleAlphaOptimizationReport:
    """Return the best accepted rule-alpha model, or a rejected report."""

    rows = list(selection_rows)
    if not rows:
        return RuleAlphaOptimizationReport(False, None, float("-inf"), None, False, None, 0, None, ())
    discovery_rows = list(ranking_rows) if ranking_rows is not None else rows
    if not discovery_rows:
        discovery_rows = rows
    objective = get_objective(objective_name)
    best_accepted: RuleAlphaCandidateResult | None = None
    best_diagnostic: RuleAlphaCandidateResult | None = None
    diagnostics: list[RuleAlphaCandidateResult] = []
    regime_scores = precompute_backtest_regime_scores(rows, strategy)
    liquidity_adjustments = precompute_backtest_liquidity_adjustments(rows, strategy)
    feature_params = rule_alpha_feature_params(feature_cfg)
    feature_dim = len(discovery_rows[0].features) if discovery_rows else 0
    bar_interval_ms = _median_row_interval_ms(discovery_rows)
    empirical_feature_scan_limit = len(_empirical_feature_scan_indices(
        feature_dim=feature_dim,
        max_feature_count=empirical_max_feature_count,
        feature_cfg=feature_cfg,
    ))
    template_replay_limit = max(1, int(max_candidates))
    event_rank_pool_limit = _rule_alpha_event_rank_pool_limit(len(discovery_rows), template_replay_limit)
    template_pool = rule_alpha_candidates(
        objective.name,
        max_candidates=event_rank_pool_limit,
        bar_interval_ms=bar_interval_ms,
    )
    template_candidates, event_rank_summary = _event_rank_rule_alpha_candidates(
        discovery_rows,
        strategy,
        template_pool,
        market_type=market_type,
        replay_limit=template_replay_limit,
        feature_params=feature_params,
        symbol_profile=symbol_profile,
        status_callback=status_callback,
    )
    if status_callback is not None:
        status_callback(
            "rule_alpha_event_rank_complete",
            {
                "selection_rows": len(rows),
                "ranking_rows": len(discovery_rows),
                "template_candidates": len(template_candidates),
                "template_pool_candidates": len(template_pool),
                "event_rank_pool_candidates": int(event_rank_summary.get("event_rank_pool_candidates", 0) or 0),
                "event_rank_positive_pool_candidates": int(
                    event_rank_summary.get("event_rank_positive_pool_candidates", 0) or 0
                ),
                "event_rank_best_candidate": str(event_rank_summary.get("event_rank_best_candidate", "") or ""),
                "event_rank_best_net_edge_bps": float(
                    event_rank_summary.get("event_rank_best_net_edge_bps", 0.0) or 0.0
                ),
            },
        )
        status_callback(
            "rule_alpha_empirical_mining_started",
            {
                "selection_rows": len(rows),
                "ranking_rows": len(discovery_rows),
                "feature_dim": int(feature_dim),
                "feature_scan_limit": int(empirical_feature_scan_limit),
                "candidate_limit": DEFAULT_EMPIRICAL_RULE_ALPHA_MAX_CANDIDATES,
                "interaction_candidate_limit": DEFAULT_EMPIRICAL_RULE_ALPHA_INTERACTION_MAX_CANDIDATES,
            },
        )
    empirical_candidates = mine_empirical_rule_alpha_candidates(
        discovery_rows,
        strategy,
        objective_name=objective.name,
        market_type=market_type,
        max_candidates=DEFAULT_EMPIRICAL_RULE_ALPHA_MAX_CANDIDATES,
        max_feature_count=empirical_max_feature_count,
        feature_cfg=feature_cfg,
        symbol_profile=symbol_profile,
        bar_interval_ms=bar_interval_ms,
        status_callback=status_callback,
    )
    if status_callback is not None:
        status_callback(
            "rule_alpha_empirical_mining_complete",
            {
                "selection_rows": len(rows),
                "ranking_rows": len(discovery_rows),
                "empirical_candidates": len(empirical_candidates),
                "empirical_interaction_candidates": sum(
                    1 for candidate in empirical_candidates if "second_feature_index" in candidate.params
                ),
            },
        )
    replay_candidates = (*template_candidates, *empirical_candidates)
    total_replays = len(replay_candidates) * 2
    replay_index = 0
    for candidate in replay_candidates:
        candidate_strategy = _candidate_strategy(
            strategy,
            candidate,
            market_type=market_type,
            symbol_profile=symbol_profile,
        )
        for inverted in (False, True):
            replay_index += 1
            if status_callback is not None:
                status_callback(
                    "rule_alpha_candidate_started",
                    {
                        "candidate_index": replay_index,
                        "candidate_count": total_replays,
                        "candidate": candidate.name,
                        "candidate_family": candidate.family,
                        "probability_inverted": bool(inverted),
                        "evaluated_candidates": len(diagnostics),
                    },
                )
            model = model_for_rule_alpha(
                rows,
                candidate,
                strategy,
                market_type=market_type,
                probability_inverted=inverted,
                feature_params=feature_params,
                symbol_profile=symbol_profile,
            )
            event_study = rule_alpha_event_study(
                rows,
                model,
                candidate_strategy,
                candidate,
                market_type=market_type,
                symbol_profile=symbol_profile,
            )
            result = run_backtest(
                rows,
                model,
                candidate_strategy,
                starting_cash=starting_cash,
                market_type=market_type,
                compute_backend=compute_backend,
                score_batch_size=score_batch_size,
                symbol_profile=symbol_profile,
                precomputed_regime_scores=regime_scores,
                precomputed_liquidity_adjustments=liquidity_adjustments,
            )
            reject_reason = objective.reject_reason(result)
            accepted = reject_reason is None
            raw_score = objective.score(result)
            score = raw_score if accepted else float("-inf")
            candidate_result = RuleAlphaCandidateResult(
                candidate=candidate,
                probability_inverted=inverted,
                accepted=accepted,
                score=float(score),
                raw_score=float(raw_score),
                reject_reason=reject_reason,
                result=result,
                event_study=event_study,
            )
            diagnostics.append(candidate_result)
            if accepted and (best_accepted is None or candidate_result.score > best_accepted.score + 1e-12):
                best_accepted = candidate_result
            if best_diagnostic is None or _diagnostic_rank_key(candidate_result) > _diagnostic_rank_key(best_diagnostic):
                best_diagnostic = candidate_result
            if status_callback is not None:
                diagnostic = best_accepted or best_diagnostic
                status_callback(
                    "rule_alpha_candidate_evaluated",
                    {
                        "candidate_index": replay_index,
                        "candidate_count": total_replays,
                        "candidate": candidate.name,
                        "candidate_family": candidate.family,
                        "probability_inverted": bool(inverted),
                        "accepted": bool(accepted),
                        "raw_score": float(raw_score),
                        "score": float(score),
                        "realized_pnl": float(result.realized_pnl),
                        "closed_trades": int(result.closed_trades),
                        "best_candidate": diagnostic.candidate.name if diagnostic is not None else "",
                        "best_accepted": bool(diagnostic.accepted) if diagnostic is not None else False,
                        "best_raw_score": float(diagnostic.raw_score) if diagnostic is not None else float("-inf"),
                        "best_pnl": float(diagnostic.result.realized_pnl) if diagnostic is not None else 0.0,
                        "best_closed_trades": int(diagnostic.result.closed_trades) if diagnostic is not None else 0,
                    },
                )

    candidate_summary = summarize_rule_alpha_candidate_distribution(diagnostics)
    candidate_summary.update(event_rank_summary)
    candidate_summary["ranking_rows"] = int(len(discovery_rows))
    candidate_summary["selection_rows"] = int(len(rows))
    candidate_summary["ranking_source"] = "external_chronological_training_slice" if ranking_rows is not None else "selection_slice"
    candidate_summary["bar_interval_ms"] = int(bar_interval_ms)
    candidate_summary["static_template_candidates"] = int(len(template_candidates))
    candidate_summary["static_template_pool_candidates"] = int(len(template_pool))
    candidate_summary["empirical_mined_candidates"] = int(len(empirical_candidates))
    candidate_summary["empirical_feature_scan_limit"] = int(empirical_feature_scan_limit)
    candidate_summary["empirical_interaction_candidates"] = int(
        sum(1 for candidate in empirical_candidates if "second_feature_index" in candidate.params)
    )
    candidate_summary["empirical_candidate_limit"] = int(DEFAULT_EMPIRICAL_RULE_ALPHA_MAX_CANDIDATES)
    winner = best_accepted or best_diagnostic
    if winner is None:
        return RuleAlphaOptimizationReport(
            False,
            None,
            float("-inf"),
            None,
            False,
            None,
            len(diagnostics),
            None,
            tuple(diagnostics),
            dict(candidate_summary),
        )
    model = model_for_rule_alpha(
        rows,
        winner.candidate,
        strategy,
        market_type=market_type,
        probability_inverted=winner.probability_inverted,
        feature_params=feature_params,
        symbol_profile=symbol_profile,
    )
    model.rule_alpha_best_score = float(winner.score if winner.accepted else winner.raw_score)
    model.rule_alpha_best_pnl = float(winner.result.realized_pnl)
    model.rule_alpha_best_closed_trades = int(winner.result.closed_trades)
    path_summary = summarize_rule_alpha_trade_path(winner.result)
    model.rule_alpha_best_win_rate = float(path_summary["win_rate"])
    model.rule_alpha_best_profit_factor = float(path_summary["profit_factor"])
    model.rule_alpha_best_max_drawdown = float(path_summary["max_drawdown"])
    model.rule_alpha_best_exit_reason_counts = dict(path_summary["exit_reason_counts"])
    model.rule_alpha_best_side_counts = dict(path_summary["side_counts"])
    model.rule_alpha_best_reject_reason = str(winner.reject_reason or "")
    model.rule_alpha_probability_inverted = bool(winner.probability_inverted)
    model.rule_alpha_evaluated_candidates = len(diagnostics)
    model.rule_alpha_candidate_summary = dict(candidate_summary)
    model.round_selection_gate_passed = bool(winner.accepted)
    model.round_selection_reject_reason = "" if winner.accepted else str(winner.reject_reason or "rule_alpha_selection_failed")
    if not winner.accepted:
        model.decision_threshold = 1.0
        model.long_decision_threshold = 1.0
        model.short_decision_threshold = None
        model.quality_warnings = [
            *list(getattr(model, "quality_warnings", [])),
            "rule_alpha_model_zoo_rejected",
            "round_selection_failed_no_entry_enforced",
        ]
    return RuleAlphaOptimizationReport(
        bool(winner.accepted),
        model if winner.accepted else None,
        float(winner.score if winner.accepted else winner.raw_score),
        winner.candidate,
        bool(winner.probability_inverted),
        winner.reject_reason,
        len(diagnostics),
        winner.result,
        tuple(diagnostics),
        dict(candidate_summary),
    )


def _candidate_strategy(
    strategy: StrategyConfig,
    candidate: RuleAlphaCandidate,
    *,
    market_type: str = "spot",
    symbol_profile: SymbolExecutionProfile | None = None,
) -> StrategyConfig:
    payload = strategy.asdict()
    stop_loss_pct, take_profit_pct = _rule_alpha_profile_barriers(
        strategy,
        market_type=market_type,
        stop_loss_multiplier=candidate.stop_loss_multiplier,
        take_profit_multiplier=candidate.take_profit_multiplier,
        symbol_profile=symbol_profile,
    )
    payload["stop_loss_pct"] = stop_loss_pct
    payload["take_profit_pct"] = take_profit_pct
    payload["cooldown_minutes"] = max(0, int(round(float(strategy.cooldown_minutes) * max(0.0, candidate.cooldown_multiplier))))
    payload["min_position_hold_bars"] = max(0, int(candidate.min_position_hold_bars))
    payload["flat_signal_exit_grace_bars"] = max(0, int(candidate.flat_signal_exit_grace_bars))
    payload["max_position_hold_bars"] = max(
        1,
        int(candidate.max_position_hold_bars or candidate.min_position_hold_bars or 1),
    )
    return StrategyConfig(**payload)


def _rule_alpha_profile_barriers(
    strategy: StrategyConfig,
    *,
    market_type: str,
    stop_loss_multiplier: float,
    take_profit_multiplier: float,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> tuple[float, float]:
    stop_floor = rule_alpha_stop_loss_floor_pct(strategy, symbol_profile=symbol_profile)
    take_floor = rule_alpha_take_profit_floor_pct(strategy, symbol_profile=symbol_profile)
    price_move_scale = _price_move_scale_for_market(strategy, market_type)
    base_stop_loss_pct = float(strategy.stop_loss_pct) * price_move_scale
    base_take_profit_pct = float(strategy.take_profit_pct) * price_move_scale
    stop_loss_pct = max(
        0.0005,
        stop_floor,
        base_stop_loss_pct * max(0.05, float(stop_loss_multiplier)),
    )
    take_profit_pct = max(
        0.0005,
        take_floor + (_RULE_ALPHA_MIN_PROFIT_BUFFER_BPS / 10_000.0),
        base_take_profit_pct * max(0.05, float(take_profit_multiplier)),
    )
    if take_profit_pct <= stop_loss_pct:
        take_profit_pct = stop_loss_pct + (_RULE_ALPHA_MIN_PROFIT_BUFFER_BPS / 10_000.0)
    return float(stop_loss_pct), float(take_profit_pct)


def _diagnostic_rank_key(result: RuleAlphaCandidateResult) -> tuple[float, ...]:
    backtest = result.result
    if result.accepted:
        return (
            1.0,
            float(result.score),
            float(backtest.realized_pnl),
            float(backtest.profit_factor),
            -float(backtest.max_drawdown),
            -float(backtest.max_consecutive_losses),
            float(backtest.closed_trades),
        )
    return (
        0.0,
        1.0 if int(backtest.closed_trades) > 0 else 0.0,
        float(backtest.realized_pnl),
        float(backtest.edge_vs_buy_hold),
        float(backtest.profit_factor),
        -float(backtest.max_drawdown),
        -float(backtest.max_consecutive_losses),
        float(backtest.closed_trades),
        float(result.raw_score),
    )


def _clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return low
    return low if value < low else (high if value > high else value)


__all__ = [
    "DEFAULT_RULE_ALPHA_MAX_CANDIDATES",
    "DEFAULT_EMPIRICAL_RULE_ALPHA_MAX_CANDIDATES",
    "RuleAlphaCandidate",
    "RuleAlphaCandidateResult",
    "RuleAlphaOptimizationReport",
    "mine_empirical_rule_alpha_candidates",
    "model_for_rule_alpha",
    "optimize_rule_alpha_model_zoo",
    "rule_alpha_candidates",
    "rule_alpha_event_study",
    "rule_alpha_feature_params",
    "rule_alpha_stop_loss_floor_pct",
    "rule_alpha_take_profit_floor_pct",
    "summarize_rule_alpha_candidate_distribution",
    "summarize_rule_alpha_trade_path",
]
