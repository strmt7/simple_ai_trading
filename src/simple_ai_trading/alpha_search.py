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
from typing import Any, Mapping, Sequence

from .advanced_model import AdvancedFeatureConfig, advanced_feature_group_spans
from .backtest import (
    BacktestResult,
    precompute_backtest_liquidity_adjustments,
    precompute_backtest_regime_scores,
    run_backtest,
)
from .execution_simulation import execution_assumptions_from_strategy
from .features import ModelRow
from .model import HybridExpert, TrainedModel
from .objective import get_objective
from .strategy_overrides import strategy_overrides_from_config
from .types import StrategyConfig


DEFAULT_RULE_ALPHA_MAX_CANDIDATES = 225
DEFAULT_EMPIRICAL_RULE_ALPHA_MAX_CANDIDATES = 18
DEFAULT_EMPIRICAL_RULE_ALPHA_INTERACTION_MAX_CANDIDATES = 18
_RULE_ALPHA_COST_FLOOR_PARTICIPATION = 0.05
_RULE_ALPHA_MIN_PROFIT_BUFFER_BPS = 2.0
_RULE_ALPHA_MIN_STOP_BUFFER_BPS = 1.0
_EMPIRICAL_RULE_ALPHA_FAMILY = "empirical_feature_edge"


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
        elif span.name == "higher_timeframe_context" and span.size > 0:
            window_count = max(0, len(tuple(feature_cfg.higher_timeframe_windows)))
            params["higher_timeframe_start"] = int(span.start)
            params["higher_timeframe_width"] = int(span.size) // window_count if window_count > 0 else int(span.size)
            params["higher_timeframe_window_count"] = window_count
    return params


def rule_alpha_candidates(objective_name: str, *, max_candidates: int | None = None) -> tuple[RuleAlphaCandidate, ...]:
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
    )
    execution_profiles = (
        ("scalp_3s", 0.06, 0.05, 0.0, 1, 0),
        ("scalp_8s", 0.08, 0.07, 0.0, 2, 0),
        ("scalp_20s", 0.10, 0.09, 0.0, 4, 1),
        ("micro", 0.14, 0.10, 0.0, 2, 0),
        ("balanced", 0.18, 0.14, 0.0, 3, 1),
        ("guarded", 0.22, 0.18, 0.25, 4, 1),
        ("held_30s", 0.18, 0.16, 0.0, 30, 10),
        ("held_90s", 0.24, 0.22, 0.0, 90, 20),
        ("held_180s", 0.30, 0.28, 0.0, 180, 30),
    )
    limit = max(1, int(max_candidates)) if max_candidates is not None else DEFAULT_RULE_ALPHA_MAX_CANDIDATES
    ranked: list[tuple[tuple[int, int, int, int, int, int], str, str, RuleAlphaCandidate]] = []
    base_by_family_profile: dict[tuple[str, str], RuleAlphaCandidate] = {}
    for threshold_index, threshold in enumerate(thresholds):
        for sensitivity_index, sensitivity in enumerate(sensitivities):
            for deadband_index, deadband in enumerate(deadbands):
                for family_index, family in enumerate(families):
                    for profile_index, (profile, stop_mult, take_mult, cooldown_mult, hold_bars, grace_bars) in enumerate(execution_profiles):
                        candidate = RuleAlphaCandidate(
                            name=f"{family}:{profile}:t{threshold:.2f}:s{sensitivity:.1f}:d{deadband:.2f}",
                            family=family,
                            threshold=float(threshold),
                            sensitivity=float(sensitivity),
                            deadband=float(deadband),
                            stop_loss_multiplier=float(stop_mult),
                            take_profit_multiplier=float(take_mult),
                            cooldown_multiplier=float(cooldown_mult),
                            min_position_hold_bars=int(hold_bars),
                            flat_signal_exit_grace_bars=int(grace_bars),
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
    model: TrainedModel,
    strategy: StrategyConfig,
    candidate: RuleAlphaCandidate,
    *,
    market_type: str,
) -> dict[str, object]:
    """Measure directional forward edge before full trade-lifecycle replay.

    This is telemetry only.  Promotion still depends on `run_backtest` and the
    objective gates, because stop/take/cooldown behavior can differ from a raw
    forward-return event study.
    """

    row_list = list(rows)
    horizon = max(1, min(300, int(candidate.min_position_hold_bars or 1)))
    if len(row_list) <= horizon:
        return {
            "horizon_bars": int(horizon),
            "signal_count": 0,
            "long_signal_count": 0,
            "short_signal_count": 0,
            "mean_edge_bps": 0.0,
            "net_mean_edge_bps": 0.0,
            "hit_rate": 0.0,
            "cost_floor_bps": float(rule_alpha_take_profit_floor_pct(strategy) * 10_000.0),
        }
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
    cost_floor_bps = float(rule_alpha_take_profit_floor_pct(strategy) * 10_000.0)
    edges: list[float] = []
    long_count = 0
    short_count = 0
    for index in range(0, len(row_list) - horizon):
        current = row_list[index]
        future = row_list[index + horizon]
        if current.close <= 0.0 or future.close <= 0.0:
            continue
        probability = float(model.predict_proba(current.features))
        side = 0
        if probability >= long_threshold:
            side = 1
            long_count += 1
        elif short_threshold is not None and probability <= short_threshold:
            side = -1
            short_count += 1
        if side == 0:
            continue
        edge_bps = float(side) * ((float(future.close) - float(current.close)) / float(current.close)) * 10_000.0
        if math.isfinite(edge_bps):
            edges.append(edge_bps)
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
    }


def mine_empirical_rule_alpha_candidates(
    rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    objective_name: str,
    market_type: str,
    max_candidates: int = DEFAULT_EMPIRICAL_RULE_ALPHA_MAX_CANDIDATES,
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
    feature_count = min(len(row_list[0].features), 512)
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
    profiles = (
        ("empirical_8s", 0.10, 0.09, 8, 2),
        ("empirical_30s", 0.18, 0.16, 30, 8),
        ("empirical_90s", 0.26, 0.24, 90, 20),
    )
    quantiles = (
        (0.08, -1.0, "low08"),
        (0.12, -1.0, "low12"),
        (0.20, -1.0, "low20"),
        (0.80, 1.0, "high80"),
        (0.88, 1.0, "high88"),
        (0.92, 1.0, "high92"),
    )
    cost_floor_bps = float(rule_alpha_take_profit_floor_pct(strategy) * 10_000.0)
    candidates: list[tuple[tuple[float, float, int, float], RuleAlphaCandidate]] = []
    condition_pools: dict[tuple[str, int, int, float], list[_EmpiricalCondition]] = {}

    for feature_index in range(feature_count):
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
        for profile, stop_mult, take_mult, horizon, grace_bars in profiles:
            if len(training) <= horizon or len(validation) <= horizon:
                continue
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
                        key = (profile, int(horizon), int(grace_bars), float(trade_side))
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
    )

    candidates.sort(key=lambda item: item[0], reverse=True)
    output: list[RuleAlphaCandidate] = []
    seen: set[tuple[int, float, float, int, int, float]] = set()
    for _score, candidate in candidates:
        key = (
            int(candidate.params.get("feature_index", -1)),
            float(candidate.params.get("tail_direction", 0.0)),
            float(candidate.params.get("trade_side", 0.0)),
            int(candidate.params.get("horizon_bars", 0)),
            int(candidate.params.get("second_feature_index", -1)),
            float(candidate.params.get("second_tail_direction", 0.0)),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
        if len(output) >= limit:
            break
    return tuple(output)


def _mine_empirical_interaction_candidates(
    candidates: list[tuple[tuple[float, float, int, float], RuleAlphaCandidate]],
    condition_pools: Mapping[tuple[str, int, int, float], list[_EmpiricalCondition]],
    *,
    training: Sequence[ModelRow],
    validation: Sequence[ModelRow],
    cost_floor_bps: float,
    probability_threshold: float,
    market_type: str,
) -> None:
    """Append validated two-condition empirical rules to ``candidates``."""

    del market_type  # side handling is already encoded in the condition-pool key.
    emitted = 0
    for (profile, horizon, grace_bars, trade_side), pool in condition_pools.items():
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
    second_feature_index: int | None = None,
    second_threshold_value: float | None = None,
    second_feature_scale: float | None = None,
    second_tail_direction: float | None = None,
) -> dict[str, object]:
    edges: list[float] = []
    side = 1.0 if trade_side >= 0.0 else -1.0
    tail = 1.0 if tail_direction >= 0.0 else -1.0
    scale = max(abs(float(feature_scale)), 1e-9)
    for index in range(0, len(rows) - horizon):
        current = rows[index]
        future = rows[index + horizon]
        if len(current.features) <= feature_index or current.close <= 0.0 or future.close <= 0.0:
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
        edge_bps = side * ((float(future.close) - float(current.close)) / float(current.close)) * 10_000.0
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
    model.strategy_overrides = strategy_overrides_from_config(_candidate_strategy(strategy, candidate))
    if probability_inverted:
        model.quality_warnings = [
            *list(getattr(model, "quality_warnings", []) or []),
            "rule_alpha_probability_inversion_variant",
        ]
    return model


def _rule_alpha_one_side_cost_bps(strategy: StrategyConfig) -> float:
    assumptions = execution_assumptions_from_strategy(strategy)
    latency_seconds = min(10.0, max(0.0, float(assumptions.latency_ms)) / 1000.0)
    impact_cost = max(0.0, float(assumptions.impact_coefficient)) * math.sqrt(_RULE_ALPHA_COST_FLOOR_PARTICIPATION)
    return max(
        0.0,
        max(0.0, float(assumptions.spread_bps)) / 2.0
        + max(0.0, float(assumptions.volatility_buffer_bps)) * latency_seconds
        + impact_cost
        + max(0.0, float(assumptions.testnet_to_live_buffer_bps)),
    )


def rule_alpha_stop_loss_floor_pct(strategy: StrategyConfig) -> float:
    """Return a minimum stop distance above one adverse modeled fill."""

    fee_bps = max(0.0, float(strategy.taker_fee_bps))
    floor_bps = _rule_alpha_one_side_cost_bps(strategy) + fee_bps + _RULE_ALPHA_MIN_STOP_BUFFER_BPS
    return max(0.0005, floor_bps / 10_000.0)


def rule_alpha_take_profit_floor_pct(strategy: StrategyConfig) -> float:
    """Return the minimum target move needed to clear round-trip modeled costs."""

    fee_bps = max(0.0, float(strategy.taker_fee_bps))
    floor_bps = (2.0 * _rule_alpha_one_side_cost_bps(strategy)) + (2.0 * fee_bps) + _RULE_ALPHA_MIN_PROFIT_BUFFER_BPS
    return max(0.0005, floor_bps / 10_000.0)


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
    feature_cfg: AdvancedFeatureConfig | None = None,
) -> RuleAlphaOptimizationReport:
    """Return the best accepted rule-alpha model, or a rejected report."""

    rows = list(selection_rows)
    if not rows:
        return RuleAlphaOptimizationReport(False, None, float("-inf"), None, False, None, 0, None, ())
    objective = get_objective(objective_name)
    best_accepted: RuleAlphaCandidateResult | None = None
    best_diagnostic: RuleAlphaCandidateResult | None = None
    diagnostics: list[RuleAlphaCandidateResult] = []
    regime_scores = precompute_backtest_regime_scores(rows, strategy)
    liquidity_adjustments = precompute_backtest_liquidity_adjustments(rows, strategy)
    feature_params = rule_alpha_feature_params(feature_cfg)
    template_candidates = rule_alpha_candidates(objective.name, max_candidates=max_candidates)
    empirical_candidates = mine_empirical_rule_alpha_candidates(
        rows,
        strategy,
        objective_name=objective.name,
        market_type=market_type,
        max_candidates=DEFAULT_EMPIRICAL_RULE_ALPHA_MAX_CANDIDATES,
    )
    for candidate in (*template_candidates, *empirical_candidates):
        candidate_strategy = _candidate_strategy(strategy, candidate)
        for inverted in (False, True):
            model = model_for_rule_alpha(
                rows,
                candidate,
                strategy,
                market_type=market_type,
                probability_inverted=inverted,
                feature_params=feature_params,
            )
            event_study = rule_alpha_event_study(
                rows,
                model,
                candidate_strategy,
                candidate,
                market_type=market_type,
            )
            result = run_backtest(
                rows,
                model,
                candidate_strategy,
                starting_cash=starting_cash,
                market_type=market_type,
                compute_backend=compute_backend,
                score_batch_size=score_batch_size,
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

    candidate_summary = summarize_rule_alpha_candidate_distribution(diagnostics)
    candidate_summary["static_template_candidates"] = int(len(template_candidates))
    candidate_summary["empirical_mined_candidates"] = int(len(empirical_candidates))
    candidate_summary["empirical_interaction_candidates"] = int(
        sum(1 for candidate in empirical_candidates if "second_feature_index" in candidate.params)
    )
    candidate_summary["empirical_candidate_limit"] = int(DEFAULT_EMPIRICAL_RULE_ALPHA_MAX_CANDIDATES)
    winner = best_accepted or best_diagnostic
    if winner is None:
        return RuleAlphaOptimizationReport(False, None, float("-inf"), None, False, None, len(diagnostics), None, tuple(diagnostics))
    model = model_for_rule_alpha(
        rows,
        winner.candidate,
        strategy,
        market_type=market_type,
        probability_inverted=winner.probability_inverted,
        feature_params=feature_params,
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
    )


def _candidate_strategy(strategy: StrategyConfig, candidate: RuleAlphaCandidate) -> StrategyConfig:
    payload = strategy.asdict()
    stop_floor = rule_alpha_stop_loss_floor_pct(strategy)
    take_floor = rule_alpha_take_profit_floor_pct(strategy)
    stop_loss_pct = max(
        0.0005,
        stop_floor,
        float(strategy.stop_loss_pct) * max(0.05, candidate.stop_loss_multiplier),
    )
    take_profit_pct = max(
        0.0005,
        take_floor,
        float(strategy.take_profit_pct) * max(0.05, candidate.take_profit_multiplier),
    )
    if take_profit_pct <= stop_loss_pct:
        take_profit_pct = stop_loss_pct + (_RULE_ALPHA_MIN_PROFIT_BUFFER_BPS / 10_000.0)
    payload["stop_loss_pct"] = stop_loss_pct
    payload["take_profit_pct"] = take_profit_pct
    payload["cooldown_minutes"] = max(0, int(round(float(strategy.cooldown_minutes) * max(0.0, candidate.cooldown_multiplier))))
    payload["min_position_hold_bars"] = max(0, int(candidate.min_position_hold_bars))
    payload["flat_signal_exit_grace_bars"] = max(0, int(candidate.flat_signal_exit_grace_bars))
    return StrategyConfig(**payload)


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
