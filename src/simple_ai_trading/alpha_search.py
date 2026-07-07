"""Interpretable intraday alpha-template search.

This module adds a Freqtrade/vectorbt-style research layer before ML model
promotion.  It searches a bounded set of original, interpretable day-trading
templates, then validates each candidate through the same execution/risk
backtest used by trained models.  A candidate is promoted only when the target
objective accepts the full selection backtest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from .backtest import (
    BacktestResult,
    precompute_backtest_liquidity_adjustments,
    precompute_backtest_regime_scores,
    run_backtest,
)
from .features import ModelRow
from .model import HybridExpert, TrainedModel
from .objective import get_objective
from .strategy_overrides import strategy_overrides_from_config
from .types import StrategyConfig


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

    def asdict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.asdict(),
            "probability_inverted": bool(self.probability_inverted),
            "accepted": bool(self.accepted),
            "score": float(self.score),
            "raw_score": float(self.raw_score),
            "reject_reason": self.reject_reason,
            "result": asdict(self.result),
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
    )
    execution_profiles = (
        ("micro", 0.14, 0.10, 0.0, 2, 0),
        ("balanced", 0.18, 0.14, 0.0, 3, 1),
        ("guarded", 0.22, 0.18, 0.25, 4, 1),
    )
    limit = max(1, int(max_candidates)) if max_candidates is not None else 48
    ranked: list[tuple[tuple[int, int, int, int, int, int], RuleAlphaCandidate]] = []
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
                            candidate,
                        ))
    ranked.sort(key=lambda item: item[0])
    return tuple(candidate for _rank, candidate in ranked[:limit])


def model_for_rule_alpha(
    rows: Sequence[ModelRow],
    candidate: RuleAlphaCandidate,
    strategy: StrategyConfig,
    *,
    market_type: str,
    probability_inverted: bool = False,
) -> TrainedModel:
    """Build a serializable model that emits only the selected rule alpha."""

    if not rows:
        raise ValueError("rule alpha model requires at least one feature row")
    feature_dim = len(rows[0].features)
    threshold = _clamp(candidate.threshold, 0.50 if market_type == "futures" else 0.05, 0.95)
    short_threshold = 1.0 - threshold if market_type == "futures" else None
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
                feature_count=min(13, feature_dim),
                params={
                    "family": candidate.family,
                    "sensitivity": candidate.sensitivity,
                    "deadband": candidate.deadband,
                },
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


def optimize_rule_alpha_model_zoo(
    selection_rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    objective_name: str,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
    max_candidates: int = 48,
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
    for candidate in rule_alpha_candidates(objective.name, max_candidates=max_candidates):
        candidate_strategy = _candidate_strategy(strategy, candidate)
        for inverted in (False, True):
            model = model_for_rule_alpha(
                rows,
                candidate,
                strategy,
                market_type=market_type,
                probability_inverted=inverted,
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
            )
            diagnostics.append(candidate_result)
            if accepted and (best_accepted is None or candidate_result.score > best_accepted.score + 1e-12):
                best_accepted = candidate_result
            if best_diagnostic is None or _diagnostic_rank_key(candidate_result) > _diagnostic_rank_key(best_diagnostic):
                best_diagnostic = candidate_result

    winner = best_accepted or best_diagnostic
    if winner is None:
        return RuleAlphaOptimizationReport(False, None, float("-inf"), None, False, None, len(diagnostics), None, tuple(diagnostics))
    model = model_for_rule_alpha(
        rows,
        winner.candidate,
        strategy,
        market_type=market_type,
        probability_inverted=winner.probability_inverted,
    )
    model.rule_alpha_best_score = float(winner.score if winner.accepted else winner.raw_score)
    model.rule_alpha_best_pnl = float(winner.result.realized_pnl)
    model.rule_alpha_best_closed_trades = int(winner.result.closed_trades)
    model.rule_alpha_best_reject_reason = str(winner.reject_reason or "")
    model.rule_alpha_probability_inverted = bool(winner.probability_inverted)
    model.rule_alpha_evaluated_candidates = len(diagnostics)
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
    payload["stop_loss_pct"] = max(0.0005, float(strategy.stop_loss_pct) * max(0.05, candidate.stop_loss_multiplier))
    payload["take_profit_pct"] = max(0.0005, float(strategy.take_profit_pct) * max(0.05, candidate.take_profit_multiplier))
    payload["cooldown_minutes"] = max(0, int(round(float(strategy.cooldown_minutes) * max(0.0, candidate.cooldown_multiplier))))
    payload["min_position_hold_bars"] = max(0, int(candidate.min_position_hold_bars))
    payload["flat_signal_exit_grace_bars"] = max(0, int(candidate.flat_signal_exit_grace_bars))
    return StrategyConfig(**payload)


def _diagnostic_rank_key(result: RuleAlphaCandidateResult) -> tuple[float, float, float, float]:
    backtest = result.result
    return (
        float(result.raw_score),
        float(backtest.realized_pnl),
        float(backtest.edge_vs_buy_hold),
        float(backtest.closed_trades),
    )


def _clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return low
    return low if value < low else (high if value > high else value)


__all__ = [
    "RuleAlphaCandidate",
    "RuleAlphaCandidateResult",
    "RuleAlphaOptimizationReport",
    "model_for_rule_alpha",
    "optimize_rule_alpha_model_zoo",
    "rule_alpha_candidates",
]
