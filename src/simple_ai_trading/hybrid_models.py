"""Adaptive model-zoo experts layered on top of the trained base model.

The experts in this module are original implementations inspired by common
free/community day-trading model families: Lorentzian nearest-neighbor voting,
rational-quadratic kernel smoothing, and technical confluence controllers
similar in spirit to SuperTrend/MACD/Bollinger-style dashboards.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from statistics import median
from typing import Sequence

from .backtest import BacktestResult, run_backtest
from .features import ModelRow
from .model import HybridExpert, HybridPrototype, TrainedModel
from .objective import get_objective
from .types import StrategyConfig


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

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["base_result"] = asdict(self.base_result) if self.base_result is not None else None
        payload["best_result"] = asdict(self.best_result) if self.best_result is not None else None
        payload["ablation_results"] = [item.asdict() for item in self.ablation_results]
        return payload


@dataclass(frozen=True)
class _WeightProfile:
    name: str
    base: float
    lorentzian: float
    kernel: float
    technical: float


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


def _prototype_limit(objective_name: str) -> int:
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    if objective_name == "conservative":
        return 220
    if objective_name == "aggressive":
        return 520
    return 360


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
            _WeightProfile("guarded_neighbors", 0.68, 0.16, 0.11, 0.05),
            _WeightProfile("kernel_confirmation", 0.62, 0.12, 0.20, 0.06),
            _WeightProfile("technical_tiebreaker", 0.72, 0.10, 0.08, 0.10),
        )
    if objective_name == "aggressive":
        return (
            _WeightProfile("base_only", 1.00, 0.00, 0.00, 0.00),
            _WeightProfile("neighbor_momentum", 0.38, 0.34, 0.16, 0.12),
            _WeightProfile("kernel_regime", 0.34, 0.20, 0.30, 0.16),
            _WeightProfile("technical_breakout", 0.30, 0.24, 0.18, 0.28),
            _WeightProfile("balanced_aggressive", 0.36, 0.26, 0.22, 0.16),
        )
    return (
        _WeightProfile("base_only", 1.00, 0.00, 0.00, 0.00),
        _WeightProfile("balanced_neighbors", 0.50, 0.24, 0.17, 0.09),
        _WeightProfile("smooth_kernel", 0.48, 0.16, 0.26, 0.10),
        _WeightProfile("technical_blend", 0.46, 0.20, 0.18, 0.16),
    )


def _experts_for_profile(
    profile: _WeightProfile,
    prototypes: Sequence[HybridPrototype],
    *,
    feature_dim: int,
    feature_count: int,
    bandwidth: float,
    objective_name: str,
) -> list[HybridExpert]:
    objective_name = "aggressive" if objective_name == "risky" else objective_name
    k = 13 if objective_name == "conservative" else (31 if objective_name == "aggressive" else 21)
    return [
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
) -> tuple[float, BacktestResult | None]:
    if not rows:
        return float("-inf"), None
    objective = get_objective(objective_name)
    result = run_backtest(
        list(rows),
        model,
        strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
    if not objective.accepts(result):
        return float("-inf"), result
    return float(objective.score(result)), result


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
        score, _result = _evaluate_model(
            candidate if candidate.hybrid_experts else base_model,
            rows,
            strategy,
            objective_name=objective_name,
            market_type=market_type,
            starting_cash=starting_cash,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
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
    base_score, base_result = _evaluate_model(
        base_model,
        selection_rows,
        strategy,
        objective_name=objective_name,
        market_type=market_type,
        starting_cash=starting_cash,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
    prototypes = _build_prototypes(training_rows, base_model, objective_name)
    bandwidth = _estimate_bandwidth(prototypes, base_model.feature_dim)
    best_model = base_model
    best_score = base_score
    best_result = base_result
    best_profile = "base_only"
    evaluated = 0
    for profile in _profiles_for(objective_name):
        evaluated += 1
        candidate = copy.deepcopy(base_model)
        candidate.model_family = "adaptive_hybrid_model_zoo"
        candidate.hybrid_base_weight = max(0.0, float(profile.base))
        candidate.hybrid_experts = _experts_for_profile(
            profile,
            prototypes,
            feature_dim=base_model.feature_dim,
            feature_count=feature_count,
            bandwidth=bandwidth,
            objective_name=objective_name,
        )
        score, result = _evaluate_model(
            candidate,
            selection_rows,
            strategy,
            objective_name=objective_name,
            market_type=market_type,
            starting_cash=starting_cash,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
        )
        if score > best_score + 1e-12:
            best_model = candidate
            best_score = score
            best_result = result
            best_profile = profile.name
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
    )
