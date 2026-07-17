"""Multi-objective training orchestrator with process-pool parallelization.

For every registered objective (Conservative / Regular / Aggressive) the suite:

1. Expands candles into candidate-specific advanced feature vectors.
2. Seals a purged terminal suffix before any model or policy selection.
3. Splits development rows into fit/calibration, selection, and validation.
4. Evaluates a curated hyperparameter grid; each candidate is an independent,
   picklable unit of work dispatched through a ``ProcessPoolExecutor`` when
   more than one worker is available.
5. Applies walk-forward, multiple-trial, and exact final-model terminal gates.
6. Writes ``data/model_<objective>.json`` only after every gate passes.

Each worker process imports this package and calls :func:`_evaluate_candidate`
with a fully self-contained payload; there are no shared globals or closures in
the worker path.  Omitted compute backends resolve to GPU-first ``auto``; an
explicit CPU request or failed GPU probe records CPU fallback metadata on the
model artifact.  Tests keep the legacy
``runner=`` injection seam so they can stub out candidate evaluation without
spawning subprocesses.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable, Sequence

from .advanced_model import (
    AdvancedFeatureConfig,
    advanced_feature_dimension,
    advanced_feature_group_spans,
    advanced_feature_signature,
    default_config_for,
    make_advanced_rows,
    train_advanced,
)
from .api import Candle
from .backtest import (
    BacktestResult,
    _backtest_probabilities,
    calibrate_threshold_for_backtest,
    precompute_backtest_liquidity_adjustments,
    precompute_backtest_regime_scores,
    row_span_days,
    run_backtest,
    trade_activity_satisfies,
)
from .assets import DEFAULT_CONSERVATIVE_LEVERAGE, is_supported_major_symbol, normalize_symbol
from .features import ModelRow
from .execution_simulation import configured_round_trip_cost_floor_bps
from .hybrid_models import optimize_hybrid_model_zoo
from .market_edge import build_market_edge_report
from .meta_label import build_meta_label_report
from .model import (
    TrainedModel,
    calibrate_probability_temperature,
    calibrate_threshold,
    effective_training_backend_name,
    evaluate_classification,
    serialize_model,
)
from .objective import (
    ObjectiveSpec,
    ObjectiveTraining,
    available_objectives,
    get_objective,
    rank_candidates,
)
from .strategy_overrides import strategy_overrides_from_config
from .storage import write_json_atomic
from .terminal_holdout_ledger import (
    TerminalHoldoutLedger,
    reservation_evidence_passed,
    terminal_model_fingerprint,
    terminal_result_fingerprint,
)
from .types import StrategyConfig

_DEFAULT_OUTPUT_DIR = Path("data")
_ENSEMBLE_REFINEMENT_CANDIDATES = 3
_SELECTION_RISK_DISPERSION_FLOOR = 1e-4
_SELECTION_RISK_RELATIVE_FLOOR = 0.03
_PBO_MAX_PROBABILITY = 0.50
_PBO_MIN_CANDIDATES = 3
_PBO_EPSILON = 1e-6
_TERMINAL_HOLDOUT_RATIO = 0.20
_TERMINAL_HOLDOUT_SCHEMA = "terminal-holdout-v1"


# ==========================================================================
# Public dataclasses
# ==========================================================================


class TrainingSuiteRejected(ValueError):
    """Raised when every candidate for an objective fails its risk gates."""

    def __init__(self, message: str, *, row_count: int, diagnostics: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.row_count = int(row_count)
        self.diagnostics = diagnostics or {}


@dataclass
class CandidateParams:
    """One grid point evaluated during the suite's per-objective search."""

    epochs: int
    learning_rate: float
    l2_penalty: float
    signal_threshold: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_per_trade: float
    confidence_beta: float = 0.85
    label_threshold_multiplier: float = 1.0
    label_lookahead_multiplier: float = 1.0
    label_mode: str = "forward_return"
    focal_gamma: float = 0.0
    seed: int = 7

    def asdict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass
class ObjectiveOutcome:
    """Summary of the training run that was picked for one objective."""

    objective: str
    model_path: Path
    feature_dim: int
    feature_signature: str
    best_score: float
    best_params: dict[str, float | int | str]
    explored_candidates: int
    rejected_candidates: int
    epochs: int
    learning_rate: float
    l2_penalty: float
    row_count: int
    positive_rate: float
    decision_threshold: float | None = None
    threshold_source: str | None = None
    calibration_score: float | None = None
    calibration_rows: int = 0
    validation_rows: int = 0
    validation_score: float | None = None
    full_sample_score: float | None = None
    walk_forward_gate: dict[str, object] | None = None
    ensemble_refined: bool = False
    ensemble_refinement_candidates: int = 0
    local_refinement_candidates: int = 0
    training_backend_requested: str = "cpu"
    training_backend_kind: str = "cpu"
    training_backend_device: str = "cpu"
    hybrid_model: bool = False
    hybrid_profile: str = "base_only"
    hybrid_base_score: float | None = None
    hybrid_best_score: float | None = None
    hybrid_evaluated_profiles: int = 0
    hybrid_ablation: list[dict[str, object]] = field(default_factory=list)
    hybrid_rescue: bool = False
    hybrid_rescue_candidates: int = 0
    feature_ablation: list[dict[str, object]] = field(default_factory=list)
    meta_label_report: dict[str, object] | None = None
    selection_risk: dict[str, object] | None = None

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_path"] = str(self.model_path)
        return payload


@dataclass
class SuiteReport:
    """End-to-end summary written after the suite finishes."""

    outcomes: list[ObjectiveOutcome]
    total_rows: int
    total_candles: int
    output_dir: Path
    summary_path: Path
    objectives_run: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, object]:
        return {
            "outcomes": [o.asdict() for o in self.outcomes],
            "total_rows": self.total_rows,
            "total_candles": self.total_candles,
            "output_dir": str(self.output_dir),
            "summary_path": str(self.summary_path),
            "objectives_run": list(self.objectives_run),
        }


# ==========================================================================
# Helpers
# ==========================================================================


def _strategy_for_candidate(base: StrategyConfig, candidate: CandidateParams,
                            training: ObjectiveTraining) -> StrategyConfig:
    """Overlay a candidate's tunables on top of the base strategy config."""

    return replace(
        base,
        leverage=training.leverage,
        risk_per_trade=candidate.risk_per_trade,
        max_position_pct=training.max_position_pct,
        stop_loss_pct=candidate.stop_loss_pct,
        take_profit_pct=candidate.take_profit_pct,
        signal_threshold=candidate.signal_threshold,
        cooldown_minutes=training.cooldown_minutes,
        max_trades_per_day=training.max_trades_per_day,
        training_epochs=candidate.epochs,
        confidence_beta=candidate.confidence_beta,
    )


def _attach_strategy_overrides(model: TrainedModel, strategy: StrategyConfig) -> TrainedModel:
    """Persist execution parameters selected alongside the model weights."""

    model.strategy_overrides = strategy_overrides_from_config(strategy)
    return model


def _minimum_cost_aware_label_threshold(strategy: StrategyConfig) -> float:
    cost_floor = configured_round_trip_cost_floor_bps(strategy) / 10_000.0
    return math.nextafter(cost_floor, math.inf) if cost_floor > 0.0 else 0.0


def _cost_aware_feature_config(
    config: AdvancedFeatureConfig,
    strategy: StrategyConfig,
) -> AdvancedFeatureConfig:
    """Raise gross label barriers above the configured round-trip cost floor."""

    strict_floor = _minimum_cost_aware_label_threshold(strategy)
    stop_threshold = (
        float(config.label_stop_threshold)
        if config.label_stop_threshold is not None
        else float(config.label_threshold)
    )
    return replace(
        config,
        label_threshold=max(strict_floor, float(config.label_threshold)),
        label_stop_threshold=max(strict_floor, stop_threshold),
    )


def _feature_config_for_candidate(
    base: AdvancedFeatureConfig,
    candidate: CandidateParams,
    *,
    strategy: StrategyConfig | None = None,
) -> AdvancedFeatureConfig:
    """Apply candidate label target/horizon multipliers to a feature config."""

    effective_base = _cost_aware_feature_config(base, strategy) if strategy is not None else base
    cost_floor = (
        _minimum_cost_aware_label_threshold(strategy)
        if strategy is not None
        else 0.0
    )
    threshold_multiplier = max(0.10, min(5.0, float(candidate.label_threshold_multiplier)))
    lookahead_multiplier = max(0.25, min(4.0, float(candidate.label_lookahead_multiplier)))
    label_mode = str(candidate.label_mode or "forward_return")
    normalized_label_mode = label_mode.strip().lower().replace("-", "_")
    volatility_window = max(0, int(effective_base.label_volatility_window))
    volatility_multiplier = max(0.0, float(effective_base.label_volatility_multiplier))
    if "event" in normalized_label_mode or "volatility" in normalized_label_mode:
        derived_window = max(6, int(round(float(effective_base.label_lookahead) * lookahead_multiplier * 3.0)))
        volatility_window = max(volatility_window, derived_window)
        if volatility_multiplier <= 0.0:
            volatility_multiplier = 0.50
    return replace(
        effective_base,
        label_threshold=max(
            0.00005,
            cost_floor,
            float(effective_base.label_threshold) * threshold_multiplier,
        ),
        label_lookahead=max(1, int(round(float(effective_base.label_lookahead) * lookahead_multiplier))),
        label_mode=label_mode,
        label_stop_threshold=max(
            0.00005,
            cost_floor,
            float(effective_base.label_threshold) * threshold_multiplier,
        ),
        label_volatility_window=volatility_window,
        label_volatility_multiplier=volatility_multiplier,
    )


def _rows_for_candidate(
    candles: Sequence[Candle] | None,
    base_rows: Sequence[ModelRow],
    base_feature_cfg: AdvancedFeatureConfig,
    candidate: CandidateParams,
    *,
    strategy: StrategyConfig | None = None,
    compute_backend: str | None = None,
) -> tuple[list[ModelRow], AdvancedFeatureConfig]:
    """Return candidate-specific rows while preserving the legacy row payload path."""

    feature_cfg = _feature_config_for_candidate(
        base_feature_cfg,
        candidate,
        strategy=strategy,
    )
    if candles is None:
        return list(base_rows), feature_cfg
    return make_advanced_rows(candles, feature_cfg, compute_backend=compute_backend), feature_cfg


def _candidate_grid(training: ObjectiveTraining) -> list[CandidateParams]:
    """Curated first-pass grid without exploding runtime.

    Three epoch budgets, two learning rates, two L2 penalties, four thresholds,
    one stop/take profile, two risk levels, three confidence shrinkage levels,
    ten label/loss profiles, and one seed.
    The suite then searches locally around the winner and checks seed ensembles
    for the best candidates, keeping GPU runs finishable while still deduping
    arithmetic collisions.
    """

    epoch_options = (
        max(80, int(training.epochs * 0.20)),
        max(120, int(training.epochs * 0.50)),
        training.epochs,
    )
    lr_options = (training.learning_rate * 0.75, training.learning_rate)
    l2_options = (training.l2_penalty, training.l2_penalty * 3.0)
    threshold_options = (
        training.signal_threshold - 0.08,
        training.signal_threshold - 0.03,
        training.signal_threshold,
        training.signal_threshold + 0.03,
    )
    stop_take_options = ((training.stop_loss_pct, training.take_profit_pct),)
    risk_options = (training.risk_per_trade * 0.50, training.risk_per_trade)
    confidence_options = (0.70, 0.85, 1.0)
    label_profile_options = (
        # Directional baseline, high-frequency scalp, and high-conviction longer lookahead.
        (1.0, 1.0, "forward_return", 0.0),
        (0.10, 0.25, "forward_return", 1.5),
        (1.40, 1.75, "forward_return", 0.0),
        # Path-aware labels benchmark whether stop/take ordering beats point-to-point return.
        (1.0, 1.0, "triple_barrier", 0.0),
        (0.10, 0.25, "triple_barrier", 1.5),
        (0.75, 0.75, "triple_barrier", 1.0),
        # CUSUM-gated volatility barriers avoid training on low-information bars.
        (0.10, 0.25, "event_volatility_triple_barrier", 1.5),
        (0.75, 0.75, "event_volatility_triple_barrier", 1.0),
        # Futures-capable short-side labels are first-class candidates, not fallback inversions.
        (0.10, 0.25, "downside_forward_return", 1.5),
        (0.10, 0.25, "downside_event_volatility_triple_barrier", 2.0),
    )
    seed_options = (7,)

    candidates: list[CandidateParams] = []
    for epochs, lr, l2, thr, stop_take, risk, confidence, label_profile, seed in product(
        epoch_options, lr_options, l2_options, threshold_options,
        stop_take_options, risk_options, confidence_options, label_profile_options, seed_options,
    ):
        stop, take = stop_take
        label_threshold_multiplier, label_lookahead_multiplier, label_mode, focal_gamma = label_profile
        candidates.append(CandidateParams(
            epochs=int(epochs),
            learning_rate=float(lr),
            l2_penalty=float(l2),
            signal_threshold=max(0.05, min(0.95, float(thr))),
            stop_loss_pct=max(0.001, float(stop)),
            take_profit_pct=max(0.001, float(take)),
            risk_per_trade=max(0.0005, min(0.05, float(risk))),
            confidence_beta=max(0.0, min(1.0, float(confidence))),
            label_threshold_multiplier=max(0.10, min(5.0, float(label_threshold_multiplier))),
            label_lookahead_multiplier=max(0.25, min(4.0, float(label_lookahead_multiplier))),
            label_mode=str(label_mode),
            focal_gamma=max(0.0, min(4.0, float(focal_gamma))),
            seed=int(seed),
        ))
    # Deduplicate collisions produced by the arithmetic above.
    seen: set[tuple[float, ...]] = set()
    unique: list[CandidateParams] = []
    for entry in candidates:
        key = tuple(entry.asdict().values())
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _calibration_split(
    rows: Sequence[ModelRow],
    *,
    ratio: float = 0.20,
    purge_gap: int = 0,
) -> tuple[list[ModelRow], list[ModelRow]]:
    ordered = list(rows)
    if len(ordered) < 30:
        return ordered, []
    calibration_size = max(8, int(len(ordered) * max(0.05, min(0.40, ratio))))
    calibration_size = min(calibration_size, len(ordered) - 10)
    calibration_start = len(ordered) - calibration_size
    fit_end = calibration_start - max(0, int(purge_gap))
    if fit_end < 10:
        return [], ordered[calibration_start:]
    return ordered[:fit_end], ordered[calibration_start:]


def _threshold_guard(baseline: object, candidate: object) -> bool:
    baseline_accuracy = float(getattr(baseline, "accuracy", 0.0))
    baseline_f1 = float(getattr(baseline, "f1", 0.0))
    baseline_precision = float(getattr(baseline, "precision", 0.0))
    candidate_accuracy = float(getattr(candidate, "accuracy", 0.0))
    candidate_f1 = float(getattr(candidate, "f1", 0.0))
    candidate_precision = float(getattr(candidate, "precision", 0.0))
    candidate_true_positive = int(getattr(candidate, "true_positive", 0))
    candidate_false_negative = int(getattr(candidate, "false_negative", 0))
    if candidate_false_negative > 0 and candidate_true_positive <= 0:
        return False
    stable_f1 = candidate_accuracy + 0.03 >= baseline_accuracy and candidate_f1 + 0.05 >= baseline_f1
    sharper_precision = candidate_accuracy >= baseline_accuracy + 0.02 and candidate_precision + 0.02 >= baseline_precision
    return bool(stable_f1 or sharper_precision)


def _effective_threshold_for_market(threshold: float, market_type: str) -> float:
    value = max(0.05, min(0.95, float(threshold)))
    if str(market_type).lower() == "futures":
        value = max(0.5, value)
    return value


def _calibrate_candidate_threshold(
    model: TrainedModel,
    rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
    min_closed_trades: int = 1,
    min_trades_per_day: float = 0.0,
) -> tuple[float, str, float | None]:
    if not rows:
        return _effective_threshold_for_market(strategy.signal_threshold, market_type), "strategy", None
    strategy_threshold = _effective_threshold_for_market(strategy.signal_threshold, market_type)
    classification_threshold = calibrate_threshold(list(rows), model, start=0.05, end=0.95, steps=61)
    baseline_report = evaluate_classification(list(rows), model, threshold=strategy_threshold)
    profit_report = calibrate_threshold_for_backtest(
        list(rows),
        model,
        strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        baseline_threshold=strategy_threshold,
        start=0.05,
        end=0.95,
        steps=121,
        min_closed_trades=min_closed_trades,
        min_trades_per_day=min_trades_per_day,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
    candidate_report = evaluate_classification(list(rows), model, threshold=profit_report.best_threshold)
    profit_backed = (
        profit_report.accepted
        and profit_report.realized_pnl > 0.0
        and profit_report.closed_trades > 0
        and profit_report.score > 0.0
    )
    if profit_backed and _threshold_guard(baseline_report, candidate_report):
        return _effective_threshold_for_market(profit_report.threshold, market_type), "profit_backtest", float(profit_report.score)
    classification_report = evaluate_classification(list(rows), model, threshold=classification_threshold)
    if (
        abs(float(classification_threshold) - strategy_threshold) > 1e-12
        and profit_report.baseline_score >= 0.0
        and _threshold_guard(baseline_report, classification_report)
    ):
        return _effective_threshold_for_market(classification_threshold, market_type), "classification_f1", float(profit_report.baseline_score)
    return strategy_threshold, "strategy", float(profit_report.baseline_score)


def _refine_threshold_on_selection_rows(
    model: TrainedModel,
    rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
    min_closed_trades: int = 1,
    min_trades_per_day: float = 0.0,
    target_trades_per_day: float = 0.0,
) -> tuple[float, str, float] | None:
    """Promote a selection-set profit threshold only when it remains directional.

    The final chronological holdout is deliberately excluded from this helper.
    It can be used to tune the threshold carried into selection, but not to
    adjust the artifact after final holdout scoring.
    """

    if len(rows) < 30:
        return None
    row_list = list(rows)
    span_days = row_span_days(row_list)
    baseline_threshold = _effective_threshold_for_market(
        float(model.decision_threshold if model.decision_threshold is not None else strategy.signal_threshold),
        market_type,
    )
    baseline_report = evaluate_classification(row_list, model, threshold=baseline_threshold)
    current_threshold = model.decision_threshold
    best_threshold: float | None = None
    best_score = float("-inf")
    best_rank = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    probabilities, score_backend = _backtest_probabilities(
        row_list,
        model,
        compute_backend=compute_backend,
        batch_size=score_batch_size,
    )
    regime_scores = precompute_backtest_regime_scores(row_list, strategy)
    liquidity_adjustments = precompute_backtest_liquidity_adjustments(row_list, strategy)
    try:
        threshold_start = 0.5 if str(market_type).lower() == "futures" else 0.05
        for threshold in _threshold_values(threshold_start, 0.95, 121, baseline_threshold):
            threshold = _effective_threshold_for_market(threshold, market_type)
            model.decision_threshold = float(threshold)
            result = run_backtest(
                row_list,
                model,
                strategy,
                starting_cash=starting_cash,
                market_type=market_type,
                compute_backend=compute_backend,
                score_batch_size=score_batch_size,
                precomputed_probabilities=probabilities,
                precomputed_score_backend=score_backend,
                precomputed_regime_scores=regime_scores,
                precomputed_liquidity_adjustments=liquidity_adjustments,
            )
            if (
                result.realized_pnl <= 0.0
                or not trade_activity_satisfies(
                    result,
                    min_closed_trades=max(1, int(min_closed_trades)),
                    min_trades_per_day=max(0.0, float(min_trades_per_day)),
                    duration_days=span_days,
                )
            ):
                continue
            trades_per_day = float(result.closed_trades) / max(1.0, span_days)
            candidate_report = evaluate_classification(row_list, model, threshold=threshold)
            if not _threshold_guard(baseline_report, candidate_report):
                continue
            rank = (
                float(result.realized_pnl),
                float(candidate_report.f1),
                -float(result.max_drawdown),
                min(trades_per_day, max(0.0, float(target_trades_per_day)) or trades_per_day),
            )
            if rank > best_rank:
                best_rank = rank
                best_score = float(result.realized_pnl)
                best_threshold = float(threshold)
    finally:
        model.decision_threshold = current_threshold
    if best_threshold is None:
        return None
    return best_threshold, "selection_profit_backtest", best_score


def _refine_threshold_on_validation_rows(
    model: TrainedModel,
    rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
) -> tuple[float, str, float] | None:
    """Compatibility wrapper for older internal tests/callers."""

    return _refine_threshold_on_selection_rows(
        model,
        rows,
        strategy,
        market_type=market_type,
        starting_cash=starting_cash,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )


def _threshold_values(start: float, end: float, steps: int, baseline: float) -> list[float]:
    if steps <= 1:
        return [max(0.0, min(1.0, float(baseline)))]
    low = max(0.0, min(1.0, float(start)))
    high = max(0.0, min(1.0, float(end)))
    if high <= low:
        high = min(1.0, low + 0.01)
    values = [low + (high - low) * i / (steps - 1) for i in range(steps)]
    values.append(max(0.0, min(1.0, float(baseline))))
    return sorted(set(round(value, 12) for value in values))


def _label_purge_gap(feature_cfg: AdvancedFeatureConfig) -> int:
    return max(1, int(feature_cfg.label_lookahead)) + max(
        0,
        int(getattr(feature_cfg, "label_entry_offset", 0) or 0),
    )


def _walk_forward_split(
    rows: Sequence[ModelRow],
    *,
    eval_ratio: float = 0.25,
    purge_gap: int = 0,
) -> tuple[list[ModelRow], list[ModelRow]]:
    if len(rows) < 10:
        return list(rows), list(rows)
    split = int(len(rows) * (1.0 - eval_ratio))
    split = max(5, min(len(rows) - 5, split))
    train_end = split - max(0, int(purge_gap))
    if train_end < 5:
        return [], list(rows[split:])
    return list(rows[:train_end]), list(rows[split:])


def _reserve_terminal_holdout(
    rows: Sequence[ModelRow],
    *,
    purge_gap: int,
) -> tuple[list[ModelRow], list[ModelRow]]:
    """Seal the terminal suffix before any candidate or policy selection."""

    ordered = list(rows)
    if any(
        int(current.timestamp) <= int(previous.timestamp)
        for previous, current in zip(ordered, ordered[1:])
    ):
        raise ValueError("terminal holdout rows must be strictly chronological")
    if len(ordered) < 10:
        # The generic walk-forward helper mirrors tiny inputs on both sides.
        # A terminal holdout must instead fail closed without any row overlap.
        return [], ordered
    return _walk_forward_split(
        ordered,
        eval_ratio=_TERMINAL_HOLDOUT_RATIO,
        purge_gap=purge_gap,
    )


def _purged_walk_forward_splits(
    rows: Sequence[ModelRow],
    training: ObjectiveTraining,
    feature_cfg: AdvancedFeatureConfig,
) -> list[dict[str, object]]:
    ordered = list(rows)
    row_count = len(ordered)
    if row_count < 320:
        return []
    purge_gap = _label_purge_gap(feature_cfg)
    train_window = min(max(80, int(training.walk_forward_train)), max(80, int(row_count * 0.60)))
    remaining = row_count - train_window - purge_gap
    if remaining < 40:
        train_window = max(80, int(row_count * 0.55))
        remaining = row_count - train_window - purge_gap
    test_window = min(max(30, int(training.walk_forward_test)), max(30, int(remaining * 0.50)))
    step = min(max(1, int(training.walk_forward_step)), max(1, test_window))
    folds: list[dict[str, object]] = []
    start = 0
    fold_index = 0
    while start + train_window + purge_gap + test_window <= row_count:
        train_start = start
        train_end = start + train_window
        test_start = train_end + purge_gap
        test_end = test_start + test_window
        folds.append({
            "index": fold_index,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "purge_gap": purge_gap,
            "train_rows": ordered[train_start:train_end],
            "test_rows": ordered[test_start:test_end],
        })
        fold_index += 1
        start += step
    return folds


def _rejection_reasons(objective: ObjectiveSpec, result: BacktestResult) -> list[str]:
    reasons = objective.rejection_reasons(result) if hasattr(objective, "rejection_reasons") else []
    if result.realized_pnl <= 0.0 and "realized_pnl<=0.0" not in reasons:
        reasons.append("realized_pnl<=0.0")
    if result.stopped_by_drawdown and "stopped_by_drawdown" not in reasons:
        reasons.append("stopped_by_drawdown")
    if (
        bool(getattr(result, "stopped_by_liquidation", False))
        or int(getattr(result, "liquidation_events", 0)) > 0
    ) and "liquidation_events>0" not in reasons:
        reasons.append("liquidation_events>0")
    return reasons


def _gate_result_payload(result: BacktestResult, objective: ObjectiveSpec | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "realized_pnl": float(result.realized_pnl),
        "max_drawdown": float(result.max_drawdown),
        "closed_trades": int(result.closed_trades),
        "win_rate": float(result.win_rate),
        "total_fees": float(result.total_fees),
        "edge_vs_buy_hold": float(result.edge_vs_buy_hold),
        "stopped_by_drawdown": bool(result.stopped_by_drawdown),
        "stopped_by_liquidation": bool(getattr(result, "stopped_by_liquidation", False)),
        "liquidation_events": int(getattr(result, "liquidation_events", 0)),
        "liquidation_loss": float(getattr(result, "liquidation_loss", 0.0)),
    }
    if objective is not None:
        reasons = _rejection_reasons(objective, result)
        payload["accepted"] = not reasons
        payload["reject_reasons"] = reasons
        payload["reject_reason"] = "; ".join(reasons) if reasons else None
        payload["market_edge"] = build_market_edge_report(result, objective).asdict()
    return payload


def _finite_number_or_none(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _model_rows_fingerprint(rows: Sequence[ModelRow]) -> str:
    digest = hashlib.sha256(_TERMINAL_HOLDOUT_SCHEMA.encode("ascii"))
    feature_format = ""
    feature_count = -1
    for row in rows:
        features = tuple(float(value) for value in row.features)
        if len(features) != feature_count:
            feature_count = len(features)
            feature_format = f"<{feature_count}d"
        digest.update(struct.pack("<qdqI", int(row.timestamp), float(row.close), int(row.label), feature_count))
        if features:
            digest.update(struct.pack(feature_format, *features))
    return digest.hexdigest()


def _terminal_holdout_gate(
    rows: Sequence[ModelRow],
    model: TrainedModel,
    strategy: StrategyConfig,
    objective: ObjectiveSpec,
    *,
    market_type: str,
    starting_cash: float,
    purge_gap: int,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
) -> tuple[float, dict[str, object]]:
    terminal_rows = list(rows)
    if not terminal_rows:
        return float("-inf"), {
            "schema_version": _TERMINAL_HOLDOUT_SCHEMA,
            "passed": False,
            "reason": "terminal_holdout_missing",
            "evaluation_count": 0,
            "rows": 0,
            "purge_gap": int(purge_gap),
            "dataset_fingerprint": "",
        }
    try:
        result = run_backtest(
            terminal_rows,
            model,
            strategy,
            starting_cash=starting_cash,
            market_type=market_type,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
        )
    except Exception as exc:
        return float("-inf"), {
            "schema_version": _TERMINAL_HOLDOUT_SCHEMA,
            "passed": False,
            "reason": "terminal_holdout_evaluation_error",
            "error": f"{exc.__class__.__name__}: {exc}",
            "evaluation_count": 1,
            "rows": len(terminal_rows),
            "start_timestamp": int(terminal_rows[0].timestamp),
            "end_timestamp": int(terminal_rows[-1].timestamp),
            "purge_gap": int(purge_gap),
            "dataset_fingerprint": _model_rows_fingerprint(terminal_rows),
        }
    reasons = _rejection_reasons(objective, result)
    raw_score = objective.score(result) if not reasons else float("-inf")
    if not math.isfinite(raw_score) or raw_score <= 0.0:
        reasons.append("terminal_objective_score<=0")
    passed = not reasons
    score = float(raw_score) if passed else float("-inf")
    return float(score), {
        "schema_version": _TERMINAL_HOLDOUT_SCHEMA,
        "passed": bool(passed),
        "reason": None if passed else "; ".join(dict.fromkeys(reasons)),
        "evaluation_count": 1,
        "rows": len(terminal_rows),
        "start_timestamp": int(terminal_rows[0].timestamp),
        "end_timestamp": int(terminal_rows[-1].timestamp),
        "purge_gap": int(purge_gap),
        "dataset_fingerprint": _model_rows_fingerprint(terminal_rows),
        "model_family": str(getattr(model, "model_family", "")),
        "probability_inverted": bool(getattr(model, "probability_inverted", False)),
        "hybrid_profile": str(getattr(model, "hybrid_profile", "base_only") or "base_only"),
        "meta_label_enabled": bool(
            isinstance(getattr(model, "meta_label_policy", None), dict)
            and getattr(model, "meta_label_policy", {}).get("enabled") is True
        ),
        "score": _finite_number_or_none(score),
        "result": _gate_result_payload(result, objective),
    }


def _payload_float(payload: object, key: str, default: float = 0.0) -> float:
    if not isinstance(payload, dict):
        return default
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def _rejected_candidate_quality(entry: dict[str, Any]) -> float:
    """Rank rejected candidates for refinement without relaxing hard gates."""

    quality = 0.0
    for weight, payload_name in ((0.45, "validation_result"), (0.55, "full_sample_result")):
        payload = entry.get(payload_name)
        realized = _payload_float(payload, "realized_pnl")
        edge = _payload_float(payload, "edge_vs_buy_hold")
        drawdown = max(0.0, _payload_float(payload, "max_drawdown"))
        trades = max(0.0, _payload_float(payload, "closed_trades"))
        win_rate = max(0.0, min(1.0, _payload_float(payload, "win_rate")))
        quality += weight * (
            realized
            + 0.25 * edge
            + 0.05 * min(trades, 30.0)
            + 2.0 * win_rate
            - 100.0 * drawdown
        )
    return float(quality)


def _candidate_rank_key(entry: dict[str, Any]) -> tuple[int, float]:
    score = _finite_number_or_none(entry.get("score"))
    if score is not None:
        return 1, score
    return 0, _rejected_candidate_quality(entry)


def _candidate_risk_snapshot(entry: dict[str, Any]) -> tuple[float, float, float]:
    payloads = [
        item
        for item in (entry.get("validation_result"), entry.get("full_sample_result"))
        if isinstance(item, dict)
    ]
    if not payloads:
        return 1.0, 0.0, 0.0
    max_drawdown = max(max(0.0, _payload_float(payload, "max_drawdown", 1.0)) for payload in payloads)
    min_pnl = min(_payload_float(payload, "realized_pnl", 0.0) for payload in payloads)
    min_edge = min(_payload_float(payload, "edge_vs_buy_hold", 0.0) for payload in payloads)
    return float(max_drawdown), float(min_pnl), float(min_edge)


def _risk_non_degrading(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    """Return True only when a higher-score refinement does not materially worsen risk."""

    candidate_drawdown, candidate_pnl, candidate_edge = _candidate_risk_snapshot(candidate)
    incumbent_drawdown, incumbent_pnl, incumbent_edge = _candidate_risk_snapshot(incumbent)
    drawdown_tolerance = max(0.0025, incumbent_drawdown * 0.05)
    pnl_tolerance = max(1e-6, abs(incumbent_pnl) * 0.05)
    edge_tolerance = max(1e-6, abs(incumbent_edge) * 0.05)
    return bool(
        candidate_drawdown <= incumbent_drawdown + drawdown_tolerance
        and candidate_pnl + pnl_tolerance >= incumbent_pnl
        and candidate_edge + edge_tolerance >= incumbent_edge
    )


def _risk_aware_best(
    incumbent: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Promote a score-improving candidate only if its risk evidence is not worse."""

    best = incumbent
    best_score = _finite_number_or_none(best.get("score"))
    for candidate in sorted(candidates, key=_candidate_rank_key, reverse=True):
        candidate_score = _finite_number_or_none(candidate.get("score"))
        if candidate_score is None or best_score is None:
            continue
        if candidate_score > best_score + 1e-12 and _risk_non_degrading(candidate, best):
            best = candidate
            best_score = candidate_score
    return best


def _score_quantile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = max(0.0, min(1.0, float(q))) * (len(clean) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return clean[low]
    fraction = position - low
    return clean[low] + (clean[high] - clean[low]) * fraction


def _score_rank_percentile(values: Sequence[float], index: int) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) != len(values) or not 0 <= int(index) < len(clean):
        return None
    if len(clean) == 1:
        return 1.0
    value = clean[int(index)]
    lower = sum(1 for item in clean if item < value)
    equal = sum(1 for item in clean if item == value)
    tie_adjusted_rank = lower + (equal - 1) / 2.0
    return max(0.0, min(1.0, tie_adjusted_rank / (len(clean) - 1)))


def _pbo_logit(rank_percentile: float) -> float:
    clipped = max(_PBO_EPSILON, min(1.0 - _PBO_EPSILON, float(rank_percentile)))
    return math.log(clipped / (1.0 - clipped))


def _mean_panel_score(entry: dict[str, Any], panel_names: Sequence[str]) -> float | None:
    values: list[float] = []
    for panel in panel_names:
        value = _finite_number_or_none(entry.get(f"{panel}_score"))
        if value is None:
            return None
        values.append(float(value))
    return sum(values) / len(values) if values else None


def _overfit_diagnostics(best: dict[str, Any], ranked_pool: Sequence[dict[str, Any]]) -> dict[str, object]:
    """Return a compact CSCV/PBO-style diagnostic over selection/validation panels.

    This is deliberately conservative in naming: with the current artifacts all
    candidates expose at most one selection panel and one holdout validation
    panel, so this is a two-panel CSCV proxy rather than full CPCV over many
    purged folds. It still catches the dangerous case where the selected
    in-sample winner ranks poorly out-of-sample in both symmetric views.
    """

    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for entry in [*ranked_pool, best]:
        entry_id = id(entry)
        if entry_id in seen:
            continue
        seen.add(entry_id)
        if (
            _finite_number_or_none(entry.get("selection_score")) is not None
            and _finite_number_or_none(entry.get("validation_score")) is not None
        ):
            entries.append(entry)

    if len(entries) < _PBO_MIN_CANDIDATES:
        return {
            "status": "skipped",
            "reason": "requires_selection_and_validation_scores",
            "method": "two_panel_cscv_proxy",
            "candidate_count": len(entries),
            "min_candidates": _PBO_MIN_CANDIDATES,
            "passed": False,
        }

    split_specs = (
        (("selection",), ("validation",)),
        (("validation",), ("selection",)),
    )
    split_reports: list[dict[str, object]] = []
    overfit_splits = 0
    degradation_values: list[float] = []
    for in_sample_panels, out_sample_panels in split_specs:
        in_scores = [_mean_panel_score(entry, in_sample_panels) for entry in entries]
        out_scores = [_mean_panel_score(entry, out_sample_panels) for entry in entries]
        if any(score is None for score in in_scores) or any(score is None for score in out_scores):
            continue
        in_values = [float(score) for score in in_scores if score is not None]
        out_values = [float(score) for score in out_scores if score is not None]
        winner_index = max(range(len(in_values)), key=lambda index: (in_values[index], -index))
        rank_percentile = _score_rank_percentile(out_values, winner_index)
        if rank_percentile is None:
            continue
        logit_rank = _pbo_logit(rank_percentile)
        degradation = out_values[winner_index] - in_values[winner_index]
        degradation_values.append(float(degradation))
        overfit = bool(logit_rank < 0.0)
        if overfit:
            overfit_splits += 1
        split_reports.append({
            "in_sample_panels": list(in_sample_panels),
            "out_sample_panels": list(out_sample_panels),
            "winner_index": int(winner_index),
            "winner_in_sample_score": float(in_values[winner_index]),
            "winner_out_sample_score": float(out_values[winner_index]),
            "winner_out_sample_rank_percentile": float(rank_percentile),
            "logit_rank": float(logit_rank),
            "performance_degradation": float(degradation),
            "overfit": overfit,
        })

    if not split_reports:
        return {
            "status": "skipped",
            "reason": "no_comparable_cscv_splits",
            "method": "two_panel_cscv_proxy",
            "candidate_count": len(entries),
            "min_candidates": _PBO_MIN_CANDIDATES,
            "passed": True,
        }

    selected_index = next((index for index, entry in enumerate(entries) if id(entry) == id(best)), None)
    selected_validation_percentile = None
    if selected_index is not None:
        validation_scores = [float(_finite_number_or_none(entry.get("validation_score")) or 0.0) for entry in entries]
        selected_validation_percentile = _score_rank_percentile(validation_scores, selected_index)
    probability = overfit_splits / len(split_reports)
    passed = probability <= _PBO_MAX_PROBABILITY
    return {
        "status": "available",
        "method": "two_panel_cscv_proxy",
        "passed": bool(passed),
        "reason": None if passed else "selection_risk_pbo>0.50",
        "probability_backtest_overfit": float(probability),
        "max_probability_backtest_overfit": float(_PBO_MAX_PROBABILITY),
        "candidate_count": len(entries),
        "split_count": len(split_reports),
        "overfit_splits": int(overfit_splits),
        "mean_performance_degradation": (
            float(sum(degradation_values) / len(degradation_values))
            if degradation_values
            else None
        ),
        "selected_validation_rank_percentile": (
            float(selected_validation_percentile)
            if selected_validation_percentile is not None
            else None
        ),
        "splits": split_reports,
    }


def _selection_risk_report(
    best: dict[str, Any],
    ranked_pool: Sequence[dict[str, Any]],
    *,
    base_candidate_count: int,
    local_refinement_candidates: int,
    ensemble_refinement_candidates: int,
    hybrid_rescue_candidates: int,
    hybrid_profile_trials: int = 0,
) -> dict[str, object]:
    """Estimate whether the selected score survives the number of tried models.

    This is a deterministic multiple-trials haircut inspired by PBO/Deflated
    Sharpe discipline. It is intentionally scale-light: use observed score
    dispersion when available, otherwise use a small relative/fixed floor.
    """

    best_score = _finite_number_or_none(best.get("score"))
    entry_ids = {id(entry) for entry in ranked_pool}
    entries = list(ranked_pool)
    if id(best) not in entry_ids:
        entries.append(best)
    internal_variants_evaluated = sum(
        max(1, int(_finite_number_or_none(entry.get("internal_variant_count")) or 1))
        for entry in ranked_pool
    )
    internal_variant_extra_trials = max(0, internal_variants_evaluated - len(ranked_pool))
    scores = [
        score
        for entry in entries
        if (score := _finite_number_or_none(entry.get("score"))) is not None
    ]
    effective_trials = max(
        1,
        int(base_candidate_count)
        + int(local_refinement_candidates)
        + int(ensemble_refinement_candidates)
        + int(hybrid_rescue_candidates)
        + int(hybrid_profile_trials)
        + internal_variant_extra_trials,
        len(entries) + internal_variant_extra_trials,
    )
    if best_score is None:
        overfit = _overfit_diagnostics(best, entries)
        return {
            "passed": False,
            "reason": "no_finite_selected_score",
            "reasons": ["no_finite_selected_score"],
            "effective_trials": effective_trials,
            "base_candidates": int(base_candidate_count),
            "local_refinement_candidates": int(local_refinement_candidates),
            "ensemble_refinement_candidates": int(ensemble_refinement_candidates),
            "hybrid_rescue_candidates": int(hybrid_rescue_candidates),
            "hybrid_profile_trials": int(hybrid_profile_trials),
            "internal_variants_evaluated": int(internal_variants_evaluated),
            "internal_variant_extra_trials": int(internal_variant_extra_trials),
            "finite_candidate_scores": len(scores),
            "overfit_diagnostics": overfit,
        }

    sorted_desc = sorted(scores, reverse=True)
    runner_up = sorted_desc[1] if len(sorted_desc) > 1 else None
    q1 = _score_quantile(scores, 0.25)
    median = _score_quantile(scores, 0.50)
    q3 = _score_quantile(scores, 0.75)
    iqr = max(0.0, float(q3 - q1)) if q1 is not None and q3 is not None else 0.0
    dispersion = max(
        iqr,
        abs(float(best_score)) * _SELECTION_RISK_RELATIVE_FLOOR,
        _SELECTION_RISK_DISPERSION_FLOOR,
    )
    trial_penalty = math.log1p(float(effective_trials)) * dispersion / math.sqrt(max(1.0, float(len(scores))))
    deflated_score = float(best_score) - trial_penalty
    overfit = _overfit_diagnostics(best, entries)
    score_passed = bool(float(best_score) > 0.0 and deflated_score > 0.0)
    overfit_passed = bool(overfit.get("passed", True))
    passed = score_passed and overfit_passed
    reasons: list[str] = []
    if not score_passed:
        reasons.append("selection_risk_deflated_score<=0")
    if not overfit_passed:
        reasons.append(str(overfit.get("reason") or "selection_risk_pbo_failed"))
    return {
        "passed": passed,
        "reason": None if passed else reasons[0],
        "reasons": reasons,
        "effective_trials": effective_trials,
        "base_candidates": int(base_candidate_count),
        "local_refinement_candidates": int(local_refinement_candidates),
        "ensemble_refinement_candidates": int(ensemble_refinement_candidates),
        "hybrid_rescue_candidates": int(hybrid_rescue_candidates),
        "hybrid_profile_trials": int(hybrid_profile_trials),
        "internal_variants_evaluated": int(internal_variants_evaluated),
        "internal_variant_extra_trials": int(internal_variant_extra_trials),
        "finite_candidate_scores": len(scores),
        "selected_score": float(best_score),
        "runner_up_score": float(runner_up) if runner_up is not None else None,
        "median_score": float(median) if median is not None else None,
        "score_iqr": float(iqr),
        "dispersion_floor": float(_SELECTION_RISK_DISPERSION_FLOOR),
        "relative_floor": float(_SELECTION_RISK_RELATIVE_FLOOR),
        "trial_penalty": float(trial_penalty),
        "deflated_score": float(deflated_score),
        "score_margin_to_runner_up": (
            float(best_score - runner_up)
            if runner_up is not None
            else None
        ),
        "overfit_diagnostics": overfit,
    }


class _FeatureAblationModel:
    """Proxy that zeroes one contiguous feature group before scoring."""

    def __init__(self, model: TrainedModel, *, group_name: str, start: int, end: int) -> None:
        self._model = model
        self.ablated_feature_group = str(group_name)
        self.ablated_feature_start = int(start)
        self.ablated_feature_end = int(end)

    def __getattr__(self, name: str) -> object:
        return getattr(self._model, name)

    def _mask(self, features: Sequence[float]) -> tuple[float, ...]:
        values = list(features)
        start = max(0, min(len(values), self.ablated_feature_start))
        end = max(start, min(len(values), self.ablated_feature_end))
        for index in range(start, end):
            values[index] = 0.0
        return tuple(values)

    def predict_proba(self, features: tuple[float, ...]) -> float:
        return self._model.predict_proba(self._mask(features))

    def predict(self, features: tuple[float, ...], threshold: float) -> int:
        return int(self.predict_proba(features) >= threshold)


def _feature_ablation_report(
    rows: Sequence[ModelRow],
    model: TrainedModel,
    strategy: StrategyConfig,
    feature_cfg: AdvancedFeatureConfig,
    objective: ObjectiveSpec,
    *,
    market_type: str,
    starting_cash: float,
    score_batch_size: int,
) -> list[dict[str, object]]:
    """Replay the selected model with broad feature groups zeroed out."""

    if not rows:
        return []
    expected_dim = advanced_feature_dimension(feature_cfg)
    if int(getattr(model, "feature_dim", -1)) != expected_dim:
        return [{
            "status": "skipped",
            "reason": "feature_dimension_mismatch",
            "model_feature_dim": int(getattr(model, "feature_dim", -1)),
            "expected_feature_dim": expected_dim,
        }]
    baseline_result = run_backtest(
        list(rows),
        model,
        strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend="cpu",
        score_batch_size=score_batch_size,
    )
    baseline_accepted = objective.accepts(baseline_result)
    baseline_score = objective.score(baseline_result) if baseline_accepted else float("-inf")
    reports: list[dict[str, object]] = []
    for span in advanced_feature_group_spans(feature_cfg):
        proxy = _FeatureAblationModel(
            model,
            group_name=span.name,
            start=span.start,
            end=span.end,
        )
        result = run_backtest(
            list(rows),
            proxy,  # type: ignore[arg-type]
            strategy,
            starting_cash=starting_cash,
            market_type=market_type,
            compute_backend="cpu",
            score_batch_size=score_batch_size,
        )
        accepted = objective.accepts(result)
        score = objective.score(result) if accepted else float("-inf")
        delta = (
            float(score - baseline_score)
            if math.isfinite(float(score)) and math.isfinite(float(baseline_score))
            else float("-inf")
        )
        reports.append({
            "status": "evaluated",
            "removed_group": span.name,
            "start": span.start,
            "end": span.end,
            "size": span.size,
            "accepted": accepted,
            "score": float(score),
            "baseline_score": float(baseline_score),
            "delta_vs_selected": delta,
            "realized_pnl": float(result.realized_pnl),
            "max_drawdown": float(result.max_drawdown),
            "closed_trades": int(result.closed_trades),
            "reject_reason": objective.reject_reason(result),
        })
    return reports


def _candidate_diagnostics(entry: dict[str, Any]) -> dict[str, object]:
    model = entry.get("model")
    feature_cfg = entry.get("feature_cfg")
    return {
        "score": _finite_number_or_none(entry.get("score")),
        "model_family": str(getattr(model, "model_family", "")) if model is not None else "",
        "probability_inverted": bool(getattr(model, "probability_inverted", False)),
        "candidate": entry["candidate"].asdict() if isinstance(entry.get("candidate"), CandidateParams) else {},
        "hybrid_model": bool(entry.get("hybrid_model", False)),
        "hybrid_rescue": bool(entry.get("hybrid_rescue", False)),
        "hybrid_profile": entry.get("hybrid_profile"),
        "hybrid_ablation": list(entry.get("hybrid_ablation", []) or []),
        "feature_ablation": list(entry.get("feature_ablation", []) or []),
        "feature_signature": str(entry.get("feature_signature") or ""),
        "label_threshold": (
            float(feature_cfg.label_threshold)
            if isinstance(feature_cfg, AdvancedFeatureConfig)
            else None
        ),
        "label_lookahead": (
            int(feature_cfg.label_lookahead)
            if isinstance(feature_cfg, AdvancedFeatureConfig)
            else None
        ),
        "label_mode": (
            str(feature_cfg.label_mode)
            if isinstance(feature_cfg, AdvancedFeatureConfig)
            else None
        ),
        "selection_score": _finite_number_or_none(entry.get("selection_score")),
        "validation_score": _finite_number_or_none(entry.get("validation_score")),
        "full_sample_score": _finite_number_or_none(entry.get("full_sample_score")),
        "inversion_score": _finite_number_or_none(entry.get("inversion_score")),
        "inversion_selection_score": _finite_number_or_none(entry.get("inversion_selection_score")),
        "selected_internal_variant": entry.get("selected_internal_variant"),
        "inversion_source_variant": entry.get("inversion_source_variant"),
        "internal_variant_count": int(entry.get("internal_variant_count") or 0),
        "internal_variant_selection": list(entry.get("internal_variant_selection", []) or []),
        "threshold": _finite_number_or_none(entry.get("threshold")),
        "threshold_source": entry.get("threshold_source"),
        "threshold_score": _finite_number_or_none(entry.get("threshold_score")),
        "calibration_rows": int(entry.get("calibration_rows") or 0),
        "validation_rows": int(entry.get("validation_rows") or 0),
        "selection_result": entry.get("selection_result") if isinstance(entry.get("selection_result"), dict) else None,
        "validation_result": entry.get("validation_result") if isinstance(entry.get("validation_result"), dict) else None,
        "full_sample_result": entry.get("full_sample_result") if isinstance(entry.get("full_sample_result"), dict) else None,
        "inversion_selection_result": (
            entry.get("inversion_selection_result")
            if isinstance(entry.get("inversion_selection_result"), dict)
            else None
        ),
        "inversion_validation_result": (
            entry.get("inversion_validation_result")
            if isinstance(entry.get("inversion_validation_result"), dict)
            else None
        ),
        "inversion_full_sample_result": (
            entry.get("inversion_full_sample_result")
            if isinstance(entry.get("inversion_full_sample_result"), dict)
            else None
        ),
        "walk_forward_gate": entry.get("walk_forward_gate") if isinstance(entry.get("walk_forward_gate"), dict) else None,
        "selection_risk": entry.get("selection_risk") if isinstance(entry.get("selection_risk"), dict) else None,
    }


def _purged_walk_forward_gate(
    candidate: CandidateParams,
    rows: Sequence[ModelRow],
    base_strategy: StrategyConfig,
    feature_cfg: AdvancedFeatureConfig,
    objective: ObjectiveSpec,
    training: ObjectiveTraining,
    *,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None = None,
    batch_size: int = 8192,
    score_batch_size: int = 8192,
) -> dict[str, object]:
    folds = _purged_walk_forward_splits(rows, training, feature_cfg)
    if not folds:
        return {
            "passed": True,
            "reason": "insufficient_rows_for_purged_walk_forward",
            "fold_count": 0,
            "accepted_folds": 0,
            "worst_score": None,
            "worst_realized_pnl": None,
            "worst_max_drawdown": None,
            "folds": [],
        }
    strategy = replace(
        _strategy_for_candidate(base_strategy, candidate, training),
        label_threshold=float(feature_cfg.label_threshold),
    )
    fold_reports: list[dict[str, object]] = []
    worst_score = float("inf")
    worst_realized = float("inf")
    worst_drawdown = 0.0
    accepted_folds = 0
    for fold in folds:
        train_rows = list(fold["train_rows"])
        test_rows = list(fold["test_rows"])
        fold_payload = {
            "index": int(fold["index"]),
            "train_start": int(fold["train_start"]),
            "train_end": int(fold["train_end"]),
            "test_start": int(fold["test_start"]),
            "test_end": int(fold["test_end"]),
            "purge_gap": int(fold["purge_gap"]),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
        }
        try:
            fit_rows, calibration_rows = _calibration_split(
                train_rows,
                purge_gap=_label_purge_gap(feature_cfg),
            )
            model, _report = train_advanced(
                fit_rows,
                feature_cfg,
                epochs=candidate.epochs,
                learning_rate=candidate.learning_rate,
                l2_penalty=candidate.l2_penalty,
                seed=candidate.seed,
                validation_rows=calibration_rows,
                early_stopping_rounds=max(10, min(40, int(candidate.epochs) // 8)) if calibration_rows else None,
                compute_backend=compute_backend,
                batch_size=batch_size,
                focal_gamma=candidate.focal_gamma,
            )
            if training.calibrate_threshold and calibration_rows:
                threshold, threshold_source, threshold_score = _calibrate_candidate_threshold(
                    model,
                    calibration_rows,
                    strategy,
                    market_type=market_type,
                    starting_cash=starting_cash,
                    compute_backend=compute_backend,
                    score_batch_size=score_batch_size,
                    min_closed_trades=objective.min_closed_trades,
                    min_trades_per_day=objective.min_trades_per_day,
                )
                model.decision_threshold = float(threshold)
                model.threshold_source = threshold_source
                model.threshold_calibration_score = threshold_score
                model.threshold_calibration_trades = len(calibration_rows)
                model.calibration_size = len(calibration_rows)
            else:
                model.decision_threshold = float(strategy.signal_threshold)
                model.threshold_source = "strategy"
            _attach_strategy_overrides(model, strategy)
            result = run_backtest(
                test_rows,
                model,
                strategy,
                starting_cash=starting_cash,
                market_type=market_type,
                compute_backend=compute_backend,
                score_batch_size=score_batch_size,
            )
            accepted = (
                objective.accepts(result)
                and result.realized_pnl > 0.0
                and not result.stopped_by_drawdown
                and not bool(getattr(result, "stopped_by_liquidation", False))
                and int(getattr(result, "liquidation_events", 0)) <= 0
            )
            score = objective.score(result) if accepted else float("-inf")
            if accepted:
                accepted_folds += 1
            worst_score = min(worst_score, score)
            worst_realized = min(worst_realized, float(result.realized_pnl))
            worst_drawdown = max(worst_drawdown, float(result.max_drawdown))
            fold_reports.append({
                **fold_payload,
                "accepted": bool(accepted),
                "score": float(score),
                "result": _gate_result_payload(result, objective),
                "threshold": model.decision_threshold,
                "threshold_source": model.threshold_source,
            })
        except Exception as exc:  # pragma: no cover - defensive gate failure path
            worst_score = float("-inf")
            worst_realized = float("-inf")
            fold_reports.append({
                **fold_payload,
                "accepted": False,
                "score": float("-inf"),
                "error": f"{exc.__class__.__name__}: {exc}",
            })
    passed = accepted_folds == len(folds)
    return {
        "passed": bool(passed),
        "reason": None if passed else "purged_walk_forward_fold_failed",
        "fold_count": len(folds),
        "accepted_folds": accepted_folds,
        "worst_score": None if worst_score == float("inf") else float(worst_score),
        "worst_realized_pnl": None if worst_realized == float("inf") else float(worst_realized),
        "worst_max_drawdown": float(worst_drawdown),
        "folds": fold_reports,
    }


def _ensemble_seed_pack(seed: int) -> tuple[int, int, int]:
    base = int(seed)
    return base, base + 17, base + 37


def _candidate_variant(candidate: CandidateParams, **updates: object) -> CandidateParams:
    payload = candidate.asdict()
    payload.update(updates)
    return CandidateParams(
        epochs=max(1, int(payload["epochs"])),
        learning_rate=max(1e-6, float(payload["learning_rate"])),
        l2_penalty=max(0.0, float(payload["l2_penalty"])),
        signal_threshold=max(0.05, min(0.95, float(payload["signal_threshold"]))),
        stop_loss_pct=max(0.001, float(payload["stop_loss_pct"])),
        take_profit_pct=max(0.001, float(payload["take_profit_pct"])),
        risk_per_trade=max(0.0005, min(0.05, float(payload["risk_per_trade"]))),
        confidence_beta=max(0.0, min(1.0, float(payload["confidence_beta"]))),
        label_threshold_multiplier=max(0.10, min(5.0, float(payload["label_threshold_multiplier"]))),
        label_lookahead_multiplier=max(0.25, min(4.0, float(payload["label_lookahead_multiplier"]))),
        label_mode=str(payload.get("label_mode") or "forward_return"),
        focal_gamma=max(0.0, min(4.0, float(payload.get("focal_gamma", 0.0)))),
        seed=int(payload["seed"]),
    )


def _alternate_label_mode(label_mode: str) -> str:
    """Return a nearby label family that changes the decision target semantics."""

    mode = str(label_mode or "forward_return").strip().lower().replace("-", "_")
    if mode == "triple_barrier":
        return "event_volatility_triple_barrier"
    if mode == "event_volatility_triple_barrier":
        return "triple_barrier"
    if mode.startswith("downside_event"):
        return "downside_forward_return"
    if mode.startswith("downside"):
        return "downside_event_volatility_triple_barrier"
    return "triple_barrier"


def _local_refinement_candidates(candidate: CandidateParams) -> list[CandidateParams]:
    """Small post-grid search around the current winner.

    The broad grid keeps runtime predictable; this local pass exists for
    boundary cases where the winner sits on a coarse grid edge.
    """

    return [
        _candidate_variant(candidate, learning_rate=candidate.learning_rate * 0.75),
        _candidate_variant(candidate, learning_rate=candidate.learning_rate * 1.25),
        _candidate_variant(candidate, l2_penalty=candidate.l2_penalty / 3.0),
        _candidate_variant(candidate, l2_penalty=candidate.l2_penalty * 3.0),
        _candidate_variant(candidate, signal_threshold=candidate.signal_threshold - 0.01),
        _candidate_variant(candidate, signal_threshold=candidate.signal_threshold + 0.01),
        _candidate_variant(candidate, confidence_beta=candidate.confidence_beta * 0.75),
        _candidate_variant(candidate, confidence_beta=min(1.0, candidate.confidence_beta * 1.15)),
        _candidate_variant(candidate, risk_per_trade=candidate.risk_per_trade * 0.50),
        _candidate_variant(candidate, risk_per_trade=candidate.risk_per_trade * 1.25),
        _candidate_variant(candidate, stop_loss_pct=candidate.stop_loss_pct * 0.85),
        _candidate_variant(candidate, stop_loss_pct=candidate.stop_loss_pct * 1.20),
        _candidate_variant(
            candidate,
            stop_loss_pct=candidate.stop_loss_pct * 0.75,
            take_profit_pct=candidate.take_profit_pct * 0.75,
        ),
        _candidate_variant(
            candidate,
            stop_loss_pct=candidate.stop_loss_pct * 0.65,
            take_profit_pct=candidate.take_profit_pct * 1.15,
        ),
        _candidate_variant(
            candidate,
            stop_loss_pct=candidate.stop_loss_pct * 0.80,
            take_profit_pct=candidate.take_profit_pct * 1.45,
        ),
        _candidate_variant(
            candidate,
            stop_loss_pct=candidate.stop_loss_pct * 1.10,
            take_profit_pct=candidate.take_profit_pct * 0.90,
        ),
        _candidate_variant(
            candidate,
            stop_loss_pct=candidate.stop_loss_pct * 1.35,
            take_profit_pct=candidate.take_profit_pct * 1.35,
        ),
        _candidate_variant(candidate, take_profit_pct=candidate.take_profit_pct * 1.10),
        _candidate_variant(candidate, take_profit_pct=candidate.take_profit_pct * 0.80),
        _candidate_variant(candidate, label_threshold_multiplier=candidate.label_threshold_multiplier * 0.80),
        _candidate_variant(candidate, label_threshold_multiplier=candidate.label_threshold_multiplier * 1.20),
        _candidate_variant(candidate, label_lookahead_multiplier=candidate.label_lookahead_multiplier * 0.75),
        _candidate_variant(candidate, label_lookahead_multiplier=candidate.label_lookahead_multiplier * 1.25),
        _candidate_variant(candidate, focal_gamma=max(0.0, candidate.focal_gamma - 0.75)),
        _candidate_variant(candidate, focal_gamma=candidate.focal_gamma + 0.75),
        _candidate_variant(
            candidate,
            label_mode=_alternate_label_mode(candidate.label_mode),
        ),
    ]


def _default_training(objective: ObjectiveSpec) -> ObjectiveTraining:
    if objective.training is not None:
        return objective.training
    return ObjectiveTraining(
        epochs=200,
        learning_rate=0.03,
        l2_penalty=1e-3,
        signal_threshold=0.6,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        max_position_pct=0.2,
        max_trades_per_day=12,
        leverage=DEFAULT_CONSERVATIVE_LEVERAGE,
        cooldown_minutes=5,
        calibrate_threshold=True,
        walk_forward_train=300,
        walk_forward_test=80,
        walk_forward_step=30,
    )


# ==========================================================================
# Worker — must be picklable (module-level function, serializable payload)
# ==========================================================================


def _evaluate_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one candidate: train advanced model + backtest + objective score.

    The payload is a plain ``dict`` so it pickles reliably across processes.
    The return value is also plain data — no closures, no live sockets, no
    file handles.
    """

    candidate: CandidateParams = payload["candidate"]
    base_rows_train: list[ModelRow] = payload["rows_train"]
    base_rows_eval: list[ModelRow] = payload["rows_eval"]
    base_feature_cfg: AdvancedFeatureConfig = payload["feature_cfg"]
    candle_payload: Sequence[Candle] | None = payload.get("candles")
    base_strategy: StrategyConfig = payload["base_strategy"]
    objective_name: str = payload["objective"]
    market_type: str = payload["market_type"]
    starting_cash: float = payload["starting_cash"]
    ensemble_seeds: tuple[int, ...] | None = payload.get("ensemble_seeds")
    compute_backend = str(payload.get("compute_backend") or "cpu")
    batch_size = int(payload.get("batch_size") or 8192)
    score_batch_size = int(payload.get("score_batch_size") or batch_size)
    include_full_fit_fallback = bool(payload.get("include_full_fit_fallback", True))
    objective = get_objective(objective_name)
    training = _default_training(objective)
    strategy = _strategy_for_candidate(base_strategy, candidate, training)
    if candle_payload is not None:
        all_rows, feature_cfg = _rows_for_candidate(
            candle_payload,
            [],
            base_feature_cfg,
            candidate,
            strategy=strategy,
            compute_backend=compute_backend,
        )
        if not all_rows:
            raise ValueError("Candidate label profile produced zero advanced training rows")
        purge_gap = _label_purge_gap(feature_cfg)
        development_rows, _terminal_rows = _reserve_terminal_holdout(
            all_rows,
            purge_gap=purge_gap,
        )
        rows_train, rows_eval = _walk_forward_split(
            development_rows,
            purge_gap=purge_gap,
        )
    else:
        rows_train = list(base_rows_train)
        rows_eval = list(base_rows_eval)
        feature_cfg = _feature_config_for_candidate(
            base_feature_cfg,
            candidate,
            strategy=strategy,
        )

    strategy = replace(strategy, label_threshold=float(feature_cfg.label_threshold))
    purge_gap = _label_purge_gap(feature_cfg)
    model_train_rows, selection_rows = _walk_forward_split(rows_train, purge_gap=purge_gap)
    fit_rows, calibration_rows = _calibration_split(model_train_rows, purge_gap=purge_gap)
    model, report = train_advanced(
        fit_rows,
        feature_cfg,
        epochs=candidate.epochs,
        learning_rate=candidate.learning_rate,
        l2_penalty=candidate.l2_penalty,
        seed=candidate.seed,
        validation_rows=calibration_rows,
        early_stopping_rounds=max(10, min(40, int(candidate.epochs) // 8)) if calibration_rows else None,
        ensemble_seeds=ensemble_seeds,
        compute_backend=compute_backend,
        batch_size=batch_size,
        focal_gamma=candidate.focal_gamma,
    )
    threshold_score = None
    if calibration_rows:
        probability_calibration = calibrate_probability_temperature(
            calibration_rows,
            model,
            compute_backend=compute_backend,
            batch_size=batch_size,
        )
        if probability_calibration.status != "fail":
            model.probability_temperature = float(probability_calibration.temperature)
            model.probability_calibration_size = int(probability_calibration.rows)
            model.probability_log_loss_before = float(probability_calibration.log_loss_before)
            model.probability_log_loss_after = float(probability_calibration.log_loss_after)
            model.probability_brier_before = float(probability_calibration.brier_before)
            model.probability_brier_after = float(probability_calibration.brier_after)
            model.probability_ece_before = float(probability_calibration.expected_calibration_error_before)
            model.probability_ece_after = float(probability_calibration.expected_calibration_error_after)
            model.probability_calibration_backend_requested = str(probability_calibration.calibration_backend_requested)
            model.probability_calibration_backend_kind = str(probability_calibration.calibration_backend_kind)
            model.probability_calibration_backend_device = str(probability_calibration.calibration_backend_device)
            model.probability_calibration_backend_reason = str(probability_calibration.calibration_backend_reason)
    if training.calibrate_threshold and calibration_rows:
        threshold, threshold_source, threshold_score = _calibrate_candidate_threshold(
            model,
            calibration_rows,
            strategy,
            market_type=market_type,
            starting_cash=starting_cash,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
            min_closed_trades=objective.min_closed_trades,
            min_trades_per_day=objective.min_trades_per_day,
        )
        model.decision_threshold = float(threshold)
        model.threshold_source = threshold_source
        model.threshold_calibration_score = threshold_score
        model.threshold_calibration_trades = len(calibration_rows)
        model.calibration_size = len(calibration_rows)
    else:
        model.decision_threshold = float(strategy.signal_threshold)
        model.threshold_source = "strategy"
    selection_threshold = _refine_threshold_on_selection_rows(
        model,
        selection_rows,
        strategy,
        market_type=market_type,
        starting_cash=starting_cash,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
        min_closed_trades=objective.min_closed_trades,
        min_trades_per_day=objective.min_trades_per_day,
        target_trades_per_day=objective.target_trades_per_day,
    )
    if selection_threshold is not None:
        threshold, threshold_source, threshold_score = selection_threshold
        model.decision_threshold = float(threshold)
        model.threshold_source = threshold_source
        model.threshold_calibration_score = threshold_score
        model.threshold_calibration_trades = len(selection_rows)
    _attach_strategy_overrides(model, strategy)
    model.validation_size = len(rows_eval)
    selection_result = run_backtest(
        selection_rows,
        model,
        strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
    selection_score = objective.score(selection_result) if objective.accepts(selection_result) else float("-inf")
    selected_selection_result = selection_result
    selected_calibration_rows = len(calibration_rows)
    selected_internal_variant = "calibration_fit" if calibration_rows else "full_fit"
    internal_variant_selection: list[dict[str, object]] = [{
        "variant": selected_internal_variant,
        "score": _finite_number_or_none(selection_score),
        "result": _gate_result_payload(selection_result, objective),
    }]

    if calibration_rows and include_full_fit_fallback:
        fallback_model, fallback_report = train_advanced(
            model_train_rows,
            feature_cfg,
            epochs=candidate.epochs,
            learning_rate=candidate.learning_rate,
            l2_penalty=candidate.l2_penalty,
            seed=candidate.seed,
            ensemble_seeds=ensemble_seeds,
            compute_backend=compute_backend,
            batch_size=batch_size,
            focal_gamma=candidate.focal_gamma,
        )
        fallback_model.decision_threshold = float(strategy.signal_threshold)
        fallback_model.threshold_source = "strategy_full_fit"
        fallback_threshold = _refine_threshold_on_selection_rows(
            fallback_model,
            selection_rows,
            strategy,
            market_type=market_type,
            starting_cash=starting_cash,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
            min_closed_trades=objective.min_closed_trades,
            min_trades_per_day=objective.min_trades_per_day,
            target_trades_per_day=objective.target_trades_per_day,
        )
        fallback_threshold_score = None
        if fallback_threshold is not None:
            threshold, threshold_source, fallback_threshold_score = fallback_threshold
            fallback_model.decision_threshold = float(threshold)
            fallback_model.threshold_source = threshold_source
            fallback_model.threshold_calibration_score = fallback_threshold_score
            fallback_model.threshold_calibration_trades = len(selection_rows)
        _attach_strategy_overrides(fallback_model, strategy)
        fallback_model.validation_size = len(rows_eval)
        fallback_selection_result = run_backtest(
            selection_rows,
            fallback_model,
            strategy,
            starting_cash=starting_cash,
            market_type=market_type,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
        )
        fallback_selection_score = (
            objective.score(fallback_selection_result)
            if objective.accepts(fallback_selection_result)
            else float("-inf")
        )
        internal_variant_selection.append({
            "variant": "full_fit",
            "score": _finite_number_or_none(fallback_selection_score),
            "result": _gate_result_payload(fallback_selection_result, objective),
        })
        if fallback_selection_score > selection_score + 1e-12:
            model = fallback_model
            report = fallback_report
            threshold_score = fallback_threshold_score
            selected_calibration_rows = 0
            selection_score = fallback_selection_score
            selected_selection_result = fallback_selection_result
            selected_internal_variant = "full_fit"

    inversion_source_variant = selected_internal_variant
    inverted_model = copy.deepcopy(model)
    inverted_model.probability_inverted = not bool(getattr(inverted_model, "probability_inverted", False))
    inverted_model.model_family = f"{inverted_model.model_family}:inverted"
    inverted_model.quality_warnings = [
        *list(getattr(inverted_model, "quality_warnings", [])),
        "probability_inversion_variant",
    ]
    inverted_selection_result = run_backtest(
        selection_rows,
        inverted_model,
        strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
    inverted_selection_score = (
        objective.score(inverted_selection_result)
        if objective.accepts(inverted_selection_result)
        else float("-inf")
    )
    internal_variant_selection.append({
        "variant": "probability_inverted",
        "source_variant": inversion_source_variant,
        "score": _finite_number_or_none(inverted_selection_score),
        "result": _gate_result_payload(inverted_selection_result, objective),
    })
    inversion_selected = inverted_selection_score > selection_score + 1e-12
    if inversion_selected:
        model = inverted_model
        selection_score = inverted_selection_score
        selected_selection_result = inverted_selection_result
        selected_internal_variant = "probability_inverted"

    selected_holdout_result = run_backtest(
        rows_eval,
        model,
        strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
    validation_score = (
        objective.score(selected_holdout_result)
        if objective.accepts(selected_holdout_result)
        else float("-inf")
    )
    selected_full_result = run_backtest(
        rows_train + rows_eval,
        model,
        strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
    full_sample_score = (
        objective.score(selected_full_result)
        if objective.accepts(selected_full_result)
        else float("-inf")
    )
    score = min(selection_score, validation_score, full_sample_score)

    return {
        "score": float(score),
        "candidate": candidate,
        "strategy": strategy,
        "model": model,
        "feature_cfg": feature_cfg,
        "feature_dim": advanced_feature_dimension(feature_cfg),
        "feature_signature": advanced_feature_signature(feature_cfg),
        "row_count": report.row_count,
        "positive_rate": report.positive_rate,
        "threshold": model.decision_threshold,
        "threshold_source": model.threshold_source,
        "threshold_score": threshold_score,
        "calibration_rows": selected_calibration_rows,
        "validation_rows": len(rows_eval),
        "selection_score": float(selection_score),
        "validation_score": float(validation_score),
        "full_sample_score": float(full_sample_score),
        "selection_result": _gate_result_payload(selected_selection_result, objective),
        "validation_result": _gate_result_payload(selected_holdout_result, objective),
        "full_sample_result": _gate_result_payload(selected_full_result, objective),
        "selected_internal_variant": selected_internal_variant,
        "internal_variant_count": len(internal_variant_selection),
        "internal_variant_selection": internal_variant_selection,
        "inversion_source_variant": inversion_source_variant,
        "inversion_score": float(inverted_selection_score),
        "inversion_selection_score": float(inverted_selection_score),
        "inversion_selection_result": _gate_result_payload(inverted_selection_result, objective),
        "inversion_validation_result": (
            _gate_result_payload(selected_holdout_result, objective)
            if inversion_selected
            else None
        ),
        "inversion_full_sample_result": (
            _gate_result_payload(selected_full_result, objective)
            if inversion_selected
            else None
        ),
        "ensemble_refined": bool(ensemble_seeds),
    }


# ==========================================================================
# Orchestration
# ==========================================================================


def _resolve_workers(max_workers: int | None, candidate_count: int) -> int:
    if candidate_count <= 0:
        return 1
    if max_workers is not None:
        return max(1, min(int(max_workers), candidate_count))
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, candidate_count))


def train_for_objective(
    candles: Sequence[Candle],
    base_strategy: StrategyConfig,
    objective: ObjectiveSpec,
    *,
    output_dir: Path,
    market_type: str,
    starting_cash: float,
    runner: Callable[[ObjectiveSpec, CandidateParams, Sequence[ModelRow], StrategyConfig,
                     AdvancedFeatureConfig, str, float],
                    tuple[float, StrategyConfig, TrainedModel, int, float]] | None = None,
    max_workers: int | None = None,
    compute_backend: str | None = None,
    batch_size: int = 8192,
    score_batch_size: int | None = None,
    max_candidates: int | None = None,
    terminal_ledger: TerminalHoldoutLedger | None = None,
    terminal_symbol: str | None = None,
) -> ObjectiveOutcome:
    """Run the training suite for one objective, returning the outcome.

    When ``runner`` is supplied (test-path), each candidate is evaluated via
    that callable sequentially in the current process.  Otherwise the real
    :func:`_evaluate_candidate` worker is dispatched; with ``max_workers > 1``
    the candidates run in parallel via a ``ProcessPoolExecutor``.
    """

    candle_list = list(candles)
    feature_cfg = _cost_aware_feature_config(
        default_config_for(objective.name, base_strategy.enabled_features),
        base_strategy,
    )
    effective_compute_backend = effective_training_backend_name(compute_backend)
    rows = make_advanced_rows(candle_list, feature_cfg, compute_backend=effective_compute_backend)
    if not rows:
        raise ValueError("Insufficient candles to build advanced training rows")
    training = _default_training(objective)
    candidates = _candidate_grid(training)
    if max_candidates is not None:
        candidates = candidates[:max(1, int(max_candidates))]
    if not candidates:
        raise ValueError("Candidate grid produced zero evaluable entries")
    base_purge_gap = _label_purge_gap(feature_cfg)
    if runner is None:
        development_rows, _terminal_rows = _reserve_terminal_holdout(
            rows,
            purge_gap=base_purge_gap,
        )
        train_rows, eval_rows = _walk_forward_split(
            development_rows,
            purge_gap=base_purge_gap,
        )
    else:
        train_rows, eval_rows = _walk_forward_split(rows, purge_gap=base_purge_gap)
    effective_score_batch_size = int(score_batch_size if score_batch_size is not None else batch_size)

    if runner is not None:
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            score, strategy, model, row_count, positive_rate = runner(
                objective, candidate, rows, base_strategy, feature_cfg,
                market_type, starting_cash,
            )
            candidate_feature_cfg = _feature_config_for_candidate(
                feature_cfg,
                candidate,
                strategy=strategy,
            )
            strategy = replace(
                strategy,
                label_threshold=float(candidate_feature_cfg.label_threshold),
            )
            _attach_strategy_overrides(model, strategy)
            results.append({
                "score": float(score),
                "candidate": candidate,
                "strategy": strategy,
                "model": model,
                "feature_cfg": candidate_feature_cfg,
                "feature_dim": advanced_feature_dimension(candidate_feature_cfg),
                "feature_signature": advanced_feature_signature(candidate_feature_cfg),
                "row_count": row_count,
                "positive_rate": positive_rate,
                "threshold": getattr(model, "decision_threshold", None),
                "threshold_source": getattr(model, "threshold_source", None),
                "threshold_score": getattr(model, "threshold_calibration_score", None),
                "calibration_rows": int(getattr(model, "calibration_size", 0)),
                "validation_rows": len(eval_rows),
                "validation_score": float(score),
                "full_sample_score": None,
                "walk_forward_gate": None,
            })
    else:
        payloads = [
            {
                "candidate": candidate,
                "rows_train": train_rows,
                "rows_eval": eval_rows,
                "candles": candle_list,
                "feature_cfg": feature_cfg,
                "base_strategy": base_strategy,
                "objective": objective.name,
                "market_type": market_type,
                "starting_cash": starting_cash,
                "compute_backend": effective_compute_backend,
                "batch_size": batch_size,
                "score_batch_size": effective_score_batch_size,
                "include_full_fit_fallback": True,
            }
            for candidate in candidates
        ]
        workers = _resolve_workers(max_workers, len(payloads))
        if effective_compute_backend != "cpu":
            workers = 1
        if workers <= 1:
            results = [_evaluate_candidate(p) for p in payloads]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_evaluate_candidate, payloads))

    results.sort(key=_candidate_rank_key, reverse=True)
    ranked_pool = list(results)
    best = results[0]
    ensemble_refinement_candidates = 0
    local_refinement_candidates = 0
    hybrid_rescue_candidates = 0
    hybrid_profile_trials = 0
    local_results: list[dict[str, Any]] = []
    if runner is None:
        for local_candidate in _local_refinement_candidates(best["candidate"]):
            local_refinement_candidates += 1
            local_result = _evaluate_candidate({
                "candidate": local_candidate,
                "rows_train": train_rows,
                "rows_eval": eval_rows,
                "candles": candle_list,
                "feature_cfg": feature_cfg,
                "base_strategy": base_strategy,
                "objective": objective.name,
                "market_type": market_type,
                "starting_cash": starting_cash,
                "compute_backend": effective_compute_backend,
                "batch_size": batch_size,
                "score_batch_size": effective_score_batch_size,
                "include_full_fit_fallback": True,
            })
            local_results.append(local_result)
        ranked_pool = sorted([*results, *local_results], key=_candidate_rank_key, reverse=True)
        best = _risk_aware_best(best, ranked_pool)
        ensemble_results: list[dict[str, Any]] = []
        for base_result in ranked_pool[:_ENSEMBLE_REFINEMENT_CANDIDATES]:
            ensemble_refinement_candidates += 1
            ensemble_result = _evaluate_candidate({
                "candidate": base_result["candidate"],
                "rows_train": train_rows,
                "rows_eval": eval_rows,
                "candles": candle_list,
                "feature_cfg": feature_cfg,
                "base_strategy": base_strategy,
                "objective": objective.name,
                "market_type": market_type,
                "starting_cash": starting_cash,
                "ensemble_seeds": _ensemble_seed_pack(int(base_result["candidate"].seed)),
                "compute_backend": effective_compute_backend,
                "batch_size": batch_size,
                "score_batch_size": effective_score_batch_size,
                "include_full_fit_fallback": True,
            })
            ensemble_results.append(ensemble_result)
            if (
                ensemble_result["score"] > best["score"] + 1e-12
                and _risk_non_degrading(ensemble_result, best)
            ):
                best = ensemble_result
        if ensemble_results:
            ranked_pool = sorted([*ranked_pool, *ensemble_results], key=_candidate_rank_key, reverse=True)

        gated_best: dict[str, Any] | None = None
        last_gate: dict[str, object] | None = None
        for candidate_result in ranked_pool:
            if not math.isfinite(float(candidate_result["score"])):
                continue
            gate_rows, gate_feature_cfg = _rows_for_candidate(
                candle_list,
                rows,
                feature_cfg,
                candidate_result["candidate"],
                strategy=candidate_result["strategy"],
                compute_backend=effective_compute_backend,
            )
            if not gate_rows:
                candidate_result["walk_forward_gate"] = {
                    "passed": False,
                    "reason": "candidate_label_profile_produced_no_rows",
                    "fold_count": 0,
                    "accepted_folds": 0,
                    "worst_score": None,
                    "worst_realized_pnl": None,
                    "worst_max_drawdown": None,
                    "folds": [],
                }
                last_gate = candidate_result["walk_forward_gate"]
                continue
            gate_development_rows, _gate_terminal_rows = _reserve_terminal_holdout(
                gate_rows,
                purge_gap=_label_purge_gap(gate_feature_cfg),
            )
            gate = _purged_walk_forward_gate(
                candidate_result["candidate"],
                gate_development_rows,
                base_strategy,
                gate_feature_cfg,
                objective,
                training,
                market_type=market_type,
                starting_cash=starting_cash,
                compute_backend=effective_compute_backend,
                batch_size=batch_size,
                score_batch_size=effective_score_batch_size,
            )
            candidate_result["walk_forward_gate"] = gate
            last_gate = gate
            if gate.get("passed"):
                worst_score = gate.get("worst_score")
                if isinstance(worst_score, (int, float)) and math.isfinite(float(worst_score)):
                    candidate_result["score"] = min(float(candidate_result["score"]), float(worst_score))
                gated_best = candidate_result
                break
        if gated_best is None:
            best = {
                **best,
                "score": float("-inf"),
                "walk_forward_gate": last_gate,
            }
        else:
            best = gated_best

        best_strategy = best["strategy"]
        best_rows, best_feature_cfg = _rows_for_candidate(
            candle_list,
            rows,
            feature_cfg,
            best["candidate"],
            strategy=best["strategy"],
            compute_backend=effective_compute_backend,
        )
        best_purge_gap = _label_purge_gap(best_feature_cfg)
        best_development_rows, best_terminal_rows = _reserve_terminal_holdout(
            best_rows,
            purge_gap=best_purge_gap,
        )
        best_train_rows, best_eval_rows = _walk_forward_split(
            best_development_rows,
            purge_gap=best_purge_gap,
        )
        model_train_rows, selection_rows = _walk_forward_split(
            best_train_rows,
            purge_gap=best_purge_gap,
        )
        can_optimize_hybrid = (
            math.isfinite(float(best["score"]))
            and bool(model_train_rows)
            and bool(selection_rows)
            and int(getattr(best["model"], "feature_dim", -1)) == len(model_train_rows[0].features)
            and objective.name in available_objectives()
            and not bool(best.get("hybrid_model"))
        )
        if can_optimize_hybrid:
            hybrid_report = optimize_hybrid_model_zoo(
                best["model"],
                model_train_rows,
                selection_rows,
                best_strategy,
                objective_name=objective.name,
                market_type=market_type,
                starting_cash=starting_cash,
                compute_backend=effective_compute_backend,
                score_batch_size=effective_score_batch_size,
                feature_count=len(base_strategy.enabled_features),
            )
        else:
            hybrid_report = None
        if hybrid_report is not None:
            hybrid_profile_trials = max(
                hybrid_profile_trials,
                int(getattr(hybrid_report, "evaluated_profiles", 0) or 0),
            )
        if hybrid_report is not None and hybrid_report.accepted:
            holdout_result = run_backtest(
                best_eval_rows,
                hybrid_report.model,
                best_strategy,
                starting_cash=starting_cash,
                market_type=market_type,
                compute_backend=effective_compute_backend,
                score_batch_size=effective_score_batch_size,
            )
            holdout_score = objective.score(holdout_result) if objective.accepts(holdout_result) else float("-inf")
            full_result = run_backtest(
                best_development_rows,
                hybrid_report.model,
                best_strategy,
                starting_cash=starting_cash,
                market_type=market_type,
                compute_backend=effective_compute_backend,
                score_batch_size=effective_score_batch_size,
            )
            full_score = objective.score(full_result) if objective.accepts(full_result) else float("-inf")
            hybrid_score = min(float(holdout_score), float(full_score))
            if hybrid_score > float(best["score"]) + 1e-12:
                hybrid_candidate = {
                    **best,
                    "score": float(hybrid_score),
                    "validation_result": _gate_result_payload(holdout_result, objective),
                    "full_sample_result": _gate_result_payload(full_result, objective),
                }
                if _risk_non_degrading(hybrid_candidate, best):
                    best = {
                        **hybrid_candidate,
                        "model": hybrid_report.model,
                        "validation_score": float(holdout_score),
                        "full_sample_score": float(full_score),
                        "hybrid_model": True,
                        "hybrid_profile": hybrid_report.best_profile,
                        "hybrid_base_score": hybrid_report.base_score,
                        "hybrid_best_score": hybrid_report.best_score,
                        "hybrid_evaluated_profiles": hybrid_report.evaluated_profiles,
                        "hybrid_ablation": [
                            item.asdict()
                            for item in getattr(hybrid_report, "ablation_results", ()) or ()
                        ],
                        "walk_forward_gate": best.get("walk_forward_gate"),
                        "feature_cfg": best_feature_cfg,
                        "feature_dim": advanced_feature_dimension(best_feature_cfg),
                        "feature_signature": advanced_feature_signature(best_feature_cfg),
                    }
    selection_risk = _selection_risk_report(
        best,
        ranked_pool,
        base_candidate_count=len(candidates),
        local_refinement_candidates=local_refinement_candidates,
        ensemble_refinement_candidates=ensemble_refinement_candidates,
        hybrid_rescue_candidates=hybrid_rescue_candidates,
        hybrid_profile_trials=hybrid_profile_trials,
    )
    selection_risk["objective"] = objective.name
    best["selection_risk"] = selection_risk
    if not selection_risk.get("passed"):
        best["score"] = float("-inf")
        if all(id(entry) != id(best) for entry in ranked_pool):
            ranked_pool = [best, *ranked_pool]

    rejected = sum(1 for entry in results if entry["score"] == float("-inf"))
    if not math.isfinite(float(best["score"])):
        diagnostics = {
            "objective": objective.name,
            "row_count": len(rows),
            "candidate_count": len(ranked_pool),
            "top_candidates": [_candidate_diagnostics(entry) for entry in ranked_pool[:5]],
        }
        raise TrainingSuiteRejected(
            f"All {objective.name} training candidates were rejected by objective gates",
            row_count=len(rows),
            diagnostics=diagnostics,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_label_report: dict[str, object] | None = None
    if runner is None:
        try:
            meta_rows = list(best_development_rows)
            if meta_rows:
                meta_result = run_backtest(
                    meta_rows,
                    best["model"],
                    best["strategy"],
                    starting_cash=starting_cash,
                    market_type=market_type,
                    compute_backend=effective_compute_backend,
                    score_batch_size=effective_score_batch_size,
                )
                report = build_meta_label_report(
                    meta_rows,
                    best["model"],
                    best["strategy"],
                    meta_result,
                    objective_name=objective.name,
                    market_type=market_type,
                )
                meta_label_report = report.asdict()
                best["model"].meta_label_policy = dict(report.policy)
        except (ValueError, RuntimeError, OSError) as exc:
            meta_label_report = {
                "status": "error",
                "reason": str(exc),
                "objective": objective.name,
            }
            warnings = list(getattr(best["model"], "quality_warnings", []))
            warnings.append("meta_label_policy_unavailable")
            best["model"].quality_warnings = warnings
    else:
        meta_label_report = {
            "status": "not_run",
            "reason": "runner_path",
            "objective": objective.name,
        }
    feature_ablation: list[dict[str, object]] = []
    if runner is None:
        try:
            feature_ablation = _feature_ablation_report(
                best_development_rows,
                best["model"],
                best["strategy"],
                best_feature_cfg,
                objective,
                market_type=market_type,
                starting_cash=starting_cash,
                score_batch_size=effective_score_batch_size,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            feature_ablation = [{
                "status": "error",
                "reason": str(exc),
                "objective": objective.name,
            }]
    if runner is None:
        development_selected_score = float(best["score"])
        terminal_dataset_fingerprint = _model_rows_fingerprint(best_terminal_rows)
        try:
            terminal_model_sha256 = terminal_model_fingerprint(best["model"])
        except (TypeError, ValueError, OverflowError) as exc:
            terminal_reason = f"terminal_model_fingerprint_error: {exc}"
            selection_risk["passed"] = False
            selection_risk["reason"] = terminal_reason
            raw_reasons = selection_risk.get("reasons")
            reasons = [str(item) for item in raw_reasons] if isinstance(raw_reasons, list) else []
            reasons.append(terminal_reason)
            selection_risk["reasons"] = list(dict.fromkeys(reasons))
            best["selection_risk"] = selection_risk
            best["score"] = float("-inf")
            raise TrainingSuiteRejected(
                f"Selected {objective.name} model could not be fingerprinted before terminal holdout",
                row_count=len(rows),
                diagnostics={"objective": objective.name, "reason": terminal_reason},
            ) from exc
        terminal_reservation: dict[str, object] | None = None
        if terminal_ledger is not None:
            if not terminal_symbol:
                raise TrainingSuiteRejected(
                    "Durable terminal governance requires an explicit symbol",
                    row_count=len(rows),
                    diagnostics={"objective": objective.name, "reason": "terminal_symbol_missing"},
                )
            if not best_terminal_rows:
                raise TrainingSuiteRejected(
                    "Durable terminal governance cannot reserve an empty holdout",
                    row_count=len(rows),
                    diagnostics={"objective": objective.name, "reason": "terminal_holdout_missing"},
                )
            try:
                terminal_reservation = terminal_ledger.reserve(
                    symbol=terminal_symbol,
                    market_type=market_type,
                    objective=objective.name,
                    first_timestamp=int(best_terminal_rows[0].timestamp),
                    last_timestamp=int(best_terminal_rows[-1].timestamp),
                    rows=len(best_terminal_rows),
                    dataset_fingerprint=terminal_dataset_fingerprint,
                    model_fingerprint=terminal_model_sha256,
                )
            except Exception as exc:
                terminal_reason = f"terminal_reservation_failed: {exc}"
                selection_risk["passed"] = False
                selection_risk["reason"] = terminal_reason
                raw_reasons = selection_risk.get("reasons")
                reasons = [str(item) for item in raw_reasons] if isinstance(raw_reasons, list) else []
                reasons.append(terminal_reason)
                selection_risk["reasons"] = list(dict.fromkeys(reasons))
                best["selection_risk"] = selection_risk
                best["score"] = float("-inf")
                raise TrainingSuiteRejected(
                    f"Selected {objective.name} model could not reserve sealed terminal holdout",
                    row_count=len(rows),
                    diagnostics={"objective": objective.name, "reason": terminal_reason},
                ) from exc
        try:
            terminal_score, terminal_holdout = _terminal_holdout_gate(
                best_terminal_rows,
                best["model"],
                best["strategy"],
                objective,
                market_type=market_type,
                starting_cash=starting_cash,
                purge_gap=best_purge_gap,
                compute_backend=effective_compute_backend,
                score_batch_size=effective_score_batch_size,
            )
        except Exception as exc:
            if terminal_reservation is not None and terminal_ledger is not None:
                try:
                    error_report = {
                        "schema_version": _TERMINAL_HOLDOUT_SCHEMA,
                        "passed": False,
                        "reason": "terminal_holdout_unhandled_error",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                    terminal_ledger.finalize(
                        str(terminal_reservation["reservation_id"]),
                        result_status="evaluation_error",
                        result_fingerprint=terminal_result_fingerprint(error_report),
                        error=f"{exc.__class__.__name__}: {exc}",
                    )
                except Exception:
                    pass
            terminal_reason = f"terminal_holdout_unhandled_error: {exc.__class__.__name__}: {exc}"
            selection_risk["passed"] = False
            selection_risk["reason"] = terminal_reason
            best["selection_risk"] = selection_risk
            best["score"] = float("-inf")
            raise TrainingSuiteRejected(
                f"Selected {objective.name} model failed sealed terminal holdout",
                row_count=len(rows),
                diagnostics={"objective": objective.name, "reason": terminal_reason},
            ) from exc
        if terminal_holdout.get("dataset_fingerprint") != terminal_dataset_fingerprint:
            terminal_holdout["passed"] = False
            terminal_holdout["reason"] = "terminal_dataset_fingerprint_drift"
            terminal_score = float("-inf")
        try:
            terminal_result_sha256 = terminal_result_fingerprint(terminal_holdout)
        except (TypeError, ValueError, OverflowError) as exc:
            terminal_reason = f"terminal_result_fingerprint_error: {exc}"
            if terminal_reservation is not None and terminal_ledger is not None:
                fallback_report = {
                    "schema_version": _TERMINAL_HOLDOUT_SCHEMA,
                    "passed": False,
                    "reason": terminal_reason,
                }
                try:
                    terminal_ledger.finalize(
                        str(terminal_reservation["reservation_id"]),
                        result_status="evaluation_error",
                        result_fingerprint=terminal_result_fingerprint(fallback_report),
                        error=terminal_reason,
                    )
                except Exception:
                    pass
            selection_risk["passed"] = False
            selection_risk["reason"] = terminal_reason
            best["selection_risk"] = selection_risk
            best["score"] = float("-inf")
            raise TrainingSuiteRejected(
                f"Selected {objective.name} model terminal result could not be fingerprinted",
                row_count=len(rows),
                diagnostics={"objective": objective.name, "reason": terminal_reason},
            ) from exc
        if terminal_reservation is not None and terminal_ledger is not None:
            terminal_result_status = (
                "accepted"
                if terminal_holdout.get("passed") is True
                else "evaluation_error"
                if terminal_holdout.get("reason") == "terminal_holdout_evaluation_error"
                else "rejected"
            )
            try:
                terminal_holdout["reservation"] = terminal_ledger.finalize(
                    str(terminal_reservation["reservation_id"]),
                    result_status=terminal_result_status,
                    result_fingerprint=terminal_result_sha256,
                    error=str(terminal_holdout.get("reason") or ""),
                )
            except Exception as exc:
                terminal_reason = f"terminal_reservation_finalize_failed: {exc}"
                selection_risk["passed"] = False
                selection_risk["reason"] = terminal_reason
                best["selection_risk"] = selection_risk
                best["score"] = float("-inf")
                raise TrainingSuiteRejected(
                    f"Selected {objective.name} model terminal audit could not be finalized",
                    row_count=len(rows),
                    diagnostics={
                        "objective": objective.name,
                        "reason": terminal_reason,
                        "terminal_holdout": terminal_holdout,
                    },
                ) from exc
            if not reservation_evidence_passed(
                terminal_holdout.get("reservation"),
                expected_dataset_fingerprint=terminal_dataset_fingerprint,
                expected_model_fingerprint=terminal_model_sha256,
                expected_result_fingerprint=terminal_result_sha256,
                expected_rows=len(best_terminal_rows),
                expected_first_timestamp=int(best_terminal_rows[0].timestamp),
                expected_last_timestamp=int(best_terminal_rows[-1].timestamp),
                expected_symbol=terminal_symbol,
                expected_market_type=market_type,
                expected_objective=objective.name,
            ) and terminal_holdout.get("passed") is True:
                terminal_reason = "terminal_reservation_evidence_mismatch"
                selection_risk["passed"] = False
                selection_risk["reason"] = terminal_reason
                best["selection_risk"] = selection_risk
                best["score"] = float("-inf")
                raise TrainingSuiteRejected(
                    f"Selected {objective.name} model terminal audit evidence mismatched",
                    row_count=len(rows),
                    diagnostics={
                        "objective": objective.name,
                        "reason": terminal_reason,
                        "terminal_holdout": terminal_holdout,
                    },
                )
        else:
            terminal_holdout["reservation"] = None
        selection_risk["development_selected_score"] = development_selected_score
        selection_risk["terminal_holdout"] = terminal_holdout
        if terminal_holdout.get("passed") is True and math.isfinite(terminal_score):
            best["score"] = min(development_selected_score, float(terminal_score))
            selection_risk["selected_score"] = float(best["score"])
            deflated_score = _finite_number_or_none(selection_risk.get("deflated_score"))
            if deflated_score is not None:
                selection_risk["deflated_score"] = min(deflated_score, float(terminal_score))
        else:
            terminal_reason = str(terminal_holdout.get("reason") or "terminal_holdout_failed")
            raw_reasons = selection_risk.get("reasons")
            reasons = [str(item) for item in raw_reasons] if isinstance(raw_reasons, list) else []
            reasons.append(terminal_reason)
            selection_risk["passed"] = False
            selection_risk["reason"] = terminal_reason
            selection_risk["reasons"] = list(dict.fromkeys(reasons))
            best["selection_risk"] = selection_risk
            best["score"] = float("-inf")
            diagnostics = {
                "objective": objective.name,
                "row_count": len(rows),
                "candidate_count": len(ranked_pool),
                "terminal_holdout": terminal_holdout,
                "top_candidates": [_candidate_diagnostics(best)],
            }
            raise TrainingSuiteRejected(
                f"Selected {objective.name} model failed sealed terminal holdout",
                row_count=len(rows),
                diagnostics=diagnostics,
            )
    else:
        selection_risk["terminal_holdout"] = {
            "schema_version": _TERMINAL_HOLDOUT_SCHEMA,
            "passed": None,
            "reason": "runner_path",
            "evaluation_count": 0,
            "rows": 0,
            "dataset_fingerprint": "",
        }
    best["selection_risk"] = selection_risk
    model_path = output_dir / f"model_{objective.name}.json"
    best["model"].selection_risk = dict(selection_risk)
    serialize_model(best["model"], model_path)

    return ObjectiveOutcome(
        objective=objective.name,
        model_path=model_path,
        feature_dim=int(best.get("feature_dim") or advanced_feature_dimension(feature_cfg)),
        feature_signature=str(best.get("feature_signature") or advanced_feature_signature(feature_cfg)),
        best_score=float(best["score"]),
        best_params=best["candidate"].asdict(),
        explored_candidates=len(candidates),
        rejected_candidates=rejected,
        epochs=int(best["candidate"].epochs),
        learning_rate=float(best["candidate"].learning_rate),
        l2_penalty=float(best["candidate"].l2_penalty),
        row_count=int(best["row_count"]),
        positive_rate=float(best["positive_rate"]),
        decision_threshold=(
            float(best["threshold"])
            if best.get("threshold") is not None
            else None
        ),
        threshold_source=(
            str(best["threshold_source"])
            if best.get("threshold_source") is not None
            else None
        ),
        calibration_score=(
            float(best["threshold_score"])
            if best.get("threshold_score") is not None
            else None
        ),
        calibration_rows=int(best.get("calibration_rows", 0)),
        validation_rows=int(best.get("validation_rows", 0)),
        validation_score=(
            float(best["validation_score"])
            if best.get("validation_score") is not None
            else None
        ),
        full_sample_score=(
            float(best["full_sample_score"])
            if best.get("full_sample_score") is not None
            else None
        ),
        walk_forward_gate=(
            dict(best["walk_forward_gate"])
            if isinstance(best.get("walk_forward_gate"), dict)
            else None
        ),
        ensemble_refined=bool(best.get("ensemble_refined", False)),
        ensemble_refinement_candidates=ensemble_refinement_candidates,
        local_refinement_candidates=local_refinement_candidates,
        training_backend_requested=str(getattr(best["model"], "training_backend_requested", "cpu")),
        training_backend_kind=str(getattr(best["model"], "training_backend_kind", "cpu")),
        training_backend_device=str(getattr(best["model"], "training_backend_device", "cpu")),
        hybrid_model=bool(best.get("hybrid_model", False)),
        hybrid_profile=str(best.get("hybrid_profile", "base_only")),
        hybrid_base_score=(
            float(best["hybrid_base_score"])
            if best.get("hybrid_base_score") is not None
            else None
        ),
        hybrid_best_score=(
            float(best["hybrid_best_score"])
            if best.get("hybrid_best_score") is not None
            else None
        ),
        hybrid_evaluated_profiles=int(best.get("hybrid_evaluated_profiles", 0)),
        hybrid_ablation=list(best.get("hybrid_ablation", []) or []),
        hybrid_rescue=bool(best.get("hybrid_rescue", False)),
        hybrid_rescue_candidates=int(hybrid_rescue_candidates),
        feature_ablation=feature_ablation,
        meta_label_report=meta_label_report,
        selection_risk=dict(selection_risk),
    )


def run_training_suite(
    candles: Sequence[Candle],
    base_strategy: StrategyConfig,
    *,
    objectives: Sequence[str] | None = None,
    market_type: str = "spot",
    starting_cash: float = 1000.0,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    summary_path: Path | None = None,
    max_workers: int | None = None,
    compute_backend: str | None = None,
    batch_size: int = 8192,
    score_batch_size: int | None = None,
    max_candidates: int | None = None,
    symbol: str | None = None,
    terminal_ledger_path: Path | None = None,
) -> SuiteReport:
    """Train one model per objective and persist a suite summary."""

    requested_specs = (
        [get_objective(name) for name in objectives]
        if objectives
        else [get_objective(name) for name in available_objectives()]
    )
    specs: list[ObjectiveSpec] = []
    seen_specs: set[str] = set()
    for spec in requested_specs:
        if spec.name in seen_specs:
            continue
        seen_specs.add(spec.name)
        specs.append(spec)
    names = tuple(spec.name for spec in specs)
    if terminal_ledger_path is not None and symbol is None:
        raise ValueError("terminal_ledger_path requires an explicit symbol")
    normalized_symbol: str | None = None
    if symbol is not None:
        normalized_symbol = normalize_symbol(symbol, default="")
        if not is_supported_major_symbol(normalized_symbol):
            raise ValueError("training-suite terminal governance supports only BTC, ETH, or SOL")
    terminal_ledger = (
        TerminalHoldoutLedger(terminal_ledger_path)
        if normalized_symbol is not None
        else None
    )
    outcomes: list[ObjectiveOutcome] = []
    total_rows = 0
    for spec in specs:
        # Pass optional args only when the caller asked for them so legacy test
        # doubles that monkey-patch ``train_for_objective`` keep working without
        # having to extend their signature.
        extra: dict[str, object] = {}
        if max_workers is not None:
            extra["max_workers"] = max_workers
        if compute_backend is not None:
            extra["compute_backend"] = compute_backend
        if int(batch_size) != 8192:
            extra["batch_size"] = int(batch_size)
        if score_batch_size is not None:
            extra["score_batch_size"] = int(score_batch_size)
        if max_candidates is not None:
            extra["max_candidates"] = max(1, int(max_candidates))
        if terminal_ledger is not None:
            extra["terminal_ledger"] = terminal_ledger
            extra["terminal_symbol"] = normalized_symbol
        outcome = train_for_objective(
            candles, base_strategy, spec,
            output_dir=output_dir,
            market_type=market_type,
            starting_cash=starting_cash,
            **extra,
        )
        outcomes.append(outcome)
        total_rows = max(total_rows, outcome.row_count)

    summary_path = summary_path or (output_dir / "training_suite_summary.json")
    report = SuiteReport(
        outcomes=outcomes,
        total_rows=total_rows,
        total_candles=len(candles),
        output_dir=output_dir,
        summary_path=summary_path,
        objectives_run=list(names),
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(summary_path, report.asdict(), indent=2, sort_keys=True)
    return report


def describe_candidate_grid(objective: ObjectiveSpec) -> list[dict[str, float | int | str]]:
    """Return the grid of hyperparameters the suite will explore for ``objective``."""

    training = _default_training(objective)
    grid = _candidate_grid(training)
    return [candidate.asdict() for candidate in grid]


def preview_candidates() -> list[dict[str, object]]:
    """Human-friendly rollup of candidate grids across all registered objectives."""

    rows: list[dict[str, object]] = []
    for name in available_objectives():
        spec = get_objective(name)
        grid = describe_candidate_grid(spec)
        rows.append({
            "objective": name,
            "candidates": len(grid),
            "first_candidate": grid[0] if grid else {},
        })
    return rows


def rank_report(
    candidates_with_results: Sequence[tuple[dict[str, object], BacktestResult]],
    objective: str | ObjectiveSpec = "regular",
) -> list[dict[str, object]]:
    """Rank precomputed backtest results under an objective.

    This is the public convenience wrapper for callers that already evaluated
    candidates and only need the same accept/reject annotations used by the
    training suite.
    """

    spec = get_objective(objective) if isinstance(objective, str) else objective
    return rank_candidates(list(candidates_with_results), spec)


__all__ = [
    "CandidateParams",
    "ObjectiveOutcome",
    "SuiteReport",
    "TrainingSuiteRejected",
    "describe_candidate_grid",
    "preview_candidates",
    "rank_candidates",
    "run_training_suite",
    "train_for_objective",
]
