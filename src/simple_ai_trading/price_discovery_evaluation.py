"""Statistical evaluation and fail-closed gate for Round 72 predictions."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .depth_stress_screen import (
    benjamini_hochberg_q_values,
    paired_blocked_permutation_test,
)
from .price_discovery_model import (
    PRICE_DISCOVERY_HEADS,
    PRIMARY_FEATURE_LAYERS,
    ROUND72_SEED,
    PriceDiscoveryFoldPrediction,
    PriceDiscoveryPredictionRun,
    binary_log_loss,
)
from .price_discovery_spec import (
    HORIZONS_SECONDS,
    PRIMARY_LOSS_METRICS,
    load_round72_implementation,
)
from .spot_perpetual_flow import FLOW_SYMBOLS


PRICE_DISCOVERY_EVALUATION_SCHEMA = "round-072-price-discovery-evaluation-v1"
_PROBABILITY_EPSILON = 1e-6
_ProgressCallback = Callable[[str, dict[str, object]], None]


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


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def binary_loss_rows(
    target: np.ndarray,
    probability: np.ndarray,
) -> dict[str, np.ndarray]:
    truth = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(probability, dtype=np.float64)
    binary_log_loss(truth, prediction)
    if np.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError("Round 72 probabilities are outside [0,1]")
    clipped = np.clip(
        prediction,
        _PROBABILITY_EPSILON,
        1.0 - _PROBABILITY_EPSILON,
    )
    return {
        "log_loss": -(
            truth * np.log(clipped) + (1.0 - truth) * np.log1p(-clipped)
        ),
        "brier_score": np.square(prediction - truth),
    }


def continuous_loss_rows(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, np.ndarray]:
    truth = np.asarray(target, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    if (
        truth.ndim != 1
        or truth.shape != forecast.shape
        or len(truth) == 0
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(forecast))
    ):
        raise ValueError("Round 72 continuous loss inputs are invalid")
    error = forecast - truth
    return {
        "mean_squared_error": np.square(error),
        "mean_absolute_error": np.abs(error),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(data, kind="stable")
    ranks = np.empty(data.size, dtype=np.float64)
    start = 0
    while start < len(data):
        stop = start + 1
        while stop < len(data) and data[order[stop]] == data[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_rank_correlation(
    target: np.ndarray,
    prediction: np.ndarray,
) -> float:
    truth = np.asarray(target, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    if (
        truth.ndim != 1
        or truth.shape != forecast.shape
        or len(truth) < 2
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(forecast))
    ):
        raise ValueError("Round 72 Spearman inputs are invalid")
    truth_rank = _average_ranks(truth)
    forecast_rank = _average_ranks(forecast)
    if np.std(truth_rank) == 0.0 or np.std(forecast_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(truth_rank, forecast_rank)[0, 1])


def _confusion_counts(target: np.ndarray, positive: np.ndarray) -> tuple[int, int, int, int]:
    truth = np.asarray(target, dtype=np.float64) == 1.0
    predicted = np.asarray(positive, dtype=bool)
    return (
        int(np.count_nonzero(truth & predicted)),
        int(np.count_nonzero(~truth & ~predicted)),
        int(np.count_nonzero(~truth & predicted)),
        int(np.count_nonzero(truth & ~predicted)),
    )


def _balanced_accuracy(target: np.ndarray, positive: np.ndarray) -> float:
    tp, tn, fp, fn = _confusion_counts(target, positive)
    if tp + fn == 0 or tn + fp == 0:
        return float("nan")
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def _matthews(target: np.ndarray, positive: np.ndarray, *, undefined_nan: bool) -> float:
    tp, tn, fp, fn = _confusion_counts(target, positive)
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    if denominator == 0.0:
        return float("nan") if undefined_nan else 0.0
    return (tp * tn - fp * fn) / denominator


def _roc_auc(target: np.ndarray, probability: np.ndarray) -> float:
    truth = np.asarray(target, dtype=np.float64)
    positive = truth == 1.0
    positive_rows = int(np.count_nonzero(positive))
    negative_rows = len(truth) - positive_rows
    if positive_rows == 0 or negative_rows == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(probability, dtype=np.float64))
    rank_sum = float(np.sum(ranks[positive]))
    return (
        rank_sum - positive_rows * (positive_rows + 1) / 2.0
    ) / (positive_rows * negative_rows)


def _average_precision(target: np.ndarray, probability: np.ndarray) -> float:
    truth = np.asarray(target, dtype=np.float64)
    score = np.asarray(probability, dtype=np.float64)
    positives = int(np.count_nonzero(truth == 1.0))
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ordered_truth = truth[order]
    ordered_score = score[order]
    total_true = 0
    total_rows = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < len(truth):
        stop = start + 1
        while stop < len(truth) and ordered_score[stop] == ordered_score[start]:
            stop += 1
        total_true += int(np.count_nonzero(ordered_truth[start:stop] == 1.0))
        total_rows += stop - start
        recall = total_true / positives
        precision = total_true / total_rows
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(area)


def _expected_calibration_error(target: np.ndarray, probability: np.ndarray) -> float:
    truth = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(probability, dtype=np.float64)
    bins = np.minimum((prediction * 10.0).astype(np.int64), 9)
    total = 0.0
    for index in range(10):
        selected = bins == index
        rows = int(np.count_nonzero(selected))
        if rows:
            total += rows * abs(
                float(np.mean(prediction[selected])) - float(np.mean(truth[selected]))
            )
    return total / len(truth)


def binary_predictive_metrics(
    target: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | None | int]:
    truth = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(probability, dtype=np.float64)
    losses = binary_loss_rows(truth, prediction)
    positive = prediction >= 0.5
    prevalence = float(np.mean(truth))
    balanced = _balanced_accuracy(truth, positive)
    return {
        "rows": len(truth),
        "log_loss": float(np.mean(losses["log_loss"])),
        "brier_score": float(np.mean(losses["brier_score"])),
        "accuracy": float(np.mean(positive == (truth == 1.0))),
        "majority_accuracy": max(prevalence, 1.0 - prevalence),
        "balanced_accuracy": _finite_or_none(balanced),
        "MCC": float(_matthews(truth, positive, undefined_nan=False)),
        "ROC_AUC": _finite_or_none(_roc_auc(truth, prediction)),
        "precision_recall_AUC": _finite_or_none(
            _average_precision(truth, prediction)
        ),
        "expected_calibration_error": _expected_calibration_error(
            truth, prediction
        ),
        "positive_prevalence": prevalence,
        "mean_predicted_probability": float(np.mean(prediction)),
    }


def continuous_predictive_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | None | int]:
    truth = np.asarray(target, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    losses = continuous_loss_rows(truth, forecast)
    return {
        "rows": len(truth),
        "mean_squared_error": float(np.mean(losses["mean_squared_error"])),
        "mean_absolute_error": float(np.mean(losses["mean_absolute_error"])),
        "Spearman": _finite_or_none(spearman_rank_correlation(truth, forecast)),
        "sign_accuracy": float(np.mean(np.sign(truth) == np.sign(forecast))),
        "mean_target_bps": float(np.mean(truth)),
        "mean_prediction_bps": float(np.mean(forecast)),
    }


def day_metric_values(
    target: np.ndarray,
    prediction: np.ndarray,
    utc_day: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    truth = np.asarray(target, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    days = np.asarray(utc_day, dtype=np.int64)
    if (
        truth.ndim != 1
        or truth.shape != forecast.shape
        or truth.shape != days.shape
        or len(truth) == 0
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(forecast))
    ):
        raise ValueError("Round 72 UTC-day metric inputs are invalid")
    output: list[float] = []
    for day in np.unique(days):
        selected = days == day
        if metric == "balanced_accuracy":
            value = _balanced_accuracy(truth[selected], forecast[selected] >= 0.5)
        elif metric == "MCC":
            value = _matthews(
                truth[selected], forecast[selected] >= 0.5, undefined_nan=True
            )
        elif metric == "Spearman":
            value = spearman_rank_correlation(truth[selected], forecast[selected])
        else:
            raise ValueError("Round 72 UTC-day metric is unknown")
        if math.isfinite(value):
            output.append(float(value))
    return np.asarray(output, dtype=np.float64)


def day_block_bootstrap_lower(
    values: np.ndarray,
    *,
    draws: int = 10_000,
    seed: int = ROUND72_SEED,
) -> dict[str, float | int | None]:
    data = np.asarray(values, dtype=np.float64)
    draw_count = int(draws)
    if (
        data.ndim != 1
        or not np.all(np.isfinite(data))
        or not 100 <= draw_count <= 1_000_000
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("Round 72 day bootstrap inputs are invalid")
    if len(data) < 10:
        return {
            "finite_days": len(data),
            "draws": draw_count,
            "seed": seed,
            "day_mean": float(np.mean(data)) if len(data) else None,
            "lower_95": None,
        }
    generator = np.random.default_rng(seed)
    means = np.empty(draw_count, dtype=np.float64)
    completed = 0
    while completed < draw_count:
        batch = min(1_024, draw_count - completed)
        indexes = generator.integers(0, len(data), size=(batch, len(data)))
        means[completed : completed + batch] = np.mean(data[indexes], axis=1)
        completed += batch
    return {
        "finite_days": len(data),
        "draws": draw_count,
        "seed": seed,
        "day_mean": float(np.mean(data)),
        "lower_95": float(np.quantile(means, 0.025, method="linear")),
    }


def _blocks(
    run: PriceDiscoveryPredictionRun,
    symbol: str,
    horizon: int,
    layer: str,
    head: str,
) -> tuple[PriceDiscoveryFoldPrediction, ...]:
    values = tuple(
        block
        for block in run.blocks
        if block.symbol == symbol
        and block.horizon_seconds == horizon
        and block.feature_layer == layer
        and block.head == head
    )
    if tuple(block.fold for block in values) != tuple(range(1, 7)):
        raise ValueError("Round 72 prediction fold group is incomplete")
    return values


def _concatenate(
    blocks: tuple[PriceDiscoveryFoldPrediction, ...],
    name: str,
) -> np.ndarray:
    return np.concatenate([np.asarray(getattr(block, name)) for block in blocks])


def _family_keys() -> tuple[tuple[str, int, str, str], ...]:
    return tuple(
        (symbol, horizon, head, metric)
        for symbol in FLOW_SYMBOLS
        for horizon in HORIZONS_SECONDS
        for head in PRICE_DISCOVERY_HEADS
        for metric in PRIMARY_LOSS_METRICS[head]
    )


def _binary_layer_report(
    blocks: tuple[PriceDiscoveryFoldPrediction, ...],
    *,
    bootstrap_draws: int,
    balanced_seed: int,
    mcc_seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    target = _concatenate(blocks, "primary_target")
    prediction = _concatenate(blocks, "primary_prediction")
    days = _concatenate(blocks, "utc_day")
    baseline = np.concatenate(
        [np.full(block.test_rows, block.training_prevalence) for block in blocks]
    )
    losses = binary_loss_rows(target, prediction)
    baseline_losses = binary_loss_rows(target, baseline)
    fold_reports = []
    positive_folds = {metric: 0 for metric in PRIMARY_LOSS_METRICS["binary_direction"]}
    for block in blocks:
        fold_loss = binary_loss_rows(block.primary_target, block.primary_prediction)
        fold_baseline = binary_loss_rows(
            block.primary_target,
            np.full(block.test_rows, block.training_prevalence),
        )
        values = {}
        for metric in PRIMARY_LOSS_METRICS["binary_direction"]:
            model_mean = float(np.mean(fold_loss[metric]))
            baseline_mean = float(np.mean(fold_baseline[metric]))
            improved = model_mean < baseline_mean
            positive_folds[metric] += int(improved)
            values[metric] = {
                "model": model_mean,
                "training_prevalence_baseline": baseline_mean,
                "relative_improvement": (baseline_mean - model_mean) / baseline_mean,
                "improved": improved,
            }
        fold_reports.append({"fold": block.fold, "rows": block.test_rows, "losses": values})
    metrics = binary_predictive_metrics(target, prediction)
    prevalence_comparison = {}
    for metric in PRIMARY_LOSS_METRICS["binary_direction"]:
        model_mean = float(np.mean(losses[metric]))
        baseline_mean = float(np.mean(baseline_losses[metric]))
        prevalence_comparison[metric] = {
            "model": model_mean,
            "training_prevalence_baseline": baseline_mean,
            "relative_improvement": (baseline_mean - model_mean) / baseline_mean,
            "positive_folds": positive_folds[metric],
        }
    balanced_values = day_metric_values(
        target, prediction, days, metric="balanced_accuracy"
    )
    mcc_values = day_metric_values(target, prediction, days, metric="MCC")
    stress_target = _concatenate(blocks, "stress_target")
    stress_prediction = _concatenate(blocks, "stress_prediction")
    stress_baseline = np.concatenate(
        [np.full(block.stress_test_rows, block.training_prevalence) for block in blocks]
    )
    stress_losses = binary_loss_rows(stress_target, stress_prediction)
    stress_baseline_losses = binary_loss_rows(stress_target, stress_baseline)
    stress = {}
    for metric in PRIMARY_LOSS_METRICS["binary_direction"]:
        model_mean = float(np.mean(stress_losses[metric]))
        baseline_mean = float(np.mean(stress_baseline_losses[metric]))
        stress[metric] = {
            "model": model_mean,
            "training_prevalence_baseline": baseline_mean,
            "skill": 1.0 - model_mean / baseline_mean,
        }
    return (
        {
            "rows": len(target),
            "utc_days": len(np.unique(days)),
            "metrics": metrics,
            "prevalence_comparison": prevalence_comparison,
            "folds": fold_reports,
            "day_bootstrap": {
                "balanced_accuracy": day_block_bootstrap_lower(
                    balanced_values, draws=bootstrap_draws, seed=balanced_seed
                ),
                "MCC": day_block_bootstrap_lower(
                    mcc_values, draws=bootstrap_draws, seed=mcc_seed
                ),
            },
            "stress_delay_seconds": 5,
            "stress_rows": len(stress_target),
            "stress_comparison": stress,
        },
        {**losses, "utc_day": days, "target": target, "prediction": prediction},
    )


def _continuous_layer_report(
    blocks: tuple[PriceDiscoveryFoldPrediction, ...],
    *,
    bootstrap_draws: int,
    spearman_seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    target = _concatenate(blocks, "primary_target")
    prediction = _concatenate(blocks, "primary_prediction")
    days = _concatenate(blocks, "utc_day")
    losses = continuous_loss_rows(target, prediction)
    zero_losses = continuous_loss_rows(target, np.zeros_like(target))
    training_mean = np.concatenate(
        [np.full(block.test_rows, block.training_mean_target_bps) for block in blocks]
    )
    training_mean_losses = continuous_loss_rows(target, training_mean)
    controls = {}
    for metric in PRIMARY_LOSS_METRICS["continuous_return_bps"]:
        model_mean = float(np.mean(losses[metric]))
        zero_mean = float(np.mean(zero_losses[metric]))
        train_mean = float(np.mean(training_mean_losses[metric]))
        controls[metric] = {
            "model": model_mean,
            "zero_return": zero_mean,
            "skill_vs_zero": (
                1.0 - model_mean / zero_mean if zero_mean > 0.0 else None
            ),
            "fold_training_mean": train_mean,
            "skill_vs_fold_training_mean": (
                1.0 - model_mean / train_mean if train_mean > 0.0 else None
            ),
        }
    fold_reports = []
    for block in blocks:
        fold_losses = continuous_loss_rows(
            block.primary_target, block.primary_prediction
        )
        fold_reports.append(
            {
                "fold": block.fold,
                "rows": block.test_rows,
                "losses": {
                    metric: float(np.mean(fold_losses[metric]))
                    for metric in PRIMARY_LOSS_METRICS["continuous_return_bps"]
                },
            }
        )
    spearman_values = day_metric_values(
        target, prediction, days, metric="Spearman"
    )
    return (
        {
            "rows": len(target),
            "utc_days": len(np.unique(days)),
            "metrics": continuous_predictive_metrics(target, prediction),
            "controls": controls,
            "folds": fold_reports,
            "day_bootstrap": {
                "Spearman": day_block_bootstrap_lower(
                    spearman_values, draws=bootstrap_draws, seed=spearman_seed
                )
            },
        },
        {**losses, "utc_day": days, "target": target, "prediction": prediction},
    )


def _model_summaries(run: PriceDiscoveryPredictionRun) -> list[dict[str, object]]:
    return [
        {
            "symbol": block.symbol,
            "horizon_seconds": block.horizon_seconds,
            "feature_layer": block.feature_layer,
            "head": block.head,
            "fold": block.fold,
            "training_rows": block.training_rows,
            "tuning_rows": block.tuning_rows,
            "test_rows": block.test_rows,
            "stress_test_rows": block.stress_test_rows,
            "best_iteration": block.best_iteration,
            "calibration_value": block.calibration_value,
            "calibration_retained": block.calibration_retained,
            "model_bytes": block.model_bytes,
            "model_sha256": block.model_sha256,
            "prediction_sha256": block.prediction_sha256,
            "reload_max_absolute_prediction_difference": block.reload_max_absolute_prediction_difference,
        }
        for block in run.blocks
    ]


def evaluate_price_discovery_primary(
    run: PriceDiscoveryPredictionRun,
    *,
    implementation_path: str | Path,
    progress: _ProgressCallback | None = None,
) -> dict[str, object]:
    """Evaluate the exact primary layers and apply all frozen gates."""

    run.validate()
    if run.feature_layers != PRIMARY_FEATURE_LAYERS:
        raise ValueError("Round 72 primary evaluation requires both frozen primary layers")
    implementation = load_round72_implementation(implementation_path)
    if implementation["implementation_sha256"] != run.implementation_sha256:
        raise ValueError("Round 72 model run and implementation identities differ")
    evaluation = implementation.get("evaluation_contract")
    if not isinstance(evaluation, dict):
        raise ValueError("Round 72 evaluation contract is missing")
    permutation_draws = int(evaluation["permutation_draws"])
    bootstrap_draws = int(evaluation["bootstrap_draws"])
    if permutation_draws != 10_000 or bootstrap_draws != 10_000:
        raise ValueError("Round 72 resampling counts differ")
    family = _family_keys()
    family_index = {key: index for index, key in enumerate(family)}
    if len(family) != int(evaluation["fdr_family_cardinality"]):
        raise ValueError("Round 72 FDR family cardinality differs")

    layer_reports: list[dict[str, object]] = []
    cache: dict[tuple[str, int, str, str], dict[str, np.ndarray]] = {}
    report_by_key: dict[tuple[str, int, str, str], dict[str, object]] = {}
    total_groups = len(FLOW_SYMBOLS) * len(HORIZONS_SECONDS) * len(
        PRIMARY_FEATURE_LAYERS
    ) * len(PRICE_DISCOVERY_HEADS)
    completed = 0
    for symbol in FLOW_SYMBOLS:
        for horizon in HORIZONS_SECONDS:
            for layer in PRIMARY_FEATURE_LAYERS:
                for head in PRICE_DISCOVERY_HEADS:
                    blocks = _blocks(run, symbol, horizon, layer, head)
                    if head == "binary_direction":
                        first_seed = ROUND72_SEED + family_index[
                            (symbol, horizon, head, "log_loss")
                        ]
                        second_seed = ROUND72_SEED + family_index[
                            (symbol, horizon, head, "brier_score")
                        ]
                        report, arrays = _binary_layer_report(
                            blocks,
                            bootstrap_draws=bootstrap_draws,
                            balanced_seed=first_seed,
                            mcc_seed=second_seed,
                        )
                    else:
                        seed = ROUND72_SEED + family_index[
                            (symbol, horizon, head, "mean_squared_error")
                        ]
                        report, arrays = _continuous_layer_report(
                            blocks,
                            bootstrap_draws=bootstrap_draws,
                            spearman_seed=seed,
                        )
                    complete_report = {
                        "symbol": symbol,
                        "horizon_seconds": horizon,
                        "feature_layer": layer,
                        "head": head,
                        **report,
                    }
                    layer_reports.append(complete_report)
                    cache[(symbol, horizon, layer, head)] = arrays
                    report_by_key[(symbol, horizon, layer, head)] = complete_report
                    completed += 1
                    if progress:
                        progress(
                            "price_discovery_group_evaluated",
                            {
                                "symbol": symbol,
                                "horizon_seconds": horizon,
                                "feature_layer": layer,
                                "head": head,
                                "completed_groups": completed,
                                "total_groups": total_groups,
                            },
                        )

    comparisons: list[dict[str, object]] = []
    for index, (symbol, horizon, head, metric) in enumerate(family):
        baseline = cache[(symbol, horizon, "perpetual_only", head)]
        challenger = cache[(symbol, horizon, "spot_perpetual", head)]
        if (
            not np.array_equal(baseline["utc_day"], challenger["utc_day"])
            or not np.array_equal(baseline["target"], challenger["target"])
        ):
            raise ValueError("Round 72 paired feature-layer rows differ")
        baseline_mean = float(np.mean(baseline[metric]))
        challenger_mean = float(np.mean(challenger[metric]))
        if baseline_mean > 0.0:
            comparison_payload: dict[str, object] = asdict(
                paired_blocked_permutation_test(
                    baseline[metric],
                    challenger[metric],
                    baseline["utc_day"],
                    draws=permutation_draws,
                    seed=ROUND72_SEED + index,
                )
            )
            comparison_payload["degenerate_baseline"] = False
        else:
            comparison_payload = {
                "rows": len(baseline[metric]),
                "blocks": len(np.unique(baseline["utc_day"])),
                "baseline_mean_loss": baseline_mean,
                "challenger_mean_loss": challenger_mean,
                "mean_loss_difference": challenger_mean - baseline_mean,
                "relative_improvement": None,
                "one_sided_p_value": 1.0,
                "permutation_draws": permutation_draws,
                "seed": ROUND72_SEED + index,
                "degenerate_baseline": True,
            }
        comparisons.append(
            {
                "family_index": index,
                "symbol": symbol,
                "horizon_seconds": horizon,
                "head": head,
                "metric": metric,
                **comparison_payload,
            }
        )
    q_values = benjamini_hochberg_q_values(
        [float(value["one_sided_p_value"]) for value in comparisons]
    )
    feature_gate_passed = True
    for comparison, q_value in zip(comparisons, q_values, strict=True):
        comparison["q_value"] = float(q_value)
        relative = comparison["relative_improvement"]
        comparison["passed"] = bool(
            relative is not None
            and float(relative) >= 0.001
            and float(q_value) <= 0.05
        )
        feature_gate_passed = feature_gate_passed and bool(comparison["passed"])

    components: list[dict[str, object]] = []
    all_predictive_components = True
    for symbol in FLOW_SYMBOLS:
        for horizon in HORIZONS_SECONDS:
            binary = report_by_key[
                (symbol, horizon, "spot_perpetual", "binary_direction")
            ]
            continuous = report_by_key[
                (symbol, horizon, "spot_perpetual", "continuous_return_bps")
            ]
            binary_comparison = binary["prevalence_comparison"]
            binary_metrics = binary["metrics"]
            binary_bootstrap = binary["day_bootstrap"]
            stress = binary["stress_comparison"]
            continuous_controls = continuous["controls"]
            continuous_bootstrap = continuous["day_bootstrap"]
            reasons: list[str] = []
            for metric in PRIMARY_LOSS_METRICS["binary_direction"]:
                values = binary_comparison[metric]
                if float(values["relative_improvement"]) < 0.002:
                    reasons.append(f"binary_{metric}_relative_improvement_below_0_002")
                if int(values["positive_folds"]) < 4:
                    reasons.append(f"binary_{metric}_positive_folds_below_4")
                if float(stress[metric]["skill"]) <= 0.0:
                    reasons.append(f"stress_{metric}_skill_not_positive")
            if float(binary_metrics["accuracy"]) <= float(
                binary_metrics["majority_accuracy"]
            ):
                reasons.append("binary_accuracy_not_above_majority")
            balanced_lower = binary_bootstrap["balanced_accuracy"]["lower_95"]
            mcc_lower = binary_bootstrap["MCC"]["lower_95"]
            if balanced_lower is None or float(balanced_lower) <= 0.5:
                reasons.append("balanced_accuracy_day_bootstrap_lower_not_above_0_5")
            if mcc_lower is None or float(mcc_lower) <= 0.0:
                reasons.append("MCC_day_bootstrap_lower_not_above_0")
            mse_skill = continuous_controls["mean_squared_error"]["skill_vs_zero"]
            if mse_skill is None or float(mse_skill) < 0.001:
                reasons.append("continuous_MSE_skill_vs_zero_below_0_001")
            spearman_lower = continuous_bootstrap["Spearman"]["lower_95"]
            if spearman_lower is None or float(spearman_lower) <= 0.0:
                reasons.append("Spearman_day_bootstrap_lower_not_above_0")
            passed = not reasons
            all_predictive_components = all_predictive_components and passed
            components.append(
                {
                    "symbol": symbol,
                    "horizon_seconds": horizon,
                    "passed": passed,
                    "reasons": reasons,
                }
            )
    passed = bool(feature_gate_passed and all_predictive_components)
    without_hash: dict[str, object] = {
        "schema_version": PRICE_DISCOVERY_EVALUATION_SCHEMA,
        "implementation_sha256": run.implementation_sha256,
        "dataset_bundle_sha256": run.dataset_bundle_sha256,
        "model_run_sha256": run.run_sha256,
        "backend": {
            "requested": run.backend_requested,
            "kind": run.backend_kind,
            "device": run.backend_device,
            "lightgbm_version": run.lightgbm_version,
        },
        "scope": {
            "symbols": list(FLOW_SYMBOLS),
            "horizons_seconds": list(HORIZONS_SECONDS),
            "feature_layers": list(PRIMARY_FEATURE_LAYERS),
            "heads": list(PRICE_DISCOVERY_HEADS),
            "terminal_holdout_read": False,
            "profit_or_execution_target": False,
        },
        "resampling": {
            "permutation_draws": permutation_draws,
            "bootstrap_draws": bootstrap_draws,
            "seed": ROUND72_SEED,
            "unit": "UTC day",
        },
        "models": _model_summaries(run),
        "layer_reports": layer_reports,
        "feature_comparisons": comparisons,
        "feature_increment_gate_passed": feature_gate_passed,
        "symbol_horizon_components": components,
        "primary_gate_passed": passed,
        "decision": (
            "open_frozen_terminal_holdout"
            if passed
            else "reject_round_072_price_discovery"
        ),
        "profitability_claim": False,
        "execution_or_fill_claim": False,
        "trading_authority": False,
        "leverage_authority": False,
    }
    report = {**without_hash, "report_sha256": _canonical_sha256(without_hash)}
    validate_price_discovery_evaluation(report)
    return report


def validate_price_discovery_evaluation(report: dict[str, object]) -> None:
    canonical = dict(report)
    observed_hash = canonical.pop("report_sha256", "")
    components = report.get("symbol_horizon_components")
    comparisons = report.get("feature_comparisons")
    layer_reports = report.get("layer_reports")
    models = report.get("models")
    expected_component_keys = [
        (symbol, horizon) for symbol in FLOW_SYMBOLS for horizon in HORIZONS_SECONDS
    ]
    if (
        report.get("schema_version") != PRICE_DISCOVERY_EVALUATION_SCHEMA
        or observed_hash != _canonical_sha256(canonical)
        or not isinstance(components, list)
        or not isinstance(comparisons, list)
        or not isinstance(layer_reports, list)
        or not isinstance(models, list)
        or len(components) != 9
        or len(comparisons) != 36
        or len(layer_reports) != 36
        or len(models) != 216
        or [
            (str(value.get("symbol")), int(value.get("horizon_seconds", -1)))
            for value in components
            if isinstance(value, dict)
        ]
        != expected_component_keys
        or any(
            report.get(name) is not False
            for name in (
                "profitability_claim",
                "execution_or_fill_claim",
                "trading_authority",
                "leverage_authority",
            )
        )
        or report.get("primary_gate_passed")
        is not bool(
            report.get("feature_increment_gate_passed")
            and all(
                isinstance(value, dict) and value.get("passed") is True
                for value in components
            )
        )
    ):
        raise ValueError("Round 72 evaluation report contract is invalid")


__all__ = [
    "PRICE_DISCOVERY_EVALUATION_SCHEMA",
    "binary_loss_rows",
    "binary_predictive_metrics",
    "continuous_loss_rows",
    "continuous_predictive_metrics",
    "day_block_bootstrap_lower",
    "day_metric_values",
    "evaluate_price_discovery_primary",
    "spearman_rank_correlation",
    "validate_price_discovery_evaluation",
]
