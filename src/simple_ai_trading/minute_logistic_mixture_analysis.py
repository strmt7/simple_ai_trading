"""Diagnostics and fixed-policy replay for the Round 48 mixture candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

from .cross_asset_cost_data import MINUTE_MS, SYMBOLS
from .derivatives_hurdle_data import EXECUTION_CHARGE_BPS
from .minute_logistic_mixture_tcn_model import (
    DECISION_INTERVAL_MINUTES,
    HORIZONS_MINUTES,
    MixtureForecastBundle,
    MinuteTemporalDataset,
    numpy_hurdle_probabilities,
    numpy_logistic_mixture_cdf,
    numpy_logistic_mixture_log_density,
)


QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
STRESS_EXECUTION_CHARGE_BPS = 16.0
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_BLOCK_STEPS = 7 * 24 * 60 // DECISION_INTERVAL_MINUTES
FAMILYWISE_LOWER_QUANTILE = 0.0125
SLEEVE_FRACTION = 1.0 / len(SYMBOLS)


@dataclass(frozen=True)
class MinuteMixtureTrade:
    trade_id: str
    candidate_id: str
    symbol: str
    symbol_index: int
    decision_index: int
    decision_time_ms: int
    exit_time_ms: int
    side: int
    horizon_minutes: int
    worst_seed_profit_probability: float
    worst_seed_expected_net_bps: float
    predicted_ensemble_mean_bps: float
    realized_signed_target_bps: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MinuteReplayResult:
    candidate_id: str
    scenario: str
    execution_charge_bps: float
    trades: tuple[MinuteMixtureTrade, ...]
    metrics: Mapping[str, object]
    monthly: tuple[Mapping[str, object], ...]
    daily_equity: tuple[Mapping[str, object], ...]
    trade_outcomes: tuple[Mapping[str, object], ...]


def _finite_spearman(actual: np.ndarray, prediction: np.ndarray) -> float:
    finite = np.isfinite(actual) & np.isfinite(prediction)
    if np.count_nonzero(finite) < 2:
        return 0.0
    actual_values = actual[finite].astype(np.float64, copy=False)
    predicted_values = prediction[finite].astype(np.float64, copy=False)
    if np.std(actual_values) <= 0.0 or np.std(predicted_values) <= 0.0:
        return 0.0
    value = float(spearmanr(actual_values, predicted_values).statistic)
    return value if math.isfinite(value) else 0.0


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool, copy=False)
    positives = int(np.count_nonzero(labels))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = rankdata(scores, method="average")
    positive_rank_sum = float(np.sum(ranks[labels]))
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _binary_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities.astype(np.float64), 1e-12, 1.0 - 1e-12)
    targets = labels.astype(np.float64)
    return float(
        -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped))
    )


def _expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = probabilities.size
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= boundaries[index]) & (
                probabilities <= boundaries[index + 1]
            )
        else:
            mask = (probabilities >= boundaries[index]) & (
                probabilities < boundaries[index + 1]
            )
        count = int(np.count_nonzero(mask))
        if count:
            value += count / total * abs(
                float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
            )
    return value


def _month(timestamp_ms: int) -> str:
    value = datetime.fromtimestamp(timestamp_ms / 1_000.0, UTC)
    return f"{value.year:04d}-{value.month:02d}"


def _date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1_000.0, UTC).date().isoformat()


def _ensemble_components(
    bundle: MixtureForecastBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.concatenate(
        [value / bundle.seed_weights.shape[0] for value in bundle.seed_weights],
        axis=-1,
    )
    locations = np.concatenate(
        [value for value in bundle.seed_locations_normalized], axis=-1
    )
    scales = np.concatenate(
        [value for value in bundle.seed_scales_normalized], axis=-1
    )
    if np.max(np.abs(np.sum(weights, axis=-1) - 1.0)) > 1e-6:
        raise RuntimeError("Round 48 reporting ensemble is not a probability mixture")
    return weights, locations, scales


def _mixture_quantiles(
    weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    quantiles: Sequence[float] = QUANTILES,
) -> np.ndarray:
    lower_bound = np.min(locations - 20.0 * scales, axis=-1)
    upper_bound = np.max(locations + 20.0 * scales, axis=-1)
    output: list[np.ndarray] = []
    for quantile in quantiles:
        lower = lower_bound.copy()
        upper = upper_bound.copy()
        for _ in range(24):
            midpoint = 0.5 * (lower + upper)
            cdf = numpy_logistic_mixture_cdf(
                weights, locations, scales, midpoint
            )
            lower = np.where(cdf < quantile, midpoint, lower)
            upper = np.where(cdf >= quantile, midpoint, upper)
        output.append(0.5 * (lower + upper))
    values = np.stack(output, axis=-1).astype(np.float32)
    if np.any(np.diff(values, axis=-1) < -1e-5):
        raise RuntimeError("Round 48 numerical mixture quantiles crossed")
    return values


def _pinball(actual: np.ndarray, predictions: np.ndarray) -> float:
    quantiles = np.asarray(QUANTILES, dtype=np.float64)
    errors = actual[..., None] - predictions
    return float(
        np.mean(
            np.maximum(
                quantiles * errors,
                (quantiles - 1.0) * errors,
            )
        )
    )


def _training_baselines(
    dataset: MinuteTemporalDataset,
    bundle: MixtureForecastBundle,
) -> dict[str, np.ndarray]:
    training = dataset.signed_target_bps[dataset.role_masks["training"]].reshape(
        -1, len(HORIZONS_MINUTES)
    ).astype(np.float64)
    normalized = (
        training - bundle.target_scaler.median_bps
    ) / bundle.target_scaler.scaled_iqr_bps
    logistic_scale = np.maximum(
        np.std(normalized, axis=0) * math.sqrt(3.0) / math.pi,
        0.05,
    )
    return {
        "mean_bps": np.mean(training, axis=0),
        "quantiles_bps": np.quantile(training, QUANTILES, axis=0).T,
        "logistic_location_normalized": np.zeros(
            len(HORIZONS_MINUTES), dtype=np.float64
        ),
        "logistic_scale_normalized": logistic_scale,
        "short_prevalence": np.mean(
            training < -EXECUTION_CHARGE_BPS, axis=0
        ),
        "long_prevalence": np.mean(
            training > EXECUTION_CHARGE_BPS, axis=0
        ),
    }


def _seed_stability(
    bundle: MixtureForecastBundle,
    local_evaluation: np.ndarray,
    short_thresholds: np.ndarray,
    long_thresholds: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    seed_means = np.sum(
        bundle.seed_weights * bundle.seed_locations_normalized, axis=-1
    )
    seed_probabilities = [
        numpy_hurdle_probabilities(
            bundle.seed_weights[index],
            bundle.seed_locations_normalized[index],
            bundle.seed_scales_normalized[index],
            short_thresholds,
            long_thresholds,
        )
        for index in range(bundle.seed_weights.shape[0])
    ]
    rows: list[dict[str, object]] = []
    minima = {
        "distribution_mean": 1.0,
        "short_probability": 1.0,
        "long_probability": 1.0,
    }
    for left in range(len(seed_probabilities)):
        for right in range(left + 1, len(seed_probabilities)):
            for horizon_index, horizon in enumerate(HORIZONS_MINUTES):
                mean_spearman = _finite_spearman(
                    seed_means[left, local_evaluation, :, horizon_index].reshape(-1),
                    seed_means[right, local_evaluation, :, horizon_index].reshape(-1),
                )
                short_spearman = _finite_spearman(
                    seed_probabilities[left][
                        local_evaluation, :, horizon_index, 0
                    ].reshape(-1),
                    seed_probabilities[right][
                        local_evaluation, :, horizon_index, 0
                    ].reshape(-1),
                )
                long_spearman = _finite_spearman(
                    seed_probabilities[left][
                        local_evaluation, :, horizon_index, 2
                    ].reshape(-1),
                    seed_probabilities[right][
                        local_evaluation, :, horizon_index, 2
                    ].reshape(-1),
                )
                minima["distribution_mean"] = min(
                    minima["distribution_mean"], mean_spearman
                )
                minima["short_probability"] = min(
                    minima["short_probability"], short_spearman
                )
                minima["long_probability"] = min(
                    minima["long_probability"], long_spearman
                )
                rows.append(
                    {
                        "candidate_id": bundle.candidate_id,
                        "left_seed": int(bundle.artifacts[left].seed),
                        "right_seed": int(bundle.artifacts[right].seed),
                        "horizon_minutes": horizon,
                        "distribution_mean_spearman": mean_spearman,
                        "short_probability_spearman": short_spearman,
                        "long_probability_spearman": long_spearman,
                    }
                )
    return rows, minima


def _routing_diagnostics(
    dataset: MinuteTemporalDataset,
    bundle: MixtureForecastBundle,
    local_evaluation: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    weights = bundle.seed_weights[:, local_evaluation]
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0)), axis=-1)
    effective = np.exp(np.mean(entropy, axis=(0, 1, 2)))
    feature_indices = {
        "realized_volatility_60m": dataset.feature_names.index(
            "target_realized_volatility_60m_bps"
        ),
        "liquidity_seasonality": dataset.feature_names.index(
            "target_same_minute_of_week_liquidity_ratio"
        ),
        "signed_taker_flow_15m": dataset.feature_names.index(
            "target_signed_taker_flow_15m"
        ),
    }
    global_indices = bundle.global_indices[local_evaluation]
    rows: list[dict[str, object]] = []
    for seed_index, artifact in enumerate(bundle.artifacts):
        for horizon_index, horizon in enumerate(HORIZONS_MINUTES):
            for component in range(bundle.components):
                component_weights = weights[
                    seed_index, :, :, horizon_index, component
                ].reshape(-1)
                for label, feature_index in feature_indices.items():
                    feature = dataset.features[
                        global_indices, :, feature_index
                    ].reshape(-1)
                    rows.append(
                        {
                            "candidate_id": bundle.candidate_id,
                            "seed": artifact.seed,
                            "horizon_minutes": horizon,
                            "component": component,
                            "state_variable": label,
                            "weight_state_spearman": _finite_spearman(
                                feature, component_weights
                            ),
                        }
                    )
    return rows, {
        str(horizon): float(effective[index])
        for index, horizon in enumerate(HORIZONS_MINUTES)
    }


def candidate_diagnostics(
    dataset: MinuteTemporalDataset,
    bundle: MixtureForecastBundle,
) -> dict[str, object]:
    indices = bundle.global_indices
    local_evaluation = dataset.role_masks["viability"][indices]
    if not np.any(local_evaluation):
        raise ValueError("Round 48 bundle has no evaluation rows")
    evaluation_indices = indices[local_evaluation]
    actual_bps = dataset.signed_target_bps[evaluation_indices].astype(np.float64)
    actual_normalized = (
        actual_bps - bundle.target_scaler.median_bps.reshape(1, 1, -1)
    ) / bundle.target_scaler.scaled_iqr_bps.reshape(1, 1, -1)
    all_weights, all_locations, all_scales = _ensemble_components(bundle)
    weights = all_weights[local_evaluation]
    locations = all_locations[local_evaluation]
    scales = all_scales[local_evaluation]
    mean_normalized = np.sum(weights * locations, axis=-1)
    mean_bps = (
        mean_normalized
        * bundle.target_scaler.scaled_iqr_bps.reshape(1, 1, -1)
        + bundle.target_scaler.median_bps.reshape(1, 1, -1)
    )
    quantiles_normalized = _mixture_quantiles(weights, locations, scales)
    quantiles_bps = (
        quantiles_normalized
        * bundle.target_scaler.scaled_iqr_bps.reshape(1, 1, -1, 1)
        + bundle.target_scaler.median_bps.reshape(1, 1, -1, 1)
    )
    pit = numpy_logistic_mixture_cdf(
        weights, locations, scales, actual_normalized
    )
    baselines = _training_baselines(dataset, bundle)
    short_thresholds, long_thresholds = bundle.target_scaler.normalized_thresholds(
        EXECUTION_CHARGE_BPS
    )
    action_probabilities = numpy_hurdle_probabilities(
        weights,
        locations,
        scales,
        short_thresholds,
        long_thresholds,
    )

    horizon_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    symbol_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    pit_rows: list[dict[str, object]] = []
    for horizon_index, horizon in enumerate(HORIZONS_MINUTES):
        target = actual_bps[:, :, horizon_index].reshape(-1)
        target_normalized = actual_normalized[:, :, horizon_index].reshape(-1)
        prediction = mean_bps[:, :, horizon_index].reshape(-1)
        model_mse = float(np.mean((prediction - target) ** 2))
        baseline_mse = float(
            np.mean((target - baselines["mean_bps"][horizon_index]) ** 2)
        )
        model_nll = float(
            -np.mean(
                numpy_logistic_mixture_log_density(
                    weights[:, :, horizon_index : horizon_index + 1],
                    locations[:, :, horizon_index : horizon_index + 1],
                    scales[:, :, horizon_index : horizon_index + 1],
                    actual_normalized[:, :, horizon_index : horizon_index + 1],
                )
            )
        )
        baseline_location = float(
            baselines["logistic_location_normalized"][horizon_index]
        )
        baseline_scale = float(
            baselines["logistic_scale_normalized"][horizon_index]
        )
        z = (target_normalized - baseline_location) / baseline_scale
        baseline_nll = float(
            np.mean(z + 2.0 * np.logaddexp(0.0, -z) + math.log(baseline_scale))
        )
        model_pinball = _pinball(
            actual_bps[:, :, horizon_index],
            quantiles_bps[:, :, horizon_index],
        )
        baseline_quantiles = np.broadcast_to(
            baselines["quantiles_bps"][horizon_index],
            quantiles_bps[:, :, horizon_index].shape,
        )
        baseline_pinball = _pinball(
            actual_bps[:, :, horizon_index], baseline_quantiles
        )
        row = {
            "candidate_id": bundle.candidate_id,
            "horizon_minutes": horizon,
            "rows": int(target.size),
            "actual_mean_bps": float(np.mean(target)),
            "prediction_mean_bps": float(np.mean(prediction)),
            "actual_standard_deviation_bps": float(np.std(target)),
            "prediction_standard_deviation_bps": float(np.std(prediction)),
            "distribution_mean_mse_bps2": model_mse,
            "training_mean_baseline_mse_bps2": baseline_mse,
            "distribution_mean_mse_skill": 1.0 - model_mse / baseline_mse,
            "distribution_mean_spearman": _finite_spearman(target, prediction),
            "negative_log_likelihood": model_nll,
            "unconditional_logistic_negative_log_likelihood": baseline_nll,
            "negative_log_likelihood_skill": 1.0 - model_nll / baseline_nll,
            "five_quantile_pinball_bps": model_pinball,
            "unconditional_five_quantile_pinball_bps": baseline_pinball,
            "five_quantile_pinball_skill": 1.0 - model_pinball / baseline_pinball,
            "central_80_coverage": float(
                np.mean(
                    (actual_bps[:, :, horizon_index] >= quantiles_bps[:, :, horizon_index, 0])
                    & (actual_bps[:, :, horizon_index] <= quantiles_bps[:, :, horizon_index, 4])
                )
            ),
            "central_50_coverage": float(
                np.mean(
                    (actual_bps[:, :, horizon_index] >= quantiles_bps[:, :, horizon_index, 1])
                    & (actual_bps[:, :, horizon_index] <= quantiles_bps[:, :, horizon_index, 3])
                )
            ),
            "pit_mean": float(np.mean(pit[:, :, horizon_index])),
            "pit_standard_deviation": float(np.std(pit[:, :, horizon_index])),
        }
        horizon_rows.append(row)
        histogram, edges = np.histogram(
            pit[:, :, horizon_index], bins=20, range=(0.0, 1.0)
        )
        for bin_index, count in enumerate(histogram):
            pit_rows.append(
                {
                    "candidate_id": bundle.candidate_id,
                    "horizon_minutes": horizon,
                    "bin": bin_index,
                    "lower": float(edges[bin_index]),
                    "upper": float(edges[bin_index + 1]),
                    "count": int(count),
                }
            )

        for side, class_index, prevalence_key in (
            ("short", 0, "short_prevalence"),
            ("long", 2, "long_prevalence"),
        ):
            label = (
                actual_bps[:, :, horizon_index] < -EXECUTION_CHARGE_BPS
                if side == "short"
                else actual_bps[:, :, horizon_index] > EXECUTION_CHARGE_BPS
            ).reshape(-1)
            score = action_probabilities[
                :, :, horizon_index, class_index
            ].reshape(-1)
            prevalence = float(baselines[prevalence_key][horizon_index])
            baseline_probability = np.full(score.size, prevalence)
            model_log_loss = _binary_log_loss(score, label)
            baseline_log_loss = _binary_log_loss(baseline_probability, label)
            model_brier = float(np.mean((score - label) ** 2))
            baseline_brier = float(
                np.mean((baseline_probability - label) ** 2)
            )
            action_rows.append(
                {
                    "candidate_id": bundle.candidate_id,
                    "horizon_minutes": horizon,
                    "side": side,
                    "rows": int(score.size),
                    "training_prevalence": prevalence,
                    "evaluation_prevalence": float(np.mean(label)),
                    "log_loss": model_log_loss,
                    "baseline_log_loss": baseline_log_loss,
                    "log_loss_skill": 1.0 - model_log_loss / baseline_log_loss,
                    "brier": model_brier,
                    "baseline_brier": baseline_brier,
                    "brier_skill": 1.0 - model_brier / baseline_brier,
                    "roc_auc": _roc_auc(label, score),
                    "expected_calibration_error_10_bin": _expected_calibration_error(
                        score, label
                    ),
                }
            )

        for symbol_index, symbol in enumerate(SYMBOLS):
            target_symbol = actual_bps[:, symbol_index, horizon_index]
            prediction_symbol = mean_bps[:, symbol_index, horizon_index]
            symbol_rows.append(
                {
                    "candidate_id": bundle.candidate_id,
                    "symbol": symbol,
                    "horizon_minutes": horizon,
                    "rows": int(target_symbol.size),
                    "distribution_mean_spearman": _finite_spearman(
                        target_symbol, prediction_symbol
                    ),
                    "distribution_mean_mse_bps2": float(
                        np.mean((prediction_symbol - target_symbol) ** 2)
                    ),
                }
            )

        months = np.asarray([_month(value) for value in dataset.timestamps_ms[evaluation_indices]])
        for month in sorted(set(months.tolist())):
            month_mask = months == month
            monthly_rows.append(
                {
                    "candidate_id": bundle.candidate_id,
                    "month": month,
                    "horizon_minutes": horizon,
                    "rows": int(np.count_nonzero(month_mask) * len(SYMBOLS)),
                    "distribution_mean_spearman": _finite_spearman(
                        actual_bps[month_mask, :, horizon_index].reshape(-1),
                        mean_bps[month_mask, :, horizon_index].reshape(-1),
                    ),
                    "distribution_mean_mse_bps2": float(
                        np.mean(
                            (
                                mean_bps[month_mask, :, horizon_index]
                                - actual_bps[month_mask, :, horizon_index]
                            )
                            ** 2
                        )
                    ),
                }
            )

    seed_rows, seed_minima = _seed_stability(
        bundle,
        local_evaluation,
        short_thresholds,
        long_thresholds,
    )
    routing_rows, effective_components = _routing_diagnostics(
        dataset, bundle, local_evaluation
    )
    distribution_reasons: list[str] = []
    if sum(float(row["negative_log_likelihood_skill"]) > 0.0 for row in horizon_rows) < 3:
        distribution_reasons.append("fewer_than_three_horizons_beat_unconditional_nll")
    if sum(float(row["five_quantile_pinball_skill"]) > 0.0 for row in horizon_rows) < 3:
        distribution_reasons.append("fewer_than_three_horizons_beat_unconditional_pinball")
    if sum(float(row["distribution_mean_mse_skill"]) > 0.0 for row in horizon_rows) < 3:
        distribution_reasons.append("fewer_than_three_horizons_beat_training_mean_mse")
    if sum(float(row["distribution_mean_spearman"]) > 0.0 for row in horizon_rows) < 3:
        distribution_reasons.append("fewer_than_three_horizons_have_positive_mean_spearman")
    if any(
        not 0.72 <= float(row["central_80_coverage"]) <= 0.88
        for row in horizon_rows
    ):
        distribution_reasons.append("central_80_coverage_outside_bounds")
    if any(
        not 0.42 <= float(row["central_50_coverage"]) <= 0.58
        for row in horizon_rows
    ):
        distribution_reasons.append("central_50_coverage_outside_bounds")
    if seed_minima["distribution_mean"] < 0.5:
        distribution_reasons.append("distribution_mean_seed_stability_below_0_5")
    if seed_minima["short_probability"] < 0.5:
        distribution_reasons.append("short_probability_seed_stability_below_0_5")
    if seed_minima["long_probability"] < 0.5:
        distribution_reasons.append("long_probability_seed_stability_below_0_5")

    action_reasons: list[str] = []
    if sum(float(row["log_loss_skill"]) > 0.0 for row in action_rows) < 6:
        action_reasons.append("fewer_than_six_side_horizons_beat_prevalence_log_loss")
    if sum(float(row["brier_skill"]) > 0.0 for row in action_rows) < 6:
        action_reasons.append("fewer_than_six_side_horizons_beat_prevalence_brier")
    if sum(float(row["roc_auc"]) > 0.5 for row in action_rows) < 6:
        action_reasons.append("fewer_than_six_side_horizons_have_auc_above_half")
    if max(float(row["expected_calibration_error_10_bin"]) for row in action_rows) > 0.05:
        action_reasons.append("maximum_expected_calibration_error_exceeds_0_05")
    return {
        "candidate_id": bundle.candidate_id,
        "components": bundle.components,
        "horizons": horizon_rows,
        "actions": action_rows,
        "symbols": symbol_rows,
        "monthly": monthly_rows,
        "pit_histogram": pit_rows,
        "seed_stability": seed_rows,
        "routing": routing_rows,
        "effective_components": effective_components,
        "distribution_gate": {
            "passed": not distribution_reasons,
            "reasons": distribution_reasons,
            "minimum_pairwise_seed_distribution_mean_spearman": seed_minima[
                "distribution_mean"
            ],
            "minimum_pairwise_seed_short_probability_spearman": seed_minima[
                "short_probability"
            ],
            "minimum_pairwise_seed_long_probability_spearman": seed_minima[
                "long_probability"
            ],
        },
        "action_gate": {
            "passed": not action_reasons,
            "reasons": action_reasons,
        },
        "prediction_summary": {
            "evaluation_rows": int(actual_bps.shape[0] * len(SYMBOLS)),
            "quantile_crossing_count": int(
                np.count_nonzero(np.diff(quantiles_bps, axis=-1) < -1e-5)
            ),
            "nonfinite_values": int(
                np.count_nonzero(~np.isfinite(mean_bps))
                + np.count_nonzero(~np.isfinite(quantiles_bps))
                + np.count_nonzero(~np.isfinite(action_probabilities))
            ),
        },
    }


def mixture_ablation_gate(
    control: Mapping[str, object],
    mixture: Mapping[str, object],
) -> dict[str, object]:
    control_horizons = control["horizons"]
    mixture_horizons = mixture["horizons"]
    control_actions = control["actions"]
    mixture_actions = mixture["actions"]
    if not all(
        isinstance(value, Sequence)
        for value in (
            control_horizons,
            mixture_horizons,
            control_actions,
            mixture_actions,
        )
    ):
        raise ValueError("Round 48 ablation diagnostics are invalid")
    control_nll = float(
        np.mean([float(row["negative_log_likelihood"]) for row in control_horizons])
    )
    mixture_nll = float(
        np.mean([float(row["negative_log_likelihood"]) for row in mixture_horizons])
    )
    control_log_loss = float(
        np.mean([float(row["log_loss"]) for row in control_actions])
    )
    mixture_log_loss = float(
        np.mean([float(row["log_loss"]) for row in mixture_actions])
    )
    relative_nll_improvement = 1.0 - mixture_nll / control_nll
    relative_log_loss_degradation = mixture_log_loss / control_log_loss - 1.0
    effective = min(
        float(value)
        for value in dict(mixture["effective_components"]).values()
    )
    reasons: list[str] = []
    if relative_nll_improvement < 0.005:
        reasons.append("relative_nll_improvement_below_0_005")
    if relative_log_loss_degradation > 0.01:
        reasons.append("hurdle_log_loss_degraded_more_than_one_percent")
    if effective < 1.5:
        reasons.append("minimum_effective_components_below_1_5")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "control_average_negative_log_likelihood": control_nll,
        "mixture_average_negative_log_likelihood": mixture_nll,
        "relative_negative_log_likelihood_improvement": relative_nll_improvement,
        "control_average_hurdle_log_loss": control_log_loss,
        "mixture_average_hurdle_log_loss": mixture_log_loss,
        "relative_hurdle_log_loss_degradation": relative_log_loss_degradation,
        "minimum_effective_components": effective,
    }


def select_fixed_policy_trades(
    dataset: MinuteTemporalDataset,
    bundle: MixtureForecastBundle,
) -> tuple[MinuteMixtureTrade, ...]:
    local_evaluation = dataset.role_masks["viability"][bundle.global_indices]
    global_indices = bundle.global_indices[local_evaluation]
    weights = bundle.seed_weights[:, local_evaluation]
    locations = bundle.seed_locations_normalized[:, local_evaluation]
    scales = bundle.seed_scales_normalized[:, local_evaluation]
    seed_mean_normalized = np.sum(weights * locations, axis=-1)
    seed_mean_bps = (
        seed_mean_normalized
        * bundle.target_scaler.scaled_iqr_bps.reshape(1, 1, 1, -1)
        + bundle.target_scaler.median_bps.reshape(1, 1, 1, -1)
    )
    short_thresholds, long_thresholds = bundle.target_scaler.normalized_thresholds(
        EXECUTION_CHARGE_BPS
    )
    seed_probabilities = np.stack(
        [
            numpy_hurdle_probabilities(
                weights[seed_index],
                locations[seed_index],
                scales[seed_index],
                short_thresholds,
                long_thresholds,
            )
            for seed_index in range(weights.shape[0])
        ]
    )
    ensemble_mean_bps = np.mean(seed_mean_bps, axis=0)
    free_at_ms = np.zeros(len(SYMBOLS), dtype=np.int64)
    trades: list[MinuteMixtureTrade] = []
    for local_index, global_index in enumerate(global_indices):
        decision_time = int(dataset.timestamps_ms[global_index])
        for symbol_index, symbol in enumerate(SYMBOLS):
            if decision_time < free_at_ms[symbol_index]:
                continue
            candidates: list[tuple[float, int, int, float]] = []
            for horizon_index, horizon in enumerate(HORIZONS_MINUTES):
                for side, class_index in ((-1, 0), (1, 2)):
                    expected_net = (
                        side * seed_mean_bps[:, local_index, symbol_index, horizon_index]
                        - EXECUTION_CHARGE_BPS
                    )
                    probability = seed_probabilities[
                        :, local_index, symbol_index, horizon_index, class_index
                    ]
                    if np.all(expected_net > 0.0) and np.all(probability >= 0.5):
                        candidates.append(
                            (
                                float(np.min(expected_net)),
                                side,
                                horizon_index,
                                float(np.min(probability)),
                            )
                        )
            if not candidates:
                continue
            selected = max(
                candidates,
                key=lambda value: (
                    value[0],
                    -HORIZONS_MINUTES[value[2]],
                    value[1],
                ),
            )
            worst_expected, side, horizon_index, worst_probability = selected
            horizon = HORIZONS_MINUTES[horizon_index]
            exit_time = decision_time + (horizon + 1) * MINUTE_MS
            trade_key = (
                f"{bundle.candidate_id}|{symbol}|{decision_time}|{side}|{horizon}"
            )
            trade_id = hashlib.sha256(trade_key.encode("ascii")).hexdigest()[:24]
            trades.append(
                MinuteMixtureTrade(
                    trade_id=trade_id,
                    candidate_id=bundle.candidate_id,
                    symbol=symbol,
                    symbol_index=symbol_index,
                    decision_index=int(global_index),
                    decision_time_ms=decision_time,
                    exit_time_ms=exit_time,
                    side=side,
                    horizon_minutes=horizon,
                    worst_seed_profit_probability=worst_probability,
                    worst_seed_expected_net_bps=worst_expected,
                    predicted_ensemble_mean_bps=float(
                        ensemble_mean_bps[local_index, symbol_index, horizon_index]
                    ),
                    realized_signed_target_bps=float(
                        dataset.signed_target_bps[
                            global_index, symbol_index, horizon_index
                        ]
                    ),
                )
            )
            free_at_ms[symbol_index] = exit_time
    return tuple(trades)


def _circular_block_bootstrap(
    returns_bps: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    if returns_bps.size == 0:
        return {
            "samples": BOOTSTRAP_SAMPLES,
            "block_steps": BOOTSTRAP_BLOCK_STEPS,
            "lower_quantile": FAMILYWISE_LOWER_QUANTILE,
            "lower_bps": 0.0,
            "median_bps": 0.0,
            "upper_bps": 0.0,
        }
    generator = np.random.default_rng(seed)
    blocks = math.ceil(returns_bps.size / BOOTSTRAP_BLOCK_STEPS)
    offsets = np.arange(BOOTSTRAP_BLOCK_STEPS, dtype=np.int64)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample in range(BOOTSTRAP_SAMPLES):
        starts = generator.integers(0, returns_bps.size, size=blocks)
        indices = (
            starts[:, None] + offsets.reshape(1, -1)
        ) % returns_bps.size
        resampled = returns_bps[indices.reshape(-1)[: returns_bps.size]]
        means[sample] = float(np.mean(resampled))
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "block_steps": BOOTSTRAP_BLOCK_STEPS,
        "lower_quantile": FAMILYWISE_LOWER_QUANTILE,
        "lower_bps": float(np.quantile(means, FAMILYWISE_LOWER_QUANTILE)),
        "median_bps": float(np.median(means)),
        "upper_bps": float(np.quantile(means, 1.0 - FAMILYWISE_LOWER_QUANTILE)),
    }


def replay_fixed_trades(
    dataset: MinuteTemporalDataset,
    trades: Sequence[MinuteMixtureTrade],
    *,
    candidate_id: str,
    scenario: str,
    execution_charge_bps: float,
) -> MinuteReplayResult:
    evaluation_indices = np.flatnonzero(dataset.role_masks["viability"])
    start_ms = int(dataset.timestamps_ms[evaluation_indices[0]])
    step_ms = DECISION_INTERVAL_MINUTES * MINUTE_MS
    latest_booked_time = max(
        (
            ((trade.exit_time_ms + step_ms - 1) // step_ms) * step_ms
            for trade in trades
        ),
        default=int(dataset.timestamps_ms[evaluation_indices[-1]]),
    )
    end_ms = max(
        int(dataset.timestamps_ms[evaluation_indices[-1]]),
        int(latest_booked_time),
    )
    timeline = np.arange(start_ms, end_ms + step_ms, step_ms, dtype=np.int64)
    portfolio_return_bps = np.zeros(timeline.size, dtype=np.float64)
    symbol_return_bps = np.zeros((timeline.size, len(SYMBOLS)), dtype=np.float64)
    outcomes: list[dict[str, object]] = []
    trade_net: list[float] = []
    free_at_ms = np.zeros(len(SYMBOLS), dtype=np.int64)
    for trade in trades:
        if (
            trade.candidate_id != candidate_id
            or trade.symbol_index < 0
            or trade.symbol_index >= len(SYMBOLS)
            or trade.symbol != SYMBOLS[trade.symbol_index]
            or trade.decision_index < 0
            or trade.decision_index >= dataset.timestamps
            or not dataset.role_masks["viability"][trade.decision_index]
            or trade.decision_time_ms
            != int(dataset.timestamps_ms[trade.decision_index])
            or trade.horizon_minutes not in HORIZONS_MINUTES
            or trade.side not in (-1, 1)
        ):
            raise RuntimeError("Round 48 trade ownership contract failed")
        horizon_index = HORIZONS_MINUTES.index(trade.horizon_minutes)
        source_target = float(
            dataset.signed_target_bps[
                trade.decision_index, trade.symbol_index, horizon_index
            ]
        )
        expected_exit = trade.decision_time_ms + (
            trade.horizon_minutes + 1
        ) * MINUTE_MS
        if (
            abs(source_target - trade.realized_signed_target_bps) > 1e-6
            or trade.exit_time_ms != expected_exit
            or trade.decision_time_ms < free_at_ms[trade.symbol_index]
        ):
            raise RuntimeError("Round 48 trade target or overlap contract failed")
        free_at_ms[trade.symbol_index] = trade.exit_time_ms
        net_bps = (
            trade.side * source_target - execution_charge_bps
        )
        expected_base = (
            trade.side * source_target - EXECUTION_CHARGE_BPS
        )
        if scenario == "base" and abs(net_bps - expected_base) > 1e-6:
            raise RuntimeError("Round 48 target-to-replay identity failed")
        booked_time = ((trade.exit_time_ms + step_ms - 1) // step_ms) * step_ms
        position = int(np.searchsorted(timeline, booked_time))
        if position >= timeline.size:
            raise RuntimeError("Round 48 trade books outside evaluation timeline")
        sleeve_net = SLEEVE_FRACTION * net_bps
        symbol_return_bps[position, trade.symbol_index] += sleeve_net
        portfolio_return_bps[position] += sleeve_net
        trade_net.append(net_bps)
        outcomes.append(
            {
                **trade.asdict(),
                "scenario": scenario,
                "execution_charge_bps": execution_charge_bps,
                "realized_net_bps": net_bps,
                "sleeve_weight": SLEEVE_FRACTION,
                "booked_time_ms": int(booked_time),
            }
        )
    fractions = portfolio_return_bps / 10_000.0
    if np.any(fractions <= -1.0):
        raise RuntimeError("Round 48 unlevered replay return is impossible")
    equity = np.cumprod(1.0 + fractions)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))
    equity_with_start = np.concatenate(([1.0], equity))
    drawdown = 1.0 - equity_with_start / peaks
    trade_net_array = np.asarray(trade_net, dtype=np.float64)
    positive_sum = float(np.sum(trade_net_array[trade_net_array > 0.0]))
    negative_sum = float(-np.sum(trade_net_array[trade_net_array < 0.0]))
    profit_factor = positive_sum / negative_sum if negative_sum > 0.0 else None
    active_dates = sorted({_date(trade.decision_time_ms) for trade in trades})
    per_day_counts: dict[str, int] = {}
    for trade in trades:
        day = _date(trade.decision_time_ms)
        per_day_counts[day] = per_day_counts.get(day, 0) + 1
    symbol_net = {
        symbol: float(
            np.sum(
                [
                    outcome["realized_net_bps"]
                    for outcome in outcomes
                    if outcome["symbol"] == symbol
                ]
            )
        )
        for symbol in SYMBOLS
    }
    absolute_symbol_total = sum(abs(value) for value in symbol_net.values())
    concentration = (
        max(abs(value) for value in symbol_net.values()) / absolute_symbol_total
        if absolute_symbol_total > 0.0
        else 0.0
    )

    months = np.asarray([_month(value) for value in timeline])
    monthly: list[Mapping[str, object]] = []
    for month in sorted(set(months.tolist())):
        mask = months == month
        monthly_return = float(np.prod(1.0 + fractions[mask]) - 1.0)
        month_trades = sum(
            1 for trade in trades if _month(trade.decision_time_ms) == month
        )
        monthly.append(
            {
                "candidate_id": candidate_id,
                "scenario": scenario,
                "month": month,
                "five_minute_steps": int(np.count_nonzero(mask)),
                "trades": month_trades,
                "total_net_return_fraction": monthly_return,
                "mean_five_minute_portfolio_bps": float(
                    np.mean(portfolio_return_bps[mask])
                ),
            }
        )

    dates = np.asarray([_date(value) for value in timeline])
    daily: list[Mapping[str, object]] = []
    for day in sorted(set(dates.tolist())):
        positions = np.flatnonzero(dates == day)
        last = int(positions[-1])
        daily.append(
            {
                "candidate_id": candidate_id,
                "scenario": scenario,
                "date": day,
                "equity": float(equity[last]),
                "drawdown_fraction": float(drawdown[last + 1]),
                "daily_return_fraction": float(
                    np.prod(1.0 + fractions[positions]) - 1.0
                ),
            }
        )
    metrics = {
        "candidate_id": candidate_id,
        "scenario": scenario,
        "execution_charge_bps": execution_charge_bps,
        "five_minute_steps": int(timeline.size),
        "trades": len(trades),
        "trades_by_symbol": {
            symbol: sum(1 for trade in trades if trade.symbol == symbol)
            for symbol in SYMBOLS
        },
        "trades_by_horizon": {
            str(horizon): sum(
                1 for trade in trades if trade.horizon_minutes == horizon
            )
            for horizon in HORIZONS_MINUTES
        },
        "long_trades": sum(1 for trade in trades if trade.side == 1),
        "short_trades": sum(1 for trade in trades if trade.side == -1),
        "active_days": len(active_dates),
        "median_trades_per_active_day": (
            float(np.median(list(per_day_counts.values())))
            if per_day_counts
            else 0.0
        ),
        "total_net_return_fraction": float(equity[-1] - 1.0),
        "mean_five_minute_portfolio_bps": float(np.mean(portfolio_return_bps)),
        "maximum_drawdown_fraction": float(np.max(drawdown)),
        "profit_factor": profit_factor,
        "positive_months": sum(
            float(row["total_net_return_fraction"]) > 0.0 for row in monthly
        ),
        "symbol_net_bps": symbol_net,
        "maximum_single_symbol_fraction_of_absolute_net_pnl": concentration,
        "bootstrap_mean_five_minute_portfolio_bps": _circular_block_bootstrap(
            portfolio_return_bps,
            seed=48_000 + (0 if scenario == "base" else 1),
        ),
    }
    return MinuteReplayResult(
        candidate_id=candidate_id,
        scenario=scenario,
        execution_charge_bps=execution_charge_bps,
        trades=tuple(trades),
        metrics=metrics,
        monthly=tuple(monthly),
        daily_equity=tuple(daily),
        trade_outcomes=tuple(outcomes),
    )


def economic_gate(
    base: MinuteReplayResult,
    stress: MinuteReplayResult,
    *,
    quality_passed: bool,
) -> dict[str, object]:
    reasons: list[str] = []
    stress_bootstrap = stress.metrics[
        "bootstrap_mean_five_minute_portfolio_bps"
    ]
    if not isinstance(stress_bootstrap, Mapping) or float(
        stress_bootstrap["lower_bps"]
    ) <= 0.0:
        reasons.append("stress_familywise_bootstrap_lower_not_positive")
    if int(base.metrics["positive_months"]) < 4:
        reasons.append("fewer_than_four_positive_months")
    if int(base.metrics["active_days"]) < 90:
        reasons.append("fewer_than_ninety_active_days")
    if int(base.metrics["trades"]) < 270:
        reasons.append("fewer_than_two_hundred_seventy_closed_trades")
    if float(base.metrics["median_trades_per_active_day"]) < 1.0:
        reasons.append("median_trades_per_active_day_below_one")
    if float(base.metrics["maximum_drawdown_fraction"]) > 0.1:
        reasons.append("maximum_drawdown_exceeds_ten_percent")
    profit_factor = base.metrics["profit_factor"]
    if profit_factor is None or float(profit_factor) < 1.05:
        reasons.append("profit_factor_below_1_05")
    trades_by_symbol = base.metrics["trades_by_symbol"]
    if not isinstance(trades_by_symbol, Mapping) or any(
        int(trades_by_symbol[symbol]) <= 0 for symbol in SYMBOLS
    ):
        reasons.append("not_all_symbols_have_activity")
    if (
        float(base.metrics["maximum_single_symbol_fraction_of_absolute_net_pnl"])
        > 0.5
    ):
        reasons.append("single_symbol_absolute_net_pnl_fraction_exceeds_half")
    if not quality_passed:
        reasons.append("distribution_or_action_quality_gate_failed")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "selection_contaminated": True,
        "promotion_permitted": False,
    }


__all__ = [
    "BOOTSTRAP_BLOCK_STEPS",
    "BOOTSTRAP_SAMPLES",
    "FAMILYWISE_LOWER_QUANTILE",
    "MinuteMixtureTrade",
    "MinuteReplayResult",
    "QUANTILES",
    "STRESS_EXECUTION_CHARGE_BPS",
    "candidate_diagnostics",
    "economic_gate",
    "mixture_ablation_gate",
    "replay_fixed_trades",
    "select_fixed_policy_trades",
]
