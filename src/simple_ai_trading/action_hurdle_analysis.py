"""Diagnostics and fixed-policy validation simulation for Round 49 candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

from .action_hurdle_tcn_model import (
    ActionHurdleForecastBundle,
    PRIMARY_HORIZON_INDEX,
    PRIMARY_HORIZON_MINUTES,
    SEEDS,
    SIDES,
    MinuteTemporalDataset,
    numpy_gamma_mean_score,
    side_net_targets,
)
from .cross_asset_cost_data import MINUTE_MS, SYMBOLS
from .derivatives_hurdle_data import EXECUTION_CHARGE_BPS


DECISION_INTERVAL_MINUTES = 5
STRESS_EXECUTION_CHARGE_BPS = 16.0
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_BLOCK_STEPS = 2_016
FAMILYWISE_LOWER_QUANTILE = 0.0125
SLEEVE_FRACTION = 1.0 / len(SYMBOLS)


@dataclass(frozen=True)
class ActionHurdleTrade:
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
    ensemble_profit_probability: float
    ensemble_expected_net_bps: float
    expected_net_seed_range_bps: float
    realized_signed_target_bps: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActionHurdleReplayResult:
    candidate_id: str
    scenario: str
    execution_charge_bps: float
    trades: tuple[ActionHurdleTrade, ...]
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
    targets = labels.astype(bool, copy=False)
    positives = int(np.count_nonzero(targets))
    negatives = int(targets.size - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = rankdata(scores, method="average")
    positive_rank_sum = float(np.sum(ranks[targets]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


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
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = (
            (probabilities >= lower) & (probabilities <= upper)
            if index == bins - 1
            else (probabilities >= lower) & (probabilities < upper)
        )
        count = int(np.count_nonzero(mask))
        if count:
            value += (
                count
                / total
                * abs(
                    float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
                )
            )
    return value


def _month(timestamp_ms: int) -> str:
    value = datetime.fromtimestamp(timestamp_ms / 1_000.0, UTC)
    return f"{value.year:04d}-{value.month:02d}"


def _date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1_000.0, UTC).date().isoformat()


def _training_baselines(
    dataset: MinuteTemporalDataset,
    bundle: ActionHurdleForecastBundle,
) -> dict[str, np.ndarray]:
    targets = side_net_targets(dataset)[dataset.role_masks["training"]].astype(
        np.float64
    )
    primary = targets[:, :, PRIMARY_HORIZON_INDEX]
    return {
        "profit_prevalence_by_symbol": np.mean(targets > 0.0, axis=0),
        "profit_prevalence_by_side": np.mean(targets > 0.0, axis=(0, 1)),
        "primary_mean_bps": np.mean(primary, axis=0),
        "gain_mean_bps": bundle.target_scaler.gain_mean_bps.astype(np.float64),
        "loss_mean_bps": bundle.target_scaler.loss_mean_bps.astype(np.float64),
    }


def _probability_row(
    *,
    candidate_id: str,
    scope: str,
    side: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    baseline_probability: np.ndarray,
    symbol: str | None = None,
    month: str | None = None,
) -> dict[str, object]:
    model_log_loss = _binary_log_loss(probabilities, labels)
    baseline_log_loss = _binary_log_loss(baseline_probability, labels)
    model_brier = float(np.mean((probabilities - labels) ** 2))
    baseline_brier = float(np.mean((baseline_probability - labels) ** 2))
    return {
        "candidate_id": candidate_id,
        "scope": scope,
        "symbol": symbol,
        "month": month,
        "horizon_minutes": PRIMARY_HORIZON_MINUTES,
        "side": side,
        "rows": int(labels.size),
        "training_prevalence": float(np.mean(baseline_probability)),
        "evaluation_prevalence": float(np.mean(labels)),
        "roc_auc": _roc_auc(labels, probabilities),
        "log_loss": model_log_loss,
        "baseline_log_loss": baseline_log_loss,
        "log_loss_skill": 1.0 - model_log_loss / baseline_log_loss,
        "brier": model_brier,
        "baseline_brier": baseline_brier,
        "brier_skill": 1.0 - model_brier / baseline_brier,
        "expected_calibration_error_10_bin": _expected_calibration_error(
            probabilities, labels
        ),
    }


def _action_row(
    *,
    candidate_id: str,
    scope: str,
    side: str,
    prediction: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    symbol: str | None = None,
    month: str | None = None,
) -> dict[str, object]:
    model_mse = float(np.mean((prediction - target) ** 2))
    baseline_mse = float(np.mean((baseline - target) ** 2))
    return {
        "candidate_id": candidate_id,
        "scope": scope,
        "symbol": symbol,
        "month": month,
        "horizon_minutes": PRIMARY_HORIZON_MINUTES,
        "side": side,
        "rows": int(target.size),
        "actual_mean_bps": float(np.mean(target)),
        "prediction_mean_bps": float(np.mean(prediction)),
        "prediction_standard_deviation_bps": float(np.std(prediction)),
        "expected_net_mse_bps2": model_mse,
        "training_mean_baseline_mse_bps2": baseline_mse,
        "expected_net_mse_skill": 1.0 - model_mse / baseline_mse,
        "expected_net_spearman": _finite_spearman(target, prediction),
    }


def _seed_stability(
    bundle: ActionHurdleForecastBundle,
    local_evaluation: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    evaluation_probabilities = bundle.seed_probabilities[:, local_evaluation]
    evaluation_actions = bundle.seed_action_values_bps[:, local_evaluation]
    rows: list[dict[str, object]] = []
    minima = {"probability": 1.0, "expected_net": 1.0}
    for left in range(len(SEEDS)):
        for right in range(left + 1, len(SEEDS)):
            for side_index, side in enumerate(("short", "long")):
                probability = _finite_spearman(
                    evaluation_probabilities[
                        left, :, :, PRIMARY_HORIZON_INDEX, side_index
                    ].reshape(-1),
                    evaluation_probabilities[
                        right, :, :, PRIMARY_HORIZON_INDEX, side_index
                    ].reshape(-1),
                )
                expected_net = _finite_spearman(
                    evaluation_actions[left, :, :, side_index].reshape(-1),
                    evaluation_actions[right, :, :, side_index].reshape(-1),
                )
                minima["probability"] = min(minima["probability"], probability)
                minima["expected_net"] = min(minima["expected_net"], expected_net)
                rows.append(
                    {
                        "candidate_id": bundle.candidate_id,
                        "left_seed": int(bundle.artifacts[left].seed),
                        "right_seed": int(bundle.artifacts[right].seed),
                        "horizon_minutes": PRIMARY_HORIZON_MINUTES,
                        "side": side,
                        "probability_spearman": probability,
                        "expected_net_spearman": expected_net,
                    }
                )
    return rows, minima


def _severity_rows(
    dataset: MinuteTemporalDataset,
    bundle: ActionHurdleForecastBundle,
    evaluation_indices: np.ndarray,
    actual_primary: np.ndarray,
) -> list[dict[str, object]]:
    if bundle.seed_gain_means_bps is None or bundle.seed_loss_means_bps is None:
        return []
    local_evaluation = dataset.role_masks["viability"][bundle.global_indices]
    gain_prediction = np.mean(
        bundle.seed_gain_means_bps[:, local_evaluation], axis=0, dtype=np.float64
    )
    loss_prediction = np.mean(
        bundle.seed_loss_means_bps[:, local_evaluation], axis=0, dtype=np.float64
    )
    rows: list[dict[str, object]] = []
    months = np.asarray(
        [_month(value) for value in dataset.timestamps_ms[evaluation_indices]]
    )
    scopes: list[tuple[str, str | None, str | None, np.ndarray]] = [
        (
            "pooled",
            None,
            None,
            np.ones(actual_primary.shape[:2], dtype=bool),
        )
    ]
    scopes.extend(
        (
            "symbol",
            symbol,
            None,
            np.broadcast_to(
                np.arange(len(SYMBOLS)) == symbol_index,
                actual_primary.shape[:2],
            ),
        )
        for symbol_index, symbol in enumerate(SYMBOLS)
    )
    scopes.extend(
        (
            "month",
            None,
            month,
            np.broadcast_to((months == month)[:, None], actual_primary.shape[:2]),
        )
        for month in sorted(set(months.tolist()))
    )
    for scope, symbol, month, row_mask in scopes:
        for side_index, side in enumerate(("short", "long")):
            target = actual_primary[..., side_index]
            gain_condition = row_mask & (target > 0.0)
            loss_condition = row_mask & (target <= 0.0)
            if not np.any(gain_condition) or not np.any(loss_condition):
                continue
            gain_target = np.clip(target, 0.0, None)
            loss_target = np.clip(-target, 0.0, None)
            gain_baseline = np.broadcast_to(
                bundle.target_scaler.gain_mean_bps[:, side_index], target.shape
            )
            loss_baseline = np.broadcast_to(
                bundle.target_scaler.loss_mean_bps[:, side_index], target.shape
            )
            model_gain = numpy_gamma_mean_score(
                gain_prediction[..., side_index], gain_target, gain_condition
            )
            baseline_gain = numpy_gamma_mean_score(
                gain_baseline, gain_target, gain_condition
            )
            model_loss = numpy_gamma_mean_score(
                loss_prediction[..., side_index], loss_target, loss_condition
            )
            baseline_loss = numpy_gamma_mean_score(
                loss_baseline, loss_target, loss_condition
            )
            rows.append(
                {
                    "candidate_id": bundle.candidate_id,
                    "scope": scope,
                    "symbol": symbol,
                    "month": month,
                    "horizon_minutes": PRIMARY_HORIZON_MINUTES,
                    "side": side,
                    "gain_rows": int(np.count_nonzero(gain_condition)),
                    "loss_rows": int(np.count_nonzero(loss_condition)),
                    "conditional_gain_gamma_score": model_gain,
                    "training_gain_baseline_gamma_score": baseline_gain,
                    "conditional_gain_gamma_score_skill": 1.0
                    - model_gain / baseline_gain,
                    "conditional_loss_gamma_score": model_loss,
                    "training_loss_baseline_gamma_score": baseline_loss,
                    "conditional_loss_gamma_score_skill": 1.0
                    - model_loss / baseline_loss,
                }
            )
    return rows


def candidate_diagnostics(
    dataset: MinuteTemporalDataset,
    bundle: ActionHurdleForecastBundle,
) -> dict[str, object]:
    indices = bundle.global_indices
    local_evaluation = dataset.role_masks["viability"][indices]
    if not np.any(local_evaluation):
        raise ValueError("Round 49 bundle has no evaluation rows")
    evaluation_indices = indices[local_evaluation]
    targets = side_net_targets(dataset)[evaluation_indices].astype(np.float64)
    actual_primary = targets[:, :, PRIMARY_HORIZON_INDEX]
    probabilities = np.mean(
        bundle.seed_probabilities[:, local_evaluation], axis=0, dtype=np.float64
    )
    action_values = np.mean(
        bundle.seed_action_values_bps[:, local_evaluation],
        axis=0,
        dtype=np.float64,
    )
    baselines = _training_baselines(dataset, bundle)
    months = np.asarray(
        [_month(value) for value in dataset.timestamps_ms[evaluation_indices]]
    )

    probability_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    for side_index, side in enumerate(("short", "long")):
        labels = actual_primary[..., side_index] > 0.0
        score = probabilities[..., PRIMARY_HORIZON_INDEX, side_index]
        pooled_prevalence = float(
            baselines["profit_prevalence_by_side"][PRIMARY_HORIZON_INDEX, side_index]
        )
        probability_rows.append(
            _probability_row(
                candidate_id=bundle.candidate_id,
                scope="pooled",
                side=side,
                probabilities=score.reshape(-1),
                labels=labels.reshape(-1),
                baseline_probability=np.full(labels.size, pooled_prevalence),
            )
        )
        primary_prediction = action_values[..., side_index]
        primary_target = actual_primary[..., side_index]
        baseline_matrix = np.broadcast_to(
            baselines["primary_mean_bps"][:, side_index], primary_target.shape
        )
        action_rows.append(
            _action_row(
                candidate_id=bundle.candidate_id,
                scope="pooled",
                side=side,
                prediction=primary_prediction.reshape(-1),
                target=primary_target.reshape(-1),
                baseline=baseline_matrix.reshape(-1),
            )
        )

        for symbol_index, symbol in enumerate(SYMBOLS):
            symbol_prevalence = float(
                baselines["profit_prevalence_by_symbol"][
                    symbol_index, PRIMARY_HORIZON_INDEX, side_index
                ]
            )
            probability_rows.append(
                _probability_row(
                    candidate_id=bundle.candidate_id,
                    scope="symbol",
                    symbol=symbol,
                    side=side,
                    probabilities=score[:, symbol_index],
                    labels=labels[:, symbol_index],
                    baseline_probability=np.full(labels.shape[0], symbol_prevalence),
                )
            )
            action_rows.append(
                _action_row(
                    candidate_id=bundle.candidate_id,
                    scope="symbol",
                    symbol=symbol,
                    side=side,
                    prediction=primary_prediction[:, symbol_index],
                    target=primary_target[:, symbol_index],
                    baseline=np.full(
                        primary_target.shape[0],
                        baselines["primary_mean_bps"][symbol_index, side_index],
                    ),
                )
            )

        for month in sorted(set(months.tolist())):
            month_mask = months == month
            month_baseline_probability = np.broadcast_to(
                baselines["profit_prevalence_by_symbol"][
                    :, PRIMARY_HORIZON_INDEX, side_index
                ],
                labels[month_mask].shape,
            )
            probability_rows.append(
                _probability_row(
                    candidate_id=bundle.candidate_id,
                    scope="month",
                    month=month,
                    side=side,
                    probabilities=score[month_mask].reshape(-1),
                    labels=labels[month_mask].reshape(-1),
                    baseline_probability=month_baseline_probability.reshape(-1),
                )
            )
            month_baseline_action = np.broadcast_to(
                baselines["primary_mean_bps"][:, side_index],
                primary_target[month_mask].shape,
            )
            action_rows.append(
                _action_row(
                    candidate_id=bundle.candidate_id,
                    scope="month",
                    month=month,
                    side=side,
                    prediction=primary_prediction[month_mask].reshape(-1),
                    target=primary_target[month_mask].reshape(-1),
                    baseline=month_baseline_action.reshape(-1),
                )
            )

    seed_rows, seed_minima = _seed_stability(bundle, local_evaluation)
    severity_rows = _severity_rows(dataset, bundle, evaluation_indices, actual_primary)
    pooled_probability = {
        str(row["side"]): row for row in probability_rows if row["scope"] == "pooled"
    }
    pooled_action = {
        str(row["side"]): row for row in action_rows if row["scope"] == "pooled"
    }
    reasons: list[str] = []
    for side in ("short", "long"):
        probability = pooled_probability[side]
        action = pooled_action[side]
        if float(probability["roc_auc"]) < 0.55:
            reasons.append(f"{side}_pooled_auc_below_0_55")
        if float(probability["log_loss_skill"]) <= 0.0:
            reasons.append(f"{side}_probability_log_loss_skill_not_positive")
        if float(probability["brier_skill"]) <= 0.0:
            reasons.append(f"{side}_probability_brier_skill_not_positive")
        if float(probability["expected_calibration_error_10_bin"]) > 0.05:
            reasons.append(f"{side}_expected_calibration_error_exceeds_0_05")
        monthly_probability = [
            row
            for row in probability_rows
            if row["scope"] == "month" and row["side"] == side
        ]
        if sum(float(row["roc_auc"]) > 0.5 for row in monthly_probability) < 4:
            reasons.append(f"{side}_fewer_than_four_months_auc_above_half")
        if float(action["expected_net_mse_skill"]) <= 0.0:
            reasons.append(f"{side}_expected_net_mse_skill_not_positive")
        if float(action["expected_net_spearman"]) < 0.03:
            reasons.append(f"{side}_expected_net_spearman_below_0_03")
        positive_symbols = sum(
            float(row["expected_net_spearman"]) > 0.0
            for row in action_rows
            if row["scope"] == "symbol" and row["side"] == side
        )
        if positive_symbols < len(SYMBOLS):
            reasons.append(f"{side}_not_all_symbols_have_positive_action_spearman")
        positive_months = sum(
            float(row["expected_net_spearman"]) > 0.0
            for row in action_rows
            if row["scope"] == "month" and row["side"] == side
        )
        if positive_months < 4:
            reasons.append(f"{side}_fewer_than_four_months_positive_action_spearman")
    if seed_minima["probability"] < 0.5:
        reasons.append("minimum_pairwise_seed_probability_spearman_below_0_5")
    if seed_minima["expected_net"] < 0.5:
        reasons.append("minimum_pairwise_seed_expected_net_spearman_below_0_5")

    arrays = [
        bundle.seed_probabilities,
        bundle.seed_action_values_bps,
        bundle.seed_auxiliary_mean_bps,
    ]
    if bundle.seed_gain_means_bps is not None:
        arrays.append(bundle.seed_gain_means_bps)
    if bundle.seed_loss_means_bps is not None:
        arrays.append(bundle.seed_loss_means_bps)
    nonfinite = sum(int(np.count_nonzero(~np.isfinite(value))) for value in arrays)
    probability_minimum = float(np.min(bundle.seed_probabilities))
    probability_maximum = float(np.max(bundle.seed_probabilities))
    numerical_reasons: list[str] = []
    if nonfinite:
        numerical_reasons.append("nonfinite_prediction_values")
    if probability_minimum <= 0.0 or probability_maximum >= 1.0:
        numerical_reasons.append("probability_not_strictly_inside_unit_interval")
    if bundle.seed_gain_means_bps is not None and np.any(
        bundle.seed_gain_means_bps <= 0.0
    ):
        numerical_reasons.append("conditional_gain_mean_not_positive")
    if bundle.seed_loss_means_bps is not None and np.any(
        bundle.seed_loss_means_bps <= 0.0
    ):
        numerical_reasons.append("conditional_loss_mean_not_positive")

    return {
        "candidate_id": bundle.candidate_id,
        "probability": probability_rows,
        "expected_net": action_rows,
        "severity": severity_rows,
        "seed_stability": seed_rows,
        "action_quality_gate": {
            "passed": not reasons,
            "reasons": reasons,
            "minimum_pairwise_seed_probability_spearman": seed_minima["probability"],
            "minimum_pairwise_seed_expected_net_spearman": seed_minima["expected_net"],
        },
        "numerical_prediction_gate": {
            "passed": not numerical_reasons,
            "reasons": numerical_reasons,
            "nonfinite_values": nonfinite,
            "minimum_probability": probability_minimum,
            "maximum_probability": probability_maximum,
        },
    }


def mechanism_ablation_gate(
    control: Mapping[str, object],
    hurdle: Mapping[str, object],
) -> dict[str, object]:
    def pooled(rows: object) -> dict[str, Mapping[str, object]]:
        if not isinstance(rows, list):
            raise ValueError("Round 49 diagnostic rows are invalid")
        selected = {
            str(row["side"]): row
            for row in rows
            if isinstance(row, Mapping) and row.get("scope") == "pooled"
        }
        if set(selected) != {"short", "long"}:
            raise ValueError("Round 49 pooled side diagnostics are incomplete")
        return selected

    control_action = pooled(control.get("expected_net"))
    hurdle_action = pooled(hurdle.get("expected_net"))
    control_probability = pooled(control.get("probability"))
    hurdle_probability = pooled(hurdle.get("probability"))
    severity = pooled(hurdle.get("severity"))
    control_spearman = float(
        np.mean(
            [control_action[side]["expected_net_spearman"] for side in SIDES_BY_NAME]
        )
    )
    hurdle_spearman = float(
        np.mean(
            [hurdle_action[side]["expected_net_spearman"] for side in SIDES_BY_NAME]
        )
    )
    control_mse = float(
        np.mean(
            [control_action[side]["expected_net_mse_bps2"] for side in SIDES_BY_NAME]
        )
    )
    hurdle_mse = float(
        np.mean(
            [hurdle_action[side]["expected_net_mse_bps2"] for side in SIDES_BY_NAME]
        )
    )
    control_log_loss = float(
        np.mean([control_probability[side]["log_loss"] for side in SIDES_BY_NAME])
    )
    hurdle_log_loss = float(
        np.mean([hurdle_probability[side]["log_loss"] for side in SIDES_BY_NAME])
    )
    spearman_improvement = hurdle_spearman - control_spearman
    relative_mse_degradation = hurdle_mse / control_mse - 1.0
    relative_log_loss_degradation = hurdle_log_loss / control_log_loss - 1.0
    reasons: list[str] = []
    for side in SIDES_BY_NAME:
        if float(severity[side]["conditional_gain_gamma_score_skill"]) <= 0.0:
            reasons.append(f"{side}_conditional_gain_gamma_skill_not_positive")
        if float(severity[side]["conditional_loss_gamma_score_skill"]) <= 0.0:
            reasons.append(f"{side}_conditional_loss_gamma_skill_not_positive")
    if spearman_improvement < 0.005:
        reasons.append("average_expected_net_spearman_improvement_below_0_005")
    if relative_mse_degradation > 0.01:
        reasons.append("relative_expected_net_mse_degradation_exceeds_0_01")
    if relative_log_loss_degradation > 0.01:
        reasons.append("relative_probability_log_loss_degradation_exceeds_0_01")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "control_average_expected_net_spearman": control_spearman,
        "hurdle_average_expected_net_spearman": hurdle_spearman,
        "average_expected_net_spearman_improvement": spearman_improvement,
        "control_average_expected_net_mse_bps2": control_mse,
        "hurdle_average_expected_net_mse_bps2": hurdle_mse,
        "relative_expected_net_mse_degradation": relative_mse_degradation,
        "control_average_probability_log_loss": control_log_loss,
        "hurdle_average_probability_log_loss": hurdle_log_loss,
        "relative_probability_log_loss_degradation": relative_log_loss_degradation,
    }


SIDES_BY_NAME = ("short", "long")


def select_fixed_policy_trades(
    dataset: MinuteTemporalDataset,
    bundle: ActionHurdleForecastBundle,
) -> tuple[ActionHurdleTrade, ...]:
    local_evaluation = dataset.role_masks["viability"][bundle.global_indices]
    global_indices = bundle.global_indices[local_evaluation]
    probabilities = bundle.seed_probabilities[:, local_evaluation][
        :, :, :, PRIMARY_HORIZON_INDEX
    ]
    action_values = bundle.seed_action_values_bps[:, local_evaluation]
    free_at_ms = np.zeros(len(SYMBOLS), dtype=np.int64)
    trades: list[ActionHurdleTrade] = []
    for local_index, global_index in enumerate(global_indices):
        decision_time = int(dataset.timestamps_ms[global_index])
        for symbol_index, symbol in enumerate(SYMBOLS):
            if decision_time < free_at_ms[symbol_index]:
                continue
            eligible: list[tuple[float, int]] = []
            for side_index, side in enumerate(SIDES):
                seed_values = action_values[:, local_index, symbol_index, side_index]
                if np.all(seed_values > 0.0):
                    eligible.append((float(np.min(seed_values)), side))
            if not eligible:
                continue
            if len(eligible) == 2 and eligible[0][0] == eligible[1][0]:
                continue
            worst_expected, side = max(eligible, key=lambda item: item[0])
            side_index = SIDES.index(side)
            seed_values = action_values[:, local_index, symbol_index, side_index]
            seed_probabilities = probabilities[:, local_index, symbol_index, side_index]
            exit_time = decision_time + (PRIMARY_HORIZON_MINUTES + 1) * MINUTE_MS
            trade_key = (
                f"{bundle.candidate_id}|{symbol}|{decision_time}|{side}|"
                f"{PRIMARY_HORIZON_MINUTES}"
            )
            trade_id = hashlib.sha256(trade_key.encode("ascii")).hexdigest()[:24]
            trades.append(
                ActionHurdleTrade(
                    trade_id=trade_id,
                    candidate_id=bundle.candidate_id,
                    symbol=symbol,
                    symbol_index=symbol_index,
                    decision_index=int(global_index),
                    decision_time_ms=decision_time,
                    exit_time_ms=exit_time,
                    side=side,
                    horizon_minutes=PRIMARY_HORIZON_MINUTES,
                    worst_seed_profit_probability=float(np.min(seed_probabilities)),
                    worst_seed_expected_net_bps=worst_expected,
                    ensemble_profit_probability=float(np.mean(seed_probabilities)),
                    ensemble_expected_net_bps=float(np.mean(seed_values)),
                    expected_net_seed_range_bps=float(np.ptp(seed_values)),
                    realized_signed_target_bps=float(
                        dataset.signed_target_bps[
                            global_index, symbol_index, PRIMARY_HORIZON_INDEX
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
        indices = (starts[:, None] + offsets.reshape(1, -1)) % returns_bps.size
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
    trades: Sequence[ActionHurdleTrade],
    *,
    candidate_id: str,
    scenario: str,
    execution_charge_bps: float,
) -> ActionHurdleReplayResult:
    if scenario not in {"base", "stress"}:
        raise ValueError("Round 49 replay scenario must be base or stress")
    evaluation_indices = np.flatnonzero(dataset.role_masks["viability"])
    if evaluation_indices.size == 0:
        raise ValueError("Round 49 replay has no evaluation timeline")
    start_ms = int(dataset.timestamps_ms[evaluation_indices[0]])
    step_ms = DECISION_INTERVAL_MINUTES * MINUTE_MS
    latest_booked_time = max(
        (((trade.exit_time_ms + step_ms - 1) // step_ms) * step_ms for trade in trades),
        default=int(dataset.timestamps_ms[evaluation_indices[-1]]),
    )
    end_ms = max(int(dataset.timestamps_ms[evaluation_indices[-1]]), latest_booked_time)
    timeline = np.arange(start_ms, end_ms + step_ms, step_ms, dtype=np.int64)
    portfolio_return_bps = np.zeros(timeline.size, dtype=np.float64)
    symbol_return_bps = np.zeros((timeline.size, len(SYMBOLS)), dtype=np.float64)
    outcomes: list[dict[str, object]] = []
    trade_net: list[float] = []
    free_at_ms = np.zeros(len(SYMBOLS), dtype=np.int64)
    seen_trade_ids: set[str] = set()
    for trade in trades:
        if (
            trade.trade_id in seen_trade_ids
            or trade.candidate_id != candidate_id
            or trade.symbol_index < 0
            or trade.symbol_index >= len(SYMBOLS)
            or trade.symbol != SYMBOLS[trade.symbol_index]
            or trade.decision_index < 0
            or trade.decision_index >= dataset.timestamps
            or not dataset.role_masks["viability"][trade.decision_index]
            or trade.decision_time_ms
            != int(dataset.timestamps_ms[trade.decision_index])
            or trade.horizon_minutes != PRIMARY_HORIZON_MINUTES
            or trade.side not in SIDES
        ):
            raise RuntimeError("Round 49 trade ownership contract failed")
        seen_trade_ids.add(trade.trade_id)
        source_target = float(
            dataset.signed_target_bps[
                trade.decision_index,
                trade.symbol_index,
                PRIMARY_HORIZON_INDEX,
            ]
        )
        expected_exit = (
            trade.decision_time_ms + (PRIMARY_HORIZON_MINUTES + 1) * MINUTE_MS
        )
        if (
            abs(source_target - trade.realized_signed_target_bps) > 1e-6
            or trade.exit_time_ms != expected_exit
            or trade.decision_time_ms < free_at_ms[trade.symbol_index]
        ):
            raise RuntimeError("Round 49 trade target or overlap contract failed")
        free_at_ms[trade.symbol_index] = trade.exit_time_ms
        net_bps = trade.side * source_target - execution_charge_bps
        expected_base = trade.side * source_target - EXECUTION_CHARGE_BPS
        if scenario == "base" and abs(net_bps - expected_base) > 1e-6:
            raise RuntimeError("Round 49 target-to-replay identity failed")
        booked_time = ((trade.exit_time_ms + step_ms - 1) // step_ms) * step_ms
        position = int(np.searchsorted(timeline, booked_time))
        if position >= timeline.size or int(timeline[position]) != booked_time:
            raise RuntimeError("Round 49 trade books outside evaluation timeline")
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
        raise RuntimeError("Round 49 unlevered replay return is impossible")
    equity = np.cumprod(1.0 + fractions)
    equity_with_start = np.concatenate(([1.0], equity))
    peaks = np.maximum.accumulate(equity_with_start)
    drawdown = 1.0 - equity_with_start / peaks
    trade_net_array = np.asarray(trade_net, dtype=np.float64)
    positive_sum = float(np.sum(trade_net_array[trade_net_array > 0.0]))
    negative_sum = float(-np.sum(trade_net_array[trade_net_array < 0.0]))
    profit_factor = positive_sum / negative_sum if negative_sum > 0.0 else None
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
        monthly.append(
            {
                "candidate_id": candidate_id,
                "scenario": scenario,
                "month": month,
                "five_minute_steps": int(np.count_nonzero(mask)),
                "trades": sum(
                    1 for trade in trades if _month(trade.decision_time_ms) == month
                ),
                "total_net_return_fraction": float(
                    np.prod(1.0 + fractions[mask]) - 1.0
                ),
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
        "trades_by_horizon": {str(PRIMARY_HORIZON_MINUTES): len(trades)},
        "long_trades": sum(1 for trade in trades if trade.side == 1),
        "short_trades": sum(1 for trade in trades if trade.side == -1),
        "active_days": len(per_day_counts),
        "median_trades_per_active_day": (
            float(np.median(list(per_day_counts.values()))) if per_day_counts else 0.0
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
            seed=49_000 + (0 if scenario == "base" else 1),
        ),
    }
    return ActionHurdleReplayResult(
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
    base: ActionHurdleReplayResult,
    stress: ActionHurdleReplayResult,
    *,
    quality_passed: bool,
) -> dict[str, object]:
    reasons: list[str] = []
    stress_bootstrap = stress.metrics["bootstrap_mean_five_minute_portfolio_bps"]
    if (
        not isinstance(stress_bootstrap, Mapping)
        or float(stress_bootstrap["lower_bps"]) <= 0.0
    ):
        reasons.append("stress_familywise_bootstrap_lower_not_positive")
    if int(base.metrics["positive_months"]) < 4:
        reasons.append("fewer_than_four_positive_months")
    if int(base.metrics["active_days"]) < 90:
        reasons.append("fewer_than_ninety_active_days")
    if int(base.metrics["trades"]) < 180:
        reasons.append("fewer_than_one_hundred_eighty_closed_trades")
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
    if float(base.metrics["maximum_single_symbol_fraction_of_absolute_net_pnl"]) > 0.5:
        reasons.append("single_symbol_absolute_net_pnl_fraction_exceeds_half")
    if not quality_passed:
        reasons.append("numerical_action_or_mechanism_quality_gate_failed")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "selection_contaminated": True,
        "promotion_permitted": False,
    }


__all__ = [
    "ActionHurdleReplayResult",
    "ActionHurdleTrade",
    "BOOTSTRAP_BLOCK_STEPS",
    "BOOTSTRAP_SAMPLES",
    "FAMILYWISE_LOWER_QUANTILE",
    "STRESS_EXECUTION_CHARGE_BPS",
    "candidate_diagnostics",
    "economic_gate",
    "mechanism_ablation_gate",
    "replay_fixed_trades",
    "select_fixed_policy_trades",
]
