"""Hash-bound predictive gates for Round 57 make/take candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np

from .make_take_action_features import MAKE_TAKE_ACTION_NAMES
from .make_take_payoff_lightgbm import (
    MakeTakeConditionalPayoffPredictionBatch,
    TrainedMakeTakePayoffLightGBMModel,
    predict_make_take_conditional_payoff_panel,
    validate_make_take_payoff_lightgbm_model,
)
from .make_take_payoff_panel import (
    MAKE_TAKE_PAYOFF_SYMBOLS,
    MakeTakeConditionalPayoffPanel,
    validate_make_take_conditional_payoff_panel,
)
from .queue_fill_lightgbm import (
    QueueFillPredictionBatch,
    TrainedQueueFillLightGBMModel,
    predict_queue_fill_lightgbm_model,
    validate_queue_fill_lightgbm_model,
)
from .queue_fill_survival import (
    PassiveFillSurvivalPanel,
    validate_passive_fill_survival_panel,
)


MAKE_TAKE_PREDICTIVE_EVALUATION_SCHEMA_VERSION = "make-take-predictive-evaluation-v1"
MAKE_TAKE_PREDICTIVE_EVALUATION_ROLES = ("policy_calibration", "evaluation")
_PASSIVE_SIDE_NAMES = {1: "long", -1: "short"}


@dataclass(frozen=True)
class FillPredictiveMetric:
    symbol: str
    side: str
    rows: int
    log_loss: float
    baseline_log_loss: float
    log_loss_skill: float
    integrated_brier: float
    baseline_integrated_brier: float
    integrated_brier_skill: float
    predicted_fill_probability: float
    observed_fill_ratio: float
    absolute_calibration_error: float
    passed: bool


@dataclass(frozen=True)
class PayoffPredictiveMetric:
    symbol: str
    action_code: int
    action_name: str
    rows: int
    mean_mse_bps2: float
    baseline_mean_mse_bps2: float
    mean_mse_skill: float
    q20_pinball_bps: float
    baseline_q20_pinball_bps: float
    q20_pinball_skill: float
    spearman: float
    top_quintile_rows: int
    top_quintile_mean_net_bps: float
    top_quintile_mean_markout_5s_bps: float
    top_quintile_mean_markout_15s_bps: float
    passed: bool


@dataclass(frozen=True)
class MakeTakePredictiveEvaluation:
    schema_version: str
    role: str
    fill_model_sha256: str
    payoff_model_sha256: str
    training_fill_panel_sha256_by_symbol: tuple[tuple[str, str], ...]
    evaluation_fill_panel_sha256_by_symbol: tuple[tuple[str, str], ...]
    training_payoff_panel_sha256_by_symbol: tuple[tuple[str, str], ...]
    evaluation_payoff_panel_sha256_by_symbol: tuple[tuple[str, str], ...]
    fill_metrics: tuple[FillPredictiveMetric, ...]
    payoff_metrics: tuple[PayoffPredictiveMetric, ...]
    payoff_early_quality_gate_passed: bool
    predictive_gate_passed: bool
    report_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def evidence(self) -> dict[str, object]:
        return asdict(self)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _report_payload(report: MakeTakePredictiveEvaluation) -> dict[str, object]:
    payload = asdict(report)
    payload.pop("report_sha256")
    return payload


def _ordered_fill_panels(
    panels: Sequence[PassiveFillSurvivalPanel],
    *,
    role: str,
) -> tuple[PassiveFillSurvivalPanel, ...]:
    values = tuple(panels)
    try:
        for panel in values:
            validate_passive_fill_survival_panel(panel)
    except ValueError as exc:
        raise ValueError(f"predictive {role} fill panel is invalid") from exc
    if (
        len(values) != len(MAKE_TAKE_PAYOFF_SYMBOLS)
        or {panel.symbol for panel in values} != set(MAKE_TAKE_PAYOFF_SYMBOLS)
    ):
        raise ValueError(f"predictive {role} fill panel set is invalid")
    return tuple(sorted(values, key=lambda panel: panel.symbol))


def _ordered_payoff_panels(
    panels: Sequence[MakeTakeConditionalPayoffPanel],
    *,
    role: str,
) -> tuple[MakeTakeConditionalPayoffPanel, ...]:
    values = tuple(panels)
    try:
        for panel in values:
            validate_make_take_conditional_payoff_panel(panel)
    except ValueError as exc:
        raise ValueError(f"predictive {role} payoff panel is invalid") from exc
    if (
        len(values) != len(MAKE_TAKE_PAYOFF_SYMBOLS)
        or {panel.symbol for panel in values} != set(MAKE_TAKE_PAYOFF_SYMBOLS)
    ):
        raise ValueError(f"predictive {role} payoff panel set is invalid")
    return tuple(sorted(values, key=lambda panel: panel.symbol))


def _ordered_fill_predictions(
    predictions: Sequence[QueueFillPredictionBatch],
) -> tuple[QueueFillPredictionBatch, ...]:
    values = tuple(predictions)
    for prediction in values:
        prediction.__post_init__()
    if (
        len(values) != len(MAKE_TAKE_PAYOFF_SYMBOLS)
        or {prediction.symbol for prediction in values} != set(MAKE_TAKE_PAYOFF_SYMBOLS)
    ):
        raise ValueError("predictive fill prediction set is invalid")
    return tuple(sorted(values, key=lambda prediction: prediction.symbol))


def _ordered_payoff_predictions(
    predictions: Sequence[MakeTakeConditionalPayoffPredictionBatch],
) -> tuple[MakeTakeConditionalPayoffPredictionBatch, ...]:
    values = tuple(predictions)
    for prediction in values:
        prediction.__post_init__()
    if (
        len(values) != len(MAKE_TAKE_PAYOFF_SYMBOLS)
        or {prediction.symbol for prediction in values} != set(MAKE_TAKE_PAYOFF_SYMBOLS)
    ):
        raise ValueError("predictive payoff prediction set is invalid")
    return tuple(sorted(values, key=lambda prediction: prediction.symbol))


def _fill_metric(
    training: PassiveFillSurvivalPanel,
    evaluation: PassiveFillSurvivalPanel,
    prediction: QueueFillPredictionBatch,
    *,
    side: int,
) -> FillPredictiveMetric:
    train_mask = training.action_side == side
    evaluate_mask = evaluation.action_side == side
    train_bucket = training.fill_bucket[train_mask]
    evaluate_bucket = evaluation.fill_bucket[evaluate_mask]
    probabilities = prediction.bucket_probabilities[evaluate_mask]
    counts = np.asarray(
        [
            np.count_nonzero(train_bucket == 1),
            np.count_nonzero(train_bucket == 2),
            np.count_nonzero(train_bucket == 3),
            np.count_nonzero(train_bucket == 0),
        ],
        dtype=np.float64,
    )
    baseline = (counts + 1.0) / (train_bucket.size + 4.0)
    class_index = np.where(evaluate_bucket == 0, 3, evaluate_bucket - 1).astype(np.int64)
    rows = int(evaluate_bucket.size)
    row_index = np.arange(rows)
    epsilon = 1e-12
    log_loss = float(
        -np.mean(np.log(np.clip(probabilities[row_index, class_index], epsilon, 1.0)))
    )
    baseline_log_loss = float(
        -np.mean(np.log(np.clip(baseline[class_index], epsilon, 1.0)))
    )
    cumulative = np.cumsum(probabilities[:, :3], axis=1)
    baseline_cumulative = np.cumsum(baseline[:3])
    observed = np.column_stack(
        tuple(
            (evaluate_bucket > 0) & (evaluate_bucket <= bucket)
            for bucket in (1, 2, 3)
        )
    ).astype(np.float64)
    integrated_brier = float(np.mean((cumulative - observed) ** 2))
    baseline_integrated_brier = float(
        np.mean((baseline_cumulative[None, :] - observed) ** 2)
    )
    log_skill = 1.0 - log_loss / baseline_log_loss
    brier_skill = 1.0 - integrated_brier / baseline_integrated_brier
    predicted_fill = float(np.mean(cumulative[:, -1]))
    observed_fill = float(np.mean(evaluate_bucket > 0))
    return FillPredictiveMetric(
        symbol=evaluation.symbol,
        side=_PASSIVE_SIDE_NAMES[side],
        rows=rows,
        log_loss=log_loss,
        baseline_log_loss=baseline_log_loss,
        log_loss_skill=log_skill,
        integrated_brier=integrated_brier,
        baseline_integrated_brier=baseline_integrated_brier,
        integrated_brier_skill=brier_skill,
        predicted_fill_probability=predicted_fill,
        observed_fill_ratio=observed_fill,
        absolute_calibration_error=abs(predicted_fill - observed_fill),
        passed=bool(log_skill > 0.0 and brier_skill > 0.0),
    )


def _pinball(truth: np.ndarray, prediction: np.ndarray, alpha: float) -> float:
    residual = truth - prediction
    return float(np.mean(np.maximum(alpha * residual, (alpha - 1.0) * residual)))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    ordering = np.argsort(data, kind="stable")
    ranks = np.empty(data.size, dtype=np.float64)
    position = 0
    while position < data.size:
        stop = position + 1
        while stop < data.size and data[ordering[stop]] == data[ordering[position]]:
            stop += 1
        ranks[ordering[position:stop]] = 0.5 * (position + stop - 1)
        position = stop
    return ranks


def _spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_rank = _average_ranks(truth)
    prediction_rank = _average_ranks(prediction)
    if np.std(truth_rank) == 0.0 or np.std(prediction_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(truth_rank, prediction_rank)[0, 1])


def _payoff_metric(
    training: MakeTakeConditionalPayoffPanel,
    evaluation: MakeTakeConditionalPayoffPanel,
    prediction: MakeTakeConditionalPayoffPredictionBatch,
    *,
    action: int,
    alpha: float = 0.20,
) -> PayoffPredictiveMetric:
    train_mask = training.action_code == action
    evaluate_mask = evaluation.action_code == action
    train_truth = training.net_bps[train_mask]
    truth = evaluation.net_bps[evaluate_mask]
    mean_prediction = prediction.conditional_mean_bps[evaluate_mask]
    q20_prediction = prediction.conditional_q20_bps[evaluate_mask]
    baseline_mean = float(np.mean(train_truth, dtype=np.float64))
    baseline_q20 = float(np.quantile(train_truth, alpha, method="higher"))
    mean_mse = float(np.mean(np.square(truth - mean_prediction)))
    baseline_mean_mse = float(np.mean(np.square(truth - baseline_mean)))
    q20_pinball = _pinball(truth, q20_prediction, alpha)
    baseline_q20_pinball = _pinball(
        truth,
        np.full(truth.size, baseline_q20),
        alpha,
    )
    mse_skill = 1.0 - mean_mse / baseline_mean_mse if baseline_mean_mse > 0.0 else 0.0
    pinball_skill = (
        1.0 - q20_pinball / baseline_q20_pinball
        if baseline_q20_pinball > 0.0
        else 0.0
    )
    spearman = _spearman(truth, mean_prediction)
    threshold = float(np.quantile(mean_prediction, 0.80, method="higher"))
    top = mean_prediction >= threshold
    if not np.any(top):
        raise RuntimeError("predictive payoff top quintile is empty")
    markout_5s = evaluation.markout_5s_bps[evaluate_mask]
    markout_15s = evaluation.markout_15s_bps[evaluate_mask]
    return PayoffPredictiveMetric(
        symbol=evaluation.symbol,
        action_code=action,
        action_name=MAKE_TAKE_ACTION_NAMES[action],
        rows=int(truth.size),
        mean_mse_bps2=mean_mse,
        baseline_mean_mse_bps2=baseline_mean_mse,
        mean_mse_skill=mse_skill,
        q20_pinball_bps=q20_pinball,
        baseline_q20_pinball_bps=baseline_q20_pinball,
        q20_pinball_skill=pinball_skill,
        spearman=spearman,
        top_quintile_rows=int(np.count_nonzero(top)),
        top_quintile_mean_net_bps=float(np.mean(truth[top])),
        top_quintile_mean_markout_5s_bps=float(np.mean(markout_5s[top])),
        top_quintile_mean_markout_15s_bps=float(np.mean(markout_15s[top])),
        passed=bool(mse_skill > 0.0 and pinball_skill > 0.0 and spearman >= 0.02),
    )


def validate_make_take_predictive_evaluation(
    report: MakeTakePredictiveEvaluation,
) -> None:
    expected_fill = len(MAKE_TAKE_PAYOFF_SYMBOLS) * 2
    expected_payoff = len(MAKE_TAKE_PAYOFF_SYMBOLS) * 4
    maps = (
        report.training_fill_panel_sha256_by_symbol,
        report.evaluation_fill_panel_sha256_by_symbol,
        report.training_payoff_panel_sha256_by_symbol,
        report.evaluation_payoff_panel_sha256_by_symbol,
    )
    numeric = [
        value
        for metric in (*report.fill_metrics, *report.payoff_metrics)
        for name, value in metric.__dict__.items()
        if name not in {"symbol", "side", "action_name", "passed"}
    ]
    expected_fill_keys = tuple(
        (symbol, side)
        for symbol in sorted(MAKE_TAKE_PAYOFF_SYMBOLS)
        for side in ("long", "short")
    )
    expected_payoff_keys = tuple(
        (symbol, action, MAKE_TAKE_ACTION_NAMES[action])
        for symbol in sorted(MAKE_TAKE_PAYOFF_SYMBOLS)
        for action in range(4)
    )
    expected_gate = bool(
        report.payoff_early_quality_gate_passed
        and all(metric.passed for metric in report.fill_metrics)
        and all(
            metric.passed
            for metric in report.payoff_metrics
            if metric.action_code < 2
        )
    )
    if (
        report.schema_version != MAKE_TAKE_PREDICTIVE_EVALUATION_SCHEMA_VERSION
        or report.role not in MAKE_TAKE_PREDICTIVE_EVALUATION_ROLES
        or not _is_sha256(report.fill_model_sha256)
        or not _is_sha256(report.payoff_model_sha256)
        or any(
            tuple(symbol for symbol, _sha in values)
            != tuple(sorted(MAKE_TAKE_PAYOFF_SYMBOLS))
            or any(not _is_sha256(sha) for _symbol, sha in values)
            for values in maps
        )
        or len(report.fill_metrics) != expected_fill
        or len(report.payoff_metrics) != expected_payoff
        or tuple((metric.symbol, metric.side) for metric in report.fill_metrics)
        != expected_fill_keys
        or tuple(
            (metric.symbol, metric.action_code, metric.action_name)
            for metric in report.payoff_metrics
        )
        != expected_payoff_keys
        or any(
            metric.rows <= 0
            or metric.baseline_log_loss <= 0.0
            or metric.baseline_integrated_brier <= 0.0
            or metric.passed
            is not bool(
                metric.log_loss_skill > 0.0
                and metric.integrated_brier_skill > 0.0
            )
            for metric in report.fill_metrics
        )
        or any(
            metric.rows <= 0
            or metric.top_quintile_rows <= 0
            or metric.baseline_mean_mse_bps2 <= 0.0
            or metric.baseline_q20_pinball_bps <= 0.0
            or metric.passed
            is not bool(
                metric.mean_mse_skill > 0.0
                and metric.q20_pinball_skill > 0.0
                and metric.spearman >= 0.02
            )
            for metric in report.payoff_metrics
        )
        or not all(math.isfinite(float(value)) for value in numeric)
        or not isinstance(report.payoff_early_quality_gate_passed, bool)
        or not isinstance(report.predictive_gate_passed, bool)
        or report.predictive_gate_passed is not expected_gate
        or report.trading_authority is not False
        or report.execution_claim is not False
        or report.profitability_claim is not False
        or report.portfolio_claim is not False
        or report.leverage_applied is not False
        or not _is_sha256(report.report_sha256)
        or report.report_sha256 != _sha256(_report_payload(report))
    ):
        raise ValueError("make/take predictive evaluation is invalid")


def build_make_take_predictive_evaluation(
    *,
    role: str,
    fill_model: TrainedQueueFillLightGBMModel,
    payoff_model: TrainedMakeTakePayoffLightGBMModel,
    training_fill_panels: Sequence[PassiveFillSurvivalPanel],
    evaluation_fill_panels: Sequence[PassiveFillSurvivalPanel],
    training_payoff_panels: Sequence[MakeTakeConditionalPayoffPanel],
    evaluation_payoff_panels: Sequence[MakeTakeConditionalPayoffPanel],
) -> MakeTakePredictiveEvaluation:
    """Apply every precommitted proper-score and rank gate by symbol/action."""

    if role not in MAKE_TAKE_PREDICTIVE_EVALUATION_ROLES:
        raise ValueError("make/take predictive evaluation role is invalid")
    validate_queue_fill_lightgbm_model(fill_model, reload=True)
    validate_make_take_payoff_lightgbm_model(payoff_model, reload=True)
    train_fill = _ordered_fill_panels(training_fill_panels, role="training")
    evaluate_fill = _ordered_fill_panels(evaluation_fill_panels, role=role)
    train_payoff = _ordered_payoff_panels(training_payoff_panels, role="training")
    evaluate_payoff = _ordered_payoff_panels(evaluation_payoff_panels, role=role)
    if (
        tuple((panel.symbol, panel.panel_sha256) for panel in train_fill)
        != fill_model.training_panel_sha256_by_symbol
        or tuple((panel.symbol, panel.panel_sha256) for panel in train_payoff)
        != payoff_model.training_panel_sha256_by_symbol
        or tuple((panel.symbol, panel.source_dataset_sha256) for panel in evaluate_fill)
        != fill_model.source_dataset_sha256_by_symbol
        or tuple((panel.symbol, panel.source_dataset_sha256) for panel in evaluate_payoff)
        != payoff_model.source_dataset_sha256_by_symbol
    ):
        raise ValueError("predictive model training or source identity drifted")
    fill_prediction = _ordered_fill_predictions(
        tuple(
            predict_queue_fill_lightgbm_model(fill_model, panel)
            for panel in evaluate_fill
        )
    )
    payoff_prediction = _ordered_payoff_predictions(
        tuple(
            predict_make_take_conditional_payoff_panel(payoff_model, panel)
            for panel in evaluate_payoff
        )
    )
    fill_metrics: list[FillPredictiveMetric] = []
    payoff_metrics: list[PayoffPredictiveMetric] = []
    for index, symbol in enumerate(sorted(MAKE_TAKE_PAYOFF_SYMBOLS)):
        fill_train = train_fill[index]
        fill_evaluation = evaluate_fill[index]
        fill_result = fill_prediction[index]
        payoff_train = train_payoff[index]
        payoff_evaluation = evaluate_payoff[index]
        payoff_result = payoff_prediction[index]
        if (
            fill_train.symbol != symbol
            or fill_evaluation.symbol != symbol
            or fill_result.symbol != symbol
            or payoff_train.symbol != symbol
            or payoff_evaluation.symbol != symbol
            or payoff_result.symbol != symbol
            or fill_train.source_dataset_sha256 != fill_evaluation.source_dataset_sha256
            or payoff_train.source_dataset_sha256 != payoff_evaluation.source_dataset_sha256
            or fill_evaluation.source_dataset_sha256
            != payoff_evaluation.source_dataset_sha256
            or fill_train.source_last_decision_time_ms + 15_000
            >= fill_evaluation.source_first_decision_time_ms
            or payoff_train.source_label_end_ms
            >= payoff_evaluation.source_first_decision_time_ms
            or fill_result.source_panel_sha256 != fill_evaluation.panel_sha256
            or payoff_result.source_panel_sha256 != payoff_evaluation.panel_sha256
            or fill_result.rows != fill_evaluation.rows
            or payoff_result.rows != payoff_evaluation.rows
            or not np.array_equal(fill_result.event_index, fill_evaluation.event_index)
            or not np.array_equal(
                fill_result.decision_time_ms,
                fill_evaluation.decision_time_ms,
            )
            or not np.array_equal(fill_result.action_side, fill_evaluation.action_side)
            or not np.array_equal(payoff_result.action_code, payoff_evaluation.action_code)
            or not np.array_equal(payoff_result.action_side, payoff_evaluation.action_side)
        ):
            raise ValueError("predictive source or chronology contract drifted")
        for side in (1, -1):
            fill_metrics.append(
                _fill_metric(fill_train, fill_evaluation, fill_result, side=side)
            )
        for action in range(4):
            payoff_metrics.append(
                _payoff_metric(
                    payoff_train,
                    payoff_evaluation,
                    payoff_result,
                    action=action,
                )
            )
    passive_payoff = tuple(metric for metric in payoff_metrics if metric.action_code < 2)
    predictive_gate = bool(
        payoff_model.early_quality.quality_gate_passed
        and all(metric.passed for metric in fill_metrics)
        and all(metric.passed for metric in passive_payoff)
    )
    provisional = MakeTakePredictiveEvaluation(
        schema_version=MAKE_TAKE_PREDICTIVE_EVALUATION_SCHEMA_VERSION,
        role=role,
        fill_model_sha256=fill_model.model_sha256,
        payoff_model_sha256=payoff_model.model_sha256,
        training_fill_panel_sha256_by_symbol=tuple(
            (panel.symbol, panel.panel_sha256) for panel in train_fill
        ),
        evaluation_fill_panel_sha256_by_symbol=tuple(
            (panel.symbol, panel.panel_sha256) for panel in evaluate_fill
        ),
        training_payoff_panel_sha256_by_symbol=tuple(
            (panel.symbol, panel.panel_sha256) for panel in train_payoff
        ),
        evaluation_payoff_panel_sha256_by_symbol=tuple(
            (panel.symbol, panel.panel_sha256) for panel in evaluate_payoff
        ),
        fill_metrics=tuple(fill_metrics),
        payoff_metrics=tuple(payoff_metrics),
        payoff_early_quality_gate_passed=(
            payoff_model.early_quality.quality_gate_passed
        ),
        predictive_gate_passed=predictive_gate,
        report_sha256="",
    )
    report = MakeTakePredictiveEvaluation(
        **{**provisional.__dict__, "report_sha256": _sha256(_report_payload(provisional))}
    )
    validate_make_take_predictive_evaluation(report)
    return report


__all__ = [
    "MAKE_TAKE_PREDICTIVE_EVALUATION_ROLES",
    "MAKE_TAKE_PREDICTIVE_EVALUATION_SCHEMA_VERSION",
    "FillPredictiveMetric",
    "MakeTakePredictiveEvaluation",
    "PayoffPredictiveMetric",
    "build_make_take_predictive_evaluation",
    "validate_make_take_predictive_evaluation",
]
