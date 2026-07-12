"""Utility-weighted, mirror-equivariant three-action LightGBM model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from .lightgbm_backend import lightgbm_backend_parameters
from .microstructure_action_architecture import (
    ActionValueEnsembleBatch,
    ActionValuePredictionBatch,
    ensemble_action_value_predictions,
)
from .microstructure_action_features import (
    ACTION_CANONICALIZATION_SHA256,
    ACTION_CONDITIONAL_FEATURE_NAMES,
    ACTION_FEATURE_SCHEMA_VERSION,
    build_action_conditional_features,
    mirror_microstructure_direction,
)
from .microstructure_barriers import (
    ADAPTIVE_BARRIER_SCHEMA_VERSION,
    ADAPTIVE_BARRIER_TARGET_MODE,
    AdaptiveBarrierSpec,
    AdaptiveBarrierTargets,
    validate_adaptive_barrier_targets,
)
from .microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
    validate_microstructure_dataset,
)
from .microstructure_outcome_lightgbm import (
    _target_arrays,
    _target_contract_sha256,
    _validate_indexes,
    _validate_weights,
)
from .microstructure_selective_action_lightgbm import (
    SelectiveActionEnsembleBatch,
)
from .microstructure_shared_action_lightgbm import (
    _paired_values,
    _paired_weights,
    _train_booster,
)
from .probability_calibration import apply_platt_scaling, fit_platt_scaling
from .storage import write_json_atomic


THREE_ACTION_LIGHTGBM_SCHEMA_VERSION = "three-action-lightgbm-hurdle-v2"
_MODEL_FAMILY = "utility_weighted_symmetric_three_action_lightgbm_hurdle"
_CLASS_NAMES = ("long_rows", "abstain_rows", "short_rows")
_LONG_CLASS = 0
_ABSTAIN_CLASS = 1
_SHORT_CLASS = 2
_HEADS = (
    "three_action_probability",
    "side_profit_probability",
    "positive_magnitude",
    "nonpositive_loss_magnitude",
    "lower_quantile",
    "upper_quantile",
)
_MINIMUM_TRAIN_CLASS_ROWS = 512
_MINIMUM_EARLY_CLASS_ROWS = 128
_MINIMUM_CALIBRATION_CLASS_ROWS = 256
_MINIMUM_SIDE_TRAIN_CLASS_ROWS = 512
_MINIMUM_SIDE_EARLY_CLASS_ROWS = 128
_MINIMUM_SIDE_CALIBRATION_CLASS_ROWS = 256
_MINIMUM_CONDITIONAL_ROWS = 128
_MINIMUM_REGRET_MULTIPLIER = 0.5
_MAXIMUM_REGRET_MULTIPLIER = 3.0
_MINIMUM_LOG_TEMPERATURE = -4.0
_MAXIMUM_LOG_TEMPERATURE = 4.0
_MINIMUM_ABSTAIN_BIAS = -5.0
_MAXIMUM_ABSTAIN_BIAS = 5.0
_CALIBRATION_MAXIMUM_ITERATIONS = 100
_CALIBRATION_GRADIENT_TOLERANCE = 1e-10
_CALIBRATION_HESSIAN_RIDGE = 1e-12
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
ThreeActionProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class ThreeActionLightGBMSpec:
    """Precommitted multiclass and conditional-distribution controls."""

    candidate_id: str
    family: str
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l1: float
    lambda_l2: float
    max_bin: int
    num_boost_round: int
    early_stopping_rounds: int
    lower_quantile: float
    upper_quantile: float
    calibration_fraction: float
    gpu_use_dp_required: bool

    def __post_init__(self) -> None:
        integral = (
            self.num_leaves,
            self.max_depth,
            self.min_data_in_leaf,
            self.bagging_freq,
            self.max_bin,
            self.num_boost_round,
            self.early_stopping_rounds,
        )
        numeric = (
            self.learning_rate,
            self.feature_fraction,
            self.bagging_fraction,
            self.lambda_l1,
            self.lambda_l2,
            self.lower_quantile,
            self.upper_quantile,
            self.calibration_fraction,
        )
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id.strip()
            or self.family != _MODEL_FAMILY
            or any(
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                for value in integral
            )
            or any(isinstance(value, (bool, np.bool_)) for value in numeric)
            or not all(math.isfinite(float(value)) for value in numeric)
            or not 0.0 < self.learning_rate <= 0.25
            or not 2 <= int(self.num_leaves) <= 255
            or not 1 <= int(self.max_depth) <= 16
            or not 32 <= int(self.min_data_in_leaf) <= 8_192
            or not 0.0 < self.feature_fraction <= 1.0
            or not 0.0 < self.bagging_fraction <= 1.0
            or not 0 <= int(self.bagging_freq) <= 100
            or self.lambda_l1 < 0.0
            or self.lambda_l2 < 0.0
            or not 31 <= int(self.max_bin) <= 255
            or not 10 <= int(self.num_boost_round) <= 10_000
            or not 5 <= int(self.early_stopping_rounds) < int(self.num_boost_round)
            or not 0.0 < self.lower_quantile < 0.5
            or not 0.5 < self.upper_quantile < 1.0
            or not 0.25 <= self.calibration_fraction <= 0.75
            or self.gpu_use_dp_required is not True
        ):
            raise ValueError("three-action LightGBM specification is invalid")


@dataclass(frozen=True)
class TrainedThreeActionLightGBMModel:
    """Hash-bound three-action member without execution authority."""

    schema_version: str
    model_family: str
    spec: ThreeActionLightGBMSpec
    source_feature_version: str
    source_feature_names: tuple[str, ...]
    action_feature_schema_version: str
    action_canonicalization_sha256: str
    action_feature_names: tuple[str, ...]
    target_schema_version: str
    target_mode: str
    target_spec: AdaptiveBarrierSpec
    target_contract_sha256: str
    target_scenario: str
    backend_requested: str
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    seed: int
    training_event_rows: int
    requested_tuning_event_rows: int
    early_stop_event_rows: int
    calibration_event_rows: int
    calibration_start_ms: int
    internal_purged_event_rows: int
    action_class_support: Mapping[str, Mapping[str, int]]
    side_profit_class_support: Mapping[str, Mapping[str, int]]
    regret_scale_bps: float
    minimum_regret_multiplier: float
    maximum_regret_multiplier: float
    training_regret_multiplier_mean: float
    early_stop_regret_multiplier_mean: float
    calibration_temperature: float
    calibration_abstain_logit_bias: float
    calibration_iterations: int
    calibration_gradient_norm: float
    calibration_multiclass_log_loss: float
    calibration_class_prior_log_loss: float
    side_profit_probability_calibration: tuple[float, float]
    best_iterations: Mapping[str, int]
    model_strings: Mapping[str, str]
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False


@dataclass(frozen=True)
class ThreeActionPredictionBatch:
    """Separate decision-class probabilities from side-profit probabilities."""

    endpoint_indexes: np.ndarray
    long_mean_bps: np.ndarray
    short_mean_bps: np.ndarray
    long_profitable_probability: np.ndarray
    short_profitable_probability: np.ndarray
    long_action_probability: np.ndarray
    abstain_action_probability: np.ndarray
    short_action_probability: np.ndarray
    opportunity_probability: np.ndarray
    conditional_long_probability: np.ndarray
    long_lower_bps: np.ndarray
    short_lower_bps: np.ndarray
    long_upper_bps: np.ndarray
    short_upper_bps: np.ndarray
    action_preference_side: np.ndarray
    decision_preference_side: np.ndarray
    side_consensus: np.ndarray

    def __post_init__(self) -> None:
        endpoints = np.asarray(self.endpoint_indexes, dtype=np.int64)
        rows = len(endpoints)
        numeric = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (
                self.long_mean_bps,
                self.short_mean_bps,
                self.long_profitable_probability,
                self.short_profitable_probability,
                self.long_action_probability,
                self.abstain_action_probability,
                self.short_action_probability,
                self.opportunity_probability,
                self.conditional_long_probability,
                self.long_lower_bps,
                self.short_lower_bps,
                self.long_upper_bps,
                self.short_upper_bps,
            )
        )
        (
            long_mean,
            short_mean,
            long_profitable,
            short_profitable,
            long_action,
            abstain_action,
            short_action,
            opportunity,
            conditional_long,
            long_lower,
            short_lower,
            long_upper,
            short_upper,
        ) = numeric
        action_side = np.asarray(self.action_preference_side, dtype=np.int8)
        decision_side = np.asarray(self.decision_preference_side, dtype=np.int8)
        consensus = np.asarray(self.side_consensus)
        probabilities = numeric[2:9]
        if (
            rows <= 0
            or endpoints.ndim != 1
            or endpoints[0] < 0
            or np.any(np.diff(endpoints) <= 0)
            or any(value.shape != (rows,) for value in numeric)
            or any(not np.all(np.isfinite(value)) for value in numeric)
            or any(
                np.any(value < 0.0) or np.any(value > 1.0) for value in probabilities
            )
            or not np.allclose(
                long_action + abstain_action + short_action,
                1.0,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                long_action + short_action,
                opportunity,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                long_action,
                opportunity * conditional_long,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                short_action,
                opportunity * (1.0 - conditional_long),
                rtol=0.0,
                atol=1e-12,
            )
            or np.any(long_lower > long_upper)
            or np.any(short_lower > short_upper)
            or action_side.shape != (rows,)
            or decision_side.shape != (rows,)
            or consensus.shape != (rows,)
            or consensus.dtype != np.bool_
            or not set(np.unique(action_side)).issubset({-1, 0, 1})
            or not set(np.unique(decision_side)).issubset({-1, 0, 1})
            or not np.array_equal(action_side, np.sign(long_mean - short_mean))
            or not np.array_equal(
                decision_side,
                np.sign(long_action - short_action),
            )
            or not np.array_equal(
                consensus,
                (action_side != 0) & (action_side == decision_side),
            )
        ):
            raise ValueError("three-action prediction batch is invalid")

    @property
    def rows(self) -> int:
        return len(self.endpoint_indexes)


@dataclass(frozen=True)
class ThreeActionEnsembleBatch:
    """Action values plus complete three-action routing evidence."""

    action_values: ActionValueEnsembleBatch
    long_action_probability_mean: np.ndarray
    abstain_action_probability_mean: np.ndarray
    short_action_probability_mean: np.ndarray
    opportunity_probability_mean: np.ndarray
    opportunity_probability_std: np.ndarray
    conditional_long_probability_mean: np.ndarray
    conditional_long_probability_std: np.ndarray
    long_action_member_probabilities: np.ndarray
    abstain_action_member_probabilities: np.ndarray
    short_action_member_probabilities: np.ndarray
    opportunity_member_probabilities: np.ndarray
    conditional_long_member_probabilities: np.ndarray
    direction_long_member_ratio: np.ndarray
    direction_short_member_ratio: np.ndarray
    side_consensus_member_ratio: np.ndarray
    member_count: int
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def __post_init__(self) -> None:
        rows = self.action_values.rows
        vectors = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (
                self.long_action_probability_mean,
                self.abstain_action_probability_mean,
                self.short_action_probability_mean,
                self.opportunity_probability_mean,
                self.opportunity_probability_std,
                self.conditional_long_probability_mean,
                self.conditional_long_probability_std,
                self.direction_long_member_ratio,
                self.direction_short_member_ratio,
                self.side_consensus_member_ratio,
            )
        )
        matrices = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (
                self.long_action_member_probabilities,
                self.abstain_action_member_probabilities,
                self.short_action_member_probabilities,
                self.opportunity_member_probabilities,
                self.conditional_long_member_probabilities,
            )
        )
        (
            long_mean,
            abstain_mean,
            short_mean,
            opportunity_mean,
            opportunity_std,
            conditional_mean,
            conditional_std,
            direction_long_ratio,
            direction_short_ratio,
            consensus_ratio,
        ) = vectors
        (
            long_members,
            abstain_members,
            short_members,
            opportunity_members,
            direction_members,
        ) = matrices
        authority = (
            self.trading_authority,
            self.execution_claim,
            self.profitability_claim,
            self.portfolio_claim,
            self.leverage_applied,
        )
        if (
            self.member_count < 2
            or self.member_count != self.action_values.member_count
            or any(value.shape != (rows,) for value in vectors)
            or any(value.shape != (self.member_count, rows) for value in matrices)
            or any(not np.all(np.isfinite(value)) for value in (*vectors, *matrices))
            or any(
                np.any(value < 0.0) or np.any(value > 1.0)
                for value in (
                    long_mean,
                    abstain_mean,
                    short_mean,
                    opportunity_mean,
                    conditional_mean,
                    direction_long_ratio,
                    direction_short_ratio,
                    consensus_ratio,
                    *matrices,
                )
            )
            or np.any(opportunity_std < 0.0)
            or np.any(conditional_std < 0.0)
            or not np.allclose(
                long_members + abstain_members + short_members,
                1.0,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                opportunity_members,
                long_members + short_members,
                rtol=0.0,
                atol=1e-12,
            )
            or np.any(opportunity_members <= 0.0)
            or not np.allclose(
                long_members,
                opportunity_members * direction_members,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                long_mean,
                np.mean(long_members, axis=0),
                rtol=0.0,
                atol=1e-15,
            )
            or not np.allclose(
                abstain_mean,
                np.mean(abstain_members, axis=0),
                rtol=0.0,
                atol=1e-15,
            )
            or not np.allclose(
                short_mean,
                np.mean(short_members, axis=0),
                rtol=0.0,
                atol=1e-15,
            )
            or not np.allclose(
                opportunity_mean,
                np.mean(opportunity_members, axis=0),
                rtol=0.0,
                atol=1e-15,
            )
            or not np.allclose(
                opportunity_std,
                np.std(opportunity_members, axis=0),
                rtol=0.0,
                atol=1e-15,
            )
            or not np.allclose(
                conditional_mean,
                np.mean(direction_members, axis=0),
                rtol=0.0,
                atol=1e-15,
            )
            or not np.allclose(
                conditional_std,
                np.std(direction_members, axis=0),
                rtol=0.0,
                atol=1e-15,
            )
            or not np.array_equal(
                direction_long_ratio, np.mean(direction_members > 0.5, axis=0)
            )
            or not np.array_equal(
                direction_short_ratio, np.mean(direction_members < 0.5, axis=0)
            )
            or any(authority)
        ):
            raise ValueError("three-action ensemble batch is invalid")

    @property
    def endpoint_indexes(self) -> np.ndarray:
        return self.action_values.endpoint_indexes

    @property
    def rows(self) -> int:
        return self.action_values.rows


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _model_payload(model: TrainedThreeActionLightGBMModel) -> dict[str, object]:
    value = asdict(model)
    value.pop("model_sha256")
    return value


def _model_sha256(model: TrainedThreeActionLightGBMModel) -> str:
    return hashlib.sha256(
        _canonical_json(_model_payload(model)).encode("ascii")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("three-action logits are invalid")
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponential = np.maximum(np.exp(shifted), np.finfo(np.float64).tiny)
    probabilities = exponential / np.sum(exponential, axis=1, keepdims=True)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("three-action softmax emitted non-finite probabilities")
    return probabilities


def _action_labels(long_values: np.ndarray, short_values: np.ndarray) -> np.ndarray:
    long_target = np.asarray(long_values, dtype=np.float64)
    short_target = np.asarray(short_values, dtype=np.float64)
    if (
        long_target.ndim != 1
        or short_target.shape != long_target.shape
        or len(long_target) == 0
        or not np.all(np.isfinite(long_target))
        or not np.all(np.isfinite(short_target))
    ):
        raise ValueError("three-action target values are invalid")
    labels = np.full(len(long_target), _ABSTAIN_CLASS, dtype=np.int8)
    labels[(long_target > 0.0) & (long_target > short_target)] = _LONG_CLASS
    labels[(short_target > 0.0) & (short_target > long_target)] = _SHORT_CLASS
    return labels


def _class_support(
    labels: np.ndarray,
    *,
    role: str,
    minimum: int,
) -> dict[str, int]:
    values = np.asarray(labels)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(~np.isin(values, (_LONG_CLASS, _ABSTAIN_CLASS, _SHORT_CLASS)))
    ):
        raise ValueError(f"{role} three-action labels are invalid")
    support = {
        name: int(np.sum(values == index)) for index, name in enumerate(_CLASS_NAMES)
    }
    if min(support.values()) < int(minimum):
        raise ValueError(f"{role} three-action class support is insufficient")
    return support


def _side_profit_class_support(
    labels: np.ndarray,
    *,
    role: str,
    minimum: int,
) -> dict[str, int]:
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0 or values.dtype != np.bool_:
        raise ValueError(f"{role} side-profit labels are invalid")
    support = {
        "profitable_rows": int(np.sum(values)),
        "non_profitable_rows": int(np.sum(~values)),
    }
    if min(support.values()) < int(minimum):
        raise ValueError(f"{role} side-profit class support is insufficient")
    return support


def _decision_regret_span(
    long_values: np.ndarray,
    short_values: np.ndarray,
) -> np.ndarray:
    long_target = np.asarray(long_values, dtype=np.float64)
    short_target = np.asarray(short_values, dtype=np.float64)
    if (
        long_target.ndim != 1
        or short_target.shape != long_target.shape
        or len(long_target) == 0
        or not np.all(np.isfinite(long_target))
        or not np.all(np.isfinite(short_target))
    ):
        raise ValueError("three-action decision utilities are invalid")
    abstain = np.zeros_like(long_target)
    utilities = np.column_stack((long_target, abstain, short_target))
    return np.max(utilities, axis=1) - np.min(utilities, axis=1)


def _regret_multiplier(span: np.ndarray, scale_bps: float) -> np.ndarray:
    values = np.asarray(span, dtype=np.float64)
    scale = float(scale_bps)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError("three-action decision-regret inputs are invalid")
    return np.clip(
        values / scale,
        _MINIMUM_REGRET_MULTIPLIER,
        _MAXIMUM_REGRET_MULTIPLIER,
    )


def _symmetrized_logits(
    booster: lgb.Booster,
    source: np.ndarray,
    *,
    iteration: int,
) -> np.ndarray:
    values = np.asarray(source, dtype=np.float32)
    mirrored = mirror_microstructure_direction(values)
    raw = np.asarray(
        booster.predict(values, num_iteration=iteration, raw_score=True),
        dtype=np.float64,
    ).reshape(len(values), 3)
    reflected = np.asarray(
        booster.predict(mirrored, num_iteration=iteration, raw_score=True),
        dtype=np.float64,
    ).reshape(len(values), 3)
    swapped_reflected = reflected[:, (_SHORT_CLASS, _ABSTAIN_CLASS, _LONG_CLASS)]
    logits = 0.5 * (raw + swapped_reflected)
    if not np.all(np.isfinite(logits)):
        raise ValueError("three-action symmetrized logits are non-finite")
    return logits


def _calibrated_probabilities(
    logits: np.ndarray,
    *,
    temperature: float,
    abstain_logit_bias: float,
) -> np.ndarray:
    calibrated_temperature = float(temperature)
    bias = float(abstain_logit_bias)
    if (
        not math.isfinite(calibrated_temperature)
        or calibrated_temperature <= 0.0
        or not math.isfinite(bias)
    ):
        raise ValueError("three-action calibration parameters are invalid")
    scale = 1.0 / calibrated_temperature
    scores = np.asarray(logits, dtype=np.float64) * scale
    scores = scores.copy()
    scores[:, _ABSTAIN_CLASS] += bias
    return _softmax(scores)


def _multiclass_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    predicted = np.asarray(probabilities, dtype=np.float64)
    raw_target = np.asarray(labels)
    target = np.asarray(raw_target, dtype=np.int64)
    if (
        predicted.ndim != 2
        or predicted.shape != (len(target), 3)
        or len(target) < 256
        or not np.issubdtype(raw_target.dtype, np.integer)
        or np.any(~np.isin(target, (_LONG_CLASS, _ABSTAIN_CLASS, _SHORT_CLASS)))
        or not np.all(np.isfinite(predicted))
        or np.any(predicted < 0.0)
        or np.any(predicted > 1.0)
        or not np.allclose(
            np.sum(predicted, axis=1),
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("three-action log-loss inputs are invalid")
    chosen = np.clip(predicted[np.arange(len(target)), target], 1e-15, 1.0)
    return float(-np.mean(np.log(chosen)))


def _fit_multiclass_calibration(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, int, float, float, float]:
    values = np.asarray(logits, dtype=np.float64)
    raw_target = np.asarray(labels)
    target = np.asarray(raw_target, dtype=np.int64)
    if (
        values.ndim != 2
        or values.shape != (len(target), 3)
        or len(target) < 256
        or not np.issubdtype(raw_target.dtype, np.integer)
        or not np.all(np.isfinite(values))
        or set(np.unique(target)) != {_LONG_CLASS, _ABSTAIN_CLASS, _SHORT_CLASS}
    ):
        raise ValueError("three-action calibration inputs are invalid")
    minimum_scale = math.exp(-_MAXIMUM_LOG_TEMPERATURE)
    maximum_scale = math.exp(-_MINIMUM_LOG_TEMPERATURE)
    scale = 1.0
    bias = 0.0
    iterations = 0
    gradient_norm = math.inf

    def objective(
        current_scale: float,
        current_bias: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        scores = values * current_scale
        scores = scores.copy()
        scores[:, _ABSTAIN_CLASS] += current_bias
        probabilities = _softmax(scores)
        loss = _multiclass_log_loss(probabilities, target)
        expected_logit = np.sum(probabilities * values, axis=1)
        true_logit = values[np.arange(len(target)), target]
        abstain_probability = probabilities[:, _ABSTAIN_CLASS]
        true_abstain = (target == _ABSTAIN_CLASS).astype(np.float64)
        gradient = np.asarray(
            [
                np.mean(expected_logit - true_logit),
                np.mean(abstain_probability - true_abstain),
            ],
            dtype=np.float64,
        )
        expected_squared_logit = np.sum(probabilities * values * values, axis=1)
        hessian_scale = np.mean(
            expected_squared_logit - expected_logit * expected_logit
        )
        hessian_bias = np.mean(abstain_probability * (1.0 - abstain_probability))
        hessian_cross = np.mean(
            abstain_probability * (values[:, _ABSTAIN_CLASS] - expected_logit)
        )
        hessian = np.asarray(
            [
                [hessian_scale, hessian_cross],
                [hessian_cross, hessian_bias],
            ],
            dtype=np.float64,
        )
        hessian.flat[::3] += _CALIBRATION_HESSIAN_RIDGE
        return loss, gradient, hessian

    loss, gradient, hessian = objective(scale, bias)
    for iteration in range(1, _CALIBRATION_MAXIMUM_ITERATIONS + 1):
        iterations = iteration
        gradient_norm = float(np.linalg.norm(gradient, ord=2))
        if gradient_norm <= _CALIBRATION_GRADIENT_TOLERANCE:
            break
        try:
            direction = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("three-action calibration Hessian is singular") from exc
        if not np.all(np.isfinite(direction)):
            raise ValueError("three-action calibration direction is non-finite")
        accepted = False
        step = 1.0
        for _line_search in range(40):
            candidate_scale = float(
                np.clip(
                    scale + step * direction[0],
                    minimum_scale,
                    maximum_scale,
                )
            )
            candidate_bias = float(
                np.clip(
                    bias + step * direction[1],
                    _MINIMUM_ABSTAIN_BIAS,
                    _MAXIMUM_ABSTAIN_BIAS,
                )
            )
            candidate_loss, candidate_gradient, candidate_hessian = objective(
                candidate_scale,
                candidate_bias,
            )
            if candidate_loss <= loss - 1e-12 * step * gradient_norm * gradient_norm:
                scale = candidate_scale
                bias = candidate_bias
                loss = candidate_loss
                gradient = candidate_gradient
                hessian = candidate_hessian
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    gradient_norm = float(np.linalg.norm(gradient, ord=2))
    temperature = 1.0 / scale
    calibrated = _calibrated_probabilities(
        values,
        temperature=temperature,
        abstain_logit_bias=bias,
    )
    calibrated_loss = _multiclass_log_loss(calibrated, target)
    class_counts = np.bincount(target, minlength=3).astype(np.float64)
    class_prior = class_counts / np.sum(class_counts)
    prior_probabilities = np.broadcast_to(class_prior, calibrated.shape)
    prior_loss = _multiclass_log_loss(prior_probabilities, target)
    if (
        not math.isfinite(temperature)
        or not math.exp(_MINIMUM_LOG_TEMPERATURE)
        <= temperature
        <= math.exp(_MAXIMUM_LOG_TEMPERATURE)
        or not _MINIMUM_ABSTAIN_BIAS <= bias <= _MAXIMUM_ABSTAIN_BIAS
        or not all(
            math.isfinite(value)
            for value in (gradient_norm, calibrated_loss, prior_loss)
        )
        or calibrated_loss > _multiclass_log_loss(_softmax(values), target) + 1e-12
    ):
        raise ValueError("three-action multiclass calibration failed")
    return (
        float(temperature),
        float(bias),
        iterations,
        gradient_norm,
        calibrated_loss,
        prior_loss,
    )


def _validate_model_contract(
    model: TrainedThreeActionLightGBMModel,
    *,
    reload_boosters: bool,
) -> None:
    expected_target_hash = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": model.target_schema_version,
                "target_mode": model.target_mode,
                "spec": asdict(model.target_spec),
            }
        ).encode("ascii")
    ).hexdigest()
    authority = (
        model.trading_authority,
        model.execution_claim,
        model.profitability_claim,
        model.portfolio_claim,
        model.leverage_applied,
    )
    if (
        model.schema_version != THREE_ACTION_LIGHTGBM_SCHEMA_VERSION
        or model.model_family != _MODEL_FAMILY
        or model.source_feature_version != MICROSTRUCTURE_FEATURE_VERSION
        or model.source_feature_names != MICROSTRUCTURE_FEATURE_NAMES
        or model.action_feature_schema_version != ACTION_FEATURE_SCHEMA_VERSION
        or model.action_canonicalization_sha256 != ACTION_CANONICALIZATION_SHA256
        or model.action_feature_names != ACTION_CONDITIONAL_FEATURE_NAMES
        or model.target_schema_version != ADAPTIVE_BARRIER_SCHEMA_VERSION
        or model.target_mode != ADAPTIVE_BARRIER_TARGET_MODE
        or model.target_scenario != "stress"
        or model.target_contract_sha256 != expected_target_hash
        or model.backend_kind not in {"cpu", "opencl"}
        or not model.backend_device
        or model.lightgbm_version != str(lgb.__version__)
        or not isinstance(model.seed, int)
        or isinstance(model.seed, bool)
        or min(
            model.training_event_rows,
            model.requested_tuning_event_rows,
            model.early_stop_event_rows,
            model.calibration_event_rows,
        )
        <= 0
        or model.early_stop_event_rows
        + model.calibration_event_rows
        + model.internal_purged_event_rows
        != model.requested_tuning_event_rows
        or model.calibration_start_ms <= 0
        or model.internal_purged_event_rows < 0
        or set(model.best_iterations) != set(_HEADS)
        or set(model.model_strings) != set(_HEADS)
        or any(
            isinstance(value, bool) or not 1 <= int(value) <= model.spec.num_boost_round
            for value in model.best_iterations.values()
        )
        or any(
            not isinstance(value, str) or not value
            for value in model.model_strings.values()
        )
        or not math.isfinite(model.regret_scale_bps)
        or model.regret_scale_bps <= 0.0
        or model.minimum_regret_multiplier != _MINIMUM_REGRET_MULTIPLIER
        or model.maximum_regret_multiplier != _MAXIMUM_REGRET_MULTIPLIER
        or not _MINIMUM_REGRET_MULTIPLIER
        <= model.training_regret_multiplier_mean
        <= _MAXIMUM_REGRET_MULTIPLIER
        or not _MINIMUM_REGRET_MULTIPLIER
        <= model.early_stop_regret_multiplier_mean
        <= _MAXIMUM_REGRET_MULTIPLIER
        or not math.exp(_MINIMUM_LOG_TEMPERATURE)
        <= model.calibration_temperature
        <= math.exp(_MAXIMUM_LOG_TEMPERATURE)
        or not _MINIMUM_ABSTAIN_BIAS
        <= model.calibration_abstain_logit_bias
        <= _MAXIMUM_ABSTAIN_BIAS
        or not 1 <= model.calibration_iterations <= _CALIBRATION_MAXIMUM_ITERATIONS
        or not math.isfinite(model.calibration_gradient_norm)
        or model.calibration_gradient_norm < 0.0
        or not math.isfinite(model.calibration_multiclass_log_loss)
        or model.calibration_multiclass_log_loss <= 0.0
        or not math.isfinite(model.calibration_class_prior_log_loss)
        or model.calibration_class_prior_log_loss <= 0.0
        or len(model.side_profit_probability_calibration) != 2
        or not 0.05 <= float(model.side_profit_probability_calibration[0]) <= 10.0
        or not -10.0 <= float(model.side_profit_probability_calibration[1]) <= 10.0
        or (
            model.backend_kind == "opencl"
            and model.spec.gpu_use_dp_required is not True
        )
        or any(authority)
        or not _is_sha256(model.model_sha256)
        or _model_sha256(model) != model.model_sha256
    ):
        raise ValueError("three-action LightGBM model contract is invalid")
    role_contract = {
        "train": (model.training_event_rows, _MINIMUM_TRAIN_CLASS_ROWS),
        "early_stop": (model.early_stop_event_rows, _MINIMUM_EARLY_CLASS_ROWS),
        "calibration": (
            model.calibration_event_rows,
            _MINIMUM_CALIBRATION_CLASS_ROWS,
        ),
    }
    if set(model.action_class_support) != set(role_contract):
        raise ValueError("three-action class-support roles are invalid")
    for role, (expected_rows, minimum) in role_contract.items():
        support = model.action_class_support[role]
        if (
            set(support) != set(_CLASS_NAMES)
            or any(
                isinstance(value, bool) or int(value) < minimum
                for value in support.values()
            )
            or sum(int(value) for value in support.values()) != expected_rows
        ):
            raise ValueError("three-action class-support values are invalid")
    side_role_contract = {
        "train": (model.training_event_rows * 2, _MINIMUM_SIDE_TRAIN_CLASS_ROWS),
        "early_stop": (
            model.early_stop_event_rows * 2,
            _MINIMUM_SIDE_EARLY_CLASS_ROWS,
        ),
        "calibration": (
            model.calibration_event_rows * 2,
            _MINIMUM_SIDE_CALIBRATION_CLASS_ROWS,
        ),
    }
    if set(model.side_profit_class_support) != set(side_role_contract):
        raise ValueError("three-action side-profit support roles are invalid")
    for role, (expected_rows, minimum) in side_role_contract.items():
        support = model.side_profit_class_support[role]
        if (
            set(support) != {"profitable_rows", "non_profitable_rows"}
            or any(
                isinstance(value, bool) or int(value) < minimum
                for value in support.values()
            )
            or sum(int(value) for value in support.values()) != expected_rows
        ):
            raise ValueError("three-action side-profit support values are invalid")
    if reload_boosters:
        try:
            expected_features = {
                "three_action_probability": len(MICROSTRUCTURE_FEATURE_NAMES),
                "side_profit_probability": len(ACTION_CONDITIONAL_FEATURE_NAMES),
                "positive_magnitude": len(ACTION_CONDITIONAL_FEATURE_NAMES),
                "nonpositive_loss_magnitude": len(ACTION_CONDITIONAL_FEATURE_NAMES),
                "lower_quantile": len(ACTION_CONDITIONAL_FEATURE_NAMES),
                "upper_quantile": len(ACTION_CONDITIONAL_FEATURE_NAMES),
            }
            for name in _HEADS:
                booster = lgb.Booster(model_str=model.model_strings[name])
                if booster.num_feature() != expected_features[name]:
                    raise ValueError("three-action booster feature count drifted")
        except lgb.basic.LightGBMError as exc:
            raise ValueError("three-action booster payload cannot be reloaded") from exc


def train_three_action_lightgbm_model(
    dataset: MicrostructureDataset,
    barrier_targets: AdaptiveBarrierTargets,
    *,
    train_endpoints: np.ndarray,
    tuning_endpoints: np.ndarray,
    spec: ThreeActionLightGBMSpec,
    compute_backend: str,
    seed: int,
    train_sample_weights: np.ndarray | None = None,
    tuning_sample_weights: np.ndarray | None = None,
    progress: ThreeActionProgressCallback | None = None,
) -> TrainedThreeActionLightGBMModel:
    """Fit one utility-weighted three-action member on prior roles only."""

    validate_microstructure_dataset(dataset)
    validate_adaptive_barrier_targets(dataset, barrier_targets)
    if (
        not isinstance(spec, ThreeActionLightGBMSpec)
        or not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        or not isinstance(compute_backend, str)
        or not compute_backend.strip()
        or dataset.feature_version != MICROSTRUCTURE_FEATURE_VERSION
        or dataset.feature_names != MICROSTRUCTURE_FEATURE_NAMES
    ):
        raise ValueError("three-action source feature contract is unsupported")
    train = _validate_indexes(
        train_endpoints,
        rows=dataset.rows,
        label="three-action training",
        minimum_rows=512,
    )
    requested_tuning = _validate_indexes(
        tuning_endpoints,
        rows=dataset.rows,
        label="three-action tuning",
        minimum_rows=1_024,
    )
    if train[-1] >= requested_tuning[0]:
        raise ValueError("three-action training and tuning roles overlap")
    train_weights = _validate_weights(
        train_sample_weights,
        rows=len(train),
        label="three-action training",
    )
    requested_tuning_weights = _validate_weights(
        tuning_sample_weights,
        rows=len(requested_tuning),
        label="three-action tuning",
    )
    targets, exits = _target_arrays(dataset, barrier_targets, scenario="stress")
    maximum_exit = np.maximum(exits["long"], exits["short"])
    if any(
        not np.all(np.isfinite(targets[side][train]))
        or not np.all(np.isfinite(targets[side][requested_tuning]))
        or np.any(exits[side][train] <= dataset.decision_time_ms[train])
        or np.any(
            exits[side][requested_tuning] <= dataset.decision_time_ms[requested_tuning]
        )
        for side in ("long", "short")
    ):
        raise ValueError("three-action roles contain invalid barrier labels")
    if np.any(maximum_exit[train] >= dataset.decision_time_ms[requested_tuning[0]]):
        raise ValueError("three-action training labels overlap tuning")
    calibration_rows = max(
        1,
        int(round(len(requested_tuning) * float(spec.calibration_fraction))),
    )
    split_at = len(requested_tuning) - calibration_rows
    calibration = requested_tuning[split_at:]
    calibration_start_ms = int(dataset.decision_time_ms[calibration[0]])
    early_candidates = requested_tuning[:split_at]
    early_candidate_weights = requested_tuning_weights[:split_at]
    early_keep = maximum_exit[early_candidates] < calibration_start_ms
    early_stop = early_candidates[early_keep]
    early_weights = early_candidate_weights[early_keep]
    internal_purged_rows = int(np.sum(~early_keep))
    if min(len(early_stop), len(calibration)) < 256:
        raise ValueError("three-action purged tuning roles are too small")

    backend_parameters, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        int(seed),
        reproducible=True,
    )
    if backend_kind == "opencl" and (
        spec.gpu_use_dp_required is not True
        or backend_parameters.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("three-action LightGBM OpenCL FP64 is required")
    common: dict[str, object] = {
        **backend_parameters,
        "learning_rate": float(spec.learning_rate),
        "num_leaves": int(spec.num_leaves),
        "max_depth": int(spec.max_depth),
        "min_data_in_leaf": int(spec.min_data_in_leaf),
        "feature_fraction": float(spec.feature_fraction),
        "bagging_fraction": float(spec.bagging_fraction),
        "bagging_freq": int(spec.bagging_freq),
        "lambda_l1": float(spec.lambda_l1),
        "lambda_l2": float(spec.lambda_l2),
        "max_bin": int(spec.max_bin),
    }
    source = np.asarray(dataset.features, dtype=np.float32)
    train_source = source[train]
    early_source = source[early_stop]
    calibration_source = source[calibration]
    mirrored_train = mirror_microstructure_direction(train_source)
    mirrored_early = mirror_microstructure_direction(early_source)
    train_labels = _action_labels(targets["long"][train], targets["short"][train])
    early_labels = _action_labels(
        targets["long"][early_stop],
        targets["short"][early_stop],
    )
    calibration_labels = _action_labels(
        targets["long"][calibration],
        targets["short"][calibration],
    )
    support = {
        "train": _class_support(
            train_labels,
            role="training",
            minimum=_MINIMUM_TRAIN_CLASS_ROWS,
        ),
        "early_stop": _class_support(
            early_labels,
            role="early-stop",
            minimum=_MINIMUM_EARLY_CLASS_ROWS,
        ),
        "calibration": _class_support(
            calibration_labels,
            role="calibration",
            minimum=_MINIMUM_CALIBRATION_CLASS_ROWS,
        ),
    }
    train_span = _decision_regret_span(
        targets["long"][train],
        targets["short"][train],
    )
    positive_span = train_span[train_span > 0.0]
    if len(positive_span) < 512:
        raise ValueError(
            "three-action training decision-regret support is insufficient"
        )
    regret_scale = float(np.median(positive_span))
    train_regret = _regret_multiplier(train_span, regret_scale)
    early_regret = _regret_multiplier(
        _decision_regret_span(
            targets["long"][early_stop],
            targets["short"][early_stop],
        ),
        regret_scale,
    )
    mirror_labels = np.asarray(
        (_SHORT_CLASS, _ABSTAIN_CLASS, _LONG_CLASS),
        dtype=np.int8,
    )[train_labels]
    mirror_early_labels = np.asarray(
        (_SHORT_CLASS, _ABSTAIN_CLASS, _LONG_CLASS),
        dtype=np.int8,
    )[early_labels]
    train_action = build_action_conditional_features(train_source)
    early_action = build_action_conditional_features(early_source)
    calibration_action = build_action_conditional_features(calibration_source)
    train_target = _paired_values(targets["long"][train], targets["short"][train])
    early_target = _paired_values(
        targets["long"][early_stop],
        targets["short"][early_stop],
    )
    action_train_weights = _paired_weights(train_weights)
    action_early_weights = _paired_weights(early_weights)
    train_positive = train_target > 0.0
    early_positive = early_target > 0.0
    calibration_target = _paired_values(
        targets["long"][calibration],
        targets["short"][calibration],
    )
    calibration_positive = calibration_target > 0.0
    side_profit_support = {
        "train": _side_profit_class_support(
            train_positive,
            role="training",
            minimum=_MINIMUM_SIDE_TRAIN_CLASS_ROWS,
        ),
        "early_stop": _side_profit_class_support(
            early_positive,
            role="early-stop",
            minimum=_MINIMUM_SIDE_EARLY_CLASS_ROWS,
        ),
        "calibration": _side_profit_class_support(
            calibration_positive,
            role="calibration",
            minimum=_MINIMUM_SIDE_CALIBRATION_CLASS_ROWS,
        ),
    }
    if (
        min(
            int(np.sum(train_positive)),
            int(np.sum(~train_positive)),
            int(np.sum(early_positive)),
            int(np.sum(~early_positive)),
        )
        < _MINIMUM_CONDITIONAL_ROWS
    ):
        raise ValueError("three-action magnitude class support is insufficient")

    boosters: dict[str, lgb.Booster] = {}
    iterations: dict[str, int] = {}
    step = 0

    def train_head(
        name: str,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        head_train_weights: np.ndarray,
        x_early: np.ndarray,
        y_early: np.ndarray,
        head_early_weights: np.ndarray,
        objective: str,
        metric: str,
        alpha: float | None = None,
        extra_parameters: Mapping[str, object] | None = None,
    ) -> None:
        nonlocal step
        if progress is not None:
            progress(name, step + 1, len(_HEADS))
        booster, iteration = _train_booster(
            x_train=x_train,
            y_train=np.asarray(y_train, dtype=np.float32),
            train_weights=np.asarray(head_train_weights, dtype=np.float32),
            x_early_stop=x_early,
            y_early_stop=np.asarray(y_early, dtype=np.float32),
            early_stop_weights=np.asarray(head_early_weights, dtype=np.float32),
            parameters={**common, **dict(extra_parameters or {})},
            objective=objective,
            metric=metric,
            num_boost_round=spec.num_boost_round,
            early_stopping_rounds=spec.early_stopping_rounds,
            alpha=alpha,
        )
        boosters[name] = booster
        iterations[name] = iteration
        step += 1

    train_head(
        "three_action_probability",
        x_train=np.concatenate((train_source, mirrored_train), axis=0),
        y_train=np.concatenate((train_labels, mirror_labels)),
        head_train_weights=np.concatenate(
            (
                train_weights * train_regret * 0.5,
                train_weights * train_regret * 0.5,
            )
        ),
        x_early=np.concatenate((early_source, mirrored_early), axis=0),
        y_early=np.concatenate((early_labels, mirror_early_labels)),
        head_early_weights=np.concatenate(
            (
                early_weights * early_regret * 0.5,
                early_weights * early_regret * 0.5,
            )
        ),
        objective="multiclass",
        metric="multi_logloss",
        extra_parameters={"num_class": 3},
    )
    calibration_logits = _symmetrized_logits(
        boosters["three_action_probability"],
        calibration_source,
        iteration=iterations["three_action_probability"],
    )
    (
        calibration_temperature,
        calibration_bias,
        calibration_iterations,
        calibration_gradient_norm,
        calibration_loss,
        prior_loss,
    ) = _fit_multiclass_calibration(calibration_logits, calibration_labels)

    train_head(
        "side_profit_probability",
        x_train=train_action.features,
        y_train=train_positive,
        head_train_weights=action_train_weights,
        x_early=early_action.features,
        y_early=early_positive,
        head_early_weights=action_early_weights,
        objective="binary",
        metric="binary_logloss",
    )
    raw_side_profit_calibration = np.asarray(
        boosters["side_profit_probability"].predict(
            calibration_action.features,
            num_iteration=iterations["side_profit_probability"],
        ),
        dtype=np.float64,
    )
    side_profit_probability_calibration = fit_platt_scaling(
        raw_side_profit_calibration,
        calibration_positive.astype(np.float64),
    )

    train_head(
        "positive_magnitude",
        x_train=train_action.features[train_positive],
        y_train=train_target[train_positive],
        head_train_weights=action_train_weights[train_positive],
        x_early=early_action.features[early_positive],
        y_early=early_target[early_positive],
        head_early_weights=action_early_weights[early_positive],
        objective="regression",
        metric="l2",
    )
    train_head(
        "nonpositive_loss_magnitude",
        x_train=train_action.features[~train_positive],
        y_train=-train_target[~train_positive],
        head_train_weights=action_train_weights[~train_positive],
        x_early=early_action.features[~early_positive],
        y_early=-early_target[~early_positive],
        head_early_weights=action_early_weights[~early_positive],
        objective="regression",
        metric="l2",
    )
    train_head(
        "lower_quantile",
        x_train=train_action.features,
        y_train=train_target,
        head_train_weights=action_train_weights,
        x_early=early_action.features,
        y_early=early_target,
        head_early_weights=action_early_weights,
        objective="quantile",
        metric="quantile",
        alpha=spec.lower_quantile,
    )
    train_head(
        "upper_quantile",
        x_train=train_action.features,
        y_train=train_target,
        head_train_weights=action_train_weights,
        x_early=early_action.features,
        y_early=early_target,
        head_early_weights=action_early_weights,
        objective="quantile",
        metric="quantile",
        alpha=spec.upper_quantile,
    )
    model_strings = {
        name: boosters[name].model_to_string(num_iteration=iterations[name])
        for name in _HEADS
    }
    provisional = TrainedThreeActionLightGBMModel(
        schema_version=THREE_ACTION_LIGHTGBM_SCHEMA_VERSION,
        model_family=_MODEL_FAMILY,
        spec=spec,
        source_feature_version=dataset.feature_version,
        source_feature_names=dataset.feature_names,
        action_feature_schema_version=ACTION_FEATURE_SCHEMA_VERSION,
        action_canonicalization_sha256=ACTION_CANONICALIZATION_SHA256,
        action_feature_names=ACTION_CONDITIONAL_FEATURE_NAMES,
        target_schema_version=barrier_targets.schema_version,
        target_mode=ADAPTIVE_BARRIER_TARGET_MODE,
        target_spec=barrier_targets.spec,
        target_contract_sha256=_target_contract_sha256(barrier_targets),
        target_scenario="stress",
        backend_requested=str(compute_backend),
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        seed=int(seed),
        training_event_rows=len(train),
        requested_tuning_event_rows=len(requested_tuning),
        early_stop_event_rows=len(early_stop),
        calibration_event_rows=len(calibration),
        calibration_start_ms=calibration_start_ms,
        internal_purged_event_rows=internal_purged_rows,
        action_class_support=support,
        side_profit_class_support=side_profit_support,
        regret_scale_bps=regret_scale,
        minimum_regret_multiplier=_MINIMUM_REGRET_MULTIPLIER,
        maximum_regret_multiplier=_MAXIMUM_REGRET_MULTIPLIER,
        training_regret_multiplier_mean=float(np.mean(train_regret)),
        early_stop_regret_multiplier_mean=float(np.mean(early_regret)),
        calibration_temperature=calibration_temperature,
        calibration_abstain_logit_bias=calibration_bias,
        calibration_iterations=calibration_iterations,
        calibration_gradient_norm=calibration_gradient_norm,
        calibration_multiclass_log_loss=calibration_loss,
        calibration_class_prior_log_loss=prior_loss,
        side_profit_probability_calibration=side_profit_probability_calibration,
        best_iterations=iterations,
        model_strings=model_strings,
        model_sha256="",
    )
    model = TrainedThreeActionLightGBMModel(
        **{**provisional.__dict__, "model_sha256": _model_sha256(provisional)}
    )
    _validate_model_contract(model, reload_boosters=False)
    return model


def predict_three_action_lightgbm_model(
    model: TrainedThreeActionLightGBMModel,
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
) -> ThreeActionPredictionBatch:
    """Predict three action probabilities and after-cost action values."""

    validate_microstructure_dataset(dataset)
    _validate_model_contract(model, reload_boosters=False)
    if (
        dataset.feature_version != model.source_feature_version
        or dataset.feature_names != model.source_feature_names
    ):
        raise ValueError("three-action prediction feature contract drifted")
    selected = _validate_indexes(
        endpoints,
        rows=dataset.rows,
        label="three-action prediction",
        minimum_rows=1,
    )
    try:
        boosters = {
            name: lgb.Booster(model_str=model.model_strings[name]) for name in _HEADS
        }
    except lgb.basic.LightGBMError as exc:
        raise ValueError("three-action booster payload cannot be reloaded") from exc
    source = np.asarray(dataset.features[selected], dtype=np.float32)
    logits = _symmetrized_logits(
        boosters["three_action_probability"],
        source,
        iteration=model.best_iterations["three_action_probability"],
    )
    probabilities = _calibrated_probabilities(
        logits,
        temperature=model.calibration_temperature,
        abstain_logit_bias=model.calibration_abstain_logit_bias,
    )
    long_action_probability = probabilities[:, _LONG_CLASS]
    abstain_action_probability = probabilities[:, _ABSTAIN_CLASS]
    short_action_probability = probabilities[:, _SHORT_CLASS]
    opportunity_probability = long_action_probability + short_action_probability
    conditional_long_probability = long_action_probability / opportunity_probability
    actions = build_action_conditional_features(source)
    raw_side_profit_probability = np.asarray(
        boosters["side_profit_probability"].predict(
            actions.features,
            num_iteration=model.best_iterations["side_profit_probability"],
        ),
        dtype=np.float64,
    )
    side_profit_probability = apply_platt_scaling(
        raw_side_profit_probability,
        model.side_profit_probability_calibration,
    ).reshape(len(selected), 2)
    positive = np.maximum(
        0.0,
        np.asarray(
            boosters["positive_magnitude"].predict(
                actions.features,
                num_iteration=model.best_iterations["positive_magnitude"],
            ),
            dtype=np.float64,
        ),
    ).reshape(len(selected), 2)
    loss = np.maximum(
        0.0,
        np.asarray(
            boosters["nonpositive_loss_magnitude"].predict(
                actions.features,
                num_iteration=model.best_iterations["nonpositive_loss_magnitude"],
            ),
            dtype=np.float64,
        ),
    ).reshape(len(selected), 2)
    expected = (
        side_profit_probability * positive - (1.0 - side_profit_probability) * loss
    )
    raw_lower = np.asarray(
        boosters["lower_quantile"].predict(
            actions.features,
            num_iteration=model.best_iterations["lower_quantile"],
        ),
        dtype=np.float64,
    ).reshape(len(selected), 2)
    raw_upper = np.asarray(
        boosters["upper_quantile"].predict(
            actions.features,
            num_iteration=model.best_iterations["upper_quantile"],
        ),
        dtype=np.float64,
    ).reshape(len(selected), 2)
    lower = np.minimum(raw_lower, raw_upper)
    upper = np.maximum(raw_lower, raw_upper)
    action_side = np.sign(expected[:, 0] - expected[:, 1]).astype(np.int8)
    decision_side = np.sign(long_action_probability - short_action_probability).astype(
        np.int8
    )
    consensus = (action_side != 0) & (action_side == decision_side)
    return ThreeActionPredictionBatch(
        endpoint_indexes=selected,
        long_mean_bps=expected[:, 0],
        short_mean_bps=expected[:, 1],
        long_profitable_probability=side_profit_probability[:, 0].copy(),
        short_profitable_probability=side_profit_probability[:, 1].copy(),
        long_action_probability=long_action_probability.copy(),
        abstain_action_probability=abstain_action_probability.copy(),
        short_action_probability=short_action_probability.copy(),
        opportunity_probability=opportunity_probability,
        conditional_long_probability=conditional_long_probability,
        long_lower_bps=lower[:, 0],
        short_lower_bps=lower[:, 1],
        long_upper_bps=upper[:, 0],
        short_upper_bps=upper[:, 1],
        action_preference_side=action_side,
        decision_preference_side=decision_side,
        side_consensus=np.asarray(consensus, dtype=bool),
    )


def ensemble_three_action_predictions(
    members: Sequence[ThreeActionPredictionBatch],
) -> ThreeActionEnsembleBatch:
    """Ensemble compatible members without aliasing probability semantics."""

    values = tuple(members)
    if len(values) < 2:
        raise ValueError("three-action ensemble requires at least two members")
    endpoints = np.asarray(values[0].endpoint_indexes, dtype=np.int64)
    if any(
        not np.array_equal(endpoints, value.endpoint_indexes) for value in values[1:]
    ):
        raise ValueError("three-action ensemble endpoint identities differ")
    action_values = ensemble_action_value_predictions(
        tuple(
            ActionValuePredictionBatch(
                endpoint_indexes=value.endpoint_indexes,
                long_mean_bps=value.long_mean_bps,
                short_mean_bps=value.short_mean_bps,
                long_profitable_probability=value.long_profitable_probability,
                short_profitable_probability=value.short_profitable_probability,
                long_lower_bps=value.long_lower_bps,
                short_lower_bps=value.short_lower_bps,
                long_upper_bps=value.long_upper_bps,
                short_upper_bps=value.short_upper_bps,
            )
            for value in values
        )
    )
    long_members = np.stack(
        [
            np.asarray(value.long_action_probability, dtype=np.float64)
            for value in values
        ]
    )
    abstain_members = np.stack(
        [
            np.asarray(value.abstain_action_probability, dtype=np.float64)
            for value in values
        ]
    )
    short_members = np.stack(
        [
            np.asarray(value.short_action_probability, dtype=np.float64)
            for value in values
        ]
    )
    opportunity_members = long_members + short_members
    direction_members = long_members / opportunity_members
    consensus = np.stack(
        [np.asarray(value.side_consensus, dtype=bool) for value in values]
    )
    return ThreeActionEnsembleBatch(
        action_values=action_values,
        long_action_probability_mean=np.mean(long_members, axis=0),
        abstain_action_probability_mean=np.mean(abstain_members, axis=0),
        short_action_probability_mean=np.mean(short_members, axis=0),
        opportunity_probability_mean=np.mean(opportunity_members, axis=0),
        opportunity_probability_std=np.std(opportunity_members, axis=0),
        conditional_long_probability_mean=np.mean(direction_members, axis=0),
        conditional_long_probability_std=np.std(direction_members, axis=0),
        long_action_member_probabilities=long_members,
        abstain_action_member_probabilities=abstain_members,
        short_action_member_probabilities=short_members,
        opportunity_member_probabilities=opportunity_members,
        conditional_long_member_probabilities=direction_members,
        direction_long_member_ratio=np.mean(direction_members > 0.5, axis=0),
        direction_short_member_ratio=np.mean(direction_members < 0.5, axis=0),
        side_consensus_member_ratio=np.mean(consensus, axis=0),
        member_count=len(values),
    )


def as_selective_action_ensemble(
    ensemble: ThreeActionEnsembleBatch,
) -> SelectiveActionEnsembleBatch:
    """Adapt verified routing evidence to the shared fail-closed policy API."""

    return SelectiveActionEnsembleBatch(
        action_values=ensemble.action_values,
        opportunity_probability_mean=ensemble.opportunity_probability_mean,
        opportunity_probability_std=ensemble.opportunity_probability_std,
        conditional_long_probability_mean=ensemble.conditional_long_probability_mean,
        conditional_long_probability_std=ensemble.conditional_long_probability_std,
        opportunity_member_probabilities=ensemble.opportunity_member_probabilities,
        conditional_long_member_probabilities=(
            ensemble.conditional_long_member_probabilities
        ),
        direction_long_member_ratio=ensemble.direction_long_member_ratio,
        direction_short_member_ratio=ensemble.direction_short_member_ratio,
        side_consensus_member_ratio=ensemble.side_consensus_member_ratio,
        member_count=ensemble.member_count,
    )


def save_three_action_lightgbm_model(
    path: str | Path,
    model: TrainedThreeActionLightGBMModel,
) -> None:
    """Atomically persist a complete authority-denied three-action model."""

    _validate_model_contract(model, reload_boosters=True)
    write_json_atomic(
        path,
        {**_model_payload(model), "model_sha256": model.model_sha256},
        indent=None,
        sort_keys=True,
    )


def load_three_action_lightgbm_model(
    path: str | Path,
) -> TrainedThreeActionLightGBMModel:
    """Load and independently validate a three-action model artifact."""

    target = Path(path)
    try:
        size = target.stat().st_size
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("three-action artifact is unreadable") from exc
    expected_fields = {field.name for field in fields(TrainedThreeActionLightGBMModel)}
    if (
        size <= 0
        or size > _MAX_ARTIFACT_BYTES
        or not isinstance(payload, dict)
        or set(payload) != expected_fields
    ):
        raise ValueError("three-action artifact size or structure is invalid")
    authority_fields = (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    )
    integer_fields = (
        "seed",
        "training_event_rows",
        "requested_tuning_event_rows",
        "early_stop_event_rows",
        "calibration_event_rows",
        "calibration_start_ms",
        "internal_purged_event_rows",
        "calibration_iterations",
    )
    if any(payload[name] is not False for name in authority_fields) or any(
        not isinstance(payload[name], int) or isinstance(payload[name], bool)
        for name in integer_fields
    ):
        raise ValueError("three-action artifact scalar types are invalid")
    try:
        model = TrainedThreeActionLightGBMModel(
            schema_version=str(payload["schema_version"]),
            model_family=str(payload["model_family"]),
            spec=ThreeActionLightGBMSpec(**payload["spec"]),
            source_feature_version=str(payload["source_feature_version"]),
            source_feature_names=tuple(payload["source_feature_names"]),
            action_feature_schema_version=str(payload["action_feature_schema_version"]),
            action_canonicalization_sha256=str(
                payload["action_canonicalization_sha256"]
            ),
            action_feature_names=tuple(payload["action_feature_names"]),
            target_schema_version=str(payload["target_schema_version"]),
            target_mode=str(payload["target_mode"]),
            target_spec=AdaptiveBarrierSpec(**payload["target_spec"]),
            target_contract_sha256=str(payload["target_contract_sha256"]),
            target_scenario=str(payload["target_scenario"]),
            backend_requested=str(payload["backend_requested"]),
            backend_kind=str(payload["backend_kind"]),
            backend_device=str(payload["backend_device"]),
            lightgbm_version=str(payload["lightgbm_version"]),
            seed=int(payload["seed"]),
            training_event_rows=int(payload["training_event_rows"]),
            requested_tuning_event_rows=int(payload["requested_tuning_event_rows"]),
            early_stop_event_rows=int(payload["early_stop_event_rows"]),
            calibration_event_rows=int(payload["calibration_event_rows"]),
            calibration_start_ms=int(payload["calibration_start_ms"]),
            internal_purged_event_rows=int(payload["internal_purged_event_rows"]),
            action_class_support={
                str(role): {str(name): int(value) for name, value in support.items()}
                for role, support in payload["action_class_support"].items()
            },
            side_profit_class_support={
                str(role): {str(name): int(value) for name, value in support.items()}
                for role, support in payload["side_profit_class_support"].items()
            },
            regret_scale_bps=float(payload["regret_scale_bps"]),
            minimum_regret_multiplier=float(payload["minimum_regret_multiplier"]),
            maximum_regret_multiplier=float(payload["maximum_regret_multiplier"]),
            training_regret_multiplier_mean=float(
                payload["training_regret_multiplier_mean"]
            ),
            early_stop_regret_multiplier_mean=float(
                payload["early_stop_regret_multiplier_mean"]
            ),
            calibration_temperature=float(payload["calibration_temperature"]),
            calibration_abstain_logit_bias=float(
                payload["calibration_abstain_logit_bias"]
            ),
            calibration_iterations=int(payload["calibration_iterations"]),
            calibration_gradient_norm=float(payload["calibration_gradient_norm"]),
            calibration_multiclass_log_loss=float(
                payload["calibration_multiclass_log_loss"]
            ),
            calibration_class_prior_log_loss=float(
                payload["calibration_class_prior_log_loss"]
            ),
            side_profit_probability_calibration=tuple(
                float(value) for value in payload["side_profit_probability_calibration"]
            ),
            best_iterations={
                str(name): int(value)
                for name, value in payload["best_iterations"].items()
            },
            model_strings={
                str(name): str(value)
                for name, value in payload["model_strings"].items()
            },
            model_sha256=str(payload["model_sha256"]),
            trading_authority=payload["trading_authority"],
            execution_claim=payload["execution_claim"],
            profitability_claim=payload["profitability_claim"],
            portfolio_claim=payload["portfolio_claim"],
            leverage_applied=payload["leverage_applied"],
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("three-action artifact values are invalid") from exc
    _validate_model_contract(model, reload_boosters=True)
    return model


__all__ = [
    "THREE_ACTION_LIGHTGBM_SCHEMA_VERSION",
    "ThreeActionEnsembleBatch",
    "ThreeActionLightGBMSpec",
    "ThreeActionPredictionBatch",
    "TrainedThreeActionLightGBMModel",
    "as_selective_action_ensemble",
    "ensemble_three_action_predictions",
    "load_three_action_lightgbm_model",
    "predict_three_action_lightgbm_model",
    "save_three_action_lightgbm_model",
    "train_three_action_lightgbm_model",
]
