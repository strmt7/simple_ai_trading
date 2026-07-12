"""Factorized opportunity, conditional-direction, and action-value LightGBM."""

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
from .microstructure_shared_action_lightgbm import (
    _paired_values,
    _paired_weights,
    _train_booster,
)
from .probability_calibration import apply_platt_scaling, fit_platt_scaling
from .storage import write_json_atomic


SELECTIVE_ACTION_LIGHTGBM_SCHEMA_VERSION = "selective-action-lightgbm-hurdle-v1"
_MODEL_FAMILY = "factorized_selective_action_lightgbm_hurdle"
_HEADS = (
    "opportunity_probability",
    "conditional_direction_probability",
    "positive_magnitude",
    "nonpositive_loss_magnitude",
    "lower_quantile",
    "upper_quantile",
)
_MINIMUM_TRAIN_CLASS_ROWS = 512
_MINIMUM_EARLY_CLASS_ROWS = 128
_MINIMUM_CALIBRATION_CLASS_ROWS = 256
_MINIMUM_DIRECTION_TRAIN_CLASS_ROWS = 256
_MINIMUM_DIRECTION_EARLY_CLASS_ROWS = 64
_MINIMUM_DIRECTION_CALIBRATION_CLASS_ROWS = 128
_MINIMUM_CONDITIONAL_ROWS = 128
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
SelectiveActionProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class SelectiveActionLightGBMSpec:
    """Precommitted factorized selective-action model controls."""

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
            or any(isinstance(value, bool) for value in (*integral, *numeric))
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
            raise ValueError("selective-action LightGBM specification is invalid")


@dataclass(frozen=True)
class TrainedSelectiveActionLightGBMModel:
    """Hash-bound selective-action member without execution authority."""

    schema_version: str
    model_family: str
    spec: SelectiveActionLightGBMSpec
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
    class_support: Mapping[str, Mapping[str, int]]
    opportunity_calibration: tuple[float, float]
    direction_temperature: float
    best_iterations: Mapping[str, int]
    model_strings: Mapping[str, str]
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False


@dataclass(frozen=True)
class SelectiveActionPredictionBatch:
    """One member's three-state probabilities and paired action values."""

    endpoint_indexes: np.ndarray
    long_mean_bps: np.ndarray
    short_mean_bps: np.ndarray
    long_profitable_probability: np.ndarray
    short_profitable_probability: np.ndarray
    abstain_probability: np.ndarray
    opportunity_probability: np.ndarray
    conditional_long_probability: np.ndarray
    long_lower_bps: np.ndarray
    short_lower_bps: np.ndarray
    long_upper_bps: np.ndarray
    short_upper_bps: np.ndarray
    action_preference_side: np.ndarray
    direction_preference_side: np.ndarray
    side_consensus: np.ndarray

    def __post_init__(self) -> None:
        endpoints = np.asarray(self.endpoint_indexes, dtype=np.int64)
        rows = len(endpoints)
        numeric = (
            self.long_mean_bps,
            self.short_mean_bps,
            self.long_profitable_probability,
            self.short_profitable_probability,
            self.abstain_probability,
            self.opportunity_probability,
            self.conditional_long_probability,
            self.long_lower_bps,
            self.short_lower_bps,
            self.long_upper_bps,
            self.short_upper_bps,
        )
        long_probability = np.asarray(
            self.long_profitable_probability,
            dtype=np.float64,
        )
        short_probability = np.asarray(
            self.short_profitable_probability,
            dtype=np.float64,
        )
        abstain_probability = np.asarray(self.abstain_probability, dtype=np.float64)
        opportunity_probability = np.asarray(
            self.opportunity_probability,
            dtype=np.float64,
        )
        conditional_long_probability = np.asarray(
            self.conditional_long_probability,
            dtype=np.float64,
        )
        long_mean = np.asarray(self.long_mean_bps, dtype=np.float64)
        short_mean = np.asarray(self.short_mean_bps, dtype=np.float64)
        action_side = np.asarray(self.action_preference_side, dtype=np.int8)
        direction_side = np.asarray(self.direction_preference_side, dtype=np.int8)
        consensus = np.asarray(self.side_consensus)
        if (
            rows <= 0
            or endpoints.ndim != 1
            or endpoints[0] < 0
            or np.any(np.diff(endpoints) <= 0)
            or any(np.asarray(value).shape != (rows,) for value in numeric)
            or any(not np.all(np.isfinite(value)) for value in numeric)
            or any(
                np.any(value < 0.0) or np.any(value > 1.0)
                for value in numeric[2:7]
            )
            or not np.allclose(
                long_probability + short_probability + abstain_probability,
                1.0,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                long_probability + short_probability,
                opportunity_probability,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                long_probability,
                opportunity_probability * conditional_long_probability,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                short_probability,
                opportunity_probability * (1.0 - conditional_long_probability),
                rtol=0.0,
                atol=1e-12,
            )
            or np.any(
                np.asarray(self.long_lower_bps, dtype=np.float64)
                > np.asarray(self.long_upper_bps, dtype=np.float64)
            )
            or np.any(
                np.asarray(self.short_lower_bps, dtype=np.float64)
                > np.asarray(self.short_upper_bps, dtype=np.float64)
            )
            or any(
                np.asarray(value).shape != (rows,)
                for value in (
                    self.action_preference_side,
                    self.direction_preference_side,
                    self.side_consensus,
                )
            )
            or not set(np.unique(self.action_preference_side)).issubset({-1, 0, 1})
            or not set(np.unique(self.direction_preference_side)).issubset(
                {-1, 0, 1}
            )
            or consensus.dtype != np.bool_
            or not np.array_equal(action_side, np.sign(long_mean - short_mean))
            or not np.array_equal(
                direction_side,
                np.sign(conditional_long_probability - 0.5),
            )
            or not np.array_equal(
                consensus,
                (action_side != 0) & (action_side == direction_side),
            )
        ):
            raise ValueError("selective-action prediction batch is invalid")

    @property
    def rows(self) -> int:
        return len(self.endpoint_indexes)


@dataclass(frozen=True)
class SelectiveActionEnsembleBatch:
    """Ensembled action values plus retained selective member probabilities."""

    action_values: ActionValueEnsembleBatch
    opportunity_probability_mean: np.ndarray
    opportunity_probability_std: np.ndarray
    conditional_long_probability_mean: np.ndarray
    conditional_long_probability_std: np.ndarray
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
        vectors = (
            self.opportunity_probability_mean,
            self.opportunity_probability_std,
            self.conditional_long_probability_mean,
            self.conditional_long_probability_std,
            self.direction_long_member_ratio,
            self.direction_short_member_ratio,
            self.side_consensus_member_ratio,
        )
        matrices = (
            self.opportunity_member_probabilities,
            self.conditional_long_member_probabilities,
        )
        opportunity_members = np.asarray(
            self.opportunity_member_probabilities,
            dtype=np.float64,
        )
        direction_members = np.asarray(
            self.conditional_long_member_probabilities,
            dtype=np.float64,
        )
        if (
            self.member_count != self.action_values.member_count
            or self.member_count < 2
            or any(np.asarray(value).shape != (rows,) for value in vectors)
            or any(np.asarray(value).shape != (self.member_count, rows) for value in matrices)
            or any(not np.all(np.isfinite(value)) for value in (*vectors, *matrices))
            or any(
                np.any(value < 0.0) or np.any(value > 1.0)
                for value in (
                    self.opportunity_probability_mean,
                    self.conditional_long_probability_mean,
                    self.direction_long_member_ratio,
                    self.direction_short_member_ratio,
                    self.side_consensus_member_ratio,
                    *matrices,
                )
            )
            or np.any(self.opportunity_probability_std < 0.0)
            or np.any(self.conditional_long_probability_std < 0.0)
            or not np.allclose(
                self.opportunity_probability_mean,
                np.mean(opportunity_members, axis=0),
                rtol=0.0,
                atol=1e-15,
            )
            or not np.allclose(
                self.opportunity_probability_std,
                np.std(opportunity_members, axis=0),
                rtol=0.0,
                atol=1e-15,
            )
            or not np.allclose(
                self.conditional_long_probability_mean,
                np.mean(direction_members, axis=0),
                rtol=0.0,
                atol=1e-15,
            )
            or not np.allclose(
                self.conditional_long_probability_std,
                np.std(direction_members, axis=0),
                rtol=0.0,
                atol=1e-15,
            )
            or not np.array_equal(
                self.direction_long_member_ratio,
                np.mean(direction_members > 0.5, axis=0),
            )
            or not np.array_equal(
                self.direction_short_member_ratio,
                np.mean(direction_members < 0.5, axis=0),
            )
            or self.trading_authority
            or self.execution_claim
            or self.profitability_claim
            or self.portfolio_claim
            or self.leverage_applied
        ):
            raise ValueError("selective-action ensemble batch is invalid")

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


def _model_payload(model: TrainedSelectiveActionLightGBMModel) -> dict[str, object]:
    value = asdict(model)
    value.pop("model_sha256")
    return value


def _model_sha256(model: TrainedSelectiveActionLightGBMModel) -> str:
    return hashlib.sha256(_canonical_json(_model_payload(model)).encode("ascii")).hexdigest()


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    value = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _binary_class_support(
    labels: np.ndarray,
    *,
    role: str,
    minimum: int,
    positive_name: str,
    negative_name: str,
) -> dict[str, int]:
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0 or values.dtype != np.bool_:
        raise ValueError(f"{role} labels are invalid")
    positive = int(np.sum(values))
    negative = int(len(values) - positive)
    if min(positive, negative) < int(minimum):
        raise ValueError(f"{role} class support is insufficient")
    return {positive_name: positive, negative_name: negative}


def _fit_direction_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    binary = np.asarray(labels, dtype=np.float64)
    if (
        values.ndim != 1
        or binary.shape != values.shape
        or len(values) < 256
        or not np.all(np.isfinite(values))
        or not set(np.unique(binary)).issubset({0.0, 1.0})
        or len(np.unique(binary)) != 2
    ):
        raise ValueError("direction temperature calibration inputs are invalid")

    def loss(log_scale: float) -> float:
        scaled = values * math.exp(log_scale)
        return float(np.mean(np.logaddexp(0.0, scaled) - binary * scaled))

    left, right = -4.0, 4.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    first = right - ratio * (right - left)
    second = left + ratio * (right - left)
    first_loss, second_loss = loss(first), loss(second)
    for _iteration in range(96):
        if first_loss <= second_loss:
            right, second, second_loss = second, first, first_loss
            first = right - ratio * (right - left)
            first_loss = loss(first)
        else:
            left, first, first_loss = first, second, second_loss
            second = left + ratio * (right - left)
            second_loss = loss(second)
    scale = math.exp((left + right) / 2.0)
    temperature = 1.0 / scale
    if not math.isfinite(temperature) or not 0.01 <= temperature <= 100.0:
        raise ValueError("direction temperature calibration failed")
    return float(temperature)


def _mirrored_opportunity_probability(
    booster: lgb.Booster,
    source: np.ndarray,
    *,
    iteration: int,
) -> np.ndarray:
    mirrored = mirror_microstructure_direction(source)
    raw = np.asarray(booster.predict(source, num_iteration=iteration), dtype=np.float64)
    reflected = np.asarray(
        booster.predict(mirrored, num_iteration=iteration), dtype=np.float64
    )
    return 0.5 * (raw + reflected)


def _antisymmetric_direction_logit(
    booster: lgb.Booster,
    source: np.ndarray,
    *,
    iteration: int,
) -> np.ndarray:
    mirrored = mirror_microstructure_direction(source)
    raw = np.asarray(booster.predict(source, num_iteration=iteration), dtype=np.float64)
    reflected = np.asarray(
        booster.predict(mirrored, num_iteration=iteration), dtype=np.float64
    )
    return 0.5 * (_logit(raw) - _logit(reflected))


def _validate_model_contract(
    model: TrainedSelectiveActionLightGBMModel,
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
        model.schema_version != SELECTIVE_ACTION_LIGHTGBM_SCHEMA_VERSION
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
        or isinstance(model.seed, bool)
        or set(model.best_iterations) != set(_HEADS)
        or set(model.model_strings) != set(_HEADS)
        or any(
            isinstance(value, bool)
            or not 1 <= int(value) <= model.spec.num_boost_round
            for value in model.best_iterations.values()
        )
        or any(
            not isinstance(value, str) or not value
            for value in model.model_strings.values()
        )
        or model.training_event_rows < 512
        or min(
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
        or not math.isfinite(model.direction_temperature)
        or not 0.01 <= model.direction_temperature <= 100.0
        or len(model.opportunity_calibration) != 2
        or not 0.05 <= float(model.opportunity_calibration[0]) <= 10.0
        or not -10.0 <= float(model.opportunity_calibration[1]) <= 10.0
        or any(authority)
        or not _is_sha256(model.model_sha256)
        or _model_sha256(model) != model.model_sha256
    ):
        raise ValueError("selective-action model contract is invalid")
    support_contract = {
        "opportunity_train": (
            "opportunity_rows",
            "abstain_rows",
            model.training_event_rows,
            _MINIMUM_TRAIN_CLASS_ROWS,
        ),
        "opportunity_early_stop": (
            "opportunity_rows",
            "abstain_rows",
            model.early_stop_event_rows,
            _MINIMUM_EARLY_CLASS_ROWS,
        ),
        "opportunity_calibration": (
            "opportunity_rows",
            "abstain_rows",
            model.calibration_event_rows,
            _MINIMUM_CALIBRATION_CLASS_ROWS,
        ),
        "direction_train": (
            "long_preferred_rows",
            "short_preferred_rows",
            None,
            _MINIMUM_DIRECTION_TRAIN_CLASS_ROWS,
        ),
        "direction_early_stop": (
            "long_preferred_rows",
            "short_preferred_rows",
            None,
            _MINIMUM_DIRECTION_EARLY_CLASS_ROWS,
        ),
        "direction_calibration": (
            "long_preferred_rows",
            "short_preferred_rows",
            None,
            _MINIMUM_DIRECTION_CALIBRATION_CLASS_ROWS,
        ),
    }
    if set(model.class_support) != set(support_contract):
        raise ValueError("selective-action class-support roles are invalid")
    for role, (positive_name, negative_name, expected_rows, minimum) in (
        support_contract.items()
    ):
        support = model.class_support[role]
        values = tuple(support.values())
        if (
            set(support) != {positive_name, negative_name}
            or any(isinstance(value, bool) or int(value) < minimum for value in values)
        ):
            raise ValueError("selective-action class-support values are invalid")
        if expected_rows is not None and sum(int(value) for value in values) != expected_rows:
            raise ValueError("selective-action class-support row count drifted")
    for suffix in ("train", "early_stop", "calibration"):
        opportunity_rows = model.class_support[f"opportunity_{suffix}"][
            "opportunity_rows"
        ]
        direction_rows = sum(
            model.class_support[f"direction_{suffix}"].values()
        )
        if int(direction_rows) != int(opportunity_rows):
            raise ValueError("conditional-direction population drifted")
    if reload_boosters:
        try:
            for name in _HEADS:
                lgb.Booster(model_str=model.model_strings[name])
        except lgb.basic.LightGBMError as exc:
            raise ValueError("selective-action booster payload cannot be reloaded") from exc


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def train_selective_action_lightgbm_model(
    dataset: MicrostructureDataset,
    barrier_targets: AdaptiveBarrierTargets,
    *,
    train_endpoints: np.ndarray,
    tuning_endpoints: np.ndarray,
    spec: SelectiveActionLightGBMSpec,
    compute_backend: str,
    seed: int,
    train_sample_weights: np.ndarray | None = None,
    tuning_sample_weights: np.ndarray | None = None,
    progress: SelectiveActionProgressCallback | None = None,
) -> TrainedSelectiveActionLightGBMModel:
    """Fit one selective-action member without accessing later roles."""

    validate_microstructure_dataset(dataset)
    validate_adaptive_barrier_targets(dataset, barrier_targets)
    if (
        dataset.feature_version != MICROSTRUCTURE_FEATURE_VERSION
        or dataset.feature_names != MICROSTRUCTURE_FEATURE_NAMES
    ):
        raise ValueError("selective-action source feature contract is unsupported")
    train = _validate_indexes(
        train_endpoints,
        rows=dataset.rows,
        label="selective-action training",
        minimum_rows=512,
    )
    requested_tuning = _validate_indexes(
        tuning_endpoints,
        rows=dataset.rows,
        label="selective-action tuning",
        minimum_rows=1_024,
    )
    if train[-1] >= requested_tuning[0]:
        raise ValueError("selective-action training and tuning roles overlap")
    train_weights = _validate_weights(
        train_sample_weights, rows=len(train), label="selective-action training"
    )
    requested_tuning_weights = _validate_weights(
        tuning_sample_weights,
        rows=len(requested_tuning),
        label="selective-action tuning",
    )
    targets, exits = _target_arrays(dataset, barrier_targets, scenario="stress")
    maximum_exit = np.maximum(exits["long"], exits["short"])
    if any(
        not np.all(np.isfinite(targets[side][train]))
        or not np.all(np.isfinite(targets[side][requested_tuning]))
        or np.any(exits[side][train] <= dataset.decision_time_ms[train])
        or np.any(exits[side][requested_tuning] <= dataset.decision_time_ms[requested_tuning])
        for side in ("long", "short")
    ):
        raise ValueError("selective-action roles contain invalid barrier labels")
    if np.any(maximum_exit[train] >= dataset.decision_time_ms[requested_tuning[0]]):
        raise ValueError("selective-action training labels overlap tuning")
    calibration_rows = max(
        1, int(round(len(requested_tuning) * float(spec.calibration_fraction)))
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
        raise ValueError("selective-action purged tuning roles are too small")

    backend_parameters, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend, int(seed), reproducible=True
    )
    if backend_kind == "opencl" and (
        spec.gpu_use_dp_required is not True
        or backend_parameters.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("selective-action LightGBM OpenCL FP64 is required")
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
    train_opportunity = (
        np.maximum(targets["long"][train], targets["short"][train]) > 0.0
    )
    early_opportunity = (
        np.maximum(
            targets["long"][early_stop], targets["short"][early_stop]
        )
        > 0.0
    )
    calibration_opportunity = (
        np.maximum(
            targets["long"][calibration], targets["short"][calibration]
        )
        > 0.0
    )
    train_direction = targets["long"][train] > targets["short"][train]
    early_direction = (
        targets["long"][early_stop] > targets["short"][early_stop]
    )
    calibration_direction = (
        targets["long"][calibration] > targets["short"][calibration]
    )
    support = {
        "opportunity_train": _binary_class_support(
            train_opportunity,
            role="opportunity training",
            minimum=_MINIMUM_TRAIN_CLASS_ROWS,
            positive_name="opportunity_rows",
            negative_name="abstain_rows",
        ),
        "opportunity_early_stop": _binary_class_support(
            early_opportunity,
            role="opportunity early-stop",
            minimum=_MINIMUM_EARLY_CLASS_ROWS,
            positive_name="opportunity_rows",
            negative_name="abstain_rows",
        ),
        "opportunity_calibration": _binary_class_support(
            calibration_opportunity,
            role="opportunity calibration",
            minimum=_MINIMUM_CALIBRATION_CLASS_ROWS,
            positive_name="opportunity_rows",
            negative_name="abstain_rows",
        ),
        "direction_train": _binary_class_support(
            train_direction[train_opportunity],
            role="conditional direction training",
            minimum=_MINIMUM_DIRECTION_TRAIN_CLASS_ROWS,
            positive_name="long_preferred_rows",
            negative_name="short_preferred_rows",
        ),
        "direction_early_stop": _binary_class_support(
            early_direction[early_opportunity],
            role="conditional direction early-stop",
            minimum=_MINIMUM_DIRECTION_EARLY_CLASS_ROWS,
            positive_name="long_preferred_rows",
            negative_name="short_preferred_rows",
        ),
        "direction_calibration": _binary_class_support(
            calibration_direction[calibration_opportunity],
            role="conditional direction calibration",
            minimum=_MINIMUM_DIRECTION_CALIBRATION_CLASS_ROWS,
            positive_name="long_preferred_rows",
            negative_name="short_preferred_rows",
        ),
    }
    train_action = build_action_conditional_features(train_source)
    early_action = build_action_conditional_features(early_source)
    train_target = _paired_values(targets["long"][train], targets["short"][train])
    early_target = _paired_values(
        targets["long"][early_stop], targets["short"][early_stop]
    )
    action_train_weights = _paired_weights(train_weights)
    action_early_weights = _paired_weights(early_weights)
    train_positive = train_target > 0.0
    early_positive = early_target > 0.0
    if min(
        int(np.sum(train_positive)),
        int(np.sum(~train_positive)),
        int(np.sum(early_positive)),
        int(np.sum(~early_positive)),
    ) < _MINIMUM_CONDITIONAL_ROWS:
        raise ValueError("selective-action magnitude class support is insufficient")

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
            parameters=common,
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
        "opportunity_probability",
        x_train=np.concatenate((train_source, mirrored_train), axis=0),
        y_train=np.concatenate((train_opportunity, train_opportunity)),
        head_train_weights=np.concatenate(
            (train_weights * 0.5, train_weights * 0.5)
        ),
        x_early=np.concatenate((early_source, mirrored_early), axis=0),
        y_early=np.concatenate((early_opportunity, early_opportunity)),
        head_early_weights=np.concatenate(
            (early_weights * 0.5, early_weights * 0.5)
        ),
        objective="binary",
        metric="binary_logloss",
    )
    opportunity_raw = _mirrored_opportunity_probability(
        boosters["opportunity_probability"],
        calibration_source,
        iteration=iterations["opportunity_probability"],
    )
    opportunity_calibration = fit_platt_scaling(
        opportunity_raw, calibration_opportunity.astype(np.float32)
    )

    train_direction_rows = train_opportunity
    early_direction_rows = early_opportunity
    train_head(
        "conditional_direction_probability",
        x_train=np.concatenate(
            (
                train_source[train_direction_rows],
                mirrored_train[train_direction_rows],
            ),
            axis=0,
        ),
        y_train=np.concatenate(
            (
                train_direction[train_direction_rows],
                ~train_direction[train_direction_rows],
            )
        ),
        head_train_weights=np.concatenate(
            (
                train_weights[train_direction_rows] * 0.5,
                train_weights[train_direction_rows] * 0.5,
            )
        ),
        x_early=np.concatenate(
            (
                early_source[early_direction_rows],
                mirrored_early[early_direction_rows],
            ),
            axis=0,
        ),
        y_early=np.concatenate(
            (
                early_direction[early_direction_rows],
                ~early_direction[early_direction_rows],
            )
        ),
        head_early_weights=np.concatenate(
            (
                early_weights[early_direction_rows] * 0.5,
                early_weights[early_direction_rows] * 0.5,
            )
        ),
        objective="binary",
        metric="binary_logloss",
    )
    calibration_direction_logits = _antisymmetric_direction_logit(
        boosters["conditional_direction_probability"],
        calibration_source[calibration_opportunity],
        iteration=iterations["conditional_direction_probability"],
    )
    direction_temperature = _fit_direction_temperature(
        calibration_direction_logits,
        calibration_direction[calibration_opportunity].astype(np.float32),
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
    provisional = TrainedSelectiveActionLightGBMModel(
        schema_version=SELECTIVE_ACTION_LIGHTGBM_SCHEMA_VERSION,
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
        class_support=support,
        opportunity_calibration=opportunity_calibration,
        direction_temperature=direction_temperature,
        best_iterations=iterations,
        model_strings=model_strings,
        model_sha256="",
    )
    model = TrainedSelectiveActionLightGBMModel(
        **{**provisional.__dict__, "model_sha256": _model_sha256(provisional)}
    )
    _validate_model_contract(model, reload_boosters=False)
    return model


def predict_selective_action_lightgbm_model(
    model: TrainedSelectiveActionLightGBMModel,
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
) -> SelectiveActionPredictionBatch:
    """Predict exact three-state probabilities and after-cost action values."""

    validate_microstructure_dataset(dataset)
    _validate_model_contract(model, reload_boosters=False)
    if (
        dataset.feature_version != model.source_feature_version
        or dataset.feature_names != model.source_feature_names
    ):
        raise ValueError("selective-action prediction feature contract drifted")
    selected = _validate_indexes(
        endpoints,
        rows=dataset.rows,
        label="selective-action prediction",
        minimum_rows=1,
    )
    try:
        boosters = {
            name: lgb.Booster(model_str=model.model_strings[name]) for name in _HEADS
        }
    except lgb.basic.LightGBMError as exc:
        raise ValueError("selective-action booster payload cannot be reloaded") from exc
    source = np.asarray(dataset.features[selected], dtype=np.float32)
    opportunity_raw = _mirrored_opportunity_probability(
        boosters["opportunity_probability"],
        source,
        iteration=model.best_iterations["opportunity_probability"],
    )
    opportunity = apply_platt_scaling(
        opportunity_raw, model.opportunity_calibration
    )
    direction_logit = _antisymmetric_direction_logit(
        boosters["conditional_direction_probability"],
        source,
        iteration=model.best_iterations["conditional_direction_probability"],
    )
    conditional_long = _sigmoid(direction_logit / model.direction_temperature)
    long_probability = opportunity * conditional_long
    short_probability = opportunity * (1.0 - conditional_long)
    abstain_probability = 1.0 - opportunity
    actions = build_action_conditional_features(source)
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
    probability = np.column_stack((long_probability, short_probability))
    expected = probability * positive - (1.0 - probability) * loss
    raw_lower = np.asarray(
        boosters["lower_quantile"].predict(
            actions.features, num_iteration=model.best_iterations["lower_quantile"]
        ),
        dtype=np.float64,
    ).reshape(len(selected), 2)
    raw_upper = np.asarray(
        boosters["upper_quantile"].predict(
            actions.features, num_iteration=model.best_iterations["upper_quantile"]
        ),
        dtype=np.float64,
    ).reshape(len(selected), 2)
    lower = np.minimum(raw_lower, raw_upper)
    upper = np.maximum(raw_lower, raw_upper)
    action_side = np.sign(expected[:, 0] - expected[:, 1]).astype(np.int8)
    direction_side = np.sign(conditional_long - 0.5).astype(np.int8)
    consensus = (action_side != 0) & (action_side == direction_side)
    return SelectiveActionPredictionBatch(
        endpoint_indexes=selected,
        long_mean_bps=expected[:, 0],
        short_mean_bps=expected[:, 1],
        long_profitable_probability=long_probability,
        short_profitable_probability=short_probability,
        abstain_probability=abstain_probability,
        opportunity_probability=opportunity,
        conditional_long_probability=conditional_long,
        long_lower_bps=lower[:, 0],
        short_lower_bps=lower[:, 1],
        long_upper_bps=upper[:, 0],
        short_upper_bps=upper[:, 1],
        action_preference_side=action_side,
        direction_preference_side=direction_side,
        side_consensus=np.asarray(consensus, dtype=bool),
    )


def ensemble_selective_action_predictions(
    members: Sequence[SelectiveActionPredictionBatch],
) -> SelectiveActionEnsembleBatch:
    """Combine members while retaining opportunity and direction probabilities."""

    values = tuple(members)
    if len(values) < 2:
        raise ValueError("selective-action ensemble requires at least two members")
    endpoints = np.asarray(values[0].endpoint_indexes, dtype=np.int64)
    if any(not np.array_equal(endpoints, value.endpoint_indexes) for value in values[1:]):
        raise ValueError("selective-action ensemble endpoint identities differ")
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
    opportunity = np.stack(
        [np.asarray(value.opportunity_probability, dtype=np.float64) for value in values]
    )
    direction = np.stack(
        [
            np.asarray(value.conditional_long_probability, dtype=np.float64)
            for value in values
        ]
    )
    consensus = np.stack(
        [np.asarray(value.side_consensus, dtype=bool) for value in values]
    )
    return SelectiveActionEnsembleBatch(
        action_values=action_values,
        opportunity_probability_mean=np.mean(opportunity, axis=0),
        opportunity_probability_std=np.std(opportunity, axis=0),
        conditional_long_probability_mean=np.mean(direction, axis=0),
        conditional_long_probability_std=np.std(direction, axis=0),
        opportunity_member_probabilities=opportunity,
        conditional_long_member_probabilities=direction,
        direction_long_member_ratio=np.mean(direction > 0.5, axis=0),
        direction_short_member_ratio=np.mean(direction < 0.5, axis=0),
        side_consensus_member_ratio=np.mean(consensus, axis=0),
        member_count=len(values),
    )


def save_selective_action_lightgbm_model(
    path: str | Path,
    model: TrainedSelectiveActionLightGBMModel,
) -> None:
    """Atomically persist a complete authority-denied selective model."""

    _validate_model_contract(model, reload_boosters=True)
    write_json_atomic(
        path,
        {**_model_payload(model), "model_sha256": model.model_sha256},
        indent=None,
        sort_keys=True,
    )


def load_selective_action_lightgbm_model(
    path: str | Path,
) -> TrainedSelectiveActionLightGBMModel:
    """Load and independently validate a selective-action artifact."""

    target = Path(path)
    try:
        size = target.stat().st_size
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("selective-action artifact is unreadable") from exc
    expected_fields = {field.name for field in fields(TrainedSelectiveActionLightGBMModel)}
    if (
        size <= 0
        or size > _MAX_ARTIFACT_BYTES
        or not isinstance(payload, dict)
        or set(payload) != expected_fields
    ):
        raise ValueError("selective-action artifact size or structure is invalid")
    authority_fields = (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    )
    if any(payload[name] is not False for name in authority_fields):
        raise ValueError("selective-action artifact contains authority")
    try:
        model = TrainedSelectiveActionLightGBMModel(
            schema_version=str(payload["schema_version"]),
            model_family=str(payload["model_family"]),
            spec=SelectiveActionLightGBMSpec(**payload["spec"]),
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
            class_support={
                str(role): {str(name): int(value) for name, value in support.items()}
                for role, support in payload["class_support"].items()
            },
            opportunity_calibration=tuple(
                float(value) for value in payload["opportunity_calibration"]
            ),
            direction_temperature=float(payload["direction_temperature"]),
            best_iterations={
                str(name): int(value) for name, value in payload["best_iterations"].items()
            },
            model_strings={
                str(name): str(value) for name, value in payload["model_strings"].items()
            },
            model_sha256=str(payload["model_sha256"]),
            trading_authority=payload["trading_authority"],
            execution_claim=payload["execution_claim"],
            profitability_claim=payload["profitability_claim"],
            portfolio_claim=payload["portfolio_claim"],
            leverage_applied=payload["leverage_applied"],
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("selective-action artifact values are invalid") from exc
    _validate_model_contract(model, reload_boosters=True)
    return model


__all__ = [
    "SELECTIVE_ACTION_LIGHTGBM_SCHEMA_VERSION",
    "SelectiveActionEnsembleBatch",
    "SelectiveActionLightGBMSpec",
    "SelectiveActionPredictionBatch",
    "TrainedSelectiveActionLightGBMModel",
    "ensemble_selective_action_predictions",
    "load_selective_action_lightgbm_model",
    "predict_selective_action_lightgbm_model",
    "save_selective_action_lightgbm_model",
    "train_selective_action_lightgbm_model",
]
