"""Shared action-conditional LightGBM hurdle and signed-advantage model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

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
from .probability_calibration import apply_platt_scaling, fit_platt_scaling
from .storage import write_json_atomic


SHARED_ACTION_LIGHTGBM_SCHEMA_VERSION = "shared-action-lightgbm-hurdle-v1"
_MODEL_FAMILY = (
    "shared_action_conditional_lightgbm_hurdle_with_signed_advantage_consensus"
)
_HEADS = (
    "probability",
    "positive_magnitude",
    "nonpositive_loss_magnitude",
    "lower_quantile",
    "upper_quantile",
    "signed_advantage",
)
_MINIMUM_TRAIN_CLASS_ROWS = 512
_MINIMUM_EARLY_STOP_CLASS_ROWS = 128
_MINIMUM_CALIBRATION_CLASS_ROWS = 512
_MINIMUM_CONDITIONAL_ROWS = 128
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
SharedActionProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class SharedActionLightGBMSpec:
    """Precommitted shared-tree, calibration, and quantile contract."""

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
            raise ValueError("shared-action LightGBM specification is invalid")


@dataclass(frozen=True)
class TrainedSharedActionLightGBMModel:
    """Hash-bound shared action model with no execution authority."""

    schema_version: str
    model_family: str
    spec: SharedActionLightGBMSpec
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
    positive_class_prevalence: float
    class_support: Mapping[str, Mapping[str, int]]
    probability_calibration: tuple[float, float]
    advantage_validation_directional_loss: float
    best_iterations: Mapping[str, int]
    model_strings: Mapping[str, str]
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False


@dataclass(frozen=True)
class SharedActionPredictionBatch:
    """One member's paired action values and independent side consensus."""

    endpoint_indexes: np.ndarray
    long_mean_bps: np.ndarray
    short_mean_bps: np.ndarray
    long_profitable_probability: np.ndarray
    short_profitable_probability: np.ndarray
    long_lower_bps: np.ndarray
    short_lower_bps: np.ndarray
    long_upper_bps: np.ndarray
    short_upper_bps: np.ndarray
    signed_advantage_bps: np.ndarray
    action_preference_side: np.ndarray
    advantage_preference_side: np.ndarray
    side_consensus: np.ndarray

    def __post_init__(self) -> None:
        endpoints = np.asarray(self.endpoint_indexes, dtype=np.int64)
        rows = len(endpoints)
        numeric = (
            self.long_mean_bps,
            self.short_mean_bps,
            self.long_profitable_probability,
            self.short_profitable_probability,
            self.long_lower_bps,
            self.short_lower_bps,
            self.long_upper_bps,
            self.short_upper_bps,
            self.signed_advantage_bps,
        )
        sides = (self.action_preference_side, self.advantage_preference_side)
        if (
            rows <= 0
            or endpoints.ndim != 1
            or np.any(np.diff(endpoints) <= 0)
            or any(np.asarray(value).shape != (rows,) for value in numeric)
            or any(not np.all(np.isfinite(value)) for value in numeric)
            or any(np.asarray(value).shape != (rows,) for value in sides)
            or any(not set(np.unique(value)).issubset({-1, 0, 1}) for value in sides)
            or np.asarray(self.side_consensus).shape != (rows,)
            or np.asarray(self.side_consensus).dtype != np.bool_
        ):
            raise ValueError("shared-action prediction batch is invalid")

    @property
    def rows(self) -> int:
        return len(self.endpoint_indexes)


@dataclass(frozen=True)
class SharedActionEnsembleBatch:
    """Ensembled action values plus signed-advantage agreement evidence."""

    action_values: ActionValueEnsembleBatch
    signed_advantage_mean_bps: np.ndarray
    signed_advantage_epistemic_std_bps: np.ndarray
    advantage_long_member_ratio: np.ndarray
    advantage_short_member_ratio: np.ndarray
    side_consensus_member_ratio: np.ndarray
    member_count: int
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def __post_init__(self) -> None:
        rows = self.action_values.rows
        arrays = (
            self.signed_advantage_mean_bps,
            self.signed_advantage_epistemic_std_bps,
            self.advantage_long_member_ratio,
            self.advantage_short_member_ratio,
            self.side_consensus_member_ratio,
        )
        ratios = arrays[2:]
        if (
            self.member_count != self.action_values.member_count
            or self.member_count < 2
            or any(np.asarray(value).shape != (rows,) for value in arrays)
            or any(not np.all(np.isfinite(value)) for value in arrays)
            or any(np.any(value < 0.0) or np.any(value > 1.0) for value in ratios)
            or np.any(self.signed_advantage_epistemic_std_bps < 0.0)
            or self.trading_authority
            or self.execution_claim
            or self.profitability_claim
            or self.portfolio_claim
            or self.leverage_applied
        ):
            raise ValueError("shared-action ensemble batch is invalid")

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


def _model_payload(model: TrainedSharedActionLightGBMModel) -> dict[str, object]:
    payload = asdict(model)
    payload.pop("model_sha256")
    return payload


def _model_sha256(model: TrainedSharedActionLightGBMModel) -> str:
    return hashlib.sha256(
        _canonical_json(_model_payload(model)).encode("ascii")
    ).hexdigest()


def _class_support(labels: np.ndarray) -> dict[str, int]:
    values = np.asarray(labels, dtype=np.float32)
    return {
        "profitable_rows": int(np.sum(values == 1.0)),
        "non_profitable_rows": int(np.sum(values == 0.0)),
    }


def _require_class_support(
    labels: np.ndarray,
    *,
    role: str,
    minimum: int,
) -> dict[str, int]:
    support = _class_support(labels)
    if min(support.values()) < minimum:
        raise ValueError(
            f"shared-action {role} class support is insufficient: "
            f"profitable={support['profitable_rows']} "
            f"non_profitable={support['non_profitable_rows']} "
            f"required_each={minimum}"
        )
    return support


def _paired_values(long_values: np.ndarray, short_values: np.ndarray) -> np.ndarray:
    if long_values.shape != short_values.shape or long_values.ndim != 1:
        raise ValueError("shared-action paired targets are invalid")
    values = np.empty(len(long_values) * 2, dtype=np.float32)
    values[0::2] = np.asarray(long_values, dtype=np.float32)
    values[1::2] = np.asarray(short_values, dtype=np.float32)
    return values


def _paired_weights(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float32)
    return np.repeat(values * np.float32(0.5), 2)


def _directional_loss(
    predictions: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    predicted = np.asarray(predictions, dtype=np.float64)
    actual = np.asarray(labels, dtype=np.float64)
    sample_weights = (
        np.ones(len(actual), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    magnitude = sample_weights * np.abs(actual)
    denominator = float(np.sum(magnitude))
    if (
        predicted.shape != actual.shape
        or sample_weights.shape != actual.shape
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(sample_weights))
        or denominator <= 0.0
    ):
        raise ValueError("shared-action directional metric inputs are invalid")
    wrong = np.signbit(predicted) != np.signbit(actual)
    wrong |= predicted == 0.0
    return float(np.sum(magnitude[wrong]) / denominator)


def _directional_loss_metric(
    predictions: np.ndarray,
    data: lgb.Dataset,
) -> tuple[str, float, bool]:
    return (
        "magnitude_weighted_directional_loss",
        _directional_loss(predictions, data.get_label(), data.get_weight()),
        False,
    )


def _train_booster(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    x_early_stop: np.ndarray,
    y_early_stop: np.ndarray,
    early_stop_weights: np.ndarray,
    parameters: Mapping[str, object],
    objective: str,
    metric: str,
    num_boost_round: int,
    early_stopping_rounds: int,
    alpha: float | None = None,
    directional_selection: bool = False,
) -> tuple[lgb.Booster, int]:
    config = {
        **parameters,
        "objective": objective,
        "metric": "None" if directional_selection else metric,
    }
    if alpha is not None:
        config["alpha"] = float(alpha)
    training = lgb.Dataset(
        x_train,
        label=y_train,
        weight=train_weights,
        free_raw_data=False,
        feature_name="auto",
    )
    early_stop = lgb.Dataset(
        x_early_stop,
        label=y_early_stop,
        weight=early_stop_weights,
        reference=training,
        free_raw_data=False,
        feature_name="auto",
    )
    booster = lgb.train(
        config,
        training,
        num_boost_round=int(num_boost_round),
        valid_sets=[early_stop],
        valid_names=["early_stop"],
        feval=_directional_loss_metric if directional_selection else None,
        callbacks=[
            lgb.early_stopping(int(early_stopping_rounds), verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    iteration = max(1, int(booster.best_iteration or booster.current_iteration()))
    return booster, iteration


def _validate_model_contract(
    model: TrainedSharedActionLightGBMModel,
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
    role_rows = {
        "train": model.training_event_rows * 2,
        "early_stop": model.early_stop_event_rows * 2,
        "calibration": model.calibration_event_rows * 2,
    }
    if (
        model.schema_version != SHARED_ACTION_LIGHTGBM_SCHEMA_VERSION
        or model.model_family != _MODEL_FAMILY
        or model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.portfolio_claim
        or model.leverage_applied
        or model.source_feature_version != MICROSTRUCTURE_FEATURE_VERSION
        or model.source_feature_names != MICROSTRUCTURE_FEATURE_NAMES
        or model.action_feature_schema_version != ACTION_FEATURE_SCHEMA_VERSION
        or model.action_canonicalization_sha256
        != ACTION_CANONICALIZATION_SHA256
        or model.action_feature_names != ACTION_CONDITIONAL_FEATURE_NAMES
        or model.target_mode != ADAPTIVE_BARRIER_TARGET_MODE
        or model.target_scenario != "stress"
        or model.target_contract_sha256 != expected_target_hash
        or model.backend_kind not in {"opencl", "cpu"}
        or not model.backend_device
        or model.lightgbm_version != str(lgb.__version__)
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
        or not math.isfinite(model.positive_class_prevalence)
        or not 0.0 < model.positive_class_prevalence < 1.0
        or set(model.class_support) != set(role_rows)
        or len(model.probability_calibration) != 2
        or not 0.05 <= float(model.probability_calibration[0]) <= 10.0
        or not -10.0 <= float(model.probability_calibration[1]) <= 10.0
        or not math.isfinite(model.advantage_validation_directional_loss)
        or not 0.0 <= model.advantage_validation_directional_loss <= 1.0
        or set(model.best_iterations) != set(_HEADS)
        or set(model.model_strings) != set(_HEADS)
        or any(
            isinstance(value, bool)
            or not 1 <= int(value) <= model.spec.num_boost_round
            for value in model.best_iterations.values()
        )
        or any(not isinstance(value, str) or not value for value in model.model_strings.values())
        or model.model_sha256 != _model_sha256(model)
    ):
        raise ValueError("shared-action LightGBM model contract is invalid")
    for role, expected_rows in role_rows.items():
        support = model.class_support[role]
        if (
            set(support) != {"profitable_rows", "non_profitable_rows"}
            or any(isinstance(value, bool) or int(value) < 0 for value in support.values())
            or sum(int(value) for value in support.values()) != expected_rows
        ):
            raise ValueError("shared-action class-support contract is invalid")
    expected_prevalence = (
        model.class_support["train"]["profitable_rows"]
        / (model.training_event_rows * 2)
    )
    if not math.isclose(
        model.positive_class_prevalence,
        expected_prevalence,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("shared-action prevalence contract is invalid")
    if reload_boosters:
        try:
            for name in _HEADS:
                lgb.Booster(model_str=model.model_strings[name])
        except lgb.basic.LightGBMError as exc:
            raise ValueError("shared-action booster payload cannot be reloaded") from exc


def train_shared_action_lightgbm_model(
    dataset: MicrostructureDataset,
    barrier_targets: AdaptiveBarrierTargets,
    *,
    train_endpoints: np.ndarray,
    tuning_endpoints: np.ndarray,
    spec: SharedActionLightGBMSpec,
    compute_backend: str,
    seed: int,
    train_sample_weights: np.ndarray | None = None,
    tuning_sample_weights: np.ndarray | None = None,
    progress: SharedActionProgressCallback | None = None,
) -> TrainedSharedActionLightGBMModel:
    """Fit one research-only member without accessing later roles."""

    validate_microstructure_dataset(dataset)
    validate_adaptive_barrier_targets(dataset, barrier_targets)
    if (
        dataset.feature_version != MICROSTRUCTURE_FEATURE_VERSION
        or dataset.feature_names != MICROSTRUCTURE_FEATURE_NAMES
    ):
        raise ValueError("shared-action source feature contract is unsupported")
    train = _validate_indexes(
        train_endpoints,
        rows=dataset.rows,
        label="shared-action training",
        minimum_rows=512,
    )
    requested_tuning = _validate_indexes(
        tuning_endpoints,
        rows=dataset.rows,
        label="shared-action tuning",
        minimum_rows=1_024,
    )
    if train[-1] >= requested_tuning[0]:
        raise ValueError("shared-action training and tuning roles overlap")
    train_weights = _validate_weights(
        train_sample_weights,
        rows=len(train),
        label="shared-action training",
    )
    requested_tuning_weights = _validate_weights(
        tuning_sample_weights,
        rows=len(requested_tuning),
        label="shared-action tuning",
    )
    targets, exits = _target_arrays(dataset, barrier_targets, scenario="stress")
    if any(
        not np.all(np.isfinite(targets[side][train]))
        or not np.all(np.isfinite(targets[side][requested_tuning]))
        or np.any(exits[side][train] <= dataset.decision_time_ms[train])
        or np.any(
            exits[side][requested_tuning]
            <= dataset.decision_time_ms[requested_tuning]
        )
        for side in ("long", "short")
    ):
        raise ValueError("shared-action roles contain invalid barrier labels")
    maximum_exit = np.maximum(exits["long"], exits["short"])
    if np.any(maximum_exit[train] >= dataset.decision_time_ms[requested_tuning[0]]):
        raise ValueError("shared-action training labels overlap tuning")

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
    early_stop_weights = early_candidate_weights[early_keep]
    internal_purged_rows = int(np.sum(~early_keep))
    if min(len(early_stop), len(calibration)) < 256:
        raise ValueError("shared-action purged tuning roles are too small")

    backend_parameters, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        int(seed),
        reproducible=True,
    )
    if backend_kind == "opencl" and (
        spec.gpu_use_dp_required is not True
        or backend_parameters.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("shared-action LightGBM OpenCL FP64 is required")
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
    source_features = np.asarray(dataset.features, dtype=np.float32)
    train_action = build_action_conditional_features(source_features[train])
    early_action = build_action_conditional_features(source_features[early_stop])
    calibration_action = build_action_conditional_features(
        source_features[calibration]
    )
    train_target = _paired_values(targets["long"][train], targets["short"][train])
    early_target = _paired_values(
        targets["long"][early_stop], targets["short"][early_stop]
    )
    calibration_target = _paired_values(
        targets["long"][calibration], targets["short"][calibration]
    )
    action_train_weights = _paired_weights(train_weights)
    action_early_weights = _paired_weights(early_stop_weights)
    train_labels = (train_target > 0.0).astype(np.float32)
    early_labels = (early_target > 0.0).astype(np.float32)
    calibration_labels = (calibration_target > 0.0).astype(np.float32)
    support = {
        "train": _require_class_support(
            train_labels,
            role="training",
            minimum=_MINIMUM_TRAIN_CLASS_ROWS,
        ),
        "early_stop": _require_class_support(
            early_labels,
            role="early-stop",
            minimum=_MINIMUM_EARLY_STOP_CLASS_ROWS,
        ),
        "calibration": _require_class_support(
            calibration_labels,
            role="calibration",
            minimum=_MINIMUM_CALIBRATION_CLASS_ROWS,
        ),
    }
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
        directional_selection: bool = False,
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
            directional_selection=directional_selection,
        )
        boosters[name] = booster
        iterations[name] = iteration
        step += 1

    train_head(
        "probability",
        x_train=train_action.features,
        y_train=train_labels,
        head_train_weights=action_train_weights,
        x_early=early_action.features,
        y_early=early_labels,
        head_early_weights=action_early_weights,
        objective="binary",
        metric="binary_logloss",
    )
    raw_calibration = np.asarray(
        boosters["probability"].predict(
            calibration_action.features,
            num_iteration=iterations["probability"],
        ),
        dtype=np.float64,
    )
    calibration_parameters = fit_platt_scaling(
        raw_calibration,
        calibration_labels,
    )
    positive_train = train_labels == 1.0
    positive_early = early_labels == 1.0
    nonpositive_train = ~positive_train
    nonpositive_early = ~positive_early
    if min(
        int(np.sum(positive_train)),
        int(np.sum(positive_early)),
        int(np.sum(nonpositive_train)),
        int(np.sum(nonpositive_early)),
    ) < _MINIMUM_CONDITIONAL_ROWS:
        raise ValueError("shared-action conditional magnitude support is insufficient")
    train_head(
        "positive_magnitude",
        x_train=train_action.features[positive_train],
        y_train=train_target[positive_train],
        head_train_weights=action_train_weights[positive_train],
        x_early=early_action.features[positive_early],
        y_early=early_target[positive_early],
        head_early_weights=action_early_weights[positive_early],
        objective="regression",
        metric="l2",
    )
    train_head(
        "nonpositive_loss_magnitude",
        x_train=train_action.features[nonpositive_train],
        y_train=-train_target[nonpositive_train],
        head_train_weights=action_train_weights[nonpositive_train],
        x_early=early_action.features[nonpositive_early],
        y_early=-early_target[nonpositive_early],
        head_early_weights=action_early_weights[nonpositive_early],
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
    train_advantage = targets["long"][train] - targets["short"][train]
    early_advantage = (
        targets["long"][early_stop] - targets["short"][early_stop]
    )
    mirrored_train = mirror_microstructure_direction(source_features[train])
    mirrored_early = mirror_microstructure_direction(source_features[early_stop])
    train_head(
        "signed_advantage",
        x_train=np.concatenate((source_features[train], mirrored_train), axis=0),
        y_train=np.concatenate((train_advantage, -train_advantage), axis=0),
        head_train_weights=np.concatenate(
            (train_weights * 0.5, train_weights * 0.5), axis=0
        ),
        x_early=np.concatenate((source_features[early_stop], mirrored_early), axis=0),
        y_early=np.concatenate((early_advantage, -early_advantage), axis=0),
        head_early_weights=np.concatenate(
            (early_stop_weights * 0.5, early_stop_weights * 0.5), axis=0
        ),
        objective="regression",
        metric="None",
        directional_selection=True,
    )
    best_score = boosters["signed_advantage"].best_score.get("early_stop", {})
    directional_loss = float(
        best_score.get("magnitude_weighted_directional_loss", math.nan)
    )
    model_strings = {
        name: boosters[name].model_to_string(num_iteration=iterations[name])
        for name in _HEADS
    }
    provisional = TrainedSharedActionLightGBMModel(
        schema_version=SHARED_ACTION_LIGHTGBM_SCHEMA_VERSION,
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
        positive_class_prevalence=float(
            support["train"]["profitable_rows"] / (len(train) * 2)
        ),
        class_support=support,
        probability_calibration=calibration_parameters,
        advantage_validation_directional_loss=directional_loss,
        best_iterations=iterations,
        model_strings=model_strings,
        model_sha256="",
    )
    model = TrainedSharedActionLightGBMModel(
        **{**provisional.__dict__, "model_sha256": _model_sha256(provisional)}
    )
    _validate_model_contract(model, reload_boosters=False)
    return model


def predict_shared_action_lightgbm_model(
    model: TrainedSharedActionLightGBMModel,
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
) -> SharedActionPredictionBatch:
    """Predict paired values and an antisymmetric signed advantage."""

    validate_microstructure_dataset(dataset)
    _validate_model_contract(model, reload_boosters=False)
    if (
        dataset.feature_version != model.source_feature_version
        or dataset.feature_names != model.source_feature_names
    ):
        raise ValueError("shared-action prediction feature contract drifted")
    selected = _validate_indexes(
        endpoints,
        rows=dataset.rows,
        label="shared-action prediction",
        minimum_rows=1,
    )
    try:
        boosters = {
            name: lgb.Booster(model_str=model.model_strings[name])
            for name in _HEADS
        }
    except lgb.basic.LightGBMError as exc:
        raise ValueError("shared-action booster payload cannot be reloaded") from exc
    source = np.asarray(dataset.features[selected], dtype=np.float32)
    actions = build_action_conditional_features(source)
    raw_probability = np.asarray(
        boosters["probability"].predict(
            actions.features,
            num_iteration=model.best_iterations["probability"],
        ),
        dtype=np.float64,
    )
    probability = apply_platt_scaling(
        raw_probability,
        model.probability_calibration,
    )
    positive = np.maximum(
        0.0,
        np.asarray(
            boosters["positive_magnitude"].predict(
                actions.features,
                num_iteration=model.best_iterations["positive_magnitude"],
            ),
            dtype=np.float64,
        ),
    )
    loss = np.maximum(
        0.0,
        np.asarray(
            boosters["nonpositive_loss_magnitude"].predict(
                actions.features,
                num_iteration=model.best_iterations[
                    "nonpositive_loss_magnitude"
                ],
            ),
            dtype=np.float64,
        ),
    )
    expected = probability * positive - (1.0 - probability) * loss
    raw_lower = np.asarray(
        boosters["lower_quantile"].predict(
            actions.features,
            num_iteration=model.best_iterations["lower_quantile"],
        ),
        dtype=np.float64,
    )
    raw_upper = np.asarray(
        boosters["upper_quantile"].predict(
            actions.features,
            num_iteration=model.best_iterations["upper_quantile"],
        ),
        dtype=np.float64,
    )
    lower = np.minimum(raw_lower, raw_upper)
    upper = np.maximum(raw_lower, raw_upper)
    mirrored = mirror_microstructure_direction(source)
    advantage_forward = np.asarray(
        boosters["signed_advantage"].predict(
            source,
            num_iteration=model.best_iterations["signed_advantage"],
        ),
        dtype=np.float64,
    )
    advantage_mirrored = np.asarray(
        boosters["signed_advantage"].predict(
            mirrored,
            num_iteration=model.best_iterations["signed_advantage"],
        ),
        dtype=np.float64,
    )
    signed_advantage = 0.5 * (advantage_forward - advantage_mirrored)
    expected = expected.reshape(len(selected), 2)
    probability = probability.reshape(len(selected), 2)
    lower = lower.reshape(len(selected), 2)
    upper = upper.reshape(len(selected), 2)
    action_side = np.sign(expected[:, 0] - expected[:, 1]).astype(np.int8)
    advantage_side = np.sign(signed_advantage).astype(np.int8)
    consensus = (action_side != 0) & (action_side == advantage_side)
    return SharedActionPredictionBatch(
        endpoint_indexes=selected,
        long_mean_bps=expected[:, 0],
        short_mean_bps=expected[:, 1],
        long_profitable_probability=probability[:, 0],
        short_profitable_probability=probability[:, 1],
        long_lower_bps=lower[:, 0],
        short_lower_bps=lower[:, 1],
        long_upper_bps=upper[:, 0],
        short_upper_bps=upper[:, 1],
        signed_advantage_bps=signed_advantage,
        action_preference_side=action_side,
        advantage_preference_side=advantage_side,
        side_consensus=np.asarray(consensus, dtype=bool),
    )


def ensemble_shared_action_predictions(
    members: list[SharedActionPredictionBatch]
    | tuple[SharedActionPredictionBatch, ...],
) -> SharedActionEnsembleBatch:
    """Combine members while retaining side-consensus diagnostics."""

    values = tuple(members)
    if len(values) < 2:
        raise ValueError("shared-action ensemble requires at least two members")
    endpoints = np.asarray(values[0].endpoint_indexes, dtype=np.int64)
    if any(
        not np.array_equal(endpoints, value.endpoint_indexes)
        for value in values[1:]
    ):
        raise ValueError("shared-action ensemble endpoint identities differ")
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
    advantage = np.stack(
        [np.asarray(value.signed_advantage_bps, dtype=np.float64) for value in values],
        axis=0,
    )
    consensus = np.stack(
        [np.asarray(value.side_consensus, dtype=bool) for value in values],
        axis=0,
    )
    if (
        advantage.shape != (len(values), len(endpoints))
        or consensus.shape != advantage.shape
        or not np.all(np.isfinite(advantage))
    ):
        raise ValueError("shared-action ensemble member evidence is invalid")
    return SharedActionEnsembleBatch(
        action_values=action_values,
        signed_advantage_mean_bps=np.mean(advantage, axis=0),
        signed_advantage_epistemic_std_bps=np.std(advantage, axis=0),
        advantage_long_member_ratio=np.mean(advantage > 0.0, axis=0),
        advantage_short_member_ratio=np.mean(advantage < 0.0, axis=0),
        side_consensus_member_ratio=np.mean(consensus, axis=0),
        member_count=len(values),
    )


def save_shared_action_lightgbm_model(
    path: str | Path,
    model: TrainedSharedActionLightGBMModel,
) -> None:
    """Atomically persist the model and authority-denial contract."""

    _validate_model_contract(model, reload_boosters=True)
    write_json_atomic(
        path,
        {**_model_payload(model), "model_sha256": model.model_sha256},
        indent=None,
        sort_keys=True,
    )


def load_shared_action_lightgbm_model(
    path: str | Path,
) -> TrainedSharedActionLightGBMModel:
    """Load and independently validate a complete shared-action artifact."""

    target = Path(path)
    try:
        size = target.stat().st_size
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("shared-action artifact is unreadable") from exc
    if size <= 0 or size > _MAX_ARTIFACT_BYTES or not isinstance(payload, dict):
        raise ValueError("shared-action artifact size or structure is invalid")
    if set(payload) != {field.name for field in fields(TrainedSharedActionLightGBMModel)}:
        raise ValueError("shared-action artifact contract is incomplete")
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
    )
    if any(payload[name] is not False for name in authority_fields) or any(
        not isinstance(payload[name], int) or isinstance(payload[name], bool)
        for name in integer_fields
    ):
        raise ValueError("shared-action artifact scalar types are invalid")
    try:
        model = TrainedSharedActionLightGBMModel(
            schema_version=str(payload["schema_version"]),
            model_family=str(payload["model_family"]),
            spec=SharedActionLightGBMSpec(**payload["spec"]),
            source_feature_version=str(payload["source_feature_version"]),
            source_feature_names=tuple(
                str(value) for value in payload["source_feature_names"]
            ),
            action_feature_schema_version=str(
                payload["action_feature_schema_version"]
            ),
            action_canonicalization_sha256=str(
                payload["action_canonicalization_sha256"]
            ),
            action_feature_names=tuple(
                str(value) for value in payload["action_feature_names"]
            ),
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
            requested_tuning_event_rows=int(
                payload["requested_tuning_event_rows"]
            ),
            early_stop_event_rows=int(payload["early_stop_event_rows"]),
            calibration_event_rows=int(payload["calibration_event_rows"]),
            calibration_start_ms=int(payload["calibration_start_ms"]),
            internal_purged_event_rows=int(
                payload["internal_purged_event_rows"]
            ),
            positive_class_prevalence=float(
                payload["positive_class_prevalence"]
            ),
            class_support={
                str(role): {
                    str(name): int(value) for name, value in support.items()
                }
                for role, support in payload["class_support"].items()
            },
            probability_calibration=tuple(
                float(value) for value in payload["probability_calibration"]
            ),
            advantage_validation_directional_loss=float(
                payload["advantage_validation_directional_loss"]
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
        raise ValueError("shared-action artifact values are invalid") from exc
    _validate_model_contract(model, reload_boosters=True)
    return model


__all__ = [
    "SHARED_ACTION_LIGHTGBM_SCHEMA_VERSION",
    "SharedActionEnsembleBatch",
    "SharedActionLightGBMSpec",
    "SharedActionPredictionBatch",
    "TrainedSharedActionLightGBMModel",
    "ensemble_shared_action_predictions",
    "load_shared_action_lightgbm_model",
    "predict_shared_action_lightgbm_model",
    "save_shared_action_lightgbm_model",
    "train_shared_action_lightgbm_model",
]
