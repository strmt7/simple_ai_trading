"""GPU LightGBM hurdle ensemble member for adaptive BBO action values."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import lightgbm as lgb
import numpy as np

from .lightgbm_backend import (
    SUPPORTED_LIGHTGBM_BACKEND_KINDS,
    lightgbm_backend_parameters,
)
from .microstructure_action_architecture import ActionValuePredictionBatch
from .microstructure_barriers import (
    ADAPTIVE_BARRIER_TARGET_MODE,
    AdaptiveBarrierSpec,
    AdaptiveBarrierTargets,
    validate_adaptive_barrier_targets,
)
from .microstructure_features import (
    MicrostructureDataset,
    microstructure_feature_names,
    validate_microstructure_dataset,
)
from .probability_calibration import apply_platt_scaling, fit_platt_scaling
from .storage import write_json_atomic


LIGHTGBM_HURDLE_SCHEMA_VERSION = "adaptive-lightgbm-hurdle-action-value-v1"
_MODEL_FAMILY = "side_specific_lightgbm_hurdle_expected_value"
_SIDES = ("long", "short")
_HEADS = (
    "probability",
    "positive_magnitude",
    "nonpositive_loss_magnitude",
    "lower_quantile",
    "upper_quantile",
)
_BOOSTER_NAMES = tuple(f"{side}_{head}" for side in _SIDES for head in _HEADS)
_MINIMUM_TRAIN_CLASS_ROWS = 256
_MINIMUM_EARLY_STOP_CLASS_ROWS = 64
_MINIMUM_CALIBRATION_CLASS_ROWS = 256
_MINIMUM_CONDITIONAL_ROWS = 64
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
LightGBMProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class LightGBMHurdleSpec:
    """Precommitted tree, calibration, and quantile contract."""

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
            or not isinstance(self.family, str)
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
            raise ValueError("LightGBM hurdle specification is invalid")


@dataclass(frozen=True)
class TrainedLightGBMHurdleModel:
    """A research-only, hash-bound side-specific hurdle model."""

    schema_version: str
    model_family: str
    spec: LightGBMHurdleSpec
    feature_version: str
    feature_names: tuple[str, ...]
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
    training_rows: int
    requested_early_stop_rows: int
    early_stop_rows: int
    calibration_rows: int
    calibration_start_ms: int
    internal_purged_rows: int
    positive_class_prevalence: tuple[float, float]
    class_support: Mapping[str, Mapping[str, Mapping[str, int]]]
    probability_calibration: Mapping[str, tuple[float, float]]
    best_iterations: Mapping[str, int]
    model_strings: Mapping[str, str]
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _target_contract_sha256(targets: AdaptiveBarrierTargets) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "schema_version": targets.schema_version,
                "target_mode": targets.target_mode,
                "spec": asdict(targets.spec),
            }
        ).encode("ascii")
    ).hexdigest()


def _model_payload(model: TrainedLightGBMHurdleModel) -> dict[str, object]:
    payload = asdict(model)
    payload.pop("model_sha256")
    return payload


def _model_sha256(model: TrainedLightGBMHurdleModel) -> str:
    return hashlib.sha256(_canonical_json(_model_payload(model)).encode("ascii")).hexdigest()


def _validate_indexes(
    indexes: np.ndarray,
    *,
    rows: int,
    label: str,
    minimum_rows: int,
) -> np.ndarray:
    values = np.asarray(indexes, dtype=np.int64)
    if (
        values.ndim != 1
        or len(values) < minimum_rows
        or values[0] < 0
        or values[-1] >= rows
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError(f"LightGBM hurdle {label} endpoints are invalid")
    return values


def _validate_weights(
    weights: np.ndarray | None,
    *,
    rows: int,
    label: str,
) -> np.ndarray:
    values = (
        np.ones(rows, dtype=np.float32)
        if weights is None
        else np.asarray(weights, dtype=np.float32)
    )
    if (
        values.shape != (rows,)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError(f"LightGBM hurdle {label} sample weights are invalid")
    return values


def _target_arrays(
    dataset: MicrostructureDataset,
    targets: AdaptiveBarrierTargets,
    *,
    scenario: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if scenario == "base":
        source_values = {
            "long": targets.base_long_net_bps,
            "short": targets.base_short_net_bps,
        }
        source_exits = {
            "long": targets.base_long_exit_time_ms,
            "short": targets.base_short_exit_time_ms,
        }
    elif scenario == "stress":
        source_values = {
            "long": targets.stress_long_net_bps,
            "short": targets.stress_short_net_bps,
        }
        source_exits = {
            "long": targets.stress_long_exit_time_ms,
            "short": targets.stress_short_exit_time_ms,
        }
    else:
        raise ValueError("LightGBM hurdle target scenario is unsupported")
    valid_positions = np.flatnonzero(targets.valid)
    valid_sources = np.asarray(targets.source_indexes[valid_positions], dtype=np.int64)
    values: dict[str, np.ndarray] = {}
    exits: dict[str, np.ndarray] = {}
    for side in _SIDES:
        side_values = np.full(dataset.rows, np.nan, dtype=np.float32)
        side_exits = np.full(dataset.rows, -1, dtype=np.int64)
        side_values[valid_sources] = np.asarray(
            source_values[side][valid_positions], dtype=np.float32
        )
        side_exits[valid_sources] = np.asarray(
            source_exits[side][valid_positions], dtype=np.int64
        )
        values[side] = side_values
        exits[side] = side_exits
    return values, exits


def _class_support(labels: np.ndarray) -> dict[str, int]:
    values = np.asarray(labels, dtype=np.float32)
    return {
        "profitable_rows": int(np.sum(values == 1.0)),
        "non_profitable_rows": int(np.sum(values == 0.0)),
    }


def _require_class_support(
    labels: np.ndarray,
    *,
    side: str,
    role: str,
    minimum: int,
) -> dict[str, int]:
    support = _class_support(labels)
    if min(support.values()) < minimum:
        raise ValueError(
            f"LightGBM hurdle {side} {role} class support is insufficient: "
            f"profitable={support['profitable_rows']} "
            f"non_profitable={support['non_profitable_rows']} required_each={minimum}"
        )
    return support


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
) -> tuple[lgb.Booster, int]:
    config = {**parameters, "objective": objective, "metric": metric}
    if alpha is not None:
        config["alpha"] = float(alpha)
    training = lgb.Dataset(
        x_train,
        label=y_train,
        weight=train_weights,
        free_raw_data=False,
    )
    early_stop = lgb.Dataset(
        x_early_stop,
        label=y_early_stop,
        weight=early_stop_weights,
        reference=training,
        free_raw_data=False,
    )
    booster = lgb.train(
        config,
        training,
        num_boost_round=int(num_boost_round),
        valid_sets=[early_stop],
        valid_names=["early_stop"],
        callbacks=[
            lgb.early_stopping(int(early_stopping_rounds), verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    iteration = max(1, int(booster.best_iteration or booster.current_iteration()))
    return booster, iteration


def _validate_model_contract(
    model: TrainedLightGBMHurdleModel,
    *,
    reload_boosters: bool,
) -> None:
    try:
        expected_names = microstructure_feature_names(model.feature_version)
    except ValueError as exc:
        raise ValueError("LightGBM hurdle feature contract is unsupported") from exc
    expected_target_hash = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": model.target_schema_version,
                "target_mode": model.target_mode,
                "spec": asdict(model.target_spec),
            }
        ).encode("ascii")
    ).hexdigest()
    expected_class_roles = {"train", "early_stop", "calibration"}
    expected_role_rows = {
        "train": model.training_rows,
        "early_stop": model.early_stop_rows,
        "calibration": model.calibration_rows,
    }
    if (
        model.schema_version != LIGHTGBM_HURDLE_SCHEMA_VERSION
        or model.model_family != _MODEL_FAMILY
        or model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.portfolio_claim
        or model.leverage_applied
        or model.feature_names != expected_names
        or model.target_mode != ADAPTIVE_BARRIER_TARGET_MODE
        or model.target_scenario not in {"base", "stress"}
        or model.target_contract_sha256 != expected_target_hash
        or model.backend_kind not in SUPPORTED_LIGHTGBM_BACKEND_KINDS
        or not str(model.backend_device).strip()
        or model.lightgbm_version != str(lgb.__version__)
        or isinstance(model.seed, bool)
        or min(
            model.training_rows,
            model.requested_early_stop_rows,
            model.early_stop_rows,
            model.calibration_rows,
        ) <= 0
        or model.early_stop_rows + model.calibration_rows + model.internal_purged_rows
        != model.requested_early_stop_rows
        or model.calibration_start_ms <= 0
        or model.internal_purged_rows < 0
        or len(model.positive_class_prevalence) != 2
        or any(
            not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0
            for value in model.positive_class_prevalence
        )
        or set(model.class_support) != set(_SIDES)
        or any(set(model.class_support[side]) != expected_class_roles for side in _SIDES)
        or set(model.probability_calibration) != set(_SIDES)
        or any(
            len(model.probability_calibration[side]) != 2
            or not 0.05 <= float(model.probability_calibration[side][0]) <= 10.0
            or not -10.0 <= float(model.probability_calibration[side][1]) <= 10.0
            for side in _SIDES
        )
        or set(model.best_iterations) != set(_BOOSTER_NAMES)
        or any(
            isinstance(value, bool)
            or not 1 <= int(value) <= model.spec.num_boost_round
            for value in model.best_iterations.values()
        )
        or set(model.model_strings) != set(_BOOSTER_NAMES)
        or any(not isinstance(value, str) or not value for value in model.model_strings.values())
        or model.model_sha256 != _model_sha256(model)
    ):
        raise ValueError("LightGBM hurdle model contract is invalid")
    for side in _SIDES:
        for role in expected_class_roles:
            support = model.class_support[side][role]
            if (
                set(support) != {"profitable_rows", "non_profitable_rows"}
                or any(isinstance(value, bool) or int(value) < 0 for value in support.values())
                or sum(int(value) for value in support.values())
                != expected_role_rows[role]
            ):
                raise ValueError("LightGBM hurdle class-support contract is invalid")
        expected_prevalence = (
            model.class_support[side]["train"]["profitable_rows"]
            / model.training_rows
        )
        side_index = _SIDES.index(side)
        if not math.isclose(
            float(model.positive_class_prevalence[side_index]),
            float(expected_prevalence),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("LightGBM hurdle prevalence contract is invalid")
    if reload_boosters:
        try:
            for name in _BOOSTER_NAMES:
                lgb.Booster(model_str=model.model_strings[name])
        except lgb.basic.LightGBMError as exc:
            raise ValueError("LightGBM hurdle booster payload cannot be reloaded") from exc


def train_lightgbm_hurdle_model(
    dataset: MicrostructureDataset,
    barrier_targets: AdaptiveBarrierTargets,
    *,
    train_endpoints: np.ndarray,
    tuning_endpoints: np.ndarray,
    spec: LightGBMHurdleSpec,
    target_scenario: str,
    compute_backend: str,
    seed: int,
    train_sample_weights: np.ndarray | None = None,
    tuning_sample_weights: np.ndarray | None = None,
    progress: LightGBMProgressCallback | None = None,
) -> TrainedLightGBMHurdleModel:
    """Fit one research-only hurdle member without accessing later roles."""

    validate_microstructure_dataset(dataset)
    validate_adaptive_barrier_targets(dataset, barrier_targets)
    train = _validate_indexes(
        train_endpoints,
        rows=dataset.rows,
        label="training",
        minimum_rows=512,
    )
    requested_tuning = _validate_indexes(
        tuning_endpoints,
        rows=dataset.rows,
        label="tuning",
        minimum_rows=1_024,
    )
    if train[-1] >= requested_tuning[0]:
        raise ValueError("LightGBM hurdle training and tuning roles overlap")
    train_weights = _validate_weights(
        train_sample_weights, rows=len(train), label="training"
    )
    requested_tuning_weights = _validate_weights(
        tuning_sample_weights, rows=len(requested_tuning), label="tuning"
    )
    targets, exits = _target_arrays(
        dataset, barrier_targets, scenario=target_scenario
    )
    if any(
        not np.all(np.isfinite(targets[side][train]))
        or not np.all(np.isfinite(targets[side][requested_tuning]))
        or np.any(exits[side][train] <= dataset.decision_time_ms[train])
        or np.any(
            exits[side][requested_tuning]
            <= dataset.decision_time_ms[requested_tuning]
        )
        for side in _SIDES
    ):
        raise ValueError("LightGBM hurdle roles contain invalid barrier labels")
    maximum_exit = np.maximum(exits["long"], exits["short"])
    if np.any(maximum_exit[train] >= dataset.decision_time_ms[requested_tuning[0]]):
        raise ValueError("LightGBM hurdle training labels overlap the tuning role")

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
        raise ValueError("LightGBM hurdle purged tuning roles are too small")
    if np.any(maximum_exit[early_stop] >= calibration_start_ms):
        raise ValueError("LightGBM hurdle early-stop labels overlap calibration")

    backend_parameters, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        int(seed),
        reproducible=True,
    )
    if backend_kind == "opencl" and (
        spec.gpu_use_dp_required is not True
        or backend_parameters.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("LightGBM hurdle OpenCL FP64 accumulation is required")
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
    features = np.asarray(dataset.features, dtype=np.float32)
    boosters: dict[str, lgb.Booster] = {}
    iterations: dict[str, int] = {}
    calibrations: dict[str, tuple[float, float]] = {}
    support: dict[str, dict[str, dict[str, int]]] = {}
    prevalence: list[float] = []
    step = 0
    total_steps = len(_BOOSTER_NAMES)

    def train_head(
        name: str,
        *,
        train_indexes: np.ndarray,
        train_labels: np.ndarray,
        train_role_weights: np.ndarray,
        early_indexes: np.ndarray,
        early_labels: np.ndarray,
        early_role_weights: np.ndarray,
        objective: str,
        metric: str,
        alpha: float | None = None,
    ) -> None:
        nonlocal step
        if progress is not None:
            progress(name, step + 1, total_steps)
        booster, iteration = _train_booster(
            x_train=features[train_indexes],
            y_train=np.asarray(train_labels, dtype=np.float32),
            train_weights=np.asarray(train_role_weights, dtype=np.float32),
            x_early_stop=features[early_indexes],
            y_early_stop=np.asarray(early_labels, dtype=np.float32),
            early_stop_weights=np.asarray(early_role_weights, dtype=np.float32),
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

    for side in _SIDES:
        train_target = targets[side][train]
        early_target = targets[side][early_stop]
        calibration_target = targets[side][calibration]
        train_labels = (train_target > 0.0).astype(np.float32)
        early_labels = (early_target > 0.0).astype(np.float32)
        calibration_labels = (calibration_target > 0.0).astype(np.float32)
        support[side] = {
            "train": _require_class_support(
                train_labels,
                side=side,
                role="train",
                minimum=_MINIMUM_TRAIN_CLASS_ROWS,
            ),
            "early_stop": _require_class_support(
                early_labels,
                side=side,
                role="early_stop",
                minimum=_MINIMUM_EARLY_STOP_CLASS_ROWS,
            ),
            "calibration": _require_class_support(
                calibration_labels,
                side=side,
                role="calibration",
                minimum=_MINIMUM_CALIBRATION_CLASS_ROWS,
            ),
        }
        prevalence.append(
            float(support[side]["train"]["profitable_rows"] / len(train_labels))
        )
        probability_name = f"{side}_probability"
        train_head(
            probability_name,
            train_indexes=train,
            train_labels=train_labels,
            train_role_weights=train_weights,
            early_indexes=early_stop,
            early_labels=early_labels,
            early_role_weights=early_stop_weights,
            objective="binary",
            metric="binary_logloss",
        )
        raw_calibration = np.asarray(
            boosters[probability_name].predict(
                features[calibration],
                num_iteration=iterations[probability_name],
            ),
            dtype=np.float64,
        )
        calibrations[side] = fit_platt_scaling(
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
            raise ValueError(
                f"LightGBM hurdle {side} conditional magnitude support is insufficient"
            )
        train_head(
            f"{side}_positive_magnitude",
            train_indexes=train[positive_train],
            train_labels=train_target[positive_train],
            train_role_weights=train_weights[positive_train],
            early_indexes=early_stop[positive_early],
            early_labels=early_target[positive_early],
            early_role_weights=early_stop_weights[positive_early],
            objective="regression",
            metric="l2",
        )
        train_head(
            f"{side}_nonpositive_loss_magnitude",
            train_indexes=train[nonpositive_train],
            train_labels=-train_target[nonpositive_train],
            train_role_weights=train_weights[nonpositive_train],
            early_indexes=early_stop[nonpositive_early],
            early_labels=-early_target[nonpositive_early],
            early_role_weights=early_stop_weights[nonpositive_early],
            objective="regression",
            metric="l2",
        )
        train_head(
            f"{side}_lower_quantile",
            train_indexes=train,
            train_labels=train_target,
            train_role_weights=train_weights,
            early_indexes=early_stop,
            early_labels=early_target,
            early_role_weights=early_stop_weights,
            objective="quantile",
            metric="quantile",
            alpha=spec.lower_quantile,
        )
        train_head(
            f"{side}_upper_quantile",
            train_indexes=train,
            train_labels=train_target,
            train_role_weights=train_weights,
            early_indexes=early_stop,
            early_labels=early_target,
            early_role_weights=early_stop_weights,
            objective="quantile",
            metric="quantile",
            alpha=spec.upper_quantile,
        )

    model_strings = {
        name: boosters[name].model_to_string(num_iteration=iterations[name])
        for name in _BOOSTER_NAMES
    }
    provisional = TrainedLightGBMHurdleModel(
        schema_version=LIGHTGBM_HURDLE_SCHEMA_VERSION,
        model_family=_MODEL_FAMILY,
        spec=spec,
        feature_version=dataset.feature_version,
        feature_names=dataset.feature_names,
        target_schema_version=barrier_targets.schema_version,
        target_mode=ADAPTIVE_BARRIER_TARGET_MODE,
        target_spec=barrier_targets.spec,
        target_contract_sha256=_target_contract_sha256(barrier_targets),
        target_scenario=target_scenario,
        backend_requested=str(compute_backend),
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        seed=int(seed),
        training_rows=len(train),
        requested_early_stop_rows=len(requested_tuning),
        early_stop_rows=len(early_stop),
        calibration_rows=len(calibration),
        calibration_start_ms=calibration_start_ms,
        internal_purged_rows=internal_purged_rows,
        positive_class_prevalence=(prevalence[0], prevalence[1]),
        class_support=support,
        probability_calibration=calibrations,
        best_iterations=iterations,
        model_strings=model_strings,
        model_sha256="",
    )
    model = TrainedLightGBMHurdleModel(
        **{
            **provisional.__dict__,
            "model_sha256": _model_sha256(provisional),
        }
    )
    _validate_model_contract(model, reload_boosters=False)
    return model


def predict_lightgbm_hurdle_model(
    model: TrainedLightGBMHurdleModel,
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
) -> ActionValuePredictionBatch:
    """Predict calibrated action values and direct conditional quantiles."""

    validate_microstructure_dataset(dataset)
    _validate_model_contract(model, reload_boosters=False)
    if (
        dataset.feature_version != model.feature_version
        or dataset.feature_names != model.feature_names
    ):
        raise ValueError("LightGBM hurdle prediction feature contract drifted")
    selected = _validate_indexes(
        endpoints,
        rows=dataset.rows,
        label="prediction",
        minimum_rows=1,
    )
    try:
        boosters = {
            name: lgb.Booster(model_str=model.model_strings[name])
            for name in _BOOSTER_NAMES
        }
    except lgb.basic.LightGBMError as exc:
        raise ValueError("LightGBM hurdle booster payload cannot be reloaded") from exc
    features = np.asarray(dataset.features[selected], dtype=np.float32)
    output: dict[str, np.ndarray] = {}
    for side in _SIDES:
        probability_name = f"{side}_probability"
        raw_probability = np.asarray(
            boosters[probability_name].predict(
                features,
                num_iteration=model.best_iterations[probability_name],
            ),
            dtype=np.float64,
        )
        probability = apply_platt_scaling(
            raw_probability,
            model.probability_calibration[side],
        )
        positive_name = f"{side}_positive_magnitude"
        loss_name = f"{side}_nonpositive_loss_magnitude"
        positive = np.maximum(
            0.0,
            np.asarray(
                boosters[positive_name].predict(
                    features,
                    num_iteration=model.best_iterations[positive_name],
                ),
                dtype=np.float64,
            ),
        )
        loss = np.maximum(
            0.0,
            np.asarray(
                boosters[loss_name].predict(
                    features,
                    num_iteration=model.best_iterations[loss_name],
                ),
                dtype=np.float64,
            ),
        )
        lower_name = f"{side}_lower_quantile"
        upper_name = f"{side}_upper_quantile"
        raw_lower = np.asarray(
            boosters[lower_name].predict(
                features,
                num_iteration=model.best_iterations[lower_name],
            ),
            dtype=np.float64,
        )
        raw_upper = np.asarray(
            boosters[upper_name].predict(
                features,
                num_iteration=model.best_iterations[upper_name],
            ),
            dtype=np.float64,
        )
        output[f"{side}_probability"] = probability
        output[f"{side}_mean"] = probability * positive - (1.0 - probability) * loss
        # Monotone rearrangement repairs finite-sample quantile crossing only.
        output[f"{side}_lower"] = np.minimum(raw_lower, raw_upper)
        output[f"{side}_upper"] = np.maximum(raw_lower, raw_upper)
    if any(
        values.shape != (len(selected),) or not np.all(np.isfinite(values))
        for values in output.values()
    ):
        raise ValueError("LightGBM hurdle model emitted invalid predictions")
    return ActionValuePredictionBatch(
        endpoint_indexes=selected,
        long_mean_bps=output["long_mean"],
        short_mean_bps=output["short_mean"],
        long_profitable_probability=output["long_probability"],
        short_profitable_probability=output["short_probability"],
        long_lower_bps=output["long_lower"],
        short_lower_bps=output["short_lower"],
        long_upper_bps=output["long_upper"],
        short_upper_bps=output["short_upper"],
    )


def save_lightgbm_hurdle_model(
    path: str | Path,
    model: TrainedLightGBMHurdleModel,
) -> None:
    """Atomically persist the complete model and its authority-denial contract."""

    _validate_model_contract(model, reload_boosters=True)
    payload = {**_model_payload(model), "model_sha256": model.model_sha256}
    write_json_atomic(path, payload, indent=None, sort_keys=True)


def load_lightgbm_hurdle_model(path: str | Path) -> TrainedLightGBMHurdleModel:
    """Load and independently validate a complete LightGBM hurdle artifact."""

    target = Path(path)
    try:
        size = target.stat().st_size
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("LightGBM hurdle artifact is unreadable") from exc
    if size <= 0 or size > _MAX_ARTIFACT_BYTES or not isinstance(payload, dict):
        raise ValueError("LightGBM hurdle artifact size or structure is invalid")
    expected_keys = {field.name for field in fields(TrainedLightGBMHurdleModel)}
    if set(payload) != expected_keys:
        raise ValueError("LightGBM hurdle artifact contract is incomplete")
    authority_fields = (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    )
    integer_fields = (
        "seed",
        "training_rows",
        "requested_early_stop_rows",
        "early_stop_rows",
        "calibration_rows",
        "calibration_start_ms",
        "internal_purged_rows",
    )
    if any(payload[name] is not False for name in authority_fields) or any(
        not isinstance(payload[name], int) or isinstance(payload[name], bool)
        for name in integer_fields
    ):
        raise ValueError("LightGBM hurdle artifact scalar types are invalid")
    try:
        model = TrainedLightGBMHurdleModel(
            schema_version=str(payload["schema_version"]),
            model_family=str(payload["model_family"]),
            spec=LightGBMHurdleSpec(**payload["spec"]),
            feature_version=str(payload["feature_version"]),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
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
            training_rows=int(payload["training_rows"]),
            requested_early_stop_rows=int(payload["requested_early_stop_rows"]),
            early_stop_rows=int(payload["early_stop_rows"]),
            calibration_rows=int(payload["calibration_rows"]),
            calibration_start_ms=int(payload["calibration_start_ms"]),
            internal_purged_rows=int(payload["internal_purged_rows"]),
            positive_class_prevalence=tuple(
                float(value) for value in payload["positive_class_prevalence"]
            ),
            class_support={
                str(side): {
                    str(role): {
                        str(name): int(value)
                        for name, value in role_support.items()
                    }
                    for role, role_support in side_support.items()
                }
                for side, side_support in payload["class_support"].items()
            },
            probability_calibration={
                str(side): tuple(float(value) for value in calibration)
                for side, calibration in payload["probability_calibration"].items()
            },
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
        raise ValueError("LightGBM hurdle artifact values are invalid") from exc
    _validate_model_contract(model, reload_boosters=True)
    return model


__all__ = [
    "LIGHTGBM_HURDLE_SCHEMA_VERSION",
    "LightGBMHurdleSpec",
    "TrainedLightGBMHurdleModel",
    "load_lightgbm_hurdle_model",
    "predict_lightgbm_hurdle_model",
    "save_lightgbm_hurdle_model",
    "train_lightgbm_hurdle_model",
]
