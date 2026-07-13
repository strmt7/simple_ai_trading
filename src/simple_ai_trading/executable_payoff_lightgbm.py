"""Support-aligned LightGBM models for exact executable barrier payoffs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from .categorical_payoff_lightgbm import (
    CategoricalPayoffDataset,
    build_categorical_payoff_dataset,
)
from .lightgbm_backend import lightgbm_backend_parameters
from .microstructure_barriers import AdaptiveBarrierTargets
from .microstructure_features import (
    MicrostructureDataset,
    validate_microstructure_dataset,
)
from .probability_calibration import apply_platt_scaling, fit_platt_scaling
from .storage import write_json_atomic


EXECUTABLE_PAYOFF_DATASET_SCHEMA_VERSION = "executable-payoff-dataset-v1"
EXECUTABLE_PAYOFF_MODEL_SCHEMA_VERSION = "executable-payoff-lightgbm-v1"
_MODEL_FAMILY = "side_specific_executable_payoff"
_ARCHITECTURES = ("direct_mean", "sign_magnitude_hurdle")
_SIDES = ("long", "short")
_ROLES = ("train", "early_stop", "probability_calibration")
_DAY_MS = 86_400_000
_MINIMUM_ROLE_ROWS = {
    "train": 256,
    "early_stop": 64,
    "probability_calibration": 32,
}
_MINIMUM_CLASS_ROWS = {
    "train": 32,
    "early_stop": 16,
    "probability_calibration": 8,
}
_MINIMUM_CONDITIONAL_ROWS = 32
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
ExecutablePayoffProgressCallback = Callable[[str, str, int, int], None]


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


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExecutablePayoffDataset:
    """Exact payoff rows plus the side-specific predicate used by replay."""

    schema_version: str
    payoff: CategoricalPayoffDataset
    long_executable: np.ndarray
    short_executable: np.ndarray
    dataset_sha256: str

    @property
    def rows(self) -> int:
        return self.payoff.rows

    def __post_init__(self) -> None:
        rows = self.rows
        long_mask = np.asarray(self.long_executable)
        short_mask = np.asarray(self.short_executable)
        expected = _sha256(
            {
                "payoff_dataset_sha256": self.payoff.dataset_sha256,
                "long_executable_sha256": _array_sha256(
                    np.asarray(long_mask, dtype=np.bool_)
                ),
                "short_executable_sha256": _array_sha256(
                    np.asarray(short_mask, dtype=np.bool_)
                ),
            }
        )
        if (
            self.schema_version != EXECUTABLE_PAYOFF_DATASET_SCHEMA_VERSION
            or rows <= 0
            or long_mask.dtype != np.bool_
            or short_mask.dtype != np.bool_
            or long_mask.shape != (rows,)
            or short_mask.shape != (rows,)
            or not np.any(long_mask)
            or not np.any(short_mask)
            or self.dataset_sha256 != expected
        ):
            raise ValueError("executable payoff dataset is invalid")


def build_executable_payoff_dataset(
    dataset: MicrostructureDataset,
    targets: AdaptiveBarrierTargets,
    *,
    target_scenario: str,
    extra_feature_names: Sequence[str] = (),
    extra_features: np.ndarray | None = None,
) -> ExecutablePayoffDataset:
    """Build labels and the exact side support later enforced by replay."""

    validate_microstructure_dataset(dataset)
    payoff = build_categorical_payoff_dataset(
        dataset,
        targets,
        target_scenario=target_scenario,
        extra_feature_names=extra_feature_names,
        extra_features=extra_features,
    )
    source = np.asarray(payoff.source_row_indexes, dtype=np.int64)
    decision_days = np.asarray(payoff.decision_time_ms, dtype=np.int64) // _DAY_MS
    long_mask = np.asarray(dataset.long_liquidity_eligible[source], dtype=np.bool_)
    short_mask = np.asarray(dataset.short_liquidity_eligible[source], dtype=np.bool_)
    long_mask &= (
        np.asarray(payoff.long_exit_time_ms, dtype=np.int64) // _DAY_MS == decision_days
    )
    short_mask &= (
        np.asarray(payoff.short_exit_time_ms, dtype=np.int64) // _DAY_MS
        == decision_days
    )
    identity = _sha256(
        {
            "payoff_dataset_sha256": payoff.dataset_sha256,
            "long_executable_sha256": _array_sha256(long_mask),
            "short_executable_sha256": _array_sha256(short_mask),
        }
    )
    return ExecutablePayoffDataset(
        schema_version=EXECUTABLE_PAYOFF_DATASET_SCHEMA_VERSION,
        payoff=payoff,
        long_executable=long_mask,
        short_executable=short_mask,
        dataset_sha256=identity,
    )


@dataclass(frozen=True)
class ExecutablePayoffSpec:
    candidate_id: str
    family: str
    architecture: str
    learning_rate: float
    num_leaves: int
    max_depth: int
    minimum_leaf_fraction: float
    minimum_leaf_rows: int
    maximum_leaf_rows: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l1: float
    lambda_l2: float
    max_bin: int
    num_boost_round: int
    early_stopping_rounds: int
    gpu_use_dp_required: bool = True

    def __post_init__(self) -> None:
        numeric = (
            self.learning_rate,
            self.minimum_leaf_fraction,
            self.feature_fraction,
            self.bagging_fraction,
            self.lambda_l1,
            self.lambda_l2,
        )
        if (
            not self.candidate_id.strip()
            or self.family != _MODEL_FAMILY
            or self.architecture not in _ARCHITECTURES
            or not all(math.isfinite(float(value)) for value in numeric)
            or not 0.0 < self.learning_rate <= 0.25
            or not 2 <= int(self.num_leaves) <= 255
            or not 1 <= int(self.max_depth) <= 16
            or not 0.0 < self.minimum_leaf_fraction <= 0.05
            or not 16 <= int(self.minimum_leaf_rows) <= int(self.maximum_leaf_rows)
            or int(self.maximum_leaf_rows) > 65_536
            or not 0.0 < self.feature_fraction <= 1.0
            or not 0.0 < self.bagging_fraction <= 1.0
            or not 0 <= int(self.bagging_freq) <= 100
            or self.lambda_l1 < 0.0
            or self.lambda_l2 < 0.0
            or not 31 <= int(self.max_bin) <= 255
            or not 10 <= int(self.num_boost_round) <= 10_000
            or not 5 <= int(self.early_stopping_rounds) < int(self.num_boost_round)
            or self.gpu_use_dp_required is not True
        ):
            raise ValueError("executable payoff specification is invalid")


@dataclass(frozen=True)
class TrainedExecutablePayoffModel:
    schema_version: str
    model_family: str
    spec: ExecutablePayoffSpec
    symbol: str
    feature_names: tuple[str, ...]
    source_dataset_sha256: str
    target_scenario: str
    backend_requested: str
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    seed: int
    role_rows: Mapping[str, Mapping[str, int]]
    rejected_role_rows: Mapping[str, Mapping[str, int]]
    role_mask_sha256: Mapping[str, Mapping[str, str]]
    class_support: Mapping[str, Mapping[str, Mapping[str, int]]]
    minimum_leaf_rows: Mapping[str, int]
    training_target_mean_bps: Mapping[str, float]
    training_profitable_prevalence: Mapping[str, float]
    training_conditional_gain_mean_bps: Mapping[str, float]
    training_conditional_loss_mean_bps: Mapping[str, float]
    probability_calibration: Mapping[str, tuple[float, float]]
    best_iterations: Mapping[str, int]
    model_strings: Mapping[str, str]
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False


@dataclass(frozen=True)
class ExecutablePayoffPredictionBatch:
    architecture: str
    endpoint_indexes: np.ndarray
    long_expected_net_bps: np.ndarray
    short_expected_net_bps: np.ndarray
    long_executable: np.ndarray
    short_executable: np.ndarray
    long_profitable_probability: np.ndarray | None = None
    short_profitable_probability: np.ndarray | None = None
    long_conditional_gain_bps: np.ndarray | None = None
    short_conditional_gain_bps: np.ndarray | None = None
    long_conditional_loss_bps: np.ndarray | None = None
    short_conditional_loss_bps: np.ndarray | None = None
    magnitude_floor_count: int = 0

    @property
    def rows(self) -> int:
        return int(len(self.endpoint_indexes))

    def __post_init__(self) -> None:
        rows = self.rows
        expected = (
            np.asarray(self.long_expected_net_bps),
            np.asarray(self.short_expected_net_bps),
        )
        masks = (
            np.asarray(self.long_executable),
            np.asarray(self.short_executable),
        )
        hurdle_values = (
            self.long_profitable_probability,
            self.short_profitable_probability,
            self.long_conditional_gain_bps,
            self.short_conditional_gain_bps,
            self.long_conditional_loss_bps,
            self.short_conditional_loss_bps,
        )
        if (
            self.architecture not in _ARCHITECTURES
            or rows <= 0
            or np.asarray(self.endpoint_indexes).shape != (rows,)
            or np.any(np.diff(self.endpoint_indexes) <= 0)
            or any(value.shape != (rows,) for value in (*expected, *masks))
            or any(not np.all(np.isfinite(value)) for value in expected)
            or any(value.dtype != np.bool_ for value in masks)
            or isinstance(self.magnitude_floor_count, bool)
            or int(self.magnitude_floor_count) < 0
        ):
            raise ValueError("executable payoff prediction batch is invalid")
        if self.architecture == "direct_mean":
            if (
                any(value is not None for value in hurdle_values)
                or self.magnitude_floor_count
            ):
                raise ValueError("direct payoff prediction contains hurdle fields")
            return
        if any(value is None for value in hurdle_values):
            raise ValueError("hurdle payoff prediction fields are incomplete")
        arrays = tuple(
            np.asarray(value) for value in hurdle_values if value is not None
        )
        if (
            any(value.shape != (rows,) for value in arrays)
            or any(not np.all(np.isfinite(value)) for value in arrays)
            or np.any(arrays[0] < 0.0)
            or np.any(arrays[0] > 1.0)
            or np.any(arrays[1] < 0.0)
            or np.any(arrays[1] > 1.0)
            or any(np.any(value < 0.0) for value in arrays[2:])
        ):
            raise ValueError("hurdle payoff prediction values are invalid")


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
        raise ValueError(f"executable payoff {label} indexes are invalid")
    return values


def _role_support(
    dataset: ExecutablePayoffDataset,
    roles: Mapping[str, np.ndarray],
    side: str,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, str]]:
    mask = np.asarray(
        dataset.long_executable if side == "long" else dataset.short_executable,
        dtype=np.bool_,
    )
    output: dict[str, np.ndarray] = {}
    rejected: dict[str, int] = {}
    digests: dict[str, str] = {}
    for role, values in roles.items():
        selected = np.asarray(values[mask[values]], dtype=np.int64)
        minimum = _MINIMUM_ROLE_ROWS[role]
        if len(selected) < minimum:
            raise ValueError(
                f"executable payoff {side} {role} support is insufficient: "
                f"{len(selected)} < {minimum}"
            )
        output[role] = selected
        rejected[role] = int(len(values) - len(selected))
        digests[role] = _array_sha256(
            np.asarray(dataset.payoff.source_row_indexes[selected], dtype=np.int64)
        )
    return output, rejected, digests


def _class_support(labels: np.ndarray) -> dict[str, int]:
    values = np.asarray(labels, dtype=np.bool_)
    return {
        "profitable_rows": int(np.sum(values)),
        "non_profitable_rows": int(np.sum(~values)),
    }


def _train_booster(
    *,
    features: np.ndarray,
    train_indexes: np.ndarray,
    train_labels: np.ndarray,
    early_indexes: np.ndarray,
    early_labels: np.ndarray,
    parameters: Mapping[str, object],
    objective: str,
    metric: str,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[lgb.Booster, int]:
    training = lgb.Dataset(
        np.asarray(features[train_indexes], dtype=np.float32),
        label=np.asarray(train_labels, dtype=np.float32),
        free_raw_data=False,
    )
    early = lgb.Dataset(
        np.asarray(features[early_indexes], dtype=np.float32),
        label=np.asarray(early_labels, dtype=np.float32),
        reference=training,
        free_raw_data=False,
    )
    booster = lgb.train(
        {**parameters, "objective": objective, "metric": metric},
        training,
        num_boost_round=int(num_boost_round),
        valid_sets=[early],
        valid_names=["early_stop"],
        callbacks=[
            lgb.early_stopping(int(early_stopping_rounds), verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    iteration = max(1, int(booster.best_iteration or booster.current_iteration()))
    return booster, iteration


def _booster_names(architecture: str) -> tuple[str, ...]:
    heads = (
        ("mean",)
        if architecture == "direct_mean"
        else (
            "probability",
            "conditional_gain",
            "conditional_loss",
        )
    )
    return tuple(f"{side}_{head}" for side in _SIDES for head in heads)


def _model_payload(model: TrainedExecutablePayoffModel) -> dict[str, object]:
    payload = asdict(model)
    payload.pop("model_sha256")
    return payload


def _model_sha256(model: TrainedExecutablePayoffModel) -> str:
    return _sha256(_model_payload(model))


def _validate_model(
    model: TrainedExecutablePayoffModel,
    *,
    reload_boosters: bool,
) -> None:
    expected_boosters = set(_booster_names(model.spec.architecture))
    expected_sides = set(_SIDES)
    expected_roles = set(_ROLES)
    hurdle = model.spec.architecture == "sign_magnitude_hurdle"
    if (
        model.schema_version != EXECUTABLE_PAYOFF_MODEL_SCHEMA_VERSION
        or model.model_family != _MODEL_FAMILY
        or model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.portfolio_claim
        or model.leverage_applied
        or not model.symbol
        or not model.feature_names
        or len(set(model.feature_names)) != len(model.feature_names)
        or len(model.source_dataset_sha256) != 64
        or model.target_scenario not in {"base", "stress"}
        or model.backend_kind not in {"opencl", "cpu"}
        or not model.backend_device.strip()
        or model.lightgbm_version != str(lgb.__version__)
        or isinstance(model.seed, bool)
        or set(model.role_rows) != expected_sides
        or set(model.rejected_role_rows) != expected_sides
        or set(model.role_mask_sha256) != expected_sides
        or set(model.class_support) != expected_sides
        or set(model.minimum_leaf_rows) != expected_sides
        or set(model.training_target_mean_bps) != expected_sides
        or set(model.training_profitable_prevalence) != expected_sides
        or set(model.training_conditional_gain_mean_bps) != expected_sides
        or set(model.training_conditional_loss_mean_bps) != expected_sides
        or set(model.probability_calibration) != (expected_sides if hurdle else set())
        or set(model.best_iterations) != expected_boosters
        or set(model.model_strings) != expected_boosters
        or model.model_sha256 != _model_sha256(model)
    ):
        raise ValueError("executable payoff model contract is invalid")
    for side in _SIDES:
        if (
            set(model.role_rows[side]) != expected_roles
            or set(model.rejected_role_rows[side]) != expected_roles
            or set(model.role_mask_sha256[side]) != expected_roles
            or set(model.class_support[side]) != expected_roles
        ):
            raise ValueError("executable payoff role contract is incomplete")
        for role in _ROLES:
            rows = int(model.role_rows[side][role])
            support = model.class_support[side][role]
            if (
                rows < _MINIMUM_ROLE_ROWS[role]
                or int(model.rejected_role_rows[side][role]) < 0
                or len(str(model.role_mask_sha256[side][role])) != 64
                or set(support) != {"profitable_rows", "non_profitable_rows"}
                or sum(int(value) for value in support.values()) != rows
            ):
                raise ValueError("executable payoff role evidence is invalid")
            if (
                hurdle
                and min(int(value) for value in support.values())
                < _MINIMUM_CLASS_ROWS[role]
            ):
                raise ValueError("executable payoff hurdle class support is invalid")
        training_rows = int(model.role_rows[side]["train"])
        expected_leaf = max(
            int(model.spec.minimum_leaf_rows),
            min(
                int(model.spec.maximum_leaf_rows),
                int(math.ceil(model.spec.minimum_leaf_fraction * training_rows)),
            ),
        )
        if int(model.minimum_leaf_rows[side]) != expected_leaf:
            raise ValueError("executable payoff leaf-size contract drifted")
        prevalence = float(model.training_profitable_prevalence[side])
        gain = float(model.training_conditional_gain_mean_bps[side])
        loss = float(model.training_conditional_loss_mean_bps[side])
        mean = float(model.training_target_mean_bps[side])
        if (
            not all(math.isfinite(value) for value in (prevalence, gain, loss, mean))
            or not 0.0 < prevalence < 1.0
            or gain <= 0.0
            or loss < 0.0
        ):
            raise ValueError("executable payoff training baseline is invalid")
        if hurdle:
            calibration = model.probability_calibration[side]
            if (
                len(calibration) != 2
                or not 0.05 <= float(calibration[0]) <= 10.0
                or not -10.0 <= float(calibration[1]) <= 10.0
            ):
                raise ValueError("executable payoff calibration is invalid")
    if any(
        isinstance(value, bool) or not 1 <= int(value) <= model.spec.num_boost_round
        for value in model.best_iterations.values()
    ) or any(not str(value).strip() for value in model.model_strings.values()):
        raise ValueError("executable payoff booster metadata is invalid")
    if reload_boosters:
        try:
            for name in _booster_names(model.spec.architecture):
                booster = lgb.Booster(model_str=model.model_strings[name])
                if booster.num_feature() != len(model.feature_names):
                    raise ValueError("executable payoff booster feature count drifted")
        except lgb.basic.LightGBMError as exc:
            raise ValueError("executable payoff booster cannot be reloaded") from exc


def train_executable_payoff_model(
    dataset: ExecutablePayoffDataset,
    *,
    train_indexes: np.ndarray,
    early_stop_indexes: np.ndarray,
    probability_calibration_indexes: np.ndarray,
    probability_calibration_end_exclusive_ms: int,
    spec: ExecutablePayoffSpec,
    target_scenario: str,
    compute_backend: str,
    seed: int,
    progress: ExecutablePayoffProgressCallback | None = None,
) -> TrainedExecutablePayoffModel:
    """Fit one support-correct, non-authoritative payoff model."""

    payoff = dataset.payoff
    roles = {
        "train": _validate_indexes(
            train_indexes,
            rows=dataset.rows,
            label="training",
            minimum_rows=1_024,
        ),
        "early_stop": _validate_indexes(
            early_stop_indexes,
            rows=dataset.rows,
            label="early-stop",
            minimum_rows=512,
        ),
        "probability_calibration": _validate_indexes(
            probability_calibration_indexes,
            rows=dataset.rows,
            label="probability-calibration",
            minimum_rows=256,
        ),
    }
    if not (
        roles["train"][-1]
        < roles["early_stop"][0]
        < roles["probability_calibration"][0]
    ):
        raise ValueError("executable payoff chronological roles overlap")
    cutoff = int(probability_calibration_end_exclusive_ms)
    if cutoff <= int(payoff.decision_time_ms[roles["probability_calibration"][-1]]):
        raise ValueError("executable payoff calibration cutoff is invalid")
    if target_scenario != payoff.target_scenario:
        raise ValueError("executable payoff target scenario drifted")

    backend_parameters, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        int(seed),
        reproducible=True,
    )
    if backend_kind == "opencl" and (
        spec.gpu_use_dp_required is not True
        or backend_parameters.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("executable payoff OpenCL FP64 accumulation is required")
    common: dict[str, object] = {
        **backend_parameters,
        "learning_rate": float(spec.learning_rate),
        "num_leaves": int(spec.num_leaves),
        "max_depth": int(spec.max_depth),
        "feature_fraction": float(spec.feature_fraction),
        "bagging_fraction": float(spec.bagging_fraction),
        "bagging_freq": int(spec.bagging_freq),
        "lambda_l1": float(spec.lambda_l1),
        "lambda_l2": float(spec.lambda_l2),
        "max_bin": int(spec.max_bin),
    }
    features = np.asarray(payoff.features, dtype=np.float32)
    targets = {
        "long": np.asarray(payoff.long_net_bps, dtype=np.float32),
        "short": np.asarray(payoff.short_net_bps, dtype=np.float32),
    }
    exits = {
        "long": np.asarray(payoff.long_exit_time_ms, dtype=np.int64),
        "short": np.asarray(payoff.short_exit_time_ms, dtype=np.int64),
    }
    role_rows: dict[str, dict[str, int]] = {}
    rejected_rows: dict[str, dict[str, int]] = {}
    role_digests: dict[str, dict[str, str]] = {}
    class_support: dict[str, dict[str, dict[str, int]]] = {}
    leaf_rows: dict[str, int] = {}
    target_means: dict[str, float] = {}
    prevalence: dict[str, float] = {}
    gain_means: dict[str, float] = {}
    loss_means: dict[str, float] = {}
    calibrations: dict[str, tuple[float, float]] = {}
    boosters: dict[str, lgb.Booster] = {}
    iterations: dict[str, int] = {}
    total_steps = len(_booster_names(spec.architecture))
    completed_steps = 0

    def train_head(
        name: str,
        *,
        side: str,
        train_rows: np.ndarray,
        train_labels: np.ndarray,
        early_rows: np.ndarray,
        early_labels: np.ndarray,
        parameters: Mapping[str, object],
        objective: str,
        metric: str,
    ) -> None:
        nonlocal completed_steps
        if progress is not None:
            progress(name, side, completed_steps + 1, total_steps)
        booster, iteration = _train_booster(
            features=features,
            train_indexes=train_rows,
            train_labels=train_labels,
            early_indexes=early_rows,
            early_labels=early_labels,
            parameters=parameters,
            objective=objective,
            metric=metric,
            num_boost_round=spec.num_boost_round,
            early_stopping_rounds=spec.early_stopping_rounds,
        )
        boosters[name] = booster
        iterations[name] = iteration
        completed_steps += 1

    for side in _SIDES:
        side_roles, side_rejected, side_digests = _role_support(
            dataset,
            roles,
            side,
        )
        role_rows[side] = {name: len(values) for name, values in side_roles.items()}
        rejected_rows[side] = side_rejected
        role_digests[side] = side_digests
        side_target = targets[side]
        side_exit = exits[side]
        if (
            np.any(
                side_exit[side_roles["train"]]
                >= payoff.decision_time_ms[roles["early_stop"][0]]
            )
            or np.any(
                side_exit[side_roles["early_stop"]]
                >= payoff.decision_time_ms[roles["probability_calibration"][0]]
            )
            or np.any(side_exit[side_roles["probability_calibration"]] >= cutoff)
        ):
            raise ValueError(f"executable payoff {side} labels cross a role boundary")
        labels_by_role = {
            role: side_target[indexes] > 0.0 for role, indexes in side_roles.items()
        }
        class_support[side] = {
            role: _class_support(labels) for role, labels in labels_by_role.items()
        }
        training_support = class_support[side]["train"]
        training_count = role_rows[side]["train"]
        profitable = labels_by_role["train"]
        training_values = side_target[side_roles["train"]]
        if min(training_support.values()) < _MINIMUM_CLASS_ROWS["train"]:
            raise ValueError(
                f"executable payoff {side} training classes are insufficient"
            )
        target_means[side] = float(np.mean(training_values, dtype=np.float64))
        prevalence[side] = float(training_support["profitable_rows"] / training_count)
        gain_means[side] = float(np.mean(training_values[profitable], dtype=np.float64))
        loss_means[side] = float(
            np.mean(-training_values[~profitable], dtype=np.float64)
        )
        leaf_rows[side] = max(
            int(spec.minimum_leaf_rows),
            min(
                int(spec.maximum_leaf_rows),
                int(math.ceil(spec.minimum_leaf_fraction * training_count)),
            ),
        )
        side_parameters = {**common, "min_data_in_leaf": leaf_rows[side]}
        train_role = side_roles["train"]
        early_role = side_roles["early_stop"]
        if spec.architecture == "direct_mean":
            train_head(
                f"{side}_mean",
                side=side,
                train_rows=train_role,
                train_labels=side_target[train_role],
                early_rows=early_role,
                early_labels=side_target[early_role],
                parameters=side_parameters,
                objective="regression",
                metric="l2",
            )
            continue

        for role, support in class_support[side].items():
            if min(support.values()) < _MINIMUM_CLASS_ROWS[role]:
                raise ValueError(
                    f"executable payoff {side} {role} classes are insufficient"
                )
        train_labels = labels_by_role["train"]
        early_labels = labels_by_role["early_stop"]
        calibration_role = side_roles["probability_calibration"]
        calibration_labels = labels_by_role["probability_calibration"]
        train_head(
            f"{side}_probability",
            side=side,
            train_rows=train_role,
            train_labels=train_labels.astype(np.float32),
            early_rows=early_role,
            early_labels=early_labels.astype(np.float32),
            parameters=side_parameters,
            objective="binary",
            metric="binary_logloss",
        )
        probability_booster = boosters[f"{side}_probability"]
        raw_calibration = np.asarray(
            probability_booster.predict(
                features[calibration_role],
                num_iteration=iterations[f"{side}_probability"],
            ),
            dtype=np.float64,
        )
        calibrations[side] = fit_platt_scaling(
            raw_calibration,
            calibration_labels.astype(np.float64),
        )
        positive_train = train_labels
        positive_early = early_labels
        if (
            min(
                int(np.sum(positive_train)),
                int(np.sum(~positive_train)),
                int(np.sum(positive_early)),
                int(np.sum(~positive_early)),
            )
            < _MINIMUM_CONDITIONAL_ROWS
        ):
            raise ValueError(
                f"executable payoff {side} conditional magnitude support is insufficient"
            )
        train_head(
            f"{side}_conditional_gain",
            side=side,
            train_rows=train_role[positive_train],
            train_labels=side_target[train_role[positive_train]],
            early_rows=early_role[positive_early],
            early_labels=side_target[early_role[positive_early]],
            parameters=side_parameters,
            objective="regression",
            metric="l2",
        )
        train_head(
            f"{side}_conditional_loss",
            side=side,
            train_rows=train_role[~positive_train],
            train_labels=-side_target[train_role[~positive_train]],
            early_rows=early_role[~positive_early],
            early_labels=-side_target[early_role[~positive_early]],
            parameters=side_parameters,
            objective="regression",
            metric="l2",
        )

    model_strings = {
        name: boosters[name].model_to_string(num_iteration=iterations[name])
        for name in _booster_names(spec.architecture)
    }
    provisional = TrainedExecutablePayoffModel(
        schema_version=EXECUTABLE_PAYOFF_MODEL_SCHEMA_VERSION,
        model_family=_MODEL_FAMILY,
        spec=spec,
        symbol=payoff.symbol,
        feature_names=payoff.feature_names,
        source_dataset_sha256=dataset.dataset_sha256,
        target_scenario=target_scenario,
        backend_requested=str(compute_backend),
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        seed=int(seed),
        role_rows=role_rows,
        rejected_role_rows=rejected_rows,
        role_mask_sha256=role_digests,
        class_support=class_support,
        minimum_leaf_rows=leaf_rows,
        training_target_mean_bps=target_means,
        training_profitable_prevalence=prevalence,
        training_conditional_gain_mean_bps=gain_means,
        training_conditional_loss_mean_bps=loss_means,
        probability_calibration=calibrations,
        best_iterations=iterations,
        model_strings=model_strings,
        model_sha256="",
    )
    model = TrainedExecutablePayoffModel(
        **{**provisional.__dict__, "model_sha256": _model_sha256(provisional)}
    )
    _validate_model(model, reload_boosters=False)
    return model


def predict_executable_payoff_model(
    model: TrainedExecutablePayoffModel,
    dataset: ExecutablePayoffDataset,
    indexes: np.ndarray,
) -> ExecutablePayoffPredictionBatch:
    """Score requested rows while preserving their executable support flags."""

    _validate_model(model, reload_boosters=False)
    payoff = dataset.payoff
    if (
        model.symbol != payoff.symbol
        or model.feature_names != payoff.feature_names
        or model.source_dataset_sha256 != dataset.dataset_sha256
        or model.target_scenario != payoff.target_scenario
    ):
        raise ValueError("executable payoff prediction dataset drifted")
    selected = _validate_indexes(
        indexes,
        rows=dataset.rows,
        label="prediction",
        minimum_rows=1,
    )
    features = np.asarray(payoff.features[selected], dtype=np.float32)

    def predict_head(name: str) -> np.ndarray:
        booster = lgb.Booster(model_str=model.model_strings[name])
        values = np.asarray(
            booster.predict(features, num_iteration=model.best_iterations[name]),
            dtype=np.float64,
        )
        if values.shape != (len(selected),) or not np.all(np.isfinite(values)):
            raise ValueError(f"executable payoff {name} predictions are invalid")
        return values

    expected: dict[str, np.ndarray] = {}
    probability: dict[str, np.ndarray] = {}
    gain: dict[str, np.ndarray] = {}
    loss: dict[str, np.ndarray] = {}
    floor_count = 0
    for side in _SIDES:
        if model.spec.architecture == "direct_mean":
            expected[side] = predict_head(f"{side}_mean")
            continue
        raw_probability = predict_head(f"{side}_probability")
        probability[side] = apply_platt_scaling(
            raw_probability,
            model.probability_calibration[side],
        )
        raw_gain = predict_head(f"{side}_conditional_gain")
        raw_loss = predict_head(f"{side}_conditional_loss")
        floor_count += int(np.sum(raw_gain < 0.0) + np.sum(raw_loss < 0.0))
        gain[side] = np.maximum(raw_gain, 0.0)
        loss[side] = np.maximum(raw_loss, 0.0)
        expected[side] = (
            probability[side] * gain[side] - (1.0 - probability[side]) * loss[side]
        )
    hurdle = model.spec.architecture == "sign_magnitude_hurdle"
    return ExecutablePayoffPredictionBatch(
        architecture=model.spec.architecture,
        endpoint_indexes=np.asarray(
            payoff.source_row_indexes[selected], dtype=np.int64
        ),
        long_expected_net_bps=expected["long"],
        short_expected_net_bps=expected["short"],
        long_executable=np.asarray(dataset.long_executable[selected], dtype=np.bool_),
        short_executable=np.asarray(dataset.short_executable[selected], dtype=np.bool_),
        long_profitable_probability=probability.get("long") if hurdle else None,
        short_profitable_probability=probability.get("short") if hurdle else None,
        long_conditional_gain_bps=gain.get("long") if hurdle else None,
        short_conditional_gain_bps=gain.get("short") if hurdle else None,
        long_conditional_loss_bps=loss.get("long") if hurdle else None,
        short_conditional_loss_bps=loss.get("short") if hurdle else None,
        magnitude_floor_count=floor_count,
    )


def save_executable_payoff_model(
    path: str | Path,
    model: TrainedExecutablePayoffModel,
) -> None:
    _validate_model(model, reload_boosters=True)
    payload = asdict(model)
    encoded = _canonical_json(payload).encode("utf-8")
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("executable payoff model artifact is too large")
    write_json_atomic(Path(path), payload, indent=None, sort_keys=True)


def load_executable_payoff_model(
    path: str | Path,
) -> TrainedExecutablePayoffModel:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("executable payoff model artifact is missing or oversized")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("executable payoff model artifact is unreadable") from exc
    expected = {field.name for field in fields(TrainedExecutablePayoffModel)}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("executable payoff model artifact fields drifted")
    if not isinstance(payload.get("spec"), dict):
        raise ValueError("executable payoff model specification is missing")
    payload["spec"] = ExecutablePayoffSpec(**payload["spec"])
    payload["feature_names"] = tuple(payload["feature_names"])
    payload["probability_calibration"] = {
        side: tuple(values)
        for side, values in payload["probability_calibration"].items()
    }
    model = TrainedExecutablePayoffModel(**payload)
    _validate_model(model, reload_boosters=True)
    return model


__all__ = [
    "EXECUTABLE_PAYOFF_DATASET_SCHEMA_VERSION",
    "EXECUTABLE_PAYOFF_MODEL_SCHEMA_VERSION",
    "ExecutablePayoffDataset",
    "ExecutablePayoffPredictionBatch",
    "ExecutablePayoffSpec",
    "TrainedExecutablePayoffModel",
    "build_executable_payoff_dataset",
    "load_executable_payoff_model",
    "predict_executable_payoff_model",
    "save_executable_payoff_model",
    "train_executable_payoff_model",
]
