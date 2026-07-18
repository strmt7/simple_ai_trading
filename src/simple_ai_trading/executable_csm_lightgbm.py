"""Executable conditioning-sign-on-magnitude payoff distributions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize_scalar

from .executable_payoff_lightgbm import ExecutablePayoffDataset
from .lightgbm_backend import (
    SUPPORTED_LIGHTGBM_BACKEND_KINDS,
    lightgbm_backend_parameters,
)
from .probability_calibration import apply_platt_scaling, fit_platt_scaling
from .storage import write_json_atomic


EXECUTABLE_CSM_MODEL_SCHEMA_VERSION = "executable-csm-lightgbm-v1"
_MODEL_FAMILY = "side_specific_executable_csm"
_SIDES = ("long", "short")
_ROLES = ("train", "early_stop", "probability_calibration")
_MINIMUM_ROLE_ROWS = {
    "train": 256,
    "early_stop": 64,
    "probability_calibration": 32,
}
_MINIMUM_MAGNITUDE_CLASS_ROWS = {
    "train": 32,
    "early_stop": 8,
    "probability_calibration": 4,
}
_MINIMUM_SIGN_CLASS_ROWS = {
    "train": 32,
    "early_stop": 16,
    "probability_calibration": 8,
}
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
ExecutableCsmProgressCallback = Callable[[str, str, int, int], None]


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
class ExecutableCsmSpec:
    candidate_id: str
    family: str
    magnitude_edge_quantiles: tuple[float, ...]
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
    temperature_minimum: float = 0.5
    temperature_maximum: float = 3.0
    gpu_use_dp_required: bool = True

    @property
    def magnitude_classes(self) -> int:
        return len(self.magnitude_edge_quantiles) + 1

    def __post_init__(self) -> None:
        quantiles = tuple(float(value) for value in self.magnitude_edge_quantiles)
        numeric = (
            self.learning_rate,
            self.minimum_leaf_fraction,
            self.feature_fraction,
            self.bagging_fraction,
            self.lambda_l1,
            self.lambda_l2,
            self.temperature_minimum,
            self.temperature_maximum,
        )
        if (
            not self.candidate_id.strip()
            or self.family != _MODEL_FAMILY
            or len(quantiles) < 3
            or quantiles != tuple(sorted(set(quantiles)))
            or not all(0.0 < value < 1.0 for value in quantiles)
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
            or not 0.0 < self.temperature_minimum <= 1.0
            or not 1.0 <= self.temperature_maximum <= 10.0
            or self.temperature_minimum >= self.temperature_maximum
            or self.gpu_use_dp_required is not True
        ):
            raise ValueError("executable CSM specification is invalid")


@dataclass(frozen=True)
class TrainedExecutableCsmModel:
    schema_version: str
    model_family: str
    spec: ExecutableCsmSpec
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
    magnitude_class_support: Mapping[str, Mapping[str, tuple[int, ...]]]
    sign_class_support: Mapping[str, Mapping[str, Mapping[str, int]]]
    minimum_leaf_rows: Mapping[str, int]
    training_target_mean_bps: Mapping[str, float]
    training_profitable_prevalence: Mapping[str, float]
    magnitude_edges_risk_units: Mapping[str, tuple[float, ...]]
    magnitude_representatives_risk_units: Mapping[str, tuple[float, ...]]
    training_joint_probabilities: Mapping[str, tuple[float, ...]]
    magnitude_temperature: Mapping[str, float]
    magnitude_calibration_log_loss_before: Mapping[str, float]
    magnitude_calibration_log_loss_after: Mapping[str, float]
    sign_probability_calibration: Mapping[str, tuple[float, float]]
    sign_calibration_log_loss_before: Mapping[str, float]
    sign_calibration_log_loss_after: Mapping[str, float]
    best_iterations: Mapping[str, int]
    model_strings: Mapping[str, str]
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False


@dataclass(frozen=True)
class ExecutableCsmPredictionBatch:
    endpoint_indexes: np.ndarray
    long_expected_net_bps: np.ndarray
    short_expected_net_bps: np.ndarray
    long_executable: np.ndarray
    short_executable: np.ndarray
    long_profitable_probability: np.ndarray
    short_profitable_probability: np.ndarray
    long_q10_net_bps: np.ndarray
    short_q10_net_bps: np.ndarray
    long_q90_net_bps: np.ndarray
    short_q90_net_bps: np.ndarray
    long_cvar10_net_bps: np.ndarray
    short_cvar10_net_bps: np.ndarray
    long_magnitude_probabilities: np.ndarray
    short_magnitude_probabilities: np.ndarray
    long_positive_probability_by_magnitude: np.ndarray
    short_positive_probability_by_magnitude: np.ndarray

    @property
    def rows(self) -> int:
        return int(len(self.endpoint_indexes))

    @property
    def magnitude_classes(self) -> int:
        return int(np.asarray(self.long_magnitude_probabilities).shape[1])

    def __post_init__(self) -> None:
        rows = self.rows
        vectors = (
            self.long_expected_net_bps,
            self.short_expected_net_bps,
            self.long_profitable_probability,
            self.short_profitable_probability,
            self.long_q10_net_bps,
            self.short_q10_net_bps,
            self.long_q90_net_bps,
            self.short_q90_net_bps,
            self.long_cvar10_net_bps,
            self.short_cvar10_net_bps,
        )
        masks = (self.long_executable, self.short_executable)
        distributions = (
            np.asarray(self.long_magnitude_probabilities, dtype=np.float64),
            np.asarray(self.short_magnitude_probabilities, dtype=np.float64),
        )
        conditional = (
            np.asarray(
                self.long_positive_probability_by_magnitude, dtype=np.float64
            ),
            np.asarray(
                self.short_positive_probability_by_magnitude, dtype=np.float64
            ),
        )
        classes = self.magnitude_classes
        if (
            rows <= 0
            or np.asarray(self.endpoint_indexes).shape != (rows,)
            or np.any(np.diff(self.endpoint_indexes) <= 0)
            or any(np.asarray(value).shape != (rows,) for value in vectors)
            or any(not np.all(np.isfinite(value)) for value in vectors)
            or any(np.asarray(value).shape != (rows,) for value in masks)
            or any(np.asarray(value).dtype != np.bool_ for value in masks)
            or classes < 4
            or any(value.shape != (rows, classes) for value in distributions)
            or any(value.shape != (rows, classes) for value in conditional)
            or any(not np.all(np.isfinite(value)) for value in distributions)
            or any(not np.all(np.isfinite(value)) for value in conditional)
            or any(np.any(value < 0.0) for value in distributions)
            or any(np.any(value < 0.0) or np.any(value > 1.0) for value in conditional)
            or any(
                not np.allclose(value.sum(axis=1), 1.0, atol=1e-8)
                for value in distributions
            )
            or np.any(np.asarray(self.long_profitable_probability) < 0.0)
            or np.any(np.asarray(self.long_profitable_probability) > 1.0)
            or np.any(np.asarray(self.short_profitable_probability) < 0.0)
            or np.any(np.asarray(self.short_profitable_probability) > 1.0)
        ):
            raise ValueError("executable CSM prediction batch is invalid")


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
        raise ValueError(f"executable CSM {label} indexes are invalid")
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
    selected: dict[str, np.ndarray] = {}
    rejected: dict[str, int] = {}
    digests: dict[str, str] = {}
    for role, indexes in roles.items():
        supported = np.asarray(indexes[mask[indexes]], dtype=np.int64)
        if len(supported) < _MINIMUM_ROLE_ROWS[role]:
            raise ValueError(
                f"executable CSM {side} {role} support is insufficient: "
                f"{len(supported)} < {_MINIMUM_ROLE_ROWS[role]}"
            )
        selected[role] = supported
        rejected[role] = int(len(indexes) - len(supported))
        digests[role] = _array_sha256(
            np.asarray(dataset.payoff.source_row_indexes[supported], dtype=np.int64)
        )
    return selected, rejected, digests


def _magnitude_bins(
    values: np.ndarray,
    quantiles: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    magnitude = np.asarray(values, dtype=np.float64)
    edges = np.unique(np.quantile(magnitude, quantiles, method="linear"))
    if len(edges) != len(tuple(quantiles)) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("executable CSM magnitude edges are degenerate")
    labels = np.searchsorted(edges, magnitude, side="right").astype(np.int32)
    classes = len(edges) + 1
    representatives = np.asarray(
        [float(np.mean(magnitude[labels == index])) for index in range(classes)],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(representatives))
        or np.any(representatives <= 0.0)
        or not np.all(np.diff(representatives) > 0.0)
    ):
        raise ValueError("executable CSM magnitude representatives are invalid")
    return edges, labels, representatives


def _labels_from_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(
        np.asarray(edges, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        side="right",
    ).astype(np.int32)


def _class_counts(labels: np.ndarray, classes: int) -> tuple[int, ...]:
    return tuple(
        int(value) for value in np.bincount(labels, minlength=int(classes))
    )


def _sign_counts(labels: np.ndarray) -> dict[str, int]:
    positive = np.asarray(labels, dtype=np.bool_)
    return {
        "profitable_rows": int(np.sum(positive)),
        "non_profitable_rows": int(np.sum(~positive)),
    }


def _multiclass_log_loss(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if (
        values.ndim != 2
        or target.shape != (len(values),)
        or len(values) == 0
        or np.any(target < 0)
        or np.any(target >= values.shape[1])
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("executable CSM multiclass score input is invalid")
    chosen = np.clip(values[np.arange(len(values)), target], 1e-15, 1.0)
    return float(-np.mean(np.log(chosen)))


def _binary_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    target = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(target * np.log(values) + (1.0 - target) * np.log1p(-values)))


def _temperature_probabilities(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    logits = np.log(np.clip(values, 1e-15, 1.0)) / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    output = np.exp(logits)
    output /= np.sum(output, axis=1, keepdims=True)
    return output


def _fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> tuple[float, float, float]:
    raw = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    before = _multiclass_log_loss(raw, target)

    def objective(log_temperature: float) -> float:
        calibrated = _temperature_probabilities(raw, math.exp(log_temperature))
        return _multiclass_log_loss(calibrated, target)

    result = minimize_scalar(
        objective,
        bounds=(math.log(float(minimum)), math.log(float(maximum))),
        method="bounded",
        options={"xatol": 1e-8},
    )
    candidate = float(math.exp(float(result.x))) if result.success else 1.0
    candidate = float(np.clip(candidate, minimum, maximum))
    after = objective(math.log(candidate))
    if after > before:
        return 1.0, before, before
    return candidate, before, after


def _train_booster(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_early: np.ndarray,
    y_early: np.ndarray,
    parameters: Mapping[str, object],
    objective: str,
    metric: str,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[lgb.Booster, int]:
    training = lgb.Dataset(
        np.asarray(x_train, dtype=np.float32),
        label=np.asarray(y_train),
        free_raw_data=False,
    )
    early = lgb.Dataset(
        np.asarray(x_early, dtype=np.float32),
        label=np.asarray(y_early),
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


def _booster_names() -> tuple[str, ...]:
    return tuple(
        f"{side}_{head}"
        for side in _SIDES
        for head in ("magnitude", "conditional_sign")
    )


def _model_payload(model: TrainedExecutableCsmModel) -> dict[str, object]:
    payload = asdict(model)
    payload.pop("model_sha256")
    return payload


def _model_sha256(model: TrainedExecutableCsmModel) -> str:
    return _sha256(_model_payload(model))


def _validate_model(
    model: TrainedExecutableCsmModel,
    *,
    reload_boosters: bool,
) -> None:
    sides = set(_SIDES)
    roles = set(_ROLES)
    boosters = set(_booster_names())
    classes = model.spec.magnitude_classes
    side_maps = (
        model.role_rows,
        model.rejected_role_rows,
        model.role_mask_sha256,
        model.magnitude_class_support,
        model.sign_class_support,
        model.minimum_leaf_rows,
        model.training_target_mean_bps,
        model.training_profitable_prevalence,
        model.magnitude_edges_risk_units,
        model.magnitude_representatives_risk_units,
        model.training_joint_probabilities,
        model.magnitude_temperature,
        model.magnitude_calibration_log_loss_before,
        model.magnitude_calibration_log_loss_after,
        model.sign_probability_calibration,
        model.sign_calibration_log_loss_before,
        model.sign_calibration_log_loss_after,
    )
    if (
        model.schema_version != EXECUTABLE_CSM_MODEL_SCHEMA_VERSION
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
        or model.backend_kind not in SUPPORTED_LIGHTGBM_BACKEND_KINDS
        or not model.backend_device.strip()
        or model.lightgbm_version != str(lgb.__version__)
        or isinstance(model.seed, bool)
        or any(set(value) != sides for value in side_maps)
        or set(model.best_iterations) != boosters
        or set(model.model_strings) != boosters
        or model.model_sha256 != _model_sha256(model)
    ):
        raise ValueError("executable CSM model contract is invalid")
    for side in _SIDES:
        if (
            set(model.role_rows[side]) != roles
            or set(model.rejected_role_rows[side]) != roles
            or set(model.role_mask_sha256[side]) != roles
            or set(model.magnitude_class_support[side]) != roles
            or set(model.sign_class_support[side]) != roles
        ):
            raise ValueError("executable CSM role contract is incomplete")
        for role in _ROLES:
            rows = int(model.role_rows[side][role])
            magnitude_support = tuple(model.magnitude_class_support[side][role])
            sign_support = model.sign_class_support[side][role]
            if (
                rows < _MINIMUM_ROLE_ROWS[role]
                or int(model.rejected_role_rows[side][role]) < 0
                or len(str(model.role_mask_sha256[side][role])) != 64
                or len(magnitude_support) != classes
                or sum(int(value) for value in magnitude_support) != rows
                or min(int(value) for value in magnitude_support)
                < _MINIMUM_MAGNITUDE_CLASS_ROWS[role]
                or set(sign_support) != {"profitable_rows", "non_profitable_rows"}
                or sum(int(value) for value in sign_support.values()) != rows
                or min(int(value) for value in sign_support.values())
                < _MINIMUM_SIGN_CLASS_ROWS[role]
            ):
                raise ValueError("executable CSM role support is invalid")
        expected_leaf = max(
            int(model.spec.minimum_leaf_rows),
            min(
                int(model.spec.maximum_leaf_rows),
                int(
                    math.ceil(
                        model.spec.minimum_leaf_fraction
                        * int(model.role_rows[side]["train"])
                    )
                ),
            ),
        )
        edges = np.asarray(model.magnitude_edges_risk_units[side], dtype=np.float64)
        representatives = np.asarray(
            model.magnitude_representatives_risk_units[side], dtype=np.float64
        )
        joint = np.asarray(model.training_joint_probabilities[side], dtype=np.float64)
        prevalence = float(model.training_profitable_prevalence[side])
        calibration = tuple(model.sign_probability_calibration[side])
        scores = (
            float(model.training_target_mean_bps[side]),
            float(model.magnitude_temperature[side]),
            float(model.magnitude_calibration_log_loss_before[side]),
            float(model.magnitude_calibration_log_loss_after[side]),
            float(model.sign_calibration_log_loss_before[side]),
            float(model.sign_calibration_log_loss_after[side]),
        )
        if (
            int(model.minimum_leaf_rows[side]) != expected_leaf
            or edges.shape != (classes - 1,)
            or representatives.shape != (classes,)
            or not np.all(np.isfinite(edges))
            or not np.all(np.diff(edges) > 0.0)
            or not np.all(np.isfinite(representatives))
            or np.any(representatives <= 0.0)
            or not np.all(np.diff(representatives) > 0.0)
            or joint.shape != (2 * classes,)
            or not np.all(np.isfinite(joint))
            or np.any(joint < 0.0)
            or not np.isclose(np.sum(joint), 1.0, atol=1e-10)
            or not 0.0 < prevalence < 1.0
            or len(calibration) != 2
            or not 0.05 <= float(calibration[0]) <= 10.0
            or not -10.0 <= float(calibration[1]) <= 10.0
            or not all(math.isfinite(value) for value in scores)
            or not model.spec.temperature_minimum
            <= float(model.magnitude_temperature[side])
            <= model.spec.temperature_maximum
            or float(model.magnitude_calibration_log_loss_after[side])
            > float(model.magnitude_calibration_log_loss_before[side]) + 1e-12
            or float(model.sign_calibration_log_loss_after[side])
            > float(model.sign_calibration_log_loss_before[side]) + 1e-12
        ):
            raise ValueError("executable CSM side metadata is invalid")
    if any(
        isinstance(value, bool) or not 1 <= int(value) <= model.spec.num_boost_round
        for value in model.best_iterations.values()
    ) or any(not str(value).strip() for value in model.model_strings.values()):
        raise ValueError("executable CSM booster metadata is invalid")
    if reload_boosters:
        try:
            for name in _booster_names():
                booster = lgb.Booster(model_str=model.model_strings[name])
                expected = len(model.feature_names) + (
                    1 if name.endswith("conditional_sign") else 0
                )
                if booster.num_feature() != expected:
                    raise ValueError("executable CSM booster feature count drifted")
        except lgb.basic.LightGBMError as exc:
            raise ValueError("executable CSM booster cannot be reloaded") from exc


def train_executable_csm_model(
    dataset: ExecutablePayoffDataset,
    *,
    train_indexes: np.ndarray,
    early_stop_indexes: np.ndarray,
    probability_calibration_indexes: np.ndarray,
    probability_calibration_end_exclusive_ms: int,
    spec: ExecutableCsmSpec,
    target_scenario: str,
    compute_backend: str,
    seed: int,
    progress: ExecutableCsmProgressCallback | None = None,
) -> TrainedExecutableCsmModel:
    """Fit a support-aligned discrete CSM joint payoff model."""

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
        raise ValueError("executable CSM chronological roles overlap")
    payoff = dataset.payoff
    cutoff = int(probability_calibration_end_exclusive_ms)
    if cutoff <= int(payoff.decision_time_ms[roles["probability_calibration"][-1]]):
        raise ValueError("executable CSM calibration cutoff is invalid")
    if target_scenario != payoff.target_scenario:
        raise ValueError("executable CSM target scenario drifted")

    backend, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        int(seed),
        reproducible=True,
    )
    if backend_kind == "opencl" and (
        spec.gpu_use_dp_required is not True or backend.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("executable CSM OpenCL FP64 accumulation is required")
    features = np.asarray(payoff.features, dtype=np.float32)
    stop_width = np.asarray(payoff.stop_width_bps, dtype=np.float64)
    targets = {
        "long": np.asarray(payoff.long_net_bps, dtype=np.float64),
        "short": np.asarray(payoff.short_net_bps, dtype=np.float64),
    }
    exits = {
        "long": np.asarray(payoff.long_exit_time_ms, dtype=np.int64),
        "short": np.asarray(payoff.short_exit_time_ms, dtype=np.int64),
    }
    role_rows: dict[str, dict[str, int]] = {}
    rejected_rows: dict[str, dict[str, int]] = {}
    role_digests: dict[str, dict[str, str]] = {}
    magnitude_support: dict[str, dict[str, tuple[int, ...]]] = {}
    sign_support: dict[str, dict[str, dict[str, int]]] = {}
    minimum_leaf_rows: dict[str, int] = {}
    target_means: dict[str, float] = {}
    prevalence: dict[str, float] = {}
    magnitude_edges: dict[str, tuple[float, ...]] = {}
    magnitude_representatives: dict[str, tuple[float, ...]] = {}
    joint_probabilities: dict[str, tuple[float, ...]] = {}
    magnitude_temperature: dict[str, float] = {}
    magnitude_loss_before: dict[str, float] = {}
    magnitude_loss_after: dict[str, float] = {}
    sign_calibration: dict[str, tuple[float, float]] = {}
    sign_loss_before: dict[str, float] = {}
    sign_loss_after: dict[str, float] = {}
    boosters: dict[str, lgb.Booster] = {}
    iterations: dict[str, int] = {}
    completed = 0
    total = len(_booster_names())

    for side in _SIDES:
        side_roles, side_rejected, side_digests = _role_support(
            dataset, roles, side
        )
        role_rows[side] = {name: len(indexes) for name, indexes in side_roles.items()}
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
            raise ValueError(f"executable CSM {side} labels cross a role boundary")
        risk = np.abs(side_target) / stop_width
        train_role = side_roles["train"]
        early_role = side_roles["early_stop"]
        calibration_role = side_roles["probability_calibration"]
        edges, train_magnitude_labels, representatives = _magnitude_bins(
            risk[train_role], spec.magnitude_edge_quantiles
        )
        magnitude_labels = {
            "train": train_magnitude_labels,
            "early_stop": _labels_from_edges(risk[early_role], edges),
            "probability_calibration": _labels_from_edges(
                risk[calibration_role], edges
            ),
        }
        sign_labels = {
            role: side_target[indexes] > 0.0
            for role, indexes in side_roles.items()
        }
        magnitude_support[side] = {
            role: _class_counts(labels, spec.magnitude_classes)
            for role, labels in magnitude_labels.items()
        }
        sign_support[side] = {
            role: _sign_counts(labels) for role, labels in sign_labels.items()
        }
        for role in _ROLES:
            if (
                min(magnitude_support[side][role])
                < _MINIMUM_MAGNITUDE_CLASS_ROWS[role]
                or min(sign_support[side][role].values())
                < _MINIMUM_SIGN_CLASS_ROWS[role]
            ):
                raise ValueError(f"executable CSM {side} {role} classes are insufficient")
        minimum_leaf_rows[side] = max(
            int(spec.minimum_leaf_rows),
            min(
                int(spec.maximum_leaf_rows),
                int(math.ceil(spec.minimum_leaf_fraction * len(train_role))),
            ),
        )
        common: dict[str, object] = {
            **backend,
            "learning_rate": float(spec.learning_rate),
            "num_leaves": int(spec.num_leaves),
            "max_depth": int(spec.max_depth),
            "min_data_in_leaf": minimum_leaf_rows[side],
            "feature_fraction": float(spec.feature_fraction),
            "bagging_fraction": float(spec.bagging_fraction),
            "bagging_freq": int(spec.bagging_freq),
            "lambda_l1": float(spec.lambda_l1),
            "lambda_l2": float(spec.lambda_l2),
            "max_bin": int(spec.max_bin),
        }

        name = f"{side}_magnitude"
        if progress is not None:
            progress("magnitude", side, completed + 1, total)
        magnitude_booster, magnitude_iteration = _train_booster(
            x_train=features[train_role],
            y_train=magnitude_labels["train"],
            x_early=features[early_role],
            y_early=magnitude_labels["early_stop"],
            parameters={**common, "num_class": spec.magnitude_classes},
            objective="multiclass",
            metric="multi_logloss",
            num_boost_round=spec.num_boost_round,
            early_stopping_rounds=spec.early_stopping_rounds,
        )
        boosters[name] = magnitude_booster
        iterations[name] = magnitude_iteration
        completed += 1

        name = f"{side}_conditional_sign"
        if progress is not None:
            progress("conditional_sign", side, completed + 1, total)
        sign_train = np.column_stack(
            (features[train_role], np.log1p(risk[train_role]))
        ).astype(np.float32)
        sign_early = np.column_stack(
            (features[early_role], np.log1p(risk[early_role]))
        ).astype(np.float32)
        sign_booster, sign_iteration = _train_booster(
            x_train=sign_train,
            y_train=sign_labels["train"].astype(np.float32),
            x_early=sign_early,
            y_early=sign_labels["early_stop"].astype(np.float32),
            parameters=common,
            objective="binary",
            metric="binary_logloss",
            num_boost_round=spec.num_boost_round,
            early_stopping_rounds=spec.early_stopping_rounds,
        )
        boosters[name] = sign_booster
        iterations[name] = sign_iteration
        completed += 1

        raw_magnitude = np.asarray(
            magnitude_booster.predict(
                features[calibration_role], num_iteration=magnitude_iteration
            ),
            dtype=np.float64,
        )
        temperature, before, after = _fit_temperature(
            raw_magnitude,
            magnitude_labels["probability_calibration"],
            minimum=spec.temperature_minimum,
            maximum=spec.temperature_maximum,
        )
        calibration_sign_features = np.column_stack(
            (features[calibration_role], np.log1p(risk[calibration_role]))
        ).astype(np.float32)
        raw_sign = np.asarray(
            sign_booster.predict(
                calibration_sign_features, num_iteration=sign_iteration
            ),
            dtype=np.float64,
        )
        platt = fit_platt_scaling(
            raw_sign,
            sign_labels["probability_calibration"].astype(np.float64),
        )
        calibrated_sign = apply_platt_scaling(raw_sign, platt)
        identity_sign = apply_platt_scaling(raw_sign, (1.0, 0.0))
        raw_sign_loss = _binary_log_loss(
            identity_sign, sign_labels["probability_calibration"]
        )
        calibrated_sign_loss = _binary_log_loss(
            calibrated_sign, sign_labels["probability_calibration"]
        )
        if calibrated_sign_loss > raw_sign_loss:
            platt = (1.0, 0.0)
            calibrated_sign_loss = raw_sign_loss
        magnitude_edges[side] = tuple(float(value) for value in edges)
        magnitude_representatives[side] = tuple(
            float(value) for value in representatives
        )
        magnitude_temperature[side] = temperature
        magnitude_loss_before[side] = before
        magnitude_loss_after[side] = after
        sign_calibration[side] = platt
        sign_loss_before[side] = raw_sign_loss
        sign_loss_after[side] = calibrated_sign_loss
        target_means[side] = float(np.mean(side_target[train_role]))
        prevalence[side] = float(np.mean(sign_labels["train"]))
        positive_joint = np.bincount(
            train_magnitude_labels[sign_labels["train"]],
            minlength=spec.magnitude_classes,
        ).astype(np.float64)
        negative_joint = np.bincount(
            train_magnitude_labels[~sign_labels["train"]],
            minlength=spec.magnitude_classes,
        ).astype(np.float64)
        ordered_joint = np.concatenate((negative_joint[::-1], positive_joint))
        joint_probabilities[side] = tuple(
            float(value / len(train_role)) for value in ordered_joint
        )

    model_strings = {
        name: boosters[name].model_to_string(num_iteration=iterations[name])
        for name in _booster_names()
    }
    provisional = TrainedExecutableCsmModel(
        schema_version=EXECUTABLE_CSM_MODEL_SCHEMA_VERSION,
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
        magnitude_class_support=magnitude_support,
        sign_class_support=sign_support,
        minimum_leaf_rows=minimum_leaf_rows,
        training_target_mean_bps=target_means,
        training_profitable_prevalence=prevalence,
        magnitude_edges_risk_units=magnitude_edges,
        magnitude_representatives_risk_units=magnitude_representatives,
        training_joint_probabilities=joint_probabilities,
        magnitude_temperature=magnitude_temperature,
        magnitude_calibration_log_loss_before=magnitude_loss_before,
        magnitude_calibration_log_loss_after=magnitude_loss_after,
        sign_probability_calibration=sign_calibration,
        sign_calibration_log_loss_before=sign_loss_before,
        sign_calibration_log_loss_after=sign_loss_after,
        best_iterations=iterations,
        model_strings=model_strings,
        model_sha256="",
    )
    model = TrainedExecutableCsmModel(
        **{**provisional.__dict__, "model_sha256": _model_sha256(provisional)}
    )
    _validate_model(model, reload_boosters=False)
    return model


def _joint_tail_statistics(
    magnitude_probability: np.ndarray,
    positive_probability: np.ndarray,
    representatives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive_mass = magnitude_probability * positive_probability
    negative_mass = magnitude_probability * (1.0 - positive_probability)
    probabilities = np.concatenate((negative_mass[:, ::-1], positive_mass), axis=1)
    support = np.concatenate((-representatives[::-1], representatives))
    cumulative = np.cumsum(probabilities, axis=1)
    q10_index = np.argmax(cumulative >= 0.10, axis=1)
    q90_index = np.argmax(cumulative >= 0.90, axis=1)
    remaining = np.full(len(probabilities), 0.10, dtype=np.float64)
    total = np.zeros(len(probabilities), dtype=np.float64)
    for index, outcome in enumerate(support):
        consumed = np.minimum(remaining, probabilities[:, index])
        total += consumed * float(outcome)
        remaining -= consumed
    if np.any(remaining > 1e-9):
        raise ValueError("executable CSM lower-tail mass is incomplete")
    return support[q10_index], support[q90_index], total / 0.10


def predict_executable_csm_model(
    model: TrainedExecutableCsmModel,
    dataset: ExecutablePayoffDataset,
    indexes: np.ndarray,
) -> ExecutableCsmPredictionBatch:
    """Integrate the fitted sign-magnitude distribution on requested rows."""

    _validate_model(model, reload_boosters=False)
    payoff = dataset.payoff
    if (
        model.symbol != payoff.symbol
        or model.feature_names != payoff.feature_names
        or model.source_dataset_sha256 != dataset.dataset_sha256
        or model.target_scenario != payoff.target_scenario
    ):
        raise ValueError("executable CSM prediction dataset drifted")
    selected = _validate_indexes(
        indexes,
        rows=dataset.rows,
        label="prediction",
        minimum_rows=1,
    )
    features = np.asarray(payoff.features[selected], dtype=np.float32)
    stop_width = np.asarray(payoff.stop_width_bps[selected], dtype=np.float64)
    output: dict[str, np.ndarray] = {}
    for side in _SIDES:
        magnitude_booster = lgb.Booster(
            model_str=model.model_strings[f"{side}_magnitude"]
        )
        raw_magnitude = np.asarray(
            magnitude_booster.predict(
                features,
                num_iteration=model.best_iterations[f"{side}_magnitude"],
            ),
            dtype=np.float64,
        )
        magnitude_probability = _temperature_probabilities(
            raw_magnitude, model.magnitude_temperature[side]
        )
        representatives = np.asarray(
            model.magnitude_representatives_risk_units[side], dtype=np.float64
        )
        sign_booster = lgb.Booster(
            model_str=model.model_strings[f"{side}_conditional_sign"]
        )
        positive_by_magnitude = np.empty_like(magnitude_probability)
        for class_index, representative in enumerate(representatives):
            magnitude_feature = np.full(
                (len(features), 1), math.log1p(float(representative)), dtype=np.float32
            )
            sign_features = np.column_stack((features, magnitude_feature))
            raw_sign = np.asarray(
                sign_booster.predict(
                    sign_features,
                    num_iteration=model.best_iterations[
                        f"{side}_conditional_sign"
                    ],
                ),
                dtype=np.float64,
            )
            positive_by_magnitude[:, class_index] = apply_platt_scaling(
                raw_sign, model.sign_probability_calibration[side]
            )
        profitable_probability = np.sum(
            magnitude_probability * positive_by_magnitude, axis=1
        )
        expected_risk_units = np.sum(
            magnitude_probability
            * representatives
            * (2.0 * positive_by_magnitude - 1.0),
            axis=1,
        )
        q10, q90, cvar10 = _joint_tail_statistics(
            magnitude_probability, positive_by_magnitude, representatives
        )
        output[f"{side}_expected"] = expected_risk_units * stop_width
        output[f"{side}_profitable"] = profitable_probability
        output[f"{side}_q10"] = q10 * stop_width
        output[f"{side}_q90"] = q90 * stop_width
        output[f"{side}_cvar10"] = cvar10 * stop_width
        output[f"{side}_magnitude"] = magnitude_probability
        output[f"{side}_positive_by_magnitude"] = positive_by_magnitude
    if any(not np.all(np.isfinite(value)) for value in output.values()):
        raise ValueError("executable CSM predictions are nonfinite")
    return ExecutableCsmPredictionBatch(
        endpoint_indexes=np.asarray(payoff.source_row_indexes[selected], dtype=np.int64),
        long_expected_net_bps=output["long_expected"],
        short_expected_net_bps=output["short_expected"],
        long_executable=np.asarray(dataset.long_executable[selected], dtype=np.bool_),
        short_executable=np.asarray(dataset.short_executable[selected], dtype=np.bool_),
        long_profitable_probability=output["long_profitable"],
        short_profitable_probability=output["short_profitable"],
        long_q10_net_bps=output["long_q10"],
        short_q10_net_bps=output["short_q10"],
        long_q90_net_bps=output["long_q90"],
        short_q90_net_bps=output["short_q90"],
        long_cvar10_net_bps=output["long_cvar10"],
        short_cvar10_net_bps=output["short_cvar10"],
        long_magnitude_probabilities=output["long_magnitude"],
        short_magnitude_probabilities=output["short_magnitude"],
        long_positive_probability_by_magnitude=output[
            "long_positive_by_magnitude"
        ],
        short_positive_probability_by_magnitude=output[
            "short_positive_by_magnitude"
        ],
    )


def save_executable_csm_model(
    path: str | Path,
    model: TrainedExecutableCsmModel,
) -> None:
    _validate_model(model, reload_boosters=True)
    payload = asdict(model)
    encoded = _canonical_json(payload).encode("utf-8")
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("executable CSM model artifact is too large")
    write_json_atomic(Path(path), payload, indent=None, sort_keys=True)


def load_executable_csm_model(path: str | Path) -> TrainedExecutableCsmModel:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("executable CSM model artifact is missing or oversized")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("executable CSM model artifact is unreadable") from exc
    expected = {field.name for field in fields(TrainedExecutableCsmModel)}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("executable CSM model artifact fields drifted")
    if not isinstance(payload.get("spec"), dict):
        raise ValueError("executable CSM model specification is missing")
    specification = dict(payload["spec"])
    specification["magnitude_edge_quantiles"] = tuple(
        specification["magnitude_edge_quantiles"]
    )
    payload["spec"] = ExecutableCsmSpec(**specification)
    payload["feature_names"] = tuple(payload["feature_names"])
    tuple_maps = (
        "magnitude_class_support",
        "magnitude_edges_risk_units",
        "magnitude_representatives_risk_units",
        "training_joint_probabilities",
        "sign_probability_calibration",
    )
    for name in tuple_maps:
        value = payload[name]
        if name == "magnitude_class_support":
            payload[name] = {
                side: {role: tuple(rows) for role, rows in roles.items()}
                for side, roles in value.items()
            }
        else:
            payload[name] = {side: tuple(rows) for side, rows in value.items()}
    model = TrainedExecutableCsmModel(**payload)
    _validate_model(model, reload_boosters=True)
    return model


__all__ = [
    "EXECUTABLE_CSM_MODEL_SCHEMA_VERSION",
    "ExecutableCsmPredictionBatch",
    "ExecutableCsmSpec",
    "TrainedExecutableCsmModel",
    "load_executable_csm_model",
    "predict_executable_csm_model",
    "save_executable_csm_model",
    "train_executable_csm_model",
]
