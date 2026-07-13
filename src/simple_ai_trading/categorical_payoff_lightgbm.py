"""Replay-aligned categorical distributions of exact barrier payoffs."""

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
from .microstructure_action_architecture import ActionValuePredictionBatch
from .microstructure_barriers import (
    AdaptiveBarrierTargets,
    validate_adaptive_barrier_targets,
)
from .microstructure_features import (
    MicrostructureDataset,
    validate_microstructure_dataset,
)
from .storage import write_json_atomic


CATEGORICAL_PAYOFF_SCHEMA_VERSION = "categorical-barrier-payoff-lightgbm-v1"
CATEGORICAL_DATASET_SCHEMA_VERSION = "categorical-barrier-payoff-dataset-v1"
_MODEL_FAMILY = "side_specific_categorical_exact_payoff"
_SIDES = ("long", "short")
_MINIMUM_TRAIN_ROWS_PER_CLASS = 64
_MINIMUM_EARLY_ROWS_PER_CLASS = 16
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
CategoricalProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class CategoricalPayoffSpec:
    candidate_id: str
    family: str
    bin_edge_quantiles: tuple[float, ...]
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
    minimum_unique_bins: int
    temperature_minimum: float = 0.5
    temperature_maximum: float = 3.0
    gpu_use_dp_required: bool = True

    def __post_init__(self) -> None:
        quantiles = tuple(float(value) for value in self.bin_edge_quantiles)
        numeric = (
            self.learning_rate,
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
            or not 32 <= int(self.min_data_in_leaf) <= 65_536
            or not 0.0 < self.feature_fraction <= 1.0
            or not 0.0 < self.bagging_fraction <= 1.0
            or not 0 <= int(self.bagging_freq) <= 100
            or self.lambda_l1 < 0.0
            or self.lambda_l2 < 0.0
            or not 31 <= int(self.max_bin) <= 255
            or not 10 <= int(self.num_boost_round) <= 10_000
            or not 5 <= int(self.early_stopping_rounds) < int(self.num_boost_round)
            or not 4 <= int(self.minimum_unique_bins) <= len(quantiles) + 1
            or not 0.0 < self.temperature_minimum <= 1.0
            or not 1.0 <= self.temperature_maximum <= 10.0
            or self.temperature_minimum >= self.temperature_maximum
            or self.gpu_use_dp_required is not True
        ):
            raise ValueError("categorical payoff specification is invalid")


@dataclass(frozen=True)
class CategoricalPayoffDataset:
    schema_version: str
    symbol: str
    target_scenario: str
    feature_names: tuple[str, ...]
    decision_time_ms: np.ndarray
    features: np.ndarray
    stop_width_bps: np.ndarray
    long_net_bps: np.ndarray
    short_net_bps: np.ndarray
    long_exit_time_ms: np.ndarray
    short_exit_time_ms: np.ndarray
    source_row_indexes: np.ndarray
    source_dataset_sha256: str
    dataset_sha256: str

    @property
    def rows(self) -> int:
        return int(len(self.decision_time_ms))

    def __post_init__(self) -> None:
        rows = self.rows
        vectors = (
            np.asarray(self.stop_width_bps),
            np.asarray(self.long_net_bps),
            np.asarray(self.short_net_bps),
            np.asarray(self.long_exit_time_ms),
            np.asarray(self.short_exit_time_ms),
            np.asarray(self.source_row_indexes),
        )
        if (
            self.schema_version != CATEGORICAL_DATASET_SCHEMA_VERSION
            or not self.symbol
            or self.target_scenario not in {"base", "stress"}
            or not self.feature_names
            or len(set(self.feature_names)) != len(self.feature_names)
            or any(not name.strip() for name in self.feature_names)
            or rows <= 0
            or np.asarray(self.decision_time_ms).shape != (rows,)
            or np.any(np.diff(self.decision_time_ms) <= 0)
            or np.asarray(self.features).shape != (rows, len(self.feature_names))
            or not np.all(np.isfinite(self.features))
            or any(value.shape != (rows,) for value in vectors)
            or not np.all(np.isfinite(self.stop_width_bps))
            or np.any(self.stop_width_bps <= 0.0)
            or not np.all(np.isfinite(self.long_net_bps))
            or not np.all(np.isfinite(self.short_net_bps))
            or np.any(self.long_exit_time_ms <= self.decision_time_ms)
            or np.any(self.short_exit_time_ms <= self.decision_time_ms)
            or np.any(np.diff(self.source_row_indexes) <= 0)
            or len(self.source_dataset_sha256) != 64
            or len(self.dataset_sha256) != 64
        ):
            raise ValueError("categorical payoff dataset is invalid")


@dataclass(frozen=True)
class TrainedCategoricalPayoffModel:
    schema_version: str
    model_family: str
    spec: CategoricalPayoffSpec
    symbol: str
    feature_names: tuple[str, ...]
    source_dataset_sha256: str
    target_scenario: str
    backend_requested: str
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    seed: int
    training_rows: int
    early_stop_rows: int
    calibration_rows: int
    bin_edges_risk_units: Mapping[str, tuple[float, ...]]
    bin_representatives_risk_units: Mapping[str, tuple[float, ...]]
    bin_positive_rates: Mapping[str, tuple[float, ...]]
    class_support: Mapping[str, Mapping[str, tuple[int, ...]]]
    temperature: Mapping[str, float]
    calibration_log_loss_before: Mapping[str, float]
    calibration_log_loss_after: Mapping[str, float]
    best_iterations: Mapping[str, int]
    model_strings: Mapping[str, str]
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False


@dataclass(frozen=True)
class CategoricalPayoffPredictionBatch:
    action_values: ActionValuePredictionBatch
    long_probabilities: np.ndarray
    short_probabilities: np.ndarray
    long_cvar10_bps: np.ndarray
    short_cvar10_bps: np.ndarray

    def __post_init__(self) -> None:
        rows = self.action_values.rows
        long_probabilities = np.asarray(self.long_probabilities, dtype=np.float64)
        short_probabilities = np.asarray(self.short_probabilities, dtype=np.float64)
        if (
            rows <= 0
            or long_probabilities.ndim != 2
            or short_probabilities.ndim != 2
            or long_probabilities.shape[0] != rows
            or short_probabilities.shape[0] != rows
            or not np.all(np.isfinite(long_probabilities))
            or not np.all(np.isfinite(short_probabilities))
            or np.any(long_probabilities < 0.0)
            or np.any(short_probabilities < 0.0)
            or not np.allclose(long_probabilities.sum(axis=1), 1.0, atol=1e-10)
            or not np.allclose(short_probabilities.sum(axis=1), 1.0, atol=1e-10)
            or np.asarray(self.long_cvar10_bps).shape != (rows,)
            or np.asarray(self.short_cvar10_bps).shape != (rows,)
            or not np.all(np.isfinite(self.long_cvar10_bps))
            or not np.all(np.isfinite(self.short_cvar10_bps))
        ):
            raise ValueError("categorical payoff prediction batch is invalid")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _array_digest(
    *,
    names: Sequence[str],
    arrays: Sequence[np.ndarray],
) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(str(name).encode("ascii"))
        digest.update(b"\0")
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _source_dataset_identity(dataset: MicrostructureDataset) -> str:
    evidence = dict(dataset.source_evidence or {})
    payload = {
        "symbol": dataset.symbol,
        "feature_version": dataset.feature_version,
        "feature_names": dataset.feature_names,
        "horizon_seconds": dataset.horizon_seconds,
        "decision_cadence_seconds": dataset.decision_cadence_seconds,
        "target_mode": dataset.target_mode,
        "source_evidence": evidence,
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def build_categorical_payoff_dataset(
    dataset: MicrostructureDataset,
    targets: AdaptiveBarrierTargets,
    *,
    target_scenario: str,
    extra_feature_names: Sequence[str] = (),
    extra_features: np.ndarray | None = None,
) -> CategoricalPayoffDataset:
    """Bind exact valid barrier targets to causal deterministic and AI features."""

    validate_microstructure_dataset(dataset)
    validate_adaptive_barrier_targets(dataset, targets)
    positions = np.flatnonzero(np.asarray(targets.valid, dtype=bool))
    source_indexes = np.asarray(targets.source_indexes[positions], dtype=np.int64)
    if target_scenario == "base":
        long_net = np.asarray(targets.base_long_net_bps[positions], dtype=np.float32)
        short_net = np.asarray(targets.base_short_net_bps[positions], dtype=np.float32)
        long_exit = np.asarray(
            targets.base_long_exit_time_ms[positions], dtype=np.int64
        )
        short_exit = np.asarray(
            targets.base_short_exit_time_ms[positions], dtype=np.int64
        )
    elif target_scenario == "stress":
        long_net = np.asarray(targets.stress_long_net_bps[positions], dtype=np.float32)
        short_net = np.asarray(
            targets.stress_short_net_bps[positions], dtype=np.float32
        )
        long_exit = np.asarray(
            targets.stress_long_exit_time_ms[positions], dtype=np.int64
        )
        short_exit = np.asarray(
            targets.stress_short_exit_time_ms[positions], dtype=np.int64
        )
    else:
        raise ValueError("categorical payoff target scenario is unsupported")
    base_features = np.asarray(dataset.features[source_indexes], dtype=np.float32)
    extra_names = tuple(str(value) for value in extra_feature_names)
    if extra_features is None:
        if extra_names:
            raise ValueError("categorical payoff extra feature names lack values")
        combined = base_features
    else:
        extra = np.asarray(extra_features, dtype=np.float32)
        if (
            not extra_names
            or len(set(extra_names)) != len(extra_names)
            or set(extra_names).intersection(dataset.feature_names)
            or extra.shape != (dataset.rows, len(extra_names))
            or not np.all(np.isfinite(extra))
        ):
            raise ValueError("categorical payoff extra features are invalid")
        combined = np.column_stack((base_features, extra[source_indexes])).astype(
            np.float32
        )
    feature_names = tuple(dataset.feature_names) + extra_names
    decision_time = np.asarray(dataset.decision_time_ms[source_indexes], dtype=np.int64)
    stop_width = np.asarray(targets.stop_barrier_bps[positions], dtype=np.float32)
    source_identity = _source_dataset_identity(dataset)
    identity = _array_digest(
        names=feature_names + (target_scenario, source_identity),
        arrays=(
            decision_time,
            combined,
            stop_width,
            long_net,
            short_net,
            long_exit,
            short_exit,
            source_indexes,
        ),
    )
    return CategoricalPayoffDataset(
        schema_version=CATEGORICAL_DATASET_SCHEMA_VERSION,
        symbol=dataset.symbol,
        target_scenario=target_scenario,
        feature_names=feature_names,
        decision_time_ms=decision_time,
        features=combined,
        stop_width_bps=stop_width,
        long_net_bps=long_net,
        short_net_bps=short_net,
        long_exit_time_ms=long_exit,
        short_exit_time_ms=short_exit,
        source_row_indexes=source_indexes,
        source_dataset_sha256=source_identity,
        dataset_sha256=identity,
    )


def _validate_indexes(
    values: np.ndarray,
    *,
    rows: int,
    label: str,
    minimum_rows: int,
) -> np.ndarray:
    indexes = np.asarray(values, dtype=np.int64)
    if (
        indexes.ndim != 1
        or len(indexes) < minimum_rows
        or indexes[0] < 0
        or indexes[-1] >= rows
        or np.any(np.diff(indexes) <= 0)
    ):
        raise ValueError(f"categorical payoff {label} indexes are invalid")
    return indexes


def _fit_bins(
    risk_units: np.ndarray,
    quantiles: Sequence[float],
    *,
    minimum_unique_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(risk_units, dtype=np.float64)
    edges = np.unique(np.quantile(values, quantiles, method="linear"))
    if len(edges) + 1 < minimum_unique_bins or not np.all(np.diff(edges) > 0.0):
        raise ValueError("categorical payoff target has insufficient unique bins")
    labels = np.searchsorted(edges, values, side="right").astype(np.int32)
    class_count = len(edges) + 1
    support = np.bincount(labels, minlength=class_count)
    if int(np.min(support)) < _MINIMUM_TRAIN_ROWS_PER_CLASS:
        raise ValueError("categorical payoff training class support is insufficient")
    representatives = np.asarray(
        [float(np.mean(values[labels == index])) for index in range(class_count)],
        dtype=np.float64,
    )
    positive_rates = np.asarray(
        [float(np.mean(values[labels == index] > 0.0)) for index in range(class_count)],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(representatives))
        or not np.all(np.diff(representatives) >= 0.0)
        or not np.all(np.isfinite(positive_rates))
        or np.any(positive_rates < 0.0)
        or np.any(positive_rates > 1.0)
    ):
        raise ValueError("categorical payoff bin statistics are invalid")
    return edges, labels, representatives, positive_rates


def _temperature_probabilities(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=np.float64)
    if raw.ndim != 2 or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("categorical payoff temperature input is invalid")
    logits = np.log(np.clip(raw, 1e-15, 1.0)) / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    output = np.exp(logits)
    output /= np.sum(output, axis=1, keepdims=True)
    return output


def multiclass_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if (
        values.ndim != 2
        or target.shape != (values.shape[0],)
        or target.size == 0
        or np.any(target < 0)
        or np.any(target >= values.shape[1])
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-10)
    ):
        raise ValueError("multiclass log-loss inputs are invalid")
    return float(
        -np.mean(np.log(np.clip(values[np.arange(len(target)), target], 1e-15, 1.0)))
    )


def ranked_probability_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] < 2
        or target.shape != (len(values),)
        or np.any(target < 0)
        or np.any(target >= values.shape[1])
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-10)
    ):
        raise ValueError("ranked-probability inputs are invalid")
    cumulative = np.cumsum(values, axis=1)[:, :-1]
    thresholds = np.arange(values.shape[1] - 1)[None, :]
    observed = (target[:, None] <= thresholds).astype(np.float64)
    return float(np.mean(np.sum((cumulative - observed) ** 2, axis=1)))


def _fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> tuple[float, float, float]:
    before = multiclass_log_loss(probabilities, labels)

    def objective(log_temperature: float) -> float:
        return multiclass_log_loss(
            _temperature_probabilities(probabilities, math.exp(log_temperature)),
            labels,
        )

    lower = math.log(minimum)
    upper = math.log(maximum)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = objective(left)
    right_value = objective(right)
    for _iteration in range(80):
        if left_value <= right_value:
            upper = right
            right = left
            right_value = left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(left)
        else:
            lower = left
            left = right
            left_value = right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)
    candidate = math.exp((lower + upper) / 2.0)
    after = objective(math.log(candidate))
    if after > before:
        return 1.0, before, before
    return float(candidate), before, after


def _train_multiclass(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_early: np.ndarray,
    y_early: np.ndarray,
    parameters: Mapping[str, object],
    class_count: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[lgb.Booster, int]:
    configuration = {
        **parameters,
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": int(class_count),
    }
    training = lgb.Dataset(
        np.asarray(x_train, dtype=np.float32),
        label=np.asarray(y_train, dtype=np.int32),
        free_raw_data=False,
    )
    early = lgb.Dataset(
        np.asarray(x_early, dtype=np.float32),
        label=np.asarray(y_early, dtype=np.int32),
        reference=training,
        free_raw_data=False,
    )
    booster = lgb.train(
        configuration,
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


def _model_payload(model: TrainedCategoricalPayoffModel) -> dict[str, object]:
    payload = asdict(model)
    payload.pop("model_sha256")
    return payload


def _model_sha256(model: TrainedCategoricalPayoffModel) -> str:
    return hashlib.sha256(
        _canonical_json(_model_payload(model)).encode("ascii")
    ).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_model(model: TrainedCategoricalPayoffModel, *, reload: bool) -> None:
    sides = set(_SIDES)
    if (
        model.schema_version != CATEGORICAL_PAYOFF_SCHEMA_VERSION
        or model.model_family != _MODEL_FAMILY
        or model.spec.family != _MODEL_FAMILY
        or model.target_scenario not in {"base", "stress"}
        or model.backend_kind not in {"opencl", "cpu"}
        or model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.portfolio_claim
        or model.leverage_applied
        or not model.symbol
        or not model.feature_names
        or len(set(model.feature_names)) != len(model.feature_names)
        or any(not name.strip() for name in model.feature_names)
        or not _is_sha256(model.source_dataset_sha256)
        or not _is_sha256(model.model_sha256)
        or model.training_rows < 1_024
        or model.early_stop_rows < 512
        or model.calibration_rows < 512
        or set(model.bin_edges_risk_units) != sides
        or set(model.bin_representatives_risk_units) != sides
        or set(model.bin_positive_rates) != sides
        or set(model.class_support) != sides
        or set(model.temperature) != sides
        or set(model.calibration_log_loss_before) != sides
        or set(model.calibration_log_loss_after) != sides
        or set(model.model_strings) != sides
        or set(model.best_iterations) != sides
        or _model_sha256(model) != model.model_sha256
    ):
        raise ValueError("categorical payoff model contract is invalid")
    for side in _SIDES:
        edges = np.asarray(model.bin_edges_risk_units[side], dtype=np.float64)
        representatives = np.asarray(
            model.bin_representatives_risk_units[side], dtype=np.float64
        )
        positive_rates = np.asarray(model.bin_positive_rates[side], dtype=np.float64)
        classes = len(edges) + 1
        role_support = model.class_support[side]
        temperature = float(model.temperature[side])
        loss_before = float(model.calibration_log_loss_before[side])
        loss_after = float(model.calibration_log_loss_after[side])
        if (
            classes < model.spec.minimum_unique_bins
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
            or representatives.shape != (classes,)
            or not np.all(np.isfinite(representatives))
            or np.any(np.diff(representatives) < 0.0)
            or positive_rates.shape != (classes,)
            or not np.all(np.isfinite(positive_rates))
            or np.any(positive_rates < 0.0)
            or np.any(positive_rates > 1.0)
            or set(role_support) != {"train", "early_stop", "calibration"}
            or any(len(values) != classes for values in role_support.values())
            or any(
                any(int(value) < 0 for value in values)
                for values in role_support.values()
            )
            or sum(role_support["train"]) != model.training_rows
            or sum(role_support["early_stop"]) != model.early_stop_rows
            or sum(role_support["calibration"]) != model.calibration_rows
            or min(role_support["train"]) < _MINIMUM_TRAIN_ROWS_PER_CLASS
            or min(role_support["early_stop"]) < _MINIMUM_EARLY_ROWS_PER_CLASS
            or not math.isfinite(temperature)
            or not model.spec.temperature_minimum
            <= temperature
            <= model.spec.temperature_maximum
            or not math.isfinite(loss_before)
            or not math.isfinite(loss_after)
            or loss_before < 0.0
            or loss_after < 0.0
            or loss_after > loss_before + 1e-12
            or not 1 <= int(model.best_iterations[side]) <= model.spec.num_boost_round
            or not model.model_strings[side].strip()
        ):
            raise ValueError("categorical payoff class contract is invalid")
    if reload:
        try:
            for side in _SIDES:
                booster = lgb.Booster(model_str=model.model_strings[side])
                if booster.num_model_per_iteration() != len(
                    model.bin_representatives_risk_units[side]
                ):
                    raise ValueError("categorical payoff booster class count drifted")
        except lgb.basic.LightGBMError as exc:
            raise ValueError("categorical payoff booster cannot be reloaded") from exc


def train_categorical_payoff_model(
    dataset: CategoricalPayoffDataset,
    *,
    train_indexes: np.ndarray,
    early_stop_indexes: np.ndarray,
    calibration_indexes: np.ndarray,
    spec: CategoricalPayoffSpec,
    target_scenario: str,
    compute_backend: str,
    seed: int,
    progress: CategoricalProgressCallback | None = None,
) -> TrainedCategoricalPayoffModel:
    train = _validate_indexes(
        train_indexes,
        rows=dataset.rows,
        label="training",
        minimum_rows=1_024,
    )
    early = _validate_indexes(
        early_stop_indexes,
        rows=dataset.rows,
        label="early-stop",
        minimum_rows=512,
    )
    calibration = _validate_indexes(
        calibration_indexes,
        rows=dataset.rows,
        label="calibration",
        minimum_rows=512,
    )
    if train[-1] >= early[0] or early[-1] >= calibration[0]:
        raise ValueError("categorical payoff chronological roles overlap")
    maximum_exit = np.maximum(dataset.long_exit_time_ms, dataset.short_exit_time_ms)
    if np.any(maximum_exit[train] >= dataset.decision_time_ms[early[0]]) or np.any(
        maximum_exit[early] >= dataset.decision_time_ms[calibration[0]]
    ):
        raise ValueError("categorical payoff labels cross a role boundary")
    if target_scenario != dataset.target_scenario:
        raise ValueError("categorical payoff target scenario drifted")
    parameters, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        int(seed),
        reproducible=True,
    )
    if backend_kind == "opencl" and (
        not spec.gpu_use_dp_required or parameters.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("categorical payoff OpenCL FP64 accumulation is required")
    common: dict[str, object] = {
        **parameters,
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
    values = {
        "long": np.asarray(dataset.long_net_bps, dtype=np.float64),
        "short": np.asarray(dataset.short_net_bps, dtype=np.float64),
    }
    edges: dict[str, tuple[float, ...]] = {}
    representatives: dict[str, tuple[float, ...]] = {}
    positive_rates: dict[str, tuple[float, ...]] = {}
    support: dict[str, dict[str, tuple[int, ...]]] = {}
    temperatures: dict[str, float] = {}
    before: dict[str, float] = {}
    after: dict[str, float] = {}
    iterations: dict[str, int] = {}
    model_strings: dict[str, str] = {}
    for side_index, side in enumerate(_SIDES):
        if progress is not None:
            progress(side, side_index + 1, len(_SIDES))
        risk = values[side] / np.asarray(dataset.stop_width_bps, dtype=np.float64)
        fitted_edges, train_labels, fitted_representatives, fitted_positive = _fit_bins(
            risk[train],
            spec.bin_edge_quantiles,
            minimum_unique_bins=spec.minimum_unique_bins,
        )
        early_labels = np.searchsorted(
            fitted_edges,
            risk[early],
            side="right",
        ).astype(np.int32)
        calibration_labels = np.searchsorted(
            fitted_edges,
            risk[calibration],
            side="right",
        ).astype(np.int32)
        class_count = len(fitted_edges) + 1
        role_support = {
            "train": tuple(
                int(value) for value in np.bincount(train_labels, minlength=class_count)
            ),
            "early_stop": tuple(
                int(value) for value in np.bincount(early_labels, minlength=class_count)
            ),
            "calibration": tuple(
                int(value)
                for value in np.bincount(calibration_labels, minlength=class_count)
            ),
        }
        if min(role_support["early_stop"]) < _MINIMUM_EARLY_ROWS_PER_CLASS:
            raise ValueError(
                f"categorical payoff {side} early-stop class support failed"
            )
        booster, best_iteration = _train_multiclass(
            x_train=features[train],
            y_train=train_labels,
            x_early=features[early],
            y_early=early_labels,
            parameters=common,
            class_count=class_count,
            num_boost_round=spec.num_boost_round,
            early_stopping_rounds=spec.early_stopping_rounds,
        )
        raw_calibration = np.asarray(
            booster.predict(features[calibration], num_iteration=best_iteration),
            dtype=np.float64,
        )
        temperature, loss_before, loss_after = _fit_temperature(
            raw_calibration,
            calibration_labels,
            minimum=spec.temperature_minimum,
            maximum=spec.temperature_maximum,
        )
        edges[side] = tuple(float(value) for value in fitted_edges)
        representatives[side] = tuple(float(value) for value in fitted_representatives)
        positive_rates[side] = tuple(float(value) for value in fitted_positive)
        support[side] = role_support
        temperatures[side] = temperature
        before[side] = loss_before
        after[side] = loss_after
        iterations[side] = best_iteration
        model_strings[side] = booster.model_to_string(num_iteration=best_iteration)
    provisional = TrainedCategoricalPayoffModel(
        schema_version=CATEGORICAL_PAYOFF_SCHEMA_VERSION,
        model_family=_MODEL_FAMILY,
        spec=spec,
        symbol=dataset.symbol,
        feature_names=dataset.feature_names,
        source_dataset_sha256=dataset.dataset_sha256,
        target_scenario=target_scenario,
        backend_requested=str(compute_backend),
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        seed=int(seed),
        training_rows=len(train),
        early_stop_rows=len(early),
        calibration_rows=len(calibration),
        bin_edges_risk_units=edges,
        bin_representatives_risk_units=representatives,
        bin_positive_rates=positive_rates,
        class_support=support,
        temperature=temperatures,
        calibration_log_loss_before=before,
        calibration_log_loss_after=after,
        best_iterations=iterations,
        model_strings=model_strings,
        model_sha256="",
    )
    model = TrainedCategoricalPayoffModel(
        **{**provisional.__dict__, "model_sha256": _model_sha256(provisional)}
    )
    _validate_model(model, reload=False)
    return model


def _discrete_quantile(
    probabilities: np.ndarray,
    representatives: np.ndarray,
    probability: float,
) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    indexes = np.argmax(cumulative >= float(probability), axis=1)
    return representatives[indexes]


def _discrete_cvar(
    probabilities: np.ndarray,
    representatives: np.ndarray,
    probability: float,
) -> np.ndarray:
    remaining = np.full(len(probabilities), float(probability), dtype=np.float64)
    total = np.zeros(len(probabilities), dtype=np.float64)
    for class_index, representative in enumerate(representatives):
        consumed = np.minimum(remaining, probabilities[:, class_index])
        total += consumed * float(representative)
        remaining -= consumed
    if np.any(remaining > 1e-10):
        raise ValueError("categorical payoff CVaR mass is incomplete")
    return total / float(probability)


def predict_categorical_payoff_model(
    model: TrainedCategoricalPayoffModel,
    dataset: CategoricalPayoffDataset,
    indexes: np.ndarray,
) -> CategoricalPayoffPredictionBatch:
    _validate_model(model, reload=False)
    if (
        model.symbol != dataset.symbol
        or model.feature_names != dataset.feature_names
        or model.source_dataset_sha256 != dataset.dataset_sha256
    ):
        raise ValueError("categorical payoff prediction dataset drifted")
    selected = _validate_indexes(
        indexes,
        rows=dataset.rows,
        label="prediction",
        minimum_rows=1,
    )
    features = np.asarray(dataset.features[selected], dtype=np.float32)
    stop_width = np.asarray(dataset.stop_width_bps[selected], dtype=np.float64)
    outputs: dict[str, np.ndarray] = {}
    probabilities_by_side: dict[str, np.ndarray] = {}
    for side in _SIDES:
        booster = lgb.Booster(model_str=model.model_strings[side])
        raw = np.asarray(
            booster.predict(features, num_iteration=model.best_iterations[side]),
            dtype=np.float64,
        )
        probabilities = _temperature_probabilities(raw, model.temperature[side])
        representatives = np.asarray(
            model.bin_representatives_risk_units[side], dtype=np.float64
        )
        positive_rates = np.asarray(model.bin_positive_rates[side], dtype=np.float64)
        probabilities_by_side[side] = probabilities
        outputs[f"{side}_mean"] = probabilities @ representatives * stop_width
        outputs[f"{side}_positive"] = probabilities @ positive_rates
        outputs[f"{side}_q10"] = (
            _discrete_quantile(probabilities, representatives, 0.10) * stop_width
        )
        outputs[f"{side}_q90"] = (
            _discrete_quantile(probabilities, representatives, 0.90) * stop_width
        )
        outputs[f"{side}_cvar10"] = (
            _discrete_cvar(probabilities, representatives, 0.10) * stop_width
        )
    if any(not np.all(np.isfinite(value)) for value in outputs.values()):
        raise ValueError("categorical payoff predictions are nonfinite")
    action_values = ActionValuePredictionBatch(
        endpoint_indexes=np.asarray(
            dataset.source_row_indexes[selected], dtype=np.int64
        ),
        long_mean_bps=outputs["long_mean"],
        short_mean_bps=outputs["short_mean"],
        long_profitable_probability=outputs["long_positive"],
        short_profitable_probability=outputs["short_positive"],
        long_lower_bps=outputs["long_q10"],
        short_lower_bps=outputs["short_q10"],
        long_upper_bps=outputs["long_q90"],
        short_upper_bps=outputs["short_q90"],
    )
    return CategoricalPayoffPredictionBatch(
        action_values=action_values,
        long_probabilities=probabilities_by_side["long"],
        short_probabilities=probabilities_by_side["short"],
        long_cvar10_bps=outputs["long_cvar10"],
        short_cvar10_bps=outputs["short_cvar10"],
    )


def save_categorical_payoff_model(
    path: str | Path,
    model: TrainedCategoricalPayoffModel,
) -> None:
    _validate_model(model, reload=True)
    payload = asdict(model)
    encoded = _canonical_json(payload).encode("utf-8")
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("categorical payoff model artifact is too large")
    write_json_atomic(Path(path), payload, indent=None, sort_keys=True)


def load_categorical_payoff_model(
    path: str | Path,
) -> TrainedCategoricalPayoffModel:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("categorical payoff model artifact is missing or oversized")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("categorical payoff model artifact must be an object")
    expected = {field.name for field in fields(TrainedCategoricalPayoffModel)}
    if set(payload) != expected or not isinstance(payload.get("spec"), dict):
        raise ValueError("categorical payoff model artifact fields drifted")
    payload["spec"] = CategoricalPayoffSpec(**payload["spec"])
    payload["feature_names"] = tuple(payload["feature_names"])
    for key in (
        "bin_edges_risk_units",
        "bin_representatives_risk_units",
        "bin_positive_rates",
    ):
        payload[key] = {
            side: tuple(float(value) for value in values)
            for side, values in payload[key].items()
        }
    payload["class_support"] = {
        side: {
            role: tuple(int(value) for value in values)
            for role, values in roles.items()
        }
        for side, roles in payload["class_support"].items()
    }
    model = TrainedCategoricalPayoffModel(**payload)
    _validate_model(model, reload=True)
    return model
