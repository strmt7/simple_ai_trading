"""Causal depth-stress transition models for research-only risk evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
from typing import Sequence

import lightgbm as lgb
from lightgbm.basic import LightGBMError
import numpy as np

from .lightgbm_backend import (
    SUPPORTED_LIGHTGBM_BACKEND_KINDS,
    lightgbm_backend_parameters,
)


DEPTH_STRESS_MODEL_SCHEMA_VERSION = "depth-stress-transition-v1"
DEPTH_STRESS_STATE_NAMES = ("calm", "mixed", "stressed")
DEPTH_STRESS_DESCRIPTOR_NAMES = (
    "near_depth_thinness",
    "near_depth_absolute_imbalance",
    "far_to_near_depth_concentration",
)
_STATE_COUNT = len(DEPTH_STRESS_STATE_NAMES)
_PROBABILITY_FLOOR = 1.0e-12


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _row_indexes(
    values: Sequence[int] | np.ndarray,
    *,
    rows: int,
    label: str,
    minimum: int = 1,
) -> np.ndarray:
    indexes = np.asarray(values, dtype=np.int64)
    if (
        indexes.ndim != 1
        or len(indexes) < minimum
        or np.any(indexes < 0)
        or np.any(indexes >= rows)
        or len(np.unique(indexes)) != len(indexes)
    ):
        raise ValueError(f"{label} row indexes are invalid")
    return np.sort(indexes)


def _state_array(values: Sequence[int] | np.ndarray, *, label: str) -> np.ndarray:
    states = np.asarray(values, dtype=np.int8)
    if states.ndim != 1 or np.any((states < 0) | (states >= _STATE_COUNT)):
        raise ValueError(f"{label} states are invalid")
    return states


def _probability_matrix(values: np.ndarray, *, rows: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if (
        probabilities.shape != (rows, _STATE_COUNT)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
    ):
        raise ValueError("depth-stress probabilities are invalid")
    totals = np.sum(probabilities, axis=1)
    if not np.allclose(totals, 1.0, rtol=0.0, atol=1.0e-8):
        raise ValueError("depth-stress probability rows must sum to one")
    return probabilities


@dataclass(frozen=True)
class DepthStressThresholds:
    """Training-only upper-tercile thresholds for oriented book descriptors."""

    descriptor_names: tuple[str, ...]
    upper_tercile: tuple[float, ...]
    fitted_rows: int
    fit_fingerprint: str

    def __post_init__(self) -> None:
        if self.descriptor_names != DEPTH_STRESS_DESCRIPTOR_NAMES:
            raise ValueError("depth-stress descriptor contract is unsupported")
        if (
            len(self.upper_tercile) != len(self.descriptor_names)
            or not all(math.isfinite(value) for value in self.upper_tercile)
            or self.fitted_rows < 30
            or len(self.fit_fingerprint) != 64
        ):
            raise ValueError("depth-stress threshold evidence is invalid")

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["descriptor_names"] = list(self.descriptor_names)
        payload["upper_tercile"] = list(self.upper_tercile)
        return payload


@dataclass(frozen=True)
class DepthStressMetrics:
    rows: int
    negative_log_likelihood: float
    multiclass_brier: float
    stressed_brier: float
    accuracy: float
    stressed_prevalence: float
    mean_predicted_stress_probability: float


@dataclass(frozen=True)
class DepthStressModelArtifact:
    schema_version: str
    model_family: str
    trading_authority: bool
    profitability_claim: bool
    feature_names: tuple[str, ...]
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    seed: int
    training_rows: int
    tuning_rows: int
    best_iteration: int
    model_sha256: str
    model_string: str

    def __post_init__(self) -> None:
        self.validate()

    def _digest_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "trading_authority": self.trading_authority,
            "profitability_claim": self.profitability_claim,
            "feature_names": list(self.feature_names),
            "backend_kind": self.backend_kind,
            "backend_device": self.backend_device,
            "lightgbm_version": self.lightgbm_version,
            "seed": self.seed,
            "training_rows": self.training_rows,
            "tuning_rows": self.tuning_rows,
            "best_iteration": self.best_iteration,
        }

    def validate(self) -> None:
        names = self.feature_names
        if (
            self.schema_version != DEPTH_STRESS_MODEL_SCHEMA_VERSION
            or self.model_family != "lightgbm_shallow_multiclass"
            or self.trading_authority is not False
            or self.profitability_claim is not False
            or not isinstance(names, tuple)
            or not names
            or any(not isinstance(name, str) or not name or name.strip() != name for name in names)
            or len(set(names)) != len(names)
            or self.backend_kind not in SUPPORTED_LIGHTGBM_BACKEND_KINDS
            or not self.backend_device.strip()
            or not self.lightgbm_version.strip()
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.training_rows < 90
            or self.tuning_rows < 30
            or not 1 <= self.best_iteration <= 2_048
            or len(self.model_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.model_sha256)
            or not self.model_string.strip()
        ):
            raise ValueError("depth-stress artifact contract is invalid")
        expected = _serialized_model_sha256(self._digest_contract(), self.model_string)
        if not hmac.compare_digest(self.model_sha256, expected):
            raise ValueError("depth-stress artifact digest does not match its model")

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        return payload


def _serialized_model_sha256(contract: dict[str, object], model_string: str) -> str:
    digest = hashlib.sha256(_canonical_json(contract).encode("ascii"))
    digest.update(b"\x00")
    digest.update(model_string.encode("utf-8"))
    return digest.hexdigest()


def orient_depth_stress_descriptors(
    *,
    bid_near_depth: Sequence[float] | np.ndarray,
    ask_near_depth: Sequence[float] | np.ndarray,
    bid_near_notional: Sequence[float] | np.ndarray,
    ask_near_notional: Sequence[float] | np.ndarray,
    bid_far_notional: Sequence[float] | np.ndarray,
    ask_far_notional: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return scale-stable descriptors where larger values mean greater stress.

    Binance's official ``bookDepth`` files contain cumulative notional bands,
    not an order-by-order L2 book. Near-depth thinness, absolute near-depth
    imbalance, and far/near concentration are therefore named and modeled as
    coarse depth stress rather than full-book liquidity.
    """

    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            bid_near_depth,
            ask_near_depth,
            bid_near_notional,
            ask_near_notional,
            bid_far_notional,
            ask_far_notional,
        )
    )
    if (
        any(values.ndim != 1 for values in arrays)
        or len({len(values) for values in arrays}) != 1
        or not len(arrays[0])
        or not all(np.all(np.isfinite(values)) for values in arrays)
        or any(np.any(values < 0.0) for values in arrays)
    ):
        raise ValueError("depth-stress source arrays are invalid")
    bid_depth, ask_depth, bid_near, ask_near, bid_far, ask_far = arrays
    near_depth = bid_depth + ask_depth
    near_notional = bid_near + ask_near
    far_notional = bid_far + ask_far
    if (
        np.any(near_depth <= 0.0)
        or np.any(near_notional <= 0.0)
        or np.any(far_notional < near_notional)
    ):
        raise ValueError("depth-stress cumulative notional bands are inconsistent")
    thinness = -np.log1p(near_notional)
    absolute_imbalance = np.abs((bid_depth - ask_depth) / near_depth)
    concentration = np.log1p(far_notional) - np.log1p(near_notional)
    descriptors = np.column_stack((thinness, absolute_imbalance, concentration))
    if not np.all(np.isfinite(descriptors)):
        raise ValueError("depth-stress descriptors are non-finite")
    return descriptors


def fit_depth_stress_thresholds(
    descriptors: np.ndarray,
    fit_rows: Sequence[int] | np.ndarray,
) -> DepthStressThresholds:
    """Fit the state encoder on explicitly supplied historical rows only."""

    matrix = np.asarray(descriptors, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(DEPTH_STRESS_DESCRIPTOR_NAMES)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("depth-stress descriptor matrix is invalid")
    indexes = _row_indexes(
        fit_rows,
        rows=len(matrix),
        label="threshold-fit",
        minimum=30,
    )
    selected = np.ascontiguousarray(matrix[indexes], dtype="<f8")
    thresholds = np.quantile(selected, 2.0 / 3.0, axis=0, method="linear")
    digest = hashlib.sha256()
    digest.update(b"depth-stress-threshold-fit-v1\x00")
    digest.update(np.asarray(selected.shape, dtype="<i8").tobytes())
    digest.update(selected.tobytes(order="C"))
    digest.update(np.ascontiguousarray(thresholds, dtype="<f8").tobytes())
    return DepthStressThresholds(
        descriptor_names=DEPTH_STRESS_DESCRIPTOR_NAMES,
        upper_tercile=tuple(float(value) for value in thresholds),
        fitted_rows=len(indexes),
        fit_fingerprint=digest.hexdigest(),
    )


def assign_depth_stress_states(
    descriptors: np.ndarray,
    thresholds: DepthStressThresholds,
    *,
    fail_closed: bool = True,
) -> np.ndarray:
    """Map zero, one, or at least two severe descriptors to three states."""

    if not isinstance(fail_closed, bool):
        raise ValueError("fail_closed must be a boolean")
    matrix = np.asarray(descriptors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(thresholds.upper_tercile):
        raise ValueError("depth-stress descriptor matrix is invalid")
    finite = np.all(np.isfinite(matrix), axis=1)
    if not fail_closed and not np.all(finite):
        raise ValueError("depth-stress descriptors contain non-finite values")
    output = np.full(len(matrix), 2, dtype=np.int8)
    if np.any(finite):
        severe_count = np.sum(
            matrix[finite] >= np.asarray(thresholds.upper_tercile, dtype=np.float64),
            axis=1,
        )
        output[finite] = np.minimum(severe_count, 2).astype(np.int8)
    return output


def fit_depth_transition_probabilities(
    pre_states: Sequence[int] | np.ndarray,
    post_states: Sequence[int] | np.ndarray,
    *,
    rows: Sequence[int] | np.ndarray | None = None,
    alpha: float = 1.0,
    condition_on_pre_state: bool = True,
) -> np.ndarray:
    """Fit a Dirichlet-smoothed marginal or conditional transition baseline."""

    pre = _state_array(pre_states, label="pre")
    post = _state_array(post_states, label="post")
    if pre.shape != post.shape or not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("depth-stress transition inputs are invalid")
    indexes = (
        np.arange(len(pre), dtype=np.int64)
        if rows is None
        else _row_indexes(rows, rows=len(pre), label="transition-fit")
    )
    counts = np.full((_STATE_COUNT, _STATE_COUNT), float(alpha), dtype=np.float64)
    if condition_on_pre_state:
        np.add.at(counts, (pre[indexes], post[indexes]), 1.0)
    else:
        marginal = np.full(_STATE_COUNT, float(alpha), dtype=np.float64)
        np.add.at(marginal, post[indexes], 1.0)
        counts[:] = marginal
    probabilities = counts / np.sum(counts, axis=1, keepdims=True)
    return _probability_matrix(probabilities, rows=_STATE_COUNT)


def predict_depth_transition_probabilities(
    transition_probabilities: np.ndarray,
    pre_states: Sequence[int] | np.ndarray,
) -> np.ndarray:
    pre = _state_array(pre_states, label="pre")
    matrix = _probability_matrix(transition_probabilities, rows=_STATE_COUNT)
    return np.asarray(matrix[pre], dtype=np.float64)


def depth_stress_metrics(
    post_states: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
) -> DepthStressMetrics:
    post = _state_array(post_states, label="post")
    predicted = _probability_matrix(probabilities, rows=len(post))
    losses = depth_stress_loss_rows(post, predicted)
    stress_actual = (post == 2).astype(np.float64)
    stress_probability = predicted[:, 2]
    metrics = DepthStressMetrics(
        rows=len(post),
        negative_log_likelihood=float(np.mean(losses["negative_log_likelihood"])),
        multiclass_brier=float(np.mean(losses["multiclass_brier"])),
        stressed_brier=float(np.mean(losses["stressed_brier"])),
        accuracy=float(np.mean(np.argmax(predicted, axis=1) == post)),
        stressed_prevalence=float(np.mean(stress_actual)),
        mean_predicted_stress_probability=float(np.mean(stress_probability)),
    )
    if not all(math.isfinite(value) for value in asdict(metrics).values()):
        raise ValueError("depth-stress metrics are non-finite")
    return metrics


def depth_stress_loss_rows(
    post_states: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return paired proper-score losses without reducing away time blocks."""

    post = _state_array(post_states, label="post")
    predicted = _probability_matrix(probabilities, rows=len(post))
    selected = np.clip(predicted[np.arange(len(post)), post], _PROBABILITY_FLOOR, 1.0)
    one_hot = np.eye(_STATE_COUNT, dtype=np.float64)[post]
    stress_actual = (post == 2).astype(np.float64)
    return {
        "negative_log_likelihood": -np.log(selected),
        "multiclass_brier": np.sum((predicted - one_hot) ** 2, axis=1),
        "stressed_brier": (predicted[:, 2] - stress_actual) ** 2,
    }


def train_depth_stress_challenger(
    features: np.ndarray,
    post_states: Sequence[int] | np.ndarray,
    *,
    train_rows: Sequence[int] | np.ndarray,
    tuning_rows: Sequence[int] | np.ndarray,
    feature_names: Sequence[str],
    compute_backend: str = "auto",
    seed: int = 20260717,
    maximum_iterations: int = 256,
) -> DepthStressModelArtifact:
    """Fit a bounded shallow classifier with no execution authority."""

    matrix = np.asarray(features, dtype=np.float32)
    labels = _state_array(post_states, label="post")
    names = tuple(str(name).strip() for name in feature_names)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(labels)
        or matrix.shape[1] != len(names)
        or not names
        or any(not name for name in names)
        or len(set(names)) != len(names)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("depth-stress model matrix is invalid")
    train = _row_indexes(train_rows, rows=len(matrix), label="training", minimum=90)
    tuning = _row_indexes(tuning_rows, rows=len(matrix), label="tuning", minimum=30)
    if np.intersect1d(train, tuning).size:
        raise ValueError("depth-stress training and tuning rows overlap")
    if len(np.unique(labels[train])) != _STATE_COUNT:
        raise ValueError("depth-stress training requires all states")
    iterations = int(maximum_iterations)
    if not 32 <= iterations <= 2_048:
        raise ValueError("maximum_iterations must lie in [32, 2048]")
    backend, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        int(seed),
        reproducible=True,
    )
    parameters = {
        **backend,
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": _STATE_COUNT,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": max(24, min(256, len(train) // 100)),
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.01,
        "lambda_l2": 0.10,
        "max_bin": 127,
    }
    model = lgb.train(
        parameters,
        lgb.Dataset(matrix[train], label=labels[train], free_raw_data=False),
        num_boost_round=iterations,
        valid_sets=[lgb.Dataset(matrix[tuning], label=labels[tuning], reference=None)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    best_iteration = int(model.best_iteration or iterations)
    model_string = model.model_to_string(num_iteration=best_iteration)
    try:
        reloaded = lgb.Booster(model_str=model_string)
    except LightGBMError as exc:
        raise ValueError("serialized depth-stress model could not be reloaded") from exc
    probabilities = np.asarray(
        reloaded.predict(matrix[tuning], num_iteration=best_iteration),
        dtype=np.float64,
    )
    _probability_matrix(probabilities, rows=len(tuning))
    contract = {
        "schema_version": DEPTH_STRESS_MODEL_SCHEMA_VERSION,
        "model_family": "lightgbm_shallow_multiclass",
        "trading_authority": False,
        "profitability_claim": False,
        "feature_names": list(names),
        "backend_kind": backend_kind,
        "backend_device": backend_device,
        "lightgbm_version": str(lgb.__version__),
        "seed": int(seed),
        "training_rows": len(train),
        "tuning_rows": len(tuning),
        "best_iteration": best_iteration,
    }
    return DepthStressModelArtifact(
        schema_version=DEPTH_STRESS_MODEL_SCHEMA_VERSION,
        model_family="lightgbm_shallow_multiclass",
        trading_authority=False,
        profitability_claim=False,
        feature_names=names,
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        seed=int(seed),
        training_rows=len(train),
        tuning_rows=len(tuning),
        best_iteration=best_iteration,
        model_sha256=_serialized_model_sha256(contract, model_string),
        model_string=model_string,
    )


def predict_depth_stress_challenger(
    artifact: DepthStressModelArtifact,
    features: np.ndarray,
) -> np.ndarray:
    artifact.validate()
    matrix = np.asarray(features, dtype=np.float32)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(artifact.feature_names)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("depth-stress prediction matrix is invalid")
    try:
        model = lgb.Booster(model_str=artifact.model_string)
    except LightGBMError as exc:
        raise ValueError("depth-stress model could not be loaded") from exc
    probabilities = np.asarray(
        model.predict(matrix, num_iteration=artifact.best_iteration),
        dtype=np.float64,
    )
    return _probability_matrix(probabilities, rows=len(matrix))


__all__ = [
    "DEPTH_STRESS_DESCRIPTOR_NAMES",
    "DEPTH_STRESS_MODEL_SCHEMA_VERSION",
    "DEPTH_STRESS_STATE_NAMES",
    "DepthStressMetrics",
    "DepthStressModelArtifact",
    "DepthStressThresholds",
    "assign_depth_stress_states",
    "depth_stress_loss_rows",
    "depth_stress_metrics",
    "fit_depth_stress_thresholds",
    "fit_depth_transition_probabilities",
    "orient_depth_stress_descriptors",
    "predict_depth_stress_challenger",
    "predict_depth_transition_probabilities",
    "train_depth_stress_challenger",
]
