"""Frozen, leakage-resistant controls for the Round 25 TWAP-native model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
)
from .polymarket_round25_dataset import (
    POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
    Round25DevelopmentDataset,
    require_round25_dataset_minimum,
)
from .polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
)
from .polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_CONDITION_DURATION_MS,
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256 = (
    "4865de354943bf3db3d8a5a3b93f348606944c3bb5d75a55a49417ff79872e54"
)
POLYMARKET_ROUND25_PHASE_ISOTONIC_SCHEMA_VERSION = (
    "polymarket-round25-phase-isotonic-control-v1"
)
POLYMARKET_ROUND25_LOGISTIC_RESIDUAL_SCHEMA_VERSION = (
    "polymarket-round25-l2-logistic-residual-control-v1"
)
POLYMARKET_ROUND25_L2_GRID = (0.01, 0.1, 1.0, 10.0)
POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND = 4.0
POLYMARKET_ROUND25_PHASE_COUNT = 4
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256 = _canonical_sha256(
    list(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
)
_BINARY_AVAILABILITY_INDICES = tuple(
    index
    for index, name in enumerate(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    if name.endswith("_available")
)


def _finite_tuple(values: Sequence[float], *, expected: int) -> tuple[float, ...]:
    if any(isinstance(value, bool) for value in values):
        raise ValueError("Round 25 numeric vector is invalid")
    output = tuple(float(value) for value in values)
    if len(output) != expected or any(not math.isfinite(value) for value in output):
        raise ValueError("Round 25 numeric vector is invalid")
    return output


def _phase_index(event_start_ms: int, decision_time_ms: int) -> int:
    if (
        type(event_start_ms) is not int
        or type(decision_time_ms) is not int
        or not event_start_ms
        <= decision_time_ms
        < event_start_ms + POLYMARKET_ROUND25_CONDITION_DURATION_MS
    ):
        raise ValueError("Round 25 event phase timestamp is invalid")
    elapsed = decision_time_ms - event_start_ms
    return min(
        POLYMARKET_ROUND25_PHASE_COUNT - 1,
        elapsed * POLYMARKET_ROUND25_PHASE_COUNT
        // POLYMARKET_ROUND25_CONDITION_DURATION_MS,
    )


@dataclass(frozen=True, slots=True)
class Round25IsotonicPhaseModel:
    phase: int
    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            type(self.phase) is not int
            or not 0 <= self.phase < POLYMARKET_ROUND25_PHASE_COUNT
            or not self.x_thresholds
            or len(self.x_thresholds) != len(self.y_thresholds)
            or any(
                not math.isfinite(value) or not 0.0 < value < 1.0
                for value in self.x_thresholds
            )
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in self.y_thresholds
            )
            or any(
                right <= left
                for left, right in zip(self.x_thresholds, self.x_thresholds[1:])
            )
            or any(
                right < left
                for left, right in zip(self.y_thresholds, self.y_thresholds[1:])
            )
        ):
            raise ValueError("Round 25 isotonic phase model is invalid")

    def payload(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "x_thresholds": list(self.x_thresholds),
            "y_thresholds": list(self.y_thresholds),
        }


@dataclass(frozen=True, slots=True)
class Round25PhaseIsotonicArtifact:
    fit_dataset_sha256: str
    resolution_authority_sha256: str
    phase_models: tuple[Round25IsotonicPhaseModel, ...]
    artifact_sha256: str
    schema_version: str = POLYMARKET_ROUND25_PHASE_ISOTONIC_SCHEMA_VERSION
    candidate_id: str = "phase-isotonic-market-prior-v1"
    fit_role: str = "calibration"
    feature_schema_version: str = POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION
    feature_names_sha256: str = POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256
    model_design_sha256: str = POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
    candidate_design_sha256: str = POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
    candidate_amendment_sha256: str = (
        POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
    )
    control_fit_contract_sha256: str = (
        POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256
    )
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "candidate_amendment_sha256": self.candidate_amendment_sha256,
            "candidate_design_sha256": self.candidate_design_sha256,
            "candidate_id": self.candidate_id,
            "control_fit_contract_sha256": self.control_fit_contract_sha256,
            "feature_names_sha256": self.feature_names_sha256,
            "feature_schema_version": self.feature_schema_version,
            "fit_dataset_sha256": self.fit_dataset_sha256,
            "fit_role": self.fit_role,
            "model_design_sha256": self.model_design_sha256,
            "phase_models": [model.payload() for model in self.phase_models],
            "resolution_authority_sha256": self.resolution_authority_sha256,
            "schema_version": self.schema_version,
            "trading_authority": self.trading_authority,
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != POLYMARKET_ROUND25_PHASE_ISOTONIC_SCHEMA_VERSION
            or self.candidate_id != "phase-isotonic-market-prior-v1"
            or self.fit_role != "calibration"
            or self.feature_schema_version
            != POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION
            or self.feature_names_sha256
            != POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256
            or self.model_design_sha256
            != POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
            or self.candidate_design_sha256
            != POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
            or self.candidate_amendment_sha256
            != POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
            or self.control_fit_contract_sha256
            != POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256
            or len(self.phase_models) != POLYMARKET_ROUND25_PHASE_COUNT
            or tuple(model.phase for model in self.phase_models) != tuple(range(4))
            or self.trading_authority is not False
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.fit_dataset_sha256,
                    self.resolution_authority_sha256,
                    self.artifact_sha256,
                )
            )
            or self.artifact_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 isotonic artifact differs")

    def validated(self) -> Round25PhaseIsotonicArtifact:
        self.__post_init__()
        return self


@dataclass(frozen=True, slots=True)
class Round25L2CalibrationScore:
    l2: float
    condition_equal_log_loss: float
    condition_equal_brier_score: float

    def __post_init__(self) -> None:
        if (
            self.l2 not in POLYMARKET_ROUND25_L2_GRID
            or not math.isfinite(self.condition_equal_log_loss)
            or self.condition_equal_log_loss < 0.0
            or not math.isfinite(self.condition_equal_brier_score)
            or not 0.0 <= self.condition_equal_brier_score <= 1.0
        ):
            raise ValueError("Round 25 L2 calibration score is invalid")

    def payload(self) -> dict[str, float]:
        return {
            "condition_equal_brier_score": self.condition_equal_brier_score,
            "condition_equal_log_loss": self.condition_equal_log_loss,
            "l2": self.l2,
        }


@dataclass(frozen=True, slots=True)
class Round25LogisticResidualArtifact:
    train_dataset_sha256: str
    calibration_dataset_sha256: str
    train_resolution_authority_sha256: str
    calibration_resolution_authority_sha256: str
    center: tuple[float, ...]
    scale: tuple[float, ...]
    selected_l2: float
    intercept: float
    coefficients: tuple[float, ...]
    calibration_scores: tuple[Round25L2CalibrationScore, ...]
    artifact_sha256: str
    schema_version: str = POLYMARKET_ROUND25_LOGISTIC_RESIDUAL_SCHEMA_VERSION
    candidate_id: str = "l2-logistic-residual-v1"
    fit_role: str = "train"
    selection_role: str = "calibration"
    residual_logit_bound: float = POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
    feature_schema_version: str = POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION
    feature_names_sha256: str = POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256
    model_design_sha256: str = POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
    candidate_design_sha256: str = POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
    candidate_amendment_sha256: str = (
        POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
    )
    control_fit_contract_sha256: str = (
        POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256
    )
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "calibration_dataset_sha256": self.calibration_dataset_sha256,
            "calibration_resolution_authority_sha256": (
                self.calibration_resolution_authority_sha256
            ),
            "calibration_scores": [score.payload() for score in self.calibration_scores],
            "candidate_amendment_sha256": self.candidate_amendment_sha256,
            "candidate_design_sha256": self.candidate_design_sha256,
            "candidate_id": self.candidate_id,
            "center": list(self.center),
            "coefficients": list(self.coefficients),
            "control_fit_contract_sha256": self.control_fit_contract_sha256,
            "feature_names_sha256": self.feature_names_sha256,
            "feature_schema_version": self.feature_schema_version,
            "fit_role": self.fit_role,
            "intercept": self.intercept,
            "model_design_sha256": self.model_design_sha256,
            "residual_logit_bound": self.residual_logit_bound,
            "scale": list(self.scale),
            "schema_version": self.schema_version,
            "selected_l2": self.selected_l2,
            "selection_role": self.selection_role,
            "trading_authority": self.trading_authority,
            "train_dataset_sha256": self.train_dataset_sha256,
            "train_resolution_authority_sha256": (
                self.train_resolution_authority_sha256
            ),
        }

    def __post_init__(self) -> None:
        width = len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
        finite_vectors = (*self.center, *self.scale, self.intercept, *self.coefficients)
        if (
            self.schema_version
            != POLYMARKET_ROUND25_LOGISTIC_RESIDUAL_SCHEMA_VERSION
            or self.candidate_id != "l2-logistic-residual-v1"
            or self.fit_role != "train"
            or self.selection_role != "calibration"
            or self.residual_logit_bound != POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
            or self.feature_schema_version
            != POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION
            or self.feature_names_sha256
            != POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256
            or self.model_design_sha256
            != POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
            or self.candidate_design_sha256
            != POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
            or self.candidate_amendment_sha256
            != POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
            or self.control_fit_contract_sha256
            != POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256
            or len(self.center) != width
            or len(self.scale) != width
            or len(self.coefficients) != width
            or any(not math.isfinite(value) for value in finite_vectors)
            or any(value <= 0.0 for value in self.scale)
            or any(
                self.center[index] != 0.0 or self.scale[index] != 1.0
                for index in _BINARY_AVAILABILITY_INDICES
            )
            or tuple(score.l2 for score in self.calibration_scores)
            != POLYMARKET_ROUND25_L2_GRID
            or self.selected_l2
            != min(
                enumerate(self.calibration_scores),
                key=lambda item: (
                    item[1].condition_equal_log_loss,
                    item[1].condition_equal_brier_score,
                    item[0],
                ),
            )[1].l2
            or self.trading_authority is not False
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.train_dataset_sha256,
                    self.calibration_dataset_sha256,
                    self.train_resolution_authority_sha256,
                    self.calibration_resolution_authority_sha256,
                    self.artifact_sha256,
                )
            )
            or self.artifact_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 logistic residual artifact differs")

    def validated(self) -> Round25LogisticResidualArtifact:
        self.__post_init__()
        return self


@dataclass(frozen=True, slots=True)
class Round25DevelopmentMatrix:
    matrix: np.ndarray
    labels: np.ndarray
    prior: np.ndarray
    weights: np.ndarray
    phases: np.ndarray
    condition_ids: tuple[str, ...]
    event_start_ms: np.ndarray


def round25_dataset_matrix(
    dataset: Round25DevelopmentDataset,
) -> Round25DevelopmentMatrix:
    if not isinstance(dataset, Round25DevelopmentDataset):
        raise TypeError("Round 25 dataset type differs")
    dataset.__post_init__()
    selected = require_round25_dataset_minimum(dataset)
    samples = selected.samples
    matrix = np.asarray([sample.feature_values for sample in samples], dtype=np.float64)
    labels = np.asarray([sample.target_up for sample in samples], dtype=np.float64)
    prior = np.asarray(
        [sample.market_prior_probability for sample in samples],
        dtype=np.float64,
    )
    weights = np.asarray([sample.endpoint_weight for sample in samples], dtype=np.float64)
    phases = np.asarray(
        [
            _phase_index(sample.event_start_ms, sample.decision_time_ms)
            for sample in samples
        ],
        dtype=np.int8,
    )
    event_start = np.asarray([sample.event_start_ms for sample in samples], dtype=np.int64)
    if (
        matrix.shape != (len(samples), len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES))
        or labels.shape != prior.shape
        or weights.shape != prior.shape
        or phases.shape != prior.shape
        or event_start.shape != prior.shape
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(prior))
        or not np.all((prior > 0.0) & (prior < 1.0))
        or not np.all(weights == 1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION)
    ):
        raise ValueError("Round 25 dataset matrix differs")
    return Round25DevelopmentMatrix(
        matrix=matrix,
        labels=labels,
        prior=prior,
        weights=weights,
        phases=phases,
        condition_ids=tuple(sample.condition_id for sample in samples),
        event_start_ms=event_start,
    )


def _weighted_isotonic_thresholds(
    values: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (
        x.ndim != 1
        or x.shape != y.shape
        or x.shape != w.shape
        or len(x) < 2
        or not np.all(np.isfinite(x))
        or not np.all((x > 0.0) & (x < 1.0))
        or not np.all((y == 0.0) | (y == 1.0))
        or len(np.unique(y)) != 2
        or not np.all(np.isfinite(w))
        or not np.all(w > 0.0)
    ):
        raise ValueError("Round 25 isotonic fit population is invalid")

    order = np.argsort(x, kind="stable")
    ordered_x = x[order]
    ordered_y = y[order]
    ordered_w = w[order]
    unique_x, first_indices = np.unique(ordered_x, return_index=True)
    grouped_weight = np.add.reduceat(ordered_w, first_indices)
    grouped_positive = np.add.reduceat(ordered_w * ordered_y, first_indices)

    blocks: list[list[float | int]] = []
    for index, (weight, positive) in enumerate(
        zip(grouped_weight, grouped_positive, strict=True)
    ):
        blocks.append([index, index, float(weight), float(positive)])
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = float(left[3]) / float(left[2])
            right_mean = float(right[3]) / float(right[2])
            if left_mean <= right_mean:
                break
            blocks[-2:] = [[
                int(left[0]),
                int(right[1]),
                float(left[2]) + float(right[2]),
                float(left[3]) + float(right[3]),
            ]]

    thresholds_x: list[float] = []
    thresholds_y: list[float] = []
    for start, end, weight, positive in blocks:
        fitted = float(positive) / float(weight)
        left_x = float(unique_x[int(start)])
        right_x = float(unique_x[int(end)])
        thresholds_x.append(left_x)
        thresholds_y.append(fitted)
        if right_x != left_x:
            thresholds_x.append(right_x)
            thresholds_y.append(fitted)
    return tuple(thresholds_x), tuple(thresholds_y)


def _apply_isotonic(
    values: np.ndarray,
    x_thresholds: Sequence[float],
    y_thresholds: Sequence[float],
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    thresholds_x = np.asarray(x_thresholds, dtype=np.float64)
    thresholds_y = np.asarray(y_thresholds, dtype=np.float64)
    if (
        not np.all(np.isfinite(x))
        or not np.all((x > 0.0) & (x < 1.0))
        or thresholds_x.ndim != 1
        or thresholds_x.shape != thresholds_y.shape
        or not len(thresholds_x)
    ):
        raise ValueError("Round 25 isotonic prediction input is invalid")
    return np.interp(
        x,
        thresholds_x,
        thresholds_y,
        left=float(thresholds_y[0]),
        right=float(thresholds_y[-1]),
    )


def fit_round25_phase_isotonic(
    calibration: Round25DevelopmentDataset,
) -> Round25PhaseIsotonicArtifact:
    if not isinstance(calibration, Round25DevelopmentDataset):
        raise TypeError("Round 25 calibration dataset type differs")
    if calibration.role != "calibration":
        raise ValueError("Round 25 isotonic fit requires the calibration role")
    values = round25_dataset_matrix(calibration)
    phase_models: list[Round25IsotonicPhaseModel] = []
    for phase in range(POLYMARKET_ROUND25_PHASE_COUNT):
        selected = values.phases == phase
        x_thresholds, y_thresholds = _weighted_isotonic_thresholds(
            values.prior[selected],
            values.labels[selected],
            values.weights[selected],
        )
        phase_models.append(Round25IsotonicPhaseModel(
            phase=phase,
            x_thresholds=x_thresholds,
            y_thresholds=y_thresholds,
        ))
    payload = {
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "candidate_id": "phase-isotonic-market-prior-v1",
        "control_fit_contract_sha256": POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
        "feature_names_sha256": POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "fit_dataset_sha256": calibration.dataset_sha256,
        "fit_role": "calibration",
        "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
        "phase_models": [model.payload() for model in phase_models],
        "resolution_authority_sha256": calibration.resolution_authority_sha256,
        "schema_version": POLYMARKET_ROUND25_PHASE_ISOTONIC_SCHEMA_VERSION,
        "trading_authority": False,
    }
    return Round25PhaseIsotonicArtifact(
        fit_dataset_sha256=calibration.dataset_sha256,
        resolution_authority_sha256=calibration.resolution_authority_sha256,
        phase_models=tuple(phase_models),
        artifact_sha256=_canonical_sha256(payload),
    )


def predict_round25_phase_isotonic_probability(
    artifact: Round25PhaseIsotonicArtifact,
    *,
    event_start_ms: int,
    decision_time_ms: int,
    market_prior_probability: float,
) -> float:
    if not isinstance(artifact, Round25PhaseIsotonicArtifact):
        raise TypeError("Round 25 isotonic artifact type differs")
    artifact.validated()
    if (
        isinstance(market_prior_probability, bool)
        or not isinstance(market_prior_probability, (int, float))
        or not math.isfinite(market_prior_probability)
        or not 0.0 < market_prior_probability < 1.0
    ):
        raise ValueError("Round 25 market prior is invalid")
    phase = _phase_index(event_start_ms, decision_time_ms)
    selected = artifact.phase_models[phase]
    prediction = _apply_isotonic(
        np.asarray([market_prior_probability], dtype=np.float64),
        selected.x_thresholds,
        selected.y_thresholds,
    )
    return float(prediction[0])


def fit_round25_feature_transform(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    width = len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    if (
        values.ndim != 2
        or values.shape[1] != width
        or not len(values)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Round 25 transform population is invalid")
    center = np.quantile(values, 0.5, axis=0, method="linear")
    lower = np.quantile(values, 0.25, axis=0, method="linear")
    upper = np.quantile(values, 0.75, axis=0, method="linear")
    scale = upper - lower
    numerical_zero = (
        16.0
        * np.finfo(np.float64).eps
        * np.maximum(np.abs(center), 1.0)
    )
    scale[scale <= numerical_zero] = 1.0
    for index in _BINARY_AVAILABILITY_INDICES:
        if not np.all((values[:, index] == 0.0) | (values[:, index] == 1.0)):
            raise ValueError("Round 25 availability feature is not binary")
        center[index] = 0.0
        scale[index] = 1.0
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)) or np.any(
        scale <= 0.0
    ):
        raise ValueError("Round 25 feature transform is invalid")
    return center, scale


def transform_round25_features(
    matrix: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    selected_center = np.asarray(center, dtype=np.float64)
    selected_scale = np.asarray(scale, dtype=np.float64)
    if (
        values.ndim != 2
        or selected_center.shape != (values.shape[1],)
        or selected_scale.shape != selected_center.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(selected_center))
        or not np.all(np.isfinite(selected_scale))
        or np.any(selected_scale <= 0.0)
    ):
        raise ValueError("Round 25 feature transform is invalid")
    normalized = (values - selected_center) / selected_scale
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Round 25 normalized features are nonfinite")
    return normalized


def round25_logit(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if not np.all(np.isfinite(values)) or not np.all(
        (values > 0.0) & (values < 1.0)
    ):
        raise ValueError("Round 25 probability is invalid")
    return np.log(values) - np.log1p(-values)


def round25_bounded_residual_linear(
    normalized: np.ndarray,
    prior: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    if (
        normalized.ndim != 2
        or coefficients.shape != (normalized.shape[1],)
        or prior.shape != (normalized.shape[0],)
        or not math.isfinite(intercept)
        or not np.all(np.isfinite(normalized))
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError("Round 25 logistic prediction input is invalid")
    raw_residual = float(intercept) + normalized @ coefficients
    return round25_apply_bounded_residual(prior, raw_residual)


def round25_apply_bounded_residual(
    prior: np.ndarray,
    raw_residual: np.ndarray,
) -> np.ndarray:
    baseline = np.asarray(prior, dtype=np.float64)
    residual = np.asarray(raw_residual, dtype=np.float64)
    if (
        baseline.ndim != 1
        or residual.shape != baseline.shape
        or not np.all(np.isfinite(residual))
    ):
        raise ValueError("Round 25 residual prediction input is invalid")
    bounded_residual = POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND * np.tanh(
        residual / POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
    )
    return round25_logit(baseline) + bounded_residual


def _fit_bounded_logistic(
    normalized: np.ndarray,
    labels: np.ndarray,
    prior: np.ndarray,
    weights: np.ndarray,
    *,
    l2: float,
) -> tuple[float, np.ndarray]:
    matrix = np.asarray(normalized, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    baseline = np.asarray(prior, dtype=np.float64)
    sample_weight = np.asarray(weights, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != target.shape[0]
        or target.shape != baseline.shape
        or target.shape != sample_weight.shape
        or not len(target)
        or l2 not in POLYMARKET_ROUND25_L2_GRID
        or not np.all(np.isfinite(matrix))
        or not np.all((target == 0.0) | (target == 1.0))
        or len(np.unique(target)) != 2
        or not np.all(np.isfinite(sample_weight))
        or not np.all(sample_weight > 0.0)
    ):
        raise ValueError("Round 25 logistic fit population is invalid")
    offset = round25_logit(baseline)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = float(parameters[0])
        coefficients = parameters[1:]
        raw_residual = intercept + matrix @ coefficients
        scaled = raw_residual / POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
        bounded_residual = POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND * np.tanh(
            scaled
        )
        linear = offset + bounded_residual
        probability = expit(linear)
        weighted_residual = sample_weight * (probability - target)
        derivative = 1.0 - np.tanh(scaled) ** 2
        chain = weighted_residual * derivative
        loss = float(
            np.sum(sample_weight * (np.logaddexp(0.0, linear) - target * linear))
            + 0.5 * l2 * float(coefficients @ coefficients)
        )
        gradient = np.concatenate((
            np.asarray([np.sum(chain)], dtype=np.float64),
            np.asarray(matrix.T @ chain, dtype=np.float64) + l2 * coefficients,
        ))
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(matrix.shape[1] + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 512, "ftol": 1e-11, "gtol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"Round 25 residual logistic fit failed: {str(result.message).strip()}"
        )
    return float(result.x[0]), np.asarray(result.x[1:], dtype=np.float64)


def round25_condition_equal_scores(
    labels: np.ndarray,
    linear: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    target = np.asarray(labels, dtype=np.float64)
    logits = np.asarray(linear, dtype=np.float64)
    sample_weight = np.asarray(weights, dtype=np.float64)
    total_weight = float(np.sum(sample_weight))
    if (
        target.shape != logits.shape
        or target.shape != sample_weight.shape
        or not total_weight > 0.0
        or not np.all(np.isfinite(target))
        or not np.all((target == 0.0) | (target == 1.0))
        or not np.all(np.isfinite(logits))
        or not np.all(np.isfinite(sample_weight))
        or not np.all(sample_weight > 0.0)
    ):
        raise ValueError("Round 25 score population is invalid")
    probability = expit(logits)
    log_loss = float(
        np.sum(
            sample_weight * (np.logaddexp(0.0, logits) - target * logits)
        )
        / total_weight
    )
    brier = float(np.sum(sample_weight * (probability - target) ** 2) / total_weight)
    return log_loss, brier


def validate_round25_fit_pair(
    train: Round25DevelopmentDataset,
    calibration: Round25DevelopmentDataset,
) -> tuple[Round25DevelopmentMatrix, Round25DevelopmentMatrix]:
    if not isinstance(train, Round25DevelopmentDataset) or not isinstance(
        calibration, Round25DevelopmentDataset
    ):
        raise TypeError("Round 25 fit dataset type differs")
    if train.role != "train" or calibration.role != "calibration":
        raise ValueError("Round 25 logistic fit roles differ")
    train_values = round25_dataset_matrix(train)
    calibration_values = round25_dataset_matrix(calibration)
    if (
        train.resolution_authority_sha256
        != calibration.resolution_authority_sha256
        or set(train_values.condition_ids) & set(calibration_values.condition_ids)
        or int(np.max(train_values.event_start_ms))
        >= int(np.min(calibration_values.event_start_ms))
    ):
        raise ValueError("Round 25 fit populations overlap or differ")
    return train_values, calibration_values


def fit_round25_logistic_residual(
    *,
    train: Round25DevelopmentDataset,
    calibration: Round25DevelopmentDataset,
) -> Round25LogisticResidualArtifact:
    train_values, calibration_values = validate_round25_fit_pair(train, calibration)
    center, scale = fit_round25_feature_transform(train_values.matrix)
    normalized_train = transform_round25_features(train_values.matrix, center, scale)
    normalized_calibration = transform_round25_features(
        calibration_values.matrix,
        center,
        scale,
    )
    candidates: list[tuple[Round25L2CalibrationScore, float, np.ndarray]] = []
    for l2 in POLYMARKET_ROUND25_L2_GRID:
        intercept, coefficients = _fit_bounded_logistic(
            normalized_train,
            train_values.labels,
            train_values.prior,
            train_values.weights,
            l2=l2,
        )
        linear = round25_bounded_residual_linear(
            normalized_calibration,
            calibration_values.prior,
            intercept,
            coefficients,
        )
        log_loss, brier = round25_condition_equal_scores(
            calibration_values.labels,
            linear,
            calibration_values.weights,
        )
        candidates.append((
            Round25L2CalibrationScore(
                l2=l2,
                condition_equal_log_loss=log_loss,
                condition_equal_brier_score=brier,
            ),
            intercept,
            coefficients,
        ))
    selected_index, selected = min(
        enumerate(candidates),
        key=lambda item: (
            item[1][0].condition_equal_log_loss,
            item[1][0].condition_equal_brier_score,
            item[0],
        ),
    )
    del selected_index
    score, intercept, coefficients = selected
    scores = tuple(item[0] for item in candidates)
    payload = {
        "calibration_dataset_sha256": calibration.dataset_sha256,
        "calibration_resolution_authority_sha256": (
            calibration.resolution_authority_sha256
        ),
        "calibration_scores": [item.payload() for item in scores],
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "candidate_id": "l2-logistic-residual-v1",
        "center": center.tolist(),
        "coefficients": coefficients.tolist(),
        "control_fit_contract_sha256": POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
        "feature_names_sha256": POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "fit_role": "train",
        "intercept": intercept,
        "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
        "residual_logit_bound": POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
        "scale": scale.tolist(),
        "schema_version": POLYMARKET_ROUND25_LOGISTIC_RESIDUAL_SCHEMA_VERSION,
        "selected_l2": score.l2,
        "selection_role": "calibration",
        "trading_authority": False,
        "train_dataset_sha256": train.dataset_sha256,
        "train_resolution_authority_sha256": train.resolution_authority_sha256,
    }
    return Round25LogisticResidualArtifact(
        train_dataset_sha256=train.dataset_sha256,
        calibration_dataset_sha256=calibration.dataset_sha256,
        train_resolution_authority_sha256=train.resolution_authority_sha256,
        calibration_resolution_authority_sha256=(
            calibration.resolution_authority_sha256
        ),
        center=tuple(float(value) for value in center),
        scale=tuple(float(value) for value in scale),
        selected_l2=score.l2,
        intercept=intercept,
        coefficients=tuple(float(value) for value in coefficients),
        calibration_scores=scores,
        artifact_sha256=_canonical_sha256(payload),
    )


def predict_round25_logistic_residual_probability(
    artifact: Round25LogisticResidualArtifact,
    *,
    feature_values: Sequence[float],
    market_prior_probability: float,
) -> float:
    if not isinstance(artifact, Round25LogisticResidualArtifact):
        raise TypeError("Round 25 logistic artifact type differs")
    artifact.validated()
    values = _finite_tuple(
        feature_values,
        expected=len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES),
    )
    if (
        isinstance(market_prior_probability, bool)
        or not isinstance(market_prior_probability, (int, float))
        or not math.isfinite(market_prior_probability)
        or not 0.0 < market_prior_probability < 1.0
    ):
        raise ValueError("Round 25 market prior is invalid")
    normalized = transform_round25_features(
        np.asarray([values], dtype=np.float64),
        np.asarray(artifact.center, dtype=np.float64),
        np.asarray(artifact.scale, dtype=np.float64),
    )
    linear = round25_bounded_residual_linear(
        normalized,
        np.asarray([market_prior_probability], dtype=np.float64),
        artifact.intercept,
        np.asarray(artifact.coefficients, dtype=np.float64),
    )
    return float(expit(linear[0]))


def round25_market_prior_predictions(
    dataset: Round25DevelopmentDataset,
) -> tuple[float, ...]:
    values = round25_dataset_matrix(dataset)
    return tuple(float(value) for value in values.prior)


__all__ = [
    "POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256",
    "POLYMARKET_ROUND25_L2_GRID",
    "POLYMARKET_ROUND25_LOGISTIC_RESIDUAL_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_PHASE_ISOTONIC_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND",
    "Round25IsotonicPhaseModel",
    "Round25DevelopmentMatrix",
    "Round25L2CalibrationScore",
    "Round25LogisticResidualArtifact",
    "Round25PhaseIsotonicArtifact",
    "fit_round25_logistic_residual",
    "fit_round25_feature_transform",
    "fit_round25_phase_isotonic",
    "predict_round25_logistic_residual_probability",
    "predict_round25_phase_isotonic_probability",
    "round25_apply_bounded_residual",
    "round25_bounded_residual_linear",
    "round25_condition_equal_scores",
    "round25_dataset_matrix",
    "round25_logit",
    "round25_market_prior_predictions",
    "transform_round25_features",
    "validate_round25_fit_pair",
]
