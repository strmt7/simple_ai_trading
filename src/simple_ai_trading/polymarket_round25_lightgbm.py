"""Frozen shallow LightGBM residual candidates for Round 25."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Sequence

import lightgbm as lgb
import numpy as np
from scipy.special import expit

from .lightgbm_backend import (
    SUPPORTED_LIGHTGBM_BACKEND_KINDS,
    lightgbm_backend_parameters,
)
from .polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
)
from .polymarket_round25_controls import (
    POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
    fit_round25_feature_transform,
    round25_apply_bounded_residual,
    round25_condition_equal_scores,
    transform_round25_features,
    validate_round25_fit_pair,
)
from .polymarket_round25_dataset import Round25DevelopmentDataset
from .polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
)
from .polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


POLYMARKET_ROUND25_LIGHTGBM_FIT_CONTRACT_SHA256 = (
    "f82bbad224f48b97a4e90c95af1f5180228a1e45590af03a75024acac29f64ec"
)
POLYMARKET_ROUND25_LIGHTGBM_ARTIFACT_SCHEMA_VERSION = (
    "polymarket-round25-lightgbm-residual-artifact-v1"
)
POLYMARKET_ROUND25_LIGHTGBM_SEED = 25_025
POLYMARKET_ROUND25_LIGHTGBM_SERIALIZATION_TOLERANCE = 1e-10
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAXIMUM_MODEL_STRING_BYTES = 16 * 1024 * 1024
_BINARY_AVAILABILITY_INDICES = tuple(
    index
    for index, name in enumerate(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    if name.endswith("_available")
)


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


@dataclass(frozen=True, slots=True)
class Round25LightGBMConfig:
    candidate_id: str
    max_depth: int
    num_leaves: int
    learning_rate: float
    maximum_trees: int
    early_stopping_rounds: int
    minimum_conditions_per_leaf: int
    minimum_rows_per_leaf: int

    def __post_init__(self) -> None:
        expected = {
            "lightgbm-residual-depth3-v1": (3, 8, 0.03, 500, 50, 25, 400),
            "lightgbm-residual-depth5-v1": (5, 16, 0.02, 750, 75, 40, 640),
        }.get(self.candidate_id)
        observed = (
            self.max_depth,
            self.num_leaves,
            self.learning_rate,
            self.maximum_trees,
            self.early_stopping_rounds,
            self.minimum_conditions_per_leaf,
            self.minimum_rows_per_leaf,
        )
        if expected is None or observed != expected:
            raise ValueError("Round 25 LightGBM configuration differs")

    def payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "early_stopping_rounds": self.early_stopping_rounds,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "maximum_trees": self.maximum_trees,
            "minimum_conditions_per_leaf": self.minimum_conditions_per_leaf,
            "minimum_rows_per_leaf": self.minimum_rows_per_leaf,
            "num_leaves": self.num_leaves,
        }


POLYMARKET_ROUND25_LIGHTGBM_CONFIGS = (
    Round25LightGBMConfig(
        candidate_id="lightgbm-residual-depth3-v1",
        max_depth=3,
        num_leaves=8,
        learning_rate=0.03,
        maximum_trees=500,
        early_stopping_rounds=50,
        minimum_conditions_per_leaf=25,
        minimum_rows_per_leaf=400,
    ),
    Round25LightGBMConfig(
        candidate_id="lightgbm-residual-depth5-v1",
        max_depth=5,
        num_leaves=16,
        learning_rate=0.02,
        maximum_trees=750,
        early_stopping_rounds=75,
        minimum_conditions_per_leaf=40,
        minimum_rows_per_leaf=640,
    ),
)


def round25_lightgbm_config(candidate_id: str) -> Round25LightGBMConfig:
    matches = [
        config
        for config in POLYMARKET_ROUND25_LIGHTGBM_CONFIGS
        if config.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError("Round 25 LightGBM candidate is not frozen")
    return matches[0]


def _validate_binary_availability(matrix: np.ndarray) -> None:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(
        POLYMARKET_ROUND25_JOINT_FEATURE_NAMES
    ):
        raise ValueError("Round 25 LightGBM feature matrix shape differs")
    if any(
        not np.all((values[:, index] == 0.0) | (values[:, index] == 1.0))
        for index in _BINARY_AVAILABILITY_INDICES
    ):
        raise ValueError("Round 25 LightGBM availability feature is not binary")


@dataclass(frozen=True, slots=True)
class _Round25LightGBMFitResult:
    best_iteration: int
    model_string: str
    lightgbm_version: str
    backend_kind: str
    backend_device: str
    calibration_condition_equal_log_loss: float
    calibration_condition_equal_brier_score: float


def _validate_fit_arrays(
    matrix: np.ndarray,
    labels: np.ndarray,
    prior: np.ndarray,
    weights: np.ndarray,
    *,
    name: str,
) -> None:
    width = len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != width
        or matrix.shape[0] != labels.shape[0]
        or labels.shape != prior.shape
        or labels.shape != weights.shape
        or not len(labels)
        or not np.all(np.isfinite(matrix))
        or not np.all((labels == 0.0) | (labels == 1.0))
        or len(np.unique(labels)) != 2
        or not np.all(np.isfinite(prior))
        or not np.all((prior > 0.0) & (prior < 1.0))
        or not np.all(np.isfinite(weights))
        or not np.all(weights > 0.0)
    ):
        raise ValueError(f"Round 25 LightGBM {name} population is invalid")


def _train_round25_lightgbm(
    *,
    config: Round25LightGBMConfig,
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_prior: np.ndarray,
    train_weights: np.ndarray,
    calibration_matrix: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_prior: np.ndarray,
    calibration_weights: np.ndarray,
    compute_backend: str,
) -> _Round25LightGBMFitResult:
    if not isinstance(config, Round25LightGBMConfig):
        raise TypeError("Round 25 LightGBM configuration type differs")
    config.__post_init__()
    train_values = np.asarray(train_matrix, dtype=np.float32, order="C")
    calibration_values = np.asarray(
        calibration_matrix,
        dtype=np.float32,
        order="C",
    )
    train_target = np.asarray(train_labels, dtype=np.float64)
    calibration_target = np.asarray(calibration_labels, dtype=np.float64)
    train_baseline = np.asarray(train_prior, dtype=np.float64)
    calibration_baseline = np.asarray(calibration_prior, dtype=np.float64)
    train_weight = np.asarray(train_weights, dtype=np.float64)
    calibration_weight = np.asarray(calibration_weights, dtype=np.float64)
    _validate_fit_arrays(
        train_values,
        train_target,
        train_baseline,
        train_weight,
        name="training",
    )
    _validate_fit_arrays(
        calibration_values,
        calibration_target,
        calibration_baseline,
        calibration_weight,
        name="calibration",
    )
    if len(train_target) < config.minimum_rows_per_leaf * 2:
        raise ValueError("Round 25 LightGBM training rows cannot form two leaves")

    backend, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        POLYMARKET_ROUND25_LIGHTGBM_SEED,
        reproducible=True,
        pin_opencl_device=True,
    )
    parameters: dict[str, object] = {
        **backend,
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": config.learning_rate,
        "max_depth": config.max_depth,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.minimum_rows_per_leaf,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "min_gain_to_split": 0.0,
        "feature_pre_filter": False,
        "verbosity": -1,
    }
    feature_names = list(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    training = lgb.Dataset(
        train_values,
        label=train_target,
        weight=train_weight,
        init_score=np.log(train_baseline) - np.log1p(-train_baseline),
        feature_name=feature_names,
        free_raw_data=False,
    )
    calibration = lgb.Dataset(
        calibration_values,
        label=calibration_target,
        weight=calibration_weight,
        init_score=(
            np.log(calibration_baseline) - np.log1p(-calibration_baseline)
        ),
        reference=training,
        feature_name=feature_names,
        free_raw_data=False,
    )
    booster = lgb.train(
        parameters,
        training,
        num_boost_round=config.maximum_trees,
        valid_sets=[calibration],
        valid_names=["calibration"],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best_iteration = max(
        1,
        int(booster.best_iteration or booster.current_iteration()),
    )
    model_string = booster.model_to_string(num_iteration=best_iteration)
    reloaded = lgb.Booster(model_str=model_string)
    original_raw = np.asarray(
        booster.predict(
            calibration_values,
            num_iteration=best_iteration,
            raw_score=True,
        ),
        dtype=np.float64,
    )
    restored_raw = np.asarray(
        reloaded.predict(calibration_values, raw_score=True),
        dtype=np.float64,
    )
    reload_error = float(
        np.max(np.abs(original_raw - restored_raw), initial=0.0)
    )
    if reload_error > POLYMARKET_ROUND25_LIGHTGBM_SERIALIZATION_TOLERANCE:
        raise RuntimeError("Round 25 LightGBM serialization identity failed")
    bounded_linear = round25_apply_bounded_residual(
        calibration_baseline,
        restored_raw,
    )
    log_loss, brier = round25_condition_equal_scores(
        calibration_target,
        bounded_linear,
        calibration_weight,
    )
    return _Round25LightGBMFitResult(
        best_iteration=best_iteration,
        model_string=model_string,
        lightgbm_version=str(lgb.__version__),
        backend_kind=backend_kind,
        backend_device=backend_device,
        calibration_condition_equal_log_loss=log_loss,
        calibration_condition_equal_brier_score=brier,
    )


@dataclass(frozen=True, slots=True)
class Round25LightGBMArtifact:
    config: Round25LightGBMConfig
    train_dataset_sha256: str
    calibration_dataset_sha256: str
    train_resolution_authority_sha256: str
    calibration_resolution_authority_sha256: str
    center: tuple[float, ...]
    scale: tuple[float, ...]
    best_iteration: int
    model_string: str
    model_string_sha256: str
    lightgbm_version: str
    backend_kind: str
    backend_device: str
    calibration_condition_equal_log_loss: float
    calibration_condition_equal_brier_score: float
    artifact_sha256: str
    schema_version: str = POLYMARKET_ROUND25_LIGHTGBM_ARTIFACT_SCHEMA_VERSION
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
    lightgbm_fit_contract_sha256: str = (
        POLYMARKET_ROUND25_LIGHTGBM_FIT_CONTRACT_SHA256
    )
    residual_logit_bound: float = POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "backend_device": self.backend_device,
            "backend_kind": self.backend_kind,
            "best_iteration": self.best_iteration,
            "calibration_condition_equal_brier_score": (
                self.calibration_condition_equal_brier_score
            ),
            "calibration_condition_equal_log_loss": (
                self.calibration_condition_equal_log_loss
            ),
            "calibration_dataset_sha256": self.calibration_dataset_sha256,
            "calibration_resolution_authority_sha256": (
                self.calibration_resolution_authority_sha256
            ),
            "candidate_amendment_sha256": self.candidate_amendment_sha256,
            "candidate_design_sha256": self.candidate_design_sha256,
            "center": list(self.center),
            "config": self.config.payload(),
            "control_fit_contract_sha256": self.control_fit_contract_sha256,
            "feature_names_sha256": self.feature_names_sha256,
            "feature_schema_version": self.feature_schema_version,
            "lightgbm_fit_contract_sha256": self.lightgbm_fit_contract_sha256,
            "lightgbm_version": self.lightgbm_version,
            "model_design_sha256": self.model_design_sha256,
            "model_string": self.model_string,
            "model_string_sha256": self.model_string_sha256,
            "residual_logit_bound": self.residual_logit_bound,
            "scale": list(self.scale),
            "schema_version": self.schema_version,
            "trading_authority": self.trading_authority,
            "train_dataset_sha256": self.train_dataset_sha256,
            "train_resolution_authority_sha256": (
                self.train_resolution_authority_sha256
            ),
        }

    def __post_init__(self) -> None:
        width = len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
        model_bytes = self.model_string.encode("utf-8")
        if (
            not isinstance(self.config, Round25LightGBMConfig)
            or self.schema_version
            != POLYMARKET_ROUND25_LIGHTGBM_ARTIFACT_SCHEMA_VERSION
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
            or self.lightgbm_fit_contract_sha256
            != POLYMARKET_ROUND25_LIGHTGBM_FIT_CONTRACT_SHA256
            or self.residual_logit_bound != POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
            or len(self.center) != width
            or len(self.scale) != width
            or any(not math.isfinite(value) for value in (*self.center, *self.scale))
            or any(value <= 0.0 for value in self.scale)
            or any(
                self.center[index] != 0.0 or self.scale[index] != 1.0
                for index in _BINARY_AVAILABILITY_INDICES
            )
            or not 1 <= self.best_iteration <= self.config.maximum_trees
            or not 32 <= len(model_bytes) <= _MAXIMUM_MODEL_STRING_BYTES
            or self.model_string_sha256
            != hashlib.sha256(model_bytes).hexdigest()
            or not self.lightgbm_version.strip()
            or len(self.lightgbm_version) > 128
            or self.backend_kind not in SUPPORTED_LIGHTGBM_BACKEND_KINDS
            or not self.backend_device.strip()
            or len(self.backend_device) > 500
            or not math.isfinite(self.calibration_condition_equal_log_loss)
            or self.calibration_condition_equal_log_loss < 0.0
            or not math.isfinite(self.calibration_condition_equal_brier_score)
            or not 0.0 <= self.calibration_condition_equal_brier_score <= 1.0
            or self.trading_authority is not False
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.train_dataset_sha256,
                    self.calibration_dataset_sha256,
                    self.train_resolution_authority_sha256,
                    self.calibration_resolution_authority_sha256,
                    self.model_string_sha256,
                    self.artifact_sha256,
                )
            )
            or self.artifact_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 LightGBM artifact differs")
        self.config.__post_init__()

    def validated(self) -> Round25LightGBMArtifact:
        self.__post_init__()
        return self


def _create_round25_lightgbm_artifact(
    *,
    config: Round25LightGBMConfig,
    train_dataset_sha256: str,
    calibration_dataset_sha256: str,
    train_resolution_authority_sha256: str,
    calibration_resolution_authority_sha256: str,
    center: np.ndarray,
    scale: np.ndarray,
    fitted: _Round25LightGBMFitResult,
) -> Round25LightGBMArtifact:
    center_values = tuple(float(value) for value in center)
    scale_values = tuple(float(value) for value in scale)
    model_sha256 = hashlib.sha256(fitted.model_string.encode("utf-8")).hexdigest()
    payload = {
        "backend_device": fitted.backend_device,
        "backend_kind": fitted.backend_kind,
        "best_iteration": fitted.best_iteration,
        "calibration_condition_equal_brier_score": (
            fitted.calibration_condition_equal_brier_score
        ),
        "calibration_condition_equal_log_loss": (
            fitted.calibration_condition_equal_log_loss
        ),
        "calibration_dataset_sha256": calibration_dataset_sha256,
        "calibration_resolution_authority_sha256": (
            calibration_resolution_authority_sha256
        ),
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "center": list(center_values),
        "config": config.payload(),
        "control_fit_contract_sha256": POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
        "feature_names_sha256": POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "lightgbm_fit_contract_sha256": (
            POLYMARKET_ROUND25_LIGHTGBM_FIT_CONTRACT_SHA256
        ),
        "lightgbm_version": fitted.lightgbm_version,
        "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
        "model_string": fitted.model_string,
        "model_string_sha256": model_sha256,
        "residual_logit_bound": POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
        "scale": list(scale_values),
        "schema_version": POLYMARKET_ROUND25_LIGHTGBM_ARTIFACT_SCHEMA_VERSION,
        "trading_authority": False,
        "train_dataset_sha256": train_dataset_sha256,
        "train_resolution_authority_sha256": train_resolution_authority_sha256,
    }
    return Round25LightGBMArtifact(
        config=config,
        train_dataset_sha256=train_dataset_sha256,
        calibration_dataset_sha256=calibration_dataset_sha256,
        train_resolution_authority_sha256=train_resolution_authority_sha256,
        calibration_resolution_authority_sha256=(
            calibration_resolution_authority_sha256
        ),
        center=center_values,
        scale=scale_values,
        best_iteration=fitted.best_iteration,
        model_string=fitted.model_string,
        model_string_sha256=model_sha256,
        lightgbm_version=fitted.lightgbm_version,
        backend_kind=fitted.backend_kind,
        backend_device=fitted.backend_device,
        calibration_condition_equal_log_loss=(
            fitted.calibration_condition_equal_log_loss
        ),
        calibration_condition_equal_brier_score=(
            fitted.calibration_condition_equal_brier_score
        ),
        artifact_sha256=_canonical_sha256(payload),
    )


def fit_round25_lightgbm_residual(
    *,
    candidate_id: str,
    train: Round25DevelopmentDataset,
    calibration: Round25DevelopmentDataset,
    compute_backend: str = "auto",
) -> Round25LightGBMArtifact:
    config = round25_lightgbm_config(candidate_id)
    train_values, calibration_values = validate_round25_fit_pair(train, calibration)
    _validate_binary_availability(train_values.matrix)
    _validate_binary_availability(calibration_values.matrix)
    center, scale = fit_round25_feature_transform(train_values.matrix)
    normalized_train = transform_round25_features(train_values.matrix, center, scale)
    normalized_calibration = transform_round25_features(
        calibration_values.matrix,
        center,
        scale,
    )
    fitted = _train_round25_lightgbm(
        config=config,
        train_matrix=normalized_train,
        train_labels=train_values.labels,
        train_prior=train_values.prior,
        train_weights=train_values.weights,
        calibration_matrix=normalized_calibration,
        calibration_labels=calibration_values.labels,
        calibration_prior=calibration_values.prior,
        calibration_weights=calibration_values.weights,
        compute_backend=compute_backend,
    )
    return _create_round25_lightgbm_artifact(
        config=config,
        train_dataset_sha256=train.dataset_sha256,
        calibration_dataset_sha256=calibration.dataset_sha256,
        train_resolution_authority_sha256=train.resolution_authority_sha256,
        calibration_resolution_authority_sha256=(
            calibration.resolution_authority_sha256
        ),
        center=center,
        scale=scale,
        fitted=fitted,
    )


class Round25CompiledLightGBM:
    """Reusable target-free runtime for one validated Round 25 tree artifact."""

    def __init__(self, artifact: Round25LightGBMArtifact) -> None:
        if not isinstance(artifact, Round25LightGBMArtifact):
            raise TypeError("Round 25 LightGBM artifact type differs")
        self._artifact = artifact.validated()
        self._center = np.asarray(artifact.center, dtype=np.float64)
        self._scale = np.asarray(artifact.scale, dtype=np.float64)
        self._booster = lgb.Booster(model_str=artifact.model_string)

    @property
    def artifact_sha256(self) -> str:
        return self._artifact.artifact_sha256

    def predict_probabilities(
        self,
        feature_values: Sequence[Sequence[float]],
        market_prior_probability: Sequence[float],
    ) -> tuple[float, ...]:
        matrix = np.asarray(feature_values, dtype=np.float64)
        prior = np.asarray(market_prior_probability, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
            or prior.shape != (matrix.shape[0],)
            or not len(matrix)
            or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(prior))
            or not np.all((prior > 0.0) & (prior < 1.0))
        ):
            raise ValueError("Round 25 LightGBM inference population is invalid")
        _validate_binary_availability(matrix)
        normalized = transform_round25_features(matrix, self._center, self._scale)
        raw_residual = np.asarray(
            self._booster.predict(normalized.astype(np.float32), raw_score=True),
            dtype=np.float64,
        )
        prediction = expit(round25_apply_bounded_residual(prior, raw_residual))
        if prediction.shape != prior.shape or not np.all(np.isfinite(prediction)):
            raise RuntimeError("Round 25 LightGBM prediction is invalid")
        return tuple(float(value) for value in prediction)

    def predict_probability(
        self,
        feature_values: Sequence[float],
        market_prior_probability: float,
    ) -> float:
        return self.predict_probabilities(
            (feature_values,),
            (market_prior_probability,),
        )[0]


__all__ = [
    "POLYMARKET_ROUND25_LIGHTGBM_ARTIFACT_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_LIGHTGBM_CONFIGS",
    "POLYMARKET_ROUND25_LIGHTGBM_FIT_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_LIGHTGBM_SEED",
    "Round25CompiledLightGBM",
    "Round25LightGBMArtifact",
    "Round25LightGBMConfig",
    "fit_round25_lightgbm_residual",
    "round25_lightgbm_config",
]
