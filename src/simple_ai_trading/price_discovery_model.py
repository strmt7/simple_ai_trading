"""Frozen walk-forward LightGBM models for the Round 72 viability screen."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize_scalar

from .lightgbm_backend import lightgbm_backend_parameters
from .price_discovery_dataset import (
    PriceDiscoveryDatasetBundle,
    PriceDiscoverySymbolDataset,
)
from .price_discovery_spec import (
    FEATURE_LAYERS,
    HORIZONS_SECONDS,
    PRIMARY_ENTRY_DELAY_SECONDS,
    PRIMARY_LOSS_METRICS,
    ROUND72_IMPLEMENTATION_SCHEMA,
    layer_feature_names,
    load_round72_implementation,
)
from .spot_perpetual_flow import FLOW_SYMBOLS


PRICE_DISCOVERY_MODEL_RUN_SCHEMA = "round-072-price-discovery-model-run-v1"
PRICE_DISCOVERY_HEADS = tuple(PRIMARY_LOSS_METRICS)
PRIMARY_FEATURE_LAYERS = ("perpetual_only", "spot_perpetual")
ROUND72_SEED = 20260722
_PROBABILITY_EPSILON = 1e-6
_ProgressCallback = Callable[[str, Mapping[str, object]], None]


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


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _array_digest(digest, value: np.ndarray) -> None:
    array = np.asarray(value)
    dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    digest.update(dtype.str.encode("ascii"))
    digest.update(int(canonical.size).to_bytes(8, "little", signed=False))
    digest.update(memoryview(canonical).cast("B"))


def _month_ordinal(value: str) -> int:
    text = str(value)
    if len(text) != 7 or text[4] != "-":
        raise ValueError("Round 72 fold month must use YYYY-MM")
    try:
        year = int(text[:4])
        month = int(text[5:])
    except ValueError as exc:
        raise ValueError("Round 72 fold month must use YYYY-MM") from exc
    if not 1 <= month <= 12:
        raise ValueError("Round 72 fold month is outside the calendar")
    return year * 12 + month - 1


def _readonly(value: np.ndarray, dtype) -> np.ndarray:
    output = np.ascontiguousarray(value, dtype=dtype)
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class PriceDiscoveryFold:
    fold: int
    training_start_month: int
    training_end_month: int
    tuning_start_month: int
    tuning_end_month: int
    test_start_month: int
    test_end_month: int

    def validate(self) -> None:
        if (
            not 1 <= self.fold <= 6
            or self.training_start_month != _month_ordinal("2020-10")
            or self.training_end_month + 1 != self.tuning_start_month
            or self.tuning_end_month + 1 != self.test_start_month
            or self.training_end_month < self.training_start_month
            or self.training_end_month - self.training_start_month + 1
            != 24 + (self.fold - 1) * 6
            or self.tuning_end_month - self.tuning_start_month != 5
            or self.test_end_month - self.test_start_month != 5
        ):
            raise ValueError("Round 72 rolling fold is invalid")


@dataclass(frozen=True)
class PriceDiscoveryRoleIndices:
    training: np.ndarray
    tuning: np.ndarray
    test: np.ndarray
    stress_test: np.ndarray

    def validate(self, rows: int) -> None:
        arrays = (self.training, self.tuning, self.test, self.stress_test)
        if any(
            value.ndim != 1
            or value.dtype != np.int64
            or len(value) == 0
            or np.any(np.diff(value) <= 0)
            or value[0] < 0
            or value[-1] >= rows
            for value in arrays
        ):
            raise ValueError("Round 72 role indexes are invalid")
        if self.training[-1] >= self.tuning[0] or self.tuning[-1] >= self.test[0]:
            raise ValueError("Round 72 chronological roles overlap")


@dataclass(frozen=True)
class PriceDiscoveryFoldPrediction:
    symbol: str
    horizon_seconds: int
    feature_layer: str
    head: str
    fold: int
    feature_count: int
    training_rows: int
    tuning_rows: int
    test_rows: int
    stress_test_rows: int
    training_positive_rows: int
    training_negative_rows: int
    training_prevalence: float
    training_mean_target_bps: float
    best_iteration: int
    calibration_value: float
    calibration_retained: bool
    tuning_loss_before_calibration: float
    tuning_loss_after_calibration: float
    model_bytes: int
    model_sha256: str
    reload_max_absolute_prediction_difference: float
    backend_requested: str
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    anchor_second_ms: np.ndarray
    utc_day: np.ndarray
    primary_target: np.ndarray
    primary_prediction: np.ndarray
    stress_anchor_second_ms: np.ndarray
    stress_utc_day: np.ndarray
    stress_target: np.ndarray
    stress_prediction: np.ndarray
    prediction_sha256: str
    profitability_claim: bool = False
    execution_or_fill_claim: bool = False
    trading_authority: bool = False

    def validate(self) -> None:
        primary_shape = (self.test_rows,)
        stress_shape = (self.stress_test_rows,)
        arrays = (
            (self.anchor_second_ms, primary_shape, np.int64),
            (self.utc_day, primary_shape, np.int32),
            (self.primary_target, primary_shape, np.float64),
            (self.primary_prediction, primary_shape, np.float64),
            (self.stress_anchor_second_ms, stress_shape, np.int64),
            (self.stress_utc_day, stress_shape, np.int32),
            (self.stress_target, stress_shape, np.float64),
            (self.stress_prediction, stress_shape, np.float64),
        )
        expected_width = len(layer_feature_names(self.feature_layer))
        if (
            self.symbol not in FLOW_SYMBOLS
            or self.horizon_seconds not in HORIZONS_SECONDS
            or self.feature_layer not in FEATURE_LAYERS
            or self.head not in PRICE_DISCOVERY_HEADS
            or not 1 <= self.fold <= 6
            or self.feature_count != expected_width
            or min(
                self.training_rows,
                self.tuning_rows,
                self.test_rows,
                self.stress_test_rows,
                self.best_iteration,
                self.model_bytes,
            )
            <= 0
            or self.training_positive_rows + self.training_negative_rows
            != self.training_rows
            or min(self.training_positive_rows, self.training_negative_rows) <= 0
            or not 0.0 < self.training_prevalence < 1.0
            or not math.isfinite(self.training_mean_target_bps)
            or not (
                0.25 <= self.calibration_value <= 4.0
                if self.head == "binary_direction"
                else 0.0 <= self.calibration_value <= 4.0
            )
            or not isinstance(self.calibration_retained, bool)
            or not math.isfinite(self.tuning_loss_before_calibration)
            or not math.isfinite(self.tuning_loss_after_calibration)
            or self.tuning_loss_after_calibration
            > self.tuning_loss_before_calibration + 1e-12
            or not _is_sha256(self.model_sha256)
            or not 0.0 <= self.reload_max_absolute_prediction_difference <= 1e-12
            or not self.backend_requested
            or self.backend_kind not in {"cpu", "opencl", "cuda"}
            or not self.backend_device
            or not self.lightgbm_version
            or any(
                value.shape != shape
                or value.dtype != dtype
                or value.flags.writeable
                or not np.all(np.isfinite(value))
                for value, shape, dtype in arrays
            )
            or np.any(np.diff(self.anchor_second_ms) <= 0)
            or np.any(np.diff(self.stress_anchor_second_ms) <= 0)
            or not np.array_equal(self.utc_day, self.anchor_second_ms // 86_400_000)
            or not np.array_equal(
                self.stress_utc_day,
                self.stress_anchor_second_ms // 86_400_000,
            )
            or any(
                (
                    self.profitability_claim,
                    self.execution_or_fill_claim,
                    self.trading_authority,
                )
            )
            or self.prediction_sha256 != _prediction_sha256(self)
        ):
            raise ValueError("Round 72 fold prediction is invalid")
        if self.head == "binary_direction" and (
            np.any((self.primary_target != 0.0) & (self.primary_target != 1.0))
            or np.any((self.stress_target != 0.0) & (self.stress_target != 1.0))
            or np.any(
                (self.primary_prediction < _PROBABILITY_EPSILON)
                | (self.primary_prediction > 1.0 - _PROBABILITY_EPSILON)
            )
            or np.any(
                (self.stress_prediction < _PROBABILITY_EPSILON)
                | (self.stress_prediction > 1.0 - _PROBABILITY_EPSILON)
            )
        ):
            raise ValueError("Round 72 binary prediction values are invalid")


@dataclass(frozen=True)
class PriceDiscoveryPredictionRun:
    schema_version: str
    implementation_sha256: str
    dataset_bundle_sha256: str
    feature_layers: tuple[str, ...]
    backend_requested: str
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    blocks: tuple[PriceDiscoveryFoldPrediction, ...]
    run_sha256: str
    profitability_claim: bool = False
    execution_or_fill_claim: bool = False
    trading_authority: bool = False

    def validate(self) -> None:
        expected = tuple(
            (symbol, horizon, layer, head, fold)
            for symbol in FLOW_SYMBOLS
            for horizon in HORIZONS_SECONDS
            for layer in self.feature_layers
            for head in PRICE_DISCOVERY_HEADS
            for fold in range(1, 7)
        )
        observed = tuple(
            (
                block.symbol,
                block.horizon_seconds,
                block.feature_layer,
                block.head,
                block.fold,
            )
            for block in self.blocks
        )
        if (
            self.schema_version != PRICE_DISCOVERY_MODEL_RUN_SCHEMA
            or not _is_sha256(self.implementation_sha256)
            or not _is_sha256(self.dataset_bundle_sha256)
            or not self.feature_layers
            or any(layer not in FEATURE_LAYERS for layer in self.feature_layers)
            or tuple(sorted(self.feature_layers, key=FEATURE_LAYERS.index))
            != self.feature_layers
            or observed != expected
            or any(
                block.backend_requested != self.backend_requested
                or block.backend_kind != self.backend_kind
                or block.backend_device != self.backend_device
                or block.lightgbm_version != self.lightgbm_version
                for block in self.blocks
            )
            or any(
                (
                    self.profitability_claim,
                    self.execution_or_fill_claim,
                    self.trading_authority,
                )
            )
        ):
            raise ValueError("Round 72 model run contract is invalid")
        for block in self.blocks:
            block.validate()
        if self.run_sha256 != _run_sha256(self):
            raise ValueError("Round 72 model run fingerprint differs")


def _prediction_sha256(value: PriceDiscoveryFoldPrediction) -> str:
    digest = hashlib.sha256()
    metadata = {
        key: item
        for key, item in value.__dict__.items()
        if not isinstance(item, np.ndarray) and key != "prediction_sha256"
    }
    digest.update(_canonical_json(metadata).encode("ascii"))
    for array in (
        value.anchor_second_ms,
        value.utc_day,
        value.primary_target,
        value.primary_prediction,
        value.stress_anchor_second_ms,
        value.stress_utc_day,
        value.stress_target,
        value.stress_prediction,
    ):
        _array_digest(digest, array)
    return digest.hexdigest()


def _run_sha256(value: PriceDiscoveryPredictionRun) -> str:
    return _canonical_sha256(
        {
            "schema_version": value.schema_version,
            "implementation_sha256": value.implementation_sha256,
            "dataset_bundle_sha256": value.dataset_bundle_sha256,
            "feature_layers": list(value.feature_layers),
            "backend_requested": value.backend_requested,
            "backend_kind": value.backend_kind,
            "backend_device": value.backend_device,
            "lightgbm_version": value.lightgbm_version,
            "block_sha256": [block.prediction_sha256 for block in value.blocks],
            "claims": {
                "profitability": value.profitability_claim,
                "execution_or_fill": value.execution_or_fill_claim,
                "trading_authority": value.trading_authority,
            },
        }
    )


def load_price_discovery_folds(
    implementation_path: str | Path,
) -> tuple[dict[str, object], tuple[PriceDiscoveryFold, ...]]:
    implementation = load_round72_implementation(implementation_path)
    if implementation.get("schema_version") != ROUND72_IMPLEMENTATION_SCHEMA:
        raise ValueError("Round 72 implementation schema differs")
    split = implementation.get("split_contract")
    if not isinstance(split, dict) or not isinstance(split.get("folds"), list):
        raise ValueError("Round 72 split contract is missing")
    folds: list[PriceDiscoveryFold] = []
    for expected_fold, raw in enumerate(split["folds"], start=1):
        if not isinstance(raw, dict):
            raise ValueError("Round 72 fold entry is invalid")
        try:
            training = raw["training_months"]
            tuning = raw["tuning_months"]
            test = raw["test_months"]
            if any(not isinstance(value, list) or len(value) != 2 for value in (training, tuning, test)):
                raise ValueError("Round 72 fold range is invalid")
            fold = PriceDiscoveryFold(
                fold=int(raw["fold"]),
                training_start_month=_month_ordinal(str(training[0])),
                training_end_month=_month_ordinal(str(training[1])),
                tuning_start_month=_month_ordinal(str(tuning[0])),
                tuning_end_month=_month_ordinal(str(tuning[1])),
                test_start_month=_month_ordinal(str(test[0])),
                test_end_month=_month_ordinal(str(test[1])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 72 fold fields are invalid") from exc
        fold.validate()
        if fold.fold != expected_fold:
            raise ValueError("Round 72 fold order differs")
        if int(raw.get("training_month_count", -1)) != 24 + (fold.fold - 1) * 6:
            raise ValueError("Round 72 training-month count differs")
        folds.append(fold)
    if len(folds) != 6 or folds[-1].test_end_month != _month_ordinal("2026-03"):
        raise ValueError("Round 72 fold set differs")
    return implementation, tuple(folds)


def build_price_discovery_roles(
    dataset: PriceDiscoverySymbolDataset,
    fold: PriceDiscoveryFold,
    *,
    horizon_seconds: int,
) -> PriceDiscoveryRoleIndices:
    """Construct exact chronological roles and purge labels at role boundaries."""

    dataset.validate()
    fold.validate()
    try:
        horizon_index = HORIZONS_SECONDS.index(int(horizon_seconds))
    except ValueError as exc:
        raise ValueError("Round 72 horizon is not frozen") from exc
    months = np.asarray(dataset.month_ordinal, dtype=np.int32)
    training_month = (months >= fold.training_start_month) & (
        months <= fold.training_end_month
    )
    tuning_month = (months >= fold.tuning_start_month) & (
        months <= fold.tuning_end_month
    )
    test_month = (months >= fold.test_start_month) & (months <= fold.test_end_month)
    if not np.any(training_month) or not np.any(tuning_month) or not np.any(test_month):
        raise ValueError("Round 72 fold has an empty calendar role")
    label_available = dataset.anchor_second_ms + (
        PRIMARY_ENTRY_DELAY_SECONDS + int(horizon_seconds) + 1
    ) * 1_000
    first_tuning_available = int(np.min(dataset.available_time_ms[tuning_month]))
    first_test_available = int(np.min(dataset.available_time_ms[test_month]))
    valid = np.asarray(dataset.primary_valid[:, horizon_index], dtype=bool)
    stress_valid = np.asarray(dataset.stress_valid[:, horizon_index], dtype=bool)
    roles = PriceDiscoveryRoleIndices(
        training=_readonly(
            np.flatnonzero(
                training_month & valid & (label_available < first_tuning_available)
            ),
            np.int64,
        ),
        tuning=_readonly(
            np.flatnonzero(tuning_month & valid & (label_available < first_test_available)),
            np.int64,
        ),
        test=_readonly(np.flatnonzero(test_month & valid), np.int64),
        stress_test=_readonly(np.flatnonzero(test_month & stress_valid), np.int64),
    )
    roles.validate(dataset.rows)
    if (
        np.any(label_available[roles.training] >= first_tuning_available)
        or np.any(label_available[roles.tuning] >= first_test_available)
        or not np.all(np.isfinite(dataset.primary_target_bps[roles.training, horizon_index]))
        or not np.all(np.isfinite(dataset.primary_target_bps[roles.tuning, horizon_index]))
        or not np.all(np.isfinite(dataset.primary_target_bps[roles.test, horizon_index]))
        or not np.all(np.isfinite(dataset.stress_target_bps[roles.stress_test, horizon_index]))
    ):
        raise ValueError("Round 72 role target or purge contract differs")
    return roles


def binary_log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    truth = np.asarray(target, dtype=np.float64)
    prediction = np.clip(
        np.asarray(probability, dtype=np.float64),
        _PROBABILITY_EPSILON,
        1.0 - _PROBABILITY_EPSILON,
    )
    if (
        truth.ndim != 1
        or truth.shape != prediction.shape
        or len(truth) == 0
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(prediction))
        or not np.all((truth == 0.0) | (truth == 1.0))
    ):
        raise ValueError("binary log-loss inputs are invalid")
    return float(-np.mean(truth * np.log(prediction) + (1.0 - truth) * np.log1p(-prediction)))


def _temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probability, dtype=np.float64),
        _PROBABILITY_EPSILON,
        1.0 - _PROBABILITY_EPSILON,
    )
    logits = np.log(clipped) - np.log1p(-clipped)
    scaled = logits / float(temperature)
    output = np.empty_like(scaled)
    positive = scaled >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-scaled[positive]))
    exponential = np.exp(scaled[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return np.clip(output, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)


def fit_binary_temperature(
    probability: np.ndarray,
    target: np.ndarray,
) -> tuple[float, bool, float, float]:
    """Fit the frozen bounded temperature and retain only a real tuning gain."""

    raw = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    before = binary_log_loss(truth, raw)
    result = minimize_scalar(
        lambda value: binary_log_loss(truth, _temperature_scale(raw, float(value))),
        bounds=(0.25, 4.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    candidate = float(result.x)
    candidate_loss = float(result.fun)
    if result.success and math.isfinite(candidate_loss) and before - candidate_loss > 1e-12:
        return candidate, True, before, candidate_loss
    return 1.0, False, before, before


def fit_continuous_slope(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, bool, float, float]:
    """Fit the frozen nonnegative slope-through-origin calibration."""

    raw = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if (
        raw.ndim != 1
        or raw.shape != truth.shape
        or len(raw) == 0
        or not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(truth))
    ):
        raise ValueError("continuous calibration inputs are invalid")
    before = float(np.mean(np.square(truth - raw)))
    denominator = float(np.dot(raw, raw))
    candidate = float(np.clip(np.dot(raw, truth) / denominator, 0.0, 4.0)) if denominator > 0.0 else 1.0
    after = float(np.mean(np.square(truth - candidate * raw)))
    if math.isfinite(after) and before - after > 1e-12:
        return candidate, True, before, after
    return 1.0, False, before, before


def _target_values(
    dataset: PriceDiscoverySymbolDataset,
    indexes: np.ndarray,
    horizon_index: int,
    *,
    stress: bool,
    head: str,
) -> np.ndarray:
    source = dataset.stress_target_bps if stress else dataset.primary_target_bps
    values = np.asarray(source[indexes, horizon_index], dtype=np.float64)
    if head == "binary_direction":
        values = (values > 0.0).astype(np.float64)
    return values


def _fit_fold_prediction(
    dataset: PriceDiscoverySymbolDataset,
    fold: PriceDiscoveryFold,
    *,
    horizon_seconds: int,
    feature_layer: str,
    head: str,
    backend_requested: str,
    backend_parameters: Mapping[str, object],
    backend_kind: str,
    backend_device: str,
    model_parameters: Mapping[str, object],
) -> PriceDiscoveryFoldPrediction:
    roles = build_price_discovery_roles(dataset, fold, horizon_seconds=horizon_seconds)
    horizon_index = HORIZONS_SECONDS.index(horizon_seconds)
    feature_names = layer_feature_names(feature_layer)
    width = len(feature_names)
    train_target_bps = _target_values(
        dataset, roles.training, horizon_index, stress=False, head="continuous_return_bps"
    )
    training_prevalence = float(np.mean(train_target_bps > 0.0))
    training_mean = float(np.mean(train_target_bps))
    train_target = _target_values(
        dataset, roles.training, horizon_index, stress=False, head=head
    )
    tune_target = _target_values(
        dataset, roles.tuning, horizon_index, stress=False, head=head
    )
    test_target = _target_values(
        dataset, roles.test, horizon_index, stress=False, head=head
    )
    stress_target = _target_values(
        dataset, roles.stress_test, horizon_index, stress=True, head=head
    )
    positive_rows = int(np.count_nonzero(train_target_bps > 0.0))
    negative_rows = len(train_target_bps) - positive_rows
    if (
        min(positive_rows, negative_rows) <= 0
        or not _PROBABILITY_EPSILON < training_prevalence < 1.0 - _PROBABILITY_EPSILON
        or (head == "binary_direction" and len(np.unique(tune_target)) != 2)
        or (head == "continuous_return_bps" and np.std(train_target) <= 0.0)
    ):
        raise ValueError("Round 72 model role support is insufficient")

    parameters: dict[str, object] = {
        **backend_parameters,
        "boosting_type": str(model_parameters["boosting"]),
        "learning_rate": float(model_parameters["learning_rate"]),
        "num_leaves": int(model_parameters["num_leaves"]),
        "max_depth": int(model_parameters["max_depth"]),
        "min_data_in_leaf": int(model_parameters["min_data_in_leaf"]),
        "feature_fraction": float(model_parameters["feature_fraction"]),
        "bagging_fraction": float(model_parameters["bagging_fraction"]),
        "bagging_freq": int(model_parameters["bagging_freq"]),
        "lambda_l1": float(model_parameters["lambda_l1"]),
        "lambda_l2": float(model_parameters["lambda_l2"]),
        "max_bin": int(model_parameters["max_bin"]),
        "histogram_pool_size": 512,
        "objective": "binary" if head == "binary_direction" else "huber",
        "metric": "binary_logloss" if head == "binary_direction" else "l2",
    }
    if head == "continuous_return_bps":
        parameters["alpha"] = float(model_parameters["huber_alpha"])
    train_features = np.asarray(dataset.features[roles.training, :width], dtype=np.float32)
    tune_features = np.asarray(dataset.features[roles.tuning, :width], dtype=np.float32)
    train_set = lgb.Dataset(
        train_features,
        label=train_target,
        feature_name=list(feature_names),
        free_raw_data=True,
    )
    tune_set = lgb.Dataset(
        tune_features,
        label=tune_target,
        reference=train_set,
        feature_name=list(feature_names),
        free_raw_data=True,
    )
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=int(model_parameters["maximum_boosting_iterations"]),
        valid_sets=[tune_set],
        valid_names=["tuning"],
        callbacks=[
            lgb.early_stopping(
                int(model_parameters["early_stopping_rounds"]), verbose=False
            ),
            lgb.log_evaluation(0),
        ],
    )
    best_iteration = int(booster.best_iteration or booster.current_iteration())
    if not 1 <= best_iteration <= int(model_parameters["maximum_boosting_iterations"]):
        raise RuntimeError("Round 72 LightGBM best iteration is invalid")
    tune_raw = np.asarray(
        booster.predict(tune_features, num_iteration=best_iteration), dtype=np.float64
    )
    if head == "binary_direction":
        calibration, retained, before, after = fit_binary_temperature(
            tune_raw, tune_target
        )
    else:
        calibration, retained, before, after = fit_continuous_slope(
            tune_raw, tune_target
        )

    primary_features = np.asarray(
        dataset.features[roles.test, :width], dtype=np.float32
    )
    stress_features = np.asarray(
        dataset.features[roles.stress_test, :width], dtype=np.float32
    )
    primary_raw = np.asarray(
        booster.predict(primary_features, num_iteration=best_iteration),
        dtype=np.float64,
    )
    stress_raw = np.asarray(
        booster.predict(stress_features, num_iteration=best_iteration),
        dtype=np.float64,
    )
    model_string = booster.model_to_string(num_iteration=best_iteration)
    model_bytes = len(model_string.encode("utf-8"))
    reloaded = lgb.Booster(model_str=model_string)
    reload_tuning = np.asarray(reloaded.predict(tune_features), dtype=np.float64)
    reload_primary = np.asarray(reloaded.predict(primary_features), dtype=np.float64)
    reload_stress = np.asarray(reloaded.predict(stress_features), dtype=np.float64)
    reload_difference = float(
        max(
            np.max(np.abs(tune_raw - reload_tuning), initial=0.0),
            np.max(np.abs(primary_raw - reload_primary), initial=0.0),
            np.max(np.abs(stress_raw - reload_stress), initial=0.0),
        )
    )
    if reload_difference > 1e-12:
        raise RuntimeError("Round 72 serialized model prediction identity failed")
    if head == "binary_direction":
        primary_prediction = _temperature_scale(primary_raw, calibration)
        stress_prediction = _temperature_scale(stress_raw, calibration)
    else:
        primary_prediction = calibration * primary_raw
        stress_prediction = calibration * stress_raw
    provisional = PriceDiscoveryFoldPrediction(
        symbol=dataset.symbol,
        horizon_seconds=horizon_seconds,
        feature_layer=feature_layer,
        head=head,
        fold=fold.fold,
        feature_count=width,
        training_rows=len(roles.training),
        tuning_rows=len(roles.tuning),
        test_rows=len(roles.test),
        stress_test_rows=len(roles.stress_test),
        training_positive_rows=positive_rows,
        training_negative_rows=negative_rows,
        training_prevalence=training_prevalence,
        training_mean_target_bps=training_mean,
        best_iteration=best_iteration,
        calibration_value=float(calibration),
        calibration_retained=retained,
        tuning_loss_before_calibration=before,
        tuning_loss_after_calibration=after,
        model_bytes=model_bytes,
        model_sha256=hashlib.sha256(model_string.encode("utf-8")).hexdigest(),
        reload_max_absolute_prediction_difference=reload_difference,
        backend_requested=backend_requested,
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        anchor_second_ms=_readonly(dataset.anchor_second_ms[roles.test], np.int64),
        utc_day=_readonly(dataset.utc_day[roles.test], np.int32),
        primary_target=_readonly(test_target, np.float64),
        primary_prediction=_readonly(primary_prediction, np.float64),
        stress_anchor_second_ms=_readonly(
            dataset.anchor_second_ms[roles.stress_test], np.int64
        ),
        stress_utc_day=_readonly(dataset.utc_day[roles.stress_test], np.int32),
        stress_target=_readonly(stress_target, np.float64),
        stress_prediction=_readonly(stress_prediction, np.float64),
        prediction_sha256="",
    )
    block = replace(
        provisional,
        prediction_sha256=_prediction_sha256(provisional),
    )
    block.validate()
    return block


def run_price_discovery_models(
    bundle: PriceDiscoveryDatasetBundle,
    *,
    implementation_path: str | Path,
    compute_backend: str = "auto",
    feature_layers: Sequence[str] = PRIMARY_FEATURE_LAYERS,
    progress: _ProgressCallback | None = None,
) -> PriceDiscoveryPredictionRun:
    """Fit the frozen models sequentially and retain only OOS predictions."""

    bundle.validate()
    implementation, folds = load_price_discovery_folds(implementation_path)
    if implementation["implementation_sha256"] != bundle.implementation_sha256:
        raise ValueError("Round 72 dataset and implementation identities differ")
    layers = tuple(str(layer) for layer in feature_layers)
    if (
        not layers
        or len(layers) != len(set(layers))
        or any(layer not in FEATURE_LAYERS for layer in layers)
        or tuple(sorted(layers, key=FEATURE_LAYERS.index)) != layers
        or ("cross_asset" in layers and layers != ("cross_asset",))
    ):
        raise ValueError("Round 72 requested feature-layer run is invalid")
    model_contract = implementation.get("model_contract")
    if not isinstance(model_contract, dict) or not isinstance(
        model_contract.get("parameters"), dict
    ):
        raise ValueError("Round 72 model contract is missing")
    parameters = model_contract["parameters"]
    if parameters.get("seed") != ROUND72_SEED:
        raise ValueError("Round 72 model seed differs")
    requested_backend = str(compute_backend or "auto").strip().lower()
    backend, backend_kind, backend_device = lightgbm_backend_parameters(
        requested_backend,
        ROUND72_SEED,
        reproducible=True,
    )
    if backend_kind == "opencl" and backend.get("gpu_use_dp") is not True:
        raise RuntimeError("Round 72 OpenCL training requires FP64 accumulation")
    blocks: list[PriceDiscoveryFoldPrediction] = []
    total = len(FLOW_SYMBOLS) * len(HORIZONS_SECONDS) * len(layers) * len(
        PRICE_DISCOVERY_HEADS
    ) * len(folds)
    completed = 0
    datasets = {dataset.symbol: dataset for dataset in bundle.symbols}
    for symbol in FLOW_SYMBOLS:
        dataset = datasets[symbol]
        for horizon in HORIZONS_SECONDS:
            for layer in layers:
                for head in PRICE_DISCOVERY_HEADS:
                    for fold in folds:
                        identity = {
                            "symbol": symbol,
                            "horizon_seconds": horizon,
                            "feature_layer": layer,
                            "head": head,
                            "fold": fold.fold,
                            "completed_models": completed,
                            "total_models": total,
                        }
                        if progress:
                            progress("price_discovery_model_started", identity)
                        block = _fit_fold_prediction(
                            dataset,
                            fold,
                            horizon_seconds=horizon,
                            feature_layer=layer,
                            head=head,
                            backend_requested=requested_backend,
                            backend_parameters=backend,
                            backend_kind=backend_kind,
                            backend_device=backend_device,
                            model_parameters=parameters,
                        )
                        blocks.append(block)
                        completed += 1
                        if progress:
                            progress(
                                "price_discovery_model_completed",
                                {
                                    **identity,
                                    "completed_models": completed,
                                    "best_iteration": block.best_iteration,
                                    "model_sha256": block.model_sha256,
                                    "test_rows": block.test_rows,
                                    "stress_test_rows": block.stress_test_rows,
                                },
                            )
    provisional = PriceDiscoveryPredictionRun(
        schema_version=PRICE_DISCOVERY_MODEL_RUN_SCHEMA,
        implementation_sha256=bundle.implementation_sha256,
        dataset_bundle_sha256=bundle.bundle_sha256,
        feature_layers=layers,
        backend_requested=requested_backend,
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        blocks=tuple(blocks),
        run_sha256="",
    )
    result = replace(provisional, run_sha256=_run_sha256(provisional))
    result.validate()
    return result


__all__ = [
    "PRICE_DISCOVERY_HEADS",
    "PRICE_DISCOVERY_MODEL_RUN_SCHEMA",
    "PRIMARY_FEATURE_LAYERS",
    "ROUND72_SEED",
    "PriceDiscoveryFold",
    "PriceDiscoveryFoldPrediction",
    "PriceDiscoveryPredictionRun",
    "PriceDiscoveryRoleIndices",
    "binary_log_loss",
    "build_price_discovery_roles",
    "fit_binary_temperature",
    "fit_continuous_slope",
    "load_price_discovery_folds",
    "run_price_discovery_models",
]
