"""Direct conditional-mean control for exact barrier payoffs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import lightgbm as lgb
import numpy as np

from .categorical_payoff_lightgbm import CategoricalPayoffDataset
from .lightgbm_backend import (
    SUPPORTED_LIGHTGBM_BACKEND_KINDS,
    lightgbm_backend_parameters,
)
from .storage import write_json_atomic


DIRECT_PAYOFF_SCHEMA_VERSION = "direct-barrier-payoff-lightgbm-v1"
_MODEL_FAMILY = "side_specific_direct_exact_payoff_mean"
_SIDES = ("long", "short")
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class DirectPayoffSpec:
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
    gpu_use_dp_required: bool = True

    def __post_init__(self) -> None:
        numeric = (
            self.learning_rate,
            self.feature_fraction,
            self.bagging_fraction,
            self.lambda_l1,
            self.lambda_l2,
        )
        if (
            not self.candidate_id.strip()
            or self.family != _MODEL_FAMILY
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
            or self.gpu_use_dp_required is not True
        ):
            raise ValueError("direct payoff specification is invalid")


@dataclass(frozen=True)
class TrainedDirectPayoffModel:
    schema_version: str
    model_family: str
    spec: DirectPayoffSpec
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
    training_target_mean_bps: Mapping[str, float]
    best_iterations: Mapping[str, int]
    model_strings: Mapping[str, str]
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False


@dataclass(frozen=True)
class DirectPayoffPredictionBatch:
    endpoint_indexes: np.ndarray
    long_mean_bps: np.ndarray
    short_mean_bps: np.ndarray

    @property
    def rows(self) -> int:
        return int(len(self.endpoint_indexes))

    def __post_init__(self) -> None:
        rows = self.rows
        if (
            rows <= 0
            or np.asarray(self.endpoint_indexes).shape != (rows,)
            or np.any(np.diff(self.endpoint_indexes) <= 0)
            or np.asarray(self.long_mean_bps).shape != (rows,)
            or np.asarray(self.short_mean_bps).shape != (rows,)
            or not np.all(np.isfinite(self.long_mean_bps))
            or not np.all(np.isfinite(self.short_mean_bps))
        ):
            raise ValueError("direct payoff prediction batch is invalid")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _model_payload(model: TrainedDirectPayoffModel) -> dict[str, object]:
    payload = asdict(model)
    payload.pop("model_sha256")
    return payload


def _model_sha256(model: TrainedDirectPayoffModel) -> str:
    return hashlib.sha256(
        _canonical_json(_model_payload(model)).encode("ascii")
    ).hexdigest()


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
        raise ValueError(f"direct payoff {label} indexes are invalid")
    return indexes


def _validate_model(model: TrainedDirectPayoffModel, *, reload: bool) -> None:
    sides = set(_SIDES)
    if (
        model.schema_version != DIRECT_PAYOFF_SCHEMA_VERSION
        or model.model_family != _MODEL_FAMILY
        or model.spec.family != _MODEL_FAMILY
        or model.target_scenario not in {"base", "stress"}
        or model.backend_kind not in SUPPORTED_LIGHTGBM_BACKEND_KINDS
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
        or set(model.training_target_mean_bps) != sides
        or set(model.best_iterations) != sides
        or set(model.model_strings) != sides
        or _model_sha256(model) != model.model_sha256
    ):
        raise ValueError("direct payoff model contract is invalid")
    for side in _SIDES:
        if (
            not math.isfinite(float(model.training_target_mean_bps[side]))
            or not 1 <= int(model.best_iterations[side]) <= model.spec.num_boost_round
            or not model.model_strings[side].strip()
        ):
            raise ValueError("direct payoff side contract is invalid")
    if reload:
        try:
            for side in _SIDES:
                booster = lgb.Booster(model_str=model.model_strings[side])
                if booster.num_feature() != len(model.feature_names):
                    raise ValueError("direct payoff booster feature count drifted")
        except lgb.basic.LightGBMError as exc:
            raise ValueError("direct payoff booster cannot be reloaded") from exc


def _train_regressor(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_early: np.ndarray,
    y_early: np.ndarray,
    parameters: Mapping[str, object],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[lgb.Booster, int]:
    training = lgb.Dataset(
        np.asarray(x_train, dtype=np.float32),
        label=np.asarray(y_train, dtype=np.float32),
        free_raw_data=False,
    )
    early = lgb.Dataset(
        np.asarray(x_early, dtype=np.float32),
        label=np.asarray(y_early, dtype=np.float32),
        reference=training,
        free_raw_data=False,
    )
    booster = lgb.train(
        {**parameters, "objective": "regression", "metric": "l2"},
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


def train_direct_payoff_model(
    dataset: CategoricalPayoffDataset,
    *,
    train_indexes: np.ndarray,
    early_stop_indexes: np.ndarray,
    spec: DirectPayoffSpec,
    target_scenario: str,
    compute_backend: str,
    seed: int,
) -> TrainedDirectPayoffModel:
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
    if train[-1] >= early[0]:
        raise ValueError("direct payoff chronological roles overlap")
    maximum_exit = np.maximum(dataset.long_exit_time_ms, dataset.short_exit_time_ms)
    if np.any(maximum_exit[train] >= dataset.decision_time_ms[early[0]]):
        raise ValueError("direct payoff labels cross the early-stop boundary")
    if target_scenario != dataset.target_scenario:
        raise ValueError("direct payoff target scenario drifted")
    parameters, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        int(seed),
        reproducible=True,
    )
    if backend_kind == "opencl" and (
        not spec.gpu_use_dp_required or parameters.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("direct payoff OpenCL FP64 accumulation is required")
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
    targets = {
        "long": np.asarray(dataset.long_net_bps, dtype=np.float32),
        "short": np.asarray(dataset.short_net_bps, dtype=np.float32),
    }
    means: dict[str, float] = {}
    iterations: dict[str, int] = {}
    model_strings: dict[str, str] = {}
    for side in _SIDES:
        booster, iteration = _train_regressor(
            x_train=features[train],
            y_train=targets[side][train],
            x_early=features[early],
            y_early=targets[side][early],
            parameters=common,
            num_boost_round=spec.num_boost_round,
            early_stopping_rounds=spec.early_stopping_rounds,
        )
        means[side] = float(np.mean(targets[side][train], dtype=np.float64))
        iterations[side] = iteration
        model_strings[side] = booster.model_to_string(num_iteration=iteration)
    provisional = TrainedDirectPayoffModel(
        schema_version=DIRECT_PAYOFF_SCHEMA_VERSION,
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
        training_target_mean_bps=means,
        best_iterations=iterations,
        model_strings=model_strings,
        model_sha256="",
    )
    model = TrainedDirectPayoffModel(
        **{**provisional.__dict__, "model_sha256": _model_sha256(provisional)}
    )
    _validate_model(model, reload=False)
    return model


def predict_direct_payoff_model(
    model: TrainedDirectPayoffModel,
    dataset: CategoricalPayoffDataset,
    indexes: np.ndarray,
) -> DirectPayoffPredictionBatch:
    _validate_model(model, reload=False)
    if (
        model.symbol != dataset.symbol
        or model.feature_names != dataset.feature_names
        or model.source_dataset_sha256 != dataset.dataset_sha256
        or model.target_scenario != dataset.target_scenario
    ):
        raise ValueError("direct payoff prediction dataset drifted")
    selected = _validate_indexes(
        indexes,
        rows=dataset.rows,
        label="prediction",
        minimum_rows=1,
    )
    features = np.asarray(dataset.features[selected], dtype=np.float32)
    outputs: dict[str, np.ndarray] = {}
    for side in _SIDES:
        booster = lgb.Booster(model_str=model.model_strings[side])
        outputs[side] = np.asarray(
            booster.predict(features, num_iteration=model.best_iterations[side]),
            dtype=np.float64,
        )
    return DirectPayoffPredictionBatch(
        endpoint_indexes=np.asarray(
            dataset.source_row_indexes[selected], dtype=np.int64
        ),
        long_mean_bps=outputs["long"],
        short_mean_bps=outputs["short"],
    )


def save_direct_payoff_model(
    path: str | Path,
    model: TrainedDirectPayoffModel,
) -> None:
    _validate_model(model, reload=True)
    payload = asdict(model)
    encoded = _canonical_json(payload).encode("utf-8")
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("direct payoff model artifact is too large")
    write_json_atomic(Path(path), payload, indent=None, sort_keys=True)


def load_direct_payoff_model(path: str | Path) -> TrainedDirectPayoffModel:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("direct payoff model artifact is missing or oversized")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("direct payoff model artifact must be an object")
    expected = {field.name for field in fields(TrainedDirectPayoffModel)}
    if set(payload) != expected or not isinstance(payload.get("spec"), dict):
        raise ValueError("direct payoff model artifact fields drifted")
    payload["spec"] = DirectPayoffSpec(**payload["spec"])
    payload["feature_names"] = tuple(payload["feature_names"])
    model = TrainedDirectPayoffModel(**payload)
    _validate_model(model, reload=True)
    return model


__all__ = [
    "DIRECT_PAYOFF_SCHEMA_VERSION",
    "DirectPayoffPredictionBatch",
    "DirectPayoffSpec",
    "TrainedDirectPayoffModel",
    "load_direct_payoff_model",
    "predict_direct_payoff_model",
    "save_direct_payoff_model",
    "train_direct_payoff_model",
]
