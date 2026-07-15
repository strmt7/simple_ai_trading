"""Pooled GPU-capable LightGBM hazards for queue-censored passive fills."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Sequence

import lightgbm as lgb
import numpy as np

from .lightgbm_backend import lightgbm_backend_parameters
from .probability_calibration import apply_platt_scaling, fit_platt_scaling
from .queue_fill_survival import (
    PASSIVE_FILL_SURVIVAL_SCHEMA_VERSION,
    PassiveFillSurvivalPanel,
    build_hazard_risk_set,
    fill_bucket_prevalence,
    hazards_to_bucket_probabilities,
    validate_passive_fill_survival_panel,
)
from .queue_censored_actions import PASSIVE_FILL_BUCKETS_MS
from .storage import write_json_atomic


QUEUE_FILL_LIGHTGBM_SCHEMA_VERSION = "queue-fill-discrete-hazard-lightgbm-v1"
QUEUE_FILL_ENSEMBLE_SCHEMA_VERSION = "queue-fill-discrete-hazard-ensemble-v1"
QUEUE_FILL_MODEL_FAMILY = "shared_three_interval_passive_fill_hazard"
QUEUE_FILL_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
QUEUE_FILL_SEEDS = (5701, 5702, 5703)
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class QueueFillLightGBMSpec:
    candidate_id: str = "queue_censored_make_take_lightgbm"
    family: str = QUEUE_FILL_MODEL_FAMILY
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = 8
    min_data_in_leaf: int = 256
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l1: float = 0.1
    lambda_l2: float = 1.0
    max_bin: int = 127
    num_boost_round: int = 1_000
    early_stopping_rounds: int = 50
    minimum_training_class_rows_per_symbol: int = 128
    minimum_early_stop_class_rows_per_symbol: int = 32
    minimum_calibration_class_rows_per_symbol: int = 32
    gpu_use_dp_required: bool = True

    def __post_init__(self) -> None:
        integral = (
            self.num_leaves,
            self.max_depth,
            self.min_data_in_leaf,
            self.bagging_freq,
            self.max_bin,
            self.num_boost_round,
            self.early_stopping_rounds,
            self.minimum_training_class_rows_per_symbol,
            self.minimum_early_stop_class_rows_per_symbol,
            self.minimum_calibration_class_rows_per_symbol,
        )
        numeric = (
            self.learning_rate,
            self.feature_fraction,
            self.bagging_fraction,
            self.lambda_l1,
            self.lambda_l2,
        )
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id.strip()
            or self.family != QUEUE_FILL_MODEL_FAMILY
            or any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                for value in integral
            )
            or any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                for value in numeric
            )
            or not all(math.isfinite(float(value)) for value in numeric)
            or not 0.0 < self.learning_rate <= 0.25
            or not 2 <= int(self.num_leaves) <= 255
            or not 1 <= int(self.max_depth) <= 16
            or not 2 <= int(self.min_data_in_leaf) <= 65_536
            or not 0.0 < self.feature_fraction <= 1.0
            or not 0.0 < self.bagging_fraction <= 1.0
            or not 0 <= int(self.bagging_freq) <= 100
            or self.lambda_l1 < 0.0
            or self.lambda_l2 < 0.0
            or not 31 <= int(self.max_bin) <= 255
            or not 10 <= int(self.num_boost_round) <= 10_000
            or not 5 <= int(self.early_stopping_rounds) < int(self.num_boost_round)
            or not 2 <= int(self.minimum_training_class_rows_per_symbol)
            or not 2 <= int(self.minimum_early_stop_class_rows_per_symbol)
            or not 2 <= int(self.minimum_calibration_class_rows_per_symbol)
            or self.gpu_use_dp_required is not True
        ):
            raise ValueError("queue-fill LightGBM specification is invalid")


@dataclass(frozen=True)
class SymbolHazardSupport:
    symbol: str
    training: tuple[tuple[int, int], ...]
    early_stop: tuple[tuple[int, int], ...]
    calibration: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TrainedQueueFillLightGBMModel:
    schema_version: str
    model_family: str
    spec: QueueFillLightGBMSpec
    feature_names: tuple[str, ...]
    source_dataset_sha256_by_symbol: tuple[tuple[str, str], ...]
    training_panel_sha256_by_symbol: tuple[tuple[str, str], ...]
    early_stop_panel_sha256_by_symbol: tuple[tuple[str, str], ...]
    calibration_panel_sha256_by_symbol: tuple[tuple[str, str], ...]
    training_end_ms: int
    early_stop_start_ms: int
    early_stop_end_ms: int
    calibration_start_ms: int
    class_support: tuple[SymbolHazardSupport, ...]
    baseline_bucket_probabilities: tuple[float, float, float, float]
    probability_calibration: tuple[tuple[float, float], ...]
    backend_requested: str
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    seed: int
    best_iterations: tuple[int, int, int]
    model_strings: tuple[str, str, str]
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False


@dataclass(frozen=True)
class TrainedQueueFillLightGBMEnsemble:
    schema_version: str
    members: tuple[TrainedQueueFillLightGBMModel, ...]
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    @property
    def spec(self) -> QueueFillLightGBMSpec:
        return self.members[0].spec

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.members[0].feature_names

    @property
    def source_dataset_sha256_by_symbol(self) -> tuple[tuple[str, str], ...]:
        return self.members[0].source_dataset_sha256_by_symbol

    @property
    def training_panel_sha256_by_symbol(self) -> tuple[tuple[str, str], ...]:
        return self.members[0].training_panel_sha256_by_symbol

    @property
    def early_stop_panel_sha256_by_symbol(self) -> tuple[tuple[str, str], ...]:
        return self.members[0].early_stop_panel_sha256_by_symbol

    @property
    def calibration_panel_sha256_by_symbol(self) -> tuple[tuple[str, str], ...]:
        return self.members[0].calibration_panel_sha256_by_symbol


QueueFillLightGBMArtifact = (
    TrainedQueueFillLightGBMModel | TrainedQueueFillLightGBMEnsemble
)


@dataclass(frozen=True)
class QueueFillPredictionBatch:
    source_action_feature_sha256: str
    source_panel_sha256: str
    model_sha256: str
    symbol: str
    event_index: np.ndarray
    decision_time_ms: np.ndarray
    action_side: np.ndarray
    hazard_probabilities: np.ndarray
    bucket_probabilities: np.ndarray
    fill_probability_15s: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.event_index.size)

    def __post_init__(self) -> None:
        rows = self.rows
        if (
            rows <= 0
            or not _is_sha256(self.source_action_feature_sha256)
            or not _is_sha256(self.source_panel_sha256)
            or not _is_sha256(self.model_sha256)
            or self.symbol not in QUEUE_FILL_SYMBOLS
            or np.asarray(self.event_index).shape != (rows,)
            or np.asarray(self.decision_time_ms).shape != (rows,)
            or np.asarray(self.action_side).shape != (rows,)
            or np.asarray(self.hazard_probabilities).shape != (rows, 3)
            or np.asarray(self.bucket_probabilities).shape != (rows, 4)
            or np.asarray(self.fill_probability_15s).shape != (rows,)
            or np.any(np.diff(self.decision_time_ms) < 0)
            or not np.all(np.isin(self.action_side, (-1, 1)))
            or not np.isfinite(self.hazard_probabilities).all()
            or not np.isfinite(self.bucket_probabilities).all()
            or not np.isfinite(self.fill_probability_15s).all()
            or np.any(self.hazard_probabilities < 0.0)
            or np.any(self.hazard_probabilities > 1.0)
            or np.any(self.bucket_probabilities < 0.0)
            or np.any(self.bucket_probabilities > 1.0)
            or not np.allclose(
                self.bucket_probabilities,
                hazards_to_bucket_probabilities(self.hazard_probabilities),
                atol=1e-12,
                rtol=1e-12,
            )
            or not np.allclose(
                np.sum(self.bucket_probabilities, axis=1), 1.0, atol=1e-10
            )
            or not np.allclose(
                self.fill_probability_15s,
                np.sum(self.bucket_probabilities[:, :3], axis=1),
                atol=1e-12,
            )
        ):
            raise ValueError("queue-fill prediction batch is invalid")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _model_payload(model: TrainedQueueFillLightGBMModel) -> dict[str, object]:
    payload = asdict(model)
    payload.pop("model_sha256")
    return payload


def _model_sha256(model: TrainedQueueFillLightGBMModel) -> str:
    return hashlib.sha256(_canonical_json(_model_payload(model)).encode("ascii")).hexdigest()


def _ordered_panels(
    panels: Sequence[PassiveFillSurvivalPanel],
    *,
    role: str,
) -> tuple[PassiveFillSurvivalPanel, ...]:
    values = tuple(panels)
    try:
        for panel in values:
            validate_passive_fill_survival_panel(panel)
    except ValueError as exc:
        raise ValueError(f"queue-fill {role} panel contract is invalid") from exc
    if (
        len(values) != len(QUEUE_FILL_SYMBOLS)
        or {panel.symbol for panel in values} != set(QUEUE_FILL_SYMBOLS)
        or any(panel.schema_version != PASSIVE_FILL_SURVIVAL_SCHEMA_VERSION for panel in values)
        or any(panel.rows <= 0 for panel in values)
        or any(np.any(np.diff(panel.decision_time_ms) < 0) for panel in values)
        or any(
            panel.source_first_decision_time_ms > int(np.min(panel.decision_time_ms))
            or panel.source_last_decision_time_ms < int(np.max(panel.decision_time_ms))
            or panel.source_first_decision_time_ms > panel.source_last_decision_time_ms
            or not _is_sha256(panel.source_dataset_sha256)
            or not _is_sha256(panel.panel_sha256)
            for panel in values
        )
    ):
        raise ValueError(f"queue-fill {role} panel contract is invalid")
    ordered = tuple(sorted(values, key=lambda panel: panel.symbol))
    feature_names = ordered[0].feature_names
    if any(panel.feature_names != feature_names for panel in ordered[1:]):
        raise ValueError(f"queue-fill {role} feature contracts drifted")
    return ordered


def _role_support(
    panels: tuple[PassiveFillSurvivalPanel, ...],
) -> dict[str, tuple[tuple[int, int], ...]]:
    output: dict[str, tuple[tuple[int, int], ...]] = {}
    for panel in panels:
        counts: list[tuple[int, int]] = []
        for head in range(3):
            labels = build_hazard_risk_set(panel, head).labels
            counts.append(
                (int(np.count_nonzero(labels == 1.0)), int(np.count_nonzero(labels == 0.0)))
            )
        output[panel.symbol] = tuple(counts)
    return output


def _risk_data(
    panels: tuple[PassiveFillSurvivalPanel, ...],
    hazard_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    row_counts: list[int] = []
    for panel in panels:
        risk = build_hazard_risk_set(panel, hazard_index)
        features.append(np.asarray(risk.features, dtype=np.float32))
        labels.append(np.asarray(risk.labels, dtype=np.float32))
        row_counts.append(risk.labels.size)
    total = int(sum(row_counts))
    weights = [
        np.full(rows, total / (len(row_counts) * rows), dtype=np.float32)
        for rows in row_counts
    ]
    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(weights, axis=0),
    )


def _validate_support(
    support: tuple[SymbolHazardSupport, ...],
    spec: QueueFillLightGBMSpec,
) -> None:
    if tuple(item.symbol for item in support) != tuple(sorted(QUEUE_FILL_SYMBOLS)):
        raise ValueError("queue-fill symbol support contract is invalid")
    role_minimums = (
        ("training", spec.minimum_training_class_rows_per_symbol),
        ("early_stop", spec.minimum_early_stop_class_rows_per_symbol),
        ("calibration", spec.minimum_calibration_class_rows_per_symbol),
    )
    for item in support:
        for role, minimum in role_minimums:
            values = getattr(item, role)
            if (
                len(values) != 3
                or any(len(pair) != 2 for pair in values)
                or any(min(pair) < minimum for pair in values)
            ):
                raise ValueError("queue-fill per-symbol class support is insufficient")


def _validate_model(model: TrainedQueueFillLightGBMModel, *, reload: bool) -> None:
    symbols = tuple(sorted(QUEUE_FILL_SYMBOLS))
    source_symbols = tuple(symbol for symbol, _sha in model.source_dataset_sha256_by_symbol)
    if (
        model.schema_version != QUEUE_FILL_LIGHTGBM_SCHEMA_VERSION
        or model.model_family != QUEUE_FILL_MODEL_FAMILY
        or model.spec.family != QUEUE_FILL_MODEL_FAMILY
        or not all(
            isinstance(value, str) and value.strip()
            for value in (
                model.backend_requested,
                model.backend_kind,
                model.backend_device,
                model.lightgbm_version,
            )
        )
        or not model.feature_names
        or len(set(model.feature_names)) != len(model.feature_names)
        or source_symbols != symbols
        or any(not _is_sha256(sha) for _symbol, sha in model.source_dataset_sha256_by_symbol)
        or any(
            tuple(symbol for symbol, _sha in values) != symbols
            or any(not _is_sha256(sha) for _symbol, sha in values)
            for values in (
                model.training_panel_sha256_by_symbol,
                model.early_stop_panel_sha256_by_symbol,
                model.calibration_panel_sha256_by_symbol,
            )
        )
        or not model.training_end_ms < model.early_stop_start_ms
        or not model.early_stop_end_ms < model.calibration_start_ms
        or model.backend_kind not in {"opencl", "cpu"}
        or isinstance(model.seed, (bool, np.bool_))
        or not isinstance(model.seed, Integral)
        or model.seed not in QUEUE_FILL_SEEDS
        or len(model.best_iterations) != 3
        or len(model.model_strings) != 3
        or len(model.probability_calibration) != 3
        or len(model.baseline_bucket_probabilities) != 4
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or not 1 <= value <= model.spec.num_boost_round
            for value in model.best_iterations
        )
        or any(not value.strip() for value in model.model_strings)
        or not np.isclose(sum(model.baseline_bucket_probabilities), 1.0, atol=1e-12)
        or any(value <= 0.0 or value >= 1.0 for value in model.baseline_bucket_probabilities)
        or model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.portfolio_claim
        or model.leverage_applied
        or not _is_sha256(model.model_sha256)
        or _model_sha256(model) != model.model_sha256
    ):
        raise ValueError("queue-fill LightGBM model contract is invalid")
    _validate_support(model.class_support, model.spec)
    for calibration in model.probability_calibration:
        apply_platt_scaling(np.asarray([0.5]), calibration)
    if reload:
        try:
            for model_string in model.model_strings:
                booster = lgb.Booster(model_str=model_string)
                if booster.num_feature() != len(model.feature_names):
                    raise ValueError("queue-fill booster feature count drifted")
        except lgb.basic.LightGBMError as exc:
            raise ValueError("queue-fill booster payload cannot be reloaded") from exc


def _ensemble_payload(
    model: TrainedQueueFillLightGBMEnsemble,
) -> dict[str, object]:
    return {
        "schema_version": model.schema_version,
        "member_model_sha256": [member.model_sha256 for member in model.members],
        "member_seeds": [member.seed for member in model.members],
        "trading_authority": model.trading_authority,
        "execution_claim": model.execution_claim,
        "profitability_claim": model.profitability_claim,
        "portfolio_claim": model.portfolio_claim,
        "leverage_applied": model.leverage_applied,
    }


def _ensemble_sha256(model: TrainedQueueFillLightGBMEnsemble) -> str:
    return hashlib.sha256(
        _canonical_json(_ensemble_payload(model)).encode("ascii")
    ).hexdigest()


def _validate_ensemble(
    model: TrainedQueueFillLightGBMEnsemble,
    *,
    reload: bool,
) -> None:
    members = tuple(model.members)
    for member in members:
        _validate_model(member, reload=reload)
    shared_fields = (
        "spec",
        "feature_names",
        "source_dataset_sha256_by_symbol",
        "training_panel_sha256_by_symbol",
        "early_stop_panel_sha256_by_symbol",
        "calibration_panel_sha256_by_symbol",
        "training_end_ms",
        "early_stop_start_ms",
        "early_stop_end_ms",
        "calibration_start_ms",
        "class_support",
        "baseline_bucket_probabilities",
        "backend_requested",
        "backend_kind",
        "backend_device",
        "lightgbm_version",
    )
    if (
        model.schema_version != QUEUE_FILL_ENSEMBLE_SCHEMA_VERSION
        or len(members) != len(QUEUE_FILL_SEEDS)
        or tuple(member.seed for member in members) != QUEUE_FILL_SEEDS
        or any(
            getattr(member, field) != getattr(members[0], field)
            for member in members[1:]
            for field in shared_fields
        )
        or len({member.model_sha256 for member in members}) != len(members)
        or model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.portfolio_claim
        or model.leverage_applied
        or not _is_sha256(model.model_sha256)
        or _ensemble_sha256(model) != model.model_sha256
    ):
        raise ValueError("queue-fill LightGBM ensemble contract is invalid")


def build_queue_fill_lightgbm_ensemble(
    members: Sequence[TrainedQueueFillLightGBMModel],
) -> TrainedQueueFillLightGBMEnsemble:
    """Bind all frozen seeds without selecting a member on held-forward outcomes."""

    ordered = tuple(sorted(tuple(members), key=lambda member: member.seed))
    provisional = TrainedQueueFillLightGBMEnsemble(
        schema_version=QUEUE_FILL_ENSEMBLE_SCHEMA_VERSION,
        members=ordered,
        model_sha256="",
    )
    model = TrainedQueueFillLightGBMEnsemble(
        **{
            **provisional.__dict__,
            "model_sha256": _ensemble_sha256(provisional),
        }
    )
    _validate_ensemble(model, reload=True)
    return model


def validate_queue_fill_lightgbm_model(
    model: QueueFillLightGBMArtifact,
    *,
    reload: bool = False,
) -> None:
    """Public artifact validator used by downstream evidence gates."""

    if isinstance(model, TrainedQueueFillLightGBMEnsemble):
        _validate_ensemble(model, reload=reload)
        return
    _validate_model(model, reload=reload)


def _train_head(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_weight: np.ndarray,
    x_early: np.ndarray,
    y_early: np.ndarray,
    early_weight: np.ndarray,
    parameters: dict[str, object],
    feature_names: tuple[str, ...],
    spec: QueueFillLightGBMSpec,
) -> tuple[lgb.Booster, int]:
    training = lgb.Dataset(
        x_train,
        label=y_train,
        weight=train_weight,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    early = lgb.Dataset(
        x_early,
        label=y_early,
        weight=early_weight,
        feature_name=list(feature_names),
        reference=training,
        free_raw_data=False,
    )
    booster = lgb.train(
        {**parameters, "objective": "binary", "metric": "binary_logloss"},
        training,
        num_boost_round=spec.num_boost_round,
        valid_sets=[early],
        valid_names=["early_stop"],
        callbacks=[
            lgb.early_stopping(spec.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    iteration = int(booster.best_iteration or spec.num_boost_round)
    if not 1 <= iteration <= spec.num_boost_round:
        raise RuntimeError("queue-fill booster selected an invalid iteration")
    return booster, iteration


def train_queue_fill_lightgbm_model(
    *,
    training_panels: Sequence[PassiveFillSurvivalPanel],
    early_stop_panels: Sequence[PassiveFillSurvivalPanel],
    calibration_panels: Sequence[PassiveFillSurvivalPanel],
    spec: QueueFillLightGBMSpec = QueueFillLightGBMSpec(),
    compute_backend: str = "auto",
    seed: int = QUEUE_FILL_SEEDS[0],
    progress: Callable[[int, int], None] | None = None,
) -> TrainedQueueFillLightGBMModel:
    """Train three shared conditional hazards on chronological BTC/ETH/SOL roles."""

    train = _ordered_panels(training_panels, role="training")
    early = _ordered_panels(early_stop_panels, role="early-stop")
    calibration = _ordered_panels(calibration_panels, role="calibration")
    feature_names = train[0].feature_names
    if any(panel.feature_names != feature_names for panel in (*early, *calibration)):
        raise ValueError("queue-fill role feature contracts drifted")
    source_by_role = tuple(
        tuple((panel.symbol, panel.source_dataset_sha256) for panel in panels)
        for panels in (train, early, calibration)
    )
    if source_by_role[0] != source_by_role[1] or source_by_role[0] != source_by_role[2]:
        raise ValueError("queue-fill source datasets drifted across roles")
    training_end = (
        max(panel.source_last_decision_time_ms for panel in train)
        + PASSIVE_FILL_BUCKETS_MS[-1]
    )
    early_start = min(panel.source_first_decision_time_ms for panel in early)
    early_end = (
        max(panel.source_last_decision_time_ms for panel in early)
        + PASSIVE_FILL_BUCKETS_MS[-1]
    )
    calibration_start = min(
        panel.source_first_decision_time_ms for panel in calibration
    )
    if not training_end < early_start or not early_end < calibration_start:
        raise ValueError("queue-fill chronological roles overlap")
    support_by_role = tuple(_role_support(panels) for panels in (train, early, calibration))
    support = tuple(
        SymbolHazardSupport(
            symbol=symbol,
            training=support_by_role[0][symbol],
            early_stop=support_by_role[1][symbol],
            calibration=support_by_role[2][symbol],
        )
        for symbol in sorted(QUEUE_FILL_SYMBOLS)
    )
    _validate_support(support, spec)
    selected_seed = int(seed)
    if selected_seed not in QUEUE_FILL_SEEDS:
        raise ValueError("queue-fill seed is outside the frozen ensemble")
    backend, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        selected_seed,
        reproducible=True,
    )
    if backend_kind == "opencl" and (
        spec.gpu_use_dp_required is not True or backend.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("queue-fill LightGBM OpenCL FP64 accumulation is required")
    common: dict[str, object] = {
        **backend,
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
    model_strings: list[str] = []
    best_iterations: list[int] = []
    calibrations: list[tuple[float, float]] = []
    for head in range(3):
        if progress is not None:
            progress(head + 1, 3)
        x_train, y_train, train_weight = _risk_data(train, head)
        x_early, y_early, early_weight = _risk_data(early, head)
        x_calibration, y_calibration, calibration_weight = _risk_data(
            calibration, head
        )
        booster, iteration = _train_head(
            x_train=x_train,
            y_train=y_train,
            train_weight=train_weight,
            x_early=x_early,
            y_early=y_early,
            early_weight=early_weight,
            parameters=common,
            feature_names=feature_names,
            spec=spec,
        )
        raw_calibration = np.asarray(
            booster.predict(x_calibration, num_iteration=iteration),
            dtype=np.float64,
        )
        calibrations.append(
            fit_platt_scaling(
                raw_calibration,
                y_calibration,
                calibration_weight,
            )
        )
        best_iterations.append(iteration)
        model_strings.append(booster.model_to_string(num_iteration=iteration))
    baseline = np.mean(
        np.stack([fill_bucket_prevalence(panel) for panel in train], axis=0),
        axis=0,
    )
    provisional = TrainedQueueFillLightGBMModel(
        schema_version=QUEUE_FILL_LIGHTGBM_SCHEMA_VERSION,
        model_family=QUEUE_FILL_MODEL_FAMILY,
        spec=spec,
        feature_names=feature_names,
        source_dataset_sha256_by_symbol=source_by_role[0],
        training_panel_sha256_by_symbol=tuple(
            (panel.symbol, panel.panel_sha256) for panel in train
        ),
        early_stop_panel_sha256_by_symbol=tuple(
            (panel.symbol, panel.panel_sha256) for panel in early
        ),
        calibration_panel_sha256_by_symbol=tuple(
            (panel.symbol, panel.panel_sha256) for panel in calibration
        ),
        training_end_ms=training_end,
        early_stop_start_ms=early_start,
        early_stop_end_ms=early_end,
        calibration_start_ms=calibration_start,
        class_support=support,
        baseline_bucket_probabilities=tuple(float(value) for value in baseline),
        probability_calibration=tuple(calibrations),
        backend_requested=str(compute_backend),
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        seed=selected_seed,
        best_iterations=tuple(best_iterations),
        model_strings=tuple(model_strings),
        model_sha256="",
    )
    model = TrainedQueueFillLightGBMModel(
        **{**provisional.__dict__, "model_sha256": _model_sha256(provisional)}
    )
    _validate_model(model, reload=False)
    return model


def predict_queue_fill_lightgbm_model(
    model: QueueFillLightGBMArtifact,
    panel: PassiveFillSurvivalPanel,
) -> QueueFillPredictionBatch:
    """Predict calibrated fill-time probabilities for one symbol panel."""

    if isinstance(model, TrainedQueueFillLightGBMEnsemble):
        _validate_ensemble(model, reload=False)
        predictions = tuple(
            predict_queue_fill_lightgbm_model(member, panel)
            for member in model.members
        )
        averaged = np.mean(
            np.stack(
                [prediction.bucket_probabilities for prediction in predictions],
                axis=0,
            ),
            axis=0,
        )
        hazards = np.zeros((panel.rows, 3), dtype=np.float64)
        hazards[:, 0] = averaged[:, 0]
        remaining = 1.0 - averaged[:, 0]
        np.divide(
            averaged[:, 1],
            remaining,
            out=hazards[:, 1],
            where=remaining > 0.0,
        )
        remaining -= averaged[:, 1]
        np.divide(
            averaged[:, 2],
            remaining,
            out=hazards[:, 2],
            where=remaining > 0.0,
        )
        hazards = np.clip(hazards, 0.0, 1.0)
        buckets = hazards_to_bucket_probabilities(hazards)
        fill_probability = np.sum(buckets[:, :3], axis=1)
        for array in (hazards, buckets, fill_probability):
            array.setflags(write=False)
        return QueueFillPredictionBatch(
            source_action_feature_sha256=panel.source_action_feature_sha256,
            source_panel_sha256=panel.panel_sha256,
            model_sha256=model.model_sha256,
            symbol=panel.symbol,
            event_index=panel.event_index,
            decision_time_ms=panel.decision_time_ms,
            action_side=panel.action_side,
            hazard_probabilities=hazards,
            bucket_probabilities=buckets,
            fill_probability_15s=fill_probability,
        )
    _validate_model(model, reload=False)
    validate_passive_fill_survival_panel(panel)
    source = dict(model.source_dataset_sha256_by_symbol)
    if (
        panel.symbol not in source
        or panel.source_dataset_sha256 != source[panel.symbol]
        or panel.feature_names != model.feature_names
    ):
        raise ValueError("queue-fill prediction panel drifted")
    hazards = np.empty((panel.rows, 3), dtype=np.float64)
    try:
        for head in range(3):
            booster = lgb.Booster(model_str=model.model_strings[head])
            raw = np.asarray(
                booster.predict(
                    panel.features,
                    num_iteration=model.best_iterations[head],
                ),
                dtype=np.float64,
            )
            hazards[:, head] = apply_platt_scaling(
                raw,
                model.probability_calibration[head],
            )
    except lgb.basic.LightGBMError as exc:
        raise ValueError("queue-fill booster payload cannot be reloaded") from exc
    buckets = hazards_to_bucket_probabilities(hazards)
    fill_probability = np.sum(buckets[:, :3], axis=1)
    retained = (
        np.asarray(panel.event_index),
        np.asarray(panel.decision_time_ms),
        np.asarray(panel.action_side),
        hazards,
        buckets,
        fill_probability,
    )
    for array in retained:
        array.setflags(write=False)
    return QueueFillPredictionBatch(
        source_action_feature_sha256=panel.source_action_feature_sha256,
        source_panel_sha256=panel.panel_sha256,
        model_sha256=model.model_sha256,
        symbol=panel.symbol,
        event_index=panel.event_index,
        decision_time_ms=panel.decision_time_ms,
        action_side=panel.action_side,
        hazard_probabilities=hazards,
        bucket_probabilities=buckets,
        fill_probability_15s=fill_probability,
    )


def save_queue_fill_lightgbm_model(
    path: str | Path,
    model: TrainedQueueFillLightGBMModel,
) -> None:
    _validate_model(model, reload=True)
    payload = asdict(model)
    encoded = _canonical_json(payload).encode("utf-8")
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("queue-fill model artifact is too large")
    write_json_atomic(Path(path), payload, indent=None, sort_keys=True)


def _support_from_payload(value: object) -> tuple[SymbolHazardSupport, ...]:
    if not isinstance(value, list):
        raise ValueError("queue-fill class support payload is invalid")
    output: list[SymbolHazardSupport] = []
    try:
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("queue-fill class support payload is invalid")
            symbol = item["symbol"]
            if not isinstance(symbol, str):
                raise ValueError("queue-fill class support payload is invalid")
            roles: dict[str, tuple[tuple[int, int], ...]] = {}
            for role in ("training", "early_stop", "calibration"):
                raw_role = item[role]
                if not isinstance(raw_role, list) or len(raw_role) != 3:
                    raise ValueError("queue-fill class support payload is invalid")
                parsed_role: list[tuple[int, int]] = []
                for pair in raw_role:
                    if (
                        not isinstance(pair, list)
                        or len(pair) != 2
                        or any(
                            isinstance(count, bool) or not isinstance(count, int)
                            for count in pair
                        )
                    ):
                        raise ValueError("queue-fill class support payload is invalid")
                    parsed_role.append((pair[0], pair[1]))
                roles[role] = tuple(parsed_role)
            output.append(
                SymbolHazardSupport(
                    symbol=symbol,
                    training=roles["training"],
                    early_stop=roles["early_stop"],
                    calibration=roles["calibration"],
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("queue-fill class support payload is invalid") from exc
    return tuple(output)


def load_queue_fill_lightgbm_model(path: str | Path) -> TrainedQueueFillLightGBMModel:
    source = Path(path)
    try:
        size = source.stat().st_size
        if size <= 0 or size > _MAX_ARTIFACT_BYTES or not source.is_file():
            raise ValueError("queue-fill model artifact size is invalid")
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("queue-fill model artifact is unreadable") from exc
    expected = {field.name for field in fields(TrainedQueueFillLightGBMModel)}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("queue-fill model artifact fields drifted")
    if not isinstance(payload.get("spec"), dict):
        raise ValueError("queue-fill model specification payload is invalid")
    try:
        payload["spec"] = QueueFillLightGBMSpec(**payload["spec"])
        payload["feature_names"] = tuple(payload["feature_names"])
        for field_name in (
            "source_dataset_sha256_by_symbol",
            "training_panel_sha256_by_symbol",
            "early_stop_panel_sha256_by_symbol",
            "calibration_panel_sha256_by_symbol",
        ):
            payload[field_name] = tuple(tuple(value) for value in payload[field_name])
        payload["class_support"] = _support_from_payload(payload["class_support"])
        payload["baseline_bucket_probabilities"] = tuple(
            float(value) for value in payload["baseline_bucket_probabilities"]
        )
        payload["probability_calibration"] = tuple(
            tuple(float(value) for value in calibration)
            for calibration in payload["probability_calibration"]
        )
        payload["best_iterations"] = tuple(payload["best_iterations"])
        payload["model_strings"] = tuple(payload["model_strings"])
        model = TrainedQueueFillLightGBMModel(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("queue-fill model artifact payload is invalid") from exc
    _validate_model(model, reload=True)
    return model


__all__ = [
    "QUEUE_FILL_ENSEMBLE_SCHEMA_VERSION",
    "QUEUE_FILL_LIGHTGBM_SCHEMA_VERSION",
    "QUEUE_FILL_MODEL_FAMILY",
    "QUEUE_FILL_SEEDS",
    "QUEUE_FILL_SYMBOLS",
    "QueueFillLightGBMArtifact",
    "QueueFillLightGBMSpec",
    "QueueFillPredictionBatch",
    "SymbolHazardSupport",
    "TrainedQueueFillLightGBMEnsemble",
    "TrainedQueueFillLightGBMModel",
    "build_queue_fill_lightgbm_ensemble",
    "load_queue_fill_lightgbm_model",
    "predict_queue_fill_lightgbm_model",
    "save_queue_fill_lightgbm_model",
    "train_queue_fill_lightgbm_model",
    "validate_queue_fill_lightgbm_model",
]
