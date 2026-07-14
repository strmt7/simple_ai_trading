"""Multi-view after-cost payoff models and bounded research replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from .cross_asset_cost_data import MINUTE_MS, SYMBOLS
from .lightgbm_backend import lightgbm_backend_parameters
from .stop_time_payoff_data import STOP_EVENT, StopTimePayoffDataset
from .storage import write_json_atomic


BOUNDED_ALPHA_MODEL_SCHEMA_VERSION = "bounded-alpha-lightgbm-model-v1"
VIEW_IDS = (
    "raw_uniform",
    "risk_normalized_uniform",
    "risk_normalized_recency_180d",
)
SIDE_NAMES = ("long", "short")
DAY_MS = 24 * 60 * MINUTE_MS
HOUR_MS = 60 * MINUTE_MS
_MODEL_FAMILY = "side_specific_multi_view_after_cost_payoff"
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class BoundedAlphaSpec:
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = 6
    min_data_in_leaf: int = 256
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 1
    lambda_l1: float = 1.0
    lambda_l2: float = 1.0
    max_bin: int = 63
    maximum_boosting_rounds: int = 1200
    early_stopping_rounds: int = 100
    recency_half_life_days: float = 180.0
    gpu_use_dp_required: bool = True

    def validate(self) -> None:
        numeric = np.asarray(
            [
                self.learning_rate,
                self.feature_fraction,
                self.bagging_fraction,
                self.lambda_l1,
                self.lambda_l2,
                self.recency_half_life_days,
            ],
            dtype=np.float64,
        )
        if (
            not np.isfinite(numeric).all()
            or not 0.0 < self.learning_rate <= 0.25
            or not 2 <= self.num_leaves <= 255
            or not 1 <= self.max_depth <= 16
            or not 32 <= self.min_data_in_leaf <= 65_536
            or not 0.0 < self.feature_fraction <= 1.0
            or not 0.0 < self.bagging_fraction <= 1.0
            or not 0 <= self.bagging_freq <= 100
            or self.lambda_l1 < 0.0
            or self.lambda_l2 < 0.0
            or not 31 <= self.max_bin <= 255
            or not 10 <= self.maximum_boosting_rounds <= 10_000
            or not 5 <= self.early_stopping_rounds < self.maximum_boosting_rounds
            or self.recency_half_life_days <= 0.0
            or self.gpu_use_dp_required is not True
        ):
            raise ValueError("bounded alpha model specification is invalid")


@dataclass(frozen=True)
class BoundedAlphaModel:
    schema_version: str
    model_family: str
    treatment_id: str
    view_id: str
    seed: int
    feature_names: tuple[str, ...]
    source_dataset_sha256: str
    payoff_dataset_sha256: str
    spec: BoundedAlphaSpec
    backend_requested: str
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    iteration_training_rows: int
    iteration_selection_rows: int
    final_refit_rows: int
    best_iterations: Mapping[str, int]
    iteration_selection_mae: Mapping[str, float]
    iteration_selection_constant_mae: Mapping[str, float]
    iteration_selection_mae_skill: Mapping[str, float]
    final_target_mean: Mapping[str, float]
    model_strings: Mapping[str, str]
    reload_max_abs_prediction_error_bps: float
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    leverage_applied: bool = False

    def asdict(self) -> dict[str, object]:
        value = asdict(self)
        value["feature_names"] = list(self.feature_names)
        return value


@dataclass(frozen=True)
class ConsensusDecisions:
    actions: np.ndarray
    score_bps: np.ndarray
    model_eligible: np.ndarray
    liquidity_eligible: np.ndarray
    volatility_eligible: np.ndarray
    view_median_bps: np.ndarray
    seed_positive_fraction: np.ndarray


@dataclass(frozen=True)
class TradePlan:
    decision_index: np.ndarray
    decision_time_ms: np.ndarray
    entry_time_ms: np.ndarray
    exit_time_ms: np.ndarray
    symbol_index: np.ndarray
    side: np.ndarray
    size_fraction: np.ndarray
    score_bps: np.ndarray
    stop_bps: np.ndarray
    event_code: np.ndarray
    stress_net_payoff_bps: np.ndarray
    blocked_daily_loss_decisions: int
    blocked_cooldown_decisions: int
    signal_count: int

    @property
    def closed_trades(self) -> int:
        return int(self.side.size)


@dataclass(frozen=True)
class ReplayResult:
    scenario: str
    round_trip_execution_charge_bps: float
    interval_timestamps_ms: np.ndarray
    hourly_return_fraction: np.ndarray
    trade_return_fraction: np.ndarray
    metrics: Mapping[str, object]


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


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _model_payload(model: BoundedAlphaModel) -> dict[str, object]:
    payload = model.asdict()
    payload.pop("model_sha256")
    return payload


def _model_sha256(model: BoundedAlphaModel) -> str:
    return _sha256(_model_payload(model))


def _validate_model(model: BoundedAlphaModel, *, reload: bool) -> None:
    sides = set(SIDE_NAMES)
    if (
        model.schema_version != BOUNDED_ALPHA_MODEL_SCHEMA_VERSION
        or model.model_family != _MODEL_FAMILY
        or not model.treatment_id.strip()
        or model.view_id not in VIEW_IDS
        or not model.feature_names
        or len(set(model.feature_names)) != len(model.feature_names)
        or any(not name.strip() for name in model.feature_names)
        or not _is_sha256(model.source_dataset_sha256)
        or not _is_sha256(model.payoff_dataset_sha256)
        or not _is_sha256(model.model_sha256)
        or model.backend_kind not in {"opencl", "cpu"}
        or model.iteration_training_rows < 1_024
        or model.iteration_selection_rows < 512
        or model.final_refit_rows <= model.iteration_training_rows
        or set(model.best_iterations) != sides
        or set(model.iteration_selection_mae) != sides
        or set(model.iteration_selection_constant_mae) != sides
        or set(model.iteration_selection_mae_skill) != sides
        or set(model.final_target_mean) != sides
        or set(model.model_strings) != sides
        or not math.isfinite(model.reload_max_abs_prediction_error_bps)
        or model.reload_max_abs_prediction_error_bps < 0.0
        or model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.leverage_applied
        or _model_sha256(model) != model.model_sha256
    ):
        raise ValueError("bounded alpha model contract is invalid")
    model.spec.validate()
    for side in SIDE_NAMES:
        if (
            not 1 <= int(model.best_iterations[side]) <= model.spec.maximum_boosting_rounds
            or not math.isfinite(float(model.iteration_selection_mae[side]))
            or float(model.iteration_selection_mae[side]) < 0.0
            or not math.isfinite(float(model.iteration_selection_constant_mae[side]))
            or float(model.iteration_selection_constant_mae[side]) <= 0.0
            or not math.isfinite(float(model.iteration_selection_mae_skill[side]))
            or not math.isfinite(float(model.final_target_mean[side]))
            or not model.model_strings[side].strip()
        ):
            raise ValueError("bounded alpha side contract is invalid")
    if reload:
        try:
            for side in SIDE_NAMES:
                booster = lgb.Booster(model_str=model.model_strings[side])
                if booster.num_feature() != len(model.feature_names):
                    raise ValueError("bounded alpha booster feature count drifted")
        except lgb.basic.LightGBMError as exc:
            raise ValueError("bounded alpha booster cannot be reloaded") from exc


def _flatten_source(
    features: np.ndarray,
    timestamps_ms: np.ndarray,
    stop_bps: np.ndarray,
    long_target_bps: np.ndarray,
    short_target_bps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    matrix = np.asarray(features, dtype=np.float32)
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    stops = np.asarray(stop_bps, dtype=np.float32)
    targets = {
        "long": np.asarray(long_target_bps, dtype=np.float32),
        "short": np.asarray(short_target_bps, dtype=np.float32),
    }
    expected = (timestamps.size, len(SYMBOLS))
    if (
        matrix.ndim != 3
        or matrix.shape[:2] != expected
        or timestamps.ndim != 1
        or timestamps.size == 0
        or np.any(np.diff(timestamps) <= 0)
        or stops.shape != expected
        or any(value.shape != expected for value in targets.values())
        or not np.isfinite(matrix).all()
        or not np.isfinite(stops).all()
        or np.any(stops <= 0.0)
        or not all(np.isfinite(value).all() for value in targets.values())
    ):
        raise ValueError("bounded alpha training source is invalid")
    rows = timestamps.size * len(SYMBOLS)
    return (
        matrix.reshape(rows, matrix.shape[-1]),
        np.repeat(timestamps, len(SYMBOLS)),
        stops.reshape(rows),
        {side: values.reshape(rows) for side, values in targets.items()},
    )


def _role_indexes(mask: np.ndarray, timestamps: int, label: str) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.shape != (timestamps,) or np.count_nonzero(values) == 0:
        raise ValueError(f"bounded alpha role is invalid: {label}")
    return np.flatnonzero(np.repeat(values, len(SYMBOLS)))


def _target_for_view(
    view_id: str,
    target_bps: np.ndarray,
    stop_bps: np.ndarray,
) -> np.ndarray:
    if view_id == "raw_uniform":
        return target_bps
    if view_id in VIEW_IDS[1:]:
        return target_bps / stop_bps
    raise KeyError(view_id)


def _sample_weight(
    view_id: str,
    timestamps_ms: np.ndarray,
    indexes: np.ndarray,
    half_life_days: float,
) -> np.ndarray | None:
    if view_id != "risk_normalized_recency_180d":
        return None
    selected = timestamps_ms[indexes].astype(np.float64)
    age_days = (float(np.max(selected)) - selected) / float(DAY_MS)
    weights = np.exp2(-age_days / half_life_days)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("bounded alpha recency weights are invalid")
    return weights.astype(np.float32)


def _parameters(
    spec: BoundedAlphaSpec,
    compute_backend: str,
    seed: int,
) -> tuple[dict[str, object], str, str]:
    backend, kind, device = lightgbm_backend_parameters(
        compute_backend,
        seed,
        reproducible=True,
    )
    if kind == "opencl" and (
        not spec.gpu_use_dp_required or backend.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("bounded alpha OpenCL FP64 accumulation is required")
    return (
        {
            **backend,
            "objective": "regression_l1",
            "metric": "l1",
            "learning_rate": spec.learning_rate,
            "num_leaves": spec.num_leaves,
            "max_depth": spec.max_depth,
            "min_data_in_leaf": spec.min_data_in_leaf,
            "feature_fraction": spec.feature_fraction,
            "bagging_fraction": spec.bagging_fraction,
            "bagging_freq": spec.bagging_freq,
            "lambda_l1": spec.lambda_l1,
            "lambda_l2": spec.lambda_l2,
            "max_bin": spec.max_bin,
            "feature_pre_filter": False,
        },
        kind,
        device,
    )


def train_bounded_alpha_model(
    *,
    treatment_id: str,
    view_id: str,
    seed: int,
    features: np.ndarray,
    feature_names: Sequence[str],
    timestamps_ms: np.ndarray,
    stop_bps: np.ndarray,
    long_target_bps: np.ndarray,
    short_target_bps: np.ndarray,
    role_masks: Mapping[str, np.ndarray],
    long_exit_time_ms: np.ndarray,
    short_exit_time_ms: np.ndarray,
    source_dataset_sha256: str,
    payoff_dataset_sha256: str,
    spec: BoundedAlphaSpec,
    compute_backend: str,
    progress: ProgressCallback | None = None,
) -> BoundedAlphaModel:
    """Select boosting length chronologically, then refit through June 2024."""

    spec.validate()
    if view_id not in VIEW_IDS or not treatment_id.strip():
        raise ValueError("bounded alpha treatment or view is invalid")
    names = tuple(str(name) for name in feature_names)
    if not names or len(names) != np.asarray(features).shape[-1]:
        raise ValueError("bounded alpha feature names are invalid")
    flat_x, flat_time, flat_stop, flat_targets = _flatten_source(
        features,
        timestamps_ms,
        stop_bps,
        long_target_bps,
        short_target_bps,
    )
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    required_roles = (
        "iteration_training",
        "iteration_selection",
        "final_refit",
        "policy_development",
    )
    if set(role_masks) != set(required_roles):
        raise ValueError("bounded alpha role set is invalid")
    indexes = {
        role: _role_indexes(role_masks[role], timestamps.size, role)
        for role in required_roles
    }
    if (
        indexes["iteration_training"][-1] >= indexes["iteration_selection"][0]
        or indexes["iteration_selection"][-1] >= indexes["final_refit"][-1]
        or indexes["final_refit"][-1] >= indexes["policy_development"][0]
    ):
        raise ValueError("bounded alpha roles are not chronological")
    maximum_exit = np.maximum(long_exit_time_ms, short_exit_time_ms)
    if (
        np.max(maximum_exit[np.asarray(role_masks["iteration_training"], dtype=bool)])
        >= np.min(timestamps[np.asarray(role_masks["iteration_selection"], dtype=bool)])
        or np.max(maximum_exit[np.asarray(role_masks["final_refit"], dtype=bool)])
        >= np.min(timestamps[np.asarray(role_masks["policy_development"], dtype=bool)])
    ):
        raise ValueError("bounded alpha labels cross a chronological boundary")

    parameters, backend_kind, backend_device = _parameters(
        spec, compute_backend, int(seed)
    )
    train_index = indexes["iteration_training"]
    selection_index = indexes["iteration_selection"]
    final_index = indexes["final_refit"]
    train_weight = _sample_weight(
        view_id,
        flat_time,
        train_index,
        spec.recency_half_life_days,
    )
    final_weight = _sample_weight(
        view_id,
        flat_time,
        final_index,
        spec.recency_half_life_days,
    )
    best_iterations: dict[str, int] = {}
    selection_mae: dict[str, float] = {}
    constant_mae: dict[str, float] = {}
    selection_skill: dict[str, float] = {}
    final_means: dict[str, float] = {}
    model_strings: dict[str, str] = {}
    reload_error = 0.0
    prediction_probe = indexes["policy_development"][: min(2_048, len(indexes["policy_development"]))]
    for side in SIDE_NAMES:
        transformed = _target_for_view(view_id, flat_targets[side], flat_stop)
        training = lgb.Dataset(
            flat_x[train_index],
            label=transformed[train_index],
            weight=train_weight,
            feature_name=list(names),
            free_raw_data=False,
        )
        selection = lgb.Dataset(
            flat_x[selection_index],
            label=transformed[selection_index],
            reference=training,
            feature_name=list(names),
            free_raw_data=False,
        )
        if progress is not None:
            progress(
                "bounded_alpha_iteration_fit",
                {
                    "treatment_id": treatment_id,
                    "view_id": view_id,
                    "seed": int(seed),
                    "side": side,
                    "status": "started",
                },
            )
        selected_booster = lgb.train(
            parameters,
            training,
            num_boost_round=spec.maximum_boosting_rounds,
            valid_sets=[selection],
            valid_names=["iteration_selection"],
            callbacks=[
                lgb.early_stopping(spec.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        iteration = max(
            1,
            int(selected_booster.best_iteration or selected_booster.current_iteration()),
        )
        selection_prediction = np.asarray(
            selected_booster.predict(flat_x[selection_index], num_iteration=iteration),
            dtype=np.float64,
        )
        selection_truth = transformed[selection_index].astype(np.float64)
        model_mae = float(np.mean(np.abs(selection_truth - selection_prediction)))
        baseline = float(
            np.median(transformed[train_index].astype(np.float64))
        )
        baseline_mae = float(np.mean(np.abs(selection_truth - baseline)))
        if baseline_mae <= 0.0:
            raise ValueError("bounded alpha constant baseline is degenerate")
        best_iterations[side] = iteration
        selection_mae[side] = model_mae
        constant_mae[side] = baseline_mae
        selection_skill[side] = 1.0 - model_mae / baseline_mae

        final_training = lgb.Dataset(
            flat_x[final_index],
            label=transformed[final_index],
            weight=final_weight,
            feature_name=list(names),
            free_raw_data=False,
        )
        final_booster = lgb.train(
            parameters,
            final_training,
            num_boost_round=iteration,
            callbacks=[lgb.log_evaluation(0)],
        )
        model_string = final_booster.model_to_string(num_iteration=iteration)
        direct_prediction = np.asarray(
            final_booster.predict(flat_x[prediction_probe], num_iteration=iteration),
            dtype=np.float64,
        )
        reloaded = lgb.Booster(model_str=model_string)
        reloaded_prediction = np.asarray(
            reloaded.predict(flat_x[prediction_probe], num_iteration=iteration),
            dtype=np.float64,
        )
        scale = flat_stop[prediction_probe] if view_id != "raw_uniform" else 1.0
        reload_error = max(
            reload_error,
            float(np.max(np.abs(direct_prediction - reloaded_prediction) * scale)),
        )
        final_means[side] = float(
            np.average(
                transformed[final_index].astype(np.float64),
                weights=final_weight,
            )
            if final_weight is not None
            else np.mean(transformed[final_index], dtype=np.float64)
        )
        model_strings[side] = model_string
        if progress is not None:
            progress(
                "bounded_alpha_iteration_fit",
                {
                    "treatment_id": treatment_id,
                    "view_id": view_id,
                    "seed": int(seed),
                    "side": side,
                    "status": "complete",
                    "best_iteration": iteration,
                    "selection_mae_skill": selection_skill[side],
                },
            )
    provisional = BoundedAlphaModel(
        schema_version=BOUNDED_ALPHA_MODEL_SCHEMA_VERSION,
        model_family=_MODEL_FAMILY,
        treatment_id=treatment_id,
        view_id=view_id,
        seed=int(seed),
        feature_names=names,
        source_dataset_sha256=source_dataset_sha256,
        payoff_dataset_sha256=payoff_dataset_sha256,
        spec=spec,
        backend_requested=compute_backend,
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        iteration_training_rows=len(train_index),
        iteration_selection_rows=len(selection_index),
        final_refit_rows=len(final_index),
        best_iterations=best_iterations,
        iteration_selection_mae=selection_mae,
        iteration_selection_constant_mae=constant_mae,
        iteration_selection_mae_skill=selection_skill,
        final_target_mean=final_means,
        model_strings=model_strings,
        reload_max_abs_prediction_error_bps=reload_error,
        model_sha256="",
    )
    model = BoundedAlphaModel(
        **{**provisional.__dict__, "model_sha256": _model_sha256(provisional)}
    )
    _validate_model(model, reload=True)
    return model


def predict_bounded_alpha_model(
    model: BoundedAlphaModel,
    features: np.ndarray,
    stop_bps: np.ndarray,
) -> np.ndarray:
    """Return long/short expected stress payoff in basis points."""

    _validate_model(model, reload=False)
    matrix = np.asarray(features, dtype=np.float32)
    stops = np.asarray(stop_bps, dtype=np.float64)
    if (
        matrix.ndim != 3
        or matrix.shape[:2] != stops.shape
        or matrix.shape[-1] != len(model.feature_names)
        or not np.isfinite(matrix).all()
        or not np.isfinite(stops).all()
        or np.any(stops <= 0.0)
    ):
        raise ValueError("bounded alpha prediction source is invalid")
    flat = matrix.reshape(-1, matrix.shape[-1])
    output = np.empty((flat.shape[0], len(SIDE_NAMES)), dtype=np.float64)
    for side_index, side in enumerate(SIDE_NAMES):
        booster = lgb.Booster(model_str=model.model_strings[side])
        values = np.asarray(
            booster.predict(flat, num_iteration=model.best_iterations[side]),
            dtype=np.float64,
        )
        if model.view_id != "raw_uniform":
            values *= stops.reshape(-1)
        output[:, side_index] = values
    if not np.isfinite(output).all():
        raise ValueError("bounded alpha prediction is nonfinite")
    return output.reshape(matrix.shape[0], matrix.shape[1], len(SIDE_NAMES))


def save_bounded_alpha_model(path: str | Path, model: BoundedAlphaModel) -> None:
    _validate_model(model, reload=True)
    payload = model.asdict()
    encoded = _canonical_json(payload).encode("utf-8")
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("bounded alpha model artifact is too large")
    write_json_atomic(Path(path), payload, indent=None, sort_keys=True)


def load_bounded_alpha_model(path: str | Path) -> BoundedAlphaModel:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("bounded alpha model artifact is missing or oversized")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bounded alpha model artifact must be an object")
    expected = {field.name for field in fields(BoundedAlphaModel)}
    if set(payload) != expected or not isinstance(payload.get("spec"), dict):
        raise ValueError("bounded alpha model artifact fields drifted")
    payload["spec"] = BoundedAlphaSpec(**payload["spec"])
    payload["feature_names"] = tuple(payload["feature_names"])
    model = BoundedAlphaModel(**payload)
    _validate_model(model, reload=True)
    return model


def consensus_decisions(
    seed_predictions_bps: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
) -> ConsensusDecisions:
    """Apply the frozen all-view preference and causal market-state gates."""

    predictions = np.asarray(seed_predictions_bps, dtype=np.float64)
    matrix = np.asarray(features, dtype=np.float64)
    names = tuple(feature_names)
    if (
        predictions.ndim != 5
        or predictions.shape[0] != len(VIEW_IDS)
        or predictions.shape[1] != 3
        or predictions.shape[2:4] != matrix.shape[:2]
        or predictions.shape[-1] != len(SIDE_NAMES)
        or matrix.ndim != 3
        or matrix.shape[-1] != len(names)
        or not np.isfinite(predictions).all()
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("bounded alpha consensus source is invalid")
    positions = {name: index for index, name in enumerate(names)}
    required = (
        "target_same_minute_of_week_liquidity_ratio",
        "target_quote_volume_vs_1440m_mean",
        "target_realized_volatility_60m_bps",
        "target_realized_volatility_1440m_bps",
    )
    if any(name not in positions for name in required):
        raise ValueError("bounded alpha consensus feature is missing")

    medians = np.median(predictions, axis=1)
    seed_positive = np.mean(predictions > 0.0, axis=1)
    long_eligible = (
        np.all(medians[..., 0] > 0.0, axis=0)
        & np.all(medians[..., 0] > medians[..., 1], axis=0)
        & np.all(seed_positive[..., 0] >= (2.0 / 3.0), axis=0)
    )
    short_eligible = (
        np.all(medians[..., 1] > 0.0, axis=0)
        & np.all(medians[..., 1] > medians[..., 0], axis=0)
        & np.all(seed_positive[..., 1] >= (2.0 / 3.0), axis=0)
    )
    if np.any(long_eligible & short_eligible):
        raise RuntimeError("bounded alpha consensus selected both sides")
    model_eligible = long_eligible | short_eligible
    liquidity = (
        matrix[..., positions[required[0]]] >= 0.5
    ) & (matrix[..., positions[required[1]]] >= 0.25)
    volatility = matrix[..., positions[required[2]]] <= (
        2.5 * matrix[..., positions[required[3]]]
    )
    eligible = model_eligible & liquidity & volatility
    actions = np.zeros(matrix.shape[:2], dtype=np.int8)
    actions[eligible & long_eligible] = 1
    actions[eligible & short_eligible] = -1
    score = np.zeros(matrix.shape[:2], dtype=np.float64)
    long_score = np.min(medians[..., 0], axis=0)
    short_score = np.min(medians[..., 1], axis=0)
    score[actions == 1] = long_score[actions == 1]
    score[actions == -1] = short_score[actions == -1]
    return ConsensusDecisions(
        actions=actions,
        score_bps=score,
        model_eligible=model_eligible,
        liquidity_eligible=liquidity,
        volatility_eligible=volatility,
        view_median_bps=np.moveaxis(medians, 0, 2),
        seed_positive_fraction=np.moveaxis(seed_positive, 0, 2),
    )


def _settle_pending(
    pending: list[dict[str, float | int]],
    *,
    through_time_ms: int,
    daily_pnl: dict[int, float],
    consecutive_losses: int,
    cooldown_until_ms: int,
    cooldown_hours: int,
) -> tuple[int, int]:
    settled = sorted(
        (item for item in pending if int(item["exit_time_ms"]) <= through_time_ms),
        key=lambda item: (int(item["exit_time_ms"]), int(item["symbol_index"])),
    )
    for item in settled:
        pending.remove(item)
        exit_time = int(item["exit_time_ms"])
        return_fraction = float(item["stress_return_fraction"])
        day = exit_time // DAY_MS
        daily_pnl[day] = daily_pnl.get(day, 0.0) + return_fraction
        if return_fraction < 0.0:
            consecutive_losses += 1
            if consecutive_losses >= 3:
                cooldown_until_ms = max(
                    cooldown_until_ms,
                    exit_time + cooldown_hours * HOUR_MS,
                )
                consecutive_losses = 0
        else:
            consecutive_losses = 0
    return consecutive_losses, cooldown_until_ms


def build_trade_plan(
    payoff: StopTimePayoffDataset,
    decisions: ConsensusDecisions,
    interval_mask: np.ndarray,
    *,
    per_position_stop_risk_fraction: float = 0.001,
    aggregate_stop_risk_fraction: float = 0.0015,
    maximum_symbol_notional_fraction: float = 1.0 / 3.0,
    maximum_gross_fraction: float = 0.5,
    maximum_net_directional_fraction: float = 1.0 / 3.0,
    daily_loss_limit_fraction: float = 0.005,
    cooldown_hours: int = 6,
) -> TradePlan:
    """Create one fixed trade plan using stress outcomes for safety state."""

    mask = np.asarray(interval_mask, dtype=bool)
    shape = (payoff.timestamps, len(SYMBOLS))
    numeric = np.asarray(
        [
            per_position_stop_risk_fraction,
            aggregate_stop_risk_fraction,
            maximum_symbol_notional_fraction,
            maximum_gross_fraction,
            maximum_net_directional_fraction,
            daily_loss_limit_fraction,
        ],
        dtype=np.float64,
    )
    if (
        mask.shape != (payoff.timestamps,)
        or decisions.actions.shape != shape
        or decisions.score_bps.shape != shape
        or not np.isfinite(numeric).all()
        or np.any(numeric <= 0.0)
        or aggregate_stop_risk_fraction < per_position_stop_risk_fraction
        or cooldown_hours <= 0
    ):
        raise ValueError("bounded alpha trade-plan contract is invalid")
    selected_indexes = np.flatnonzero(mask)
    if selected_indexes.size == 0:
        raise ValueError("bounded alpha trade-plan interval is empty")

    trades: list[dict[str, float | int]] = []
    pending: list[dict[str, float | int]] = []
    daily_pnl: dict[int, float] = {}
    consecutive_losses = 0
    cooldown_until_ms = 0
    blocked_daily = 0
    blocked_cooldown = 0
    signal_count = 0
    for timestamp_index in selected_indexes:
        decision_time = int(payoff.timestamps_ms[timestamp_index])
        entry_time = decision_time + MINUTE_MS
        consecutive_losses, cooldown_until_ms = _settle_pending(
            pending,
            through_time_ms=entry_time,
            daily_pnl=daily_pnl,
            consecutive_losses=consecutive_losses,
            cooldown_until_ms=cooldown_until_ms,
            cooldown_hours=cooldown_hours,
        )
        actions = decisions.actions[timestamp_index].astype(np.int8, copy=True)
        signal_count += int(np.count_nonzero(actions))
        if not np.any(actions):
            continue
        current_day = entry_time // DAY_MS
        if daily_pnl.get(current_day, 0.0) <= -daily_loss_limit_fraction:
            blocked_daily += 1
            continue
        if entry_time < cooldown_until_ms:
            blocked_cooldown += 1
            continue
        stop = payoff.stop_bps[timestamp_index].astype(np.float64)
        raw_size = np.where(
            actions != 0,
            np.minimum(
                maximum_symbol_notional_fraction,
                per_position_stop_risk_fraction * 10_000.0 / stop,
            ),
            0.0,
        )
        total_stop_risk = float(np.sum(raw_size * stop / 10_000.0))
        gross = float(np.sum(raw_size))
        net = abs(float(np.sum(raw_size * actions)))
        scale = min(
            1.0,
            aggregate_stop_risk_fraction / max(total_stop_risk, 1e-15),
            maximum_gross_fraction / max(gross, 1e-15),
            maximum_net_directional_fraction / max(net, 1e-15),
        )
        sizes = raw_size * scale
        for symbol_index in np.flatnonzero(actions):
            side = int(actions[symbol_index])
            if side == 1:
                exit_time = int(payoff.long_exit_time_ms[timestamp_index, symbol_index])
                event_code = int(payoff.long_event_code[timestamp_index, symbol_index])
                net_bps = float(
                    payoff.long_net_payoff_bps[timestamp_index, symbol_index]
                )
            else:
                exit_time = int(payoff.short_exit_time_ms[timestamp_index, symbol_index])
                event_code = int(payoff.short_event_code[timestamp_index, symbol_index])
                net_bps = float(
                    payoff.short_net_payoff_bps[timestamp_index, symbol_index]
                )
            trade = {
                "decision_index": int(timestamp_index),
                "decision_time_ms": decision_time,
                "entry_time_ms": entry_time,
                "exit_time_ms": exit_time,
                "symbol_index": int(symbol_index),
                "side": side,
                "size_fraction": float(sizes[symbol_index]),
                "score_bps": float(decisions.score_bps[timestamp_index, symbol_index]),
                "stop_bps": float(stop[symbol_index]),
                "event_code": event_code,
                "stress_net_payoff_bps": net_bps,
                "stress_return_fraction": float(sizes[symbol_index] * net_bps / 10_000.0),
            }
            trades.append(trade)
            pending.append(trade)
    consecutive_losses, cooldown_until_ms = _settle_pending(
        pending,
        through_time_ms=np.iinfo(np.int64).max,
        daily_pnl=daily_pnl,
        consecutive_losses=consecutive_losses,
        cooldown_until_ms=cooldown_until_ms,
        cooldown_hours=cooldown_hours,
    )
    del consecutive_losses, cooldown_until_ms
    if pending:
        raise RuntimeError("bounded alpha replay left an unsettled position")

    def column(name: str, dtype: np.dtype[object]) -> np.ndarray:
        return np.asarray([trade[name] for trade in trades], dtype=dtype)

    return TradePlan(
        decision_index=column("decision_index", np.dtype(np.int64)),
        decision_time_ms=column("decision_time_ms", np.dtype(np.int64)),
        entry_time_ms=column("entry_time_ms", np.dtype(np.int64)),
        exit_time_ms=column("exit_time_ms", np.dtype(np.int64)),
        symbol_index=column("symbol_index", np.dtype(np.int8)),
        side=column("side", np.dtype(np.int8)),
        size_fraction=column("size_fraction", np.dtype(np.float64)),
        score_bps=column("score_bps", np.dtype(np.float64)),
        stop_bps=column("stop_bps", np.dtype(np.float64)),
        event_code=column("event_code", np.dtype(np.int8)),
        stress_net_payoff_bps=column(
            "stress_net_payoff_bps", np.dtype(np.float64)
        ),
        blocked_daily_loss_decisions=blocked_daily,
        blocked_cooldown_decisions=blocked_cooldown,
        signal_count=signal_count,
    )


def _drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.max(np.maximum(0.0, (peak - equity) / np.maximum(peak, 1e-12))))


def _profit_factor(returns: np.ndarray) -> float | None:
    gains = float(np.sum(returns[returns > 0.0]))
    losses = float(-np.sum(returns[returns < 0.0]))
    return None if losses <= 0.0 else gains / losses


def replay_trade_plan(
    payoff: StopTimePayoffDataset,
    plan: TradePlan,
    interval_mask: np.ndarray,
    *,
    scenario: str,
    round_trip_execution_charge_bps: float,
) -> ReplayResult:
    """Replay one fixed plan under a named execution-charge scenario."""

    mask = np.asarray(interval_mask, dtype=bool)
    if (
        scenario not in {"base", "stress"}
        or mask.shape != (payoff.timestamps,)
        or not math.isfinite(round_trip_execution_charge_bps)
        or round_trip_execution_charge_bps < 0.0
    ):
        raise ValueError("bounded alpha replay scenario is invalid")
    interval_indexes = np.flatnonzero(mask)
    index_position = {int(index): offset for offset, index in enumerate(interval_indexes)}
    cost_delta = (
        payoff.specification.round_trip_execution_charge_bps
        - round_trip_execution_charge_bps
    )
    trade_bps = plan.stress_net_payoff_bps + cost_delta
    trade_returns = plan.size_fraction * trade_bps / 10_000.0
    hourly = np.zeros(interval_indexes.size, dtype=np.float64)
    for trade_index, decision_index in enumerate(plan.decision_index):
        if int(decision_index) not in index_position:
            raise ValueError("bounded alpha trade falls outside its replay interval")
        hourly[index_position[int(decision_index)]] += trade_returns[trade_index]
    equity = 1.0 + np.cumsum(hourly)
    if np.any(equity <= 0.0):
        raise RuntimeError("bounded alpha fixed-capital replay exhausted equity")
    mean = float(np.mean(hourly))
    standard_deviation = float(np.std(hourly, ddof=1)) if hourly.size > 1 else 0.0
    downside = hourly[hourly < 0.0]
    downside_deviation = float(np.sqrt(np.mean(downside * downside))) if downside.size else 0.0
    annualizer = math.sqrt(24.0 * 365.0)
    symbol_pnl = {
        symbol: float(np.sum(trade_returns[plan.symbol_index == symbol_index]))
        for symbol_index, symbol in enumerate(SYMBOLS)
    }
    absolute_symbol_pnl = sum(abs(value) for value in symbol_pnl.values())
    holding_minutes = (plan.exit_time_ms - plan.entry_time_ms) / MINUTE_MS
    metrics: dict[str, object] = {
        "scenario": scenario,
        "closed_trades": plan.closed_trades,
        "signals_before_cooldowns": plan.signal_count,
        "blocked_daily_loss_decisions": plan.blocked_daily_loss_decisions,
        "blocked_cooldown_decisions": plan.blocked_cooldown_decisions,
        "active_days": int(np.unique(plan.decision_time_ms // DAY_MS).size),
        "total_return_fraction": float(np.sum(hourly)),
        "maximum_drawdown_fraction": _drawdown(equity),
        "profit_factor": _profit_factor(trade_returns),
        "win_rate": float(np.mean(trade_returns > 0.0)) if trade_returns.size else 0.0,
        "mean_trade_initial_capital_bps": float(np.mean(trade_returns) * 10_000.0)
        if trade_returns.size
        else 0.0,
        "mean_hourly_initial_capital_bps": mean * 10_000.0,
        "annualized_sharpe": mean / standard_deviation * annualizer
        if standard_deviation > 0.0
        else None,
        "annualized_sortino": mean / downside_deviation * annualizer
        if downside_deviation > 0.0
        else None,
        "gross_round_trip_turnover_fraction": float(
            2.0 * np.sum(plan.size_fraction)
        ),
        "maximum_position_fraction": float(np.max(plan.size_fraction))
        if plan.closed_trades
        else 0.0,
        "maximum_holding_minutes": float(np.max(holding_minutes))
        if plan.closed_trades
        else 0.0,
        "stop_loss_trades": int(np.count_nonzero(plan.event_code == STOP_EVENT)),
        "timeout_trades": int(np.count_nonzero(plan.event_code != STOP_EVENT)),
        "symbol_return_fraction": symbol_pnl,
        "maximum_single_symbol_fraction_of_absolute_net_pnl": max(
            (abs(value) / absolute_symbol_pnl for value in symbol_pnl.values()),
            default=0.0,
        )
        if absolute_symbol_pnl > 0.0
        else 0.0,
        "symbols_with_trades": int(np.unique(plan.symbol_index).size),
        "fixed_initial_capital": True,
        "profit_reinvestment": False,
        "leverage": 1.0,
    }
    return ReplayResult(
        scenario=scenario,
        round_trip_execution_charge_bps=round_trip_execution_charge_bps,
        interval_timestamps_ms=payoff.timestamps_ms[interval_indexes].copy(),
        hourly_return_fraction=hourly,
        trade_return_fraction=trade_returns,
        metrics=metrics,
    )


def block_bootstrap_mean_bps(
    hourly_return_fraction: np.ndarray,
    *,
    samples: int = 2_000,
    block_hours: int = 168,
    lower_quantile: float = 0.0125,
    seed: int = 55_091,
) -> dict[str, float | int]:
    values = np.asarray(hourly_return_fraction, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 24
        or not np.isfinite(values).all()
        or samples < 100
        or block_hours <= 0
        or not 0.0 < lower_quantile < 0.5
    ):
        raise ValueError("bounded alpha bootstrap source is invalid")
    rng = np.random.default_rng(seed)
    starts = np.arange(values.size, dtype=np.int64)
    offsets = np.arange(block_hours, dtype=np.int64)
    block_count = math.ceil(values.size / block_hours)
    means = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected_starts = rng.choice(starts, size=block_count, replace=True)
        indexes = (selected_starts[:, None] + offsets[None, :]) % values.size
        resampled = values[indexes.reshape(-1)[: values.size]]
        means[sample] = np.mean(resampled) * 10_000.0
    return {
        "samples": int(samples),
        "block_hours": int(block_hours),
        "lower_quantile": float(lower_quantile),
        "lower_bps": float(np.quantile(means, lower_quantile)),
        "median_bps": float(np.median(means)),
        "upper_bps": float(np.quantile(means, 1.0 - lower_quantile)),
    }


def report_generated_at_utc() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "BOUNDED_ALPHA_MODEL_SCHEMA_VERSION",
    "SIDE_NAMES",
    "VIEW_IDS",
    "BoundedAlphaModel",
    "BoundedAlphaSpec",
    "ConsensusDecisions",
    "ReplayResult",
    "TradePlan",
    "block_bootstrap_mean_bps",
    "build_trade_plan",
    "consensus_decisions",
    "load_bounded_alpha_model",
    "predict_bounded_alpha_model",
    "replay_trade_plan",
    "report_generated_at_utc",
    "save_bounded_alpha_model",
    "train_bounded_alpha_model",
]
