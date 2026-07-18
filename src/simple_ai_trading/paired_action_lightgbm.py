"""Paired long/short action-value models with held-forward calibration."""

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
from scipy.optimize import isotonic_regression

from .cross_asset_cost_data import SYMBOLS
from .lightgbm_backend import (
    SUPPORTED_LIGHTGBM_BACKEND_KINDS,
    lightgbm_backend_parameters,
)
from .storage import write_json_atomic


PAIRED_ACTION_MODEL_SCHEMA_VERSION = "paired-action-distributional-lightgbm-model-v1"
PAIRED_ACTION_CALIBRATION_SCHEMA_VERSION = "paired-action-calibration-v1"
VIEW_IDS = ("raw_stress_payoff", "stop_normalized_stress_payoff")
OBJECTIVE_IDS = ("conditional_mean", "lower_tail_q20")
ACTION_NAMES = ("long", "short")
ACTION_SIGNS = np.asarray((1.0, -1.0), dtype=np.float32)
Q20_ALPHA = 0.2
_MODEL_FAMILY = "shared_paired_action_distributional_lightgbm"
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
ProgressCallback = Callable[[str, Mapping[str, object]], None]

SIGNED_FEATURES = (
    "target_return_1m_bps",
    "target_return_3m_bps",
    "target_return_5m_bps",
    "target_return_15m_bps",
    "target_return_30m_bps",
    "target_return_60m_bps",
    "target_return_240m_bps",
    "target_return_1440m_bps",
    "target_taker_buy_minus_sell_share",
    "target_signed_taker_flow_1m",
    "target_signed_taker_flow_5m",
    "target_signed_taker_flow_15m",
    "target_signed_taker_flow_60m",
    "target_return_zscore_240m",
    "btcusdt_return_1m_bps",
    "btcusdt_return_5m_bps",
    "btcusdt_return_15m_bps",
    "btcusdt_return_60m_bps",
    "ethusdt_return_1m_bps",
    "ethusdt_return_5m_bps",
    "ethusdt_return_15m_bps",
    "ethusdt_return_60m_bps",
    "solusdt_return_1m_bps",
    "solusdt_return_5m_bps",
    "solusdt_return_15m_bps",
    "solusdt_return_60m_bps",
    "target_beta_residual_return_60m_bps",
    "target_beta_residual_return_240m_bps",
    "target_beta_residual_return_1440m_bps",
    "cross_asset_taker_flow_mean",
    "target_return_vs_cross_mean_1m_bps",
)
CENTERED_SIGNED_FEATURES = {"target_close_location": 0.5}
SEMIVOLATILITY_FEATURES = {
    60: (
        "target_downside_semivolatility_60m_bps",
        "target_upside_semivolatility_60m_bps",
    ),
    240: (
        "target_downside_semivolatility_240m_bps",
        "target_upside_semivolatility_240m_bps",
    ),
}


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
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class PairedActionSpec:
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
    factor_lower_quantile: float = 0.005
    factor_upper_quantile: float = 0.995
    gpu_use_dp_required: bool = True

    def validate(self) -> None:
        numeric = np.asarray(
            [
                self.learning_rate,
                self.feature_fraction,
                self.bagging_fraction,
                self.lambda_l1,
                self.lambda_l2,
                self.factor_lower_quantile,
                self.factor_upper_quantile,
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
            or not 0.0 <= self.factor_lower_quantile < self.factor_upper_quantile <= 1.0
            or self.gpu_use_dp_required is not True
        ):
            raise ValueError("paired-action model specification is invalid")


@dataclass(frozen=True)
class PairedActionPanel:
    timestamps_ms: np.ndarray
    source_feature_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    features: np.ndarray
    stop_bps: np.ndarray
    target_bps: np.ndarray
    exit_time_ms: np.ndarray

    @property
    def timestamps(self) -> int:
        return int(self.timestamps_ms.size)

    @property
    def rows(self) -> int:
        return int(self.timestamps * len(SYMBOLS) * len(ACTION_NAMES))

    def flat_features(self) -> np.ndarray:
        return self.features.reshape(self.rows, self.features.shape[-1])

    def flat_stop_bps(self) -> np.ndarray:
        return self.stop_bps.reshape(self.rows)

    def flat_target_bps(self) -> np.ndarray:
        return self.target_bps.reshape(self.rows)


@dataclass(frozen=True)
class OuterFold:
    fold_id: str
    training_mask: np.ndarray
    early_stopping_mask: np.ndarray
    outer_mask: np.ndarray
    training_start_ms: int
    early_stopping_start_ms: int
    outer_start_ms: int
    outer_end_ms: int

    def evidence(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "training_timestamps": int(np.count_nonzero(self.training_mask)),
            "early_stopping_timestamps": int(
                np.count_nonzero(self.early_stopping_mask)
            ),
            "outer_timestamps": int(np.count_nonzero(self.outer_mask)),
            "training_start_utc": _iso_ms(self.training_start_ms),
            "early_stopping_start_utc": _iso_ms(self.early_stopping_start_ms),
            "outer_start_utc": _iso_ms(self.outer_start_ms),
            "outer_end_utc": _iso_ms(self.outer_end_ms),
        }


@dataclass(frozen=True)
class PairedActionModel:
    schema_version: str
    model_family: str
    treatment_id: str
    view_id: str
    objective_id: str
    seed: int
    base_feature_names: tuple[str, ...]
    factor_feature_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    source_dataset_sha256: str
    payoff_dataset_sha256: str
    design_sha256: str
    spec: PairedActionSpec
    backend_requested: str
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    outer_fold_diagnostics: tuple[Mapping[str, object], ...]
    final_iterations: int
    final_refit_rows: int
    final_factor_lower_bounds: tuple[float, ...]
    final_factor_upper_bounds: tuple[float, ...]
    final_target_mean: float
    model_string: str
    reload_max_abs_prediction_error_bps: float
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    leverage_applied: bool = False

    def asdict(self) -> dict[str, object]:
        value = asdict(self)
        for name in (
            "base_feature_names",
            "factor_feature_names",
            "feature_names",
            "final_factor_lower_bounds",
            "final_factor_upper_bounds",
            "outer_fold_diagnostics",
        ):
            value[name] = list(value[name])
        return value


@dataclass(frozen=True)
class PairedActionTrainingResult:
    model: PairedActionModel
    oof_prediction_bps: np.ndarray
    oof_causal_baseline_bps: np.ndarray
    oof_mask: np.ndarray


@dataclass(frozen=True)
class IsotonicCalibration:
    upper_score_bounds: tuple[float, ...]
    calibrated_values_bps: tuple[float, ...]
    training_rows: int
    calibration_sha256: str

    def asdict(self) -> dict[str, object]:
        return {
            "upper_score_bounds": list(self.upper_score_bounds),
            "calibrated_values_bps": list(self.calibrated_values_bps),
            "training_rows": self.training_rows,
            "calibration_sha256": self.calibration_sha256,
        }


@dataclass(frozen=True)
class QuantileCalibration:
    alpha: float
    additive_offset_bps: float
    training_rows: int
    empirical_training_coverage: float
    calibration_sha256: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PairedActionCalibration:
    schema_version: str
    treatment_id: str
    view_ids: tuple[str, ...]
    point: tuple[IsotonicCalibration, ...]
    lower_tail: tuple[QuantileCalibration, ...]
    fit_start_ms: int
    fit_end_ms: int
    calibration_sha256: str
    trading_authority: bool = False
    profitability_claim: bool = False

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "treatment_id": self.treatment_id,
            "view_ids": list(self.view_ids),
            "point": [value.asdict() for value in self.point],
            "lower_tail": [value.asdict() for value in self.lower_tail],
            "fit_start_ms": self.fit_start_ms,
            "fit_end_ms": self.fit_end_ms,
            "calibration_sha256": self.calibration_sha256,
            "trading_authority": self.trading_authority,
            "profitability_claim": self.profitability_claim,
        }


@dataclass(frozen=True)
class PairedActionDecisions:
    actions: np.ndarray
    score_bps: np.ndarray
    lower_tail_bps: np.ndarray
    size_multiplier: np.ndarray
    model_eligible: np.ndarray
    liquidity_eligible: np.ndarray
    volatility_eligible: np.ndarray
    calibrated_view_median_bps: np.ndarray
    raw_positive_count: np.ndarray


def _iso_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000.0, UTC).isoformat()


def _date_ms(value: str) -> int:
    return int(datetime.fromisoformat(f"{value}T00:00:00+00:00").timestamp() * 1000)


def action_conditioned_feature_names(
    source_feature_names: Sequence[str],
) -> tuple[str, ...]:
    """Return the frozen paired-action names without reading market values."""

    names = tuple(str(name) for name in source_feature_names)
    positions = {name: index for index, name in enumerate(names)}
    required = set(SIGNED_FEATURES) | set(CENTERED_SIGNED_FEATURES)
    required.update(
        name for pair in SEMIVOLATILITY_FEATURES.values() for name in pair
    )
    if (
        not names
        or len(positions) != len(names)
        or any(not name for name in names)
        or not required.issubset(positions)
        or "action_sign" in positions
    ):
        raise ValueError("paired-action source feature names are invalid")
    transformed = list(names)
    for name in SIGNED_FEATURES:
        transformed[positions[name]] = f"action_aligned_{name}"
    for name in CENTERED_SIGNED_FEATURES:
        transformed[positions[name]] = f"action_aligned_{name}"
    for horizon, (downside, upside) in SEMIVOLATILITY_FEATURES.items():
        transformed[positions[downside]] = (
            f"action_favorable_semivolatility_{horizon}m_bps"
        )
        transformed[positions[upside]] = (
            f"action_adverse_semivolatility_{horizon}m_bps"
        )
    transformed.append("action_sign")
    if len(set(transformed)) != len(transformed):
        raise ValueError("paired-action output feature names collide")
    return tuple(transformed)


def build_paired_action_panel(
    *,
    features: np.ndarray,
    feature_names: Sequence[str],
    timestamps_ms: np.ndarray,
    stop_bps: np.ndarray,
    long_target_bps: np.ndarray,
    short_target_bps: np.ndarray,
    long_exit_time_ms: np.ndarray,
    short_exit_time_ms: np.ndarray,
) -> PairedActionPanel:
    """Build exact paired long/short rows from one causal state tensor."""

    matrix = np.asarray(features, dtype=np.float32)
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    names = tuple(str(name) for name in feature_names)
    expected = (timestamps.size, len(SYMBOLS))
    numeric_arrays = (
        np.asarray(stop_bps, dtype=np.float32),
        np.asarray(long_target_bps, dtype=np.float32),
        np.asarray(short_target_bps, dtype=np.float32),
    )
    exit_arrays = (
        np.asarray(long_exit_time_ms, dtype=np.int64),
        np.asarray(short_exit_time_ms, dtype=np.int64),
    )
    if (
        matrix.ndim != 3
        or matrix.shape != (*expected, len(names))
        or timestamps.ndim != 1
        or timestamps.size == 0
        or np.any(np.diff(timestamps) <= 0)
        or any(value.shape != expected for value in numeric_arrays + exit_arrays)
        or not np.isfinite(matrix).all()
        or not all(np.isfinite(value).all() for value in numeric_arrays)
        or np.any(numeric_arrays[0] <= 0.0)
        or any(np.any(value <= timestamps[:, None]) for value in exit_arrays)
    ):
        raise ValueError("paired-action source panel is invalid")

    output_names = action_conditioned_feature_names(names)
    positions = {name: index for index, name in enumerate(names)}
    paired = np.repeat(matrix[:, :, None, :], len(ACTION_NAMES), axis=2)
    signs = ACTION_SIGNS.reshape(1, 1, len(ACTION_NAMES))
    for name in SIGNED_FEATURES:
        index = positions[name]
        paired[..., index] *= signs
    for name, center in CENTERED_SIGNED_FEATURES.items():
        index = positions[name]
        paired[..., index] = (paired[..., index] - center) * signs
    for _horizon, (downside, upside) in SEMIVOLATILITY_FEATURES.items():
        downside_values = matrix[..., positions[downside]]
        upside_values = matrix[..., positions[upside]]
        paired[..., positions[downside]] = np.stack(
            (upside_values, downside_values), axis=2
        )
        paired[..., positions[upside]] = np.stack(
            (downside_values, upside_values), axis=2
        )
    action_column = np.broadcast_to(
        signs[..., None], (*expected, len(ACTION_NAMES), 1)
    )
    paired = np.concatenate((paired, action_column), axis=-1).astype(np.float32)
    targets = np.stack(numeric_arrays[1:], axis=2).astype(np.float32)
    exits = np.stack(exit_arrays, axis=2).astype(np.int64)
    stops = np.repeat(numeric_arrays[0][..., None], len(ACTION_NAMES), axis=2)
    if (
        paired.shape != (*expected, len(ACTION_NAMES), len(output_names))
        or not np.isfinite(paired).all()
        or not np.array_equal(paired[..., -1], np.broadcast_to(signs, targets.shape))
    ):
        raise RuntimeError("paired-action transform invariant failed")
    return PairedActionPanel(
        timestamps_ms=timestamps.copy(),
        source_feature_names=names,
        feature_names=output_names,
        features=paired,
        stop_bps=stops.astype(np.float32),
        target_bps=targets,
        exit_time_ms=exits,
    )


def build_monthly_outer_folds(
    panel: PairedActionPanel,
    *,
    first_outer_month: str = "2023-07-01",
    outer_months: int = 12,
    training_start: str = "2022-01-01",
) -> tuple[OuterFold, ...]:
    """Build expanding monthly folds with label-path separation."""

    if outer_months < 2 or outer_months > 60:
        raise ValueError("paired-action outer fold count is invalid")
    timestamps = panel.timestamps_ms
    maximum_exit = np.max(panel.exit_time_ms, axis=(1, 2))
    training_start_ms = _date_ms(training_start)
    first = np.datetime64(first_outer_month, "M")
    folds: list[OuterFold] = []
    seen_outer = np.zeros(timestamps.size, dtype=bool)
    for offset in range(outer_months):
        outer_month = first + np.timedelta64(offset, "M")
        outer_start_text = str(outer_month.astype("datetime64[D]"))
        outer_end_text = str(
            (outer_month + np.timedelta64(1, "M")).astype("datetime64[D]")
        )
        early_month = outer_month - np.timedelta64(1, "M")
        early_start_text = str(early_month.astype("datetime64[D]"))
        early_start_ms = _date_ms(early_start_text)
        outer_start_ms = _date_ms(outer_start_text)
        outer_end_ms = _date_ms(outer_end_text)
        training_mask = (
            (timestamps >= training_start_ms)
            & (timestamps < early_start_ms)
            & (maximum_exit < early_start_ms)
        )
        early_mask = (
            (timestamps >= early_start_ms)
            & (timestamps < outer_start_ms)
            & (maximum_exit < outer_start_ms)
        )
        outer_mask = (timestamps >= outer_start_ms) & (timestamps < outer_end_ms)
        if (
            np.count_nonzero(training_mask) < 1_000
            or np.count_nonzero(early_mask) < 100
            or np.count_nonzero(outer_mask) < 100
            or np.any(seen_outer & outer_mask)
            or np.max(maximum_exit[training_mask]) >= np.min(timestamps[early_mask])
            or np.max(maximum_exit[early_mask]) >= np.min(timestamps[outer_mask])
        ):
            raise ValueError(f"paired-action fold is invalid: {outer_start_text}")
        seen_outer |= outer_mask
        folds.append(
            OuterFold(
                fold_id=outer_start_text[:7],
                training_mask=training_mask,
                early_stopping_mask=early_mask,
                outer_mask=outer_mask,
                training_start_ms=training_start_ms,
                early_stopping_start_ms=early_start_ms,
                outer_start_ms=outer_start_ms,
                outer_end_ms=outer_end_ms,
            )
        )
    return tuple(folds)


def embargoed_interval_mask(
    panel: PairedActionPanel,
    *,
    start: str,
    end: str,
) -> np.ndarray:
    """Select timestamps whose complete label path ends before the next role."""

    start_ms = _date_ms(start)
    end_ms = _date_ms(end)
    maximum_exit = np.max(panel.exit_time_ms, axis=(1, 2))
    mask = (
        (panel.timestamps_ms >= start_ms)
        & (panel.timestamps_ms < end_ms)
        & (maximum_exit < end_ms)
    )
    if np.count_nonzero(mask) == 0:
        raise ValueError("paired-action embargoed interval is empty")
    return mask


def _row_mask(timestamp_mask: np.ndarray, timestamps: int) -> np.ndarray:
    mask = np.asarray(timestamp_mask, dtype=bool)
    if mask.shape != (timestamps,) or np.count_nonzero(mask) == 0:
        raise ValueError("paired-action timestamp mask is invalid")
    return np.repeat(mask, len(SYMBOLS) * len(ACTION_NAMES))


def _validate_outer_fold(panel: PairedActionPanel, fold: OuterFold) -> None:
    masks = (
        np.asarray(fold.training_mask, dtype=bool),
        np.asarray(fold.early_stopping_mask, dtype=bool),
        np.asarray(fold.outer_mask, dtype=bool),
    )
    if (
        not fold.fold_id.strip()
        or any(mask.shape != (panel.timestamps,) for mask in masks)
        or any(np.count_nonzero(mask) == 0 for mask in masks)
        or np.any(masks[0] & masks[1])
        or np.any(masks[0] & masks[2])
        or np.any(masks[1] & masks[2])
    ):
        raise ValueError(f"paired-action outer fold masks are invalid: {fold.fold_id}")
    timestamps = panel.timestamps_ms
    maximum_exit = np.max(panel.exit_time_ms, axis=(1, 2))
    if (
        np.max(timestamps[masks[0]]) >= np.min(timestamps[masks[1]])
        or np.max(timestamps[masks[1]]) >= np.min(timestamps[masks[2]])
        or np.max(maximum_exit[masks[0]]) >= np.min(timestamps[masks[1]])
        or np.max(maximum_exit[masks[1]]) >= np.min(timestamps[masks[2]])
    ):
        raise ValueError(
            f"paired-action outer fold labels cross a boundary: {fold.fold_id}"
        )


def _validate_factor_values(
    panel: PairedActionPanel,
    factor_values: np.ndarray | None,
    factor_names: Sequence[str],
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    names = tuple(str(name) for name in factor_names)
    if factor_values is None:
        if names:
            raise ValueError("paired-action factor names have no values")
        return None, ()
    values = np.asarray(factor_values, dtype=np.float32)
    if (
        values.ndim != 4
        or values.shape[:3] != panel.target_bps.shape
        or values.shape[-1] != len(names)
        or not names
        or len(set(names)) != len(names)
        or set(names) & set(panel.feature_names)
        or not np.isfinite(values).all()
    ):
        raise ValueError("paired-action factor matrix is invalid")
    return values, names


def _fit_factor_bounds(
    values: np.ndarray | None,
    training_rows: np.ndarray,
    spec: PairedActionSpec,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if values is None:
        return (), ()
    selected = values.reshape(-1, values.shape[-1])[training_rows]
    lower = np.quantile(selected, spec.factor_lower_quantile, axis=0)
    upper = np.quantile(selected, spec.factor_upper_quantile, axis=0)
    if (
        not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(upper - lower <= 1e-12)
    ):
        raise ValueError("paired-action factor clipping distribution is degenerate")
    return tuple(float(value) for value in lower), tuple(float(value) for value in upper)


def _feature_matrix(
    panel: PairedActionPanel,
    factor_values: np.ndarray | None,
    lower: Sequence[float],
    upper: Sequence[float],
) -> np.ndarray:
    base = panel.flat_features()
    if factor_values is None:
        if lower or upper:
            raise ValueError("paired-action factor bounds have no values")
        return base
    if len(lower) != factor_values.shape[-1] or len(upper) != factor_values.shape[-1]:
        raise ValueError("paired-action factor bound count differs")
    clipped = np.clip(
        factor_values.reshape(-1, factor_values.shape[-1]),
        np.asarray(lower, dtype=np.float32),
        np.asarray(upper, dtype=np.float32),
    )
    return np.concatenate((base, clipped.astype(np.float32)), axis=1)


def _target_for_view(
    view_id: str,
    target_bps: np.ndarray,
    stop_bps: np.ndarray,
) -> np.ndarray:
    if view_id == VIEW_IDS[0]:
        return target_bps
    if view_id == VIEW_IDS[1]:
        return target_bps / stop_bps
    raise KeyError(view_id)


def _prediction_to_bps(
    view_id: str,
    prediction: np.ndarray,
    stop_bps: np.ndarray,
) -> np.ndarray:
    if view_id == VIEW_IDS[0]:
        return prediction
    if view_id == VIEW_IDS[1]:
        return prediction * stop_bps
    raise KeyError(view_id)


def pinball_loss(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    alpha: float = Q20_ALPHA,
) -> float:
    actual = np.asarray(truth, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    if (
        actual.shape != forecast.shape
        or actual.size == 0
        or not np.isfinite(actual).all()
        or not np.isfinite(forecast).all()
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("pinball loss source is invalid")
    residual = actual - forecast
    return float(np.mean(np.maximum(alpha * residual, (alpha - 1.0) * residual)))


def _loss(objective_id: str, truth: np.ndarray, prediction: np.ndarray) -> float:
    if objective_id == OBJECTIVE_IDS[0]:
        return float(np.mean(np.square(truth - prediction)))
    if objective_id == OBJECTIVE_IDS[1]:
        return pinball_loss(truth, prediction)
    raise KeyError(objective_id)


def _parameters(
    spec: PairedActionSpec,
    compute_backend: str,
    seed: int,
    objective_id: str,
) -> tuple[dict[str, object], str, str]:
    backend, kind, device = lightgbm_backend_parameters(
        compute_backend,
        seed,
        reproducible=True,
    )
    if kind == "opencl" and (
        not spec.gpu_use_dp_required or backend.get("gpu_use_dp") is not True
    ):
        raise RuntimeError("paired-action OpenCL FP64 accumulation is required")
    objective = "regression_l2" if objective_id == OBJECTIVE_IDS[0] else "quantile"
    parameters: dict[str, object] = {
        **backend,
        "objective": objective,
        "metric": "l2" if objective_id == OBJECTIVE_IDS[0] else "quantile",
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
    }
    if objective_id == OBJECTIVE_IDS[1]:
        parameters["alpha"] = Q20_ALPHA
    return parameters, kind, device


def _causal_constant(
    transformed_target: np.ndarray,
    training_rows: np.ndarray,
    prediction_rows: np.ndarray,
    objective_id: str,
) -> np.ndarray:
    action_index = np.tile(
        np.arange(len(ACTION_NAMES), dtype=np.int8),
        transformed_target.size // len(ACTION_NAMES),
    )
    output = np.empty(np.count_nonzero(prediction_rows), dtype=np.float64)
    selected_actions = action_index[prediction_rows]
    for action in range(len(ACTION_NAMES)):
        sample = transformed_target[training_rows & (action_index == action)]
        if sample.size == 0:
            raise ValueError("paired-action causal baseline action is empty")
        constant = (
            float(np.mean(sample))
            if objective_id == OBJECTIVE_IDS[0]
            else float(np.quantile(sample, Q20_ALPHA, method="higher"))
        )
        output[selected_actions == action] = constant
    return output


def _model_payload(model: PairedActionModel) -> dict[str, object]:
    payload = model.asdict()
    payload.pop("model_sha256")
    return payload


def _model_sha256(model: PairedActionModel) -> str:
    return _sha256(_model_payload(model))


def _validate_model(model: PairedActionModel, *, reload: bool) -> None:
    if (
        model.schema_version != PAIRED_ACTION_MODEL_SCHEMA_VERSION
        or model.model_family != _MODEL_FAMILY
        or not model.treatment_id.strip()
        or model.view_id not in VIEW_IDS
        or model.objective_id not in OBJECTIVE_IDS
        or not model.base_feature_names
        or tuple(model.base_feature_names + model.factor_feature_names)
        != model.feature_names
        or len(set(model.feature_names)) != len(model.feature_names)
        or not _is_sha256(model.source_dataset_sha256)
        or not _is_sha256(model.payoff_dataset_sha256)
        or not _is_sha256(model.design_sha256)
        or not _is_sha256(model.model_sha256)
        or model.backend_kind not in SUPPORTED_LIGHTGBM_BACKEND_KINDS
        or not model.outer_fold_diagnostics
        or not 1 <= model.final_iterations <= model.spec.maximum_boosting_rounds
        or model.final_refit_rows < 1
        or len(model.final_factor_lower_bounds) != len(model.factor_feature_names)
        or len(model.final_factor_upper_bounds) != len(model.factor_feature_names)
        or not math.isfinite(model.final_target_mean)
        or not model.model_string.strip()
        or not math.isfinite(model.reload_max_abs_prediction_error_bps)
        or model.reload_max_abs_prediction_error_bps < 0.0
        or model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.leverage_applied
        or _model_sha256(model) != model.model_sha256
    ):
        raise ValueError("paired-action model contract is invalid")
    model.spec.validate()
    if reload:
        try:
            booster = lgb.Booster(model_str=model.model_string)
        except lgb.basic.LightGBMError as exc:
            raise ValueError("paired-action booster cannot be reloaded") from exc
        if booster.num_feature() != len(model.feature_names):
            raise ValueError("paired-action booster feature count drifted")


def train_paired_action_model(
    *,
    treatment_id: str,
    view_id: str,
    objective_id: str,
    seed: int,
    panel: PairedActionPanel,
    outer_folds: Sequence[OuterFold],
    final_refit_mask: np.ndarray,
    prediction_probe_mask: np.ndarray,
    factor_values: np.ndarray | None,
    factor_feature_names: Sequence[str],
    source_dataset_sha256: str,
    payoff_dataset_sha256: str,
    design_sha256: str,
    spec: PairedActionSpec,
    compute_backend: str,
    progress: ProgressCallback | None = None,
) -> PairedActionTrainingResult:
    """Fit rolling models, retain honest OOF scores, then perform one final refit."""

    spec.validate()
    if view_id not in VIEW_IDS or objective_id not in OBJECTIVE_IDS:
        raise ValueError("paired-action model view or objective is invalid")
    if not treatment_id.strip() or len(outer_folds) < 2:
        raise ValueError("paired-action treatment or folds are invalid")
    factors, factor_names = _validate_factor_values(
        panel, factor_values, factor_feature_names
    )
    full_feature_names = panel.feature_names + factor_names
    flat_target_bps = panel.flat_target_bps().astype(np.float64)
    flat_stop = panel.flat_stop_bps().astype(np.float64)
    transformed_target = _target_for_view(
        view_id, flat_target_bps, flat_stop
    ).astype(np.float64)
    parameters, backend_kind, backend_device = _parameters(
        spec, compute_backend, int(seed), objective_id
    )
    oof_prediction = np.full(panel.rows, np.nan, dtype=np.float64)
    oof_baseline = np.full(panel.rows, np.nan, dtype=np.float64)
    fold_diagnostics: list[dict[str, object]] = []
    best_iterations: list[int] = []
    row_weight = np.full(panel.rows, 0.5, dtype=np.float32)

    for fold_index, fold in enumerate(outer_folds, start=1):
        _validate_outer_fold(panel, fold)
        train_rows = _row_mask(fold.training_mask, panel.timestamps)
        early_rows = _row_mask(fold.early_stopping_mask, panel.timestamps)
        outer_rows = _row_mask(fold.outer_mask, panel.timestamps)
        if np.any(np.isfinite(oof_prediction[outer_rows])):
            raise ValueError("paired-action OOF folds overlap")
        lower, upper = _fit_factor_bounds(factors, train_rows, spec)
        matrix = _feature_matrix(panel, factors, lower, upper)
        training = lgb.Dataset(
            matrix[train_rows],
            label=transformed_target[train_rows],
            weight=row_weight[train_rows],
            feature_name=list(full_feature_names),
            free_raw_data=False,
        )
        early = lgb.Dataset(
            matrix[early_rows],
            label=transformed_target[early_rows],
            weight=row_weight[early_rows],
            reference=training,
            feature_name=list(full_feature_names),
            free_raw_data=False,
        )
        if progress is not None:
            progress(
                "paired_action_outer_fit",
                {
                    "treatment_id": treatment_id,
                    "view_id": view_id,
                    "objective_id": objective_id,
                    "seed": int(seed),
                    "fold": fold.fold_id,
                    "index": fold_index,
                    "total": len(outer_folds),
                    "status": "started",
                },
            )
        booster = lgb.train(
            parameters,
            training,
            num_boost_round=spec.maximum_boosting_rounds,
            valid_sets=[early],
            valid_names=["early_stopping"],
            callbacks=[
                lgb.early_stopping(spec.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        iteration = max(1, int(booster.best_iteration or booster.current_iteration()))
        prediction_view = np.asarray(
            booster.predict(matrix[outer_rows], num_iteration=iteration),
            dtype=np.float64,
        )
        baseline_view = _causal_constant(
            transformed_target,
            train_rows,
            outer_rows,
            objective_id,
        )
        prediction_bps = _prediction_to_bps(
            view_id, prediction_view, flat_stop[outer_rows]
        )
        baseline_bps = _prediction_to_bps(
            view_id, baseline_view, flat_stop[outer_rows]
        )
        truth_bps = flat_target_bps[outer_rows]
        model_loss = _loss(objective_id, truth_bps, prediction_bps)
        baseline_loss = _loss(objective_id, truth_bps, baseline_bps)
        if not math.isfinite(model_loss) or not math.isfinite(baseline_loss):
            raise ValueError("paired-action outer-fold loss is nonfinite")
        oof_prediction[outer_rows] = prediction_bps
        oof_baseline[outer_rows] = baseline_bps
        best_iterations.append(iteration)
        fold_diagnostics.append(
            {
                **fold.evidence(),
                "training_rows": int(np.count_nonzero(train_rows)),
                "early_stopping_rows": int(np.count_nonzero(early_rows)),
                "outer_rows": int(np.count_nonzero(outer_rows)),
                "best_iterations": iteration,
                "loss_name": "mse"
                if objective_id == OBJECTIVE_IDS[0]
                else "pinball_q20",
                "loss_units": "bps_squared"
                if objective_id == OBJECTIVE_IDS[0]
                else "bps",
                "model_loss": model_loss,
                "causal_constant_loss": baseline_loss,
                "loss_skill": 1.0 - model_loss / baseline_loss
                if baseline_loss > 0.0
                else None,
                "factor_lower_bounds": list(lower),
                "factor_upper_bounds": list(upper),
            }
        )
        if progress is not None:
            progress(
                "paired_action_outer_fit",
                {
                    "treatment_id": treatment_id,
                    "view_id": view_id,
                    "objective_id": objective_id,
                    "seed": int(seed),
                    "fold": fold.fold_id,
                    "index": fold_index,
                    "total": len(outer_folds),
                    "status": "complete",
                    "best_iterations": iteration,
                    "loss_skill": fold_diagnostics[-1]["loss_skill"],
                },
            )

    oof_mask = np.isfinite(oof_prediction)
    if not np.array_equal(oof_mask, np.isfinite(oof_baseline)):
        raise RuntimeError("paired-action OOF prediction and baseline masks differ")
    final_timestamp_mask = np.asarray(final_refit_mask, dtype=bool)
    probe_timestamp_mask = np.asarray(prediction_probe_mask, dtype=bool)
    final_rows = _row_mask(final_timestamp_mask, panel.timestamps)
    probe_rows = _row_mask(probe_timestamp_mask, panel.timestamps)
    if (
        np.any(final_timestamp_mask & probe_timestamp_mask)
        or np.max(panel.timestamps_ms[final_timestamp_mask])
        >= np.min(panel.timestamps_ms[probe_timestamp_mask])
        or np.max(
            np.max(panel.exit_time_ms, axis=(1, 2))[final_timestamp_mask]
        )
        >= np.min(panel.timestamps_ms[probe_timestamp_mask])
    ):
        raise ValueError("paired-action final-refit labels cross the prediction role")
    final_lower, final_upper = _fit_factor_bounds(factors, final_rows, spec)
    final_matrix = _feature_matrix(panel, factors, final_lower, final_upper)
    final_iterations = max(1, int(np.rint(np.median(best_iterations))))
    final_training = lgb.Dataset(
        final_matrix[final_rows],
        label=transformed_target[final_rows],
        weight=row_weight[final_rows],
        feature_name=list(full_feature_names),
        free_raw_data=False,
    )
    final_booster = lgb.train(
        parameters,
        final_training,
        num_boost_round=final_iterations,
        callbacks=[lgb.log_evaluation(0)],
    )
    model_string = final_booster.model_to_string(num_iteration=final_iterations)
    direct = np.asarray(
        final_booster.predict(final_matrix[probe_rows], num_iteration=final_iterations),
        dtype=np.float64,
    )
    reloaded = lgb.Booster(model_str=model_string)
    restored = np.asarray(
        reloaded.predict(final_matrix[probe_rows], num_iteration=final_iterations),
        dtype=np.float64,
    )
    reload_error = float(
        np.max(
            np.abs(
                _prediction_to_bps(view_id, direct, flat_stop[probe_rows])
                - _prediction_to_bps(view_id, restored, flat_stop[probe_rows])
            )
        )
    )
    provisional = PairedActionModel(
        schema_version=PAIRED_ACTION_MODEL_SCHEMA_VERSION,
        model_family=_MODEL_FAMILY,
        treatment_id=treatment_id,
        view_id=view_id,
        objective_id=objective_id,
        seed=int(seed),
        base_feature_names=panel.feature_names,
        factor_feature_names=factor_names,
        feature_names=full_feature_names,
        source_dataset_sha256=source_dataset_sha256,
        payoff_dataset_sha256=payoff_dataset_sha256,
        design_sha256=design_sha256,
        spec=spec,
        backend_requested=compute_backend,
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=lgb.__version__,
        outer_fold_diagnostics=tuple(fold_diagnostics),
        final_iterations=final_iterations,
        final_refit_rows=int(np.count_nonzero(final_rows)),
        final_factor_lower_bounds=final_lower,
        final_factor_upper_bounds=final_upper,
        final_target_mean=float(np.mean(transformed_target[final_rows])),
        model_string=model_string,
        reload_max_abs_prediction_error_bps=reload_error,
        model_sha256="",
    )
    model = PairedActionModel(
        **{
            **provisional.__dict__,
            "model_sha256": _model_sha256(provisional),
        }
    )
    _validate_model(model, reload=True)
    return PairedActionTrainingResult(
        model=model,
        oof_prediction_bps=oof_prediction.reshape(panel.target_bps.shape),
        oof_causal_baseline_bps=oof_baseline.reshape(panel.target_bps.shape),
        oof_mask=oof_mask.reshape(panel.target_bps.shape),
    )


def predict_paired_action_model(
    model: PairedActionModel,
    panel: PairedActionPanel,
    factor_values: np.ndarray | None = None,
) -> np.ndarray:
    """Predict both candidate actions with the final frozen clipping bounds."""

    _validate_model(model, reload=True)
    if panel.feature_names != model.base_feature_names:
        raise ValueError("paired-action prediction base feature identity differs")
    factors, names = _validate_factor_values(
        panel,
        factor_values,
        model.factor_feature_names,
    )
    if names != model.factor_feature_names:
        raise ValueError("paired-action prediction factor identity differs")
    matrix = _feature_matrix(
        panel,
        factors,
        model.final_factor_lower_bounds,
        model.final_factor_upper_bounds,
    )
    booster = lgb.Booster(model_str=model.model_string)
    prediction = np.asarray(
        booster.predict(matrix, num_iteration=model.final_iterations),
        dtype=np.float64,
    )
    prediction = _prediction_to_bps(
        model.view_id,
        prediction,
        panel.flat_stop_bps().astype(np.float64),
    )
    if prediction.shape != (panel.rows,) or not np.isfinite(prediction).all():
        raise RuntimeError("paired-action model produced invalid predictions")
    return prediction.reshape(panel.target_bps.shape)


def save_paired_action_model(path: str | Path, model: PairedActionModel) -> None:
    _validate_model(model, reload=True)
    payload = model.asdict()
    encoded = _canonical_json(payload).encode("utf-8")
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("paired-action model artifact is too large")
    write_json_atomic(Path(path), payload, indent=None, sort_keys=True)


def load_paired_action_model(path: str | Path) -> PairedActionModel:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("paired-action model artifact is missing or oversized")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("paired-action model artifact must be an object")
    expected = {field.name for field in fields(PairedActionModel)}
    if set(payload) != expected or not isinstance(payload.get("spec"), dict):
        raise ValueError("paired-action model artifact fields drifted")
    payload["spec"] = PairedActionSpec(**payload["spec"])
    for name in (
        "base_feature_names",
        "factor_feature_names",
        "feature_names",
        "final_factor_lower_bounds",
        "final_factor_upper_bounds",
        "outer_fold_diagnostics",
    ):
        payload[name] = tuple(payload[name])
    model = PairedActionModel(**payload)
    _validate_model(model, reload=True)
    return model


def _isotonic_payload(calibration: IsotonicCalibration) -> dict[str, object]:
    payload = calibration.asdict()
    payload.pop("calibration_sha256")
    return payload


def fit_isotonic_calibration(
    score_bps: np.ndarray,
    truth_bps: np.ndarray,
) -> IsotonicCalibration:
    score = np.asarray(score_bps, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth_bps, dtype=np.float64).reshape(-1)
    if (
        score.shape != truth.shape
        or score.size < 100
        or not np.isfinite(score).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError("isotonic calibration source is invalid")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_truth = truth[order]
    unique_score, first, counts = np.unique(
        sorted_score,
        return_index=True,
        return_counts=True,
    )
    truth_sum = np.add.reduceat(sorted_truth, first)
    grouped_truth = truth_sum / counts
    result = isotonic_regression(
        grouped_truth,
        weights=counts.astype(np.float64),
        increasing=True,
    )
    blocks = np.asarray(result.blocks, dtype=np.int64)
    fitted = np.asarray(result.x, dtype=np.float64)
    upper = tuple(float(unique_score[blocks[index + 1] - 1]) for index in range(len(blocks) - 1))
    values = tuple(float(fitted[blocks[index]]) for index in range(len(blocks) - 1))
    provisional = IsotonicCalibration(
        upper_score_bounds=upper,
        calibrated_values_bps=values,
        training_rows=int(score.size),
        calibration_sha256="",
    )
    return IsotonicCalibration(
        **{
            **provisional.__dict__,
            "calibration_sha256": _sha256(_isotonic_payload(provisional)),
        }
    )


def apply_isotonic_calibration(
    calibration: IsotonicCalibration,
    score_bps: np.ndarray,
) -> np.ndarray:
    if (
        not calibration.upper_score_bounds
        or len(calibration.upper_score_bounds)
        != len(calibration.calibrated_values_bps)
        or calibration.training_rows < 100
        or calibration.calibration_sha256 != _sha256(_isotonic_payload(calibration))
    ):
        raise ValueError("isotonic calibration identity is invalid")
    score = np.asarray(score_bps, dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError("isotonic calibration score is nonfinite")
    upper = np.asarray(calibration.upper_score_bounds, dtype=np.float64)
    values = np.asarray(calibration.calibrated_values_bps, dtype=np.float64)
    indexes = np.searchsorted(upper, score, side="left")
    indexes = np.clip(indexes, 0, len(values) - 1)
    return values[indexes]


def _quantile_payload(calibration: QuantileCalibration) -> dict[str, object]:
    payload = calibration.asdict()
    payload.pop("calibration_sha256")
    return payload


def fit_quantile_calibration(
    prediction_bps: np.ndarray,
    truth_bps: np.ndarray,
    *,
    alpha: float = Q20_ALPHA,
) -> QuantileCalibration:
    prediction = np.asarray(prediction_bps, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth_bps, dtype=np.float64).reshape(-1)
    if (
        prediction.shape != truth.shape
        or prediction.size < 100
        or not np.isfinite(prediction).all()
        or not np.isfinite(truth).all()
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("quantile calibration source is invalid")
    offset = float(np.quantile(truth - prediction, alpha, method="higher"))
    coverage = float(np.mean(truth <= prediction + offset))
    provisional = QuantileCalibration(
        alpha=float(alpha),
        additive_offset_bps=offset,
        training_rows=int(prediction.size),
        empirical_training_coverage=coverage,
        calibration_sha256="",
    )
    return QuantileCalibration(
        **{
            **provisional.__dict__,
            "calibration_sha256": _sha256(_quantile_payload(provisional)),
        }
    )


def apply_quantile_calibration(
    calibration: QuantileCalibration,
    prediction_bps: np.ndarray,
) -> np.ndarray:
    if (
        not 0.0 < calibration.alpha < 1.0
        or calibration.training_rows < 100
        or not math.isfinite(calibration.additive_offset_bps)
        or not 0.0 <= calibration.empirical_training_coverage <= 1.0
        or calibration.calibration_sha256 != _sha256(_quantile_payload(calibration))
    ):
        raise ValueError("quantile calibration identity is invalid")
    prediction = np.asarray(prediction_bps, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("quantile calibration prediction is nonfinite")
    return prediction + calibration.additive_offset_bps


def _calibration_payload(calibration: PairedActionCalibration) -> dict[str, object]:
    payload = calibration.asdict()
    payload.pop("calibration_sha256")
    return payload


def _validate_calibration(calibration: PairedActionCalibration) -> None:
    if (
        calibration.schema_version != PAIRED_ACTION_CALIBRATION_SCHEMA_VERSION
        or not calibration.treatment_id.strip()
        or calibration.view_ids != VIEW_IDS
        or len(calibration.point) != len(VIEW_IDS)
        or len(calibration.lower_tail) != len(VIEW_IDS)
        or calibration.fit_start_ms >= calibration.fit_end_ms
        or calibration.trading_authority
        or calibration.profitability_claim
        or calibration.calibration_sha256
        != _sha256(_calibration_payload(calibration))
    ):
        raise ValueError("paired-action calibration contract is invalid")
    for point in calibration.point:
        if (
            not point.upper_score_bounds
            or len(point.upper_score_bounds) != len(point.calibrated_values_bps)
            or point.training_rows < 100
            or not np.isfinite(point.upper_score_bounds).all()
            or not np.isfinite(point.calibrated_values_bps).all()
            or np.any(np.diff(point.upper_score_bounds) <= 0.0)
            or np.any(np.diff(point.calibrated_values_bps) < 0.0)
            or point.calibration_sha256 != _sha256(_isotonic_payload(point))
        ):
            raise ValueError("paired-action point calibration is invalid")
    for quantile in calibration.lower_tail:
        if (
            not 0.0 < quantile.alpha < 1.0
            or quantile.training_rows < 100
            or not math.isfinite(quantile.additive_offset_bps)
            or not 0.0 <= quantile.empirical_training_coverage <= 1.0
            or quantile.calibration_sha256
            != _sha256(_quantile_payload(quantile))
        ):
            raise ValueError("paired-action quantile calibration is invalid")


def fit_paired_action_calibration(
    *,
    treatment_id: str,
    point_predictions_bps: np.ndarray,
    q20_predictions_bps: np.ndarray,
    truth_bps: np.ndarray,
    timestamp_mask: np.ndarray,
    timestamps_ms: np.ndarray,
) -> PairedActionCalibration:
    """Fit one point and lower-tail calibrator per view from OOF predictions."""

    point = np.asarray(point_predictions_bps, dtype=np.float64)
    quantile = np.asarray(q20_predictions_bps, dtype=np.float64)
    truth = np.asarray(truth_bps, dtype=np.float64)
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    mask = np.asarray(timestamp_mask, dtype=bool)
    expected = (len(VIEW_IDS), 3, *truth.shape)
    if (
        point.shape != expected
        or quantile.shape != expected
        or truth.ndim != 3
        or truth.shape[-1] != len(ACTION_NAMES)
        or timestamps.shape != (truth.shape[0],)
        or mask.shape != (truth.shape[0],)
        or np.count_nonzero(mask) < 100
        or not treatment_id.strip()
    ):
        raise ValueError("paired-action calibration source shape is invalid")
    selected_truth = truth[mask].reshape(-1)
    point_calibrations: list[IsotonicCalibration] = []
    quantile_calibrations: list[QuantileCalibration] = []
    for view_index in range(len(VIEW_IDS)):
        point_median = np.median(point[view_index], axis=0)
        q20_median = np.median(quantile[view_index], axis=0)
        selected_point = point_median[mask].reshape(-1)
        selected_q20 = q20_median[mask].reshape(-1)
        if (
            not np.isfinite(selected_point).all()
            or not np.isfinite(selected_q20).all()
            or not np.isfinite(selected_truth).all()
        ):
            raise ValueError("paired-action calibration source is nonfinite")
        point_calibrations.append(
            fit_isotonic_calibration(selected_point, selected_truth)
        )
        quantile_calibrations.append(
            fit_quantile_calibration(selected_q20, selected_truth)
        )
    provisional = PairedActionCalibration(
        schema_version=PAIRED_ACTION_CALIBRATION_SCHEMA_VERSION,
        treatment_id=treatment_id,
        view_ids=VIEW_IDS,
        point=tuple(point_calibrations),
        lower_tail=tuple(quantile_calibrations),
        fit_start_ms=int(np.min(timestamps[mask])),
        fit_end_ms=int(np.max(timestamps[mask])),
        calibration_sha256="",
    )
    return PairedActionCalibration(
        **{
            **provisional.__dict__,
            "calibration_sha256": _sha256(_calibration_payload(provisional)),
        }
    )


def apply_paired_action_calibration(
    calibration: PairedActionCalibration,
    point_predictions_bps: np.ndarray,
    q20_predictions_bps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_calibration(calibration)
    point = np.asarray(point_predictions_bps, dtype=np.float64)
    quantile = np.asarray(q20_predictions_bps, dtype=np.float64)
    if (
        point.shape != quantile.shape
        or point.ndim != 5
        or point.shape[:2] != (len(VIEW_IDS), 3)
        or point.shape[-1] != len(ACTION_NAMES)
    ):
        raise ValueError("paired-action calibration prediction shape is invalid")
    calibrated_point = np.empty((len(VIEW_IDS), *point.shape[2:]), dtype=np.float64)
    calibrated_q20 = np.empty_like(calibrated_point)
    for view_index in range(len(VIEW_IDS)):
        calibrated_point[view_index] = apply_isotonic_calibration(
            calibration.point[view_index],
            np.median(point[view_index], axis=0),
        )
        calibrated_q20[view_index] = apply_quantile_calibration(
            calibration.lower_tail[view_index],
            np.median(quantile[view_index], axis=0),
        )
    return calibrated_point, calibrated_q20


def save_paired_action_calibration(
    path: str | Path,
    calibration: PairedActionCalibration,
) -> None:
    _validate_calibration(calibration)
    write_json_atomic(Path(path), calibration.asdict(), indent=None, sort_keys=True)


def load_paired_action_calibration(path: str | Path) -> PairedActionCalibration:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("paired-action calibration artifact is missing or oversized")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("paired-action calibration artifact must be an object")
    payload["view_ids"] = tuple(payload["view_ids"])
    payload["point"] = tuple(
        IsotonicCalibration(
            **{
                **value,
                "upper_score_bounds": tuple(value["upper_score_bounds"]),
                "calibrated_values_bps": tuple(value["calibrated_values_bps"]),
            }
        )
        for value in payload["point"]
    )
    payload["lower_tail"] = tuple(
        QuantileCalibration(**value) for value in payload["lower_tail"]
    )
    calibration = PairedActionCalibration(**payload)
    _validate_calibration(calibration)
    return calibration


def paired_action_decisions(
    *,
    point_predictions_bps: np.ndarray,
    q20_predictions_bps: np.ndarray,
    calibration: PairedActionCalibration,
    source_features: np.ndarray,
    source_feature_names: Sequence[str],
    stop_bps: np.ndarray,
) -> PairedActionDecisions:
    """Apply frozen calibrated action comparison and causal market-state gates."""

    point = np.asarray(point_predictions_bps, dtype=np.float64)
    source = np.asarray(source_features, dtype=np.float64)
    stop = np.asarray(stop_bps, dtype=np.float64)
    names = tuple(str(name) for name in source_feature_names)
    if (
        point.ndim != 5
        or point.shape[:2] != (len(VIEW_IDS), 3)
        or point.shape[-1] != len(ACTION_NAMES)
        or source.ndim != 3
        or point.shape[2:4] != source.shape[:2]
        or source.shape[-1] != len(names)
        or stop.shape != source.shape[:2]
        or not np.isfinite(point).all()
        or not np.isfinite(source).all()
        or not np.isfinite(stop).all()
        or np.any(stop <= 0.0)
    ):
        raise ValueError("paired-action decision source is invalid")
    calibrated_point, calibrated_q20 = apply_paired_action_calibration(
        calibration,
        point,
        q20_predictions_bps,
    )
    combined = np.mean(calibrated_point, axis=0)
    selected_index = np.argmax(combined, axis=-1)
    selected_score = np.take_along_axis(
        combined, selected_index[..., None], axis=-1
    )[..., 0]
    opposite_score = np.take_along_axis(
        combined, (1 - selected_index)[..., None], axis=-1
    )[..., 0]
    raw_positive = np.sum(point > 0.0, axis=(0, 1))
    selected_positive = np.take_along_axis(
        raw_positive, selected_index[..., None], axis=-1
    )[..., 0]
    model_eligible = (
        (selected_score > 0.0)
        & (selected_score > opposite_score)
        & (selected_positive >= 4)
    )
    positions = {name: index for index, name in enumerate(names)}
    required = (
        "target_same_minute_of_week_liquidity_ratio",
        "target_quote_volume_vs_1440m_mean",
        "target_realized_volatility_60m_bps",
        "target_realized_volatility_1440m_bps",
    )
    if any(name not in positions for name in required):
        raise ValueError("paired-action decision feature is missing")
    liquidity = (
        source[..., positions[required[0]]] >= 0.5
    ) & (source[..., positions[required[1]]] >= 0.25)
    volatility = source[..., positions[required[2]]] <= (
        2.5 * source[..., positions[required[3]]]
    )
    eligible = model_eligible & liquidity & volatility
    actions = np.zeros(source.shape[:2], dtype=np.int8)
    actions[eligible & (selected_index == 0)] = 1
    actions[eligible & (selected_index == 1)] = -1
    score = np.where(eligible, selected_score, 0.0)
    conservative_q20 = np.min(calibrated_q20, axis=0)
    selected_q20 = np.take_along_axis(
        conservative_q20, selected_index[..., None], axis=-1
    )[..., 0]
    downside = np.abs(np.minimum(selected_q20, 0.0))
    size_multiplier = np.minimum(1.0, stop / np.maximum(stop, downside))
    size_multiplier = np.where(eligible, size_multiplier, 0.0)
    if (
        np.any((actions != 0) & (size_multiplier <= 0.0))
        or np.any(size_multiplier > 1.0)
        or not np.isfinite(size_multiplier).all()
    ):
        raise RuntimeError("paired-action decision sizing invariant failed")
    return PairedActionDecisions(
        actions=actions,
        score_bps=score,
        lower_tail_bps=np.where(eligible, selected_q20, 0.0),
        size_multiplier=size_multiplier,
        model_eligible=model_eligible,
        liquidity_eligible=liquidity,
        volatility_eligible=volatility,
        calibrated_view_median_bps=np.moveaxis(calibrated_point, 0, 2),
        raw_positive_count=raw_positive,
    )


__all__ = [
    "ACTION_NAMES",
    "CENTERED_SIGNED_FEATURES",
    "OBJECTIVE_IDS",
    "PAIRED_ACTION_CALIBRATION_SCHEMA_VERSION",
    "PAIRED_ACTION_MODEL_SCHEMA_VERSION",
    "Q20_ALPHA",
    "SEMIVOLATILITY_FEATURES",
    "SIGNED_FEATURES",
    "VIEW_IDS",
    "IsotonicCalibration",
    "OuterFold",
    "PairedActionCalibration",
    "PairedActionDecisions",
    "PairedActionModel",
    "PairedActionPanel",
    "PairedActionSpec",
    "PairedActionTrainingResult",
    "QuantileCalibration",
    "action_conditioned_feature_names",
    "apply_isotonic_calibration",
    "apply_paired_action_calibration",
    "apply_quantile_calibration",
    "build_monthly_outer_folds",
    "build_paired_action_panel",
    "embargoed_interval_mask",
    "fit_isotonic_calibration",
    "fit_paired_action_calibration",
    "fit_quantile_calibration",
    "load_paired_action_calibration",
    "load_paired_action_model",
    "paired_action_decisions",
    "pinball_loss",
    "predict_paired_action_model",
    "save_paired_action_calibration",
    "save_paired_action_model",
    "train_paired_action_model",
]
