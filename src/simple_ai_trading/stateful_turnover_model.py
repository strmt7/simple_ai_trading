"""Stateful turnover-aware hourly forecasting and replay for Round 43."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import math
from pathlib import Path
from typing import Callable, Mapping

import lightgbm as lgb
import numpy as np

from .cross_asset_cost_data import (
    MINUTE_MS,
    SYMBOLS,
    MinuteSeries,
    _feature_arrays,
)
from .cross_asset_cost_model import prediction_metrics
from .derivatives_hurdle_data import (
    DerivativesSourceEvidence,
    EXECUTION_CHARGE_BPS,
    FundingState,
    _funding_in_holding_window,
)
from .lightgbm_backend import lightgbm_backend_parameters


ROUND = 43
SEED = 4301
HORIZON_MINUTES = 60
FEATURE_SETS = ("baseline_71", "ai_research_augmented_77")
MODES = ("long_only", "long_short")
AI_FACTOR_NAMES = (
    "trend_quality_60m",
    "flow_confirmation_15m",
    "conditional_mean_reversion_240m",
    "residual_risk_adjusted_60m",
    "liquidity_conditioned_flow_5m",
    "downside_asymmetry_60m",
)
BASE_ONE_WAY_COST_BPS = 6.0
STRESS_ONE_WAY_COST_BPS = 8.0
COST_FILTER_LAMBDA = 2.0
MAXIMUM_HOLDING_HOURS = 24
SLEEVE_FRACTION = 1.0 / len(SYMBOLS)
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_BLOCK_HOURS = 168
FAMILYWISE_LOWER_QUANTILE = 0.0125


@dataclass(frozen=True)
class MonthlySchedule:
    evaluation_month: str
    training_start: str
    training_end: str
    early_stop_start: str
    early_stop_end: str
    calibration_start: str
    calibration_end: str
    evaluation_start: str
    evaluation_end: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


SCHEDULES = (
    MonthlySchedule(
        "2025-01",
        "2022-10-01",
        "2024-09-30",
        "2024-10-01",
        "2024-11-30",
        "2024-12-01",
        "2024-12-31",
        "2025-01-01",
        "2025-01-31",
    ),
    MonthlySchedule(
        "2025-02",
        "2022-11-01",
        "2024-10-31",
        "2024-11-01",
        "2024-12-31",
        "2025-01-01",
        "2025-01-31",
        "2025-02-01",
        "2025-02-28",
    ),
    MonthlySchedule(
        "2025-03",
        "2022-12-01",
        "2024-11-30",
        "2024-12-01",
        "2025-01-31",
        "2025-02-01",
        "2025-02-28",
        "2025-03-01",
        "2025-03-31",
    ),
    MonthlySchedule(
        "2025-04",
        "2023-01-01",
        "2024-12-31",
        "2025-01-01",
        "2025-02-28",
        "2025-03-01",
        "2025-03-31",
        "2025-04-01",
        "2025-04-30",
    ),
    MonthlySchedule(
        "2025-05",
        "2023-02-01",
        "2025-01-31",
        "2025-02-01",
        "2025-03-31",
        "2025-04-01",
        "2025-04-30",
        "2025-05-01",
        "2025-05-31",
    ),
    MonthlySchedule(
        "2025-06",
        "2023-03-01",
        "2025-02-28",
        "2025-03-01",
        "2025-04-30",
        "2025-05-01",
        "2025-05-31",
        "2025-06-01",
        "2025-06-30",
    ),
)


@dataclass(frozen=True)
class StatefulHourlyDataset:
    feature_names: tuple[str, ...]
    baseline_features: np.ndarray
    augmented_features: np.ndarray
    decision_time_ms: np.ndarray
    symbol_index: np.ndarray
    signed_pre_transition_utility_bps: np.ndarray
    funding_cash_flow_bps: np.ndarray
    source_evidence: DerivativesSourceEvidence
    dataset_sha256: str

    @property
    def rows(self) -> int:
        return int(self.baseline_features.shape[0])

    def feature_view(self, feature_set: str) -> np.ndarray:
        if feature_set == FEATURE_SETS[0]:
            return self.baseline_features
        if feature_set == FEATURE_SETS[1]:
            return self.augmented_features
        raise KeyError(feature_set)

    def names_for(self, feature_set: str) -> tuple[str, ...]:
        if feature_set == FEATURE_SETS[0]:
            return self.feature_names
        if feature_set == FEATURE_SETS[1]:
            return self.feature_names + AI_FACTOR_NAMES
        raise KeyError(feature_set)


@dataclass(frozen=True)
class StatefulModelArtifact:
    model_id: str
    feature_set: str
    evaluation_month: str
    feature_count: int
    training_rows: int
    early_stop_rows: int
    calibration_rows: int
    evaluation_rows: int
    amplitude_slope: float
    best_iteration: int
    backend_kind: str
    backend_device: str
    path: str
    bytes: int
    sha256: str
    reload_max_abs_prediction_error: float
    top_feature_gain: tuple[tuple[str, float], ...]

    def asdict(self) -> dict[str, object]:
        value = asdict(self)
        value["top_feature_gain"] = [
            {"feature": name, "gain": gain} for name, gain in self.top_feature_gain
        ]
        return value


@dataclass(frozen=True)
class ForecastBundle:
    predictions: Mapping[str, np.ndarray]
    artifacts: tuple[StatefulModelArtifact, ...]
    diagnostics: tuple[Mapping[str, object], ...]
    backend_kind: str
    backend_device: str


@dataclass(frozen=True)
class ReplayResult:
    candidate_id: str
    feature_set: str
    mode: str
    cost_scenario: str
    cost_bps: float
    timestamps_ms: np.ndarray
    forecasts_bps: np.ndarray
    target_bps: np.ndarray
    positions: np.ndarray
    position_age_hours: np.ndarray
    transition_units: np.ndarray
    transition_reasons: np.ndarray
    symbol_net_bps: np.ndarray
    portfolio_return_bps: np.ndarray
    metrics: Mapping[str, object]

    def summary(self) -> dict[str, object]:
        return dict(self.metrics)


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _date_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1000)


def _end_exclusive_ms(value: str) -> int:
    return _date_ms(value) + 86_400_000


def _array_identity(*arrays: np.ndarray, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    floor: float,
) -> np.ndarray:
    bounded = np.maximum(denominator, floor)
    return numerator / bounded


def build_ai_factor_matrix(
    baseline_features: np.ndarray,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    """Evaluate the six frozen finite factor programs without dynamic code."""

    if baseline_features.ndim != 2 or baseline_features.shape[1] != len(feature_names):
        raise ValueError("Round 43 baseline feature dimensions are inconsistent")
    positions = {name: index for index, name in enumerate(feature_names)}

    def feature(name: str) -> np.ndarray:
        if name not in positions:
            raise ValueError(f"Round 43 factor input is missing: {name}")
        return baseline_features[:, positions[name]].astype(np.float64)

    return_60 = feature("target_return_60m_bps")
    efficiency_60 = feature("target_path_efficiency_60m")
    return_15 = feature("target_return_15m_bps")
    flow_15 = feature("target_signed_taker_flow_15m")
    zscore_240 = feature("target_return_zscore_240m")
    residual_60 = feature("target_beta_residual_return_60m_bps")
    volatility_60 = feature("target_realized_volatility_60m_bps")
    flow_5 = feature("target_signed_taker_flow_5m")
    liquidity = feature("target_same_minute_of_week_liquidity_ratio")
    upside = feature("target_upside_semivolatility_60m_bps")
    downside = feature("target_downside_semivolatility_60m_bps")
    factors = np.column_stack(
        (
            np.tanh(return_60 / 50.0) * efficiency_60,
            np.tanh(return_15 / 25.0) * flow_15,
            -np.tanh(zscore_240) * (1.0 - efficiency_60),
            _safe_divide(residual_60, volatility_60, 5.0),
            flow_5 * np.minimum(liquidity, 2.0),
            _safe_divide(upside - downside, upside + downside, 1.0),
        )
    ).astype(np.float32)
    if factors.shape[1] != len(AI_FACTOR_NAMES) or not np.isfinite(factors).all():
        raise ValueError("Round 43 AI factor matrix is nonfinite or incomplete")
    return factors


def build_stateful_hourly_dataset(
    panel: Mapping[str, MinuteSeries],
    funding: Mapping[str, FundingState],
    source_evidence: DerivativesSourceEvidence,
    *,
    progress: ProgressCallback | None = None,
) -> StatefulHourlyDataset:
    """Build the compact causal hourly matrix directly from verified minute data."""

    reference = panel[SYMBOLS[0]]
    index = np.arange(reference.open_time_ms.size, dtype=np.int64)
    decision_mask = (
        (reference.open_time_ms >= _date_ms("2022-01-01"))
        & (reference.open_time_ms < _end_exclusive_ms("2025-06-30"))
        & ((reference.open_time_ms // MINUTE_MS) % 60 == 0)
        & (index + 1 + HORIZON_MINUTES < reference.open_time_ms.size)
    )
    decision_indices = np.flatnonzero(decision_mask)
    if decision_indices.size == 0:
        raise ValueError("Round 43 has no hourly decision rows")
    feature_blocks: list[np.ndarray] = []
    time_blocks: list[np.ndarray] = []
    symbol_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    funding_blocks: list[np.ndarray] = []
    expected_names: tuple[str, ...] | None = None
    for symbol_index, symbol in enumerate(SYMBOLS):
        if progress is not None:
            progress("round43_feature_build", {"status": "started", "symbol": symbol})
        series = panel[symbol]
        if not np.array_equal(series.open_time_ms, reference.open_time_ms):
            raise ValueError(f"Round 43 minute grid differs for {symbol}")
        names, arrays = _feature_arrays(panel, symbol)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f"Round 43 feature order differs for {symbol}")
        block = np.column_stack([values[decision_indices] for values in arrays]).astype(
            np.float32
        )
        if not np.isfinite(block).all():
            raise ValueError(f"Round 43 features contain nonfinite values for {symbol}")
        entry_indices = decision_indices + 1
        exit_indices = entry_indices + HORIZON_MINUTES
        entry_price = series.open[entry_indices]
        exit_price = series.open[exit_indices]
        entry_time = series.open_time_ms[entry_indices]
        exit_time = series.open_time_ms[exit_indices]
        settled_funding = _funding_in_holding_window(
            funding[symbol], entry_time, exit_time
        )
        long_gross = 10_000.0 * (exit_price / entry_price - 1.0)
        long_net = long_gross - EXECUTION_CHARGE_BPS - settled_funding
        short_net = -long_gross - EXECUTION_CHARGE_BPS + settled_funding
        signed_target = (long_net - short_net) / 2.0
        direct_target = long_gross - settled_funding
        if not np.allclose(signed_target, direct_target, rtol=0.0, atol=1e-10):
            raise ValueError(f"Round 43 signed target identity failed for {symbol}")
        feature_blocks.append(block)
        time_blocks.append(series.open_time_ms[decision_indices].copy())
        symbol_blocks.append(
            np.full(decision_indices.size, symbol_index, dtype=np.int8)
        )
        target_blocks.append(signed_target.astype(np.float32))
        funding_blocks.append(settled_funding.astype(np.float32))
        if progress is not None:
            progress(
                "round43_feature_build",
                {
                    "status": "complete",
                    "symbol": symbol,
                    "rows": int(block.shape[0]),
                    "feature_count": int(block.shape[1]),
                },
            )
    if expected_names is None:
        raise RuntimeError("Round 43 feature build produced no names")
    baseline = np.concatenate(feature_blocks, axis=0)
    decision_time_ms = np.concatenate(time_blocks)
    symbol_index = np.concatenate(symbol_blocks)
    target = np.concatenate(target_blocks)
    funding_values = np.concatenate(funding_blocks)
    order = np.lexsort((symbol_index, decision_time_ms))
    baseline = baseline[order]
    decision_time_ms = decision_time_ms[order]
    symbol_index = symbol_index[order]
    target = target[order]
    funding_values = funding_values[order]
    if baseline.shape[1] != 71:
        raise ValueError(
            f"Round 43 expected 71 baseline features, got {baseline.shape[1]}"
        )
    if baseline.shape[0] % len(SYMBOLS):
        raise ValueError("Round 43 hourly rows do not form complete symbol groups")
    grouped_time = decision_time_ms.reshape(-1, len(SYMBOLS))
    grouped_symbol = symbol_index.reshape(-1, len(SYMBOLS))
    if not np.all(grouped_time == grouped_time[:, :1]) or not np.array_equal(
        grouped_symbol,
        np.tile(np.arange(len(SYMBOLS), dtype=np.int8), (grouped_symbol.shape[0], 1)),
    ):
        raise ValueError("Round 43 chronological symbol grid is incomplete")
    factor_matrix = build_ai_factor_matrix(baseline, expected_names)
    augmented = np.column_stack((baseline, factor_matrix)).astype(np.float32)
    identity = _array_identity(
        decision_time_ms,
        symbol_index,
        baseline,
        factor_matrix,
        target,
        funding_values,
        names=expected_names + AI_FACTOR_NAMES,
    )
    return StatefulHourlyDataset(
        feature_names=expected_names,
        baseline_features=baseline,
        augmented_features=augmented,
        decision_time_ms=decision_time_ms,
        symbol_index=symbol_index,
        signed_pre_transition_utility_bps=target,
        funding_cash_flow_bps=funding_values,
        source_evidence=source_evidence,
        dataset_sha256=identity,
    )


def _window_mask(
    dataset: StatefulHourlyDataset,
    *,
    start: str,
    end: str,
) -> np.ndarray:
    start_ms = _date_ms(start)
    end_ms = _end_exclusive_ms(end)
    exit_time_ms = dataset.decision_time_ms + (HORIZON_MINUTES + 1) * MINUTE_MS
    return (
        (dataset.decision_time_ms >= start_ms)
        & (dataset.decision_time_ms < end_ms)
        & (exit_time_ms < end_ms)
    )


def schedule_masks(
    dataset: StatefulHourlyDataset,
    schedule: MonthlySchedule,
) -> dict[str, np.ndarray]:
    masks = {
        "training": _window_mask(
            dataset, start=schedule.training_start, end=schedule.training_end
        ),
        "early_stop": _window_mask(
            dataset, start=schedule.early_stop_start, end=schedule.early_stop_end
        ),
        "calibration": _window_mask(
            dataset, start=schedule.calibration_start, end=schedule.calibration_end
        ),
        "evaluation": _window_mask(
            dataset, start=schedule.evaluation_start, end=schedule.evaluation_end
        ),
    }
    combined = np.zeros(dataset.rows, dtype=np.int8)
    for mask in masks.values():
        combined += mask.astype(np.int8)
    if np.any(combined > 1) or any(not np.any(mask) for mask in masks.values()):
        raise ValueError(f"Round 43 invalid role masks for {schedule.evaluation_month}")
    return masks


def _amplitude_slope(actual: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.dot(prediction, prediction))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return 0.0
    slope = float(np.dot(actual, prediction) / denominator)
    if not math.isfinite(slope):
        raise ValueError("Round 43 amplitude slope is nonfinite")
    return min(4.0, max(0.0, slope))


def _diagnostic_rows(
    dataset: StatefulHourlyDataset,
    *,
    feature_set: str,
    evaluation_month: str,
    role: str,
    mask: np.ndarray,
    predictions: np.ndarray,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for symbol_name, symbol_value in (
        ("ALL", None),
        *zip(SYMBOLS, range(3), strict=True),
    ):
        selected = mask.copy()
        if symbol_value is not None:
            selected &= dataset.symbol_index == symbol_value
        metrics = prediction_metrics(
            dataset.signed_pre_transition_utility_bps[selected].astype(np.float64),
            predictions[selected].astype(np.float64),
        )
        output.append(
            {
                "feature_set": feature_set,
                "evaluation_month": evaluation_month,
                "role": role,
                "symbol": symbol_name,
                **metrics.asdict(),
            }
        )
    return output


def train_stateful_forecasts(
    dataset: StatefulHourlyDataset,
    *,
    model_dir: Path,
    compute_backend: str,
    progress: ProgressCallback | None = None,
) -> ForecastBundle:
    """Fit the twelve frozen monthly OpenCL-first regressors."""

    predictions = {
        feature_set: np.full(dataset.rows, np.nan, dtype=np.float32)
        for feature_set in FEATURE_SETS
    }
    artifacts: list[StatefulModelArtifact] = []
    diagnostics: list[Mapping[str, object]] = []
    backend_kinds: set[str] = set()
    backend_devices: set[str] = set()
    model_dir.mkdir(parents=True, exist_ok=True)
    target = dataset.signed_pre_transition_utility_bps
    for feature_set_index, feature_set in enumerate(FEATURE_SETS):
        features = dataset.feature_view(feature_set)
        feature_names = list(dataset.names_for(feature_set))
        for schedule_index, schedule in enumerate(SCHEDULES):
            masks = schedule_masks(dataset, schedule)
            seed = SEED + feature_set_index * 100 + schedule_index
            parameters, backend_kind, backend_device = lightgbm_backend_parameters(
                compute_backend,
                seed,
                reproducible=True,
            )
            parameters.update(
                {
                    "objective": "regression_l1",
                    "metric": "l1",
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "min_data_in_leaf": 100,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "lambda_l1": 0.1,
                    "lambda_l2": 1.0,
                    "max_bin": 255,
                    "feature_pre_filter": False,
                }
            )
            backend_kinds.add(backend_kind)
            backend_devices.add(backend_device)
            model_id = (
                f"round43_{feature_set}_{schedule.evaluation_month.replace('-', '')}"
            )
            if progress is not None:
                progress(
                    "round43_model_training",
                    {
                        "status": "started",
                        "model_id": model_id,
                        "training_rows": int(np.count_nonzero(masks["training"])),
                        "early_stop_rows": int(np.count_nonzero(masks["early_stop"])),
                        "backend_kind": backend_kind,
                        "backend_device": backend_device,
                    },
                )
            train_set = lgb.Dataset(
                features[masks["training"]],
                label=target[masks["training"]],
                feature_name=feature_names,
                free_raw_data=True,
            )
            validation_set = lgb.Dataset(
                features[masks["early_stop"]],
                label=target[masks["early_stop"]],
                feature_name=feature_names,
                reference=train_set,
                free_raw_data=True,
            )
            booster = lgb.train(
                parameters,
                train_set,
                num_boost_round=1_000,
                valid_sets=[validation_set],
                valid_names=["early_stop"],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            calibration_raw = np.asarray(
                booster.predict(
                    features[masks["calibration"]],
                    num_iteration=booster.best_iteration,
                ),
                dtype=np.float64,
            )
            slope = _amplitude_slope(
                target[masks["calibration"]].astype(np.float64),
                calibration_raw,
            )
            evaluation_raw = np.asarray(
                booster.predict(
                    features[masks["evaluation"]],
                    num_iteration=booster.best_iteration,
                ),
                dtype=np.float64,
            )
            calibrated = np.full(dataset.rows, np.nan, dtype=np.float32)
            calibrated[masks["calibration"]] = (calibration_raw * slope).astype(
                np.float32
            )
            calibrated[masks["evaluation"]] = (evaluation_raw * slope).astype(
                np.float32
            )
            predictions[feature_set][masks["evaluation"]] = calibrated[
                masks["evaluation"]
            ]
            model_path = model_dir / f"{model_id}.txt"
            booster.save_model(str(model_path), num_iteration=booster.best_iteration)
            reloaded = lgb.Booster(model_file=str(model_path))
            probe = np.flatnonzero(masks["early_stop"])[:4096]
            original_probe = np.asarray(
                booster.predict(features[probe], num_iteration=booster.best_iteration)
            )
            reload_probe = np.asarray(reloaded.predict(features[probe]))
            reload_error = float(np.max(np.abs(original_probe - reload_probe)))
            if not math.isfinite(reload_error) or reload_error > 1e-12:
                raise RuntimeError(f"{model_id} reload error is {reload_error}")
            gains = booster.feature_importance(importance_type="gain")
            order = np.argsort(gains)[::-1][:20]
            artifact = StatefulModelArtifact(
                model_id=model_id,
                feature_set=feature_set,
                evaluation_month=schedule.evaluation_month,
                feature_count=features.shape[1],
                training_rows=int(np.count_nonzero(masks["training"])),
                early_stop_rows=int(np.count_nonzero(masks["early_stop"])),
                calibration_rows=int(np.count_nonzero(masks["calibration"])),
                evaluation_rows=int(np.count_nonzero(masks["evaluation"])),
                amplitude_slope=slope,
                best_iteration=int(booster.best_iteration),
                backend_kind=backend_kind,
                backend_device=backend_device,
                path=str(model_path),
                bytes=model_path.stat().st_size,
                sha256=_file_sha256(model_path),
                reload_max_abs_prediction_error=reload_error,
                top_feature_gain=tuple(
                    (feature_names[int(item)], float(gains[int(item)]))
                    for item in order
                ),
            )
            artifacts.append(artifact)
            diagnostics.extend(
                _diagnostic_rows(
                    dataset,
                    feature_set=feature_set,
                    evaluation_month=schedule.evaluation_month,
                    role="calibration",
                    mask=masks["calibration"],
                    predictions=calibrated,
                )
            )
            diagnostics.extend(
                _diagnostic_rows(
                    dataset,
                    feature_set=feature_set,
                    evaluation_month=schedule.evaluation_month,
                    role="evaluation",
                    mask=masks["evaluation"],
                    predictions=calibrated,
                )
            )
            if progress is not None:
                progress(
                    "round43_model_training",
                    {
                        "status": "complete",
                        "model_id": model_id,
                        "best_iteration": artifact.best_iteration,
                        "amplitude_slope": slope,
                        "artifact_sha256": artifact.sha256,
                    },
                )
    if len(artifacts) != 12 or len(backend_kinds) != 1 or len(backend_devices) != 1:
        raise RuntimeError("Round 43 model artifact or backend count is inconsistent")
    evaluation_mask = _window_mask(dataset, start="2025-01-01", end="2025-06-30")
    for feature_set, values in predictions.items():
        if not np.isfinite(values[evaluation_mask]).all():
            raise ValueError(
                f"Round 43 {feature_set} evaluation predictions are incomplete"
            )
    return ForecastBundle(
        predictions=predictions,
        artifacts=tuple(artifacts),
        diagnostics=tuple(diagnostics),
        backend_kind=next(iter(backend_kinds)),
        backend_device=next(iter(backend_devices)),
    )


def _circular_block_bootstrap(
    values: np.ndarray,
    *,
    samples: int,
    block_size: int,
    seed: int,
    lower_quantile: float,
) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Round 43 bootstrap values are invalid")
    generator = np.random.default_rng(seed)
    block_count = math.ceil(values.size / block_size)
    offsets = np.arange(block_size, dtype=np.int64)
    means = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        starts = generator.integers(0, values.size, size=block_count)
        indexes = (starts[:, None] + offsets[None, :]) % values.size
        means[sample] = float(np.mean(values[indexes.ravel()[: values.size]]))
    lower, median, upper = np.quantile(
        means, (lower_quantile, 0.5, 1.0 - lower_quantile)
    )
    return {
        "lower_bps": float(lower),
        "median_bps": float(median),
        "upper_bps": float(upper),
        "lower_quantile": lower_quantile,
        "samples": samples,
        "block_hours": block_size,
    }


def _month_labels(timestamps_ms: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            datetime.fromtimestamp(value / 1000.0, UTC).strftime("%Y-%m")
            for value in timestamps_ms
        ]
    )


def _day_labels(timestamps_ms: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            datetime.fromtimestamp(value / 1000.0, UTC).date().isoformat()
            for value in timestamps_ms
        ]
    )


def _economic_metrics(
    *,
    candidate_id: str,
    feature_set: str,
    mode: str,
    cost_scenario: str,
    cost_bps: float,
    timestamps_ms: np.ndarray,
    positions: np.ndarray,
    transition_units: np.ndarray,
    transition_reasons: np.ndarray,
    final_boundary_exit_mask: np.ndarray,
    symbol_net_bps: np.ndarray,
    holding_durations: list[int],
    seed: int,
    independent_round_trips: bool = False,
) -> dict[str, object]:
    portfolio_bps = np.sum(symbol_net_bps, axis=1)
    returns = portfolio_bps / 10_000.0
    equity = np.cumprod(1.0 + returns)
    if np.any(equity <= 0.0) or not np.isfinite(equity).all():
        raise ValueError(f"Round 43 {candidate_id} produced invalid equity")
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))
    equity_with_start = np.concatenate(([1.0], equity))
    drawdown = 1.0 - equity_with_start / running_peak
    total_return = float(equity[-1] - 1.0)
    annualized_return = float(
        math.exp(math.log1p(total_return) * 8_760.0 / returns.size) - 1.0
    )
    standard_deviation = float(np.std(returns, ddof=1))
    annualized_volatility = standard_deviation * math.sqrt(8_760.0)
    sharpe = (
        float(np.mean(returns) / standard_deviation * math.sqrt(8_760.0))
        if standard_deviation > 0.0
        else 0.0
    )
    downside = returns[returns < 0.0]
    downside_deviation = (
        float(np.sqrt(np.mean(downside * downside))) if downside.size else 0.0
    )
    sortino = (
        float(np.mean(returns) / downside_deviation * math.sqrt(8_760.0))
        if downside_deviation > 0.0
        else 0.0
    )
    maximum_drawdown = float(np.max(drawdown))
    calmar = annualized_return / maximum_drawdown if maximum_drawdown > 0.0 else 0.0
    gains = float(np.sum(portfolio_bps[portfolio_bps > 0.0]))
    losses = float(-np.sum(portfolio_bps[portfolio_bps < 0.0]))
    profit_factor = gains / losses if losses > 0.0 else None
    profit_factor_is_infinite = losses == 0.0 and gains > 0.0
    quantile_05 = float(np.quantile(returns, 0.05))
    expected_shortfall = float(np.mean(returns[returns <= quantile_05]))
    months = _month_labels(timestamps_ms)
    days = _day_labels(timestamps_ms)
    monthly: list[dict[str, object]] = []
    for month in sorted(set(months)):
        selected = months == month
        month_return = float(np.prod(1.0 + returns[selected]) - 1.0)
        monthly.append(
            {
                "month": month,
                "hours": int(np.count_nonzero(selected)),
                "total_net_return_fraction": month_return,
                "mean_hourly_net_bps": float(np.mean(portfolio_bps[selected])),
                "transition_events": int(
                    np.count_nonzero(transition_units[selected] > 0.0)
                ),
            }
        )
    symbol_rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        if independent_round_trips:
            symbol_transition_events = 2 * int(
                np.count_nonzero(positions[:, symbol_index] != 0)
            )
        else:
            symbol_transition_events = int(
                np.count_nonzero(transition_reasons[:, symbol_index])
            ) + int(final_boundary_exit_mask[symbol_index])
        symbol_rows.append(
            {
                "symbol": symbol,
                "weighted_total_net_bps": float(
                    np.sum(symbol_net_bps[:, symbol_index])
                ),
                "transition_events": symbol_transition_events,
                "transition_units": float(np.sum(transition_units[:, symbol_index])),
                "active_hours": int(np.count_nonzero(positions[:, symbol_index] != 0)),
            }
        )
    active_day_count = len(
        {
            str(day)
            for day, active in zip(days, np.any(positions != 0, axis=1), strict=True)
            if active
        }
    )
    bootstrap = _circular_block_bootstrap(
        portfolio_bps,
        samples=BOOTSTRAP_SAMPLES,
        block_size=BOOTSTRAP_BLOCK_HOURS,
        seed=seed,
        lower_quantile=FAMILYWISE_LOWER_QUANTILE,
    )
    if independent_round_trips:
        entries = int(np.count_nonzero(positions))
        exits = entries
        reversals = 0
        forced_exits = 0
        final_exits = 0
        transition_events = entries + exits
    else:
        entries = int(np.count_nonzero(transition_reasons == 1))
        exits = int(np.count_nonzero(transition_reasons == 2))
        reversals = int(np.count_nonzero(transition_reasons == 3))
        forced_exits = int(np.count_nonzero(transition_reasons == 4))
        final_exits = int(np.count_nonzero(final_boundary_exit_mask))
        transition_events = int(np.count_nonzero(transition_reasons)) + final_exits
    return {
        "candidate_id": candidate_id,
        "feature_set": feature_set,
        "mode": mode,
        "cost_scenario": cost_scenario,
        "one_way_cost_bps": cost_bps,
        "hours": int(returns.size),
        "total_net_return_fraction": total_return,
        "annualized_net_return_fraction": annualized_return,
        "annualized_volatility_fraction": annualized_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "maximum_drawdown_fraction": maximum_drawdown,
        "profit_factor": profit_factor,
        "profit_factor_is_infinite": profit_factor_is_infinite,
        "mean_hourly_net_bps": float(np.mean(portfolio_bps)),
        "median_hourly_net_bps": float(np.median(portfolio_bps)),
        "positive_hour_fraction": float(np.mean(portfolio_bps > 0.0)),
        "hourly_value_at_risk_95_fraction": max(0.0, -quantile_05),
        "hourly_expected_shortfall_95_fraction": max(0.0, -expected_shortfall),
        "transition_events": transition_events,
        "transition_units": float(np.sum(transition_units)),
        "entries": entries,
        "exits": exits,
        "reversals": reversals,
        "maximum_hold_forced_exits": forced_exits,
        "final_boundary_exits": final_exits,
        "active_hours": int(np.count_nonzero(np.any(positions != 0, axis=1))),
        "active_days": active_day_count,
        "average_gross_exposure": float(np.mean(np.sum(np.abs(positions), axis=1)))
        * SLEEVE_FRACTION,
        "maximum_gross_exposure": float(np.max(np.sum(np.abs(positions), axis=1)))
        * SLEEVE_FRACTION,
        "mean_completed_holding_hours": (
            float(np.mean(holding_durations)) if holding_durations else 0.0
        ),
        "maximum_completed_holding_hours": max(holding_durations, default=0),
        "positive_months": int(
            sum(float(row["total_net_return_fraction"]) > 0.0 for row in monthly)
        ),
        "monthly": monthly,
        "symbols": symbol_rows,
        "bootstrap_mean_hourly_net_bps": bootstrap,
    }


def _evaluation_grid(
    dataset: StatefulHourlyDataset,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = _window_mask(dataset, start="2025-01-01", end="2025-06-30")
    indexes = np.flatnonzero(mask)
    if indexes.size % len(SYMBOLS):
        raise ValueError("Round 43 evaluation grid has incomplete symbol groups")
    timestamps = dataset.decision_time_ms[indexes].reshape(-1, len(SYMBOLS))
    if not np.all(timestamps == timestamps[:, :1]):
        raise ValueError("Round 43 evaluation timestamps are not aligned")
    forecast_grid = predictions[indexes].reshape(-1, len(SYMBOLS)).astype(np.float64)
    target_grid = (
        dataset.signed_pre_transition_utility_bps[indexes]
        .reshape(-1, len(SYMBOLS))
        .astype(np.float64)
    )
    if not np.isfinite(forecast_grid).all() or not np.isfinite(target_grid).all():
        raise ValueError("Round 43 evaluation grid contains nonfinite values")
    return timestamps[:, 0].copy(), forecast_grid, target_grid


def replay_stateful_policy(
    dataset: StatefulHourlyDataset,
    predictions: np.ndarray,
    *,
    feature_set: str,
    mode: str,
    cost_scenario: str,
    cost_bps: float,
    seed: int,
) -> ReplayResult:
    if mode not in MODES or cost_bps <= 0.0:
        raise ValueError("Round 43 replay mode or cost is invalid")
    timestamps, forecasts, targets = _evaluation_grid(dataset, predictions)
    hours = timestamps.size
    positions = np.zeros((hours, len(SYMBOLS)), dtype=np.int8)
    ages = np.zeros((hours, len(SYMBOLS)), dtype=np.int16)
    transitions = np.zeros((hours, len(SYMBOLS)), dtype=np.float64)
    reasons = np.zeros((hours, len(SYMBOLS)), dtype=np.int8)
    symbol_net = np.zeros((hours, len(SYMBOLS)), dtype=np.float64)
    current = np.zeros(len(SYMBOLS), dtype=np.int8)
    current_age = np.zeros(len(SYMBOLS), dtype=np.int16)
    holding_durations: list[int] = []
    for hour in range(hours):
        forecast = forecasts[hour]
        if mode == "long_only":
            proposed = np.where(forecast > 0.0, 1, 0).astype(np.int8)
        else:
            proposed = np.sign(forecast).astype(np.int8)
        next_position = current.copy()
        for symbol_index in range(len(SYMBOLS)):
            previous = int(current[symbol_index])
            desired = int(proposed[symbol_index])
            if (
                previous != 0
                and int(current_age[symbol_index]) >= MAXIMUM_HOLDING_HOURS
            ):
                next_position[symbol_index] = 0
                reasons[hour, symbol_index] = 4
                holding_durations.append(int(current_age[symbol_index]))
                continue
            if desired == previous:
                continue
            units = abs(desired - previous)
            hurdle = COST_FILTER_LAMBDA * cost_bps * units
            if abs(float(forecast[symbol_index])) <= hurdle:
                continue
            next_position[symbol_index] = desired
            if previous == 0 and desired != 0:
                reasons[hour, symbol_index] = 1
            elif previous != 0 and desired == 0:
                reasons[hour, symbol_index] = 2
                holding_durations.append(int(current_age[symbol_index]))
            else:
                reasons[hour, symbol_index] = 3
                holding_durations.append(int(current_age[symbol_index]))
        transition = np.abs(next_position - current).astype(np.float64)
        transitions[hour] = transition
        positions[hour] = next_position
        symbol_net[hour] = (
            next_position.astype(np.float64) * targets[hour] - cost_bps * transition
        ) * SLEEVE_FRACTION
        for symbol_index in range(len(SYMBOLS)):
            if next_position[symbol_index] == 0:
                current_age[symbol_index] = 0
            elif next_position[symbol_index] == current[symbol_index]:
                current_age[symbol_index] += 1
            else:
                current_age[symbol_index] = 1
        ages[hour] = current_age
        current = next_position
    final_transition = np.abs(current).astype(np.float64)
    if np.any(final_transition > 0.0):
        transitions[-1] += final_transition
        symbol_net[-1] -= cost_bps * final_transition * SLEEVE_FRACTION
        for symbol_index in np.flatnonzero(final_transition > 0.0):
            holding_durations.append(int(current_age[symbol_index]))
    candidate_id = f"{feature_set}_{mode}"
    metrics = _economic_metrics(
        candidate_id=candidate_id,
        feature_set=feature_set,
        mode=mode,
        cost_scenario=cost_scenario,
        cost_bps=cost_bps,
        timestamps_ms=timestamps,
        positions=positions,
        transition_units=transitions,
        transition_reasons=reasons,
        final_boundary_exit_mask=final_transition > 0.0,
        symbol_net_bps=symbol_net,
        holding_durations=holding_durations,
        seed=seed,
    )
    return ReplayResult(
        candidate_id=candidate_id,
        feature_set=feature_set,
        mode=mode,
        cost_scenario=cost_scenario,
        cost_bps=cost_bps,
        timestamps_ms=timestamps,
        forecasts_bps=forecasts,
        target_bps=targets,
        positions=positions,
        position_age_hours=ages,
        transition_units=transitions,
        transition_reasons=reasons,
        symbol_net_bps=symbol_net,
        portfolio_return_bps=np.sum(symbol_net, axis=1),
        metrics=metrics,
    )


def replay_independent_hourly(
    dataset: StatefulHourlyDataset,
    predictions: np.ndarray,
    *,
    feature_set: str,
    mode: str,
    cost_scenario: str,
    cost_bps: float,
    seed: int,
) -> ReplayResult:
    timestamps, forecasts, targets = _evaluation_grid(dataset, predictions)
    if mode == "long_only":
        desired = np.where(forecasts > 0.0, 1, 0).astype(np.int8)
    elif mode == "long_short":
        desired = np.sign(forecasts).astype(np.int8)
    else:
        raise ValueError(mode)
    accepted = np.abs(forecasts) > COST_FILTER_LAMBDA * cost_bps * np.abs(desired)
    positions = np.where(accepted, desired, 0).astype(np.int8)
    transitions = 2.0 * np.abs(positions).astype(np.float64)
    reasons = np.where(positions != 0, 1, 0).astype(np.int8)
    symbol_net = (
        positions.astype(np.float64) * targets - cost_bps * transitions
    ) * SLEEVE_FRACTION
    ages = np.abs(positions).astype(np.int16)
    holding = [1] * int(np.count_nonzero(positions))
    candidate_id = f"independent_hourly_{feature_set}_{mode}"
    metrics = _economic_metrics(
        candidate_id=candidate_id,
        feature_set=feature_set,
        mode=mode,
        cost_scenario=cost_scenario,
        cost_bps=cost_bps,
        timestamps_ms=timestamps,
        positions=positions,
        transition_units=transitions,
        transition_reasons=reasons,
        final_boundary_exit_mask=np.zeros(len(SYMBOLS), dtype=bool),
        symbol_net_bps=symbol_net,
        holding_durations=holding,
        seed=seed,
        independent_round_trips=True,
    )
    return ReplayResult(
        candidate_id=candidate_id,
        feature_set=feature_set,
        mode=mode,
        cost_scenario=cost_scenario,
        cost_bps=cost_bps,
        timestamps_ms=timestamps,
        forecasts_bps=forecasts,
        target_bps=targets,
        positions=positions,
        position_age_hours=ages,
        transition_units=transitions,
        transition_reasons=reasons,
        symbol_net_bps=symbol_net,
        portfolio_return_bps=np.sum(symbol_net, axis=1),
        metrics=metrics,
    )


def replay_always_long(
    dataset: StatefulHourlyDataset,
    *,
    cost_scenario: str,
    cost_bps: float,
    seed: int,
) -> ReplayResult:
    evaluation = np.zeros(dataset.rows, dtype=np.float32)
    timestamps, forecasts, targets = _evaluation_grid(dataset, evaluation)
    positions = np.ones_like(targets, dtype=np.int8)
    transitions = np.zeros_like(targets, dtype=np.float64)
    transitions[0] = 1.0
    transitions[-1] += 1.0
    reasons = np.zeros_like(positions, dtype=np.int8)
    reasons[0] = 1
    symbol_net = (targets - cost_bps * transitions) * SLEEVE_FRACTION
    ages = np.repeat(
        np.arange(1, timestamps.size + 1, dtype=np.int16)[:, None],
        len(SYMBOLS),
        axis=1,
    )
    metrics = _economic_metrics(
        candidate_id="always_long_fixed_sleeves",
        feature_set="comparator",
        mode="long_only",
        cost_scenario=cost_scenario,
        cost_bps=cost_bps,
        timestamps_ms=timestamps,
        positions=positions,
        transition_units=transitions,
        transition_reasons=reasons,
        final_boundary_exit_mask=np.ones(len(SYMBOLS), dtype=bool),
        symbol_net_bps=symbol_net,
        holding_durations=[timestamps.size] * len(SYMBOLS),
        seed=seed,
    )
    return ReplayResult(
        candidate_id="always_long_fixed_sleeves",
        feature_set="comparator",
        mode="long_only",
        cost_scenario=cost_scenario,
        cost_bps=cost_bps,
        timestamps_ms=timestamps,
        forecasts_bps=forecasts,
        target_bps=targets,
        positions=positions,
        position_age_hours=ages,
        transition_units=transitions,
        transition_reasons=reasons,
        symbol_net_bps=symbol_net,
        portfolio_return_bps=np.sum(symbol_net, axis=1),
        metrics=metrics,
    )


def evaluate_stateful_gate(
    base: ReplayResult,
    stress: ReplayResult,
    *,
    all_models_exact_reload: bool,
) -> dict[str, object]:
    stress_symbols = {
        str(row["symbol"]): row
        for row in stress.metrics["symbols"]  # type: ignore[index]
    }
    checks = {
        "base_total_net_return_positive": float(
            base.metrics["total_net_return_fraction"]
        )
        > 0.0,
        "stress_total_net_return_positive": float(
            stress.metrics["total_net_return_fraction"]
        )
        > 0.0,
        "stress_profit_factor_at_least_1_05": (
            bool(stress.metrics["profit_factor_is_infinite"])
            or (
                stress.metrics["profit_factor"] is not None
                and float(stress.metrics["profit_factor"]) >= 1.05
            )
        ),
        "stress_familywise_bootstrap_lower_positive": float(
            stress.metrics["bootstrap_mean_hourly_net_bps"]["lower_bps"]  # type: ignore[index]
        )
        > 0.0,
        "stress_positive_months_at_least_4": int(stress.metrics["positive_months"])
        >= 4,
        "stress_maximum_drawdown_at_most_10_percent": float(
            stress.metrics["maximum_drawdown_fraction"]
        )
        <= 0.10,
        "stress_transition_events_at_least_60": int(stress.metrics["transition_events"])
        >= 60,
        "stress_each_symbol_transition_events_at_least_10": all(
            int(stress_symbols[symbol]["transition_events"]) >= 10 for symbol in SYMBOLS
        ),
        "stress_active_days_at_least_90": int(stress.metrics["active_days"]) >= 90,
        "all_models_exact_reload": all_models_exact_reload,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reasons": [name for name, passed in checks.items() if not passed],
    }


def evaluate_ai_uplift(
    baseline_base: ReplayResult,
    augmented_base: ReplayResult,
    baseline_stress: ReplayResult,
    augmented_stress: ReplayResult,
) -> dict[str, object]:
    for result in (
        baseline_base,
        augmented_base,
        baseline_stress,
        augmented_stress,
    ):
        if result.mode != "long_only":
            raise ValueError("Round 43 AI uplift is registered for long-only only")
    paired_delta = (
        augmented_stress.portfolio_return_bps - baseline_stress.portfolio_return_bps
    )
    interval = _circular_block_bootstrap(
        paired_delta,
        samples=BOOTSTRAP_SAMPLES,
        block_size=BOOTSTRAP_BLOCK_HOURS,
        seed=SEED + 900,
        lower_quantile=0.025,
    )
    checks = {
        "augmented_base_total_return_exceeds_baseline": float(
            augmented_base.metrics["total_net_return_fraction"]
        )
        > float(baseline_base.metrics["total_net_return_fraction"]),
        "augmented_stress_total_return_exceeds_baseline": float(
            augmented_stress.metrics["total_net_return_fraction"]
        )
        > float(baseline_stress.metrics["total_net_return_fraction"]),
        "paired_stress_bootstrap_lower_delta_positive": interval["lower_bps"] > 0.0,
        "augmented_stress_drawdown_not_worse": float(
            augmented_stress.metrics["maximum_drawdown_fraction"]
        )
        <= float(baseline_stress.metrics["maximum_drawdown_fraction"]),
        "augmented_stress_positive_months_not_lower": int(
            augmented_stress.metrics["positive_months"]
        )
        >= int(baseline_stress.metrics["positive_months"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reasons": [name for name, passed in checks.items() if not passed],
        "paired_stress_hourly_delta_bps": {
            "observed_mean": float(np.mean(paired_delta)),
            **interval,
        },
    }


__all__ = [
    "AI_FACTOR_NAMES",
    "BASE_ONE_WAY_COST_BPS",
    "BOOTSTRAP_BLOCK_HOURS",
    "BOOTSTRAP_SAMPLES",
    "COST_FILTER_LAMBDA",
    "FAMILYWISE_LOWER_QUANTILE",
    "FEATURE_SETS",
    "ForecastBundle",
    "HORIZON_MINUTES",
    "MAXIMUM_HOLDING_HOURS",
    "MODES",
    "MonthlySchedule",
    "ReplayResult",
    "SCHEDULES",
    "SEED",
    "STRESS_ONE_WAY_COST_BPS",
    "StatefulHourlyDataset",
    "StatefulModelArtifact",
    "build_ai_factor_matrix",
    "build_stateful_hourly_dataset",
    "evaluate_ai_uplift",
    "evaluate_stateful_gate",
    "replay_always_long",
    "replay_independent_hourly",
    "replay_stateful_policy",
    "schedule_masks",
    "train_stateful_forecasts",
]
