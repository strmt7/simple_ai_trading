"""Causal post-pretraining benchmark for optional financial foundation forecasts."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import sqlite3
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .foundation_forecast import (
    KRONOS_MAX_CONTEXT,
    KRONOS_PRETRAINING_CUTOFF,
)
from .foundation_model_source import provision_kronos_source, verify_kronos_source
from .foundation_worker_client import FoundationWorkerError, FoundationWorkerSupervisor
from .storage import write_json_atomic


FOUNDATION_BENCHMARK_VERSION = "kronos-postcutoff-causal-v1"
FOUNDATION_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
FOUNDATION_SELECTION_START_MS = 1_719_792_000_000  # 2024-07-01T00:00:00Z
FOUNDATION_SELECTION_END_EXCLUSIVE_MS = 1_767_225_600_000  # 2026-01-01T00:00:00Z
BINANCE_ARCHIVE_SOURCE = "binance_public_archive"

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class FoundationBenchmarkConfig:
    database_path: str = "data/market_data.sqlite"
    source_cache_root: str | None = None
    model_size: str = "base"
    backend: str = "directml"
    bootstrap_source: bool = False
    repair_source: bool = False
    require_accelerator: bool = True
    start_ms: int = FOUNDATION_SELECTION_START_MS
    end_exclusive_ms: int = FOUNDATION_SELECTION_END_EXCLUSIVE_MS
    samples_per_symbol: int = 128
    bar_minutes: int = 5
    lookback_bars: int = 480
    prediction_bars: int = 12
    batch_size: int = 3
    inference_samples: int = 10
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 0.9
    include_volume: bool = False
    seed: int = 17
    bootstrap_samples: int = 2_000
    worker_timeout_seconds: float = 60.0
    max_worker_restarts: int = 5
    worker_rotation_batches: int = 20

    def validated(self) -> "FoundationBenchmarkConfig":
        if tuple(FOUNDATION_SYMBOLS) != ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            raise RuntimeError("foundation benchmark symbol contract changed unexpectedly")
        if self.start_ms < FOUNDATION_SELECTION_START_MS:
            raise ValueError("foundation benchmark starts before the model pretraining cutoff")
        if self.end_exclusive_ms > FOUNDATION_SELECTION_END_EXCLUSIVE_MS:
            raise ValueError("2026 foundation-model observations are sealed terminal evidence")
        if self.end_exclusive_ms <= self.start_ms:
            raise ValueError("foundation benchmark end must be after start")
        if self.bar_minutes != 5:
            raise ValueError("Kronos evaluation is currently pinned to its validated 5-minute floor")
        if self.samples_per_symbol < 1:
            raise ValueError("samples_per_symbol must be positive")
        if self.lookback_bars < 32 or self.prediction_bars < 1:
            raise ValueError("lookback and prediction windows are too short")
        if self.lookback_bars + self.prediction_bars > KRONOS_MAX_CONTEXT:
            raise ValueError(
                "lookback_bars + prediction_bars exceeds the no-fallback Kronos context"
            )
        if self.batch_size < 1 or self.inference_samples < 1:
            raise ValueError("batch_size and inference_samples must be positive")
        if str(self.backend).strip().lower() == "directml" and self.batch_size > 3:
            raise ValueError(
                "DirectML foundation batches are capped at 3 after live-host resource failures"
            )
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not math.isfinite(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.seed < 0 or self.bootstrap_samples < 100:
            raise ValueError("seed must be non-negative and bootstrap_samples must be at least 100")
        if self.worker_timeout_seconds < 1.0 or self.max_worker_restarts < 0:
            raise ValueError("worker timeout/restart contract is invalid")
        if self.worker_rotation_batches < 0:
            raise ValueError("worker_rotation_batches must be non-negative")
        return self

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArchiveCoverageEvidence:
    symbol: str
    start_ms: int
    end_exclusive_ms: int
    row_count: int
    expected_rows: int
    minimum_open_time: int
    maximum_open_time: int
    sources: tuple[str, ...]
    required_months: tuple[str, ...]
    verified_archive_periods: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastContext:
    symbol: str
    decision_ms: int
    frame: object
    history_timestamps: object
    future_timestamps: object
    last_close: float
    actual_closes: tuple[float, ...]


@dataclass(frozen=True)
class ForecastObservation:
    symbol: str
    decision_ms: int
    decision_time_utc: str
    last_close: float
    predicted_average_return: float
    actual_average_return: float
    predicted_final_return: float
    actual_final_return: float
    absolute_error: float
    random_walk_absolute_error: float
    direction_correct: bool
    inference_batch: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FoundationBenchmarkReport:
    version: str
    generated_at_ms: int
    status: str
    trading_authority: bool
    reasons: tuple[str, ...]
    config: dict[str, object]
    evaluation: dict[str, object]
    source_evidence: tuple[dict[str, object], ...]
    engine: dict[str, object]
    metrics: dict[str, object]
    calibration: dict[str, object]
    bootstrap: dict[str, object]
    inference: dict[str, object]
    observation_count: int
    observations_sha256: str
    observations_path: str
    chart_sha256: str
    chart_path: str

    @property
    def predictive_candidate(self) -> bool:
        return self.status == "predictive_candidate"

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["predictive_candidate"] = self.predictive_candidate
        return payload


def parse_utc_ms(value: str) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError("UTC timestamp is empty")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("UTC timestamp must include a timezone")
    return int(parsed.astimezone(UTC).timestamp() * 1_000)


def _iso_utc(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1_000.0, tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def _month_periods(start_ms: int, end_exclusive_ms: int) -> tuple[str, ...]:
    cursor = datetime.fromtimestamp(start_ms / 1_000.0, tz=UTC).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    end = datetime.fromtimestamp((end_exclusive_ms - 1) / 1_000.0, tz=UTC)
    values: list[str] = []
    while cursor <= end:
        values.append(cursor.strftime("%Y-%m"))
        year = cursor.year + (1 if cursor.month == 12 else 0)
        month = 1 if cursor.month == 12 else cursor.month + 1
        cursor = cursor.replace(year=year, month=month)
    return tuple(values)


def _day_periods(start_ms: int, end_exclusive_ms: int) -> tuple[str, ...]:
    cursor = datetime.fromtimestamp(start_ms / 1_000.0, tz=UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end = datetime.fromtimestamp((end_exclusive_ms - 1) / 1_000.0, tz=UTC)
    values: list[str] = []
    while cursor <= end:
        values.append(cursor.strftime("%Y-%m-%d"))
        cursor = datetime.fromtimestamp(cursor.timestamp() + 86_400, tz=UTC)
    return tuple(values)


def _read_only_connection(path: str | Path) -> sqlite3.Connection:
    target = Path(path).resolve()
    if not target.is_file():
        raise RuntimeError(f"market database is missing: {target}")
    connection = sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True, timeout=30.0)
    connection.execute("PRAGMA query_only = ON")
    return connection


def decision_timestamps(config: FoundationBenchmarkConfig) -> tuple[int, ...]:
    step_ms = int(config.bar_minutes) * 60_000
    first = ((int(config.start_ms) + step_ms - 1) // step_ms) * step_ms
    last = ((int(config.end_exclusive_ms) - int(config.prediction_bars) * step_ms) // step_ms) * step_ms
    if last < first:
        raise ValueError("foundation benchmark window has no forecastable decision timestamps")
    slot_count = ((last - first) // step_ms) + 1
    requested = min(int(config.samples_per_symbol), slot_count)
    if requested == 1:
        return (first,)
    indexes = np.linspace(0, slot_count - 1, num=requested, dtype=np.int64)
    values = tuple(first + int(index) * step_ms for index in indexes)
    if len(set(values)) != requested:
        raise RuntimeError("deterministic foundation benchmark sampling produced duplicates")
    return values


def audit_archive_coverage(
    connection: sqlite3.Connection,
    config: FoundationBenchmarkConfig,
    decisions: Sequence[int],
) -> tuple[ArchiveCoverageEvidence, ...]:
    step_ms = int(config.bar_minutes) * 60_000
    coverage_start = int(decisions[0]) - int(config.lookback_bars) * step_ms
    coverage_end = int(decisions[-1]) + int(config.prediction_bars) * step_ms
    expected_rows = (coverage_end - coverage_start) // 60_000
    required_months = _month_periods(coverage_start, coverage_end)
    required_days = _day_periods(coverage_start, coverage_end)
    evidence: list[ArchiveCoverageEvidence] = []
    for symbol in FOUNDATION_SYMBOLS:
        row = connection.execute(
            """
            SELECT COUNT(*), MIN(open_time), MAX(open_time)
            FROM candles
            WHERE symbol = ? AND market_type = 'futures' AND interval = '1m'
              AND open_time >= ? AND open_time < ?
            """,
            (symbol, coverage_start, coverage_end),
        ).fetchone()
        row_count = int(row[0] or 0)
        minimum = int(row[1] or -1)
        maximum = int(row[2] or -1)
        if row_count != expected_rows:
            raise RuntimeError(
                f"{symbol} one-minute coverage mismatch: expected {expected_rows}, "
                f"received {row_count}"
            )
        if minimum != coverage_start or maximum != coverage_end - 60_000:
            raise RuntimeError(f"{symbol} one-minute coverage boundaries are not exact")
        source_rows = connection.execute(
            """
            SELECT source, COUNT(*)
            FROM candles
            WHERE symbol = ? AND market_type = 'futures' AND interval = '1m'
              AND open_time >= ? AND open_time < ?
            GROUP BY source
            """,
            (symbol, coverage_start, coverage_end),
        ).fetchall()
        sources = tuple(sorted(str(item[0]) for item in source_rows))
        if sources != (BINANCE_ARCHIVE_SOURCE,):
            raise RuntimeError(f"{symbol} source contract failed: {sources}")
        archive_rows = connection.execute(
            """
            SELECT DISTINCT period
            FROM archive_files
            WHERE symbol = ? AND market_type = 'futures' AND interval = '1m'
              AND status = 'complete' AND checksum_status = 'verified'
            """,
            (symbol,),
        ).fetchall()
        verified = {str(item[0]) for item in archive_rows}
        accepted_periods: list[str] = []
        missing_periods: list[str] = []
        for month in required_months:
            if month in verified:
                accepted_periods.append(month)
                continue
            month_days = tuple(day for day in required_days if day.startswith(f"{month}-"))
            missing_days = tuple(day for day in month_days if day not in verified)
            if missing_days:
                missing_periods.extend(missing_days)
            else:
                accepted_periods.extend(month_days)
        if missing_periods:
            raise RuntimeError(
                f"{symbol} has unverified archive periods: {', '.join(missing_periods)}"
            )
        evidence.append(
            ArchiveCoverageEvidence(
                symbol=symbol,
                start_ms=coverage_start,
                end_exclusive_ms=coverage_end,
                row_count=row_count,
                expected_rows=expected_rows,
                minimum_open_time=minimum,
                maximum_open_time=maximum,
                sources=sources,
                required_months=required_months,
                verified_archive_periods=tuple(accepted_periods),
            )
        )
    return tuple(evidence)


def _aggregate_context(
    connection: sqlite3.Connection,
    symbol: str,
    decision_ms: int,
    config: FoundationBenchmarkConfig,
) -> ForecastContext:
    import pandas as pd

    step_ms = int(config.bar_minutes) * 60_000
    start_ms = int(decision_ms) - int(config.lookback_bars) * step_ms
    end_ms = int(decision_ms) + int(config.prediction_bars) * step_ms
    rows = connection.execute(
        """
        SELECT open_time, open, high, low, close, volume, quote_volume, source
        FROM candles
        WHERE symbol = ? AND market_type = 'futures' AND interval = '1m'
          AND open_time >= ? AND open_time < ?
        ORDER BY open_time
        """,
        (symbol, start_ms, end_ms),
    ).fetchall()
    expected_minutes = (int(config.lookback_bars) + int(config.prediction_bars)) * int(
        config.bar_minutes
    )
    if len(rows) != expected_minutes:
        raise RuntimeError(
            f"{symbol} context {decision_ms} has {len(rows)}/{expected_minutes} one-minute rows"
        )
    timestamps = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    expected_timestamps = np.arange(start_ms, end_ms, 60_000, dtype=np.int64)
    if not np.array_equal(timestamps, expected_timestamps):
        raise RuntimeError(f"{symbol} context {decision_ms} has a one-minute timestamp gap")
    if any(str(row[7]) != BINANCE_ARCHIVE_SOURCE for row in rows):
        raise RuntimeError(f"{symbol} context {decision_ms} contains an invalid source")
    values = np.asarray([row[1:7] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{symbol} context {decision_ms} contains non-finite values")
    if np.any(values[:, 1] < values[:, 2]) or np.any(values[:, :6] < 0.0):
        raise RuntimeError(f"{symbol} context {decision_ms} violates candle invariants")
    per_bar = int(config.bar_minutes)
    blocks = values.reshape(-1, per_bar, values.shape[1])
    aggregated = np.column_stack(
        (
            blocks[:, 0, 0],
            blocks[:, :, 1].max(axis=1),
            blocks[:, :, 2].min(axis=1),
            blocks[:, -1, 3],
            blocks[:, :, 4].sum(axis=1),
            blocks[:, :, 5].sum(axis=1),
        )
    )
    if np.any(aggregated[:, 1] < aggregated[:, 2]):
        raise RuntimeError(f"{symbol} aggregated context violates OHLC invariants")
    bar_times = timestamps[::per_bar]
    history_count = int(config.lookback_bars)
    history_values = aggregated[:history_count]
    future_values = aggregated[history_count:]
    columns = ("open", "high", "low", "close", "volume", "amount")
    frame = pd.DataFrame(history_values, columns=columns)
    if not config.include_volume:
        frame = frame.loc[:, ("open", "high", "low", "close")]
    history_timestamps = pd.Series(pd.to_datetime(bar_times[:history_count], unit="ms", utc=True))
    future_timestamps = pd.Series(pd.to_datetime(bar_times[history_count:], unit="ms", utc=True))
    return ForecastContext(
        symbol=symbol,
        decision_ms=int(decision_ms),
        frame=frame,
        history_timestamps=history_timestamps,
        future_timestamps=future_timestamps,
        last_close=float(history_values[-1, 3]),
        actual_closes=tuple(float(value) for value in future_values[:, 3]),
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + end - 1) / 2.0
        ranks[order[index:end]] = average_rank
        index = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) != len(left):
        return 0.0
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _metrics(observations: Sequence[ForecastObservation]) -> dict[str, object]:
    predicted = np.asarray([item.predicted_average_return for item in observations])
    actual = np.asarray([item.actual_average_return for item in observations])
    final_predicted = np.asarray([item.predicted_final_return for item in observations])
    final_actual = np.asarray([item.actual_final_return for item in observations])
    model_errors = np.abs(predicted - actual)
    baseline_errors = np.abs(actual)
    direction_mask = (predicted != 0.0) & (actual != 0.0)
    direction_accuracy = (
        float(np.mean(np.sign(predicted[direction_mask]) == np.sign(actual[direction_mask])))
        if bool(np.any(direction_mask))
        else 0.0
    )
    model_mae = float(np.mean(model_errors))
    baseline_mae = float(np.mean(baseline_errors))
    return {
        "observations": len(observations),
        "mean_predicted_average_return": float(np.mean(predicted)),
        "mean_actual_average_return": float(np.mean(actual)),
        "model_mae": model_mae,
        "random_walk_mae": baseline_mae,
        "mae_improvement": baseline_mae - model_mae,
        "mae_improvement_pct": (
            (baseline_mae - model_mae) / baseline_mae if baseline_mae > 0.0 else 0.0
        ),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "information_coefficient": _correlation(predicted, actual),
        "rank_information_coefficient": _correlation(
            _rankdata(predicted), _rankdata(actual)
        ),
        "direction_accuracy": direction_accuracy,
        "direction_observations": int(np.sum(direction_mask)),
        "final_return_information_coefficient": _correlation(final_predicted, final_actual),
        "final_return_mae": float(np.mean(np.abs(final_predicted - final_actual))),
        "random_walk_final_return_mae": float(np.mean(np.abs(final_actual))),
    }


def summarize_metrics(
    observations: Sequence[ForecastObservation],
) -> dict[str, object]:
    by_symbol = {
        symbol: _metrics([item for item in observations if item.symbol == symbol])
        for symbol in FOUNDATION_SYMBOLS
    }
    return {"overall": _metrics(observations), "by_symbol": by_symbol}


def _fit_nonnegative_scale(observations: Sequence[ForecastObservation]) -> float:
    numerator = sum(
        item.predicted_average_return * item.actual_average_return for item in observations
    )
    denominator = sum(item.predicted_average_return**2 for item in observations)
    if denominator <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, numerator / denominator)))


def _scale_observation(item: ForecastObservation, scale: float) -> ForecastObservation:
    predicted_average = float(scale) * item.predicted_average_return
    predicted_final = float(scale) * item.predicted_final_return
    return replace(
        item,
        predicted_average_return=predicted_average,
        predicted_final_return=predicted_final,
        absolute_error=abs(predicted_average - item.actual_average_return),
        direction_correct=(
            predicted_average != 0.0
            and item.actual_average_return != 0.0
            and math.copysign(1.0, predicted_average)
            == math.copysign(1.0, item.actual_average_return)
        ),
    )


def calibrate_forecast_amplitude(
    observations: Sequence[ForecastObservation],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, object], tuple[ForecastObservation, ...]]:
    """Fit symbol-specific nonnegative scales on the earlier half only."""

    timestamps = tuple(sorted({item.decision_ms for item in observations}))
    if len(timestamps) < 4:
        raise RuntimeError("foundation calibration needs at least four decision timestamps")
    boundary = timestamps[len(timestamps) // 2]
    tuning = tuple(item for item in observations if item.decision_ms < boundary)
    selection = tuple(item for item in observations if item.decision_ms >= boundary)
    scales: dict[str, float] = {}
    calibrated: list[ForecastObservation] = []
    symbol_evidence: dict[str, object] = {}
    eligible_symbols: list[str] = []
    for symbol in FOUNDATION_SYMBOLS:
        symbol_tuning = tuple(item for item in tuning if item.symbol == symbol)
        symbol_selection = tuple(item for item in selection if item.symbol == symbol)
        scale = _fit_nonnegative_scale(symbol_tuning)
        scales[symbol] = scale
        transformed = tuple(_scale_observation(item, scale) for item in symbol_selection)
        calibrated.extend(transformed)
        raw_metrics = _metrics(symbol_selection)
        calibrated_metrics = _metrics(transformed)
        reasons: list[str] = []
        if scale <= 0.0:
            reasons.append("calibration_scale_is_zero_abstain")
        if float(calibrated_metrics["mae_improvement"]) <= 0.0:
            reasons.append("calibrated_mae_not_above_random_walk")
        if float(calibrated_metrics["information_coefficient"]) <= 0.0:
            reasons.append("calibrated_information_coefficient_not_positive")
        if float(calibrated_metrics["direction_accuracy"]) <= 0.50:
            reasons.append("calibrated_direction_accuracy_not_above_half")
        eligible = not reasons
        if eligible:
            eligible_symbols.append(symbol)
        symbol_evidence[symbol] = {
            "scale": scale,
            "eligible": eligible,
            "reasons": reasons,
            "tuning_observations": len(symbol_tuning),
            "selection_observations": len(symbol_selection),
            "raw_selection_metrics": raw_metrics,
            "calibrated_selection_metrics": calibrated_metrics,
        }
    calibrated_tuple = tuple(sorted(calibrated, key=lambda item: (item.decision_ms, item.symbol)))
    bootstrap = day_block_bootstrap(
        calibrated_tuple,
        samples=bootstrap_samples,
        seed=seed,
    )
    return (
        {
            "method": "zero-intercept nonnegative OLS scale, bounded to [0,1]",
            "split": "earlier-half tuning; later-half selection",
            "boundary_utc": _iso_utc(boundary),
            "tuning_observations": len(tuning),
            "selection_observations": len(selection),
            "scales": scales,
            "eligible_symbols": eligible_symbols,
            "raw_selection_metrics": summarize_metrics(selection),
            "calibrated_selection_metrics": summarize_metrics(calibrated_tuple),
            "selection_bootstrap": bootstrap,
        },
        calibrated_tuple,
    )


def day_block_bootstrap(
    observations: Sequence[ForecastObservation],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    blocks: dict[str, list[float]] = {}
    for item in observations:
        day = item.decision_time_utc[:10]
        blocks.setdefault(day, []).append(
            float(item.random_walk_absolute_error - item.absolute_error)
        )
    ordered = tuple(sorted(blocks))
    if not ordered:
        raise RuntimeError("foundation benchmark produced no UTC day blocks")
    generator = random.Random(int(seed))
    estimates: list[float] = []
    for _ in range(int(samples)):
        values: list[float] = []
        for _block in ordered:
            sampled_day = ordered[generator.randrange(len(ordered))]
            values.extend(blocks[sampled_day])
        estimates.append(float(sum(values) / len(values)))
    distribution = np.asarray(estimates, dtype=np.float64)
    return {
        "method": "UTC-day block bootstrap with replacement",
        "samples": int(samples),
        "day_blocks": len(ordered),
        "mean_mae_improvement": float(np.mean(distribution)),
        "ci_95_low": float(np.quantile(distribution, 0.025)),
        "ci_95_high": float(np.quantile(distribution, 0.975)),
        "positive_probability": float(np.mean(distribution > 0.0)),
        "seed": int(seed),
    }


def _write_observations(path: Path, observations: Sequence[ForecastObservation]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            fieldnames = tuple(ForecastObservation.__dataclass_fields__)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for observation in observations:
                writer.writerow(observation.asdict())
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, path)
        return digest
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _cumulative_error_improvement(
    observations: Sequence[ForecastObservation], symbol: str
) -> tuple[tuple[int, float], ...]:
    cumulative = 0.0
    values: list[tuple[int, float]] = []
    for item in sorted(
        (row for row in observations if row.symbol == symbol),
        key=lambda row: row.decision_ms,
    ):
        cumulative += (item.random_walk_absolute_error - item.absolute_error) * 10_000.0
        values.append((item.decision_ms, cumulative))
    return tuple(values)


def write_foundation_benchmark_chart(
    path: str | Path,
    *,
    raw_observations: Sequence[ForecastObservation],
    calibrated_selection: Sequence[ForecastObservation],
    model_label: str,
) -> str:
    """Write a deterministic SVG of paired forecast-error improvement, not P&L."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1_200, 720
    left, right = 92, 35
    panel_top = (118, 408)
    panel_height = 220
    colors = {"BTCUSDT": "#d97706", "ETHUSDT": "#2563eb", "SOLUSDT": "#15803d"}
    panels = (
        (
            "Raw forecast vs random walk",
            tuple(raw_observations),
            panel_top[0],
        ),
        (
            "Causally calibrated selection half (zero scale means abstain)",
            tuple(calibrated_selection),
            panel_top[1],
        ),
    )
    all_timestamps = [item.decision_ms for item in raw_observations]
    if not all_timestamps:
        raise RuntimeError("foundation chart requires observations")
    minimum_time = min(all_timestamps)
    maximum_time = max(all_timestamps)
    time_span = max(1, maximum_time - minimum_time)
    plot_width = width - left - right

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#111827;letter-spacing:0}.axis{font-size:12px}.label{font-size:13px}.panel{font-size:16px;font-weight:600}.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#4b5563}</style>',
        f'<text class="title" x="{left}" y="38">Kronos post-cutoff forecast benchmark</text>',
        f'<text class="sub" x="{left}" y="62">{model_label} | BTCUSDT, ETHUSDT, SOLUSDT | cumulative paired absolute-error improvement in bps | not P&amp;L</text>',
        f'<text class="sub" x="{left}" y="82">Above zero beats a zero-return random-walk forecast; source rows and SHA-256 remain the numerical authority.</text>',
    ]
    for title, panel_observations, top in panels:
        series = {
            symbol: _cumulative_error_improvement(panel_observations, symbol)
            for symbol in FOUNDATION_SYMBOLS
        }
        y_values = [value for values in series.values() for _timestamp, value in values]
        bound = max(1e-9, max((abs(value) for value in y_values), default=0.0)) * 1.10
        bottom = top + panel_height
        zero_y = top + panel_height / 2.0
        lines.extend(
            (
                f'<text class="panel" x="{left}" y="{top - 14}">{title}</text>',
                f'<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="#f9fafb" stroke="#d1d5db"/>',
            )
        )
        for tick in range(5):
            fraction = tick / 4.0
            y = top + fraction * panel_height
            value = bound * (1.0 - 2.0 * fraction)
            lines.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb"/>'
            )
            lines.append(
                f'<text class="axis" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.2f}</text>'
            )
        lines.append(
            f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" y2="{zero_y:.2f}" stroke="#111827" stroke-width="1.2"/>'
        )
        for symbol in FOUNDATION_SYMBOLS:
            points: list[str] = []
            for timestamp, value in series[symbol]:
                x = left + (timestamp - minimum_time) / time_span * plot_width
                y = top + (bound - value) / (2.0 * bound) * panel_height
                points.append(f"{x:.2f},{y:.2f}")
            if points:
                lines.append(
                    f'<polyline fill="none" stroke="{colors[symbol]}" stroke-width="2" points="{" ".join(points)}"/>'
                )
        for fraction in (0.0, 0.5, 1.0):
            timestamp = minimum_time + int(time_span * fraction)
            x = left + plot_width * fraction
            lines.append(
                f'<text class="axis" x="{x:.2f}" y="{bottom + 20}" text-anchor="middle">{_iso_utc(timestamp)[:10]}</text>'
            )
    legend_x = left
    for symbol in FOUNDATION_SYMBOLS:
        lines.append(
            f'<line x1="{legend_x}" y1="692" x2="{legend_x + 24}" y2="692" stroke="{colors[symbol]}" stroke-width="3"/>'
        )
        lines.append(
            f'<text class="label" x="{legend_x + 31}" y="697">{symbol}</text>'
        )
        legend_x += 150
    lines.append('</svg>')
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(payload).hexdigest()
        os.replace(temporary, target)
        return digest
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _batches(values: Sequence[ForecastContext], size: int) -> Iterable[tuple[ForecastContext, ...]]:
    for index in range(0, len(values), size):
        yield tuple(values[index : index + size])


def _engine_identity(report: dict[str, object]) -> tuple[object, ...]:
    source = report.get("source") if isinstance(report.get("source"), dict) else {}
    backend = report.get("backend") if isinstance(report.get("backend"), dict) else {}
    return (
        report.get("provider"),
        report.get("model_size"),
        report.get("model_artifact"),
        report.get("tokenizer_artifact"),
        source.get("commit"),
        source.get("files"),
        backend.get("kind"),
        backend.get("device"),
        report.get("max_context"),
        report.get("model_parameters"),
        report.get("tokenizer_parameters"),
        report.get("rng_seed_control"),
    )


def run_foundation_forecast_benchmark(
    config: FoundationBenchmarkConfig,
    *,
    observations_path: str | Path,
    chart_path: str | Path,
    report_path: str | Path,
    progress: ProgressCallback | None = None,
) -> FoundationBenchmarkReport:
    """Run a bounded, post-pretraining predictive benchmark without placing orders."""

    cfg = config.validated()
    notify = progress or (lambda _message: None)
    decisions = decision_timestamps(cfg)
    notify(f"data-audit decisions_per_symbol={len(decisions)}")
    with _read_only_connection(cfg.database_path) as connection:
        source_evidence = audit_archive_coverage(connection, cfg, decisions)
        notify("data-audit complete source=binance_public_archive checksums=verified")
        contexts: list[ForecastContext] = []
        for symbol in FOUNDATION_SYMBOLS:
            for index, decision in enumerate(decisions, start=1):
                contexts.append(_aggregate_context(connection, symbol, decision, cfg))
                if index == len(decisions) or index % max(1, len(decisions) // 4) == 0:
                    notify(f"contexts symbol={symbol} {index}/{len(decisions)}")

    if cfg.bootstrap_source:
        provision_kronos_source(
            cfg.source_cache_root,
            repair=cfg.repair_source,
        )
    else:
        verify_kronos_source(cfg.source_cache_root)
    notify(f"worker-start model=kronos-{cfg.model_size} backend={cfg.backend}")
    supervisor = FoundationWorkerSupervisor(
        model_size=cfg.model_size,
        backend=cfg.backend,
        source_cache_root=cfg.source_cache_root,
        require_accelerator=cfg.require_accelerator,
        startup_timeout_seconds=max(120.0, cfg.worker_timeout_seconds),
        request_timeout_seconds=cfg.worker_timeout_seconds,
    )
    engine_report = supervisor.start()
    engine_identity = _engine_identity(engine_report)
    worker_pids: list[int] = [int(supervisor.pid or 0)]
    restart_events: list[dict[str, object]] = []
    rotation_events: list[dict[str, object]] = []
    notify(
        f"worker-ready pid={supervisor.pid} kind="
        f"{dict(engine_report['backend'])['kind']}"
    )
    observations: list[ForecastObservation] = []
    durations: list[float] = []
    worker_durations: list[float] = []
    repeatability: dict[str, object] = {
        "checked": False,
        "exact": False,
        "batch": 1,
        "seed": cfg.seed,
        "maximum_absolute_close_difference": None,
    }
    batch_values = tuple(_batches(contexts, int(cfg.batch_size)))
    try:
        for batch_number, batch in enumerate(batch_values, start=1):
            if (
                cfg.worker_rotation_batches > 0
                and batch_number > 1
                and (batch_number - 1) % cfg.worker_rotation_batches == 0
            ):
                previous_pid = int(supervisor.pid or 0)
                supervisor.stop()
                notify(
                    f"worker-rotate before_batch={batch_number}/{len(batch_values)} "
                    f"previous_pid={previous_pid}"
                )
                replacement_report = supervisor.start()
                if _engine_identity(replacement_report) != engine_identity:
                    supervisor.stop()
                    raise RuntimeError("foundation worker identity changed after planned rotation")
                worker_pids.append(int(supervisor.pid or 0))
                rotation_events.append(
                    {
                        "before_batch": batch_number,
                        "previous_pid": previous_pid,
                        "replacement_pid": worker_pids[-1],
                    }
                )
            batch_restart_attempts = 0
            while True:
                started = time.perf_counter()
                try:
                    worker_prediction = supervisor.predict(
                        batch,
                        prediction_length=cfg.prediction_bars,
                        temperature=cfg.temperature,
                        top_k=cfg.top_k,
                        top_p=cfg.top_p,
                        sample_count=cfg.inference_samples,
                        seed=cfg.seed + batch_number - 1,
                    )
                    if batch_number == 1 and not bool(repeatability["checked"]):
                        repeated = supervisor.predict(
                            batch,
                            prediction_length=cfg.prediction_bars,
                            temperature=cfg.temperature,
                            top_k=cfg.top_k,
                            top_p=cfg.top_p,
                            sample_count=cfg.inference_samples,
                            seed=cfg.seed + batch_number - 1,
                        )
                        first_values = np.asarray(
                            worker_prediction.predicted_closes, dtype=np.float64
                        )
                        repeated_values = np.asarray(
                            repeated.predicted_closes, dtype=np.float64
                        )
                        maximum_difference = float(
                            np.max(np.abs(first_values - repeated_values))
                        )
                        exact = bool(np.array_equal(first_values, repeated_values))
                        repeatability.update(
                            {
                                "checked": True,
                                "exact": exact,
                                "maximum_absolute_close_difference": maximum_difference,
                            }
                        )
                        worker_durations.append(repeated.worker_seconds)
                        if not exact:
                            raise FoundationWorkerError(
                                "seeded foundation forecast failed exact repeatability gate",
                                restartable=False,
                            )
                    break
                except FoundationWorkerError as exc:
                    supervisor.stop()
                    if (
                        not exc.restartable
                        or batch_restart_attempts >= 1
                        or len(restart_events) >= cfg.max_worker_restarts
                    ):
                        raise RuntimeError(
                            f"foundation worker batch {batch_number}/{len(batch_values)} "
                            f"failed without recoverable restart capacity: {exc}"
                        ) from exc
                    batch_restart_attempts += 1
                    event = {
                        "batch": batch_number,
                        "attempt": batch_restart_attempts,
                        "previous_pid": worker_pids[-1],
                        "error": str(exc),
                    }
                    restart_events.append(event)
                    notify(
                        f"worker-restart batch={batch_number}/{len(batch_values)} "
                        f"attempt={batch_restart_attempts}"
                    )
                    replacement_report = supervisor.start()
                    if _engine_identity(replacement_report) != engine_identity:
                        supervisor.stop()
                        raise RuntimeError("foundation worker identity changed after restart")
                    worker_pids.append(int(supervisor.pid or 0))
                    event["replacement_pid"] = worker_pids[-1]
            duration = time.perf_counter() - started
            durations.append(duration)
            worker_durations.append(worker_prediction.worker_seconds)
            for context, close_path in zip(
                batch, worker_prediction.predicted_closes, strict=True
            ):
                predicted_closes = np.asarray(close_path, dtype=np.float64)
                actual_closes = np.asarray(context.actual_closes, dtype=np.float64)
                predicted_average_return = float(
                    np.mean(predicted_closes) / context.last_close - 1.0
                )
                actual_average_return = float(
                    np.mean(actual_closes) / context.last_close - 1.0
                )
                predicted_final_return = float(
                    predicted_closes[-1] / context.last_close - 1.0
                )
                actual_final_return = float(actual_closes[-1] / context.last_close - 1.0)
                observations.append(
                    ForecastObservation(
                        symbol=context.symbol,
                        decision_ms=context.decision_ms,
                        decision_time_utc=_iso_utc(context.decision_ms),
                        last_close=context.last_close,
                        predicted_average_return=predicted_average_return,
                        actual_average_return=actual_average_return,
                        predicted_final_return=predicted_final_return,
                        actual_final_return=actual_final_return,
                        absolute_error=abs(predicted_average_return - actual_average_return),
                        random_walk_absolute_error=abs(actual_average_return),
                        direction_correct=(
                            predicted_average_return != 0.0
                            and actual_average_return != 0.0
                            and math.copysign(1.0, predicted_average_return)
                            == math.copysign(1.0, actual_average_return)
                        ),
                        inference_batch=batch_number,
                    )
                )
            notify(
                f"inference batch={batch_number}/{len(batch_values)} "
                f"rows={len(batch)} seconds={duration:.3f} pid={worker_prediction.worker_pid}"
            )
    finally:
        supervisor.stop()

    metrics = summarize_metrics(observations)
    raw_bootstrap = day_block_bootstrap(
        observations,
        samples=cfg.bootstrap_samples,
        seed=cfg.seed,
    )
    calibration, calibrated_selection = calibrate_forecast_amplitude(
        observations,
        bootstrap_samples=cfg.bootstrap_samples,
        seed=cfg.seed,
    )
    bootstrap = dict(calibration["selection_bootstrap"])
    reasons: list[str] = []
    if not calibration["eligible_symbols"]:
        reasons.append("no_symbol_passed_causal_amplitude_calibration")
    if int(bootstrap["day_blocks"]) < 30:
        reasons.append("fewer_than_30_utc_day_blocks")
    if float(bootstrap["ci_95_low"]) <= 0.0:
        reasons.append("calibrated_mae_uplift_day_block_ci_not_strictly_positive")
    status = "predictive_candidate" if not reasons else "rejected"
    observation_target = Path(observations_path)
    observations_sha256 = _write_observations(observation_target, observations)
    chart_target = Path(chart_path)
    chart_sha256 = write_foundation_benchmark_chart(
        chart_target,
        raw_observations=observations,
        calibrated_selection=calibrated_selection,
        model_label=(
            f"Kronos-{cfg.model_size}; {cfg.lookback_bars}x5m lookback; "
            f"{cfg.prediction_bars}x5m horizon"
        ),
    )
    inference = {
        "batches": len(durations),
        "total_seconds": float(sum(durations)),
        "average_batch_seconds": float(np.mean(durations)),
        "minimum_batch_seconds": float(np.min(durations)),
        "maximum_batch_seconds": float(np.max(durations)),
        "worker_seconds_total": float(sum(worker_durations)),
        "worker_process_starts": len(worker_pids),
        "worker_pids": worker_pids,
        "worker_restart_count": len(restart_events),
        "worker_restarts": restart_events,
        "planned_worker_rotation_count": len(rotation_events),
        "planned_worker_rotations": rotation_events,
        "seeded_repeatability": repeatability,
        "in_process_retries": 0,
    }
    report = FoundationBenchmarkReport(
        version=FOUNDATION_BENCHMARK_VERSION,
        generated_at_ms=int(time.time() * 1_000),
        status=status,
        trading_authority=False,
        reasons=tuple(reasons),
        config=cfg.asdict(),
        evaluation={
            "pretraining_cutoff": KRONOS_PRETRAINING_CUTOFF,
            "selection_start": _iso_utc(cfg.start_ms),
            "selection_end_exclusive": _iso_utc(cfg.end_exclusive_ms),
            "terminal_period": "2026-01-01T00:00:00Z onward; not accessed",
            "sampling_mode": "deterministic_evenly_spaced_candidate_diagnostic",
            "target": "mean future close return over prediction_bars",
            "random_walk_baseline": "zero expected return from the last observed close",
            "after_cost_trading_evidence": False,
            "orders_allowed": False,
        },
        source_evidence=tuple(item.asdict() for item in source_evidence),
        engine=engine_report,
        metrics=metrics,
        calibration={**calibration, "raw_full_bootstrap": raw_bootstrap},
        bootstrap=bootstrap,
        inference=inference,
        observation_count=len(observations),
        observations_sha256=observations_sha256,
        observations_path=str(observation_target),
        chart_sha256=chart_sha256,
        chart_path=str(chart_target),
    )
    write_json_atomic(Path(report_path), report.asdict(), sort_keys=True)
    notify(f"report status={status} observations={len(observations)}")
    return report


__all__ = [
    "ArchiveCoverageEvidence",
    "FOUNDATION_BENCHMARK_VERSION",
    "FOUNDATION_SELECTION_END_EXCLUSIVE_MS",
    "FOUNDATION_SELECTION_START_MS",
    "FOUNDATION_SYMBOLS",
    "ForecastObservation",
    "FoundationBenchmarkConfig",
    "FoundationBenchmarkReport",
    "audit_archive_coverage",
    "calibrate_forecast_amplitude",
    "day_block_bootstrap",
    "decision_timestamps",
    "parse_utc_ms",
    "run_foundation_forecast_benchmark",
    "summarize_metrics",
    "write_foundation_benchmark_chart",
]
