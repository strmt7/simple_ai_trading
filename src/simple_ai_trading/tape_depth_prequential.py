"""Timestamp-defined rolling evaluation for the tape/depth gross forecaster."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Callable, Sequence

import numpy as np

from .assets import is_supported_major_symbol, normalize_symbol
from .microstructure_warehouse import MicrostructureWarehouse
from .storage import write_json_atomic
from .tape_depth_cache import (
    TAPE_DEPTH_CACHE_SCHEMA_VERSION,
    load_tape_depth_dataset_cache,
    save_tape_depth_dataset_cache,
    tape_depth_dataset_cache_key,
)
from .tape_depth_features import (
    TapeDepthForecastDataset,
    build_tape_depth_forecast_dataset,
    slice_tape_depth_forecast_dataset,
    tape_depth_dataset_source_evidence,
)
from .tape_depth_model import (
    TapeDepthModelArtifact,
    TapeDepthPredictionBatch,
    load_tape_depth_model_artifact,
    save_tape_depth_model_artifact,
    score_tape_depth_evaluation,
    train_tape_depth_forecaster,
)


TAPE_DEPTH_PREQUENTIAL_SCHEMA_VERSION = "tape-depth-prequential-plan-v2"
TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION = "tape-depth-prequential-evidence-v2"
_DAY_MS = 86_400_000


@dataclass(frozen=True)
class TapeDepthFoldPlan:
    symbol: str
    fold_index: int
    dataset_start_ms: int
    tuning_start_ms: int
    calibration_start_ms: int
    evaluation_start_ms: int
    evaluation_end_ms: int
    estimated_dataset_rows: int
    estimated_evaluation_rows: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TapeDepthSymbolPlan:
    schema_version: str
    symbol: str
    source_first_second_ms: int
    source_last_second_ms: int
    first_eligible_decision_ms: int
    last_eligible_decision_ms: int
    training_window_days: int
    tuning_window_days: int
    calibration_window_days: int
    evaluation_window_days: int
    horizon_seconds: int
    total_latency_ms: int
    decision_cadence_seconds: int
    maximum_rows: int
    fold_start: int
    max_folds: int
    available_fold_count: int
    folds: tuple[TapeDepthFoldPlan, ...]
    coverage_fingerprint: str
    plan_fingerprint: str

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["folds"] = [fold.asdict() for fold in self.folds]
        return payload


@dataclass(frozen=True)
class TapeDepthFoldEvaluation:
    plan: TapeDepthFoldPlan
    artifact: TapeDepthModelArtifact
    predictions: TapeDepthPredictionBatch


def _align_up(value: int, quantum: int) -> int:
    return ((int(value) + int(quantum) - 1) // int(quantum)) * int(quantum)


def _align_down(value: int, quantum: int) -> int:
    return (int(value) // int(quantum)) * int(quantum)


def _plan_fingerprint(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def plan_tape_depth_folds(
    *,
    symbol: str,
    source_first_second_ms: int,
    source_last_second_ms: int,
    training_window_days: int = 730,
    tuning_window_days: int = 30,
    calibration_window_days: int = 30,
    evaluation_window_days: int = 90,
    horizon_seconds: int = 60,
    total_latency_ms: int = 750,
    decision_cadence_seconds: int = 20,
    maximum_rows: int = 5_000_000,
    fold_start: int = 0,
    max_folds: int = 0,
) -> TapeDepthSymbolPlan:
    """Plan non-overlapping calendar evaluation folds over exact source coverage."""

    normalized = normalize_symbol(symbol)
    if not is_supported_major_symbol(normalized):
        raise ValueError(f"unsupported tape/depth symbol: {normalized}")
    training_days = int(training_window_days)
    tuning_days = int(tuning_window_days)
    calibration_days = int(calibration_window_days)
    evaluation_days = int(evaluation_window_days)
    horizon = int(horizon_seconds)
    latency = int(total_latency_ms)
    cadence = int(decision_cadence_seconds)
    row_limit = int(maximum_rows)
    first_fold = int(fold_start)
    fold_limit = int(max_folds)
    if not 365 <= training_days <= 3_650:
        raise ValueError("training_window_days must lie in [365, 3650]")
    if any(not 7 <= value <= 180 for value in (tuning_days, calibration_days, evaluation_days)):
        raise ValueError("tuning, calibration, and evaluation days must lie in [7, 180]")
    if not 1 <= horizon <= 3_600:
        raise ValueError("horizon_seconds must lie in [1, 3600]")
    if not 0 <= latency <= 60_000:
        raise ValueError("total_latency_ms must lie in [0, 60000]")
    if not 1 <= cadence <= 60 or 60 % cadence != 0:
        raise ValueError("decision_cadence_seconds must divide 60 and lie in [1, 60]")
    if row_limit < 1 or first_fold < 0 or fold_limit < 0:
        raise ValueError(
            "maximum_rows must be positive; fold_start and max_folds must be non-negative"
        )
    first_second = int(source_first_second_ms)
    last_second = int(source_last_second_ms)
    if first_second >= last_second:
        raise ValueError("trade source coverage is empty")
    cadence_ms = cadence * 1_000
    entry_delay_seconds = max(1, int(math.ceil(latency / 1_000.0)))
    target_offset_seconds = entry_delay_seconds + horizon
    first_eligible = _align_up(first_second + 901_000, cadence_ms)
    last_eligible = _align_down(
        last_second - target_offset_seconds * 1_000 + 1_000,
        cadence_ms,
    )
    first_dataset_start = _align_up(first_eligible, _DAY_MS)
    pre_evaluation_days = training_days + tuning_days + calibration_days
    evaluation_start = first_dataset_start + pre_evaluation_days * _DAY_MS
    available_folds: list[TapeDepthFoldPlan] = []
    fold_index = 0
    while True:
        evaluation_end = evaluation_start + evaluation_days * _DAY_MS - cadence_ms
        if evaluation_end > last_eligible:
            break
        dataset_start = evaluation_start - pre_evaluation_days * _DAY_MS
        tuning_start = dataset_start + training_days * _DAY_MS
        calibration_start = tuning_start + tuning_days * _DAY_MS
        estimated_rows = (evaluation_end - dataset_start) // cadence_ms + 1
        estimated_evaluation_rows = (
            evaluation_end - evaluation_start
        ) // cadence_ms + 1
        if estimated_rows > row_limit:
            raise ValueError(
                f"planned fold may emit {estimated_rows} rows; maximum_rows={row_limit}"
            )
        available_folds.append(
            TapeDepthFoldPlan(
                symbol=normalized,
                fold_index=fold_index,
                dataset_start_ms=dataset_start,
                tuning_start_ms=tuning_start,
                calibration_start_ms=calibration_start,
                evaluation_start_ms=evaluation_start,
                evaluation_end_ms=evaluation_end,
                estimated_dataset_rows=estimated_rows,
                estimated_evaluation_rows=estimated_evaluation_rows,
            )
        )
        fold_index += 1
        evaluation_start += evaluation_days * _DAY_MS
    coverage_payload: dict[str, object] = {
        "schema_version": TAPE_DEPTH_PREQUENTIAL_SCHEMA_VERSION,
        "symbol": normalized,
        "source_first_second_ms": first_second,
        "source_last_second_ms": last_second,
        "first_eligible_decision_ms": first_eligible,
        "last_eligible_decision_ms": last_eligible,
        "training_window_days": training_days,
        "tuning_window_days": tuning_days,
        "calibration_window_days": calibration_days,
        "evaluation_window_days": evaluation_days,
        "horizon_seconds": horizon,
        "total_latency_ms": latency,
        "decision_cadence_seconds": cadence,
        "maximum_rows": row_limit,
        "folds": [fold.asdict() for fold in available_folds],
    }
    coverage_fingerprint = _plan_fingerprint(coverage_payload)
    selected_end = None if fold_limit == 0 else first_fold + fold_limit
    folds = available_folds[first_fold:selected_end]
    payload: dict[str, object] = {
        **coverage_payload,
        "fold_start": first_fold,
        "max_folds": fold_limit,
        "available_fold_count": len(available_folds),
        "coverage_fingerprint": coverage_fingerprint,
        "folds": [fold.asdict() for fold in folds],
    }
    return TapeDepthSymbolPlan(
        schema_version=TAPE_DEPTH_PREQUENTIAL_SCHEMA_VERSION,
        symbol=normalized,
        source_first_second_ms=first_second,
        source_last_second_ms=last_second,
        first_eligible_decision_ms=first_eligible,
        last_eligible_decision_ms=last_eligible,
        training_window_days=training_days,
        tuning_window_days=tuning_days,
        calibration_window_days=calibration_days,
        evaluation_window_days=evaluation_days,
        horizon_seconds=horizon,
        total_latency_ms=latency,
        decision_cadence_seconds=cadence,
        maximum_rows=row_limit,
        fold_start=first_fold,
        max_folds=fold_limit,
        available_fold_count=len(available_folds),
        folds=tuple(folds),
        coverage_fingerprint=coverage_fingerprint,
        plan_fingerprint=_plan_fingerprint(payload),
    )


def plan_tape_depth_warehouse(
    warehouse: MicrostructureWarehouse,
    *,
    symbols: Sequence[str],
    **options: object,
) -> tuple[TapeDepthSymbolPlan, ...]:
    plans: list[TapeDepthSymbolPlan] = []
    for raw_symbol in symbols:
        symbol = normalize_symbol(raw_symbol)
        row = warehouse.connect().execute(
            "SELECT min(second_ms), max(second_ms) FROM current_trade_1s WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            raise ValueError(f"no current one-second trade rows exist for {symbol}")
        plan = plan_tape_depth_folds(
            symbol=symbol,
            source_first_second_ms=int(row[0]),
            source_last_second_ms=int(row[1]),
            **options,
        )
        if not plan.folds:
            raise ValueError(f"source coverage is too short for one {symbol} fold")
        plans.append(plan)
    return tuple(plans)


def _dataset_range_source_evidence(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    horizon_seconds: int,
    total_latency_ms: int,
) -> dict[str, object]:
    entry_delay_seconds = max(
        1,
        int(math.ceil(int(total_latency_ms) / 1_000.0)),
    )
    target_offset_seconds = entry_delay_seconds + int(horizon_seconds)
    source_start_ms = int(start_ms) - 901_000
    prior = warehouse.connect().execute(
        "SELECT max(second_ms) FROM current_trade_1s "
        "WHERE symbol = ? AND second_ms <= ?",
        [symbol, source_start_ms],
    ).fetchone()
    if prior is None or prior[0] is None:
        raise ValueError("tape/depth interval has no prior verified trade reference")
    return tape_depth_dataset_source_evidence(
        warehouse,
        symbol,
        required_start_ms=min(source_start_ms, int(prior[0])),
        required_end_ms=(
            int(end_ms) - 1_000 + target_offset_seconds * 1_000
        ),
        peer_feature_start_ms=int(start_ms) - 1_000,
        peer_feature_end_ms=int(end_ms) - 1_000,
    )


def evaluate_tape_depth_fold(
    warehouse: MicrostructureWarehouse,
    *,
    plan: TapeDepthFoldPlan,
    horizon_seconds: int,
    total_latency_ms: int,
    decision_cadence_seconds: int,
    maximum_depth_age_ms: int,
    maximum_rows: int,
    risk_level: str,
    model_profile: str,
    feature_set: str,
    compute_backend: str,
    minimum_segment_rows: int,
    prefetched_dataset: TapeDepthForecastDataset | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> TapeDepthFoldEvaluation:
    if prefetched_dataset is None:
        dataset = build_tape_depth_forecast_dataset(
            warehouse,
            symbol=plan.symbol,
            start_ms=plan.dataset_start_ms,
            end_ms=plan.evaluation_end_ms,
            horizon_seconds=horizon_seconds,
            total_latency_ms=total_latency_ms,
            decision_cadence_seconds=decision_cadence_seconds,
            maximum_depth_age_ms=maximum_depth_age_ms,
            maximum_rows=maximum_rows,
        )
    else:
        if (
            prefetched_dataset.symbol != plan.symbol
            or prefetched_dataset.horizon_seconds != int(horizon_seconds)
            or prefetched_dataset.total_latency_ms != int(total_latency_ms)
            or prefetched_dataset.decision_cadence_seconds
            != int(decision_cadence_seconds)
            or prefetched_dataset.maximum_depth_age_ms != int(maximum_depth_age_ms)
        ):
            raise ValueError("prefetched tape/depth dataset contract differs from fold")
        source_evidence = _dataset_range_source_evidence(
            warehouse,
            symbol=plan.symbol,
            start_ms=plan.dataset_start_ms,
            end_ms=plan.evaluation_end_ms,
            horizon_seconds=horizon_seconds,
            total_latency_ms=total_latency_ms,
        )
        dataset = slice_tape_depth_forecast_dataset(
            prefetched_dataset,
            start_ms=plan.dataset_start_ms,
            end_ms=plan.evaluation_end_ms,
            source_evidence=source_evidence,
        )
    if dataset.rows != plan.estimated_dataset_rows:
        raise ValueError(
            f"{plan.symbol} fold {plan.fold_index} has {dataset.rows} complete rows; "
            f"expected {plan.estimated_dataset_rows}"
        )
    artifact = train_tape_depth_forecaster(
        dataset,
        risk_level=risk_level,
        model_profile=model_profile,
        feature_set=feature_set,
        compute_backend=compute_backend,
        minimum_segment_rows=minimum_segment_rows,
        split_boundaries_ms=(
            plan.tuning_start_ms,
            plan.calibration_start_ms,
            plan.evaluation_start_ms,
        ),
        progress=progress,
    )
    predictions = score_tape_depth_evaluation(artifact, dataset)
    if (
        predictions.rows != plan.estimated_evaluation_rows
        or int(predictions.decision_time_ms[0]) != plan.evaluation_start_ms
        or int(predictions.decision_time_ms[-1]) != plan.evaluation_end_ms
    ):
        raise ValueError("tape/depth fold output does not match its timestamp plan")
    return TapeDepthFoldEvaluation(
        plan=plan,
        artifact=artifact,
        predictions=predictions,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def write_tape_depth_predictions(
    batch: TapeDepthPredictionBatch,
    path: str | Path,
) -> str:
    """Atomically write every fold prediction as deterministic gzip CSV."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    arrays = (
        batch.decision_time_ms,
        batch.target_entry_time_ms,
        batch.target_exit_time_ms,
        batch.actual_gross_return_bps,
        batch.direction_probability,
        batch.mean_prediction_bps,
        batch.lower_prediction_bps,
        batch.upper_prediction_bps,
    )
    if batch.rows <= 0 or any(len(values) != batch.rows for values in arrays):
        raise ValueError("tape/depth prediction arrays have inconsistent lengths")
    if (
        np.any(np.diff(batch.decision_time_ms) <= 0)
        or np.any(batch.target_entry_time_ms <= batch.decision_time_ms)
        or np.any(batch.target_exit_time_ms <= batch.target_entry_time_ms)
        or not all(np.all(np.isfinite(values)) for values in arrays[3:])
        or np.any(
            (batch.direction_probability < 0.0)
            | (batch.direction_probability > 1.0)
        )
    ):
        raise ValueError("tape/depth prediction batch failed its numeric contract")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=raw,
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(
                    compressed,
                    encoding="ascii",
                    newline="",
                ) as text:
                    writer = csv.writer(text, lineterminator="\n")
                    writer.writerow(
                        (
                            "decision_time_ms",
                            "target_entry_time_ms",
                            "target_exit_time_ms",
                            "actual_gross_return_bps",
                            "direction_probability",
                            "mean_prediction_bps",
                            "lower_prediction_bps",
                            "upper_prediction_bps",
                        )
                    )
                    for row in zip(*arrays, strict=True):
                        writer.writerow(
                            (
                                int(row[0]),
                                int(row[1]),
                                int(row[2]),
                                format(float(row[3]), ".17g"),
                                format(float(row[4]), ".17g"),
                                format(float(row[5]), ".17g"),
                                format(float(row[6]), ".17g"),
                                format(float(row[7]), ".17g"),
                            )
                        )
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return _sha256_file(destination)


def read_tape_depth_predictions(path: str | Path) -> TapeDepthPredictionBatch:
    """Load and validate a deterministic fold prediction table for resume."""

    expected_fields = (
        "decision_time_ms",
        "target_entry_time_ms",
        "target_exit_time_ms",
        "actual_gross_return_bps",
        "direction_probability",
        "mean_prediction_bps",
        "lower_prediction_bps",
        "upper_prediction_bps",
    )
    columns: dict[str, list[int] | list[float]] = {
        name: [] for name in expected_fields
    }
    try:
        with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != expected_fields:
                raise ValueError("tape/depth prediction header drifted")
            for row in reader:
                for name in expected_fields[:3]:
                    columns[name].append(int(row[name]))
                for name in expected_fields[3:]:
                    columns[name].append(float(row[name]))
    except (
        csv.Error,
        EOFError,
        KeyError,
        OSError,
        OverflowError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise ValueError("tape/depth prediction table is invalid") from exc
    rows = len(columns[expected_fields[0]])
    if rows <= 0 or any(len(columns[name]) != rows for name in expected_fields):
        raise ValueError("tape/depth prediction table has inconsistent rows")
    batch = TapeDepthPredictionBatch(
        decision_time_ms=np.asarray(columns["decision_time_ms"], dtype=np.int64),
        target_entry_time_ms=np.asarray(
            columns["target_entry_time_ms"], dtype=np.int64
        ),
        target_exit_time_ms=np.asarray(columns["target_exit_time_ms"], dtype=np.int64),
        actual_gross_return_bps=np.asarray(
            columns["actual_gross_return_bps"], dtype=np.float64
        ),
        direction_probability=np.asarray(
            columns["direction_probability"], dtype=np.float64
        ),
        mean_prediction_bps=np.asarray(
            columns["mean_prediction_bps"], dtype=np.float64
        ),
        lower_prediction_bps=np.asarray(
            columns["lower_prediction_bps"], dtype=np.float64
        ),
        upper_prediction_bps=np.asarray(
            columns["upper_prediction_bps"], dtype=np.float64
        ),
    )
    numeric_arrays = (
        batch.actual_gross_return_bps,
        batch.direction_probability,
        batch.mean_prediction_bps,
        batch.lower_prediction_bps,
        batch.upper_prediction_bps,
    )
    if (
        np.any(np.diff(batch.decision_time_ms) <= 0)
        or np.any(batch.target_entry_time_ms <= batch.decision_time_ms)
        or np.any(batch.target_exit_time_ms <= batch.target_entry_time_ms)
        or not all(np.all(np.isfinite(values)) for values in numeric_arrays)
        or np.any((batch.direction_probability < 0.0) | (batch.direction_probability > 1.0))
    ):
        raise ValueError("tape/depth prediction table failed its numeric contract")
    return batch


def _fold_summary(
    evaluation: TapeDepthFoldEvaluation,
    *,
    artifact_path: Path,
    artifact_sha256: str,
    predictions_path: Path,
    predictions_sha256: str,
) -> dict[str, object]:
    artifact = evaluation.artifact
    return {
        **evaluation.plan.asdict(),
        "status": artifact.status,
        "rejection_reasons": list(artifact.rejection_reasons),
        "trading_authority": False,
        "execution_claim": False,
        "backend_kind": artifact.backend_kind,
        "backend_device": artifact.backend_device,
        "risk_level": artifact.risk_level,
        "model_profile": artifact.model_profile,
        "feature_set": artifact.feature_set,
        "dataset_fingerprint": artifact.dataset_fingerprint,
        "source_manifest_fingerprint": artifact.dataset_summary["source_evidence"][
            "manifest_fingerprint"
        ],
        "prediction_fingerprint": evaluation.predictions.fingerprint(),
        "artifact_path": artifact_path.as_posix(),
        "artifact_sha256": artifact_sha256,
        "predictions_path": predictions_path.as_posix(),
        "predictions_sha256": predictions_sha256,
        "metrics": asdict(artifact.evaluation_metrics),
    }


def _aggregate_fold_metrics(folds: Sequence[dict[str, object]]) -> dict[str, object]:
    if not folds:
        return {"folds": 0, "rows": 0}
    metrics = [dict(fold["metrics"]) for fold in folds]
    row_counts = np.asarray([int(item["rows"]) for item in metrics], dtype=np.float64)
    total_rows = int(np.sum(row_counts))
    weighted_names = (
        "direction_auc",
        "direction_brier",
        "prevalence_brier",
        "mean_absolute_error_bps",
        "zero_baseline_mae_bps",
        "spearman_information_coefficient",
        "interval_80_coverage",
        "interval_crossing_rate",
        "top_decile_mean_signed_gross_bps",
        "top_decile_positive_rate",
    )
    output: dict[str, object] = {
        "folds": len(folds),
        "rows": total_rows,
        "research_candidate_folds": sum(
            1 for fold in folds if fold["status"] == "research_candidate"
        ),
    }
    for name in weighted_names:
        values = np.asarray([float(item[name]) for item in metrics], dtype=np.float64)
        output[f"weighted_{name}"] = float(np.average(values, weights=row_counts))
        output[f"minimum_{name}"] = float(np.min(values))
        output[f"maximum_{name}"] = float(np.max(values))
    return output


def _verified_relative_evidence_path(
    root: Path,
    payload: dict[str, object],
    *,
    path_name: str,
    hash_name: str,
) -> tuple[Path, Path]:
    relative = Path(str(payload.get(path_name) or ""))
    resolved = (root / relative).resolve()
    if (
        not relative.parts
        or relative.is_absolute()
        or root not in resolved.parents
        or not resolved.is_file()
        or not _is_sha256(payload.get(hash_name))
        or _sha256_file(resolved) != payload.get(hash_name)
    ):
        raise ValueError(f"tape/depth report has invalid {path_name} evidence")
    return relative, resolved


def verify_tape_depth_prequential_report(
    report_path: str | Path,
    report: dict[str, object],
) -> None:
    """Recompute every fold binding from serialized row-level evidence."""

    if report.get("schema_version") != TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION:
        raise ValueError("tape/depth report schema is unsupported")
    if (
        report.get("trading_authority") is not False
        or report.get("execution_claim") is not False
        or report.get("profitability_claim") is not False
    ):
        raise ValueError("tape/depth report carries forbidden authority")
    root = Path(report_path).resolve().parent
    plan_path = root / "plan.json"
    try:
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("tape/depth report plan evidence is unreadable") from exc
    folds = report.get("folds")
    config = report.get("config")
    plans = plan_payload.get("plans") if isinstance(plan_payload, dict) else None
    if (
        not isinstance(folds, list)
        or not folds
        or not all(isinstance(fold, dict) for fold in folds)
        or not isinstance(config, dict)
        or not isinstance(plans, list)
        or plan_payload.get("schema_version") != TAPE_DEPTH_PREQUENTIAL_SCHEMA_VERSION
        or plan_payload.get("config") != config
        or plan_payload.get("total_folds") != len(folds)
        or plan_payload.get("coverage_fingerprints")
        != report.get("coverage_fingerprints")
        or plan_payload.get("available_fold_counts")
        != report.get("available_fold_counts")
    ):
        raise ValueError("tape/depth report plan evidence differs")
    flat_plan_folds: list[dict[str, object]] = []
    plan_fingerprints: dict[str, object] = {}
    for raw_plan in plans:
        if not isinstance(raw_plan, dict) or not isinstance(raw_plan.get("folds"), list):
            raise ValueError("tape/depth report plan evidence differs")
        symbol = str(raw_plan.get("symbol") or "")
        plan_fingerprints[symbol] = raw_plan.get("plan_fingerprint")
        flat_plan_folds.extend(dict(fold) for fold in raw_plan["folds"])
    if (
        len(flat_plan_folds) != len(folds)
        or plan_fingerprints != report.get("plan_fingerprints")
        or int(report.get("total_folds", -1)) != len(folds)
        or int(report.get("completed_folds", -1)) != len(folds)
    ):
        raise ValueError("tape/depth report fold plan is incomplete")

    verified_summaries: list[dict[str, object]] = []
    for index, (raw_summary, raw_plan) in enumerate(
        zip(folds, flat_plan_folds, strict=True)
    ):
        summary = dict(raw_summary)
        plan_keys = (
            "symbol",
            "fold_index",
            "dataset_start_ms",
            "tuning_start_ms",
            "calibration_start_ms",
            "evaluation_start_ms",
            "evaluation_end_ms",
            "estimated_dataset_rows",
            "estimated_evaluation_rows",
        )
        if any(summary.get(name) != raw_plan.get(name) for name in plan_keys):
            raise ValueError(f"tape/depth report fold {index} differs from its plan")
        try:
            fold_plan = TapeDepthFoldPlan(
                symbol=str(raw_plan["symbol"]),
                fold_index=int(raw_plan["fold_index"]),
                dataset_start_ms=int(raw_plan["dataset_start_ms"]),
                tuning_start_ms=int(raw_plan["tuning_start_ms"]),
                calibration_start_ms=int(raw_plan["calibration_start_ms"]),
                evaluation_start_ms=int(raw_plan["evaluation_start_ms"]),
                evaluation_end_ms=int(raw_plan["evaluation_end_ms"]),
                estimated_dataset_rows=int(raw_plan["estimated_dataset_rows"]),
                estimated_evaluation_rows=int(raw_plan["estimated_evaluation_rows"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"tape/depth report fold {index} plan is invalid") from exc
        artifact_relative, artifact_path = _verified_relative_evidence_path(
            root,
            summary,
            path_name="artifact_path",
            hash_name="artifact_sha256",
        )
        predictions_relative, predictions_path = _verified_relative_evidence_path(
            root,
            summary,
            path_name="predictions_path",
            hash_name="predictions_sha256",
        )
        artifact = load_tape_depth_model_artifact(artifact_path)
        predictions = read_tape_depth_predictions(predictions_path)
        if (
            predictions.rows != fold_plan.estimated_evaluation_rows
            or int(predictions.decision_time_ms[0]) != fold_plan.evaluation_start_ms
            or int(predictions.decision_time_ms[-1]) != fold_plan.evaluation_end_ms
            or predictions.metrics() != artifact.evaluation_metrics
        ):
            raise ValueError(f"tape/depth report fold {index} prediction replay differs")
        expected = _fold_summary(
            TapeDepthFoldEvaluation(
                plan=fold_plan,
                artifact=artifact,
                predictions=predictions,
            ),
            artifact_path=artifact_relative,
            artifact_sha256=str(summary["artifact_sha256"]),
            predictions_path=predictions_relative,
            predictions_sha256=str(summary["predictions_sha256"]),
        )
        if summary != expected:
            raise ValueError(f"tape/depth report fold {index} binding differs")
        if (
            summary.get("risk_level") != config.get("risk_level")
            or summary.get("model_profile") != config.get("model_profile")
            or summary.get("feature_set") != config.get("feature_set")
        ):
            raise ValueError(f"tape/depth report fold {index} config differs")
        verified_summaries.append(summary)

    for path_name, hash_name in (
        ("fold_metrics_path", "fold_metrics_sha256"),
        ("forecast_diagnostics_path", "forecast_diagnostics_sha256"),
    ):
        _verified_relative_evidence_path(
            root,
            report,
            path_name=path_name,
            hash_name=hash_name,
        )
    expected_status = (
        "research_candidate"
        if all(fold["status"] == "research_candidate" for fold in verified_summaries)
        else "rejected"
    )
    if (
        report.get("status") != expected_status
        or report.get("aggregate_forecast_metrics")
        != _aggregate_fold_metrics(verified_summaries)
    ):
        raise ValueError("tape/depth report aggregate replay differs")


def _write_fold_metrics_csv(folds: Sequence[dict[str, object]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    metric_names = tuple(dict(folds[0]["metrics"]).keys()) if folds else ()
    fieldnames = (
        "symbol",
        "fold_index",
        "evaluation_start_ms",
        "evaluation_end_ms",
        "status",
        "backend_kind",
        "backend_device",
        *metric_names,
        "dataset_fingerprint",
        "prediction_fingerprint",
        "artifact_sha256",
        "predictions_sha256",
    )
    try:
        with temporary.open("w", encoding="ascii", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for fold in folds:
                metrics = dict(fold["metrics"])
                writer.writerow(
                    {
                        "symbol": fold["symbol"],
                        "fold_index": fold["fold_index"],
                        "evaluation_start_ms": fold["evaluation_start_ms"],
                        "evaluation_end_ms": fold["evaluation_end_ms"],
                        "status": fold["status"],
                        "backend_kind": fold["backend_kind"],
                        "backend_device": fold["backend_device"],
                        **metrics,
                        "dataset_fingerprint": fold["dataset_fingerprint"],
                        "prediction_fingerprint": fold["prediction_fingerprint"],
                        "artifact_sha256": fold["artifact_sha256"],
                        "predictions_sha256": fold["predictions_sha256"],
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return _sha256_file(path)


def render_tape_depth_prequential_svg(
    folds: Sequence[dict[str, object]],
    path: str | Path,
) -> str:
    """Render truthful time-based forecast diagnostics from the fold table."""

    if not folds:
        raise ValueError("cannot chart an empty tape/depth fold table")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 760
    left, right = 94.0, 42.0
    panel_top = (120.0, 330.0, 540.0)
    panel_height = 150.0
    plot_width = width - left - right
    colors = {"BTCUSDT": "#0072B2", "ETHUSDT": "#009E73", "SOLUSDT": "#D55E00"}
    metrics = (
        ("direction_auc", "Direction AUC", 0.5, "random = 0.5"),
        (
            "spearman_information_coefficient",
            "Spearman information coefficient",
            0.0,
            "no rank association = 0",
        ),
        (
            "top_decile_mean_signed_gross_bps",
            "Top-decile mean signed gross return (bps)",
            0.0,
            "gross only; no spread, fees, fills, or ROI",
        ),
    )
    timestamps = [int(fold["evaluation_end_ms"]) for fold in folds]
    minimum_time, maximum_time = min(timestamps), max(timestamps)
    if minimum_time == maximum_time:
        maximum_time += 1

    def x_position(timestamp: int) -> float:
        return left + (timestamp - minimum_time) / (maximum_time - minimum_time) * plot_width

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Rolling tape and coarse-depth forecast diagnostics</title>',
        '<desc id="desc">Time-based out-of-sample AUC, rank correlation, and gross forecast return by BTC, ETH, and SOL fold. This is not executable profit evidence.</desc>',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="42" y="42" font-family="Segoe UI,Arial,sans-serif" font-size="24" fill="#17212B">Rolling forecast evidence</text>',
        '<text x="42" y="70" font-family="Segoe UI,Arial,sans-serif" font-size="14" fill="#4B5B68">UTC evaluation windows; hash-bound real rows; no execution or profitability claim</text>',
    ]
    grouped: dict[str, list[dict[str, object]]] = {}
    for fold in folds:
        grouped.setdefault(str(fold["symbol"]), []).append(dict(fold))
    for values in grouped.values():
        values.sort(key=lambda item: int(item["evaluation_end_ms"]))

    for panel_index, (metric_name, label, reference, caveat) in enumerate(metrics):
        top = panel_top[panel_index]
        bottom = top + panel_height
        values = [float(dict(fold["metrics"])[metric_name]) for fold in folds]
        minimum_value = min(min(values), reference)
        maximum_value = max(max(values), reference)
        span = maximum_value - minimum_value
        padding = max(0.01, span * 0.12)
        minimum_value -= padding
        maximum_value += padding

        def y_position(value: float) -> float:
            return bottom - (value - minimum_value) / (maximum_value - minimum_value) * panel_height

        svg.extend(
            [
                f'<text x="{left:.1f}" y="{top - 18:.1f}" font-family="Segoe UI,Arial,sans-serif" font-size="15" fill="#17212B">{label}</text>',
                f'<text x="{width - right:.1f}" y="{top - 18:.1f}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#667784">{caveat}</text>',
                f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{width-right:.1f}" y2="{bottom:.1f}" stroke="#CBD4DB"/>',
                f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" stroke="#CBD4DB"/>',
                f'<line x1="{left:.1f}" y1="{y_position(reference):.1f}" x2="{width-right:.1f}" y2="{y_position(reference):.1f}" stroke="#8C9AA5" stroke-dasharray="5 5"/>',
                f'<text x="{left-10:.1f}" y="{y_position(reference)+4:.1f}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="11" fill="#667784">{reference:.3g}</text>',
            ]
        )
        for symbol_index, (symbol, symbol_folds) in enumerate(sorted(grouped.items())):
            color = colors.get(symbol, "#6B7280")
            points = [
                (
                    x_position(int(fold["evaluation_end_ms"])),
                    y_position(float(dict(fold["metrics"])[metric_name])),
                )
                for fold in symbol_folds
            ]
            point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            svg.append(
                f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            for x_value, y_value in points:
                svg.append(
                    f'<circle cx="{x_value:.2f}" cy="{y_value:.2f}" r="2.8" fill="{color}"/>'
                )
            final_x, final_y = points[-1]
            label_y = final_y + (symbol_index - 1) * 13
            svg.append(
                f'<text x="{min(final_x + 7, width - right - 52):.2f}" y="{label_y:.2f}" font-family="Segoe UI,Arial,sans-serif" font-size="11" font-weight="600" fill="{color}">{symbol}</text>'
            )
        if panel_index == len(metrics) - 1:
            for tick_index in range(5):
                fraction = tick_index / 4
                timestamp = int(minimum_time + fraction * (maximum_time - minimum_time))
                x_value = x_position(timestamp)
                date_label = datetime.fromtimestamp(timestamp / 1_000, tz=UTC).strftime(
                    "%Y-%m-%d"
                )
                svg.extend(
                    [
                        f'<line x1="{x_value:.1f}" y1="{bottom:.1f}" x2="{x_value:.1f}" y2="{bottom+5:.1f}" stroke="#667784"/>',
                        f'<text x="{x_value:.1f}" y="{bottom+22:.1f}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="11" fill="#4B5B68">{date_label}</text>',
                    ]
                )
            svg.append(
                f'<text x="{left + plot_width/2:.1f}" y="{bottom+48:.1f}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="#4B5B68">UTC evaluation end</text>'
            )
    svg.append("</svg>\n")
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with temporary.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("".join(svg))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return _sha256_file(destination)


def run_tape_depth_prequential(
    warehouse: MicrostructureWarehouse,
    *,
    symbols: Sequence[str],
    output_dir: str | Path,
    training_window_days: int = 730,
    tuning_window_days: int = 30,
    calibration_window_days: int = 30,
    evaluation_window_days: int = 90,
    horizon_seconds: int = 60,
    total_latency_ms: int = 750,
    decision_cadence_seconds: int = 20,
    maximum_depth_age_ms: int = 60_000,
    maximum_rows: int = 5_000_000,
    maximum_cached_rows: int = 15_000_000,
    dataset_cache: bool = True,
    study_stage: str = "development",
    max_folds: int = 0,
    risk_level: str = "conservative",
    model_profile: str | None = None,
    feature_set: str | None = None,
    compute_backend: str = "auto",
    minimum_segment_rows: int = 10_000,
    selection_lock: str | Path | None = None,
    plan_only: bool = False,
    resume: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    """Run or plan full rolling evidence without creating trading authority."""

    stage = str(study_stage).strip().lower()
    if stage not in {"development", "screening", "confirmation"}:
        raise ValueError("study_stage must be development, screening, or confirmation")
    requested_fold_start = 0
    requested_max_folds = int(max_folds)
    selected_profile = str(model_profile or "regularized")
    selected_feature_set = str(feature_set or "full")
    selection_lock_payload: dict[str, object] | None = None
    selection_lock_sha256: str | None = None
    if stage == "confirmation":
        if selection_lock is None:
            raise ValueError("confirmation requires a verified selection lock")
        if requested_max_folds != 0:
            raise ValueError("confirmation fold boundaries come only from the selection lock")
        from .tape_depth_comparison import load_verified_tape_depth_selection

        selection_lock_payload, selection_lock_sha256 = (
            load_verified_tape_depth_selection(selection_lock)
        )
        locked_profile = str(selection_lock_payload["selected_model_profile"])
        locked_feature_set = str(selection_lock_payload["selected_feature_set"])
        if model_profile is not None and str(model_profile) != locked_profile:
            raise ValueError("confirmation model profile differs from the frozen winner")
        if feature_set is not None and str(feature_set) != locked_feature_set:
            raise ValueError("confirmation feature set differs from the frozen winner")
        selected_profile = locked_profile
        selected_feature_set = locked_feature_set
        requested_fold_start = int(
            selection_lock_payload["confirmation_fold_start"]
        )
    elif selection_lock is not None:
        raise ValueError("selection_lock is valid only for confirmation")
    if stage == "screening" and requested_max_folds not in {4, 6, 8, 10}:
        raise ValueError(
            "screening must start at fold 0 and include 4, 6, 8, or 10 folds"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plans = plan_tape_depth_warehouse(
        warehouse,
        symbols=symbols,
        training_window_days=training_window_days,
        tuning_window_days=tuning_window_days,
        calibration_window_days=calibration_window_days,
        evaluation_window_days=evaluation_window_days,
        horizon_seconds=horizon_seconds,
        total_latency_ms=total_latency_ms,
        decision_cadence_seconds=decision_cadence_seconds,
        maximum_rows=maximum_rows,
        fold_start=requested_fold_start,
        max_folds=requested_max_folds,
    )
    if stage == "screening" and any(
        len(plan.folds) != requested_max_folds
        or plan.available_fold_count - len(plan.folds) < 2
        for plan in plans
    ):
        raise ValueError(
            "screening requires its declared folds plus at least two sealed folds per symbol"
        )
    if stage == "confirmation" and any(len(plan.folds) < 2 for plan in plans):
        raise ValueError("confirmation requires at least two untouched folds per symbol")
    total_folds = sum(len(plan.folds) for plan in plans)
    config = {
        "symbols": [plan.symbol for plan in plans],
        "training_window_days": int(training_window_days),
        "tuning_window_days": int(tuning_window_days),
        "calibration_window_days": int(calibration_window_days),
        "evaluation_window_days": int(evaluation_window_days),
        "horizon_seconds": int(horizon_seconds),
        "total_latency_ms": int(total_latency_ms),
        "decision_cadence_seconds": int(decision_cadence_seconds),
        "maximum_depth_age_ms": int(maximum_depth_age_ms),
        "maximum_rows": int(maximum_rows),
        "maximum_cached_rows": int(maximum_cached_rows),
        "dataset_cache": bool(dataset_cache),
        "study_stage": stage,
        "fold_start": requested_fold_start,
        "max_folds": requested_max_folds,
        "risk_level": str(risk_level),
        "model_profile": selected_profile,
        "feature_set": selected_feature_set,
        "compute_backend": str(compute_backend),
        "minimum_segment_rows": int(minimum_segment_rows),
        "selection_lock_sha256": selection_lock_sha256,
    }
    if stage == "confirmation":
        from .tape_depth_comparison import validate_tape_depth_confirmation_request

        assert selection_lock_payload is not None
        validate_tape_depth_confirmation_request(
            selection_lock_payload,
            config=config,
            plans=plans,
        )
    plan_payload: dict[str, object] = {
        "schema_version": TAPE_DEPTH_PREQUENTIAL_SCHEMA_VERSION,
        "truth_basis": "warehouse_coverage_and_timestamp_defined_purged_folds",
        "trading_authority": False,
        "execution_claim": False,
        "plan_only": bool(plan_only),
        "config": config,
        "total_folds": total_folds,
        "coverage_fingerprints": {
            plan.symbol: plan.coverage_fingerprint for plan in plans
        },
        "available_fold_counts": {
            plan.symbol: plan.available_fold_count for plan in plans
        },
        "plans": [plan.asdict() for plan in plans],
    }
    occupied = [
        path
        for path in destination.iterdir()
        if path.name not in {"plan.json"}
    ]
    if plan_only and resume:
        raise ValueError("--plan-only cannot be combined with resume")
    if occupied and not resume:
        raise ValueError(
            "tape/depth prequential output directory contains prior evidence"
        )
    if resume and (destination / "report.json").exists():
        raise ValueError("completed tape/depth evidence cannot be resumed")
    existing_plan_path = destination / "plan.json"
    if resume and existing_plan_path.exists():
        try:
            existing_plan = json.loads(existing_plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("existing tape/depth plan is unreadable") from exc
        if not isinstance(existing_plan, dict) or any(
            existing_plan.get(name) != plan_payload.get(name)
            for name in ("config", "total_folds", "plans")
        ):
            raise ValueError("resume plan differs from the requested tape/depth plan")
    write_json_atomic(destination / "plan.json", plan_payload, indent=2, sort_keys=True)
    if plan_only:
        return plan_payload
    status_path = destination / "run-status.json"
    started_at = datetime.now(tz=UTC).isoformat()
    summaries_path = destination / "fold-summaries.json"
    cache_events_path = destination / "dataset-cache-events.json"
    fold_summaries: list[dict[str, object]] = []
    dataset_cache_events: list[dict[str, object]] = []
    if resume and summaries_path.exists():
        try:
            loaded_summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("resume fold summaries are unreadable") from exc
        if not isinstance(loaded_summaries, list) or not all(
            isinstance(item, dict) for item in loaded_summaries
        ):
            raise ValueError("resume fold summaries must be a JSON list")
        fold_summaries = [dict(item) for item in loaded_summaries]
    if resume and cache_events_path.exists():
        try:
            loaded_cache_events = json.loads(cache_events_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("resume dataset cache events are unreadable") from exc
        if (
            not isinstance(loaded_cache_events, dict)
            or loaded_cache_events.get("schema_version")
            != TAPE_DEPTH_CACHE_SCHEMA_VERSION
            or not isinstance(loaded_cache_events.get("events"), list)
            or not all(
                isinstance(item, dict) for item in loaded_cache_events["events"]
            )
        ):
            raise ValueError("resume dataset cache events have an invalid contract")
        dataset_cache_events = [
            dict(item) for item in loaded_cache_events["events"]
        ]

    flat_folds = tuple(fold for plan in plans for fold in plan.folds)
    if len(fold_summaries) > len(flat_folds):
        raise ValueError("resume has more fold summaries than the current plan")
    destination_root = destination.resolve()
    for index, summary in enumerate(fold_summaries):
        expected = flat_folds[index]
        for name in (
            "symbol",
            "fold_index",
            "dataset_start_ms",
            "tuning_start_ms",
            "calibration_start_ms",
            "evaluation_start_ms",
            "evaluation_end_ms",
        ):
            if summary.get(name) != expected.asdict()[name]:
                raise ValueError(f"resume fold {index} differs at {name}")
        evidence_paths: dict[str, Path] = {}
        for path_name, hash_name in (
            ("artifact_path", "artifact_sha256"),
            ("predictions_path", "predictions_sha256"),
        ):
            relative = Path(str(summary.get(path_name) or ""))
            resolved = (destination / relative).resolve()
            if (
                not relative.parts
                or relative.is_absolute()
                or destination_root not in resolved.parents
                or not resolved.is_file()
                or _sha256_file(resolved) != summary.get(hash_name)
            ):
                raise ValueError(
                    f"resume fold {index} has invalid {path_name} evidence"
                )
            evidence_paths[path_name] = resolved
        artifact = load_tape_depth_model_artifact(evidence_paths["artifact_path"])
        predictions = read_tape_depth_predictions(evidence_paths["predictions_path"])
        source_evidence = artifact.dataset_summary.get("source_evidence")
        source_fingerprint = (
            source_evidence.get("manifest_fingerprint")
            if isinstance(source_evidence, dict)
            else None
        )
        if (
            summary.get("trading_authority") is not False
            or summary.get("execution_claim") is not False
            or artifact.trading_authority
            or artifact.execution_claim
            or summary.get("status") != artifact.status
            or summary.get("risk_level") != artifact.risk_level
            or summary.get("model_profile") != artifact.model_profile
            or summary.get("feature_set") != artifact.feature_set
            or summary.get("backend_kind") != artifact.backend_kind
            or summary.get("backend_device") != artifact.backend_device
            or summary.get("dataset_fingerprint") != artifact.dataset_fingerprint
            or summary.get("source_manifest_fingerprint") != source_fingerprint
            or summary.get("prediction_fingerprint") != predictions.fingerprint()
            or summary.get("metrics") != asdict(artifact.evaluation_metrics)
            or predictions.metrics() != artifact.evaluation_metrics
            or predictions.rows != expected.estimated_evaluation_rows
            or int(predictions.decision_time_ms[0]) != expected.evaluation_start_ms
            or int(predictions.decision_time_ms[-1]) != expected.evaluation_end_ms
        ):
            raise ValueError(f"resume fold {index} evidence binding is invalid")

    def persist_status(state: str, **extra: object) -> None:
        write_json_atomic(
            status_path,
            {
                "schema_version": TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION,
                "state": state,
                "started_at": started_at,
                "updated_at": datetime.now(tz=UTC).isoformat(),
                "completed_folds": len(fold_summaries),
                "total_folds": total_folds,
                "trading_authority": False,
                "execution_claim": False,
                **extra,
            },
            indent=2,
            sort_keys=True,
        )

    persist_status("running")
    try:
        resume_fold_count = len(fold_summaries)
        global_offset = 0
        for symbol_plan in plans:
            symbol_fold_count = len(symbol_plan.folds)
            first_unfinished = max(
                0,
                min(symbol_fold_count, resume_fold_count - global_offset),
            )
            for local_index in range(first_unfinished):
                if progress:
                    progress(
                        "resume-verified",
                        global_offset + local_index + 1,
                        total_folds,
                    )
            if first_unfinished >= symbol_fold_count:
                global_offset += symbol_fold_count
                continue

            remaining_folds = symbol_plan.folds[first_unfinished:]
            cached_start_ms = remaining_folds[0].dataset_start_ms
            cached_end_ms = remaining_folds[-1].evaluation_end_ms
            estimated_cached_rows = (
                (cached_end_ms - cached_start_ms)
                // (int(decision_cadence_seconds) * 1_000)
                + 1
            )
            if not 1 <= estimated_cached_rows <= int(maximum_cached_rows):
                raise ValueError(
                    f"{symbol_plan.symbol} cached feature build may emit "
                    f"{estimated_cached_rows} rows; maximum_cached_rows="
                    f"{int(maximum_cached_rows)}"
                )
            feature_phase = "cache-lookup" if dataset_cache else "feature-build"
            persist_status(
                "running",
                symbol=symbol_plan.symbol,
                model_phase=feature_phase,
                cached_start_ms=cached_start_ms,
                cached_end_ms=cached_end_ms,
                estimated_cached_rows=estimated_cached_rows,
            )
            if progress:
                progress(
                    f"{symbol_plan.symbol}-{feature_phase}",
                    global_offset + first_unfinished,
                    total_folds,
                )
            heartbeat_stop = threading.Event()
            feature_started = time.monotonic()

            def feature_heartbeat() -> None:
                while not heartbeat_stop.wait(30.0):
                    elapsed_seconds = int(time.monotonic() - feature_started)
                    persist_status(
                        "running",
                        symbol=symbol_plan.symbol,
                        model_phase=feature_phase,
                        feature_build_elapsed_seconds=elapsed_seconds,
                        estimated_cached_rows=estimated_cached_rows,
                    )
                    if progress:
                        progress(
                            f"{symbol_plan.symbol}-feature-build-heartbeat-"
                            f"{elapsed_seconds}s",
                            global_offset + first_unfinished,
                            total_folds,
                        )

            heartbeat_thread = threading.Thread(
                target=feature_heartbeat,
                name=f"tape-depth-{symbol_plan.symbol}-heartbeat",
                daemon=True,
            )
            heartbeat_thread.start()
            dataset_cache_key: str | None = None
            dataset_cache_state = "disabled"

            def validate_prefetched_dataset(
                candidate: TapeDepthForecastDataset,
            ) -> None:
                if (
                    candidate.rows != estimated_cached_rows
                    or int(candidate.decision_time_ms[0]) != cached_start_ms
                    or int(candidate.decision_time_ms[-1]) != cached_end_ms
                ):
                    raise ValueError(
                        f"{symbol_plan.symbol} cached feature build differs from "
                        "the planned row/time interval"
                    )

            try:
                expected_source_evidence = _dataset_range_source_evidence(
                    warehouse,
                    symbol=symbol_plan.symbol,
                    start_ms=cached_start_ms,
                    end_ms=cached_end_ms,
                    horizon_seconds=horizon_seconds,
                    total_latency_ms=total_latency_ms,
                )
                prefetched_dataset = None
                if dataset_cache:
                    prefetched_dataset = load_tape_depth_dataset_cache(
                        warehouse,
                        symbol=symbol_plan.symbol,
                        requested_start_ms=cached_start_ms,
                        requested_end_ms=cached_end_ms,
                        horizon_seconds=horizon_seconds,
                        total_latency_ms=total_latency_ms,
                        decision_cadence_seconds=decision_cadence_seconds,
                        maximum_depth_age_ms=maximum_depth_age_ms,
                        source_evidence=expected_source_evidence,
                    )
                    if prefetched_dataset is not None:
                        validate_prefetched_dataset(prefetched_dataset)
                        dataset_cache_state = "hit"
                        feature_phase = "cache-hit"
                if prefetched_dataset is None:
                    feature_phase = "feature-build"
                    prefetched_dataset = build_tape_depth_forecast_dataset(
                        warehouse,
                        symbol=symbol_plan.symbol,
                        start_ms=cached_start_ms,
                        end_ms=cached_end_ms,
                        horizon_seconds=horizon_seconds,
                        total_latency_ms=total_latency_ms,
                        decision_cadence_seconds=decision_cadence_seconds,
                        maximum_depth_age_ms=maximum_depth_age_ms,
                        maximum_rows=int(maximum_cached_rows),
                    )
                    if dict(prefetched_dataset.source_evidence) != dict(
                        expected_source_evidence
                    ):
                        raise ValueError(
                            "tape/depth feature build source evidence drifted"
                        )
                    validate_prefetched_dataset(prefetched_dataset)
                    if dataset_cache:
                        feature_phase = "cache-write"
                        dataset_cache_key = save_tape_depth_dataset_cache(
                            warehouse,
                            prefetched_dataset,
                        )
                        dataset_cache_state = "written"
                elif dataset_cache:
                    dataset_cache_key = tape_depth_dataset_cache_key(
                        symbol=symbol_plan.symbol,
                        requested_start_ms=cached_start_ms,
                        requested_end_ms=cached_end_ms,
                        horizon_seconds=horizon_seconds,
                        total_latency_ms=total_latency_ms,
                        decision_cadence_seconds=decision_cadence_seconds,
                        maximum_depth_age_ms=maximum_depth_age_ms,
                        source_evidence=expected_source_evidence,
                    )
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=5.0)
            dataset_cache_events.append(
                {
                    "symbol": symbol_plan.symbol,
                    "requested_start_ms": cached_start_ms,
                    "requested_end_ms": cached_end_ms,
                    "state": dataset_cache_state,
                    "cache_key": dataset_cache_key,
                    "rows": prefetched_dataset.rows,
                    "feature_version": prefetched_dataset.feature_version,
                    "source_manifest_fingerprint": prefetched_dataset.source_evidence[
                        "manifest_fingerprint"
                    ],
                }
            )
            write_json_atomic(
                cache_events_path,
                {
                    "schema_version": TAPE_DEPTH_CACHE_SCHEMA_VERSION,
                    "truth_basis": "verified_derived_dataset_cache_events",
                    "events": dataset_cache_events,
                },
                indent=2,
                sort_keys=True,
            )
            persist_status(
                "running",
                symbol=symbol_plan.symbol,
                model_phase="feature-build-complete",
                cached_rows=prefetched_dataset.rows,
                dataset_cache_state=dataset_cache_state,
                dataset_cache_key=dataset_cache_key,
            )
            if progress:
                progress(
                    f"{symbol_plan.symbol}-feature-build-complete",
                    global_offset + first_unfinished,
                    total_folds,
                )

            for local_index in range(first_unfinished, symbol_fold_count):
                fold = symbol_plan.folds[local_index]
                global_fold_index = global_offset + local_index

                def fold_progress(phase: str, completed: int, total: int) -> None:
                    persist_status(
                        "running",
                        symbol=fold.symbol,
                        fold_index=fold.fold_index,
                        model_phase=phase,
                        model_phase_completed=int(completed),
                        model_phase_total=int(total),
                    )
                    if progress:
                        progress(
                            f"{fold.symbol}-fold-{fold.fold_index}-{phase}",
                            global_fold_index,
                            total_folds,
                        )

                evaluation = evaluate_tape_depth_fold(
                    warehouse,
                    plan=fold,
                    horizon_seconds=horizon_seconds,
                    total_latency_ms=total_latency_ms,
                    decision_cadence_seconds=decision_cadence_seconds,
                    maximum_depth_age_ms=maximum_depth_age_ms,
                    maximum_rows=maximum_rows,
                    risk_level=risk_level,
                    model_profile=selected_profile,
                    feature_set=selected_feature_set,
                    compute_backend=compute_backend,
                    minimum_segment_rows=minimum_segment_rows,
                    prefetched_dataset=prefetched_dataset,
                    progress=fold_progress,
                )
                stem = f"{fold.symbol.lower()}-fold-{fold.fold_index:04d}"
                artifact_relative = Path("models") / f"{stem}.json"
                prediction_relative = Path("predictions") / f"{stem}.csv.gz"
                artifact_path = destination / artifact_relative
                prediction_path = destination / prediction_relative
                save_tape_depth_model_artifact(evaluation.artifact, artifact_path)
                artifact_sha256 = _sha256_file(artifact_path)
                predictions_sha256 = write_tape_depth_predictions(
                    evaluation.predictions,
                    prediction_path,
                )
                fold_summaries.append(
                    _fold_summary(
                        evaluation,
                        artifact_path=artifact_relative,
                        artifact_sha256=artifact_sha256,
                        predictions_path=prediction_relative,
                        predictions_sha256=predictions_sha256,
                    )
                )
                write_json_atomic(
                    summaries_path,
                    fold_summaries,
                    indent=2,
                    sort_keys=True,
                )
                persist_status(
                    "running",
                    last_symbol=fold.symbol,
                    last_fold_index=fold.fold_index,
                )
                if progress:
                    progress("fold-complete", len(fold_summaries), total_folds)
            del evaluation
            del prefetched_dataset
            global_offset += symbol_fold_count
    except Exception as exc:
        persist_status(
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise

    fold_metrics_path = destination / "fold-metrics.csv"
    fold_metrics_sha256 = _write_fold_metrics_csv(
        fold_summaries,
        fold_metrics_path,
    )
    diagnostics_path = destination / "forecast-diagnostics.svg"
    diagnostics_sha256 = render_tape_depth_prequential_svg(
        fold_summaries,
        diagnostics_path,
    )
    all_candidates = all(
        fold["status"] == "research_candidate" for fold in fold_summaries
    )
    report: dict[str, object] = {
        "schema_version": TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION,
        "status": "research_candidate" if all_candidates else "rejected",
        "truth_basis": "hash_bound_serialized_model_replay_on_nonoverlapping_rolling_folds",
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "started_at": started_at,
        "completed_at": datetime.now(tz=UTC).isoformat(),
        "config": config,
        "plan_fingerprints": {
            plan.symbol: plan.plan_fingerprint for plan in plans
        },
        "coverage_fingerprints": {
            plan.symbol: plan.coverage_fingerprint for plan in plans
        },
        "available_fold_counts": {
            plan.symbol: plan.available_fold_count for plan in plans
        },
        "total_folds": total_folds,
        "completed_folds": len(fold_summaries),
        "fold_metrics_path": fold_metrics_path.name,
        "fold_metrics_sha256": fold_metrics_sha256,
        "forecast_diagnostics_path": diagnostics_path.name,
        "forecast_diagnostics_sha256": diagnostics_sha256,
        "aggregate_forecast_metrics": _aggregate_fold_metrics(fold_summaries),
        "dataset_cache": {
            "enabled": bool(dataset_cache),
            "schema_version": TAPE_DEPTH_CACHE_SCHEMA_VERSION,
            "truth_basis": "transactional_source_bound_duckdb_derived_dataset_cache",
            "events": dataset_cache_events,
        },
        "folds": fold_summaries,
        "limitations": [
            "gross trade-reference targets are not executable fills or after-cost PnL",
            "overlapping forecast horizons are not summed into ROI",
            "exact BBO replay and current no-order shadow remain mandatory",
        ],
    }
    write_json_atomic(destination / "report.json", report, indent=2, sort_keys=True)
    persist_status("complete", report_status=report["status"])
    return report


__all__ = [
    "TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION",
    "TAPE_DEPTH_PREQUENTIAL_SCHEMA_VERSION",
    "TapeDepthFoldEvaluation",
    "TapeDepthFoldPlan",
    "TapeDepthSymbolPlan",
    "evaluate_tape_depth_fold",
    "plan_tape_depth_folds",
    "plan_tape_depth_warehouse",
    "read_tape_depth_predictions",
    "render_tape_depth_prequential_svg",
    "run_tape_depth_prequential",
    "verify_tape_depth_prequential_report",
    "write_tape_depth_predictions",
]
