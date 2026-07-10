"""Causal rolling-refit evaluation for microstructure action-value candidates."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from .microstructure_features import MicrostructureDataset
from .microstructure_model import (
    MICROSTRUCTURE_PREQUENTIAL_EVIDENCE_VERSION,
    MicrostructureModelArtifact,
    PrequentialValidationEvidence,
    ThresholdPolicy,
    TradingMetrics,
    _apply_platt_scaling,
    _backend_parameters,
    _baseline_metrics,
    _fit_platt_scaling,
    _minimum_evaluation_trades,
    _model_strings_sha256,
    _performance_confidence,
    _risk_parameters,
    _risk_utility,
    _select_threshold,
    _side_probability_quality,
    _simulate_non_overlapping_trace,
    _train_fixed_booster,
    _trading_metrics,
    _validated_source_evidence,
    microstructure_candidate_sha256,
)
from .storage import write_json_atomic


PREQUENTIAL_EVIDENCE_VERSION = MICROSTRUCTURE_PREQUENTIAL_EVIDENCE_VERSION
_DAY_MS = 86_400_000
_SIDES = ("long", "short")
PrequentialProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class PrequentialConfig:
    training_window_days: int = 180
    minimum_training_days: int = 60
    calibration_days: int = 14
    policy_days: int = 14
    evaluation_block_days: int = 7
    minimum_segment_rows: int = 256
    minimum_class_rows: int = 128
    bootstrap_samples: int = 2_000
    max_folds: int = 0

    def validated(self) -> "PrequentialConfig":
        if self.training_window_days < 0:
            raise ValueError("training_window_days must be zero (expanding) or positive")
        if self.minimum_training_days < 5:
            raise ValueError("minimum_training_days must be at least 5")
        if min(self.calibration_days, self.policy_days, self.evaluation_block_days) < 1:
            raise ValueError("calibration, policy, and evaluation windows must be positive")
        if self.minimum_segment_rows < 32 or self.minimum_class_rows < 16:
            raise ValueError("prequential row/class support is too small")
        if self.bootstrap_samples < 1_000:
            raise ValueError("prequential confidence requires at least 1,000 bootstrap samples")
        if self.max_folds < 0:
            raise ValueError("max_folds must be non-negative")
        if 0 < self.training_window_days < self.minimum_training_days:
            raise ValueError(
                "training_window_days cannot be shorter than minimum_training_days"
            )
        return self

    def asdict(self) -> dict[str, object]:
        return asdict(self)


PROMOTION_PREQUENTIAL_CONFIG = PrequentialConfig()


def prequential_protocol_sha256(config: PrequentialConfig) -> str:
    payload = {
        "version": PREQUENTIAL_EVIDENCE_VERSION,
        "config": config.validated().asdict(),
        "fold_refit": "rolling_full_retrain_fixed_hyperparameters",
        "terminal_holdout": "not_accessed",
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PrequentialFoldPlan:
    fold: int
    calibration_start_ms: int
    policy_start_ms: int
    evaluation_start_ms: int
    evaluation_end_exclusive_ms: int
    training_indexes: np.ndarray
    calibration_indexes: np.ndarray
    policy_indexes: np.ndarray
    evaluation_indexes: np.ndarray

    def summary(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "calibration_start_ms": self.calibration_start_ms,
            "policy_start_ms": self.policy_start_ms,
            "evaluation_start_ms": self.evaluation_start_ms,
            "evaluation_end_exclusive_ms": self.evaluation_end_exclusive_ms,
            "training_rows": len(self.training_indexes),
            "calibration_rows": len(self.calibration_indexes),
            "policy_rows": len(self.policy_indexes),
            "evaluation_rows": len(self.evaluation_indexes),
        }


@dataclass(frozen=True)
class PrequentialEvaluationReport:
    version: str
    generated_at_ms: int
    status: str
    trading_authority: bool
    reasons: tuple[str, ...]
    candidate_sha256: str
    symbol: str
    risk_level: str
    config: Mapping[str, object]
    data_contract: Mapping[str, object]
    coverage: Mapping[str, object]
    folds: tuple[Mapping[str, object], ...]
    aggregate: Mapping[str, object]
    predictions_path: str
    predictions_sha256: str
    chart_path: str
    chart_sha256: str

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["passed"] = self.passed
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class _FoldModels:
    models: Mapping[str, object]
    calibration: Mapping[str, tuple[float, float]]
    model_sha256: str
    backend_kind: str
    backend_device: str
    side_training_rows: Mapping[str, int]
    side_calibration_rows: Mapping[str, int]


@dataclass(frozen=True)
class _PredictionBatch:
    long_edge: np.ndarray
    short_edge: np.ndarray
    long_probability: np.ndarray
    short_probability: np.ndarray


@dataclass(frozen=True)
class _VariablePolicyTrace:
    metrics: TradingMetrics
    pnls: tuple[float, ...]
    sides: tuple[int, ...]
    timestamps: tuple[int, ...]
    selected_side: np.ndarray
    executed: np.ndarray


def _iso_utc(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1_000.0, tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_lf_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    last_byte = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            if b"\r" in chunk:
                raise ValueError(f"{path} is not canonical LF-only text")
            digest.update(chunk)
            last_byte = chunk[-1:]
    if last_byte != b"\n":
        raise ValueError(f"{path} is not LF-terminated text")
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_candidate_contract(
    artifact: MicrostructureModelArtifact,
    dataset: MicrostructureDataset,
) -> dict[str, object]:
    if artifact.status != "candidate" or artifact.rejection_reasons:
        raise ValueError("prequential evaluation requires an unrejected candidate artifact")
    if artifact.terminal_evaluated_at is not None or artifact.terminal_metrics is not None:
        raise ValueError("prequential evaluation must run before terminal evidence is consumed")
    if artifact.deployment_model_strings is not None or artifact.deployment_refit is not None:
        raise ValueError("prequential evaluation cannot use a deployment-refit artifact")
    if artifact.model_family != "side_specific_hurdle_expected_value":
        raise ValueError("prequential candidate model family is unsupported")
    if artifact.lightgbm_version != str(lgb.__version__):
        raise ValueError("prequential LightGBM version differs from candidate training")
    if (
        artifact.target_mode != "exchange_trigger_market_exit_1s_adverse_first"
        or artifact.stop_loss_bps is None
        or artifact.stop_loss_bps <= 0.0
        or artifact.take_profit_bps is None
        or artifact.take_profit_bps <= 0.0
        or artifact.trigger_execution_slippage_bps is None
        or artifact.trigger_execution_slippage_bps < 0.0
        or artifact.path_resolution_ms != 1_000
    ):
        raise ValueError("prequential candidate lacks the required path-aware exit contract")
    expected_models = {
        f"{side}_{component}"
        for side in _SIDES
        for component in ("probability", "win_magnitude", "loss_magnitude")
    }
    if set(artifact.best_iterations) != expected_models or any(
        int(value) <= 0 for value in artifact.best_iterations.values()
    ):
        raise ValueError("prequential candidate iteration contract is incomplete")
    exact_fields = (
        (artifact.symbol, dataset.symbol, "symbol"),
        (artifact.feature_version, dataset.feature_version, "feature version"),
        (artifact.feature_names, dataset.feature_names, "feature names"),
        (artifact.horizon_seconds, dataset.horizon_seconds, "horizon"),
        (artifact.total_latency_ms, dataset.total_latency_ms, "latency"),
        (artifact.target_mode, dataset.target_mode, "target mode"),
        (artifact.decision_cadence_seconds, dataset.decision_cadence_seconds, "cadence"),
        (artifact.max_quote_age_ms, dataset.max_quote_age_ms, "maximum quote age"),
        (artifact.stop_loss_bps, dataset.stop_loss_bps, "stop loss"),
        (artifact.take_profit_bps, dataset.take_profit_bps, "take profit"),
        (
            artifact.trigger_execution_slippage_bps,
            dataset.trigger_execution_slippage_bps,
            "trigger execution slippage",
        ),
        (artifact.path_resolution_ms, dataset.path_resolution_ms, "path resolution"),
        (
            artifact.dataset_summary.get("trade_feature_embargo_ms"),
            dataset.trade_feature_embargo_ms,
            "trade feature embargo",
        ),
    )
    for expected, actual, label in exact_fields:
        if expected != actual:
            raise ValueError(f"prequential dataset {label} drifted from the candidate")
    float_fields = (
        (artifact.taker_fee_bps, dataset.taker_fee_bps, "taker fee"),
        (
            artifact.reference_order_notional_quote,
            dataset.reference_order_notional_quote,
            "reference order notional",
        ),
        (artifact.max_l1_participation, dataset.max_l1_participation, "L1 participation"),
    )
    for expected, actual, label in float_fields:
        if not math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"prequential dataset {label} drifted from the candidate")
    if dataset.rows <= 0 or not np.all(np.isfinite(dataset.features)):
        raise ValueError("prequential dataset is empty or non-finite")
    if np.any(np.diff(dataset.decision_time_ms) <= 0):
        raise ValueError("prequential decisions must be strictly increasing")
    source = _validated_source_evidence(dataset.source_evidence, symbol=dataset.symbol)
    candidate_source = artifact.dataset_summary.get("source_evidence")
    if not isinstance(candidate_source, Mapping):
        raise ValueError("candidate is missing source evidence")
    if (
        source["build_id"] != candidate_source.get("build_id")
        or source["manifest_fingerprint"]
        != candidate_source.get("manifest_fingerprint")
    ):
        raise ValueError("prequential source provenance drifted from the candidate")
    return source


def plan_prequential_folds(
    artifact: MicrostructureModelArtifact,
    dataset: MicrostructureDataset,
    config: PrequentialConfig,
) -> tuple[PrequentialFoldPlan, ...]:
    cfg = config.validated()
    _validate_candidate_contract(artifact, dataset)
    times = np.asarray(dataset.decision_time_ms, dtype=np.int64)
    available_ms = np.maximum(
        np.asarray(dataset.long_exit_time_ms, dtype=np.int64),
        np.asarray(dataset.short_exit_time_ms, dtype=np.int64),
    ) + int(dataset.trade_feature_embargo_ms)
    selection_start_ms = int(artifact.split.selection_start_ms)
    terminal_start_ms = int(artifact.split.terminal_start_ms)
    if terminal_start_ms <= selection_start_ms:
        raise ValueError("candidate selection/terminal time contract is invalid")
    all_plans: list[PrequentialFoldPlan] = []
    fold = 0
    evaluation_start_ms = selection_start_ms
    while evaluation_start_ms < terminal_start_ms:
        fold += 1
        evaluation_end_ms = min(
            terminal_start_ms,
            evaluation_start_ms + int(cfg.evaluation_block_days) * _DAY_MS,
        )
        policy_start_ms = evaluation_start_ms - int(cfg.policy_days) * _DAY_MS
        calibration_start_ms = policy_start_ms - int(cfg.calibration_days) * _DAY_MS
        training_start_ms = (
            int(times[0])
            if cfg.training_window_days <= 0
            else calibration_start_ms - int(cfg.training_window_days) * _DAY_MS
        )
        training = np.flatnonzero(
            (times >= training_start_ms)
            & (times < calibration_start_ms)
            & (available_ms < calibration_start_ms)
        ).astype(np.int64)
        calibration = np.flatnonzero(
            (times >= calibration_start_ms)
            & (times < policy_start_ms)
            & (available_ms < policy_start_ms)
        ).astype(np.int64)
        policy = np.flatnonzero(
            (times >= policy_start_ms)
            & (times < evaluation_start_ms)
            & (available_ms < evaluation_start_ms)
        ).astype(np.int64)
        evaluation = np.flatnonzero(
            (times >= evaluation_start_ms)
            & (times < evaluation_end_ms)
            & (available_ms < artifact.split.terminal_start_ms)
        ).astype(np.int64)
        all_plans.append(
            PrequentialFoldPlan(
                fold=fold,
                calibration_start_ms=calibration_start_ms,
                policy_start_ms=policy_start_ms,
                evaluation_start_ms=evaluation_start_ms,
                evaluation_end_exclusive_ms=evaluation_end_ms,
                training_indexes=training,
                calibration_indexes=calibration,
                policy_indexes=policy,
                evaluation_indexes=evaluation,
            )
        )
        evaluation_start_ms = evaluation_end_ms
    if not all_plans:
        raise ValueError("candidate has no pre-terminal selection days for prequential scoring")
    if cfg.max_folds > 0:
        return tuple(all_plans[: cfg.max_folds])
    return tuple(all_plans)


def _require_segment_support(
    plan: PrequentialFoldPlan,
    dataset: MicrostructureDataset,
    config: PrequentialConfig,
) -> None:
    for label, indexes in (
        ("training", plan.training_indexes),
        ("calibration", plan.calibration_indexes),
        ("policy", plan.policy_indexes),
        ("evaluation", plan.evaluation_indexes),
    ):
        if len(indexes) < config.minimum_segment_rows:
            raise ValueError(
                f"fold {plan.fold} {label} rows {len(indexes)}<"
                f"{config.minimum_segment_rows}"
            )
    training_days = len(
        np.unique(dataset.decision_time_ms[plan.training_indexes] // _DAY_MS)
    )
    if training_days < config.minimum_training_days:
        raise ValueError(
            f"fold {plan.fold} training days {training_days}<"
            f"{config.minimum_training_days}"
        )


def _fit_fold_models(
    artifact: MicrostructureModelArtifact,
    dataset: MicrostructureDataset,
    plan: PrequentialFoldPlan,
    config: PrequentialConfig,
    *,
    compute_backend: str,
    seed: int,
) -> _FoldModels:
    _require_segment_support(plan, dataset, config)
    x = np.asarray(dataset.features, dtype=np.float32)
    parameters, backend_kind, backend_device = _backend_parameters(compute_backend, seed)
    parameters = {
        **parameters,
        **_risk_parameters(artifact.risk_level, int(artifact.split.train_rows)),
    }
    targets = {"long": dataset.long_net_bps, "short": dataset.short_net_bps}
    eligible = {
        "long": np.asarray(dataset.long_liquidity_eligible, dtype=bool),
        "short": np.asarray(dataset.short_liquidity_eligible, dtype=bool),
    }
    models: dict[str, object] = {}
    calibration: dict[str, tuple[float, float]] = {}
    side_training_rows: dict[str, int] = {}
    side_calibration_rows: dict[str, int] = {}
    for side in _SIDES:
        train = plan.training_indexes[eligible[side][plan.training_indexes]]
        calibrate = plan.calibration_indexes[eligible[side][plan.calibration_indexes]]
        labels = (targets[side][train] > 0.0).astype(np.float32)
        calibration_labels = (targets[side][calibrate] > 0.0).astype(np.float32)
        support = (
            int(np.sum(labels == 0.0)),
            int(np.sum(labels == 1.0)),
            int(np.sum(calibration_labels == 0.0)),
            int(np.sum(calibration_labels == 1.0)),
        )
        if min(support) < config.minimum_class_rows:
            raise ValueError(
                f"fold {plan.fold} {side} classifier class support {support}<"
                f"{config.minimum_class_rows}"
            )
        side_training_rows[side] = len(train)
        side_calibration_rows[side] = len(calibrate)
        probability_name = f"{side}_probability"
        classifier = _train_fixed_booster(
            features=x[train],
            targets=labels,
            objective="binary",
            metric="binary_logloss",
            parameters=parameters,
            iterations=int(artifact.best_iterations[probability_name]),
        )
        models[probability_name] = classifier
        calibration[side] = _fit_platt_scaling(
            classifier.predict(x[calibrate]),
            calibration_labels,
        )
        for outcome, keep, magnitude in (
            ("win", labels == 1.0, targets[side][train]),
            ("loss", labels == 0.0, -targets[side][train]),
        ):
            if int(np.sum(keep)) < config.minimum_class_rows:
                raise ValueError(
                    f"fold {plan.fold} {side} {outcome} rows<"
                    f"{config.minimum_class_rows}"
                )
            name = f"{side}_{outcome}_magnitude"
            models[name] = _train_fixed_booster(
                features=x[train][keep],
                targets=np.asarray(magnitude[keep], dtype=np.float32),
                objective="regression",
                metric="l2",
                parameters=parameters,
                iterations=int(artifact.best_iterations[name]),
            )
    model_strings = {
        name: model.model_to_string(num_iteration=int(artifact.best_iterations[name]))
        for name, model in models.items()
    }
    return _FoldModels(
        models=models,
        calibration=calibration,
        model_sha256=_model_strings_sha256(model_strings),
        backend_kind=backend_kind,
        backend_device=backend_device,
        side_training_rows=side_training_rows,
        side_calibration_rows=side_calibration_rows,
    )


def _predict_models(
    artifact: MicrostructureModelArtifact,
    dataset: MicrostructureDataset,
    fitted: _FoldModels,
    indexes: np.ndarray,
) -> _PredictionBatch:
    x = np.asarray(dataset.features[indexes], dtype=np.float32)
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in _SIDES:
        probability = _apply_platt_scaling(
            fitted.models[f"{side}_probability"].predict(x),
            fitted.calibration[side],
        )
        win = np.maximum(0.0, fitted.models[f"{side}_win_magnitude"].predict(x))
        loss = np.maximum(0.0, fitted.models[f"{side}_loss_magnitude"].predict(x))
        output[side] = (
            probability * win - (1.0 - probability) * loss,
            probability,
        )
    return _PredictionBatch(
        long_edge=np.asarray(output["long"][0], dtype=np.float64),
        short_edge=np.asarray(output["short"][0], dtype=np.float64),
        long_probability=np.asarray(output["long"][1], dtype=np.float64),
        short_probability=np.asarray(output["short"][1], dtype=np.float64),
    )


def _variable_policy_trace(
    dataset: MicrostructureDataset,
    indexes: np.ndarray,
    predictions: _PredictionBatch,
    edge_thresholds: np.ndarray,
    probability_thresholds: np.ndarray,
) -> _VariablePolicyTrace:
    rows = len(indexes)
    if not (
        len(predictions.long_edge)
        == len(predictions.short_edge)
        == len(predictions.long_probability)
        == len(predictions.short_probability)
        == len(edge_thresholds)
        == len(probability_thresholds)
        == rows
    ):
        raise ValueError("prequential prediction/policy arrays are inconsistent")
    selected_side = np.zeros(rows, dtype=np.int8)
    executed = np.zeros(rows, dtype=bool)
    pnls: list[float] = []
    sides: list[int] = []
    timestamps: list[int] = []
    next_available_ms = -1
    for offset, index in enumerate(indexes):
        long_ok = (
            bool(dataset.long_liquidity_eligible[index])
            and predictions.long_edge[offset] >= edge_thresholds[offset]
            and predictions.long_probability[offset] >= probability_thresholds[offset]
        )
        short_ok = (
            bool(dataset.short_liquidity_eligible[index])
            and predictions.short_edge[offset] >= edge_thresholds[offset]
            and predictions.short_probability[offset] >= probability_thresholds[offset]
        )
        if not long_ok and not short_ok:
            continue
        side = (
            1
            if long_ok and (not short_ok or predictions.long_edge[offset] >= predictions.short_edge[offset])
            else -1
        )
        selected_side[offset] = side
        timestamp = int(dataset.decision_time_ms[index])
        if timestamp < next_available_ms:
            continue
        executed[offset] = True
        pnl = (
            float(dataset.long_net_bps[index])
            if side > 0
            else float(dataset.short_net_bps[index])
        )
        exit_time = (
            int(dataset.long_exit_time_ms[index])
            if side > 0
            else int(dataset.short_exit_time_ms[index])
        )
        pnls.append(pnl)
        sides.append(side)
        timestamps.append(timestamp)
        next_available_ms = exit_time
    return _VariablePolicyTrace(
        metrics=_trading_metrics(pnls, sides, timestamps),
        pnls=tuple(pnls),
        sides=tuple(sides),
        timestamps=tuple(timestamps),
        selected_side=selected_side,
        executed=executed,
    )


def _prediction_rows(
    dataset: MicrostructureDataset,
    fold_ids: np.ndarray,
    indexes: np.ndarray,
    predictions: _PredictionBatch,
    policies: Sequence[ThresholdPolicy],
    trace: _VariablePolicyTrace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, index in enumerate(indexes):
        policy = policies[offset]
        side = int(trace.selected_side[offset])
        actual = (
            float(dataset.long_net_bps[index])
            if side > 0
            else (float(dataset.short_net_bps[index]) if side < 0 else 0.0)
        )
        exit_time = (
            int(dataset.long_exit_time_ms[index])
            if side > 0
            else (
                int(dataset.short_exit_time_ms[index])
                if side < 0
                else int(dataset.decision_time_ms[index])
            )
        )
        rows.append(
            {
                "fold": int(fold_ids[offset]),
                "symbol": dataset.symbol,
                "decision_time_ms": int(dataset.decision_time_ms[index]),
                "decision_time_utc": _iso_utc(int(dataset.decision_time_ms[index])),
                "long_expected_net_bps": float(predictions.long_edge[offset]),
                "short_expected_net_bps": float(predictions.short_edge[offset]),
                "long_profitable_probability": float(predictions.long_probability[offset]),
                "short_profitable_probability": float(predictions.short_probability[offset]),
                "minimum_predicted_edge_bps": float(policy.minimum_predicted_edge_bps),
                "minimum_profitable_probability": float(
                    policy.minimum_profitable_probability
                ),
                "selected_side": side,
                "executed_nonoverlap": bool(trace.executed[offset]),
                "realized_selected_net_bps": actual,
                "selected_exit_time_ms": exit_time,
            }
        )
    return rows


def _write_predictions(path: Path, rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        raise ValueError("prequential prediction output is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            fields = tuple(rows[0])
            writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256(temporary)
        os.replace(temporary, path)
        return digest
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _render_chart(
    rows: Sequence[Mapping[str, object]],
    folds: Sequence[Mapping[str, object]],
) -> bytes:
    width, height = 1_200, 680
    left, right = 88, 34
    plot_width = width - left - right
    top, panel_height = 110, 330
    fold_top, fold_height = 500, 110
    executed_rows = [row for row in rows if bool(row["executed_nonoverlap"])]
    cumulative: list[tuple[int, float]] = []
    total = 0.0
    for row in executed_rows:
        total += float(row["realized_selected_net_bps"])
        cumulative.append((int(row["decision_time_ms"]), total))
    all_times = [int(row["decision_time_ms"]) for row in rows]
    minimum_time = min(all_times)
    maximum_time = max(all_times)
    time_span = max(1, maximum_time - minimum_time)
    values = [value for _timestamp, value in cumulative]
    lower = min(0.0, min(values, default=0.0))
    upper = max(0.0, max(values, default=0.0))
    if math.isclose(lower, upper):
        upper = lower + 1.0
    padding = max(1.0, (upper - lower) * 0.10)
    lower -= padding
    upper += padding

    def x_position(timestamp: int) -> float:
        return left + (timestamp - minimum_time) / time_span * plot_width

    def y_position(value: float) -> float:
        return top + (upper - value) / (upper - lower) * panel_height

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#111827;letter-spacing:0}.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#4b5563}.axis{font-size:12px}.panel{font-size:16px;font-weight:600}</style>',
        f'<text class="title" x="{left}" y="38">Prequential microstructure selection evidence</text>',
        f'<text class="sub" x="{left}" y="62">{rows[0]["symbol"]} | rolling fixed-hyperparameter refits | after-cost executable targets | terminal holdout sealed</text>',
        f'<text class="sub" x="{left}" y="82">Cumulative selected net bps is simulated research evidence, not exchange P&amp;L or trading authority.</text>',
        f'<text class="panel" x="{left}" y="{top - 14}">Cumulative non-overlapping selected net bps</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="#f9fafb" stroke="#d1d5db"/>',
    ]
    for tick in range(5):
        fraction = tick / 4.0
        value = upper - fraction * (upper - lower)
        y = top + fraction * panel_height
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        lines.append(
            f'<text class="axis" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{value:.2f}</text>'
        )
    if lower <= 0.0 <= upper:
        zero_y = y_position(0.0)
        lines.append(
            f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#111827"/>'
        )
    if cumulative:
        points = " ".join(
            f"{x_position(timestamp):.2f},{y_position(value):.2f}"
            for timestamp, value in cumulative
        )
        lines.append(
            f'<polyline fill="none" stroke="#0f766e" stroke-width="2.4" points="{points}"/>'
        )
    for fraction in (0.0, 0.5, 1.0):
        timestamp = minimum_time + int(time_span * fraction)
        x = left + plot_width * fraction
        lines.append(
            f'<text class="axis" x="{x:.2f}" y="{top+panel_height+22}" text-anchor="middle">{_iso_utc(timestamp)[:10]}</text>'
        )
    successful = [fold for fold in folds if fold.get("status") == "complete"]
    fold_values = [float(fold["evaluation_metrics"]["total_net_bps"]) for fold in successful]
    fold_bound = max(1.0, max((abs(value) for value in fold_values), default=0.0) * 1.1)
    lines.extend(
        (
            f'<text class="panel" x="{left}" y="{fold_top - 14}">Net bps by untouched evaluation block</text>',
            f'<line x1="{left}" y1="{fold_top + fold_height/2:.2f}" x2="{width-right}" y2="{fold_top + fold_height/2:.2f}" stroke="#111827"/>',
        )
    )
    bar_width = plot_width / max(1, len(successful)) * 0.65
    for offset, (fold, value) in enumerate(zip(successful, fold_values, strict=True)):
        center = left + (offset + 0.5) / max(1, len(successful)) * plot_width
        zero = fold_top + fold_height / 2.0
        value_height = abs(value) / fold_bound * fold_height / 2.0
        y = zero - value_height if value >= 0.0 else zero
        color = "#0f766e" if value >= 0.0 else "#b91c1c"
        lines.append(
            f'<rect x="{center-bar_width/2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{max(1.0,value_height):.2f}" fill="{color}"/>'
        )
        lines.append(
            f'<text class="axis" x="{center:.2f}" y="{fold_top+fold_height+18}" text-anchor="middle">F{fold["fold"]}</text>'
        )
    lines.append("</svg>")
    return ("\n".join(lines) + "\n").encode("utf-8")


def evaluate_prequential_microstructure_model(
    artifact: MicrostructureModelArtifact,
    dataset: MicrostructureDataset,
    *,
    config: PrequentialConfig | None = None,
    compute_backend: str = "auto",
    predictions_path: str | Path = "data/microstructure-prequential-predictions.csv",
    chart_path: str | Path = "data/microstructure-prequential.svg",
    report_path: str | Path = "data/microstructure-prequential.json",
    progress: PrequentialProgressCallback | None = None,
) -> PrequentialEvaluationReport:
    """Evaluate repeated fixed-hyperparameter refits before the terminal holdout."""

    cfg = (config or PrequentialConfig()).validated()
    source = _validate_candidate_contract(artifact, dataset)
    plans = plan_prequential_folds(artifact, dataset, cfg)
    all_planned = plan_prequential_folds(
        artifact,
        dataset,
        PrequentialConfig(**{**cfg.asdict(), "max_folds": 0}),
    )
    candidate_sha = microstructure_candidate_sha256(artifact)
    fold_reports: list[dict[str, object]] = []
    complete_indexes: list[np.ndarray] = []
    complete_predictions: list[_PredictionBatch] = []
    complete_fold_ids: list[np.ndarray] = []
    complete_policies: list[ThresholdPolicy] = []
    backend_kinds: set[str] = set()
    backend_devices: set[str] = set()
    total_steps = len(plans)
    for offset, plan in enumerate(plans, start=1):
        if progress:
            progress(f"fold-{plan.fold}-start", offset - 1, total_steps)
        started = time.perf_counter()
        base_report = plan.summary()
        try:
            fitted = _fit_fold_models(
                artifact,
                dataset,
                plan,
                cfg,
                compute_backend=compute_backend,
                seed=int(artifact.seed) + plan.fold * 101,
            )
            backend_kinds.add(fitted.backend_kind)
            backend_devices.add(fitted.backend_device)
            policy_predictions = _predict_models(
                artifact,
                dataset,
                fitted,
                plan.policy_indexes,
            )
            policy, policy_metrics, policy_search = _select_threshold(
                risk_level=artifact.risk_level,
                timestamps=dataset.decision_time_ms[plan.policy_indexes],
                long_exit_times=dataset.long_exit_time_ms[plan.policy_indexes],
                short_exit_times=dataset.short_exit_time_ms[plan.policy_indexes],
                long_targets=dataset.long_net_bps[plan.policy_indexes],
                short_targets=dataset.short_net_bps[plan.policy_indexes],
                long_edge=policy_predictions.long_edge,
                short_edge=policy_predictions.short_edge,
                long_probability=policy_predictions.long_probability,
                short_probability=policy_predictions.short_probability,
                long_eligible=dataset.long_liquidity_eligible[plan.policy_indexes],
                short_eligible=dataset.short_liquidity_eligible[plan.policy_indexes],
            )
            evaluation_predictions = _predict_models(
                artifact,
                dataset,
                fitted,
                plan.evaluation_indexes,
            )
            evaluation_trace = _simulate_non_overlapping_trace(
                timestamps=dataset.decision_time_ms[plan.evaluation_indexes],
                long_exit_times=dataset.long_exit_time_ms[plan.evaluation_indexes],
                short_exit_times=dataset.short_exit_time_ms[plan.evaluation_indexes],
                long_targets=dataset.long_net_bps[plan.evaluation_indexes],
                short_targets=dataset.short_net_bps[plan.evaluation_indexes],
                long_edge=evaluation_predictions.long_edge,
                short_edge=evaluation_predictions.short_edge,
                long_probability=evaluation_predictions.long_probability,
                short_probability=evaluation_predictions.short_probability,
                edge_threshold=policy.minimum_predicted_edge_bps,
                probability_threshold=policy.minimum_profitable_probability,
                long_eligible=dataset.long_liquidity_eligible[plan.evaluation_indexes],
                short_eligible=dataset.short_liquidity_eligible[plan.evaluation_indexes],
            )
            auc, brier = _side_probability_quality(
                dataset,
                plan.evaluation_indexes,
                evaluation_predictions.long_probability,
                evaluation_predictions.short_probability,
            )
            fold_reports.append(
                {
                    **base_report,
                    "status": "complete",
                    "error": None,
                    "backend_kind": fitted.backend_kind,
                    "backend_device": fitted.backend_device,
                    "model_sha256": fitted.model_sha256,
                    "side_training_rows": dict(fitted.side_training_rows),
                    "side_calibration_rows": dict(fitted.side_calibration_rows),
                    "probability_calibration": dict(fitted.calibration),
                    "policy": asdict(policy),
                    "policy_metrics": asdict(policy_metrics),
                    "policy_search": asdict(policy_search),
                    "evaluation_metrics_local_reset": asdict(
                        evaluation_trace.metrics
                    ),
                    "evaluation_auc": auc,
                    "evaluation_brier": brier,
                    "seconds": float(time.perf_counter() - started),
                }
            )
            complete_indexes.append(plan.evaluation_indexes)
            complete_predictions.append(evaluation_predictions)
            complete_fold_ids.append(
                np.full(len(plan.evaluation_indexes), plan.fold, dtype=np.int32)
            )
            complete_policies.extend([policy] * len(plan.evaluation_indexes))
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            fold_reports.append(
                {
                    **base_report,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": float(time.perf_counter() - started),
                }
            )
        gc.collect()
        if progress:
            progress(f"fold-{plan.fold}-complete", offset, total_steps)

    reasons: list[str] = []
    if cfg.asdict() != PROMOTION_PREQUENTIAL_CONFIG.asdict():
        reasons.append("prequential_protocol_deviates_from_locked_promotion_config")
    failed = [fold for fold in fold_reports if fold["status"] != "complete"]
    if failed:
        reasons.append(f"prequential_fold_failures:{len(failed)}")
    if len(plans) != len(all_planned):
        reasons.append("max_folds_truncated_selection_coverage")
    if not complete_indexes:
        reasons.append("no_complete_prequential_folds")
        raise RuntimeError("prequential evaluation produced no complete folds")
    indexes = np.concatenate(complete_indexes)
    fold_ids = np.concatenate(complete_fold_ids)
    predictions = _PredictionBatch(
        long_edge=np.concatenate([item.long_edge for item in complete_predictions]),
        short_edge=np.concatenate([item.short_edge for item in complete_predictions]),
        long_probability=np.concatenate(
            [item.long_probability for item in complete_predictions]
        ),
        short_probability=np.concatenate(
            [item.short_probability for item in complete_predictions]
        ),
    )
    order = np.argsort(dataset.decision_time_ms[indexes], kind="stable")
    indexes = indexes[order]
    fold_ids = fold_ids[order]
    predictions = _PredictionBatch(
        long_edge=predictions.long_edge[order],
        short_edge=predictions.short_edge[order],
        long_probability=predictions.long_probability[order],
        short_probability=predictions.short_probability[order],
    )
    policies = [complete_policies[int(value)] for value in order]
    edge_thresholds = np.asarray(
        [item.minimum_predicted_edge_bps for item in policies], dtype=np.float64
    )
    probability_thresholds = np.asarray(
        [item.minimum_profitable_probability for item in policies], dtype=np.float64
    )
    trace = _variable_policy_trace(
        dataset,
        indexes,
        predictions,
        edge_thresholds,
        probability_thresholds,
    )
    for fold in fold_reports:
        if fold["status"] != "complete":
            continue
        fold_mask = fold_ids == int(fold["fold"])
        executed_mask = fold_mask & trace.executed
        executed_indexes = indexes[executed_mask]
        executed_sides = trace.selected_side[executed_mask]
        executed_pnls = np.where(
            executed_sides > 0,
            dataset.long_net_bps[executed_indexes],
            dataset.short_net_bps[executed_indexes],
        )
        fold["evaluation_metrics"] = asdict(
            _trading_metrics(
                executed_pnls.tolist(),
                executed_sides.tolist(),
                dataset.decision_time_ms[executed_indexes].tolist(),
            )
        )
        fold["cross_boundary_suppressed_actions"] = int(
            np.sum(fold_mask & (trace.selected_side != 0) & ~trace.executed)
        )
    confidence = _performance_confidence(
        trace,
        dataset.decision_time_ms[indexes],
        bootstrap_samples=cfg.bootstrap_samples,
    )
    auc, brier = _side_probability_quality(
        dataset,
        indexes,
        predictions.long_probability,
        predictions.short_probability,
    )
    baselines = _baseline_metrics(dataset, indexes)
    available_ms = np.maximum(dataset.long_exit_time_ms, dataset.short_exit_time_ms) + int(
        dataset.trade_feature_embargo_ms
    )
    expected_indexes = np.flatnonzero(
        (dataset.decision_time_ms >= artifact.split.selection_start_ms)
        & (dataset.decision_time_ms < artifact.split.terminal_start_ms)
        & (available_ms < artifact.split.terminal_start_ms)
    )
    unique_indexes = np.unique(indexes)
    matched_indexes = np.intersect1d(unique_indexes, expected_indexes, assume_unique=True)
    missing_indexes = np.setdiff1d(expected_indexes, unique_indexes, assume_unique=True)
    extra_indexes = np.setdiff1d(unique_indexes, expected_indexes, assume_unique=True)
    coverage_ratio = len(matched_indexes) / max(1, len(expected_indexes))
    if len(unique_indexes) != len(indexes):
        reasons.append("duplicate_prequential_evaluation_rows")
    if len(extra_indexes):
        reasons.append("prequential_evaluation_contains_out_of_contract_rows")
    if len(missing_indexes) or not np.array_equal(indexes, expected_indexes):
        reasons.append("prequential_selection_coverage_is_not_complete")
    if len(fold_reports) < 3:
        reasons.append("fewer_than_three_prequential_folds")
    if len(backend_kinds) != 1 or len(backend_devices) != 1:
        reasons.append("prequential_backend_identity_changed")
    elif (
        backend_kinds != {artifact.training_backend_kind}
        or backend_devices != {artifact.training_backend_device}
    ):
        reasons.append("prequential_backend_differs_from_candidate_training")
    minimum_trades = _minimum_evaluation_trades(dataset.decision_time_ms[indexes])
    if trace.metrics.trades < minimum_trades:
        reasons.append("prequential_trade_count_below_statistical_minimum")
    if trace.metrics.total_net_bps <= 0.0 or _risk_utility(
        trace.metrics, artifact.risk_level
    ) <= 0.0:
        reasons.append("prequential_not_profitable_after_drawdown_penalty")
    if trace.metrics.profit_factor is None or trace.metrics.profit_factor <= 1.0:
        reasons.append("prequential_profit_factor_not_above_one")
    if confidence.mean_daily_net_bps_ci_lower <= 0.0:
        reasons.append("prequential_daily_edge_lower_confidence_bound_not_positive")
    strongest_baseline = max(item.total_net_bps for item in baselines.values())
    if trace.metrics.total_net_bps <= strongest_baseline:
        reasons.append("prequential_not_above_directional_baselines")
    if trace.metrics.long_trades and float(auc["long"]) <= 0.5:
        reasons.append("prequential_long_action_auc_not_above_random")
    if trace.metrics.short_trades and float(auc["short"]) <= 0.5:
        reasons.append("prequential_short_action_auc_not_above_random")
    complete_folds = [fold for fold in fold_reports if fold["status"] == "complete"]
    profitable_folds = sum(
        float(fold["evaluation_metrics"]["total_net_bps"]) > 0.0
        for fold in complete_folds
    )
    profitable_fold_ratio = profitable_folds / max(1, len(complete_folds))
    minimum_profitable_ratio = {
        "conservative": 0.70,
        "regular": 0.60,
        "aggressive": 0.50,
    }[artifact.risk_level]
    if profitable_fold_ratio < minimum_profitable_ratio:
        reasons.append("prequential_profitable_fold_ratio_below_risk_gate")
    if (
        artifact.selection_confidence.mean_daily_net_bps > 0.0
        and confidence.mean_daily_net_bps
        < artifact.selection_confidence.mean_daily_net_bps
    ):
        reasons.append("prequential_mean_daily_edge_below_static_selection")

    prediction_rows = _prediction_rows(
        dataset,
        fold_ids,
        indexes,
        predictions,
        policies,
        trace,
    )
    prediction_target = Path(predictions_path)
    prediction_sha = _write_predictions(prediction_target, prediction_rows)
    chart_target = Path(chart_path)
    chart_payload = _render_chart(prediction_rows, fold_reports)
    _write_bytes_atomic(chart_target, chart_payload)
    chart_sha = hashlib.sha256(chart_payload).hexdigest()
    protocol_sha = prequential_protocol_sha256(cfg)
    fold_models_payload = [
        {
            "fold": int(fold["fold"]),
            "model_sha256": str(fold["model_sha256"]),
        }
        for fold in complete_folds
    ]
    fold_models_sha = hashlib.sha256(
        json.dumps(
            fold_models_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    aggregate = {
        "metrics": asdict(trace.metrics),
        "confidence": asdict(confidence),
        "auc": auc,
        "brier": brier,
        "baselines": {name: asdict(value) for name, value in baselines.items()},
        "minimum_required_trades": minimum_trades,
        "profitable_folds": profitable_folds,
        "complete_folds": len(complete_folds),
        "profitable_fold_ratio": profitable_fold_ratio,
        "minimum_profitable_fold_ratio": minimum_profitable_ratio,
        "static_selection_mean_daily_net_bps": (
            artifact.selection_confidence.mean_daily_net_bps
        ),
        "prequential_minus_static_mean_daily_net_bps": (
            confidence.mean_daily_net_bps
            - artifact.selection_confidence.mean_daily_net_bps
        ),
        "fold_models_sha256": fold_models_sha,
    }
    report = PrequentialEvaluationReport(
        version=PREQUENTIAL_EVIDENCE_VERSION,
        generated_at_ms=int(time.time() * 1_000),
        status="passed" if not reasons else "rejected",
        trading_authority=False,
        reasons=tuple(reasons),
        candidate_sha256=candidate_sha,
        symbol=artifact.symbol,
        risk_level=artifact.risk_level,
        config=cfg.asdict(),
        data_contract={
            "source_feature_build_id": source["build_id"],
            "source_manifest_fingerprint": source["manifest_fingerprint"],
            "feature_version": dataset.feature_version,
            "feature_names": list(dataset.feature_names),
            "target_mode": dataset.target_mode,
            "selection_start_ms": artifact.split.selection_start_ms,
            "terminal_start_ms": artifact.split.terminal_start_ms,
            "terminal_holdout": "not accessed",
            "refit_mode": "rolling_full_retrain_fixed_hyperparameters",
            "lightgbm_leaf_only_refit_used": False,
            "orders_allowed": False,
            "protocol_sha256": protocol_sha,
        },
        coverage={
            "planned_folds": len(all_planned),
            "executed_folds": len(plans),
            "complete_folds": len(complete_folds),
            "failed_folds": len(failed),
            "expected_selection_rows": len(expected_indexes),
            "evaluated_rows": len(indexes),
            "unique_evaluated_rows": len(unique_indexes),
            "matched_selection_rows": len(matched_indexes),
            "missing_selection_rows": len(missing_indexes),
            "out_of_contract_rows": len(extra_indexes),
            "selection_coverage_ratio": coverage_ratio,
            "first_evaluation_ms": int(dataset.decision_time_ms[indexes[0]]),
            "last_evaluation_ms": int(dataset.decision_time_ms[indexes[-1]]),
        },
        folds=tuple(fold_reports),
        aggregate=aggregate,
        predictions_path=str(prediction_target),
        predictions_sha256=prediction_sha,
        chart_path=str(chart_target),
        chart_sha256=chart_sha,
    )
    write_json_atomic(Path(report_path), report.asdict(), sort_keys=True)
    return report


_PREDICTION_FIELDS = (
    "fold",
    "symbol",
    "decision_time_ms",
    "decision_time_utc",
    "long_expected_net_bps",
    "short_expected_net_bps",
    "long_profitable_probability",
    "short_profitable_probability",
    "minimum_predicted_edge_bps",
    "minimum_profitable_probability",
    "selected_side",
    "executed_nonoverlap",
    "realized_selected_net_bps",
    "selected_exit_time_ms",
)


def _finite_csv_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"prequential predictions contain invalid {label}") from exc
    if not math.isfinite(result):
        raise ValueError(f"prequential predictions contain non-finite {label}")
    return result


def _csv_bool(value: object, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"prequential predictions contain invalid {label}")


def _assert_record_matches(
    actual: object,
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError(f"prequential report is missing {label}")
    for key, expected_value in expected.items():
        if key not in actual:
            raise ValueError(f"prequential report {label} is missing {key}")
        actual_value = actual[key]
        if isinstance(expected_value, float):
            if actual_value is None:
                raise ValueError(f"prequential report {label}.{key} is null")
            observed = _finite_csv_float(actual_value, f"{label}.{key}")
            if not math.isclose(
                observed,
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(f"prequential report {label}.{key} drifted")
        elif expected_value is None:
            if actual_value is not None:
                raise ValueError(f"prequential report {label}.{key} drifted")
        elif actual_value != expected_value:
            raise ValueError(f"prequential report {label}.{key} drifted")


def _resolve_evidence_path(
    report_path: Path,
    recorded: object,
    explicit: str | Path | None,
    *,
    label: str,
) -> Path:
    if explicit is not None:
        target = Path(explicit)
    else:
        if not isinstance(recorded, str) or not recorded.strip():
            raise ValueError(f"prequential report is missing its {label} path")
        target = Path(recorded)
        if not target.exists() and not target.is_absolute():
            sibling = report_path.parent / target.name
            if sibling.exists():
                target = sibling
    if not target.is_file():
        raise ValueError(f"prequential {label} artifact does not exist")
    return target


def attach_verified_prequential_evidence(
    artifact: MicrostructureModelArtifact,
    dataset: MicrostructureDataset,
    *,
    report_path: str | Path,
    predictions_path: str | Path | None = None,
    chart_path: str | Path | None = None,
) -> MicrostructureModelArtifact:
    """Verify persisted prequential evidence and bind it to the exact candidate."""

    source = _validate_candidate_contract(artifact, dataset)
    if artifact.prequential_validation is not None:
        raise ValueError("candidate already has attached prequential evidence")
    report_target = Path(report_path)
    try:
        report_bytes = report_target.read_bytes()
        report_text = report_bytes.decode("utf-8")
        payload = json.loads(report_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prequential report is not valid UTF-8 JSON") from exc
    if b"\r\n" in report_bytes or not report_bytes.endswith(b"\n"):
        raise ValueError("prequential report is not canonical LF-terminated JSON")
    if not isinstance(payload, Mapping):
        raise ValueError("prequential report must be a JSON object")
    canonical_report = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if report_bytes != canonical_report:
        raise ValueError("prequential report is not canonical deterministic JSON")
    if (
        payload.get("version") != PREQUENTIAL_EVIDENCE_VERSION
        or payload.get("status") != "passed"
        or payload.get("passed") is not True
        or payload.get("trading_authority") is not False
        or payload.get("reasons") != []
    ):
        raise ValueError("prequential report did not pass its locked no-order protocol")
    candidate_sha = microstructure_candidate_sha256(artifact)
    if payload.get("candidate_sha256") != candidate_sha:
        raise ValueError("prequential report belongs to a different candidate")
    if payload.get("symbol") != artifact.symbol or payload.get("risk_level") != artifact.risk_level:
        raise ValueError("prequential report symbol or risk profile drifted")
    generated_at_ms = int(payload.get("generated_at_ms") or 0)
    try:
        trained_at_ms = int(
            datetime.fromisoformat(artifact.trained_at.replace("Z", "+00:00")).timestamp()
            * 1_000
        )
    except ValueError as exc:
        raise ValueError("candidate training time is invalid") from exc
    if (
        generated_at_ms < trained_at_ms
        or generated_at_ms > int(time.time() * 1_000) + 300_000
    ):
        raise ValueError("prequential report generation time is invalid")
    config_payload = payload.get("config")
    locked_config = PROMOTION_PREQUENTIAL_CONFIG.validated()
    if config_payload != locked_config.asdict():
        raise ValueError("prequential report did not use the locked promotion protocol")
    protocol_sha = prequential_protocol_sha256(locked_config)
    data_contract = payload.get("data_contract")
    if not isinstance(data_contract, Mapping):
        raise ValueError("prequential report is missing its data contract")
    if (
        data_contract.get("source_feature_build_id") != source["build_id"]
        or data_contract.get("source_manifest_fingerprint")
        != source["manifest_fingerprint"]
        or data_contract.get("feature_version") != dataset.feature_version
        or tuple(data_contract.get("feature_names") or ()) != dataset.feature_names
        or data_contract.get("target_mode") != dataset.target_mode
        or data_contract.get("selection_start_ms") != artifact.split.selection_start_ms
        or data_contract.get("terminal_start_ms") != artifact.split.terminal_start_ms
        or data_contract.get("terminal_holdout") != "not accessed"
        or data_contract.get("refit_mode")
        != "rolling_full_retrain_fixed_hyperparameters"
        or data_contract.get("lightgbm_leaf_only_refit_used") is not False
        or data_contract.get("orders_allowed") is not False
        or data_contract.get("protocol_sha256") != protocol_sha
    ):
        raise ValueError("prequential report data contract drifted from the candidate")

    available_ms = np.maximum(dataset.long_exit_time_ms, dataset.short_exit_time_ms) + int(
        dataset.trade_feature_embargo_ms
    )
    expected_indexes = np.flatnonzero(
        (dataset.decision_time_ms >= artifact.split.selection_start_ms)
        & (dataset.decision_time_ms < artifact.split.terminal_start_ms)
        & (available_ms < artifact.split.terminal_start_ms)
    ).astype(np.int64)
    if not len(expected_indexes):
        raise ValueError("prequential candidate has no expected selection rows")
    expected_plans = plan_prequential_folds(artifact, dataset, locked_config)
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("prequential report is missing coverage evidence")
    planned_folds = int(coverage.get("planned_folds") or 0)
    if (
        planned_folds < 3
        or planned_folds != len(expected_plans)
        or int(coverage.get("executed_folds") or 0) != planned_folds
        or int(coverage.get("complete_folds") or 0) != planned_folds
        or int(coverage.get("failed_folds") or 0) != 0
        or int(coverage.get("expected_selection_rows") or 0) != len(expected_indexes)
        or int(coverage.get("evaluated_rows") or 0) != len(expected_indexes)
        or int(coverage.get("unique_evaluated_rows") or 0) != len(expected_indexes)
        or int(coverage.get("matched_selection_rows") or 0) != len(expected_indexes)
        or coverage.get("missing_selection_rows") != 0
        or coverage.get("out_of_contract_rows") != 0
        or float(coverage.get("selection_coverage_ratio") or 0.0) != 1.0
        or int(coverage.get("first_evaluation_ms") or 0)
        != int(dataset.decision_time_ms[expected_indexes[0]])
        or int(coverage.get("last_evaluation_ms") or 0)
        != int(dataset.decision_time_ms[expected_indexes[-1]])
    ):
        raise ValueError("prequential report coverage is incomplete or inconsistent")

    raw_folds = payload.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != planned_folds:
        raise ValueError("prequential report fold evidence is incomplete")
    folds: list[Mapping[str, object]] = []
    prior_end = int(artifact.split.selection_start_ms)
    fold_models: list[dict[str, object]] = []
    for expected_fold, (raw_fold, expected_plan) in enumerate(
        zip(raw_folds, expected_plans, strict=True),
        start=1,
    ):
        if not isinstance(raw_fold, Mapping):
            raise ValueError("prequential report contains an invalid fold")
        model_sha = str(raw_fold.get("model_sha256") or "")
        start_ms = int(raw_fold.get("evaluation_start_ms") or 0)
        end_ms = int(raw_fold.get("evaluation_end_exclusive_ms") or 0)
        if (
            int(raw_fold.get("fold") or 0) != expected_fold
            or raw_fold.get("status") != "complete"
            or raw_fold.get("error") is not None
            or raw_fold.get("backend_kind") != artifact.training_backend_kind
            or raw_fold.get("backend_device") != artifact.training_backend_device
            or start_ms != prior_end
            or end_ms <= start_ms
            or end_ms > artifact.split.terminal_start_ms
            or len(model_sha) != 64
            or any(char not in "0123456789abcdef" for char in model_sha)
        ):
            raise ValueError("prequential report fold contract is invalid")
        expected_summary = expected_plan.summary()
        if any(raw_fold.get(key) != value for key, value in expected_summary.items()):
            raise ValueError("prequential report fold plan does not reproduce")
        expected_fold_rows = int(
            np.sum(
                (dataset.decision_time_ms[expected_indexes] >= start_ms)
                & (dataset.decision_time_ms[expected_indexes] < end_ms)
            )
        )
        if int(raw_fold.get("evaluation_rows") or 0) != expected_fold_rows:
            raise ValueError("prequential report fold row count drifted")
        policy = raw_fold.get("policy")
        if not isinstance(policy, Mapping):
            raise ValueError("prequential report fold is missing its policy")
        folds.append(raw_fold)
        fold_models.append({"fold": expected_fold, "model_sha256": model_sha})
        prior_end = end_ms
    if prior_end != artifact.split.terminal_start_ms:
        raise ValueError("prequential folds do not cover the complete selection interval")
    fold_models_sha = hashlib.sha256(
        json.dumps(
            fold_models,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    prediction_target = _resolve_evidence_path(
        report_target,
        payload.get("predictions_path"),
        predictions_path,
        label="predictions",
    )
    prediction_sha = _canonical_lf_sha256(prediction_target)
    if prediction_sha != payload.get("predictions_sha256"):
        raise ValueError("prequential prediction artifact hash drifted")
    chart_target = _resolve_evidence_path(
        report_target,
        payload.get("chart_path"),
        chart_path,
        label="chart",
    )
    chart_bytes = chart_target.read_bytes()
    chart_sha = hashlib.sha256(chart_bytes).hexdigest()
    if (
        chart_sha != payload.get("chart_sha256")
        or b"\r\n" in chart_bytes
        or b"terminal holdout sealed" not in chart_bytes
    ):
        raise ValueError("prequential chart artifact is invalid or drifted")

    long_probability = np.empty(len(expected_indexes), dtype=np.float64)
    short_probability = np.empty(len(expected_indexes), dtype=np.float64)
    global_pnls: list[float] = []
    global_sides: list[int] = []
    global_times: list[int] = []
    fold_global: dict[int, tuple[list[float], list[int], list[int]]] = {
        number: ([], [], []) for number in range(1, planned_folds + 1)
    }
    fold_local: dict[int, tuple[list[float], list[int], list[int]]] = {
        number: ([], [], []) for number in range(1, planned_folds + 1)
    }
    fold_local_next = {number: -1 for number in range(1, planned_folds + 1)}
    fold_suppressed = {number: 0 for number in range(1, planned_folds + 1)}
    global_next = -1
    fold_cursor = 0
    with prediction_target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _PREDICTION_FIELDS:
            raise ValueError("prequential prediction columns drifted")
        for offset, row in enumerate(reader):
            if offset >= len(expected_indexes):
                raise ValueError("prequential predictions contain extra rows")
            index = int(expected_indexes[offset])
            timestamp = int(dataset.decision_time_ms[index])
            while timestamp >= int(folds[fold_cursor]["evaluation_end_exclusive_ms"]):
                fold_cursor += 1
                if fold_cursor >= len(folds):
                    raise ValueError("prequential prediction timestamp is outside all folds")
            fold = folds[fold_cursor]
            fold_number = int(fold["fold"])
            if (
                int(row["fold"]) != fold_number
                or row["symbol"] != dataset.symbol
                or int(row["decision_time_ms"]) != timestamp
                or row["decision_time_utc"] != _iso_utc(timestamp)
            ):
                raise ValueError("prequential prediction identity or ordering drifted")
            long_edge = _finite_csv_float(
                row["long_expected_net_bps"], "long expected net bps"
            )
            short_edge = _finite_csv_float(
                row["short_expected_net_bps"], "short expected net bps"
            )
            long_prob = _finite_csv_float(
                row["long_profitable_probability"], "long probability"
            )
            short_prob = _finite_csv_float(
                row["short_profitable_probability"], "short probability"
            )
            edge_threshold = _finite_csv_float(
                row["minimum_predicted_edge_bps"], "edge threshold"
            )
            probability_threshold = _finite_csv_float(
                row["minimum_profitable_probability"], "probability threshold"
            )
            policy = fold["policy"]
            if (
                not 0.0 <= long_prob <= 1.0
                or not 0.0 <= short_prob <= 1.0
                or edge_threshold < 0.0
                or not 0.0 <= probability_threshold <= 1.0
                or not math.isclose(
                    edge_threshold,
                    float(policy["minimum_predicted_edge_bps"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    probability_threshold,
                    float(policy["minimum_profitable_probability"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("prequential prediction policy or probability drifted")
            long_probability[offset] = long_prob
            short_probability[offset] = short_prob
            long_ok = (
                bool(dataset.long_liquidity_eligible[index])
                and long_edge >= edge_threshold
                and long_prob >= probability_threshold
            )
            short_ok = (
                bool(dataset.short_liquidity_eligible[index])
                and short_edge >= edge_threshold
                and short_prob >= probability_threshold
            )
            selected_side = (
                0
                if not long_ok and not short_ok
                else (
                    1
                    if long_ok and (not short_ok or long_edge >= short_edge)
                    else -1
                )
            )
            if int(row["selected_side"]) != selected_side:
                raise ValueError("prequential selected action does not reproduce")
            expected_exit = (
                int(dataset.long_exit_time_ms[index])
                if selected_side > 0
                else (
                    int(dataset.short_exit_time_ms[index])
                    if selected_side < 0
                    else timestamp
                )
            )
            expected_pnl = (
                float(dataset.long_net_bps[index])
                if selected_side > 0
                else (
                    float(dataset.short_net_bps[index])
                    if selected_side < 0
                    else 0.0
                )
            )
            global_executed = selected_side != 0 and timestamp >= global_next
            if _csv_bool(row["executed_nonoverlap"], "execution flag") != global_executed:
                raise ValueError("prequential non-overlap execution does not reproduce")
            if (
                int(row["selected_exit_time_ms"]) != expected_exit
                or not math.isclose(
                    _finite_csv_float(row["realized_selected_net_bps"], "realized net bps"),
                    expected_pnl,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError("prequential realized target does not match the dataset")
            if global_executed:
                global_pnls.append(expected_pnl)
                global_sides.append(selected_side)
                global_times.append(timestamp)
                fold_global[fold_number][0].append(expected_pnl)
                fold_global[fold_number][1].append(selected_side)
                fold_global[fold_number][2].append(timestamp)
                global_next = expected_exit
            elif selected_side != 0:
                fold_suppressed[fold_number] += 1
            if selected_side != 0 and timestamp >= fold_local_next[fold_number]:
                fold_local[fold_number][0].append(expected_pnl)
                fold_local[fold_number][1].append(selected_side)
                fold_local[fold_number][2].append(timestamp)
                fold_local_next[fold_number] = expected_exit
        observed_rows = int(reader.line_num - 1)
    if observed_rows != len(expected_indexes):
        raise ValueError("prequential prediction row count is incomplete")

    metrics = _trading_metrics(global_pnls, global_sides, global_times)
    trace = _VariablePolicyTrace(
        metrics=metrics,
        pnls=tuple(global_pnls),
        sides=tuple(global_sides),
        timestamps=tuple(global_times),
        selected_side=np.empty(0, dtype=np.int8),
        executed=np.empty(0, dtype=bool),
    )
    confidence = _performance_confidence(
        trace,
        dataset.decision_time_ms[expected_indexes],
        bootstrap_samples=locked_config.bootstrap_samples,
    )
    auc, brier = _side_probability_quality(
        dataset,
        expected_indexes,
        long_probability,
        short_probability,
    )
    baselines = _baseline_metrics(dataset, expected_indexes)
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("prequential report is missing aggregate financial evidence")
    _assert_record_matches(aggregate.get("metrics"), asdict(metrics), label="metrics")
    _assert_record_matches(
        aggregate.get("confidence"),
        asdict(confidence),
        label="confidence",
    )
    _assert_record_matches(aggregate.get("auc"), auc, label="AUC")
    _assert_record_matches(aggregate.get("brier"), brier, label="Brier")
    raw_baselines = aggregate.get("baselines")
    if not isinstance(raw_baselines, Mapping) or set(raw_baselines) != set(baselines):
        raise ValueError("prequential report baseline set drifted")
    for name, baseline in baselines.items():
        _assert_record_matches(
            raw_baselines[name],
            asdict(baseline),
            label=f"baseline {name}",
        )
    if aggregate.get("fold_models_sha256") != fold_models_sha:
        raise ValueError("prequential fold-model fingerprint drifted")
    for fold in folds:
        fold_number = int(fold["fold"])
        global_fold_metrics = _trading_metrics(*fold_global[fold_number])
        local_fold_metrics = _trading_metrics(*fold_local[fold_number])
        _assert_record_matches(
            fold.get("evaluation_metrics"),
            asdict(global_fold_metrics),
            label=f"fold {fold_number} global metrics",
        )
        _assert_record_matches(
            fold.get("evaluation_metrics_local_reset"),
            asdict(local_fold_metrics),
            label=f"fold {fold_number} local-reset metrics",
        )
        if int(fold.get("cross_boundary_suppressed_actions") or 0) != fold_suppressed[
            fold_number
        ]:
            raise ValueError("prequential fold suppression evidence drifted")
        fold_mask = np.flatnonzero(
            (dataset.decision_time_ms[expected_indexes] >= int(fold["evaluation_start_ms"]))
            & (
                dataset.decision_time_ms[expected_indexes]
                < int(fold["evaluation_end_exclusive_ms"])
            )
        )
        fold_auc, fold_brier = _side_probability_quality(
            dataset,
            expected_indexes[fold_mask],
            long_probability[fold_mask],
            short_probability[fold_mask],
        )
        _assert_record_matches(
            fold.get("evaluation_auc"), fold_auc, label=f"fold {fold_number} AUC"
        )
        _assert_record_matches(
            fold.get("evaluation_brier"),
            fold_brier,
            label=f"fold {fold_number} Brier",
        )

    profit_factor = metrics.profit_factor
    minimum_trades = _minimum_evaluation_trades(
        dataset.decision_time_ms[expected_indexes]
    )
    profitable_folds = sum(
        _trading_metrics(*fold_global[number]).total_net_bps > 0.0
        for number in fold_global
    )
    profitable_ratio = profitable_folds / planned_folds
    minimum_profitable_ratio = {
        "conservative": 0.70,
        "regular": 0.60,
        "aggressive": 0.50,
    }[artifact.risk_level]
    if (
        int(aggregate.get("minimum_required_trades") or 0) != minimum_trades
        or int(aggregate.get("profitable_folds") or -1) != profitable_folds
        or int(aggregate.get("complete_folds") or 0) != planned_folds
        or not math.isclose(
            float(aggregate.get("profitable_fold_ratio") or 0.0),
            profitable_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(aggregate.get("minimum_profitable_fold_ratio") or 0.0),
            minimum_profitable_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(aggregate.get("static_selection_mean_daily_net_bps") or 0.0),
            artifact.selection_confidence.mean_daily_net_bps,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(aggregate.get("prequential_minus_static_mean_daily_net_bps") or 0.0),
            confidence.mean_daily_net_bps
            - artifact.selection_confidence.mean_daily_net_bps,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("prequential aggregate gate evidence drifted")
    if (
        metrics.trades < minimum_trades
        or metrics.total_net_bps <= 0.0
        or _risk_utility(metrics, artifact.risk_level) <= 0.0
        or profit_factor is None
        or profit_factor <= 1.0
        or confidence.mean_daily_net_bps_ci_lower <= 0.0
        or metrics.total_net_bps <= max(item.total_net_bps for item in baselines.values())
        or (metrics.long_trades > 0 and float(auc["long"]) <= 0.5)
        or (metrics.short_trades > 0 and float(auc["short"]) <= 0.5)
        or profitable_ratio < minimum_profitable_ratio
        or (
            artifact.selection_confidence.mean_daily_net_bps > 0.0
            and confidence.mean_daily_net_bps
            < artifact.selection_confidence.mean_daily_net_bps
        )
    ):
        raise ValueError("prequential financial promotion gates do not reproduce")
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    evidence = PrequentialValidationEvidence(
        version=PREQUENTIAL_EVIDENCE_VERSION,
        report_sha256=report_sha,
        predictions_sha256=prediction_sha,
        chart_sha256=chart_sha,
        candidate_sha256=candidate_sha,
        protocol_sha256=protocol_sha,
        fold_models_sha256=fold_models_sha,
        source_feature_build_id=str(source["build_id"]),
        source_manifest_fingerprint=str(source["manifest_fingerprint"]),
        generated_at_ms=generated_at_ms,
        planned_folds=planned_folds,
        complete_folds=planned_folds,
        evaluated_rows=len(expected_indexes),
        selection_coverage_ratio=1.0,
        total_net_bps=metrics.total_net_bps,
        profit_factor=float(profit_factor),
        max_drawdown_bps=metrics.max_drawdown_bps,
        mean_daily_net_bps_ci_lower=confidence.mean_daily_net_bps_ci_lower,
        attached_at=datetime.now(tz=UTC).isoformat(),
    )
    return replace(artifact, prequential_validation=evidence)


__all__ = [
    "PREQUENTIAL_EVIDENCE_VERSION",
    "PROMOTION_PREQUENTIAL_CONFIG",
    "PrequentialConfig",
    "PrequentialEvaluationReport",
    "PrequentialFoldPlan",
    "attach_verified_prequential_evidence",
    "evaluate_prequential_microstructure_model",
    "plan_prequential_folds",
    "prequential_protocol_sha256",
]
