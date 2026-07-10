"""Purged, cost-aware LightGBM research models for L1/tape day trading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from .compute import resolve_backend
from .microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_TRADE_EMBARGO_MS,
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
)
from .microstructure_warehouse import (
    BOOK_TICKER_FEATURE_BUILD_VERSION,
    TICK_WAREHOUSE_SCHEMA_VERSION,
)


MICROSTRUCTURE_MODEL_SCHEMA_VERSION = "microstructure-action-value-v11"
MICROSTRUCTURE_PREQUENTIAL_EVIDENCE_VERSION = (
    "microstructure-prequential-fixed-refit-v1"
)
_RISK_LEVELS = frozenset({"conservative", "regular", "aggressive"})
ModelProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class PurgedSplitEvidence:
    train_rows: int
    tuning_rows: int
    policy_rows: int
    selection_rows: int
    terminal_rows: int
    train_end_ms: int
    tuning_start_ms: int
    policy_start_ms: int
    selection_start_ms: int
    terminal_start_ms: int
    purge_ms: int
    purged_rows: int
    tuning_early_stop_rows: int
    tuning_calibration_rows: int
    tuning_calibration_start_ms: int
    tuning_internal_purged_rows: int


@dataclass(frozen=True)
class TradingMetrics:
    trades: int
    total_net_bps: float
    mean_net_bps: float
    median_net_bps: float
    win_rate: float
    profit_factor: float | None
    max_drawdown_bps: float
    worst_trade_bps: float
    best_trade_bps: float
    long_trades: int
    short_trades: int
    active_days: int
    trades_per_active_day: float


@dataclass(frozen=True)
class PerformanceConfidence:
    calendar_days: int
    active_days: int
    block_length_days: int
    bootstrap_samples: int
    mean_daily_net_bps: float
    median_daily_net_bps: float
    worst_daily_net_bps: float
    best_daily_net_bps: float
    annualized_daily_sharpe: float | None
    mean_daily_net_bps_ci_lower: float
    mean_daily_net_bps_ci_upper: float
    bootstrap_probability_mean_positive: float


@dataclass(frozen=True)
class ThresholdPolicy:
    minimum_predicted_edge_bps: float
    minimum_profitable_probability: float
    selection_utility_bps: float


@dataclass(frozen=True)
class ThresholdSearchEvidence:
    rows: int
    long_positive_edge_rows: int
    short_positive_edge_rows: int
    minimum_required_trades: int
    evaluated_policy_count: int
    policies_meeting_trade_minimum: int
    best_observed_utility_bps: float | None
    best_observed_metrics: TradingMetrics | None


@dataclass(frozen=True)
class DeploymentRefitEvidence:
    refit_mode: str
    backend_kind: str
    backend_device: str
    lightgbm_version: str
    training_rows: int
    calibration_days: int
    calibration_start_ms: int
    calibration_end_ms: int
    side_training_rows: Mapping[str, int]
    side_calibration_rows: Mapping[str, int]
    probability_calibration: Mapping[str, tuple[float, float]]
    training_cutoff_ms: int
    maximum_model_age_seconds: int
    expires_at_ms: int
    source_feature_build_id: str
    source_manifest_fingerprint: str
    validation_model_sha256: str
    deployment_model_sha256: str
    fitted_at: str


@dataclass(frozen=True)
class PrequentialValidationEvidence:
    version: str
    report_sha256: str
    predictions_sha256: str
    chart_sha256: str
    candidate_sha256: str
    protocol_sha256: str
    fold_models_sha256: str
    source_feature_build_id: str
    source_manifest_fingerprint: str
    generated_at_ms: int
    planned_folds: int
    complete_folds: int
    evaluated_rows: int
    selection_coverage_ratio: float
    total_net_bps: float
    profit_factor: float
    max_drawdown_bps: float
    mean_daily_net_bps_ci_lower: float
    attached_at: str


@dataclass(frozen=True)
class MicrostructureModelArtifact:
    schema_version: str
    model_family: str
    status: str
    rejection_reasons: tuple[str, ...]
    symbol: str
    feature_version: str
    feature_names: tuple[str, ...]
    risk_level: str
    horizon_seconds: int
    total_latency_ms: int
    taker_fee_bps: float
    reference_order_notional_quote: float
    max_l1_participation: float
    max_quote_age_ms: int
    decision_cadence_seconds: int
    target_mode: str
    stop_loss_bps: float | None
    take_profit_bps: float | None
    trigger_execution_slippage_bps: float | None
    path_resolution_ms: int | None
    training_backend_kind: str
    training_backend_device: str
    lightgbm_version: str
    seed: int
    unique_utc_days: int
    calendar_span_days: int
    calendar_day_coverage_ratio: float
    minimum_rows_per_utc_day: int
    daily_rows_p10: float
    median_rows_per_utc_day: float
    minimum_promotion_days: int
    deployment_calibration_days: int
    maximum_model_age_seconds: int
    split: PurgedSplitEvidence
    best_iterations: Mapping[str, int]
    probability_calibration: Mapping[str, tuple[float, float]]
    threshold_policy: ThresholdPolicy
    policy_search: ThresholdSearchEvidence
    tuning_auc: Mapping[str, float]
    tuning_brier: Mapping[str, float]
    selection_auc: Mapping[str, float]
    selection_brier: Mapping[str, float]
    terminal_auc: Mapping[str, float] | None
    terminal_brier: Mapping[str, float] | None
    policy_metrics: TradingMetrics
    selection_metrics: TradingMetrics
    terminal_metrics: TradingMetrics | None
    selection_confidence: PerformanceConfidence
    terminal_confidence: PerformanceConfidence | None
    selection_baselines: Mapping[str, TradingMetrics]
    terminal_baselines: Mapping[str, TradingMetrics] | None
    model_strings: Mapping[str, str]
    deployment_model_strings: Mapping[str, str] | None
    deployment_refit: DeploymentRefitEvidence | None
    dataset_summary: Mapping[str, object]
    trained_at: str
    terminal_evaluated_at: str | None
    prequential_validation: PrequentialValidationEvidence | None = None

    @property
    def promotion_eligible(self) -> bool:
        return self.status == "accepted"

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        payload["feature_names"] = list(self.feature_names)
        return payload


@dataclass(frozen=True)
class MicrostructureActionPrediction:
    side: str
    long_expected_net_bps: float
    short_expected_net_bps: float
    long_profitable_probability: float
    short_profitable_probability: float
    minimum_predicted_edge_bps: float
    minimum_profitable_probability: float
    long_l1_participation: float
    short_l1_participation: float
    reason: str


class MicrostructureModelExpiredError(RuntimeError):
    """The promoted estimator is older than its validated live-use contract."""


_RUNTIME_MODEL_NAMES = tuple(
    f"{side}_{component}"
    for side in ("long", "short")
    for component in ("probability", "win_magnitude", "loss_magnitude")
)


class MicrostructureActionScorer:
    """Validated inference-only view of one promoted microstructure artifact."""

    def __init__(
        self,
        *,
        artifact_sha256: str,
        symbol: str,
        risk_level: str,
        horizon_seconds: int,
        total_latency_ms: int,
        taker_fee_bps: float,
        reference_order_notional_quote: float,
        max_l1_participation: float,
        max_quote_age_ms: int,
        decision_cadence_seconds: int,
        stop_loss_bps: float,
        take_profit_bps: float,
        trigger_execution_slippage_bps: float,
        path_resolution_ms: int,
        minimum_predicted_edge_bps: float,
        minimum_profitable_probability: float,
        probability_calibration: Mapping[str, tuple[float, float]],
        models: Mapping[str, lgb.Booster],
        training_cutoff_ms: int | None = None,
        expires_at_ms: int | None = None,
        enforce_model_freshness: bool = False,
    ) -> None:
        self.artifact_sha256 = artifact_sha256
        self.symbol = symbol
        self.risk_level = risk_level
        self.horizon_seconds = horizon_seconds
        self.total_latency_ms = total_latency_ms
        self.taker_fee_bps = taker_fee_bps
        self.reference_order_notional_quote = reference_order_notional_quote
        self.max_l1_participation = max_l1_participation
        self.max_quote_age_ms = max_quote_age_ms
        self.decision_cadence_seconds = decision_cadence_seconds
        self.stop_loss_bps = stop_loss_bps
        self.take_profit_bps = take_profit_bps
        self.trigger_execution_slippage_bps = trigger_execution_slippage_bps
        self.path_resolution_ms = path_resolution_ms
        self.minimum_predicted_edge_bps = minimum_predicted_edge_bps
        self.minimum_profitable_probability = minimum_profitable_probability
        self.probability_calibration = dict(probability_calibration)
        self.models = dict(models)
        self.training_cutoff_ms = training_cutoff_ms
        self.expires_at_ms = expires_at_ms
        self.enforce_model_freshness = bool(enforce_model_freshness)

    def score(
        self,
        features: Sequence[float] | np.ndarray,
        *,
        decision_time_ms: int,
        order_notional_quote: float,
        close_bid: float,
        close_ask: float,
        close_bid_qty: float,
        close_ask_qty: float,
        quote_time_ms: int,
        observation_time_ms: int,
    ) -> MicrostructureActionPrediction:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.shape != (len(MICROSTRUCTURE_FEATURE_NAMES),):
            raise ValueError(
                "microstructure inference feature shape mismatch: "
                f"expected={len(MICROSTRUCTURE_FEATURE_NAMES)} actual={matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("microstructure inference features contain non-finite values")
        decision_time = int(decision_time_ms)
        if (
            decision_time < 0
            or decision_time % 1_000 != 0
            or (decision_time // 1_000) % self.decision_cadence_seconds != 0
        ):
            raise ValueError("microstructure inference timestamp violates decision cadence")
        if (
            self.enforce_model_freshness
            and self.expires_at_ms is not None
            and decision_time > self.expires_at_ms
        ):
            raise MicrostructureModelExpiredError(
                "microstructure deployment model expired at "
                f"{self.expires_at_ms}; decision={decision_time}"
            )
        liquidity_values = np.asarray(
            [order_notional_quote, close_bid, close_ask, close_bid_qty, close_ask_qty],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(liquidity_values)) or np.any(liquidity_values <= 0.0):
            raise ValueError("microstructure inference liquidity inputs must be finite and positive")
        if close_bid >= close_ask:
            raise ValueError("microstructure inference quote is crossed")
        quote_time = int(quote_time_ms)
        observation_time = int(observation_time_ms)
        quote_age_ms = observation_time - quote_time
        if (
            quote_time <= 0
            or observation_time < decision_time
            or quote_age_ms < 0
        ):
            raise ValueError("microstructure inference quote timing is invalid")
        long_participation = float(
            (float(order_notional_quote) / float(close_ask)) / float(close_ask_qty)
        )
        short_participation = float(
            (float(order_notional_quote) / float(close_bid)) / float(close_bid_qty)
        )
        if quote_age_ms > self.max_quote_age_ms:
            return MicrostructureActionPrediction(
                side="FLAT",
                long_expected_net_bps=0.0,
                short_expected_net_bps=0.0,
                long_profitable_probability=0.0,
                short_profitable_probability=0.0,
                minimum_predicted_edge_bps=self.minimum_predicted_edge_bps,
                minimum_profitable_probability=self.minimum_profitable_probability,
                long_l1_participation=long_participation,
                short_l1_participation=short_participation,
                reason="quote_age_exceeds_validated_limit",
            )
        within_validated_notional = (
            float(order_notional_quote) <= self.reference_order_notional_quote
        )
        batch = matrix.reshape(1, -1)
        values: dict[str, tuple[float, float]] = {}
        for side in ("long", "short"):
            raw_probability = float(self.models[f"{side}_probability"].predict(batch)[0])
            probability = float(
                _apply_platt_scaling(
                    np.asarray([raw_probability], dtype=np.float64),
                    self.probability_calibration[side],
                )[0]
            )
            win = max(0.0, float(self.models[f"{side}_win_magnitude"].predict(batch)[0]))
            loss = max(0.0, float(self.models[f"{side}_loss_magnitude"].predict(batch)[0]))
            expected_net = probability * win - (1.0 - probability) * loss
            if not all(math.isfinite(value) for value in (probability, win, loss, expected_net)):
                raise ValueError(f"{side} microstructure model emitted non-finite inference")
            values[side] = (expected_net, probability)

        long_edge, long_probability = values["long"]
        short_edge, short_probability = values["short"]
        long_eligible = (
            within_validated_notional
            and long_participation <= self.max_l1_participation
            and long_edge >= self.minimum_predicted_edge_bps
            and long_probability >= self.minimum_profitable_probability
        )
        short_eligible = (
            within_validated_notional
            and short_participation <= self.max_l1_participation
            and short_edge >= self.minimum_predicted_edge_bps
            and short_probability >= self.minimum_profitable_probability
        )
        if long_eligible and (not short_eligible or long_edge >= short_edge):
            side = "LONG"
            reason = "long_policy_pass"
        elif short_eligible:
            side = "SHORT"
            reason = "short_policy_pass"
        else:
            side = "FLAT"
            reason = (
                "order_notional_exceeds_validated_reference"
                if not within_validated_notional
                else "no_side_passed_policy_and_liquidity_gates"
            )
        return MicrostructureActionPrediction(
            side=side,
            long_expected_net_bps=long_edge,
            short_expected_net_bps=short_edge,
            long_profitable_probability=long_probability,
            short_profitable_probability=short_probability,
            minimum_predicted_edge_bps=self.minimum_predicted_edge_bps,
            minimum_profitable_probability=self.minimum_profitable_probability,
            long_l1_participation=long_participation,
            short_l1_participation=short_participation,
            reason=reason,
        )


def _artifact_float(value: object, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"microstructure model {label} is invalid") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"microstructure model {label} is invalid")
    return parsed


def _artifact_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"microstructure model {label} is invalid")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"microstructure model {label} is invalid") from exc


def _model_strings_sha256(models: Mapping[str, object]) -> str:
    payload: dict[str, str] = {}
    for name in _RUNTIME_MODEL_NAMES:
        value = models.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"microstructure model is missing estimator {name}")
        payload[name] = value
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validated_source_evidence(
    value: object,
    *,
    symbol: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("microstructure model is missing source provenance")
    if (
        value.get("verified") is not True
        or value.get("manifest_current") is not True
        or value.get("status") != "complete"
        or value.get("is_current") is not True
    ):
        raise ValueError("microstructure model source provenance was not verified at training time")
    if value.get("schema_version") != TICK_WAREHOUSE_SCHEMA_VERSION:
        raise ValueError("microstructure model source warehouse schema is not supported")
    if value.get("feature_build_version") != BOOK_TICKER_FEATURE_BUILD_VERSION:
        raise ValueError("microstructure model source aggregation contract is not supported")
    if value.get("availability_clock") != "event_time_ms":
        raise ValueError("microstructure model source does not use the availability clock")
    if str(value.get("symbol") or "").upper() != symbol:
        raise ValueError("microstructure model source symbol does not match the artifact")
    for key in ("build_id", "manifest_fingerprint"):
        digest = str(value.get(key) or "").lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"microstructure model source {key} is invalid")
    archive_count = _artifact_int(value.get("source_archive_count"), "source archive count")
    manifest_rows = _artifact_int(value.get("source_manifest_rows"), "source manifest rows")
    raw_rows = _artifact_int(value.get("source_raw_rows"), "source raw rows")
    quote_updates = _artifact_int(value.get("quote_update_sum"), "source quote update sum")
    feature_rows = _artifact_int(value.get("feature_rows"), "source feature rows")
    duplicate_seconds = _artifact_int(value.get("duplicate_seconds"), "source duplicate seconds")
    invalid_rows = _artifact_int(value.get("invalid_feature_rows"), "source invalid feature rows")
    if (
        archive_count <= 0
        or manifest_rows <= 0
        or raw_rows != manifest_rows
        or quote_updates != raw_rows
        or feature_rows <= 0
        or feature_rows > raw_rows
        or duplicate_seconds != 0
        or invalid_rows != 0
    ):
        raise ValueError("microstructure model source row provenance is inconsistent")
    first_second = _artifact_int(
        value.get("first_feature_second_ms"),
        "first source feature second",
    )
    last_second = _artifact_int(
        value.get("last_feature_second_ms"),
        "last source feature second",
    )
    if first_second > last_second:
        raise ValueError("microstructure model source time range is invalid")
    return value


def _validated_performance_confidence(
    value: object,
    *,
    label: str,
    require_positive_lower_bound: bool,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"microstructure model is missing {label} confidence evidence")
    calendar_days = _artifact_int(value.get("calendar_days"), f"{label} calendar days")
    active_days = _artifact_int(value.get("active_days"), f"{label} active days")
    block_length = _artifact_int(value.get("block_length_days"), f"{label} block length")
    samples = _artifact_int(value.get("bootstrap_samples"), f"{label} bootstrap samples")
    lower = _artifact_float(
        value.get("mean_daily_net_bps_ci_lower"),
        f"{label} confidence lower bound",
    )
    upper = _artifact_float(
        value.get("mean_daily_net_bps_ci_upper"),
        f"{label} confidence upper bound",
    )
    probability = _artifact_float(
        value.get("bootstrap_probability_mean_positive"),
        f"{label} positive-mean probability",
    )
    if (
        calendar_days <= 0
        or active_days < 0
        or active_days > calendar_days
        or block_length <= 0
        or block_length > calendar_days
        or samples < 2_000
        or lower > upper
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError(f"microstructure model {label} confidence evidence is inconsistent")
    if require_positive_lower_bound and lower <= 0.0:
        raise ValueError(f"microstructure model {label} lower confidence bound is not positive")
    return value


def _record_from_mapping(record_type, value: object, label: str):
    if not isinstance(value, Mapping):
        raise ValueError(f"microstructure model is missing {label}")
    try:
        return record_type(
            **{item.name: value[item.name] for item in fields(record_type)}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"microstructure model {label} is invalid") from exc


def load_microstructure_model_artifact(path: str | Path) -> MicrostructureModelArtifact:
    """Load a research artifact for deterministic terminal/refit recovery workflows."""

    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("microstructure model artifact is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("microstructure model artifact must be a JSON object")
    if payload.get("schema_version") != MICROSTRUCTURE_MODEL_SCHEMA_VERSION:
        raise ValueError("microstructure model schema is not supported")
    status = str(payload.get("status") or "")
    if status not in {"candidate", "validated", "accepted", "rejected"}:
        raise ValueError("microstructure model artifact status is unsupported")
    feature_names = tuple(payload.get("feature_names") or ())
    if feature_names != MICROSTRUCTURE_FEATURE_NAMES:
        raise ValueError("microstructure model feature names do not match the current contract")
    symbol = str(payload.get("symbol") or "").strip().upper()
    dataset_summary = payload.get("dataset_summary")
    if not isinstance(dataset_summary, Mapping):
        raise ValueError("microstructure model is missing dataset evidence")
    _validated_source_evidence(dataset_summary.get("source_evidence"), symbol=symbol)
    model_strings = payload.get("model_strings")
    if not isinstance(model_strings, Mapping):
        raise ValueError("microstructure model is missing serialized estimators")
    _model_strings_sha256(model_strings)

    metrics_fields = ("policy_metrics", "selection_metrics", "terminal_metrics")
    confidence_fields = ("selection_confidence", "terminal_confidence")
    auc_fields = ("tuning_auc", "tuning_brier", "selection_auc", "selection_brier", "terminal_auc", "terminal_brier")
    values = {item.name: payload.get(item.name) for item in fields(MicrostructureModelArtifact)}
    values["rejection_reasons"] = tuple(payload.get("rejection_reasons") or ())
    values["feature_names"] = feature_names
    values["split"] = _record_from_mapping(
        PurgedSplitEvidence,
        payload.get("split"),
        "purged split evidence",
    )
    values["threshold_policy"] = _record_from_mapping(
        ThresholdPolicy,
        payload.get("threshold_policy"),
        "threshold policy",
    )
    raw_search = payload.get("policy_search")
    if not isinstance(raw_search, Mapping):
        raise ValueError("microstructure model is missing threshold-search evidence")
    search_values = dict(raw_search)
    if search_values.get("best_observed_metrics") is not None:
        search_values["best_observed_metrics"] = _record_from_mapping(
            TradingMetrics,
            search_values["best_observed_metrics"],
            "threshold-search metrics",
        )
    values["policy_search"] = _record_from_mapping(
        ThresholdSearchEvidence,
        search_values,
        "threshold-search evidence",
    )
    for name in metrics_fields:
        raw = payload.get(name)
        values[name] = (
            None
            if raw is None
            else _record_from_mapping(TradingMetrics, raw, name.replace("_", " "))
        )
    for name in confidence_fields:
        raw = payload.get(name)
        values[name] = (
            None
            if raw is None
            else _record_from_mapping(
                PerformanceConfidence,
                raw,
                name.replace("_", " "),
            )
        )
    for name in auc_fields:
        raw = payload.get(name)
        values[name] = None if raw is None else dict(raw)
    for name in ("selection_baselines", "terminal_baselines"):
        raw = payload.get(name)
        if raw is None:
            values[name] = None
        elif isinstance(raw, Mapping):
            values[name] = {
                str(key): _record_from_mapping(
                    TradingMetrics,
                    item,
                    f"{name} {key}",
                )
                for key, item in raw.items()
            }
        else:
            raise ValueError(f"microstructure model {name} is invalid")
    raw_refit = payload.get("deployment_refit")
    values["deployment_refit"] = (
        None
        if raw_refit is None
        else _record_from_mapping(
            DeploymentRefitEvidence,
            raw_refit,
            "deployment refit evidence",
        )
    )
    raw_prequential = payload.get("prequential_validation")
    values["prequential_validation"] = (
        None
        if raw_prequential is None
        else _record_from_mapping(
            PrequentialValidationEvidence,
            raw_prequential,
            "prequential validation evidence",
        )
    )
    values["probability_calibration"] = {
        str(key): tuple(item)
        for key, item in dict(payload.get("probability_calibration") or {}).items()
    }
    values["best_iterations"] = dict(payload.get("best_iterations") or {})
    values["model_strings"] = dict(model_strings)
    deployment_models = payload.get("deployment_model_strings")
    values["deployment_model_strings"] = (
        None if deployment_models is None else dict(deployment_models)
    )
    values["dataset_summary"] = dict(dataset_summary)
    try:
        return MicrostructureModelArtifact(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError("microstructure model artifact fields are incomplete") from exc


def load_microstructure_action_scorer(
    path: str | Path,
    *,
    require_accepted: bool = True,
    as_of_ms: int | None = None,
) -> MicrostructureActionScorer:
    """Load an inference scorer only after validating the full live contract."""

    target = Path(path)
    encoded = target.read_bytes()
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("microstructure model artifact is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("microstructure model artifact must be a JSON object")
    if payload.get("schema_version") != MICROSTRUCTURE_MODEL_SCHEMA_VERSION:
        raise ValueError("microstructure model schema is not supported")
    status = str(payload.get("status") or "")
    rejection_reasons = payload.get("rejection_reasons")
    if status == "rejected" or rejection_reasons:
        raise ValueError("rejected microstructure artifacts cannot be loaded for inference")
    if status not in {"candidate", "validated", "accepted"}:
        raise ValueError("microstructure model artifact status is unsupported")
    if require_accepted and (status != "accepted" or rejection_reasons):
        raise ValueError("live microstructure inference requires an accepted artifact")
    if payload.get("feature_version") != MICROSTRUCTURE_FEATURE_VERSION:
        raise ValueError("microstructure model feature version does not match the live engine")
    if tuple(payload.get("feature_names") or ()) != MICROSTRUCTURE_FEATURE_NAMES:
        raise ValueError("microstructure model feature names do not match the live engine")
    symbol = str(payload.get("symbol") or "").strip().upper()
    risk_level = str(payload.get("risk_level") or "").strip().lower()
    if not symbol or risk_level not in _RISK_LEVELS:
        raise ValueError("microstructure model symbol or risk level is invalid")
    dataset_summary = payload.get("dataset_summary")
    if not isinstance(dataset_summary, Mapping):
        raise ValueError("microstructure model is missing dataset evidence")
    if _artifact_int(
        dataset_summary.get("trade_feature_embargo_ms"),
        "trade feature embargo",
    ) != MICROSTRUCTURE_TRADE_EMBARGO_MS:
        raise ValueError("microstructure model trade feature embargo does not match the live engine")
    source_evidence = _validated_source_evidence(
        dataset_summary.get("source_evidence"),
        symbol=symbol,
    )
    _validated_performance_confidence(
        payload.get("selection_confidence"),
        label="selection",
        require_positive_lower_bound=require_accepted,
    )
    if require_accepted and payload.get("terminal_evaluated_at") is None:
        raise ValueError("accepted microstructure model is missing terminal evaluation evidence")
    if require_accepted:
        unique_days = _artifact_int(payload.get("unique_utc_days"), "unique-day evidence")
        minimum_days = _artifact_int(payload.get("minimum_promotion_days"), "promotion-day gate")
        coverage = _artifact_float(payload.get("calendar_day_coverage_ratio"), "calendar coverage")
        terminal_metrics = payload.get("terminal_metrics")
        terminal_auc = payload.get("terminal_auc")
        terminal_brier = payload.get("terminal_brier")
        if unique_days < max(1, minimum_days) or not 0.95 <= coverage <= 1.0:
            raise ValueError("accepted microstructure model does not reproduce promotion coverage gates")
        if not isinstance(terminal_metrics, Mapping):
            raise ValueError("accepted microstructure model is missing terminal metrics")
        _validated_performance_confidence(
            payload.get("terminal_confidence"),
            label="terminal",
            require_positive_lower_bound=True,
        )
        terminal_trades = _artifact_int(terminal_metrics.get("trades"), "terminal trade count")
        terminal_net = _artifact_float(terminal_metrics.get("total_net_bps"), "terminal net return")
        terminal_drawdown = _artifact_float(terminal_metrics.get("max_drawdown_bps"), "terminal drawdown")
        terminal_profit_factor = _artifact_float(
            terminal_metrics.get("profit_factor"),
            "terminal profit factor",
        )
        if (
            terminal_trades < 20
            or terminal_net <= 0.0
            or terminal_drawdown < 0.0
            or terminal_profit_factor <= 1.0
        ):
            raise ValueError("accepted microstructure model does not reproduce terminal promotion gates")
        if not isinstance(terminal_auc, Mapping) or not isinstance(terminal_brier, Mapping):
            raise ValueError("accepted microstructure model is missing terminal probability evidence")
        for side in ("long", "short"):
            auc = _artifact_float(terminal_auc.get(side), f"terminal {side} AUC")
            brier = _artifact_float(terminal_brier.get(side), f"terminal {side} Brier score")
            if not 0.0 <= auc <= 1.0 or not 0.0 <= brier <= 1.0:
                raise ValueError("accepted microstructure terminal probability evidence is out of range")

    policy = payload.get("threshold_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("microstructure model is missing threshold policy")
    edge_threshold = _artifact_float(
        policy.get("minimum_predicted_edge_bps"),
        "edge threshold",
    )
    probability_threshold = _artifact_float(
        policy.get("minimum_profitable_probability"),
        "probability threshold",
    )
    if edge_threshold < 0.0:
        raise ValueError("microstructure edge threshold is invalid")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("microstructure probability threshold is invalid")

    raw_calibration = payload.get("probability_calibration")
    if not isinstance(raw_calibration, Mapping):
        raise ValueError("microstructure model is missing probability calibration")
    calibration: dict[str, tuple[float, float]] = {}
    for side in ("long", "short"):
        values = raw_calibration.get(side)
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"microstructure model is missing {side} probability calibration")
        pair = (
            _artifact_float(values[0], f"{side} calibration slope"),
            _artifact_float(values[1], f"{side} calibration intercept"),
        )
        if pair[0] <= 0.0:
            raise ValueError(f"microstructure model has invalid {side} probability calibration")
        calibration[side] = pair

    horizon_seconds = _artifact_int(payload.get("horizon_seconds"), "horizon")
    total_latency_ms = _artifact_int(payload.get("total_latency_ms"), "latency")
    taker_fee_bps = _artifact_float(payload.get("taker_fee_bps"), "taker fee")
    reference_notional = _artifact_float(
        payload.get("reference_order_notional_quote"),
        "reference order notional",
    )
    max_participation = _artifact_float(
        payload.get("max_l1_participation"),
        "maximum L1 participation",
    )
    max_quote_age_ms = _artifact_int(payload.get("max_quote_age_ms"), "maximum quote age")
    decision_cadence = _artifact_int(payload.get("decision_cadence_seconds"), "decision cadence")
    target_mode = str(payload.get("target_mode") or "")
    stop_loss_bps = _artifact_float(payload.get("stop_loss_bps"), "stop loss")
    take_profit_bps = _artifact_float(payload.get("take_profit_bps"), "take profit")
    trigger_slippage_bps = _artifact_float(
        payload.get("trigger_execution_slippage_bps"),
        "trigger execution slippage",
    )
    path_resolution_ms = _artifact_int(payload.get("path_resolution_ms"), "path resolution")
    if horizon_seconds <= 0 or total_latency_ms < 0:
        raise ValueError("microstructure model timing contract is invalid")
    if taker_fee_bps < 0.0:
        raise ValueError("microstructure model fee contract is invalid")
    if (
        reference_notional <= 0.0
        or not 0.0 < max_participation <= 1.0
        or max_quote_age_ms <= 0
    ):
        raise ValueError("microstructure model liquidity contract is invalid")
    if decision_cadence <= 0 or decision_cadence > 60:
        raise ValueError("microstructure model decision cadence is invalid")
    if (
        target_mode != "exchange_trigger_market_exit_1s_adverse_first"
        or stop_loss_bps <= 0.0
        or take_profit_bps <= 0.0
        or trigger_slippage_bps < 0.0
        or path_resolution_ms != 1_000
    ):
        raise ValueError("microstructure model protective-exit contract is invalid")
    if (
        _artifact_float(
            dataset_summary.get("reference_order_notional_quote"),
            "dataset reference order notional",
        )
        != reference_notional
        or _artifact_float(
            dataset_summary.get("max_l1_participation"),
            "dataset maximum L1 participation",
        )
        != max_participation
        or _artifact_int(
            dataset_summary.get("max_quote_age_ms"),
            "dataset maximum quote age",
        )
        != max_quote_age_ms
        or _artifact_int(
            dataset_summary.get("decision_cadence_seconds"),
            "dataset decision cadence",
        )
        != decision_cadence
    ):
        raise ValueError("microstructure model liquidity evidence does not match the artifact")
    raw_models = payload.get("model_strings")
    if not isinstance(raw_models, Mapping):
        raise ValueError("microstructure model is missing serialized estimators")
    active_models: Mapping[str, object] = raw_models
    training_cutoff_ms: int | None = None
    expires_at_ms: int | None = None
    enforce_model_freshness = False
    if status == "accepted":
        refit = payload.get("deployment_refit")
        deployment_models = payload.get("deployment_model_strings")
        if not isinstance(refit, Mapping) or not isinstance(deployment_models, Mapping):
            raise ValueError("accepted microstructure model is missing deployment refit evidence")
        if refit.get("refit_mode") != "full_history_fixed_hyperparameters":
            raise ValueError("microstructure deployment refit mode is unsupported")
        if (
            not str(refit.get("backend_kind") or "").strip()
            or not str(refit.get("backend_device") or "").strip()
            or not str(refit.get("lightgbm_version") or "").strip()
        ):
            raise ValueError("microstructure deployment backend evidence is missing")
        calibration_days = _artifact_int(
            payload.get("deployment_calibration_days"),
            "deployment calibration days",
        )
        maximum_age_seconds = _artifact_int(
            payload.get("maximum_model_age_seconds"),
            "maximum model age",
        )
        if calibration_days < 2 or maximum_age_seconds <= 0:
            raise ValueError("microstructure deployment age or calibration contract is invalid")
        if (
            _artifact_int(refit.get("calibration_days"), "refit calibration days")
            != calibration_days
            or _artifact_int(refit.get("maximum_model_age_seconds"), "refit maximum age")
            != maximum_age_seconds
        ):
            raise ValueError("microstructure deployment refit settings drifted from the artifact")
        training_cutoff_ms = _artifact_int(refit.get("training_cutoff_ms"), "refit cutoff")
        expires_at_ms = _artifact_int(refit.get("expires_at_ms"), "refit expiration")
        if (
            training_cutoff_ms <= 0
            or expires_at_ms != training_cutoff_ms + maximum_age_seconds * 1_000
        ):
            raise ValueError("microstructure deployment expiration evidence is inconsistent")
        training_rows = _artifact_int(refit.get("training_rows"), "refit training rows")
        calibration_start_ms = _artifact_int(
            refit.get("calibration_start_ms"),
            "refit calibration start",
        )
        calibration_end_ms = _artifact_int(
            refit.get("calibration_end_ms"),
            "refit calibration end",
        )
        if (
            training_rows <= 0
            or calibration_start_ms <= 0
            or calibration_start_ms > calibration_end_ms
            or calibration_end_ms > training_cutoff_ms
        ):
            raise ValueError("microstructure deployment refit row or time evidence is invalid")
        for field in ("side_training_rows", "side_calibration_rows"):
            values = refit.get(field)
            if not isinstance(values, Mapping) or any(
                _artifact_int(values.get(side), f"refit {field} {side}") <= 0
                for side in ("long", "short")
            ):
                raise ValueError(f"microstructure deployment {field} evidence is invalid")
        if (
            str(refit.get("source_feature_build_id") or "") != source_evidence["build_id"]
            or str(refit.get("source_manifest_fingerprint") or "")
            != source_evidence["manifest_fingerprint"]
        ):
            raise ValueError("microstructure deployment source provenance drifted")
        if str(refit.get("validation_model_sha256") or "") != _model_strings_sha256(
            raw_models
        ):
            raise ValueError("microstructure validation estimator hash drifted")
        if str(refit.get("deployment_model_sha256") or "") != _model_strings_sha256(
            deployment_models
        ):
            raise ValueError("microstructure deployment estimator hash drifted")
        refit_calibration = refit.get("probability_calibration")
        if not isinstance(refit_calibration, Mapping):
            raise ValueError("microstructure deployment calibration is missing")
        deployment_calibration: dict[str, tuple[float, float]] = {}
        for side in ("long", "short"):
            values = refit_calibration.get(side)
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError(f"microstructure deployment {side} calibration is missing")
            pair = (
                _artifact_float(values[0], f"deployment {side} calibration slope"),
                _artifact_float(values[1], f"deployment {side} calibration intercept"),
            )
            if pair[0] <= 0.0:
                raise ValueError(f"microstructure deployment {side} calibration is invalid")
            deployment_calibration[side] = pair
        if as_of_ms is not None and int(as_of_ms) > expires_at_ms:
            raise MicrostructureModelExpiredError(
                f"microstructure deployment model expired at {expires_at_ms}; as_of={int(as_of_ms)}"
            )
        active_models = deployment_models
        calibration = deployment_calibration
        enforce_model_freshness = True
        _validated_prequential_binding(load_microstructure_model_artifact(target))
    models: dict[str, lgb.Booster] = {}
    try:
        for name in _RUNTIME_MODEL_NAMES:
            model_string = active_models.get(name)
            if not isinstance(model_string, str) or not model_string.strip():
                raise ValueError(f"microstructure model is missing estimator {name}")
            models[name] = lgb.Booster(model_str=model_string)
    except (TypeError, ValueError) as exc:
        raise ValueError("microstructure model contains an unreadable estimator") from exc
    return MicrostructureActionScorer(
        artifact_sha256=hashlib.sha256(encoded).hexdigest(),
        symbol=symbol,
        risk_level=risk_level,
        horizon_seconds=horizon_seconds,
        total_latency_ms=total_latency_ms,
        taker_fee_bps=taker_fee_bps,
        reference_order_notional_quote=reference_notional,
        max_l1_participation=max_participation,
        max_quote_age_ms=max_quote_age_ms,
        decision_cadence_seconds=decision_cadence,
        stop_loss_bps=stop_loss_bps,
        take_profit_bps=take_profit_bps,
        trigger_execution_slippage_bps=trigger_slippage_bps,
        path_resolution_ms=path_resolution_ms,
        minimum_predicted_edge_bps=edge_threshold,
        minimum_profitable_probability=probability_threshold,
        probability_calibration=calibration,
        models=models,
        training_cutoff_ms=training_cutoff_ms,
        expires_at_ms=expires_at_ms,
        enforce_model_freshness=enforce_model_freshness,
    )


def concatenate_microstructure_datasets(
    datasets: Sequence[MicrostructureDataset],
) -> MicrostructureDataset:
    if not datasets:
        raise ValueError("at least one microstructure dataset is required")
    first = datasets[0]
    for item in datasets[1:]:
        contract = (
            item.symbol,
            item.feature_version,
            item.feature_names,
            item.horizon_seconds,
            item.total_latency_ms,
            item.taker_fee_bps,
            item.reference_order_notional_quote,
            item.max_l1_participation,
            item.max_quote_age_ms,
            item.decision_cadence_seconds,
            item.target_mode,
            item.stop_loss_bps,
            item.take_profit_bps,
            item.trigger_execution_slippage_bps,
            item.path_resolution_ms,
            item.source_evidence,
            item.trade_feature_embargo_ms,
        )
        expected = (
            first.symbol,
            first.feature_version,
            first.feature_names,
            first.horizon_seconds,
            first.total_latency_ms,
            first.taker_fee_bps,
            first.reference_order_notional_quote,
            first.max_l1_participation,
            first.max_quote_age_ms,
            first.decision_cadence_seconds,
            first.target_mode,
            first.stop_loss_bps,
            first.take_profit_bps,
            first.trigger_execution_slippage_bps,
            first.path_resolution_ms,
            first.source_evidence,
            first.trade_feature_embargo_ms,
        )
        if contract != expected:
            raise ValueError("microstructure datasets do not share one model contract")
    order = np.argsort(np.concatenate([item.decision_time_ms for item in datasets]), kind="stable")

    def combine(name: str) -> np.ndarray:
        return np.concatenate([getattr(item, name) for item in datasets])[order]

    combined = MicrostructureDataset(
        symbol=first.symbol,
        feature_version=first.feature_version,
        feature_names=first.feature_names,
        horizon_seconds=first.horizon_seconds,
        total_latency_ms=first.total_latency_ms,
        taker_fee_bps=first.taker_fee_bps,
        reference_order_notional_quote=first.reference_order_notional_quote,
        max_l1_participation=first.max_l1_participation,
        max_quote_age_ms=first.max_quote_age_ms,
        decision_cadence_seconds=first.decision_cadence_seconds,
        target_mode=first.target_mode,
        stop_loss_bps=first.stop_loss_bps,
        take_profit_bps=first.take_profit_bps,
        trigger_execution_slippage_bps=first.trigger_execution_slippage_bps,
        path_resolution_ms=first.path_resolution_ms,
        decision_time_ms=combine("decision_time_ms"),
        long_exit_time_ms=combine("long_exit_time_ms"),
        short_exit_time_ms=combine("short_exit_time_ms"),
        features=np.concatenate([item.features for item in datasets], axis=0)[order],
        long_net_bps=combine("long_net_bps"),
        short_net_bps=combine("short_net_bps"),
        entry_spread_bps=combine("entry_spread_bps"),
        exit_spread_bps=combine("exit_spread_bps"),
        entry_quote_age_ms=combine("entry_quote_age_ms"),
        exit_quote_age_ms=combine("exit_quote_age_ms"),
        entry_bid_price=combine("entry_bid_price"),
        entry_ask_price=combine("entry_ask_price"),
        fixed_exit_bid_price=combine("fixed_exit_bid_price"),
        fixed_exit_ask_price=combine("fixed_exit_ask_price"),
        entry_bid_qty=combine("entry_bid_qty"),
        entry_ask_qty=combine("entry_ask_qty"),
        fixed_exit_bid_qty=combine("fixed_exit_bid_qty"),
        fixed_exit_ask_qty=combine("fixed_exit_ask_qty"),
        long_l1_participation=combine("long_l1_participation"),
        short_l1_participation=combine("short_l1_participation"),
        long_liquidity_eligible=combine("long_liquidity_eligible"),
        short_liquidity_eligible=combine("short_liquidity_eligible"),
        source_evidence=first.source_evidence,
        trade_feature_embargo_ms=first.trade_feature_embargo_ms,
    )
    if len(np.unique(combined.decision_time_ms)) != combined.rows:
        raise ValueError("combined microstructure datasets contain duplicate decision timestamps")
    return combined


def _utc_day_ids(timestamps_ms: np.ndarray) -> np.ndarray:
    return timestamps_ms // 86_400_000


def _purged_split(dataset: MicrostructureDataset) -> tuple[dict[str, np.ndarray], PurgedSplitEvidence]:
    rows = dataset.rows
    if rows < 2_000:
        raise ValueError("microstructure training requires at least 2,000 executable rows")
    times = dataset.decision_time_ms
    day_ids = _utc_day_ids(times)
    unique_days = np.unique(day_ids)
    if len(unique_days) >= 5:
        train_day_count = min(len(unique_days) - 4, max(1, int(len(unique_days) * 0.50)))
        tuning_day_end = min(
            len(unique_days) - 3,
            max(train_day_count + 1, int(len(unique_days) * 0.65)),
        )
        policy_day_end = min(
            len(unique_days) - 2,
            max(tuning_day_end + 1, int(len(unique_days) * 0.75)),
        )
        selection_day_end = min(
            len(unique_days) - 1,
            max(policy_day_end + 1, int(len(unique_days) * 0.85)),
        )
        train_boundary = int(np.searchsorted(day_ids, unique_days[train_day_count], side="left"))
        tuning_boundary = int(np.searchsorted(day_ids, unique_days[tuning_day_end], side="left"))
        policy_boundary = int(np.searchsorted(day_ids, unique_days[policy_day_end], side="left"))
        selection_boundary = int(
            np.searchsorted(day_ids, unique_days[selection_day_end], side="left")
        )
    else:
        train_boundary = max(1, min(rows - 4, int(rows * 0.50)))
        tuning_boundary = max(train_boundary + 1, min(rows - 3, int(rows * 0.65)))
        policy_boundary = max(tuning_boundary + 1, min(rows - 2, int(rows * 0.75)))
        selection_boundary = max(policy_boundary + 1, min(rows - 1, int(rows * 0.85)))
    purge_ms = (dataset.horizon_seconds * 1000) + dataset.total_latency_ms + 60_000

    def before_boundary(start: int, boundary: int) -> np.ndarray:
        cutoff = int(times[boundary]) - purge_ms
        return np.arange(start, boundary, dtype=np.int64)[times[start:boundary] < cutoff]

    train = before_boundary(0, train_boundary)
    tuning = before_boundary(train_boundary, tuning_boundary)
    policy = before_boundary(tuning_boundary, policy_boundary)
    selection = before_boundary(policy_boundary, selection_boundary)
    terminal = np.arange(selection_boundary, rows, dtype=np.int64)
    if min(len(train), len(tuning), len(policy), len(selection), len(terminal)) < 256:
        raise ValueError("purged train/tuning/policy/selection/terminal segments are too small")
    tuning_early_stop, tuning_calibration = _purged_tuning_subsplit(
        times,
        tuning,
        purge_ms=purge_ms,
    )
    used = len(train) + len(tuning) + len(policy) + len(selection) + len(terminal)
    evidence = PurgedSplitEvidence(
        train_rows=len(train),
        tuning_rows=len(tuning),
        policy_rows=len(policy),
        selection_rows=len(selection),
        terminal_rows=len(terminal),
        train_end_ms=int(times[train[-1]]),
        tuning_start_ms=int(times[train_boundary]),
        policy_start_ms=int(times[tuning_boundary]),
        selection_start_ms=int(times[policy_boundary]),
        terminal_start_ms=int(times[selection_boundary]),
        purge_ms=purge_ms,
        purged_rows=rows - used,
        tuning_early_stop_rows=len(tuning_early_stop),
        tuning_calibration_rows=len(tuning_calibration),
        tuning_calibration_start_ms=int(times[tuning_calibration[0]]),
        tuning_internal_purged_rows=(
            len(tuning) - len(tuning_early_stop) - len(tuning_calibration)
        ),
    )
    return {
        "train": train,
        "tuning": tuning,
        "policy": policy,
        "selection": selection,
        "terminal": terminal,
    }, evidence


def _purged_tuning_subsplit(
    times: np.ndarray,
    tuning_indexes: np.ndarray,
    *,
    purge_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    indexes = np.asarray(tuning_indexes, dtype=np.int64)
    if len(indexes) < 512:
        raise ValueError("tuning segment is too small for separate calibration")
    calibration_offset = len(indexes) // 2
    calibration = indexes[calibration_offset:]
    cutoff = int(times[calibration[0]]) - int(purge_ms)
    early_stop = indexes[:calibration_offset][times[indexes[:calibration_offset]] < cutoff]
    if min(len(early_stop), len(calibration)) < 256:
        raise ValueError("purged tuning early-stop/calibration segments are too small")
    return early_stop, calibration


def _backend_parameters(compute_backend: str, seed: int) -> tuple[dict[str, object], str, str]:
    backend = resolve_backend(compute_backend)
    use_gpu = backend.kind != "cpu"
    parameters: dict[str, object] = {
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "num_threads": max(1, min(16, os.cpu_count() or 1)),
        "device_type": "gpu" if use_gpu else "cpu",
    }
    if not use_gpu:
        return parameters, "cpu", "cpu"
    try:
        platform_id = int(os.getenv("SIMPLE_AI_TRADING_OPENCL_PLATFORM_ID", "0"))
        device_id = int(os.getenv("SIMPLE_AI_TRADING_OPENCL_DEVICE_ID", "0"))
    except ValueError as exc:
        raise ValueError("invalid OpenCL platform or device id") from exc
    parameters.update(
        {
            "gpu_platform_id": platform_id,
            "gpu_device_id": device_id,
            "gpu_use_dp": False,
        }
    )
    return parameters, "opencl", f"opencl:{platform_id}:{device_id}"


def _risk_parameters(risk_level: str, train_rows: int) -> dict[str, object]:
    if risk_level == "conservative":
        leaves, depth, min_leaf, l2 = 31, 6, 256, 0.05
    elif risk_level == "regular":
        leaves, depth, min_leaf, l2 = 47, 7, 160, 0.025
    else:
        leaves, depth, min_leaf, l2 = 63, 8, 96, 0.015
    return {
        "learning_rate": 0.025,
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": max(32, min(min_leaf, train_rows // 100)),
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.005,
        "lambda_l2": l2,
        "max_bin": 127,
    }


def _economic_weights(targets: np.ndarray) -> np.ndarray:
    scale = max(1.0, float(np.quantile(np.abs(targets), 0.90)))
    return (1.0 + np.clip(np.abs(targets) / scale, 0.0, 3.0)).astype(np.float32)


def _train_booster(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_tuning: np.ndarray,
    y_tuning: np.ndarray,
    objective: str,
    metric: str,
    parameters: Mapping[str, object],
    train_weights: np.ndarray | None = None,
    tuning_weights: np.ndarray | None = None,
) -> tuple[lgb.Booster, int]:
    config = dict(parameters)
    config.update({"objective": objective, "metric": metric})
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=_economic_weights(y_train) if train_weights is None else train_weights,
        free_raw_data=False,
    )
    tuning_set = lgb.Dataset(
        x_tuning,
        label=y_tuning,
        weight=tuning_weights,
        reference=train_set,
        free_raw_data=False,
    )
    booster = lgb.train(
        config,
        train_set,
        num_boost_round=1_500,
        valid_sets=[tuning_set],
        valid_names=["tuning"],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    iteration = max(1, int(booster.best_iteration or booster.current_iteration()))
    return booster, iteration


def _train_fixed_booster(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    objective: str,
    metric: str,
    parameters: Mapping[str, object],
    iterations: int,
) -> lgb.Booster:
    rounds = int(iterations)
    if rounds <= 0:
        raise ValueError("deployment refit iterations must be positive")
    config = dict(parameters)
    config.update({"objective": objective, "metric": metric})
    dataset = lgb.Dataset(
        features,
        label=targets,
        weight=np.ones(len(targets), dtype=np.float32),
        free_raw_data=False,
    )
    return lgb.train(
        config,
        dataset,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _apply_platt_scaling(
    probabilities: np.ndarray,
    calibration: tuple[float, float],
) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(values / (1.0 - values))
    slope, intercept = calibration
    scaled = np.clip(slope * logits + intercept, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def _fit_platt_scaling(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit a bounded logistic calibration on the chronological tuning segment."""

    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    outcomes = np.asarray(labels, dtype=np.float64)
    if values.shape != outcomes.shape or values.ndim != 1:
        raise ValueError("probability calibration arrays are inconsistent")
    if min(int(np.sum(outcomes == 0.0)), int(np.sum(outcomes == 1.0))) < 2:
        raise ValueError("probability calibration requires both outcomes")
    logits = np.log(values / (1.0 - values))
    slope = 1.0
    intercept = 0.0
    regularization = 1e-3
    for _ in range(50):
        linear = np.clip(slope * logits + intercept, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        residual = fitted - outcomes
        weights = np.maximum(fitted * (1.0 - fitted), 1e-8)
        gradient = np.asarray(
            [np.sum(residual * logits) + regularization * (slope - 1.0), np.sum(residual)],
            dtype=np.float64,
        )
        hessian = np.asarray(
            [
                [np.sum(weights * logits * logits) + regularization, np.sum(weights * logits)],
                [np.sum(weights * logits), np.sum(weights) + regularization],
            ],
            dtype=np.float64,
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        slope = float(np.clip(slope - step[0], 0.05, 10.0))
        intercept = float(np.clip(intercept - step[1], -10.0, 10.0))
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return slope, intercept


def _trading_metrics(pnls: Sequence[float], sides: Sequence[int], timestamps: Sequence[int]) -> TradingMetrics:
    values = np.asarray(pnls, dtype=np.float64)
    side_values = np.asarray(sides, dtype=np.int8)
    time_values = np.asarray(timestamps, dtype=np.int64)
    if values.size == 0:
        return TradingMetrics(0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0, 0, 0, 0.0)
    cumulative = np.cumsum(values)
    peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[:-1]
    drawdown = peak - cumulative
    gains = float(np.sum(values[values > 0.0]))
    losses = float(-np.sum(values[values < 0.0]))
    days = len(np.unique(_utc_day_ids(time_values)))
    return TradingMetrics(
        trades=int(values.size),
        total_net_bps=float(np.sum(values)),
        mean_net_bps=float(np.mean(values)),
        median_net_bps=float(np.median(values)),
        win_rate=float(np.mean(values > 0.0)),
        profit_factor=(gains / losses if losses > 0.0 else (None if gains <= 0.0 else 1.0e12)),
        max_drawdown_bps=float(np.max(drawdown, initial=0.0)),
        worst_trade_bps=float(np.min(values)),
        best_trade_bps=float(np.max(values)),
        long_trades=int(np.sum(side_values == 1)),
        short_trades=int(np.sum(side_values == -1)),
        active_days=days,
        trades_per_active_day=float(values.size / max(1, days)),
    )


@dataclass(frozen=True)
class _SimulationTrace:
    metrics: TradingMetrics
    pnls: tuple[float, ...]
    sides: tuple[int, ...]
    timestamps: tuple[int, ...]


def _simulate_non_overlapping_trace(
    *,
    timestamps: np.ndarray,
    long_exit_times: np.ndarray,
    short_exit_times: np.ndarray,
    long_targets: np.ndarray,
    short_targets: np.ndarray,
    long_edge: np.ndarray,
    short_edge: np.ndarray,
    long_probability: np.ndarray,
    short_probability: np.ndarray,
    edge_threshold: float,
    probability_threshold: float,
    long_eligible: np.ndarray | None = None,
    short_eligible: np.ndarray | None = None,
) -> _SimulationTrace:
    if long_eligible is not None and len(long_eligible) != len(timestamps):
        raise ValueError("long liquidity eligibility length is inconsistent")
    if short_eligible is not None and len(short_eligible) != len(timestamps):
        raise ValueError("short liquidity eligibility length is inconsistent")
    next_available_ms = -1
    pnls: list[float] = []
    sides: list[int] = []
    executed_times: list[int] = []
    for index, timestamp in enumerate(timestamps):
        ts = int(timestamp)
        if ts < next_available_ms:
            continue
        long_is_eligible = (
            (long_eligible is None or bool(long_eligible[index]))
            and long_edge[index] >= edge_threshold
            and long_probability[index] >= probability_threshold
        )
        short_is_eligible = (
            (short_eligible is None or bool(short_eligible[index]))
            and short_edge[index] >= edge_threshold
            and short_probability[index] >= probability_threshold
        )
        if not long_is_eligible and not short_is_eligible:
            continue
        if long_is_eligible and (
            not short_is_eligible or long_edge[index] >= short_edge[index]
        ):
            side = 1
            pnl = float(long_targets[index])
            exit_time = int(long_exit_times[index])
        else:
            side = -1
            pnl = float(short_targets[index])
            exit_time = int(short_exit_times[index])
        pnls.append(pnl)
        sides.append(side)
        executed_times.append(ts)
        next_available_ms = exit_time
    return _SimulationTrace(
        metrics=_trading_metrics(pnls, sides, executed_times),
        pnls=tuple(pnls),
        sides=tuple(sides),
        timestamps=tuple(executed_times),
    )


def _simulate_non_overlapping(
    *,
    timestamps: np.ndarray,
    long_exit_times: np.ndarray,
    short_exit_times: np.ndarray,
    long_targets: np.ndarray,
    short_targets: np.ndarray,
    long_edge: np.ndarray,
    short_edge: np.ndarray,
    long_probability: np.ndarray,
    short_probability: np.ndarray,
    edge_threshold: float,
    probability_threshold: float,
    long_eligible: np.ndarray | None = None,
    short_eligible: np.ndarray | None = None,
) -> TradingMetrics:
    return _simulate_non_overlapping_trace(
        timestamps=timestamps,
        long_exit_times=long_exit_times,
        short_exit_times=short_exit_times,
        long_targets=long_targets,
        short_targets=short_targets,
        long_edge=long_edge,
        short_edge=short_edge,
        long_probability=long_probability,
        short_probability=short_probability,
        edge_threshold=edge_threshold,
        probability_threshold=probability_threshold,
        long_eligible=long_eligible,
        short_eligible=short_eligible,
    ).metrics


def _performance_confidence(
    trace: _SimulationTrace,
    segment_timestamps: np.ndarray,
    *,
    bootstrap_samples: int = 2_000,
) -> PerformanceConfidence:
    segment = np.asarray(segment_timestamps, dtype=np.int64)
    samples = int(bootstrap_samples)
    if segment.size == 0:
        raise ValueError("performance confidence requires a non-empty evaluation segment")
    if samples < 1_000:
        raise ValueError("performance confidence requires at least 1,000 bootstrap samples")
    first_day = int(np.min(segment) // 86_400_000)
    last_day = int(np.max(segment) // 86_400_000)
    calendar_days = last_day - first_day + 1
    daily = np.zeros(calendar_days, dtype=np.float64)
    if trace.timestamps:
        executed_days = np.asarray(trace.timestamps, dtype=np.int64) // 86_400_000
        executed_pnls = np.asarray(trace.pnls, dtype=np.float64)
        np.add.at(daily, executed_days - first_day, executed_pnls)
    block_length = max(1, min(calendar_days, int(math.ceil(calendar_days ** (1.0 / 3.0)))))
    block_count = int(math.ceil(calendar_days / block_length))
    seed_material = (
        f"{first_day}:{last_day}:{len(trace.pnls)}:{float(np.sum(daily)):.12f}"
    ).encode("ascii")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    offsets = np.arange(block_length, dtype=np.int64)
    for sample in range(samples):
        starts = rng.integers(0, calendar_days, size=block_count)
        indexes = ((starts[:, None] + offsets[None, :]) % calendar_days).reshape(-1)
        bootstrap_means[sample] = float(np.mean(daily[indexes[:calendar_days]]))
    daily_std = float(np.std(daily, ddof=1)) if calendar_days > 1 else 0.0
    return PerformanceConfidence(
        calendar_days=calendar_days,
        active_days=trace.metrics.active_days,
        block_length_days=block_length,
        bootstrap_samples=samples,
        mean_daily_net_bps=float(np.mean(daily)),
        median_daily_net_bps=float(np.median(daily)),
        worst_daily_net_bps=float(np.min(daily)),
        best_daily_net_bps=float(np.max(daily)),
        annualized_daily_sharpe=(
            float(np.mean(daily) / daily_std * math.sqrt(365.0)) if daily_std > 0.0 else None
        ),
        mean_daily_net_bps_ci_lower=float(np.quantile(bootstrap_means, 0.025)),
        mean_daily_net_bps_ci_upper=float(np.quantile(bootstrap_means, 0.975)),
        bootstrap_probability_mean_positive=float(np.mean(bootstrap_means > 0.0)),
    )


def _select_threshold(
    *,
    risk_level: str,
    timestamps: np.ndarray,
    long_exit_times: np.ndarray,
    short_exit_times: np.ndarray,
    long_targets: np.ndarray,
    short_targets: np.ndarray,
    long_edge: np.ndarray,
    short_edge: np.ndarray,
    long_probability: np.ndarray,
    short_probability: np.ndarray,
    long_eligible: np.ndarray | None = None,
    short_eligible: np.ndarray | None = None,
) -> tuple[ThresholdPolicy, TradingMetrics, ThresholdSearchEvidence]:
    long_mask = (
        np.ones(len(long_edge), dtype=bool)
        if long_eligible is None
        else np.asarray(long_eligible, dtype=bool)
    )
    short_mask = (
        np.ones(len(short_edge), dtype=bool)
        if short_eligible is None
        else np.asarray(short_eligible, dtype=bool)
    )
    strongest = np.maximum(
        np.where(long_mask, long_edge, -math.inf),
        np.where(short_mask, short_edge, -math.inf),
    )
    finite = strongest[np.isfinite(strongest)]
    if finite.size == 0:
        raise ValueError("model produced no finite action values")
    positive = finite[finite > 0.0]
    quantiles = (0.0, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 1.0)
    edge_thresholds = sorted(
        {0.0, *(float(np.quantile(positive, value)) for value in quantiles)}
        if positive.size
        else {0.0}
    )
    strongest_probability = np.where(
        long_mask & (~short_mask | (long_edge >= short_edge)),
        long_probability,
        short_probability,
    )
    probability_population = strongest_probability[strongest > 0.0]
    if probability_population.size == 0:
        probability_population = strongest_probability
    probability_quantiles = {
        "conservative": (0.75, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995),
        "regular": (0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98),
        "aggressive": (0.0, 0.25, 0.50, 0.65, 0.75, 0.85, 0.90),
    }[risk_level]
    probability_floors = sorted(
        {float(np.quantile(probability_population, value)) for value in probability_quantiles}
    )
    minimum_trades = _minimum_evaluation_trades(timestamps)
    best_policy: ThresholdPolicy | None = None
    best_metrics: TradingMetrics | None = None
    best_utility = -math.inf
    evaluated_policy_count = 0
    policies_meeting_trade_minimum = 0
    for edge_threshold in edge_thresholds:
        for probability_threshold in probability_floors:
            evaluated_policy_count += 1
            metrics = _simulate_non_overlapping(
                timestamps=timestamps,
                long_exit_times=long_exit_times,
                short_exit_times=short_exit_times,
                long_targets=long_targets,
                short_targets=short_targets,
                long_edge=long_edge,
                short_edge=short_edge,
                long_probability=long_probability,
                short_probability=short_probability,
                edge_threshold=edge_threshold,
                probability_threshold=probability_threshold,
                long_eligible=long_eligible,
                short_eligible=short_eligible,
            )
            if metrics.trades < minimum_trades:
                continue
            policies_meeting_trade_minimum += 1
            utility = _risk_utility(metrics, risk_level)
            if utility > best_utility:
                best_utility = utility
                best_metrics = metrics
                best_policy = ThresholdPolicy(edge_threshold, probability_threshold, utility)
    evidence = ThresholdSearchEvidence(
        rows=len(timestamps),
        long_positive_edge_rows=int(np.sum((np.asarray(long_edge) > 0.0) & long_mask)),
        short_positive_edge_rows=int(np.sum((np.asarray(short_edge) > 0.0) & short_mask)),
        minimum_required_trades=minimum_trades,
        evaluated_policy_count=evaluated_policy_count,
        policies_meeting_trade_minimum=policies_meeting_trade_minimum,
        best_observed_utility_bps=(float(best_utility) if math.isfinite(best_utility) else None),
        best_observed_metrics=best_metrics,
    )
    if best_policy is None or best_metrics is None or best_utility <= 0.0:
        return ThresholdPolicy(1.0e12, 1.0, 0.0), _trading_metrics([], [], []), evidence
    return best_policy, best_metrics, evidence


def _baseline_metrics(dataset: MicrostructureDataset, indexes: np.ndarray) -> dict[str, TradingMetrics]:
    times = dataset.decision_time_ms[indexes]
    ones = np.ones(len(indexes), dtype=np.float64)
    zeros = np.zeros(len(indexes), dtype=np.float64)
    always_long = _simulate_non_overlapping(
        timestamps=times,
        long_exit_times=dataset.long_exit_time_ms[indexes],
        short_exit_times=dataset.short_exit_time_ms[indexes],
        long_targets=dataset.long_net_bps[indexes],
        short_targets=dataset.short_net_bps[indexes],
        long_edge=ones,
        short_edge=zeros,
        long_probability=ones,
        short_probability=ones,
        edge_threshold=0.5,
        probability_threshold=0.5,
        long_eligible=dataset.long_liquidity_eligible[indexes],
        short_eligible=dataset.short_liquidity_eligible[indexes],
    )
    always_short = _simulate_non_overlapping(
        timestamps=times,
        long_exit_times=dataset.long_exit_time_ms[indexes],
        short_exit_times=dataset.short_exit_time_ms[indexes],
        long_targets=dataset.long_net_bps[indexes],
        short_targets=dataset.short_net_bps[indexes],
        long_edge=zeros,
        short_edge=ones,
        long_probability=ones,
        short_probability=ones,
        edge_threshold=0.5,
        probability_threshold=0.5,
        long_eligible=dataset.long_liquidity_eligible[indexes],
        short_eligible=dataset.short_liquidity_eligible[indexes],
    )
    return {
        "no_trade": _trading_metrics([], [], []),
        "always_long": always_long,
        "always_short": always_short,
    }


def _side_probability_quality(
    dataset: MicrostructureDataset,
    indexes: np.ndarray,
    long_probability: np.ndarray,
    short_probability: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    probabilities = {
        "long": np.clip(np.asarray(long_probability, dtype=np.float64), 0.0, 1.0),
        "short": np.clip(np.asarray(short_probability, dtype=np.float64), 0.0, 1.0),
    }
    targets = {"long": dataset.long_net_bps, "short": dataset.short_net_bps}
    eligible = {
        "long": np.asarray(dataset.long_liquidity_eligible[indexes], dtype=bool),
        "short": np.asarray(dataset.short_liquidity_eligible[indexes], dtype=bool),
    }
    auc: dict[str, float] = {}
    brier: dict[str, float] = {}
    for side in ("long", "short"):
        labels = (targets[side][indexes][eligible[side]] > 0.0).astype(np.int8)
        scores = probabilities[side][eligible[side]]
        auc[side] = _auc(labels, scores)
        brier[side] = (
            float(np.mean(np.square(scores - labels))) if labels.size else 1.0
        )
    return auc, brier


def _risk_utility(metrics: TradingMetrics, risk_level: str) -> float:
    penalty = {"conservative": 2.0, "regular": 1.5, "aggressive": 1.0}[risk_level]
    return metrics.total_net_bps - penalty * metrics.max_drawdown_bps


def _minimum_evaluation_trades(timestamps: np.ndarray) -> int:
    return max(20, len(np.unique(_utc_day_ids(timestamps))) * 5)


def _calendar_coverage(timestamps: np.ndarray) -> tuple[int, int, float, int, float, float]:
    day_ids, counts = np.unique(_utc_day_ids(timestamps), return_counts=True)
    if len(day_ids) == 0:
        return 0, 0, 0.0, 0, 0.0, 0.0
    span = int(day_ids[-1] - day_ids[0] + 1)
    return (
        int(len(day_ids)),
        span,
        float(len(day_ids) / max(1, span)),
        int(np.min(counts)),
        float(np.quantile(counts, 0.10)),
        float(np.median(counts)),
    )


def train_microstructure_action_value_model(
    dataset: MicrostructureDataset,
    *,
    risk_level: str = "conservative",
    compute_backend: str = "auto",
    seed: int = 20260710,
    minimum_promotion_days: int = 365,
    deployment_calibration_days: int = 14,
    maximum_model_age_seconds: int = 86_400,
    evaluate_terminal: bool = False,
    progress: ModelProgressCallback | None = None,
) -> MicrostructureModelArtifact:
    risk = str(risk_level).strip().lower()
    if risk not in _RISK_LEVELS:
        raise ValueError("risk_level must be conservative, regular, or aggressive")
    if dataset.rows <= 0 or not np.all(np.isfinite(dataset.features)):
        raise ValueError("microstructure dataset is empty or non-finite")
    calibration_days = int(deployment_calibration_days)
    maximum_age = int(maximum_model_age_seconds)
    if calibration_days < 2:
        raise ValueError("deployment_calibration_days must be at least 2")
    if maximum_age <= 0:
        raise ValueError("maximum_model_age_seconds must be positive")
    _validated_source_evidence(dataset.source_evidence, symbol=dataset.symbol)
    splits, split_evidence = _purged_split(dataset)
    x = np.asarray(dataset.features, dtype=np.float32)
    backend_params, backend_kind, backend_device = _backend_parameters(compute_backend, seed)
    parameters = {**backend_params, **_risk_parameters(risk, len(splits["train"]))}

    targets = {"long": dataset.long_net_bps, "short": dataset.short_net_bps}
    liquidity_eligible = {
        "long": np.asarray(dataset.long_liquidity_eligible, dtype=bool),
        "short": np.asarray(dataset.short_liquidity_eligible, dtype=bool),
    }
    models: dict[str, lgb.Booster] = {}
    iterations: dict[str, int] = {}
    probability_calibration: dict[str, tuple[float, float]] = {}
    training_step = 0
    total_training_steps = 6
    train_indexes = splits["train"]
    tuning_indexes, calibration_indexes = _purged_tuning_subsplit(
        dataset.decision_time_ms,
        splits["tuning"],
        purge_ms=split_evidence.purge_ms,
    )
    for side, target in targets.items():
        side_train_indexes = train_indexes[liquidity_eligible[side][train_indexes]]
        side_tuning_indexes = tuning_indexes[liquidity_eligible[side][tuning_indexes]]
        side_calibration_indexes = calibration_indexes[
            liquidity_eligible[side][calibration_indexes]
        ]
        train_labels = (target[side_train_indexes] > 0.0).astype(np.float32)
        tuning_labels = (target[side_tuning_indexes] > 0.0).astype(np.float32)
        calibration_labels = (
            target[side_calibration_indexes] > 0.0
        ).astype(np.float32)
        if min(
            int(np.sum(train_labels == 0.0)),
            int(np.sum(train_labels == 1.0)),
            int(np.sum(tuning_labels == 0.0)),
            int(np.sum(tuning_labels == 1.0)),
            int(np.sum(calibration_labels == 0.0)),
            int(np.sum(calibration_labels == 1.0)),
        ) < 256:
            raise ValueError(f"{side} hurdle classifier has insufficient class support")
        if progress:
            progress(f"train-{side}-probability", training_step, total_training_steps)
        classifier, classifier_iteration = _train_booster(
            x_train=x[side_train_indexes],
            y_train=train_labels,
            x_tuning=x[side_tuning_indexes],
            y_tuning=tuning_labels,
            objective="binary",
            metric="binary_logloss",
            parameters=parameters,
            train_weights=np.ones(len(side_train_indexes), dtype=np.float32),
        )
        training_step += 1
        models[f"{side}_probability"] = classifier
        iterations[f"{side}_probability"] = classifier_iteration
        raw_calibration_probability = classifier.predict(
            x[side_calibration_indexes], num_iteration=classifier_iteration
        )
        probability_calibration[side] = _fit_platt_scaling(
            raw_calibration_probability,
            calibration_labels,
        )

        for outcome, keep, magnitude in (
            ("win", train_labels == 1.0, target[side_train_indexes]),
            ("loss", train_labels == 0.0, -target[side_train_indexes]),
        ):
            tuning_keep = tuning_labels == (1.0 if outcome == "win" else 0.0)
            if min(int(np.sum(keep)), int(np.sum(tuning_keep))) < 256:
                raise ValueError(f"{side} conditional {outcome} regressor has insufficient rows")
            if progress:
                progress(
                    f"train-{side}-{outcome}-magnitude",
                    training_step,
                    total_training_steps,
                )
            tuning_magnitude = (
                target[side_tuning_indexes][tuning_keep]
                if outcome == "win"
                else -target[side_tuning_indexes][tuning_keep]
            )
            regressor, regressor_iteration = _train_booster(
                x_train=x[side_train_indexes][keep],
                y_train=np.asarray(magnitude[keep], dtype=np.float32),
                x_tuning=x[side_tuning_indexes][tuning_keep],
                y_tuning=np.asarray(tuning_magnitude, dtype=np.float32),
                objective="regression",
                metric="l2",
                parameters=parameters,
                train_weights=np.ones(int(np.sum(keep)), dtype=np.float32),
            )
            training_step += 1
            name = f"{side}_{outcome}_magnitude"
            models[name] = regressor
            iterations[name] = regressor_iteration

    if progress:
        progress("select-threshold", training_step, total_training_steps)

    def predictions(indexes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side in ("long", "short"):
            probability = _apply_platt_scaling(
                models[f"{side}_probability"].predict(
                    x[indexes], num_iteration=iterations[f"{side}_probability"]
                ),
                probability_calibration[side],
            )
            win = np.maximum(
                0.0,
                models[f"{side}_win_magnitude"].predict(
                    x[indexes], num_iteration=iterations[f"{side}_win_magnitude"]
                ),
            )
            loss = np.maximum(
                0.0,
                models[f"{side}_loss_magnitude"].predict(
                    x[indexes], num_iteration=iterations[f"{side}_loss_magnitude"]
                ),
            )
            output[side] = (probability * win - (1.0 - probability) * loss, probability)
        return output["long"][0], output["short"][0], output["long"][1], output["short"][1]

    tuning_predictions = predictions(calibration_indexes)
    tuning_auc, tuning_brier = _side_probability_quality(
        dataset,
        calibration_indexes,
        tuning_predictions[2],
        tuning_predictions[3],
    )

    policy_indexes = splits["policy"]
    policy_predictions = predictions(policy_indexes)
    policy, policy_metrics, policy_search = _select_threshold(
        risk_level=risk,
        timestamps=dataset.decision_time_ms[policy_indexes],
        long_exit_times=dataset.long_exit_time_ms[policy_indexes],
        short_exit_times=dataset.short_exit_time_ms[policy_indexes],
        long_targets=dataset.long_net_bps[policy_indexes],
        short_targets=dataset.short_net_bps[policy_indexes],
        long_edge=policy_predictions[0],
        short_edge=policy_predictions[1],
        long_probability=policy_predictions[2],
        short_probability=policy_predictions[3],
        long_eligible=dataset.long_liquidity_eligible[policy_indexes],
        short_eligible=dataset.short_liquidity_eligible[policy_indexes],
    )
    selection = splits["selection"]
    selection_predictions = predictions(selection)
    selection_auc, selection_brier = _side_probability_quality(
        dataset,
        selection,
        selection_predictions[2],
        selection_predictions[3],
    )
    selection_trace = _simulate_non_overlapping_trace(
        timestamps=dataset.decision_time_ms[selection],
        long_exit_times=dataset.long_exit_time_ms[selection],
        short_exit_times=dataset.short_exit_time_ms[selection],
        long_targets=dataset.long_net_bps[selection],
        short_targets=dataset.short_net_bps[selection],
        long_edge=selection_predictions[0],
        short_edge=selection_predictions[1],
        long_probability=selection_predictions[2],
        short_probability=selection_predictions[3],
        edge_threshold=policy.minimum_predicted_edge_bps,
        probability_threshold=policy.minimum_profitable_probability,
        long_eligible=dataset.long_liquidity_eligible[selection],
        short_eligible=dataset.short_liquidity_eligible[selection],
    )
    selection_metrics = selection_trace.metrics
    selection_confidence = _performance_confidence(
        selection_trace,
        dataset.decision_time_ms[selection],
    )
    selection_baselines = _baseline_metrics(dataset, selection)
    (
        unique_days,
        calendar_span_days,
        calendar_day_coverage_ratio,
        minimum_rows_per_utc_day,
        daily_rows_p10,
        median_rows_per_utc_day,
    ) = _calendar_coverage(dataset.decision_time_ms)
    reasons: list[str] = []
    if policy_metrics.trades <= 0 or _risk_utility(policy_metrics, risk) <= 0.0:
        reasons.append("policy_segment_has_no_positive_drawdown_adjusted_threshold")
    if selection_metrics.trades < _minimum_evaluation_trades(
        dataset.decision_time_ms[selection]
    ):
        reasons.append("selection_trade_count_below_statistical_minimum")
    if selection_metrics.total_net_bps <= 0.0 or _risk_utility(selection_metrics, risk) <= 0.0:
        reasons.append("selection_not_profitable_after_drawdown_penalty")
    if selection_metrics.profit_factor is None or selection_metrics.profit_factor <= 1.0:
        reasons.append("selection_profit_factor_not_above_one")
    if selection_confidence.mean_daily_net_bps_ci_lower <= 0.0:
        reasons.append("selection_daily_edge_lower_confidence_bound_not_positive")
    strongest_selection_baseline = max(
        item.total_net_bps for item in selection_baselines.values()
    )
    if selection_metrics.total_net_bps <= strongest_selection_baseline:
        reasons.append("selection_not_above_directional_baselines")
    if selection_metrics.long_trades and selection_auc["long"] <= 0.5:
        reasons.append("selection_long_action_auc_not_above_random")
    if selection_metrics.short_trades and selection_auc["short"] <= 0.5:
        reasons.append("selection_short_action_auc_not_above_random")
    status = "candidate" if not reasons else "rejected"
    model_strings = {
        name: model.model_to_string(num_iteration=iterations[name]) for name, model in models.items()
    }
    artifact = MicrostructureModelArtifact(
        schema_version=MICROSTRUCTURE_MODEL_SCHEMA_VERSION,
        model_family="side_specific_hurdle_expected_value",
        status=status,
        rejection_reasons=tuple(reasons),
        symbol=dataset.symbol,
        feature_version=dataset.feature_version,
        feature_names=dataset.feature_names,
        risk_level=risk,
        horizon_seconds=dataset.horizon_seconds,
        total_latency_ms=dataset.total_latency_ms,
        taker_fee_bps=dataset.taker_fee_bps,
        reference_order_notional_quote=dataset.reference_order_notional_quote,
        max_l1_participation=dataset.max_l1_participation,
        max_quote_age_ms=dataset.max_quote_age_ms,
        decision_cadence_seconds=dataset.decision_cadence_seconds,
        target_mode=dataset.target_mode,
        stop_loss_bps=dataset.stop_loss_bps,
        take_profit_bps=dataset.take_profit_bps,
        trigger_execution_slippage_bps=dataset.trigger_execution_slippage_bps,
        path_resolution_ms=dataset.path_resolution_ms,
        training_backend_kind=backend_kind,
        training_backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        seed=int(seed),
        unique_utc_days=unique_days,
        calendar_span_days=calendar_span_days,
        calendar_day_coverage_ratio=calendar_day_coverage_ratio,
        minimum_rows_per_utc_day=minimum_rows_per_utc_day,
        daily_rows_p10=daily_rows_p10,
        median_rows_per_utc_day=median_rows_per_utc_day,
        minimum_promotion_days=max(1, int(minimum_promotion_days)),
        deployment_calibration_days=calibration_days,
        maximum_model_age_seconds=maximum_age,
        split=split_evidence,
        best_iterations=iterations,
        probability_calibration=probability_calibration,
        threshold_policy=policy,
        policy_search=policy_search,
        tuning_auc=tuning_auc,
        tuning_brier=tuning_brier,
        selection_auc=selection_auc,
        selection_brier=selection_brier,
        terminal_auc=None,
        terminal_brier=None,
        policy_metrics=policy_metrics,
        selection_metrics=selection_metrics,
        terminal_metrics=None,
        selection_confidence=selection_confidence,
        terminal_confidence=None,
        selection_baselines=selection_baselines,
        terminal_baselines=None,
        model_strings=model_strings,
        deployment_model_strings=None,
        deployment_refit=None,
        dataset_summary=dataset.summary(),
        trained_at=datetime.now(timezone.utc).isoformat(),
        terminal_evaluated_at=None,
    )
    if evaluate_terminal and artifact.status == "candidate":
        validated = evaluate_microstructure_model_terminal(
            artifact,
            dataset,
            progress=progress,
        )
        if validated.status == "validated":
            return refit_validated_microstructure_model(
                validated,
                dataset,
                compute_backend=compute_backend,
                progress=progress,
            )
        return validated
    return artifact


def _candidate_payload_sha256(artifact: MicrostructureModelArtifact) -> str:
    payload = artifact.asdict()
    payload.update(
        {
            "status": "candidate",
            "rejection_reasons": [],
            "terminal_auc": None,
            "terminal_brier": None,
            "terminal_metrics": None,
            "terminal_confidence": None,
            "terminal_baselines": None,
            "deployment_model_strings": None,
            "deployment_refit": None,
            "terminal_evaluated_at": None,
            "prequential_validation": None,
        }
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def microstructure_candidate_sha256(artifact: MicrostructureModelArtifact) -> str:
    if artifact.status != "candidate" or artifact.rejection_reasons:
        raise ValueError("only an unrejected candidate can be fingerprinted for terminal evaluation")
    if artifact.terminal_evaluated_at is not None or artifact.terminal_metrics is not None:
        raise ValueError("terminal-evaluated artifacts cannot be candidate fingerprints")
    return _candidate_payload_sha256(artifact)


def _validated_prequential_binding(
    artifact: MicrostructureModelArtifact,
) -> PrequentialValidationEvidence:
    evidence = artifact.prequential_validation
    if evidence is None:
        raise ValueError("microstructure model is missing prequential validation evidence")
    if evidence.version != MICROSTRUCTURE_PREQUENTIAL_EVIDENCE_VERSION:
        raise ValueError("microstructure prequential evidence version is unsupported")
    for label, digest in (
        ("report", evidence.report_sha256),
        ("predictions", evidence.predictions_sha256),
        ("chart", evidence.chart_sha256),
        ("candidate", evidence.candidate_sha256),
        ("protocol", evidence.protocol_sha256),
        ("fold models", evidence.fold_models_sha256),
        ("source feature build", evidence.source_feature_build_id),
        ("source manifest", evidence.source_manifest_fingerprint),
    ):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"microstructure prequential {label} digest is invalid")
    if evidence.candidate_sha256 != _candidate_payload_sha256(artifact):
        raise ValueError("microstructure prequential candidate fingerprint drifted")
    source = _validated_source_evidence(
        artifact.dataset_summary.get("source_evidence"),
        symbol=artifact.symbol,
    )
    if (
        evidence.source_feature_build_id != source["build_id"]
        or evidence.source_manifest_fingerprint != source["manifest_fingerprint"]
    ):
        raise ValueError("microstructure prequential source provenance drifted")
    numeric = (
        evidence.selection_coverage_ratio,
        evidence.total_net_bps,
        evidence.profit_factor,
        evidence.max_drawdown_bps,
        evidence.mean_daily_net_bps_ci_lower,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("microstructure prequential financial evidence is non-finite")
    if (
        evidence.generated_at_ms <= 0
        or evidence.planned_folds < 3
        or evidence.complete_folds != evidence.planned_folds
        or evidence.evaluated_rows <= 0
        or evidence.selection_coverage_ratio != 1.0
        or evidence.total_net_bps <= 0.0
        or evidence.profit_factor <= 1.0
        or evidence.max_drawdown_bps < 0.0
        or evidence.mean_daily_net_bps_ci_lower <= 0.0
        or not evidence.attached_at.strip()
    ):
        raise ValueError("microstructure prequential promotion gates are not satisfied")
    return evidence


def refit_validated_microstructure_model(
    artifact: MicrostructureModelArtifact,
    dataset: MicrostructureDataset,
    *,
    compute_backend: str = "auto",
    progress: ModelProgressCallback | None = None,
) -> MicrostructureModelArtifact:
    """Refit fixed validated estimators on all labeled rows for bounded live use."""

    if artifact.status != "validated" or artifact.rejection_reasons:
        raise ValueError("deployment refit requires a terminal-validated artifact")
    if artifact.terminal_metrics is None or artifact.terminal_evaluated_at is None:
        raise ValueError("deployment refit requires terminal evaluation evidence")
    _validated_prequential_binding(artifact)
    if artifact.deployment_refit is not None or artifact.deployment_model_strings is not None:
        raise ValueError("deployment refit has already been applied")
    if (
        dataset.symbol != artifact.symbol
        or dataset.feature_version != artifact.feature_version
        or dataset.feature_names != artifact.feature_names
        or dataset.horizon_seconds != artifact.horizon_seconds
        or dataset.total_latency_ms != artifact.total_latency_ms
        or dataset.taker_fee_bps != artifact.taker_fee_bps
        or dataset.reference_order_notional_quote
        != artifact.reference_order_notional_quote
        or dataset.max_l1_participation != artifact.max_l1_participation
        or dataset.max_quote_age_ms != artifact.max_quote_age_ms
        or dataset.decision_cadence_seconds != artifact.decision_cadence_seconds
        or dataset.target_mode != artifact.target_mode
        or dataset.stop_loss_bps != artifact.stop_loss_bps
        or dataset.take_profit_bps != artifact.take_profit_bps
        or dataset.trigger_execution_slippage_bps
        != artifact.trigger_execution_slippage_bps
        or dataset.path_resolution_ms != artifact.path_resolution_ms
    ):
        raise ValueError("deployment refit dataset does not match the validated contract")
    source = _validated_source_evidence(dataset.source_evidence, symbol=dataset.symbol)
    artifact_source = _validated_source_evidence(
        artifact.dataset_summary.get("source_evidence"),
        symbol=artifact.symbol,
    )
    if (
        source["build_id"] != artifact_source["build_id"]
        or source["manifest_fingerprint"] != artifact_source["manifest_fingerprint"]
    ):
        raise ValueError("deployment refit source provenance does not match validation")

    calibration_days = int(artifact.deployment_calibration_days)
    maximum_age_seconds = int(artifact.maximum_model_age_seconds)
    times = np.asarray(dataset.decision_time_ms, dtype=np.int64)
    day_ids = _utc_day_ids(times)
    unique_days = np.unique(day_ids)
    if len(unique_days) <= calibration_days + 2:
        raise ValueError("deployment refit has insufficient days before its calibration tail")
    calibration_first_day = int(unique_days[-calibration_days])
    calibration_indexes = np.flatnonzero(day_ids >= calibration_first_day).astype(np.int64)
    calibration_start_ms = int(times[calibration_indexes[0]])
    provisional_cutoff_ms = calibration_start_ms - int(artifact.split.purge_ms)
    provisional_indexes = np.flatnonzero(times < provisional_cutoff_ms).astype(np.int64)
    if len(provisional_indexes) < 2_000 or len(calibration_indexes) < 512:
        raise ValueError("deployment refit train/calibration segments are too small")

    x = np.asarray(dataset.features, dtype=np.float32)
    backend_parameters, backend_kind, backend_device = _backend_parameters(
        compute_backend,
        artifact.seed + 10_000,
    )
    parameters = {
        **backend_parameters,
        **_risk_parameters(artifact.risk_level, int(artifact.split.train_rows)),
    }
    targets = {"long": dataset.long_net_bps, "short": dataset.short_net_bps}
    eligible = {
        "long": np.asarray(dataset.long_liquidity_eligible, dtype=bool),
        "short": np.asarray(dataset.short_liquidity_eligible, dtype=bool),
    }
    deployment_models: dict[str, lgb.Booster] = {}
    deployment_calibration: dict[str, tuple[float, float]] = {}
    side_training_rows: dict[str, int] = {}
    side_calibration_rows: dict[str, int] = {}
    completed = 0
    total_steps = 8
    for side, target in targets.items():
        provisional_side = provisional_indexes[eligible[side][provisional_indexes]]
        calibration_side = calibration_indexes[eligible[side][calibration_indexes]]
        final_side = np.flatnonzero(eligible[side]).astype(np.int64)
        provisional_labels = (target[provisional_side] > 0.0).astype(np.float32)
        calibration_labels = (target[calibration_side] > 0.0).astype(np.float32)
        final_labels = (target[final_side] > 0.0).astype(np.float32)
        if min(
            int(np.sum(provisional_labels == 0.0)),
            int(np.sum(provisional_labels == 1.0)),
            int(np.sum(calibration_labels == 0.0)),
            int(np.sum(calibration_labels == 1.0)),
            int(np.sum(final_labels == 0.0)),
            int(np.sum(final_labels == 1.0)),
        ) < 256:
            raise ValueError(f"deployment refit {side} classifier lacks class support")
        if progress:
            progress(f"refit-{side}-calibration-model", completed, total_steps)
        provisional_classifier = _train_fixed_booster(
            features=x[provisional_side],
            targets=provisional_labels,
            objective="binary",
            metric="binary_logloss",
            parameters=parameters,
            iterations=int(artifact.best_iterations[f"{side}_probability"]),
        )
        completed += 1
        raw_calibration = provisional_classifier.predict(x[calibration_side])
        deployment_calibration[side] = _fit_platt_scaling(
            raw_calibration,
            calibration_labels,
        )

        if progress:
            progress(f"refit-{side}-probability", completed, total_steps)
        deployment_models[f"{side}_probability"] = _train_fixed_booster(
            features=x[final_side],
            targets=final_labels,
            objective="binary",
            metric="binary_logloss",
            parameters=parameters,
            iterations=int(artifact.best_iterations[f"{side}_probability"]),
        )
        completed += 1
        for outcome, keep, magnitude in (
            ("win", final_labels == 1.0, target[final_side]),
            ("loss", final_labels == 0.0, -target[final_side]),
        ):
            if int(np.sum(keep)) < 256:
                raise ValueError(f"deployment refit {side} {outcome} regressor lacks support")
            if progress:
                progress(f"refit-{side}-{outcome}-magnitude", completed, total_steps)
            name = f"{side}_{outcome}_magnitude"
            deployment_models[name] = _train_fixed_booster(
                features=x[final_side][keep],
                targets=np.asarray(magnitude[keep], dtype=np.float32),
                objective="regression",
                metric="l2",
                parameters=parameters,
                iterations=int(artifact.best_iterations[name]),
            )
            completed += 1
        side_training_rows[side] = int(len(final_side))
        side_calibration_rows[side] = int(len(calibration_side))

    deployment_model_strings = {
        name: model.model_to_string(num_iteration=int(artifact.best_iterations[name]))
        for name, model in deployment_models.items()
    }
    training_cutoff_ms = int(times[-1])
    expires_at_ms = training_cutoff_ms + maximum_age_seconds * 1_000
    evidence = DeploymentRefitEvidence(
        refit_mode="full_history_fixed_hyperparameters",
        backend_kind=backend_kind,
        backend_device=backend_device,
        lightgbm_version=str(lgb.__version__),
        training_rows=int(dataset.rows),
        calibration_days=calibration_days,
        calibration_start_ms=calibration_start_ms,
        calibration_end_ms=int(times[calibration_indexes[-1]]),
        side_training_rows=side_training_rows,
        side_calibration_rows=side_calibration_rows,
        probability_calibration=deployment_calibration,
        training_cutoff_ms=training_cutoff_ms,
        maximum_model_age_seconds=maximum_age_seconds,
        expires_at_ms=expires_at_ms,
        source_feature_build_id=str(source["build_id"]),
        source_manifest_fingerprint=str(source["manifest_fingerprint"]),
        validation_model_sha256=_model_strings_sha256(artifact.model_strings),
        deployment_model_sha256=_model_strings_sha256(deployment_model_strings),
        fitted_at=datetime.now(timezone.utc).isoformat(),
    )
    if progress:
        progress("deployment-refit-complete", total_steps, total_steps)
    return replace(
        artifact,
        status="accepted",
        deployment_model_strings=deployment_model_strings,
        deployment_refit=evidence,
    )


def evaluate_microstructure_model_terminal(
    artifact: MicrostructureModelArtifact,
    dataset: MicrostructureDataset,
    *,
    progress: ModelProgressCallback | None = None,
) -> MicrostructureModelArtifact:
    """Evaluate one selected candidate exactly once on its terminal segment."""

    if artifact.schema_version != MICROSTRUCTURE_MODEL_SCHEMA_VERSION:
        raise ValueError("microstructure model schema is not supported")
    if artifact.status != "candidate" or artifact.rejection_reasons:
        raise ValueError("only an unrejected candidate can consume the terminal holdout")
    if artifact.terminal_evaluated_at is not None or artifact.terminal_metrics is not None:
        raise ValueError("terminal holdout has already been evaluated")
    _validated_prequential_binding(artifact)
    artifact_source = _validated_source_evidence(
        artifact.dataset_summary.get("source_evidence"),
        symbol=artifact.symbol,
    )
    dataset_source = _validated_source_evidence(dataset.source_evidence, symbol=dataset.symbol)
    if (
        artifact_source.get("build_id") != dataset_source.get("build_id")
        or artifact_source.get("manifest_fingerprint")
        != dataset_source.get("manifest_fingerprint")
    ):
        raise ValueError("terminal dataset source provenance does not match the candidate")
    expected_contract = (
        artifact.symbol,
        artifact.feature_version,
        artifact.feature_names,
        artifact.horizon_seconds,
        artifact.total_latency_ms,
        artifact.taker_fee_bps,
        artifact.reference_order_notional_quote,
        artifact.max_l1_participation,
        artifact.max_quote_age_ms,
        artifact.decision_cadence_seconds,
        artifact.target_mode,
        artifact.stop_loss_bps,
        artifact.take_profit_bps,
        artifact.trigger_execution_slippage_bps,
        artifact.path_resolution_ms,
        _artifact_int(
            artifact.dataset_summary.get("trade_feature_embargo_ms"),
            "trade feature embargo",
        ),
    )
    dataset_contract = (
        dataset.symbol,
        dataset.feature_version,
        dataset.feature_names,
        dataset.horizon_seconds,
        dataset.total_latency_ms,
        dataset.taker_fee_bps,
        dataset.reference_order_notional_quote,
        dataset.max_l1_participation,
        dataset.max_quote_age_ms,
        dataset.decision_cadence_seconds,
        dataset.target_mode,
        dataset.stop_loss_bps,
        dataset.take_profit_bps,
        dataset.trigger_execution_slippage_bps,
        dataset.path_resolution_ms,
        dataset.trade_feature_embargo_ms,
    )
    if dataset_contract != expected_contract:
        raise ValueError("terminal dataset does not match the trained model contract")
    splits, split_evidence = _purged_split(dataset)
    if split_evidence != artifact.split:
        raise ValueError("terminal dataset does not reproduce the recorded purged split")

    x = np.asarray(dataset.features, dtype=np.float32)
    indexes = splits["terminal"]
    try:
        models = {
            name: lgb.Booster(model_str=artifact.model_strings[name])
            for side in ("long", "short")
            for name in (
                f"{side}_probability",
                f"{side}_win_magnitude",
                f"{side}_loss_magnitude",
            )
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidate artifact contains invalid model payloads") from exc
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in ("long", "short"):
        calibration_values = artifact.probability_calibration.get(side)
        if calibration_values is None or len(calibration_values) != 2:
            raise ValueError(f"candidate artifact is missing {side} probability calibration")
        probability = _apply_platt_scaling(
            models[f"{side}_probability"].predict(x[indexes]),
            (float(calibration_values[0]), float(calibration_values[1])),
        )
        win = np.maximum(0.0, models[f"{side}_win_magnitude"].predict(x[indexes]))
        loss = np.maximum(0.0, models[f"{side}_loss_magnitude"].predict(x[indexes]))
        predictions[side] = (probability * win - (1.0 - probability) * loss, probability)
    terminal_trace = _simulate_non_overlapping_trace(
        timestamps=dataset.decision_time_ms[indexes],
        long_exit_times=dataset.long_exit_time_ms[indexes],
        short_exit_times=dataset.short_exit_time_ms[indexes],
        long_targets=dataset.long_net_bps[indexes],
        short_targets=dataset.short_net_bps[indexes],
        long_edge=predictions["long"][0],
        short_edge=predictions["short"][0],
        long_probability=predictions["long"][1],
        short_probability=predictions["short"][1],
        edge_threshold=artifact.threshold_policy.minimum_predicted_edge_bps,
        probability_threshold=artifact.threshold_policy.minimum_profitable_probability,
        long_eligible=dataset.long_liquidity_eligible[indexes],
        short_eligible=dataset.short_liquidity_eligible[indexes],
    )
    terminal_metrics = terminal_trace.metrics
    terminal_confidence = _performance_confidence(
        terminal_trace,
        dataset.decision_time_ms[indexes],
    )
    terminal_auc, terminal_brier = _side_probability_quality(
        dataset,
        indexes,
        predictions["long"][1],
        predictions["short"][1],
    )
    terminal_baselines = _baseline_metrics(dataset, indexes)
    reasons: list[str] = []
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
        reasons.append("terminal_protective_exit_contract_missing_or_invalid")
    if artifact.unique_utc_days < artifact.minimum_promotion_days:
        reasons.append(
            f"unique_utc_days={artifact.unique_utc_days}<{artifact.minimum_promotion_days}"
        )
    if artifact.calendar_day_coverage_ratio < 0.95:
        reasons.append(
            "calendar_day_coverage_ratio="
            f"{artifact.calendar_day_coverage_ratio:.6f}<0.950000"
        )
    expected_usable_rows = max(
        1,
        (
            86_400
            - 900
            - artifact.horizon_seconds
            - math.ceil(artifact.total_latency_ms / 1000)
        )
        // artifact.decision_cadence_seconds,
    )
    minimum_dense_day_rows = int(expected_usable_rows * 0.80)
    if artifact.daily_rows_p10 < minimum_dense_day_rows:
        reasons.append(
            f"daily_rows_p10={artifact.daily_rows_p10:.1f}<{minimum_dense_day_rows}"
        )
    if terminal_metrics.trades < _minimum_evaluation_trades(dataset.decision_time_ms[indexes]):
        reasons.append("terminal_trade_count_below_statistical_minimum")
    if terminal_metrics.total_net_bps <= 0.0 or _risk_utility(
        terminal_metrics, artifact.risk_level
    ) <= 0.0:
        reasons.append("terminal_not_profitable_after_drawdown_penalty")
    if terminal_metrics.profit_factor is None or terminal_metrics.profit_factor <= 1.0:
        reasons.append("terminal_profit_factor_not_above_one")
    if terminal_confidence.mean_daily_net_bps_ci_lower <= 0.0:
        reasons.append("terminal_daily_edge_lower_confidence_bound_not_positive")
    if terminal_metrics.total_net_bps <= max(
        item.total_net_bps for item in terminal_baselines.values()
    ):
        reasons.append("terminal_not_above_directional_baselines")
    if terminal_metrics.long_trades and terminal_auc["long"] <= 0.5:
        reasons.append("terminal_long_action_auc_not_above_random")
    if terminal_metrics.short_trades and terminal_auc["short"] <= 0.5:
        reasons.append("terminal_short_action_auc_not_above_random")
    if progress:
        progress("terminal-evaluation", 6, 6)
    return replace(
        artifact,
        status="validated" if not reasons else "rejected",
        rejection_reasons=tuple(reasons),
        terminal_auc=terminal_auc,
        terminal_brier=terminal_brier,
        terminal_metrics=terminal_metrics,
        terminal_confidence=terminal_confidence,
        terminal_baselines=terminal_baselines,
        terminal_evaluated_at=datetime.now(timezone.utc).isoformat(),
    )


def save_microstructure_model_artifact(
    artifact: MicrostructureModelArtifact,
    path: str | Path,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(artifact.asdict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return digest


__all__ = [
    "MICROSTRUCTURE_MODEL_SCHEMA_VERSION",
    "MICROSTRUCTURE_PREQUENTIAL_EVIDENCE_VERSION",
    "DeploymentRefitEvidence",
    "MicrostructureActionPrediction",
    "MicrostructureActionScorer",
    "MicrostructureModelExpiredError",
    "MicrostructureModelArtifact",
    "PurgedSplitEvidence",
    "PrequentialValidationEvidence",
    "ThresholdPolicy",
    "ThresholdSearchEvidence",
    "TradingMetrics",
    "concatenate_microstructure_datasets",
    "evaluate_microstructure_model_terminal",
    "load_microstructure_action_scorer",
    "load_microstructure_model_artifact",
    "microstructure_candidate_sha256",
    "refit_validated_microstructure_model",
    "save_microstructure_model_artifact",
    "train_microstructure_action_value_model",
]
