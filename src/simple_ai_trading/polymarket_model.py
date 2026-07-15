"""Leakage-safe market-anchored probability research for Polymarket 5-minute data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

from .assets import SUPPORTED_MAJOR_BASE_ASSETS
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_features import (
    POLYMARKET_FEATURE_NAMES,
    PolymarketFeatureDataset,
    PolymarketFeatureRow,
)


POLYMARKET_MODEL_SAMPLE_SCHEMA_VERSION = "polymarket-model-sample-v4"
POLYMARKET_INFERENCE_INPUT_SCHEMA_VERSION = "polymarket-inference-input-v1"
POLYMARKET_MODEL_DATASET_SCHEMA_VERSION = "polymarket-model-dataset-v4"
POLYMARKET_OFFSET_MODEL_SCHEMA_VERSION = "polymarket-market-anchored-logit-v4"
POLYMARKET_MODEL_SPLIT_SCHEMA_VERSION = "polymarket-purged-time-split-v1"
POLYMARKET_MODEL_REPORT_SCHEMA_VERSION = "polymarket-probability-report-v2"
POLYMARKET_PROFILE_MODEL_SCHEMA_VERSION = (
    "polymarket-market-anchored-profile-logit-v1"
)
POLYMARKET_PROFILE_REPORT_SCHEMA_VERSION = (
    "polymarket-profile-challenger-report-v1"
)
POLYMARKET_PROFILE_CHALLENGER_SCHEMA_VERSION = (
    "polymarket-profile-challenger-assessment-v1"
)
POLYMARKET_PROFILE_CONTRACT_SHA256 = (
    "ae983f4f0cfbcaa130cc5a7d4ec0d3b08cb240ee631e44921b869e69508974ec"
)
POLYMARKET_LIVE_INFERENCE_CONTRACT_SHA256 = (
    "9e9a139cc28d988a5f45ed160a00cc3ac363657c41e3394d1169031164fc2eb3"
)
POLYMARKET_PROFILE_L2_CANDIDATES = (0.001, 0.01, 0.1, 1.0, 10.0)

POLYMARKET_MODEL_FEATURE_NAMES = (
    "remaining_seconds",
    "direct_distance_from_chainlink_open_bps",
    "direct_chainlink_basis_bps",
    "direct_return_100ms_bps",
    "direct_return_250ms_bps",
    "direct_return_1000ms_bps",
    "direct_return_5000ms_bps",
    "direct_realized_volatility_100ms_bps",
    "direct_realized_volatility_1000ms_bps",
    "direct_realized_volatility_5000ms_bps",
    "direct_diffusion_market_logit_gap",
    "chainlink_diffusion_market_logit_gap",
    "direct_trade_imbalance_100ms",
    "direct_trade_imbalance_250ms",
    "direct_trade_imbalance_1000ms",
    "direct_trade_imbalance_5000ms",
    "direct_top_imbalance",
    "direct_spread_bps",
    "up_microprice_deviation_bps",
    "down_microprice_deviation_bps",
    "up_top_imbalance",
    "down_top_imbalance",
    "outcome_midpoint_sum_error_bps",
    "executable_ask_pair_premium_bps",
    "executable_bid_pair_discount_bps",
    "asset_is_eth",
    "asset_is_sol",
)
POLYMARKET_MODEL_RISK_CONTEXT_NAMES = (
    "up_book_age_ms",
    "down_book_age_ms",
    "direct_binance_age_ms",
    "chainlink_source_age_ms",
    "chainlink_arrival_age_ms",
    "chainlink_anchor_gap_ms",
    "up_bid_depth_3_contracts",
    "up_ask_depth_3_contracts",
    "down_bid_depth_3_contracts",
    "down_ask_depth_3_contracts",
    "log1p_market_liquidity_quote",
    "log1p_market_volume_quote",
)
POLYMARKET_PROFILE_FEATURES = (
    (
        "diffusion_core",
        (
            "remaining_seconds",
            "direct_distance_from_chainlink_open_bps",
            "direct_chainlink_basis_bps",
            "direct_realized_volatility_100ms_bps",
            "direct_realized_volatility_1000ms_bps",
            "direct_realized_volatility_5000ms_bps",
            "direct_diffusion_market_logit_gap",
            "chainlink_diffusion_market_logit_gap",
            "direct_spread_bps",
            "asset_is_eth",
            "asset_is_sol",
        ),
    ),
    (
        "fast_cross_venue_flow",
        (
            "remaining_seconds",
            "direct_distance_from_chainlink_open_bps",
            "direct_chainlink_basis_bps",
            "direct_return_100ms_bps",
            "direct_return_250ms_bps",
            "direct_return_1000ms_bps",
            "direct_return_5000ms_bps",
            "direct_realized_volatility_100ms_bps",
            "direct_realized_volatility_1000ms_bps",
            "direct_realized_volatility_5000ms_bps",
            "direct_trade_imbalance_100ms",
            "direct_trade_imbalance_250ms",
            "direct_trade_imbalance_1000ms",
            "direct_trade_imbalance_5000ms",
            "direct_top_imbalance",
            "direct_spread_bps",
            "asset_is_eth",
            "asset_is_sol",
        ),
    ),
    ("full", POLYMARKET_MODEL_FEATURE_NAMES),
    (
        "prediction_book_state",
        (
            "remaining_seconds",
            "direct_distance_from_chainlink_open_bps",
            "direct_chainlink_basis_bps",
            "direct_diffusion_market_logit_gap",
            "chainlink_diffusion_market_logit_gap",
            "up_microprice_deviation_bps",
            "down_microprice_deviation_bps",
            "up_top_imbalance",
            "down_top_imbalance",
            "outcome_midpoint_sum_error_bps",
            "executable_ask_pair_premium_bps",
            "executable_bid_pair_discount_bps",
            "asset_is_eth",
            "asset_is_sol",
        ),
    ),
)

_ASSETS = tuple(SUPPORTED_MAJOR_BASE_ASSETS)
_FEATURE_INDEX = {name: index for index, name in enumerate(POLYMARKET_FEATURE_NAMES)}


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


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _format_floats(values: Sequence[float]) -> list[str]:
    return [format(_finite(value, name="model value"), ".17g") for value in values]


@dataclass(frozen=True)
class PolymarketModelConfig:
    """Precommitted sampling, split, and regularization contract."""

    decision_horizons_seconds: tuple[int, ...] = (240, 180, 120, 60, 30)
    maximum_horizon_error_ms: int = 1_000
    minimum_markets_per_asset: int = 30
    minimum_time_groups: int = 30
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    purge_time_groups: int = 1
    minimum_train_time_groups: int = 16
    minimum_validation_time_groups: int = 5
    minimum_test_time_groups: int = 5
    minimum_outcome_markets_per_split: int = 2
    inner_fold_count: int = 3
    inner_validation_time_groups: int = 2
    inner_purge_time_groups: int = 1
    minimum_inner_train_time_groups: int = 8
    l2_candidates: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    minimum_validation_log_loss_improvement: float = 0.0001
    maximum_absolute_logit_correction: float = 2.0
    maximum_iterations: int = 100
    convergence_tolerance: float = 1e-9

    def validated(self) -> "PolymarketModelConfig":
        horizons = tuple(int(value) for value in self.decision_horizons_seconds)
        l2_values = tuple(float(value) for value in self.l2_candidates)
        fractions = (float(self.validation_fraction), float(self.test_fraction))
        if (
            not horizons
            or horizons != tuple(sorted(set(horizons), reverse=True))
            or any(value < 5 or value > 295 for value in horizons)
            or not 50 <= int(self.maximum_horizon_error_ms) <= 10_000
            or not 3 <= int(self.minimum_markets_per_asset) <= 100_000
            or not 10 <= int(self.minimum_time_groups) <= 100_000
            or any(not math.isfinite(value) or not 0.05 <= value <= 0.35 for value in fractions)
            or sum(fractions) > 0.60
            or not 1 <= int(self.purge_time_groups) <= 100
            or not 5 <= int(self.minimum_train_time_groups) <= 100_000
            or not 2 <= int(self.minimum_validation_time_groups) <= 100_000
            or not 2 <= int(self.minimum_test_time_groups) <= 100_000
            or not 1 <= int(self.minimum_outcome_markets_per_split) <= 100_000
            or not 2 <= int(self.inner_fold_count) <= 8
            or not 1 <= int(self.inner_validation_time_groups) <= 20
            or not 1 <= int(self.inner_purge_time_groups) <= 20
            or not 5 <= int(self.minimum_inner_train_time_groups) <= 100_000
            or int(self.minimum_train_time_groups)
            < int(self.minimum_inner_train_time_groups)
            + int(self.inner_fold_count) * int(self.inner_validation_time_groups)
            + int(self.inner_purge_time_groups)
            or not l2_values
            or l2_values != tuple(sorted(set(l2_values)))
            or any(not math.isfinite(value) or value <= 0.0 for value in l2_values)
            or not math.isfinite(float(self.minimum_validation_log_loss_improvement))
            or not 0.0 <= float(self.minimum_validation_log_loss_improvement) <= 0.1
            or not math.isfinite(float(self.maximum_absolute_logit_correction))
            or not 0.25 <= float(self.maximum_absolute_logit_correction) <= 5.0
            or not 10 <= int(self.maximum_iterations) <= 1_000
            or not math.isfinite(float(self.convergence_tolerance))
            or not 1e-12 <= float(self.convergence_tolerance) <= 1e-3
        ):
            raise ValueError("Polymarket model configuration is invalid")
        return self

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["decision_horizons_seconds"] = list(self.decision_horizons_seconds)
        payload["l2_candidates"] = list(self.l2_candidates)
        return payload


@dataclass(frozen=True)
class PolymarketModelSample:
    """One causal decision point; its label is used only after time splitting."""

    sample_id: str
    source_run_id: str
    source_feature_id: str
    condition_id: str
    market_id: str
    asset: str
    event_start_ms: int
    end_ms: int
    decision_received_wall_ms: int
    decision_received_monotonic_ns: int
    decision_event_id: str
    horizon_seconds: int
    feature_values: tuple[float, ...]
    risk_context_values: tuple[float, ...]
    baseline_up_probability: float
    up_best_bid: float
    up_best_ask: float
    down_best_bid: float
    down_best_ask: float
    official_up: bool
    resolution_event_id: str
    market_weight: float
    input_provenance_sha256: str
    sample_sha256: str

    def feature_map(self) -> dict[str, float]:
        return dict(zip(POLYMARKET_MODEL_FEATURE_NAMES, self.feature_values, strict=True))

    def risk_context_map(self) -> dict[str, float]:
        return dict(
            zip(
                POLYMARKET_MODEL_RISK_CONTEXT_NAMES,
                self.risk_context_values,
                strict=True,
            )
        )

    def identity_payload(self) -> dict[str, object]:
        return _sample_payload(self)

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "sample_sha256": self.sample_sha256,
        }

    def validated(self) -> "PolymarketModelSample":
        if (
            len(self.feature_values) != len(POLYMARKET_MODEL_FEATURE_NAMES)
            or len(self.risk_context_values)
            != len(POLYMARKET_MODEL_RISK_CONTEXT_NAMES)
            or not all(math.isfinite(value) for value in self.feature_values)
            or not all(
                math.isfinite(value) and value >= 0.0
                for value in self.risk_context_values
            )
            or not 0.0 < self.baseline_up_probability < 1.0
            or len(self.sample_sha256) != 64
            or self.sample_sha256 != _canonical_sha256(_sample_payload(self))
        ):
            raise ValueError("Polymarket model sample identity is invalid")
        return self


@dataclass(frozen=True)
class PolymarketInferenceInput:
    """Hash-bound causal model input that cannot carry a future label."""

    input_id: str
    source_run_id: str
    source_feature_id: str
    source_row_sha256: str
    model_config_sha256: str
    live_inference_contract_sha256: str
    condition_id: str
    market_id: str
    asset: str
    event_start_ms: int
    end_ms: int
    decision_received_wall_ms: int
    decision_received_monotonic_ns: int
    decision_event_id: str
    horizon_seconds: int
    feature_values: tuple[float, ...]
    risk_context_values: tuple[float, ...]
    baseline_up_probability: float
    up_best_bid: float
    up_best_ask: float
    down_best_bid: float
    down_best_ask: float
    input_provenance_sha256: str
    input_sha256: str

    def feature_map(self) -> dict[str, float]:
        return dict(zip(POLYMARKET_MODEL_FEATURE_NAMES, self.feature_values, strict=True))

    def risk_context_map(self) -> dict[str, float]:
        return dict(
            zip(
                POLYMARKET_MODEL_RISK_CONTEXT_NAMES,
                self.risk_context_values,
                strict=True,
            )
        )

    def identity_payload(self) -> dict[str, object]:
        return _inference_input_payload(self)

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "input_sha256": self.input_sha256}

    def validated(self) -> "PolymarketInferenceInput":
        if (
            len(self.input_id) != 64
            or len(self.source_row_sha256) != 64
            or len(self.model_config_sha256) != 64
            or self.live_inference_contract_sha256
            != POLYMARKET_LIVE_INFERENCE_CONTRACT_SHA256
            or len(self.input_provenance_sha256) != 64
            or self.asset not in _ASSETS
            or self.end_ms <= self.event_start_ms
            or not (
                self.event_start_ms
                <= self.decision_received_wall_ms
                < self.end_ms
            )
            or len(self.feature_values) != len(POLYMARKET_MODEL_FEATURE_NAMES)
            or len(self.risk_context_values)
            != len(POLYMARKET_MODEL_RISK_CONTEXT_NAMES)
            or not all(math.isfinite(value) for value in self.feature_values)
            or not all(
                math.isfinite(value) and value >= 0.0
                for value in self.risk_context_values
            )
            or not 0.0 < self.baseline_up_probability < 1.0
            or self.input_id
            != _canonical_sha256(
                {
                    "schema_version": POLYMARKET_INFERENCE_INPUT_SCHEMA_VERSION,
                    "source_row_sha256": self.source_row_sha256,
                    "horizon_seconds": self.horizon_seconds,
                    "model_config_sha256": self.model_config_sha256,
                    "live_inference_contract_sha256": (
                        self.live_inference_contract_sha256
                    ),
                }
            )
            or self.input_sha256 != _canonical_sha256(_inference_input_payload(self))
        ):
            raise ValueError("Polymarket inference input identity is invalid")
        return self


PolymarketPredictable = PolymarketModelSample | PolymarketInferenceInput


@dataclass(frozen=True)
class PolymarketModelDataset:
    schema_version: str
    source_dataset_id: str
    source_dataset_sha256: str
    source_run_id: str
    config: PolymarketModelConfig
    samples: tuple[PolymarketModelSample, ...]
    market_counts: Mapping[str, int]
    time_group_count: int
    skipped_counts: Mapping[str, int]
    training_errors: tuple[str, ...]
    dataset_sha256: str

    @property
    def training_ready(self) -> bool:
        return not self.training_errors

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_dataset_id": self.source_dataset_id,
            "source_dataset_sha256": self.source_dataset_sha256,
            "source_run_id": self.source_run_id,
            "config": self.config.asdict(),
            "model_feature_names": list(POLYMARKET_MODEL_FEATURE_NAMES),
            "risk_context_names": list(POLYMARKET_MODEL_RISK_CONTEXT_NAMES),
            "sample_count": len(self.samples),
            "market_counts": dict(self.market_counts),
            "time_group_count": self.time_group_count,
            "skipped_counts": dict(self.skipped_counts),
            "training_ready": self.training_ready,
            "training_errors": list(self.training_errors),
            "dataset_sha256": self.dataset_sha256,
        }


@dataclass(frozen=True)
class PolymarketModelSplit:
    schema_version: str
    source_dataset_sha256: str
    train: tuple[PolymarketModelSample, ...]
    validation: tuple[PolymarketModelSample, ...]
    test: tuple[PolymarketModelSample, ...]
    train_group_starts_ms: tuple[int, ...]
    validation_group_starts_ms: tuple[int, ...]
    test_group_starts_ms: tuple[int, ...]
    purged_group_starts_ms: tuple[int, ...]
    split_sha256: str

    def samples_for(self, role: str) -> tuple[PolymarketModelSample, ...]:
        if role not in {"train", "validation", "test"}:
            raise ValueError("unknown Polymarket split role")
        return tuple(getattr(self, role))

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_dataset_sha256": self.source_dataset_sha256,
            "sample_counts": {
                role: len(self.samples_for(role))
                for role in ("train", "validation", "test")
            },
            "market_counts": {
                role: len(
                    {
                        item.condition_id
                        for item in self.samples_for(role)
                    }
                )
                for role in ("train", "validation", "test")
            },
            "train_group_starts_ms": list(self.train_group_starts_ms),
            "validation_group_starts_ms": list(
                self.validation_group_starts_ms
            ),
            "test_group_starts_ms": list(self.test_group_starts_ms),
            "purged_group_starts_ms": list(self.purged_group_starts_ms),
            "split_sha256": self.split_sha256,
        }


@dataclass(frozen=True)
class PolymarketProbabilityMetrics:
    row_count: int
    market_count: int
    time_group_count: int
    effective_market_weight: float
    weighted_log_loss: float
    weighted_brier_score: float
    weighted_calibration_error: float
    weighted_accuracy: float
    weighted_sharpness: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainedPolymarketOffsetModel:
    """Hash-bound research model with no order or profitability authority."""

    schema_version: str
    source_dataset_sha256: str
    source_split_sha256: str
    config: PolymarketModelConfig
    feature_names: tuple[str, ...]
    winsor_lower: tuple[float, ...]
    winsor_upper: tuple[float, ...]
    robust_center: tuple[float, ...]
    robust_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    selected_candidate: str
    selected_l2: float | None
    inner_selected_candidate: str
    candidate_inner_log_losses: tuple[tuple[str, float], ...]
    validation_gate_log_losses: tuple[tuple[str, float], ...]
    inner_fold_count: int
    inner_fold_boundaries_ms: tuple[tuple[int, int, int, int], ...]
    training_sample_count: int
    training_market_count: int
    training_time_group_count: int
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def asdict(self) -> dict[str, object]:
        payload = _model_payload(self)
        payload["model_sha256"] = self.model_sha256
        return payload


@dataclass(frozen=True)
class PolymarketModelReport:
    schema_version: str
    source_dataset_sha256: str
    source_split_sha256: str
    model_sha256: str
    selected_candidate: str
    baseline_metrics: Mapping[str, PolymarketProbabilityMetrics]
    model_metrics: Mapping[str, PolymarketProbabilityMetrics]
    validation_log_loss_delta: float
    test_log_loss_delta: float
    test_brier_delta: float
    report_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_dataset_sha256": self.source_dataset_sha256,
            "source_split_sha256": self.source_split_sha256,
            "model_sha256": self.model_sha256,
            "selected_candidate": self.selected_candidate,
            "baseline_metrics": {
                key: value.asdict() for key, value in self.baseline_metrics.items()
            },
            "model_metrics": {
                key: value.asdict() for key, value in self.model_metrics.items()
            },
            "validation_log_loss_delta": self.validation_log_loss_delta,
            "test_log_loss_delta": self.test_log_loss_delta,
            "test_brier_delta": self.test_brier_delta,
            "report_sha256": self.report_sha256,
            "trading_authority": self.trading_authority,
            "execution_claim": self.execution_claim,
            "profitability_claim": self.profitability_claim,
            "portfolio_claim": self.portfolio_claim,
            "leverage_applied": self.leverage_applied,
        }


@dataclass(frozen=True)
class TrainedPolymarketProfileModel:
    """Hash-bound reduced-profile challenger with no order authority."""

    schema_version: str
    contract_sha256: str
    source_dataset_sha256: str
    source_split_sha256: str
    control_model_sha256: str
    config: PolymarketModelConfig
    feature_names: tuple[str, ...]
    selected_profile: str | None
    selected_feature_names: tuple[str, ...]
    winsor_lower: tuple[float, ...]
    winsor_upper: tuple[float, ...]
    robust_center: tuple[float, ...]
    robust_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    selected_candidate: str
    selected_l2: float | None
    inner_selected_candidate: str
    candidate_inner_log_losses: tuple[tuple[str, float], ...]
    validation_gate_log_losses: tuple[tuple[str, float], ...]
    inner_fold_count: int
    inner_fold_boundaries_ms: tuple[tuple[int, int, int, int], ...]
    training_sample_count: int
    training_market_count: int
    training_time_group_count: int
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def asdict(self) -> dict[str, object]:
        payload = _profile_model_payload(self)
        payload["model_sha256"] = self.model_sha256
        return payload


@dataclass(frozen=True)
class PolymarketProfileModelReport:
    """Proper-score comparison of the frozen profile challenger and control."""

    schema_version: str
    contract_sha256: str
    source_dataset_sha256: str
    source_split_sha256: str
    control_model_sha256: str
    challenger_model_sha256: str
    selected_candidate: str
    baseline_metrics: Mapping[str, PolymarketProbabilityMetrics]
    control_metrics: Mapping[str, PolymarketProbabilityMetrics]
    challenger_metrics: Mapping[str, PolymarketProbabilityMetrics]
    validation_log_loss_delta_vs_control: float
    test_log_loss_delta_vs_control: float
    test_brier_delta_vs_control: float
    report_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_sha256": self.contract_sha256,
            "source_dataset_sha256": self.source_dataset_sha256,
            "source_split_sha256": self.source_split_sha256,
            "control_model_sha256": self.control_model_sha256,
            "challenger_model_sha256": self.challenger_model_sha256,
            "selected_candidate": self.selected_candidate,
            "baseline_metrics": {
                key: value.asdict() for key, value in self.baseline_metrics.items()
            },
            "control_metrics": {
                key: value.asdict() for key, value in self.control_metrics.items()
            },
            "challenger_metrics": {
                key: value.asdict() for key, value in self.challenger_metrics.items()
            },
            "validation_log_loss_delta_vs_control": (
                self.validation_log_loss_delta_vs_control
            ),
            "test_log_loss_delta_vs_control": self.test_log_loss_delta_vs_control,
            "test_brier_delta_vs_control": self.test_brier_delta_vs_control,
            "report_sha256": self.report_sha256,
            "trading_authority": self.trading_authority,
            "execution_claim": self.execution_claim,
            "profitability_claim": self.profitability_claim,
            "portfolio_claim": self.portfolio_claim,
            "leverage_applied": self.leverage_applied,
        }


def _row_map(row: PolymarketFeatureRow) -> dict[str, float]:
    if len(row.feature_values) != len(POLYMARKET_FEATURE_NAMES):
        raise ValueError("Polymarket feature row width is invalid")
    values = tuple(_finite(value, name="Polymarket feature") for value in row.feature_values)
    return dict(zip(POLYMARKET_FEATURE_NAMES, values, strict=True))


def _relative_bps(value: float, reference: float, *, name: str) -> float:
    if reference <= 0.0:
        raise ValueError(f"{name} reference must be positive")
    return 10_000.0 * (value - reference) / reference


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("logit probability must lie inside (0, 1)")
    return math.log(probability) - math.log1p(-probability)


def _diffusion_market_logit_gap(
    *,
    distance_from_open_bps: float,
    realized_volatility_5000ms_bps: float,
    remaining_seconds: float,
    market_probability: float,
) -> float:
    """Return a bounded driftless-diffusion proxy minus the market logit."""

    if realized_volatility_5000ms_bps < 0.0 or remaining_seconds <= 0.0:
        raise ValueError("diffusion proxy inputs are outside their domains")
    projected_volatility_bps = max(realized_volatility_5000ms_bps, 1e-6) * math.sqrt(
        remaining_seconds / 5.0
    )
    z_score = max(
        -8.0,
        min(8.0, distance_from_open_bps / projected_volatility_bps),
    )
    diffusion_probability = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    diffusion_probability = max(1e-6, min(1.0 - 1e-6, diffusion_probability))
    return max(
        -12.0,
        min(12.0, _logit(diffusion_probability) - _logit(market_probability)),
    )


def build_polymarket_model_features(
    values: Mapping[str, float],
    asset: str,
    *,
    baseline_up_probability: float,
) -> tuple[float, ...]:
    up_midpoint = values["up_midpoint"]
    down_midpoint = values["down_midpoint"]
    direct_distance = values["binance_distance_from_chainlink_open_bps"]
    direct_chainlink_basis = values["binance_chainlink_basis_bps"]
    remaining_seconds = values["remaining_seconds"]
    volatility_5000ms = values["binance_realized_volatility_5000ms_bps"]
    derived = (
        remaining_seconds,
        direct_distance,
        direct_chainlink_basis,
        values["binance_return_100ms_bps"],
        values["binance_return_250ms_bps"],
        values["binance_return_1000ms_bps"],
        values["binance_return_5000ms_bps"],
        values["binance_realized_volatility_100ms_bps"],
        values["binance_realized_volatility_1000ms_bps"],
        volatility_5000ms,
        _diffusion_market_logit_gap(
            distance_from_open_bps=direct_distance,
            realized_volatility_5000ms_bps=volatility_5000ms,
            remaining_seconds=remaining_seconds,
            market_probability=baseline_up_probability,
        ),
        _diffusion_market_logit_gap(
            distance_from_open_bps=direct_distance - direct_chainlink_basis,
            realized_volatility_5000ms_bps=volatility_5000ms,
            remaining_seconds=remaining_seconds,
            market_probability=baseline_up_probability,
        ),
        values["binance_trade_imbalance_100ms"],
        values["binance_trade_imbalance_250ms"],
        values["binance_trade_imbalance_1000ms"],
        values["binance_trade_imbalance_5000ms"],
        values["binance_top_imbalance"],
        values["binance_spread_bps"],
        _relative_bps(
            values["up_microprice"], up_midpoint, name="Up microprice"
        ),
        _relative_bps(
            values["down_microprice"], down_midpoint, name="Down microprice"
        ),
        values["up_top_imbalance"],
        values["down_top_imbalance"],
        10_000.0 * (up_midpoint + down_midpoint - 1.0),
        10_000.0 * (values["ask_pair_cost"] - 1.0),
        10_000.0 * (1.0 - values["bid_pair_value"]),
        1.0 if asset == "ETH" else 0.0,
        1.0 if asset == "SOL" else 0.0,
    )
    if len(derived) != len(POLYMARKET_MODEL_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in derived
    ):
        raise ValueError("Polymarket model feature vector is invalid")
    return tuple(float(value) for value in derived)


def build_polymarket_risk_context(
    values: Mapping[str, float],
) -> tuple[float, ...]:
    context = (
        values["up_book_age_ms"],
        values["down_book_age_ms"],
        values["direct_binance_age_ms"],
        values["chainlink_source_age_ms"],
        values["chainlink_arrival_age_ms"],
        values["chainlink_anchor_gap_ms"],
        values["up_bid_depth_3"],
        values["up_ask_depth_3"],
        values["down_bid_depth_3"],
        values["down_ask_depth_3"],
        values["log1p_market_liquidity_quote"],
        values["log1p_market_volume_quote"],
    )
    if len(context) != len(POLYMARKET_MODEL_RISK_CONTEXT_NAMES) or not all(
        math.isfinite(value) and value >= 0.0 for value in context
    ):
        raise ValueError("Polymarket model risk context is invalid")
    return tuple(float(value) for value in context)


def _sample_payload(sample: PolymarketModelSample) -> dict[str, object]:
    return {
        "schema_version": POLYMARKET_MODEL_SAMPLE_SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "source_run_id": sample.source_run_id,
        "source_feature_id": sample.source_feature_id,
        "condition_id": sample.condition_id,
        "market_id": sample.market_id,
        "asset": sample.asset,
        "event_start_ms": sample.event_start_ms,
        "end_ms": sample.end_ms,
        "decision_received_wall_ms": sample.decision_received_wall_ms,
        "decision_received_monotonic_ns": sample.decision_received_monotonic_ns,
        "decision_event_id": sample.decision_event_id,
        "horizon_seconds": sample.horizon_seconds,
        "feature_names": list(POLYMARKET_MODEL_FEATURE_NAMES),
        "feature_values": _format_floats(sample.feature_values),
        "risk_context_names": list(POLYMARKET_MODEL_RISK_CONTEXT_NAMES),
        "risk_context_values": _format_floats(sample.risk_context_values),
        "baseline_up_probability": format(sample.baseline_up_probability, ".17g"),
        "up_best_bid": format(sample.up_best_bid, ".17g"),
        "up_best_ask": format(sample.up_best_ask, ".17g"),
        "down_best_bid": format(sample.down_best_bid, ".17g"),
        "down_best_ask": format(sample.down_best_ask, ".17g"),
        "official_up": sample.official_up,
        "resolution_event_id": sample.resolution_event_id,
        "market_weight": format(sample.market_weight, ".17g"),
        "input_provenance_sha256": sample.input_provenance_sha256,
    }


def _inference_input_payload(
    model_input: PolymarketInferenceInput,
) -> dict[str, object]:
    return {
        "schema_version": POLYMARKET_INFERENCE_INPUT_SCHEMA_VERSION,
        "input_id": model_input.input_id,
        "source_run_id": model_input.source_run_id,
        "source_feature_id": model_input.source_feature_id,
        "source_row_sha256": model_input.source_row_sha256,
        "model_config_sha256": model_input.model_config_sha256,
        "live_inference_contract_sha256": (
            model_input.live_inference_contract_sha256
        ),
        "condition_id": model_input.condition_id,
        "market_id": model_input.market_id,
        "asset": model_input.asset,
        "event_start_ms": model_input.event_start_ms,
        "end_ms": model_input.end_ms,
        "decision_received_wall_ms": model_input.decision_received_wall_ms,
        "decision_received_monotonic_ns": (
            model_input.decision_received_monotonic_ns
        ),
        "decision_event_id": model_input.decision_event_id,
        "horizon_seconds": model_input.horizon_seconds,
        "feature_names": list(POLYMARKET_MODEL_FEATURE_NAMES),
        "feature_values": _format_floats(model_input.feature_values),
        "risk_context_names": list(POLYMARKET_MODEL_RISK_CONTEXT_NAMES),
        "risk_context_values": _format_floats(model_input.risk_context_values),
        "baseline_up_probability": format(
            model_input.baseline_up_probability,
            ".17g",
        ),
        "up_best_bid": format(model_input.up_best_bid, ".17g"),
        "up_best_ask": format(model_input.up_best_ask, ".17g"),
        "down_best_bid": format(model_input.down_best_bid, ".17g"),
        "down_best_ask": format(model_input.down_best_ask, ".17g"),
        "input_provenance_sha256": model_input.input_provenance_sha256,
    }


def build_polymarket_inference_input(
    row: PolymarketFeatureRow,
    market: PolymarketFiveMinuteMarket,
    *,
    config: PolymarketModelConfig | None = None,
) -> PolymarketInferenceInput:
    """Build one fixed-horizon scoring input without accepting future state."""

    cfg = (config or PolymarketModelConfig()).validated()
    row.validated()
    condition = row.condition_id.lower()
    if row.official_up is not None or row.resolution_event_id:
        raise ValueError("Polymarket live inference input must be label-free")
    if (
        market.condition_id.lower() != condition
        or market.market_id != row.market_id
        or market.asset != row.asset
    ):
        raise ValueError("Polymarket inference feature and market metadata disagree")
    if not (
        market.event_start_ms
        <= row.decision_received_wall_ms
        < market.end_ms
    ):
        raise ValueError("Polymarket inference row lies outside its market window")
    values = row.feature_map()
    remaining_ms = market.end_ms - row.decision_received_wall_ms
    if abs(values["remaining_seconds"] * 1_000.0 - remaining_ms) > 2.0:
        raise ValueError("Polymarket inference row has inconsistent remaining time")
    eligible_horizons = sorted(
        (
            abs(remaining_ms - horizon * 1_000),
            horizon,
        )
        for horizon in cfg.decision_horizons_seconds
        if abs(remaining_ms - horizon * 1_000) <= cfg.maximum_horizon_error_ms
    )
    if not eligible_horizons or (
        len(eligible_horizons) > 1
        and eligible_horizons[0][0] == eligible_horizons[1][0]
    ):
        raise ValueError("Polymarket inference row does not match one fixed horizon")
    horizon = eligible_horizons[0][1]
    midpoint_total = values["up_midpoint"] + values["down_midpoint"]
    if (
        midpoint_total <= 0.0
        or not 0.0 < values["up_best_bid"] < values["up_best_ask"] < 1.0
        or not 0.0 < values["down_best_bid"] < values["down_best_ask"] < 1.0
    ):
        raise ValueError("Polymarket inference row has non-executable outcome quotes")
    baseline = values["up_midpoint"] / midpoint_total
    if not 0.0 < baseline < 1.0:
        raise ValueError("Polymarket inference market probability is invalid")
    model_config_sha256 = _canonical_sha256(cfg.asdict())
    input_id = _canonical_sha256(
        {
            "schema_version": POLYMARKET_INFERENCE_INPUT_SCHEMA_VERSION,
            "source_row_sha256": row.row_sha256,
            "horizon_seconds": horizon,
            "model_config_sha256": model_config_sha256,
            "live_inference_contract_sha256": (
                POLYMARKET_LIVE_INFERENCE_CONTRACT_SHA256
            ),
        }
    )
    provisional = PolymarketInferenceInput(
        input_id=input_id,
        source_run_id=row.run_id,
        source_feature_id=row.feature_id,
        source_row_sha256=row.row_sha256,
        model_config_sha256=model_config_sha256,
        live_inference_contract_sha256=(
            POLYMARKET_LIVE_INFERENCE_CONTRACT_SHA256
        ),
        condition_id=condition,
        market_id=market.market_id,
        asset=market.asset,
        event_start_ms=market.event_start_ms,
        end_ms=market.end_ms,
        decision_received_wall_ms=row.decision_received_wall_ms,
        decision_received_monotonic_ns=row.decision_received_monotonic_ns,
        decision_event_id=row.decision_event_id,
        horizon_seconds=horizon,
        feature_values=build_polymarket_model_features(
            values,
            market.asset,
            baseline_up_probability=baseline,
        ),
        risk_context_values=build_polymarket_risk_context(values),
        baseline_up_probability=baseline,
        up_best_bid=values["up_best_bid"],
        up_best_ask=values["up_best_ask"],
        down_best_bid=values["down_best_bid"],
        down_best_ask=values["down_best_ask"],
        input_provenance_sha256=row.input_provenance_sha256,
        input_sha256="",
    )
    return replace(
        provisional,
        input_sha256=_canonical_sha256(_inference_input_payload(provisional)),
    ).validated()


def _increment(values: dict[str, int], key: str) -> None:
    values[key] = values.get(key, 0) + 1


def build_polymarket_model_dataset(
    source: PolymarketFeatureDataset,
    markets: Sequence[PolymarketFiveMinuteMarket],
    *,
    config: PolymarketModelConfig | None = None,
) -> PolymarketModelDataset:
    """Select fixed horizons without allowing one market to dominate by update rate."""

    cfg = (config or PolymarketModelConfig()).validated()
    if source.dataset_id != source.dataset_sha256 or len(source.dataset_sha256) != 64:
        raise ValueError("source Polymarket feature dataset identity is invalid")
    market_by_condition: dict[str, PolymarketFiveMinuteMarket] = {}
    for market in markets:
        condition = market.condition_id.lower()
        if condition in market_by_condition:
            raise ValueError("Polymarket model market metadata is duplicated")
        market_by_condition[condition] = market

    rows_by_condition: dict[str, list[PolymarketFeatureRow]] = {}
    for row in source.rows:
        if row.run_id != source.run_id:
            raise ValueError("Polymarket feature row belongs to another recorder run")
        if row.official_up is None:
            continue
        rows_by_condition.setdefault(row.condition_id.lower(), []).append(row)

    samples: list[PolymarketModelSample] = []
    skipped: dict[str, int] = {}
    for condition in sorted(rows_by_condition):
        market = market_by_condition.get(condition)
        if market is None:
            _increment(skipped, "missing_market_metadata")
            continue
        rows = rows_by_condition[condition]
        if any(row.asset != market.asset or row.market_id != market.market_id for row in rows):
            raise ValueError("Polymarket feature and market metadata disagree")
        outcomes = {bool(row.official_up) for row in rows}
        resolutions = {row.resolution_event_id for row in rows}
        if len(outcomes) != 1 or len(resolutions) != 1 or "" in resolutions:
            raise ValueError("Polymarket model market labels are inconsistent")

        candidates: list[tuple[int, PolymarketFeatureRow, dict[str, float]]] = []
        for row in rows:
            values = _row_map(row)
            if not (
                market.event_start_ms
                <= row.decision_received_wall_ms
                < market.end_ms
            ):
                raise ValueError("Polymarket model row lies outside its market window")
            remaining_ms = market.end_ms - row.decision_received_wall_ms
            if abs(values["remaining_seconds"] * 1_000.0 - remaining_ms) > 2.0:
                raise ValueError("Polymarket model row has inconsistent remaining time")
            candidates.append((remaining_ms, row, values))

        selected: list[tuple[int, PolymarketFeatureRow, dict[str, float]]] = []
        used_feature_ids: set[str] = set()
        for horizon in cfg.decision_horizons_seconds:
            target_ms = horizon * 1_000
            ranked = sorted(
                candidates,
                key=lambda item: (
                    abs(item[0] - target_ms),
                    item[1].decision_received_wall_ms,
                    item[1].decision_received_monotonic_ns,
                    item[1].feature_id,
                ),
            )
            chosen = next(
                (
                    item
                    for item in ranked
                    if item[1].feature_id not in used_feature_ids
                    and abs(item[0] - target_ms) <= cfg.maximum_horizon_error_ms
                ),
                None,
            )
            if chosen is None:
                selected = []
                break
            selected.append((horizon, chosen[1], chosen[2]))
            used_feature_ids.add(chosen[1].feature_id)
        if len(selected) != len(cfg.decision_horizons_seconds):
            _increment(skipped, "incomplete_fixed_horizons")
            continue

        market_weight = 1.0 / len(selected)
        for horizon, row, values in selected:
            midpoint_total = values["up_midpoint"] + values["down_midpoint"]
            if (
                midpoint_total <= 0.0
                or not 0.0 < values["up_best_bid"] < values["up_best_ask"] < 1.0
                or not 0.0 < values["down_best_bid"] < values["down_best_ask"] < 1.0
            ):
                raise ValueError("Polymarket model row has non-executable outcome quotes")
            baseline = values["up_midpoint"] / midpoint_total
            if not 0.0 < baseline < 1.0:
                raise ValueError("Polymarket normalized market probability is invalid")
            sample_id = _canonical_sha256(
                {
                    "source_dataset_sha256": source.dataset_sha256,
                    "source_feature_id": row.feature_id,
                    "horizon_seconds": horizon,
                    "config": cfg.asdict(),
                }
            )
            sample = PolymarketModelSample(
                sample_id=sample_id,
                source_run_id=source.run_id,
                source_feature_id=row.feature_id,
                condition_id=condition,
                market_id=market.market_id,
                asset=market.asset,
                event_start_ms=market.event_start_ms,
                end_ms=market.end_ms,
                decision_received_wall_ms=row.decision_received_wall_ms,
                decision_received_monotonic_ns=row.decision_received_monotonic_ns,
                decision_event_id=row.decision_event_id,
                horizon_seconds=horizon,
                feature_values=build_polymarket_model_features(
                    values,
                    market.asset,
                    baseline_up_probability=baseline,
                ),
                risk_context_values=build_polymarket_risk_context(values),
                baseline_up_probability=baseline,
                up_best_bid=values["up_best_bid"],
                up_best_ask=values["up_best_ask"],
                down_best_bid=values["down_best_bid"],
                down_best_ask=values["down_best_ask"],
                official_up=bool(row.official_up),
                resolution_event_id=row.resolution_event_id,
                market_weight=market_weight,
                input_provenance_sha256=row.input_provenance_sha256,
                sample_sha256="",
            )
            samples.append(
                replace(sample, sample_sha256=_canonical_sha256(_sample_payload(sample)))
            )

    samples.sort(
        key=lambda item: (
            item.event_start_ms,
            item.asset,
            item.condition_id,
            -item.horizon_seconds,
            item.sample_id,
        )
    )
    market_counts = {
        asset: len({item.condition_id for item in samples if item.asset == asset})
        for asset in _ASSETS
    }
    time_group_count = len({item.event_start_ms for item in samples})
    errors = list(source.training_errors)
    for asset, count in market_counts.items():
        if count < cfg.minimum_markets_per_asset:
            errors.append(
                f"insufficient_model_markets:{asset}:"
                f"{count}/{cfg.minimum_markets_per_asset}"
            )
    if time_group_count < cfg.minimum_time_groups:
        errors.append(
            f"insufficient_model_time_groups:"
            f"{time_group_count}/{cfg.minimum_time_groups}"
        )
    for condition in {item.condition_id for item in samples}:
        weight = sum(item.market_weight for item in samples if item.condition_id == condition)
        if not math.isclose(weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Polymarket model market weights are not equalized")

    payload = {
        "schema_version": POLYMARKET_MODEL_DATASET_SCHEMA_VERSION,
        "source_dataset_id": source.dataset_id,
        "source_dataset_sha256": source.dataset_sha256,
        "source_run_id": source.run_id,
        "config": cfg.asdict(),
        "model_feature_names": list(POLYMARKET_MODEL_FEATURE_NAMES),
        "risk_context_names": list(POLYMARKET_MODEL_RISK_CONTEXT_NAMES),
        "sample_sha256": [item.sample_sha256 for item in samples],
        "market_counts": market_counts,
        "time_group_count": time_group_count,
        "skipped_counts": dict(sorted(skipped.items())),
        "training_errors": sorted(set(errors)),
    }
    dataset_sha256 = _canonical_sha256(payload)
    return PolymarketModelDataset(
        schema_version=POLYMARKET_MODEL_DATASET_SCHEMA_VERSION,
        source_dataset_id=source.dataset_id,
        source_dataset_sha256=source.dataset_sha256,
        source_run_id=source.run_id,
        config=cfg,
        samples=tuple(samples),
        market_counts=market_counts,
        time_group_count=time_group_count,
        skipped_counts=dict(sorted(skipped.items())),
        training_errors=tuple(sorted(set(errors))),
        dataset_sha256=dataset_sha256,
    )


def _split_payload(split: PolymarketModelSplit) -> dict[str, object]:
    return {
        "schema_version": POLYMARKET_MODEL_SPLIT_SCHEMA_VERSION,
        "source_dataset_sha256": split.source_dataset_sha256,
        "train_sample_ids": [item.sample_id for item in split.train],
        "validation_sample_ids": [item.sample_id for item in split.validation],
        "test_sample_ids": [item.sample_id for item in split.test],
        "train_group_starts_ms": list(split.train_group_starts_ms),
        "validation_group_starts_ms": list(split.validation_group_starts_ms),
        "test_group_starts_ms": list(split.test_group_starts_ms),
        "purged_group_starts_ms": list(split.purged_group_starts_ms),
    }


def _validate_model_split_identity(
    dataset: PolymarketModelDataset,
    split: PolymarketModelSplit,
) -> None:
    if split.source_dataset_sha256 != dataset.dataset_sha256:
        raise ValueError("Polymarket model split belongs to another dataset")
    if split.split_sha256 != _canonical_sha256(_split_payload(split)):
        raise ValueError("Polymarket model split identity is invalid")
    dataset_by_id = {item.sample_id: item for item in dataset.samples}
    if len(dataset_by_id) != len(dataset.samples):
        raise ValueError("Polymarket model dataset contains duplicate sample IDs")
    role_groups = {
        "train": set(split.train_group_starts_ms),
        "validation": set(split.validation_group_starts_ms),
        "test": set(split.test_group_starts_ms),
    }
    seen_ids: set[str] = set()
    for role, groups in role_groups.items():
        actual = split.samples_for(role)
        expected = tuple(
            item for item in dataset.samples if item.event_start_ms in groups
        )
        if tuple((item.sample_id, item.sample_sha256) for item in actual) != tuple(
            (item.sample_id, item.sample_sha256) for item in expected
        ):
            raise ValueError("Polymarket model split sample provenance is invalid")
        for sample in actual:
            if dataset_by_id.get(sample.sample_id) != sample:
                raise ValueError("Polymarket model split substituted a sample")
            sample.validated()
            if sample.sample_id in seen_ids:
                raise ValueError("Polymarket model split repeats a sample")
            seen_ids.add(sample.sample_id)


def _market_outcome_counts(
    samples: Sequence[PolymarketModelSample],
) -> dict[bool, int]:
    outcomes = {
        condition: bool(next(item.official_up for item in samples if item.condition_id == condition))
        for condition in {item.condition_id for item in samples}
    }
    return {
        outcome: sum(value is outcome for value in outcomes.values())
        for outcome in (False, True)
    }


def _asset_market_outcome_counts(
    samples: Sequence[PolymarketModelSample],
    asset: str,
) -> dict[bool, int]:
    return _market_outcome_counts(tuple(item for item in samples if item.asset == asset))


def split_polymarket_model_dataset(
    dataset: PolymarketModelDataset,
) -> PolymarketModelSplit:
    """Create contiguous splits by shared market start, with purged boundary groups."""

    if not dataset.training_ready:
        raise ValueError(
            "Polymarket model dataset is not training-ready: "
            + "; ".join(dataset.training_errors)
        )
    cfg = dataset.config.validated()
    groups = tuple(sorted({item.event_start_ms for item in dataset.samples}))
    validation_count = max(
        cfg.minimum_validation_time_groups,
        int(round(len(groups) * cfg.validation_fraction)),
    )
    test_count = max(
        cfg.minimum_test_time_groups,
        int(round(len(groups) * cfg.test_fraction)),
    )
    train_count = (
        len(groups)
        - validation_count
        - test_count
        - 2 * cfg.purge_time_groups
    )
    if train_count < cfg.minimum_train_time_groups:
        raise ValueError("insufficient time groups for the purged chronological split")
    train_groups = groups[:train_count]
    first_purge = groups[
        train_count : train_count + cfg.purge_time_groups
    ]
    validation_start = train_count + cfg.purge_time_groups
    validation_groups = groups[
        validation_start : validation_start + validation_count
    ]
    second_purge = groups[
        validation_start
        + validation_count : validation_start
        + validation_count
        + cfg.purge_time_groups
    ]
    test_groups = groups[-test_count:]
    purged_groups = (*first_purge, *second_purge)
    role_groups = (set(train_groups), set(validation_groups), set(test_groups))
    if any(left & right for index, left in enumerate(role_groups) for right in role_groups[index + 1 :]):
        raise ValueError("Polymarket chronological split groups overlap")
    if set(purged_groups) & set().union(*role_groups):
        raise ValueError("Polymarket purged groups leaked into a split role")
    if not max(train_groups) < min(validation_groups) < max(validation_groups) < min(test_groups):
        raise ValueError("Polymarket chronological split order is invalid")

    def select(group_values: Sequence[int]) -> tuple[PolymarketModelSample, ...]:
        allowed = set(group_values)
        return tuple(item for item in dataset.samples if item.event_start_ms in allowed)

    train = select(train_groups)
    validation = select(validation_groups)
    test = select(test_groups)
    for role, role_samples in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        counts = _market_outcome_counts(role_samples)
        if min(counts.values()) < cfg.minimum_outcome_markets_per_split:
            raise ValueError(
                f"Polymarket {role} split lacks both official outcome classes"
            )
        for asset in _ASSETS:
            asset_counts = _asset_market_outcome_counts(role_samples, asset)
            if min(asset_counts.values()) < cfg.minimum_outcome_markets_per_split:
                raise ValueError(
                    f"Polymarket {role} split lacks both official outcome classes "
                    f"for {asset}"
                )
    provisional = PolymarketModelSplit(
        schema_version=POLYMARKET_MODEL_SPLIT_SCHEMA_VERSION,
        source_dataset_sha256=dataset.dataset_sha256,
        train=train,
        validation=validation,
        test=test,
        train_group_starts_ms=train_groups,
        validation_group_starts_ms=validation_groups,
        test_group_starts_ms=test_groups,
        purged_group_starts_ms=tuple(purged_groups),
        split_sha256="",
    )
    return replace(
        provisional,
        split_sha256=_canonical_sha256(_split_payload(provisional)),
    )


def _raw_matrix(samples: Sequence[PolymarketPredictable]) -> np.ndarray:
    for sample in samples:
        sample.validated()
    matrix = np.asarray([item.feature_values for item in samples], dtype=np.float64)
    if matrix.shape != (len(samples), len(POLYMARKET_MODEL_FEATURE_NAMES)) or not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError("Polymarket model design matrix is invalid")
    return matrix


def _fit_transform(
    samples: Sequence[PolymarketModelSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = _raw_matrix(samples)
    lower = np.quantile(raw, 0.005, axis=0)
    upper = np.quantile(raw, 0.995, axis=0)
    clipped = np.clip(raw, lower, upper)
    center = np.median(clipped, axis=0)
    first_quartile = np.quantile(clipped, 0.25, axis=0)
    third_quartile = np.quantile(clipped, 0.75, axis=0)
    scale = (third_quartile - first_quartile) / 1.349
    scale = np.where(scale > 1e-12, scale, 1.0)
    if not all(np.all(np.isfinite(value)) for value in (lower, upper, center, scale)):
        raise ValueError("Polymarket robust feature transform is non-finite")
    return lower, upper, center, scale


def _apply_transform(
    samples: Sequence[PolymarketPredictable],
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    raw = _raw_matrix(samples)
    transformed = (np.clip(raw, lower, upper) - center) / scale
    return np.clip(transformed, -8.0, 8.0)


def _baseline_offsets(samples: Sequence[PolymarketPredictable]) -> np.ndarray:
    probabilities = np.clip(
        np.asarray(
            [item.baseline_up_probability for item in samples],
            dtype=np.float64,
        ),
        1e-6,
        1.0 - 1e-6,
    )
    if len(samples) == 0 or not np.all(np.isfinite(probabilities)):
        raise ValueError("Polymarket model priors are invalid")
    return np.log(probabilities / (1.0 - probabilities))


def _targets_weights_offsets(
    samples: Sequence[PolymarketModelSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([float(item.official_up) for item in samples], dtype=np.float64)
    weights = np.asarray([item.market_weight for item in samples], dtype=np.float64)
    if (
        len(samples) == 0
        or not np.all(np.isin(labels, (0.0, 1.0)))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("Polymarket model labels or market weights are invalid")
    return labels, weights, _baseline_offsets(samples)


def _weighted_log_loss_arrays(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return float(
        np.sum(
            weights
            * (-(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped)))
        )
        / np.sum(weights)
    )


def _fit_offset_coefficients(
    design: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    offsets: np.ndarray,
    *,
    l2: float,
    maximum_iterations: int,
    tolerance: float,
) -> np.ndarray:
    augmented = np.column_stack((np.ones(len(design), dtype=np.float64), design))
    coefficients = np.zeros(augmented.shape[1], dtype=np.float64)
    regularization = np.full(augmented.shape[1], float(l2), dtype=np.float64)
    regularization[0] = 0.01 * float(l2)
    weight_sum = float(np.sum(weights))

    def objective(candidate: np.ndarray) -> float:
        linear = offsets + augmented @ candidate
        likelihood = np.sum(
            weights * (np.logaddexp(0.0, linear) - labels * linear)
        ) / weight_sum
        return float(likelihood + 0.5 * np.sum(regularization * candidate**2))

    for _ in range(maximum_iterations):
        linear = offsets + augmented @ coefficients
        probabilities = np.exp(-np.logaddexp(0.0, -linear))
        residual = probabilities - labels
        variance = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
        gradient = (
            augmented.T @ (weights * residual) / weight_sum
            + regularization * coefficients
        )
        hessian = (
            (augmented.T * (weights * variance)) @ augmented / weight_sum
            + np.diag(regularization)
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        current = objective(coefficients)
        accepted = 0.0
        for line_search_step in range(30):
            scale = 2.0 ** (-line_search_step)
            candidate = coefficients - scale * step
            if objective(candidate) <= current:
                coefficients = candidate
                accepted = scale
                break
        if accepted == 0.0 or float(np.max(np.abs(accepted * step))) < tolerance:
            break
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Polymarket offset model coefficients are non-finite")
    return coefficients


def _predict_arrays(
    samples: Sequence[PolymarketPredictable],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    maximum_absolute_correction: float,
) -> np.ndarray:
    design = _apply_transform(samples, lower, upper, center, scale)
    offsets = _baseline_offsets(samples)
    correction = coefficients[0] + design @ coefficients[1:]
    correction = np.clip(
        correction,
        -float(maximum_absolute_correction),
        float(maximum_absolute_correction),
    )
    linear = offsets + correction
    return np.exp(-np.logaddexp(0.0, -linear))


def _validate_inference_config(
    samples: Sequence[PolymarketPredictable],
    config: PolymarketModelConfig,
) -> None:
    expected = _canonical_sha256(config.asdict())
    if any(
        isinstance(item, PolymarketInferenceInput)
        and item.model_config_sha256 != expected
        for item in samples
    ):
        raise ValueError("Polymarket inference input uses another model configuration")


def predict_polymarket_probabilities(
    model: TrainedPolymarketOffsetModel,
    samples: Sequence[PolymarketPredictable],
) -> np.ndarray:
    """Apply a validated, bounded correction to each market-implied prior."""

    if model.schema_version != POLYMARKET_OFFSET_MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported Polymarket offset model schema")
    if model.model_sha256 != _canonical_sha256(_model_payload(model)):
        raise ValueError("Polymarket offset model identity is invalid")
    if model.feature_names != POLYMARKET_MODEL_FEATURE_NAMES:
        raise ValueError("Polymarket offset model feature contract is inconsistent")
    width = len(POLYMARKET_MODEL_FEATURE_NAMES)
    arrays = (
        model.winsor_lower,
        model.winsor_upper,
        model.robust_center,
        model.robust_scale,
    )
    if any(len(value) != width for value in arrays) or len(model.coefficients) != width + 1:
        raise ValueError("Polymarket offset model parameter width is invalid")
    _validate_inference_config(samples, model.config)
    return _predict_arrays(
        samples,
        lower=np.asarray(model.winsor_lower, dtype=np.float64),
        upper=np.asarray(model.winsor_upper, dtype=np.float64),
        center=np.asarray(model.robust_center, dtype=np.float64),
        scale=np.asarray(model.robust_scale, dtype=np.float64),
        coefficients=np.asarray(model.coefficients, dtype=np.float64),
        maximum_absolute_correction=model.config.maximum_absolute_logit_correction,
    )


def predict_polymarket_profile_probabilities(
    model: TrainedPolymarketProfileModel,
    samples: Sequence[PolymarketPredictable],
) -> np.ndarray:
    """Apply a hash-bound frozen-profile challenger with no execution authority."""

    if model.schema_version != POLYMARKET_PROFILE_MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported Polymarket profile model schema")
    if model.contract_sha256 != POLYMARKET_PROFILE_CONTRACT_SHA256:
        raise ValueError("Polymarket profile model contract is inconsistent")
    if model.model_sha256 != _canonical_sha256(_profile_model_payload(model)):
        raise ValueError("Polymarket profile model identity is invalid")
    if model.feature_names != POLYMARKET_MODEL_FEATURE_NAMES:
        raise ValueError("Polymarket profile model feature contract is inconsistent")
    if any(
        (
            model.trading_authority,
            model.execution_claim,
            model.profitability_claim,
            model.portfolio_claim,
            model.leverage_applied,
        )
    ):
        raise ValueError("Polymarket profile model cannot carry trading authority")
    width = len(POLYMARKET_MODEL_FEATURE_NAMES)
    arrays = (
        model.winsor_lower,
        model.winsor_upper,
        model.robust_center,
        model.robust_scale,
    )
    if (
        any(len(value) != width for value in arrays)
        or len(model.coefficients) != width + 1
        or len(model.control_model_sha256) != 64
    ):
        raise ValueError("Polymarket profile model parameter width is invalid")
    _validate_inference_config(samples, model.config)
    if model.selected_candidate == "market_baseline":
        if (
            model.selected_profile is not None
            or model.selected_feature_names
            or model.selected_l2 is not None
            or any(value != 0.0 for value in model.coefficients)
        ):
            raise ValueError("Polymarket profile baseline fallback is inconsistent")
    else:
        profile, feature_names, l2 = _profile_candidate_details(
            model.selected_candidate
        )
        if (
            model.selected_profile != profile
            or model.selected_feature_names != feature_names
            or model.selected_l2 != l2
        ):
            raise ValueError("Polymarket profile selection is inconsistent")
        active_indexes = set(_profile_feature_indexes(feature_names))
        if any(
            model.coefficients[index + 1] != 0.0
            for index in range(width)
            if index not in active_indexes
        ):
            raise ValueError("Polymarket excluded profile coefficients must be zero")
    _profile_candidate_details(model.inner_selected_candidate)
    return _predict_arrays(
        samples,
        lower=np.asarray(model.winsor_lower, dtype=np.float64),
        upper=np.asarray(model.winsor_upper, dtype=np.float64),
        center=np.asarray(model.robust_center, dtype=np.float64),
        scale=np.asarray(model.robust_scale, dtype=np.float64),
        coefficients=np.asarray(model.coefficients, dtype=np.float64),
        maximum_absolute_correction=model.config.maximum_absolute_logit_correction,
    )


def evaluate_polymarket_probabilities(
    samples: Sequence[PolymarketModelSample],
    probabilities: Sequence[float] | np.ndarray,
) -> PolymarketProbabilityMetrics:
    """Calculate proper scores with every market carrying equal total weight."""

    labels, weights, _ = _targets_weights_offsets(samples)
    predicted = np.asarray(probabilities, dtype=np.float64)
    if predicted.shape != labels.shape or not np.all(np.isfinite(predicted)):
        raise ValueError("Polymarket probability predictions are invalid")
    if np.any(predicted <= 0.0) or np.any(predicted >= 1.0):
        raise ValueError("Polymarket probability predictions must lie inside (0, 1)")
    weight_sum = float(np.sum(weights))
    brier = float(np.sum(weights * (predicted - labels) ** 2) / weight_sum)
    accuracy = float(
        np.sum(weights * ((predicted >= 0.5) == labels)) / weight_sum
    )
    sharpness = float(np.sum(weights * np.abs(predicted - 0.5)) / weight_sum)
    calibration_error = 0.0
    bin_indexes = np.minimum((predicted * 10).astype(np.int64), 9)
    for index in range(10):
        selected = bin_indexes == index
        if not np.any(selected):
            continue
        bin_weight = float(np.sum(weights[selected]))
        observed = float(np.sum(weights[selected] * labels[selected]) / bin_weight)
        expected = float(np.sum(weights[selected] * predicted[selected]) / bin_weight)
        calibration_error += bin_weight / weight_sum * abs(observed - expected)
    return PolymarketProbabilityMetrics(
        row_count=len(samples),
        market_count=len({item.condition_id for item in samples}),
        time_group_count=len({item.event_start_ms for item in samples}),
        effective_market_weight=weight_sum,
        weighted_log_loss=_weighted_log_loss_arrays(labels, predicted, weights),
        weighted_brier_score=brier,
        weighted_calibration_error=calibration_error,
        weighted_accuracy=accuracy,
        weighted_sharpness=sharpness,
    )


def _model_payload(model: TrainedPolymarketOffsetModel) -> dict[str, object]:
    return {
        "schema_version": model.schema_version,
        "source_dataset_sha256": model.source_dataset_sha256,
        "source_split_sha256": model.source_split_sha256,
        "config": model.config.asdict(),
        "feature_names": list(model.feature_names),
        "winsor_lower": _format_floats(model.winsor_lower),
        "winsor_upper": _format_floats(model.winsor_upper),
        "robust_center": _format_floats(model.robust_center),
        "robust_scale": _format_floats(model.robust_scale),
        "coefficients": _format_floats(model.coefficients),
        "selected_candidate": model.selected_candidate,
        "selected_l2": (
            None if model.selected_l2 is None else format(model.selected_l2, ".17g")
        ),
        "inner_selected_candidate": model.inner_selected_candidate,
        "candidate_inner_log_losses": [
            [name, format(value, ".17g")]
            for name, value in model.candidate_inner_log_losses
        ],
        "validation_gate_log_losses": [
            [name, format(value, ".17g")]
            for name, value in model.validation_gate_log_losses
        ],
        "inner_fold_count": model.inner_fold_count,
        "inner_fold_boundaries_ms": [
            list(value) for value in model.inner_fold_boundaries_ms
        ],
        "training_sample_count": model.training_sample_count,
        "training_market_count": model.training_market_count,
        "training_time_group_count": model.training_time_group_count,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def _profile_model_payload(
    model: TrainedPolymarketProfileModel,
) -> dict[str, object]:
    return {
        "schema_version": model.schema_version,
        "contract_sha256": model.contract_sha256,
        "source_dataset_sha256": model.source_dataset_sha256,
        "source_split_sha256": model.source_split_sha256,
        "control_model_sha256": model.control_model_sha256,
        "config": model.config.asdict(),
        "feature_names": list(model.feature_names),
        "selected_profile": model.selected_profile,
        "selected_feature_names": list(model.selected_feature_names),
        "winsor_lower": _format_floats(model.winsor_lower),
        "winsor_upper": _format_floats(model.winsor_upper),
        "robust_center": _format_floats(model.robust_center),
        "robust_scale": _format_floats(model.robust_scale),
        "coefficients": _format_floats(model.coefficients),
        "selected_candidate": model.selected_candidate,
        "selected_l2": (
            None if model.selected_l2 is None else format(model.selected_l2, ".17g")
        ),
        "inner_selected_candidate": model.inner_selected_candidate,
        "candidate_inner_log_losses": [
            [name, format(value, ".17g")]
            for name, value in model.candidate_inner_log_losses
        ],
        "validation_gate_log_losses": [
            [name, format(value, ".17g")]
            for name, value in model.validation_gate_log_losses
        ],
        "inner_fold_count": model.inner_fold_count,
        "inner_fold_boundaries_ms": [
            list(value) for value in model.inner_fold_boundaries_ms
        ],
        "training_sample_count": model.training_sample_count,
        "training_market_count": model.training_market_count,
        "training_time_group_count": model.training_time_group_count,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def _profile_candidate_name(profile: str, l2: float) -> str:
    return f"offset_profile_{profile}_l2_{format(float(l2), '.17g')}"


def _profile_candidate_details(
    name: str,
) -> tuple[str, tuple[str, ...], float]:
    for profile, feature_names in POLYMARKET_PROFILE_FEATURES:
        for l2 in POLYMARKET_PROFILE_L2_CANDIDATES:
            if name == _profile_candidate_name(profile, l2):
                return profile, feature_names, l2
    raise ValueError("unknown frozen Polymarket profile candidate")


def _profile_feature_indexes(feature_names: Sequence[str]) -> tuple[int, ...]:
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Polymarket profile contains duplicate features")
    index_by_name = {
        name: index for index, name in enumerate(POLYMARKET_MODEL_FEATURE_NAMES)
    }
    try:
        return tuple(index_by_name[name] for name in feature_names)
    except KeyError as exc:
        raise ValueError("Polymarket profile contains an undeclared feature") from exc


def _expand_profile_coefficients(
    active_coefficients: np.ndarray,
    feature_indexes: Sequence[int],
) -> np.ndarray:
    if active_coefficients.shape != (len(feature_indexes) + 1,):
        raise ValueError("Polymarket profile coefficient width is invalid")
    expanded = np.zeros(len(POLYMARKET_MODEL_FEATURE_NAMES) + 1, dtype=np.float64)
    expanded[0] = active_coefficients[0]
    expanded[1 + np.asarray(feature_indexes, dtype=np.int64)] = active_coefficients[1:]
    return expanded


def _baseline_probabilities(
    samples: Sequence[PolymarketModelSample],
) -> np.ndarray:
    return np.asarray(
        [item.baseline_up_probability for item in samples], dtype=np.float64
    )


def _inner_rolling_folds(
    samples: Sequence[PolymarketModelSample],
    config: PolymarketModelConfig,
) -> tuple[
    tuple[tuple[PolymarketModelSample, ...], tuple[PolymarketModelSample, ...]],
    ...,
]:
    groups = tuple(sorted({item.event_start_ms for item in samples}))
    validation_size = int(config.inner_validation_time_groups)
    first_validation = len(groups) - int(config.inner_fold_count) * validation_size
    folds: list[
        tuple[
            tuple[PolymarketModelSample, ...],
            tuple[PolymarketModelSample, ...],
        ]
    ] = []
    for fold_index in range(int(config.inner_fold_count)):
        validation_start = first_validation + fold_index * validation_size
        train_end = validation_start - int(config.inner_purge_time_groups)
        train_groups = groups[:train_end]
        validation_groups = groups[
            validation_start : validation_start + validation_size
        ]
        if (
            len(train_groups) < int(config.minimum_inner_train_time_groups)
            or len(validation_groups) != validation_size
            or not max(train_groups) < min(validation_groups)
        ):
            raise ValueError("insufficient groups for inner rolling model selection")
        train_allowed = set(train_groups)
        validation_allowed = set(validation_groups)
        train_rows = tuple(
            item for item in samples if item.event_start_ms in train_allowed
        )
        validation_rows = tuple(
            item for item in samples if item.event_start_ms in validation_allowed
        )
        if not train_rows or not validation_rows:
            raise ValueError("inner rolling model fold is empty")
        folds.append((train_rows, validation_rows))
    return tuple(folds)


def fit_polymarket_offset_model(
    dataset: PolymarketModelDataset,
    split: PolymarketModelSplit,
) -> tuple[TrainedPolymarketOffsetModel, PolymarketModelReport]:
    """Select inside training, gate on validation, and score untouched test data."""

    _validate_model_split_identity(dataset, split)
    cfg = dataset.config.validated()
    inner_folds = _inner_rolling_folds(split.train, cfg)
    inner_weight = 0.0
    baseline_inner_sum = 0.0
    candidate_inner_sums = {float(l2): 0.0 for l2 in cfg.l2_candidates}
    for inner_train, inner_validation in inner_folds:
        inner_lower, inner_upper, inner_center, inner_scale = _fit_transform(
            inner_train
        )
        inner_design = _apply_transform(
            inner_train,
            inner_lower,
            inner_upper,
            inner_center,
            inner_scale,
        )
        inner_labels, inner_weights, inner_offsets = _targets_weights_offsets(
            inner_train
        )
        fold_labels, fold_weights, _ = _targets_weights_offsets(inner_validation)
        fold_weight = float(np.sum(fold_weights))
        inner_weight += fold_weight
        baseline_inner_sum += fold_weight * _weighted_log_loss_arrays(
            fold_labels,
            _baseline_probabilities(inner_validation),
            fold_weights,
        )
        for l2 in cfg.l2_candidates:
            coefficients = _fit_offset_coefficients(
                inner_design,
                inner_labels,
                inner_weights,
                inner_offsets,
                l2=l2,
                maximum_iterations=cfg.maximum_iterations,
                tolerance=cfg.convergence_tolerance,
            )
            predictions = _predict_arrays(
                inner_validation,
                lower=inner_lower,
                upper=inner_upper,
                center=inner_center,
                scale=inner_scale,
                coefficients=coefficients,
                maximum_absolute_correction=cfg.maximum_absolute_logit_correction,
            )
            candidate_inner_sums[float(l2)] += (
                fold_weight
                * _weighted_log_loss_arrays(
                    fold_labels,
                    predictions,
                    fold_weights,
                )
            )
    if inner_weight <= 0.0:
        raise ValueError("inner rolling model selection has no effective weight")
    candidate_inner_losses = [
        ("market_baseline", baseline_inner_sum / inner_weight),
        *[
            (
                f"offset_l2_{format(l2, '.17g')}",
                candidate_inner_sums[float(l2)] / inner_weight,
            )
            for l2 in cfg.l2_candidates
        ],
    ]
    inner_name, _inner_loss = min(
        candidate_inner_losses[1:],
        key=lambda item: (
            item[1],
            -float(item[0].removeprefix("offset_l2_")),
            item[0],
        ),
    )
    inner_l2 = float(inner_name.removeprefix("offset_l2_"))

    lower, upper, center, scale = _fit_transform(split.train)
    train_design = _apply_transform(split.train, lower, upper, center, scale)
    train_labels, train_weights, train_offsets = _targets_weights_offsets(split.train)
    candidate_coefficients = _fit_offset_coefficients(
        train_design,
        train_labels,
        train_weights,
        train_offsets,
        l2=inner_l2,
        maximum_iterations=cfg.maximum_iterations,
        tolerance=cfg.convergence_tolerance,
    )
    validation_labels, validation_weights, _ = _targets_weights_offsets(split.validation)
    baseline_validation = _baseline_probabilities(split.validation)
    baseline_validation_loss = _weighted_log_loss_arrays(
        validation_labels,
        baseline_validation,
        validation_weights,
    )
    candidate_validation_predictions = _predict_arrays(
        split.validation,
        lower=lower,
        upper=upper,
        center=center,
        scale=scale,
        coefficients=candidate_coefficients,
        maximum_absolute_correction=cfg.maximum_absolute_logit_correction,
    )
    candidate_validation_loss = _weighted_log_loss_arrays(
        validation_labels,
        candidate_validation_predictions,
        validation_weights,
    )
    validation_gate_losses = (
        ("market_baseline", baseline_validation_loss),
        (inner_name, candidate_validation_loss),
    )
    required = cfg.minimum_validation_log_loss_improvement
    if candidate_validation_loss <= baseline_validation_loss - required:
        selected_name = inner_name
        selected_l2 = inner_l2
        selected_coefficients = candidate_coefficients
    else:
        selected_name = "market_baseline"
        selected_l2 = None
        selected_coefficients = np.zeros(
            len(POLYMARKET_MODEL_FEATURE_NAMES) + 1,
            dtype=np.float64,
        )
    provisional = TrainedPolymarketOffsetModel(
        schema_version=POLYMARKET_OFFSET_MODEL_SCHEMA_VERSION,
        source_dataset_sha256=dataset.dataset_sha256,
        source_split_sha256=split.split_sha256,
        config=cfg,
        feature_names=POLYMARKET_MODEL_FEATURE_NAMES,
        winsor_lower=tuple(float(value) for value in lower),
        winsor_upper=tuple(float(value) for value in upper),
        robust_center=tuple(float(value) for value in center),
        robust_scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in selected_coefficients),
        selected_candidate=selected_name,
        selected_l2=selected_l2,
        inner_selected_candidate=inner_name,
        candidate_inner_log_losses=tuple(candidate_inner_losses),
        validation_gate_log_losses=validation_gate_losses,
        inner_fold_count=len(inner_folds),
        inner_fold_boundaries_ms=tuple(
            (
                min(item.event_start_ms for item in inner_train),
                max(item.event_start_ms for item in inner_train),
                min(item.event_start_ms for item in inner_validation),
                max(item.event_start_ms for item in inner_validation),
            )
            for inner_train, inner_validation in inner_folds
        ),
        training_sample_count=len(split.train),
        training_market_count=len({item.condition_id for item in split.train}),
        training_time_group_count=len(split.train_group_starts_ms),
        model_sha256="",
    )
    model = replace(
        provisional,
        model_sha256=_canonical_sha256(_model_payload(provisional)),
    )
    baseline_metrics: dict[str, PolymarketProbabilityMetrics] = {}
    model_metrics: dict[str, PolymarketProbabilityMetrics] = {}
    for role in ("train", "validation", "test"):
        samples = split.samples_for(role)
        baseline_metrics[role] = evaluate_polymarket_probabilities(
            samples,
            _baseline_probabilities(samples),
        )
        model_metrics[role] = evaluate_polymarket_probabilities(
            samples,
            predict_polymarket_probabilities(model, samples),
        )
    report_payload = {
        "schema_version": POLYMARKET_MODEL_REPORT_SCHEMA_VERSION,
        "source_dataset_sha256": dataset.dataset_sha256,
        "source_split_sha256": split.split_sha256,
        "model_sha256": model.model_sha256,
        "selected_candidate": selected_name,
        "baseline_metrics": {
            key: value.asdict() for key, value in baseline_metrics.items()
        },
        "model_metrics": {
            key: value.asdict() for key, value in model_metrics.items()
        },
        "validation_log_loss_delta": (
            model_metrics["validation"].weighted_log_loss
            - baseline_metrics["validation"].weighted_log_loss
        ),
        "test_log_loss_delta": (
            model_metrics["test"].weighted_log_loss
            - baseline_metrics["test"].weighted_log_loss
        ),
        "test_brier_delta": (
            model_metrics["test"].weighted_brier_score
            - baseline_metrics["test"].weighted_brier_score
        ),
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }
    report = PolymarketModelReport(
        schema_version=POLYMARKET_MODEL_REPORT_SCHEMA_VERSION,
        source_dataset_sha256=dataset.dataset_sha256,
        source_split_sha256=split.split_sha256,
        model_sha256=model.model_sha256,
        selected_candidate=selected_name,
        baseline_metrics=baseline_metrics,
        model_metrics=model_metrics,
        validation_log_loss_delta=float(report_payload["validation_log_loss_delta"]),
        test_log_loss_delta=float(report_payload["test_log_loss_delta"]),
        test_brier_delta=float(report_payload["test_brier_delta"]),
        report_sha256=_canonical_sha256(report_payload),
    )
    return model, report


def fit_polymarket_profile_challenger(
    dataset: PolymarketModelDataset,
    split: PolymarketModelSplit,
    control_model: TrainedPolymarketOffsetModel,
) -> tuple[TrainedPolymarketProfileModel, PolymarketProfileModelReport]:
    """Fit the preregistered profile grid without consulting test outcomes."""

    _validate_model_split_identity(dataset, split)
    cfg = dataset.config.validated()
    if (
        tuple(cfg.l2_candidates) != POLYMARKET_PROFILE_L2_CANDIDATES
        or cfg.inner_fold_count != 3
        or not math.isclose(
            cfg.minimum_validation_log_loss_improvement,
            0.0001,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("Polymarket profile challenger requires its frozen grid")
    if (
        control_model.schema_version != POLYMARKET_OFFSET_MODEL_SCHEMA_VERSION
        or control_model.source_dataset_sha256 != dataset.dataset_sha256
        or control_model.source_split_sha256 != split.split_sha256
        or control_model.config != cfg
    ):
        raise ValueError("Polymarket profile control model is inconsistent")
    predict_polymarket_probabilities(control_model, split.train)

    profile_indexes = {
        profile: _profile_feature_indexes(feature_names)
        for profile, feature_names in POLYMARKET_PROFILE_FEATURES
    }
    inner_folds = _inner_rolling_folds(split.train, cfg)
    candidate_sums = {
        _profile_candidate_name(profile, l2): 0.0
        for profile, _feature_names in POLYMARKET_PROFILE_FEATURES
        for l2 in POLYMARKET_PROFILE_L2_CANDIDATES
    }
    inner_weight = 0.0
    baseline_inner_sum = 0.0
    for inner_train, inner_validation in inner_folds:
        lower, upper, center, scale = _fit_transform(inner_train)
        full_design = _apply_transform(
            inner_train,
            lower,
            upper,
            center,
            scale,
        )
        labels, weights, offsets = _targets_weights_offsets(inner_train)
        fold_labels, fold_weights, _ = _targets_weights_offsets(inner_validation)
        fold_weight = float(np.sum(fold_weights))
        inner_weight += fold_weight
        baseline_inner_sum += fold_weight * _weighted_log_loss_arrays(
            fold_labels,
            _baseline_probabilities(inner_validation),
            fold_weights,
        )
        for profile, _feature_names in POLYMARKET_PROFILE_FEATURES:
            indexes = profile_indexes[profile]
            active_design = full_design[:, indexes]
            for l2 in POLYMARKET_PROFILE_L2_CANDIDATES:
                name = _profile_candidate_name(profile, l2)
                active_coefficients = _fit_offset_coefficients(
                    active_design,
                    labels,
                    weights,
                    offsets,
                    l2=l2,
                    maximum_iterations=cfg.maximum_iterations,
                    tolerance=cfg.convergence_tolerance,
                )
                coefficients = _expand_profile_coefficients(
                    active_coefficients,
                    indexes,
                )
                predictions = _predict_arrays(
                    inner_validation,
                    lower=lower,
                    upper=upper,
                    center=center,
                    scale=scale,
                    coefficients=coefficients,
                    maximum_absolute_correction=(
                        cfg.maximum_absolute_logit_correction
                    ),
                )
                candidate_sums[name] += fold_weight * _weighted_log_loss_arrays(
                    fold_labels,
                    predictions,
                    fold_weights,
                )
    if inner_weight <= 0.0:
        raise ValueError("inner profile selection has no effective weight")

    candidate_inner_losses = [
        ("market_baseline", baseline_inner_sum / inner_weight),
        *[
            (
                _profile_candidate_name(profile, l2),
                candidate_sums[_profile_candidate_name(profile, l2)]
                / inner_weight,
            )
            for profile, _feature_names in POLYMARKET_PROFILE_FEATURES
            for l2 in POLYMARKET_PROFILE_L2_CANDIDATES
        ],
    ]

    def selection_key(item: tuple[str, float]) -> tuple[float, int, float, str]:
        profile, feature_names, l2 = _profile_candidate_details(item[0])
        return item[1], len(feature_names), -l2, profile

    inner_name, _inner_loss = min(candidate_inner_losses[1:], key=selection_key)
    inner_profile, inner_feature_names, inner_l2 = _profile_candidate_details(
        inner_name
    )
    inner_indexes = profile_indexes[inner_profile]

    lower, upper, center, scale = _fit_transform(split.train)
    full_train_design = _apply_transform(
        split.train,
        lower,
        upper,
        center,
        scale,
    )
    train_labels, train_weights, train_offsets = _targets_weights_offsets(split.train)
    active_coefficients = _fit_offset_coefficients(
        full_train_design[:, inner_indexes],
        train_labels,
        train_weights,
        train_offsets,
        l2=inner_l2,
        maximum_iterations=cfg.maximum_iterations,
        tolerance=cfg.convergence_tolerance,
    )
    candidate_coefficients = _expand_profile_coefficients(
        active_coefficients,
        inner_indexes,
    )
    validation_labels, validation_weights, _ = _targets_weights_offsets(
        split.validation
    )
    baseline_validation_loss = _weighted_log_loss_arrays(
        validation_labels,
        _baseline_probabilities(split.validation),
        validation_weights,
    )
    candidate_validation_loss = _weighted_log_loss_arrays(
        validation_labels,
        _predict_arrays(
            split.validation,
            lower=lower,
            upper=upper,
            center=center,
            scale=scale,
            coefficients=candidate_coefficients,
            maximum_absolute_correction=cfg.maximum_absolute_logit_correction,
        ),
        validation_weights,
    )
    validation_gate_losses = (
        ("market_baseline", baseline_validation_loss),
        (inner_name, candidate_validation_loss),
    )
    if (
        candidate_validation_loss
        <= baseline_validation_loss
        - cfg.minimum_validation_log_loss_improvement
    ):
        selected_name = inner_name
        selected_profile: str | None = inner_profile
        selected_feature_names = inner_feature_names
        selected_l2: float | None = inner_l2
        selected_coefficients = candidate_coefficients
    else:
        selected_name = "market_baseline"
        selected_profile = None
        selected_feature_names = ()
        selected_l2 = None
        selected_coefficients = np.zeros(
            len(POLYMARKET_MODEL_FEATURE_NAMES) + 1,
            dtype=np.float64,
        )

    provisional = TrainedPolymarketProfileModel(
        schema_version=POLYMARKET_PROFILE_MODEL_SCHEMA_VERSION,
        contract_sha256=POLYMARKET_PROFILE_CONTRACT_SHA256,
        source_dataset_sha256=dataset.dataset_sha256,
        source_split_sha256=split.split_sha256,
        control_model_sha256=control_model.model_sha256,
        config=cfg,
        feature_names=POLYMARKET_MODEL_FEATURE_NAMES,
        selected_profile=selected_profile,
        selected_feature_names=selected_feature_names,
        winsor_lower=tuple(float(value) for value in lower),
        winsor_upper=tuple(float(value) for value in upper),
        robust_center=tuple(float(value) for value in center),
        robust_scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in selected_coefficients),
        selected_candidate=selected_name,
        selected_l2=selected_l2,
        inner_selected_candidate=inner_name,
        candidate_inner_log_losses=tuple(candidate_inner_losses),
        validation_gate_log_losses=validation_gate_losses,
        inner_fold_count=len(inner_folds),
        inner_fold_boundaries_ms=tuple(
            (
                min(item.event_start_ms for item in inner_train),
                max(item.event_start_ms for item in inner_train),
                min(item.event_start_ms for item in inner_validation),
                max(item.event_start_ms for item in inner_validation),
            )
            for inner_train, inner_validation in inner_folds
        ),
        training_sample_count=len(split.train),
        training_market_count=len({item.condition_id for item in split.train}),
        training_time_group_count=len(split.train_group_starts_ms),
        model_sha256="",
    )
    model = replace(
        provisional,
        model_sha256=_canonical_sha256(_profile_model_payload(provisional)),
    )

    baseline_metrics: dict[str, PolymarketProbabilityMetrics] = {}
    control_metrics: dict[str, PolymarketProbabilityMetrics] = {}
    challenger_metrics: dict[str, PolymarketProbabilityMetrics] = {}
    for role in ("train", "validation", "test"):
        samples = split.samples_for(role)
        baseline_metrics[role] = evaluate_polymarket_probabilities(
            samples,
            _baseline_probabilities(samples),
        )
        control_metrics[role] = evaluate_polymarket_probabilities(
            samples,
            predict_polymarket_probabilities(control_model, samples),
        )
        challenger_metrics[role] = evaluate_polymarket_probabilities(
            samples,
            predict_polymarket_profile_probabilities(model, samples),
        )
    validation_delta = (
        challenger_metrics["validation"].weighted_log_loss
        - control_metrics["validation"].weighted_log_loss
    )
    test_log_loss_delta = (
        challenger_metrics["test"].weighted_log_loss
        - control_metrics["test"].weighted_log_loss
    )
    test_brier_delta = (
        challenger_metrics["test"].weighted_brier_score
        - control_metrics["test"].weighted_brier_score
    )
    report_payload = {
        "schema_version": POLYMARKET_PROFILE_REPORT_SCHEMA_VERSION,
        "contract_sha256": POLYMARKET_PROFILE_CONTRACT_SHA256,
        "source_dataset_sha256": dataset.dataset_sha256,
        "source_split_sha256": split.split_sha256,
        "control_model_sha256": control_model.model_sha256,
        "challenger_model_sha256": model.model_sha256,
        "selected_candidate": selected_name,
        "baseline_metrics": {
            key: value.asdict() for key, value in baseline_metrics.items()
        },
        "control_metrics": {
            key: value.asdict() for key, value in control_metrics.items()
        },
        "challenger_metrics": {
            key: value.asdict() for key, value in challenger_metrics.items()
        },
        "validation_log_loss_delta_vs_control": validation_delta,
        "test_log_loss_delta_vs_control": test_log_loss_delta,
        "test_brier_delta_vs_control": test_brier_delta,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }
    report = PolymarketProfileModelReport(
        schema_version=POLYMARKET_PROFILE_REPORT_SCHEMA_VERSION,
        contract_sha256=POLYMARKET_PROFILE_CONTRACT_SHA256,
        source_dataset_sha256=dataset.dataset_sha256,
        source_split_sha256=split.split_sha256,
        control_model_sha256=control_model.model_sha256,
        challenger_model_sha256=model.model_sha256,
        selected_candidate=selected_name,
        baseline_metrics=baseline_metrics,
        control_metrics=control_metrics,
        challenger_metrics=challenger_metrics,
        validation_log_loss_delta_vs_control=validation_delta,
        test_log_loss_delta_vs_control=test_log_loss_delta,
        test_brier_delta_vs_control=test_brier_delta,
        report_sha256=_canonical_sha256(report_payload),
    )
    return model, report


__all__ = [
    "POLYMARKET_INFERENCE_INPUT_SCHEMA_VERSION",
    "POLYMARKET_LIVE_INFERENCE_CONTRACT_SHA256",
    "POLYMARKET_MODEL_DATASET_SCHEMA_VERSION",
    "POLYMARKET_MODEL_FEATURE_NAMES",
    "POLYMARKET_MODEL_RISK_CONTEXT_NAMES",
    "POLYMARKET_MODEL_REPORT_SCHEMA_VERSION",
    "POLYMARKET_MODEL_SAMPLE_SCHEMA_VERSION",
    "POLYMARKET_MODEL_SPLIT_SCHEMA_VERSION",
    "POLYMARKET_OFFSET_MODEL_SCHEMA_VERSION",
    "POLYMARKET_PROFILE_CONTRACT_SHA256",
    "POLYMARKET_PROFILE_CHALLENGER_SCHEMA_VERSION",
    "POLYMARKET_PROFILE_FEATURES",
    "POLYMARKET_PROFILE_L2_CANDIDATES",
    "POLYMARKET_PROFILE_MODEL_SCHEMA_VERSION",
    "POLYMARKET_PROFILE_REPORT_SCHEMA_VERSION",
    "PolymarketModelConfig",
    "PolymarketModelDataset",
    "PolymarketInferenceInput",
    "PolymarketModelReport",
    "PolymarketModelSample",
    "PolymarketModelSplit",
    "PolymarketProbabilityMetrics",
    "PolymarketProfileModelReport",
    "TrainedPolymarketOffsetModel",
    "TrainedPolymarketProfileModel",
    "build_polymarket_inference_input",
    "build_polymarket_model_features",
    "build_polymarket_model_dataset",
    "build_polymarket_risk_context",
    "evaluate_polymarket_probabilities",
    "fit_polymarket_offset_model",
    "fit_polymarket_profile_challenger",
    "predict_polymarket_probabilities",
    "predict_polymarket_profile_probabilities",
    "split_polymarket_model_dataset",
]
