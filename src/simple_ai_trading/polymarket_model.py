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


POLYMARKET_MODEL_SAMPLE_SCHEMA_VERSION = "polymarket-model-sample-v1"
POLYMARKET_MODEL_DATASET_SCHEMA_VERSION = "polymarket-model-dataset-v1"
POLYMARKET_OFFSET_MODEL_SCHEMA_VERSION = "polymarket-market-anchored-logit-v2"
POLYMARKET_MODEL_SPLIT_SCHEMA_VERSION = "polymarket-purged-time-split-v1"
POLYMARKET_MODEL_REPORT_SCHEMA_VERSION = "polymarket-probability-report-v2"

POLYMARKET_MODEL_FEATURE_NAMES = (
    "remaining_seconds",
    "direct_distance_from_chainlink_open_bps",
    "direct_chainlink_basis_bps",
    "direct_return_250ms_bps",
    "direct_return_1000ms_bps",
    "direct_return_5000ms_bps",
    "direct_realized_volatility_1000ms_bps",
    "direct_realized_volatility_5000ms_bps",
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


def _row_map(row: PolymarketFeatureRow) -> dict[str, float]:
    if len(row.feature_values) != len(POLYMARKET_FEATURE_NAMES):
        raise ValueError("Polymarket feature row width is invalid")
    values = tuple(_finite(value, name="Polymarket feature") for value in row.feature_values)
    return dict(zip(POLYMARKET_FEATURE_NAMES, values, strict=True))


def _relative_bps(value: float, reference: float, *, name: str) -> float:
    if reference <= 0.0:
        raise ValueError(f"{name} reference must be positive")
    return 10_000.0 * (value - reference) / reference


def _model_features(values: Mapping[str, float], asset: str) -> tuple[float, ...]:
    up_midpoint = values["up_midpoint"]
    down_midpoint = values["down_midpoint"]
    derived = (
        values["remaining_seconds"],
        values["binance_distance_from_chainlink_open_bps"],
        values["binance_chainlink_basis_bps"],
        values["binance_return_250ms_bps"],
        values["binance_return_1000ms_bps"],
        values["binance_return_5000ms_bps"],
        values["binance_realized_volatility_1000ms_bps"],
        values["binance_realized_volatility_5000ms_bps"],
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
                feature_values=_model_features(values, market.asset),
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
            if not any(item.asset == asset for item in role_samples):
                raise ValueError(f"Polymarket {role} split has no {asset} markets")
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


def _raw_matrix(samples: Sequence[PolymarketModelSample]) -> np.ndarray:
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
    samples: Sequence[PolymarketModelSample],
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    raw = _raw_matrix(samples)
    transformed = (np.clip(raw, lower, upper) - center) / scale
    return np.clip(transformed, -8.0, 8.0)


def _targets_weights_offsets(
    samples: Sequence[PolymarketModelSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([float(item.official_up) for item in samples], dtype=np.float64)
    weights = np.asarray([item.market_weight for item in samples], dtype=np.float64)
    probabilities = np.clip(
        np.asarray(
            [item.baseline_up_probability for item in samples], dtype=np.float64
        ),
        1e-6,
        1.0 - 1e-6,
    )
    if (
        len(samples) == 0
        or not np.all(np.isin(labels, (0.0, 1.0)))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("Polymarket model labels or market weights are invalid")
    return labels, weights, np.log(probabilities / (1.0 - probabilities))


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
    samples: Sequence[PolymarketModelSample],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    maximum_absolute_correction: float,
) -> np.ndarray:
    design = _apply_transform(samples, lower, upper, center, scale)
    _, _, offsets = _targets_weights_offsets(samples)
    correction = coefficients[0] + design @ coefficients[1:]
    correction = np.clip(
        correction,
        -float(maximum_absolute_correction),
        float(maximum_absolute_correction),
    )
    linear = offsets + correction
    return np.exp(-np.logaddexp(0.0, -linear))


def predict_polymarket_probabilities(
    model: TrainedPolymarketOffsetModel,
    samples: Sequence[PolymarketModelSample],
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

    if split.source_dataset_sha256 != dataset.dataset_sha256:
        raise ValueError("Polymarket model split belongs to another dataset")
    if split.split_sha256 != _canonical_sha256(_split_payload(split)):
        raise ValueError("Polymarket model split identity is invalid")
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


__all__ = [
    "POLYMARKET_MODEL_DATASET_SCHEMA_VERSION",
    "POLYMARKET_MODEL_FEATURE_NAMES",
    "POLYMARKET_MODEL_REPORT_SCHEMA_VERSION",
    "POLYMARKET_MODEL_SAMPLE_SCHEMA_VERSION",
    "POLYMARKET_MODEL_SPLIT_SCHEMA_VERSION",
    "POLYMARKET_OFFSET_MODEL_SCHEMA_VERSION",
    "PolymarketModelConfig",
    "PolymarketModelDataset",
    "PolymarketModelReport",
    "PolymarketModelSample",
    "PolymarketModelSplit",
    "PolymarketProbabilityMetrics",
    "TrainedPolymarketOffsetModel",
    "build_polymarket_model_dataset",
    "evaluate_polymarket_probabilities",
    "fit_polymarket_offset_model",
    "predict_polymarket_probabilities",
    "split_polymarket_model_dataset",
]
