"""Chronological tuning split and probability calibration for Round 74."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .impact_absorption_event_dataset import (
    Round74EventRunPartition,
)
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SYMBOLS,
)


ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION = "round-074-tuning-subpartition-v1"
ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION = "round-074-temperature-calibration-v5"
ROUND74_TEMPERATURE_CALIBRATION_LEGACY_SCHEMA_VERSION = (
    "round-074-temperature-calibration-v2"
)
ROUND74_TEMPERATURE_CALIBRATION_PRIOR_SCHEMA_VERSION = (
    "round-074-temperature-calibration-v3"
)
ROUND74_TEMPERATURE_CALIBRATION_RISK_PRIOR_SCHEMA_VERSION = (
    "round-074-temperature-calibration-v4"
)
ROUND74_RISK_QUANTILE_CALIBRATION_SCHEMA_VERSION = (
    "round-074-risk-quantile-calibration-v2"
)
ROUND74_RISK_QUANTILE_CALIBRATION_PRIOR_SCHEMA_VERSION = (
    "round-074-risk-quantile-calibration-v1"
)
ROUND74_NO_INFORMATION_QUANTILE_BASELINE_SCHEMA_VERSION = (
    "round-074-no-information-quantile-baseline-v1"
)
ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS = ("capture_run", "eligible_target")
ROUND74_TUNING_EXPECTED_RUNS = 24
ROUND74_TUNING_MODEL_SELECTION_RUNS = 12
ROUND74_TUNING_CALIBRATION_RUNS = 6
ROUND74_TUNING_POLICY_SELECTION_RUNS = 6
ROUND74_TEMPERATURE_MINIMUM = 0.05
ROUND74_TEMPERATURE_MAXIMUM = 20.0
ROUND74_TEMPERATURE_CANDIDATE_COUNT = 257
ROUND74_TEMPERATURE_ECE_BINS = 20
ROUND74_PAYOFF_LOWER_CALIBRATION_QUANTILES = (0.10, 0.25)
ROUND74_MAE_UPPER_CALIBRATION_QUANTILE = 0.90


def _valid_population_run_count(
    count: object,
    *,
    optimization_population: str,
    capture_run_count: int,
) -> bool:
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or optimization_population not in ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS
    ):
        return False
    if optimization_population == "capture_run":
        return count == capture_run_count
    return count >= capture_run_count


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"Round 74 calibration {label} digest differs")
    return str(value)


@dataclass(frozen=True)
class Round74TuningSubpartition:
    """Disjoint chronological tuning roles fixed before target access."""

    parent_partition_sha256: str
    model_selection_run_ids: tuple[str, ...]
    calibration_run_ids: tuple[str, ...]
    policy_selection_run_ids: tuple[str, ...]
    schema_version: str = ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION

    def validate(self) -> None:
        groups = (
            self.model_selection_run_ids,
            self.calibration_run_ids,
            self.policy_selection_run_ids,
        )
        expected_counts = (
            ROUND74_TUNING_MODEL_SELECTION_RUNS,
            ROUND74_TUNING_CALIBRATION_RUNS,
            ROUND74_TUNING_POLICY_SELECTION_RUNS,
        )
        if (
            self.schema_version != ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION
            or tuple(len(group) for group in groups) != expected_counts
            or any(
                len(value) != 32
                or any(character not in "0123456789abcdef" for character in value)
                for group in groups
                for value in group
            )
            or len({value for group in groups for value in group})
            != ROUND74_TUNING_EXPECTED_RUNS
        ):
            raise ValueError("Round 74 tuning subpartition differs")
        _require_sha256(self.parent_partition_sha256, "parent partition")

    @property
    def subpartition_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "parent_partition_sha256": self.parent_partition_sha256,
            "split_unit": "whole_capture_run",
            "chronological": True,
            "random_row_split_permitted": False,
            "model_selection_run_ids": list(self.model_selection_run_ids),
            "calibration_run_ids": list(self.calibration_run_ids),
            "policy_selection_run_ids": list(self.policy_selection_run_ids),
            "sealed_test_run_accessed": False,
        }
        if include_sha256:
            payload["subpartition_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74TuningSubpartition:
        payload = dict(value)
        claimed = str(payload.pop("subpartition_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 tuning subpartition digest differs")
        try:
            selected = cls(
                parent_partition_sha256=str(payload["parent_partition_sha256"]),
                model_selection_run_ids=tuple(
                    str(item) for item in payload["model_selection_run_ids"]
                ),
                calibration_run_ids=tuple(
                    str(item) for item in payload["calibration_run_ids"]
                ),
                policy_selection_run_ids=tuple(
                    str(item) for item in payload["policy_selection_run_ids"]
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 tuning subpartition payload differs") from exc
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 tuning subpartition policy differs")
        selected.validate()
        return selected


def build_round74_tuning_subpartition(
    partition: Round74EventRunPartition,
) -> Round74TuningSubpartition:
    """Split exactly 24 chronological tuning runs without row reuse."""

    partition.validate()
    run_ids = tuple(
        entry.run_id for entry in partition.entries if entry.role == "tuning"
    )
    if len(run_ids) != ROUND74_TUNING_EXPECTED_RUNS:
        raise ValueError("Round 74 tuning run count differs")
    model_end = ROUND74_TUNING_MODEL_SELECTION_RUNS
    calibration_end = model_end + ROUND74_TUNING_CALIBRATION_RUNS
    selected = Round74TuningSubpartition(
        parent_partition_sha256=partition.partition_sha256,
        model_selection_run_ids=run_ids[:model_end],
        calibration_run_ids=run_ids[model_end:calibration_end],
        policy_selection_run_ids=run_ids[calibration_end:],
    )
    selected.validate()
    return selected


@dataclass(frozen=True)
class Round74TemperatureFit:
    """One bounded scalar temperature selected on calibration data."""

    temperature: float
    eligible_observations: int
    positive_observations: int
    calibration_runs: int
    minimum_run_observations: int
    maximum_run_observations: int
    uncalibrated_run_balanced_nll: float
    calibrated_run_balanced_nll: float
    uncalibrated_nll: float
    calibrated_nll: float
    uncalibrated_brier: float
    calibrated_brier: float
    uncalibrated_ece: float
    calibrated_ece: float

    def validate(self) -> None:
        metrics = (
            self.temperature,
            self.uncalibrated_run_balanced_nll,
            self.calibrated_run_balanced_nll,
            self.uncalibrated_nll,
            self.calibrated_nll,
            self.uncalibrated_brier,
            self.calibrated_brier,
            self.uncalibrated_ece,
            self.calibrated_ece,
        )
        if (
            any(not math.isfinite(float(value)) for value in metrics)
            or not ROUND74_TEMPERATURE_MINIMUM
            <= self.temperature
            <= ROUND74_TEMPERATURE_MAXIMUM
            or isinstance(self.eligible_observations, bool)
            or not isinstance(self.eligible_observations, int)
            or self.eligible_observations < 2
            or isinstance(self.positive_observations, bool)
            or not isinstance(self.positive_observations, int)
            or not 0 < self.positive_observations < self.eligible_observations
            or isinstance(self.calibration_runs, bool)
            or not isinstance(self.calibration_runs, int)
            or self.calibration_runs < 1
            or isinstance(self.minimum_run_observations, bool)
            or self.minimum_run_observations < 1
            or isinstance(self.maximum_run_observations, bool)
            or self.maximum_run_observations < self.minimum_run_observations
            or min(
                self.uncalibrated_run_balanced_nll,
                self.calibrated_run_balanced_nll,
                self.uncalibrated_nll,
                self.calibrated_nll,
                self.uncalibrated_brier,
                self.calibrated_brier,
                self.uncalibrated_ece,
                self.calibrated_ece,
            )
            < 0.0
        ):
            raise ValueError("Round 74 temperature fit differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "temperature": self.temperature,
            "eligible_observations": self.eligible_observations,
            "positive_observations": self.positive_observations,
            "calibration_runs": self.calibration_runs,
            "minimum_run_observations": self.minimum_run_observations,
            "maximum_run_observations": self.maximum_run_observations,
            "uncalibrated_run_balanced_nll": (self.uncalibrated_run_balanced_nll),
            "calibrated_run_balanced_nll": self.calibrated_run_balanced_nll,
            "uncalibrated_nll": self.uncalibrated_nll,
            "calibrated_nll": self.calibrated_nll,
            "uncalibrated_brier": self.uncalibrated_brier,
            "calibrated_brier": self.calibrated_brier,
            "uncalibrated_ece": self.uncalibrated_ece,
            "calibrated_ece": self.calibrated_ece,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Round74TemperatureFit:
        payload = dict(value)
        expected_keys = {
            "temperature",
            "eligible_observations",
            "positive_observations",
            "calibration_runs",
            "minimum_run_observations",
            "maximum_run_observations",
            "uncalibrated_run_balanced_nll",
            "calibrated_run_balanced_nll",
            "uncalibrated_nll",
            "calibrated_nll",
            "uncalibrated_brier",
            "calibrated_brier",
            "uncalibrated_ece",
            "calibrated_ece",
        }
        if set(payload) != expected_keys:
            raise ValueError("Round 74 temperature fit payload differs")

        def integer(name: str) -> int:
            selected = payload[name]
            if isinstance(selected, bool) or not isinstance(selected, int):
                raise ValueError("Round 74 temperature fit integer differs")
            return selected

        def number(name: str) -> float:
            selected = payload[name]
            if (
                isinstance(selected, bool)
                or not isinstance(selected, (int, float))
                or not math.isfinite(float(selected))
            ):
                raise ValueError("Round 74 temperature fit number differs")
            return float(selected)

        selected = cls(
            temperature=number("temperature"),
            eligible_observations=integer("eligible_observations"),
            positive_observations=integer("positive_observations"),
            calibration_runs=integer("calibration_runs"),
            minimum_run_observations=integer("minimum_run_observations"),
            maximum_run_observations=integer("maximum_run_observations"),
            uncalibrated_run_balanced_nll=number("uncalibrated_run_balanced_nll"),
            calibrated_run_balanced_nll=number("calibrated_run_balanced_nll"),
            uncalibrated_nll=number("uncalibrated_nll"),
            calibrated_nll=number("calibrated_nll"),
            uncalibrated_brier=number("uncalibrated_brier"),
            calibrated_brier=number("calibrated_brier"),
            uncalibrated_ece=number("uncalibrated_ece"),
            calibrated_ece=number("calibrated_ece"),
        )
        selected.validate()
        if _canonical_json(selected.as_dict()) != _canonical_json(payload):
            raise ValueError("Round 74 temperature fit encoding differs")
        return selected


def _risk_panel_shape() -> tuple[int, int]:
    return (
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )


def _validate_risk_matrix(
    value: Sequence[Sequence[float | int]],
    *,
    label: str,
    integer: bool = False,
) -> None:
    horizons, sides = _risk_panel_shape()
    if len(value) != horizons or any(len(row) != sides for row in value):
        raise ValueError(f"Round 74 risk calibration {label} shape differs")
    flattened = tuple(item for row in value for item in row)
    if integer:
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in flattened
        ):
            raise ValueError(f"Round 74 risk calibration {label} differs")
    elif any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in flattened
    ):
        raise ValueError(f"Round 74 risk calibration {label} differs")


@dataclass(frozen=True)
class Round74RiskQuantileCalibration:
    """One-sided calibration-run widening for deployed payoff and MAE tails."""

    payoff_lower_offsets_bps: tuple[
        tuple[tuple[float, float], ...],
        ...,
    ]
    mae_upper_offsets_bps: tuple[tuple[float, ...], ...]
    eligible_observations: tuple[tuple[int, ...], ...]
    payoff_lower_empirical_coverage_before: tuple[
        tuple[tuple[float, float], ...],
        ...,
    ]
    payoff_lower_empirical_coverage_after: tuple[
        tuple[tuple[float, float], ...],
        ...,
    ]
    mae_upper_empirical_coverage_before: tuple[tuple[float, ...], ...]
    mae_upper_empirical_coverage_after: tuple[tuple[float, ...], ...]
    calibration_runs: int
    optimization_population: str
    schema_version: str = ROUND74_RISK_QUANTILE_CALIBRATION_SCHEMA_VERSION

    def validate(self) -> None:
        horizons, sides = _risk_panel_shape()
        lower_panels = (
            self.payoff_lower_offsets_bps,
            self.payoff_lower_empirical_coverage_before,
            self.payoff_lower_empirical_coverage_after,
        )
        if (
            self.schema_version
            not in {
                ROUND74_RISK_QUANTILE_CALIBRATION_SCHEMA_VERSION,
                ROUND74_RISK_QUANTILE_CALIBRATION_PRIOR_SCHEMA_VERSION,
            }
            or self.optimization_population
            not in ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS
            or not _valid_population_run_count(
                self.calibration_runs,
                optimization_population=self.optimization_population,
                capture_run_count=ROUND74_TUNING_CALIBRATION_RUNS,
            )
            or any(len(panel) != horizons for panel in lower_panels)
            or any(len(row) != sides for panel in lower_panels for row in panel)
            or any(
                len(pair) != len(ROUND74_PAYOFF_LOWER_CALIBRATION_QUANTILES)
                for panel in lower_panels
                for row in panel
                for pair in row
            )
        ):
            raise ValueError("Round 74 risk quantile calibration differs")
        _validate_risk_matrix(
            self.mae_upper_offsets_bps,
            label="MAE offsets",
        )
        _validate_risk_matrix(
            self.eligible_observations,
            label="observation counts",
            integer=True,
        )
        _validate_risk_matrix(
            self.mae_upper_empirical_coverage_before,
            label="MAE coverage before",
        )
        _validate_risk_matrix(
            self.mae_upper_empirical_coverage_after,
            label="MAE coverage after",
        )
        offsets = (
            *(
                float(value)
                for row in self.payoff_lower_offsets_bps
                for pair in row
                for value in pair
            ),
            *(float(value) for row in self.mae_upper_offsets_bps for value in row),
        )
        coverage = (
            *(
                float(value)
                for panel in (
                    self.payoff_lower_empirical_coverage_before,
                    self.payoff_lower_empirical_coverage_after,
                )
                for row in panel
                for pair in row
                for value in pair
            ),
            *(
                float(value)
                for panel in (
                    self.mae_upper_empirical_coverage_before,
                    self.mae_upper_empirical_coverage_after,
                )
                for row in panel
                for value in row
            ),
        )
        if (
            any(value < 0.0 for value in offsets)
            or any(not 0.0 <= value <= 1.0 for value in coverage)
            or any(
                after + 1e-12 < before
                for before_panel, after_panel in zip(
                    self.payoff_lower_empirical_coverage_before,
                    self.payoff_lower_empirical_coverage_after,
                    strict=True,
                )
                for before_row, after_row in zip(
                    before_panel,
                    after_panel,
                    strict=True,
                )
                for before, after in zip(before_row, after_row, strict=True)
            )
            or any(
                after + 1e-12 < before
                for before_row, after_row in zip(
                    self.mae_upper_empirical_coverage_before,
                    self.mae_upper_empirical_coverage_after,
                    strict=True,
                )
                for before, after in zip(before_row, after_row, strict=True)
            )
            or any(
                after + 1e-12 < 1.0 - nominal
                for row in self.payoff_lower_empirical_coverage_after
                for pair in row
                for nominal, after in zip(
                    ROUND74_PAYOFF_LOWER_CALIBRATION_QUANTILES,
                    pair,
                    strict=True,
                )
            )
            or any(
                after + 1e-12 < ROUND74_MAE_UPPER_CALIBRATION_QUANTILE
                for row in self.mae_upper_empirical_coverage_after
                for after in row
            )
        ):
            raise ValueError("Round 74 risk quantile calibration bounds differ")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "method": "one_sided_split_conformal_nonnegative_widening",
            "payoff_lower_nominal_quantiles": list(
                ROUND74_PAYOFF_LOWER_CALIBRATION_QUANTILES
            ),
            "mae_upper_nominal_quantile": (ROUND74_MAE_UPPER_CALIBRATION_QUANTILE),
            "payoff_lower_offsets_bps": [
                [list(pair) for pair in row] for row in self.payoff_lower_offsets_bps
            ],
            "mae_upper_offsets_bps": [list(row) for row in self.mae_upper_offsets_bps],
            "eligible_observations": [list(row) for row in self.eligible_observations],
            "payoff_lower_empirical_coverage_before": [
                [list(pair) for pair in row]
                for row in self.payoff_lower_empirical_coverage_before
            ],
            "payoff_lower_empirical_coverage_after": [
                [list(pair) for pair in row]
                for row in self.payoff_lower_empirical_coverage_after
            ],
            "mae_upper_empirical_coverage_before": [
                list(row) for row in self.mae_upper_empirical_coverage_before
            ],
            "mae_upper_empirical_coverage_after": [
                list(row) for row in self.mae_upper_empirical_coverage_after
            ],
            "calibration_runs": self.calibration_runs,
            "optimization_population": self.optimization_population,
            **(
                {
                    "capture_run_grouping": "capture_run_x_symbol",
                    "capture_run_coverage_aggregation": (
                        "minimum_over_capture_run_symbol_groups"
                    ),
                    "eligible_target_grouping": (
                        "pooled_with_complete_capture_run_x_symbol_support_required"
                    ),
                    "calibration_symbols": list(ROUND74_EVENT_SYMBOLS),
                }
                if self.schema_version
                == ROUND74_RISK_QUANTILE_CALIBRATION_SCHEMA_VERSION
                else {}
            ),
            "nonnegative_widening_only": True,
            "exchangeability_or_future_coverage_claim": False,
            "sealed_test_accessed": False,
            "calibration_implies_financial_edge": False,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74RiskQuantileCalibration:
        payload = dict(value)
        expected = {
            "schema_version",
            "method",
            "payoff_lower_nominal_quantiles",
            "mae_upper_nominal_quantile",
            "payoff_lower_offsets_bps",
            "mae_upper_offsets_bps",
            "eligible_observations",
            "payoff_lower_empirical_coverage_before",
            "payoff_lower_empirical_coverage_after",
            "mae_upper_empirical_coverage_before",
            "mae_upper_empirical_coverage_after",
            "calibration_runs",
            "optimization_population",
            "nonnegative_widening_only",
            "exchangeability_or_future_coverage_claim",
            "sealed_test_accessed",
            "calibration_implies_financial_edge",
        }
        current = (
            payload.get("schema_version")
            == ROUND74_RISK_QUANTILE_CALIBRATION_SCHEMA_VERSION
        )
        if current:
            expected.update(
                {
                    "capture_run_grouping",
                    "capture_run_coverage_aggregation",
                    "eligible_target_grouping",
                    "calibration_symbols",
                }
            )
        if set(payload) != expected:
            raise ValueError("Round 74 risk quantile payload differs")

        def matrix(name: str, *, integer: bool = False):
            raw = payload[name]
            if not isinstance(raw, list) or any(
                not isinstance(row, list) for row in raw
            ):
                raise ValueError("Round 74 risk quantile matrix differs")
            items = tuple(item for row in raw for item in row)
            if integer:
                invalid = any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in items
                )
            else:
                invalid = any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in items
                )
            if invalid:
                raise ValueError("Round 74 risk quantile matrix values differ")
            conversion = int if integer else float
            return tuple(tuple(conversion(item) for item in row) for row in raw)

        def lower(name: str):
            raw = payload[name]
            if (
                not isinstance(raw, list)
                or any(not isinstance(row, list) for row in raw)
                or any(not isinstance(pair, list) for row in raw for pair in row)
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for row in raw
                    for pair in row
                    for item in pair
                )
            ):
                raise ValueError("Round 74 risk quantile lower panel differs")
            return tuple(
                tuple(tuple(float(item) for item in pair) for pair in row)
                for row in raw
            )

        selected = cls(
            payoff_lower_offsets_bps=lower("payoff_lower_offsets_bps"),
            mae_upper_offsets_bps=matrix("mae_upper_offsets_bps"),
            eligible_observations=matrix(
                "eligible_observations",
                integer=True,
            ),
            payoff_lower_empirical_coverage_before=lower(
                "payoff_lower_empirical_coverage_before"
            ),
            payoff_lower_empirical_coverage_after=lower(
                "payoff_lower_empirical_coverage_after"
            ),
            mae_upper_empirical_coverage_before=matrix(
                "mae_upper_empirical_coverage_before"
            ),
            mae_upper_empirical_coverage_after=matrix(
                "mae_upper_empirical_coverage_after"
            ),
            calibration_runs=int(payload["calibration_runs"]),
            optimization_population=str(payload["optimization_population"]),
            schema_version=str(payload["schema_version"]),
        )
        selected.validate()
        if (
            payload["method"] != "one_sided_split_conformal_nonnegative_widening"
            or payload["payoff_lower_nominal_quantiles"]
            != list(ROUND74_PAYOFF_LOWER_CALIBRATION_QUANTILES)
            or payload["mae_upper_nominal_quantile"]
            != ROUND74_MAE_UPPER_CALIBRATION_QUANTILE
            or payload["nonnegative_widening_only"] is not True
            or payload["exchangeability_or_future_coverage_claim"] is not False
            or payload["sealed_test_accessed"] is not False
            or payload["calibration_implies_financial_edge"] is not False
            or current
            and (
                payload["capture_run_grouping"] != "capture_run_x_symbol"
                or payload["capture_run_coverage_aggregation"]
                != "minimum_over_capture_run_symbol_groups"
                or payload["eligible_target_grouping"]
                != "pooled_with_complete_capture_run_x_symbol_support_required"
                or payload["calibration_symbols"] != list(ROUND74_EVENT_SYMBOLS)
            )
            or selected.as_dict() != payload
        ):
            raise ValueError("Round 74 risk quantile policy differs")
        return selected


def _freeze_quantile_baseline_panel(
    value: np.ndarray,
) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]:
    return tuple(
        tuple(
            tuple(
                tuple(float(item) for item in value[symbol, horizon, side])
                for side in range(value.shape[2])
            )
            for horizon in range(value.shape[1])
        )
        for symbol in range(value.shape[0])
    )


def _is_strict_numeric_tree(value: object, *, integer: bool) -> bool:
    if isinstance(value, (list, tuple)):
        return bool(value) and all(
            _is_strict_numeric_tree(item, integer=integer) for item in value
        )
    if isinstance(value, bool):
        return False
    if integer:
        return isinstance(value, int)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _equal_run_empirical_quantiles(
    values: np.ndarray,
    run_ids: np.ndarray,
    expected_run_ids: tuple[str, ...],
    quantiles: np.ndarray,
) -> np.ndarray:
    selected_values = np.asarray(values, dtype=np.float64)
    selected_runs = np.asarray(run_ids, dtype=object)
    if (
        selected_values.ndim != 1
        or selected_runs.shape != selected_values.shape
        or not selected_values.size
        or not np.isfinite(selected_values).all()
        or set(str(value) for value in selected_runs) != set(expected_run_ids)
    ):
        raise ValueError("Round 74 equal-run empirical distribution differs")
    run_counts = {
        run_id: int(np.count_nonzero(selected_runs == run_id))
        for run_id in expected_run_ids
    }
    if any(count < 1 for count in run_counts.values()):
        raise ValueError("Round 74 equal-run empirical support differs")
    observation_weights = np.asarray(
        tuple(1.0 / run_counts[str(run_id)] for run_id in selected_runs),
        dtype=np.float64,
    )
    support, inverse = np.unique(selected_values, return_inverse=True)
    support_weights = np.bincount(
        inverse,
        weights=observation_weights,
        minlength=support.size,
    )
    cumulative_probability = np.cumsum(support_weights, dtype=np.float64)
    cumulative_probability /= cumulative_probability[-1]
    cumulative_probability[-1] = 1.0
    quantile_indices = np.searchsorted(
        cumulative_probability + 1e-12,
        quantiles,
        side="left",
    )
    return support[np.minimum(quantile_indices, support.size - 1)]


@dataclass(frozen=True)
class Round74NoInformationQuantileBaseline:
    """Calibration-only unconditional distributions for sealed skill scoring."""

    payoff_quantiles_bps: tuple[
        tuple[tuple[tuple[float, ...], ...], ...],
        ...,
    ]
    maximum_adverse_excursion_quantiles_bps: tuple[
        tuple[tuple[tuple[float, ...], ...], ...],
        ...,
    ]
    eligible_observations: tuple[tuple[tuple[int, ...], ...], ...]
    calibration_runs: int
    optimization_population: str
    schema_version: str = ROUND74_NO_INFORMATION_QUANTILE_BASELINE_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            not _is_strict_numeric_tree(
                self.payoff_quantiles_bps,
                integer=False,
            )
            or not _is_strict_numeric_tree(
                self.maximum_adverse_excursion_quantiles_bps,
                integer=False,
            )
            or not _is_strict_numeric_tree(
                self.eligible_observations,
                integer=True,
            )
        ):
            raise ValueError("Round 74 no-information quantile types differ")
        expected_quantile_shape = (
            len(ROUND74_EVENT_SYMBOLS),
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            len(ROUND74_EVENT_PAYOFF_SIDES),
            len(ROUND74_EVENT_PAYOFF_QUANTILES),
        )
        expected_count_shape = expected_quantile_shape[:3]
        payoff = np.asarray(self.payoff_quantiles_bps, dtype=np.float64)
        adverse = np.asarray(
            self.maximum_adverse_excursion_quantiles_bps,
            dtype=np.float64,
        )
        observations = np.asarray(self.eligible_observations, dtype=np.int64)
        if (
            self.schema_version
            != ROUND74_NO_INFORMATION_QUANTILE_BASELINE_SCHEMA_VERSION
            or self.optimization_population
            not in ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS
            or not _valid_population_run_count(
                self.calibration_runs,
                optimization_population=self.optimization_population,
                capture_run_count=ROUND74_TUNING_CALIBRATION_RUNS,
            )
            or payoff.shape != expected_quantile_shape
            or adverse.shape != expected_quantile_shape
            or observations.shape != expected_count_shape
            or not np.isfinite(payoff).all()
            or not np.isfinite(adverse).all()
            or np.any(observations < 1)
            or np.any(np.diff(payoff, axis=3) < 0.0)
            or np.any(np.diff(adverse, axis=3) < 0.0)
            or np.any(np.diff(adverse, axis=1) < 0.0)
            or np.any(adverse < 0.0)
        ):
            raise ValueError("Round 74 no-information quantile baseline differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "symbols": list(ROUND74_EVENT_SYMBOLS),
            "horizons_seconds": list(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            "sides": list(ROUND74_EVENT_PAYOFF_SIDES),
            "quantiles": list(ROUND74_EVENT_PAYOFF_QUANTILES),
            "payoff_quantiles_bps": [
                [[list(values) for values in horizon] for horizon in symbol]
                for symbol in self.payoff_quantiles_bps
            ],
            "maximum_adverse_excursion_quantiles_bps": [
                [[list(values) for values in horizon] for horizon in symbol]
                for symbol in self.maximum_adverse_excursion_quantiles_bps
            ],
            "eligible_observations": [
                [list(horizon) for horizon in symbol]
                for symbol in self.eligible_observations
            ],
            "calibration_runs": self.calibration_runs,
            "optimization_population": self.optimization_population,
            "fit_population": "disjoint_probability_calibration_runs_only",
            "capture_run_quantile_method": (
                "equal_capture_run_mass_weighted_empirical_inverse_cdf"
            ),
            "eligible_target_quantile_method": (
                "pooled_eligible_target_linear_empirical_quantile"
            ),
            "capture_run_x_symbol_support_required": True,
            "sealed_test_accessed": False,
            "test_labels_used_for_baseline_fit": False,
            "baseline_implies_predictive_or_financial_edge": False,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74NoInformationQuantileBaseline:
        payload = dict(value)
        expected = {
            "schema_version",
            "symbols",
            "horizons_seconds",
            "sides",
            "quantiles",
            "payoff_quantiles_bps",
            "maximum_adverse_excursion_quantiles_bps",
            "eligible_observations",
            "calibration_runs",
            "optimization_population",
            "fit_population",
            "capture_run_quantile_method",
            "eligible_target_quantile_method",
            "capture_run_x_symbol_support_required",
            "sealed_test_accessed",
            "test_labels_used_for_baseline_fit",
            "baseline_implies_predictive_or_financial_edge",
        }
        if set(payload) != expected:
            raise ValueError("Round 74 no-information quantile payload differs")
        if (
            not _is_strict_numeric_tree(
                payload["payoff_quantiles_bps"],
                integer=False,
            )
            or not _is_strict_numeric_tree(
                payload["maximum_adverse_excursion_quantiles_bps"],
                integer=False,
            )
            or not _is_strict_numeric_tree(
                payload["eligible_observations"],
                integer=True,
            )
            or isinstance(payload["calibration_runs"], bool)
            or not isinstance(payload["calibration_runs"], int)
        ):
            raise ValueError("Round 74 no-information quantile types differ")
        try:
            payoff = np.asarray(payload["payoff_quantiles_bps"], dtype=np.float64)
            adverse = np.asarray(
                payload["maximum_adverse_excursion_quantiles_bps"],
                dtype=np.float64,
            )
            observations = np.asarray(
                payload["eligible_observations"],
                dtype=np.int64,
            )
            selected = cls(
                payoff_quantiles_bps=_freeze_quantile_baseline_panel(payoff),
                maximum_adverse_excursion_quantiles_bps=(
                    _freeze_quantile_baseline_panel(adverse)
                ),
                eligible_observations=tuple(
                    tuple(
                        tuple(int(item) for item in observations[symbol, horizon])
                        for horizon in range(observations.shape[1])
                    )
                    for symbol in range(observations.shape[0])
                ),
                calibration_runs=int(payload["calibration_runs"]),
                optimization_population=str(payload["optimization_population"]),
                schema_version=str(payload["schema_version"]),
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 no-information quantile payload differs"
            ) from exc
        selected.validate()
        if (
            payload["symbols"] != list(ROUND74_EVENT_SYMBOLS)
            or payload["horizons_seconds"]
            != list(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
            or payload["sides"] != list(ROUND74_EVENT_PAYOFF_SIDES)
            or payload["quantiles"] != list(ROUND74_EVENT_PAYOFF_QUANTILES)
            or payload["fit_population"] != "disjoint_probability_calibration_runs_only"
            or payload["capture_run_quantile_method"]
            != "equal_capture_run_mass_weighted_empirical_inverse_cdf"
            or payload["eligible_target_quantile_method"]
            != "pooled_eligible_target_linear_empirical_quantile"
            or payload["capture_run_x_symbol_support_required"] is not True
            or payload["sealed_test_accessed"] is not False
            or payload["test_labels_used_for_baseline_fit"] is not False
            or payload["baseline_implies_predictive_or_financial_edge"] is not False
            or selected.as_dict() != payload
        ):
            raise ValueError("Round 74 no-information quantile policy differs")
        return selected


def fit_round74_no_information_quantile_baseline(
    *,
    net_payoff_bps: torch.Tensor,
    maximum_adverse_excursion_bps: torch.Tensor,
    action_eligibility: torch.Tensor,
    row_run_ids: tuple[str, ...],
    row_symbols: tuple[str, ...],
    expected_run_ids: tuple[str, ...],
    optimization_population: str,
) -> Round74NoInformationQuantileBaseline:
    """Fit fixed empirical quantiles on disjoint calibration runs only."""

    selected_runs = tuple(str(value) for value in row_run_ids)
    selected_symbols = tuple(str(value) for value in row_symbols)
    expected = tuple(str(value) for value in expected_run_ids)
    selected_population = str(optimization_population)
    rows = len(selected_runs)
    target_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    tensors = (
        net_payoff_bps,
        maximum_adverse_excursion_bps,
        action_eligibility,
    )
    if (
        selected_population not in ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS
        or expected != tuple(dict.fromkeys(expected))
        or not _valid_population_run_count(
            len(expected),
            optimization_population=selected_population,
            capture_run_count=ROUND74_TUNING_CALIBRATION_RUNS,
        )
        or set(selected_runs) != set(expected)
        or len(selected_symbols) != rows
        or set(selected_symbols) != set(ROUND74_EVENT_SYMBOLS)
        or any(value not in ROUND74_EVENT_SYMBOLS for value in selected_symbols)
        or any(value.shape != target_shape for value in tensors)
        or any(not value.is_floating_point() for value in tensors)
        or len({value.device for value in tensors}) != 1
        or any(not bool(torch.isfinite(value).all()) for value in tensors)
        or bool((maximum_adverse_excursion_bps < 0.0).any())
        or bool(((action_eligibility != 0.0) & (action_eligibility != 1.0)).any())
    ):
        raise ValueError("Round 74 no-information quantile panel differs")
    payoff = net_payoff_bps.detach().to(device="cpu", dtype=torch.float64).numpy()
    adverse = (
        maximum_adverse_excursion_bps.detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )
    eligible = (
        action_eligibility.detach().to(device="cpu", dtype=torch.float64).numpy() == 1.0
    )
    run_array = np.asarray(selected_runs, dtype=object)
    symbol_array = np.asarray(selected_symbols, dtype=object)
    quantiles = np.asarray(ROUND74_EVENT_PAYOFF_QUANTILES, dtype=np.float64)
    baseline_shape = (
        len(ROUND74_EVENT_SYMBOLS),
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
        len(ROUND74_EVENT_PAYOFF_QUANTILES),
    )
    payoff_baseline = np.zeros(baseline_shape, dtype=np.float64)
    adverse_baseline = np.zeros_like(payoff_baseline)
    observations = np.zeros(baseline_shape[:3], dtype=np.int64)
    for symbol_index, symbol in enumerate(ROUND74_EVENT_SYMBOLS):
        symbol_mask = symbol_array == symbol
        for horizon_index in range(baseline_shape[1]):
            for side_index in range(baseline_shape[2]):
                mask = symbol_mask & eligible[:, horizon_index, side_index]
                if not bool(mask.any()) or any(
                    not bool((mask & (run_array == run_id)).any())
                    for run_id in expected
                ):
                    raise ValueError("Round 74 no-information quantile support differs")
                observations[symbol_index, horizon_index, side_index] = int(mask.sum())
                if selected_population == "capture_run":
                    payoff_baseline[symbol_index, horizon_index, side_index] = (
                        _equal_run_empirical_quantiles(
                            payoff[mask, horizon_index, side_index],
                            run_array[mask],
                            expected,
                            quantiles,
                        )
                    )
                    adverse_baseline[symbol_index, horizon_index, side_index] = (
                        _equal_run_empirical_quantiles(
                            adverse[mask, horizon_index, side_index],
                            run_array[mask],
                            expected,
                            quantiles,
                        )
                    )
                else:
                    payoff_baseline[symbol_index, horizon_index, side_index] = (
                        np.quantile(
                            payoff[mask, horizon_index, side_index],
                            quantiles,
                            method="linear",
                        )
                    )
                    adverse_baseline[symbol_index, horizon_index, side_index] = (
                        np.quantile(
                            adverse[mask, horizon_index, side_index],
                            quantiles,
                            method="linear",
                        )
                    )
    adverse_baseline = np.maximum.accumulate(adverse_baseline, axis=1)
    result = Round74NoInformationQuantileBaseline(
        payoff_quantiles_bps=_freeze_quantile_baseline_panel(payoff_baseline),
        maximum_adverse_excursion_quantiles_bps=(
            _freeze_quantile_baseline_panel(adverse_baseline)
        ),
        eligible_observations=tuple(
            tuple(
                tuple(int(item) for item in observations[symbol, horizon])
                for horizon in range(observations.shape[1])
            )
            for symbol in range(observations.shape[0])
        ),
        calibration_runs=len(expected),
        optimization_population=selected_population,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class Round74ProbabilityCalibration:
    """Hash-bound probability temperatures and optional risk-tail widening."""

    pretest_policy_sha256: str
    tuning_subpartition_sha256: str
    calibration_source_sha256: str
    calibration_data_sha256: str
    calibration_run_ids: tuple[str, ...]
    calibration_row_run_ids_sha256: str
    positive_payoff: Round74TemperatureFit
    adverse_selection: Round74TemperatureFit
    regime_unpredictability: Round74TemperatureFit
    backend_kind: str
    backend_device: str
    risk_quantiles: Round74RiskQuantileCalibration | None = None
    quantile_baseline: Round74NoInformationQuantileBaseline | None = None
    optimization_population: str = "capture_run"
    schema_version: str = ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version
            not in {
                ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION,
                ROUND74_TEMPERATURE_CALIBRATION_RISK_PRIOR_SCHEMA_VERSION,
                ROUND74_TEMPERATURE_CALIBRATION_PRIOR_SCHEMA_VERSION,
                ROUND74_TEMPERATURE_CALIBRATION_LEGACY_SCHEMA_VERSION,
            }
            or self.optimization_population
            not in ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS
            or (
                self.schema_version
                == ROUND74_TEMPERATURE_CALIBRATION_LEGACY_SCHEMA_VERSION
                and self.optimization_population != "capture_run"
            )
            or (
                self.schema_version == ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION
                and self.risk_quantiles is None
            )
            or (
                self.schema_version
                == ROUND74_TEMPERATURE_CALIBRATION_RISK_PRIOR_SCHEMA_VERSION
                and self.risk_quantiles is None
            )
            or (
                self.schema_version
                not in {
                    ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION,
                    ROUND74_TEMPERATURE_CALIBRATION_RISK_PRIOR_SCHEMA_VERSION,
                }
                and self.risk_quantiles is not None
            )
            or (
                self.schema_version == ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION
                and self.quantile_baseline is None
            )
            or (
                self.schema_version != ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION
                and self.quantile_baseline is not None
            )
            or not self.backend_kind.strip()
            or not self.backend_device.strip()
        ):
            raise ValueError("Round 74 probability calibration differs")
        _require_sha256(self.pretest_policy_sha256, "pretest policy")
        _require_sha256(
            self.tuning_subpartition_sha256,
            "tuning subpartition",
        )
        _require_sha256(
            self.calibration_source_sha256,
            "calibration source",
        )
        _require_sha256(self.calibration_data_sha256, "calibration data")
        calibration_run_count = len(self.calibration_run_ids)
        if (
            not _valid_population_run_count(
                calibration_run_count,
                optimization_population=self.optimization_population,
                capture_run_count=ROUND74_TUNING_CALIBRATION_RUNS,
            )
            or len(set(self.calibration_run_ids)) != calibration_run_count
            or any(
                len(value) != 32
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.calibration_run_ids
            )
        ):
            raise ValueError("Round 74 probability calibration run identity differs")
        _require_sha256(
            self.calibration_row_run_ids_sha256,
            "calibration row run ids",
        )
        self.positive_payoff.validate()
        self.adverse_selection.validate()
        self.regime_unpredictability.validate()
        if self.risk_quantiles is not None:
            self.risk_quantiles.validate()
            if (
                self.risk_quantiles.optimization_population
                != self.optimization_population
            ):
                raise ValueError(
                    "Round 74 probability and risk calibration populations differ"
                )
        if self.quantile_baseline is not None:
            self.quantile_baseline.validate()
            if (
                self.quantile_baseline.optimization_population
                != self.optimization_population
            ):
                raise ValueError(
                    "Round 74 probability and quantile baseline populations differ"
                )
        fits = (
            self.positive_payoff,
            self.adverse_selection,
            self.regime_unpredictability,
        )
        if any(fit.calibration_runs != calibration_run_count for fit in fits):
            raise ValueError("Round 74 calibration fit run count differs")
        if (
            self.risk_quantiles is not None
            and self.risk_quantiles.calibration_runs != calibration_run_count
        ):
            raise ValueError("Round 74 risk calibration run count differs")
        if (
            self.quantile_baseline is not None
            and self.quantile_baseline.calibration_runs != calibration_run_count
        ):
            raise ValueError("Round 74 quantile baseline run count differs")
        if self.optimization_population == "capture_run":
            worsened = any(
                fit.calibrated_run_balanced_nll
                > fit.uncalibrated_run_balanced_nll + 1e-7
                for fit in fits
            )
        else:
            worsened = any(
                fit.calibrated_nll > fit.uncalibrated_nll + 1e-7 for fit in fits
            )
        if worsened:
            raise ValueError("Round 74 calibration optimization objective worsened")

    @property
    def calibration_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "tuning_subpartition_sha256": (self.tuning_subpartition_sha256),
            "calibration_source_sha256": (self.calibration_source_sha256),
            "calibration_data_sha256": self.calibration_data_sha256,
            "calibration_run_ids": list(self.calibration_run_ids),
            "calibration_row_run_ids_sha256": (self.calibration_row_run_ids_sha256),
            "positive_payoff": self.positive_payoff.as_dict(),
            "adverse_selection": self.adverse_selection.as_dict(),
            "regime_unpredictability": (self.regime_unpredictability.as_dict()),
            "backend_kind": self.backend_kind,
            "backend_device": self.backend_device,
            "candidate_temperature_count": (ROUND74_TEMPERATURE_CANDIDATE_COUNT),
            "candidate_temperature_minimum": (ROUND74_TEMPERATURE_MINIMUM),
            "candidate_temperature_maximum": (ROUND74_TEMPERATURE_MAXIMUM),
            "selection_objective": (
                "equal_run_weight_binary_cross_entropy_on_calibration_runs_only"
                if self.optimization_population == "capture_run"
                else "eligible_target_weight_binary_cross_entropy_on_calibration_runs_only"
            ),
            "pooled_metrics_are_diagnostic_only": (
                self.optimization_population == "capture_run"
            ),
            "sealed_test_accessed": False,
            "calibration_implies_financial_edge": False,
        }
        if self.schema_version != ROUND74_TEMPERATURE_CALIBRATION_LEGACY_SCHEMA_VERSION:
            payload["optimization_population"] = self.optimization_population
        if self.schema_version in {
            ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION,
            ROUND74_TEMPERATURE_CALIBRATION_RISK_PRIOR_SCHEMA_VERSION,
        }:
            assert self.risk_quantiles is not None
            payload["risk_quantiles"] = self.risk_quantiles.as_dict()
        if self.schema_version == ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION:
            assert self.quantile_baseline is not None
            payload["quantile_baseline"] = self.quantile_baseline.as_dict()
        if include_sha256:
            payload["calibration_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74ProbabilityCalibration:
        payload = dict(value)
        claimed = payload.pop("calibration_sha256", None)
        if not _is_sha256(claimed) or claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 probability calibration digest differs")
        schema_version = payload.get("schema_version")
        legacy = schema_version == ROUND74_TEMPERATURE_CALIBRATION_LEGACY_SCHEMA_VERSION
        current = schema_version == ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION
        risk_prior = (
            schema_version == ROUND74_TEMPERATURE_CALIBRATION_RISK_PRIOR_SCHEMA_VERSION
        )
        has_risk = current or risk_prior
        expected_keys = {
            "schema_version",
            "pretest_policy_sha256",
            "tuning_subpartition_sha256",
            "calibration_source_sha256",
            "calibration_data_sha256",
            "calibration_run_ids",
            "calibration_row_run_ids_sha256",
            "positive_payoff",
            "adverse_selection",
            "regime_unpredictability",
            "backend_kind",
            "backend_device",
            "candidate_temperature_count",
            "candidate_temperature_minimum",
            "candidate_temperature_maximum",
            "selection_objective",
            "pooled_metrics_are_diagnostic_only",
            "sealed_test_accessed",
            "calibration_implies_financial_edge",
        }
        if not legacy:
            expected_keys.add("optimization_population")
        if has_risk:
            expected_keys.add("risk_quantiles")
        if current:
            expected_keys.add("quantile_baseline")
        if set(payload) != expected_keys:
            raise ValueError("Round 74 probability calibration payload differs")
        run_ids = payload["calibration_run_ids"]
        nested = (
            payload["positive_payoff"],
            payload["adverse_selection"],
            payload["regime_unpredictability"],
        )
        risk_quantiles = payload.get("risk_quantiles")
        quantile_baseline = payload.get("quantile_baseline")
        if (
            not isinstance(run_ids, list)
            or any(not isinstance(item, str) for item in run_ids)
            or any(not isinstance(item, Mapping) for item in nested)
            or (has_risk and not isinstance(risk_quantiles, Mapping))
            or (current and not isinstance(quantile_baseline, Mapping))
        ):
            raise ValueError("Round 74 probability calibration types differ")
        try:
            selected = cls(
                pretest_policy_sha256=str(payload["pretest_policy_sha256"]),
                tuning_subpartition_sha256=str(payload["tuning_subpartition_sha256"]),
                calibration_source_sha256=str(payload["calibration_source_sha256"]),
                calibration_data_sha256=str(payload["calibration_data_sha256"]),
                calibration_run_ids=tuple(run_ids),
                calibration_row_run_ids_sha256=str(
                    payload["calibration_row_run_ids_sha256"]
                ),
                positive_payoff=Round74TemperatureFit.from_dict(nested[0]),
                adverse_selection=Round74TemperatureFit.from_dict(nested[1]),
                regime_unpredictability=Round74TemperatureFit.from_dict(nested[2]),
                backend_kind=str(payload["backend_kind"]),
                backend_device=str(payload["backend_device"]),
                risk_quantiles=(
                    Round74RiskQuantileCalibration.from_dict(risk_quantiles)
                    if has_risk and isinstance(risk_quantiles, Mapping)
                    else None
                ),
                quantile_baseline=(
                    Round74NoInformationQuantileBaseline.from_dict(quantile_baseline)
                    if current and isinstance(quantile_baseline, Mapping)
                    else None
                ),
                optimization_population=(
                    "capture_run" if legacy else str(payload["optimization_population"])
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 probability calibration payload differs"
            ) from exc
        selected.validate()
        if _canonical_json(selected.as_dict(include_sha256=False)) != _canonical_json(
            payload
        ):
            raise ValueError("Round 74 probability calibration policy differs")
        if selected.calibration_sha256 != claimed:
            raise ValueError("Round 74 probability calibration identity differs")
        return selected


def _validate_binary_panel(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eligibility: torch.Tensor,
    *,
    row_run_ids: tuple[str, ...],
    expected_run_ids: tuple[str, ...],
    label: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[tuple[torch.Tensor, torch.Tensor], ...],
]:
    if (
        logits.shape != labels.shape
        or logits.shape != eligibility.shape
        or not logits.is_floating_point()
        or not labels.is_floating_point()
        or not eligibility.is_floating_point()
        or logits.device != labels.device
        or logits.device != eligibility.device
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(labels).all())
        or not bool(torch.isfinite(eligibility).all())
        or bool(((eligibility != 0.0) & (eligibility != 1.0)).any())
        or bool(((labels < 0.0) | (labels > 1.0)).any())
    ):
        raise ValueError(f"Round 74 {label} calibration panel differs")
    mask = eligibility == 1.0
    selected_logits = logits.detach()[mask].to(dtype=torch.float32)
    selected_labels = labels.detach()[mask].to(dtype=torch.float32)
    if (
        selected_logits.numel() < 2
        or not bool((selected_labels == 0.0).any())
        or not bool((selected_labels == 1.0).any())
    ):
        raise ValueError(f"Round 74 {label} calibration class support differs")
    row_shape = (len(row_run_ids),) + (1,) * (logits.ndim - 1)
    run_panels: list[tuple[torch.Tensor, torch.Tensor]] = []
    for run_id in expected_run_ids:
        row_mask = torch.tensor(
            tuple(value == run_id for value in row_run_ids),
            dtype=torch.bool,
            device=logits.device,
        ).reshape(row_shape)
        run_mask = mask & row_mask
        run_logits = logits.detach()[run_mask].to(dtype=torch.float32)
        run_labels = labels.detach()[run_mask].to(dtype=torch.float32)
        if run_logits.numel() < 1:
            raise ValueError(f"Round 74 {label} calibration run support differs")
        run_panels.append((run_logits, run_labels))
    return selected_logits, selected_labels, tuple(run_panels)


def _expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    boundaries = torch.linspace(
        0.0,
        1.0,
        ROUND74_TEMPERATURE_ECE_BINS + 1,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    total = float(probabilities.numel())
    ece = torch.zeros(
        (),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    for index in range(ROUND74_TEMPERATURE_ECE_BINS):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index + 1 == ROUND74_TEMPERATURE_ECE_BINS:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum().item())
        if count:
            confidence = probabilities[mask].mean()
            frequency = labels[mask].mean()
            ece = ece + abs(confidence - frequency) * (count / total)
    return float(ece.item())


def _finite_sample_higher_quantile(
    values: np.ndarray,
    *,
    level: float,
) -> float:
    selected = np.asarray(values, dtype=np.float64)
    if (
        selected.ndim != 1
        or selected.size < 1
        or not np.isfinite(selected).all()
        or not 0.0 < float(level) < 1.0
    ):
        raise ValueError("Round 74 risk quantile sample differs")
    ordered = np.sort(selected)
    rank = min(
        selected.size,
        int(math.ceil((selected.size + 1) * float(level))),
    )
    return float(ordered[rank - 1])


def _risk_quantile_offset(
    residuals: np.ndarray,
    row_run_ids: tuple[str, ...],
    row_symbols: tuple[str, ...],
    *,
    expected_run_ids: tuple[str, ...],
    eligibility: np.ndarray,
    level: float,
    optimization_population: str,
) -> float:
    selected = np.asarray(residuals, dtype=np.float64)
    mask = np.asarray(eligibility, dtype=np.bool_)
    if (
        selected.shape != mask.shape
        or selected.shape != (len(row_run_ids),)
        or len(row_symbols) != len(row_run_ids)
        or not np.isfinite(selected).all()
    ):
        raise ValueError("Round 74 risk quantile residual panel differs")
    group_panels = tuple(
        selected[
            np.asarray(
                [
                    run_id == expected_run_id
                    and symbol == expected_symbol
                    and bool(eligible)
                    for run_id, symbol, eligible in zip(
                        row_run_ids,
                        row_symbols,
                        mask,
                        strict=True,
                    )
                ],
                dtype=np.bool_,
            )
        ]
        for expected_run_id in expected_run_ids
        for expected_symbol in ROUND74_EVENT_SYMBOLS
    )
    if any(panel.size < 1 for panel in group_panels):
        raise ValueError("Round 74 risk quantile run-symbol support differs")
    if optimization_population == "capture_run":
        correction = max(
            _finite_sample_higher_quantile(panel, level=level) for panel in group_panels
        )
    elif optimization_population == "eligible_target":
        correction = _finite_sample_higher_quantile(
            selected[mask],
            level=level,
        )
    else:
        raise ValueError("Round 74 risk quantile population differs")
    return max(0.0, float(correction))


def _risk_empirical_coverage(
    covered: np.ndarray,
    row_run_ids: tuple[str, ...],
    row_symbols: tuple[str, ...],
    *,
    expected_run_ids: tuple[str, ...],
    eligibility: np.ndarray,
    optimization_population: str,
) -> float:
    selected = np.asarray(covered, dtype=np.bool_)
    mask = np.asarray(eligibility, dtype=np.bool_)
    if (
        selected.shape != mask.shape
        or selected.shape != (len(row_run_ids),)
        or len(row_symbols) != len(row_run_ids)
    ):
        raise ValueError("Round 74 risk coverage panel differs")
    if optimization_population == "eligible_target":
        if not bool(mask.any()):
            raise ValueError("Round 74 risk coverage has no eligible target")
        return float(selected[mask].mean())
    if optimization_population != "capture_run":
        raise ValueError("Round 74 risk coverage population differs")
    group_coverage: list[float] = []
    for expected_run_id in expected_run_ids:
        for expected_symbol in ROUND74_EVENT_SYMBOLS:
            group_mask = np.asarray(
                [
                    run_id == expected_run_id
                    and symbol == expected_symbol
                    and bool(eligible)
                    for run_id, symbol, eligible in zip(
                        row_run_ids,
                        row_symbols,
                        mask,
                        strict=True,
                    )
                ],
                dtype=np.bool_,
            )
            if not bool(group_mask.any()):
                raise ValueError("Round 74 risk coverage run-symbol support differs")
            group_coverage.append(float(selected[group_mask].mean()))
    return float(np.min(np.asarray(group_coverage, dtype=np.float64)))


def fit_round74_risk_quantile_calibration(
    *,
    payoff_quantiles_bps: torch.Tensor,
    net_payoff_bps: torch.Tensor,
    maximum_adverse_excursion_quantiles_bps: torch.Tensor,
    maximum_adverse_excursion_bps: torch.Tensor,
    action_eligibility: torch.Tensor,
    row_run_ids: tuple[str, ...],
    row_symbols: tuple[str, ...],
    expected_run_ids: tuple[str, ...],
    optimization_population: str,
) -> Round74RiskQuantileCalibration:
    """Fit nonnegative one-sided tail widening on calibration runs only."""

    selected_population = str(optimization_population)
    expected = tuple(str(value) for value in expected_run_ids)
    selected_runs = tuple(str(value) for value in row_run_ids)
    selected_symbols = tuple(str(value) for value in row_symbols)
    horizons, sides = _risk_panel_shape()
    quantile_shape = (
        len(selected_runs),
        horizons,
        sides,
        len(ROUND74_EVENT_PAYOFF_QUANTILES),
    )
    target_shape = (len(selected_runs), horizons, sides)
    tensors = (
        payoff_quantiles_bps,
        net_payoff_bps,
        maximum_adverse_excursion_quantiles_bps,
        maximum_adverse_excursion_bps,
        action_eligibility,
    )
    if (
        selected_population not in ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS
        or expected != tuple(dict.fromkeys(expected))
        or not _valid_population_run_count(
            len(expected),
            optimization_population=selected_population,
            capture_run_count=ROUND74_TUNING_CALIBRATION_RUNS,
        )
        or set(selected_runs) != set(expected)
        or len(selected_symbols) != len(selected_runs)
        or set(selected_symbols) != set(ROUND74_EVENT_SYMBOLS)
        or any(value not in ROUND74_EVENT_SYMBOLS for value in selected_symbols)
        or tuple(ROUND74_EVENT_PAYOFF_QUANTILES[:2])
        != ROUND74_PAYOFF_LOWER_CALIBRATION_QUANTILES
        or float(ROUND74_EVENT_PAYOFF_QUANTILES[-1])
        != ROUND74_MAE_UPPER_CALIBRATION_QUANTILE
        or payoff_quantiles_bps.shape != quantile_shape
        or maximum_adverse_excursion_quantiles_bps.shape != quantile_shape
        or net_payoff_bps.shape != target_shape
        or maximum_adverse_excursion_bps.shape != target_shape
        or action_eligibility.shape != target_shape
        or any(not value.is_floating_point() for value in tensors)
        or len({value.device for value in tensors}) != 1
        or any(not bool(torch.isfinite(value).all()) for value in tensors)
        or bool(
            (payoff_quantiles_bps[..., 1:] - payoff_quantiles_bps[..., :-1] < 0.0).any()
        )
        or bool(
            (
                maximum_adverse_excursion_quantiles_bps[..., 1:]
                - maximum_adverse_excursion_quantiles_bps[..., :-1]
                < 0.0
            ).any()
        )
        or bool((maximum_adverse_excursion_quantiles_bps < 0.0).any())
        or bool(((action_eligibility != 0.0) & (action_eligibility != 1.0)).any())
        or bool((maximum_adverse_excursion_bps < 0.0).any())
    ):
        raise ValueError("Round 74 risk quantile calibration panel differs")
    payoff_forecast = (
        payoff_quantiles_bps.detach().to(device="cpu", dtype=torch.float64).numpy()
    )
    payoff_target = (
        net_payoff_bps.detach().to(device="cpu", dtype=torch.float64).numpy()
    )
    mae_forecast = (
        maximum_adverse_excursion_quantiles_bps.detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )
    mae_target = (
        maximum_adverse_excursion_bps.detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )
    eligible = (
        action_eligibility.detach().to(device="cpu", dtype=torch.float64).numpy() == 1.0
    )
    lower_offsets = np.zeros((horizons, sides, 2), dtype=np.float64)
    lower_before = np.zeros_like(lower_offsets)
    lower_after = np.zeros_like(lower_offsets)
    mae_offsets = np.zeros((horizons, sides), dtype=np.float64)
    mae_before = np.zeros_like(mae_offsets)
    mae_after = np.zeros_like(mae_offsets)
    observations = np.zeros((horizons, sides), dtype=np.int64)
    for horizon_index in range(horizons):
        for side_index in range(sides):
            mask = eligible[:, horizon_index, side_index]
            observations[horizon_index, side_index] = int(mask.sum())
            truth = payoff_target[:, horizon_index, side_index]
            for lower_index, nominal in enumerate(
                ROUND74_PAYOFF_LOWER_CALIBRATION_QUANTILES
            ):
                prediction = payoff_forecast[
                    :,
                    horizon_index,
                    side_index,
                    lower_index,
                ]
                offset = _risk_quantile_offset(
                    prediction - truth,
                    selected_runs,
                    selected_symbols,
                    expected_run_ids=expected,
                    eligibility=mask,
                    level=1.0 - nominal,
                    optimization_population=selected_population,
                )
                lower_offsets[horizon_index, side_index, lower_index] = offset
                lower_before[horizon_index, side_index, lower_index] = (
                    _risk_empirical_coverage(
                        truth >= prediction,
                        selected_runs,
                        selected_symbols,
                        expected_run_ids=expected,
                        eligibility=mask,
                        optimization_population=selected_population,
                    )
                )
                lower_after[horizon_index, side_index, lower_index] = (
                    _risk_empirical_coverage(
                        truth >= prediction - offset,
                        selected_runs,
                        selected_symbols,
                        expected_run_ids=expected,
                        eligibility=mask,
                        optimization_population=selected_population,
                    )
                )
            mae_prediction = mae_forecast[:, horizon_index, side_index, -1]
            mae_truth = mae_target[:, horizon_index, side_index]
            mae_offset = _risk_quantile_offset(
                mae_truth - mae_prediction,
                selected_runs,
                selected_symbols,
                expected_run_ids=expected,
                eligibility=mask,
                level=ROUND74_MAE_UPPER_CALIBRATION_QUANTILE,
                optimization_population=selected_population,
            )
            mae_offsets[horizon_index, side_index] = mae_offset
            mae_before[horizon_index, side_index] = _risk_empirical_coverage(
                mae_truth <= mae_prediction,
                selected_runs,
                selected_symbols,
                expected_run_ids=expected,
                eligibility=mask,
                optimization_population=selected_population,
            )
            mae_after[horizon_index, side_index] = _risk_empirical_coverage(
                mae_truth <= mae_prediction + mae_offset,
                selected_runs,
                selected_symbols,
                expected_run_ids=expected,
                eligibility=mask,
                optimization_population=selected_population,
            )
    result = Round74RiskQuantileCalibration(
        payoff_lower_offsets_bps=tuple(
            tuple(
                tuple(float(value) for value in lower_offsets[horizon, side])
                for side in range(sides)
            )
            for horizon in range(horizons)
        ),
        mae_upper_offsets_bps=tuple(
            tuple(float(value) for value in mae_offsets[horizon])
            for horizon in range(horizons)
        ),
        eligible_observations=tuple(
            tuple(int(value) for value in observations[horizon])
            for horizon in range(horizons)
        ),
        payoff_lower_empirical_coverage_before=tuple(
            tuple(
                tuple(float(value) for value in lower_before[horizon, side])
                for side in range(sides)
            )
            for horizon in range(horizons)
        ),
        payoff_lower_empirical_coverage_after=tuple(
            tuple(
                tuple(float(value) for value in lower_after[horizon, side])
                for side in range(sides)
            )
            for horizon in range(horizons)
        ),
        mae_upper_empirical_coverage_before=tuple(
            tuple(float(value) for value in mae_before[horizon])
            for horizon in range(horizons)
        ),
        mae_upper_empirical_coverage_after=tuple(
            tuple(float(value) for value in mae_after[horizon])
            for horizon in range(horizons)
        ),
        calibration_runs=len(expected),
        optimization_population=selected_population,
    )
    result.validate()
    return result


def apply_round74_risk_quantile_calibration(
    calibration: Round74RiskQuantileCalibration,
    *,
    payoff_quantiles_bps: torch.Tensor,
    maximum_adverse_excursion_quantiles_bps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply frozen nonnegative tail widening without target access."""

    calibration.validate()
    payoff = payoff_quantiles_bps.clone()
    mae = maximum_adverse_excursion_quantiles_bps.clone()
    horizons, sides = _risk_panel_shape()
    expected = (
        payoff.shape[0],
        horizons,
        sides,
        len(ROUND74_EVENT_PAYOFF_QUANTILES),
    )
    if (
        payoff.shape != expected
        or mae.shape != expected
        or not payoff.is_floating_point()
        or not mae.is_floating_point()
        or payoff.device != mae.device
        or not bool(torch.isfinite(payoff).all())
        or not bool(torch.isfinite(mae).all())
    ):
        raise ValueError("Round 74 risk quantile inference input differs")
    lower_offsets = torch.as_tensor(
        calibration.payoff_lower_offsets_bps,
        dtype=payoff.dtype,
        device=payoff.device,
    )
    mae_offsets = torch.as_tensor(
        calibration.mae_upper_offsets_bps,
        dtype=mae.dtype,
        device=mae.device,
    )
    adjusted_q25 = payoff[..., 1] - lower_offsets[..., 1]
    payoff[..., 1] = adjusted_q25
    payoff[..., 0] = torch.minimum(
        payoff[..., 0] - lower_offsets[..., 0],
        adjusted_q25,
    )
    mae[..., -1] = mae[..., -1] + mae_offsets
    if (
        not bool(torch.isfinite(payoff).all())
        or not bool(torch.isfinite(mae).all())
        or bool((payoff[..., 1:] - payoff[..., :-1] < 0.0).any())
        or bool((mae[..., 1:] - mae[..., :-1] < 0.0).any())
        or bool((mae < 0.0).any())
    ):
        raise RuntimeError("Round 74 calibrated risk quantiles differ")
    return payoff, mae


def _fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    run_panels: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    *,
    optimization_population: str,
) -> Round74TemperatureFit:
    if optimization_population not in ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS:
        raise ValueError("Round 74 calibration optimization population differs")
    log_temperatures = torch.linspace(
        math.log(ROUND74_TEMPERATURE_MINIMUM),
        math.log(ROUND74_TEMPERATURE_MAXIMUM),
        ROUND74_TEMPERATURE_CANDIDATE_COUNT,
        dtype=logits.dtype,
        device=logits.device,
    )
    temperatures = torch.exp(log_temperatures)
    losses: list[torch.Tensor] = []
    for chunk in temperatures.split(16):
        if optimization_population == "capture_run":
            run_losses: list[torch.Tensor] = []
            for run_logits, run_labels in run_panels:
                scaled = run_logits.unsqueeze(0) / chunk.unsqueeze(1)
                target = run_labels.unsqueeze(0).expand_as(scaled)
                run_losses.append((F.softplus(scaled) - target * scaled).mean(dim=1))
            losses.append(torch.stack(run_losses, dim=1).mean(dim=1))
        else:
            scaled = logits.unsqueeze(0) / chunk.unsqueeze(1)
            target = labels.unsqueeze(0).expand_as(scaled)
            losses.append((F.softplus(scaled) - target * scaled).mean(dim=1))
    candidate_loss = torch.cat(losses)
    selected_index = int(torch.argmin(candidate_loss).item())
    temperature = min(
        ROUND74_TEMPERATURE_MAXIMUM,
        max(
            ROUND74_TEMPERATURE_MINIMUM,
            float(temperatures[selected_index].item()),
        ),
    )
    uncalibrated_probability = torch.sigmoid(logits)
    calibrated_probability = torch.sigmoid(logits / temperature)
    uncalibrated_run_balanced_nll = float(
        torch.stack(
            tuple(
                (F.softplus(run_logits) - run_labels * run_logits).mean()
                for run_logits, run_labels in run_panels
            )
        )
        .mean()
        .item()
    )
    run_counts = tuple(int(run_labels.numel()) for _, run_labels in run_panels)
    uncalibrated_nll = float((F.softplus(logits) - labels * logits).mean().item())
    scaled_logits = logits / temperature
    calibrated_nll = float(
        (F.softplus(scaled_logits) - labels * scaled_logits).mean().item()
    )
    calibrated_run_balanced_nll = float(
        torch.stack(
            tuple(
                (
                    F.softplus(run_logits / temperature)
                    - run_labels * (run_logits / temperature)
                ).mean()
                for run_logits, run_labels in run_panels
            )
        )
        .mean()
        .item()
    )
    fit = Round74TemperatureFit(
        temperature=temperature,
        eligible_observations=int(labels.numel()),
        positive_observations=int((labels == 1.0).sum().item()),
        calibration_runs=len(run_panels),
        minimum_run_observations=min(run_counts),
        maximum_run_observations=max(run_counts),
        uncalibrated_run_balanced_nll=uncalibrated_run_balanced_nll,
        calibrated_run_balanced_nll=calibrated_run_balanced_nll,
        uncalibrated_nll=uncalibrated_nll,
        calibrated_nll=calibrated_nll,
        uncalibrated_brier=float(
            torch.mean((uncalibrated_probability - labels) ** 2).item()
        ),
        calibrated_brier=float(
            torch.mean((calibrated_probability - labels) ** 2).item()
        ),
        uncalibrated_ece=_expected_calibration_error(
            uncalibrated_probability,
            labels,
        ),
        calibrated_ece=_expected_calibration_error(
            calibrated_probability,
            labels,
        ),
    )
    fit.validate()
    return fit


def _update_tensor_digest(
    digest: object,
    value: torch.Tensor,
) -> None:
    array = np.ascontiguousarray(
        value.detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(int(array.ndim).to_bytes(2, "little", signed=False))
    for size in array.shape:
        digest.update(int(size).to_bytes(8, "little", signed=False))
    digest.update(memoryview(array).cast("B"))


def fit_round74_probability_calibration(
    *,
    positive_payoff_logits: torch.Tensor,
    positive_payoff_labels: torch.Tensor,
    adverse_selection_logits: torch.Tensor,
    adverse_selection_labels: torch.Tensor,
    action_eligibility: torch.Tensor,
    regime_unpredictability_logits: torch.Tensor,
    regime_unpredictability_labels: torch.Tensor,
    regime_eligibility: torch.Tensor,
    payoff_quantiles_bps: torch.Tensor | None = None,
    net_payoff_bps: torch.Tensor | None = None,
    maximum_adverse_excursion_quantiles_bps: torch.Tensor | None = None,
    maximum_adverse_excursion_bps: torch.Tensor | None = None,
    row_symbols: tuple[str, ...] | None = None,
    row_run_ids: tuple[str, ...],
    tuning_subpartition: Round74TuningSubpartition,
    pretest_policy_sha256: str,
    calibration_source_sha256: str,
    backend_kind: str,
    backend_device: str,
    optimization_population: str = "capture_run",
) -> Round74ProbabilityCalibration:
    """Fit probability temperatures and optional risk tails on calibration runs."""

    _require_sha256(pretest_policy_sha256, "pretest policy")
    tuning_subpartition.validate()
    _require_sha256(calibration_source_sha256, "calibration source")
    selected_population = str(optimization_population)
    if selected_population not in ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS:
        raise ValueError("Round 74 calibration optimization population differs")
    selected_row_run_ids = tuple(str(value) for value in row_run_ids)
    expected_runs = tuning_subpartition.calibration_run_ids
    risk_values = (
        payoff_quantiles_bps,
        net_payoff_bps,
        maximum_adverse_excursion_quantiles_bps,
        maximum_adverse_excursion_bps,
    )
    risk_present = any(value is not None for value in risk_values) or (
        row_symbols is not None
    )
    risk_requested = all(value is not None for value in risk_values) and (
        row_symbols is not None
    )
    if (
        risk_present != risk_requested
        or positive_payoff_logits.ndim < 1
        or adverse_selection_logits.ndim < 1
        or regime_unpredictability_logits.ndim < 1
    ):
        raise ValueError("Round 74 calibration run panel differs")
    if (
        positive_payoff_logits.shape[-1] != len(ROUND74_EVENT_PAYOFF_SIDES)
        or positive_payoff_labels.shape != positive_payoff_logits.shape
        or action_eligibility.shape != positive_payoff_logits.shape
        or bool((positive_payoff_logits.sum(dim=-1) > 1e-6).any())
        or bool(
            (
                (action_eligibility[..., 0] == 1.0)
                & (action_eligibility[..., 1] == 1.0)
                & (positive_payoff_labels[..., 0] == 1.0)
                & (positive_payoff_labels[..., 1] == 1.0)
            ).any()
        )
    ):
        raise ValueError("Round 74 positive-payoff outcome simplex differs")
    rows = int(positive_payoff_logits.shape[0])
    if (
        len(selected_row_run_ids) != rows
        or not _valid_population_run_count(
            len(expected_runs),
            optimization_population=selected_population,
            capture_run_count=ROUND74_TUNING_CALIBRATION_RUNS,
        )
        or len(set(expected_runs)) != len(expected_runs)
        or set(selected_row_run_ids) != set(expected_runs)
        or any(
            len(value) != 32
            or any(character not in "0123456789abcdef" for character in value)
            for value in (*selected_row_run_ids, *expected_runs)
        )
        or adverse_selection_logits.shape[0] != rows
        or regime_unpredictability_logits.shape[0] != rows
    ):
        raise ValueError("Round 74 calibration run panel differs")
    positive_logits, positive_labels, positive_run_panels = _validate_binary_panel(
        positive_payoff_logits,
        positive_payoff_labels,
        action_eligibility,
        row_run_ids=selected_row_run_ids,
        expected_run_ids=expected_runs,
        label="positive-payoff",
    )
    adverse_logits, adverse_labels, adverse_run_panels = _validate_binary_panel(
        adverse_selection_logits,
        adverse_selection_labels,
        action_eligibility,
        row_run_ids=selected_row_run_ids,
        expected_run_ids=expected_runs,
        label="adverse-selection",
    )
    regime_logits, regime_labels, regime_run_panels = _validate_binary_panel(
        regime_unpredictability_logits,
        regime_unpredictability_labels,
        regime_eligibility,
        row_run_ids=selected_row_run_ids,
        expected_run_ids=expected_runs,
        label="regime-unpredictability",
    )
    devices = {
        positive_logits.device,
        adverse_logits.device,
        regime_logits.device,
    }
    if len(devices) != 1:
        raise ValueError("Round 74 calibration devices differ")
    schema_version = (
        ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION
        if risk_requested
        else ROUND74_TEMPERATURE_CALIBRATION_PRIOR_SCHEMA_VERSION
    )
    identity = {
        "schema_version": schema_version,
        "pretest_policy_sha256": pretest_policy_sha256,
        "tuning_subpartition_sha256": tuning_subpartition.subpartition_sha256,
        "calibration_source_sha256": calibration_source_sha256,
        "backend_kind": str(backend_kind),
        "backend_device": str(backend_device),
        "optimization_population": selected_population,
        "calibration_run_ids": list(expected_runs),
        "calibration_row_run_ids": list(selected_row_run_ids),
    }
    if risk_requested:
        assert row_symbols is not None
        identity["calibration_row_symbols"] = list(row_symbols)
    digest = hashlib.sha256(_canonical_json(identity).encode("ascii"))
    for value in (
        positive_payoff_logits,
        positive_payoff_labels,
        adverse_selection_logits,
        adverse_selection_labels,
        action_eligibility,
        regime_unpredictability_logits,
        regime_unpredictability_labels,
        regime_eligibility,
        *(value for value in risk_values if value is not None),
    ):
        _update_tensor_digest(digest, value)
    risk_quantiles = None
    quantile_baseline = None
    if risk_requested:
        assert payoff_quantiles_bps is not None
        assert net_payoff_bps is not None
        assert maximum_adverse_excursion_quantiles_bps is not None
        assert maximum_adverse_excursion_bps is not None
        assert row_symbols is not None
        risk_quantiles = fit_round74_risk_quantile_calibration(
            payoff_quantiles_bps=payoff_quantiles_bps,
            net_payoff_bps=net_payoff_bps,
            maximum_adverse_excursion_quantiles_bps=(
                maximum_adverse_excursion_quantiles_bps
            ),
            maximum_adverse_excursion_bps=maximum_adverse_excursion_bps,
            action_eligibility=action_eligibility,
            row_run_ids=selected_row_run_ids,
            row_symbols=tuple(row_symbols),
            expected_run_ids=expected_runs,
            optimization_population=selected_population,
        )
        quantile_baseline = fit_round74_no_information_quantile_baseline(
            net_payoff_bps=net_payoff_bps,
            maximum_adverse_excursion_bps=maximum_adverse_excursion_bps,
            action_eligibility=action_eligibility,
            row_run_ids=selected_row_run_ids,
            row_symbols=tuple(row_symbols),
            expected_run_ids=expected_runs,
            optimization_population=selected_population,
        )
    result = Round74ProbabilityCalibration(
        pretest_policy_sha256=pretest_policy_sha256,
        tuning_subpartition_sha256=tuning_subpartition.subpartition_sha256,
        calibration_source_sha256=calibration_source_sha256,
        calibration_data_sha256=digest.hexdigest(),
        calibration_run_ids=expected_runs,
        calibration_row_run_ids_sha256=_canonical_sha256(list(selected_row_run_ids)),
        positive_payoff=_fit_temperature(
            positive_logits,
            positive_labels,
            positive_run_panels,
            optimization_population=selected_population,
        ),
        adverse_selection=_fit_temperature(
            adverse_logits,
            adverse_labels,
            adverse_run_panels,
            optimization_population=selected_population,
        ),
        regime_unpredictability=_fit_temperature(
            regime_logits,
            regime_labels,
            regime_run_panels,
            optimization_population=selected_population,
        ),
        backend_kind=str(backend_kind),
        backend_device=str(backend_device),
        risk_quantiles=risk_quantiles,
        quantile_baseline=quantile_baseline,
        optimization_population=selected_population,
        schema_version=schema_version,
    )
    result.validate()
    return result


def apply_round74_probability_calibration(
    calibration: Round74ProbabilityCalibration,
    *,
    positive_payoff_logits: torch.Tensor,
    adverse_selection_logits: torch.Tensor,
    regime_unpredictability_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply frozen temperatures without fitting or target access."""

    calibration.validate()
    tensors = (
        positive_payoff_logits,
        adverse_selection_logits,
        regime_unpredictability_logits,
    )
    if (
        any(not value.is_floating_point() for value in tensors)
        or any(not bool(torch.isfinite(value).all()) for value in tensors)
        or len({value.device for value in tensors}) != 1
        or positive_payoff_logits.ndim < 1
        or positive_payoff_logits.shape[-1] != len(ROUND74_EVENT_PAYOFF_SIDES)
        or bool((positive_payoff_logits.sum(dim=-1) > 1e-6).any())
    ):
        raise ValueError("Round 74 calibration inference input differs")
    return (
        torch.sigmoid(positive_payoff_logits / calibration.positive_payoff.temperature),
        torch.sigmoid(
            adverse_selection_logits / calibration.adverse_selection.temperature
        ),
        torch.sigmoid(
            regime_unpredictability_logits
            / calibration.regime_unpredictability.temperature
        ),
    )


__all__ = [
    "ROUND74_CALIBRATION_OPTIMIZATION_POPULATIONS",
    "ROUND74_MAE_UPPER_CALIBRATION_QUANTILE",
    "ROUND74_NO_INFORMATION_QUANTILE_BASELINE_SCHEMA_VERSION",
    "ROUND74_PAYOFF_LOWER_CALIBRATION_QUANTILES",
    "ROUND74_RISK_QUANTILE_CALIBRATION_SCHEMA_VERSION",
    "ROUND74_RISK_QUANTILE_CALIBRATION_PRIOR_SCHEMA_VERSION",
    "ROUND74_TEMPERATURE_CALIBRATION_LEGACY_SCHEMA_VERSION",
    "ROUND74_TEMPERATURE_CALIBRATION_PRIOR_SCHEMA_VERSION",
    "ROUND74_TEMPERATURE_CALIBRATION_RISK_PRIOR_SCHEMA_VERSION",
    "ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION",
    "ROUND74_TEMPERATURE_CANDIDATE_COUNT",
    "ROUND74_TEMPERATURE_MAXIMUM",
    "ROUND74_TEMPERATURE_MINIMUM",
    "ROUND74_TUNING_CALIBRATION_RUNS",
    "ROUND74_TUNING_EXPECTED_RUNS",
    "ROUND74_TUNING_MODEL_SELECTION_RUNS",
    "ROUND74_TUNING_POLICY_SELECTION_RUNS",
    "ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION",
    "Round74NoInformationQuantileBaseline",
    "Round74ProbabilityCalibration",
    "Round74RiskQuantileCalibration",
    "Round74TemperatureFit",
    "Round74TuningSubpartition",
    "apply_round74_probability_calibration",
    "apply_round74_risk_quantile_calibration",
    "build_round74_tuning_subpartition",
    "fit_round74_no_information_quantile_baseline",
    "fit_round74_probability_calibration",
    "fit_round74_risk_quantile_calibration",
]
