"""Development-only selective-risk evaluation for Round 74 ensemble telemetry.

The evaluator consumes only disjoint tuning policy-selection batches. It
measures whether ensemble disagreement orders calibrated prediction errors;
it cannot change candidates, sizing, leverage, or execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import numpy as np
import torch

from .impact_absorption_event_calibration import (
    Round74ProbabilityCalibration,
    apply_round74_probability_calibration,
    apply_round74_risk_quantile_calibration,
)
from .impact_absorption_event_dataset import Round74EventTrainingBatch
from .impact_absorption_event_model import Round74EventModelOutput
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SYMBOLS,
)
from .statistical_resampling import moving_block_bootstrap_mean


ROUND74_EPISTEMIC_EVALUATION_BATCH_SCHEMA_VERSION = (
    "round-074-epistemic-evaluation-batch-v1"
)
ROUND74_EPISTEMIC_RISK_COVERAGE_SCHEMA_VERSION = "round-074-epistemic-risk-coverage-v1"
ROUND74_EPISTEMIC_RISK_COVERAGE_METRIC_IDS = (
    "payoff_quantile_peer_dispersion",
    "adverse_excursion_quantile_peer_dispersion",
    "positive_payoff_probability_peer_dispersion",
    "adverse_selection_probability_peer_dispersion",
    "regime_unpredictability_probability_peer_dispersion",
)
ROUND74_EPISTEMIC_CURVE_TARGET_COVERAGES = tuple(index / 20.0 for index in range(1, 21))
ROUND74_EPISTEMIC_FIXED_RISK_RATIOS = (0.50, 0.75, 0.90, 1.00)
ROUND74_EPISTEMIC_MINIMUM_STRATUM_ROWS = 256
ROUND74_EPISTEMIC_REQUIRED_POLICY_SELECTION_RUNS = 6
ROUND74_EPISTEMIC_BOOTSTRAP_SAMPLES = 10_000
ROUND74_EPISTEMIC_FAMILYWISE_CONFIDENCE = 0.95
ROUND74_EPISTEMIC_PER_METRIC_CONFIDENCE = 1.0 - (
    (1.0 - ROUND74_EPISTEMIC_FAMILYWISE_CONFIDENCE)
    / len(ROUND74_EPISTEMIC_RISK_COVERAGE_METRIC_IDS)
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_EPSILON = np.finfo(np.float64).eps


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


def _readonly(value: np.ndarray) -> np.ndarray:
    selected = np.ascontiguousarray(value)
    selected.setflags(write=False)
    return selected


def _json_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Round 74 epistemic {label} number differs")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"Round 74 epistemic {label} number differs")
    return selected


def _json_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Round 74 epistemic {label} integer differs")
    return value


def _json_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Round 74 epistemic {label} flag differs")
    return value


def _tensor_array(value: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(
        value.detach().to(device="cpu", dtype=torch.float64).numpy()
    )


def _update_array_digest(digest: object, value: np.ndarray) -> None:
    array = np.asarray(value)
    canonical = np.ascontiguousarray(
        array.astype(array.dtype.newbyteorder("<"), copy=False)
    )
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(int(canonical.ndim).to_bytes(2, "little", signed=False))
    for size in canonical.shape:
        digest.update(int(size).to_bytes(8, "little", signed=False))
    digest.update(memoryview(canonical).cast("B"))


def _model_output_sha256(output: Round74EventModelOutput) -> str:
    output.validate(int(output.payoff_quantiles_bps.shape[0]))
    digest = hashlib.sha256(b"round-074-epistemic-model-output-v1")
    for value in (
        output.payoff_quantiles_bps,
        output.maximum_adverse_excursion_quantiles_bps,
        output.positive_payoff_logits,
        output.adverse_selection_logits,
        output.regime_unpredictability_logits,
    ):
        _update_array_digest(digest, _tensor_array(value))
    diagnostics = output.epistemic_diagnostics
    if diagnostics is None:
        digest.update(b"\x00")
    else:
        digest.update(b"\x01")
        digest.update(int(diagnostics.peer_count).to_bytes(4, "little", signed=False))
        for value in (
            diagnostics.payoff_quantile_standard_deviation_bps,
            diagnostics.maximum_adverse_excursion_quantile_standard_deviation_bps,
            diagnostics.positive_payoff_probability_standard_deviation,
            diagnostics.adverse_selection_probability_standard_deviation,
            diagnostics.regime_unpredictability_probability_standard_deviation,
        ):
            _update_array_digest(digest, _tensor_array(value))
    return digest.hexdigest()


@dataclass(frozen=True)
class Round74EpistemicEvaluationBatch:
    """Calibrated predictions, immutable targets, and ensemble disagreement."""

    batch_sha256: str
    model_output_sha256: str
    probability_calibration_sha256: str
    tuning_subpartition_sha256: str
    run_id: tuple[str, ...]
    symbol: tuple[str, ...]
    net_payoff_bps: np.ndarray
    maximum_adverse_excursion_bps: np.ndarray
    adverse_selection: np.ndarray
    regime_unpredictability: np.ndarray
    action_eligibility: np.ndarray
    regime_unpredictability_eligibility: np.ndarray
    payoff_quantiles_bps: np.ndarray
    maximum_adverse_excursion_quantiles_bps: np.ndarray
    positive_payoff_probability: np.ndarray
    adverse_selection_probability: np.ndarray
    regime_unpredictability_probability: np.ndarray
    payoff_quantile_peer_dispersion_bps: np.ndarray
    adverse_excursion_quantile_peer_dispersion_bps: np.ndarray
    positive_payoff_probability_peer_dispersion: np.ndarray
    adverse_selection_probability_peer_dispersion: np.ndarray
    regime_unpredictability_probability_peer_dispersion: np.ndarray
    peer_count: int
    schema_version: str = ROUND74_EPISTEMIC_EVALUATION_BATCH_SCHEMA_VERSION

    @property
    def rows(self) -> int:
        return len(self.run_id)

    def validate(self) -> None:
        action_shape = (
            self.rows,
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            len(ROUND74_EVENT_PAYOFF_SIDES),
        )
        quantile_shape = (*action_shape, len(ROUND74_EVENT_PAYOFF_QUANTILES))
        regime_shape = action_shape[:2]
        action_arrays = (
            self.net_payoff_bps,
            self.maximum_adverse_excursion_bps,
            self.adverse_selection,
            self.action_eligibility,
            self.positive_payoff_probability,
            self.adverse_selection_probability,
            self.payoff_quantile_peer_dispersion_bps,
            self.adverse_excursion_quantile_peer_dispersion_bps,
            self.positive_payoff_probability_peer_dispersion,
            self.adverse_selection_probability_peer_dispersion,
        )
        quantile_arrays = (
            self.payoff_quantiles_bps,
            self.maximum_adverse_excursion_quantiles_bps,
        )
        regime_arrays = (
            self.regime_unpredictability,
            self.regime_unpredictability_eligibility,
            self.regime_unpredictability_probability,
            self.regime_unpredictability_probability_peer_dispersion,
        )
        arrays = (*action_arrays, *quantile_arrays, *regime_arrays)
        if (
            self.schema_version != ROUND74_EPISTEMIC_EVALUATION_BATCH_SCHEMA_VERSION
            or self.rows < 1
            or len(self.symbol) != self.rows
            or any(_RUN_ID.fullmatch(value) is None for value in self.run_id)
            or any(value not in ROUND74_EVENT_SYMBOLS for value in self.symbol)
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.batch_sha256,
                    self.model_output_sha256,
                    self.probability_calibration_sha256,
                    self.tuning_subpartition_sha256,
                )
            )
            or isinstance(self.peer_count, bool)
            or self.peer_count < 2
            or any(value.shape != action_shape for value in action_arrays)
            or any(value.shape != quantile_shape for value in quantile_arrays)
            or any(value.shape != regime_shape for value in regime_arrays)
            or any(value.dtype != np.float64 for value in arrays)
            or any(value.flags.writeable for value in arrays)
            or not all(np.isfinite(value).all() for value in arrays)
            or np.any(
                (self.action_eligibility != 0.0) & (self.action_eligibility != 1.0)
            )
            or np.any(
                (self.regime_unpredictability_eligibility != 0.0)
                & (self.regime_unpredictability_eligibility != 1.0)
            )
            or any(
                np.any(value < 0.0)
                for value in (
                    self.payoff_quantile_peer_dispersion_bps,
                    self.adverse_excursion_quantile_peer_dispersion_bps,
                    self.positive_payoff_probability_peer_dispersion,
                    self.adverse_selection_probability_peer_dispersion,
                    self.regime_unpredictability_probability_peer_dispersion,
                )
            )
            or any(
                np.any((value < 0.0) | (value > 1.0))
                for value in (
                    self.positive_payoff_probability,
                    self.adverse_selection_probability,
                    self.regime_unpredictability_probability,
                    self.adverse_selection,
                    self.regime_unpredictability,
                )
            )
            or any(
                np.any(value > 0.5 + 1e-6)
                for value in (
                    self.positive_payoff_probability_peer_dispersion,
                    self.adverse_selection_probability_peer_dispersion,
                    self.regime_unpredictability_probability_peer_dispersion,
                )
            )
        ):
            raise ValueError("Round 74 epistemic evaluation batch differs")


def prepare_round74_epistemic_evaluation_batch(
    batch: Round74EventTrainingBatch,
    output: Round74EventModelOutput,
    calibration: Round74ProbabilityCalibration,
) -> Round74EpistemicEvaluationBatch:
    """Apply frozen calibration and detach one policy-selection batch."""

    batch.validate()
    output.validate(batch.rows)
    calibration.validate()
    diagnostics = output.epistemic_diagnostics
    if (
        batch.role != "tuning"
        or diagnostics is None
        or calibration.risk_quantiles is None
    ):
        raise ValueError("Round 74 epistemic evaluation source differs")
    payoff, adverse_excursion = apply_round74_risk_quantile_calibration(
        calibration.risk_quantiles,
        payoff_quantiles_bps=output.payoff_quantiles_bps,
        maximum_adverse_excursion_quantiles_bps=(
            output.maximum_adverse_excursion_quantiles_bps
        ),
    )
    positive, adverse, unpredictable = apply_round74_probability_calibration(
        calibration,
        positive_payoff_logits=output.positive_payoff_logits,
        adverse_selection_logits=output.adverse_selection_logits,
        regime_unpredictability_logits=output.regime_unpredictability_logits,
    )

    def target(name: str) -> np.ndarray:
        return _readonly(np.asarray(getattr(batch, name), dtype=np.float64))

    def prediction(value: torch.Tensor) -> np.ndarray:
        return _readonly(_tensor_array(value))

    result = Round74EpistemicEvaluationBatch(
        batch_sha256=batch.batch_sha256,
        model_output_sha256=_model_output_sha256(output),
        probability_calibration_sha256=calibration.calibration_sha256,
        tuning_subpartition_sha256=calibration.tuning_subpartition_sha256,
        run_id=tuple(batch.run_id),
        symbol=tuple(batch.symbol),
        net_payoff_bps=target("net_payoff_bps"),
        maximum_adverse_excursion_bps=target("maximum_adverse_excursion_bps"),
        adverse_selection=target("adverse_selection"),
        regime_unpredictability=target("regime_unpredictability"),
        action_eligibility=target("action_eligibility"),
        regime_unpredictability_eligibility=target(
            "regime_unpredictability_eligibility"
        ),
        payoff_quantiles_bps=prediction(payoff),
        maximum_adverse_excursion_quantiles_bps=prediction(adverse_excursion),
        positive_payoff_probability=prediction(positive),
        adverse_selection_probability=prediction(adverse),
        regime_unpredictability_probability=prediction(unpredictable),
        payoff_quantile_peer_dispersion_bps=prediction(
            torch.sqrt(
                torch.mean(
                    torch.square(diagnostics.payoff_quantile_standard_deviation_bps),
                    dim=-1,
                )
            )
        ),
        adverse_excursion_quantile_peer_dispersion_bps=prediction(
            torch.sqrt(
                torch.mean(
                    torch.square(
                        diagnostics.maximum_adverse_excursion_quantile_standard_deviation_bps
                    ),
                    dim=-1,
                )
            )
        ),
        positive_payoff_probability_peer_dispersion=prediction(
            diagnostics.positive_payoff_probability_standard_deviation
        ),
        adverse_selection_probability_peer_dispersion=prediction(
            diagnostics.adverse_selection_probability_standard_deviation
        ),
        regime_unpredictability_probability_peer_dispersion=prediction(
            diagnostics.regime_unpredictability_probability_standard_deviation
        ),
        peer_count=diagnostics.peer_count,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class Round74RiskCoveragePoint:
    target_coverage: float
    attained_coverage: float
    accepted_rows: int
    uncertainty_threshold: float
    selective_loss: float
    generalized_loss: float

    def validate(self, population_rows: int) -> None:
        values = (
            self.target_coverage,
            self.attained_coverage,
            self.uncertainty_threshold,
            self.selective_loss,
            self.generalized_loss,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or not 0.0 < self.target_coverage <= 1.0
            or not self.target_coverage <= self.attained_coverage <= 1.0
            or isinstance(self.accepted_rows, bool)
            or not isinstance(self.accepted_rows, int)
            or not 1 <= self.accepted_rows <= population_rows
            or self.selective_loss < 0.0
            or self.generalized_loss < 0.0
        ):
            raise ValueError("Round 74 risk-coverage point differs")

    def as_dict(self) -> dict[str, object]:
        return {
            "target_coverage": self.target_coverage,
            "attained_coverage": self.attained_coverage,
            "accepted_rows": self.accepted_rows,
            "uncertainty_threshold": self.uncertainty_threshold,
            "selective_loss": self.selective_loss,
            "generalized_loss": self.generalized_loss,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Round74RiskCoveragePoint:
        if set(value) != {
            "target_coverage",
            "attained_coverage",
            "accepted_rows",
            "uncertainty_threshold",
            "selective_loss",
            "generalized_loss",
        }:
            raise ValueError("Round 74 risk-coverage point payload differs")
        return cls(
            target_coverage=_json_float(value["target_coverage"], "target coverage"),
            attained_coverage=_json_float(
                value["attained_coverage"], "attained coverage"
            ),
            accepted_rows=_json_int(value["accepted_rows"], "accepted rows"),
            uncertainty_threshold=_json_float(
                value["uncertainty_threshold"], "uncertainty threshold"
            ),
            selective_loss=_json_float(value["selective_loss"], "selective loss"),
            generalized_loss=_json_float(value["generalized_loss"], "generalized loss"),
        )


@dataclass(frozen=True)
class Round74FixedRiskCoverage:
    risk_ratio_to_full_coverage: float
    risk_limit: float
    maximum_attained_coverage: float
    accepted_rows: int
    selective_loss: float
    uncertainty_threshold: float | None

    def validate(self, population_rows: int, full_coverage_loss: float) -> None:
        optional = (
            () if self.uncertainty_threshold is None else (self.uncertainty_threshold,)
        )
        if (
            any(
                not math.isfinite(value)
                for value in (
                    self.risk_ratio_to_full_coverage,
                    self.risk_limit,
                    self.maximum_attained_coverage,
                    self.selective_loss,
                    *optional,
                )
            )
            or self.risk_ratio_to_full_coverage
            not in ROUND74_EPISTEMIC_FIXED_RISK_RATIOS
            or not math.isclose(
                self.risk_limit,
                self.risk_ratio_to_full_coverage * full_coverage_loss,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not 0.0 <= self.maximum_attained_coverage <= 1.0
            or isinstance(self.accepted_rows, bool)
            or not isinstance(self.accepted_rows, int)
            or not 0 <= self.accepted_rows <= population_rows
            or self.selective_loss < 0.0
            or (self.accepted_rows == 0) != (self.uncertainty_threshold is None)
            or not math.isclose(
                self.maximum_attained_coverage,
                self.accepted_rows / population_rows,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or (
                self.accepted_rows > 0 and self.selective_loss > self.risk_limit + 1e-12
            )
        ):
            raise ValueError("Round 74 fixed-risk coverage differs")

    def as_dict(self) -> dict[str, object]:
        return {
            "risk_ratio_to_full_coverage": self.risk_ratio_to_full_coverage,
            "risk_limit": self.risk_limit,
            "maximum_attained_coverage": self.maximum_attained_coverage,
            "accepted_rows": self.accepted_rows,
            "selective_loss": self.selective_loss,
            "uncertainty_threshold": self.uncertainty_threshold,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Round74FixedRiskCoverage:
        if set(value) != {
            "risk_ratio_to_full_coverage",
            "risk_limit",
            "maximum_attained_coverage",
            "accepted_rows",
            "selective_loss",
            "uncertainty_threshold",
        }:
            raise ValueError("Round 74 fixed-risk coverage payload differs")
        threshold = value["uncertainty_threshold"]
        return cls(
            risk_ratio_to_full_coverage=_json_float(
                value["risk_ratio_to_full_coverage"], "fixed-risk ratio"
            ),
            risk_limit=_json_float(value["risk_limit"], "fixed-risk limit"),
            maximum_attained_coverage=_json_float(
                value["maximum_attained_coverage"], "fixed-risk coverage"
            ),
            accepted_rows=_json_int(value["accepted_rows"], "fixed-risk accepted rows"),
            selective_loss=_json_float(
                value["selective_loss"], "fixed-risk selective loss"
            ),
            uncertainty_threshold=(
                None
                if threshold is None
                else _json_float(threshold, "fixed-risk uncertainty threshold")
            ),
        )


@dataclass(frozen=True)
class Round74EpistemicRiskCoverageMetric:
    metric_id: str
    loss_id: str
    scope: str
    symbol: str
    horizon_seconds: int
    side: str
    outcome_class: str
    population_rows: int
    capture_runs: int
    unique_uncertainty_thresholds: int
    full_coverage_loss: float
    area_under_risk_coverage: float
    area_under_generalized_risk_coverage: float
    aurc_improvement_over_no_information: float
    aurc_improvement_ratio: float
    uncertainty_loss_rank_correlation: float
    run_mean_aurc_improvement: float
    run_mean_aurc_improvement_ci_lower: float
    run_mean_aurc_improvement_ci_upper: float
    bootstrap_samples: int
    bootstrap_confidence: float
    bootstrap_block_length: int
    curve_sha256: str
    curve_points: tuple[Round74RiskCoveragePoint, ...]
    fixed_risk_coverage: tuple[Round74FixedRiskCoverage, ...]
    ordering_assessable: bool
    ordering_supported: bool

    def validate(self) -> None:
        finite = (
            self.full_coverage_loss,
            self.area_under_risk_coverage,
            self.area_under_generalized_risk_coverage,
            self.aurc_improvement_over_no_information,
            self.aurc_improvement_ratio,
            self.uncertainty_loss_rank_correlation,
            self.run_mean_aurc_improvement,
            self.run_mean_aurc_improvement_ci_lower,
            self.run_mean_aurc_improvement_ci_upper,
            self.bootstrap_confidence,
        )
        if (
            self.metric_id not in ROUND74_EPISTEMIC_RISK_COVERAGE_METRIC_IDS
            or not self.loss_id
            or self.scope not in {"aggregate", "conditional"}
            or self.symbol not in {*ROUND74_EVENT_SYMBOLS, "all"}
            or self.horizon_seconds
            not in {
                *ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
                0,
            }
            or self.side not in {*ROUND74_EVENT_PAYOFF_SIDES, "shared", "all"}
            or not self.outcome_class
            or isinstance(self.population_rows, bool)
            or self.population_rows < 1
            or isinstance(self.capture_runs, bool)
            or self.capture_runs < 1
            or isinstance(self.unique_uncertainty_thresholds, bool)
            or not 1 <= self.unique_uncertainty_thresholds <= self.population_rows
            or any(not math.isfinite(value) for value in finite)
            or any(
                value < 0.0
                for value in (
                    self.full_coverage_loss,
                    self.area_under_risk_coverage,
                    self.area_under_generalized_risk_coverage,
                )
            )
            or not -1.0 <= self.uncertainty_loss_rank_correlation <= 1.0
            or self.run_mean_aurc_improvement_ci_lower
            > self.run_mean_aurc_improvement_ci_upper
            or isinstance(self.bootstrap_samples, bool)
            or self.bootstrap_samples < 200
            or not 0.80 <= self.bootstrap_confidence <= 0.999
            or isinstance(self.bootstrap_block_length, bool)
            or not 1 <= self.bootstrap_block_length <= self.capture_runs
            or _SHA256.fullmatch(self.curve_sha256) is None
            or len(self.curve_points) != len(ROUND74_EPISTEMIC_CURVE_TARGET_COVERAGES)
            or tuple(point.target_coverage for point in self.curve_points)
            != ROUND74_EPISTEMIC_CURVE_TARGET_COVERAGES
            or len(self.fixed_risk_coverage) != len(ROUND74_EPISTEMIC_FIXED_RISK_RATIOS)
            or tuple(
                point.risk_ratio_to_full_coverage for point in self.fixed_risk_coverage
            )
            != ROUND74_EPISTEMIC_FIXED_RISK_RATIOS
            or not isinstance(self.ordering_assessable, bool)
            or not isinstance(self.ordering_supported, bool)
            or (self.ordering_supported and not self.ordering_assessable)
        ):
            raise ValueError("Round 74 epistemic risk-coverage metric differs")
        for point in self.curve_points:
            point.validate(self.population_rows)
        for point in self.fixed_risk_coverage:
            point.validate(self.population_rows, self.full_coverage_loss)
        expected_assessable = (
            self.population_rows >= ROUND74_EPISTEMIC_MINIMUM_STRATUM_ROWS
            and self.capture_runs >= ROUND74_EPISTEMIC_REQUIRED_POLICY_SELECTION_RUNS
            and self.unique_uncertainty_thresholds >= 2
            and self.full_coverage_loss > _EPSILON
        )
        expected_supported = expected_assessable and (
            self.run_mean_aurc_improvement_ci_lower > 0.0
            and self.uncertainty_loss_rank_correlation > 0.0
        )
        if (
            self.ordering_assessable != expected_assessable
            or self.ordering_supported != expected_supported
        ):
            raise ValueError("Round 74 epistemic ordering conclusion differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "metric_id": self.metric_id,
            "loss_id": self.loss_id,
            "scope": self.scope,
            "symbol": self.symbol,
            "horizon_seconds": self.horizon_seconds,
            "side": self.side,
            "outcome_class": self.outcome_class,
            "population_rows": self.population_rows,
            "capture_runs": self.capture_runs,
            "unique_uncertainty_thresholds": (self.unique_uncertainty_thresholds),
            "full_coverage_loss": self.full_coverage_loss,
            "area_under_risk_coverage": self.area_under_risk_coverage,
            "area_under_generalized_risk_coverage": (
                self.area_under_generalized_risk_coverage
            ),
            "aurc_improvement_over_no_information": (
                self.aurc_improvement_over_no_information
            ),
            "aurc_improvement_ratio": self.aurc_improvement_ratio,
            "uncertainty_loss_rank_correlation": (
                self.uncertainty_loss_rank_correlation
            ),
            "run_mean_aurc_improvement": self.run_mean_aurc_improvement,
            "run_mean_aurc_improvement_ci_lower": (
                self.run_mean_aurc_improvement_ci_lower
            ),
            "run_mean_aurc_improvement_ci_upper": (
                self.run_mean_aurc_improvement_ci_upper
            ),
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_confidence": self.bootstrap_confidence,
            "bootstrap_block_length": self.bootstrap_block_length,
            "curve_sha256": self.curve_sha256,
            "curve_points": [point.as_dict() for point in self.curve_points],
            "fixed_risk_coverage": [
                point.as_dict() for point in self.fixed_risk_coverage
            ],
            "ordering_assessable": self.ordering_assessable,
            "ordering_supported": self.ordering_supported,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74EpistemicRiskCoverageMetric:
        expected = {
            "metric_id",
            "loss_id",
            "scope",
            "symbol",
            "horizon_seconds",
            "side",
            "outcome_class",
            "population_rows",
            "capture_runs",
            "unique_uncertainty_thresholds",
            "full_coverage_loss",
            "area_under_risk_coverage",
            "area_under_generalized_risk_coverage",
            "aurc_improvement_over_no_information",
            "aurc_improvement_ratio",
            "uncertainty_loss_rank_correlation",
            "run_mean_aurc_improvement",
            "run_mean_aurc_improvement_ci_lower",
            "run_mean_aurc_improvement_ci_upper",
            "bootstrap_samples",
            "bootstrap_confidence",
            "bootstrap_block_length",
            "curve_sha256",
            "curve_points",
            "fixed_risk_coverage",
            "ordering_assessable",
            "ordering_supported",
        }
        curves = value.get("curve_points")
        fixed = value.get("fixed_risk_coverage")
        if (
            set(value) != expected
            or not isinstance(curves, list)
            or not all(isinstance(item, Mapping) for item in curves)
            or not isinstance(fixed, list)
            or not all(isinstance(item, Mapping) for item in fixed)
        ):
            raise ValueError("Round 74 epistemic metric payload differs")
        selected = cls(
            metric_id=str(value["metric_id"]),
            loss_id=str(value["loss_id"]),
            scope=str(value["scope"]),
            symbol=str(value["symbol"]),
            horizon_seconds=_json_int(value["horizon_seconds"], "metric horizon"),
            side=str(value["side"]),
            outcome_class=str(value["outcome_class"]),
            population_rows=_json_int(
                value["population_rows"], "metric population rows"
            ),
            capture_runs=_json_int(value["capture_runs"], "metric capture runs"),
            unique_uncertainty_thresholds=_json_int(
                value["unique_uncertainty_thresholds"],
                "metric uncertainty thresholds",
            ),
            full_coverage_loss=_json_float(
                value["full_coverage_loss"], "full-coverage loss"
            ),
            area_under_risk_coverage=_json_float(
                value["area_under_risk_coverage"], "AURC"
            ),
            area_under_generalized_risk_coverage=_json_float(
                value["area_under_generalized_risk_coverage"], "AUGRC"
            ),
            aurc_improvement_over_no_information=_json_float(
                value["aurc_improvement_over_no_information"],
                "AURC improvement",
            ),
            aurc_improvement_ratio=_json_float(
                value["aurc_improvement_ratio"], "AURC improvement ratio"
            ),
            uncertainty_loss_rank_correlation=_json_float(
                value["uncertainty_loss_rank_correlation"],
                "uncertainty-loss rank correlation",
            ),
            run_mean_aurc_improvement=_json_float(
                value["run_mean_aurc_improvement"],
                "run-mean AURC improvement",
            ),
            run_mean_aurc_improvement_ci_lower=_json_float(
                value["run_mean_aurc_improvement_ci_lower"],
                "run-mean AURC lower confidence bound",
            ),
            run_mean_aurc_improvement_ci_upper=_json_float(
                value["run_mean_aurc_improvement_ci_upper"],
                "run-mean AURC upper confidence bound",
            ),
            bootstrap_samples=_json_int(
                value["bootstrap_samples"], "bootstrap samples"
            ),
            bootstrap_confidence=_json_float(
                value["bootstrap_confidence"], "bootstrap confidence"
            ),
            bootstrap_block_length=_json_int(
                value["bootstrap_block_length"], "bootstrap block length"
            ),
            curve_sha256=str(value["curve_sha256"]),
            curve_points=tuple(
                Round74RiskCoveragePoint.from_dict(item) for item in curves
            ),
            fixed_risk_coverage=tuple(
                Round74FixedRiskCoverage.from_dict(item) for item in fixed
            ),
            ordering_assessable=_json_bool(
                value["ordering_assessable"], "ordering assessable"
            ),
            ordering_supported=_json_bool(
                value["ordering_supported"], "ordering supported"
            ),
        )
        selected.validate()
        return selected


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centered = left_ranks - float(left_ranks.mean())
    right_centered = right_ranks - float(right_ranks.mean())
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator <= _EPSILON:
        return 0.0
    return max(
        -1.0,
        min(1.0, float(np.dot(left_centered, right_centered)) / denominator),
    )


@dataclass(frozen=True)
class _TieSafeCurve:
    thresholds: np.ndarray
    accepted_rows: np.ndarray
    coverage: np.ndarray
    selective_loss: np.ndarray
    generalized_loss: np.ndarray
    full_coverage_loss: float
    aurc: float
    augrc: float
    rank_correlation: float
    curve_sha256: str


def _tie_safe_curve(
    uncertainty: np.ndarray,
    loss: np.ndarray,
) -> _TieSafeCurve:
    if (
        uncertainty.ndim != 1
        or loss.shape != uncertainty.shape
        or len(loss) < 1
        or not np.isfinite(uncertainty).all()
        or not np.isfinite(loss).all()
        or np.any(uncertainty < 0.0)
        or np.any(loss < 0.0)
    ):
        raise ValueError("Round 74 risk-coverage input differs")
    order = np.argsort(uncertainty, kind="stable")
    scores = uncertainty[order]
    losses = loss[order]
    group_end = np.flatnonzero(np.r_[scores[1:] != scores[:-1], np.array([True])])
    accepted = group_end.astype(np.int64) + 1
    cumulative = np.cumsum(losses, dtype=np.float64)[group_end]
    population = len(loss)
    coverage = accepted.astype(np.float64) / population
    selective = cumulative / accepted
    generalized = cumulative / population
    delta_coverage = np.diff(np.r_[0.0, coverage])
    full_loss = float(loss.mean(dtype=np.float64))
    digest = hashlib.sha256(b"round-074-tie-safe-risk-coverage-v1")
    for value in (
        scores[group_end],
        accepted,
        selective,
        generalized,
    ):
        _update_array_digest(digest, value)
    return _TieSafeCurve(
        thresholds=_readonly(scores[group_end]),
        accepted_rows=_readonly(accepted),
        coverage=_readonly(coverage),
        selective_loss=_readonly(selective),
        generalized_loss=_readonly(generalized),
        full_coverage_loss=full_loss,
        aurc=float(np.dot(selective, delta_coverage)),
        augrc=float(np.dot(generalized, delta_coverage)),
        rank_correlation=_rank_correlation(uncertainty, loss),
        curve_sha256=digest.hexdigest(),
    )


def _quantile_loss(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    quantiles = np.asarray(ROUND74_EVENT_PAYOFF_QUANTILES, dtype=np.float64)
    residual = np.expand_dims(target, axis=-1) - prediction
    return np.mean(
        np.maximum(quantiles * residual, (quantiles - 1.0) * residual),
        axis=-1,
    )


def _curve_points(curve: _TieSafeCurve) -> tuple[Round74RiskCoveragePoint, ...]:
    selected: list[Round74RiskCoveragePoint] = []
    for target in ROUND74_EPISTEMIC_CURVE_TARGET_COVERAGES:
        index = min(
            int(np.searchsorted(curve.coverage, target, side="left")),
            len(curve.coverage) - 1,
        )
        selected.append(
            Round74RiskCoveragePoint(
                target_coverage=target,
                attained_coverage=float(curve.coverage[index]),
                accepted_rows=int(curve.accepted_rows[index]),
                uncertainty_threshold=float(curve.thresholds[index]),
                selective_loss=float(curve.selective_loss[index]),
                generalized_loss=float(curve.generalized_loss[index]),
            )
        )
    return tuple(selected)


def _fixed_risk_coverage(
    curve: _TieSafeCurve,
) -> tuple[Round74FixedRiskCoverage, ...]:
    selected: list[Round74FixedRiskCoverage] = []
    for ratio in ROUND74_EPISTEMIC_FIXED_RISK_RATIOS:
        limit = ratio * curve.full_coverage_loss
        eligible = np.flatnonzero(curve.selective_loss <= limit + 1e-12)
        if len(eligible) == 0:
            selected.append(
                Round74FixedRiskCoverage(
                    risk_ratio_to_full_coverage=ratio,
                    risk_limit=limit,
                    maximum_attained_coverage=0.0,
                    accepted_rows=0,
                    selective_loss=0.0,
                    uncertainty_threshold=None,
                )
            )
            continue
        index = int(eligible[-1])
        selected.append(
            Round74FixedRiskCoverage(
                risk_ratio_to_full_coverage=ratio,
                risk_limit=limit,
                maximum_attained_coverage=float(curve.coverage[index]),
                accepted_rows=int(curve.accepted_rows[index]),
                selective_loss=float(curve.selective_loss[index]),
                uncertainty_threshold=float(curve.thresholds[index]),
            )
        )
    return tuple(selected)


def _evaluate_metric(
    *,
    metric_id: str,
    loss_id: str,
    scope: str,
    symbol: str,
    horizon_seconds: int,
    side: str,
    outcome_class: str,
    uncertainty: np.ndarray,
    loss: np.ndarray,
    run_index: np.ndarray,
    seed_material: str,
) -> Round74EpistemicRiskCoverageMetric:
    curve = _tie_safe_curve(uncertainty, loss)
    unique_runs = tuple(dict.fromkeys(int(value) for value in run_index))
    run_improvements: list[float] = []
    for selected_run in unique_runs:
        mask = run_index == selected_run
        run_curve = _tie_safe_curve(uncertainty[mask], loss[mask])
        run_improvements.append(run_curve.full_coverage_loss - run_curve.aurc)
    bootstrap = moving_block_bootstrap_mean(
        run_improvements,
        samples=ROUND74_EPISTEMIC_BOOTSTRAP_SAMPLES,
        confidence=ROUND74_EPISTEMIC_PER_METRIC_CONFIDENCE,
        seed_material=seed_material,
    )
    improvement = curve.full_coverage_loss - curve.aurc
    improvement_ratio = (
        improvement / curve.full_coverage_loss
        if curve.full_coverage_loss > _EPSILON
        else 0.0
    )
    assessable = bool(
        len(loss) >= ROUND74_EPISTEMIC_MINIMUM_STRATUM_ROWS
        and len(unique_runs) >= ROUND74_EPISTEMIC_REQUIRED_POLICY_SELECTION_RUNS
        and len(curve.thresholds) >= 2
        and curve.full_coverage_loss > _EPSILON
    )
    supported = bool(
        assessable
        and float(bootstrap["mean_ci_lower"]) > 0.0
        and curve.rank_correlation > 0.0
    )
    result = Round74EpistemicRiskCoverageMetric(
        metric_id=metric_id,
        loss_id=loss_id,
        scope=scope,
        symbol=symbol,
        horizon_seconds=horizon_seconds,
        side=side,
        outcome_class=outcome_class,
        population_rows=len(loss),
        capture_runs=len(unique_runs),
        unique_uncertainty_thresholds=len(curve.thresholds),
        full_coverage_loss=curve.full_coverage_loss,
        area_under_risk_coverage=curve.aurc,
        area_under_generalized_risk_coverage=curve.augrc,
        aurc_improvement_over_no_information=improvement,
        aurc_improvement_ratio=improvement_ratio,
        uncertainty_loss_rank_correlation=curve.rank_correlation,
        run_mean_aurc_improvement=float(np.mean(run_improvements)),
        run_mean_aurc_improvement_ci_lower=float(bootstrap["mean_ci_lower"]),
        run_mean_aurc_improvement_ci_upper=float(bootstrap["mean_ci_upper"]),
        bootstrap_samples=int(bootstrap["samples"]),
        bootstrap_confidence=float(bootstrap["confidence"]),
        bootstrap_block_length=int(bootstrap["block_length"]),
        curve_sha256=curve.curve_sha256,
        curve_points=_curve_points(curve),
        fixed_risk_coverage=_fixed_risk_coverage(curve),
        ordering_assessable=assessable,
        ordering_supported=supported,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class _MetricPanel:
    metric_id: str
    loss_id: str
    uncertainty: np.ndarray
    loss: np.ndarray
    run_index: np.ndarray
    symbol_index: np.ndarray
    horizon_index: np.ndarray
    side_index: np.ndarray
    outcome_index: np.ndarray
    outcome_labels: tuple[str, str]
    shared_side: bool = False


def _action_metric_panel(
    batches: Sequence[Round74EpistemicEvaluationBatch],
    *,
    metric_id: str,
) -> _MetricPanel:
    uncertainty_values: list[np.ndarray] = []
    loss_values: list[np.ndarray] = []
    run_values: list[np.ndarray] = []
    symbol_values: list[np.ndarray] = []
    horizon_values: list[np.ndarray] = []
    side_values: list[np.ndarray] = []
    outcome_values: list[np.ndarray] = []
    for run_index, batch in enumerate(batches):
        mask = batch.action_eligibility.astype(bool)
        if metric_id == "payoff_quantile_peer_dispersion":
            uncertainty = batch.payoff_quantile_peer_dispersion_bps
            loss = _quantile_loss(
                batch.payoff_quantiles_bps,
                batch.net_payoff_bps,
            )
            outcome = (batch.net_payoff_bps > 0.0).astype(np.int8)
            outcome_labels = ("non_positive_payoff", "positive_payoff")
            loss_id = "mean_pinball_loss_bps"
        elif metric_id == "adverse_excursion_quantile_peer_dispersion":
            uncertainty = batch.adverse_excursion_quantile_peer_dispersion_bps
            loss = _quantile_loss(
                batch.maximum_adverse_excursion_quantiles_bps,
                batch.maximum_adverse_excursion_bps,
            )
            outcome = batch.adverse_selection.astype(np.int8)
            outcome_labels = ("not_adverse_selection", "adverse_selection")
            loss_id = "mean_pinball_loss_bps"
        elif metric_id == "positive_payoff_probability_peer_dispersion":
            uncertainty = batch.positive_payoff_probability_peer_dispersion
            target = (batch.net_payoff_bps > 0.0).astype(np.float64)
            loss = np.square(batch.positive_payoff_probability - target)
            outcome = target.astype(np.int8)
            outcome_labels = ("non_positive_payoff", "positive_payoff")
            loss_id = "brier_loss"
        elif metric_id == "adverse_selection_probability_peer_dispersion":
            uncertainty = batch.adverse_selection_probability_peer_dispersion
            loss = np.square(
                batch.adverse_selection_probability - batch.adverse_selection
            )
            outcome = batch.adverse_selection.astype(np.int8)
            outcome_labels = ("not_adverse_selection", "adverse_selection")
            loss_id = "brier_loss"
        else:
            raise ValueError("Round 74 action epistemic metric differs")
        shape = mask.shape
        symbols = np.broadcast_to(
            np.asarray(
                [ROUND74_EVENT_SYMBOLS.index(value) for value in batch.symbol],
                dtype=np.int8,
            ).reshape(batch.rows, 1, 1),
            shape,
        )
        horizons = np.broadcast_to(
            np.arange(shape[1], dtype=np.int8).reshape(1, shape[1], 1),
            shape,
        )
        sides = np.broadcast_to(
            np.arange(shape[2], dtype=np.int8).reshape(1, 1, shape[2]),
            shape,
        )
        uncertainty_values.append(uncertainty[mask])
        loss_values.append(loss[mask])
        run_values.append(np.full(int(mask.sum()), run_index, dtype=np.int16))
        symbol_values.append(symbols[mask])
        horizon_values.append(horizons[mask])
        side_values.append(sides[mask])
        outcome_values.append(outcome[mask])
    return _MetricPanel(
        metric_id=metric_id,
        loss_id=loss_id,
        uncertainty=np.concatenate(uncertainty_values),
        loss=np.concatenate(loss_values),
        run_index=np.concatenate(run_values),
        symbol_index=np.concatenate(symbol_values),
        horizon_index=np.concatenate(horizon_values),
        side_index=np.concatenate(side_values),
        outcome_index=np.concatenate(outcome_values),
        outcome_labels=outcome_labels,
    )


def _regime_metric_panel(
    batches: Sequence[Round74EpistemicEvaluationBatch],
) -> _MetricPanel:
    uncertainty_values: list[np.ndarray] = []
    loss_values: list[np.ndarray] = []
    run_values: list[np.ndarray] = []
    symbol_values: list[np.ndarray] = []
    horizon_values: list[np.ndarray] = []
    outcome_values: list[np.ndarray] = []
    for run_index, batch in enumerate(batches):
        mask = batch.regime_unpredictability_eligibility.astype(bool)
        shape = mask.shape
        uncertainty_values.append(
            batch.regime_unpredictability_probability_peer_dispersion[mask]
        )
        loss_values.append(
            np.square(
                batch.regime_unpredictability_probability
                - batch.regime_unpredictability
            )[mask]
        )
        run_values.append(np.full(int(mask.sum()), run_index, dtype=np.int16))
        symbol_values.append(
            np.broadcast_to(
                np.asarray(
                    [ROUND74_EVENT_SYMBOLS.index(value) for value in batch.symbol],
                    dtype=np.int8,
                ).reshape(batch.rows, 1),
                shape,
            )[mask]
        )
        horizon_values.append(
            np.broadcast_to(
                np.arange(shape[1], dtype=np.int8).reshape(1, shape[1]),
                shape,
            )[mask]
        )
        outcome_values.append(
            (batch.regime_unpredictability >= 0.5).astype(np.int8)[mask]
        )
    uncertainty = np.concatenate(uncertainty_values)
    return _MetricPanel(
        metric_id="regime_unpredictability_probability_peer_dispersion",
        loss_id="soft_brier_loss",
        uncertainty=uncertainty,
        loss=np.concatenate(loss_values),
        run_index=np.concatenate(run_values),
        symbol_index=np.concatenate(symbol_values),
        horizon_index=np.concatenate(horizon_values),
        side_index=np.full(len(uncertainty), -1, dtype=np.int8),
        outcome_index=np.concatenate(outcome_values),
        outcome_labels=("lower_path_unpredictability", "higher_path_unpredictability"),
        shared_side=True,
    )


@dataclass(frozen=True)
class Round74EpistemicRiskCoverageReport:
    tuning_subpartition_sha256: str
    probability_calibration_sha256: str
    policy_selection_run_ids: tuple[str, ...]
    policy_selection_batch_sha256: tuple[str, ...]
    model_output_sha256: tuple[str, ...]
    peer_count: int
    metrics: tuple[Round74EpistemicRiskCoverageMetric, ...]
    missing_required_strata: tuple[str, ...]
    required_strata_complete: bool
    aggregate_ordering_supported: bool
    conditional_ordering_supported: bool
    policy_challenge_eligible: bool
    schema_version: str = ROUND74_EPISTEMIC_RISK_COVERAGE_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_EPISTEMIC_RISK_COVERAGE_SCHEMA_VERSION
            or _SHA256.fullmatch(self.tuning_subpartition_sha256) is None
            or _SHA256.fullmatch(self.probability_calibration_sha256) is None
            or len(self.policy_selection_run_ids)
            != ROUND74_EPISTEMIC_REQUIRED_POLICY_SELECTION_RUNS
            or any(
                _RUN_ID.fullmatch(value) is None
                for value in self.policy_selection_run_ids
            )
            or len(set(self.policy_selection_run_ids))
            != len(self.policy_selection_run_ids)
            or len(self.policy_selection_batch_sha256)
            != len(self.policy_selection_run_ids)
            or len(self.model_output_sha256) != len(self.policy_selection_run_ids)
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    *self.policy_selection_batch_sha256,
                    *self.model_output_sha256,
                )
            )
            or isinstance(self.peer_count, bool)
            or self.peer_count < 2
            or not self.metrics
            or len({metric.metric_id for metric in self.metrics})
            != len(ROUND74_EPISTEMIC_RISK_COVERAGE_METRIC_IDS)
            or len(
                {
                    (
                        metric.metric_id,
                        metric.scope,
                        metric.symbol,
                        metric.horizon_seconds,
                        metric.side,
                        metric.outcome_class,
                    )
                    for metric in self.metrics
                }
            )
            != len(self.metrics)
            or tuple(sorted(set(self.missing_required_strata)))
            != self.missing_required_strata
            or any(not value for value in self.missing_required_strata)
            or not isinstance(self.required_strata_complete, bool)
            or not isinstance(self.aggregate_ordering_supported, bool)
            or not isinstance(self.conditional_ordering_supported, bool)
            or not isinstance(self.policy_challenge_eligible, bool)
        ):
            raise ValueError("Round 74 epistemic risk-coverage report differs")
        for metric in self.metrics:
            metric.validate()
        aggregate = tuple(
            metric for metric in self.metrics if metric.scope == "aggregate"
        )
        conditional = tuple(
            metric for metric in self.metrics if metric.scope == "conditional"
        )
        expected_complete = not self.missing_required_strata
        expected_aggregate = len(aggregate) == len(
            ROUND74_EPISTEMIC_RISK_COVERAGE_METRIC_IDS
        ) and all(metric.ordering_supported for metric in aggregate)
        expected_conditional = (
            expected_complete
            and bool(conditional)
            and all(metric.ordering_supported for metric in conditional)
        )
        expected_eligible = (
            self.peer_count == 3
            and expected_complete
            and expected_aggregate
            and expected_conditional
        )
        if (
            self.required_strata_complete != expected_complete
            or self.aggregate_ordering_supported != expected_aggregate
            or self.conditional_ordering_supported != expected_conditional
            or self.policy_challenge_eligible != expected_eligible
        ):
            raise ValueError("Round 74 epistemic risk-coverage conclusion differs")

    @property
    def report_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "tuning_subpartition_sha256": self.tuning_subpartition_sha256,
            "probability_calibration_sha256": (self.probability_calibration_sha256),
            "policy_selection_run_ids": list(self.policy_selection_run_ids),
            "policy_selection_batch_sha256": list(self.policy_selection_batch_sha256),
            "model_output_sha256": list(self.model_output_sha256),
            "peer_count": self.peer_count,
            "metrics": [metric.as_dict() for metric in self.metrics],
            "missing_required_strata": list(self.missing_required_strata),
            "required_strata_complete": self.required_strata_complete,
            "aggregate_ordering_supported": self.aggregate_ordering_supported,
            "conditional_ordering_supported": self.conditional_ordering_supported,
            "policy_challenge_eligible": self.policy_challenge_eligible,
            "evaluation_contract": {
                "source_role": "disjoint_tuning_policy_selection_runs",
                "tie_handling": "whole_equal_uncertainty_blocks",
                "curve_area": "exact_over_all_attainable_tie_safe_thresholds",
                "curve_plot_points": list(ROUND74_EPISTEMIC_CURVE_TARGET_COVERAGES),
                "fixed_risk_ratios_to_full_coverage": list(
                    ROUND74_EPISTEMIC_FIXED_RISK_RATIOS
                ),
                "minimum_rows_per_required_stratum": (
                    ROUND74_EPISTEMIC_MINIMUM_STRATUM_ROWS
                ),
                "required_capture_runs": (
                    ROUND74_EPISTEMIC_REQUIRED_POLICY_SELECTION_RUNS
                ),
                "bootstrap_samples": ROUND74_EPISTEMIC_BOOTSTRAP_SAMPLES,
                "familywise_confidence": (ROUND74_EPISTEMIC_FAMILYWISE_CONFIDENCE),
                "per_metric_confidence": (ROUND74_EPISTEMIC_PER_METRIC_CONFIDENCE),
                "aggregate_only_promotion_permitted": False,
                "sealed_test_accessed": False,
            },
            "policy_effects": {
                "candidate_eligibility_changed": False,
                "candidate_ranking_changed": False,
                "position_size_changed": False,
                "leverage_changed": False,
                "automatic_policy_gate_enabled": False,
            },
            "authority": {
                "financial_edge_tested": False,
                "profitability_claim": False,
                "paper_trading_authority": False,
                "testnet_trading_authority": False,
                "live_trading_authority": False,
            },
        }
        if include_sha256:
            value["report_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74EpistemicRiskCoverageReport:
        payload = dict(value)
        claimed = payload.pop("report_sha256", None)
        metrics = payload.get("metrics")
        contract = payload.get("evaluation_contract")
        effects = payload.get("policy_effects")
        authority = payload.get("authority")
        expected_keys = {
            "schema_version",
            "tuning_subpartition_sha256",
            "probability_calibration_sha256",
            "policy_selection_run_ids",
            "policy_selection_batch_sha256",
            "model_output_sha256",
            "peer_count",
            "metrics",
            "missing_required_strata",
            "required_strata_complete",
            "aggregate_ordering_supported",
            "conditional_ordering_supported",
            "policy_challenge_eligible",
            "evaluation_contract",
            "policy_effects",
            "authority",
        }
        if (
            _SHA256.fullmatch(str(claimed)) is None
            or claimed != _canonical_sha256(payload)
            or set(payload) != expected_keys
            or not isinstance(metrics, list)
            or not all(isinstance(item, Mapping) for item in metrics)
            or not isinstance(contract, Mapping)
            or not isinstance(effects, Mapping)
            or not isinstance(authority, Mapping)
            or set(effects)
            != {
                "candidate_eligibility_changed",
                "candidate_ranking_changed",
                "position_size_changed",
                "leverage_changed",
                "automatic_policy_gate_enabled",
            }
            or any(item is not False for item in effects.values())
            or set(authority)
            != {
                "financial_edge_tested",
                "profitability_claim",
                "paper_trading_authority",
                "testnet_trading_authority",
                "live_trading_authority",
            }
            or any(item is not False for item in authority.values())
        ):
            raise ValueError("Round 74 epistemic report payload differs")
        expected_effects = {
            "candidate_eligibility_changed": False,
            "candidate_ranking_changed": False,
            "position_size_changed": False,
            "leverage_changed": False,
            "automatic_policy_gate_enabled": False,
        }
        if dict(effects) != expected_effects:
            raise ValueError("Round 74 epistemic report payload differs")
        expected_contract = {
            "source_role": "disjoint_tuning_policy_selection_runs",
            "tie_handling": "whole_equal_uncertainty_blocks",
            "curve_area": "exact_over_all_attainable_tie_safe_thresholds",
            "curve_plot_points": list(ROUND74_EPISTEMIC_CURVE_TARGET_COVERAGES),
            "fixed_risk_ratios_to_full_coverage": list(
                ROUND74_EPISTEMIC_FIXED_RISK_RATIOS
            ),
            "minimum_rows_per_required_stratum": (
                ROUND74_EPISTEMIC_MINIMUM_STRATUM_ROWS
            ),
            "required_capture_runs": (ROUND74_EPISTEMIC_REQUIRED_POLICY_SELECTION_RUNS),
            "bootstrap_samples": ROUND74_EPISTEMIC_BOOTSTRAP_SAMPLES,
            "familywise_confidence": ROUND74_EPISTEMIC_FAMILYWISE_CONFIDENCE,
            "per_metric_confidence": ROUND74_EPISTEMIC_PER_METRIC_CONFIDENCE,
            "aggregate_only_promotion_permitted": False,
            "sealed_test_accessed": False,
        }
        if dict(contract) != expected_contract:
            raise ValueError("Round 74 epistemic evaluation contract differs")

        def strings(name: str) -> tuple[str, ...]:
            selected = payload[name]
            if not isinstance(selected, list) or any(
                not isinstance(item, str) for item in selected
            ):
                raise ValueError("Round 74 epistemic report sequence differs")
            return tuple(selected)

        selected = cls(
            tuning_subpartition_sha256=str(payload["tuning_subpartition_sha256"]),
            probability_calibration_sha256=str(
                payload["probability_calibration_sha256"]
            ),
            policy_selection_run_ids=strings("policy_selection_run_ids"),
            policy_selection_batch_sha256=strings("policy_selection_batch_sha256"),
            model_output_sha256=strings("model_output_sha256"),
            peer_count=_json_int(payload["peer_count"], "peer count"),
            metrics=tuple(
                Round74EpistemicRiskCoverageMetric.from_dict(item) for item in metrics
            ),
            missing_required_strata=strings("missing_required_strata"),
            required_strata_complete=_json_bool(
                payload["required_strata_complete"], "required strata complete"
            ),
            aggregate_ordering_supported=_json_bool(
                payload["aggregate_ordering_supported"],
                "aggregate ordering supported",
            ),
            conditional_ordering_supported=_json_bool(
                payload["conditional_ordering_supported"],
                "conditional ordering supported",
            ),
            policy_challenge_eligible=_json_bool(
                payload["policy_challenge_eligible"],
                "policy challenge eligible",
            ),
            schema_version=str(payload["schema_version"]),
        )
        selected.validate()
        if selected.report_sha256 != claimed:
            raise ValueError("Round 74 epistemic report identity differs")
        return selected


def _conditional_key(
    metric_id: str,
    symbol: str,
    horizon_seconds: int,
    side: str,
    outcome_class: str,
) -> str:
    return f"{metric_id}:{symbol}:{horizon_seconds}:{side}:{outcome_class}"


def evaluate_round74_epistemic_risk_coverage(
    batches: Sequence[Round74EpistemicEvaluationBatch],
    *,
    expected_policy_selection_run_ids: Sequence[str],
) -> Round74EpistemicRiskCoverageReport:
    """Evaluate aggregate and class-conditional ordering without policy effects."""

    selected = tuple(batches)
    expected_runs = tuple(str(value) for value in expected_policy_selection_run_ids)
    for batch in selected:
        batch.validate()
    observed_runs = tuple(
        dict.fromkeys(run_id for batch in selected for run_id in batch.run_id)
    )
    if (
        len(selected) != ROUND74_EPISTEMIC_REQUIRED_POLICY_SELECTION_RUNS
        or len(expected_runs) != ROUND74_EPISTEMIC_REQUIRED_POLICY_SELECTION_RUNS
        or observed_runs != expected_runs
        or tuple(tuple(dict.fromkeys(batch.run_id)) for batch in selected)
        != tuple((run_id,) for run_id in expected_runs)
        or len({batch.probability_calibration_sha256 for batch in selected}) != 1
        or len({batch.tuning_subpartition_sha256 for batch in selected}) != 1
        or len({batch.peer_count for batch in selected}) != 1
        or len({batch.batch_sha256 for batch in selected}) != len(selected)
    ):
        raise ValueError("Round 74 epistemic policy-selection panel differs")
    panels = tuple(
        _action_metric_panel(selected, metric_id=metric_id)
        for metric_id in ROUND74_EPISTEMIC_RISK_COVERAGE_METRIC_IDS[:-1]
    ) + (_regime_metric_panel(selected),)
    metrics: list[Round74EpistemicRiskCoverageMetric] = []
    missing: list[str] = []
    source_seed = _canonical_sha256(
        {
            "schema_version": ROUND74_EPISTEMIC_RISK_COVERAGE_SCHEMA_VERSION,
            "run_ids": list(expected_runs),
            "batch_sha256": [batch.batch_sha256 for batch in selected],
            "model_output_sha256": [batch.model_output_sha256 for batch in selected],
            "probability_calibration_sha256": (
                selected[0].probability_calibration_sha256
            ),
        }
    )
    for panel in panels:
        metrics.append(
            _evaluate_metric(
                metric_id=panel.metric_id,
                loss_id=panel.loss_id,
                scope="aggregate",
                symbol="all",
                horizon_seconds=0,
                side="all",
                outcome_class="all",
                uncertainty=panel.uncertainty,
                loss=panel.loss,
                run_index=panel.run_index,
                seed_material=f"{source_seed}:{panel.metric_id}:aggregate",
            )
        )
        side_values = (
            (("shared", -1),)
            if panel.shared_side
            else tuple(
                (side, index) for index, side in enumerate(ROUND74_EVENT_PAYOFF_SIDES)
            )
        )
        for symbol_index, symbol in enumerate(ROUND74_EVENT_SYMBOLS):
            for horizon_index, horizon in enumerate(
                ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS
            ):
                for side, side_index in side_values:
                    for outcome_index, outcome_class in enumerate(panel.outcome_labels):
                        mask = (
                            (panel.symbol_index == symbol_index)
                            & (panel.horizon_index == horizon_index)
                            & (panel.side_index == side_index)
                            & (panel.outcome_index == outcome_index)
                        )
                        key = _conditional_key(
                            panel.metric_id,
                            symbol,
                            horizon,
                            side,
                            outcome_class,
                        )
                        if not bool(mask.any()):
                            missing.append(key)
                            continue
                        metrics.append(
                            _evaluate_metric(
                                metric_id=panel.metric_id,
                                loss_id=panel.loss_id,
                                scope="conditional",
                                symbol=symbol,
                                horizon_seconds=horizon,
                                side=side,
                                outcome_class=outcome_class,
                                uncertainty=panel.uncertainty[mask],
                                loss=panel.loss[mask],
                                run_index=panel.run_index[mask],
                                seed_material=f"{source_seed}:{key}",
                            )
                        )
    missing_strata = tuple(sorted(missing))
    aggregate = tuple(metric for metric in metrics if metric.scope == "aggregate")
    conditional = tuple(metric for metric in metrics if metric.scope == "conditional")
    complete = not missing_strata
    aggregate_supported = len(aggregate) == len(
        ROUND74_EPISTEMIC_RISK_COVERAGE_METRIC_IDS
    ) and all(metric.ordering_supported for metric in aggregate)
    conditional_supported = (
        complete
        and bool(conditional)
        and all(metric.ordering_supported for metric in conditional)
    )
    peer_count = selected[0].peer_count
    result = Round74EpistemicRiskCoverageReport(
        tuning_subpartition_sha256=selected[0].tuning_subpartition_sha256,
        probability_calibration_sha256=(selected[0].probability_calibration_sha256),
        policy_selection_run_ids=expected_runs,
        policy_selection_batch_sha256=tuple(batch.batch_sha256 for batch in selected),
        model_output_sha256=tuple(batch.model_output_sha256 for batch in selected),
        peer_count=peer_count,
        metrics=tuple(metrics),
        missing_required_strata=missing_strata,
        required_strata_complete=complete,
        aggregate_ordering_supported=aggregate_supported,
        conditional_ordering_supported=conditional_supported,
        policy_challenge_eligible=(
            peer_count == 3
            and complete
            and aggregate_supported
            and conditional_supported
        ),
    )
    result.validate()
    return result


__all__ = [
    "ROUND74_EPISTEMIC_BOOTSTRAP_SAMPLES",
    "ROUND74_EPISTEMIC_CURVE_TARGET_COVERAGES",
    "ROUND74_EPISTEMIC_EVALUATION_BATCH_SCHEMA_VERSION",
    "ROUND74_EPISTEMIC_FAMILYWISE_CONFIDENCE",
    "ROUND74_EPISTEMIC_FIXED_RISK_RATIOS",
    "ROUND74_EPISTEMIC_MINIMUM_STRATUM_ROWS",
    "ROUND74_EPISTEMIC_PER_METRIC_CONFIDENCE",
    "ROUND74_EPISTEMIC_REQUIRED_POLICY_SELECTION_RUNS",
    "ROUND74_EPISTEMIC_RISK_COVERAGE_METRIC_IDS",
    "ROUND74_EPISTEMIC_RISK_COVERAGE_SCHEMA_VERSION",
    "Round74EpistemicEvaluationBatch",
    "Round74EpistemicRiskCoverageMetric",
    "Round74EpistemicRiskCoverageReport",
    "Round74FixedRiskCoverage",
    "Round74RiskCoveragePoint",
    "evaluate_round74_epistemic_risk_coverage",
    "prepare_round74_epistemic_evaluation_batch",
]
