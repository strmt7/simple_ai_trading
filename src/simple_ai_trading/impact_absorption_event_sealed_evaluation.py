"""One-use, output-bound Round 74 sealed ML and local-AI evaluation.

The public entry point reserves the immutable test identity before loading the
model, deriving candidates, or reading realized payoffs. AI review coverage is
defined on every target-free candidate above the frozen tuning threshold. The
financial replay then preserves the baseline ML action sequence; AI may only
retain, reduce, or veto those same observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Protocol
import warnings

import numpy as np
import torch

from .compute import require_backend, resolve_backend, torch_device_for_backend
from .impact_absorption_ai_uplift import (
    ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS,
    Round74AIExecutionReplayEvidence,
    Round74AIPairedReviewEvidence,
    Round74AIPretestQualificationPanel,
)
from .impact_absorption_ai_execution_replay import (
    Round74AIExecutionReplayInstruction,
    build_round74_ai_execution_replay_instructions,
)
from .impact_absorption_event_action_policy import (
    Round74ActionCandidateBatch,
    Round74ActionInferenceContext,
    Round74ActionPolicySelection,
    Round74ActionTrace,
    _model_output_sha256,
    _simulate_round74_action_trace_batches,
    build_round74_action_inference_context,
    derive_round74_action_candidates,
    round74_action_profile,
)
from .impact_absorption_event_calibration import (
    Round74ProbabilityCalibration,
    apply_round74_probability_calibration,
    apply_round74_risk_quantile_calibration,
)
from .impact_absorption_event_cohort import (
    ROUND74_EVENT_COHORT_DEFAULT_ROLE_COUNTS,
)
from .impact_absorption_event_dataset import Round74EventTrainingBatch
from .impact_absorption_event_financial_metrics import (
    round74_conservative_maximum_drawdown_bps,
    round74_maximum_concurrent_adverse_excursion_bps,
    round74_maximum_realized_drawdown_bps,
)
from .impact_absorption_event_model import Round74EventModelOutput
from .impact_absorption_event_sealed_ledger import (
    ROUND74_SEALED_OPTIMIZATION_POPULATIONS,
    Round74SealedDatasetIdentity,
    Round74SealedEvaluationClaim,
    Round74SealedEvaluationLedger,
    build_round74_sealed_dataset_identity,
)
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SYMBOLS,
)
from .impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS,
)
from .impact_absorption_event_training import load_round74_pretest_policy


ROUND74_SEALED_EVALUATION_SCHEMA_VERSION = "round-074-sealed-evaluation-v20"
ROUND74_TARGET_FREE_INFERENCE_SCHEMA_VERSION = (
    "round-074-target-free-candidate-inference-v3"
)
ROUND74_TARGET_FREE_INFERENCE_DATA_SCOPES = (
    "sealed_test",
    "ai_qualification_tuning",
)
ROUND74_SEALED_BOOTSTRAP_DRAWS = 10_000
ROUND74_SEALED_BOOTSTRAP_SEED = 7_474_011
ROUND74_SEALED_INFERENCE_MINIBATCH_ROWS = 2_048
ROUND74_SEALED_ECE_BINS = 10
ROUND74_SEALED_TEST_RUNS = ROUND74_EVENT_COHORT_DEFAULT_ROLE_COUNTS["test"]
ROUND74_SEALED_FAMILYWISE_ALPHA = 0.05
ROUND74_SEALED_QUALIFICATION_CONFIGURATION_COUNT = 3
ROUND74_SEALED_AI_MODEL_COUNT = 2
ROUND74_SEALED_AI_REVIEW_HORIZONS_SECONDS = (30, 300)
ROUND74_SEALED_BINARY_PREDICTIVE_TASKS = (
    "positive_payoff",
    "adverse_selection",
    "regime_unpredictability",
)
ROUND74_SEALED_QUANTILE_PREDICTIVE_TASKS = (
    "net_payoff_quantiles",
    "maximum_adverse_excursion_quantiles",
)
ROUND74_SEALED_PREDICTIVE_TASKS = (
    *ROUND74_SEALED_BINARY_PREDICTIVE_TASKS,
    *ROUND74_SEALED_QUANTILE_PREDICTIVE_TASKS,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_ACTIVITY_REGIMES = ("predictable", "unpredictable", "unavailable")


class Round74SealedTestBatchLoader(Protocol):
    """Load the target-bearing test panel only after a live reservation."""

    def __call__(
        self,
        *,
        claim: Round74SealedEvaluationClaim,
    ) -> Sequence[Round74EventTrainingBatch]: ...


class Round74SealedAIReviewProvider(Protocol):
    """Review only target-free contexts after the test access is consumed."""

    def __call__(
        self,
        *,
        claim: Round74SealedEvaluationClaim,
        manifests: tuple[str, ...],
        inference: Round74TargetFreeCandidateInference,
        action_selection: Round74ActionPolicySelection,
    ) -> Mapping[str, Sequence[Round74AIPairedReviewEvidence]]: ...


class Round74SealedAIExecutionReplayProvider(Protocol):
    """Perform target-bearing replay only from post-reservation instructions."""

    def __call__(
        self,
        *,
        claim: Round74SealedEvaluationClaim,
        instructions_by_manifest: Mapping[
            str,
            Sequence[Round74AIExecutionReplayInstruction],
        ],
    ) -> Mapping[str, Sequence[Round74AIExecutionReplayEvidence]]: ...


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


def _require_sha256(value: object, label: str) -> str:
    selected = str(value)
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"Round 74 sealed evaluation {label} digest differs")
    return selected


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _tensor_array(value: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(
        value.detach().to(device="cpu", dtype=torch.float32).numpy()
    ).astype(np.float64)


@dataclass(frozen=True)
class Round74BinaryForecastMetrics:
    observations: int
    positive_observations: int
    prevalence: float
    brier_score: float
    expected_calibration_error: float
    accuracy_at_0_5: float
    balanced_accuracy_at_0_5: float
    matthews_correlation_coefficient_at_0_5: float
    single_class: bool

    def validate(self) -> None:
        finite = (
            self.prevalence,
            self.brier_score,
            self.expected_calibration_error,
            self.accuracy_at_0_5,
            self.balanced_accuracy_at_0_5,
            self.matthews_correlation_coefficient_at_0_5,
        )
        if (
            isinstance(self.observations, bool)
            or self.observations < 1
            or isinstance(self.positive_observations, bool)
            or not 0 <= self.positive_observations <= self.observations
            or any(not math.isfinite(float(value)) for value in finite)
            or any(not 0.0 <= float(value) <= 1.0 for value in finite[:-1])
            or not -1.0 <= self.matthews_correlation_coefficient_at_0_5 <= 1.0
            or self.single_class
            != (self.positive_observations in (0, self.observations))
        ):
            raise ValueError("Round 74 sealed binary metrics differ")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return dict(self.__dict__)


@dataclass(frozen=True)
class Round74QuantileForecastMetrics:
    observations: int
    mean_pinball_loss_bps: float
    no_information_mean_pinball_loss_bps: float
    pinball_skill_score: float
    empirical_coverage: tuple[float, ...]

    def validate(self) -> None:
        if (
            isinstance(self.observations, bool)
            or self.observations < 1
            or not math.isfinite(float(self.mean_pinball_loss_bps))
            or not math.isfinite(float(self.no_information_mean_pinball_loss_bps))
            or not math.isfinite(float(self.pinball_skill_score))
            or self.mean_pinball_loss_bps < 0.0
            or self.no_information_mean_pinball_loss_bps < 0.0
            or self.pinball_skill_score > 1.0
            or (
                self.no_information_mean_pinball_loss_bps == 0.0
                and self.pinball_skill_score != 0.0
            )
            or len(self.empirical_coverage) != len(ROUND74_EVENT_PAYOFF_QUANTILES)
            or any(
                not math.isfinite(float(value)) or not 0.0 <= value <= 1.0
                for value in self.empirical_coverage
            )
        ):
            raise ValueError("Round 74 sealed quantile metrics differ")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "observations": self.observations,
            "quantiles": list(ROUND74_EVENT_PAYOFF_QUANTILES),
            "mean_pinball_loss_bps": self.mean_pinball_loss_bps,
            "no_information_mean_pinball_loss_bps": (
                self.no_information_mean_pinball_loss_bps
            ),
            "pinball_skill_score": self.pinball_skill_score,
            "empirical_coverage": list(self.empirical_coverage),
        }


@dataclass(frozen=True)
class Round74ActionForecastSlice:
    symbol: str
    horizon_seconds: int
    side: str
    activity_regime: str
    payoff: Round74QuantileForecastMetrics
    maximum_adverse_excursion: Round74QuantileForecastMetrics
    positive_payoff: Round74BinaryForecastMetrics
    adverse_selection: Round74BinaryForecastMetrics

    def validate(self) -> None:
        if (
            self.symbol not in ROUND74_EVENT_SYMBOLS
            or self.horizon_seconds not in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS
            or self.side not in ROUND74_EVENT_PAYOFF_SIDES
            or self.activity_regime not in _ACTIVITY_REGIMES
        ):
            raise ValueError("Round 74 sealed action forecast slice differs")
        self.payoff.validate()
        self.maximum_adverse_excursion.validate()
        self.positive_payoff.validate()
        self.adverse_selection.validate()
        counts = {
            self.payoff.observations,
            self.maximum_adverse_excursion.observations,
            self.positive_payoff.observations,
            self.adverse_selection.observations,
        }
        if len(counts) != 1:
            raise ValueError("Round 74 sealed action slice counts differ")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "symbol": self.symbol,
            "horizon_seconds": self.horizon_seconds,
            "side": self.side,
            "activity_regime": self.activity_regime,
            "payoff": self.payoff.as_dict(),
            "maximum_adverse_excursion": (self.maximum_adverse_excursion.as_dict()),
            "positive_payoff": self.positive_payoff.as_dict(),
            "adverse_selection": self.adverse_selection.as_dict(),
        }


@dataclass(frozen=True)
class Round74RegimeForecastSlice:
    symbol: str
    horizon_seconds: int
    regime_unpredictability: Round74BinaryForecastMetrics

    def validate(self) -> None:
        if (
            self.symbol not in ROUND74_EVENT_SYMBOLS
            or self.horizon_seconds not in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS
        ):
            raise ValueError("Round 74 sealed regime forecast slice differs")
        self.regime_unpredictability.validate()

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "symbol": self.symbol,
            "horizon_seconds": self.horizon_seconds,
            "regime_unpredictability": self.regime_unpredictability.as_dict(),
        }


@dataclass(frozen=True)
class Round74SealedPredictiveDiagnostics:
    action_slices: tuple[Round74ActionForecastSlice, ...]
    regime_slices: tuple[Round74RegimeForecastSlice, ...]
    eligible_action_targets: int
    eligible_regime_targets: int
    scope: str = "all_eligible_test_heads_before_action_threshold"

    def validate(self) -> None:
        if (
            not self.action_slices
            or not self.regime_slices
            or isinstance(self.eligible_action_targets, bool)
            or self.eligible_action_targets < 1
            or isinstance(self.eligible_regime_targets, bool)
            or self.eligible_regime_targets < 1
            or self.scope != "all_eligible_test_heads_before_action_threshold"
            or tuple(
                (
                    value.symbol,
                    value.horizon_seconds,
                    value.side,
                    value.activity_regime,
                )
                for value in self.action_slices
            )
            != tuple(
                sorted(
                    (
                        value.symbol,
                        value.horizon_seconds,
                        value.side,
                        value.activity_regime,
                    )
                    for value in self.action_slices
                )
            )
            or tuple(
                (value.symbol, value.horizon_seconds) for value in self.regime_slices
            )
            != tuple(
                sorted(
                    (value.symbol, value.horizon_seconds)
                    for value in self.regime_slices
                )
            )
        ):
            raise ValueError("Round 74 sealed predictive diagnostics differ")
        for value in self.action_slices:
            value.validate()
        for value in self.regime_slices:
            value.validate()

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "scope": self.scope,
            "eligible_action_targets": self.eligible_action_targets,
            "eligible_regime_targets": self.eligible_regime_targets,
            "action_slices": [value.as_dict() for value in self.action_slices],
            "regime_slices": [value.as_dict() for value in self.regime_slices],
        }


def _predictive_brier_gate_reasons(
    *,
    observations: int,
    evaluable_slices: int,
    capture_runs: int,
    covered_capture_runs: int,
    no_information_brier_score: float,
    brier_skill_score: float,
    familywise_lower_mean_run_brier_improvement: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if observations < 1 or evaluable_slices < 1:
        reasons.append("non_single_class_evidence_missing")
    if capture_runs < 2 or covered_capture_runs != capture_runs:
        reasons.append("capture_run_coverage_incomplete")
    if no_information_brier_score <= 0.0:
        reasons.append("no_information_brier_not_positive")
    if brier_skill_score <= 0.0:
        reasons.append("positive_brier_skill_not_met")
    if familywise_lower_mean_run_brier_improvement <= 0.0:
        reasons.append("positive_familywise_run_brier_improvement_lower_bound_not_met")
    return tuple(reasons)


@dataclass(frozen=True)
class Round74PredictiveBrierSkill:
    """Proper-score evidence against a slice-specific prevalence forecast."""

    task: str
    observations: int
    evaluable_slices: int
    capture_runs: int
    covered_capture_runs: int
    model_brier_score: float
    no_information_brier_score: float
    brier_skill_score: float
    mean_run_brier_improvement: float
    familywise_lower_mean_run_brier_improvement: float
    mean_block_length_runs: int
    restart_probability: float
    gate_passed: bool
    gate_reasons: tuple[str, ...]

    def validate(self) -> None:
        integers = (
            self.observations,
            self.evaluable_slices,
            self.capture_runs,
            self.covered_capture_runs,
            self.mean_block_length_runs,
        )
        finite = (
            self.model_brier_score,
            self.no_information_brier_score,
            self.brier_skill_score,
            self.mean_run_brier_improvement,
            self.familywise_lower_mean_run_brier_improvement,
            self.restart_probability,
        )
        expected_reasons = _predictive_brier_gate_reasons(
            observations=self.observations,
            evaluable_slices=self.evaluable_slices,
            capture_runs=self.capture_runs,
            covered_capture_runs=self.covered_capture_runs,
            no_information_brier_score=self.no_information_brier_score,
            brier_skill_score=self.brier_skill_score,
            familywise_lower_mean_run_brier_improvement=(
                self.familywise_lower_mean_run_brier_improvement
            ),
        )
        if (
            self.task not in ROUND74_SEALED_BINARY_PREDICTIVE_TASKS
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in integers
            )
            or any(value < 0 for value in integers)
            or self.covered_capture_runs > self.capture_runs
            or any(not math.isfinite(float(value)) for value in finite)
            or not 0.0 <= self.model_brier_score <= 1.0
            or not 0.0 <= self.no_information_brier_score <= 1.0
            or self.brier_skill_score > 1.0
            or (
                self.capture_runs >= 2
                and not 2 <= self.mean_block_length_runs <= self.capture_runs
            )
            or (self.capture_runs < 2 and self.mean_block_length_runs != 0)
            or (
                self.mean_block_length_runs > 0
                and not math.isclose(
                    self.restart_probability,
                    1.0 / self.mean_block_length_runs,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
            or (self.mean_block_length_runs == 0 and self.restart_probability != 0.0)
            or self.gate_reasons != expected_reasons
            or self.gate_passed != (not expected_reasons)
        ):
            raise ValueError("Round 74 sealed predictive Brier skill differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task": self.task,
            "observations": self.observations,
            "evaluable_slices": self.evaluable_slices,
            "capture_runs": self.capture_runs,
            "covered_capture_runs": self.covered_capture_runs,
            "model_brier_score": self.model_brier_score,
            "no_information_brier_score": self.no_information_brier_score,
            "brier_skill_score": self.brier_skill_score,
            "mean_run_brier_improvement": self.mean_run_brier_improvement,
            "familywise_alpha": (
                ROUND74_SEALED_FAMILYWISE_ALPHA / len(ROUND74_SEALED_PREDICTIVE_TASKS)
            ),
            "familywise_lower_mean_run_brier_improvement": (
                self.familywise_lower_mean_run_brier_improvement
            ),
            "mean_block_length_runs": self.mean_block_length_runs,
            "restart_probability": self.restart_probability,
            "gate_passed": self.gate_passed,
            "gate_reasons": list(self.gate_reasons),
        }


def _predictive_quantile_gate_reasons(
    *,
    observations: int,
    evaluable_slices: int,
    capture_runs: int,
    covered_capture_runs: int,
    no_information_mean_pinball_loss_bps: float,
    pinball_skill_score: float,
    familywise_lower_mean_run_pinball_improvement_bps: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if observations < 1 or evaluable_slices < 1:
        reasons.append("quantile_evidence_missing")
    if capture_runs < 2 or covered_capture_runs != capture_runs:
        reasons.append("capture_run_coverage_incomplete")
    if no_information_mean_pinball_loss_bps <= 0.0:
        reasons.append("no_information_pinball_loss_not_positive")
    if pinball_skill_score <= 0.0:
        reasons.append("positive_pinball_skill_not_met")
    if familywise_lower_mean_run_pinball_improvement_bps <= 0.0:
        reasons.append(
            "positive_familywise_run_pinball_improvement_lower_bound_not_met"
        )
    return tuple(reasons)


@dataclass(frozen=True)
class Round74PredictiveQuantileSkill:
    """Pinball skill against a calibration-only unconditional distribution."""

    task: str
    observations: int
    evaluable_slices: int
    capture_runs: int
    covered_capture_runs: int
    model_mean_pinball_loss_bps: float
    no_information_mean_pinball_loss_bps: float
    pinball_skill_score: float
    mean_run_pinball_improvement_bps: float
    familywise_lower_mean_run_pinball_improvement_bps: float
    mean_block_length_runs: int
    restart_probability: float
    gate_passed: bool
    gate_reasons: tuple[str, ...]

    def validate(self) -> None:
        integers = (
            self.observations,
            self.evaluable_slices,
            self.capture_runs,
            self.covered_capture_runs,
            self.mean_block_length_runs,
        )
        finite = (
            self.model_mean_pinball_loss_bps,
            self.no_information_mean_pinball_loss_bps,
            self.pinball_skill_score,
            self.mean_run_pinball_improvement_bps,
            self.familywise_lower_mean_run_pinball_improvement_bps,
            self.restart_probability,
        )
        expected_reasons = _predictive_quantile_gate_reasons(
            observations=self.observations,
            evaluable_slices=self.evaluable_slices,
            capture_runs=self.capture_runs,
            covered_capture_runs=self.covered_capture_runs,
            no_information_mean_pinball_loss_bps=(
                self.no_information_mean_pinball_loss_bps
            ),
            pinball_skill_score=self.pinball_skill_score,
            familywise_lower_mean_run_pinball_improvement_bps=(
                self.familywise_lower_mean_run_pinball_improvement_bps
            ),
        )
        if (
            self.task not in ROUND74_SEALED_QUANTILE_PREDICTIVE_TASKS
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in integers
            )
            or any(value < 0 for value in integers)
            or self.covered_capture_runs > self.capture_runs
            or any(not math.isfinite(float(value)) for value in finite)
            or self.model_mean_pinball_loss_bps < 0.0
            or self.no_information_mean_pinball_loss_bps < 0.0
            or self.pinball_skill_score > 1.0
            or (
                self.capture_runs >= 2
                and not 2 <= self.mean_block_length_runs <= self.capture_runs
            )
            or (self.capture_runs < 2 and self.mean_block_length_runs != 0)
            or (
                self.mean_block_length_runs > 0
                and not math.isclose(
                    self.restart_probability,
                    1.0 / self.mean_block_length_runs,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
            or (self.mean_block_length_runs == 0 and self.restart_probability != 0.0)
            or self.gate_reasons != expected_reasons
            or self.gate_passed != (not expected_reasons)
        ):
            raise ValueError("Round 74 sealed predictive quantile skill differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task": self.task,
            "observations": self.observations,
            "evaluable_slices": self.evaluable_slices,
            "capture_runs": self.capture_runs,
            "covered_capture_runs": self.covered_capture_runs,
            "model_mean_pinball_loss_bps": self.model_mean_pinball_loss_bps,
            "no_information_mean_pinball_loss_bps": (
                self.no_information_mean_pinball_loss_bps
            ),
            "pinball_skill_score": self.pinball_skill_score,
            "mean_run_pinball_improvement_bps": (self.mean_run_pinball_improvement_bps),
            "familywise_alpha": (
                ROUND74_SEALED_FAMILYWISE_ALPHA / len(ROUND74_SEALED_PREDICTIVE_TASKS)
            ),
            "familywise_lower_mean_run_pinball_improvement_bps": (
                self.familywise_lower_mean_run_pinball_improvement_bps
            ),
            "mean_block_length_runs": self.mean_block_length_runs,
            "restart_probability": self.restart_probability,
            "gate_passed": self.gate_passed,
            "gate_reasons": list(self.gate_reasons),
        }


@dataclass(frozen=True)
class Round74SealedPredictiveGate:
    """Independent predictive-validity gate for every modeled forecast task."""

    task_skills: tuple[
        Round74PredictiveBrierSkill | Round74PredictiveQuantileSkill,
        ...,
    ]
    gate_passed: bool
    gate_reasons: tuple[str, ...]
    scope: str = "all_evaluable_test_slices_before_action_threshold"

    def validate(self) -> None:
        for value in self.task_skills:
            value.validate()
        expected_tasks = tuple(value.task for value in self.task_skills)
        expected_reasons = tuple(
            f"{value.task}:{reason}"
            for value in self.task_skills
            for reason in value.gate_reasons
        )
        if (
            expected_tasks != ROUND74_SEALED_PREDICTIVE_TASKS
            or self.scope != "all_evaluable_test_slices_before_action_threshold"
            or self.gate_reasons != expected_reasons
            or self.gate_passed != (not expected_reasons)
        ):
            raise ValueError("Round 74 sealed predictive gate differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "scope": self.scope,
            "task_skills": [value.as_dict() for value in self.task_skills],
            "gate_passed": self.gate_passed,
            "gate_reasons": list(self.gate_reasons),
            "test_labels_used_for_threshold_selection": False,
            "binary_no_information_baseline": (
                "within_slice_test_prevalence_as_a_conservative_scoring_benchmark"
            ),
            "quantile_no_information_baseline": (
                "fixed_symbol_horizon_side_empirical_quantiles_from_disjoint_"
                "calibration_runs"
            ),
            "test_labels_used_for_quantile_baseline_fit": False,
        }


class _BinaryAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.positive = 0
        self.brier_sum = 0.0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.bin_count = np.zeros(ROUND74_SEALED_ECE_BINS, dtype=np.int64)
        self.bin_probability = np.zeros(
            ROUND74_SEALED_ECE_BINS,
            dtype=np.float64,
        )
        self.bin_positive = np.zeros(
            ROUND74_SEALED_ECE_BINS,
            dtype=np.float64,
        )
        self.run_count: dict[str, int] = {}
        self.run_positive: dict[str, int] = {}
        self.run_brier_sum: dict[str, float] = {}

    def update(
        self,
        target: np.ndarray,
        probability: np.ndarray,
        *,
        run_ids: Sequence[str],
    ) -> None:
        truth = np.asarray(target, dtype=np.bool_)
        estimate = np.asarray(probability, dtype=np.float64)
        selected_run_ids = np.asarray(run_ids, dtype=object)
        if (
            truth.ndim != 1
            or estimate.shape != truth.shape
            or selected_run_ids.shape != truth.shape
            or not estimate.size
            or not np.isfinite(estimate).all()
            or np.any((estimate < 0.0) | (estimate > 1.0))
            or any(_RUN_ID.fullmatch(str(value)) is None for value in selected_run_ids)
        ):
            raise ValueError("Round 74 sealed binary update differs")
        predicted = estimate >= 0.5
        squared_error = np.square(estimate - truth.astype(float))
        self.count += int(truth.size)
        self.positive += int(truth.sum())
        self.brier_sum += float(squared_error.sum())
        self.tp += int(np.sum(predicted & truth))
        self.tn += int(np.sum(~predicted & ~truth))
        self.fp += int(np.sum(predicted & ~truth))
        self.fn += int(np.sum(~predicted & truth))
        for run_id in dict.fromkeys(str(value) for value in selected_run_ids):
            run_mask = selected_run_ids == run_id
            self.run_count[run_id] = self.run_count.get(run_id, 0) + int(run_mask.sum())
            self.run_positive[run_id] = self.run_positive.get(run_id, 0) + int(
                truth[run_mask].sum()
            )
            self.run_brier_sum[run_id] = self.run_brier_sum.get(
                run_id,
                0.0,
            ) + float(squared_error[run_mask].sum())
        bins = np.minimum(
            (estimate * ROUND74_SEALED_ECE_BINS).astype(np.int64),
            ROUND74_SEALED_ECE_BINS - 1,
        )
        self.bin_count += np.bincount(
            bins,
            minlength=ROUND74_SEALED_ECE_BINS,
        )
        self.bin_probability += np.bincount(
            bins,
            weights=estimate,
            minlength=ROUND74_SEALED_ECE_BINS,
        )
        self.bin_positive += np.bincount(
            bins,
            weights=truth.astype(np.float64),
            minlength=ROUND74_SEALED_ECE_BINS,
        )

    def result(self) -> Round74BinaryForecastMetrics:
        if self.count < 1:
            raise ValueError("Round 74 sealed binary accumulator is empty")
        active = self.bin_count > 0
        confidence = np.divide(
            self.bin_probability,
            self.bin_count,
            out=np.zeros_like(self.bin_probability),
            where=active,
        )
        frequency = np.divide(
            self.bin_positive,
            self.bin_count,
            out=np.zeros_like(self.bin_positive),
            where=active,
        )
        ece = float(
            np.sum(
                np.abs(confidence[active] - frequency[active]) * self.bin_count[active]
            )
            / self.count
        )
        positive_recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else None
        negative_recall = self.tn / (self.tn + self.fp) if self.tn + self.fp else None
        recalls = tuple(
            value for value in (positive_recall, negative_recall) if value is not None
        )
        denominator = math.sqrt(
            (self.tp + self.fp)
            * (self.tp + self.fn)
            * (self.tn + self.fp)
            * (self.tn + self.fn)
        )
        result = Round74BinaryForecastMetrics(
            observations=self.count,
            positive_observations=self.positive,
            prevalence=self.positive / self.count,
            brier_score=self.brier_sum / self.count,
            expected_calibration_error=ece,
            accuracy_at_0_5=(self.tp + self.tn) / self.count,
            balanced_accuracy_at_0_5=float(np.mean(recalls)),
            matthews_correlation_coefficient_at_0_5=(
                (self.tp * self.tn - self.fp * self.fn) / denominator
                if denominator > 0.0
                else 0.0
            ),
            single_class=self.positive in (0, self.count),
        )
        result.validate()
        return result


class _QuantileAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.pinball_sum = 0.0
        self.no_information_pinball_sum = 0.0
        self.coverage = np.zeros(
            len(ROUND74_EVENT_PAYOFF_QUANTILES),
            dtype=np.int64,
        )
        self.run_count: dict[str, int] = {}
        self.run_pinball_sum: dict[str, float] = {}
        self.run_no_information_pinball_sum: dict[str, float] = {}

    def update(
        self,
        target: np.ndarray,
        forecast: np.ndarray,
        *,
        no_information_forecast: np.ndarray,
        run_ids: Sequence[str],
    ) -> None:
        truth = np.asarray(target, dtype=np.float64)
        estimate = np.asarray(forecast, dtype=np.float64)
        baseline = np.asarray(no_information_forecast, dtype=np.float64)
        selected_run_ids = np.asarray(run_ids, dtype=object)
        expected_shape = (truth.size, len(ROUND74_EVENT_PAYOFF_QUANTILES))
        if (
            truth.ndim != 1
            or not truth.size
            or estimate.shape != expected_shape
            or baseline.shape != (len(ROUND74_EVENT_PAYOFF_QUANTILES),)
            or selected_run_ids.shape != truth.shape
            or not np.isfinite(truth).all()
            or not np.isfinite(estimate).all()
            or not np.isfinite(baseline).all()
            or np.any(np.diff(estimate, axis=1) < 0.0)
            or np.any(np.diff(baseline) < 0.0)
            or any(_RUN_ID.fullmatch(str(value)) is None for value in selected_run_ids)
        ):
            raise ValueError("Round 74 sealed quantile update differs")
        quantiles = np.asarray(
            ROUND74_EVENT_PAYOFF_QUANTILES,
            dtype=np.float64,
        )
        error = truth[:, None] - estimate
        pinball = np.maximum(
            quantiles * error,
            (quantiles - 1.0) * error,
        )
        baseline_error = truth[:, None] - baseline
        baseline_pinball = np.maximum(
            quantiles * baseline_error,
            (quantiles - 1.0) * baseline_error,
        )
        row_pinball = pinball.mean(axis=1)
        row_baseline_pinball = baseline_pinball.mean(axis=1)
        self.pinball_sum += float(pinball.sum())
        self.no_information_pinball_sum += float(baseline_pinball.sum())
        self.coverage += np.sum(truth[:, None] <= estimate, axis=0)
        self.count += int(truth.size)
        for run_id in dict.fromkeys(str(value) for value in selected_run_ids):
            run_mask = selected_run_ids == run_id
            self.run_count[run_id] = self.run_count.get(run_id, 0) + int(run_mask.sum())
            self.run_pinball_sum[run_id] = self.run_pinball_sum.get(
                run_id,
                0.0,
            ) + float(row_pinball[run_mask].sum())
            self.run_no_information_pinball_sum[run_id] = (
                self.run_no_information_pinball_sum.get(run_id, 0.0)
                + float(row_baseline_pinball[run_mask].sum())
            )

    def result(self) -> Round74QuantileForecastMetrics:
        if self.count < 1:
            raise ValueError("Round 74 sealed quantile accumulator is empty")
        model_loss = self.pinball_sum / self.count / len(ROUND74_EVENT_PAYOFF_QUANTILES)
        baseline_loss = (
            self.no_information_pinball_sum
            / self.count
            / len(ROUND74_EVENT_PAYOFF_QUANTILES)
        )
        result = Round74QuantileForecastMetrics(
            observations=self.count,
            mean_pinball_loss_bps=model_loss,
            no_information_mean_pinball_loss_bps=baseline_loss,
            pinball_skill_score=(
                1.0 - model_loss / baseline_loss if baseline_loss > 0.0 else 0.0
            ),
            empirical_coverage=tuple(
                float(value / self.count) for value in self.coverage
            ),
        )
        result.validate()
        return result


class _ActionAccumulator:
    def __init__(self) -> None:
        self.payoff = _QuantileAccumulator()
        self.mae = _QuantileAccumulator()
        self.positive = _BinaryAccumulator()
        self.adverse = _BinaryAccumulator()

    def update(
        self,
        *,
        payoff_target: np.ndarray,
        payoff_forecast: np.ndarray,
        payoff_no_information_forecast: np.ndarray,
        mae_target: np.ndarray,
        mae_forecast: np.ndarray,
        mae_no_information_forecast: np.ndarray,
        positive_probability: np.ndarray,
        adverse_target: np.ndarray,
        adverse_probability: np.ndarray,
        run_ids: Sequence[str],
    ) -> None:
        self.payoff.update(
            payoff_target,
            payoff_forecast,
            no_information_forecast=payoff_no_information_forecast,
            run_ids=run_ids,
        )
        self.mae.update(
            mae_target,
            mae_forecast,
            no_information_forecast=mae_no_information_forecast,
            run_ids=run_ids,
        )
        self.positive.update(
            payoff_target > 0.0,
            positive_probability,
            run_ids=run_ids,
        )
        self.adverse.update(
            adverse_target > 0.5,
            adverse_probability,
            run_ids=run_ids,
        )

    def result(
        self,
        *,
        symbol: str,
        horizon_seconds: int,
        side: str,
        activity_regime: str,
    ) -> Round74ActionForecastSlice:
        result = Round74ActionForecastSlice(
            symbol=symbol,
            horizon_seconds=horizon_seconds,
            side=side,
            activity_regime=activity_regime,
            payoff=self.payoff.result(),
            maximum_adverse_excursion=self.mae.result(),
            positive_payoff=self.positive.result(),
            adverse_selection=self.adverse.result(),
        )
        result.validate()
        return result


def _build_predictive_brier_skill(
    task: str,
    accumulators: Sequence[_BinaryAccumulator],
    *,
    expected_run_ids: tuple[str, ...],
    seed: int,
) -> Round74PredictiveBrierSkill:
    if (
        task not in ROUND74_SEALED_BINARY_PREDICTIVE_TASKS
        or len(expected_run_ids) < 2
        or len(set(expected_run_ids)) != len(expected_run_ids)
        or any(_RUN_ID.fullmatch(value) is None for value in expected_run_ids)
    ):
        raise ValueError("Round 74 sealed predictive skill population differs")
    selected = tuple(
        accumulator
        for accumulator in accumulators
        if 0 < accumulator.positive < accumulator.count
    )
    run_index = {run_id: index for index, run_id in enumerate(expected_run_ids)}
    run_count = np.zeros(len(expected_run_ids), dtype=np.int64)
    run_model_sse = np.zeros(len(expected_run_ids), dtype=np.float64)
    run_baseline_sse = np.zeros(len(expected_run_ids), dtype=np.float64)
    for accumulator in selected:
        prevalence = accumulator.positive / accumulator.count
        if any(run_id not in run_index for run_id in accumulator.run_count):
            raise ValueError("Round 74 sealed predictive skill run differs")
        for run_id, count in accumulator.run_count.items():
            index = run_index[run_id]
            positive = accumulator.run_positive[run_id]
            run_count[index] += count
            run_model_sse[index] += accumulator.run_brier_sum[run_id]
            run_baseline_sse[index] += (
                positive * (1.0 - prevalence) ** 2 + (count - positive) * prevalence**2
            )
    observations = int(run_count.sum())
    model_sse = float(run_model_sse.sum())
    baseline_sse = float(run_baseline_sse.sum())
    model_brier = model_sse / observations if observations else 0.0
    no_information_brier = baseline_sse / observations if observations else 0.0
    brier_skill = (
        1.0 - model_brier / no_information_brier if no_information_brier > 0.0 else 0.0
    )
    run_improvement = np.divide(
        run_baseline_sse - run_model_sse,
        run_count,
        out=np.zeros_like(run_model_sse),
        where=run_count > 0,
    )
    mean_block_length = _stationary_bootstrap_mean_block_length(len(expected_run_ids))
    sampled = _stationary_bootstrap_means(
        run_improvement,
        draws=ROUND74_SEALED_BOOTSTRAP_DRAWS,
        seed=seed,
        mean_block_length=mean_block_length,
    )
    lower = float(
        np.quantile(
            sampled,
            ROUND74_SEALED_FAMILYWISE_ALPHA / len(ROUND74_SEALED_PREDICTIVE_TASKS),
        )
    )
    covered_runs = int(np.count_nonzero(run_count))
    reasons = _predictive_brier_gate_reasons(
        observations=observations,
        evaluable_slices=len(selected),
        capture_runs=len(expected_run_ids),
        covered_capture_runs=covered_runs,
        no_information_brier_score=no_information_brier,
        brier_skill_score=brier_skill,
        familywise_lower_mean_run_brier_improvement=lower,
    )
    result = Round74PredictiveBrierSkill(
        task=task,
        observations=observations,
        evaluable_slices=len(selected),
        capture_runs=len(expected_run_ids),
        covered_capture_runs=covered_runs,
        model_brier_score=model_brier,
        no_information_brier_score=no_information_brier,
        brier_skill_score=brier_skill,
        mean_run_brier_improvement=float(run_improvement.mean()),
        familywise_lower_mean_run_brier_improvement=lower,
        mean_block_length_runs=mean_block_length,
        restart_probability=1.0 / mean_block_length,
        gate_passed=not reasons,
        gate_reasons=reasons,
    )
    result.validate()
    return result


def _build_predictive_quantile_skill(
    task: str,
    accumulators: Sequence[_QuantileAccumulator],
    *,
    expected_run_ids: tuple[str, ...],
    seed: int,
) -> Round74PredictiveQuantileSkill:
    if (
        task not in ROUND74_SEALED_QUANTILE_PREDICTIVE_TASKS
        or len(expected_run_ids) < 2
        or len(set(expected_run_ids)) != len(expected_run_ids)
        or any(_RUN_ID.fullmatch(value) is None for value in expected_run_ids)
    ):
        raise ValueError("Round 74 sealed predictive quantile population differs")
    selected = tuple(accumulator for accumulator in accumulators if accumulator.count)
    run_index = {run_id: index for index, run_id in enumerate(expected_run_ids)}
    run_count = np.zeros(len(expected_run_ids), dtype=np.int64)
    run_model_loss = np.zeros(len(expected_run_ids), dtype=np.float64)
    run_baseline_loss = np.zeros(len(expected_run_ids), dtype=np.float64)
    for accumulator in selected:
        if any(run_id not in run_index for run_id in accumulator.run_count):
            raise ValueError("Round 74 sealed predictive quantile run differs")
        for run_id, count in accumulator.run_count.items():
            index = run_index[run_id]
            run_count[index] += count
            run_model_loss[index] += accumulator.run_pinball_sum[run_id]
            run_baseline_loss[index] += accumulator.run_no_information_pinball_sum[
                run_id
            ]
    observations = int(run_count.sum())
    model_loss = float(run_model_loss.sum()) / observations if observations else 0.0
    baseline_loss = (
        float(run_baseline_loss.sum()) / observations if observations else 0.0
    )
    pinball_skill = 1.0 - model_loss / baseline_loss if baseline_loss > 0.0 else 0.0
    run_improvement = np.divide(
        run_baseline_loss - run_model_loss,
        run_count,
        out=np.zeros_like(run_model_loss),
        where=run_count > 0,
    )
    mean_block_length = _stationary_bootstrap_mean_block_length(len(expected_run_ids))
    sampled = _stationary_bootstrap_means(
        run_improvement,
        draws=ROUND74_SEALED_BOOTSTRAP_DRAWS,
        seed=seed,
        mean_block_length=mean_block_length,
    )
    lower = float(
        np.quantile(
            sampled,
            ROUND74_SEALED_FAMILYWISE_ALPHA / len(ROUND74_SEALED_PREDICTIVE_TASKS),
        )
    )
    covered_runs = int(np.count_nonzero(run_count))
    reasons = _predictive_quantile_gate_reasons(
        observations=observations,
        evaluable_slices=len(selected),
        capture_runs=len(expected_run_ids),
        covered_capture_runs=covered_runs,
        no_information_mean_pinball_loss_bps=baseline_loss,
        pinball_skill_score=pinball_skill,
        familywise_lower_mean_run_pinball_improvement_bps=lower,
    )
    result = Round74PredictiveQuantileSkill(
        task=task,
        observations=observations,
        evaluable_slices=len(selected),
        capture_runs=len(expected_run_ids),
        covered_capture_runs=covered_runs,
        model_mean_pinball_loss_bps=model_loss,
        no_information_mean_pinball_loss_bps=baseline_loss,
        pinball_skill_score=pinball_skill,
        mean_run_pinball_improvement_bps=float(run_improvement.mean()),
        familywise_lower_mean_run_pinball_improvement_bps=lower,
        mean_block_length_runs=mean_block_length,
        restart_probability=1.0 / mean_block_length,
        gate_passed=not reasons,
        gate_reasons=reasons,
    )
    result.validate()
    return result


class _PredictiveAccumulator:
    def __init__(self) -> None:
        self.action: dict[tuple[str, int, str, str], _ActionAccumulator] = {}
        self.regime: dict[tuple[str, int], _BinaryAccumulator] = {}
        self.eligible_action_targets = 0
        self.eligible_regime_targets = 0

    def update(
        self,
        batch: Round74EventTrainingBatch,
        output: Round74EventModelOutput,
        calibration: Round74ProbabilityCalibration,
    ) -> None:
        output.validate(batch.rows)
        calibration.validate()
        if calibration.quantile_baseline is None:
            raise ValueError(
                "Round 74 sealed no-information quantile baseline is missing"
            )
        positive, adverse, unpredictable = apply_round74_probability_calibration(
            calibration,
            positive_payoff_logits=output.positive_payoff_logits,
            adverse_selection_logits=output.adverse_selection_logits,
            regime_unpredictability_logits=output.regime_unpredictability_logits,
        )
        calibrated_payoff = output.payoff_quantiles_bps
        calibrated_mae = output.maximum_adverse_excursion_quantiles_bps
        if calibration.risk_quantiles is not None:
            calibrated_payoff, calibrated_mae = apply_round74_risk_quantile_calibration(
                calibration.risk_quantiles,
                payoff_quantiles_bps=calibrated_payoff,
                maximum_adverse_excursion_quantiles_bps=calibrated_mae,
            )
        payoff_forecast = _tensor_array(calibrated_payoff)
        mae_forecast = _tensor_array(calibrated_mae)
        positive_probability = _tensor_array(positive)
        adverse_probability = _tensor_array(adverse)
        unpredictable_probability = _tensor_array(unpredictable)
        payoff_no_information = np.asarray(
            calibration.quantile_baseline.payoff_quantiles_bps,
            dtype=np.float64,
        )
        mae_no_information = np.asarray(
            calibration.quantile_baseline.maximum_adverse_excursion_quantiles_bps,
            dtype=np.float64,
        )
        symbols = np.asarray(batch.symbol, dtype=object)
        run_ids = np.asarray(batch.run_id, dtype=object)
        for horizon_index, horizon in enumerate(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS):
            regime_eligible = (
                batch.regime_unpredictability_eligibility[:, horizon_index] == 1.0
            )
            for symbol_index, symbol in enumerate(ROUND74_EVENT_SYMBOLS):
                symbol_mask = symbols == symbol
                selected_regime = symbol_mask & regime_eligible
                if np.any(selected_regime):
                    key = (symbol, horizon)
                    accumulator = self.regime.setdefault(
                        key,
                        _BinaryAccumulator(),
                    )
                    accumulator.update(
                        batch.regime_unpredictability[
                            selected_regime,
                            horizon_index,
                        ]
                        > 0.5,
                        unpredictable_probability[
                            selected_regime,
                            horizon_index,
                        ],
                        run_ids=run_ids[selected_regime],
                    )
                    self.eligible_regime_targets += int(selected_regime.sum())
                for side_index, side in enumerate(ROUND74_EVENT_PAYOFF_SIDES):
                    action_eligible = (
                        batch.action_eligibility[
                            :,
                            horizon_index,
                            side_index,
                        ]
                        == 1.0
                    )
                    base = symbol_mask & action_eligible
                    if not np.any(base):
                        continue
                    activity_masks = {
                        "predictable": (
                            base
                            & regime_eligible
                            & (
                                batch.regime_unpredictability[
                                    :,
                                    horizon_index,
                                ]
                                <= 0.5
                            )
                        ),
                        "unpredictable": (
                            base
                            & regime_eligible
                            & (
                                batch.regime_unpredictability[
                                    :,
                                    horizon_index,
                                ]
                                > 0.5
                            )
                        ),
                        "unavailable": base & ~regime_eligible,
                    }
                    for activity, mask in activity_masks.items():
                        if not np.any(mask):
                            continue
                        key = (symbol, horizon, side, activity)
                        accumulator = self.action.setdefault(
                            key,
                            _ActionAccumulator(),
                        )
                        accumulator.update(
                            payoff_target=batch.net_payoff_bps[
                                mask,
                                horizon_index,
                                side_index,
                            ],
                            payoff_forecast=payoff_forecast[
                                mask,
                                horizon_index,
                                side_index,
                                :,
                            ],
                            payoff_no_information_forecast=(
                                payoff_no_information[
                                    symbol_index,
                                    horizon_index,
                                    side_index,
                                ]
                            ),
                            mae_target=batch.maximum_adverse_excursion_bps[
                                mask,
                                horizon_index,
                                side_index,
                            ],
                            mae_forecast=mae_forecast[
                                mask,
                                horizon_index,
                                side_index,
                                :,
                            ],
                            mae_no_information_forecast=mae_no_information[
                                symbol_index,
                                horizon_index,
                                side_index,
                            ],
                            positive_probability=positive_probability[
                                mask,
                                horizon_index,
                                side_index,
                            ],
                            adverse_target=batch.adverse_selection[
                                mask,
                                horizon_index,
                                side_index,
                            ],
                            adverse_probability=adverse_probability[
                                mask,
                                horizon_index,
                                side_index,
                            ],
                            run_ids=run_ids[mask],
                        )
                        self.eligible_action_targets += int(mask.sum())

    def result(
        self,
        *,
        expected_run_ids: tuple[str, ...],
    ) -> tuple[Round74SealedPredictiveDiagnostics, Round74SealedPredictiveGate]:
        diagnostics = Round74SealedPredictiveDiagnostics(
            action_slices=tuple(
                self.action[key].result(
                    symbol=key[0],
                    horizon_seconds=key[1],
                    side=key[2],
                    activity_regime=key[3],
                )
                for key in sorted(self.action)
            ),
            regime_slices=tuple(
                Round74RegimeForecastSlice(
                    symbol=key[0],
                    horizon_seconds=key[1],
                    regime_unpredictability=self.regime[key].result(),
                )
                for key in sorted(self.regime)
            ),
            eligible_action_targets=self.eligible_action_targets,
            eligible_regime_targets=self.eligible_regime_targets,
        )
        diagnostics.validate()
        action_accumulators = tuple(self.action.values())
        task_skills = (
            _build_predictive_brier_skill(
                "positive_payoff",
                tuple(value.positive for value in action_accumulators),
                expected_run_ids=expected_run_ids,
                seed=ROUND74_SEALED_BOOTSTRAP_SEED + 101,
            ),
            _build_predictive_brier_skill(
                "adverse_selection",
                tuple(value.adverse for value in action_accumulators),
                expected_run_ids=expected_run_ids,
                seed=ROUND74_SEALED_BOOTSTRAP_SEED + 102,
            ),
            _build_predictive_brier_skill(
                "regime_unpredictability",
                tuple(self.regime.values()),
                expected_run_ids=expected_run_ids,
                seed=ROUND74_SEALED_BOOTSTRAP_SEED + 103,
            ),
            _build_predictive_quantile_skill(
                "net_payoff_quantiles",
                tuple(value.payoff for value in action_accumulators),
                expected_run_ids=expected_run_ids,
                seed=ROUND74_SEALED_BOOTSTRAP_SEED + 104,
            ),
            _build_predictive_quantile_skill(
                "maximum_adverse_excursion_quantiles",
                tuple(value.mae for value in action_accumulators),
                expected_run_ids=expected_run_ids,
                seed=ROUND74_SEALED_BOOTSTRAP_SEED + 105,
            ),
        )
        gate_reasons = tuple(
            f"{value.task}:{reason}"
            for value in task_skills
            for reason in value.gate_reasons
        )
        gate = Round74SealedPredictiveGate(
            task_skills=task_skills,
            gate_passed=not gate_reasons,
            gate_reasons=gate_reasons,
        )
        gate.validate()
        return diagnostics, gate


@dataclass(frozen=True)
class Round74RunBlockBootstrap:
    blocks: int
    draws: int
    seed: int
    point_mean_run_net_bps: float
    three_configuration_bonferroni_lower_mean_run_net_bps: float
    two_ai_model_bonferroni_lower_mean_run_net_bps: float
    one_sided_95_lower_mean_run_net_bps: float
    one_sided_95_upper_mean_run_net_bps: float
    mean_block_length_runs: int
    restart_probability: float
    resampling_method: str = "circular_stationary_bootstrap"
    optimization_population: str = "capture_run"

    def validate(self) -> None:
        values = (
            self.point_mean_run_net_bps,
            self.three_configuration_bonferroni_lower_mean_run_net_bps,
            self.two_ai_model_bonferroni_lower_mean_run_net_bps,
            self.one_sided_95_lower_mean_run_net_bps,
            self.one_sided_95_upper_mean_run_net_bps,
        )
        if (
            self.optimization_population not in ROUND74_SEALED_OPTIMIZATION_POPULATIONS
            or (
                self.optimization_population == "capture_run"
                and self.blocks != ROUND74_SEALED_TEST_RUNS
            )
            or (
                self.optimization_population == "eligible_target"
                and self.blocks < ROUND74_SEALED_TEST_RUNS
            )
            or self.draws != ROUND74_SEALED_BOOTSTRAP_DRAWS
            or self.seed < 0
            or isinstance(self.mean_block_length_runs, bool)
            or not isinstance(self.mean_block_length_runs, int)
            or self.mean_block_length_runs
            != _stationary_bootstrap_mean_block_length(self.blocks)
            or not math.isclose(
                self.restart_probability,
                1.0 / self.mean_block_length_runs,
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
            or self.resampling_method != "circular_stationary_bootstrap"
            or any(not math.isfinite(float(value)) for value in values)
            or not self.one_sided_95_lower_mean_run_net_bps
            <= self.one_sided_95_upper_mean_run_net_bps
            or not self.three_configuration_bonferroni_lower_mean_run_net_bps
            <= self.two_ai_model_bonferroni_lower_mean_run_net_bps
            <= self.one_sided_95_lower_mean_run_net_bps
        ):
            raise ValueError("Round 74 sealed bootstrap differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            **self.__dict__,
            "familywise_alpha": ROUND74_SEALED_FAMILYWISE_ALPHA,
            "qualification_configuration_count": (
                ROUND74_SEALED_QUALIFICATION_CONFIGURATION_COUNT
            ),
            "paired_ai_model_count": ROUND74_SEALED_AI_MODEL_COUNT,
            "resampling_unit": "whole_capture_run",
            "block_length_policy": "ceil_sqrt_expected_capture_run_count",
            "chronological_dependence_preserved": True,
            "iid_capture_run_resampling_permitted": False,
            "circular_wraparound": True,
        }


@dataclass(frozen=True)
class Round74SealedStrategyMetrics:
    paired_observations: int
    selected_action_target_ineligible: int
    executed_trades: int
    active_runs: int
    distinct_symbols: int
    total_net_bps: float
    mean_paired_net_bps: float
    mean_executed_trade_net_bps: float
    median_executed_trade_net_bps: float
    win_rate: float
    profit_factor: float | None
    maximum_drawdown_bps: float
    realized_maximum_drawdown_bps: float
    maximum_concurrent_adverse_excursion_bps: float
    gross_profit_bps: float
    gross_loss_bps: float
    expected_shortfall_95_bps: float
    mean_maximum_adverse_excursion_bps: float
    adverse_selection_rate: float
    profitable_run_ratio: float
    maximum_symbol_trade_share: float
    optimization_population: str
    policy_selection_runs: int
    run_block_bootstrap: Round74RunBlockBootstrap
    financial_gate_passed: bool
    gate_reasons: tuple[str, ...]

    def validate(self) -> None:
        self.run_block_bootstrap.validate()
        valid_policy_runs = (
            self.policy_selection_runs == 6
            if self.optimization_population == "capture_run"
            else self.policy_selection_runs >= 6
        )
        finite = (
            self.total_net_bps,
            self.mean_paired_net_bps,
            self.mean_executed_trade_net_bps,
            self.median_executed_trade_net_bps,
            self.win_rate,
            self.maximum_drawdown_bps,
            self.realized_maximum_drawdown_bps,
            self.maximum_concurrent_adverse_excursion_bps,
            self.gross_profit_bps,
            self.gross_loss_bps,
            self.expected_shortfall_95_bps,
            self.mean_maximum_adverse_excursion_bps,
            self.adverse_selection_rate,
            self.profitable_run_ratio,
            self.maximum_symbol_trade_share,
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (
                    self.paired_observations,
                    self.selected_action_target_ineligible,
                    self.executed_trades,
                    self.active_runs,
                    self.distinct_symbols,
                )
            )
            or self.executed_trades > self.paired_observations
            or self.optimization_population
            not in ROUND74_SEALED_OPTIMIZATION_POPULATIONS
            or isinstance(self.policy_selection_runs, bool)
            or not isinstance(self.policy_selection_runs, int)
            or not valid_policy_runs
            or self.run_block_bootstrap.optimization_population
            != self.optimization_population
            or self.active_runs > self.run_block_bootstrap.blocks
            or self.distinct_symbols > len(ROUND74_EVENT_SYMBOLS)
            or any(not math.isfinite(float(value)) for value in finite)
            or any(
                not 0.0 <= float(value) <= 1.0
                for value in (
                    self.win_rate,
                    self.adverse_selection_rate,
                    self.profitable_run_ratio,
                    self.maximum_symbol_trade_share,
                )
            )
            or min(
                self.maximum_drawdown_bps,
                self.realized_maximum_drawdown_bps,
                self.maximum_concurrent_adverse_excursion_bps,
                self.gross_profit_bps,
                self.gross_loss_bps,
                self.mean_maximum_adverse_excursion_bps,
            )
            < 0.0
            or self.maximum_drawdown_bps + 1e-12 < self.realized_maximum_drawdown_bps
            or self.maximum_drawdown_bps + 1e-12
            < self.maximum_concurrent_adverse_excursion_bps
            or (
                self.profit_factor is not None
                and (
                    not math.isfinite(float(self.profit_factor))
                    or self.profit_factor < 0.0
                )
            )
            or self.financial_gate_passed == bool(self.gate_reasons)
        ):
            raise ValueError("Round 74 sealed strategy metrics differ")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            **{
                key: value
                for key, value in self.__dict__.items()
                if key != "run_block_bootstrap"
            },
            "run_block_bootstrap": self.run_block_bootstrap.as_dict(),
            "gate_reasons": list(self.gate_reasons),
            "annualized_return_reported": False,
            "sharpe_sortino_calmar_reported": False,
            "reason": (
                "the fixed unlevered sleeve path and sealed horizon are too "
                "short for defensible annualization"
            ),
        }


@dataclass(frozen=True)
class Round74SealedPairedRunDelta:
    run_id: str
    paired_observations: int
    baseline_net_bps: float
    ai_net_bps: float
    delta_net_bps: float

    def validate(self) -> None:
        values = (
            self.baseline_net_bps,
            self.ai_net_bps,
            self.delta_net_bps,
        )
        if (
            _RUN_ID.fullmatch(self.run_id) is None
            or isinstance(self.paired_observations, bool)
            or not isinstance(self.paired_observations, int)
            or self.paired_observations < 0
            or any(not math.isfinite(float(value)) for value in values)
            or not math.isclose(
                self.delta_net_bps,
                self.ai_net_bps - self.baseline_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Round 74 sealed paired run delta differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return dict(self.__dict__)


@dataclass(frozen=True)
class Round74SealedPairedSymbolHorizonDelta:
    symbol: str
    horizon_seconds: int
    paired_observations: int
    baseline_net_bps: float
    ai_net_bps: float
    delta_net_bps: float

    def validate(self) -> None:
        values = (
            self.baseline_net_bps,
            self.ai_net_bps,
            self.delta_net_bps,
        )
        if (
            self.symbol not in ROUND74_EVENT_SYMBOLS
            or isinstance(self.horizon_seconds, bool)
            or not isinstance(self.horizon_seconds, int)
            or self.horizon_seconds not in ROUND74_SEALED_AI_REVIEW_HORIZONS_SECONDS
            or isinstance(self.paired_observations, bool)
            or not isinstance(self.paired_observations, int)
            or self.paired_observations <= 0
            or any(not math.isfinite(float(value)) for value in values)
            or not math.isclose(
                self.delta_net_bps,
                self.ai_net_bps - self.baseline_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Round 74 sealed paired symbol-horizon delta differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return dict(self.__dict__)


@dataclass(frozen=True)
class Round74SealedAIOverlay:
    model_manifest_sha256: str
    review_sha256: tuple[str, ...]
    execution_replay_sha256: tuple[str, ...]
    reviewed_candidates: int
    runtime_accepted_reviews: int
    runtime_success_rate: float
    action_latency_eligible_reviews: int
    action_latency_eligibility_rate: float
    exact_replay_required_reviews: int
    exact_replay_completed_reviews: int
    exact_replay_target_ineligible_reviews: int
    delayed_overlap_vetoes: int
    retained_trades: int
    reduced_trades: int
    vetoed_trades: int
    retained_trade_ratio: float
    strategy_metrics: Round74SealedStrategyMetrics
    paired_delta_bootstrap: Round74RunBlockBootstrap
    paired_runs: tuple[Round74SealedPairedRunDelta, ...]
    paired_symbol_horizons: tuple[
        Round74SealedPairedSymbolHorizonDelta,
        ...,
    ]
    uplift_gate_passed: bool
    gate_reasons: tuple[str, ...]

    def validate(self) -> None:
        self.strategy_metrics.validate()
        self.paired_delta_bootstrap.validate()
        for value in self.paired_runs:
            value.validate()
        for value in self.paired_symbol_horizons:
            value.validate()
        run_ids = tuple(value.run_id for value in self.paired_runs)
        symbol_horizon_keys = tuple(
            (value.symbol, value.horizon_seconds)
            for value in self.paired_symbol_horizons
        )
        paired_run_baseline = math.fsum(
            value.baseline_net_bps for value in self.paired_runs
        )
        paired_run_ai = math.fsum(value.ai_net_bps for value in self.paired_runs)
        paired_group_baseline = math.fsum(
            value.baseline_net_bps for value in self.paired_symbol_horizons
        )
        paired_group_ai = math.fsum(
            value.ai_net_bps for value in self.paired_symbol_horizons
        )
        if (
            _SHA256.fullmatch(self.model_manifest_sha256) is None
            or len(self.review_sha256) != self.reviewed_candidates
            or len(set(self.review_sha256)) != len(self.review_sha256)
            or any(_SHA256.fullmatch(value) is None for value in self.review_sha256)
            or len(self.execution_replay_sha256)
            != self.strategy_metrics.paired_observations
            or len(set(self.execution_replay_sha256))
            != len(self.execution_replay_sha256)
            or any(
                _SHA256.fullmatch(value) is None
                for value in self.execution_replay_sha256
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (
                    self.reviewed_candidates,
                    self.runtime_accepted_reviews,
                    self.action_latency_eligible_reviews,
                    self.exact_replay_required_reviews,
                    self.exact_replay_completed_reviews,
                    self.exact_replay_target_ineligible_reviews,
                    self.delayed_overlap_vetoes,
                    self.retained_trades,
                    self.reduced_trades,
                    self.vetoed_trades,
                )
            )
            or self.runtime_accepted_reviews > self.reviewed_candidates
            or self.action_latency_eligible_reviews > self.runtime_accepted_reviews
            or self.exact_replay_completed_reviews != self.exact_replay_required_reviews
            or self.exact_replay_required_reviews
            > self.strategy_metrics.paired_observations
            or self.exact_replay_target_ineligible_reviews
            > self.exact_replay_completed_reviews
            or self.delayed_overlap_vetoes > self.exact_replay_completed_reviews
            or self.retained_trades + self.vetoed_trades
            != self.strategy_metrics.paired_observations
            or self.reduced_trades > self.retained_trades
            or len(self.paired_runs) != self.paired_delta_bootstrap.blocks
            or len(set(run_ids)) != len(run_ids)
            or self.strategy_metrics.optimization_population
            != self.paired_delta_bootstrap.optimization_population
            or self.strategy_metrics.run_block_bootstrap.blocks
            != self.paired_delta_bootstrap.blocks
            or not self.paired_symbol_horizons
            or len(set(symbol_horizon_keys)) != len(symbol_horizon_keys)
            or sum(value.paired_observations for value in self.paired_runs)
            != self.strategy_metrics.paired_observations
            or sum(value.paired_observations for value in self.paired_symbol_horizons)
            != self.strategy_metrics.paired_observations
            or not math.isclose(
                paired_run_baseline,
                paired_group_baseline,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                paired_run_ai,
                self.strategy_metrics.total_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                paired_group_ai,
                self.strategy_metrics.total_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not 0.0 <= self.runtime_success_rate <= 1.0
            or not 0.0 <= self.action_latency_eligibility_rate <= 1.0
            or not 0.0 <= self.retained_trade_ratio <= 1.0
            or (
                self.reviewed_candidates > 0
                and (
                    not math.isclose(
                        self.runtime_success_rate,
                        self.runtime_accepted_reviews / self.reviewed_candidates,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        self.action_latency_eligibility_rate,
                        self.action_latency_eligible_reviews / self.reviewed_candidates,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
            )
            or self.uplift_gate_passed == bool(self.gate_reasons)
        ):
            raise ValueError("Round 74 sealed AI overlay differs")

    def validate_against_baseline(self, trace: Round74ActionTrace) -> None:
        self.validate()
        trace.validate()
        expected_group_keys = tuple(
            (symbol, horizon)
            for symbol in ROUND74_EVENT_SYMBOLS
            for horizon in ROUND74_SEALED_AI_REVIEW_HORIZONS_SECONDS
            if any(
                observed_symbol == symbol and observed_horizon == horizon
                for observed_symbol, observed_horizon in zip(
                    trace.symbol,
                    trace.horizon_seconds,
                    strict=True,
                )
            )
        )
        if (
            tuple(value.run_id for value in self.paired_runs) != trace.expected_run_ids
            or tuple(
                (value.symbol, value.horizon_seconds)
                for value in self.paired_symbol_horizons
            )
            != expected_group_keys
            or not math.isclose(
                math.fsum(value.baseline_net_bps for value in self.paired_runs),
                trace.metrics.total_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                math.fsum(
                    value.baseline_net_bps for value in self.paired_symbol_horizons
                ),
                trace.metrics.total_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Round 74 sealed AI baseline pairing differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "model_manifest_sha256": self.model_manifest_sha256,
            "review_sha256": list(self.review_sha256),
            "execution_replay_sha256": list(self.execution_replay_sha256),
            "reviewed_candidates": self.reviewed_candidates,
            "runtime_accepted_reviews": self.runtime_accepted_reviews,
            "runtime_success_rate": self.runtime_success_rate,
            "action_validity_maximum_ns": ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS,
            "action_latency_eligible_reviews": (self.action_latency_eligible_reviews),
            "action_latency_eligibility_rate": (self.action_latency_eligibility_rate),
            "exact_replay_required_reviews": self.exact_replay_required_reviews,
            "exact_replay_completed_reviews": self.exact_replay_completed_reviews,
            "exact_replay_target_ineligible_reviews": (
                self.exact_replay_target_ineligible_reviews
            ),
            "delayed_overlap_vetoes": self.delayed_overlap_vetoes,
            "retained_trades": self.retained_trades,
            "reduced_trades": self.reduced_trades,
            "vetoed_trades": self.vetoed_trades,
            "retained_trade_ratio": self.retained_trade_ratio,
            "strategy_metrics": self.strategy_metrics.as_dict(),
            "paired_delta_bootstrap": self.paired_delta_bootstrap.as_dict(),
            "paired_runs": [value.as_dict() for value in self.paired_runs],
            "paired_symbol_horizons": [
                value.as_dict() for value in self.paired_symbol_horizons
            ],
            "uplift_gate_passed": self.uplift_gate_passed,
            "gate_reasons": list(self.gate_reasons),
            "may_create_or_replace_ml_actions": False,
            "action_validity_policy": (
                "minimum_of_forecast_horizon_and_target_maximum_delayed_entry"
            ),
            "action_latency_includes_historical_queue_delay": True,
            "expired_action_policy": "paired_zero_exposure_not_observation_deletion",
            "latency_adjusted_replay_performed": True,
            "baseline_payoff_scaled_without_rewalking_book": False,
        }


def _qualified_configurations(
    predictive_gate: Round74SealedPredictiveGate,
    baseline_metrics: Round74SealedStrategyMetrics,
    ai_overlays: Sequence[Round74SealedAIOverlay],
) -> tuple[str, ...]:
    predictive_gate.validate()
    baseline_metrics.validate()
    if not predictive_gate.gate_passed:
        return ()
    result: list[str] = []
    if baseline_metrics.financial_gate_passed:
        result.append("ml_baseline")
    result.extend(
        f"ai:{value.model_manifest_sha256}"
        for value in ai_overlays
        if value.uplift_gate_passed
    )
    return tuple(result)


@dataclass(frozen=True)
class Round74SealedEvaluationReport:
    reserved_claim_sha256: str
    reservation_id: str
    test_access_sha256: str
    dataset_sha256: str
    pretest_policy_sha256: str
    pretest_model_sha256: str
    probability_calibration_sha256: str
    action_selection_sha256: str
    ai_pretest_qualification_sha256: str
    profile: str
    optimization_population: str
    test_batch_sha256: tuple[str, ...]
    model_output_sha256: tuple[str, ...]
    candidate_sha256: tuple[str, ...]
    inference_backend_kind: str
    inference_backend_device: str
    inference_backend_vendor: str
    inference_warning_count: int
    predictive_diagnostics: Round74SealedPredictiveDiagnostics
    predictive_gate: Round74SealedPredictiveGate
    baseline_trace: Round74ActionTrace
    baseline_metrics: Round74SealedStrategyMetrics
    ai_overlays: tuple[Round74SealedAIOverlay, ...]
    qualified_configuration: tuple[str, ...]
    result_outcome: str
    schema_version: str = ROUND74_SEALED_EVALUATION_SCHEMA_VERSION
    test_access_consumed: bool = True
    promotion_authority: bool = False
    trading_authority: bool = False
    profitability_claim: bool = False
    leverage_applied: bool = False

    def validate(self) -> None:
        self.predictive_diagnostics.validate()
        self.predictive_gate.validate()
        self.baseline_trace.validate()
        self.baseline_metrics.validate()
        for value in self.ai_overlays:
            value.validate_against_baseline(self.baseline_trace)
        digests = (
            self.reserved_claim_sha256,
            self.reservation_id,
            self.test_access_sha256,
            self.dataset_sha256,
            self.pretest_policy_sha256,
            self.pretest_model_sha256,
            self.probability_calibration_sha256,
            self.action_selection_sha256,
            self.ai_pretest_qualification_sha256,
            *self.test_batch_sha256,
            *self.model_output_sha256,
            *self.candidate_sha256,
        )
        passed = _qualified_configurations(
            self.predictive_gate,
            self.baseline_metrics,
            self.ai_overlays,
        )
        expected_outcome = (
            "candidate_passed_predeclared_gates"
            if passed
            else "candidate_failed_predeclared_gates"
        )
        if (
            self.schema_version != ROUND74_SEALED_EVALUATION_SCHEMA_VERSION
            or any(_SHA256.fullmatch(value) is None for value in digests)
            or not self.test_batch_sha256
            or len(self.ai_overlays) != ROUND74_SEALED_AI_MODEL_COUNT
            or len({value.model_manifest_sha256 for value in self.ai_overlays})
            != ROUND74_SEALED_AI_MODEL_COUNT
            or len(self.test_batch_sha256) != len(self.model_output_sha256)
            or len(self.test_batch_sha256) != len(self.candidate_sha256)
            or len(set(self.test_batch_sha256)) != len(self.test_batch_sha256)
            or len(set(self.candidate_sha256)) != len(self.candidate_sha256)
            or self.profile not in ("conservative", "regular", "aggressive")
            or self.optimization_population
            not in ROUND74_SEALED_OPTIMIZATION_POPULATIONS
            or self.baseline_metrics.optimization_population
            != self.optimization_population
            or len(self.baseline_trace.expected_run_ids)
            != self.baseline_metrics.run_block_bootstrap.blocks
            or any(
                overlay.strategy_metrics.optimization_population
                != self.optimization_population
                for overlay in self.ai_overlays
            )
            or any(
                overlay.strategy_metrics.policy_selection_runs
                != self.baseline_metrics.policy_selection_runs
                for overlay in self.ai_overlays
            )
            or not self.inference_backend_kind
            or not self.inference_backend_device
            or not self.inference_backend_vendor
            or isinstance(self.inference_warning_count, bool)
            or self.inference_warning_count < 0
            or tuple(value.model_manifest_sha256 for value in self.ai_overlays)
            != tuple(sorted(value.model_manifest_sha256 for value in self.ai_overlays))
            or self.qualified_configuration != passed
            or self.result_outcome != expected_outcome
            or any(
                (
                    not self.test_access_consumed,
                    self.promotion_authority,
                    self.trading_authority,
                    self.profitability_claim,
                    self.leverage_applied,
                )
            )
        ):
            raise ValueError("Round 74 sealed evaluation report differs")

    @property
    def report_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "reserved_claim_sha256": self.reserved_claim_sha256,
            "reservation_id": self.reservation_id,
            "test_access_sha256": self.test_access_sha256,
            "dataset_sha256": self.dataset_sha256,
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "pretest_model_sha256": self.pretest_model_sha256,
            "probability_calibration_sha256": (self.probability_calibration_sha256),
            "action_selection_sha256": self.action_selection_sha256,
            "ai_pretest_qualification_sha256": (self.ai_pretest_qualification_sha256),
            "profile": self.profile,
            "optimization_population": self.optimization_population,
            "test_batch_sha256": list(self.test_batch_sha256),
            "model_output_sha256": list(self.model_output_sha256),
            "candidate_sha256": list(self.candidate_sha256),
            "inference_backend": {
                "kind": self.inference_backend_kind,
                "device": self.inference_backend_device,
                "vendor": self.inference_backend_vendor,
                "warning_count": self.inference_warning_count,
            },
            "predictive_diagnostics": self.predictive_diagnostics.as_dict(),
            "predictive_gate": self.predictive_gate.as_dict(),
            "baseline_trace": self.baseline_trace.as_dict(),
            "baseline_metrics": self.baseline_metrics.as_dict(),
            "ai_overlays": [value.as_dict() for value in self.ai_overlays],
            "qualified_configuration": list(self.qualified_configuration),
            "result_outcome": self.result_outcome,
            "test_access_consumed": True,
            "promotion_authority": False,
            "trading_authority": False,
            "profitability_claim": False,
            "leverage_applied": False,
        }
        if include_sha256:
            value["report_sha256"] = _canonical_sha256(value)
        return value


@dataclass(frozen=True)
class Round74SealedEvaluationOutcome:
    report: Round74SealedEvaluationReport
    finalized_claim: Round74SealedEvaluationClaim

    def validate(self) -> None:
        self.report.validate()
        self.finalized_claim.validate()
        if (
            self.finalized_claim.status != "complete"
            or self.finalized_claim.result_outcome != self.report.result_outcome
            or self.finalized_claim.result_sha256 != self.report.report_sha256
            or self.finalized_claim.reservation_id != self.report.reservation_id
            or self.finalized_claim.dataset_sha256 != self.report.dataset_sha256
            or self.finalized_claim.test_access_sha256 != self.report.test_access_sha256
            or self.finalized_claim.optimization_population
            != self.report.optimization_population
        ):
            raise ValueError("Round 74 sealed evaluation outcome differs")


def _stationary_bootstrap_mean_block_length(blocks: int) -> int:
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 2:
        raise ValueError("Round 74 stationary bootstrap block count differs")
    return max(2, math.ceil(math.sqrt(blocks)))


def _stationary_bootstrap_means(
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
    mean_block_length: int,
) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float64)
    if (
        selected.ndim != 1
        or len(selected) < 2
        or not np.isfinite(selected).all()
        or isinstance(draws, bool)
        or not isinstance(draws, int)
        or draws < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(mean_block_length, bool)
        or not isinstance(mean_block_length, int)
        or not 2 <= mean_block_length <= len(selected)
    ):
        raise ValueError("Round 74 stationary bootstrap inputs differ")
    generator = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    restart_probability = 1.0 / mean_block_length
    completed = 0
    while completed < draws:
        rows = min(512, draws - completed)
        indexes = np.empty((rows, len(selected)), dtype=np.intp)
        indexes[:, 0] = generator.integers(
            0,
            len(selected),
            size=rows,
            endpoint=False,
        )
        restart = generator.random((rows, len(selected) - 1)) < restart_probability
        restart_at = generator.integers(
            0,
            len(selected),
            size=(rows, len(selected) - 1),
            endpoint=False,
        )
        for column in range(1, len(selected)):
            indexes[:, column] = np.where(
                restart[:, column - 1],
                restart_at[:, column - 1],
                (indexes[:, column - 1] + 1) % len(selected),
            )
        sampled[completed : completed + rows] = selected[indexes].mean(axis=1)
        completed += rows
    return sampled


def _run_bootstrap(
    run_ids: Sequence[str],
    values: np.ndarray,
    *,
    expected_run_ids: tuple[str, ...],
    seed: int,
    optimization_population: str = "capture_run",
) -> Round74RunBlockBootstrap:
    selected = np.asarray(values, dtype=np.float64)
    if selected.shape != (len(run_ids),) or not np.isfinite(selected).all():
        raise ValueError("Round 74 sealed bootstrap values differ")
    totals = np.zeros(len(expected_run_ids), dtype=np.float64)
    run_index = {run_id: index for index, run_id in enumerate(expected_run_ids)}
    for run_id, value in zip(run_ids, selected, strict=True):
        try:
            totals[run_index[run_id]] += value
        except KeyError as exc:
            raise ValueError("Round 74 sealed bootstrap run differs") from exc
    mean_block_length = _stationary_bootstrap_mean_block_length(len(totals))
    sampled = _stationary_bootstrap_means(
        totals,
        draws=ROUND74_SEALED_BOOTSTRAP_DRAWS,
        seed=seed,
        mean_block_length=mean_block_length,
    )
    result = Round74RunBlockBootstrap(
        blocks=len(totals),
        draws=ROUND74_SEALED_BOOTSTRAP_DRAWS,
        seed=seed,
        point_mean_run_net_bps=float(totals.mean()),
        three_configuration_bonferroni_lower_mean_run_net_bps=float(
            np.quantile(
                sampled,
                ROUND74_SEALED_FAMILYWISE_ALPHA
                / ROUND74_SEALED_QUALIFICATION_CONFIGURATION_COUNT,
            )
        ),
        two_ai_model_bonferroni_lower_mean_run_net_bps=float(
            np.quantile(
                sampled,
                ROUND74_SEALED_FAMILYWISE_ALPHA / ROUND74_SEALED_AI_MODEL_COUNT,
            )
        ),
        one_sided_95_lower_mean_run_net_bps=float(np.quantile(sampled, 0.05)),
        one_sided_95_upper_mean_run_net_bps=float(np.quantile(sampled, 0.95)),
        mean_block_length_runs=mean_block_length,
        restart_probability=1.0 / mean_block_length,
        optimization_population=optimization_population,
    )
    result.validate()
    return result


def _financial_gate_reasons(
    *,
    metrics: Round74SealedStrategyMetrics,
    profile: str,
) -> tuple[str, ...]:
    spec = round74_action_profile(profile)
    scale = metrics.run_block_bootstrap.blocks / metrics.policy_selection_runs
    reasons: list[str] = []
    if metrics.selected_action_target_ineligible > 0:
        reasons.append("selected_action_target_coverage_incomplete")
    if metrics.executed_trades < math.ceil(spec.minimum_trades * scale):
        reasons.append("scaled_minimum_trades_not_met")
    if metrics.active_runs < math.ceil(spec.minimum_active_runs * scale):
        reasons.append("scaled_minimum_active_runs_not_met")
    if metrics.distinct_symbols != len(ROUND74_EVENT_SYMBOLS):
        reasons.append("asset_diversification_not_met")
    if metrics.total_net_bps <= 0.0 or metrics.mean_paired_net_bps <= 0.0:
        reasons.append("positive_after_cost_payoff_not_met")
    if (
        metrics.run_block_bootstrap.three_configuration_bonferroni_lower_mean_run_net_bps
        <= 0.0
    ):
        reasons.append("positive_familywise_run_block_confidence_lower_bound_not_met")
    if metrics.profitable_run_ratio < spec.minimum_profitable_run_ratio:
        reasons.append("profitable_run_ratio_not_met")
    if metrics.gross_loss_bps > 0.0 and (
        metrics.profit_factor is None
        or metrics.profit_factor < spec.minimum_profit_factor
    ):
        reasons.append("profit_factor_not_met")
    drawdown_ratio = (
        metrics.maximum_drawdown_bps / metrics.gross_profit_bps
        if metrics.gross_profit_bps > 0.0
        else math.inf
    )
    if drawdown_ratio > spec.maximum_drawdown_to_gross_profit:
        reasons.append("drawdown_to_gross_profit_not_met")
    if metrics.adverse_selection_rate > spec.maximum_adverse_selection_rate:
        reasons.append("adverse_selection_rate_not_met")
    if metrics.maximum_symbol_trade_share > spec.maximum_symbol_trade_share:
        reasons.append("symbol_concentration_not_met")
    return tuple(reasons)


def _strategy_metrics_from_execution_values(
    trace: Round74ActionTrace,
    net_payoff_bps: np.ndarray,
    maximum_adverse_excursion_bps: np.ndarray,
    retained: np.ndarray,
    adverse_selection: np.ndarray,
    entry_monotonic_ns: Sequence[int],
    exit_monotonic_ns: Sequence[int],
    *,
    profile: str,
    seed: int,
    optimization_population: str = "capture_run",
    policy_selection_runs: int = 6,
) -> Round74SealedStrategyMetrics:
    scaled = np.asarray(net_payoff_bps, dtype=np.float64)
    scaled_mae = np.asarray(
        maximum_adverse_excursion_bps,
        dtype=np.float64,
    )
    retained_mask = np.asarray(retained, dtype=np.bool_)
    adverse = np.asarray(adverse_selection, dtype=np.bool_)
    entries = tuple(int(value) for value in entry_monotonic_ns)
    exits = tuple(int(value) for value in exit_monotonic_ns)
    shape = (trace.metrics.trades,)
    if (
        scaled.shape != shape
        or scaled_mae.shape != shape
        or retained_mask.shape != shape
        or adverse.shape != shape
        or len(exits) != trace.metrics.trades
        or not np.isfinite(scaled).all()
        or not np.isfinite(scaled_mae).all()
        or np.any(scaled_mae < 0.0)
        or np.any(scaled[~retained_mask] != 0.0)
        or np.any(scaled_mae[~retained_mask] != 0.0)
        or np.any(adverse[~retained_mask])
        or len(entries) != trace.metrics.trades
        or any(value < 0 for value in exits)
        or any(value < 0 for value in entries)
        or any(exit_value < entry for entry, exit_value in zip(entries, exits))
    ):
        raise ValueError("Round 74 sealed strategy execution values differ")
    executed = scaled[retained_mask]
    executed_mae = scaled_mae[retained_mask]
    executed_adverse = adverse[retained_mask]
    retained_runs = tuple(
        run_id for run_id, keep in zip(trace.run_id, retained_mask, strict=True) if keep
    )
    retained_symbols = tuple(
        symbol for symbol, keep in zip(trace.symbol, retained_mask, strict=True) if keep
    )
    gross_profit = float(executed[executed > 0.0].sum())
    gross_loss = float(-executed[executed < 0.0].sum())
    run_pnl = {run_id: 0.0 for run_id in trace.expected_run_ids}
    for run_id, value in zip(trace.run_id, scaled, strict=True):
        run_pnl[run_id] += float(value)
    if executed.size:
        tail_threshold = float(np.quantile(executed, 0.05))
        expected_shortfall = float(executed[executed <= tail_threshold].mean())
        maximum_symbol_share = max(
            retained_symbols.count(symbol) for symbol in ROUND74_EVENT_SYMBOLS
        ) / len(retained_symbols)
    else:
        expected_shortfall = 0.0
        maximum_symbol_share = 0.0
    bootstrap = _run_bootstrap(
        trace.run_id,
        scaled,
        expected_run_ids=trace.expected_run_ids,
        seed=seed,
        optimization_population=optimization_population,
    )
    realized_drawdown = round74_maximum_realized_drawdown_bps(
        scaled,
        run_ids=trace.run_id,
        exit_monotonic_ns=exits,
        expected_run_ids=trace.expected_run_ids,
    )
    concurrent_adverse_excursion = round74_maximum_concurrent_adverse_excursion_bps(
        scaled_mae,
        run_ids=trace.run_id,
        entry_monotonic_ns=entries,
        exit_monotonic_ns=exits,
        expected_run_ids=trace.expected_run_ids,
    )
    conservative_drawdown = round74_conservative_maximum_drawdown_bps(
        scaled,
        scaled_mae,
        run_ids=trace.run_id,
        entry_monotonic_ns=entries,
        exit_monotonic_ns=exits,
        expected_run_ids=trace.expected_run_ids,
    )
    provisional = Round74SealedStrategyMetrics(
        paired_observations=trace.metrics.trades,
        selected_action_target_ineligible=trace.skipped_target_ineligible,
        executed_trades=int(retained_mask.sum()),
        active_runs=len(set(retained_runs)),
        distinct_symbols=len(set(retained_symbols)),
        total_net_bps=float(scaled.sum()),
        mean_paired_net_bps=float(scaled.mean()) if scaled.size else 0.0,
        mean_executed_trade_net_bps=(float(executed.mean()) if executed.size else 0.0),
        median_executed_trade_net_bps=(
            float(np.median(executed)) if executed.size else 0.0
        ),
        win_rate=float(np.mean(executed > 0.0)) if executed.size else 0.0,
        profit_factor=(gross_profit / gross_loss if gross_loss > 0.0 else None),
        maximum_drawdown_bps=conservative_drawdown,
        realized_maximum_drawdown_bps=realized_drawdown,
        maximum_concurrent_adverse_excursion_bps=(concurrent_adverse_excursion),
        gross_profit_bps=gross_profit,
        gross_loss_bps=gross_loss,
        expected_shortfall_95_bps=expected_shortfall,
        mean_maximum_adverse_excursion_bps=(
            float(executed_mae.mean()) if executed.size else 0.0
        ),
        adverse_selection_rate=(
            float(executed_adverse.mean()) if executed.size else 0.0
        ),
        profitable_run_ratio=float(np.mean(np.asarray(tuple(run_pnl.values())) > 0.0)),
        maximum_symbol_trade_share=float(maximum_symbol_share),
        optimization_population=optimization_population,
        policy_selection_runs=policy_selection_runs,
        run_block_bootstrap=bootstrap,
        financial_gate_passed=False,
        gate_reasons=("not_evaluated",),
    )
    reasons = _financial_gate_reasons(metrics=provisional, profile=profile)
    result = Round74SealedStrategyMetrics(
        **{
            **provisional.__dict__,
            "financial_gate_passed": not reasons,
            "gate_reasons": reasons,
        }
    )
    result.validate()
    return result


def _baseline_strategy_metrics(
    trace: Round74ActionTrace,
    *,
    profile: str,
    seed: int,
    optimization_population: str = "capture_run",
    policy_selection_runs: int = 6,
) -> Round74SealedStrategyMetrics:
    """Score the immutable baseline trace without execution substitution."""

    retained = np.ones(trace.metrics.trades, dtype=np.bool_)
    return _strategy_metrics_from_execution_values(
        trace,
        np.asarray(trace.net_payoff_bps, dtype=np.float64),
        np.asarray(
            trace.maximum_adverse_excursion_bps,
            dtype=np.float64,
        ),
        retained,
        np.asarray(trace.adverse_selection, dtype=np.bool_),
        trace.entry_monotonic_ns,
        trace.exit_monotonic_ns,
        profile=profile,
        seed=seed,
        optimization_population=optimization_population,
        policy_selection_runs=policy_selection_runs,
    )


def _exact_replay_strategy_metrics(
    trace: Round74ActionTrace,
    executions: Sequence[Round74AIExecutionReplayEvidence],
    *,
    profile: str,
    seed: int,
    optimization_population: str = "capture_run",
    policy_selection_runs: int = 6,
) -> Round74SealedStrategyMetrics:
    rows = tuple(executions)
    for row in rows:
        row.validate()
    if len(rows) != trace.metrics.trades:
        raise ValueError("Round 74 sealed AI execution coverage differs")
    retained = np.asarray(
        [row.status == "executed" for row in rows],
        dtype=np.bool_,
    )
    return _strategy_metrics_from_execution_values(
        trace,
        np.asarray(
            [
                trace.position_capital_fraction * row.capital_scaled_net_payoff_bps
                for row in rows
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                trace.position_capital_fraction
                * row.capital_scaled_maximum_adverse_excursion_bps
                for row in rows
            ],
            dtype=np.float64,
        ),
        retained,
        np.asarray([row.adverse_selection for row in rows], dtype=np.bool_),
        tuple(
            (
                row.actual_entry_monotonic_ns
                if row.actual_entry_monotonic_ns is not None
                else baseline_entry
            )
            for row, baseline_entry in zip(
                rows,
                trace.entry_monotonic_ns,
                strict=True,
            )
        ),
        tuple(
            (
                row.actual_exit_monotonic_ns
                if row.actual_exit_monotonic_ns is not None
                else baseline_exit
            )
            for row, baseline_exit in zip(
                rows,
                trace.exit_monotonic_ns,
                strict=True,
            )
        ),
        profile=profile,
        seed=seed,
        optimization_population=optimization_population,
        policy_selection_runs=policy_selection_runs,
    )


def _cpu_output(output: Round74EventModelOutput) -> Round74EventModelOutput:
    selected = Round74EventModelOutput(
        payoff_quantiles_bps=output.payoff_quantiles_bps.detach().cpu(),
        maximum_adverse_excursion_quantiles_bps=(
            output.maximum_adverse_excursion_quantiles_bps.detach().cpu()
        ),
        positive_payoff_logits=output.positive_payoff_logits.detach().cpu(),
        adverse_selection_logits=output.adverse_selection_logits.detach().cpu(),
        regime_unpredictability_logits=(
            output.regime_unpredictability_logits.detach().cpu()
        ),
    )
    selected.validate(int(selected.payoff_quantiles_bps.shape[0]))
    return selected


def _concat_outputs(
    outputs: Sequence[Round74EventModelOutput],
    *,
    rows: int,
) -> Round74EventModelOutput:
    selected = tuple(outputs)
    if not selected:
        raise ValueError("Round 74 sealed model output is missing")
    result = Round74EventModelOutput(
        payoff_quantiles_bps=torch.cat(
            tuple(value.payoff_quantiles_bps for value in selected),
            dim=0,
        ),
        maximum_adverse_excursion_quantiles_bps=torch.cat(
            tuple(value.maximum_adverse_excursion_quantiles_bps for value in selected),
            dim=0,
        ),
        positive_payoff_logits=torch.cat(
            tuple(value.positive_payoff_logits for value in selected),
            dim=0,
        ),
        adverse_selection_logits=torch.cat(
            tuple(value.adverse_selection_logits for value in selected),
            dim=0,
        ),
        regime_unpredictability_logits=torch.cat(
            tuple(value.regime_unpredictability_logits for value in selected),
            dim=0,
        ),
    )
    result.validate(rows)
    return result


@dataclass(frozen=True)
class Round74TargetFreeCandidateInference:
    """Verified model outputs and candidates with no realized-target access."""

    contexts: tuple[Round74ActionInferenceContext, ...]
    model_outputs: tuple[Round74EventModelOutput, ...]
    candidates: tuple[Round74ActionCandidateBatch, ...]
    pretest_policy_sha256: str
    pretest_model_sha256: str
    probability_calibration_sha256: str
    action_selection_sha256: str
    profile: str
    inference_backend_kind: str
    inference_backend_device: str
    inference_backend_vendor: str
    inference_warning_count: int
    optimization_population: str = "capture_run"
    data_scope: str = "sealed_test"
    expected_run_ids: tuple[str, ...] = ()
    schema_version: str = ROUND74_TARGET_FREE_INFERENCE_SCHEMA_VERSION
    target_fields_accessed: bool = False
    trading_authority: bool = False

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_TARGET_FREE_INFERENCE_SCHEMA_VERSION
            or not self.contexts
            or len(self.contexts) != len(self.model_outputs)
            or len(self.contexts) != len(self.candidates)
            or _SHA256.fullmatch(self.pretest_policy_sha256) is None
            or _SHA256.fullmatch(self.pretest_model_sha256) is None
            or _SHA256.fullmatch(self.probability_calibration_sha256) is None
            or _SHA256.fullmatch(self.action_selection_sha256) is None
            or self.profile not in ("conservative", "regular", "aggressive")
            or not self.inference_backend_kind
            or not self.inference_backend_device
            or not self.inference_backend_vendor
            or self.optimization_population
            not in ROUND74_SEALED_OPTIMIZATION_POPULATIONS
            or self.data_scope not in ROUND74_TARGET_FREE_INFERENCE_DATA_SCOPES
            or any(_RUN_ID.fullmatch(value) is None for value in self.expected_run_ids)
            or len(set(self.expected_run_ids)) != len(self.expected_run_ids)
            or isinstance(self.inference_warning_count, bool)
            or self.inference_warning_count < 0
            or self.target_fields_accessed
            or self.trading_authority
        ):
            raise ValueError("Round 74 target-free inference differs")
        first = self.contexts[0]
        run_ids: set[str] = set()
        ordered_run_ids: list[str] = []
        feature_rows: set[str] = set()
        prior_key: tuple[int, str, int, int, int, str, int] | None = None
        required_role = "test" if self.data_scope == "sealed_test" else "tuning"
        for context, output, candidates in zip(
            self.contexts,
            self.model_outputs,
            self.candidates,
            strict=True,
        ):
            context.validate()
            output.validate(context.rows)
            candidates.validate()
            first_key = (
                int(context.decision_wall_ns[0]),
                context.run_id[0],
                int(context.decision_monotonic_ns[0]),
                int(context.endpoint_frame_index[0]),
                int(context.endpoint_message_index[0]),
                context.symbol[0],
                int(context.anchor_index[0]),
            )
            last_index = context.rows - 1
            last_key = (
                int(context.decision_wall_ns[last_index]),
                context.run_id[last_index],
                int(context.decision_monotonic_ns[last_index]),
                int(context.endpoint_frame_index[last_index]),
                int(context.endpoint_message_index[last_index]),
                context.symbol[last_index],
                int(context.anchor_index[last_index]),
            )
            current_features = set(context.feature_row_sha256)
            if (
                context.role != required_role
                or context.partition_sha256 != first.partition_sha256
                or context.scaler_sha256 != first.scaler_sha256
                or candidates.context_sha256 != context.context_sha256
                or candidates.run_id != context.run_id
                or candidates.symbol != context.symbol
                or candidates.feature_row_sha256 != context.feature_row_sha256
                or candidates.model_output_sha256 != _model_output_sha256(output)
                or candidates.pretest_policy_sha256 != self.pretest_policy_sha256
                or candidates.probability_calibration_sha256
                != self.probability_calibration_sha256
                or candidates.profile != self.profile
                or len(current_features) != context.rows
                or feature_rows.intersection(current_features)
                or (prior_key is not None and first_key <= prior_key)
            ):
                raise ValueError("Round 74 target-free inference identity differs")
            feature_rows.update(current_features)
            for run_id in context.run_id:
                if run_id not in run_ids:
                    run_ids.add(run_id)
                    ordered_run_ids.append(run_id)
            prior_key = last_key
        expected_run_ids = (
            self.expected_run_ids if self.expected_run_ids else tuple(ordered_run_ids)
        )
        if (
            tuple(ordered_run_ids) != expected_run_ids
            or (
                self.data_scope == "sealed_test"
                and self.optimization_population == "capture_run"
                and len(run_ids) != ROUND74_SEALED_TEST_RUNS
            )
            or (
                self.data_scope == "sealed_test"
                and self.optimization_population == "eligible_target"
                and len(run_ids) < ROUND74_SEALED_TEST_RUNS
            )
            or (
                self.data_scope == "ai_qualification_tuning"
                and (
                    not self.expected_run_ids
                    or self.optimization_population != "eligible_target"
                )
            )
        ):
            raise ValueError("Round 74 target-free inference run coverage differs")

    @property
    def inference_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "pretest_model_sha256": self.pretest_model_sha256,
            "probability_calibration_sha256": (self.probability_calibration_sha256),
            "action_selection_sha256": self.action_selection_sha256,
            "profile": self.profile,
            "optimization_population": self.optimization_population,
            "data_scope": self.data_scope,
            "expected_run_ids": list(
                self.expected_run_ids
                if self.expected_run_ids
                else tuple(
                    dict.fromkeys(
                        run_id for context in self.contexts for run_id in context.run_id
                    )
                )
            ),
            "context_sha256": [context.context_sha256 for context in self.contexts],
            "model_output_sha256": [
                candidates.model_output_sha256 for candidates in self.candidates
            ],
            "candidate_sha256": [
                candidates.candidate_sha256 for candidates in self.candidates
            ],
            "inference_backend": {
                "kind": self.inference_backend_kind,
                "device": self.inference_backend_device,
                "vendor": self.inference_backend_vendor,
                "warning_count": self.inference_warning_count,
            },
            "target_fields_accessed": False,
            "trading_authority": False,
        }
        if include_sha256:
            value["inference_sha256"] = _canonical_sha256(value)
        return value


def infer_round74_target_free_candidates(
    contexts: Sequence[Round74ActionInferenceContext],
    *,
    action_selection: Round74ActionPolicySelection,
    probability_calibration: Round74ProbabilityCalibration,
    pretest_policy_path: str | Path,
    compute_backend: str = "auto",
    minibatch_rows: int = ROUND74_SEALED_INFERENCE_MINIBATCH_ROWS,
    data_scope: str = "sealed_test",
    expected_run_ids: Sequence[str] | None = None,
) -> Round74TargetFreeCandidateInference:
    """Run the frozen model on one explicitly scoped target-free population."""

    selected_contexts = tuple(contexts)
    selected_scope = str(data_scope)
    selected_expected_run_ids = (
        tuple(expected_run_ids) if expected_run_ids is not None else ()
    )
    if isinstance(minibatch_rows, bool) or minibatch_rows < 1:
        raise ValueError("Round 74 sealed inference minibatch differs")
    if not selected_contexts:
        raise ValueError("Round 74 target-free inference contexts are missing")
    for context in selected_contexts:
        context.validate()
    observed_run_ids = tuple(
        dict.fromkeys(
            run_id for context in selected_contexts for run_id in context.run_id
        )
    )
    required_role = "test" if selected_scope == "sealed_test" else "tuning"
    if (
        selected_scope not in ROUND74_TARGET_FREE_INFERENCE_DATA_SCOPES
        or any(context.role != required_role for context in selected_contexts)
        or (
            selected_scope == "ai_qualification_tuning"
            and (
                not selected_expected_run_ids
                or selected_expected_run_ids != observed_run_ids
            )
        )
        or (
            selected_scope == "sealed_test"
            and selected_expected_run_ids
            and selected_expected_run_ids != observed_run_ids
        )
    ):
        raise ValueError("Round 74 target-free inference scope differs")
    action_selection.validate()
    if (
        not action_selection.accepted
        or action_selection.selected_threshold_score is None
    ):
        raise ValueError("Round 74 target-free inference policy is not accepted")
    model, policy = load_round74_pretest_policy(Path(pretest_policy_path))
    policy_sha256 = _require_sha256(policy["policy_sha256"], "pretest policy")
    model_artifact = policy.get("model_artifact")
    development = policy.get("development_data")
    if not isinstance(model_artifact, Mapping) or not isinstance(
        development,
        Mapping,
    ):
        raise ValueError("Round 74 sealed pretest policy sections differ")
    model_sha256 = _require_sha256(model_artifact.get("sha256"), "pretest model")
    probability_calibration.validate()
    if (
        policy_sha256 != action_selection.pretest_policy_sha256
        or probability_calibration.pretest_policy_sha256 != policy_sha256
        or probability_calibration.calibration_sha256
        != action_selection.probability_calibration_sha256
        or probability_calibration.optimization_population
        != action_selection.optimization_population
        or development.get("partition_sha256") != selected_contexts[0].partition_sha256
        or development.get("scaler_sha256") != selected_contexts[0].scaler_sha256
        or {context.window_representation for context in selected_contexts}
        != {development.get("window_representation")}
    ):
        raise ValueError("Round 74 sealed pretest identity differs")
    backend = require_backend(resolve_backend(compute_backend))
    device = torch_device_for_backend(backend)
    candidates: list[Round74ActionCandidateBatch] = []
    model_outputs: list[Round74EventModelOutput] = []
    warning_messages: list[str] = []
    try:
        model = model.to(device)
        model.eval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.inference_mode():
                for context in selected_contexts:
                    outputs: list[Round74EventModelOutput] = []
                    for start in range(0, context.rows, minibatch_rows):
                        stop = min(context.rows, start + minibatch_rows)
                        feature_copy = np.array(
                            context.feature_values[start:stop],
                            dtype=np.float32,
                            copy=True,
                            order="C",
                        )
                        values = torch.from_numpy(feature_copy).to(device)
                        outputs.append(_cpu_output(model(values)))
                        del values
                    output = _concat_outputs(outputs, rows=context.rows)
                    candidate = derive_round74_action_candidates(
                        output,
                        context,
                        probability_calibration,
                        pretest_policy_sha256=policy_sha256,
                        profile=action_selection.profile,
                    )
                    model_outputs.append(output)
                    candidates.append(candidate)
            warning_messages.extend(str(value.message) for value in caught)
    finally:
        model.to("cpu")
    fallback_messages = tuple(
        message
        for message in warning_messages
        if "not currently supported on the DML backend" in message
        or "fall back to run on the CPU" in message
    )
    if fallback_messages:
        raise RuntimeError(
            f"Round 74 sealed inference used CPU fallback: {fallback_messages}"
        )
    result = Round74TargetFreeCandidateInference(
        contexts=selected_contexts,
        model_outputs=tuple(model_outputs),
        candidates=tuple(candidates),
        pretest_policy_sha256=policy_sha256,
        pretest_model_sha256=model_sha256,
        probability_calibration_sha256=(probability_calibration.calibration_sha256),
        action_selection_sha256=action_selection.selection_sha256,
        profile=action_selection.profile,
        inference_backend_kind=backend.kind,
        inference_backend_device=str(device),
        inference_backend_vendor=backend.vendor,
        inference_warning_count=len(warning_messages),
        optimization_population=action_selection.optimization_population,
        data_scope=selected_scope,
        expected_run_ids=(
            selected_expected_run_ids if selected_expected_run_ids else observed_run_ids
        ),
    )
    result.validate()
    return result


def _derive_test_candidates(
    batches: tuple[Round74EventTrainingBatch, ...],
    *,
    action_selection: Round74ActionPolicySelection,
    probability_calibration: Round74ProbabilityCalibration,
    pretest_policy_path: Path,
    compute_backend: str,
    minibatch_rows: int,
) -> tuple[
    Round74TargetFreeCandidateInference,
    Round74SealedPredictiveDiagnostics,
    Round74SealedPredictiveGate,
]:
    contexts = tuple(build_round74_action_inference_context(batch) for batch in batches)
    inference = infer_round74_target_free_candidates(
        contexts,
        action_selection=action_selection,
        probability_calibration=probability_calibration,
        pretest_policy_path=pretest_policy_path,
        compute_backend=compute_backend,
        minibatch_rows=minibatch_rows,
    )
    predictive = _PredictiveAccumulator()
    for batch, output in zip(batches, inference.model_outputs, strict=True):
        predictive.update(batch, output, probability_calibration)
    diagnostics, gate = predictive.result(
        expected_run_ids=inference.expected_run_ids,
    )
    return inference, diagnostics, gate


def _target_free_review_rows(
    candidates: Sequence[Round74ActionCandidateBatch],
    *,
    threshold_score: float,
) -> tuple[
    tuple[int, str, str, str, int, int],
    ...,
]:
    rows: list[tuple[int, str, str, str, int, int]] = []
    offset = 0
    for batch in candidates:
        for index in range(batch.rows):
            if batch.eligible[index] and batch.quality_score[index] >= threshold_score:
                rows.append(
                    (
                        offset + index,
                        batch.feature_row_sha256[index],
                        batch.run_id[index],
                        batch.symbol[index],
                        int(batch.side[index]),
                        int(batch.horizon_seconds[index]),
                    )
                )
        offset += batch.rows
    return tuple(rows)


def _validate_ai_reviews(
    ai_reviews_by_manifest: Mapping[
        str,
        Sequence[Round74AIPairedReviewEvidence],
    ],
    *,
    manifests: tuple[str, ...],
    expected_rows: tuple[tuple[int, str, str, str, int, int], ...],
    action_selection: Round74ActionPolicySelection,
) -> dict[str, tuple[Round74AIPairedReviewEvidence, ...]]:
    normalized = {
        _require_sha256(key, "AI manifest"): tuple(value)
        for key, value in ai_reviews_by_manifest.items()
    }
    if set(normalized) != set(manifests):
        raise ValueError("Round 74 sealed AI review panel differs")
    expected_index = tuple(value[0] for value in expected_rows)
    for manifest, reviews in normalized.items():
        for review in reviews:
            review.validate()
        if tuple(value.row_index for value in reviews) != expected_index or len(
            {value.review_sha256 for value in reviews}
        ) != len(reviews):
            raise ValueError("Round 74 sealed AI review coverage differs")
        for review, expected in zip(reviews, expected_rows, strict=True):
            if (
                review.model_manifest_sha256 != manifest
                or (
                    review.row_index,
                    review.feature_row_sha256,
                    review.run_id,
                    review.symbol,
                    review.side,
                    review.horizon_seconds,
                )
                != expected
                or review.pretest_policy_sha256
                != action_selection.pretest_policy_sha256
                or review.probability_calibration_sha256
                != action_selection.probability_calibration_sha256
            ):
                raise ValueError("Round 74 sealed AI review identity differs")
    return normalized


def _validate_ai_execution_replays(
    ai_execution_replays_by_manifest: Mapping[
        str,
        Sequence[Round74AIExecutionReplayEvidence],
    ],
    *,
    manifests: tuple[str, ...],
    instructions_by_manifest: Mapping[
        str,
        Sequence[Round74AIExecutionReplayInstruction],
    ],
) -> dict[str, tuple[Round74AIExecutionReplayEvidence, ...]]:
    normalized = {
        _require_sha256(key, "AI manifest"): tuple(value)
        for key, value in ai_execution_replays_by_manifest.items()
    }
    if set(normalized) != set(manifests):
        raise ValueError("Round 74 sealed AI execution replay panel differs")
    if set(instructions_by_manifest) != set(manifests):
        raise ValueError("Round 74 sealed AI execution instruction panel differs")
    for manifest, rows in normalized.items():
        instructions = tuple(instructions_by_manifest[manifest])
        for row in rows:
            row.validate()
        for instruction in instructions:
            instruction.validate()
        if len({row.replay_sha256 for row in rows}) != len(rows) or len(rows) != len(
            instructions
        ):
            raise ValueError("Round 74 sealed AI execution replay is duplicated")
        for row, instruction in zip(rows, instructions, strict=True):
            permitted_statuses = (
                {"target_ineligible", "delayed_overlap_veto", "executed"}
                if instruction.pre_replay_status == "replay_required"
                else {instruction.pre_replay_status}
            )
            if (
                row.row_index,
                row.feature_row_sha256,
                row.run_id,
                row.symbol,
                row.side,
                row.horizon_seconds,
                row.source_review_sha256,
                row.partition_sha256,
                row.requested_size_multiplier_bps,
            ) != (
                instruction.row_index,
                instruction.feature_row_sha256,
                instruction.run_id,
                instruction.symbol,
                instruction.side,
                instruction.horizon_seconds,
                instruction.source_review_sha256,
                instruction.partition_sha256,
                instruction.requested_size_multiplier_bps,
            ) or row.status not in permitted_statuses:
                raise ValueError("Round 74 sealed AI execution instruction differs")
    return normalized


def _paired_ai_delta_panels(
    trace: Round74ActionTrace,
    exact_values: np.ndarray,
) -> tuple[
    tuple[Round74SealedPairedRunDelta, ...],
    tuple[Round74SealedPairedSymbolHorizonDelta, ...],
]:
    baseline = np.asarray(trace.net_payoff_bps, dtype=np.float64)
    exact = np.asarray(exact_values, dtype=np.float64)
    if (
        baseline.shape != (trace.metrics.trades,)
        or exact.shape != baseline.shape
        or not np.isfinite(exact).all()
    ):
        raise ValueError("Round 74 sealed paired AI values differ")
    run_baseline = {run_id: 0.0 for run_id in trace.expected_run_ids}
    run_ai = {run_id: 0.0 for run_id in trace.expected_run_ids}
    run_observations = {run_id: 0 for run_id in trace.expected_run_ids}
    for run_id, baseline_value, ai_value in zip(
        trace.run_id,
        baseline,
        exact,
        strict=True,
    ):
        run_baseline[run_id] += float(baseline_value)
        run_ai[run_id] += float(ai_value)
        run_observations[run_id] += 1
    paired_runs = tuple(
        Round74SealedPairedRunDelta(
            run_id=run_id,
            paired_observations=run_observations[run_id],
            baseline_net_bps=run_baseline[run_id],
            ai_net_bps=run_ai[run_id],
            delta_net_bps=run_ai[run_id] - run_baseline[run_id],
        )
        for run_id in trace.expected_run_ids
    )
    symbols = np.asarray(trace.symbol, dtype=object)
    horizons = np.asarray(trace.horizon_seconds, dtype=np.int64)
    paired_symbol_horizons: list[Round74SealedPairedSymbolHorizonDelta] = []
    for symbol in ROUND74_EVENT_SYMBOLS:
        for horizon in ROUND74_SEALED_AI_REVIEW_HORIZONS_SECONDS:
            mask = (symbols == symbol) & (horizons == horizon)
            observations = int(mask.sum())
            if observations == 0:
                continue
            baseline_value = float(baseline[mask].sum())
            ai_value = float(exact[mask].sum())
            paired_symbol_horizons.append(
                Round74SealedPairedSymbolHorizonDelta(
                    symbol=symbol,
                    horizon_seconds=horizon,
                    paired_observations=observations,
                    baseline_net_bps=baseline_value,
                    ai_net_bps=ai_value,
                    delta_net_bps=ai_value - baseline_value,
                )
            )
    for value in (*paired_runs, *paired_symbol_horizons):
        value.validate()
    return paired_runs, tuple(paired_symbol_horizons)


def _ai_overlay(
    trace: Round74ActionTrace,
    all_reviews: tuple[Round74AIPairedReviewEvidence, ...],
    executions: tuple[Round74AIExecutionReplayEvidence, ...],
    *,
    manifest: str,
    expected_partition_sha256: str,
    profile: str,
    seed: int,
    optimization_population: str = "capture_run",
    policy_selection_runs: int = 6,
) -> Round74SealedAIOverlay:
    by_row = {value.row_index: value for value in all_reviews}
    try:
        reviews = tuple(by_row[row_index] for row_index in trace.row_index)
    except KeyError as exc:
        raise ValueError("Round 74 sealed AI trace review is missing") from exc
    if (
        tuple(value.row_index for value in executions) != trace.row_index
        or len({value.row_index for value in executions}) != len(executions)
        or {value.partition_sha256 for value in executions}
        != {expected_partition_sha256}
        or len({value.target_spec_sha256 for value in executions}) != 1
    ):
        raise ValueError("Round 74 sealed AI execution trace coverage differs")
    capture_report_by_run: dict[str, str] = {}
    for index, (review, execution) in enumerate(zip(reviews, executions, strict=True)):
        requested_multiplier = (
            review.decision.size_multiplier_bps
            if review.runtime_status == "accepted" and review.decision is not None
            else 0
        )
        previous_report = capture_report_by_run.setdefault(
            execution.run_id,
            execution.source_capture_report_sha256,
        )
        if (
            (
                execution.feature_row_sha256,
                execution.run_id,
                execution.symbol,
                execution.side,
                execution.horizon_seconds,
            )
            != (
                trace.feature_row_sha256[index],
                trace.run_id[index],
                trace.symbol[index],
                trace.side[index],
                trace.horizon_seconds[index],
            )
            or execution.source_review_sha256 != review.review_sha256
            or execution.requested_size_multiplier_bps != requested_multiplier
            or previous_report != execution.source_capture_report_sha256
            or (
                review.runtime_status != "accepted"
                and execution.status != "runtime_veto"
            )
            or (
                review.runtime_status == "accepted"
                and requested_multiplier == 0
                and execution.status != "ai_veto"
            )
            or (
                review.runtime_status == "accepted"
                and requested_multiplier > 0
                and review.effective_review_latency_ns
                > ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS
                and execution.status != "historical_review_expired"
            )
            or (
                review.runtime_status == "accepted"
                and requested_multiplier > 0
                and review.effective_review_latency_ns
                <= ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS
                and execution.status
                not in {
                    "target_ineligible",
                    "delayed_overlap_veto",
                    "executed",
                }
            )
        ):
            raise ValueError("Round 74 sealed AI execution identity differs")
    strategy = _exact_replay_strategy_metrics(
        trace,
        executions,
        profile=profile,
        seed=seed,
        optimization_population=optimization_population,
        policy_selection_runs=policy_selection_runs,
    )
    baseline_values = np.asarray(trace.net_payoff_bps, dtype=np.float64)
    exact_values = np.asarray(
        [
            trace.position_capital_fraction * value.capital_scaled_net_payoff_bps
            for value in executions
        ],
        dtype=np.float64,
    )
    paired_runs, paired_symbol_horizons = _paired_ai_delta_panels(
        trace,
        exact_values,
    )
    delta = exact_values - baseline_values
    delta_bootstrap = _run_bootstrap(
        trace.run_id,
        delta,
        expected_run_ids=trace.expected_run_ids,
        seed=seed + 500_000,
        optimization_population=optimization_population,
    )
    runtime_success_rate = (
        sum(value.runtime_status == "accepted" for value in all_reviews)
        / len(all_reviews)
        if all_reviews
        else 0.0
    )
    latency_eligible_reviews = sum(
        value.action_latency_eligible for value in all_reviews
    )
    latency_eligibility_rate = (
        latency_eligible_reviews / len(all_reviews) if all_reviews else 0.0
    )
    retained = sum(value.status == "executed" for value in executions)
    reduced = sum(
        value.status == "executed" and value.applied_size_multiplier_bps < 10_000
        for value in executions
    )
    reasons: list[str] = []
    if not strategy.financial_gate_passed:
        reasons.extend(f"financial:{value}" for value in strategy.gate_reasons)
    if runtime_success_rate < 0.99:
        reasons.append("runtime_success_rate_not_met")
    retained_ratio = retained / trace.metrics.trades if trace.metrics.trades else 0.0
    minimum_retained = {
        "conservative": 0.60,
        "regular": 0.50,
        "aggressive": 0.40,
    }[profile]
    if retained_ratio < minimum_retained:
        reasons.append("retained_trade_ratio_not_met")
    if strategy.total_net_bps <= trace.metrics.total_net_bps:
        reasons.append("positive_paired_after_cost_uplift_not_met")
    if delta_bootstrap.two_ai_model_bonferroni_lower_mean_run_net_bps <= 0.0:
        reasons.append(
            "positive_paired_delta_familywise_confidence_lower_bound_not_met"
        )
    if any(value.delta_net_bps < -1e-12 for value in paired_runs):
        reasons.append("paired_run_noninferiority_not_met")
    if any(value.delta_net_bps < -1e-12 for value in paired_symbol_horizons):
        reasons.append("paired_symbol_horizon_noninferiority_not_met")
    if strategy.maximum_drawdown_bps > trace.metrics.maximum_drawdown_bps:
        reasons.append("maximum_drawdown_noninferiority_not_met")
    result = Round74SealedAIOverlay(
        model_manifest_sha256=manifest,
        review_sha256=tuple(value.review_sha256 for value in all_reviews),
        execution_replay_sha256=tuple(value.replay_sha256 for value in executions),
        reviewed_candidates=len(all_reviews),
        runtime_accepted_reviews=sum(
            value.runtime_status == "accepted" for value in all_reviews
        ),
        runtime_success_rate=runtime_success_rate,
        action_latency_eligible_reviews=latency_eligible_reviews,
        action_latency_eligibility_rate=latency_eligibility_rate,
        exact_replay_required_reviews=sum(
            value.requested_size_multiplier_bps > 0
            and value.status
            not in {"runtime_veto", "ai_veto", "historical_review_expired"}
            for value in executions
        ),
        exact_replay_completed_reviews=sum(
            value.exact_l2_replay_performed for value in executions
        ),
        exact_replay_target_ineligible_reviews=sum(
            value.status == "target_ineligible" for value in executions
        ),
        delayed_overlap_vetoes=sum(
            value.status == "delayed_overlap_veto" for value in executions
        ),
        retained_trades=retained,
        reduced_trades=reduced,
        vetoed_trades=trace.metrics.trades - retained,
        retained_trade_ratio=retained_ratio,
        strategy_metrics=strategy,
        paired_delta_bootstrap=delta_bootstrap,
        paired_runs=paired_runs,
        paired_symbol_horizons=paired_symbol_horizons,
        uplift_gate_passed=not reasons,
        gate_reasons=tuple(reasons),
    )
    result.validate()
    return result


def _evaluate_reserved(
    claim: Round74SealedEvaluationClaim,
    *,
    ledger: Round74SealedEvaluationLedger,
    test_batches: tuple[Round74EventTrainingBatch, ...],
    action_selection: Round74ActionPolicySelection,
    probability_calibration: Round74ProbabilityCalibration,
    pretest_policy_path: Path,
    ai_review_provider: Round74SealedAIReviewProvider,
    ai_execution_replay_provider: Round74SealedAIExecutionReplayProvider,
    compute_backend: str,
    inference_minibatch_rows: int,
) -> Round74SealedEvaluationReport:
    if not ledger.claim_matches(claim, required_status="reserved"):
        raise ValueError("Round 74 sealed reservation is not live")
    policy_run_counts = {
        len(value.trace.expected_run_ids) for value in action_selection.evaluations
    }
    if (
        tuple(batch.batch_sha256 for batch in test_batches) != claim.batch_sha256
        or action_selection.selection_sha256 != claim.action_selection_sha256
        or action_selection.profile != claim.profile
        or action_selection.selected_threshold_score is None
        or action_selection.optimization_population != claim.optimization_population
        or probability_calibration.optimization_population
        != claim.optimization_population
        or not claim.ai_pretest_qualification_required
        or _SHA256.fullmatch(claim.ai_pretest_qualification_sha256) is None
        or len(policy_run_counts) != 1
    ):
        raise ValueError("Round 74 sealed reserved input identity differs")
    policy_selection_runs = next(iter(policy_run_counts))
    inference, predictive, predictive_gate = _derive_test_candidates(
        test_batches,
        action_selection=action_selection,
        probability_calibration=probability_calibration,
        pretest_policy_path=pretest_policy_path,
        compute_backend=compute_backend,
        minibatch_rows=inference_minibatch_rows,
    )
    candidates = inference.candidates
    contexts = inference.contexts
    threshold = action_selection.selected_threshold_score
    assert threshold is not None
    target_free_rows = _target_free_review_rows(
        candidates,
        threshold_score=threshold,
    )
    if not ledger.claim_matches(claim, required_status="reserved"):
        raise ValueError("Round 74 sealed reservation expired before AI review")
    ai_reviews_by_manifest = ai_review_provider(
        claim=claim,
        manifests=claim.ai_manifest_sha256,
        inference=inference,
        action_selection=action_selection,
    )
    reviews = _validate_ai_reviews(
        ai_reviews_by_manifest,
        manifests=claim.ai_manifest_sha256,
        expected_rows=target_free_rows,
        action_selection=action_selection,
    )
    trace = _simulate_round74_action_trace_batches(
        test_batches,
        candidates,
        threshold_score=threshold,
        expected_run_ids=claim.test_run_ids,
        required_role="test",
        expected_run_count=len(claim.test_run_ids),
    )
    baseline = _baseline_strategy_metrics(
        trace,
        profile=action_selection.profile,
        seed=ROUND74_SEALED_BOOTSTRAP_SEED,
        optimization_population=claim.optimization_population,
        policy_selection_runs=policy_selection_runs,
    )
    instructions_by_manifest: dict[
        str,
        tuple[Round74AIExecutionReplayInstruction, ...],
    ] = {}
    for manifest, manifest_reviews in reviews.items():
        by_row = {review.row_index: review for review in manifest_reviews}
        try:
            selected_reviews = tuple(by_row[row_index] for row_index in trace.row_index)
        except KeyError as exc:
            raise ValueError("Round 74 sealed AI execution review is missing") from exc
        instructions_by_manifest[manifest] = (
            build_round74_ai_execution_replay_instructions(
                action_selection,
                contexts=contexts,
                reviews=selected_reviews,
                trace=trace,
            )
        )
    if not ledger.claim_matches(claim, required_status="reserved"):
        raise ValueError("Round 74 sealed reservation expired before exact replay")
    ai_execution_replays_by_manifest = ai_execution_replay_provider(
        claim=claim,
        instructions_by_manifest=instructions_by_manifest,
    )
    executions = _validate_ai_execution_replays(
        ai_execution_replays_by_manifest,
        manifests=claim.ai_manifest_sha256,
        instructions_by_manifest=instructions_by_manifest,
    )
    partition_sha256 = {batch.partition_sha256 for batch in test_batches}
    if len(partition_sha256) != 1:
        raise ValueError("Round 74 sealed test partition differs")
    expected_partition_sha256 = next(iter(partition_sha256))
    overlays = tuple(
        _ai_overlay(
            trace,
            reviews[manifest],
            executions[manifest],
            manifest=manifest,
            expected_partition_sha256=expected_partition_sha256,
            profile=action_selection.profile,
            seed=ROUND74_SEALED_BOOTSTRAP_SEED + index + 1,
            optimization_population=claim.optimization_population,
            policy_selection_runs=policy_selection_runs,
        )
        for index, manifest in enumerate(sorted(reviews))
    )
    qualified = _qualified_configurations(
        predictive_gate,
        baseline,
        overlays,
    )
    report = Round74SealedEvaluationReport(
        reserved_claim_sha256=claim.claim_sha256,
        reservation_id=claim.reservation_id,
        test_access_sha256=claim.test_access_sha256,
        dataset_sha256=claim.dataset_sha256,
        pretest_policy_sha256=inference.pretest_policy_sha256,
        pretest_model_sha256=inference.pretest_model_sha256,
        probability_calibration_sha256=(probability_calibration.calibration_sha256),
        action_selection_sha256=action_selection.selection_sha256,
        ai_pretest_qualification_sha256=(claim.ai_pretest_qualification_sha256),
        profile=action_selection.profile,
        optimization_population=claim.optimization_population,
        test_batch_sha256=claim.batch_sha256,
        model_output_sha256=tuple(value.model_output_sha256 for value in candidates),
        candidate_sha256=tuple(value.candidate_sha256 for value in candidates),
        inference_backend_kind=inference.inference_backend_kind,
        inference_backend_device=inference.inference_backend_device,
        inference_backend_vendor=inference.inference_backend_vendor,
        inference_warning_count=inference.inference_warning_count,
        predictive_diagnostics=predictive,
        predictive_gate=predictive_gate,
        baseline_trace=trace,
        baseline_metrics=baseline,
        ai_overlays=overlays,
        qualified_configuration=qualified,
        result_outcome=(
            "candidate_passed_predeclared_gates"
            if qualified
            else "candidate_failed_predeclared_gates"
        ),
    )
    report.validate()
    return report


def evaluate_round74_sealed_once(
    test_identity: Round74SealedDatasetIdentity,
    *,
    test_batch_loader: Round74SealedTestBatchLoader,
    action_selection: Round74ActionPolicySelection,
    probability_calibration: Round74ProbabilityCalibration,
    pretest_policy_path: str | Path,
    ai_pretest_qualification: Round74AIPretestQualificationPanel,
    ai_review_provider: Round74SealedAIReviewProvider,
    ai_execution_replay_provider: Round74SealedAIExecutionReplayProvider,
    ledger: Round74SealedEvaluationLedger,
    compute_backend: str = "auto",
    inference_minibatch_rows: int = ROUND74_SEALED_INFERENCE_MINIBATCH_ROWS,
) -> Round74SealedEvaluationOutcome:
    """Reserve metadata, load targets, replay AI, and finalize exactly once."""

    ai_pretest_qualification.validate()
    manifests = ai_pretest_qualification.model_manifest_sha256
    if (
        len(manifests) != ROUND74_SEALED_AI_MODEL_COUNT
        or len(set(manifests)) != ROUND74_SEALED_AI_MODEL_COUNT
    ):
        raise ValueError(
            "Round 74 sealed evaluation requires the exact two-model AI family"
        )
    if not ai_pretest_qualification.qualification_passed:
        raise ValueError(
            "Round 74 sealed evaluation requires both AI models to pass pretest"
        )
    if (
        ai_pretest_qualification.action_selection_sha256
        != action_selection.selection_sha256
        or ai_pretest_qualification.pretest_policy_sha256
        != action_selection.pretest_policy_sha256
        or ai_pretest_qualification.probability_calibration_sha256
        != probability_calibration.calibration_sha256
        or ai_pretest_qualification.profile != action_selection.profile
    ):
        raise ValueError("Round 74 sealed AI pretest qualification identity differs")
    claim = ledger.reserve_identity(
        test_identity=test_identity,
        action_selection=action_selection,
        ai_pretest_qualification=ai_pretest_qualification,
    )
    try:
        if not ledger.claim_matches(claim, required_status="reserved"):
            raise ValueError("Round 74 sealed reservation is not live")
        batches = tuple(test_batch_loader(claim=claim))
        loaded_identity = build_round74_sealed_dataset_identity(
            batches,
            optimization_population=claim.optimization_population,
            expected_test_run_ids=claim.test_run_ids,
            test_population_sha256=claim.test_population_sha256,
        )
        if (
            loaded_identity.as_dict() != test_identity.as_dict()
            or loaded_identity.dataset_sha256 != claim.dataset_sha256
            or loaded_identity.test_access_sha256 != claim.test_access_sha256
            or loaded_identity.partition_sha256 != claim.partition_sha256
            or loaded_identity.scaler_sha256 != claim.scaler_sha256
            or loaded_identity.test_run_ids != claim.test_run_ids
            or loaded_identity.batch_sha256 != claim.batch_sha256
        ):
            raise ValueError("Round 74 sealed loaded test identity differs")
        report = _evaluate_reserved(
            claim,
            ledger=ledger,
            test_batches=batches,
            action_selection=action_selection,
            probability_calibration=probability_calibration,
            pretest_policy_path=Path(pretest_policy_path),
            ai_review_provider=ai_review_provider,
            ai_execution_replay_provider=ai_execution_replay_provider,
            compute_backend=compute_backend,
            inference_minibatch_rows=inference_minibatch_rows,
        )
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        error_sha256 = _canonical_sha256(
            {
                "schema_version": ROUND74_SEALED_EVALUATION_SCHEMA_VERSION,
                "reservation_id": claim.reservation_id,
                "outcome": "evaluation_error",
                "error": " ".join(error.split())[:2_000],
            }
        )
        ledger.finalize(
            claim.reservation_id,
            result_outcome="evaluation_error",
            result_sha256=error_sha256,
            error=error,
        )
        raise
    finalized = ledger.finalize(
        claim.reservation_id,
        result_outcome=report.result_outcome,
        result_sha256=report.report_sha256,
    )
    outcome = Round74SealedEvaluationOutcome(
        report=report,
        finalized_claim=finalized,
    )
    outcome.validate()
    return outcome


__all__ = [
    "ROUND74_SEALED_BOOTSTRAP_DRAWS",
    "ROUND74_SEALED_BOOTSTRAP_SEED",
    "ROUND74_SEALED_AI_MODEL_COUNT",
    "ROUND74_SEALED_AI_REVIEW_HORIZONS_SECONDS",
    "ROUND74_SEALED_EVALUATION_SCHEMA_VERSION",
    "ROUND74_SEALED_FAMILYWISE_ALPHA",
    "ROUND74_SEALED_INFERENCE_MINIBATCH_ROWS",
    "ROUND74_SEALED_QUALIFICATION_CONFIGURATION_COUNT",
    "ROUND74_TARGET_FREE_INFERENCE_DATA_SCOPES",
    "ROUND74_TARGET_FREE_INFERENCE_SCHEMA_VERSION",
    "Round74ActionForecastSlice",
    "Round74BinaryForecastMetrics",
    "Round74QuantileForecastMetrics",
    "Round74RegimeForecastSlice",
    "Round74RunBlockBootstrap",
    "Round74SealedAIOverlay",
    "Round74SealedAIExecutionReplayProvider",
    "Round74SealedAIReviewProvider",
    "Round74SealedEvaluationOutcome",
    "Round74SealedEvaluationReport",
    "Round74SealedPairedRunDelta",
    "Round74SealedPairedSymbolHorizonDelta",
    "Round74SealedPredictiveDiagnostics",
    "Round74SealedStrategyMetrics",
    "Round74SealedTestBatchLoader",
    "Round74TargetFreeCandidateInference",
    "evaluate_round74_sealed_once",
    "infer_round74_target_free_candidates",
]
