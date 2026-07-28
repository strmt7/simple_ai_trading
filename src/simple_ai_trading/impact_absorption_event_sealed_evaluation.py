"""One-use, output-bound Round 74 sealed ML and local-AI evaluation.

The public entry point reserves the immutable test identity before loading the
model, deriving candidates, or reading realized payoffs. AI review coverage is
defined on every target-free candidate above the frozen tuning threshold. The
financial replay then preserves the baseline ML action sequence; AI may only
retain, reduce, or veto those same observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
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
    Round74AIExecutionReplayEvidence,
    Round74AIPairedReviewEvidence,
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
)
from .impact_absorption_event_cohort import (
    ROUND74_EVENT_COHORT_DEFAULT_ROLE_COUNTS,
)
from .impact_absorption_event_dataset import Round74EventTrainingBatch
from .impact_absorption_event_financial_metrics import (
    round74_maximum_realized_drawdown_bps,
)
from .impact_absorption_event_model import Round74EventModelOutput
from .impact_absorption_event_sealed_ledger import (
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


ROUND74_SEALED_EVALUATION_SCHEMA_VERSION = "round-074-sealed-evaluation-v10"
ROUND74_TARGET_FREE_INFERENCE_SCHEMA_VERSION = (
    "round-074-target-free-candidate-inference-v1"
)
ROUND74_SEALED_BOOTSTRAP_DRAWS = 10_000
ROUND74_SEALED_BOOTSTRAP_SEED = 7_474_011
ROUND74_SEALED_INFERENCE_MINIBATCH_ROWS = 2_048
ROUND74_SEALED_ECE_BINS = 10
ROUND74_SEALED_TEST_RUNS = ROUND74_EVENT_COHORT_DEFAULT_ROLE_COUNTS["test"]
ROUND74_SEALED_FAMILYWISE_ALPHA = 0.05
ROUND74_SEALED_QUALIFICATION_CONFIGURATION_COUNT = 3
ROUND74_SEALED_AI_MODEL_COUNT = 2

_SHA256 = re.compile(r"[0-9a-f]{64}")
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
    empirical_coverage: tuple[float, ...]

    def validate(self) -> None:
        if (
            isinstance(self.observations, bool)
            or self.observations < 1
            or not math.isfinite(float(self.mean_pinball_loss_bps))
            or self.mean_pinball_loss_bps < 0.0
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

    def update(self, target: np.ndarray, probability: np.ndarray) -> None:
        truth = np.asarray(target, dtype=np.bool_)
        estimate = np.asarray(probability, dtype=np.float64)
        if (
            truth.ndim != 1
            or estimate.shape != truth.shape
            or not estimate.size
            or not np.isfinite(estimate).all()
            or np.any((estimate < 0.0) | (estimate > 1.0))
        ):
            raise ValueError("Round 74 sealed binary update differs")
        predicted = estimate >= 0.5
        self.count += int(truth.size)
        self.positive += int(truth.sum())
        self.brier_sum += float(np.square(estimate - truth.astype(float)).sum())
        self.tp += int(np.sum(predicted & truth))
        self.tn += int(np.sum(~predicted & ~truth))
        self.fp += int(np.sum(predicted & ~truth))
        self.fn += int(np.sum(~predicted & truth))
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
        self.coverage = np.zeros(
            len(ROUND74_EVENT_PAYOFF_QUANTILES),
            dtype=np.int64,
        )

    def update(self, target: np.ndarray, forecast: np.ndarray) -> None:
        truth = np.asarray(target, dtype=np.float64)
        estimate = np.asarray(forecast, dtype=np.float64)
        expected_shape = (truth.size, len(ROUND74_EVENT_PAYOFF_QUANTILES))
        if (
            truth.ndim != 1
            or not truth.size
            or estimate.shape != expected_shape
            or not np.isfinite(truth).all()
            or not np.isfinite(estimate).all()
            or np.any(np.diff(estimate, axis=1) < 0.0)
        ):
            raise ValueError("Round 74 sealed quantile update differs")
        quantiles = np.asarray(
            ROUND74_EVENT_PAYOFF_QUANTILES,
            dtype=np.float64,
        )
        error = truth[:, None] - estimate
        self.pinball_sum += float(
            np.maximum(quantiles * error, (quantiles - 1.0) * error).sum()
        )
        self.coverage += np.sum(truth[:, None] <= estimate, axis=0)
        self.count += int(truth.size)

    def result(self) -> Round74QuantileForecastMetrics:
        if self.count < 1:
            raise ValueError("Round 74 sealed quantile accumulator is empty")
        result = Round74QuantileForecastMetrics(
            observations=self.count,
            mean_pinball_loss_bps=(
                self.pinball_sum / self.count / len(ROUND74_EVENT_PAYOFF_QUANTILES)
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
        mae_target: np.ndarray,
        mae_forecast: np.ndarray,
        positive_probability: np.ndarray,
        adverse_target: np.ndarray,
        adverse_probability: np.ndarray,
    ) -> None:
        self.payoff.update(payoff_target, payoff_forecast)
        self.mae.update(mae_target, mae_forecast)
        self.positive.update(payoff_target > 0.0, positive_probability)
        self.adverse.update(adverse_target > 0.5, adverse_probability)

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
        positive, adverse, unpredictable = apply_round74_probability_calibration(
            calibration,
            positive_payoff_logits=output.positive_payoff_logits,
            adverse_selection_logits=output.adverse_selection_logits,
            regime_unpredictability_logits=output.regime_unpredictability_logits,
        )
        payoff_forecast = _tensor_array(output.payoff_quantiles_bps)
        mae_forecast = _tensor_array(output.maximum_adverse_excursion_quantiles_bps)
        positive_probability = _tensor_array(positive)
        adverse_probability = _tensor_array(adverse)
        unpredictable_probability = _tensor_array(unpredictable)
        symbols = np.asarray(batch.symbol, dtype=object)
        for horizon_index, horizon in enumerate(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS):
            regime_eligible = (
                batch.regime_unpredictability_eligibility[:, horizon_index] == 1.0
            )
            for symbol in ROUND74_EVENT_SYMBOLS:
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
                        )
                        self.eligible_action_targets += int(mask.sum())

    def result(self) -> Round74SealedPredictiveDiagnostics:
        result = Round74SealedPredictiveDiagnostics(
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
        result.validate()
        return result


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

    def validate(self) -> None:
        values = (
            self.point_mean_run_net_bps,
            self.three_configuration_bonferroni_lower_mean_run_net_bps,
            self.two_ai_model_bonferroni_lower_mean_run_net_bps,
            self.one_sided_95_lower_mean_run_net_bps,
            self.one_sided_95_upper_mean_run_net_bps,
        )
        if (
            self.blocks != ROUND74_SEALED_TEST_RUNS
            or self.draws != ROUND74_SEALED_BOOTSTRAP_DRAWS
            or self.seed < 0
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
    gross_profit_bps: float
    gross_loss_bps: float
    expected_shortfall_95_bps: float
    mean_maximum_adverse_excursion_bps: float
    adverse_selection_rate: float
    profitable_run_ratio: float
    maximum_symbol_trade_share: float
    run_block_bootstrap: Round74RunBlockBootstrap
    financial_gate_passed: bool
    gate_reasons: tuple[str, ...]

    def validate(self) -> None:
        self.run_block_bootstrap.validate()
        finite = (
            self.total_net_bps,
            self.mean_paired_net_bps,
            self.mean_executed_trade_net_bps,
            self.median_executed_trade_net_bps,
            self.win_rate,
            self.maximum_drawdown_bps,
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
            or self.active_runs > ROUND74_SEALED_TEST_RUNS
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
                self.gross_profit_bps,
                self.gross_loss_bps,
                self.mean_maximum_adverse_excursion_bps,
            )
            < 0.0
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
                "unlevered trade-payoff evidence has no capital allocation path "
                "and the sealed horizon is too short for defensible annualization"
            ),
        }


@dataclass(frozen=True)
class Round74SealedAIOverlay:
    model_manifest_sha256: str
    review_sha256: tuple[str, ...]
    execution_replay_sha256: tuple[str, ...]
    reviewed_candidates: int
    runtime_accepted_reviews: int
    runtime_success_rate: float
    same_entry_latency_budget_ns: int
    same_entry_latency_eligible_reviews: int
    same_entry_latency_eligibility_rate: float
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
    uplift_gate_passed: bool
    gate_reasons: tuple[str, ...]

    def validate(self) -> None:
        self.strategy_metrics.validate()
        self.paired_delta_bootstrap.validate()
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
                    self.same_entry_latency_eligible_reviews,
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
            or self.same_entry_latency_eligible_reviews > self.runtime_accepted_reviews
            or self.exact_replay_completed_reviews != self.exact_replay_required_reviews
            or self.exact_replay_required_reviews
            > self.strategy_metrics.paired_observations
            or self.exact_replay_target_ineligible_reviews
            > self.exact_replay_completed_reviews
            or self.delayed_overlap_vetoes > self.exact_replay_completed_reviews
            or isinstance(self.same_entry_latency_budget_ns, bool)
            or not isinstance(self.same_entry_latency_budget_ns, int)
            or self.same_entry_latency_budget_ns <= 0
            or self.retained_trades + self.vetoed_trades
            != self.strategy_metrics.paired_observations
            or self.reduced_trades > self.retained_trades
            or not 0.0 <= self.runtime_success_rate <= 1.0
            or not 0.0 <= self.same_entry_latency_eligibility_rate <= 1.0
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
                        self.same_entry_latency_eligibility_rate,
                        self.same_entry_latency_eligible_reviews
                        / self.reviewed_candidates,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
            )
            or self.uplift_gate_passed == bool(self.gate_reasons)
        ):
            raise ValueError("Round 74 sealed AI overlay differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "model_manifest_sha256": self.model_manifest_sha256,
            "review_sha256": list(self.review_sha256),
            "execution_replay_sha256": list(self.execution_replay_sha256),
            "reviewed_candidates": self.reviewed_candidates,
            "runtime_accepted_reviews": self.runtime_accepted_reviews,
            "runtime_success_rate": self.runtime_success_rate,
            "same_entry_latency_budget_ns": self.same_entry_latency_budget_ns,
            "same_entry_latency_eligible_reviews": (
                self.same_entry_latency_eligible_reviews
            ),
            "same_entry_latency_eligibility_rate": (
                self.same_entry_latency_eligibility_rate
            ),
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
            "uplift_gate_passed": self.uplift_gate_passed,
            "gate_reasons": list(self.gate_reasons),
            "may_create_or_replace_ml_actions": False,
            "same_entry_fill_requires_measured_latency_eligibility": False,
            "same_entry_latency_includes_historical_queue_delay": True,
            "same_entry_latency_is_diagnostic_only": True,
            "latency_adjusted_replay_performed": True,
            "baseline_payoff_scaled_without_rewalking_book": False,
        }


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
    profile: str
    test_batch_sha256: tuple[str, ...]
    model_output_sha256: tuple[str, ...]
    candidate_sha256: tuple[str, ...]
    inference_backend_kind: str
    inference_backend_device: str
    inference_backend_vendor: str
    inference_warning_count: int
    predictive_diagnostics: Round74SealedPredictiveDiagnostics
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
        self.baseline_trace.validate()
        self.baseline_metrics.validate()
        for value in self.ai_overlays:
            value.validate()
        digests = (
            self.reserved_claim_sha256,
            self.reservation_id,
            self.test_access_sha256,
            self.dataset_sha256,
            self.pretest_policy_sha256,
            self.pretest_model_sha256,
            self.probability_calibration_sha256,
            self.action_selection_sha256,
            *self.test_batch_sha256,
            *self.model_output_sha256,
            *self.candidate_sha256,
        )
        passed = tuple(
            (
                "ml_baseline",
                *(
                    f"ai:{value.model_manifest_sha256}"
                    for value in self.ai_overlays
                    if value.uplift_gate_passed
                ),
            )
            if self.baseline_metrics.financial_gate_passed
            else tuple(
                f"ai:{value.model_manifest_sha256}"
                for value in self.ai_overlays
                if value.uplift_gate_passed
            )
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
            or len(
                {value.model_manifest_sha256 for value in self.ai_overlays}
            )
            != ROUND74_SEALED_AI_MODEL_COUNT
            or len(self.test_batch_sha256) != len(self.model_output_sha256)
            or len(self.test_batch_sha256) != len(self.candidate_sha256)
            or len(set(self.test_batch_sha256)) != len(self.test_batch_sha256)
            or len(set(self.candidate_sha256)) != len(self.candidate_sha256)
            or self.profile not in ("conservative", "regular", "aggressive")
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
            "profile": self.profile,
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
        ):
            raise ValueError("Round 74 sealed evaluation outcome differs")


def _run_bootstrap(
    run_ids: Sequence[str],
    values: np.ndarray,
    *,
    expected_run_ids: tuple[str, ...],
    seed: int,
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
    generator = np.random.default_rng(seed)
    sampled = np.empty(ROUND74_SEALED_BOOTSTRAP_DRAWS, dtype=np.float64)
    completed = 0
    while completed < ROUND74_SEALED_BOOTSTRAP_DRAWS:
        rows = min(512, ROUND74_SEALED_BOOTSTRAP_DRAWS - completed)
        indexes = generator.integers(
            0,
            len(totals),
            size=(rows, len(totals)),
            endpoint=False,
        )
        sampled[completed : completed + rows] = totals[indexes].mean(axis=1)
        completed += rows
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
    )
    result.validate()
    return result


def _financial_gate_reasons(
    *,
    metrics: Round74SealedStrategyMetrics,
    profile: str,
) -> tuple[str, ...]:
    spec = round74_action_profile(profile)
    scale = ROUND74_SEALED_TEST_RUNS / 6
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
    exit_monotonic_ns: Sequence[int],
    *,
    profile: str,
    seed: int,
) -> Round74SealedStrategyMetrics:
    scaled = np.asarray(net_payoff_bps, dtype=np.float64)
    scaled_mae = np.asarray(
        maximum_adverse_excursion_bps,
        dtype=np.float64,
    )
    retained_mask = np.asarray(retained, dtype=np.bool_)
    adverse = np.asarray(adverse_selection, dtype=np.bool_)
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
        or any(value < 0 for value in exits)
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
        maximum_drawdown_bps=round74_maximum_realized_drawdown_bps(
            scaled,
            run_ids=trace.run_id,
            exit_monotonic_ns=exits,
            expected_run_ids=trace.expected_run_ids,
        ),
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
        trace.exit_monotonic_ns,
        profile=profile,
        seed=seed,
    )


def _exact_replay_strategy_metrics(
    trace: Round74ActionTrace,
    executions: Sequence[Round74AIExecutionReplayEvidence],
    *,
    profile: str,
    seed: int,
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
                trace.position_capital_fraction
                * row.capital_scaled_net_payoff_bps
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
            or isinstance(self.inference_warning_count, bool)
            or self.inference_warning_count < 0
            or self.target_fields_accessed
            or self.trading_authority
        ):
            raise ValueError("Round 74 target-free inference differs")
        first = self.contexts[0]
        run_ids: set[str] = set()
        feature_rows: set[str] = set()
        prior_key: tuple[int, str, int, int, int, str, int] | None = None
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
                context.role != "test"
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
            run_ids.update(context.run_id)
            prior_key = last_key
        if len(run_ids) != ROUND74_SEALED_TEST_RUNS:
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
) -> Round74TargetFreeCandidateInference:
    """Run the frozen model on immutable causal contexts without target access."""

    selected_contexts = tuple(contexts)
    if isinstance(minibatch_rows, bool) or minibatch_rows < 1:
        raise ValueError("Round 74 sealed inference minibatch differs")
    if not selected_contexts:
        raise ValueError("Round 74 target-free inference contexts are missing")
    for context in selected_contexts:
        context.validate()
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
    return inference, predictive.result()


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


def _ai_overlay(
    trace: Round74ActionTrace,
    all_reviews: tuple[Round74AIPairedReviewEvidence, ...],
    executions: tuple[Round74AIExecutionReplayEvidence, ...],
    *,
    manifest: str,
    expected_partition_sha256: str,
    profile: str,
    seed: int,
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
    )
    baseline_values = np.asarray(trace.net_payoff_bps, dtype=np.float64)
    exact_values = np.asarray(
        [
            trace.position_capital_fraction
            * value.capital_scaled_net_payoff_bps
            for value in executions
        ],
        dtype=np.float64,
    )
    delta = exact_values - baseline_values
    delta_bootstrap = _run_bootstrap(
        trace.run_id,
        delta,
        expected_run_ids=trace.expected_run_ids,
        seed=seed + 500_000,
    )
    runtime_success_rate = (
        sum(value.runtime_status == "accepted" for value in all_reviews)
        / len(all_reviews)
        if all_reviews
        else 0.0
    )
    latency_budgets = {value.same_entry_latency_budget_ns for value in all_reviews}
    if len(latency_budgets) != 1:
        raise ValueError("Round 74 sealed AI latency budget differs")
    latency_eligible_reviews = sum(
        value.same_entry_latency_eligible for value in all_reviews
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
        same_entry_latency_budget_ns=next(iter(latency_budgets)),
        same_entry_latency_eligible_reviews=latency_eligible_reviews,
        same_entry_latency_eligibility_rate=latency_eligibility_rate,
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
    if (
        tuple(batch.batch_sha256 for batch in test_batches) != claim.batch_sha256
        or action_selection.selection_sha256 != claim.action_selection_sha256
        or action_selection.profile != claim.profile
        or action_selection.selected_threshold_score is None
    ):
        raise ValueError("Round 74 sealed reserved input identity differs")
    inference, predictive = _derive_test_candidates(
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
        expected_run_count=ROUND74_SEALED_TEST_RUNS,
    )
    baseline = _baseline_strategy_metrics(
        trace,
        profile=action_selection.profile,
        seed=ROUND74_SEALED_BOOTSTRAP_SEED,
    )
    replay_selection = replace(
        action_selection,
        evaluations=tuple(
            replace(value, trace=trace) for value in action_selection.evaluations
        ),
    )
    replay_selection.validate()
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
                replay_selection,
                contexts=contexts,
                reviews=selected_reviews,
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
        )
        for index, manifest in enumerate(sorted(reviews))
    )
    qualified = tuple(
        (
            "ml_baseline",
            *(
                f"ai:{value.model_manifest_sha256}"
                for value in overlays
                if value.uplift_gate_passed
            ),
        )
        if baseline.financial_gate_passed
        else tuple(
            f"ai:{value.model_manifest_sha256}"
            for value in overlays
            if value.uplift_gate_passed
        )
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
        profile=action_selection.profile,
        test_batch_sha256=claim.batch_sha256,
        model_output_sha256=tuple(value.model_output_sha256 for value in candidates),
        candidate_sha256=tuple(value.candidate_sha256 for value in candidates),
        inference_backend_kind=inference.inference_backend_kind,
        inference_backend_device=inference.inference_backend_device,
        inference_backend_vendor=inference.inference_backend_vendor,
        inference_warning_count=inference.inference_warning_count,
        predictive_diagnostics=predictive,
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
    ai_manifest_sha256: Sequence[str],
    ai_review_provider: Round74SealedAIReviewProvider,
    ai_execution_replay_provider: Round74SealedAIExecutionReplayProvider,
    ledger: Round74SealedEvaluationLedger,
    compute_backend: str = "auto",
    inference_minibatch_rows: int = ROUND74_SEALED_INFERENCE_MINIBATCH_ROWS,
) -> Round74SealedEvaluationOutcome:
    """Reserve metadata, load targets, replay AI, and finalize exactly once."""

    manifests = tuple(
        sorted(_require_sha256(value, "AI manifest") for value in ai_manifest_sha256)
    )
    if (
        len(manifests) != ROUND74_SEALED_AI_MODEL_COUNT
        or len(set(manifests)) != ROUND74_SEALED_AI_MODEL_COUNT
    ):
        raise ValueError(
            "Round 74 sealed evaluation requires the exact two-model AI family"
        )
    claim = ledger.reserve_identity(
        test_identity=test_identity,
        action_selection=action_selection,
        ai_manifest_sha256=manifests,
    )
    try:
        if not ledger.claim_matches(claim, required_status="reserved"):
            raise ValueError("Round 74 sealed reservation is not live")
        batches = tuple(test_batch_loader(claim=claim))
        loaded_identity = build_round74_sealed_dataset_identity(batches)
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
    "ROUND74_SEALED_EVALUATION_SCHEMA_VERSION",
    "ROUND74_SEALED_FAMILYWISE_ALPHA",
    "ROUND74_SEALED_INFERENCE_MINIBATCH_ROWS",
    "ROUND74_SEALED_QUALIFICATION_CONFIGURATION_COUNT",
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
    "Round74SealedPredictiveDiagnostics",
    "Round74SealedStrategyMetrics",
    "Round74SealedTestBatchLoader",
    "Round74TargetFreeCandidateInference",
    "evaluate_round74_sealed_once",
    "infer_round74_target_free_candidates",
]
