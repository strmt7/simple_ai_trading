"""Paired, development-only Round 74 ML-versus-AI overlay evaluation.

The evaluator preserves the frozen ML action sequence. AI may only retain,
scale down, or veto each already selected action. Failed reviews remain paired
observations with zero AI exposure. Missing reviews invalidate the evaluation
instead of silently disappearing.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from .impact_absorption_ai_protocol import (
    Round74AIReviewDecision,
    Round74AIReviewRequest,
)
from .impact_absorption_ai_runtime import (
    ROUND74_AI_RUNTIME_STATUSES,
    Round74AIRuntimeOutcome,
)
from .impact_absorption_ai_worker import Round74AIWorkerResult
from .impact_absorption_event_action_policy import (
    Round74ActionPolicySelection,
    Round74ActionTrace,
    round74_action_profile,
)
from .impact_absorption_event_financial_metrics import (
    round74_conservative_maximum_drawdown_bps,
    round74_maximum_concurrent_adverse_excursion_bps,
    round74_maximum_realized_drawdown_bps,
)
from .impact_absorption_event_sequence import ROUND74_EVENT_SYMBOLS
from .impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS,
)
from .storage import write_json_atomic


ROUND74_AI_UPLIFT_SCHEMA_VERSION = "round-074-ai-uplift-development-v15"
ROUND74_AI_QUALIFICATION_POPULATION_SCHEMA_VERSION = (
    "round-074-ai-qualification-population-v1"
)
ROUND74_AI_PRETEST_QUALIFICATION_SCHEMA_VERSION = (
    "round-074-ai-pretest-qualification-v3"
)
ROUND74_AI_EXECUTION_REPLAY_EVIDENCE_SCHEMA_VERSION = (
    "round-074-ai-execution-replay-evidence-v2"
)
ROUND74_AI_UPLIFT_MINIMUM_RUNTIME_SUCCESS_RATE = 0.99
ROUND74_AI_UPLIFT_MINIMUM_RETAINED_TRADE_RATIO = {
    "conservative": 0.60,
    "regular": 0.50,
    "aggressive": 0.40,
}
ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS = (
    ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
ROUND74_AI_EXECUTION_REPLAY_STATUSES = frozenset(
    {
        "runtime_veto",
        "ai_veto",
        "historical_review_expired",
        "target_ineligible",
        "delayed_overlap_veto",
        "executed",
    }
)


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
        raise ValueError(f"Round 74 AI uplift {label} digest differs")
    return selected


def round74_ai_action_validity_latency_ns(horizon_seconds: int) -> int:
    """Return the source-bound maximum age of an AI action."""

    if (
        isinstance(horizon_seconds, bool)
        or not isinstance(horizon_seconds, int)
        or horizon_seconds not in (30, 300)
    ):
        raise ValueError("Round 74 AI action-validity horizon differs")
    return min(
        horizon_seconds * 1_000_000_000,
        ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS,
    )


def _trace_run_population_matches(
    trace: Round74ActionTrace,
    expected_run_ids: tuple[str, ...],
) -> bool:
    observed = tuple(dict.fromkeys(trace.run_id))
    observed_set = set(observed)
    return observed_set.issubset(set(expected_run_ids)) and observed == tuple(
        run_id for run_id in expected_run_ids if run_id in observed_set
    )


@dataclass(frozen=True)
class Round74AIPairedReviewEvidence:
    """Target-free runtime evidence aligned to one frozen ML action."""

    row_index: int
    feature_row_sha256: str
    run_id: str
    symbol: str
    side: int
    horizon_seconds: int
    pretest_policy_sha256: str
    probability_calibration_sha256: str
    request_sha256: str
    runtime_outcome_sha256: str
    model_manifest_sha256: str
    runtime_status: str
    runtime_elapsed_ns: int
    queue_delay_ns: int
    effective_review_latency_ns: int
    action_validity_latency_ns: int
    action_latency_eligible: bool
    size_multiplier_bps: int
    decision: Round74AIReviewDecision | None

    def validate(self) -> None:
        digests = (
            self.feature_row_sha256,
            self.pretest_policy_sha256,
            self.probability_calibration_sha256,
            self.request_sha256,
            self.runtime_outcome_sha256,
            self.model_manifest_sha256,
        )
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
            or any(_SHA256.fullmatch(value) is None for value in digests)
            or _RUN_ID.fullmatch(self.run_id) is None
            or self.symbol not in ROUND74_EVENT_SYMBOLS
            or self.side not in (-1, 1)
            or self.horizon_seconds not in (30, 300)
            or self.runtime_status not in ROUND74_AI_RUNTIME_STATUSES
            or isinstance(self.runtime_elapsed_ns, bool)
            or not isinstance(self.runtime_elapsed_ns, int)
            or self.runtime_elapsed_ns < 0
            or isinstance(self.queue_delay_ns, bool)
            or not isinstance(self.queue_delay_ns, int)
            or self.queue_delay_ns < 0
            or isinstance(self.effective_review_latency_ns, bool)
            or not isinstance(self.effective_review_latency_ns, int)
            or self.effective_review_latency_ns < 0
            or self.effective_review_latency_ns
            != self.runtime_elapsed_ns + self.queue_delay_ns
            or self.action_validity_latency_ns
            != round74_ai_action_validity_latency_ns(self.horizon_seconds)
            or not isinstance(self.action_latency_eligible, bool)
            or isinstance(self.size_multiplier_bps, bool)
            or not isinstance(self.size_multiplier_bps, int)
            or not 0 <= self.size_multiplier_bps <= 10_000
        ):
            raise ValueError("Round 74 AI paired review differs")
        expected_latency_eligible = (
            self.runtime_status == "accepted"
            and self.effective_review_latency_ns <= self.action_validity_latency_ns
        )
        if self.action_latency_eligible != expected_latency_eligible:
            raise ValueError("Round 74 AI action-latency eligibility differs")
        if self.queue_expired_before_inference and (
            self.runtime_status != "blocked_expired"
            or self.decision is not None
            or self.size_multiplier_bps != 0
        ):
            raise ValueError("Round 74 expired AI queue request differs")
        if self.runtime_status == "accepted":
            if self.decision is None:
                raise ValueError("Round 74 AI accepted review lacks a decision")
            self.decision.validate()
            expected_multiplier = self.decision.size_multiplier_bps
            if self.size_multiplier_bps != expected_multiplier:
                raise ValueError("Round 74 AI paired decision size differs")
        elif self.decision is not None or self.size_multiplier_bps != 0:
            raise ValueError("Round 74 AI failed review did not veto")

    @property
    def review_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    @property
    def queue_expired_before_inference(self) -> bool:
        return self.queue_delay_ns >= self.action_validity_latency_ns

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "row_index": self.row_index,
            "feature_row_sha256": self.feature_row_sha256,
            "run_id": self.run_id,
            "symbol": self.symbol,
            "side": self.side,
            "horizon_seconds": self.horizon_seconds,
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "probability_calibration_sha256": (self.probability_calibration_sha256),
            "request_sha256": self.request_sha256,
            "runtime_outcome_sha256": self.runtime_outcome_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "runtime_status": self.runtime_status,
            "runtime_elapsed_ns": self.runtime_elapsed_ns,
            "queue_delay_ns": self.queue_delay_ns,
            "effective_review_latency_ns": self.effective_review_latency_ns,
            "action_validity_latency_ns": self.action_validity_latency_ns,
            "action_latency_eligible": self.action_latency_eligible,
            "queue_expired_before_inference": (self.queue_expired_before_inference),
            "size_multiplier_bps": self.size_multiplier_bps,
            "decision": (
                self.decision.as_dict() if self.decision is not None else None
            ),
            "realized_target_exposed_to_ai": False,
            "may_change_side_entry_exit_or_overlap_order": False,
            "late_accepted_review_policy": (
                "retain_decision_for_exact_delayed_book_replay"
            ),
            "action_validity_policy": (
                "minimum_of_forecast_horizon_and_target_maximum_delayed_entry"
            ),
            "action_latency_includes_historical_queue_delay": True,
            "action_latency_eligibility_controls_replay": True,
            "latency_adjusted_replay_performed": False,
        }
        if include_sha256:
            value["review_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_runtime(
        cls,
        *,
        row_index: int,
        feature_row_sha256: str,
        run_id: str,
        symbol: str,
        side: int,
        horizon_seconds: int,
        request: Round74AIReviewRequest,
        outcome: Round74AIRuntimeOutcome,
        queue_delay_ns: int,
    ) -> Round74AIPairedReviewEvidence:
        """Validate parent/worker evidence before reducing it to paired data."""

        request.validate()
        outcome.validate()
        if (
            outcome.request_sha256 != request.request_sha256
            or request.sample_sha256 != feature_row_sha256
            or request.pretest_policy_sha256 == ""
            or request.probability_calibration_sha256 == ""
            or request.asset_slot != ROUND74_EVENT_SYMBOLS.index(symbol)
            or request.side != ("long" if side == 1 else "short")
            or request.horizon_seconds != horizon_seconds
        ):
            raise ValueError("Round 74 AI runtime review identity differs")
        if (
            isinstance(queue_delay_ns, bool)
            or not isinstance(queue_delay_ns, int)
            or queue_delay_ns < 0
        ):
            raise ValueError("Round 74 AI queue delay differs")
        effective_latency_ns = outcome.elapsed_ns + queue_delay_ns
        action_validity_latency_ns = round74_ai_action_validity_latency_ns(
            horizon_seconds
        )
        queue_expired_before_inference = queue_delay_ns >= action_validity_latency_ns
        if queue_expired_before_inference and outcome.status != "blocked_expired":
            raise ValueError("Round 74 expired AI queue request differs")
        decision: Round74AIReviewDecision | None = None
        multiplier = 0
        latency_eligible = (
            outcome.status == "accepted"
            and effective_latency_ns <= action_validity_latency_ns
        )
        if outcome.status == "accepted":
            assert outcome.worker_result is not None
            worker = Round74AIWorkerResult.from_dict(outcome.worker_result)
            decision = worker.decision
            multiplier = decision.size_multiplier_bps
            expected_approved = (
                request.proposed_risk_size_bps * decision.size_multiplier_bps // 10_000
            )
            if outcome.approved_risk_size_bps != expected_approved:
                raise ValueError("Round 74 AI approved risk size differs")
        selected = cls(
            row_index=int(row_index),
            feature_row_sha256=_require_sha256(
                feature_row_sha256,
                "feature row",
            ),
            run_id=str(run_id),
            symbol=str(symbol),
            side=int(side),
            horizon_seconds=int(horizon_seconds),
            pretest_policy_sha256=request.pretest_policy_sha256,
            probability_calibration_sha256=(request.probability_calibration_sha256),
            request_sha256=request.request_sha256,
            runtime_outcome_sha256=outcome.outcome_sha256,
            model_manifest_sha256=outcome.manifest_sha256,
            runtime_status=outcome.status,
            runtime_elapsed_ns=outcome.elapsed_ns,
            queue_delay_ns=queue_delay_ns,
            effective_review_latency_ns=effective_latency_ns,
            action_validity_latency_ns=action_validity_latency_ns,
            action_latency_eligible=latency_eligible,
            size_multiplier_bps=multiplier,
            decision=decision,
        )
        selected.validate()
        return selected


@dataclass(frozen=True)
class Round74AIExecutionReplayEvidence:
    """Exact delayed-entry L2 replay aligned to one paired AI review."""

    row_index: int
    feature_row_sha256: str
    run_id: str
    symbol: str
    side: int
    horizon_seconds: int
    source_review_sha256: str
    partition_sha256: str
    source_capture_report_sha256: str
    target_spec_sha256: str
    status: str
    requested_size_multiplier_bps: int
    applied_size_multiplier_bps: int
    exact_l2_replay_performed: bool
    target_outcome_sha256: str | None
    target_context_sha256: str | None
    target_ineligible_reason: str
    requested_entry_monotonic_ns: int | None
    actual_entry_monotonic_ns: int | None
    actual_exit_monotonic_ns: int | None
    reference_quote_notional: float | None
    actual_entry_quote_notional: float | None
    actual_deployed_capital_bps: float
    position_net_payoff_bps: float
    position_maximum_adverse_excursion_bps: float
    capital_scaled_net_payoff_bps: float
    capital_scaled_maximum_adverse_excursion_bps: float
    adverse_selection: bool
    schema_version: str = ROUND74_AI_EXECUTION_REPLAY_EVIDENCE_SCHEMA_VERSION

    def validate(self) -> None:
        replay_status = self.status in {
            "target_ineligible",
            "delayed_overlap_veto",
            "executed",
        }
        target_bound = (
            isinstance(self.target_outcome_sha256, str)
            and _SHA256.fullmatch(self.target_outcome_sha256) is not None
            and isinstance(self.target_context_sha256, str)
            and _SHA256.fullmatch(self.target_context_sha256) is not None
        )
        actual_entry = self.actual_entry_monotonic_ns
        actual_exit = self.actual_exit_monotonic_ns
        capital_values = (
            self.actual_deployed_capital_bps,
            self.position_net_payoff_bps,
            self.position_maximum_adverse_excursion_bps,
            self.capital_scaled_net_payoff_bps,
            self.capital_scaled_maximum_adverse_excursion_bps,
        )
        if (
            self.schema_version != ROUND74_AI_EXECUTION_REPLAY_EVIDENCE_SCHEMA_VERSION
            or isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
            or _SHA256.fullmatch(self.feature_row_sha256) is None
            or _RUN_ID.fullmatch(self.run_id) is None
            or self.symbol not in ROUND74_EVENT_SYMBOLS
            or self.side not in (-1, 1)
            or self.horizon_seconds not in (30, 300)
            or _SHA256.fullmatch(self.source_review_sha256) is None
            or _SHA256.fullmatch(self.partition_sha256) is None
            or _SHA256.fullmatch(self.source_capture_report_sha256) is None
            or _SHA256.fullmatch(self.target_spec_sha256) is None
            or self.status not in ROUND74_AI_EXECUTION_REPLAY_STATUSES
            or isinstance(self.requested_size_multiplier_bps, bool)
            or not isinstance(self.requested_size_multiplier_bps, int)
            or not 0 <= self.requested_size_multiplier_bps <= 10_000
            or isinstance(self.applied_size_multiplier_bps, bool)
            or not isinstance(self.applied_size_multiplier_bps, int)
            or not 0 <= self.applied_size_multiplier_bps <= 10_000
            or self.applied_size_multiplier_bps > self.requested_size_multiplier_bps
            or self.exact_l2_replay_performed != replay_status
            or target_bound != replay_status
            or (
                self.requested_entry_monotonic_ns is not None
                and (
                    isinstance(self.requested_entry_monotonic_ns, bool)
                    or not isinstance(self.requested_entry_monotonic_ns, int)
                    or self.requested_entry_monotonic_ns < 0
                )
            )
            or (
                actual_entry is not None
                and (
                    isinstance(actual_entry, bool)
                    or not isinstance(actual_entry, int)
                    or actual_entry < 0
                )
            )
            or (
                actual_exit is not None
                and (
                    isinstance(actual_exit, bool)
                    or not isinstance(actual_exit, int)
                    or actual_exit < 0
                )
            )
            or any(not math.isfinite(float(value)) for value in capital_values)
            or self.actual_deployed_capital_bps < 0.0
            or self.position_maximum_adverse_excursion_bps < 0.0
            or self.capital_scaled_maximum_adverse_excursion_bps < 0.0
            or not isinstance(self.adverse_selection, bool)
            or (replay_status and self.requested_size_multiplier_bps <= 0)
            or (replay_status and self.requested_entry_monotonic_ns is None)
            or (
                actual_entry is not None
                and self.requested_entry_monotonic_ns is not None
                and actual_entry < self.requested_entry_monotonic_ns
            )
            or (actual_exit is not None and actual_entry is None)
        ):
            raise ValueError("Round 74 AI execution replay evidence differs")
        if self.status == "executed":
            if (
                self.requested_size_multiplier_bps <= 0
                or self.applied_size_multiplier_bps
                != self.requested_size_multiplier_bps
                or self.reference_quote_notional is None
                or self.actual_entry_quote_notional is None
                or self.reference_quote_notional <= 0.0
                or self.actual_entry_quote_notional <= 0.0
                or self.requested_entry_monotonic_ns is None
                or actual_entry is None
                or actual_exit is None
                or actual_exit < actual_entry
                or self.target_ineligible_reason
                or not math.isclose(
                    self.actual_entry_quote_notional
                    / self.reference_quote_notional
                    * 10_000.0,
                    self.actual_deployed_capital_bps,
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                )
                or not math.isclose(
                    self.position_net_payoff_bps
                    * self.actual_entry_quote_notional
                    / self.reference_quote_notional,
                    self.capital_scaled_net_payoff_bps,
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                )
                or not math.isclose(
                    self.position_maximum_adverse_excursion_bps
                    * self.actual_entry_quote_notional
                    / self.reference_quote_notional,
                    self.capital_scaled_maximum_adverse_excursion_bps,
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                )
            ):
                raise ValueError("Round 74 AI executed replay evidence differs")
        elif (
            self.applied_size_multiplier_bps != 0
            or self.reference_quote_notional is not None
            or self.actual_entry_quote_notional is not None
            or self.actual_deployed_capital_bps != 0.0
            or self.position_net_payoff_bps != 0.0
            or self.position_maximum_adverse_excursion_bps != 0.0
            or self.capital_scaled_net_payoff_bps != 0.0
            or self.capital_scaled_maximum_adverse_excursion_bps != 0.0
            or self.adverse_selection
        ):
            raise ValueError("Round 74 AI veto replay exposure differs")
        if self.status == "target_ineligible":
            if not self.target_ineligible_reason:
                raise ValueError("Round 74 AI target ineligibility differs")
        elif self.target_ineligible_reason:
            raise ValueError("Round 74 AI replay reason differs")
        if self.status == "delayed_overlap_veto" and (
            actual_entry is None or actual_exit is None
        ):
            raise ValueError("Round 74 AI overlap replay timing differs")
        if self.status in {"runtime_veto", "ai_veto"} and (
            self.requested_size_multiplier_bps != 0
        ):
            raise ValueError("Round 74 AI pre-replay veto size differs")
        if self.status == "historical_review_expired" and (
            self.requested_size_multiplier_bps <= 0
        ):
            raise ValueError("Round 74 AI expired replay size differs")
        if not replay_status and any(
            value is not None
            for value in (
                self.target_outcome_sha256,
                self.target_context_sha256,
                self.requested_entry_monotonic_ns,
                actual_entry,
                actual_exit,
            )
        ):
            raise ValueError("Round 74 AI non-replay evidence contains a target")

    @property
    def replay_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            **self.__dict__,
            "baseline_payoff_scaled_without_rewalking_book": False,
            "latency_adjusted_book_rewalk_required_for_exposure": True,
            "trading_authority": False,
            "profitability_claim": False,
        }
        if include_sha256:
            value["replay_sha256"] = _canonical_sha256(value)
        return value


@dataclass(frozen=True)
class Round74AIOverlayMetrics:
    """Capital-scaled trace metrics for the immutable baseline sequence."""

    baseline_trades: int
    retained_trades: int
    vetoed_trades: int
    reduced_trades: int
    runtime_accepted_reviews: int
    runtime_success_rate: float
    action_latency_eligible_reviews: int
    action_latency_eligibility_rate: float
    exact_replay_required_reviews: int
    exact_replay_completed_reviews: int
    exact_replay_target_ineligible_reviews: int
    delayed_overlap_vetoes: int
    retained_trade_ratio: float
    distinct_retained_symbols: int
    maximum_retained_symbol_share: float
    total_net_bps: float
    mean_paired_net_bps: float
    maximum_drawdown_bps: float
    realized_maximum_drawdown_bps: float
    maximum_concurrent_adverse_excursion_bps: float
    mean_maximum_adverse_excursion_bps: float
    profitable_run_ratio: float

    def validate(self) -> None:
        counts = (
            self.baseline_trades,
            self.retained_trades,
            self.vetoed_trades,
            self.reduced_trades,
            self.runtime_accepted_reviews,
            self.action_latency_eligible_reviews,
            self.exact_replay_required_reviews,
            self.exact_replay_completed_reviews,
            self.exact_replay_target_ineligible_reviews,
            self.delayed_overlap_vetoes,
            self.distinct_retained_symbols,
        )
        ratios = (
            self.runtime_success_rate,
            self.action_latency_eligibility_rate,
            self.retained_trade_ratio,
            self.maximum_retained_symbol_share,
            self.profitable_run_ratio,
        )
        finite = (
            *ratios,
            self.total_net_bps,
            self.mean_paired_net_bps,
            self.maximum_drawdown_bps,
            self.realized_maximum_drawdown_bps,
            self.maximum_concurrent_adverse_excursion_bps,
            self.mean_maximum_adverse_excursion_bps,
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts
            )
            or self.retained_trades + self.vetoed_trades != self.baseline_trades
            or self.reduced_trades > self.retained_trades
            or self.runtime_accepted_reviews > self.baseline_trades
            or self.action_latency_eligible_reviews > self.runtime_accepted_reviews
            or self.exact_replay_completed_reviews != self.exact_replay_required_reviews
            or self.exact_replay_target_ineligible_reviews
            > self.exact_replay_completed_reviews
            or self.delayed_overlap_vetoes > self.exact_replay_completed_reviews
            or self.distinct_retained_symbols > len(ROUND74_EVENT_SYMBOLS)
            or any(not math.isfinite(float(value)) for value in finite)
            or any(not 0.0 <= float(value) <= 1.0 for value in ratios)
            or min(
                self.maximum_drawdown_bps,
                self.realized_maximum_drawdown_bps,
                self.maximum_concurrent_adverse_excursion_bps,
                self.mean_maximum_adverse_excursion_bps,
            )
            < 0.0
            or self.maximum_drawdown_bps + 1e-12 < self.realized_maximum_drawdown_bps
            or self.maximum_drawdown_bps + 1e-12
            < self.maximum_concurrent_adverse_excursion_bps
            or (
                self.baseline_trades > 0
                and (
                    not math.isclose(
                        self.runtime_success_rate,
                        self.runtime_accepted_reviews / self.baseline_trades,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        self.action_latency_eligibility_rate,
                        self.action_latency_eligible_reviews / self.baseline_trades,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        self.retained_trade_ratio,
                        self.retained_trades / self.baseline_trades,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
            )
        ):
            raise ValueError("Round 74 AI overlay metrics differ")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Round74AIOverlayMetrics:
        payload = dict(value)
        try:
            selected = cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("Round 74 AI overlay metrics payload differs") from exc
        selected.validate()
        if selected.as_dict() != payload:
            raise ValueError("Round 74 AI overlay metrics payload differs")
        return selected


def _scaled_metrics(
    trace: Round74ActionTrace,
    reviews: Sequence[Round74AIPairedReviewEvidence],
    executions: Sequence[Round74AIExecutionReplayEvidence],
) -> tuple[
    Round74AIOverlayMetrics,
    tuple[float, ...],
    tuple[float, ...],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    baseline = np.asarray(trace.net_payoff_bps, dtype=np.float64)
    baseline_mae = np.asarray(
        trace.maximum_adverse_excursion_bps,
        dtype=np.float64,
    )
    scaled = np.asarray(
        [
            trace.position_capital_fraction * value.capital_scaled_net_payoff_bps
            for value in executions
        ],
        dtype=np.float64,
    )
    scaled_mae = np.asarray(
        [
            trace.position_capital_fraction
            * value.capital_scaled_maximum_adverse_excursion_bps
            for value in executions
        ],
        dtype=np.float64,
    )
    retained = np.asarray(
        [value.status == "executed" for value in executions],
        dtype=np.bool_,
    )
    retained_symbols = tuple(
        symbol for symbol, keep in zip(trace.symbol, retained, strict=True) if keep
    )
    maximum_symbol_share = (
        max(retained_symbols.count(symbol) for symbol in ROUND74_EVENT_SYMBOLS)
        / len(retained_symbols)
        if retained_symbols
        else 0.0
    )
    run_baseline = {run_id: 0.0 for run_id in trace.expected_run_ids}
    run_ai = {run_id: 0.0 for run_id in trace.expected_run_ids}
    for run_id, baseline_value, ai_value in zip(
        trace.run_id,
        baseline,
        scaled,
        strict=True,
    ):
        run_baseline[run_id] += float(baseline_value)
        run_ai[run_id] += float(ai_value)
    paired_runs = tuple(
        {
            "run_id": run_id,
            "baseline_net_bps": run_baseline[run_id],
            "ai_net_bps": run_ai[run_id],
            "delta_net_bps": run_ai[run_id] - run_baseline[run_id],
        }
        for run_id in trace.expected_run_ids
    )
    symbols = np.asarray(trace.symbol, dtype=object)
    horizons = np.asarray(trace.horizon_seconds, dtype=np.int64)
    grouped_values: list[Mapping[str, object]] = []
    for symbol in ROUND74_EVENT_SYMBOLS:
        for horizon in (30, 300):
            mask = (symbols == symbol) & (horizons == horizon)
            observations = int(mask.sum())
            if observations == 0:
                continue
            baseline_value = float(baseline[mask].sum())
            ai_value = float(scaled[mask].sum())
            grouped_values.append(
                {
                    "symbol": symbol,
                    "horizon_seconds": horizon,
                    "paired_observations": observations,
                    "baseline_net_bps": baseline_value,
                    "ai_net_bps": ai_value,
                    "delta_net_bps": ai_value - baseline_value,
                }
            )
    paired_symbol_horizons = tuple(grouped_values)
    run_symbol_horizon_values: list[Mapping[str, object]] = []
    run_ids = np.asarray(trace.run_id, dtype=object)
    for run_id in trace.expected_run_ids:
        for symbol in ROUND74_EVENT_SYMBOLS:
            for horizon in (30, 300):
                mask = (run_ids == run_id) & (symbols == symbol) & (horizons == horizon)
                observations = int(mask.sum())
                if observations == 0:
                    continue
                baseline_value = float(baseline[mask].sum())
                ai_value = float(scaled[mask].sum())
                baseline_mae_value = float(baseline_mae[mask].sum())
                ai_mae_value = float(scaled_mae[mask].sum())
                run_symbol_horizon_values.append(
                    {
                        "run_id": run_id,
                        "symbol": symbol,
                        "horizon_seconds": horizon,
                        "paired_observations": observations,
                        "baseline_net_bps": baseline_value,
                        "ai_net_bps": ai_value,
                        "delta_net_bps": ai_value - baseline_value,
                        "baseline_aggregate_adverse_excursion_bps": (
                            baseline_mae_value
                        ),
                        "ai_aggregate_adverse_excursion_bps": ai_mae_value,
                        "delta_aggregate_adverse_excursion_bps": (
                            ai_mae_value - baseline_mae_value
                        ),
                    }
                )
    paired_run_symbol_horizons = tuple(run_symbol_horizon_values)
    actual_entries = tuple(
        execution.actual_entry_monotonic_ns
        if execution.actual_entry_monotonic_ns is not None
        else baseline_entry
        for execution, baseline_entry in zip(
            executions,
            trace.entry_monotonic_ns,
            strict=True,
        )
    )
    actual_exits = tuple(
        execution.actual_exit_monotonic_ns
        if execution.actual_exit_monotonic_ns is not None
        else baseline_exit
        for execution, baseline_exit in zip(
            executions,
            trace.exit_monotonic_ns,
            strict=True,
        )
    )
    realized_drawdown = round74_maximum_realized_drawdown_bps(
        scaled,
        run_ids=trace.run_id,
        exit_monotonic_ns=actual_exits,
        expected_run_ids=trace.expected_run_ids,
    )
    concurrent_adverse_excursion = round74_maximum_concurrent_adverse_excursion_bps(
        scaled_mae,
        run_ids=trace.run_id,
        entry_monotonic_ns=actual_entries,
        exit_monotonic_ns=actual_exits,
        expected_run_ids=trace.expected_run_ids,
    )
    conservative_drawdown = round74_conservative_maximum_drawdown_bps(
        scaled,
        scaled_mae,
        run_ids=trace.run_id,
        entry_monotonic_ns=actual_entries,
        exit_monotonic_ns=actual_exits,
        expected_run_ids=trace.expected_run_ids,
    )
    metrics = Round74AIOverlayMetrics(
        baseline_trades=len(reviews),
        retained_trades=int(retained.sum()),
        vetoed_trades=int((~retained).sum()),
        reduced_trades=sum(
            0 < execution.applied_size_multiplier_bps < 10_000
            for execution in executions
        ),
        runtime_accepted_reviews=sum(
            review.runtime_status == "accepted" for review in reviews
        ),
        runtime_success_rate=float(
            np.mean([review.runtime_status == "accepted" for review in reviews])
        ),
        action_latency_eligible_reviews=sum(
            review.action_latency_eligible for review in reviews
        ),
        action_latency_eligibility_rate=float(
            np.mean([review.action_latency_eligible for review in reviews])
        ),
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
        retained_trade_ratio=float(retained.mean()),
        distinct_retained_symbols=len(set(retained_symbols)),
        maximum_retained_symbol_share=float(maximum_symbol_share),
        total_net_bps=float(scaled.sum()),
        mean_paired_net_bps=float(scaled.mean()),
        maximum_drawdown_bps=conservative_drawdown,
        realized_maximum_drawdown_bps=realized_drawdown,
        maximum_concurrent_adverse_excursion_bps=(concurrent_adverse_excursion),
        mean_maximum_adverse_excursion_bps=float(scaled_mae.mean()),
        profitable_run_ratio=float(np.mean(np.asarray(tuple(run_ai.values())) > 0.0)),
    )
    metrics.validate()
    return (
        metrics,
        tuple(float(value) for value in scaled),
        tuple(float(value) for value in scaled_mae),
        paired_runs,
        paired_symbol_horizons,
        paired_run_symbol_horizons,
    )


def _development_gate_reasons(
    *,
    profile: str,
    metrics: Round74AIOverlayMetrics,
    baseline_trace: Round74ActionTrace,
    paired_runs: Sequence[Mapping[str, object]],
    paired_symbol_horizons: Sequence[Mapping[str, object]],
    paired_run_symbol_horizons: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    action_profile = round74_action_profile(profile)
    reasons: list[str] = []
    if metrics.runtime_success_rate < ROUND74_AI_UPLIFT_MINIMUM_RUNTIME_SUCCESS_RATE:
        reasons.append("runtime_success_rate_not_met")
    if (
        metrics.retained_trade_ratio
        < ROUND74_AI_UPLIFT_MINIMUM_RETAINED_TRADE_RATIO[action_profile.profile]
    ):
        reasons.append("retained_trade_ratio_not_met")
    minimum_retained_trades = math.ceil(
        action_profile.minimum_trades
        * ROUND74_AI_UPLIFT_MINIMUM_RETAINED_TRADE_RATIO[action_profile.profile]
    )
    if metrics.retained_trades < minimum_retained_trades:
        reasons.append("minimum_retained_trades_not_met")
    if metrics.distinct_retained_symbols != len(ROUND74_EVENT_SYMBOLS):
        reasons.append("retained_asset_diversification_not_met")
    if (
        metrics.maximum_retained_symbol_share
        > action_profile.maximum_symbol_trade_share
    ):
        reasons.append("retained_symbol_concentration_not_met")
    if metrics.total_net_bps <= baseline_trace.metrics.total_net_bps:
        reasons.append("positive_paired_after_cost_uplift_not_met")
    if any(float(value["delta_net_bps"]) < -1e-12 for value in paired_runs):
        reasons.append("paired_run_noninferiority_not_met")
    if any(float(value["delta_net_bps"]) < -1e-12 for value in paired_symbol_horizons):
        reasons.append("paired_symbol_horizon_noninferiority_not_met")
    if any(
        float(value["delta_net_bps"]) < -1e-12 for value in paired_run_symbol_horizons
    ):
        reasons.append("paired_run_symbol_horizon_noninferiority_not_met")
    if any(
        float(value["delta_aggregate_adverse_excursion_bps"]) > 1e-12
        for value in paired_run_symbol_horizons
    ):
        reasons.append(
            "paired_run_symbol_horizon_adverse_excursion_noninferiority_not_met"
        )
    if metrics.maximum_drawdown_bps > baseline_trace.metrics.maximum_drawdown_bps:
        reasons.append("maximum_drawdown_noninferiority_not_met")
    return tuple(reasons)


@dataclass(frozen=True)
class Round74AIQualificationPopulation:
    """Target-blind, chronologically later tuning runs reserved for AI."""

    parent_tuning_subpartition_sha256: str
    prior_run_ids: tuple[str, ...]
    prior_slot_ordinals: tuple[int, ...]
    run_ids: tuple[str, ...]
    slot_ordinals: tuple[int, ...]
    eligible_anchor_ns: tuple[int, ...]
    schema_version: str = ROUND74_AI_QUALIFICATION_POPULATION_SCHEMA_VERSION
    optimization_population: str = "eligible_target"

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_AI_QUALIFICATION_POPULATION_SCHEMA_VERSION
            or self.optimization_population != "eligible_target"
            or _SHA256.fullmatch(self.parent_tuning_subpartition_sha256) is None
            or not self.prior_run_ids
            or len(self.prior_run_ids) != len(self.prior_slot_ordinals)
            or not self.run_ids
            or len(self.run_ids) != len(self.slot_ordinals)
            or len(self.run_ids) != len(self.eligible_anchor_ns)
            or any(
                len(value) != 32
                or any(character not in "0123456789abcdef" for character in value)
                for value in (*self.prior_run_ids, *self.run_ids)
            )
            or len(set((*self.prior_run_ids, *self.run_ids)))
            != len(self.prior_run_ids) + len(self.run_ids)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (*self.prior_slot_ordinals, *self.slot_ordinals)
            )
            or len(set((*self.prior_slot_ordinals, *self.slot_ordinals)))
            != len(self.prior_slot_ordinals) + len(self.slot_ordinals)
            or any(
                current <= prior
                for prior, current in zip(
                    (*self.prior_slot_ordinals, *self.slot_ordinals),
                    (*self.prior_slot_ordinals, *self.slot_ordinals)[1:],
                    strict=False,
                )
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.eligible_anchor_ns
            )
        ):
            raise ValueError("Round 74 AI qualification population differs")

    @property
    def population_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "parent_tuning_subpartition_sha256": (
                self.parent_tuning_subpartition_sha256
            ),
            "optimization_population": self.optimization_population,
            "prior_run_ids": list(self.prior_run_ids),
            "prior_slot_ordinals": list(self.prior_slot_ordinals),
            "run_ids": list(self.run_ids),
            "slot_ordinals": list(self.slot_ordinals),
            "eligible_anchor_ns": list(self.eligible_anchor_ns),
            "data_scope": "ai_qualification_tuning_runs_only",
            "assignment_basis": "immutable_scheduled_slot_ordinal_range",
            "target_or_model_output_used_for_assignment": False,
            "calibration_or_policy_selection_run_reuse_permitted": False,
            "sealed_test_accessed": False,
        }
        if include_sha256:
            payload["population_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74AIQualificationPopulation:
        original = dict(value)
        payload = dict(original)
        claimed = str(payload.pop("population_sha256", ""))
        fixed_policy = {
            "data_scope": "ai_qualification_tuning_runs_only",
            "assignment_basis": "immutable_scheduled_slot_ordinal_range",
            "target_or_model_output_used_for_assignment": False,
            "calibration_or_policy_selection_run_reuse_permitted": False,
            "sealed_test_accessed": False,
        }
        if any(
            type(observed := payload.pop(key, None)) is not type(expected)
            or observed != expected
            for key, expected in fixed_policy.items()
        ):
            raise ValueError("Round 74 AI qualification population policy differs")
        if (
            _SHA256.fullmatch(claimed) is None
            or not isinstance(payload.get("prior_run_ids"), list)
            or not isinstance(payload.get("run_ids"), list)
            or any(not isinstance(item, str) for item in payload["prior_run_ids"])
            or any(not isinstance(item, str) for item in payload["run_ids"])
            or not isinstance(payload.get("prior_slot_ordinals"), list)
            or not isinstance(payload.get("slot_ordinals"), list)
            or not isinstance(payload.get("eligible_anchor_ns"), list)
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for key in (
                    "prior_slot_ordinals",
                    "slot_ordinals",
                    "eligible_anchor_ns",
                )
                for item in payload[key]
            )
        ):
            raise ValueError("Round 74 AI qualification population payload differs")
        try:
            selected = cls(
                parent_tuning_subpartition_sha256=str(
                    payload["parent_tuning_subpartition_sha256"]
                ),
                prior_run_ids=tuple(payload["prior_run_ids"]),
                prior_slot_ordinals=tuple(payload["prior_slot_ordinals"]),
                run_ids=tuple(payload["run_ids"]),
                slot_ordinals=tuple(payload["slot_ordinals"]),
                eligible_anchor_ns=tuple(payload["eligible_anchor_ns"]),
                schema_version=str(payload["schema_version"]),
                optimization_population=str(payload["optimization_population"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 AI qualification population payload differs"
            ) from exc
        selected.validate()
        if selected.population_sha256 != claimed or selected.as_dict() != original:
            raise ValueError("Round 74 AI qualification population identity differs")
        return selected


@dataclass(frozen=True)
class Round74AIUpliftDevelopmentReport:
    """Paired tuning diagnostic with no model-selection or trading authority."""

    profile: str
    action_selection_sha256: str
    candidate_sha256: tuple[str, ...]
    pretest_policy_sha256: str
    probability_calibration_sha256: str
    model_manifest_sha256: str
    qualification_population: Round74AIQualificationPopulation
    baseline_trace: Round74ActionTrace
    review_sha256: tuple[str, ...]
    execution_replay_sha256: tuple[str, ...]
    ai_scaled_net_payoff_bps: tuple[float, ...]
    ai_scaled_maximum_adverse_excursion_bps: tuple[float, ...]
    paired_runs: tuple[Mapping[str, object], ...]
    paired_symbol_horizons: tuple[Mapping[str, object], ...]
    paired_run_symbol_horizons: tuple[Mapping[str, object], ...]
    ai_metrics: Round74AIOverlayMetrics
    development_gate_passed: bool
    gate_reasons: tuple[str, ...]
    schema_version: str = ROUND74_AI_UPLIFT_SCHEMA_VERSION
    sealed_test_accessed: bool = False
    ai_model_selection_permitted: bool = False
    promotion_authority: bool = False
    trading_authority: bool = False
    profitability_claim: bool = False

    def validate(self) -> None:
        self.qualification_population.validate()
        self.baseline_trace.validate()
        self.ai_metrics.validate()
        scaled = np.asarray(self.ai_scaled_net_payoff_bps, dtype=np.float64)
        scaled_mae = np.asarray(
            self.ai_scaled_maximum_adverse_excursion_bps,
            dtype=np.float64,
        )
        paired_run_ids: list[str] = []
        paired_valid = True
        for raw in self.paired_runs:
            if set(raw) != {
                "run_id",
                "baseline_net_bps",
                "ai_net_bps",
                "delta_net_bps",
            }:
                paired_valid = False
                continue
            try:
                run_id = str(raw["run_id"])
                baseline_value = float(raw["baseline_net_bps"])
                ai_value = float(raw["ai_net_bps"])
                delta = float(raw["delta_net_bps"])
            except (TypeError, ValueError, OverflowError):
                paired_valid = False
                continue
            paired_run_ids.append(run_id)
            if not all(
                math.isfinite(value) for value in (baseline_value, ai_value, delta)
            ) or not math.isclose(
                delta,
                ai_value - baseline_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                paired_valid = False
        paired_group_keys: list[tuple[str, int]] = []
        paired_group_observations = 0
        paired_groups_valid = True
        for raw in self.paired_symbol_horizons:
            if set(raw) != {
                "symbol",
                "horizon_seconds",
                "paired_observations",
                "baseline_net_bps",
                "ai_net_bps",
                "delta_net_bps",
            }:
                paired_groups_valid = False
                continue
            try:
                symbol = str(raw["symbol"])
                horizon = int(raw["horizon_seconds"])
                observations = int(raw["paired_observations"])
                baseline_value = float(raw["baseline_net_bps"])
                ai_value = float(raw["ai_net_bps"])
                delta = float(raw["delta_net_bps"])
            except (TypeError, ValueError, OverflowError):
                paired_groups_valid = False
                continue
            paired_group_keys.append((symbol, horizon))
            paired_group_observations += observations
            if (
                not isinstance(raw["symbol"], str)
                or symbol not in ROUND74_EVENT_SYMBOLS
                or not isinstance(raw["horizon_seconds"], int)
                or horizon not in (30, 300)
                or isinstance(raw["horizon_seconds"], bool)
                or not isinstance(raw["paired_observations"], int)
                or isinstance(raw["paired_observations"], bool)
                or observations <= 0
                or not all(
                    math.isfinite(value) for value in (baseline_value, ai_value, delta)
                )
                or not math.isclose(
                    delta,
                    ai_value - baseline_value,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                paired_groups_valid = False
        expected_group_keys = tuple(
            (symbol, horizon)
            for symbol in ROUND74_EVENT_SYMBOLS
            for horizon in (30, 300)
            if any(
                observed_symbol == symbol and observed_horizon == horizon
                for observed_symbol, observed_horizon in zip(
                    self.baseline_trace.symbol,
                    self.baseline_trace.horizon_seconds,
                    strict=True,
                )
            )
        )
        paired_cell_keys: list[tuple[str, str, int]] = []
        paired_cell_observations = 0
        paired_cells_valid = True
        for raw in self.paired_run_symbol_horizons:
            if set(raw) != {
                "run_id",
                "symbol",
                "horizon_seconds",
                "paired_observations",
                "baseline_net_bps",
                "ai_net_bps",
                "delta_net_bps",
                "baseline_aggregate_adverse_excursion_bps",
                "ai_aggregate_adverse_excursion_bps",
                "delta_aggregate_adverse_excursion_bps",
            }:
                paired_cells_valid = False
                continue
            try:
                run_id = str(raw["run_id"])
                symbol = str(raw["symbol"])
                horizon = int(raw["horizon_seconds"])
                observations = int(raw["paired_observations"])
                baseline_value = float(raw["baseline_net_bps"])
                ai_value = float(raw["ai_net_bps"])
                delta = float(raw["delta_net_bps"])
                baseline_mae_value = float(
                    raw["baseline_aggregate_adverse_excursion_bps"]
                )
                ai_mae_value = float(raw["ai_aggregate_adverse_excursion_bps"])
                delta_mae = float(raw["delta_aggregate_adverse_excursion_bps"])
            except (TypeError, ValueError, OverflowError):
                paired_cells_valid = False
                continue
            paired_cell_keys.append((run_id, symbol, horizon))
            paired_cell_observations += observations
            if (
                not isinstance(raw["run_id"], str)
                or run_id not in self.baseline_trace.expected_run_ids
                or not isinstance(raw["symbol"], str)
                or symbol not in ROUND74_EVENT_SYMBOLS
                or not isinstance(raw["horizon_seconds"], int)
                or isinstance(raw["horizon_seconds"], bool)
                or horizon not in (30, 300)
                or not isinstance(raw["paired_observations"], int)
                or isinstance(raw["paired_observations"], bool)
                or observations <= 0
                or not all(
                    math.isfinite(value)
                    for value in (
                        baseline_value,
                        ai_value,
                        delta,
                        baseline_mae_value,
                        ai_mae_value,
                        delta_mae,
                    )
                )
                or baseline_mae_value < 0.0
                or ai_mae_value < 0.0
                or not math.isclose(
                    delta,
                    ai_value - baseline_value,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    delta_mae,
                    ai_mae_value - baseline_mae_value,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                paired_cells_valid = False
        expected_cell_keys = tuple(
            (run_id, symbol, horizon)
            for run_id in self.baseline_trace.expected_run_ids
            for symbol in ROUND74_EVENT_SYMBOLS
            for horizon in (30, 300)
            if any(
                observed_run == run_id
                and observed_symbol == symbol
                and observed_horizon == horizon
                for observed_run, observed_symbol, observed_horizon in zip(
                    self.baseline_trace.run_id,
                    self.baseline_trace.symbol,
                    self.baseline_trace.horizon_seconds,
                    strict=True,
                )
            )
        )
        expected_reasons = (
            _development_gate_reasons(
                profile=self.profile,
                metrics=self.ai_metrics,
                baseline_trace=self.baseline_trace,
                paired_runs=self.paired_runs,
                paired_symbol_horizons=self.paired_symbol_horizons,
                paired_run_symbol_horizons=self.paired_run_symbol_horizons,
            )
            if paired_valid and paired_groups_valid and paired_cells_valid
            else ()
        )
        if (
            self.schema_version != ROUND74_AI_UPLIFT_SCHEMA_VERSION
            or self.profile not in ROUND74_AI_UPLIFT_MINIMUM_RETAINED_TRADE_RATIO
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.action_selection_sha256,
                    self.pretest_policy_sha256,
                    self.probability_calibration_sha256,
                    self.model_manifest_sha256,
                    *self.candidate_sha256,
                    *self.review_sha256,
                    *self.execution_replay_sha256,
                )
            )
            or not self.candidate_sha256
            or self.baseline_trace.expected_run_ids
            != self.qualification_population.run_ids
            or not _trace_run_population_matches(
                self.baseline_trace,
                self.qualification_population.run_ids,
            )
            or len(set(self.candidate_sha256)) != len(self.candidate_sha256)
            or len(self.review_sha256) != self.baseline_trace.metrics.trades
            or len(set(self.review_sha256)) != len(self.review_sha256)
            or len(self.execution_replay_sha256) != self.baseline_trace.metrics.trades
            or len(set(self.execution_replay_sha256))
            != len(self.execution_replay_sha256)
            or len(self.ai_scaled_net_payoff_bps) != self.baseline_trace.metrics.trades
            or len(self.ai_scaled_maximum_adverse_excursion_bps)
            != self.baseline_trace.metrics.trades
            or len(self.paired_runs) != len(self.baseline_trace.expected_run_ids)
            or not paired_valid
            or tuple(paired_run_ids) != self.baseline_trace.expected_run_ids
            or not paired_groups_valid
            or tuple(paired_group_keys) != expected_group_keys
            or paired_group_observations != self.baseline_trace.metrics.trades
            or not paired_cells_valid
            or tuple(paired_cell_keys) != expected_cell_keys
            or paired_cell_observations != self.baseline_trace.metrics.trades
            or not np.isfinite(scaled).all()
            or not np.isfinite(scaled_mae).all()
            or np.any(scaled_mae < 0.0)
            or not math.isclose(
                float(scaled.sum()),
                self.ai_metrics.total_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                sum(
                    float(value["baseline_net_bps"])
                    for value in self.paired_run_symbol_horizons
                ),
                self.baseline_trace.metrics.total_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                sum(
                    float(value["ai_net_bps"])
                    for value in self.paired_run_symbol_horizons
                ),
                self.ai_metrics.total_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                sum(
                    float(value["baseline_aggregate_adverse_excursion_bps"])
                    for value in self.paired_run_symbol_horizons
                ),
                sum(self.baseline_trace.maximum_adverse_excursion_bps),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                sum(
                    float(value["ai_aggregate_adverse_excursion_bps"])
                    for value in self.paired_run_symbol_horizons
                ),
                float(scaled_mae.sum()),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                sum(float(value["baseline_net_bps"]) for value in self.paired_runs),
                self.baseline_trace.metrics.total_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                sum(float(value["ai_net_bps"]) for value in self.paired_runs),
                self.ai_metrics.total_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                sum(
                    float(value["baseline_net_bps"])
                    for value in self.paired_symbol_horizons
                ),
                self.baseline_trace.metrics.total_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                sum(
                    float(value["ai_net_bps"]) for value in self.paired_symbol_horizons
                ),
                self.ai_metrics.total_net_bps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or self.ai_metrics.baseline_trades != self.baseline_trace.metrics.trades
            or not isinstance(self.development_gate_passed, bool)
            or self.gate_reasons != expected_reasons
            or self.development_gate_passed != (not expected_reasons)
            or any(
                not isinstance(value, bool)
                for value in (
                    self.sealed_test_accessed,
                    self.ai_model_selection_permitted,
                    self.promotion_authority,
                    self.trading_authority,
                    self.profitability_claim,
                )
            )
            or any(
                (
                    self.sealed_test_accessed,
                    self.ai_model_selection_permitted,
                    self.promotion_authority,
                    self.trading_authority,
                    self.profitability_claim,
                )
            )
        ):
            raise ValueError("Round 74 AI uplift report differs")

    @property
    def report_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "action_selection_sha256": self.action_selection_sha256,
            "candidate_sha256": list(self.candidate_sha256),
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "probability_calibration_sha256": (self.probability_calibration_sha256),
            "model_manifest_sha256": self.model_manifest_sha256,
            "action_validity_maximum_ns": ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS,
            "qualification_population": self.qualification_population.as_dict(),
            "qualification_population_sha256": (
                self.qualification_population.population_sha256
            ),
            "baseline_trace": self.baseline_trace.as_dict(),
            "review_sha256": list(self.review_sha256),
            "execution_replay_sha256": list(self.execution_replay_sha256),
            "ai_scaled_net_payoff_bps": list(self.ai_scaled_net_payoff_bps),
            "ai_scaled_maximum_adverse_excursion_bps": list(
                self.ai_scaled_maximum_adverse_excursion_bps
            ),
            "paired_runs": [dict(value) for value in self.paired_runs],
            "paired_symbol_horizons": [
                dict(value) for value in self.paired_symbol_horizons
            ],
            "paired_run_symbol_horizons": [
                dict(value) for value in self.paired_run_symbol_horizons
            ],
            "ai_metrics": self.ai_metrics.as_dict(),
            "development_gate_passed": self.development_gate_passed,
            "gate_reasons": list(self.gate_reasons),
            "blocked_or_failed_review_policy": (
                "paired_zero_exposure_veto_not_observation_deletion"
            ),
            "missing_review_policy": "invalidate_entire_evaluation",
            "same_side_entry_exit_and_overlap_order": True,
            "action_validity_policy": (
                "minimum_of_forecast_horizon_and_target_maximum_delayed_entry"
            ),
            "action_latency_includes_historical_queue_delay": True,
            "queue_timeout_action": "reject_before_model_inference",
            "queue_expired_observation_policy": (
                "paired_zero_exposure_not_observation_deletion"
            ),
            "expired_action_policy": "paired_zero_exposure_not_observation_deletion",
            "latency_adjusted_replay_performed": True,
            "baseline_payoff_scaled_without_rewalking_book": False,
            "sealed_test_accessed": False,
            "ai_model_selection_permitted": False,
            "promotion_authority": False,
            "trading_authority": False,
            "profitability_claim": False,
        }
        if include_sha256:
            value["report_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74AIUpliftDevelopmentReport:
        original = dict(value)
        payload = dict(original)
        claimed = str(payload.pop("report_sha256", ""))
        policy = {
            "blocked_or_failed_review_policy": (
                "paired_zero_exposure_veto_not_observation_deletion"
            ),
            "missing_review_policy": "invalidate_entire_evaluation",
            "same_side_entry_exit_and_overlap_order": True,
            "action_validity_maximum_ns": ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS,
            "action_validity_policy": (
                "minimum_of_forecast_horizon_and_target_maximum_delayed_entry"
            ),
            "action_latency_includes_historical_queue_delay": True,
            "queue_timeout_action": "reject_before_model_inference",
            "queue_expired_observation_policy": (
                "paired_zero_exposure_not_observation_deletion"
            ),
            "expired_action_policy": "paired_zero_exposure_not_observation_deletion",
            "latency_adjusted_replay_performed": True,
            "baseline_payoff_scaled_without_rewalking_book": False,
            "sealed_test_accessed": False,
            "ai_model_selection_permitted": False,
            "promotion_authority": False,
            "trading_authority": False,
            "profitability_claim": False,
        }
        if any(
            type(observed := payload.pop(key, None)) is not type(expected)
            or observed != expected
            for key, expected in policy.items()
        ):
            raise ValueError("Round 74 AI uplift report policy differs")
        baseline_trace = payload.get("baseline_trace")
        ai_metrics = payload.get("ai_metrics")
        qualification_population = payload.get("qualification_population")
        claimed_population = payload.pop("qualification_population_sha256", None)
        sequence_keys = (
            "candidate_sha256",
            "review_sha256",
            "execution_replay_sha256",
            "ai_scaled_net_payoff_bps",
            "ai_scaled_maximum_adverse_excursion_bps",
            "paired_runs",
            "paired_symbol_horizons",
            "paired_run_symbol_horizons",
            "gate_reasons",
        )
        if (
            _SHA256.fullmatch(claimed) is None
            or not isinstance(baseline_trace, Mapping)
            or not isinstance(ai_metrics, Mapping)
            or not isinstance(qualification_population, Mapping)
            or not isinstance(payload.get("development_gate_passed"), bool)
            or any(not isinstance(payload.get(key), list) for key in sequence_keys)
        ):
            raise ValueError("Round 74 AI uplift report payload differs")
        try:
            population = Round74AIQualificationPopulation.from_dict(
                qualification_population
            )
            if claimed_population != population.population_sha256:
                raise ValueError("qualification population digest differs")
            selected = cls(
                profile=str(payload["profile"]),
                action_selection_sha256=str(payload["action_selection_sha256"]),
                candidate_sha256=tuple(
                    str(item) for item in payload["candidate_sha256"]
                ),
                pretest_policy_sha256=str(payload["pretest_policy_sha256"]),
                probability_calibration_sha256=str(
                    payload["probability_calibration_sha256"]
                ),
                model_manifest_sha256=str(payload["model_manifest_sha256"]),
                qualification_population=population,
                baseline_trace=Round74ActionTrace.from_dict(baseline_trace),
                review_sha256=tuple(str(item) for item in payload["review_sha256"]),
                execution_replay_sha256=tuple(
                    str(item) for item in payload["execution_replay_sha256"]
                ),
                ai_scaled_net_payoff_bps=tuple(
                    float(item) for item in payload["ai_scaled_net_payoff_bps"]
                ),
                ai_scaled_maximum_adverse_excursion_bps=tuple(
                    float(item)
                    for item in payload["ai_scaled_maximum_adverse_excursion_bps"]
                ),
                paired_runs=tuple(dict(item) for item in payload["paired_runs"]),
                paired_symbol_horizons=tuple(
                    dict(item) for item in payload["paired_symbol_horizons"]
                ),
                paired_run_symbol_horizons=tuple(
                    dict(item) for item in payload["paired_run_symbol_horizons"]
                ),
                ai_metrics=Round74AIOverlayMetrics.from_dict(ai_metrics),
                development_gate_passed=payload["development_gate_passed"],
                gate_reasons=tuple(str(item) for item in payload["gate_reasons"]),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Round 74 AI uplift report payload differs") from exc
        selected.validate()
        if selected.report_sha256 != claimed or selected.as_dict() != original:
            raise ValueError("Round 74 AI uplift report identity differs")
        return selected


@dataclass(frozen=True)
class Round74AIPretestQualificationPanel:
    """Bind two real development reports before any sealed-test reservation."""

    development_reports: tuple[Round74AIUpliftDevelopmentReport, ...]
    qualification_passed: bool
    gate_reasons: tuple[str, ...]
    schema_version: str = ROUND74_AI_PRETEST_QUALIFICATION_SCHEMA_VERSION
    sealed_test_accessed: bool = False
    model_selection_performed: bool = False
    promotion_authority: bool = False
    trading_authority: bool = False
    profitability_claim: bool = False

    def validate(self) -> None:
        reports = tuple(self.development_reports)
        for report in reports:
            report.validate()
        identities = {
            (
                report.profile,
                report.action_selection_sha256,
                report.candidate_sha256,
                report.pretest_policy_sha256,
                report.probability_calibration_sha256,
                report.qualification_population.population_sha256,
                _canonical_sha256(report.baseline_trace.as_dict()),
            )
            for report in reports
        }
        manifests = tuple(report.model_manifest_sha256 for report in reports)
        expected_reasons = tuple(
            sorted(
                f"model:{report.model_manifest_sha256}:{reason}"
                for report in reports
                if not report.development_gate_passed
                for reason in (
                    report.gate_reasons
                    if report.gate_reasons
                    else ("development_gate_not_met",)
                )
            )
        )
        if (
            self.schema_version != ROUND74_AI_PRETEST_QUALIFICATION_SCHEMA_VERSION
            or len(reports) != 2
            or len(identities) != 1
            or manifests != tuple(sorted(manifests))
            or len(set(manifests)) != 2
            or self.gate_reasons != expected_reasons
            or self.qualification_passed != (not expected_reasons)
            or not isinstance(self.qualification_passed, bool)
            or any(
                not isinstance(value, bool)
                for value in (
                    self.sealed_test_accessed,
                    self.model_selection_performed,
                    self.promotion_authority,
                    self.trading_authority,
                    self.profitability_claim,
                )
            )
            or any(
                (
                    self.sealed_test_accessed,
                    self.model_selection_performed,
                    self.promotion_authority,
                    self.trading_authority,
                    self.profitability_claim,
                )
            )
        ):
            raise ValueError("Round 74 AI pretest qualification differs")

    @property
    def profile(self) -> str:
        self.validate()
        return self.development_reports[0].profile

    @property
    def action_selection_sha256(self) -> str:
        self.validate()
        return self.development_reports[0].action_selection_sha256

    @property
    def pretest_policy_sha256(self) -> str:
        self.validate()
        return self.development_reports[0].pretest_policy_sha256

    @property
    def probability_calibration_sha256(self) -> str:
        self.validate()
        return self.development_reports[0].probability_calibration_sha256

    @property
    def model_manifest_sha256(self) -> tuple[str, ...]:
        self.validate()
        return tuple(
            report.model_manifest_sha256 for report in self.development_reports
        )

    @property
    def qualification_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "development_reports": [
                report.as_dict() for report in self.development_reports
            ],
            "qualification_passed": self.qualification_passed,
            "gate_reasons": list(self.gate_reasons),
            "model_manifest_sha256": list(self.model_manifest_sha256),
            "development_report_sha256": [
                report.report_sha256 for report in self.development_reports
            ],
            "development_data_scope": "ai_qualification_tuning_runs_only",
            "sealed_test_accessed": False,
            "model_selection_performed": False,
            "promotion_authority": False,
            "trading_authority": False,
            "profitability_claim": False,
        }
        if include_sha256:
            payload["qualification_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74AIPretestQualificationPanel:
        original = dict(value)
        payload = dict(original)
        claimed = str(payload.pop("qualification_sha256", ""))
        raw_reports = payload.get("development_reports")
        raw_reasons = payload.get("gate_reasons")
        if (
            _SHA256.fullmatch(claimed) is None
            or not isinstance(raw_reports, list)
            or not isinstance(raw_reasons, list)
            or not isinstance(payload.get("qualification_passed"), bool)
            or any(not isinstance(item, Mapping) for item in raw_reports)
        ):
            raise ValueError("Round 74 AI pretest qualification payload differs")
        reports = tuple(
            Round74AIUpliftDevelopmentReport.from_dict(item) for item in raw_reports
        )
        expected_policy = {
            "model_manifest_sha256": [
                report.model_manifest_sha256 for report in reports
            ],
            "development_report_sha256": [report.report_sha256 for report in reports],
            "development_data_scope": "ai_qualification_tuning_runs_only",
            "sealed_test_accessed": False,
            "model_selection_performed": False,
            "promotion_authority": False,
            "trading_authority": False,
            "profitability_claim": False,
        }
        if any(
            type(observed := payload.pop(key, None)) is not type(expected)
            or observed != expected
            for key, expected in expected_policy.items()
        ):
            raise ValueError("Round 74 AI pretest qualification policy differs")
        selected = cls(
            development_reports=reports,
            qualification_passed=payload["qualification_passed"],
            gate_reasons=tuple(str(item) for item in raw_reasons),
            schema_version=str(payload["schema_version"]),
        )
        selected.validate()
        if selected.qualification_sha256 != claimed or selected.as_dict() != original:
            raise ValueError("Round 74 AI pretest qualification identity differs")
        return selected


def build_round74_ai_pretest_qualification(
    reports: Sequence[Round74AIUpliftDevelopmentReport],
) -> Round74AIPretestQualificationPanel:
    """Create a target-bound development screen with no sealed-test access."""

    selected_reports = tuple(
        sorted(tuple(reports), key=lambda report: report.model_manifest_sha256)
    )
    for report in selected_reports:
        report.validate()
    reasons = tuple(
        sorted(
            f"model:{report.model_manifest_sha256}:{reason}"
            for report in selected_reports
            if not report.development_gate_passed
            for reason in (
                report.gate_reasons
                if report.gate_reasons
                else ("development_gate_not_met",)
            )
        )
    )
    selected = Round74AIPretestQualificationPanel(
        development_reports=selected_reports,
        qualification_passed=not reasons,
        gate_reasons=reasons,
    )
    selected.validate()
    return selected


def write_round74_ai_pretest_qualification(
    qualification: Round74AIPretestQualificationPanel,
    path: str | Path,
) -> Path:
    qualification.validate()
    selected = Path(path)
    if selected.is_symlink():
        raise ValueError("Round 74 AI pretest qualification path differs")
    if selected.exists():
        restored = load_round74_ai_pretest_qualification(selected)
        if restored.qualification_sha256 != qualification.qualification_sha256:
            raise FileExistsError("Round 74 immutable AI pretest qualification differs")
        return selected
    write_json_atomic(selected, qualification.as_dict(), indent=2, sort_keys=True)
    restored = load_round74_ai_pretest_qualification(selected)
    if restored.qualification_sha256 != qualification.qualification_sha256:
        raise RuntimeError("Round 74 AI pretest qualification persistence differs")
    return selected


def load_round74_ai_pretest_qualification(
    path: str | Path,
) -> Round74AIPretestQualificationPanel:
    selected = Path(path)
    raw = selected.read_bytes()
    if not raw or len(raw) > 64 * 1024 * 1024:
        raise ValueError("Round 74 AI pretest qualification file size differs")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for key, item in pairs:
            if key in parsed:
                raise ValueError("Round 74 AI pretest qualification has duplicate keys")
            parsed[key] = item
        return parsed

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 74 AI pretest qualification JSON differs") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Round 74 AI pretest qualification root differs")
    return Round74AIPretestQualificationPanel.from_dict(payload)


def evaluate_round74_ai_overlay_development(
    action_selection: Round74ActionPolicySelection,
    reviews: Sequence[Round74AIPairedReviewEvidence],
    executions: Sequence[Round74AIExecutionReplayEvidence],
    *,
    qualification_population: Round74AIQualificationPopulation,
    qualification_trace: Round74ActionTrace,
    qualification_candidate_sha256: Sequence[str],
) -> Round74AIUpliftDevelopmentReport:
    """Compare one AI overlay on a disjoint, preassigned tuning population."""

    action_selection.validate()
    qualification_population.validate()
    qualification_trace.validate()
    selected = [
        value
        for value in action_selection.evaluations
        if value.accepted
        and value.quantile == action_selection.selected_quantile
        and value.threshold_score == action_selection.selected_threshold_score
    ]
    if not action_selection.accepted or len(selected) != 1:
        raise ValueError("Round 74 AI uplift lacks an accepted action policy")
    selected_candidate_sha256 = tuple(
        _require_sha256(value, "qualification candidate")
        for value in qualification_candidate_sha256
    )
    policy_run_ids = {
        run_id
        for evaluation in action_selection.evaluations
        for run_id in evaluation.trace.expected_run_ids
    }
    trace = qualification_trace
    if (
        not selected_candidate_sha256
        or len(set(selected_candidate_sha256)) != len(selected_candidate_sha256)
        or qualification_population.parent_tuning_subpartition_sha256
        != action_selection.tuning_subpartition_sha256
        or trace.expected_run_ids != qualification_population.run_ids
        or not _trace_run_population_matches(
            trace,
            qualification_population.run_ids,
        )
        or not policy_run_ids.issubset(set(qualification_population.prior_run_ids))
        or policy_run_ids.intersection(qualification_population.run_ids)
        or trace.threshold_score != action_selection.selected_threshold_score
    ):
        raise ValueError("Round 74 AI qualification population identity differs")
    review_rows = tuple(reviews)
    execution_rows = tuple(executions)
    for review in review_rows:
        review.validate()
    for execution in execution_rows:
        execution.validate()
    if (
        len(review_rows) != trace.metrics.trades
        or len(execution_rows) != trace.metrics.trades
        or tuple(review.row_index for review in review_rows) != trace.row_index
        or tuple(value.row_index for value in execution_rows) != trace.row_index
        or len({review.row_index for review in review_rows}) != len(review_rows)
        or len({value.row_index for value in execution_rows}) != len(execution_rows)
    ):
        raise ValueError("Round 74 AI paired review coverage differs")
    manifest_values = {review.model_manifest_sha256 for review in review_rows}
    if len(manifest_values) != 1:
        raise ValueError("Round 74 AI paired model identity differs")
    for index, review in enumerate(review_rows):
        execution = execution_rows[index]
        requested_multiplier = (
            review.decision.size_multiplier_bps
            if review.runtime_status == "accepted" and review.decision is not None
            else 0
        )
        if (
            review.feature_row_sha256 != trace.feature_row_sha256[index]
            or review.run_id != trace.run_id[index]
            or review.symbol != trace.symbol[index]
            or review.side != trace.side[index]
            or review.horizon_seconds != trace.horizon_seconds[index]
            or review.pretest_policy_sha256 != action_selection.pretest_policy_sha256
            or review.probability_calibration_sha256
            != action_selection.probability_calibration_sha256
            or execution.feature_row_sha256 != trace.feature_row_sha256[index]
            or execution.run_id != trace.run_id[index]
            or execution.symbol != trace.symbol[index]
            or execution.side != trace.side[index]
            or execution.horizon_seconds != trace.horizon_seconds[index]
            or execution.source_review_sha256 != review.review_sha256
            or execution.requested_size_multiplier_bps != requested_multiplier
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
                and not review.action_latency_eligible
                and execution.status != "historical_review_expired"
            )
            or (
                review.runtime_status == "accepted"
                and requested_multiplier > 0
                and review.action_latency_eligible
                and execution.status
                not in {
                    "target_ineligible",
                    "delayed_overlap_veto",
                    "executed",
                }
            )
        ):
            raise ValueError("Round 74 AI paired action identity differs")
    (
        metrics,
        scaled,
        scaled_mae,
        paired_runs,
        paired_symbol_horizons,
        paired_run_symbol_horizons,
    ) = _scaled_metrics(trace, review_rows, execution_rows)
    profile = round74_action_profile(action_selection.profile)
    reasons = _development_gate_reasons(
        profile=profile.profile,
        metrics=metrics,
        baseline_trace=trace,
        paired_runs=paired_runs,
        paired_symbol_horizons=paired_symbol_horizons,
        paired_run_symbol_horizons=paired_run_symbol_horizons,
    )
    result = Round74AIUpliftDevelopmentReport(
        profile=profile.profile,
        action_selection_sha256=action_selection.selection_sha256,
        candidate_sha256=selected_candidate_sha256,
        pretest_policy_sha256=action_selection.pretest_policy_sha256,
        probability_calibration_sha256=(
            action_selection.probability_calibration_sha256
        ),
        model_manifest_sha256=next(iter(manifest_values)),
        qualification_population=qualification_population,
        baseline_trace=trace,
        review_sha256=tuple(review.review_sha256 for review in review_rows),
        execution_replay_sha256=tuple(value.replay_sha256 for value in execution_rows),
        ai_scaled_net_payoff_bps=scaled,
        ai_scaled_maximum_adverse_excursion_bps=scaled_mae,
        paired_runs=paired_runs,
        paired_symbol_horizons=paired_symbol_horizons,
        paired_run_symbol_horizons=paired_run_symbol_horizons,
        ai_metrics=metrics,
        development_gate_passed=not reasons,
        gate_reasons=reasons,
    )
    result.validate()
    return result


__all__ = [
    "ROUND74_AI_EXECUTION_REPLAY_EVIDENCE_SCHEMA_VERSION",
    "ROUND74_AI_EXECUTION_REPLAY_STATUSES",
    "ROUND74_AI_ACTION_VALIDITY_MAXIMUM_NS",
    "ROUND74_AI_PRETEST_QUALIFICATION_SCHEMA_VERSION",
    "ROUND74_AI_QUALIFICATION_POPULATION_SCHEMA_VERSION",
    "ROUND74_AI_UPLIFT_MINIMUM_RETAINED_TRADE_RATIO",
    "ROUND74_AI_UPLIFT_MINIMUM_RUNTIME_SUCCESS_RATE",
    "ROUND74_AI_UPLIFT_SCHEMA_VERSION",
    "Round74AIExecutionReplayEvidence",
    "Round74AIOverlayMetrics",
    "Round74AIPairedReviewEvidence",
    "Round74AIPretestQualificationPanel",
    "Round74AIQualificationPopulation",
    "Round74AIUpliftDevelopmentReport",
    "build_round74_ai_pretest_qualification",
    "evaluate_round74_ai_overlay_development",
    "load_round74_ai_pretest_qualification",
    "write_round74_ai_pretest_qualification",
    "round74_ai_action_validity_latency_ns",
]
