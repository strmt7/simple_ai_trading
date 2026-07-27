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
    round74_maximum_realized_drawdown_bps,
)
from .impact_absorption_event_sequence import ROUND74_EVENT_SYMBOLS


ROUND74_AI_UPLIFT_SCHEMA_VERSION = "round-074-ai-uplift-development-v4"
ROUND74_AI_UPLIFT_MINIMUM_RUNTIME_SUCCESS_RATE = 0.99
ROUND74_AI_UPLIFT_MINIMUM_SAME_ENTRY_LATENCY_ELIGIBILITY_RATE = 0.99
ROUND74_AI_UPLIFT_MINIMUM_RETAINED_TRADE_RATIO = {
    "conservative": 0.60,
    "regular": 0.50,
    "aggressive": 0.40,
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")


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
    same_entry_latency_budget_ns: int
    same_entry_latency_eligible: bool
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
            or isinstance(self.same_entry_latency_budget_ns, bool)
            or not isinstance(self.same_entry_latency_budget_ns, int)
            or self.same_entry_latency_budget_ns <= 0
            or not isinstance(self.same_entry_latency_eligible, bool)
            or isinstance(self.size_multiplier_bps, bool)
            or not isinstance(self.size_multiplier_bps, int)
            or not 0 <= self.size_multiplier_bps <= 10_000
        ):
            raise ValueError("Round 74 AI paired review differs")
        expected_latency_eligible = (
            self.runtime_status == "accepted"
            and self.runtime_elapsed_ns <= self.same_entry_latency_budget_ns
        )
        if self.same_entry_latency_eligible != expected_latency_eligible:
            raise ValueError("Round 74 AI same-entry latency eligibility differs")
        if self.runtime_status == "accepted":
            if self.decision is None:
                raise ValueError("Round 74 AI accepted review lacks a decision")
            self.decision.validate()
            expected_multiplier = (
                self.decision.size_multiplier_bps
                if self.same_entry_latency_eligible
                else 0
            )
            if self.size_multiplier_bps != expected_multiplier:
                raise ValueError("Round 74 AI paired decision size differs")
        elif self.decision is not None or self.size_multiplier_bps != 0:
            raise ValueError("Round 74 AI failed review did not veto")

    @property
    def review_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

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
            "same_entry_latency_budget_ns": self.same_entry_latency_budget_ns,
            "same_entry_latency_eligible": self.same_entry_latency_eligible,
            "size_multiplier_bps": self.size_multiplier_bps,
            "decision": (
                self.decision.as_dict() if self.decision is not None else None
            ),
            "realized_target_exposed_to_ai": False,
            "may_change_side_entry_exit_or_overlap_order": False,
            "late_accepted_review_policy": (
                "retain_audit_decision_but_apply_zero_same_entry_exposure"
            ),
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
        same_entry_latency_budget_ns: int,
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
            isinstance(same_entry_latency_budget_ns, bool)
            or not isinstance(same_entry_latency_budget_ns, int)
            or same_entry_latency_budget_ns <= 0
        ):
            raise ValueError("Round 74 AI same-entry latency budget differs")
        decision: Round74AIReviewDecision | None = None
        multiplier = 0
        latency_eligible = (
            outcome.status == "accepted"
            and outcome.elapsed_ns <= same_entry_latency_budget_ns
        )
        if outcome.status == "accepted":
            assert outcome.worker_result is not None
            worker = Round74AIWorkerResult.from_dict(outcome.worker_result)
            decision = worker.decision
            multiplier = decision.size_multiplier_bps if latency_eligible else 0
            expected_approved = (
                request.proposed_risk_size_bps
                * decision.size_multiplier_bps
                // 10_000
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
            same_entry_latency_budget_ns=same_entry_latency_budget_ns,
            same_entry_latency_eligible=latency_eligible,
            size_multiplier_bps=multiplier,
            decision=decision,
        )
        selected.validate()
        return selected


@dataclass(frozen=True)
class Round74AIOverlayMetrics:
    """Capital-scaled trace metrics for the immutable baseline sequence."""

    baseline_trades: int
    retained_trades: int
    vetoed_trades: int
    reduced_trades: int
    runtime_accepted_reviews: int
    runtime_success_rate: float
    same_entry_latency_eligible_reviews: int
    same_entry_latency_eligibility_rate: float
    retained_trade_ratio: float
    distinct_retained_symbols: int
    maximum_retained_symbol_share: float
    total_net_bps: float
    mean_paired_net_bps: float
    maximum_drawdown_bps: float
    mean_maximum_adverse_excursion_bps: float
    profitable_run_ratio: float

    def validate(self) -> None:
        counts = (
            self.baseline_trades,
            self.retained_trades,
            self.vetoed_trades,
            self.reduced_trades,
            self.runtime_accepted_reviews,
            self.same_entry_latency_eligible_reviews,
            self.distinct_retained_symbols,
        )
        ratios = (
            self.runtime_success_rate,
            self.same_entry_latency_eligibility_rate,
            self.retained_trade_ratio,
            self.maximum_retained_symbol_share,
            self.profitable_run_ratio,
        )
        finite = (
            *ratios,
            self.total_net_bps,
            self.mean_paired_net_bps,
            self.maximum_drawdown_bps,
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
            or self.same_entry_latency_eligible_reviews > self.runtime_accepted_reviews
            or self.distinct_retained_symbols > len(ROUND74_EVENT_SYMBOLS)
            or any(not math.isfinite(float(value)) for value in finite)
            or any(not 0.0 <= float(value) <= 1.0 for value in ratios)
            or min(
                self.maximum_drawdown_bps,
                self.mean_maximum_adverse_excursion_bps,
            )
            < 0.0
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
                        self.same_entry_latency_eligibility_rate,
                        self.same_entry_latency_eligible_reviews / self.baseline_trades,
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


def _scaled_metrics(
    trace: Round74ActionTrace,
    reviews: Sequence[Round74AIPairedReviewEvidence],
) -> tuple[
    Round74AIOverlayMetrics,
    tuple[float, ...],
    tuple[float, ...],
    tuple[Mapping[str, object], ...],
]:
    multipliers = np.asarray(
        [review.size_multiplier_bps / 10_000.0 for review in reviews],
        dtype=np.float64,
    )
    baseline = np.asarray(trace.net_payoff_bps, dtype=np.float64)
    baseline_mae = np.asarray(
        trace.maximum_adverse_excursion_bps,
        dtype=np.float64,
    )
    scaled = baseline * multipliers
    scaled_mae = baseline_mae * multipliers
    retained = multipliers > 0.0
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
    metrics = Round74AIOverlayMetrics(
        baseline_trades=len(reviews),
        retained_trades=int(retained.sum()),
        vetoed_trades=int((~retained).sum()),
        reduced_trades=sum(
            0 < review.size_multiplier_bps < 10_000 for review in reviews
        ),
        runtime_accepted_reviews=sum(
            review.runtime_status == "accepted" for review in reviews
        ),
        runtime_success_rate=float(
            np.mean([review.runtime_status == "accepted" for review in reviews])
        ),
        same_entry_latency_eligible_reviews=sum(
            review.same_entry_latency_eligible for review in reviews
        ),
        same_entry_latency_eligibility_rate=float(
            np.mean([review.same_entry_latency_eligible for review in reviews])
        ),
        retained_trade_ratio=float(retained.mean()),
        distinct_retained_symbols=len(set(retained_symbols)),
        maximum_retained_symbol_share=float(maximum_symbol_share),
        total_net_bps=float(scaled.sum()),
        mean_paired_net_bps=float(scaled.mean()),
        maximum_drawdown_bps=round74_maximum_realized_drawdown_bps(
            scaled,
            run_ids=trace.run_id,
            exit_monotonic_ns=trace.exit_monotonic_ns,
            expected_run_ids=trace.expected_run_ids,
        ),
        mean_maximum_adverse_excursion_bps=float(scaled_mae.mean()),
        profitable_run_ratio=float(np.mean(np.asarray(tuple(run_ai.values())) > 0.0)),
    )
    metrics.validate()
    return (
        metrics,
        tuple(float(value) for value in scaled),
        tuple(float(value) for value in scaled_mae),
        paired_runs,
    )


@dataclass(frozen=True)
class Round74AIUpliftDevelopmentReport:
    """Paired tuning diagnostic with no model-selection or trading authority."""

    profile: str
    action_selection_sha256: str
    candidate_sha256: tuple[str, ...]
    pretest_policy_sha256: str
    probability_calibration_sha256: str
    model_manifest_sha256: str
    same_entry_latency_budget_ns: int
    baseline_trace: Round74ActionTrace
    review_sha256: tuple[str, ...]
    ai_scaled_net_payoff_bps: tuple[float, ...]
    ai_scaled_maximum_adverse_excursion_bps: tuple[float, ...]
    paired_runs: tuple[Mapping[str, object], ...]
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
                )
            )
            or not self.candidate_sha256
            or isinstance(self.same_entry_latency_budget_ns, bool)
            or not isinstance(self.same_entry_latency_budget_ns, int)
            or self.same_entry_latency_budget_ns <= 0
            or len(set(self.candidate_sha256)) != len(self.candidate_sha256)
            or len(self.review_sha256) != self.baseline_trace.metrics.trades
            or len(set(self.review_sha256)) != len(self.review_sha256)
            or len(self.ai_scaled_net_payoff_bps) != self.baseline_trace.metrics.trades
            or len(self.ai_scaled_maximum_adverse_excursion_bps)
            != self.baseline_trace.metrics.trades
            or len(self.paired_runs) != len(self.baseline_trace.expected_run_ids)
            or not paired_valid
            or tuple(paired_run_ids) != self.baseline_trace.expected_run_ids
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
            or self.ai_metrics.baseline_trades != self.baseline_trace.metrics.trades
            or self.development_gate_passed == bool(self.gate_reasons)
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
            "same_entry_latency_budget_ns": self.same_entry_latency_budget_ns,
            "baseline_trace": self.baseline_trace.as_dict(),
            "review_sha256": list(self.review_sha256),
            "ai_scaled_net_payoff_bps": list(self.ai_scaled_net_payoff_bps),
            "ai_scaled_maximum_adverse_excursion_bps": list(
                self.ai_scaled_maximum_adverse_excursion_bps
            ),
            "paired_runs": [dict(value) for value in self.paired_runs],
            "ai_metrics": self.ai_metrics.as_dict(),
            "development_gate_passed": self.development_gate_passed,
            "gate_reasons": list(self.gate_reasons),
            "blocked_or_failed_review_policy": (
                "paired_zero_exposure_veto_not_observation_deletion"
            ),
            "missing_review_policy": "invalidate_entire_evaluation",
            "same_side_entry_exit_and_overlap_order": True,
            "same_entry_fill_requires_measured_latency_eligibility": True,
            "latency_adjusted_replay_performed": False,
            "sealed_test_accessed": False,
            "ai_model_selection_permitted": False,
            "promotion_authority": False,
            "trading_authority": False,
            "profitability_claim": False,
        }
        if include_sha256:
            value["report_sha256"] = _canonical_sha256(value)
        return value


def evaluate_round74_ai_overlay_development(
    action_selection: Round74ActionPolicySelection,
    reviews: Sequence[Round74AIPairedReviewEvidence],
) -> Round74AIUpliftDevelopmentReport:
    """Compare one AI overlay on the already-consumed tuning trace only."""

    action_selection.validate()
    selected = [
        value
        for value in action_selection.evaluations
        if value.accepted
        and value.quantile == action_selection.selected_quantile
        and value.threshold_score == action_selection.selected_threshold_score
    ]
    if not action_selection.accepted or len(selected) != 1:
        raise ValueError("Round 74 AI uplift lacks an accepted action policy")
    trace = selected[0].trace
    review_rows = tuple(reviews)
    for review in review_rows:
        review.validate()
    if (
        len(review_rows) != trace.metrics.trades
        or tuple(review.row_index for review in review_rows) != trace.row_index
        or len({review.row_index for review in review_rows}) != len(review_rows)
    ):
        raise ValueError("Round 74 AI paired review coverage differs")
    manifest_values = {review.model_manifest_sha256 for review in review_rows}
    if len(manifest_values) != 1:
        raise ValueError("Round 74 AI paired model identity differs")
    latency_budgets = {review.same_entry_latency_budget_ns for review in review_rows}
    if len(latency_budgets) != 1:
        raise ValueError("Round 74 AI paired latency budget differs")
    for index, review in enumerate(review_rows):
        if (
            review.feature_row_sha256 != trace.feature_row_sha256[index]
            or review.run_id != trace.run_id[index]
            or review.symbol != trace.symbol[index]
            or review.side != trace.side[index]
            or review.horizon_seconds != trace.horizon_seconds[index]
            or review.pretest_policy_sha256 != action_selection.pretest_policy_sha256
            or review.probability_calibration_sha256
            != action_selection.probability_calibration_sha256
        ):
            raise ValueError("Round 74 AI paired action identity differs")
    metrics, scaled, scaled_mae, paired_runs = _scaled_metrics(
        trace,
        review_rows,
    )
    profile = round74_action_profile(action_selection.profile)
    reasons: list[str] = []
    if metrics.runtime_success_rate < ROUND74_AI_UPLIFT_MINIMUM_RUNTIME_SUCCESS_RATE:
        reasons.append("runtime_success_rate_not_met")
    if (
        metrics.same_entry_latency_eligibility_rate
        < ROUND74_AI_UPLIFT_MINIMUM_SAME_ENTRY_LATENCY_ELIGIBILITY_RATE
    ):
        reasons.append("same_entry_latency_eligibility_rate_not_met")
    if (
        metrics.retained_trade_ratio
        < ROUND74_AI_UPLIFT_MINIMUM_RETAINED_TRADE_RATIO[profile.profile]
    ):
        reasons.append("retained_trade_ratio_not_met")
    minimum_retained_trades = math.ceil(
        profile.minimum_trades
        * ROUND74_AI_UPLIFT_MINIMUM_RETAINED_TRADE_RATIO[profile.profile]
    )
    if metrics.retained_trades < minimum_retained_trades:
        reasons.append("minimum_retained_trades_not_met")
    if metrics.distinct_retained_symbols != len(ROUND74_EVENT_SYMBOLS):
        reasons.append("retained_asset_diversification_not_met")
    if metrics.maximum_retained_symbol_share > profile.maximum_symbol_trade_share:
        reasons.append("retained_symbol_concentration_not_met")
    if metrics.total_net_bps <= trace.metrics.total_net_bps:
        reasons.append("positive_paired_after_cost_uplift_not_met")
    if metrics.maximum_drawdown_bps > trace.metrics.maximum_drawdown_bps:
        reasons.append("maximum_drawdown_noninferiority_not_met")
    result = Round74AIUpliftDevelopmentReport(
        profile=profile.profile,
        action_selection_sha256=action_selection.selection_sha256,
        candidate_sha256=action_selection.candidate_sha256,
        pretest_policy_sha256=action_selection.pretest_policy_sha256,
        probability_calibration_sha256=(
            action_selection.probability_calibration_sha256
        ),
        model_manifest_sha256=next(iter(manifest_values)),
        same_entry_latency_budget_ns=next(iter(latency_budgets)),
        baseline_trace=trace,
        review_sha256=tuple(review.review_sha256 for review in review_rows),
        ai_scaled_net_payoff_bps=scaled,
        ai_scaled_maximum_adverse_excursion_bps=scaled_mae,
        paired_runs=paired_runs,
        ai_metrics=metrics,
        development_gate_passed=not reasons,
        gate_reasons=tuple(reasons),
    )
    result.validate()
    return result


__all__ = [
    "ROUND74_AI_UPLIFT_MINIMUM_RETAINED_TRADE_RATIO",
    "ROUND74_AI_UPLIFT_MINIMUM_RUNTIME_SUCCESS_RATE",
    "ROUND74_AI_UPLIFT_MINIMUM_SAME_ENTRY_LATENCY_ELIGIBILITY_RATE",
    "ROUND74_AI_UPLIFT_SCHEMA_VERSION",
    "Round74AIOverlayMetrics",
    "Round74AIPairedReviewEvidence",
    "Round74AIUpliftDevelopmentReport",
    "evaluate_round74_ai_overlay_development",
]
