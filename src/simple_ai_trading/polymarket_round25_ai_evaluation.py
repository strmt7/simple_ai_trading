"""Disjoint prospective matched evaluation of the Round 25 AI risk overlay."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import numpy as np

from .polymarket_round25_ai import (
    POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256,
    Round25AIReviewResult,
)
from .polymarket_round25_candidate_design import POLYMARKET_ROUND25_CANDIDATE_IDS
from .polymarket_round25_evaluation import Round25PredictiveEvaluationResult
from .polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_CONDITION_DURATION_MS,
)


POLYMARKET_ROUND25_AI_UPLIFT_CONTRACT_V1_SHA256 = (
    "117b17da6f31d6d90d2730cae10d003cd424bcad9455fdabb6c16dc7cf61cbdc"
)
POLYMARKET_ROUND25_AI_UPLIFT_CONTRACT_SHA256 = (
    "47a51c97d723b37f278e1ef1f5860acae187d3a4747158d8525d2f16734d432a"
)
POLYMARKET_ROUND25_AI_MATCHED_CONDITION_SCHEMA_VERSION = (
    "polymarket-round25-ai-matched-replay-condition-v1"
)
POLYMARKET_ROUND25_AI_UPLIFT_PANEL_SCHEMA_VERSION = (
    "polymarket-round25-ai-uplift-panel-v1"
)
POLYMARKET_ROUND25_AI_UPLIFT_RESULT_SCHEMA_VERSION = (
    "polymarket-round25-ai-uplift-result-v1"
)
POLYMARKET_ROUND25_AI_UPLIFT_MINIMUM_CONDITIONS = 500
POLYMARKET_ROUND25_AI_UPLIFT_MINIMUM_INTERVENTIONS = 50
POLYMARKET_ROUND25_AI_UPLIFT_MINIMUM_VALID_RESPONSE_RATIO = 0.99
POLYMARKET_ROUND25_AI_UPLIFT_BOOTSTRAP_REPLICATES = 10_000
POLYMARKET_ROUND25_AI_UPLIFT_BOOTSTRAP_SEED = 25_025
POLYMARKET_ROUND25_AI_UPLIFT_BLOCK_CONDITIONS = 12
POLYMARKET_ROUND25_AI_UPLIFT_ALPHA = 0.05
POLYMARKET_ROUND25_AI_UPLIFT_BOOTSTRAP_CHUNK = 128
POLYMARKET_ROUND25_AI_UPLIFT_GATE_REASONS = (
    "minimum_condition_count_not_met",
    "minimum_intervention_count_not_met",
    "valid_response_ratio_not_met",
    "schema_or_coherence_violation_observed",
    "paired_after_cost_return_uplift_confidence_not_met",
    "expected_shortfall_noninferiority_not_met",
    "maximum_drawdown_noninferiority_not_met",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")


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


def _array_sha256(value: np.ndarray) -> str:
    selected = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(selected.tobytes(order="C")).hexdigest()


def _finite_return(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Round 25 AI uplift {name} is not numeric")
    selected = float(value)
    if not math.isfinite(selected) or selected <= -1.0:
        raise ValueError(f"Round 25 AI uplift {name} is outside its domain")
    return selected


@dataclass(frozen=True, slots=True)
class Round25AIMatchedReplayCondition:
    condition_id: str
    event_start_ms: int
    selected_candidate_id: str
    selected_model_prediction_sha256: str
    deterministic_decision_sha256: str
    matched_execution_scenario_sha256: str
    resolution_authority_sha256: str
    control_trace_sha256: str
    ai_trace_sha256: str
    ai_advisory_sha256: str
    control_after_cost_return: float
    ai_after_cost_return: float
    valid_model_response: bool
    schema_or_coherence_violation: bool
    ai_intervened: bool
    ai_veto_new_entries: bool
    ai_size_multiplier: float
    ai_cooldown_ms: int
    ai_response_latency_ms: float | None
    row_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        return {
            "ai_advisory_contract_sha256": (
                POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256
            ),
            "ai_advisory_sha256": self.ai_advisory_sha256,
            "ai_after_cost_return": self.ai_after_cost_return,
            "ai_cooldown_ms": self.ai_cooldown_ms,
            "ai_intervened": self.ai_intervened,
            "ai_response_latency_ms": self.ai_response_latency_ms,
            "ai_size_multiplier": self.ai_size_multiplier,
            "ai_trace_sha256": self.ai_trace_sha256,
            "ai_veto_new_entries": self.ai_veto_new_entries,
            "condition_id": self.condition_id,
            "control_after_cost_return": self.control_after_cost_return,
            "control_trace_sha256": self.control_trace_sha256,
            "deterministic_decision_sha256": self.deterministic_decision_sha256,
            "event_start_ms": self.event_start_ms,
            "live_authority": False,
            "matched_execution_scenario_sha256": (
                self.matched_execution_scenario_sha256
            ),
            "orders_submitted": False,
            "paper_authority": False,
            "resolution_authority_sha256": self.resolution_authority_sha256,
            "schema_or_coherence_violation": self.schema_or_coherence_violation,
            "schema_version": POLYMARKET_ROUND25_AI_MATCHED_CONDITION_SCHEMA_VERSION,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_model_prediction_sha256": (
                self.selected_model_prediction_sha256
            ),
            "valid_model_response": self.valid_model_response,
        }

    def __post_init__(self) -> None:
        control_return = _finite_return(
            self.control_after_cost_return,
            name="control return",
        )
        ai_return = _finite_return(self.ai_after_cost_return, name="AI return")
        multiplier = float(self.ai_size_multiplier)
        expected_intervention = bool(
            self.ai_veto_new_entries
            or multiplier < 1.0
            or self.ai_cooldown_ms > 0
        )
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or self.event_start_ms < 0
            or self.event_start_ms % POLYMARKET_ROUND25_CONDITION_DURATION_MS != 0
            or self.selected_candidate_id not in POLYMARKET_ROUND25_CANDIDATE_IDS[1:]
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.selected_model_prediction_sha256,
                    self.deterministic_decision_sha256,
                    self.matched_execution_scenario_sha256,
                    self.resolution_authority_sha256,
                    self.control_trace_sha256,
                    self.ai_trace_sha256,
                    self.ai_advisory_sha256,
                )
            )
            or not isinstance(self.valid_model_response, bool)
            or not isinstance(self.schema_or_coherence_violation, bool)
            or self.valid_model_response and self.schema_or_coherence_violation
            or not isinstance(self.ai_veto_new_entries, bool)
            or not math.isfinite(multiplier)
            or not 0.0 <= multiplier <= 1.0
            or isinstance(self.ai_cooldown_ms, bool)
            or not isinstance(self.ai_cooldown_ms, int)
            or not 0 <= self.ai_cooldown_ms <= 300_000
            or (multiplier == 0.0) is not self.ai_veto_new_entries
            or self.ai_intervened is not expected_intervention
            or (
                self.valid_model_response
                and (
                    self.ai_response_latency_ms is None
                    or not math.isfinite(float(self.ai_response_latency_ms))
                    or not 0.0 <= float(self.ai_response_latency_ms) <= 10_000.0
                )
            )
            or (not self.valid_model_response and self.ai_response_latency_ms is not None)
        ):
            raise ValueError("Round 25 AI matched replay condition differs")
        object.__setattr__(self, "control_after_cost_return", control_return)
        object.__setattr__(self, "ai_after_cost_return", ai_return)
        object.__setattr__(self, "ai_size_multiplier", multiplier)
        expected = _canonical_sha256(self.identity_payload())
        if not self.row_sha256:
            object.__setattr__(self, "row_sha256", expected)
        elif self.row_sha256 != expected:
            raise ValueError("Round 25 AI matched replay condition hash differs")

    def validated(self) -> Round25AIMatchedReplayCondition:
        self.__post_init__()
        return self


def create_round25_ai_matched_condition(
    *,
    condition_id: str,
    event_start_ms: int,
    selected_candidate_id: str,
    selected_model_prediction_sha256: str,
    deterministic_decision_sha256: str,
    matched_execution_scenario_sha256: str,
    resolution_authority_sha256: str,
    control_trace_sha256: str,
    ai_trace_sha256: str,
    control_after_cost_return: float,
    ai_after_cost_return: float,
    ai_review: Round25AIReviewResult,
) -> Round25AIMatchedReplayCondition:
    """Bind one completed matched replay to its validated AI review."""

    result = ai_review.validated()
    advisory = result.advisory
    latency_ms = None
    if result.telemetry is not None:
        latency_ms = result.telemetry.measured_latency_seconds * 1_000.0
    return Round25AIMatchedReplayCondition(
        condition_id=condition_id,
        event_start_ms=event_start_ms,
        selected_candidate_id=selected_candidate_id,
        selected_model_prediction_sha256=selected_model_prediction_sha256,
        deterministic_decision_sha256=deterministic_decision_sha256,
        matched_execution_scenario_sha256=matched_execution_scenario_sha256,
        resolution_authority_sha256=resolution_authority_sha256,
        control_trace_sha256=control_trace_sha256,
        ai_trace_sha256=ai_trace_sha256,
        ai_advisory_sha256=advisory.advisory_sha256,
        control_after_cost_return=control_after_cost_return,
        ai_after_cost_return=ai_after_cost_return,
        valid_model_response=advisory.valid_model_response,
        schema_or_coherence_violation=advisory.failure_code == "schema_failure",
        ai_intervened=(
            advisory.veto_new_entries
            or advisory.maximum_size_multiplier < 1.0
            or advisory.cooldown_ms > 0
        ),
        ai_veto_new_entries=advisory.veto_new_entries,
        ai_size_multiplier=advisory.maximum_size_multiplier,
        ai_cooldown_ms=advisory.cooldown_ms,
        ai_response_latency_ms=latency_ms,
    )


@dataclass(frozen=True, slots=True)
class Round25AIUpliftPanel:
    selected_candidate_id: str
    predictive_result_sha256: str
    selection_dataset_sha256: str
    target_access_receipt_sha256: str
    resolution_authority_sha256: str
    selection_condition_count: int
    selection_end_ms: int
    selection_condition_root_sha256: str
    rows: tuple[Round25AIMatchedReplayCondition, ...]
    uplift_population_root_sha256: str
    panel_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "ai_uplift_contract_sha256": POLYMARKET_ROUND25_AI_UPLIFT_CONTRACT_SHA256,
            "live_authority": False,
            "orders_submitted": False,
            "paper_authority": False,
            "predictive_result_sha256": self.predictive_result_sha256,
            "resolution_authority_sha256": self.resolution_authority_sha256,
            "row_sha256": [row.row_sha256 for row in self.rows],
            "schema_version": POLYMARKET_ROUND25_AI_UPLIFT_PANEL_SCHEMA_VERSION,
            "selected_candidate_id": self.selected_candidate_id,
            "selection_condition_count": self.selection_condition_count,
            "selection_condition_root_sha256": self.selection_condition_root_sha256,
            "selection_dataset_sha256": self.selection_dataset_sha256,
            "selection_end_ms": self.selection_end_ms,
            "target_access_receipt_sha256": self.target_access_receipt_sha256,
            "uplift_population_root_sha256": self.uplift_population_root_sha256,
        }

    def __post_init__(self) -> None:
        rows = tuple(row.validated() for row in self.rows)
        if (
            not rows
            or tuple((row.event_start_ms, row.condition_id) for row in rows)
            != tuple(sorted((row.event_start_ms, row.condition_id) for row in rows))
            or len({row.condition_id for row in rows}) != len(rows)
            or len({row.row_sha256 for row in rows}) != len(rows)
            or any(row.event_start_ms <= self.selection_end_ms for row in rows)
            or any(row.selected_candidate_id != self.selected_candidate_id for row in rows)
            or any(
                row.resolution_authority_sha256 != self.resolution_authority_sha256
                for row in rows
            )
            or self.selected_candidate_id not in POLYMARKET_ROUND25_CANDIDATE_IDS[1:]
            or self.selection_condition_count < 1
            or self.selection_end_ms < 0
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.predictive_result_sha256,
                    self.selection_dataset_sha256,
                    self.target_access_receipt_sha256,
                    self.resolution_authority_sha256,
                    self.selection_condition_root_sha256,
                    self.uplift_population_root_sha256,
                    self.panel_sha256,
                )
            )
            or self.uplift_population_root_sha256
            != _canonical_sha256([row.row_sha256 for row in rows])
            or self.panel_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 AI uplift panel differs")

    def validated(self) -> Round25AIUpliftPanel:
        self.__post_init__()
        return self


def create_round25_ai_uplift_panel(
    *,
    predictive_result: Round25PredictiveEvaluationResult,
    selection_conditions: Sequence[tuple[str, int]],
    rows: Sequence[Round25AIMatchedReplayCondition],
) -> Round25AIUpliftPanel:
    """Create a hash-bound panel only from conditions after the selection period."""

    result = predictive_result.validated()
    if not result.predictive_gate_passed or result.nominated_candidate_id is None:
        raise ValueError("Round 25 AI uplift requires a frozen predictive nomination")
    selection = tuple((str(condition_id), int(start_ms)) for condition_id, start_ms in selection_conditions)
    if (
        not selection
        or len({condition_id for condition_id, _ in selection}) != len(selection)
        or any(_CONDITION_ID.fullmatch(condition_id) is None for condition_id, _ in selection)
        or any(
            start_ms < 0 or start_ms % POLYMARKET_ROUND25_CONDITION_DURATION_MS != 0
            for _, start_ms in selection
        )
    ):
        raise ValueError("Round 25 AI selection condition identity differs")
    ordered_selection = tuple(sorted(selection, key=lambda value: (value[1], value[0])))
    selected_rows = tuple(
        sorted(
            (row.validated() for row in rows),
            key=lambda row: (row.event_start_ms, row.condition_id),
        )
    )
    if (
        not selected_rows
        or {condition_id for condition_id, _ in ordered_selection}
        & {row.condition_id for row in selected_rows}
    ):
        raise ValueError("Round 25 AI uplift population is not disjoint")
    selection_end = max(start_ms for _, start_ms in ordered_selection)
    if any(row.event_start_ms <= selection_end for row in selected_rows):
        raise ValueError("Round 25 AI uplift population does not follow selection")
    selection_root = _canonical_sha256([
        {"condition_id": condition_id, "event_start_ms": start_ms}
        for condition_id, start_ms in ordered_selection
    ])
    uplift_root = _canonical_sha256([row.row_sha256 for row in selected_rows])
    values = {
        "ai_uplift_contract_sha256": POLYMARKET_ROUND25_AI_UPLIFT_CONTRACT_SHA256,
        "live_authority": False,
        "orders_submitted": False,
        "paper_authority": False,
        "predictive_result_sha256": result.result_sha256,
        "resolution_authority_sha256": result.resolution_authority_sha256,
        "row_sha256": [row.row_sha256 for row in selected_rows],
        "schema_version": POLYMARKET_ROUND25_AI_UPLIFT_PANEL_SCHEMA_VERSION,
        "selected_candidate_id": result.nominated_candidate_id,
        "selection_condition_count": len(ordered_selection),
        "selection_condition_root_sha256": selection_root,
        "selection_dataset_sha256": result.selection_dataset_sha256,
        "selection_end_ms": selection_end,
        "target_access_receipt_sha256": result.target_access_receipt_sha256,
        "uplift_population_root_sha256": uplift_root,
    }
    return Round25AIUpliftPanel(
        selected_candidate_id=result.nominated_candidate_id,
        predictive_result_sha256=result.result_sha256,
        selection_dataset_sha256=result.selection_dataset_sha256,
        target_access_receipt_sha256=result.target_access_receipt_sha256,
        resolution_authority_sha256=result.resolution_authority_sha256,
        selection_condition_count=len(ordered_selection),
        selection_end_ms=selection_end,
        selection_condition_root_sha256=selection_root,
        rows=selected_rows,
        uplift_population_root_sha256=uplift_root,
        panel_sha256=_canonical_sha256(values),
    ).validated()


def _expected_shortfall_loss(returns: np.ndarray) -> float:
    tail_count = max(1, math.ceil(len(returns) * 0.05))
    losses = -np.asarray(returns, dtype=np.float64)
    threshold = len(losses) - tail_count
    return float(np.mean(np.partition(losses, threshold)[threshold:]))


def _maximum_drawdown(returns: np.ndarray) -> tuple[float, float]:
    equity = np.concatenate((
        np.ones(1, dtype=np.float64),
        np.cumprod(1.0 + np.asarray(returns, dtype=np.float64)),
    ))
    if not np.all(np.isfinite(equity)) or np.any(equity <= 0.0):
        raise ValueError("Round 25 AI uplift compounded equity differs")
    peaks = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peaks)), float(equity[-1] - 1.0)


def _bootstrap_deltas(
    control: np.ndarray,
    challenger: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    condition_count = len(control)
    block_count = math.ceil(
        condition_count / POLYMARKET_ROUND25_AI_UPLIFT_BLOCK_CONDITIONS
    )
    offsets = np.arange(
        POLYMARKET_ROUND25_AI_UPLIFT_BLOCK_CONDITIONS,
        dtype=np.int64,
    )
    rng = np.random.Generator(np.random.PCG64(POLYMARKET_ROUND25_AI_UPLIFT_BOOTSTRAP_SEED))
    pnl_delta = np.empty(POLYMARKET_ROUND25_AI_UPLIFT_BOOTSTRAP_REPLICATES, dtype=np.float64)
    es_delta = np.empty_like(pnl_delta)
    tail_count = max(1, math.ceil(condition_count * 0.05))
    threshold = condition_count - tail_count
    for first in range(
        0,
        POLYMARKET_ROUND25_AI_UPLIFT_BOOTSTRAP_REPLICATES,
        POLYMARKET_ROUND25_AI_UPLIFT_BOOTSTRAP_CHUNK,
    ):
        count = min(
            POLYMARKET_ROUND25_AI_UPLIFT_BOOTSTRAP_CHUNK,
            POLYMARKET_ROUND25_AI_UPLIFT_BOOTSTRAP_REPLICATES - first,
        )
        starts = rng.integers(
            0,
            condition_count,
            size=(count, block_count),
            dtype=np.int64,
        )
        indices = (starts[:, :, None] + offsets[None, None, :]) % condition_count
        indices = indices.reshape(count, -1)[:, :condition_count]
        control_sample = control[indices]
        challenger_sample = challenger[indices]
        pnl_delta[first : first + count] = np.mean(
            challenger_sample - control_sample,
            axis=1,
        )
        control_losses = np.partition(-control_sample, threshold, axis=1)[
            :, threshold:
        ]
        challenger_losses = np.partition(-challenger_sample, threshold, axis=1)[
            :, threshold:
        ]
        es_delta[first : first + count] = np.mean(
            challenger_losses,
            axis=1,
        ) - np.mean(control_losses, axis=1)
    return pnl_delta, es_delta


def _gate_reasons(
    *,
    condition_count: int,
    intervention_count: int,
    valid_response_ratio: float,
    schema_or_coherence_violation_count: int,
    pnl_delta_lower: float,
    expected_shortfall_delta_upper: float,
    maximum_drawdown_delta: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if condition_count < POLYMARKET_ROUND25_AI_UPLIFT_MINIMUM_CONDITIONS:
        reasons.append("minimum_condition_count_not_met")
    if intervention_count < POLYMARKET_ROUND25_AI_UPLIFT_MINIMUM_INTERVENTIONS:
        reasons.append("minimum_intervention_count_not_met")
    if valid_response_ratio < POLYMARKET_ROUND25_AI_UPLIFT_MINIMUM_VALID_RESPONSE_RATIO:
        reasons.append("valid_response_ratio_not_met")
    if schema_or_coherence_violation_count > 0:
        reasons.append("schema_or_coherence_violation_observed")
    if pnl_delta_lower <= 0.0:
        reasons.append("paired_after_cost_return_uplift_confidence_not_met")
    if expected_shortfall_delta_upper > 0.0:
        reasons.append("expected_shortfall_noninferiority_not_met")
    if maximum_drawdown_delta > 0.0:
        reasons.append("maximum_drawdown_noninferiority_not_met")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class Round25AIUpliftResult:
    panel_sha256: str
    selected_candidate_id: str
    condition_count: int
    intervention_count: int
    valid_response_ratio: float
    schema_or_coherence_violation_count: int
    control_mean_after_cost_return: float
    ai_mean_after_cost_return: float
    paired_mean_after_cost_return_delta: float
    paired_mean_after_cost_return_delta_ci_lower: float
    paired_mean_after_cost_return_delta_ci_upper: float
    control_expected_shortfall_95: float
    ai_expected_shortfall_95: float
    expected_shortfall_95_delta: float
    expected_shortfall_95_delta_ci_lower: float
    expected_shortfall_95_delta_ci_upper: float
    control_cumulative_net_return: float
    ai_cumulative_net_return: float
    control_maximum_drawdown: float
    ai_maximum_drawdown: float
    maximum_drawdown_delta: float
    pnl_bootstrap_sha256: str
    expected_shortfall_bootstrap_sha256: str
    gate_reasons: tuple[str, ...]
    development_uplift_gate_passed: bool
    result_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "ai_cumulative_net_return": self.ai_cumulative_net_return,
            "ai_expected_shortfall_95": self.ai_expected_shortfall_95,
            "ai_mean_after_cost_return": self.ai_mean_after_cost_return,
            "ai_maximum_drawdown": self.ai_maximum_drawdown,
            "ai_uplift_contract_sha256": POLYMARKET_ROUND25_AI_UPLIFT_CONTRACT_SHA256,
            "ai_uplift_verified": False,
            "condition_count": self.condition_count,
            "control_cumulative_net_return": self.control_cumulative_net_return,
            "control_expected_shortfall_95": self.control_expected_shortfall_95,
            "control_mean_after_cost_return": self.control_mean_after_cost_return,
            "control_maximum_drawdown": self.control_maximum_drawdown,
            "development_evidence_only": True,
            "development_uplift_gate_passed": self.development_uplift_gate_passed,
            "expected_shortfall_95_delta": self.expected_shortfall_95_delta,
            "expected_shortfall_95_delta_ci_lower": self.expected_shortfall_95_delta_ci_lower,
            "expected_shortfall_95_delta_ci_upper": self.expected_shortfall_95_delta_ci_upper,
            "expected_shortfall_bootstrap_sha256": self.expected_shortfall_bootstrap_sha256,
            "gate_reasons": list(self.gate_reasons),
            "intervention_count": self.intervention_count,
            "live_authority": False,
            "maximum_drawdown_delta": self.maximum_drawdown_delta,
            "orders_submitted": False,
            "paired_mean_after_cost_return_delta": self.paired_mean_after_cost_return_delta,
            "paired_mean_after_cost_return_delta_ci_lower": self.paired_mean_after_cost_return_delta_ci_lower,
            "paired_mean_after_cost_return_delta_ci_upper": self.paired_mean_after_cost_return_delta_ci_upper,
            "panel_sha256": self.panel_sha256,
            "paper_authority": False,
            "pnl_bootstrap_sha256": self.pnl_bootstrap_sha256,
            "profitability_verified": False,
            "schema_or_coherence_violation_count": self.schema_or_coherence_violation_count,
            "schema_version": POLYMARKET_ROUND25_AI_UPLIFT_RESULT_SCHEMA_VERSION,
            "selected_candidate_id": self.selected_candidate_id,
            "valid_response_ratio": self.valid_response_ratio,
        }

    def __post_init__(self) -> None:
        expected_reasons = _gate_reasons(
            condition_count=self.condition_count,
            intervention_count=self.intervention_count,
            valid_response_ratio=self.valid_response_ratio,
            schema_or_coherence_violation_count=(
                self.schema_or_coherence_violation_count
            ),
            pnl_delta_lower=self.paired_mean_after_cost_return_delta_ci_lower,
            expected_shortfall_delta_upper=(
                self.expected_shortfall_95_delta_ci_upper
            ),
            maximum_drawdown_delta=self.maximum_drawdown_delta,
        )
        numeric = (
            self.valid_response_ratio,
            self.control_mean_after_cost_return,
            self.ai_mean_after_cost_return,
            self.paired_mean_after_cost_return_delta,
            self.paired_mean_after_cost_return_delta_ci_lower,
            self.paired_mean_after_cost_return_delta_ci_upper,
            self.control_expected_shortfall_95,
            self.ai_expected_shortfall_95,
            self.expected_shortfall_95_delta,
            self.expected_shortfall_95_delta_ci_lower,
            self.expected_shortfall_95_delta_ci_upper,
            self.control_cumulative_net_return,
            self.ai_cumulative_net_return,
            self.control_maximum_drawdown,
            self.ai_maximum_drawdown,
            self.maximum_drawdown_delta,
        )
        if (
            _SHA256.fullmatch(self.panel_sha256) is None
            or self.selected_candidate_id not in POLYMARKET_ROUND25_CANDIDATE_IDS[1:]
            or self.condition_count < 1
            or not 0 <= self.intervention_count <= self.condition_count
            or not 0 <= self.schema_or_coherence_violation_count <= self.condition_count
            or not all(math.isfinite(value) for value in numeric)
            or not 0.0 <= self.valid_response_ratio <= 1.0
            or not 0.0 <= self.control_maximum_drawdown <= 1.0
            or not 0.0 <= self.ai_maximum_drawdown <= 1.0
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.pnl_bootstrap_sha256,
                    self.expected_shortfall_bootstrap_sha256,
                    self.result_sha256,
                )
            )
            or self.gate_reasons != expected_reasons
            or self.development_uplift_gate_passed is not (not expected_reasons)
            or self.result_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 AI uplift result differs")

    def validated(self) -> Round25AIUpliftResult:
        self.__post_init__()
        return self


def evaluate_round25_ai_uplift(
    panel: Round25AIUpliftPanel,
) -> Round25AIUpliftResult:
    """Evaluate one frozen, chronological, disjoint prospective AI panel."""

    selected = panel.validated()
    control = np.asarray(
        [row.control_after_cost_return for row in selected.rows],
        dtype=np.float64,
    )
    challenger = np.asarray(
        [row.ai_after_cost_return for row in selected.rows],
        dtype=np.float64,
    )
    pnl_bootstrap, es_bootstrap = _bootstrap_deltas(control, challenger)
    lower_probability = POLYMARKET_ROUND25_AI_UPLIFT_ALPHA / 2.0
    upper_probability = 1.0 - lower_probability
    pnl_lower = float(np.quantile(pnl_bootstrap, lower_probability, method="lower"))
    pnl_upper = float(np.quantile(pnl_bootstrap, upper_probability, method="higher"))
    es_lower = float(np.quantile(es_bootstrap, lower_probability, method="lower"))
    es_upper = float(np.quantile(es_bootstrap, upper_probability, method="higher"))
    control_drawdown, control_cumulative = _maximum_drawdown(control)
    ai_drawdown, ai_cumulative = _maximum_drawdown(challenger)
    control_es = _expected_shortfall_loss(control)
    ai_es = _expected_shortfall_loss(challenger)
    condition_count = len(selected.rows)
    intervention_count = sum(row.ai_intervened for row in selected.rows)
    valid_ratio = sum(row.valid_model_response for row in selected.rows) / condition_count
    violation_count = sum(
        row.schema_or_coherence_violation for row in selected.rows
    )
    mean_control = float(np.mean(control))
    mean_ai = float(np.mean(challenger))
    mean_delta = mean_ai - mean_control
    es_delta = ai_es - control_es
    drawdown_delta = ai_drawdown - control_drawdown
    reasons = _gate_reasons(
        condition_count=condition_count,
        intervention_count=intervention_count,
        valid_response_ratio=valid_ratio,
        schema_or_coherence_violation_count=violation_count,
        pnl_delta_lower=pnl_lower,
        expected_shortfall_delta_upper=es_upper,
        maximum_drawdown_delta=drawdown_delta,
    )
    values = {
        "ai_cumulative_net_return": ai_cumulative,
        "ai_expected_shortfall_95": ai_es,
        "ai_mean_after_cost_return": mean_ai,
        "ai_maximum_drawdown": ai_drawdown,
        "ai_uplift_contract_sha256": POLYMARKET_ROUND25_AI_UPLIFT_CONTRACT_SHA256,
        "ai_uplift_verified": False,
        "condition_count": condition_count,
        "control_cumulative_net_return": control_cumulative,
        "control_expected_shortfall_95": control_es,
        "control_mean_after_cost_return": mean_control,
        "control_maximum_drawdown": control_drawdown,
        "development_evidence_only": True,
        "development_uplift_gate_passed": not reasons,
        "expected_shortfall_95_delta": es_delta,
        "expected_shortfall_95_delta_ci_lower": es_lower,
        "expected_shortfall_95_delta_ci_upper": es_upper,
        "expected_shortfall_bootstrap_sha256": _array_sha256(es_bootstrap),
        "gate_reasons": list(reasons),
        "intervention_count": intervention_count,
        "live_authority": False,
        "maximum_drawdown_delta": drawdown_delta,
        "orders_submitted": False,
        "paired_mean_after_cost_return_delta": mean_delta,
        "paired_mean_after_cost_return_delta_ci_lower": pnl_lower,
        "paired_mean_after_cost_return_delta_ci_upper": pnl_upper,
        "panel_sha256": selected.panel_sha256,
        "paper_authority": False,
        "pnl_bootstrap_sha256": _array_sha256(pnl_bootstrap),
        "profitability_verified": False,
        "schema_or_coherence_violation_count": violation_count,
        "schema_version": POLYMARKET_ROUND25_AI_UPLIFT_RESULT_SCHEMA_VERSION,
        "selected_candidate_id": selected.selected_candidate_id,
        "valid_response_ratio": valid_ratio,
    }
    return Round25AIUpliftResult(
        panel_sha256=selected.panel_sha256,
        selected_candidate_id=selected.selected_candidate_id,
        condition_count=condition_count,
        intervention_count=intervention_count,
        valid_response_ratio=valid_ratio,
        schema_or_coherence_violation_count=violation_count,
        control_mean_after_cost_return=mean_control,
        ai_mean_after_cost_return=mean_ai,
        paired_mean_after_cost_return_delta=mean_delta,
        paired_mean_after_cost_return_delta_ci_lower=pnl_lower,
        paired_mean_after_cost_return_delta_ci_upper=pnl_upper,
        control_expected_shortfall_95=control_es,
        ai_expected_shortfall_95=ai_es,
        expected_shortfall_95_delta=es_delta,
        expected_shortfall_95_delta_ci_lower=es_lower,
        expected_shortfall_95_delta_ci_upper=es_upper,
        control_cumulative_net_return=control_cumulative,
        ai_cumulative_net_return=ai_cumulative,
        control_maximum_drawdown=control_drawdown,
        ai_maximum_drawdown=ai_drawdown,
        maximum_drawdown_delta=drawdown_delta,
        pnl_bootstrap_sha256=str(values["pnl_bootstrap_sha256"]),
        expected_shortfall_bootstrap_sha256=str(
            values["expected_shortfall_bootstrap_sha256"]
        ),
        gate_reasons=reasons,
        development_uplift_gate_passed=not reasons,
        result_sha256=_canonical_sha256(values),
    ).validated()


credentials_used = False
account_connected = False
binance_execution_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND25_AI_UPLIFT_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_AI_UPLIFT_CONTRACT_V1_SHA256",
    "POLYMARKET_ROUND25_AI_UPLIFT_MINIMUM_CONDITIONS",
    "POLYMARKET_ROUND25_AI_UPLIFT_MINIMUM_INTERVENTIONS",
    "Round25AIMatchedReplayCondition",
    "Round25AIUpliftPanel",
    "Round25AIUpliftResult",
    "create_round25_ai_matched_condition",
    "create_round25_ai_uplift_panel",
    "evaluate_round25_ai_uplift",
]
