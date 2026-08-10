"""Cluster-aware prospective uplift evaluation for the Round 25 AI hierarchy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Literal

import numpy as np

from .polymarket_round25_ai_supervisor import (
    POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256,
    POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_CONTRACT_SHA256,
)
from .polymarket_round25_candidate_design import POLYMARKET_ROUND25_CANDIDATE_IDS
from .polymarket_round25_twap_features import POLYMARKET_ROUND25_CONDITION_DURATION_MS


POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_MINIMUM_CONDITIONS = 500
POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_MINIMUM_WINDOWS = 120
POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_MINIMUM_FAST_INTERVENTIONS = 50
POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_MINIMUM_SLOW_INTERVENTIONS = 30
POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_MINIMUM_VALID_RATIO = 0.99
POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_BOOTSTRAP_REPLICATES = 10_000
POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_BOOTSTRAP_SEED = 25_026
POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_BLOCK_WINDOWS = 12
POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_ALPHA = 0.01
POLYMARKET_ROUND25_AI_SUPERVISOR_MATCHED_ROW_SCHEMA_VERSION = (
    "polymarket-round25-fin-r1-matched-condition-v1"
)
POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_PANEL_SCHEMA_VERSION = (
    "polymarket-round25-fin-r1-uplift-panel-v1"
)
POLYMARKET_ROUND25_AI_SUPERVISOR_COMPARISON_SCHEMA_VERSION = (
    "polymarket-round25-fin-r1-arm-comparison-v1"
)
POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_RESULT_SCHEMA_VERSION = (
    "polymarket-round25-fin-r1-uplift-result-v1"
)
Round25AIArm = Literal[
    "ml_control",
    "fast_qwen3_4b",
    "slow_fin_r1_8b",
    "hierarchical_minimum_risk",
]
_ARMS: tuple[Round25AIArm, ...] = (
    "ml_control",
    "fast_qwen3_4b",
    "slow_fin_r1_8b",
    "hierarchical_minimum_risk",
)
_COMPARISONS: tuple[tuple[Round25AIArm, Round25AIArm], ...] = (
    ("ml_control", "fast_qwen3_4b"),
    ("ml_control", "slow_fin_r1_8b"),
    ("fast_qwen3_4b", "hierarchical_minimum_risk"),
    ("ml_control", "hierarchical_minimum_risk"),
)
_PRIMARY = ("fast_qwen3_4b", "hierarchical_minimum_risk")
_SECONDARY = ("ml_control", "hierarchical_minimum_risk")
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
        raise ValueError(f"Round 25 AI supervisor uplift {name} is not numeric")
    selected = float(value)
    if not math.isfinite(selected) or selected <= -1.0:
        raise ValueError(f"Round 25 AI supervisor uplift {name} differs")
    return selected


@dataclass(frozen=True, slots=True)
class Round25AISupervisorMatchedCondition:
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    supervisor_window_start_ms: int
    supervisor_packet_sha256: str
    selected_candidate_id: str
    selected_model_prediction_sha256: str
    deterministic_decision_sha256: str
    matched_execution_scenario_sha256: str
    resolution_authority_sha256: str
    control_trace_sha256: str
    fast_trace_sha256: str
    slow_trace_sha256: str
    hierarchical_trace_sha256: str
    fast_advisory_sha256: str
    supervisor_advisory_sha256: str
    combined_decision_sha256: str
    control_after_cost_return: float
    fast_after_cost_return: float
    slow_after_cost_return: float
    hierarchical_after_cost_return: float
    fast_valid_response: bool
    slow_valid_response: bool
    fast_schema_or_coherence_violation: bool
    slow_schema_or_coherence_violation: bool
    fast_intervened: bool
    slow_intervened: bool
    row_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("row_sha256")
        return {
            **payload,
            "schema_version": POLYMARKET_ROUND25_AI_SUPERVISOR_MATCHED_ROW_SCHEMA_VERSION,
            "supervisor_contract_sha256": POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256,
            "supervisor_uplift_contract_sha256": (
                POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_CONTRACT_SHA256
            ),
            "paper_authority": False,
            "live_authority": False,
            "orders_submitted": False,
        }

    def validated(self) -> Round25AISupervisorMatchedCondition:
        returns = tuple(
            _finite_return(value, name=name)
            for name, value in (
                ("control return", self.control_after_cost_return),
                ("fast return", self.fast_after_cost_return),
                ("slow return", self.slow_after_cost_return),
                ("hierarchical return", self.hierarchical_after_cost_return),
            )
        )
        hashes = (
            self.supervisor_packet_sha256,
            self.selected_model_prediction_sha256,
            self.deterministic_decision_sha256,
            self.matched_execution_scenario_sha256,
            self.resolution_authority_sha256,
            self.control_trace_sha256,
            self.fast_trace_sha256,
            self.slow_trace_sha256,
            self.hierarchical_trace_sha256,
            self.fast_advisory_sha256,
            self.supervisor_advisory_sha256,
            self.combined_decision_sha256,
        )
        booleans = (
            self.fast_valid_response,
            self.slow_valid_response,
            self.fast_schema_or_coherence_violation,
            self.slow_schema_or_coherence_violation,
            self.fast_intervened,
            self.slow_intervened,
        )
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or isinstance(self.event_start_ms, bool)
            or not isinstance(self.event_start_ms, int)
            or self.event_start_ms < 0
            or self.event_start_ms % POLYMARKET_ROUND25_CONDITION_DURATION_MS != 0
            or isinstance(self.decision_time_ms, bool)
            or not isinstance(self.decision_time_ms, int)
            or isinstance(self.supervisor_window_start_ms, bool)
            or not isinstance(self.supervisor_window_start_ms, int)
            or self.supervisor_window_start_ms < 0
            or self.supervisor_window_start_ms % 60_000 != 0
            or not self.supervisor_window_start_ms
            <= self.decision_time_ms
            < self.supervisor_window_start_ms + 60_000
            or self.selected_candidate_id not in POLYMARKET_ROUND25_CANDIDATE_IDS[1:]
            or any(_SHA256.fullmatch(value) is None for value in hashes)
            or any(not isinstance(value, bool) for value in booleans)
            or self.fast_valid_response and self.fast_schema_or_coherence_violation
            or self.slow_valid_response and self.slow_schema_or_coherence_violation
        ):
            raise ValueError("Round 25 AI supervisor matched condition differs")
        object.__setattr__(self, "control_after_cost_return", returns[0])
        object.__setattr__(self, "fast_after_cost_return", returns[1])
        object.__setattr__(self, "slow_after_cost_return", returns[2])
        object.__setattr__(self, "hierarchical_after_cost_return", returns[3])
        expected = _canonical_sha256(self.identity_payload())
        if not self.row_sha256:
            object.__setattr__(self, "row_sha256", expected)
        elif self.row_sha256 != expected:
            raise ValueError("Round 25 AI supervisor matched condition hash differs")
        return self

    def __post_init__(self) -> None:
        self.validated()


@dataclass(frozen=True, slots=True)
class Round25AISupervisorUpliftPanel:
    selected_candidate_id: str
    selection_population_end_ms: int
    selection_condition_root_sha256: str
    uplift_condition_root_sha256: str
    rows: tuple[Round25AISupervisorMatchedCondition, ...]
    panel_sha256: str = ""
    schema_version: str = POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_PANEL_SCHEMA_VERSION

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selected_candidate_id": self.selected_candidate_id,
            "selection_population_end_ms": self.selection_population_end_ms,
            "selection_condition_root_sha256": self.selection_condition_root_sha256,
            "uplift_condition_root_sha256": self.uplift_condition_root_sha256,
            "row_sha256s": [row.row_sha256 for row in self.rows],
            "supervisor_uplift_contract_sha256": (
                POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_CONTRACT_SHA256
            ),
            "paper_authority": False,
            "live_authority": False,
            "orders_submitted": False,
        }

    def validated(self) -> Round25AISupervisorUpliftPanel:
        rows = tuple(row.validated() for row in self.rows)
        condition_ids = tuple(row.condition_id for row in rows)
        decision_times = tuple(row.decision_time_ms for row in rows)
        if (
            self.schema_version != POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_PANEL_SCHEMA_VERSION
            or self.selected_candidate_id not in POLYMARKET_ROUND25_CANDIDATE_IDS[1:]
            or isinstance(self.selection_population_end_ms, bool)
            or not isinstance(self.selection_population_end_ms, int)
            or self.selection_population_end_ms < 0
            or _SHA256.fullmatch(self.selection_condition_root_sha256) is None
            or _SHA256.fullmatch(self.uplift_condition_root_sha256) is None
            or self.selection_condition_root_sha256
            == self.uplift_condition_root_sha256
            or not rows
            or any(row.selected_candidate_id != self.selected_candidate_id for row in rows)
            or len(set(condition_ids)) != len(condition_ids)
            or decision_times != tuple(sorted(decision_times))
            or decision_times[0] <= self.selection_population_end_ms
        ):
            raise ValueError("Round 25 AI supervisor uplift panel differs")
        expected_root = _canonical_sha256(list(condition_ids))
        if self.uplift_condition_root_sha256 != expected_root:
            raise ValueError("Round 25 AI supervisor uplift condition root differs")
        expected = _canonical_sha256(self.identity_payload())
        if not self.panel_sha256:
            object.__setattr__(self, "panel_sha256", expected)
        elif self.panel_sha256 != expected:
            raise ValueError("Round 25 AI supervisor uplift panel hash differs")
        return self

    def __post_init__(self) -> None:
        self.validated()


def create_round25_ai_supervisor_uplift_panel(
    *,
    selected_candidate_id: str,
    selection_population_end_ms: int,
    selection_condition_root_sha256: str,
    rows: tuple[Round25AISupervisorMatchedCondition, ...],
) -> Round25AISupervisorUpliftPanel:
    selected_rows = tuple(row.validated() for row in rows)
    uplift_root = _canonical_sha256([row.condition_id for row in selected_rows])
    return Round25AISupervisorUpliftPanel(
        selected_candidate_id=selected_candidate_id,
        selection_population_end_ms=selection_population_end_ms,
        selection_condition_root_sha256=selection_condition_root_sha256,
        uplift_condition_root_sha256=uplift_root,
        rows=selected_rows,
    )


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
        raise ValueError("Round 25 AI supervisor compounded equity differs")
    peaks = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peaks)), float(equity[-1] - 1.0)


def _arm_returns(
    rows: tuple[Round25AISupervisorMatchedCondition, ...],
    arm: Round25AIArm,
) -> np.ndarray:
    field = {
        "ml_control": "control_after_cost_return",
        "fast_qwen3_4b": "fast_after_cost_return",
        "slow_fin_r1_8b": "slow_after_cost_return",
        "hierarchical_minimum_risk": "hierarchical_after_cost_return",
    }[arm]
    return np.asarray([getattr(row, field) for row in rows], dtype=np.float64)


def _window_row_indices(
    rows: tuple[Round25AISupervisorMatchedCondition, ...],
) -> tuple[np.ndarray, ...]:
    windows: list[list[int]] = []
    window_ids: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        window_id = row.supervisor_packet_sha256
        if window_id not in seen:
            seen.add(window_id)
            window_ids.append(window_id)
            windows.append([])
        elif window_ids[-1] != window_id:
            raise ValueError("Round 25 AI supervisor window is not contiguous")
        windows[-1].append(index)
    return tuple(np.asarray(indices, dtype=np.int64) for indices in windows)


def _bootstrap_deltas(
    baseline: np.ndarray,
    challenger: np.ndarray,
    windows: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    window_count = len(windows)
    block_count = math.ceil(
        window_count / POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_BLOCK_WINDOWS
    )
    offsets = np.arange(
        POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_BLOCK_WINDOWS,
        dtype=np.int64,
    )
    rng = np.random.Generator(
        np.random.PCG64(POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_BOOTSTRAP_SEED)
    )
    mean_delta = np.empty(
        POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_BOOTSTRAP_REPLICATES,
        dtype=np.float64,
    )
    es_delta = np.empty_like(mean_delta)
    for replicate in range(POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_BOOTSTRAP_REPLICATES):
        starts = rng.integers(0, window_count, size=block_count, dtype=np.int64)
        sampled_windows = (
            starts[:, None] + offsets[None, :]
        ).reshape(-1)[:window_count] % window_count
        indices = np.concatenate([windows[int(index)] for index in sampled_windows])
        baseline_sample = baseline[indices]
        challenger_sample = challenger[indices]
        mean_delta[replicate] = float(np.mean(challenger_sample - baseline_sample))
        es_delta[replicate] = _expected_shortfall_loss(
            challenger_sample
        ) - _expected_shortfall_loss(baseline_sample)
    return mean_delta, es_delta


@dataclass(frozen=True, slots=True)
class Round25AIArmComparison:
    baseline_arm: Round25AIArm
    challenger_arm: Round25AIArm
    baseline_mean_after_cost_return: float
    challenger_mean_after_cost_return: float
    mean_after_cost_return_delta: float
    mean_delta_ci_lower_99: float
    mean_delta_ci_upper_99: float
    baseline_expected_shortfall_95: float
    challenger_expected_shortfall_95: float
    expected_shortfall_delta: float
    expected_shortfall_delta_ci_lower_99: float
    expected_shortfall_delta_ci_upper_99: float
    baseline_maximum_drawdown: float
    challenger_maximum_drawdown: float
    maximum_drawdown_delta: float
    baseline_cumulative_net_return: float
    challenger_cumulative_net_return: float
    mean_bootstrap_sha256: str
    expected_shortfall_bootstrap_sha256: str
    comparison_gate_passed: bool
    comparison_sha256: str = ""
    schema_version: str = POLYMARKET_ROUND25_AI_SUPERVISOR_COMPARISON_SCHEMA_VERSION

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("comparison_sha256")
        return payload

    def validated(self) -> Round25AIArmComparison:
        numeric = tuple(
            value
            for key, value in asdict(self).items()
            if key not in {
                "baseline_arm",
                "challenger_arm",
                "mean_bootstrap_sha256",
                "expected_shortfall_bootstrap_sha256",
                "comparison_gate_passed",
                "comparison_sha256",
                "schema_version",
            }
        )
        expected_pass = (
            self.mean_delta_ci_lower_99 > 0.0
            and self.expected_shortfall_delta_ci_upper_99 <= 0.0
            and self.maximum_drawdown_delta <= 0.0
        )
        if (
            self.schema_version != POLYMARKET_ROUND25_AI_SUPERVISOR_COMPARISON_SCHEMA_VERSION
            or (self.baseline_arm, self.challenger_arm) not in _COMPARISONS
            or not all(math.isfinite(float(value)) for value in numeric)
            or not 0.0 <= self.baseline_maximum_drawdown <= 1.0
            or not 0.0 <= self.challenger_maximum_drawdown <= 1.0
            or _SHA256.fullmatch(self.mean_bootstrap_sha256) is None
            or _SHA256.fullmatch(self.expected_shortfall_bootstrap_sha256) is None
            or self.comparison_gate_passed is not expected_pass
        ):
            raise ValueError("Round 25 AI supervisor arm comparison differs")
        expected = _canonical_sha256(self.identity_payload())
        if not self.comparison_sha256:
            object.__setattr__(self, "comparison_sha256", expected)
        elif self.comparison_sha256 != expected:
            raise ValueError("Round 25 AI supervisor arm comparison hash differs")
        return self

    def __post_init__(self) -> None:
        self.validated()


def _compare_arms(
    *,
    rows: tuple[Round25AISupervisorMatchedCondition, ...],
    windows: tuple[np.ndarray, ...],
    baseline_arm: Round25AIArm,
    challenger_arm: Round25AIArm,
) -> Round25AIArmComparison:
    baseline = _arm_returns(rows, baseline_arm)
    challenger = _arm_returns(rows, challenger_arm)
    mean_bootstrap, es_bootstrap = _bootstrap_deltas(baseline, challenger, windows)
    lower = POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_ALPHA / 2.0
    upper = 1.0 - lower
    mean_lower = float(np.quantile(mean_bootstrap, lower, method="lower"))
    mean_upper = float(np.quantile(mean_bootstrap, upper, method="higher"))
    es_lower = float(np.quantile(es_bootstrap, lower, method="lower"))
    es_upper = float(np.quantile(es_bootstrap, upper, method="higher"))
    baseline_drawdown, baseline_cumulative = _maximum_drawdown(baseline)
    challenger_drawdown, challenger_cumulative = _maximum_drawdown(challenger)
    baseline_es = _expected_shortfall_loss(baseline)
    challenger_es = _expected_shortfall_loss(challenger)
    mean_baseline = float(np.mean(baseline))
    mean_challenger = float(np.mean(challenger))
    mean_delta = mean_challenger - mean_baseline
    es_delta = challenger_es - baseline_es
    drawdown_delta = challenger_drawdown - baseline_drawdown
    return Round25AIArmComparison(
        baseline_arm=baseline_arm,
        challenger_arm=challenger_arm,
        baseline_mean_after_cost_return=mean_baseline,
        challenger_mean_after_cost_return=mean_challenger,
        mean_after_cost_return_delta=mean_delta,
        mean_delta_ci_lower_99=mean_lower,
        mean_delta_ci_upper_99=mean_upper,
        baseline_expected_shortfall_95=baseline_es,
        challenger_expected_shortfall_95=challenger_es,
        expected_shortfall_delta=es_delta,
        expected_shortfall_delta_ci_lower_99=es_lower,
        expected_shortfall_delta_ci_upper_99=es_upper,
        baseline_maximum_drawdown=baseline_drawdown,
        challenger_maximum_drawdown=challenger_drawdown,
        maximum_drawdown_delta=drawdown_delta,
        baseline_cumulative_net_return=baseline_cumulative,
        challenger_cumulative_net_return=challenger_cumulative,
        mean_bootstrap_sha256=_array_sha256(mean_bootstrap),
        expected_shortfall_bootstrap_sha256=_array_sha256(es_bootstrap),
        comparison_gate_passed=(
            mean_lower > 0.0 and es_upper <= 0.0 and drawdown_delta <= 0.0
        ),
    )


def _population_gate_reasons(
    *,
    condition_count: int,
    window_count: int,
    fast_intervention_count: int,
    slow_intervention_count: int,
    fast_valid_ratio: float,
    slow_valid_ratio: float,
    violation_count: int,
    comparisons: tuple[Round25AIArmComparison, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if condition_count < POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_MINIMUM_CONDITIONS:
        reasons.append("minimum_condition_count_not_met")
    if window_count < POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_MINIMUM_WINDOWS:
        reasons.append("minimum_supervisor_window_count_not_met")
    if fast_intervention_count < POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_MINIMUM_FAST_INTERVENTIONS:
        reasons.append("minimum_fast_intervention_count_not_met")
    if slow_intervention_count < POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_MINIMUM_SLOW_INTERVENTIONS:
        reasons.append("minimum_supervisor_intervention_count_not_met")
    if fast_valid_ratio < POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_MINIMUM_VALID_RATIO:
        reasons.append("fast_valid_response_ratio_not_met")
    if slow_valid_ratio < POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_MINIMUM_VALID_RATIO:
        reasons.append("supervisor_valid_response_ratio_not_met")
    if violation_count > 0:
        reasons.append("schema_or_coherence_violation_observed")
    by_pair = {(item.baseline_arm, item.challenger_arm): item for item in comparisons}
    if not by_pair[_PRIMARY].comparison_gate_passed:
        reasons.append("incremental_supervisor_uplift_gate_not_met")
    if not by_pair[_SECONDARY].comparison_gate_passed:
        reasons.append("hierarchical_total_uplift_gate_not_met")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class Round25AISupervisorUpliftResult:
    panel_sha256: str
    selected_candidate_id: str
    condition_count: int
    supervisor_window_count: int
    fast_intervention_count: int
    slow_intervention_count: int
    fast_valid_response_ratio: float
    slow_valid_response_ratio: float
    schema_or_coherence_violation_count: int
    comparisons: tuple[Round25AIArmComparison, ...]
    gate_reasons: tuple[str, ...]
    development_nomination_passed: bool
    result_sha256: str = ""
    schema_version: str = POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_RESULT_SCHEMA_VERSION
    ai_uplift_verified: bool = False
    predictive_edge_verified: bool = False
    profitability_verified: bool = False
    paper_authority: bool = False
    live_authority: bool = False
    orders_submitted: bool = False

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("result_sha256")
        return payload

    def validated(self) -> Round25AISupervisorUpliftResult:
        comparisons = tuple(item.validated() for item in self.comparisons)
        expected_pairs = tuple(
            (item.baseline_arm, item.challenger_arm) for item in comparisons
        )
        violation_count = self.schema_or_coherence_violation_count
        expected_reasons = _population_gate_reasons(
            condition_count=self.condition_count,
            window_count=self.supervisor_window_count,
            fast_intervention_count=self.fast_intervention_count,
            slow_intervention_count=self.slow_intervention_count,
            fast_valid_ratio=self.fast_valid_response_ratio,
            slow_valid_ratio=self.slow_valid_response_ratio,
            violation_count=violation_count,
            comparisons=comparisons,
        )
        if (
            self.schema_version != POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_RESULT_SCHEMA_VERSION
            or _SHA256.fullmatch(self.panel_sha256) is None
            or self.selected_candidate_id not in POLYMARKET_ROUND25_CANDIDATE_IDS[1:]
            or self.condition_count < 1
            or not 1 <= self.supervisor_window_count <= self.condition_count
            or not 0 <= self.fast_intervention_count <= self.condition_count
            or not 0 <= self.slow_intervention_count <= self.condition_count
            or not 0.0 <= self.fast_valid_response_ratio <= 1.0
            or not 0.0 <= self.slow_valid_response_ratio <= 1.0
            or not 0 <= violation_count <= self.condition_count * 2
            or expected_pairs != _COMPARISONS
            or self.gate_reasons != expected_reasons
            or self.development_nomination_passed is not (not expected_reasons)
            or any(
                (
                    self.ai_uplift_verified,
                    self.predictive_edge_verified,
                    self.profitability_verified,
                    self.paper_authority,
                    self.live_authority,
                    self.orders_submitted,
                )
            )
        ):
            raise ValueError("Round 25 AI supervisor uplift result differs")
        expected = _canonical_sha256(self.identity_payload())
        if not self.result_sha256:
            object.__setattr__(self, "result_sha256", expected)
        elif self.result_sha256 != expected:
            raise ValueError("Round 25 AI supervisor uplift result hash differs")
        return self

    def __post_init__(self) -> None:
        self.validated()


def evaluate_round25_ai_supervisor_uplift(
    panel: Round25AISupervisorUpliftPanel,
) -> Round25AISupervisorUpliftResult:
    """Evaluate all four matched arms without granting trading authority."""

    selected = panel.validated()
    windows = _window_row_indices(selected.rows)
    comparisons = tuple(
        _compare_arms(
            rows=selected.rows,
            windows=windows,
            baseline_arm=baseline,
            challenger_arm=challenger,
        )
        for baseline, challenger in _COMPARISONS
    )
    condition_count = len(selected.rows)
    fast_interventions = sum(row.fast_intervened for row in selected.rows)
    slow_interventions = sum(row.slow_intervened for row in selected.rows)
    fast_valid_ratio = sum(row.fast_valid_response for row in selected.rows) / condition_count
    slow_valid_ratio = sum(row.slow_valid_response for row in selected.rows) / condition_count
    violations = sum(
        row.fast_schema_or_coherence_violation
        + row.slow_schema_or_coherence_violation
        for row in selected.rows
    )
    reasons = _population_gate_reasons(
        condition_count=condition_count,
        window_count=len(windows),
        fast_intervention_count=fast_interventions,
        slow_intervention_count=slow_interventions,
        fast_valid_ratio=fast_valid_ratio,
        slow_valid_ratio=slow_valid_ratio,
        violation_count=violations,
        comparisons=comparisons,
    )
    return Round25AISupervisorUpliftResult(
        panel_sha256=selected.panel_sha256,
        selected_candidate_id=selected.selected_candidate_id,
        condition_count=condition_count,
        supervisor_window_count=len(windows),
        fast_intervention_count=fast_interventions,
        slow_intervention_count=slow_interventions,
        fast_valid_response_ratio=fast_valid_ratio,
        slow_valid_response_ratio=slow_valid_ratio,
        schema_or_coherence_violation_count=violations,
        comparisons=comparisons,
        gate_reasons=reasons,
        development_nomination_passed=not reasons,
    )


__all__ = [
    "Round25AIArmComparison",
    "Round25AISupervisorMatchedCondition",
    "Round25AISupervisorUpliftPanel",
    "Round25AISupervisorUpliftResult",
    "create_round25_ai_supervisor_uplift_panel",
    "evaluate_round25_ai_supervisor_uplift",
]
