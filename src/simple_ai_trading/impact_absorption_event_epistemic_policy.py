"""Target-free ensemble-disagreement filter for Round 74 action candidates.

Thresholds are fitted only after the disjoint tuning policy-selection report
supports uncertainty ordering. Runtime application consumes no realized target
field and can only turn an existing candidate into an abstention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import numpy as np
import torch

from .impact_absorption_event_action_policy import (
    ROUND74_ACTION_HORIZONS_SECONDS,
    ROUND74_ACTION_PROFILES,
    Round74ActionCandidateBatch,
    round74_action_model_output_sha256,
)
from .impact_absorption_event_epistemic_evaluation import (
    ROUND74_EPISTEMIC_MINIMUM_STRATUM_ROWS,
    ROUND74_EPISTEMIC_REQUIRED_POLICY_SELECTION_RUNS,
    Round74EpistemicEvaluationBatch,
    Round74EpistemicRiskCoverageReport,
)
from .impact_absorption_event_model import Round74EventModelOutput
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SYMBOLS,
)


ROUND74_EPISTEMIC_ACTION_FILTER_SCHEMA_VERSION = (
    "round-074-epistemic-action-filter-v1"
)
ROUND74_EPISTEMIC_ACTION_FILTER_APPLICATION_SCHEMA_VERSION = (
    "round-074-epistemic-action-filter-application-v1"
)
ROUND74_EPISTEMIC_ACTION_FILTER_COMPONENTS = (
    "payoff_quantile_peer_dispersion",
    "adverse_excursion_quantile_peer_dispersion",
    "positive_payoff_probability_peer_dispersion",
    "adverse_selection_probability_peer_dispersion",
    "regime_unpredictability_probability_peer_dispersion",
)
ROUND74_EPISTEMIC_ACTION_REJECTION_BUDGETS = {
    "conservative": 0.25,
    "regular": 0.15,
    "aggressive": 0.05,
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


def _readonly(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    selected = np.ascontiguousarray(value, dtype=dtype)
    selected.setflags(write=False)
    return selected


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


def _tensor_array(value: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(
        value.detach().to(device="cpu", dtype=torch.float32).numpy(),
        dtype=np.float64,
    )


def _strict_float(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Round 74 epistemic {label} differs")
    return float(value)


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Round 74 epistemic {label} integer differs")
    return int(value)


@dataclass(frozen=True)
class Round74EpistemicActionFilter:
    """Hash-bound empirical thresholds for one risk profile."""

    profile: str
    risk_coverage_report_sha256: str
    tuning_subpartition_sha256: str
    probability_calibration_sha256: str
    source_run_ids: tuple[str, ...]
    source_batch_sha256: tuple[str, ...]
    source_model_output_sha256: tuple[str, ...]
    peer_count: int
    total_rejection_budget: float
    component_tail_budget: float
    component_quantile: float
    action_thresholds: np.ndarray
    regime_thresholds: np.ndarray
    action_fit_rows: np.ndarray
    regime_fit_rows: np.ndarray
    schema_version: str = ROUND74_EPISTEMIC_ACTION_FILTER_SCHEMA_VERSION

    def validate(self) -> None:
        action_shape = (
            len(ROUND74_EVENT_SYMBOLS),
            len(ROUND74_ACTION_HORIZONS_SECONDS),
            len(ROUND74_EVENT_PAYOFF_SIDES),
            len(ROUND74_EPISTEMIC_ACTION_FILTER_COMPONENTS) - 1,
        )
        regime_shape = action_shape[:2]
        expected_budget = ROUND74_EPISTEMIC_ACTION_REJECTION_BUDGETS.get(
            self.profile
        )
        if (
            self.schema_version != ROUND74_EPISTEMIC_ACTION_FILTER_SCHEMA_VERSION
            or self.profile not in ROUND74_ACTION_PROFILES
            or expected_budget is None
            or _SHA256.fullmatch(self.risk_coverage_report_sha256) is None
            or _SHA256.fullmatch(self.tuning_subpartition_sha256) is None
            or _SHA256.fullmatch(self.probability_calibration_sha256) is None
            or len(self.source_run_ids)
            != ROUND74_EPISTEMIC_REQUIRED_POLICY_SELECTION_RUNS
            or len(set(self.source_run_ids)) != len(self.source_run_ids)
            or any(_RUN_ID.fullmatch(value) is None for value in self.source_run_ids)
            or len(self.source_batch_sha256) != len(self.source_run_ids)
            or len(self.source_model_output_sha256) != len(self.source_run_ids)
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    *self.source_batch_sha256,
                    *self.source_model_output_sha256,
                )
            )
            or isinstance(self.peer_count, bool)
            or self.peer_count != 3
            or not math.isclose(
                self.total_rejection_budget,
                expected_budget,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                self.component_tail_budget,
                expected_budget / len(ROUND74_EPISTEMIC_ACTION_FILTER_COMPONENTS),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                self.component_quantile,
                1.0 - self.component_tail_budget,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or self.action_thresholds.shape != action_shape
            or self.action_thresholds.dtype != np.float64
            or self.action_thresholds.flags.writeable
            or self.regime_thresholds.shape != regime_shape
            or self.regime_thresholds.dtype != np.float64
            or self.regime_thresholds.flags.writeable
            or self.action_fit_rows.shape != action_shape[:3]
            or self.action_fit_rows.dtype != np.int64
            or self.action_fit_rows.flags.writeable
            or self.regime_fit_rows.shape != regime_shape
            or self.regime_fit_rows.dtype != np.int64
            or self.regime_fit_rows.flags.writeable
            or not np.isfinite(self.action_thresholds).all()
            or not np.isfinite(self.regime_thresholds).all()
            or np.any(self.action_thresholds < 0.0)
            or np.any(self.regime_thresholds < 0.0)
            or np.any(self.action_thresholds[..., 2:] > 0.5 + 1e-6)
            or np.any(self.regime_thresholds > 0.5 + 1e-6)
            or np.any(
                self.action_fit_rows < ROUND74_EPISTEMIC_MINIMUM_STRATUM_ROWS
            )
            or np.any(
                self.regime_fit_rows < ROUND74_EPISTEMIC_MINIMUM_STRATUM_ROWS
            )
        ):
            raise ValueError("Round 74 epistemic action filter differs")

    @property
    def filter_sha256(self) -> str:
        self.validate()
        value = self.as_dict(include_sha256=False)
        digest = hashlib.sha256(_canonical_json(value).encode("ascii"))
        for array in (
            self.action_thresholds,
            self.regime_thresholds,
            self.action_fit_rows,
            self.regime_fit_rows,
        ):
            _update_array_digest(digest, array)
        return digest.hexdigest()

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "risk_coverage_report_sha256": self.risk_coverage_report_sha256,
            "tuning_subpartition_sha256": self.tuning_subpartition_sha256,
            "probability_calibration_sha256": self.probability_calibration_sha256,
            "source_run_ids": list(self.source_run_ids),
            "source_batch_sha256": list(self.source_batch_sha256),
            "source_model_output_sha256": list(self.source_model_output_sha256),
            "peer_count": self.peer_count,
            "total_rejection_budget": self.total_rejection_budget,
            "component_tail_budget": self.component_tail_budget,
            "component_quantile": self.component_quantile,
            "component_ids": list(ROUND74_EPISTEMIC_ACTION_FILTER_COMPONENTS),
            "strata": {
                "symbols": list(ROUND74_EVENT_SYMBOLS),
                "horizons_seconds": list(ROUND74_ACTION_HORIZONS_SECONDS),
                "sides": list(ROUND74_EVENT_PAYOFF_SIDES),
            },
            "action_thresholds": self.action_thresholds.tolist(),
            "regime_thresholds": self.regime_thresholds.tolist(),
            "action_fit_rows": self.action_fit_rows.tolist(),
            "regime_fit_rows": self.regime_fit_rows.tolist(),
            "fit_contract": {
                "source_role": "disjoint_tuning_policy_selection_runs",
                "quantile_method": "higher",
                "tail_allocation": "bonferroni_equal_across_five_components",
                "future_rejection_rate_guaranteed": False,
                "exchangeability_claim": False,
                "target_fields_used_for_threshold_fit": False,
                "policy_effect_enabled": False,
                "sealed_test_accessed": False,
            },
        }
        if include_sha256:
            value["filter_sha256"] = self.filter_sha256
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Round74EpistemicActionFilter:
        payload = dict(value)
        claimed = payload.pop("filter_sha256", None)
        component_ids = payload.pop("component_ids", None)
        strata = payload.pop("strata", None)
        contract = payload.pop("fit_contract", None)
        expected_keys = {
            "schema_version",
            "profile",
            "risk_coverage_report_sha256",
            "tuning_subpartition_sha256",
            "probability_calibration_sha256",
            "source_run_ids",
            "source_batch_sha256",
            "source_model_output_sha256",
            "peer_count",
            "total_rejection_budget",
            "component_tail_budget",
            "component_quantile",
            "action_thresholds",
            "regime_thresholds",
            "action_fit_rows",
            "regime_fit_rows",
        }
        if (
            set(payload) != expected_keys
            or _SHA256.fullmatch(str(claimed)) is None
            or component_ids != list(ROUND74_EPISTEMIC_ACTION_FILTER_COMPONENTS)
            or strata
            != {
                "symbols": list(ROUND74_EVENT_SYMBOLS),
                "horizons_seconds": list(ROUND74_ACTION_HORIZONS_SECONDS),
                "sides": list(ROUND74_EVENT_PAYOFF_SIDES),
            }
            or contract
            != {
                "source_role": "disjoint_tuning_policy_selection_runs",
                "quantile_method": "higher",
                "tail_allocation": "bonferroni_equal_across_five_components",
                "future_rejection_rate_guaranteed": False,
                "exchangeability_claim": False,
                "target_fields_used_for_threshold_fit": False,
                "policy_effect_enabled": False,
                "sealed_test_accessed": False,
            }
        ):
            raise ValueError("Round 74 epistemic action filter payload differs")

        def strings(name: str) -> tuple[str, ...]:
            selected = payload[name]
            if not isinstance(selected, list) or any(
                not isinstance(item, str) for item in selected
            ):
                raise ValueError("Round 74 epistemic action filter strings differ")
            return tuple(selected)

        result = cls(
            schema_version=str(payload["schema_version"]),
            profile=str(payload["profile"]),
            risk_coverage_report_sha256=str(
                payload["risk_coverage_report_sha256"]
            ),
            tuning_subpartition_sha256=str(payload["tuning_subpartition_sha256"]),
            probability_calibration_sha256=str(
                payload["probability_calibration_sha256"]
            ),
            source_run_ids=strings("source_run_ids"),
            source_batch_sha256=strings("source_batch_sha256"),
            source_model_output_sha256=strings("source_model_output_sha256"),
            peer_count=_strict_int(payload["peer_count"], "peer count"),
            total_rejection_budget=_strict_float(
                payload["total_rejection_budget"], "rejection budget"
            ),
            component_tail_budget=_strict_float(
                payload["component_tail_budget"], "component tail budget"
            ),
            component_quantile=_strict_float(
                payload["component_quantile"], "component quantile"
            ),
            action_thresholds=_readonly(
                np.asarray(payload["action_thresholds"]),
                dtype=np.dtype(np.float64),
            ),
            regime_thresholds=_readonly(
                np.asarray(payload["regime_thresholds"]),
                dtype=np.dtype(np.float64),
            ),
            action_fit_rows=_readonly(
                np.asarray(payload["action_fit_rows"]),
                dtype=np.dtype(np.int64),
            ),
            regime_fit_rows=_readonly(
                np.asarray(payload["regime_fit_rows"]),
                dtype=np.dtype(np.int64),
            ),
        )
        result.validate()
        if result.filter_sha256 != claimed:
            raise ValueError("Round 74 epistemic action filter identity differs")
        return result


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    selected = np.asarray(values, dtype=np.float64)
    if selected.ndim != 1 or selected.size < 1 or not np.isfinite(selected).all():
        raise ValueError("Round 74 epistemic threshold population differs")
    return float(np.quantile(selected, quantile, method="higher"))


def fit_round74_epistemic_action_filter(
    batches: Sequence[Round74EpistemicEvaluationBatch],
    report: Round74EpistemicRiskCoverageReport,
    *,
    profile: str,
) -> Round74EpistemicActionFilter:
    """Fit target-free, stratum-specific thresholds after the ordering gate."""

    selected = tuple(batches)
    report.validate()
    for batch in selected:
        batch.validate()
    expected_runs = report.policy_selection_run_ids
    if (
        not report.policy_challenge_eligible
        or profile not in ROUND74_ACTION_PROFILES
        or len(selected) != ROUND74_EPISTEMIC_REQUIRED_POLICY_SELECTION_RUNS
        or tuple(batch.batch_sha256 for batch in selected)
        != report.policy_selection_batch_sha256
        or tuple(batch.model_output_sha256 for batch in selected)
        != report.model_output_sha256
        or tuple(tuple(dict.fromkeys(batch.run_id)) for batch in selected)
        != tuple((run_id,) for run_id in expected_runs)
        or any(
            batch.tuning_subpartition_sha256
            != report.tuning_subpartition_sha256
            or batch.probability_calibration_sha256
            != report.probability_calibration_sha256
            or batch.peer_count != report.peer_count
            for batch in selected
        )
    ):
        raise ValueError("Round 74 epistemic action-filter source differs")
    budget = ROUND74_EPISTEMIC_ACTION_REJECTION_BUDGETS[profile]
    component_tail = budget / len(ROUND74_EPISTEMIC_ACTION_FILTER_COMPONENTS)
    quantile = 1.0 - component_tail
    action_shape = (
        len(ROUND74_EVENT_SYMBOLS),
        len(ROUND74_ACTION_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    thresholds = np.empty(
        (*action_shape, len(ROUND74_EPISTEMIC_ACTION_FILTER_COMPONENTS) - 1),
        dtype=np.float64,
    )
    action_rows = np.zeros(action_shape, dtype=np.int64)
    regime_thresholds = np.empty(action_shape[:2], dtype=np.float64)
    regime_rows = np.zeros(action_shape[:2], dtype=np.int64)
    action_sources = (
        "payoff_quantile_peer_dispersion_bps",
        "adverse_excursion_quantile_peer_dispersion_bps",
        "positive_payoff_probability_peer_dispersion",
        "adverse_selection_probability_peer_dispersion",
    )
    for symbol_index, symbol in enumerate(ROUND74_EVENT_SYMBOLS):
        for action_horizon_index, horizon in enumerate(
            ROUND74_ACTION_HORIZONS_SECONDS
        ):
            horizon_index = ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS.index(horizon)
            for side_index, _side in enumerate(ROUND74_EVENT_PAYOFF_SIDES):
                populations: list[list[np.ndarray]] = [
                    [] for _ in range(len(action_sources))
                ]
                rows = 0
                for batch in selected:
                    symbol_mask = np.asarray(batch.symbol) == symbol
                    mask = symbol_mask & (
                        batch.action_eligibility[:, horizon_index, side_index] == 1.0
                    )
                    if not bool(mask.any()):
                        raise ValueError(
                            "Round 74 epistemic action-filter run stratum is empty"
                        )
                    rows += int(mask.sum())
                    for component_index, name in enumerate(action_sources):
                        populations[component_index].append(
                            getattr(batch, name)[:, horizon_index, side_index][mask]
                        )
                if rows < ROUND74_EPISTEMIC_MINIMUM_STRATUM_ROWS:
                    raise ValueError(
                        "Round 74 epistemic action-filter stratum is undersized"
                    )
                action_rows[
                    symbol_index,
                    action_horizon_index,
                    side_index,
                ] = rows
                for component_index, population in enumerate(populations):
                    thresholds[
                        symbol_index,
                        action_horizon_index,
                        side_index,
                        component_index,
                    ] = _higher_quantile(
                        np.concatenate(population),
                        quantile,
                    )
            regime_population: list[np.ndarray] = []
            regime_count = 0
            for batch in selected:
                symbol_mask = np.asarray(batch.symbol) == symbol
                mask = symbol_mask & (
                    batch.regime_unpredictability_eligibility[:, horizon_index]
                    == 1.0
                )
                if not bool(mask.any()):
                    raise ValueError(
                        "Round 74 epistemic regime-filter run stratum is empty"
                    )
                values = batch.regime_unpredictability_probability_peer_dispersion[
                    :, horizon_index
                ][mask]
                regime_population.append(values)
                regime_count += int(values.size)
            if regime_count < ROUND74_EPISTEMIC_MINIMUM_STRATUM_ROWS:
                raise ValueError(
                    "Round 74 epistemic regime-filter stratum is undersized"
                )
            regime_rows[symbol_index, action_horizon_index] = regime_count
            regime_thresholds[symbol_index, action_horizon_index] = _higher_quantile(
                np.concatenate(regime_population),
                quantile,
            )
    result = Round74EpistemicActionFilter(
        profile=profile,
        risk_coverage_report_sha256=report.report_sha256,
        tuning_subpartition_sha256=report.tuning_subpartition_sha256,
        probability_calibration_sha256=report.probability_calibration_sha256,
        source_run_ids=report.policy_selection_run_ids,
        source_batch_sha256=report.policy_selection_batch_sha256,
        source_model_output_sha256=report.model_output_sha256,
        peer_count=report.peer_count,
        total_rejection_budget=budget,
        component_tail_budget=component_tail,
        component_quantile=quantile,
        action_thresholds=_readonly(thresholds, dtype=np.dtype(np.float64)),
        regime_thresholds=_readonly(
            regime_thresholds,
            dtype=np.dtype(np.float64),
        ),
        action_fit_rows=_readonly(action_rows, dtype=np.dtype(np.int64)),
        regime_fit_rows=_readonly(regime_rows, dtype=np.dtype(np.int64)),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class Round74EpistemicActionFilterApplication:
    filter_sha256: str
    source_candidate_sha256: str
    filtered_candidate_sha256: str
    source_model_output_sha256: str
    rows: int
    eligible_rows_before: int
    eligible_rows_after: int
    blocked_distinct_rows: int
    blocked_rows_by_component: tuple[int, ...]
    schema_version: str = ROUND74_EPISTEMIC_ACTION_FILTER_APPLICATION_SCHEMA_VERSION
    target_fields_consumed: bool = False
    candidate_set_only_reduced: bool = True
    position_size_changed: bool = False
    leverage_changed: bool = False
    trading_authority: bool = False

    def validate(self) -> None:
        if (
            self.schema_version
            != ROUND74_EPISTEMIC_ACTION_FILTER_APPLICATION_SCHEMA_VERSION
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.filter_sha256,
                    self.source_candidate_sha256,
                    self.filtered_candidate_sha256,
                    self.source_model_output_sha256,
                )
            )
            or isinstance(self.rows, bool)
            or self.rows < 1
            or any(
                isinstance(value, bool) or not 0 <= value <= self.rows
                for value in (
                    self.eligible_rows_before,
                    self.eligible_rows_after,
                    self.blocked_distinct_rows,
                    *self.blocked_rows_by_component,
                )
            )
            or len(self.blocked_rows_by_component)
            != len(ROUND74_EPISTEMIC_ACTION_FILTER_COMPONENTS)
            or self.eligible_rows_after > self.eligible_rows_before
            or self.blocked_distinct_rows
            != self.eligible_rows_before - self.eligible_rows_after
            or self.target_fields_consumed
            or not self.candidate_set_only_reduced
            or self.position_size_changed
            or self.leverage_changed
            or self.trading_authority
        ):
            raise ValueError("Round 74 epistemic filter application differs")

    @property
    def application_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "filter_sha256": self.filter_sha256,
            "source_candidate_sha256": self.source_candidate_sha256,
            "filtered_candidate_sha256": self.filtered_candidate_sha256,
            "source_model_output_sha256": self.source_model_output_sha256,
            "rows": self.rows,
            "eligible_rows_before": self.eligible_rows_before,
            "eligible_rows_after": self.eligible_rows_after,
            "blocked_distinct_rows": self.blocked_distinct_rows,
            "blocked_rows_by_component": list(self.blocked_rows_by_component),
            "component_ids": list(ROUND74_EPISTEMIC_ACTION_FILTER_COMPONENTS),
            "target_fields_consumed": self.target_fields_consumed,
            "candidate_set_only_reduced": self.candidate_set_only_reduced,
            "position_size_changed": self.position_size_changed,
            "leverage_changed": self.leverage_changed,
            "trading_authority": self.trading_authority,
        }
        if include_sha256:
            value["application_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74EpistemicActionFilterApplication:
        unsigned = dict(value)
        claimed = unsigned.pop("application_sha256", None)
        if claimed != _canonical_sha256(unsigned):
            raise ValueError("Round 74 epistemic filter application payload differs")
        payload = dict(unsigned)
        component_ids = payload.pop("component_ids", None)
        expected = {
            "schema_version",
            "filter_sha256",
            "source_candidate_sha256",
            "filtered_candidate_sha256",
            "source_model_output_sha256",
            "rows",
            "eligible_rows_before",
            "eligible_rows_after",
            "blocked_distinct_rows",
            "blocked_rows_by_component",
            "target_fields_consumed",
            "candidate_set_only_reduced",
            "position_size_changed",
            "leverage_changed",
            "trading_authority",
        }
        blocked = payload.get("blocked_rows_by_component")
        boolean_fields = (
            "target_fields_consumed",
            "candidate_set_only_reduced",
            "position_size_changed",
            "leverage_changed",
            "trading_authority",
        )
        if (
            set(payload) != expected
            or component_ids != list(ROUND74_EPISTEMIC_ACTION_FILTER_COMPONENTS)
            or not isinstance(blocked, list)
            or any(not isinstance(payload[name], bool) for name in boolean_fields)
        ):
            raise ValueError("Round 74 epistemic filter application payload differs")
        result = cls(
            schema_version=str(payload["schema_version"]),
            filter_sha256=str(payload["filter_sha256"]),
            source_candidate_sha256=str(payload["source_candidate_sha256"]),
            filtered_candidate_sha256=str(payload["filtered_candidate_sha256"]),
            source_model_output_sha256=str(payload["source_model_output_sha256"]),
            rows=_strict_int(payload["rows"], "application rows"),
            eligible_rows_before=_strict_int(
                payload["eligible_rows_before"],
                "eligible rows before",
            ),
            eligible_rows_after=_strict_int(
                payload["eligible_rows_after"],
                "eligible rows after",
            ),
            blocked_distinct_rows=_strict_int(
                payload["blocked_distinct_rows"],
                "blocked rows",
            ),
            blocked_rows_by_component=tuple(
                _strict_int(item, "component blocked rows") for item in blocked
            ),
            target_fields_consumed=payload["target_fields_consumed"],
            candidate_set_only_reduced=payload["candidate_set_only_reduced"],
            position_size_changed=payload["position_size_changed"],
            leverage_changed=payload["leverage_changed"],
            trading_authority=payload["trading_authority"],
        )
        result.validate()
        if result.application_sha256 != claimed:
            raise ValueError("Round 74 epistemic filter application identity differs")
        return result


def apply_round74_epistemic_action_filter(
    candidate: Round74ActionCandidateBatch,
    model_output: Round74EventModelOutput,
    action_filter: Round74EpistemicActionFilter,
) -> tuple[
    Round74ActionCandidateBatch,
    Round74EpistemicActionFilterApplication,
]:
    """Remove high-disagreement candidates without reading realized outcomes."""

    candidate.validate()
    model_output.validate(candidate.rows)
    action_filter.validate()
    diagnostics = model_output.epistemic_diagnostics
    output_sha256 = round74_action_model_output_sha256(model_output)
    if (
        diagnostics is None
        or diagnostics.peer_count != action_filter.peer_count
        or candidate.profile != action_filter.profile
        or candidate.model_output_sha256 != output_sha256
        or candidate.tuning_subpartition_sha256
        != action_filter.tuning_subpartition_sha256
        or candidate.probability_calibration_sha256
        != action_filter.probability_calibration_sha256
    ):
        raise ValueError("Round 74 epistemic runtime filter source differs")
    diagnostics.validate(candidate.rows)
    action_values = (
        _tensor_array(
            diagnostics.payoff_quantile_standard_deviation_bps.square()
            .mean(dim=-1)
            .sqrt()
        ),
        _tensor_array(
            diagnostics.maximum_adverse_excursion_quantile_standard_deviation_bps.square()
            .mean(dim=-1)
            .sqrt()
        ),
        _tensor_array(diagnostics.positive_payoff_probability_standard_deviation),
        _tensor_array(diagnostics.adverse_selection_probability_standard_deviation),
    )
    regime_values = _tensor_array(
        diagnostics.regime_unpredictability_probability_standard_deviation
    )
    component_masks = [
        np.zeros(candidate.rows, dtype=np.bool_)
        for _ in ROUND74_EPISTEMIC_ACTION_FILTER_COMPONENTS
    ]
    active_rows = np.flatnonzero(candidate.eligible)
    for row_index in active_rows:
        symbol_index = ROUND74_EVENT_SYMBOLS.index(candidate.symbol[row_index])
        horizon = int(candidate.horizon_seconds[row_index])
        action_horizon_index = ROUND74_ACTION_HORIZONS_SECONDS.index(horizon)
        horizon_index = ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS.index(horizon)
        side_index = 0 if int(candidate.side[row_index]) == 1 else 1
        for component_index, values in enumerate(action_values):
            observed = values[row_index, horizon_index, side_index]
            threshold = action_filter.action_thresholds[
                symbol_index,
                action_horizon_index,
                side_index,
                component_index,
            ]
            component_masks[component_index][row_index] = observed > threshold
        regime_value = regime_values[row_index, horizon_index]
        component_masks[-1][row_index] = (
            regime_value
            > action_filter.regime_thresholds[
                symbol_index,
                action_horizon_index,
            ]
        )
    blocked = np.logical_or.reduce(component_masks)
    retained = np.asarray(candidate.eligible & ~blocked, dtype=np.bool_)
    retained.setflags(write=False)

    def filtered_array(value: np.ndarray) -> np.ndarray:
        selected = np.array(value, copy=True)
        selected[~retained] = 0
        selected.setflags(write=False)
        return selected

    filtered = replace(
        candidate,
        horizon_seconds=filtered_array(candidate.horizon_seconds),
        side=filtered_array(candidate.side),
        risk_adjusted_strength_bps=filtered_array(
            candidate.risk_adjusted_strength_bps
        ),
        quality_score=filtered_array(candidate.quality_score),
        positive_payoff_probability=filtered_array(
            candidate.positive_payoff_probability
        ),
        adverse_selection_probability=filtered_array(
            candidate.adverse_selection_probability
        ),
        regime_unpredictability_probability=filtered_array(
            candidate.regime_unpredictability_probability
        ),
        payoff_quantiles_bps=filtered_array(candidate.payoff_quantiles_bps),
        maximum_adverse_excursion_quantiles_bps=filtered_array(
            candidate.maximum_adverse_excursion_quantiles_bps
        ),
        eligible=retained,
    )
    filtered.validate()
    if np.any(filtered.eligible & ~candidate.eligible):
        raise RuntimeError("Round 74 epistemic filter expanded the candidate set")
    application = Round74EpistemicActionFilterApplication(
        filter_sha256=action_filter.filter_sha256,
        source_candidate_sha256=candidate.candidate_sha256,
        filtered_candidate_sha256=filtered.candidate_sha256,
        source_model_output_sha256=output_sha256,
        rows=candidate.rows,
        eligible_rows_before=int(candidate.eligible.sum()),
        eligible_rows_after=int(filtered.eligible.sum()),
        blocked_distinct_rows=int((candidate.eligible & blocked).sum()),
        blocked_rows_by_component=tuple(
            int((candidate.eligible & mask).sum()) for mask in component_masks
        ),
    )
    application.validate()
    return filtered, application


__all__ = [
    "ROUND74_EPISTEMIC_ACTION_FILTER_APPLICATION_SCHEMA_VERSION",
    "ROUND74_EPISTEMIC_ACTION_FILTER_COMPONENTS",
    "ROUND74_EPISTEMIC_ACTION_FILTER_SCHEMA_VERSION",
    "ROUND74_EPISTEMIC_ACTION_REJECTION_BUDGETS",
    "Round74EpistemicActionFilter",
    "Round74EpistemicActionFilterApplication",
    "apply_round74_epistemic_action_filter",
    "fit_round74_epistemic_action_filter",
]
