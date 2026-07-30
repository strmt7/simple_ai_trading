"""Frozen chronological cohort assembly for Polymarket Round 17."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_replay import PolymarketResolutionEvidence
from .polymarket_round17_dataset import PolymarketRound17ConditionDataset
from .polymarket_round17_features import (
    POLYMARKET_ROUND17_CONTRACT_SHA256,
    POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
)
from .polymarket_round17_model import Round17DevelopmentPanel


POLYMARKET_ROUND17_COHORT_PLAN_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-cohort-plan-v1"
)
POLYMARKET_ROUND17_COHORT_PLAN_SHA256 = (
    "37fede4da0d6c504bce7cb763b9bd49032e0252a8cede045f29f05acff67fc00"
)
POLYMARKET_ROUND17_COHORT_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-cohort-manifest-v1"
)
POLYMARKET_ROUND17_CONDITION_LABEL_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-condition-label-v1"
)
POLYMARKET_ROUND17_DEVELOPMENT_TARGET_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-development-target-manifest-v1"
)
POLYMARKET_ROUND17_CAMPAIGN_PLAN_SHA256 = (
    "c19ef7733efb86202742f045a45b3d92e8e17bb922c3c5f780240243889609b5"
)
POLYMARKET_ROUND17_CAPTURE_CONTRACT_SHA256 = (
    "60cde01112a749a9971447368b3a5d73b203d095e62a974327004c16cb021f1b"
)
_CAMPAIGN_START_MS = 1_785_344_400_000
_CAMPAIGN_END_MS = 1_787_936_400_000
_SLOT_DURATION_MS = 1_800_000
_EVENT_DURATION_MS = 300_000
_TOTAL_SLOTS = 1_440
_DEVELOPMENT_ROLES = (
    "train",
    "tune_calibration",
    "tune_selection",
    "tune_uncertainty",
    "tune_economic",
)
_ROLE_WINDOWS = (
    ("train", 0, 671),
    ("train_tune_embargo", 672, 673),
    ("tune_calibration", 674, 745),
    ("calibration_selection_embargo", 746, 747),
    ("tune_selection", 748, 819),
    ("selection_uncertainty_embargo", 820, 821),
    ("tune_uncertainty", 822, 893),
    ("uncertainty_economic_embargo", 894, 895),
    ("tune_economic", 896, 1009),
    ("tune_test_embargo", 1010, 1011),
    ("test", 1012, 1439),
)
_RESOLUTION_SOURCES = frozenset({"clob_websocket", "clob_gamma_crosscheck"})
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9:._-]{1,160}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAXIMUM_PLAN_BYTES = 64 * 1024


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 17 cohort plan contains duplicate JSON keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 17 cohort plan contains {value}")


@dataclass(frozen=True, slots=True)
class Round17CohortRoleWindow:
    name: str
    first_slot: int
    last_slot: int
    start_ms: int
    end_ms_exclusive: int

    @property
    def slot_count(self) -> int:
        return self.last_slot - self.first_slot + 1


@dataclass(frozen=True, slots=True)
class Round17CohortPlan:
    plan_sha256: str
    roles: tuple[Round17CohortRoleWindow, ...]

    def role_for_condition(
        self,
        *,
        event_start_ms: int,
        event_end_ms: int,
        source_slot_index: int,
    ) -> Round17CohortRoleWindow:
        start = int(event_start_ms)
        end = int(event_end_ms)
        source_slot = int(source_slot_index)
        if (
            start < _CAMPAIGN_START_MS
            or start >= _CAMPAIGN_END_MS
            or start % _EVENT_DURATION_MS
            or end - start != _EVENT_DURATION_MS
            or source_slot < 0
            or source_slot >= _TOTAL_SLOTS
        ):
            raise ValueError("Round 17 condition lies outside the frozen campaign")
        containing_slot = (start - _CAMPAIGN_START_MS) // _SLOT_DURATION_MS
        if source_slot != containing_slot:
            raise ValueError("Round 17 condition source slot differs")
        role = next(
            item
            for item in self.roles
            if item.first_slot <= containing_slot <= item.last_slot
        )
        if not role.start_ms <= start < role.end_ms_exclusive:
            raise ValueError("Round 17 condition role boundary differs")
        return role


def validate_round17_cohort_plan(
    value: Mapping[str, object],
) -> Round17CohortPlan:
    payload = dict(value)
    claimed = str(payload.pop("plan_sha256", "")).strip().lower()
    parents = payload.get("parents")
    campaign = payload.get("campaign")
    raw_roles = payload.get("roles")
    assignment = payload.get("assignment")
    test_policy = payload.get("test_policy")
    authority = payload.get("authority")
    expected_roles = [
        {
            "name": name,
            "first_slot": first,
            "last_slot": last,
            "slot_count": last - first + 1,
            "start_ms": _CAMPAIGN_START_MS + first * _SLOT_DURATION_MS,
            "end_ms_exclusive": (_CAMPAIGN_START_MS + (last + 1) * _SLOT_DURATION_MS),
        }
        for name, first, last in _ROLE_WINDOWS
    ]
    if (
        claimed != POLYMARKET_ROUND17_COHORT_PLAN_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND17_COHORT_PLAN_SCHEMA_VERSION
        or payload.get("round") != 17
        or payload.get("status")
        != "preregistered_before_active_campaign_outcome_or_model_score_access"
        or not isinstance(parents, Mapping)
        or parents
        != {
            "round14_capture_contract_sha256": (
                POLYMARKET_ROUND17_CAPTURE_CONTRACT_SHA256
            ),
            "round14_campaign_plan_sha256": (POLYMARKET_ROUND17_CAMPAIGN_PLAN_SHA256),
            "round17_causal_model_contract_sha256": (
                POLYMARKET_ROUND17_CONTRACT_SHA256
            ),
            "active_campaign_outcomes_consulted": False,
            "active_campaign_model_scores_consulted": False,
            "active_campaign_execution_scores_consulted": False,
        }
        or not isinstance(campaign, Mapping)
        or campaign
        != {
            "scheduled_start_ms": _CAMPAIGN_START_MS,
            "scheduled_end_ms": _CAMPAIGN_END_MS,
            "slot_duration_ms": _SLOT_DURATION_MS,
            "total_slots": _TOTAL_SLOTS,
            "event_duration_ms": _EVENT_DURATION_MS,
        }
        or raw_roles != expected_roles
        or not isinstance(assignment, Mapping)
        or assignment
        != {
            "method": "condition_event_start_to_containing_campaign_slot",
            "source_capture_slot_must_equal_event_containing_slot": True,
            "accepted_conditions_only": True,
            "admission_is_target_free": True,
            "conditions_in_embargo_roles_are_excluded": True,
            "development_manifest_may_contain_test_conditions": False,
            "development_target_manifest_may_contain_test_labels": False,
            "missing_or_excluded_conditions_are_never_backfilled_from_another_role": (
                True
            ),
            "data_dependent_role_assignment": False,
            "role_reassignment_after_model_or_outcome_access": False,
        }
        or not isinstance(test_policy, Mapping)
        or test_policy
        != {
            "minimum_calendar_days": 7,
            "minimum_slot_count": 336,
            "actual_reserved_slot_count": 428,
            "access": (
                "one_use_after_immutable_model_probability_and_economic_artifacts"
            ),
            "failed_test_returns_to_development": False,
        }
        or not isinstance(authority, Mapping)
        or authority
        != {
            "labels_consulted": False,
            "outcomes_consulted": False,
            "model_scores_consulted": False,
            "execution_scores_consulted": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
    ):
        raise ValueError("Round 17 cohort plan integrity differs")
    roles = tuple(
        Round17CohortRoleWindow(
            name=name,
            first_slot=first,
            last_slot=last,
            start_ms=_CAMPAIGN_START_MS + first * _SLOT_DURATION_MS,
            end_ms_exclusive=_CAMPAIGN_START_MS + (last + 1) * _SLOT_DURATION_MS,
        )
        for name, first, last in _ROLE_WINDOWS
    )
    return Round17CohortPlan(plan_sha256=claimed, roles=roles)


def load_round17_cohort_plan(path: str | Path) -> Round17CohortPlan:
    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= _MAXIMUM_PLAN_BYTES
    ):
        raise ValueError("Round 17 cohort plan is unavailable")
    try:
        payload = json.loads(
            selected.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 17 cohort plan is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Round 17 cohort plan is not an object")
    return validate_round17_cohort_plan(payload)


@dataclass(frozen=True, slots=True)
class Round17CohortCondition:
    source_run_id: str
    source_slot_index: int
    condition_id: str
    event_start_ms: int
    event_end_ms: int
    role: str
    admission_sha256: str
    condition_dataset_sha256: str
    feature_row_count: int
    binance_layer_eligible: bool
    condition_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        return {
            "source_run_id": self.source_run_id,
            "source_slot_index": self.source_slot_index,
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "event_end_ms": self.event_end_ms,
            "role": self.role,
            "admission_sha256": self.admission_sha256,
            "condition_dataset_sha256": self.condition_dataset_sha256,
            "feature_row_count": self.feature_row_count,
            "binance_layer_eligible": self.binance_layer_eligible,
        }

    def validated(self, plan: Round17CohortPlan) -> Round17CohortCondition:
        role = plan.role_for_condition(
            event_start_ms=self.event_start_ms,
            event_end_ms=self.event_end_ms,
            source_slot_index=self.source_slot_index,
        )
        if (
            _RUN_ID.fullmatch(self.source_run_id) is None
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or self.role != role.name
            or self.role not in _DEVELOPMENT_ROLES
            or _SHA256.fullmatch(self.admission_sha256) is None
            or _SHA256.fullmatch(self.condition_dataset_sha256) is None
            or self.feature_row_count < 1
            or type(self.binance_layer_eligible) is not bool
            or self.condition_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 cohort condition is invalid")
        return self


def build_round17_cohort_condition(
    plan: Round17CohortPlan,
    dataset: PolymarketRound17ConditionDataset,
    *,
    source_slot_index: int,
) -> Round17CohortCondition:
    if not isinstance(dataset, PolymarketRound17ConditionDataset):
        raise TypeError("Round 17 cohort source dataset type differs")
    selected = dataset.validated()
    role = plan.role_for_condition(
        event_start_ms=selected.event_start_ms,
        event_end_ms=selected.event_end_ms,
        source_slot_index=source_slot_index,
    )
    if role.name not in _DEVELOPMENT_ROLES:
        raise ValueError("Round 17 condition is reserved outside development")
    provisional = Round17CohortCondition(
        source_run_id=selected.run_id,
        source_slot_index=int(source_slot_index),
        condition_id=selected.condition_id,
        event_start_ms=selected.event_start_ms,
        event_end_ms=selected.event_end_ms,
        role=role.name,
        admission_sha256=selected.admission_sha256,
        condition_dataset_sha256=selected.dataset_sha256,
        feature_row_count=len(selected.rows),
        binance_layer_eligible=selected.binance_layer_eligible,
    )
    return replace(
        provisional,
        condition_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated(plan)


@dataclass(frozen=True, slots=True)
class Round17CohortManifest:
    plan_sha256: str
    conditions: tuple[Round17CohortCondition, ...]
    role_condition_counts: Mapping[str, int]
    cohort_dataset_sha256: str
    manifest_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND17_COHORT_MANIFEST_SCHEMA_VERSION,
            "plan_sha256": self.plan_sha256,
            "campaign_plan_sha256": POLYMARKET_ROUND17_CAMPAIGN_PLAN_SHA256,
            "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
            "feature_names_sha256": POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
            "conditions": [
                {
                    **item.identity_payload(),
                    "condition_sha256": item.condition_sha256,
                }
                for item in self.conditions
            ],
            "role_condition_counts": dict(self.role_condition_counts),
            "cohort_dataset_sha256": self.cohort_dataset_sha256,
            "labels_consulted": False,
            "outcomes_consulted": False,
            "model_scores_consulted": False,
            "execution_scores_consulted": False,
            "test_features_accessed": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def validated(self, plan: Round17CohortPlan) -> Round17CohortManifest:
        if self.plan_sha256 != plan.plan_sha256 or not self.conditions:
            raise ValueError("Round 17 cohort manifest identity differs")
        for item in self.conditions:
            item.validated(plan)
        expected_order = tuple(
            sorted(
                self.conditions,
                key=lambda item: (
                    item.event_start_ms,
                    item.condition_id,
                    item.source_run_id,
                ),
            )
        )
        expected_counts = {
            role.name: sum(item.role == role.name for item in self.conditions)
            for role in plan.roles
        }
        condition_payload = [
            [item.condition_id, item.condition_dataset_sha256, item.condition_sha256]
            for item in self.conditions
        ]
        expected_dataset_sha256 = _canonical_sha256(
            {
                "schema_version": "polymarket-round17-cohort-dataset-v1",
                "plan_sha256": plan.plan_sha256,
                "conditions": condition_payload,
            }
        )
        if (
            expected_order != self.conditions
            or len({item.condition_id for item in self.conditions})
            != len(self.conditions)
            or dict(self.role_condition_counts) != expected_counts
            or self.cohort_dataset_sha256 != expected_dataset_sha256
            or self.manifest_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 cohort manifest integrity differs")
        return self


def build_round17_cohort_manifest(
    plan: Round17CohortPlan,
    conditions: Sequence[Round17CohortCondition],
) -> Round17CohortManifest:
    selected = tuple(
        sorted(
            (item.validated(plan) for item in conditions),
            key=lambda item: (
                item.event_start_ms,
                item.condition_id,
                item.source_run_id,
            ),
        )
    )
    if not selected or len({item.condition_id for item in selected}) != len(selected):
        raise ValueError("Round 17 cohort conditions are empty or duplicated")
    counts = {
        role.name: sum(item.role == role.name for item in selected)
        for role in plan.roles
    }
    dataset_sha256 = _canonical_sha256(
        {
            "schema_version": "polymarket-round17-cohort-dataset-v1",
            "plan_sha256": plan.plan_sha256,
            "conditions": [
                [
                    item.condition_id,
                    item.condition_dataset_sha256,
                    item.condition_sha256,
                ]
                for item in selected
            ],
        }
    )
    provisional = Round17CohortManifest(
        plan_sha256=plan.plan_sha256,
        conditions=selected,
        role_condition_counts=counts,
        cohort_dataset_sha256=dataset_sha256,
    )
    return replace(
        provisional,
        manifest_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated(plan)


@dataclass(frozen=True, slots=True)
class Round17ConditionLabel:
    source_run_id: str
    condition_id: str
    event_start_ms: int
    resolved_at_ms: int
    winning_outcome: str
    source: str
    resolution_event_sha256: str
    label_sha256: str = ""

    @property
    def target_up(self) -> float:
        return 1.0 if self.winning_outcome == "Up" else 0.0

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND17_CONDITION_LABEL_SCHEMA_VERSION,
            "source_run_id": self.source_run_id,
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "resolved_at_ms": self.resolved_at_ms,
            "winning_outcome": self.winning_outcome,
            "source": self.source,
            "resolution_event_sha256": self.resolution_event_sha256,
        }

    def validated(self) -> Round17ConditionLabel:
        if (
            _RUN_ID.fullmatch(self.source_run_id) is None
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or self.event_start_ms <= 0
            or self.event_start_ms % _EVENT_DURATION_MS
            or self.resolved_at_ms < self.event_start_ms + _EVENT_DURATION_MS
            or self.winning_outcome not in {"Up", "Down"}
            or self.source not in _RESOLUTION_SOURCES
            or _SHA256.fullmatch(self.resolution_event_sha256) is None
            or self.label_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 condition label is invalid")
        return self


def build_round17_condition_label(
    dataset: PolymarketRound17ConditionDataset,
    market: PolymarketFiveMinuteMarket,
    resolution: PolymarketResolutionEvidence,
) -> Round17ConditionLabel:
    selected = dataset.validated()
    if not isinstance(market, PolymarketFiveMinuteMarket):
        raise TypeError("Round 17 label market type differs")
    if not isinstance(resolution, PolymarketResolutionEvidence):
        raise TypeError("Round 17 resolution evidence type differs")
    expected_winner = (
        market.up_token_id
        if resolution.winning_outcome == "Up"
        else market.down_token_id
        if resolution.winning_outcome == "Down"
        else ""
    )
    if (
        market.asset != "BTC"
        or market.condition_id != selected.condition_id
        or market.event_start_ms != selected.event_start_ms
        or market.end_ms != selected.event_end_ms
        or resolution.run_id != selected.run_id
        or resolution.condition_id != selected.condition_id
        or resolution.winning_outcome not in {"Up", "Down"}
        or resolution.winning_asset_id != expected_winner
        or resolution.resolved_at_ms < selected.event_end_ms
        or resolution.received_wall_ms < selected.event_end_ms
    ):
        raise ValueError("Round 17 resolution evidence identity differs")
    provisional = Round17ConditionLabel(
        source_run_id=selected.run_id,
        condition_id=selected.condition_id,
        event_start_ms=selected.event_start_ms,
        resolved_at_ms=resolution.resolved_at_ms,
        winning_outcome=resolution.winning_outcome,
        source=resolution.source,
        resolution_event_sha256=resolution.event_sha256,
    )
    return replace(
        provisional,
        label_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


@dataclass(frozen=True, slots=True)
class Round17DevelopmentTargetManifest:
    plan_sha256: str
    cohort_manifest_sha256: str
    labels: tuple[Round17ConditionLabel, ...]
    development_dataset_sha256: str
    target_manifest_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                POLYMARKET_ROUND17_DEVELOPMENT_TARGET_MANIFEST_SCHEMA_VERSION
            ),
            "plan_sha256": self.plan_sha256,
            "cohort_manifest_sha256": self.cohort_manifest_sha256,
            "labels": [
                {**item.identity_payload(), "label_sha256": item.label_sha256}
                for item in self.labels
            ],
            "development_dataset_sha256": self.development_dataset_sha256,
            "development_labels_consulted": True,
            "test_features_accessed": False,
            "test_targets_accessed": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def validated(
        self,
        plan: Round17CohortPlan,
        cohort: Round17CohortManifest,
    ) -> Round17DevelopmentTargetManifest:
        verified_cohort = cohort.validated(plan)
        for label in self.labels:
            label.validated()
        expected_order = tuple(
            sorted(
                self.labels,
                key=lambda item: (item.event_start_ms, item.condition_id),
            )
        )
        references = {item.condition_id: item for item in verified_cohort.conditions}
        expected_dataset_sha256 = _canonical_sha256(
            {
                "schema_version": "polymarket-round17-development-dataset-v1",
                "cohort_dataset_sha256": verified_cohort.cohort_dataset_sha256,
                "labels": [
                    [item.condition_id, item.label_sha256] for item in self.labels
                ],
            }
        )
        if (
            self.plan_sha256 != plan.plan_sha256
            or self.cohort_manifest_sha256 != verified_cohort.manifest_sha256
            or not self.labels
            or expected_order != self.labels
            or len({item.condition_id for item in self.labels}) != len(self.labels)
            or set(references) != {item.condition_id for item in self.labels}
            or any(
                label.source_run_id != references[label.condition_id].source_run_id
                or label.event_start_ms != references[label.condition_id].event_start_ms
                for label in self.labels
            )
            or self.development_dataset_sha256 != expected_dataset_sha256
            or self.target_manifest_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 development target manifest integrity differs")
        return self


def build_round17_development_target_manifest(
    plan: Round17CohortPlan,
    cohort: Round17CohortManifest,
    labels: Sequence[Round17ConditionLabel],
) -> Round17DevelopmentTargetManifest:
    verified_cohort = cohort.validated(plan)
    selected = tuple(
        sorted(
            (item.validated() for item in labels),
            key=lambda item: (item.event_start_ms, item.condition_id),
        )
    )
    if (
        not selected
        or len({item.condition_id for item in selected}) != len(selected)
        or {item.condition_id for item in selected}
        != {item.condition_id for item in verified_cohort.conditions}
    ):
        raise ValueError("Round 17 development labels are incomplete or duplicated")
    dataset_sha256 = _canonical_sha256(
        {
            "schema_version": "polymarket-round17-development-dataset-v1",
            "cohort_dataset_sha256": verified_cohort.cohort_dataset_sha256,
            "labels": [[item.condition_id, item.label_sha256] for item in selected],
        }
    )
    provisional = Round17DevelopmentTargetManifest(
        plan_sha256=plan.plan_sha256,
        cohort_manifest_sha256=verified_cohort.manifest_sha256,
        labels=selected,
        development_dataset_sha256=dataset_sha256,
    )
    return replace(
        provisional,
        target_manifest_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated(plan, verified_cohort)


def build_round17_development_panel(
    plan: Round17CohortPlan,
    manifest: Round17CohortManifest,
    target_manifest: Round17DevelopmentTargetManifest,
    *,
    role: str,
    datasets: Sequence[PolymarketRound17ConditionDataset],
) -> Round17DevelopmentPanel:
    """Build one development role while structurally refusing test-role input."""

    selected_role = str(role or "").strip()
    if selected_role not in _DEVELOPMENT_ROLES:
        raise ValueError("Round 17 development panel role differs")
    verified_manifest = manifest.validated(plan)
    verified_targets = target_manifest.validated(plan, verified_manifest)
    references = tuple(
        item for item in verified_manifest.conditions if item.role == selected_role
    )
    if not references:
        raise ValueError("Round 17 development role has no admitted conditions")
    reference_by_condition = {item.condition_id: item for item in references}
    selected_datasets = tuple(datasets)
    dataset_by_condition = {
        item.condition_id: item.validated() for item in selected_datasets
    }
    label_by_condition = {
        item.condition_id: item
        for item in verified_targets.labels
        if item.condition_id in reference_by_condition
    }
    if (
        len(dataset_by_condition) != len(selected_datasets)
        or set(dataset_by_condition) != set(reference_by_condition)
        or set(label_by_condition) != set(reference_by_condition)
    ):
        raise ValueError("Round 17 development role inputs differ")
    condition_ids: list[str] = []
    event_starts: list[int] = []
    decision_times: list[int] = []
    features: list[tuple[float, ...]] = []
    targets: list[float] = []
    for reference in references:
        dataset = dataset_by_condition[reference.condition_id]
        label = label_by_condition[reference.condition_id]
        if (
            dataset.run_id != reference.source_run_id
            or dataset.event_start_ms != reference.event_start_ms
            or dataset.event_end_ms != reference.event_end_ms
            or dataset.admission_sha256 != reference.admission_sha256
            or dataset.dataset_sha256 != reference.condition_dataset_sha256
            or len(dataset.rows) != reference.feature_row_count
            or label.source_run_id != reference.source_run_id
            or label.event_start_ms != reference.event_start_ms
        ):
            raise ValueError("Round 17 development condition evidence differs")
        for row in dataset.rows:
            condition_ids.append(reference.condition_id)
            event_starts.append(reference.event_start_ms)
            decision_times.append(row.decision_time_ms)
            features.append(row.values)
            targets.append(label.target_up)
    return Round17DevelopmentPanel(
        role=selected_role,
        condition_ids=np.asarray(condition_ids, dtype=object),
        event_start_ms=np.asarray(event_starts, dtype=np.int64),
        decision_time_ms=np.asarray(decision_times, dtype=np.int64),
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(targets, dtype=np.float64),
        dataset_sha256=verified_targets.development_dataset_sha256,
        target_manifest_sha256=verified_targets.target_manifest_sha256,
    ).validate()


__all__ = [
    "POLYMARKET_ROUND17_COHORT_MANIFEST_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_COHORT_PLAN_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_COHORT_PLAN_SHA256",
    "POLYMARKET_ROUND17_CONDITION_LABEL_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_DEVELOPMENT_TARGET_MANIFEST_SCHEMA_VERSION",
    "Round17CohortCondition",
    "Round17CohortManifest",
    "Round17CohortPlan",
    "Round17CohortRoleWindow",
    "Round17ConditionLabel",
    "Round17DevelopmentTargetManifest",
    "build_round17_cohort_condition",
    "build_round17_cohort_manifest",
    "build_round17_condition_label",
    "build_round17_development_panel",
    "build_round17_development_target_manifest",
    "load_round17_cohort_plan",
    "validate_round17_cohort_plan",
]
