"""Leakage-resistant Round 25 development dataset and endpoint boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Sequence

from .polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
)
from .polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
    Round25JointFeatureSnapshot,
)
from .polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_CONDITION_DURATION_MS,
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


POLYMARKET_ROUND25_DATASET_SCHEMA_VERSION = (
    "polymarket-round25-development-dataset-v1"
)
POLYMARKET_ROUND25_RESOLUTION_AUTHORITY_SCHEMA_VERSION = (
    "polymarket-round25-official-resolution-authority-v1"
)
POLYMARKET_ROUND25_CAPTURE_PLAN_SHA256 = (
    "a0b5525697c3c1e1b175bd0f0ac724fdb62845638d2040e9964221031d3e7b20"
)
POLYMARKET_ROUND25_CAMPAIGN_START_MS = 1_786_406_400_000
POLYMARKET_ROUND25_TRAIN_END_MS = 1_787_443_200_000
POLYMARKET_ROUND25_CALIBRATION_END_MS = 1_787_745_600_000
POLYMARKET_ROUND25_SELECTION_END_MS = 1_788_046_800_000
POLYMARKET_ROUND25_DEVELOPMENT_ROLES = ("train", "calibration", "selection")
POLYMARKET_ROUND25_MINIMUM_CONDITIONS = {
    "train": 2000,
    "calibration": 400,
    "selection": 400,
}
POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION = 16
POLYMARKET_ROUND25_ENDPOINTS_PER_PHASE = 4
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def round25_development_role(event_start_ms: int) -> str:
    if type(event_start_ms) is not int:
        raise ValueError("Round 25 event start is invalid")
    start = event_start_ms
    if (
        start < POLYMARKET_ROUND25_CAMPAIGN_START_MS
        or start >= POLYMARKET_ROUND25_SELECTION_END_MS
        or start % POLYMARKET_ROUND25_CONDITION_DURATION_MS
    ):
        raise ValueError("Round 25 event start is outside the development campaign")
    purge = {
        POLYMARKET_ROUND25_TRAIN_END_MS - POLYMARKET_ROUND25_CONDITION_DURATION_MS,
        POLYMARKET_ROUND25_TRAIN_END_MS,
        POLYMARKET_ROUND25_CALIBRATION_END_MS
        - POLYMARKET_ROUND25_CONDITION_DURATION_MS,
        POLYMARKET_ROUND25_CALIBRATION_END_MS,
    }
    if start in purge:
        return "purged"
    if start < POLYMARKET_ROUND25_TRAIN_END_MS:
        return "train"
    if start < POLYMARKET_ROUND25_CALIBRATION_END_MS:
        return "calibration"
    return "selection"


@dataclass(frozen=True, slots=True)
class Round25ResolutionAuthority:
    terminal_transport_sha256: str
    official_resolution_audit_sha256: str
    created_at_ms: int
    official_resolution_semantics_verified: bool
    source_campaign_plan_sha256: str = POLYMARKET_ROUND25_CAPTURE_PLAN_SHA256
    candidate_design_sha256: str = POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
    candidate_amendment_sha256: str = (
        POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
    )
    schema_version: str = POLYMARKET_ROUND25_RESOLUTION_AUTHORITY_SCHEMA_VERSION
    authority_sha256: str = ""
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "candidate_amendment_sha256": self.candidate_amendment_sha256,
            "candidate_design_sha256": self.candidate_design_sha256,
            "created_at_ms": self.created_at_ms,
            "official_resolution_audit_sha256": (
                self.official_resolution_audit_sha256
            ),
            "official_resolution_semantics_verified": (
                self.official_resolution_semantics_verified
            ),
            "schema_version": self.schema_version,
            "source_campaign_plan_sha256": self.source_campaign_plan_sha256,
            "terminal_transport_sha256": self.terminal_transport_sha256,
            "trading_authority": self.trading_authority,
        }

    def validated(self) -> Round25ResolutionAuthority:
        expected = _canonical_sha256(self.identity_payload())
        if (
            any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in (
                    self.terminal_transport_sha256,
                    self.official_resolution_audit_sha256,
                    self.source_campaign_plan_sha256,
                    self.candidate_design_sha256,
                    self.candidate_amendment_sha256,
                )
            )
            or type(self.created_at_ms) is not int
            or self.created_at_ms <= POLYMARKET_ROUND25_SELECTION_END_MS
            or self.official_resolution_semantics_verified is not True
            or self.source_campaign_plan_sha256
            != POLYMARKET_ROUND25_CAPTURE_PLAN_SHA256
            or self.candidate_design_sha256
            != POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
            or self.candidate_amendment_sha256
            != POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
            or self.schema_version
            != POLYMARKET_ROUND25_RESOLUTION_AUTHORITY_SCHEMA_VERSION
            or self.trading_authority is not False
            or self.authority_sha256 != expected
        ):
            raise ValueError("Round 25 resolution authority differs")
        return self

    @classmethod
    def create(
        cls,
        *,
        terminal_transport_sha256: str,
        official_resolution_audit_sha256: str,
        created_at_ms: int,
        official_resolution_semantics_verified: bool,
    ) -> Round25ResolutionAuthority:
        provisional = cls(
            terminal_transport_sha256=terminal_transport_sha256,
            official_resolution_audit_sha256=official_resolution_audit_sha256,
            created_at_ms=created_at_ms,
            official_resolution_semantics_verified=(
                official_resolution_semantics_verified
            ),
        )
        return replace(
            provisional,
            authority_sha256=_canonical_sha256(provisional.identity_payload()),
        ).validated()


@dataclass(frozen=True, slots=True)
class Round25OfficialResolution:
    condition_id: str
    event_start_ms: int
    up_token_id: str
    down_token_id: str
    winning_token_id: str
    resolved_at_ms: int
    official_payload_sha256: str
    resolution_authority_sha256: str
    target_origin: str = "official_polymarket_resolved_outcome"
    resolution_sha256: str = ""

    @property
    def target_up(self) -> bool:
        return self.winning_token_id == self.up_token_id

    def identity_payload(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "down_token_id": self.down_token_id,
            "event_start_ms": self.event_start_ms,
            "official_payload_sha256": self.official_payload_sha256,
            "resolution_authority_sha256": self.resolution_authority_sha256,
            "resolved_at_ms": self.resolved_at_ms,
            "target_origin": self.target_origin,
            "up_token_id": self.up_token_id,
            "winning_token_id": self.winning_token_id,
        }

    def validated(
        self, authority: Round25ResolutionAuthority
    ) -> Round25OfficialResolution:
        authority.validated()
        if (
            not isinstance(self.condition_id, str)
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or type(self.event_start_ms) is not int
            or round25_development_role(self.event_start_ms) == "purged"
            or any(
                not isinstance(value, str) or _TOKEN_ID.fullmatch(value) is None
                for value in (
                    self.up_token_id,
                    self.down_token_id,
                    self.winning_token_id,
                )
            )
            or self.up_token_id == self.down_token_id
            or self.winning_token_id not in {self.up_token_id, self.down_token_id}
            or type(self.resolved_at_ms) is not int
            or self.resolved_at_ms
            < self.event_start_ms + POLYMARKET_ROUND25_CONDITION_DURATION_MS
            or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in (
                    self.official_payload_sha256,
                    self.resolution_authority_sha256,
                    self.resolution_sha256,
                )
            )
            or self.resolution_authority_sha256 != authority.authority_sha256
            or self.target_origin != "official_polymarket_resolved_outcome"
            or self.resolution_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 official resolution differs")
        return self

    @classmethod
    def create(
        cls,
        *,
        condition_id: str,
        event_start_ms: int,
        up_token_id: str,
        down_token_id: str,
        winning_token_id: str,
        resolved_at_ms: int,
        official_payload_sha256: str,
        authority: Round25ResolutionAuthority,
    ) -> Round25OfficialResolution:
        provisional = cls(
            condition_id=condition_id,
            event_start_ms=event_start_ms,
            up_token_id=up_token_id,
            down_token_id=down_token_id,
            winning_token_id=winning_token_id,
            resolved_at_ms=resolved_at_ms,
            official_payload_sha256=official_payload_sha256,
            resolution_authority_sha256=authority.authority_sha256,
        )
        return replace(
            provisional,
            resolution_sha256=_canonical_sha256(provisional.identity_payload()),
        ).validated(authority)


def select_round25_condition_endpoints(
    snapshots: Sequence[Round25JointFeatureSnapshot],
) -> tuple[Round25JointFeatureSnapshot, ...]:
    rows = tuple(snapshots)
    if not rows:
        raise ValueError("Round 25 condition feature rows are empty")
    first = rows[0]
    if any(
        not isinstance(row, Round25JointFeatureSnapshot)
        or not row.available
        or row.condition_id != first.condition_id
        or row.event_start_ms != first.event_start_ms
        or row.model_design_sha256 != POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
        for row in rows
    ):
        raise ValueError("Round 25 condition feature rows differ")
    decisions = [row.decision_time_ms for row in rows]
    if len(set(decisions)) != len(decisions):
        raise ValueError("Round 25 condition feature decisions are duplicated")
    phases: list[list[Round25JointFeatureSnapshot]] = [[], [], [], []]
    for row in rows:
        offset = row.decision_time_ms - row.event_start_ms
        if not 0 <= offset < POLYMARKET_ROUND25_CONDITION_DURATION_MS:
            raise ValueError("Round 25 condition feature decision is outside event")
        phase = min(3, offset * 4 // POLYMARKET_ROUND25_CONDITION_DURATION_MS)
        phases[phase].append(row)
    selected: list[Round25JointFeatureSnapshot] = []
    for phase, values in enumerate(phases):
        if len(values) < POLYMARKET_ROUND25_ENDPOINTS_PER_PHASE:
            raise ValueError(f"Round 25 condition phase {phase} is underfilled")
        ranked = sorted(
            values,
            key=lambda row: _canonical_sha256(
                {
                    "candidate_amendment_sha256": (
                        POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
                    ),
                    "condition_id": row.condition_id,
                    "decision_time_ms": row.decision_time_ms,
                    "phase": phase,
                }
            ),
        )
        selected.extend(ranked[:POLYMARKET_ROUND25_ENDPOINTS_PER_PHASE])
    return tuple(sorted(selected, key=lambda row: row.decision_time_ms))


@dataclass(frozen=True, slots=True)
class Round25DevelopmentSample:
    role: str
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    feature_values: tuple[float, ...]
    market_prior_probability: float
    target_up: bool
    endpoint_weight: float
    feature_source_chain_sha256: str
    resolution_sha256: str
    sample_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "decision_time_ms": self.decision_time_ms,
            "endpoint_weight": self.endpoint_weight,
            "event_start_ms": self.event_start_ms,
            "feature_source_chain_sha256": self.feature_source_chain_sha256,
            "feature_values": list(self.feature_values),
            "market_prior_probability": self.market_prior_probability,
            "resolution_sha256": self.resolution_sha256,
            "role": self.role,
            "target_up": self.target_up,
        }

    def __post_init__(self) -> None:
        if (
            self.role not in POLYMARKET_ROUND25_DEVELOPMENT_ROLES
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or type(self.event_start_ms) is not int
            or round25_development_role(self.event_start_ms) != self.role
            or type(self.decision_time_ms) is not int
            or not self.event_start_ms
            <= self.decision_time_ms
            < self.event_start_ms + POLYMARKET_ROUND25_CONDITION_DURATION_MS
            or len(self.feature_values) != len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in self.feature_values)
            or not 0.0 < self.market_prior_probability < 1.0
            or type(self.target_up) is not bool
            or self.endpoint_weight != 1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.feature_source_chain_sha256,
                    self.resolution_sha256,
                    self.sample_sha256,
                )
            )
            or self.sample_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 development sample differs")


@dataclass(frozen=True, slots=True)
class Round25DevelopmentDataset:
    role: str
    samples: tuple[Round25DevelopmentSample, ...]
    condition_count: int
    minimum_condition_count: int
    minimum_gate_passed: bool
    resolution_authority_sha256: str
    dataset_sha256: str
    schema_version: str = POLYMARKET_ROUND25_DATASET_SCHEMA_VERSION
    feature_schema_version: str = POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION
    candidate_design_sha256: str = POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
    candidate_amendment_sha256: str = POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "candidate_amendment_sha256": self.candidate_amendment_sha256,
            "candidate_design_sha256": self.candidate_design_sha256,
            "condition_count": self.condition_count,
            "feature_schema_version": self.feature_schema_version,
            "minimum_condition_count": self.minimum_condition_count,
            "minimum_gate_passed": self.minimum_gate_passed,
            "resolution_authority_sha256": self.resolution_authority_sha256,
            "role": self.role,
            "sample_sha256": [sample.sample_sha256 for sample in self.samples],
            "schema_version": self.schema_version,
            "trading_authority": self.trading_authority,
        }

    def __post_init__(self) -> None:
        conditions = {sample.condition_id for sample in self.samples}
        if (
            self.role not in POLYMARKET_ROUND25_DEVELOPMENT_ROLES
            or not self.samples
            or any(sample.role != self.role for sample in self.samples)
            or len(self.samples)
            != len(conditions) * POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
            or self.condition_count != len(conditions)
            or self.minimum_condition_count
            != POLYMARKET_ROUND25_MINIMUM_CONDITIONS[self.role]
            or self.minimum_gate_passed
            is not (self.condition_count >= self.minimum_condition_count)
            or _SHA256.fullmatch(self.resolution_authority_sha256) is None
            or self.schema_version != POLYMARKET_ROUND25_DATASET_SCHEMA_VERSION
            or self.feature_schema_version
            != POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION
            or self.candidate_design_sha256
            != POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
            or self.candidate_amendment_sha256
            != POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
            or self.trading_authority is not False
            or self.dataset_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 development dataset differs")


def build_round25_development_dataset(
    *,
    role: str,
    snapshots: Sequence[Round25JointFeatureSnapshot],
    resolutions: Sequence[Round25OfficialResolution],
    resolution_authority: Round25ResolutionAuthority,
) -> Round25DevelopmentDataset:
    selected_role = str(role or "").strip()
    if selected_role not in POLYMARKET_ROUND25_DEVELOPMENT_ROLES:
        raise ValueError("Round 25 development role is invalid or sealed")
    authority = resolution_authority.validated()
    grouped: dict[str, list[Round25JointFeatureSnapshot]] = {}
    for row in snapshots:
        if not isinstance(row, Round25JointFeatureSnapshot) or not row.available:
            raise ValueError("Round 25 development feature row is ineligible")
        observed_role = round25_development_role(row.event_start_ms)
        if observed_role == "purged":
            continue
        if observed_role != selected_role:
            raise ValueError("Round 25 development feature role differs")
        grouped.setdefault(row.condition_id, []).append(row)
    resolution_map: dict[str, Round25OfficialResolution] = {}
    for resolution in resolutions:
        validated = resolution.validated(authority)
        observed_role = round25_development_role(validated.event_start_ms)
        if observed_role == "purged":
            continue
        if observed_role != selected_role or validated.condition_id in resolution_map:
            raise ValueError("Round 25 development resolution role differs")
        resolution_map[validated.condition_id] = validated
    if not grouped or set(grouped) != set(resolution_map):
        raise ValueError("Round 25 feature and official-resolution populations differ")

    samples: list[Round25DevelopmentSample] = []
    for condition_id in sorted(
        grouped,
        key=lambda item: (grouped[item][0].event_start_ms, item),
    ):
        resolution = resolution_map[condition_id]
        endpoints = select_round25_condition_endpoints(grouped[condition_id])
        if resolution.event_start_ms != endpoints[0].event_start_ms:
            raise ValueError("Round 25 feature and official resolution event differ")
        for row in endpoints:
            sample_payload = {
                "condition_id": condition_id,
                "decision_time_ms": row.decision_time_ms,
                "endpoint_weight": (
                    1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
                ),
                "event_start_ms": row.event_start_ms,
                "feature_source_chain_sha256": row.source_chain_sha256,
                "feature_values": list(row.values),
                "market_prior_probability": row.market_prior_probability,
                "resolution_sha256": resolution.resolution_sha256,
                "role": selected_role,
                "target_up": resolution.target_up,
            }
            samples.append(Round25DevelopmentSample(
                role=selected_role,
                condition_id=condition_id,
                event_start_ms=row.event_start_ms,
                decision_time_ms=row.decision_time_ms,
                feature_values=row.values,
                market_prior_probability=row.market_prior_probability,
                target_up=resolution.target_up,
                endpoint_weight=1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
                feature_source_chain_sha256=row.source_chain_sha256,
                resolution_sha256=resolution.resolution_sha256,
                sample_sha256=_canonical_sha256(sample_payload),
            ))
    ordered = tuple(samples)
    condition_count = len(grouped)
    dataset_payload = {
        "candidate_amendment_sha256": (
            POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
        ),
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "condition_count": condition_count,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "minimum_condition_count": (
            POLYMARKET_ROUND25_MINIMUM_CONDITIONS[selected_role]
        ),
        "minimum_gate_passed": (
            condition_count >= POLYMARKET_ROUND25_MINIMUM_CONDITIONS[selected_role]
        ),
        "resolution_authority_sha256": authority.authority_sha256,
        "role": selected_role,
        "sample_sha256": [sample.sample_sha256 for sample in ordered],
        "schema_version": POLYMARKET_ROUND25_DATASET_SCHEMA_VERSION,
        "trading_authority": False,
    }
    return Round25DevelopmentDataset(
        role=selected_role,
        samples=ordered,
        condition_count=condition_count,
        minimum_condition_count=POLYMARKET_ROUND25_MINIMUM_CONDITIONS[selected_role],
        minimum_gate_passed=(
            condition_count >= POLYMARKET_ROUND25_MINIMUM_CONDITIONS[selected_role]
        ),
        resolution_authority_sha256=authority.authority_sha256,
        dataset_sha256=_canonical_sha256(dataset_payload),
    )


def require_round25_dataset_minimum(
    dataset: Round25DevelopmentDataset,
) -> Round25DevelopmentDataset:
    if not isinstance(dataset, Round25DevelopmentDataset):
        raise TypeError("Round 25 dataset type differs")
    if not dataset.minimum_gate_passed:
        raise ValueError("Round 25 dataset minimum condition gate failed")
    return dataset


__all__ = [
    "POLYMARKET_ROUND25_CALIBRATION_END_MS",
    "POLYMARKET_ROUND25_CAMPAIGN_START_MS",
    "POLYMARKET_ROUND25_CAPTURE_PLAN_SHA256",
    "POLYMARKET_ROUND25_DATASET_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_DEVELOPMENT_ROLES",
    "POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION",
    "POLYMARKET_ROUND25_MINIMUM_CONDITIONS",
    "POLYMARKET_ROUND25_SELECTION_END_MS",
    "POLYMARKET_ROUND25_TRAIN_END_MS",
    "Round25DevelopmentDataset",
    "Round25DevelopmentSample",
    "Round25OfficialResolution",
    "Round25ResolutionAuthority",
    "build_round25_development_dataset",
    "require_round25_dataset_minimum",
    "round25_development_role",
    "select_round25_condition_endpoints",
]
