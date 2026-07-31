"""Causal, role-scoped dataset boundary for Polymarket Round 21."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .polymarket_round21_contract import POLYMARKET_ROUND21_CONTRACT_SHA256
from .polymarket_round21_model import (
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    Round21DevelopmentPanel,
)


POLYMARKET_ROUND21_DATASET_DESIGN_SCHEMA_VERSION = (
    "polymarket-round21-causal-dataset-design-v1"
)
POLYMARKET_ROUND21_DATASET_SCHEMA_VERSION = (
    "polymarket-round21-causal-development-dataset-v1"
)
POLYMARKET_ROUND21_CAMPAIGN_DURATION_MS = 30 * 86_400_000
POLYMARKET_ROUND21_CONDITION_DURATION_MS = 300_000
POLYMARKET_ROUND21_DECISION_CADENCE_MS = 250
POLYMARKET_ROUND21_MAXIMUM_LOOKBACK_MS = 120_000
POLYMARKET_ROUND21_PURGE_MS = 1_800_000
POLYMARKET_ROUND21_ROLE_INTERVALS = (
    ("train", 0, 1_555_200_000),
    ("purge_train_to_tune", 1_555_200_000, 1_557_000_000),
    ("tune_calibration", 1_557_000_000, 1_771_200_000),
    ("tune_selection", 1_771_200_000, 1_987_200_000),
    ("purge_tune_to_test", 1_987_200_000, 1_989_000_000),
    ("test", 1_989_000_000, 2_592_000_000),
)
_DEVELOPMENT_ROLES = frozenset(
    ("train", "tune_calibration", "tune_selection")
)
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_MAX_DESIGN_BYTES = 256 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Round 21 dataset design contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 dataset design contains {value}")


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


def _section(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    selected = value.get(name)
    if not isinstance(selected, Mapping):
        raise ValueError(f"Round 21 dataset {name} section is unavailable")
    return selected


def _finite_values(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f"Round 21 {name} feature vector is invalid")
    selected = tuple(float(value) for value in values)
    if not selected or any(not math.isfinite(value) for value in selected):
        raise ValueError(f"Round 21 {name} feature vector is invalid")
    return selected


def _feature_names(values: Sequence[str], *, layer: str) -> tuple[str, ...]:
    selected = tuple(str(value or "").strip() for value in values)
    if (
        not selected
        or any(not value or len(value) > 160 for value in selected)
        or len(set(selected)) != len(selected)
        or any(not value.startswith(f"{layer}.") for value in selected)
    ):
        raise ValueError(f"Round 21 {layer} feature names are invalid")
    return selected


def validate_round21_dataset_design(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Reject rehashed changes to role, causality, optionality, or authority."""

    design = dict(value)
    claimed = str(design.pop("design_sha256", "")).strip().lower()
    expected_top_level = {
        "schema_version",
        "round",
        "status",
        "parents",
        "partition",
        "feature_rows",
        "join",
        "targets",
        "artifacts",
        "authority",
    }
    parents = _section(design, "parents")
    partition = _section(design, "partition")
    feature_rows = _section(design, "feature_rows")
    join = _section(design, "join")
    targets = _section(design, "targets")
    artifacts = _section(design, "artifacts")
    authority = _section(design, "authority")
    intervals = partition.get("intervals")
    expected_intervals = [
        {"role": role, "start_offset_ms": start, "end_offset_ms": end}
        for role, start, end in POLYMARKET_ROUND21_ROLE_INTERVALS
    ]
    if (
        set(design) != expected_top_level
        or claimed != POLYMARKET_ROUND21_DATASET_DESIGN_SHA256
        or claimed != _canonical_sha256(design)
        or design.get("schema_version")
        != POLYMARKET_ROUND21_DATASET_DESIGN_SCHEMA_VERSION
        or design.get("round") != 21
        or design.get("status")
        != "preregistered_during_target_and_model_blind_capture"
        or parents.get("round21_contract_sha256")
        != POLYMARKET_ROUND21_CONTRACT_SHA256
        or partition.get("campaign_duration_ms")
        != POLYMARKET_ROUND21_CAMPAIGN_DURATION_MS
        or partition.get("condition_duration_ms")
        != POLYMARKET_ROUND21_CONDITION_DURATION_MS
        or partition.get("decision_cadence_ms")
        != POLYMARKET_ROUND21_DECISION_CADENCE_MS
        or intervals != expected_intervals
        or partition.get("whole_condition_assignment") is not True
        or partition.get("purge_seconds_between_calendar_roles") != 1800
        or partition.get("calendar_windows")
        != {"train_days": 18, "tune_days": 5, "test_days": 7}
        or partition.get("test_role_is_sealed") is not True
        or partition.get("development_builder_may_access_test_features")
        is not False
        or partition.get("development_builder_may_access_test_targets")
        is not False
        or feature_rows.get("maximum_lookback_ms")
        != POLYMARKET_ROUND21_MAXIMUM_LOOKBACK_MS
        or feature_rows.get("core_required") is not True
        or feature_rows.get("optional_spot") is not True
        or feature_rows.get("optional_usdm_requires_spot") is not True
        or feature_rows.get("missing_optional_values")
        != "exact_zero_vector_plus_false_availability"
        or feature_rows.get("forward_or_backward_fill") is not False
        or feature_rows.get("cross_gap_carry") is not False
        or feature_rows.get("source_timestamp_use") != "audit_only"
        or feature_rows.get("inference_clock") != "local_utc_receipt_time"
        or feature_rows.get("maximum_source_receipt_ms")
        != "less_than_or_equal_to_decision_time_ms"
        or feature_rows.get("future_receipts") != "rejected"
        or feature_rows.get("future_books") != "rejected"
        or feature_rows.get("future_reference_prices") != "rejected"
        or feature_rows.get("outcomes_or_resolution_in_features") != "rejected"
        or feature_rows.get("fees_fills_orders_or_pnl_in_features") != "rejected"
        or join.get("optional_rows_may_expand_core_population") is not False
        or join.get("key") != ["condition_id", "decision_time_ms"]
        or join.get("optional_join")
        != "exact_key_and_causally_available_receipts_only"
        or join.get("optional_rows_may_change_condition_role") is not False
        or join.get("optional_rows_may_change_core_admission") is not False
        or join.get("duplicate_keys") != "rejected"
        or join.get("unmatched_optional_rows") != "rejected"
        or targets.get("one_official_resolution_per_condition") is not True
        or targets.get("resolution_must_be_observed_after_event_end") is not True
        or targets.get("condition_and_event_start_must_match_feature_identity")
        is not True
        or targets.get("target_source_and_payload_sha256_required") is not True
        or targets.get("condition_equal_weighting") is not True
        or targets.get("test_target_access_before_one_use_unlock") is not False
        or artifacts
        != {
            "canonical_json": True,
            "feature_name_hashes": True,
            "row_value_hashes": True,
            "source_chain_hashes": True,
            "partition_policy_hash": True,
            "dataset_hash": True,
            "target_manifest_hash": True,
            "hand_edited_numeric_result_authority": False,
        }
        or authority
        != {
            "model_data_eligible": False,
            "model_selected": False,
            "ai_edge_claim": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
    ):
        raise ValueError("Round 21 dataset design differs")
    return {**design, "design_sha256": claimed}


def load_round21_dataset_design(path: str | Path) -> dict[str, object]:
    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= _MAX_DESIGN_BYTES
    ):
        raise ValueError("Round 21 dataset design is unavailable")
    try:
        value = json.loads(
            selected.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 dataset design is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 dataset design is not an object")
    return validate_round21_dataset_design(value)


@dataclass(frozen=True, slots=True)
class Round21FeatureSchema:
    core_names: tuple[str, ...]
    spot_names: tuple[str, ...]
    usdm_names: tuple[str, ...]
    feature_policy_sha256: str
    schema_sha256: str

    @classmethod
    def create(
        cls,
        *,
        core_names: Sequence[str],
        spot_names: Sequence[str],
        usdm_names: Sequence[str],
        feature_policy_sha256: str,
    ) -> Round21FeatureSchema:
        core = _feature_names(core_names, layer="core")
        spot = _feature_names(spot_names, layer="spot")
        usdm = _feature_names(usdm_names, layer="usdm")
        policy_sha256 = str(feature_policy_sha256 or "").strip().lower()
        if (
            _SHA256.fullmatch(policy_sha256) is None
            or policy_sha256 == _EMPTY_SHA256
        ):
            raise ValueError("Round 21 feature policy identity is invalid")
        payload = {
            "schema_version": POLYMARKET_ROUND21_DATASET_SCHEMA_VERSION,
            "dataset_design_sha256": POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
            "feature_policy_sha256": policy_sha256,
            "core_names": list(core),
            "spot_names": list(spot),
            "usdm_names": list(usdm),
        }
        return cls(
            core_names=core,
            spot_names=spot,
            usdm_names=usdm,
            feature_policy_sha256=policy_sha256,
            schema_sha256=_canonical_sha256(payload),
        )

    def validated(self) -> Round21FeatureSchema:
        rebuilt = self.create(
            core_names=self.core_names,
            spot_names=self.spot_names,
            usdm_names=self.usdm_names,
            feature_policy_sha256=self.feature_policy_sha256,
        )
        if self != rebuilt:
            raise ValueError("Round 21 feature schema differs")
        return self

    @property
    def core_names_sha256(self) -> str:
        return _canonical_sha256(list(self.core_names))

    @property
    def spot_names_sha256(self) -> str:
        return _canonical_sha256(list(self.spot_names))

    @property
    def usdm_names_sha256(self) -> str:
        return _canonical_sha256(list(self.usdm_names))


@dataclass(frozen=True, slots=True)
class Round21PartitionPolicy:
    campaign_start_ms: int
    campaign_end_ms: int
    policy_sha256: str

    @classmethod
    def create(
        cls,
        *,
        campaign_start_ms: int,
        campaign_end_ms: int,
    ) -> Round21PartitionPolicy:
        start = int(campaign_start_ms)
        end = int(campaign_end_ms)
        if (
            start <= 0
            or start % POLYMARKET_ROUND21_CONDITION_DURATION_MS
            or end - start != POLYMARKET_ROUND21_CAMPAIGN_DURATION_MS
        ):
            raise ValueError("Round 21 campaign boundary is invalid")
        payload = {
            "schema_version": POLYMARKET_ROUND21_DATASET_SCHEMA_VERSION,
            "dataset_design_sha256": POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
            "campaign_start_ms": start,
            "campaign_end_ms": end,
            "role_intervals": [
                {"role": role, "start_offset_ms": first, "end_offset_ms": last}
                for role, first, last in POLYMARKET_ROUND21_ROLE_INTERVALS
            ],
        }
        return cls(
            campaign_start_ms=start,
            campaign_end_ms=end,
            policy_sha256=_canonical_sha256(payload),
        )

    def validated(self) -> Round21PartitionPolicy:
        rebuilt = self.create(
            campaign_start_ms=self.campaign_start_ms,
            campaign_end_ms=self.campaign_end_ms,
        )
        if self != rebuilt:
            raise ValueError("Round 21 partition policy differs")
        return self

    def role_for_event_start(self, event_start_ms: int) -> str:
        event_start = int(event_start_ms)
        if (
            event_start % POLYMARKET_ROUND21_CONDITION_DURATION_MS
            or not self.campaign_start_ms <= event_start < self.campaign_end_ms
        ):
            raise ValueError("Round 21 event start lies outside the campaign")
        offset = event_start - self.campaign_start_ms
        for role, start, end in POLYMARKET_ROUND21_ROLE_INTERVALS:
            if start <= offset < end:
                return role
        raise RuntimeError("Round 21 role intervals do not cover the campaign")


@dataclass(frozen=True, slots=True)
class Round21CausalFeatureRow:
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    structural_probability: float
    market_prior_probability: float
    core_values: tuple[float, ...]
    spot_values: tuple[float, ...]
    usdm_values: tuple[float, ...]
    spot_available: bool
    usdm_available: bool
    feature_schema_sha256: str
    core_source_chain_sha256: str
    spot_source_chain_sha256: str
    usdm_source_chain_sha256: str
    core_maximum_receipt_ms: int
    spot_maximum_receipt_ms: int
    usdm_maximum_receipt_ms: int
    row_sha256: str
    trading_authority: bool = False

    @classmethod
    def create(
        cls,
        *,
        condition_id: str,
        event_start_ms: int,
        decision_time_ms: int,
        structural_probability: float,
        market_prior_probability: float,
        core_values: Sequence[float],
        spot_values: Sequence[float],
        usdm_values: Sequence[float],
        spot_available: bool,
        usdm_available: bool,
        feature_schema: Round21FeatureSchema,
        core_source_chain_sha256: str,
        spot_source_chain_sha256: str = _EMPTY_SHA256,
        usdm_source_chain_sha256: str = _EMPTY_SHA256,
        core_maximum_receipt_ms: int,
        spot_maximum_receipt_ms: int = 0,
        usdm_maximum_receipt_ms: int = 0,
    ) -> Round21CausalFeatureRow:
        schema = feature_schema.validated()
        condition = str(condition_id or "").strip().lower()
        event_start = int(event_start_ms)
        decision = int(decision_time_ms)
        structural = float(structural_probability)
        market_prior = float(market_prior_probability)
        core = _finite_values(core_values, name="core")
        spot = _finite_values(spot_values, name="spot")
        usdm = _finite_values(usdm_values, name="usdm")
        if type(spot_available) is not bool or type(usdm_available) is not bool:
            raise ValueError("Round 21 causal feature row is invalid")
        spot_flag = bool(spot_available)
        usdm_flag = bool(usdm_available)
        core_chain = str(core_source_chain_sha256 or "").strip().lower()
        spot_chain = str(spot_source_chain_sha256 or "").strip().lower()
        usdm_chain = str(usdm_source_chain_sha256 or "").strip().lower()
        core_receipt = int(core_maximum_receipt_ms)
        spot_receipt = int(spot_maximum_receipt_ms)
        usdm_receipt = int(usdm_maximum_receipt_ms)
        if (
            _CONDITION_ID.fullmatch(condition) is None
            or event_start <= 0
            or event_start % POLYMARKET_ROUND21_CONDITION_DURATION_MS
            or not event_start
            <= decision
            < event_start + POLYMARKET_ROUND21_CONDITION_DURATION_MS
            or (decision - event_start) % POLYMARKET_ROUND21_DECISION_CADENCE_MS
            or not 0.0 < structural < 1.0
            or not 0.0 < market_prior < 1.0
            or len(core) != len(schema.core_names)
            or len(spot) != len(schema.spot_names)
            or len(usdm) != len(schema.usdm_names)
            or _SHA256.fullmatch(core_chain) is None
            or _SHA256.fullmatch(spot_chain) is None
            or _SHA256.fullmatch(usdm_chain) is None
            or core_chain == _EMPTY_SHA256
            or not 0 < core_receipt <= decision
            or usdm_flag and not spot_flag
            or (
                not spot_flag
                and (
                    any(spot)
                    or spot_receipt != 0
                    or spot_chain != _EMPTY_SHA256
                )
            )
            or (
                spot_flag
                and (
                    not 0 < spot_receipt <= decision
                    or spot_chain == _EMPTY_SHA256
                )
            )
            or (
                not usdm_flag
                and (
                    any(usdm)
                    or usdm_receipt != 0
                    or usdm_chain != _EMPTY_SHA256
                )
            )
            or (
                usdm_flag
                and (
                    not 0 < usdm_receipt <= decision
                    or usdm_chain == _EMPTY_SHA256
                )
            )
        ):
            raise ValueError("Round 21 causal feature row is invalid")
        payload = {
            "schema_version": POLYMARKET_ROUND21_DATASET_SCHEMA_VERSION,
            "dataset_design_sha256": POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
            "condition_id": condition,
            "event_start_ms": event_start,
            "decision_time_ms": decision,
            "structural_probability": structural,
            "market_prior_probability": market_prior,
            "core_values_sha256": _canonical_sha256(list(core)),
            "spot_values_sha256": _canonical_sha256(list(spot)),
            "usdm_values_sha256": _canonical_sha256(list(usdm)),
            "spot_available": spot_flag,
            "usdm_available": usdm_flag,
            "feature_schema_sha256": schema.schema_sha256,
            "source_chain_sha256": {
                "core": core_chain,
                "spot": spot_chain,
                "usdm": usdm_chain,
            },
            "maximum_receipt_ms": {
                "core": core_receipt,
                "spot": spot_receipt,
                "usdm": usdm_receipt,
            },
            "trading_authority": False,
        }
        return cls(
            condition_id=condition,
            event_start_ms=event_start,
            decision_time_ms=decision,
            structural_probability=structural,
            market_prior_probability=market_prior,
            core_values=core,
            spot_values=spot,
            usdm_values=usdm,
            spot_available=spot_flag,
            usdm_available=usdm_flag,
            feature_schema_sha256=schema.schema_sha256,
            core_source_chain_sha256=core_chain,
            spot_source_chain_sha256=spot_chain,
            usdm_source_chain_sha256=usdm_chain,
            core_maximum_receipt_ms=core_receipt,
            spot_maximum_receipt_ms=spot_receipt,
            usdm_maximum_receipt_ms=usdm_receipt,
            row_sha256=_canonical_sha256(payload),
        )

    def validated(
        self,
        feature_schema: Round21FeatureSchema,
    ) -> Round21CausalFeatureRow:
        rebuilt = self.create(
            condition_id=self.condition_id,
            event_start_ms=self.event_start_ms,
            decision_time_ms=self.decision_time_ms,
            structural_probability=self.structural_probability,
            market_prior_probability=self.market_prior_probability,
            core_values=self.core_values,
            spot_values=self.spot_values,
            usdm_values=self.usdm_values,
            spot_available=self.spot_available,
            usdm_available=self.usdm_available,
            feature_schema=feature_schema,
            core_source_chain_sha256=self.core_source_chain_sha256,
            spot_source_chain_sha256=self.spot_source_chain_sha256,
            usdm_source_chain_sha256=self.usdm_source_chain_sha256,
            core_maximum_receipt_ms=self.core_maximum_receipt_ms,
            spot_maximum_receipt_ms=self.spot_maximum_receipt_ms,
            usdm_maximum_receipt_ms=self.usdm_maximum_receipt_ms,
        )
        if self != rebuilt or self.trading_authority:
            raise ValueError("Round 21 causal feature row differs")
        return self


@dataclass(frozen=True, slots=True)
class Round21OfficialOutcome:
    condition_id: str
    event_start_ms: int
    resolved_up: bool
    observed_at_ms: int
    source: str
    source_payload_sha256: str
    outcome_sha256: str
    trading_authority: bool = False

    @classmethod
    def create(
        cls,
        *,
        condition_id: str,
        event_start_ms: int,
        resolved_up: bool,
        observed_at_ms: int,
        source: str,
        source_payload_sha256: str,
    ) -> Round21OfficialOutcome:
        condition = str(condition_id or "").strip().lower()
        event_start = int(event_start_ms)
        resolved = resolved_up
        observed = int(observed_at_ms)
        selected_source = str(source or "").strip()
        source_sha = str(source_payload_sha256 or "").strip().lower()
        if (
            _CONDITION_ID.fullmatch(condition) is None
            or event_start <= 0
            or event_start % POLYMARKET_ROUND21_CONDITION_DURATION_MS
            or type(resolved) is not bool
            or observed < event_start + POLYMARKET_ROUND21_CONDITION_DURATION_MS
            or not selected_source
            or len(selected_source) > 160
            or _SHA256.fullmatch(source_sha) is None
            or source_sha == _EMPTY_SHA256
        ):
            raise ValueError("Round 21 official outcome is invalid")
        payload = {
            "schema_version": POLYMARKET_ROUND21_DATASET_SCHEMA_VERSION,
            "condition_id": condition,
            "event_start_ms": event_start,
            "resolved_up": resolved,
            "observed_at_ms": observed,
            "source": selected_source,
            "source_payload_sha256": source_sha,
            "trading_authority": False,
        }
        return cls(
            condition_id=condition,
            event_start_ms=event_start,
            resolved_up=resolved,
            observed_at_ms=observed,
            source=selected_source,
            source_payload_sha256=source_sha,
            outcome_sha256=_canonical_sha256(payload),
        )

    def validated(self) -> Round21OfficialOutcome:
        rebuilt = self.create(
            condition_id=self.condition_id,
            event_start_ms=self.event_start_ms,
            resolved_up=self.resolved_up,
            observed_at_ms=self.observed_at_ms,
            source=self.source,
            source_payload_sha256=self.source_payload_sha256,
        )
        if self != rebuilt or self.trading_authority:
            raise ValueError("Round 21 official outcome differs")
        return self


def build_round21_development_panel(
    *,
    role: str,
    feature_schema: Round21FeatureSchema,
    partition_policy: Round21PartitionPolicy,
    feature_rows: Sequence[Round21CausalFeatureRow],
    outcomes: Sequence[Round21OfficialOutcome],
) -> Round21DevelopmentPanel:
    """Join one development role without accepting sealed-test inputs."""

    selected_role = str(role or "").strip()
    if selected_role not in _DEVELOPMENT_ROLES:
        raise ValueError("Round 21 development role is invalid or sealed")
    schema = feature_schema.validated()
    policy = partition_policy.validated()
    unverified_rows = tuple(feature_rows)
    if not unverified_rows:
        raise ValueError("Round 21 development feature rows are empty")
    if any(
        not isinstance(row, Round21CausalFeatureRow)
        or policy.role_for_event_start(row.event_start_ms) != selected_role
        for row in unverified_rows
    ):
        raise ValueError("Round 21 development rows differ from the frozen role")
    rows = tuple(row.validated(schema) for row in unverified_rows)
    ordered = tuple(
        sorted(rows, key=lambda row: (row.event_start_ms, row.decision_time_ms))
    )
    keys = tuple((row.condition_id, row.decision_time_ms) for row in ordered)
    if len(set(keys)) != len(keys):
        raise ValueError("Round 21 development rows differ from the frozen role")
    unverified_outcomes = tuple(outcomes)
    if any(
        not isinstance(outcome, Round21OfficialOutcome)
        or policy.role_for_event_start(outcome.event_start_ms) != selected_role
        for outcome in unverified_outcomes
    ):
        raise ValueError("Round 21 development outcome population differs")
    verified_outcomes = tuple(
        outcome.validated() for outcome in unverified_outcomes
    )
    outcome_map = {outcome.condition_id: outcome for outcome in verified_outcomes}
    row_conditions = {row.condition_id for row in ordered}
    if (
        len(outcome_map) != len(verified_outcomes)
        or set(outcome_map) != row_conditions
    ):
        raise ValueError("Round 21 development outcome population differs")
    for row in ordered:
        outcome = outcome_map[row.condition_id]
        if outcome.event_start_ms != row.event_start_ms:
            raise ValueError("Round 21 feature and outcome identities differ")
    dataset_sha256 = _canonical_sha256(
        {
            "schema_version": POLYMARKET_ROUND21_DATASET_SCHEMA_VERSION,
            "dataset_design_sha256": POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
            "partition_policy_sha256": policy.policy_sha256,
            "feature_schema_sha256": schema.schema_sha256,
            "role": selected_role,
            "row_sha256": [row.row_sha256 for row in ordered],
        }
    )
    target_manifest_sha256 = _canonical_sha256(
        {
            "schema_version": POLYMARKET_ROUND21_DATASET_SCHEMA_VERSION,
            "dataset_design_sha256": POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
            "partition_policy_sha256": policy.policy_sha256,
            "role": selected_role,
            "outcome_sha256": [
                outcome_map[condition].outcome_sha256
                for condition in sorted(row_conditions)
            ],
        }
    )
    panel = Round21DevelopmentPanel(
        role=selected_role,
        condition_ids=np.asarray(
            [row.condition_id for row in ordered],
            dtype=object,
        ),
        event_start_ms=np.asarray(
            [row.event_start_ms for row in ordered],
            dtype=np.int64,
        ),
        decision_time_ms=np.asarray(
            [row.decision_time_ms for row in ordered],
            dtype=np.int64,
        ),
        labels=np.asarray(
            [
                float(outcome_map[row.condition_id].resolved_up)
                for row in ordered
            ],
            dtype=np.float64,
        ),
        structural_probability=np.asarray(
            [row.structural_probability for row in ordered],
            dtype=np.float64,
        ),
        market_prior_probability=np.asarray(
            [row.market_prior_probability for row in ordered],
            dtype=np.float64,
        ),
        core_features=np.asarray(
            [row.core_values for row in ordered],
            dtype=np.float32,
        ),
        spot_features=np.asarray(
            [row.spot_values for row in ordered],
            dtype=np.float32,
        ),
        usdm_features=np.asarray(
            [row.usdm_values for row in ordered],
            dtype=np.float32,
        ),
        spot_available=np.asarray(
            [row.spot_available for row in ordered],
            dtype=np.bool_,
        ),
        usdm_available=np.asarray(
            [row.usdm_available for row in ordered],
            dtype=np.bool_,
        ),
        core_feature_names_sha256=schema.core_names_sha256,
        spot_feature_names_sha256=schema.spot_names_sha256,
        usdm_feature_names_sha256=schema.usdm_names_sha256,
        dataset_sha256=dataset_sha256,
        target_manifest_sha256=target_manifest_sha256,
        dataset_design_sha256=POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    )
    return panel.validate()


__all__ = [
    "POLYMARKET_ROUND21_CAMPAIGN_DURATION_MS",
    "POLYMARKET_ROUND21_CONDITION_DURATION_MS",
    "POLYMARKET_ROUND21_DATASET_DESIGN_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_DATASET_DESIGN_SHA256",
    "POLYMARKET_ROUND21_DATASET_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_DECISION_CADENCE_MS",
    "POLYMARKET_ROUND21_MAXIMUM_LOOKBACK_MS",
    "POLYMARKET_ROUND21_PURGE_MS",
    "POLYMARKET_ROUND21_ROLE_INTERVALS",
    "Round21CausalFeatureRow",
    "Round21FeatureSchema",
    "Round21OfficialOutcome",
    "Round21PartitionPolicy",
    "build_round21_development_panel",
    "load_round21_dataset_design",
    "validate_round21_dataset_design",
]
