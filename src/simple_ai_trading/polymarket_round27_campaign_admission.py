"""Target-free campaign admission before any Round 27 outcome access."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

from .polymarket_round27_feature_store import (
    POLYMARKET_ROUND27_FEATURE_STORE_SCHEMA_VERSION,
)
from .polymarket_round27_features import Round27FeatureRow
from .polymarket_round27_model import Round27RoleInterval
from .polymarket_round27_model_amendment import (
    POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD,
    POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256,
)
from .polymarket_round27_model_contract import (
    POLYMARKET_ROUND27_MODEL_CONTRACT_SCHEMA_VERSION,
    POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256,
)


POLYMARKET_ROUND27_CAMPAIGN_ADMISSION_SCHEMA_VERSION = (
    "polymarket-round27-campaign-admission-v1"
)
_PRIMARY_SLOTS = ("stage1-a", "stage1-b", "stage1-c")
_CONTINGENCY_SLOT = "stage1-d"
_MODEL_ROLES = ("train", "calibration", "selection", "sealed")
_ALL_ROLES = (*_MODEL_ROLES, "purged")
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_AUTHORITY = {
    "credentials_used": False,
    "edge_claim": False,
    "execution_connected": False,
    "live_trading_authority": False,
    "orders_submitted": False,
    "paper_trading_authority": False,
    "profitability_claim": False,
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 27 campaign {name} SHA-256 differs")
    return selected


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 27 campaign admission has duplicate JSON keys")
        output[key] = value
    return output


def _validated_contract(
    contract: Mapping[str, object],
) -> tuple[dict[str, object], str, str]:
    payload = dict(contract)
    contract_sha256 = _sha256(
        payload.pop("contract_sha256", ""),
        name="model contract",
    )
    amendment_sha256 = _sha256(
        payload.pop(POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD, ""),
        name="model amendment",
    )
    if (
        contract_sha256 != POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256
        or contract_sha256 != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND27_MODEL_CONTRACT_SCHEMA_VERSION
        or amendment_sha256 != POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256
    ):
        raise ValueError("Round 27 campaign model contract differs")
    return payload, contract_sha256, amendment_sha256


def _validated_feature_audit(
    feature_store_audit: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(feature_store_audit)
    claimed = _sha256(payload.pop("audit_sha256", ""), name="feature audit")
    slots = payload.get("slots")
    if (
        claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND27_FEATURE_STORE_SCHEMA_VERSION
        or payload.get("target_columns_present") is not False
        or payload.get("target_accessed") is not False
        or payload.get("trading_authority") is not False
        or not isinstance(slots, list)
        or payload.get("slot_count") != len(slots)
    ):
        raise ValueError("Round 27 campaign feature-store audit differs")
    return {**payload, "audit_sha256": claimed}


def _minimum_population(contract: Mapping[str, object]) -> dict[str, int]:
    raw = contract.get("minimum_population")
    expected = {
        "campaign_eligible_conditions",
        "train_conditions",
        "calibration_conditions",
        "selection_conditions",
        "sealed_conditions",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != expected
        or any(type(raw[name]) is not int or int(raw[name]) <= 0 for name in expected)
    ):
        raise ValueError("Round 27 campaign minimum population differs")
    return {name: int(raw[name]) for name in sorted(expected)}


def _slot_reports(audit: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    raw_slots = audit.get("slots")
    if not isinstance(raw_slots, list):
        raise ValueError("Round 27 campaign feature slots differ")
    output: list[dict[str, object]] = []
    for item in raw_slots:
        if not isinstance(item, Mapping):
            raise ValueError("Round 27 campaign feature slots differ")
        selected = {
            "slot_id": str(item.get("slot_id") or "").lower(),
            "run_id": str(item.get("run_id") or ""),
            "condition_audit_sha256": _sha256(
                item.get("condition_audit_sha256"),
                name="condition audit",
            ),
            "feature_report_sha256": _sha256(
                item.get("feature_report_sha256"),
                name="feature report",
            ),
            "condition_count": item.get("condition_count"),
            "row_count": item.get("row_count"),
            "row_chain_sha256": _sha256(
                item.get("row_chain_sha256"),
                name="row chain",
            ),
        }
        if (
            selected["slot_id"] not in {*_PRIMARY_SLOTS, _CONTINGENCY_SLOT}
            or not selected["run_id"]
            or type(selected["condition_count"]) is not int
            or int(selected["condition_count"]) <= 0
            or type(selected["row_count"]) is not int
            or int(selected["row_count"]) <= 0
        ):
            raise ValueError("Round 27 campaign feature slots differ")
        output.append(selected)
    output.sort(key=lambda item: str(item["slot_id"]))
    slot_ids = [str(item["slot_id"]) for item in output]
    run_ids = [str(item["run_id"]) for item in output]
    if len(slot_ids) != len(set(slot_ids)) or len(run_ids) != len(set(run_ids)):
        raise ValueError("Round 27 campaign feature slots overlap")
    return tuple(output)


def round27_campaign_role_population_sha256(
    rows: Sequence[Round27FeatureRow],
) -> str:
    """Hash one exact chronological role population without outcomes."""

    grouped: dict[str, list[Round27FeatureRow]] = {}
    for raw in rows:
        row = raw.validated()
        grouped.setdefault(row.condition_id, []).append(row)
    conditions: list[dict[str, object]] = []
    for condition_id, condition_rows in grouped.items():
        ordered = sorted(condition_rows, key=lambda row: row.decision_time_ms)
        if (
            len({row.decision_time_ms for row in ordered}) != len(ordered)
            or len({row.event_start_ms for row in ordered}) != 1
        ):
            raise ValueError("Round 27 campaign role feature rows are duplicated")
        chain = hashlib.sha256(b"").hexdigest()
        for row in ordered:
            chain = hashlib.sha256(
                bytes.fromhex(chain) + bytes.fromhex(row.row_sha256)
            ).hexdigest()
        conditions.append(
            {
                "condition_id": condition_id,
                "event_start_ms": ordered[0].event_start_ms,
                "feature_row_chain_sha256": chain,
            }
        )
    if not conditions:
        raise ValueError("Round 27 campaign role feature population is empty")
    conditions.sort(key=lambda item: (item["event_start_ms"], item["condition_id"]))
    return _canonical_sha256(
        {
            "schema_version": "polymarket-round27-role-feature-population-v1",
            "condition_count": len(conditions),
            "conditions": conditions,
        }
    )


def _condition_roles(
    *,
    rows: Sequence[Round27FeatureRow],
    slot_reports: Sequence[Mapping[str, object]],
    partitions: Sequence[Mapping[str, object]],
) -> tuple[dict[str, int], dict[str, str]]:
    intervals = tuple(Round27RoleInterval.from_mapping(item) for item in partitions)
    slot_by_run = {str(item["run_id"]): str(item["slot_id"]) for item in slot_reports}
    conditions: dict[str, tuple[str, int]] = {}
    rows_by_condition: dict[str, list[Round27FeatureRow]] = {}
    row_counts = {str(item["slot_id"]): 0 for item in slot_reports}
    for raw in rows:
        row = raw.validated()
        slot_id = slot_by_run.get(row.run_id)
        if slot_id is None:
            raise ValueError("Round 27 campaign feature row run differs")
        row_counts[slot_id] += 1
        identity = (slot_id, row.event_start_ms)
        existing = conditions.setdefault(row.condition_id, identity)
        if existing != identity:
            raise ValueError("Round 27 campaign condition ownership differs")
        rows_by_condition.setdefault(row.condition_id, []).append(row)
    condition_counts = {str(item["slot_id"]): 0 for item in slot_reports}
    role_counts = {role: 0 for role in _ALL_ROLES}
    rows_by_role: dict[str, list[Round27FeatureRow]] = {
        role: [] for role in _MODEL_ROLES
    }
    for condition_id, (slot_id, event_start_ms) in conditions.items():
        matches = [
            item
            for item in intervals
            if item.slot_id == slot_id and item.start_ms <= event_start_ms < item.end_ms
        ]
        if len(matches) != 1 or matches[0].role not in role_counts:
            raise ValueError("Round 27 campaign condition role differs")
        condition_counts[slot_id] += 1
        role_counts[matches[0].role] += 1
        if matches[0].role in rows_by_role:
            rows_by_role[matches[0].role].extend(rows_by_condition[condition_id])
    for report in slot_reports:
        slot_id = str(report["slot_id"])
        if (
            condition_counts[slot_id] != report["condition_count"]
            or row_counts[slot_id] != report["row_count"]
        ):
            raise ValueError("Round 27 campaign feature population differs")
    return role_counts, {
        role: round27_campaign_role_population_sha256(role_rows)
        for role, role_rows in rows_by_role.items()
    }


def build_round27_campaign_admission(
    *,
    contract: Mapping[str, object],
    feature_store_audit: Mapping[str, object],
    feature_rows: Sequence[Round27FeatureRow],
    admitted_at_ms: int,
) -> dict[str, object]:
    """Build the one target-free gate that precedes every target-store role."""

    contract_body, contract_sha256, amendment_sha256 = _validated_contract(contract)
    audit = _validated_feature_audit(feature_store_audit)
    reports = _slot_reports(audit)
    minimum = _minimum_population(contract_body)
    partitions = contract_body.get("partitions")
    if not isinstance(partitions, list):
        raise ValueError("Round 27 campaign partitions differ")
    roles, role_populations = _condition_roles(
        rows=feature_rows,
        slot_reports=reports,
        partitions=[item for item in partitions if isinstance(item, Mapping)],
    )
    report_by_slot = {str(item["slot_id"]): item for item in reports}
    if any(slot not in report_by_slot for slot in _PRIMARY_SLOTS):
        raise ValueError("Round 27 campaign primary audits are incomplete")
    primary_count = sum(
        int(report_by_slot[slot]["condition_count"]) for slot in _PRIMARY_SLOTS
    )
    minimum_total = minimum["campaign_eligible_conditions"]
    contingency_required = primary_count < minimum_total
    contingency_used = _CONTINGENCY_SLOT in report_by_slot
    expected_slots = (
        {*_PRIMARY_SLOTS, _CONTINGENCY_SLOT}
        if contingency_required
        else set(_PRIMARY_SLOTS)
    )
    total_count = sum(int(item["condition_count"]) for item in reports)
    contingency_count = (
        int(report_by_slot[_CONTINGENCY_SLOT]["condition_count"])
        if contingency_used
        else 0
    )
    if (
        set(report_by_slot) != expected_slots
        or contingency_used is not contingency_required
        or total_count != audit.get("condition_count")
        or len(feature_rows) != audit.get("row_count")
        or len(feature_rows) != sum(int(item["row_count"]) for item in reports)
        or total_count != sum(roles.values())
        or total_count < minimum_total
        or any(
            roles[role] < minimum[f"{role}_conditions"] for role in _MODEL_ROLES
        )
        or type(admitted_at_ms) is not int
        or admitted_at_ms <= max(row.validated().event_start_ms for row in feature_rows) + 300_000
    ):
        raise ValueError("Round 27 campaign admission population differs")
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND27_CAMPAIGN_ADMISSION_SCHEMA_VERSION,
        "admitted_at_ms": admitted_at_ms,
        "campaign_contract_sha256": contract_body["campaign_contract_sha256"],
        "round27_model_contract_sha256": contract_sha256,
        "round27_model_amendment_sha256": amendment_sha256,
        "feature_store_audit_sha256": audit["audit_sha256"],
        "audited_slot_ids": [item["slot_id"] for item in reports],
        "primary_slot_ids": list(_PRIMARY_SLOTS),
        "slot_reports": list(reports),
        "primary_eligible_condition_count": primary_count,
        "contingency_condition_count": contingency_count,
        "eligible_condition_count": total_count,
        "role_condition_counts": roles,
        "role_population_sha256": role_populations,
        "minimum_population": minimum,
        "contingency_required": contingency_required,
        "contingency_used": contingency_used,
        "all_primary_target_free_audits_present": True,
        "target_free": True,
        "target_accessed": False,
        "target_access_authorized": True,
        "authority": _AUTHORITY,
    }
    body["admission_sha256"] = _canonical_sha256(body)
    return body


def validate_round27_campaign_admission(
    value: Mapping[str, object],
    *,
    contract: Mapping[str, object],
    feature_store_audit_sha256: str,
) -> dict[str, object]:
    contract_body, contract_sha256, amendment_sha256 = _validated_contract(contract)
    minimum = _minimum_population(contract_body)
    payload = dict(value)
    claimed = _sha256(payload.pop("admission_sha256", ""), name="admission")
    reports = payload.get("slot_reports")
    roles = payload.get("role_condition_counts")
    role_populations = payload.get("role_population_sha256")
    audited_slots = payload.get("audited_slot_ids")
    if (
        not isinstance(reports, list)
        or not isinstance(roles, Mapping)
        or not isinstance(role_populations, Mapping)
    ):
        raise ValueError("Round 27 campaign admission differs")
    try:
        validated_reports = _slot_reports({"slots": reports})
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Round 27 campaign admission differs") from exc
    if reports != list(validated_reports):
        raise ValueError("Round 27 campaign admission differs")
    report_by_slot = {
        str(item.get("slot_id")): item for item in reports if isinstance(item, Mapping)
    }
    primary_count = payload.get("primary_eligible_condition_count")
    contingency_count = payload.get("contingency_condition_count")
    total_count = payload.get("eligible_condition_count")
    contingency_required = payload.get("contingency_required")
    expected_slots = (
        {*_PRIMARY_SLOTS, _CONTINGENCY_SLOT}
        if contingency_required is True
        else set(_PRIMARY_SLOTS)
    )
    if (
        set(payload)
        != {
            "admitted_at_ms",
            "all_primary_target_free_audits_present",
            "audited_slot_ids",
            "authority",
            "campaign_contract_sha256",
            "contingency_condition_count",
            "contingency_required",
            "contingency_used",
            "eligible_condition_count",
            "feature_store_audit_sha256",
            "minimum_population",
            "primary_eligible_condition_count",
            "primary_slot_ids",
            "role_condition_counts",
            "role_population_sha256",
            "round27_model_amendment_sha256",
            "round27_model_contract_sha256",
            "schema_version",
            "slot_reports",
            "target_access_authorized",
            "target_accessed",
            "target_free",
        }
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND27_CAMPAIGN_ADMISSION_SCHEMA_VERSION
        or payload.get("campaign_contract_sha256")
        != contract_body.get("campaign_contract_sha256")
        or payload.get("round27_model_contract_sha256") != contract_sha256
        or payload.get("round27_model_amendment_sha256") != amendment_sha256
        or payload.get("feature_store_audit_sha256")
        != _sha256(feature_store_audit_sha256, name="feature audit")
        or payload.get("minimum_population") != minimum
        or payload.get("primary_slot_ids") != list(_PRIMARY_SLOTS)
        or not isinstance(audited_slots, list)
        or audited_slots != [item.get("slot_id") for item in reports]
        or len(report_by_slot) != len(reports)
        or set(report_by_slot) != expected_slots
        or any(slot not in report_by_slot for slot in _PRIMARY_SLOTS)
        or type(primary_count) is not int
        or primary_count
        != sum(int(report_by_slot[slot].get("condition_count", -1)) for slot in _PRIMARY_SLOTS)
        or type(contingency_count) is not int
        or contingency_count
        != (
            int(report_by_slot[_CONTINGENCY_SLOT].get("condition_count", -1))
            if _CONTINGENCY_SLOT in report_by_slot
            else 0
        )
        or type(total_count) is not int
        or total_count != primary_count + contingency_count
        or total_count < minimum["campaign_eligible_conditions"]
        or set(roles) != set(_ALL_ROLES)
        or any(type(roles[role]) is not int or int(roles[role]) < 0 for role in _ALL_ROLES)
        or sum(int(roles[role]) for role in _ALL_ROLES) != total_count
        or any(int(roles[role]) < minimum[f"{role}_conditions"] for role in _MODEL_ROLES)
        or set(role_populations) != set(_MODEL_ROLES)
        or any(
            _sha256(role_populations[role], name="role population")
            != role_populations[role]
            for role in _MODEL_ROLES
        )
        or contingency_required is not (primary_count < minimum["campaign_eligible_conditions"])
        or payload.get("contingency_used") is not contingency_required
        or payload.get("all_primary_target_free_audits_present") is not True
        or payload.get("target_free") is not True
        or payload.get("target_accessed") is not False
        or payload.get("target_access_authorized") is not True
        or payload.get("authority") != _AUTHORITY
        or type(payload.get("admitted_at_ms")) is not int
        or int(payload["admitted_at_ms"]) <= 0
    ):
        raise ValueError("Round 27 campaign admission differs")
    return {**payload, "admission_sha256": claimed}


def load_round27_campaign_admission(
    path: str | Path,
    *,
    contract: Mapping[str, object],
    feature_store_audit_sha256: str,
) -> dict[str, object]:
    try:
        value = json.loads(
            Path(path).resolve(strict=True).read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {raw}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 27 campaign admission is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 27 campaign admission must be an object")
    return validate_round27_campaign_admission(
        value,
        contract=contract,
        feature_store_audit_sha256=feature_store_audit_sha256,
    )


__all__ = [
    "POLYMARKET_ROUND27_CAMPAIGN_ADMISSION_SCHEMA_VERSION",
    "build_round27_campaign_admission",
    "load_round27_campaign_admission",
    "round27_campaign_role_population_sha256",
    "validate_round27_campaign_admission",
]
