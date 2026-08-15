"""Restart-safe artifact validation for the Round 28 selection operator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json

from .polymarket_round27_economics import (
    POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
    POLYMARKET_ROUND27_FIXED_DELAYS_MS,
)
from .polymarket_round28_economics import (
    POLYMARKET_ROUND28_ECONOMIC_SCHEMA_VERSION,
    POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION,
)
from .polymarket_round28_model import Round28ModelSample


POLYMARKET_ROUND28_SELECTION_INPUT_SCHEMA_VERSION = (
    "polymarket-round28-selection-input-manifest-v1"
)
_OPERATOR_AMENDMENT_SCHEMA_VERSION = (
    "polymarket-round28-operator-implementation-amendment-v1"
)
_ROLES = ("train", "calibration", "selection")
_AUTHORITY = {
    "edge_claim": False,
    "profitability_claim": False,
    "paper_trading_authority": False,
    "live_trading_authority": False,
    "orders_submitted": False,
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


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


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 28 {name} SHA-256 differs")
    return selected


def _validated_claim(
    value: Mapping[str, object],
    *,
    hash_field: str,
    name: str,
) -> dict[str, object]:
    payload = dict(value)
    claimed = _sha256(payload.pop(hash_field, None), name=name)
    if claimed != _canonical_sha256(payload):
        raise ValueError(f"Round 28 {name} hash differs")
    return {**payload, hash_field: claimed}


def _gate_result(value: Mapping[str, object], *, result_field: str) -> bool:
    checks = value.get("gate_checks")
    return bool(
        isinstance(checks, Mapping)
        and checks
        and all(type(item) is bool for item in checks.values())
        and value.get(result_field) is all(checks.values())
    )


def _sample_identity(sample: Round28ModelSample) -> dict[str, object]:
    selected = sample.validated()
    return {
        "slot_id": selected.slot_id,
        "role": selected.role,
        "condition_id": selected.condition_id,
        "event_start_ms": selected.event_start_ms,
        "decision_time_ms": selected.decision_time_ms,
        "feature_row_sha256": selected.feature_row_sha256,
        "target_up": selected.target_up,
        "condition_weight": format(selected.condition_weight, ".17g"),
    }


def build_round28_selection_input_manifest(
    *,
    samples: Sequence[Round28ModelSample],
    feature_store_audit: Mapping[str, object],
    overlay_report: Mapping[str, object],
    target_store_audit: Mapping[str, object],
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    selection_implementation_amendment: Mapping[str, object],
    economic_implementation_amendment: Mapping[str, object],
    operator_implementation_amendment: Mapping[str, object],
) -> dict[str, object]:
    """Bind exact development inputs before model fitting starts."""

    feature_audit = _validated_claim(
        feature_store_audit,
        hash_field="audit_sha256",
        name="feature-store audit",
    )
    overlay = _validated_claim(
        overlay_report,
        hash_field="report_sha256",
        name="BBO overlay report",
    )
    target_audit = _validated_claim(
        target_store_audit,
        hash_field="audit_sha256",
        name="target-store audit",
    )
    model_contract = _validated_claim(
        contract,
        hash_field="contract_sha256",
        name="model contract",
    )
    prereg = _validated_claim(
        preregistration,
        hash_field="preregistration_sha256",
        name="preregistration",
    )
    selection_amendment = _validated_claim(
        selection_implementation_amendment,
        hash_field="amendment_sha256",
        name="selection implementation amendment",
    )
    economic_amendment = _validated_claim(
        economic_implementation_amendment,
        hash_field="amendment_sha256",
        name="economic implementation amendment",
    )
    operator_amendment = _validated_claim(
        operator_implementation_amendment,
        hash_field="amendment_sha256",
        name="operator implementation amendment",
    )
    if (
        operator_amendment.get("schema_version") != _OPERATOR_AMENDMENT_SCHEMA_VERSION
        or operator_amendment.get("status")
        != "frozen_before_stage1_feature_or_outcome_access"
    ):
        raise ValueError("Round 28 operator implementation amendment differs")
    selected_samples = tuple(sample.validated() for sample in samples)
    if not selected_samples or {sample.role for sample in selected_samples} != set(
        _ROLES
    ):
        raise ValueError("Round 28 selection input roles differ")
    role_reports: list[dict[str, object]] = []
    seen_conditions: dict[str, str] = {}
    for role in _ROLES:
        role_samples = tuple(
            sorted(
                (sample for sample in selected_samples if sample.role == role),
                key=lambda item: (
                    item.event_start_ms,
                    item.condition_id,
                    item.decision_time_ms,
                ),
            )
        )
        if not role_samples:
            raise ValueError("Round 28 selection input role is empty")
        identities = [_sample_identity(sample) for sample in role_samples]
        condition_targets: dict[str, int] = {}
        for sample in role_samples:
            prior_role = seen_conditions.setdefault(sample.condition_id, role)
            if prior_role != role:
                raise ValueError("Round 28 selection condition crosses roles")
            prior_target = condition_targets.setdefault(
                sample.condition_id,
                sample.target_up,
            )
            if prior_target != sample.target_up:
                raise ValueError("Round 28 selection condition target differs")
        ordered_targets = [
            {"condition_id": condition_id, "target_up": target}
            for condition_id, target in sorted(condition_targets.items())
        ]
        role_reports.append(
            {
                "role": role,
                "condition_count": len(condition_targets),
                "row_count": len(role_samples),
                "sample_identity_sha256": _canonical_sha256(identities),
                "target_population_sha256": _canonical_sha256(ordered_targets),
                "first_event_start_ms": min(
                    sample.event_start_ms for sample in role_samples
                ),
                "last_event_start_ms": max(
                    sample.event_start_ms for sample in role_samples
                ),
            }
        )
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_SELECTION_INPUT_SCHEMA_VERSION,
        "round27_feature_store_audit_sha256": feature_audit["audit_sha256"],
        "round28_overlay_report_sha256": overlay["report_sha256"],
        "round27_target_store_audit_sha256": target_audit["audit_sha256"],
        "round27_model_contract_sha256": model_contract["contract_sha256"],
        "round28_preregistration_sha256": prereg["preregistration_sha256"],
        "round28_selection_implementation_amendment_sha256": (
            selection_amendment["amendment_sha256"]
        ),
        "round28_economic_implementation_amendment_sha256": economic_amendment[
            "amendment_sha256"
        ],
        "round28_operator_implementation_amendment_sha256": operator_amendment[
            "amendment_sha256"
        ],
        "roles": role_reports,
        "condition_roles_disjoint": True,
        "matched_base_and_augmented_rows": True,
        "sealed_partition_accessed": False,
        "economic_metrics_computed": False,
        "ai_assist_evaluated": False,
        **_AUTHORITY,
    }
    body["manifest_sha256"] = _canonical_sha256(body)
    return body


def validate_round28_selection_input_manifest(
    value: Mapping[str, object],
) -> dict[str, object]:
    manifest = _validated_claim(
        value,
        hash_field="manifest_sha256",
        name="selection input manifest",
    )
    roles = manifest.get("roles")
    if (
        manifest.get("schema_version")
        != POLYMARKET_ROUND28_SELECTION_INPUT_SCHEMA_VERSION
        or not isinstance(roles, list)
        or [item.get("role") for item in roles if isinstance(item, Mapping)]
        != list(_ROLES)
        or len(roles) != len(_ROLES)
        or any(
            not isinstance(item, Mapping)
            or type(item.get("condition_count")) is not int
            or int(item["condition_count"]) <= 0
            or type(item.get("row_count")) is not int
            or int(item["row_count"]) < int(item["condition_count"])
            or _sha256(item.get("sample_identity_sha256"), name="sample identity")
            != item.get("sample_identity_sha256")
            or _sha256(
                item.get("target_population_sha256"),
                name="target population",
            )
            != item.get("target_population_sha256")
            for item in roles
        )
        or any(
            _sha256(manifest.get(field), name=field) != manifest.get(field)
            for field in (
                "round27_feature_store_audit_sha256",
                "round28_overlay_report_sha256",
                "round27_target_store_audit_sha256",
                "round27_model_contract_sha256",
                "round28_preregistration_sha256",
                "round28_selection_implementation_amendment_sha256",
                "round28_economic_implementation_amendment_sha256",
                "round28_operator_implementation_amendment_sha256",
            )
        )
        or manifest.get("condition_roles_disjoint") is not True
        or manifest.get("matched_base_and_augmented_rows") is not True
        or manifest.get("sealed_partition_accessed") is not False
        or manifest.get("economic_metrics_computed") is not False
        or manifest.get("ai_assist_evaluated") is not False
        or any(manifest.get(key) is not value for key, value in _AUTHORITY.items())
    ):
        raise ValueError("Round 28 selection input manifest differs")
    return manifest


def _validated_round27_economic_report(
    value: Mapping[str, object],
    *,
    source_audit_sha256: str,
    resolution_evidence_sha256: str,
) -> dict[str, object]:
    report = _validated_claim(
        value,
        hash_field="report_sha256",
        name="Round 27 economic report",
    )
    scenarios = report.get("scenarios")
    if (
        report.get("schema_version") != POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION
        or report.get("source_audit_sha256") != source_audit_sha256
        or report.get("resolution_evidence_sha256") != resolution_evidence_sha256
        or not isinstance(scenarios, list)
        or [item.get("delay_ms") for item in scenarios if isinstance(item, Mapping)]
        != list(POLYMARKET_ROUND27_FIXED_DELAYS_MS)
        or len(scenarios) != len(POLYMARKET_ROUND27_FIXED_DELAYS_MS)
        or any(
            not isinstance(item, Mapping)
            or not _gate_result(
                _validated_claim(
                    item,
                    hash_field="scenario_sha256",
                    name="economic scenario",
                ),
                result_field="scenario_edge_gate_passed",
            )
            for item in scenarios
        )
        or report.get("economic_edge_gate_passed")
        is not all(
            isinstance(item, Mapping) and item.get("scenario_edge_gate_passed") is True
            for item in scenarios
        )
        or any(
            report.get(field) is not False
            for field in (
                "edge_claim",
                "profitability_claim",
                "orders_submitted",
                "trading_authority",
            )
        )
    ):
        raise ValueError("Round 28 nested economic report differs")
    return report


def validate_round28_economic_report(
    value: Mapping[str, object],
    *,
    input_manifest: Mapping[str, object],
    selection_claim: Mapping[str, object],
    resolution_evidence_sha256: str,
) -> dict[str, object]:
    """Validate a restart checkpoint against recomputed source identities."""

    manifest = validate_round28_selection_input_manifest(input_manifest)
    selection = _validated_claim(
        selection_claim,
        hash_field="claim_sha256",
        name="selection claim",
    )
    report = _validated_claim(
        value,
        hash_field="report_sha256",
        name="matched economic report",
    )
    source_audit_sha256 = str(manifest["manifest_sha256"])
    resolution_sha256 = _sha256(
        resolution_evidence_sha256,
        name="resolution evidence",
    )
    base = report.get("base_economic_report")
    augmented = report.get("augmented_economic_report")
    paired = report.get("paired_scenarios")
    if not isinstance(base, Mapping) or not isinstance(augmented, Mapping):
        raise ValueError("Round 28 matched economic reports differ")
    selected_base = _validated_round27_economic_report(
        base,
        source_audit_sha256=source_audit_sha256,
        resolution_evidence_sha256=resolution_sha256,
    )
    selected_augmented = _validated_round27_economic_report(
        augmented,
        source_audit_sha256=source_audit_sha256,
        resolution_evidence_sha256=resolution_sha256,
    )
    if not isinstance(paired, list):
        raise ValueError("Round 28 paired economic scenarios differ")
    base_scenarios = {
        int(item["delay_ms"]): item
        for item in selected_base["scenarios"]
        if isinstance(item, Mapping)
    }
    augmented_scenarios = {
        int(item["delay_ms"]): item
        for item in selected_augmented["scenarios"]
        if isinstance(item, Mapping)
    }
    validated_paired = tuple(
        _validated_claim(
            item,
            hash_field="paired_scenario_sha256",
            name="paired economic scenario",
        )
        for item in paired
        if isinstance(item, Mapping)
    )
    if (
        report.get("schema_version") != POLYMARKET_ROUND28_ECONOMIC_SCHEMA_VERSION
        or report.get("partition_role") != "selection"
        or selection.get("status") != "matched_probability_candidate_selected"
        or report.get("selected_model_family") != selection.get("selected_model_family")
        or report.get("round27_model_contract_sha256")
        != manifest["round27_model_contract_sha256"]
        or report.get("round28_preregistration_sha256")
        != manifest["round28_preregistration_sha256"]
        or report.get("round28_selection_implementation_amendment_sha256")
        != manifest["round28_selection_implementation_amendment_sha256"]
        or report.get("round28_economic_implementation_amendment_sha256")
        != manifest["round28_economic_implementation_amendment_sha256"]
        or report.get("round28_selection_claim_sha256") != selection["claim_sha256"]
        or report.get("source_audit_sha256") != source_audit_sha256
        or report.get("resolution_evidence_sha256") != resolution_sha256
        or report.get("base_model_sha256") != selected_base.get("model_sha256")
        or report.get("augmented_model_sha256")
        != selected_augmented.get("model_sha256")
        or selected_base.get("config") != selected_augmented.get("config")
        or len(validated_paired) != len(POLYMARKET_ROUND27_FIXED_DELAYS_MS)
        or [item.get("delay_ms") for item in validated_paired]
        != list(POLYMARKET_ROUND27_FIXED_DELAYS_MS)
        or any(
            item.get("schema_version")
            != POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION
            or item.get("base_scenario_sha256")
            != base_scenarios[int(item["delay_ms"])]["scenario_sha256"]
            or item.get("augmented_scenario_sha256")
            != augmented_scenarios[int(item["delay_ms"])]["scenario_sha256"]
            or not _gate_result(
                item,
                result_field="scenario_uplift_gate_passed",
            )
            for item in validated_paired
        )
        or report.get("economic_uplift_gate_passed")
        is not (
            selected_augmented["economic_edge_gate_passed"] is True
            and all(
                item.get("scenario_uplift_gate_passed") is True
                for item in validated_paired
            )
        )
        or report.get("sealed_partition_accessed") is not False
        or report.get("economic_metrics_computed") is not True
        or report.get("ai_assist_evaluated") is not False
        or any(report.get(key) is not expected for key, expected in _AUTHORITY.items())
    ):
        raise ValueError("Round 28 matched economic report differs")
    return report


__all__ = [
    "POLYMARKET_ROUND28_SELECTION_INPUT_SCHEMA_VERSION",
    "build_round28_selection_input_manifest",
    "validate_round28_economic_report",
    "validate_round28_selection_input_manifest",
]
