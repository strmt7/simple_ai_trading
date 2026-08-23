"""Source-bound operator controls for Polymarket Round 29."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

from .polymarket_round27_economics import (
    POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
    POLYMARKET_ROUND27_FIXED_DELAYS_MS,
)
from .polymarket_round28_contract_binding import (
    validate_loaded_round27_model_contract,
)
from .polymarket_round28_economics import (
    POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION,
)
from .polymarket_round29_economics import (
    POLYMARKET_ROUND29_ECONOMIC_SCHEMA_VERSION,
    POLYMARKET_ROUND29_PAIRED_SCENARIO_SCHEMA_VERSION,
)
from .polymarket_round29_model import Round29ModelSample
from .polymarket_round29_selection import load_round29_selected_pair


POLYMARKET_ROUND29_IMPLEMENTATION_AMENDMENT_SCHEMA_VERSION = (
    "polymarket-round29-model-economic-operator-implementation-amendment-v1"
)
POLYMARKET_ROUND29_SELECTION_INPUT_SCHEMA_VERSION = (
    "polymarket-round29-selection-input-manifest-v1"
)
_ROLES = ("train", "calibration", "selection")
_REQUIRED_SOURCES = frozenset(
    {
        "docs/model-research/cross-regime-edge-acceptance-contract-v1.json",
        "src/simple_ai_trading/polymarket_round29_model.py",
        "src/simple_ai_trading/polymarket_round29_selection.py",
        "src/simple_ai_trading/polymarket_round29_economics.py",
        "src/simple_ai_trading/polymarket_round29_operator.py",
        "src/simple_ai_trading/polymarket_round29_source.py",
        "tests/test_polymarket_round29_model.py",
        "tests/test_polymarket_round29_selection.py",
        "tests/test_polymarket_round29_economics.py",
        "tests/test_polymarket_round29_operator.py",
        "tests/test_polymarket_round29_source.py",
        "tests/test_run_polymarket_round29_selection.py",
        "tests/test_cross_regime_edge_acceptance_contract.py",
        "tools/run_polymarket_round29_selection.py",
    }
)
_AUTHORITY = {
    "credentials_used": False,
    "edge_claim": False,
    "execution_connected": False,
    "live_trading_authority": False,
    "orders_submitted": False,
    "paper_trading_authority": False,
    "profitability_claim": False,
}
_KNOWLEDGE = {
    "ai_assist_economic_metrics_computed": False,
    "model_fitted_on_stage1": False,
    "official_outcomes_accessed": False,
    "performance_metrics_computed": False,
    "round27_stage1_feature_rows_accessed_or_materialized": False,
    "sealed_partition_accessed": False,
    "selection_partition_accessed": False,
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
        raise ValueError(f"Round 29 {name} SHA-256 differs")
    return selected


def _validated_claim(
    value: Mapping[str, object],
    *,
    hash_field: str,
    name: str,
) -> dict[str, object]:
    body = dict(value)
    claimed = _sha256(body.pop(hash_field, None), name=f"{name} claim")
    if claimed != _canonical_sha256(body):
        raise ValueError(f"Round 29 {name} hash differs")
    return {**body, hash_field: claimed}


def _required_int(
    value: Mapping[str, object],
    field: str,
    *,
    name: str,
) -> int:
    selected = value.get(field)
    if type(selected) is not int:
        raise ValueError(f"Round 29 {name} differs")
    return selected


def _boolean_checks(
    value: Mapping[str, object],
    *,
    name: str,
) -> dict[str, bool]:
    raw_checks = value.get("gate_checks")
    if not isinstance(raw_checks, Mapping) or not raw_checks:
        raise ValueError(f"Round 29 {name} gate checks differ")
    checks: dict[str, bool] = {}
    for raw_name, raw_check in raw_checks.items():
        if not isinstance(raw_name, str) or type(raw_check) is not bool:
            raise ValueError(f"Round 29 {name} gate checks differ")
        checks[raw_name] = raw_check
    return checks


def validate_round29_implementation_amendment(
    value: Mapping[str, object],
) -> dict[str, object]:
    amendment = _validated_claim(
        value,
        hash_field="amendment_sha256",
        name="implementation amendment",
    )
    sources = amendment.get("source_text_sha256")
    if (
        amendment.get("schema_version")
        != POLYMARKET_ROUND29_IMPLEMENTATION_AMENDMENT_SCHEMA_VERSION
        or amendment.get("status") != "frozen_before_stage1_feature_or_outcome_access"
        or not isinstance(sources, Mapping)
        or set(sources) != _REQUIRED_SOURCES
        or any(
            _sha256(source_hash, name="implementation source") != source_hash
            for source_hash in sources.values()
        )
        or amendment.get("authority") != _AUTHORITY
        or amendment.get("knowledge_at_freeze") != _KNOWLEDGE
    ):
        raise ValueError("Round 29 implementation amendment differs")
    return amendment


def verify_round29_implementation_sources(
    value: Mapping[str, object],
    *,
    root: Path,
) -> dict[str, object]:
    amendment = validate_round29_implementation_amendment(value)
    sources = amendment["source_text_sha256"]
    if not isinstance(sources, Mapping):
        raise ValueError("Round 29 implementation source map differs")
    selected_root = root.resolve(strict=True)
    for relative, expected in sources.items():
        path = (selected_root / str(relative)).resolve(strict=True)
        if selected_root not in path.parents:
            raise ValueError("Round 29 implementation source escapes root")
        actual = hashlib.sha256(
            path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Round 29 implementation source differs: {relative}")
    return amendment


def _sample_identity(sample: Round29ModelSample) -> dict[str, object]:
    selected = sample.validated()
    return {
        "slot_id": selected.slot_id,
        "role": selected.role,
        "condition_id": selected.condition_id,
        "event_start_ms": selected.event_start_ms,
        "decision_time_ms": selected.decision_time_ms,
        "diagnostic_feature_row_sha256": selected.diagnostic_feature_row_sha256,
        "primary_feature_row_sha256": selected.primary_feature_row_sha256,
        "target_up": selected.target_up,
        "condition_weight": format(selected.condition_weight, ".17g"),
    }


def build_round29_selection_input_manifest(
    *,
    samples: Sequence[Round29ModelSample],
    feature_store_audit: Mapping[str, object],
    bbo_overlay_report: Mapping[str, object],
    settlement_overlay_report: Mapping[str, object],
    target_store_audit: Mapping[str, object],
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    implementation_amendment: Mapping[str, object],
) -> dict[str, object]:
    """Bind exact development inputs before fitting or economic scoring."""

    feature_audit = _validated_claim(
        feature_store_audit,
        hash_field="audit_sha256",
        name="feature-store audit",
    )
    bbo_overlay = _validated_claim(
        bbo_overlay_report,
        hash_field="report_sha256",
        name="BBO overlay report",
    )
    settlement_overlay = _validated_claim(
        settlement_overlay_report,
        hash_field="report_sha256",
        name="settlement overlay report",
    )
    target_audit = _validated_claim(
        target_store_audit,
        hash_field="audit_sha256",
        name="target-store audit",
    )
    model_contract = validate_loaded_round27_model_contract(contract)
    prereg = _validated_claim(
        preregistration,
        hash_field="preregistration_sha256",
        name="preregistration",
    )
    amendment = validate_round29_implementation_amendment(implementation_amendment)
    if amendment.get("base_preregistration_sha256") != prereg["preregistration_sha256"]:
        raise ValueError("Round 29 amendment preregistration differs")
    selected_samples = tuple(sample.validated() for sample in samples)
    if not selected_samples or {sample.role for sample in selected_samples} != set(
        _ROLES
    ):
        raise ValueError("Round 29 selection input roles differ")
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
        condition_targets: dict[str, int] = {}
        for sample in role_samples:
            prior_role = seen_conditions.setdefault(sample.condition_id, role)
            if prior_role != role:
                raise ValueError("Round 29 selection condition crosses roles")
            prior_target = condition_targets.setdefault(
                sample.condition_id,
                sample.target_up,
            )
            if prior_target != sample.target_up:
                raise ValueError("Round 29 selection condition target differs")
        identities = [_sample_identity(sample) for sample in role_samples]
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
        "schema_version": POLYMARKET_ROUND29_SELECTION_INPUT_SCHEMA_VERSION,
        "round27_feature_store_audit_sha256": feature_audit["audit_sha256"],
        "round28_bbo_overlay_report_sha256": bbo_overlay["report_sha256"],
        "round29_settlement_overlay_report_sha256": settlement_overlay["report_sha256"],
        "round27_target_store_audit_sha256": target_audit["audit_sha256"],
        "round27_model_contract_sha256": model_contract["contract_sha256"],
        "round27_model_implementation_amendment_sha256": model_contract[
            "model_implementation_amendment_sha256"
        ],
        "round29_preregistration_sha256": prereg["preregistration_sha256"],
        "round29_implementation_amendment_sha256": amendment["amendment_sha256"],
        "roles": role_reports,
        "condition_roles_disjoint": True,
        "diagnostic_and_primary_rows_matched": True,
        "sealed_partition_accessed": False,
        "economic_metrics_computed": False,
        "ai_assist_evaluated": False,
        **_AUTHORITY,
    }
    body["manifest_sha256"] = _canonical_sha256(body)
    return body


def validate_round29_selection_input_manifest(
    value: Mapping[str, object],
) -> dict[str, object]:
    manifest = _validated_claim(
        value,
        hash_field="manifest_sha256",
        name="selection input manifest",
    )
    roles = manifest.get("roles")
    hash_fields = (
        "round27_feature_store_audit_sha256",
        "round28_bbo_overlay_report_sha256",
        "round29_settlement_overlay_report_sha256",
        "round27_target_store_audit_sha256",
        "round27_model_contract_sha256",
        "round27_model_implementation_amendment_sha256",
        "round29_preregistration_sha256",
        "round29_implementation_amendment_sha256",
    )
    if (
        manifest.get("schema_version")
        != POLYMARKET_ROUND29_SELECTION_INPUT_SCHEMA_VERSION
        or not isinstance(roles, list)
        or len(roles) != len(_ROLES)
        or [item.get("role") for item in roles if isinstance(item, Mapping)]
        != list(_ROLES)
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
            for field in hash_fields
        )
        or manifest.get("condition_roles_disjoint") is not True
        or manifest.get("diagnostic_and_primary_rows_matched") is not True
        or manifest.get("sealed_partition_accessed") is not False
        or manifest.get("economic_metrics_computed") is not False
        or manifest.get("ai_assist_evaluated") is not False
        or any(
            manifest.get(key) is not expected for key, expected in _AUTHORITY.items()
        )
    ):
        raise ValueError("Round 29 selection input manifest differs")
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
        or len(scenarios) != len(POLYMARKET_ROUND27_FIXED_DELAYS_MS)
        or [item.get("delay_ms") for item in scenarios if isinstance(item, Mapping)]
        != list(POLYMARKET_ROUND27_FIXED_DELAYS_MS)
        or any(
            not isinstance(item, Mapping)
            or _validated_claim(
                item,
                hash_field="scenario_sha256",
                name="economic scenario",
            ).get("scenario_edge_gate_passed")
            is not True
            for item in scenarios
        )
        or report.get("economic_edge_gate_passed") is not True
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
        raise ValueError("Round 29 nested economic report differs")
    return report


def _scenario_by_delay(
    report: Mapping[str, object],
) -> dict[int, Mapping[str, object]]:
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Round 29 nested economic scenarios differ")
    return {
        int(item["delay_ms"]): item
        for item in scenarios
        if isinstance(item, Mapping) and type(item.get("delay_ms")) is int
    }


def validate_round29_economic_report(
    value: Mapping[str, object],
    *,
    input_manifest: Mapping[str, object],
    selection_claim: Mapping[str, object],
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    resolution_evidence_sha256: str,
) -> dict[str, object]:
    manifest = validate_round29_selection_input_manifest(input_manifest)
    pair = load_round29_selected_pair(
        selection_claim,
        contract=contract,
        preregistration=preregistration,
        selection_input_manifest_sha256=str(manifest["manifest_sha256"]),
    )
    if pair is None:
        raise ValueError("Round 29 validated economics lacks a selected pair")
    report = _validated_claim(
        value,
        hash_field="report_sha256",
        name="matched economic report",
    )
    source_sha256 = str(manifest["manifest_sha256"])
    resolution_sha256 = _sha256(
        resolution_evidence_sha256,
        name="resolution evidence",
    )
    raw_base = report.get("base_economic_report")
    raw_augmented = report.get("augmented_economic_report")
    raw_paired = report.get("paired_scenarios")
    if not isinstance(raw_base, Mapping) or not isinstance(raw_augmented, Mapping):
        raise ValueError("Round 29 matched economic reports differ")
    base = _validated_round27_economic_report(
        raw_base,
        source_audit_sha256=source_sha256,
        resolution_evidence_sha256=resolution_sha256,
    )
    augmented = _validated_round27_economic_report(
        raw_augmented,
        source_audit_sha256=source_sha256,
        resolution_evidence_sha256=resolution_sha256,
    )
    if not isinstance(raw_paired, list):
        raise ValueError("Round 29 paired economic scenarios differ")
    paired_list: list[dict[str, object]] = []
    for item in raw_paired:
        if not isinstance(item, Mapping):
            raise ValueError("Round 29 paired economic scenario differs")
        paired_list.append(
            _validated_claim(
                item,
                hash_field="paired_scenario_sha256",
                name="paired economic scenario",
            )
        )
    paired = tuple(paired_list)
    inherited_list: list[dict[str, object]] = []
    for item in paired:
        raw_inherited = item.get("inherited_round28_matched_scenario")
        if not isinstance(raw_inherited, Mapping):
            raise ValueError("Round 29 inherited paired economic scenario differs")
        inherited_list.append(
            _validated_claim(
                raw_inherited,
                hash_field="paired_scenario_sha256",
                name="inherited paired economic scenario",
            )
        )
    inherited = tuple(inherited_list)
    base_scenarios = _scenario_by_delay(base)
    augmented_scenarios = _scenario_by_delay(augmented)
    paired_delays: list[int] = []
    for item, inherited_item in zip(paired, inherited, strict=True):
        delay = _required_int(item, "delay_ms", name="paired economic delay")
        inherited_delay = _required_int(
            inherited_item,
            "delay_ms",
            name="inherited paired economic delay",
        )
        checks = _boolean_checks(
            inherited_item,
            name="inherited paired economic scenario",
        )
        if delay not in base_scenarios or delay not in augmented_scenarios:
            raise ValueError("Round 29 paired economic delay population differs")
        if (
            item.get("schema_version")
            != POLYMARKET_ROUND29_PAIRED_SCENARIO_SCHEMA_VERSION
            or delay != inherited_delay
            or item.get("scenario_uplift_gate_passed")
            is not inherited_item.get("scenario_uplift_gate_passed")
            or inherited_item.get("schema_version")
            != POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION
            or inherited_item.get("base_scenario_sha256")
            != base_scenarios[delay]["scenario_sha256"]
            or inherited_item.get("augmented_scenario_sha256")
            != augmented_scenarios[delay]["scenario_sha256"]
            or inherited_item.get("scenario_uplift_gate_passed")
            is not all(checks.values())
        ):
            raise ValueError("Round 29 matched economic report differs")
        paired_delays.append(delay)
    if (
        report.get("schema_version") != POLYMARKET_ROUND29_ECONOMIC_SCHEMA_VERSION
        or report.get("partition_role") != "selection"
        or report.get("round27_model_contract_sha256")
        != manifest["round27_model_contract_sha256"]
        or report.get("round29_preregistration_sha256")
        != manifest["round29_preregistration_sha256"]
        or report.get("round29_implementation_amendment_sha256")
        != manifest["round29_implementation_amendment_sha256"]
        or report.get("round29_selection_claim_sha256")
        != selection_claim.get("claim_sha256")
        or report.get("selected_model_family") != pair.model_family
        or report.get("base_model_sha256") != pair.base_model.model_sha256
        or report.get("augmented_model_sha256") != pair.augmented_model.model_sha256
        or report.get("source_audit_sha256") != source_sha256
        or report.get("resolution_evidence_sha256") != resolution_sha256
        or len(paired) != len(POLYMARKET_ROUND27_FIXED_DELAYS_MS)
        or len(inherited) != len(paired)
        or paired_delays != list(POLYMARKET_ROUND27_FIXED_DELAYS_MS)
        or report.get("economic_uplift_gate_passed")
        is not (
            augmented.get("economic_edge_gate_passed") is True
            and all(item.get("scenario_uplift_gate_passed") is True for item in paired)
        )
        or report.get("sealed_partition_accessed") is not False
        or report.get("economic_metrics_computed") is not True
        or report.get("ai_assist_evaluated") is not False
        or any(report.get(key) is not expected for key, expected in _AUTHORITY.items())
        or base.get("config") != augmented.get("config")
    ):
        raise ValueError("Round 29 matched economic report differs")
    return report


__all__ = [
    "POLYMARKET_ROUND29_IMPLEMENTATION_AMENDMENT_SCHEMA_VERSION",
    "POLYMARKET_ROUND29_SELECTION_INPUT_SCHEMA_VERSION",
    "build_round29_selection_input_manifest",
    "validate_round29_economic_report",
    "validate_round29_implementation_amendment",
    "validate_round29_selection_input_manifest",
    "verify_round29_implementation_sources",
]
