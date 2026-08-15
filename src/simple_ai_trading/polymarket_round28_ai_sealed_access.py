"""Sanitized authorization receipt for Round 28 sealed AI inference."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

from .polymarket_round28_ai_cases import (
    Round28AICasePanel,
    round28_ai_case_panel_from_mapping,
)
from .polymarket_round28_ai_selection import (
    Round28AICandidateSelection,
    round28_ai_candidate_selection_from_mapping,
)
from .polymarket_round28_operator import (
    validate_round28_selection_input_manifest,
)
from .polymarket_round28_sealed import (
    validate_round28_sealed_access_artifacts,
)
from .polymarket_round28_selection import (
    Round28SelectedPair,
    load_round28_selected_pair,
)


POLYMARKET_ROUND28_AI_SEALED_ACCESS_SCHEMA_VERSION = (
    "polymarket-round28-ai-sealed-access-v1"
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_FALSE_FLAGS = (
    "selection_metrics_exposed_to_case_process",
    "target_data_exposed_to_case_process",
    "outcome_data_exposed_to_case_process",
    "resolution_data_exposed_to_case_process",
    "pnl_data_exposed_to_case_process",
    "credentials_used",
    "orders_submitted",
    "trading_authority",
)


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
        raise ValueError(f"Round 28 sealed AI access {name} SHA-256 differs")
    return selected


def _validate_selection_lineage(
    *,
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    selection_input_manifest: Mapping[str, object],
    selection_claim: Mapping[str, object],
    selection_ai_case_panel: Mapping[str, object],
    ai_selection_claim: Mapping[str, object],
) -> tuple[
    dict[str, object],
    Round28SelectedPair,
    Round28AICasePanel,
    Round28AICandidateSelection,
]:
    manifest = validate_round28_selection_input_manifest(
        selection_input_manifest
    )
    pair = load_round28_selected_pair(
        selection_claim,
        contract=contract,
        preregistration=preregistration,
    )
    if pair is None:
        raise ValueError("Round 28 sealed AI access has no selected model pair")
    panel = round28_ai_case_panel_from_mapping(selection_ai_case_panel)
    selection = round28_ai_candidate_selection_from_mapping(
        ai_selection_claim
    )
    if (
        panel.partition_role != "selection"
        or panel.panel_sha256 != selection.case_panel_sha256
        or panel.selection_claim_sha256 != selection_claim.get("claim_sha256")
        or panel.source_audit_sha256 != manifest.get("manifest_sha256")
        or panel.model_name != pair.augmented_model.model_name
        or panel.model_sha256 != pair.augmented_model.model_sha256
    ):
        raise ValueError("Round 28 sealed AI access selection lineage differs")
    return manifest, pair, panel, selection


def build_round28_ai_sealed_access_receipt(
    *,
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    selection_input_manifest: Mapping[str, object],
    selection_claim: Mapping[str, object],
    selection_economic_report: Mapping[str, object],
    selection_ai_case_panel: Mapping[str, object],
    ai_selection_claim: Mapping[str, object],
    selection_resolution_evidence_sha256: str,
) -> dict[str, object]:
    """Validate selection gates, then expose only identities to case inference."""

    authorized_pair = validate_round28_sealed_access_artifacts(
        contract=contract,
        preregistration=preregistration,
        selection_input_manifest=selection_input_manifest,
        selection_claim=selection_claim,
        selection_economic_report=selection_economic_report,
        selection_resolution_evidence_sha256=(
            selection_resolution_evidence_sha256
        ),
    )
    manifest, pair, panel, selection = _validate_selection_lineage(
        contract=contract,
        preregistration=preregistration,
        selection_input_manifest=selection_input_manifest,
        selection_claim=selection_claim,
        selection_ai_case_panel=selection_ai_case_panel,
        ai_selection_claim=ai_selection_claim,
    )
    if (
        authorized_pair != pair
        or selection.round28_economic_report_sha256
        != selection_economic_report.get("report_sha256")
    ):
        raise ValueError("Round 28 sealed AI access gate lineage differs")
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_AI_SEALED_ACCESS_SCHEMA_VERSION,
        "round27_model_contract_sha256": _sha256(
            contract.get("contract_sha256"),
            name="model contract",
        ),
        "round28_preregistration_sha256": _sha256(
            preregistration.get("preregistration_sha256"),
            name="preregistration",
        ),
        "selection_input_manifest_sha256": manifest["manifest_sha256"],
        "selection_claim_sha256": _sha256(
            selection_claim.get("claim_sha256"),
            name="selection claim",
        ),
        "selection_after_cost_gate_report_sha256": _sha256(
            selection_economic_report.get("report_sha256"),
            name="selection after-cost gate report",
        ),
        "selection_ai_case_panel_sha256": panel.panel_sha256,
        "ai_selection_sha256": selection.selection_sha256,
        "selected_model_family": pair.model_family,
        "selected_augmented_model_name": pair.augmented_model.model_name,
        "selected_augmented_model_sha256": pair.augmented_model.model_sha256,
        "nominated_model_id": selection.nominated_model_id,
        "nominated_runtime_digest": selection.nominated_runtime_digest,
        "selection_probability_gate_passed": True,
        "selection_after_cost_gate_passed": True,
        "selection_ai_lineage_validated": True,
        "case_process_receives_sanitized_receipt_only": True,
        **{field: False for field in _FALSE_FLAGS},
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    validate_round28_ai_sealed_access_receipt(
        body,
        contract=contract,
        preregistration=preregistration,
        selection_input_manifest=selection_input_manifest,
        selection_claim=selection_claim,
        selection_ai_case_panel=selection_ai_case_panel,
        ai_selection_claim=ai_selection_claim,
    )
    return body


def validate_round28_ai_sealed_access_receipt(
    value: Mapping[str, object],
    *,
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    selection_input_manifest: Mapping[str, object],
    selection_claim: Mapping[str, object],
    selection_ai_case_panel: Mapping[str, object],
    ai_selection_claim: Mapping[str, object],
) -> tuple[
    dict[str, object],
    Round28SelectedPair,
    Round28AICasePanel,
    Round28AICandidateSelection,
]:
    """Validate a sanitized receipt without loading outcomes or PnL."""

    receipt = dict(value)
    claimed = _sha256(
        receipt.pop("receipt_sha256", None),
        name="receipt",
    )
    manifest, pair, panel, selection = _validate_selection_lineage(
        contract=contract,
        preregistration=preregistration,
        selection_input_manifest=selection_input_manifest,
        selection_claim=selection_claim,
        selection_ai_case_panel=selection_ai_case_panel,
        ai_selection_claim=ai_selection_claim,
    )
    expected_fields = {
        "schema_version",
        "round27_model_contract_sha256",
        "round28_preregistration_sha256",
        "selection_input_manifest_sha256",
        "selection_claim_sha256",
        "selection_after_cost_gate_report_sha256",
        "selection_ai_case_panel_sha256",
        "ai_selection_sha256",
        "selected_model_family",
        "selected_augmented_model_name",
        "selected_augmented_model_sha256",
        "nominated_model_id",
        "nominated_runtime_digest",
        "selection_probability_gate_passed",
        "selection_after_cost_gate_passed",
        "selection_ai_lineage_validated",
        "case_process_receives_sanitized_receipt_only",
        *_FALSE_FLAGS,
    }
    if (
        set(receipt) != expected_fields
        or claimed != _canonical_sha256(receipt)
        or receipt.get("schema_version")
        != POLYMARKET_ROUND28_AI_SEALED_ACCESS_SCHEMA_VERSION
        or receipt.get("round27_model_contract_sha256")
        != contract.get("contract_sha256")
        or receipt.get("round28_preregistration_sha256")
        != preregistration.get("preregistration_sha256")
        or receipt.get("selection_input_manifest_sha256")
        != manifest["manifest_sha256"]
        or receipt.get("selection_claim_sha256")
        != selection_claim.get("claim_sha256")
        or receipt.get("selection_after_cost_gate_report_sha256")
        != selection.round28_economic_report_sha256
        or receipt.get("selection_ai_case_panel_sha256")
        != panel.panel_sha256
        or receipt.get("ai_selection_sha256") != selection.selection_sha256
        or receipt.get("selected_model_family") != pair.model_family
        or receipt.get("selected_augmented_model_name")
        != pair.augmented_model.model_name
        or receipt.get("selected_augmented_model_sha256")
        != pair.augmented_model.model_sha256
        or receipt.get("nominated_model_id") != selection.nominated_model_id
        or receipt.get("nominated_runtime_digest")
        != selection.nominated_runtime_digest
        or any(
            receipt.get(field) is not True
            for field in (
                "selection_probability_gate_passed",
                "selection_after_cost_gate_passed",
                "selection_ai_lineage_validated",
                "case_process_receives_sanitized_receipt_only",
            )
        )
        or any(receipt.get(field) is not False for field in _FALSE_FLAGS)
    ):
        raise ValueError("Round 28 sealed AI access receipt differs")
    return {**receipt, "receipt_sha256": claimed}, pair, panel, selection


__all__ = [
    "POLYMARKET_ROUND28_AI_SEALED_ACCESS_SCHEMA_VERSION",
    "build_round28_ai_sealed_access_receipt",
    "validate_round28_ai_sealed_access_receipt",
]
