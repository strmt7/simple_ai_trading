from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round27_economics import (
    POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
    POLYMARKET_ROUND27_FIXED_DELAYS_MS,
)
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.polymarket_round28_book_ticker import (
    POLYMARKET_ROUND28_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round28_economics import (
    POLYMARKET_ROUND28_ECONOMIC_SCHEMA_VERSION,
    POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION,
)
from simple_ai_trading.polymarket_round28_model import Round28ModelSample
from simple_ai_trading.polymarket_round28_operator import (
    build_round28_selection_input_manifest,
    validate_round28_economic_report,
    validate_round28_selection_input_manifest,
)


_START_MS = 1_786_784_400_000
ROOT = Path(__file__).resolve().parents[1]
_AUTHORITY = {
    "edge_claim": False,
    "profitability_claim": False,
    "paper_trading_authority": False,
    "live_trading_authority": False,
    "orders_submitted": False,
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


def _artifact(hash_field: str, marker: str) -> dict[str, object]:
    body: dict[str, object] = {"marker": marker}
    body[hash_field] = _canonical_sha256(body)
    return body


def _operator_amendment() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "polymarket-round28-operator-implementation-amendment-v1",
        "status": "frozen_before_stage1_feature_or_outcome_access",
    }
    body["amendment_sha256"] = _canonical_sha256(body)
    return body


def _contract_binding_correction() -> dict[str, object]:
    return json.loads(
        (
            ROOT
            / "docs/model-research/polymarket/"
            "round-028-loaded-contract-binding-correction-v1.json"
        ).read_text(encoding="ascii")
    )


def _sample(index: int, role: str) -> Round28ModelSample:
    base = (float(index),) + (0.0,) * (len(POLYMARKET_ROUND27_FEATURE_NAMES) - 1)
    augmented = (
        base
        + (float(index % 2),)
        + (0.0,) * (len(POLYMARKET_ROUND28_FEATURE_NAMES) - len(base) - 1)
    )
    start = _START_MS + index * 300_000
    return Round28ModelSample(
        slot_id="stage1-a" if role != "selection" else "stage1-b",
        role=role,
        condition_id="0x" + format(index + 1, "064x"),
        event_start_ms=start,
        decision_time_ms=start + 30_000,
        market_prior_probability=0.5,
        base_values=base,
        augmented_values=augmented,
        target_up=index % 2,
        condition_weight=1.0,
        feature_row_sha256=hashlib.sha256(
            f"operator-sample-{index}".encode("ascii")
        ).hexdigest(),
    ).validated()


def _input_manifest() -> dict[str, object]:
    samples = (
        _sample(0, "train"),
        _sample(1, "train"),
        _sample(2, "calibration"),
        _sample(3, "calibration"),
        _sample(4, "selection"),
        _sample(5, "selection"),
    )
    return build_round28_selection_input_manifest(
        samples=samples,
        feature_store_audit=_artifact("audit_sha256", "feature"),
        overlay_report=_artifact("report_sha256", "overlay"),
        target_store_audit=_artifact("audit_sha256", "target"),
        contract=load_round27_model_contract(ROOT),
        preregistration=_artifact("preregistration_sha256", "prereg"),
        selection_implementation_amendment=_artifact(
            "amendment_sha256",
            "selection-amendment",
        ),
        economic_implementation_amendment=_artifact(
            "amendment_sha256",
            "economic-amendment",
        ),
        operator_implementation_amendment=_operator_amendment(),
        contract_binding_correction=_contract_binding_correction(),
    )


def _scenario(delay_ms: int, *, passed: bool) -> dict[str, object]:
    body: dict[str, object] = {
        "delay_ms": delay_ms,
        "gate_checks": {"mechanics": passed},
        "scenario_edge_gate_passed": passed,
    }
    body["scenario_sha256"] = _canonical_sha256(body)
    return body


def _round27_report(
    *,
    source_sha256: str,
    resolution_sha256: str,
    model_name: str,
    model_sha256: str,
) -> dict[str, object]:
    scenarios = [
        _scenario(delay, passed=True) for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS
    ]
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
        "model_name": model_name,
        "model_sha256": model_sha256,
        "source_audit_sha256": source_sha256,
        "resolution_evidence_sha256": resolution_sha256,
        "config": {"fixed": True},
        "scenarios": scenarios,
        "economic_edge_gate_passed": True,
        "edge_claim": False,
        "profitability_claim": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


def _selection_claim(manifest: dict[str, object]) -> dict[str, object]:
    body: dict[str, object] = {
        "status": "matched_probability_candidate_selected",
        "selected_model_family": "l2_offset_logistic",
        "round27_model_contract_sha256": manifest[
            "round27_model_contract_sha256"
        ],
        "round27_model_implementation_amendment_sha256": manifest[
            "round27_model_implementation_amendment_sha256"
        ],
    }
    body["claim_sha256"] = _canonical_sha256(body)
    return body


def _economic_report(
    manifest: dict[str, object],
    selection: dict[str, object],
    resolution_sha256: str,
) -> dict[str, object]:
    source_sha256 = str(manifest["manifest_sha256"])
    base = _round27_report(
        source_sha256=source_sha256,
        resolution_sha256=resolution_sha256,
        model_name="l2_offset_logistic:round27_base",
        model_sha256="b" * 64,
    )
    augmented = _round27_report(
        source_sha256=source_sha256,
        resolution_sha256=resolution_sha256,
        model_name="l2_offset_logistic:round28_bbo_augmented",
        model_sha256="c" * 64,
    )
    paired: list[dict[str, object]] = []
    for base_scenario, augmented_scenario in zip(
        base["scenarios"],
        augmented["scenarios"],
        strict=True,
    ):
        body: dict[str, object] = {
            "schema_version": POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION,
            "delay_ms": base_scenario["delay_ms"],
            "base_scenario_sha256": base_scenario["scenario_sha256"],
            "augmented_scenario_sha256": augmented_scenario["scenario_sha256"],
            "gate_checks": {"uplift": True},
            "scenario_uplift_gate_passed": True,
        }
        body["paired_scenario_sha256"] = _canonical_sha256(body)
        paired.append(body)
    report: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_ECONOMIC_SCHEMA_VERSION,
        "partition_role": "selection",
        "round27_model_contract_sha256": manifest["round27_model_contract_sha256"],
        "round28_preregistration_sha256": manifest["round28_preregistration_sha256"],
        "round28_selection_implementation_amendment_sha256": manifest[
            "round28_selection_implementation_amendment_sha256"
        ],
        "round28_economic_implementation_amendment_sha256": manifest[
            "round28_economic_implementation_amendment_sha256"
        ],
        "round28_selection_claim_sha256": selection["claim_sha256"],
        "selected_model_family": selection["selected_model_family"],
        "base_model_sha256": base["model_sha256"],
        "augmented_model_sha256": augmented["model_sha256"],
        "source_audit_sha256": source_sha256,
        "resolution_evidence_sha256": resolution_sha256,
        "base_economic_report": base,
        "augmented_economic_report": augmented,
        "paired_scenarios": paired,
        "economic_uplift_gate_passed": True,
        "sealed_partition_accessed": False,
        "economic_metrics_computed": True,
        "ai_assist_evaluated": False,
        **_AUTHORITY,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def test_round28_selection_input_manifest_binds_roles_and_sources() -> None:
    manifest = _input_manifest()

    assert validate_round28_selection_input_manifest(manifest) == manifest
    assert [item["role"] for item in manifest["roles"]] == [
        "train",
        "calibration",
        "selection",
    ]
    assert manifest["sealed_partition_accessed"] is False
    assert manifest["matched_base_and_augmented_rows"] is True
    assert manifest["round27_model_implementation_amendment_sha256"] == (
        load_round27_model_contract(ROOT)[
            "model_implementation_amendment_sha256"
        ]
    )

    tampered = json.loads(json.dumps(manifest))
    tampered["roles"][0]["row_count"] += 1
    with pytest.raises(ValueError, match="manifest hash differs"):
        validate_round28_selection_input_manifest(tampered)


def test_round28_restart_report_requires_exact_nested_lineage_and_gates() -> None:
    manifest = _input_manifest()
    selection = _selection_claim(manifest)
    resolution_sha256 = "d" * 64
    report = _economic_report(manifest, selection, resolution_sha256)

    assert (
        validate_round28_economic_report(
            report,
            input_manifest=manifest,
            selection_claim=selection,
            resolution_evidence_sha256=resolution_sha256,
        )
        == report
    )

    tampered = json.loads(json.dumps(report))
    tampered["paired_scenarios"][0]["scenario_uplift_gate_passed"] = False
    tampered.pop("report_sha256")
    tampered["report_sha256"] = _canonical_sha256(tampered)
    with pytest.raises(ValueError, match="paired economic scenario hash differs"):
        validate_round28_economic_report(
            tampered,
            input_manifest=manifest,
            selection_claim=selection,
            resolution_evidence_sha256=resolution_sha256,
        )


def test_round28_restart_report_rejects_rehashed_wrong_source_manifest() -> None:
    manifest = _input_manifest()
    selection = _selection_claim(manifest)
    resolution_sha256 = "d" * 64
    report = _economic_report(manifest, selection, resolution_sha256)
    wrong_manifest = json.loads(json.dumps(manifest))
    wrong_manifest["round28_overlay_report_sha256"] = "e" * 64
    wrong_manifest.pop("manifest_sha256")
    wrong_manifest["manifest_sha256"] = _canonical_sha256(wrong_manifest)

    with pytest.raises(ValueError, match="nested economic report differs"):
        validate_round28_economic_report(
            report,
            input_manifest=wrong_manifest,
            selection_claim=selection,
            resolution_evidence_sha256=resolution_sha256,
        )
