from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v68.json"
)
PREVIOUS = DESIGN.with_name("round-074-event-sequence-model-design-v67.json")


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


def _normalized_lf_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest()


def test_round74_v68_binds_source_replayed_target_manifests() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    manifest = design["target_assembly_manifest"]
    assert claimed == _canonical_sha256(design)
    assert design["base_design"]["normalized_lf_sha256"] == (
        _normalized_lf_sha256(PREVIOUS)
    )
    assert (
        design["base_design"]["design_sha256"]
        == json.loads(PREVIOUS.read_text(encoding="ascii"))["design_sha256"]
    )
    assert manifest["source_normalized_lf_sha256"] == _normalized_lf_sha256(
        ROOT / manifest["source_path"]
    )
    assert manifest["test_normalized_lf_sha256"] == _normalized_lf_sha256(
        ROOT / manifest["test_path"]
    )
    assert manifest["required_source_labels"] == [
        "cohort_capture",
        "exchange_info",
        "commission",
        "funding",
        "execution_calibration",
        "execution_scenario",
    ]
    assert manifest["bare_self_consistent_assembly_accepted"] is False
    assert manifest["testnet_calibration_may_supply_mainnet_assembly_evidence"] is False


def test_round74_v68_binds_guarded_runtime_v2_sources() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    runtime = design["development_input_runtime"]
    for label in (
        "input_source",
        "input_test",
        "runtime",
        "runtime_test",
        "tool",
        "tool_test",
    ):
        assert runtime[f"{label}_normalized_lf_sha256"] == _normalized_lf_sha256(
            ROOT / runtime[f"{label}_path"]
        )
    assert runtime["input_schema_version"] == "round-074-development-inputs-v2"
    assert runtime["required_binding_count"] == 168
    assert runtime["required_development_manifest_count"] == 144
    assert runtime["sealed_test_manifest_count_read"] == 0
    assert runtime["manifest_sha256_in_inputs_sha256"] is True
    assert runtime["all_manifest_and_source_validation_before_database_open"] is True
    assert runtime["database_open_mode"] == "read_only"


def test_round74_v68_preserves_claim_blocks() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    assert design["verification"]["representative_target_manifest_available"] is False
    assert design["verification"]["development_training_executed"] is False
    assert design["verification"]["sealed_test_accessed"] is False
    assert design["unchanged_model_contract"]["candidate_panel_changed"] is False
    assert design["unchanged_model_contract"]["default_profile"] == "conservative"
    assert set(design["authority"].values()) == {False}
