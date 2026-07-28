from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.impact_absorption_execution_scenario import (
    ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v102.json"
)


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round74_v102_binds_no_transfer_execution_scenario_sources() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    base = design["base_design"]
    base_path = ROOT / base["path"]
    assert base["file_sha256"] == _file_sha256(base_path)
    assert base["design_sha256"] == json.loads(
        base_path.read_text(encoding="ascii")
    )["design_sha256"]
    for source in design["source_binding"].values():
        if not isinstance(source, dict):
            continue
        assert len(source["sha256"]) == 64
        assert (ROOT / source["path"]).is_file()
    delta = design["declared_delta"]
    assert delta["scenario_contract_sha256"] == (
        ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256
    )
    assert delta["testnet_execution_equivalence_claim"] is False
    assert delta["testnet_to_mainnet_multiplier_permitted"] is False
    assert delta["synthetic_mainnet_fill_point_estimate_permitted"] is False
    assert delta["mainnet_fill_evidence_claim"] is False
    assert delta["orders_submitted_by_scenario_builder"] is False


def test_round74_v102_preserves_evidence_and_authority_boundaries() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))

    assert design["verification"]["focused_execution_target_manifest_tests_passed"] == 43
    assert design["verification"]["live_scenario_artifact_created"] is False
    assert design["verification"]["complete_testnet_execution_aggregate_available"] is False
    assert design["verification"]["representative_training_performed"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert set(design["authority"].values()) == {False}
