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
    / "round-074-event-sequence-model-design-v103.json"
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


def test_round74_v103_binds_complete_public_target_operator_sources() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    base = design["base_design"]
    base_path = ROOT / base["path"]
    assert base["file_sha256"] == _file_sha256(base_path)
    assert (
        base["design_sha256"]
        == json.loads(base_path.read_text(encoding="ascii"))["design_sha256"]
    )
    for source in design["source_binding"].values():
        if not isinstance(source, dict):
            continue
        assert source["sha256"] == _file_sha256(ROOT / source["path"])


def test_round74_v103_preserves_no_authority_and_evidence_limits() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    verification = design["verification"]

    assert delta["source_artifact_count"] == 6
    assert delta["manifest_reload_reopens_and_reaudits_all_six_sources"] is True
    assert (
        delta["testnet_execution_calibration_transferred_as_mainnet_fill_evidence"]
        is False
    )
    assert (
        delta[
            "caller_supplied_fees_latency_slippage_funding_or_quantity_values_permitted"
        ]
        is False
    )
    assert delta["orders_submitted_by_target_assembler"] is False
    assert verification["focused_current_source_pipeline_tests_passed"] == 51
    assert verification["focused_plus_historical_design_tests_passed"] == 87
    assert verification["complete_testnet_execution_aggregate_available"] is False
    assert (
        verification["account_specific_mainnet_commission_artifact_available"] is False
    )
    assert verification["live_target_manifest_created"] is False
    assert verification["representative_training_performed"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert set(design["authority"].values()) == {False}
