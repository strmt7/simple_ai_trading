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
    / "round-074-event-sequence-model-design-v67.json"
)
PREVIOUS = DESIGN.with_name("round-074-event-sequence-model-design-v66.json")


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
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_round74_v67_binds_complete_testnet_aggregation() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["base_design"]["normalized_lf_sha256"] == (
        _normalized_lf_sha256(PREVIOUS)
    )
    assert (
        design["base_design"]["design_sha256"]
        == json.loads(PREVIOUS.read_text(encoding="ascii"))["design_sha256"]
    )

    aggregate = design["execution_calibration_aggregation"]
    assert aggregate["required_campaign_slots"] == 900
    assert aggregate["required_source_records"] == 1800
    assert aggregate["complete_campaign_required"] is True
    assert aggregate["source_capture_file_sha256_bound"] is True
    assert aggregate["source_order_records_revalidated"] is True
    assert aggregate["mainnet_execution_equivalence"] is False
    assert aggregate["mainnet_transfer_permitted"] is False
    assert aggregate["tool_normalized_lf_sha256"] == _normalized_lf_sha256(
        ROOT / aggregate["tool_path"]
    )
    assert aggregate["test_normalized_lf_sha256"] == _normalized_lf_sha256(
        ROOT / aggregate["test_path"]
    )


def test_round74_v67_forbids_unmeasured_mainnet_transfer() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    decision = design["no_transfer_decision"]
    assert decision["mainnet_execution_equivalence_claim"] is False
    assert decision["single_testnet_to_mainnet_multiplier_permitted"] is False
    assert decision["synthetic_mainnet_point_estimate_permitted"] is False
    assert decision["scenario_bounds_may_be_tuned_on_sealed_test"] is False
    assert decision["testnet_fill_pnl_is_financial_edge_evidence"] is False
    assert (
        decision["development_training_unblocked_by_testnet_aggregate_alone"] is False
    )
    assert {row["block"] for row in design["remaining_implementation_blocks"]} == {
        "mainnet public-paper execution scenario contract",
        "target assembly source manifest",
        "prospective cohort",
    }


def test_round74_v67_preserves_every_claim_and_authority_block() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    unchanged = design["unchanged_model_contract"]
    assert all(
        unchanged[key] is False
        for key in (
            "candidate_panel_changed",
            "feature_contract_changed",
            "target_contract_changed",
            "training_tuning_test_roles_changed",
            "action_policy_rule_changed",
            "ai_overlay_rule_changed",
            "sealed_test_rule_changed",
        )
    )
    assert design["authority"] == {
        "representative_market_training_completed": False,
        "sealed_test_accessed": False,
        "ai_uplift_evaluated": False,
        "market_edge_established": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "testnet_trading_authority": False,
        "live_trading_authority": False,
    }
