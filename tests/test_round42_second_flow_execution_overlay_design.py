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
    / "round-042-second-flow-execution-overlay-design.json"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_round42_design_is_hash_frozen_and_fail_closed() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    canonical = dict(design)
    claimed = canonical.pop("design_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "2695f9064a5622658f772f44cdc3e8fcc0f8f19ef7ae594e937a899533085588"
    assert design["schema_version"] == "second-flow-execution-overlay-design-v1"
    assert design["round"] == 42
    assert design["status"] == "frozen_before_implementation_or_outcome_access"

    governance = design["governance"]
    assert governance["single_registered_experiment"] is True
    for field in (
        "unregistered_hyperparameter_search_permitted",
        "round_41_primary_side_or_probability_change_permitted",
        "future_delay_or_outcome_feature_use_permitted",
        "oracle_delay_use_for_training_selection_or_claims_permitted",
        "maker_execution_assumption_permitted",
        "fee_or_slippage_reduction_permitted",
        "risk_gate_relaxation_permitted",
        "historical_data_expansion_before_pilot_gate_permitted",
        "ai_inference_during_pilot_permitted",
        "promotion_permitted",
        "trading_authority_permitted",
        "leverage_permitted",
        "profitability_portfolio_roi_or_drawdown_claim_permitted",
        "selection_confirmation_or_terminal_2026_access_permitted",
    ):
        assert governance[field] is False


def test_round42_design_preserves_source_model_cost_and_walk_forward_contracts() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    source = design["source_contract"]
    proposal = design["frozen_primary_proposal"]
    outcome = design["delay_option_and_outcome_contract"]
    model = design["overlay_model"]
    walk = design["walk_forward_contract"]
    gate = design["pilot_gate"]
    ai = design["ai_contract"]

    assert source["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert source["seconds_per_symbol"] == 604_800
    assert source["second_rows_total"] == 1_814_400
    assert source["archive_files_per_symbol"] == 7
    assert source["database_read_only_during_execution"] is True
    assert source["persistent_feature_matrix_or_raw_trade_copy_permitted"] is False
    assert source["api_calls_required"] is False
    assert proposal["model_artifact_ids"] == [
        "round41_202406_primary_btcusdt",
        "round41_202406_primary_ethusdt",
        "round41_202406_primary_solusdt",
    ]
    assert proposal["temperature_scale"] == 1.06538314761829
    assert proposal["minimum_direction_probability_margin"] == 0.10
    assert proposal["holding_horizon_seconds"] == 1800
    assert outcome["entry_delay_seconds"] == [0, 5, 15, 30]
    assert outcome["base_round_trip_charge_bps"] == 12.0
    assert outcome["stress_round_trip_charge_bps"] == 16.0
    assert outcome["same_symbol_position_overlap_permitted"] is False
    assert outcome["maximum_entries_per_symbol_per_utc_day"] == 8
    assert model["models_per_walk_forward_fold"] == 3
    assert model["opencl_gpu_first_required"] is True
    assert len(walk["folds"]) == 2
    assert walk["threshold_cells_per_fold"] == 27
    assert walk["threshold_cells_total"] == 54
    assert walk["evaluation_outcome_access_before_threshold_freeze_permitted"] is False
    assert gate["all_requirements_mandatory"] is True
    assert gate["stress_16_bps_mean_net_bps_must_exceed"] == 0.0
    assert gate["passing_is_not_promotion_or_profitability_evidence"] is True
    assert ai["pilot_ai_cases"] == 0
    assert ai["ai_improvement_claim_permitted_in_round_42"] is False

    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "roi_claim",
        "drawdown_claim",
        "leverage_applied",
        "ai_uplift_claim",
    ):
        assert design[field] is False
