from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-v2-forensic-diagnostic-evaluation-contract-v1.json"
)
FILE_SHA256 = "8e82d5bfe3c9cf43aa1fe810b39308980dd0cc88122f0dced5dea241b0d29c3a"
CONTRACT_SHA256 = "f80d0396d7b51afdc63868a1e259099c2621ef45af7a3a31ab28b64967534896"


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


def test_forensic_diagnostic_contract_is_self_hashed_and_target_blind() -> None:
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == FILE_SHA256
    value = json.loads(CONTRACT.read_text(encoding="ascii"))
    canonical = dict(value)
    claimed = canonical.pop("contract_sha256")
    assert claimed == CONTRACT_SHA256
    assert claimed == _canonical_sha256(canonical)
    assert value["status"] == (
        "frozen_target_and_model_score_blind_before_feature_store_publication"
    )
    assert value["truth_state"]["train_or_calibration_targets_accessed"] is False
    assert value["truth_state"]["selection_targets_accessed"] is False
    assert value["truth_state"]["models_fitted"] is False


def test_forensic_diagnostic_contract_corrects_selection_isolation() -> None:
    value = json.loads(CONTRACT.read_text(encoding="ascii"))
    access = value["corrected_access_order"]
    assert access["condition_salvage_contract_access_order_superseded"] is True
    assert access["feature_materialization_rules_changed"] is False
    assert access["selection_resolution_requested_before_prediction_freeze"] is False
    assert access["selection_target_reuse_for_candidate_selection"] is False
    assert access["steps"].index(
        "freeze_selection_predictions_trade_policy_and_cost_parameters"
    ) < access["steps"].index("access_selection_official_resolutions_once")
    gate = value["diagnostic_pass_gate"]
    assert gate["net_profit_quote_strictly_positive"] is True
    assert gate["predictive_gate_passed"] is True
    assert gate["grants_financial_edge_claim"] is False
    assert gate["grants_profitability_claim"] is False
    assert gate["grants_trading_authority"] is False
