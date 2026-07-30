from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_round21_contract as contract_module


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-independent-matched-edge-contract-v1.json"
)


def _payload() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _rehash(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("contract_sha256", None)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    body["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def test_round21_is_polymarket_independent_with_matched_optional_predictors() -> None:
    raw = _payload()
    program = contract_module.validate_round21_contract(raw)

    assert program.decision_cadence_ms == 250
    assert (program.train_calendar_days, program.tune_calendar_days) == (18, 5)
    assert program.sealed_test_calendar_days == 7
    independence = raw["venue_independence"]
    assert independence["execution_venue"] == "polymarket"
    assert independence["binance_process_and_storage"] == "independent_sidecar"
    assert independence["binance_credentials_allowed"] is False
    assert independence["binance_execution_allowed"] is False
    assert independence["binance_failure_blocks_core_capture"] is False
    assert independence["binance_failure_blocks_polymarket_risk_or_close"] is False
    assert (
        independence["matched_uplift_population"]
        == "identical_core_eligible_decisions_with_causally_available_optional_receipts"
    )


def test_round21_separates_prediction_execution_fill_and_ai_claims() -> None:
    raw = _payload()
    tasks = raw["prediction_tasks"]
    passive = tasks["passive_fill_survival"]
    ai = raw["ai_assist"]
    execution = raw["execution_evaluation"]

    assert tasks["aggressive_action_value"]["forced_activity"] is False
    assert tasks["aggressive_action_value"]["multiple_actions_per_condition_allowed"]
    assert passive["right_censoring_required"] is True
    assert passive["public_book_depletion_as_fill_truth"] is False
    assert passive["maker_fill_or_rebate_credit_before_qualification"] is False
    assert execution["maker_rebate_credit"] is False
    assert execution["fees"] == (
        "market_specific_dynamic_platform_fee_from_captured_market_rules"
    )
    assert ai["per_tick_direction_or_order_generation"] is False
    assert ai["allowed_actions"] == ["preserve", "reduce", "abstain"]
    assert ai["increase_risk_or_override_safety"] is False
    assert ai["block_close_stop_or_recovery"] is False
    assert ai["minimum_matched_decisions"] == 300
    assert ai["minimum_non_tied_matched_actions"] == 30


def test_round21_keeps_proper_scores_and_after_cost_edge_as_separate_gates() -> None:
    raw = _payload()
    candidates = raw["candidate_program"]
    evaluation = raw["sealed_evaluation"]

    assert candidates["accuracy_without_executable_after_cost_utility_is_rejected"]
    assert candidates["profitability_target_used_for_parameter_selection"] is False
    assert evaluation["minimum_resolved_test_conditions"] == 1800
    assert evaluation["minimum_test_calendar_days"] == 7
    assert "log_loss_and_brier" in evaluation["predictive_gate"]
    assert "positive_net_pnl" in evaluation["core_economic_gate"]
    assert "exact_matched_population" in evaluation["optional_binance_gate"]
    assert raw["authority"] == {
        "model_data_eligible": False,
        "model_selected": False,
        "ai_edge_claim": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }


def test_round21_rejects_rehashed_independence_fill_or_authority_drift() -> None:
    mutations = (
        ("venue_independence", "binance_execution_allowed", True),
        (
            "venue_independence",
            "binance_failure_blocks_polymarket_stop_or_recovery",
            True,
        ),
        ("execution_evaluation", "maker_rebate_credit", True),
        ("ai_assist", "increase_risk_or_override_safety", True),
        ("authority", "paper_trading_authority", True),
    )
    for section, key, value in mutations:
        changed = _payload()
        nested = changed[section]
        assert isinstance(nested, dict)
        nested[key] = value
        with pytest.raises(ValueError, match="contract differs"):
            contract_module.validate_round21_contract(_rehash(changed))


def test_round21_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"round":21,"round":22}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON keys"):
        contract_module.load_round21_contract(path)
