from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
DESIGN = RESEARCH / "round-036-multi-horizon-signal-decay-design.json"


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _canonical_sha256(payload: dict[str, object], field: str) -> str:
    canonical = dict(payload)
    canonical.pop(field)
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round36_design_is_hash_bound_to_round35_failure_and_registry() -> None:
    design = _read(DESIGN)
    predecessor = design["predecessor"]
    governance = design["governance"]
    failure_path = RESEARCH / predecessor["failure_analysis"]
    registry_path = RESEARCH / governance["consumed_period_registry"]
    failure = _read(failure_path)
    registry = _read(registry_path)

    assert design["design_sha256"] == _canonical_sha256(design, "design_sha256")
    assert design["schema_version"] == "multi-horizon-signal-decay-design-v1"
    assert design["round"] == 36
    assert design["phase"] == "pre_model_consumed_data_diagnostic"
    assert (
        predecessor["failure_analysis_canonical_sha256"] == failure["analysis_sha256"]
    )
    assert predecessor["failure_analysis_file_sha256"] == _file_sha256(failure_path)
    assert (
        governance["consumed_period_registry_canonical_sha256"]
        == registry["registry_sha256"]
    )
    assert governance["consumed_period_registry_file_sha256"] == _file_sha256(
        registry_path
    )


def test_round36_design_freezes_signal_horizon_and_cost_contracts() -> None:
    design = _read(DESIGN)
    source = design["source_contract"]
    signals = design["signals"]
    costs = design["execution_cost_contract"]
    statistics = design["statistical_contract"]

    names = [item["name"] for item in signals]
    assert source["feature_version"] == MICROSTRUCTURE_FEATURE_VERSION
    assert source["feature_count"] == len(MICROSTRUCTURE_FEATURE_NAMES) == 107
    assert len(names) == len(set(names)) == 13
    assert set(names) <= set(MICROSTRUCTURE_FEATURE_NAMES)
    assert all(
        item["positive_orientation"] == "higher_future_midquote" for item in signals
    )
    assert design["horizons_seconds"] == [5, 15, 30, 60, 120, 300, 900]
    assert costs["delayed_entry_arrival_ms"] == 750
    assert costs["taker_fee_bps_per_side"] == 5.0
    assert costs["additional_adverse_slippage_bps_per_side"] == 1.0
    assert costs["zero_latency_counterfactual"]["purpose"] == (
        "decompose_historical_latency_drag_only"
    )
    assert statistics["primary_pooling_weight"] == (
        "average_label_uniqueness_for_each_horizon"
    )
    assert statistics["placebo"]["replicates"] == 200
    assert statistics["placebo"]["seed"] == 3601
    assert statistics["ranked_tail_counts"] == [100, 500, 1000]
    assert statistics["ranked_tail_outputs_are_event_outcomes_not_executable_trades"]


def test_round36_design_denies_selection_promotion_and_trading_claims() -> None:
    design = _read(DESIGN)
    governance = design["governance"]
    claims = design["claims"]

    assert governance["post_hoc_diagnostic_only"] is True
    assert governance["all_evaluated_dates_already_consumed"] is True
    for field in (
        "untouched_period_access_permitted",
        "model_training_permitted",
        "model_architecture_selection_permitted",
        "signal_sign_reversal_permitted",
        "signal_threshold_search_permitted",
        "signal_combination_weight_search_permitted",
        "horizon_selection_permitted",
        "promotion_permitted",
        "trading_policy_selection_permitted",
        "risk_gate_relaxation_permitted",
        "leverage_permitted",
        "maker_execution_assumption_permitted",
        "oracle_feature_or_runtime_label_use_permitted",
    ):
        assert governance[field] is False
    assert all(value is False for value in claims.values())
    assert design["decision_contract"]["this_round_can_create_model_candidate"] is False
    assert design["decision_contract"]["this_round_can_create_trading_authority"] is (
        False
    )


def test_round36_design_is_explicit_about_data_and_inference_limits() -> None:
    design = _read(DESIGN)
    access = design["data_access"]
    source = design["source_contract"]

    assert access["metric_start"] == "2023-06-21"
    assert access["metric_end"] == "2023-06-25"
    assert access["policy_prediction_or_metric_access_permitted"] is False
    assert access["development_prediction_or_metric_access_permitted"] is False
    assert access["distant_confirmation_source_materialization_permitted"] is False
    assert source["full_level_two_order_book_claim"] is False
    assert source["queue_position_claim"] is False
    assert source["hidden_liquidity_claim"] is False
    assert len(design["research_basis"]) >= 8
    assert len(design["limitations"]) >= 8
    assert any("multiple-testing" in item for item in design["limitations"])
    assert any("ETHUSDT and SOLUSDT" in item for item in design["limitations"])
