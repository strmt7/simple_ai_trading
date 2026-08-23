from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from simple_ai_trading.impact_absorption_event_scaling import (
    ROUND74_EVENT_BINARY_FEATURE_COUNT,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_FEATURE_NAMES_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT / "docs/model-research/action-value/"
    "round-076-mark-conditioned-duration-preregistration-v1.json"
)
V136_DESIGN = (
    ROOT / "docs/model-research/action-value/"
    "round-074-event-sequence-model-design-v136.json"
)
ROUND75_CONTRACT = (
    ROOT / "docs/model-research/action-value/"
    "round-075-continuous-capture-contract-v4.json"
)
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


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


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    ).hexdigest()


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _strings(child)]
    return []


def _load_preregistration() -> dict[str, object]:
    return json.loads(PREREGISTRATION.read_text(encoding="ascii"))


def test_round76_preregistration_is_canonical_source_bound_and_non_authoritative() -> (
    None
):
    value = _load_preregistration()
    claimed = value.pop("preregistration_sha256")

    assert claimed == _canonical_sha256(value)
    assert value["status"] == (
        "frozen_target_blind_before_round75_terminal_adjudication"
    )
    assert value["authority"] == {
        "credentials_used": False,
        "edge_claim": False,
        "execution_connected": False,
        "live_trading_authority": False,
        "orders_submitted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
    }
    assert not any(
        WINDOWS_ABSOLUTE.match(item) or item.startswith("/") for item in _strings(value)
    )

    source_binding = value["source_binding"]
    assert source_binding["file_sha256_normalization"] == "canonical_lf_bytes"
    for relative, expected in source_binding[
        "current_main_canonical_lf_sha256"
    ].items():
        assert _text_sha256(ROOT / relative) == expected


def test_round76_preregistration_binds_v136_and_round75_without_hiding_drift() -> None:
    value = _load_preregistration()
    v136 = json.loads(V136_DESIGN.read_text(encoding="ascii"))
    round75 = json.loads(ROUND75_CONTRACT.read_text(encoding="ascii"))
    source = value["source_binding"]
    capture = source["round75_capture_contract"]

    assert _text_sha256(V136_DESIGN) == source["v136_design_file_sha256"]
    assert v136["design_sha256"] == source["v136_design_sha256"]
    canonical_v136 = dict(v136)
    claimed_v136 = canonical_v136.pop("design_sha256")
    assert claimed_v136 == _canonical_sha256(canonical_v136)
    assert _text_sha256(ROUND75_CONTRACT) == capture["file_sha256"]
    assert round75["artifact_sha256"] == capture["artifact_sha256"]
    assert (
        round75["source_bindings"]["capture_source_git_base_commit"]
        == capture["capture_source_git_base_commit"]
    )
    assert (
        round75["source_bindings"]["frozen_capture_source_file_sha256"][
            "src/simple_ai_trading/impact_absorption_event_sequence.py"
        ]
        == capture["frozen_event_sequence_file_sha256"]
    )
    assert source["round75_frozen_and_current_event_sequence_hashes_differ"] is True
    implementation_gate = value["implementation_gate"]
    assert implementation_gate["capture_source_compatibility_proven"] is False
    assert implementation_gate["implementation_permitted_now"] is False
    assert implementation_gate["round75_terminal_adjudication_completed"] is False
    assert (
        implementation_gate["source_hash_difference_is_permission_to_bulk_copy"]
        is False
    )


def test_round76_duration_target_is_causal_mask_safe_and_censor_aware() -> None:
    value = _load_preregistration()
    data = value["data_contract"]
    loss = value["loss_contract"]
    candidate = value["candidate_contract"]
    target_index = ROUND74_EVENT_FEATURE_NAMES.index(data["target_feature_name"])

    assert data["feature_names_sha256"] == ROUND74_EVENT_FEATURE_NAMES_SHA256
    assert target_index >= ROUND74_EVENT_BINARY_FEATURE_COUNT
    assert data["capture_clock"] == "local_monotonic_receipt_time"
    assert data["target_source"] == (
        "unmasked_original_feature_values[:,1:,target_feature_index]"
    )
    assert data["forbidden_target_clocks"] == [
        "exchange_event_time_ms",
        "received_wall_ns",
        "decision_wall_ns",
    ]
    assert data["masked_timing_view_policy"].startswith("zero_timing_input")
    assert "censored" in data["winsorization_policy"]
    assert "cdf" in loss["duration_left_boundary"]
    assert "survival" in loss["duration_right_boundary"]
    assert loss["loss_weight_tuning"] is False
    assert loss["supervised_financial_targets_used"] is False
    assert candidate["candidate_count"] == 1
    assert candidate["duration_head"]["mixture_components"] == 8
    assert candidate["duration_head"][
        "target_mark_is_not_an_input_to_encoder_or_next_mark_head"
    ]


def test_round76_requires_proper_score_and_after_cost_promotion() -> None:
    value = _load_preregistration()
    comparison = value["comparison_contract"]
    gates = value["promotion_gates"]
    evidence = value["evidence_boundary"]

    assert value["baseline_contract"]["unchanged_for_comparison"] is True
    assert comparison["candidate_multiplicity"].startswith("one_candidate")
    assert comparison["selection_data_reuse_for_terminal_claim"] is False
    assert "capture_runs" in comparison["identical_fields"]
    assert "initial_encoder_state" in comparison["identical_fields"]
    assert "cost_and_latency_scenarios" in comparison["identical_fields"]
    assert "duration_nll_must_improve" in gates["proper_duration_score"]
    assert "net_pnl_roi_and_expectancy" in gates["after_cost"]
    assert "maximum_drawdown" in gates["economic_risk"]
    assert "familywise_95_percent" in gates["simultaneous_inference"]
    assert "no_required_symbol_mark" in gates["subgroup_non_degradation"]
    assert "no_pretraining_or_predictive_result_grants" in gates["trading_authority"]
    assert evidence == {
        "ai_uplift_established": False,
        "candidate_implemented": False,
        "candidate_trained": False,
        "financial_edge_established": False,
        "predictive_accuracy_established": False,
        "profitability_established": False,
        "representative_training_completed": False,
        "round75_outcomes_accessed": False,
        "sealed_test_accessed": False,
    }
