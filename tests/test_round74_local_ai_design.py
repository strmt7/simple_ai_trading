from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.impact_absorption_ai_protocol import (
    ROUND74_AI_MODEL_MANIFEST_SCHEMA_VERSION,
    ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION,
    ROUND74_AI_REVIEW_HORIZONS_SECONDS,
    ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_bridge import (
    ROUND74_AI_BRIDGE_SCHEMA_VERSION,
    ROUND74_AI_RECENT_BLOCK_EVENTS,
)
from simple_ai_trading.impact_absorption_event_calibration import (
    ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION,
    ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_action_policy import (
    ROUND74_ACTION_CONTEXT_SCHEMA_VERSION,
    ROUND74_ACTION_POLICY_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_runtime import (
    ROUND74_AI_RUNTIME_OUTCOME_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_review_preparation import (
    ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION,
    round74_default_ai_review_model_panel,
)
from simple_ai_trading.impact_absorption_ai_uplift import (
    ROUND74_AI_UPLIFT_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_worker import (
    ROUND74_AI_WORKER_ENVELOPE_SCHEMA_VERSION,
    ROUND74_AI_WORKER_RESULT_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_sealed_evaluation import (
    ROUND74_SEALED_BOOTSTRAP_DRAWS,
    ROUND74_SEALED_EVALUATION_SCHEMA_VERSION,
    ROUND74_TARGET_FREE_INFERENCE_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_sealed_ledger import (
    ROUND74_SEALED_CLAIM_SCHEMA_VERSION,
    ROUND74_SEALED_LEDGER_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_financial_metrics import (
    ROUND74_REALIZED_METRICS_SCHEMA_VERSION,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-local-ai-review-design-v13.json"
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


def test_round74_local_ai_design_is_source_bound_and_fail_closed() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    source = artifact["source_binding"]
    for label in (
        "protocol",
        "bridge",
        "calibration",
        "action_policy",
        "uplift_evaluator",
        "sealed_ledger",
        "sealed_evaluator",
        "financial_metrics",
        "review_preparation",
        "worker",
        "runtime",
    ):
        assert (
            source[f"{label}_sha256"]
            == hashlib.sha256(
                (REPOSITORY / source[f"{label}_path"]).read_bytes()
            ).hexdigest()
        )
    assert source["model_manifest_schema_version"] == (
        ROUND74_AI_MODEL_MANIFEST_SCHEMA_VERSION
    )
    assert source["review_request_schema_version"] == (
        ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION
    )
    assert source["review_decision_schema_version"] == (
        ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION
    )
    assert source["worker_envelope_schema_version"] == (
        ROUND74_AI_WORKER_ENVELOPE_SCHEMA_VERSION
    )
    assert source["worker_result_schema_version"] == (
        ROUND74_AI_WORKER_RESULT_SCHEMA_VERSION
    )
    assert source["runtime_outcome_schema_version"] == (
        ROUND74_AI_RUNTIME_OUTCOME_SCHEMA_VERSION
    )
    assert source["bridge_schema_version"] == (ROUND74_AI_BRIDGE_SCHEMA_VERSION)
    assert source["tuning_subpartition_schema_version"] == (
        ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION
    )
    assert source["temperature_calibration_schema_version"] == (
        ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION
    )
    assert source["action_context_schema_version"] == (
        ROUND74_ACTION_CONTEXT_SCHEMA_VERSION
    )
    assert source["action_policy_schema_version"] == (
        ROUND74_ACTION_POLICY_SCHEMA_VERSION
    )
    assert source["uplift_evaluator_schema_version"] == (
        ROUND74_AI_UPLIFT_SCHEMA_VERSION
    )
    assert source["sealed_ledger_schema_version"] == (
        ROUND74_SEALED_LEDGER_SCHEMA_VERSION
    )
    assert source["sealed_claim_schema_version"] == (
        ROUND74_SEALED_CLAIM_SCHEMA_VERSION
    )
    assert source["sealed_evaluator_schema_version"] == (
        ROUND74_SEALED_EVALUATION_SCHEMA_VERSION
    )
    assert source["financial_metrics_schema_version"] == (
        ROUND74_REALIZED_METRICS_SCHEMA_VERSION
    )
    assert source["target_free_inference_schema_version"] == (
        ROUND74_TARGET_FREE_INFERENCE_SCHEMA_VERSION
    )
    assert source["review_panel_schema_version"] == (
        ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION
    )
    architecture = artifact["architecture"]
    assert architecture["supported_review_horizons_seconds"] == list(
        ROUND74_AI_REVIEW_HORIZONS_SECONDS
    )
    assert architecture["event_by_event_execution_loop_member"] is False
    assert architecture["one_and_five_second_ml_paths_wait_for_ai"] is False
    assert architecture["absolute_date_exposed_to_ai"] is False
    assert architecture["real_symbol_exposed_to_ai"] is False
    assert architecture["future_outcome_exposed_to_ai"] is False
    assert architecture["causal_recent_direction_summary_implemented"] is True
    assert architecture["causal_recent_direction_block_events"] == (
        ROUND74_AI_RECENT_BLOCK_EVENTS
    )
    assert (
        architecture["causal_recent_direction_realized_target_or_future_access"]
        is False
    )
    assert architecture["ai_prompt_summary_values_per_feature"] == 4
    assert architecture["ai_review_requires_preexisting_ml_candidate"] is True
    assert (
        architecture["calibrated_probability_selection_uses_equal_capture_run_weights"]
        is True
    )
    assert (
        architecture["calibration_pooled_observation_metrics_used_for_selection"]
        is False
    )
    assert architecture["calibration_exact_run_ids_bound"] is True
    assert architecture["runtime_elapsed_nanoseconds_bound_into_each_review_hash"] is True
    assert (
        architecture["same_entry_latency_budget_bound_into_each_review_and_panel_hash"]
        is True
    )
    assert (
        architecture[
            "same_entry_latency_budget_may_be_derived_from_forecast_horizon_or_request_validity"
        ]
        is False
    )
    assert architecture["late_accepted_review_retains_auditable_decision"] is True
    assert architecture["late_accepted_review_receives_same_entry_exposure"] is False
    assert architecture["latency_adjusted_delayed_entry_replay_implemented"] is False
    assert (
        architecture["future_target_eligibility_may_delete_a_model_selected_action"]
        is False
    )
    assert architecture["selected_action_with_incomplete_executable_target_policy"] == (
        "invalidate the threshold or sealed configuration"
    )
    assert (
        architecture["fixed_payoff_imputation_for_censored_selected_action_permitted"]
        is False
    )
    assert architecture["bounded_capture_run_panel_replay_implemented"] is True
    assert architecture["durable_one_use_sealed_ledger_implemented"] is True
    assert architecture["sealed_ml_ai_evaluator_implemented"] is True
    assert architecture["ml_ai_drawdown_uses_shared_actual_exit_order"] is True
    assert architecture["drawdown_order"] == (
        "cohort run then actual exit monotonic time then signal-order tie break"
    )
    assert architecture["reusable_target_free_candidate_inference_implemented"] is True
    assert architecture["two_model_target_free_review_preparation_implemented"] is True
    assert architecture["ai_may_revive_ml_abstention"] is False
    for key in (
        "ai_may_create_trade_side",
        "ai_may_increase_risk",
        "ai_may_set_leverage",
        "ai_may_submit_cancel_or_close_orders",
    ):
        assert architecture[key] is False
    assert (
        architecture["deterministic_data_execution_and_risk_gates_remain_authoritative"]
        is True
    )


def test_round74_local_ai_candidates_are_pinned_but_unpromoted() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    candidates = artifact["candidate_preselection"]
    finance = candidates["finance_candidate"]
    control = candidates["general_control"]

    assert finance["model_id"] == "TheFinAI/Fino1-8B"
    assert len(finance["repository_revision"]) == 40
    assert finance["license_id"] == "llama3.1"
    assert finance["trading_edge_established"] is False
    assert finance["pinned_quantized_artifact_ready"] is True
    default_models = round74_default_ai_review_model_panel()
    assert (
        finance["local_ollama_artifact"]["review_protocol_manifest_sha256"]
        == default_models[0].manifest.manifest_sha256
    )
    assert len(finance["local_ollama_artifact"]["manifest_sha256"]) == 64
    assert control["model_id"] == "Qwen/Qwen3-8B"
    assert len(control["repository_revision"]) == 40
    assert control["license_id"] == "Apache-2.0"
    assert (
        control["pinned_onnx_conversion"]["int4_external_data_observed_bytes"]
        == 6_076_825_600
    )
    assert control["trading_edge_established"] is False
    assert (
        control["local_ollama_control_artifact"]["review_protocol_manifest_sha256"]
        == default_models[1].manifest.manifest_sha256
    )
    assert candidates["finance_brand_or_parameter_count_implies_edge"] is False
    assert candidates["promotion_requires_our_paired_sealed_market_evidence"] is True
    challengers = candidates["researched_deferred_challengers"]
    assert [value["model_id"] for value in challengers] == [
        "OpenDataArena/ODA-Fin-RL-8B",
        "TheFinAI/Fin-o1-8B",
    ]
    assert all(value["license_id"] == "Apache-2.0" for value in challengers)
    assert all(
        value["trading_packet_edge_established"] is False
        and value["local_artifact_downloaded"] is False
        and len(value["repository_revision"]) == 40
        for value in challengers
    )
    assert all(
        value["source_safetensors_bytes"] > 16_000_000_000 for value in challengers
    )

    status = artifact["status"]
    assert status["protocol_implemented"] is True
    for key in (
        "candidate_weight_hash_verified",
        "actual_multibillion_parameter_inference_completed",
        "host_latency_preflight_completed",
        "representative_market_ai_evaluation_completed",
        "ai_uplift_established",
        "financial_edge_established",
        "profitability_claim",
        "paper_trading_authority",
        "testnet_trading_authority",
        "live_trading_authority",
    ):
        assert status[key] is False
    assert status["candidate_weights_downloaded"] is True
    assert status["candidate_manifest_hash_verified"] is True
    assert status["isolated_worker_implemented"] is True
    assert status["fail_closed_parent_runtime_implemented"] is True
    assert status["causal_calibrated_bridge_implemented"] is True
    assert status["target_free_action_preselection_implemented"] is True
    assert status["paired_uplift_evaluator_implemented"] is True
    assert status["durable_one_use_sealed_ledger_implemented"] is True
    assert status["sealed_ml_ai_evaluator_implemented"] is True
    assert status["target_free_candidate_inference_implemented"] is True
    assert status["two_model_review_preparation_implemented"] is True
    assert status["same_entry_latency_uplift_gate_implemented"] is True
    assert status["future_censorship_promotion_gate_implemented"] is True
    assert artifact["host_preflight"]["actual_model_inference_attempted"] is False
    assert artifact["host_preflight"]["approved_risk_size_bps"] == 0
    assert artifact["host_preflight"]["request_schema_version"] == (
        ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION
    )


def test_round74_local_ai_evaluation_cannot_win_by_all_veto() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    evaluation = artifact["paired_evaluation_contract"]

    assert evaluation["same_events_targets_costs_and_timing"] is True
    assert evaluation["same_frozen_risk_profile_and_action_threshold"] is True
    assert evaluation["historical_symbol_and_absolute_date_masked"] is True
    assert evaluation["ai_can_only_preserve_reduce_veto_or_abstain"] is True
    assert (
        evaluation[
            "timeout_blocked_or_invalid_present_ai_review_is_a_fail_closed_veto_not_a_dropped_observation"
        ]
        is True
    )
    assert evaluation["implemented_missing_review_policy"] == (
        "invalidate entire evaluation"
    )
    assert (
        evaluation["runtime_elapsed_nanoseconds_and_same_entry_budget_are_hash_bound"]
        is True
    )
    assert (
        evaluation[
            "same_entry_budget_must_not_exceed_independently_measured_residual_signal_to_entry_slack"
        ]
        is True
    )
    assert (
        evaluation[
            "forecast_horizon_or_request_validity_may_not_substitute_for_a_same_entry_budget"
        ]
        is True
    )
    assert evaluation["late_accepted_review_policy"] == (
        "retain the validated decision for audit but apply zero exposure at the ML entry"
    )
    assert evaluation["same_entry_latency_eligibility_rate_gate"] == 0.99
    assert (
        evaluation["same_entry_latency_eligibility_is_distinct_from_runtime_success"]
        is True
    )
    assert evaluation["latency_adjusted_delayed_entry_replay_implemented"] is False
    assert (
        evaluation[
            "future_target_eligibility_may_delete_a_model_selected_observation"
        ]
        is False
    )
    assert (
        evaluation[
            "complete_executable_target_coverage_required_for_every_selected_action"
        ]
        is True
    )
    assert evaluation["censored_selected_action_policy"] == (
        "configuration is unscorable and cannot be promoted"
    )
    assert (
        evaluation[
            "fixed_zero_loss_or_profit_imputation_for_censored_selected_action_permitted"
        ]
        is False
    )
    assert evaluation["development_evaluator_is_promotional"] is False
    assert evaluation["ai_may_change_candidate_overlap_order"] is False
    assert evaluation["sealed_test_used_once_after_ai_policy_freeze"] is True
    assert (
        evaluation["sealed_access_is_reserved_and_permanently_consumed_before_scoring"]
        is True
    )
    assert evaluation["crash_after_reservation_may_reset_or_reuse_the_test"] is False
    assert evaluation["sealed_bootstrap_draws"] == ROUND74_SEALED_BOOTSTRAP_DRAWS
    assert (
        evaluation[
            "target_eligibility_and_realized_exit_timing_may_not_choose_which_ai_reviews_exist"
        ]
        is True
    )
    assert (
        evaluation["ai_veto_of_an_earlier_action_may_admit_a_later_overlapping_action"]
        is False
    )
    assert evaluation["accuracy_ignored"] is False
    assert (
        evaluation["profitability_required_by_data_filter_or_test_assertion"] is False
    )
    assert evaluation["ai_receives_causal_recent_direction_summary"] is True
    assert evaluation["ai_recent_direction_summary_block_events"] == (
        ROUND74_AI_RECENT_BLOCK_EVENTS
    )
    assert (
        "enough remaining executable trades to avoid trivial all-veto behavior"
        in evaluation["promotion_requires"]
    )
