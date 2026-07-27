from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.impact_absorption_ai_protocol import (
    ROUND74_AI_MODEL_MANIFEST_SCHEMA_VERSION,
    ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION,
    ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_bridge import (
    ROUND74_AI_BRIDGE_SCHEMA_VERSION,
    ROUND74_AI_RECENT_BLOCK_EVENTS,
)
from simple_ai_trading.impact_absorption_ai_uplift import (
    ROUND74_AI_UPLIFT_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_runtime import (
    ROUND74_AI_RUNTIME_OUTCOME_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_review_preparation import (
    ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_worker import (
    ROUND74_AI_WORKER_ENVELOPE_SCHEMA_VERSION,
    ROUND74_AI_WORKER_RESULT_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_cohort import (
    ROUND74_EVENT_COHORT_BINDING_SCHEMA_VERSION,
    ROUND74_EVENT_COHORT_PLAN_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_calibration import (
    ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION,
    ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_action_policy import (
    ROUND74_ACTION_CONTEXT_SCHEMA_VERSION,
    ROUND74_ACTION_DEFAULT_PROFILE,
    ROUND74_ACTION_HORIZONS_SECONDS,
    ROUND74_ACTION_POLICY_SCHEMA_VERSION,
    ROUND74_ACTION_PROFILES,
)
from simple_ai_trading.impact_absorption_event_dataset import (
    ROUND74_EVENT_DATASET_SCHEMA_VERSION,
    ROUND74_EVENT_PARTITION_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_scaling import (
    ROUND74_EVENT_SCALER_SCHEMA_VERSION,
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
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_FEATURE_NAMES_SHA256,
    ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_training import (
    ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION,
    ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION,
    ROUND74_EVENT_TRAINING_DEFAULT_SEEDS,
    ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_model import (
    ROUND74_EVENT_TCN_DILATIONS,
    ROUND74_EVENT_TCN_RECEPTIVE_FIELD,
)
from simple_ai_trading.impact_absorption_event_financial_metrics import (
    ROUND74_REALIZED_METRICS_SCHEMA_VERSION,
)


REPOSITORY = Path(__file__).resolve().parents[1]
RESEARCH = REPOSITORY / "docs" / "model-research" / "action-value"
DESIGN_PATH = RESEARCH / "round-074-event-sequence-model-design-v24.json"
DIRECTML_PATH = (
    RESEARCH / "round-074-event-model-directml-preflight-2026-07-26.json"
)
REPLAY_PATH = (
    RESEARCH / "round-074-event-sequence-host-replay-2026-07-26.json"
)
TRAINING_PATH = (
    RESEARCH / "round-074-event-training-directml-preflight-2026-07-27.json"
)
CALIBRATION_PATH = (
    RESEARCH
    / "round-074-temperature-calibration-directml-preflight-2026-07-26.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _load_hash_bound(path: Path, field: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(field)
    assert claimed == _canonical_sha256(value)
    value[field] = claimed
    return value


def _file_sha256(relative_path: str) -> str:
    return hashlib.sha256((REPOSITORY / relative_path).read_bytes()).hexdigest()


def test_round74_event_model_design_is_source_bound_and_causal() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    source = design["source_binding"]

    assert source["event_sequence_sha256"] == _file_sha256(
        source["event_sequence_path"]
    )
    assert source["event_model_sha256"] == _file_sha256(
        source["event_model_path"]
    )
    assert source["event_scaler_sha256"] == _file_sha256(
        source["event_scaler_path"]
    )
    assert source["event_target_sha256"] == _file_sha256(
        source["event_target_path"]
    )
    assert source["event_dataset_sha256"] == _file_sha256(
        source["event_dataset_path"]
    )
    assert source["event_action_policy_sha256"] == _file_sha256(
        source["event_action_policy_path"]
    )
    assert source["ai_uplift_evaluator_sha256"] == _file_sha256(
        source["ai_uplift_evaluator_path"]
    )
    assert source["sealed_ledger_sha256"] == _file_sha256(
        source["sealed_ledger_path"]
    )
    assert source["sealed_evaluator_sha256"] == _file_sha256(
        source["sealed_evaluator_path"]
    )
    assert source["financial_metrics_sha256"] == _file_sha256(
        source["financial_metrics_path"]
    )
    assert source["ai_review_preparation_sha256"] == _file_sha256(
        source["ai_review_preparation_path"]
    )
    assert source["event_cohort_sha256"] == _file_sha256(
        source["event_cohort_path"]
    )
    assert source["ai_protocol_sha256"] == _file_sha256(
        source["ai_protocol_path"]
    )
    assert source["ai_bridge_sha256"] == _file_sha256(
        source["ai_bridge_path"]
    )
    assert source["event_calibration_sha256"] == _file_sha256(
        source["event_calibration_path"]
    )
    assert source["ai_worker_sha256"] == _file_sha256(
        source["ai_worker_path"]
    )
    assert source["ai_runtime_sha256"] == _file_sha256(
        source["ai_runtime_path"]
    )
    assert source["event_training_sha256"] == _file_sha256(
        source["event_training_path"]
    )
    assert source["storage_sha256"] == _file_sha256(source["storage_path"])
    assert (
        source["event_sequence_schema_version"]
        == ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION
    )
    assert (
        source["event_scaler_schema_version"]
        == ROUND74_EVENT_SCALER_SCHEMA_VERSION
    )
    assert (
        source["event_target_schema_version"]
        == ROUND74_EVENT_TARGET_SCHEMA_VERSION
    )
    assert (
        source["event_dataset_schema_version"]
        == ROUND74_EVENT_DATASET_SCHEMA_VERSION
    )
    assert (
        source["event_partition_schema_version"]
        == ROUND74_EVENT_PARTITION_SCHEMA_VERSION
    )
    assert (
        source["event_action_context_schema_version"]
        == ROUND74_ACTION_CONTEXT_SCHEMA_VERSION
    )
    assert (
        source["event_action_policy_schema_version"]
        == ROUND74_ACTION_POLICY_SCHEMA_VERSION
    )
    assert (
        source["ai_uplift_evaluator_schema_version"]
        == ROUND74_AI_UPLIFT_SCHEMA_VERSION
    )
    assert (
        source["sealed_ledger_schema_version"]
        == ROUND74_SEALED_LEDGER_SCHEMA_VERSION
    )
    assert (
        source["sealed_claim_schema_version"]
        == ROUND74_SEALED_CLAIM_SCHEMA_VERSION
    )
    assert (
        source["sealed_evaluator_schema_version"]
        == ROUND74_SEALED_EVALUATION_SCHEMA_VERSION
    )
    assert source["financial_metrics_schema_version"] == (
        ROUND74_REALIZED_METRICS_SCHEMA_VERSION
    )
    assert (
        source["target_free_inference_schema_version"]
        == ROUND74_TARGET_FREE_INFERENCE_SCHEMA_VERSION
    )
    assert (
        source["ai_review_panel_schema_version"]
        == ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION
    )
    assert (
        source["event_cohort_plan_schema_version"]
        == ROUND74_EVENT_COHORT_PLAN_SCHEMA_VERSION
    )
    assert (
        source["event_cohort_binding_schema_version"]
        == ROUND74_EVENT_COHORT_BINDING_SCHEMA_VERSION
    )
    assert (
        source["ai_model_manifest_schema_version"]
        == ROUND74_AI_MODEL_MANIFEST_SCHEMA_VERSION
    )
    assert (
        source["ai_review_request_schema_version"]
        == ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION
    )
    assert (
        source["ai_review_decision_schema_version"]
        == ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION
    )
    assert (
        source["ai_worker_envelope_schema_version"]
        == ROUND74_AI_WORKER_ENVELOPE_SCHEMA_VERSION
    )
    assert (
        source["ai_worker_result_schema_version"]
        == ROUND74_AI_WORKER_RESULT_SCHEMA_VERSION
    )
    assert (
        source["ai_runtime_outcome_schema_version"]
        == ROUND74_AI_RUNTIME_OUTCOME_SCHEMA_VERSION
    )
    assert (
        source["ai_bridge_schema_version"]
        == ROUND74_AI_BRIDGE_SCHEMA_VERSION
    )
    assert (
        source["tuning_subpartition_schema_version"]
        == ROUND74_TUNING_SUBPARTITION_SCHEMA_VERSION
    )
    assert (
        source["temperature_calibration_schema_version"]
        == ROUND74_TEMPERATURE_CALIBRATION_SCHEMA_VERSION
    )
    assert (
        source["event_training_schema_version"]
        == ROUND74_EVENT_TRAINING_SCHEMA_VERSION
    )
    assert (
        source["pretest_policy_schema_version"]
        == ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION
    )
    assert source["target_context_panel_schema_version"] == (
        ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION
    )
    assert source["feature_count"] == len(ROUND74_EVENT_FEATURE_NAMES) == 43
    assert source["feature_names_sha256"] == (
        ROUND74_EVENT_FEATURE_NAMES_SHA256
    )
    data_scope = design["data_scope"]
    assert data_scope["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert data_scope["exact_local_receipt_order_required"] is True
    assert data_scope["one_second_or_minute_collapse_permitted"] is False
    assert data_scope["listed_venue_calendar_may_create_crypto_close"] is False
    cohort = design["cohort_admission_contract"]
    assert cohort["implemented_now"] is True
    assert cohort["plan_sha256"] == (
        "c19caee6042531f0a86b9c2f3ef9b1de9380889eeb34758e6a27922fc6fee9e9"
    )
    assert cohort["role_counts"] == {
        "training": 120,
        "tuning": 24,
        "test": 24,
    }
    assert cohort["active_prerequisite_passed_now"] is False
    assert cohort["failed_or_missed_slot_replacement_permitted"] is False
    assert cohort["partition_hash_must_bind_plan_sha256"] is True
    assert cohort["maximum_target_span_seconds"] == 310.5
    assert cohort["minimum_purge_seconds"] == 310.5
    assert cohort["minimum_embargo_seconds"] == 310.5
    assert cohort["representative_market_data_collected_now"] is False
    features = design["causal_feature_contract"]
    assert features["per_event_asset_identity_retained"] is True
    assert features["window_may_cross_long_gap"] is False
    scaler = features["feature_scaler"]
    assert scaler["implemented_now"] is True
    assert scaler["fit_on_training_partition_only"] is True
    assert scaler["validation_or_test_statistics_permitted"] is False
    assert scaler["sample_indices_hash_bound"] is True
    assert scaler["sampling_algorithm"] == (
        "splitmix64-smallest-priority-v1"
    )
    assert scaler["serialized_scaler_digest_verified_on_load"] is True
    assert scaler["model_bundle_must_bind_scaler_hash"] is True


def test_round74_event_target_and_evaluation_contracts_fail_closed() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    targets = design["prospective_target_contract"]
    assert targets["implemented_now"] is True
    assert targets["real_market_targets_generated_now"] is False
    assert targets["decision_order_key"] == [
        "received_monotonic_ns",
        "frame_index",
        "message_index",
    ]
    assert (
        targets[
            "later_messages_with_the_same_receipt_timestamp_permitted_in_decision_state"
        ]
        is False
    )
    assert (
        targets[
            "target_context_hash_binds_spec_quantity_filters_and_funding_schedule"
        ]
        is True
    )
    latency = targets["decision_to_entry_and_exit_execution_latency"]
    assert latency["must_be_measured_on_the_execution_host"] is True
    assert latency["entry_and_exit_must_be_measured_separately"] is True
    assert latency["maximum_entry_latency_nanoseconds"] == 5_000_000_000
    assert latency["maximum_exit_latency_nanoseconds"] == 5_000_000_000
    assert latency["maximum_entry_state_lateness_nanoseconds"] == 250_000_000
    assert latency["maximum_exit_state_lateness_nanoseconds"] == 250_000_000
    assert latency["fixed_unverified_latency_assumption_permitted"] is False
    entry = targets["entry_and_exit"]
    assert entry["initial_supported_execution"] == "marketable orders only"
    assert entry["passive_maker_fill_target_permitted_from_l2"] is False
    assert entry["insufficient_visible_depth_policy"] == (
        "censor target and forbid action"
    )
    costs = targets["costs"]
    assert costs["commission_evidence_digest_required"] is True
    assert (
        costs["additional_residual_slippage_evidence_digest_required"]
        is True
    )
    assert costs["missing_account_fee_policy"] == "fail closed"
    assert costs["runtime_fee_mismatch_policy"] == (
        "model bundle is incompatible and cannot trade"
    )
    assert targets["leverage"][
        "leverage_is_applied_only_by_independent_risk_controller"
    ] is True
    assert targets["path_risk"]["maximum_path_state_gap_nanoseconds"] == (
        250_000_000
    )
    assert targets["path_risk"]["excessive_path_state_gap_policy"] == (
        "censor target"
    )
    dataset = design["dataset_assembly_contract"]
    assert dataset["implemented_now"] is True
    assert dataset["representative_market_dataset_built_now"] is False
    assert dataset["database_access"] == "read only"
    assert dataset["cohort_plan_digest_must_match_partition"] is True
    tuning = dataset["tuning_subpartition"]
    assert tuning["implemented_now"] is True
    assert tuning["expected_tuning_runs"] == 24
    assert tuning["model_selection_runs"] == 12
    assert tuning["probability_calibration_runs"] == 6
    assert tuning["action_policy_selection_runs"] == 6
    assert tuning["run_reuse_permitted"] is False
    assert tuning["sealed_test_accessed"] is False
    assert dataset["split_unit"] == "whole capture run"
    assert dataset["random_row_split_permitted"] is False
    assert dataset["minimum_purge_seconds"] == 310.5
    assert dataset["minimum_embargo_seconds"] == 310.5
    assert dataset[
        "maximum_target_span_seconds_including_latency_and_state_lateness_ceilings"
    ] == (
        310.5
    )
    assert dataset["batch_hash_binds_run_symbol_and_exact_decision_order"] is True
    assert (
        dataset[
            "batch_hash_binds_exact_target_entry_and_exit_monotonic_times"
        ]
        is True
    )
    assert (
        dataset["assembled_test_batch_retains_single_one_use_access_digest"]
        is True
    )
    assert dataset["development_batch_may_carry_test_access_digest"] is False
    assert dataset["per_capture_run_target_context_uniformity_required"] is True
    assert dataset["cross_run_target_context_change_permitted"] is True
    assert dataset["pretest_policy_binds_exact_sorted_target_context_panel"] is True
    action = design["action_policy_contract"]
    assert action["implemented_now"] is True
    assert action["representative_market_policy_selected_now"] is False
    assert action["profiles"] == list(ROUND74_ACTION_PROFILES)
    assert action["default_profile"] == ROUND74_ACTION_DEFAULT_PROFILE
    assert action["candidate_horizons_seconds"] == list(
        ROUND74_ACTION_HORIZONS_SECONDS
    )
    assert action["candidate_derivation_receives_realized_targets"] is False
    assert (
        action["candidate_input_context_contains_realized_target_fields"]
        is False
    )
    assert action["calibrated_probabilities_required"] is True
    assert action["maximum_candidates_per_row"] == 1
    assert (
        action["threshold_selection_data"]
        == "six whole chronological policy-selection tuning runs only"
    )
    assert action["sealed_test_accessed"] is False
    assert (
        action[
            "replay_uses_exact_captured_entry_and_exit_monotonic_times"
        ]
        is True
    )
    assert action["horizon_duration_used_as_exit_time_proxy"] is False
    assert action["drawdown_uses_actual_payoff_realization_order"] is True
    assert action["drawdown_order"] == (
        "cohort run then actual exit monotonic time then signal-order tie break"
    )
    assert (
        action[
            "shared_drawdown_implementation_required_for_ml_ai_and_sealed_evaluation"
        ]
        is True
    )
    assert action["diversification_requires_all_btc_eth_sol_symbols"] is True
    assert action["position_sizing_or_leverage_applied_here"] is False
    assert action["candidate_input_context_duplicates_full_feature_tensor"] is False
    assert action["selection_concatenates_full_feature_tensors"] is False
    assert action["selection_binds_every_target_batch_and_candidate_digest"] is True
    training = design["development_training_contract"]
    assert training["implemented_now"] is True
    assert training["representative_market_training_run_completed_now"] is False
    assert training["accepted_roles"] == ["training", "tuning"]
    assert training["test_role_rejected_before_backend_initialization"] is True
    assert training["training_and_tuning_sample_overlap_permitted"] is False
    assert training["minimum_role_transition_purge_seconds"] == 310.5
    assert training["mixed_partition_scaler_or_target_context_permitted"] is False
    assert training["one_capture_run_per_batch_required"] is True
    assert training["mixed_capture_run_batch_permitted"] is False
    assert training["repeated_capture_run_batch_permitted"] is False
    assert training["candidate_panel"] == [
        "event_pooling_mlp",
        "causal_event_tcn",
    ]
    assert training["default_seed_panel"] == list(
        ROUND74_EVENT_TRAINING_DEFAULT_SEEDS
    )
    assert training["backtest_roi_used_for_gradient_or_model_selection"] is False
    assert training["seed_ensemble_method"] == (
        "equal peer weights; arithmetic mean of peer quantiles for continuous "
        "heads and arithmetic mean of peer probabilities then logit conversion "
        "for classification heads"
    )
    assert training["classification_mean_logit_pooling_permitted"] is False
    assert training["run_balanced_loss_primary"] is True
    assert training["checkpoint_selection_metric"] == "run_balanced_loss"
    assert training["checkpoint_reload_verification_metric"] == (
        "run_balanced_loss"
    )
    assert training["pooled_loss_used_for_checkpoint_reload_verification"] is False
    assert training["unequal_tuning_run_size_regression_tested"] is True
    assert training["per_capture_run_target_context_uniformity_required"] is True
    assert training["cross_run_target_context_change_permitted"] is True
    assert training["exact_sorted_target_context_panel_bound_into_policy"] is True
    assert training["fully_censored_minibatch_policy"] == (
        "skip before device transfer and record minibatch and row counts"
    )
    assert training["fully_censored_capture_run_policy"] == "reject"
    assert training["censored_targets_used_as_negative_labels"] is False
    assert training["fully_censored_minibatch_regression_tested"] is True
    assert training["high_activity_run_receives_extra_selection_weight"] is False
    assert training["worst_run_loss_is_reported"] is True
    assert training["worst_run_loss_is_primary_optimization_objective"] is False
    assert training["probability_calibration_order"] == (
        "after seed-ensemble probability aggregation"
    )
    assert training["pickle_permitted"] is False
    assert training["policy_binds_entire_causal_source_chain"] is True
    assert training["cross_platform_bitwise_reproducibility_claim"] is False
    temporal = design["candidate_panel"]["causal_event_tcn"]
    assert temporal["dilations"] == list(ROUND74_EVENT_TCN_DILATIONS)
    assert temporal["causal_receptive_field_events"] == (
        ROUND74_EVENT_TCN_RECEPTIVE_FIELD
    )
    assert temporal["frozen_sequence_fully_covered"] is True
    assert temporal["frozen_sequence_length_events"] <= (
        temporal["causal_receptive_field_events"]
    )
    evaluation = design["prospective_evaluation_contract"]
    assert "design-consumed" in evaluation["quiet_qualification_run_status"]
    assert evaluation["random_row_split_permitted"] is False
    assert evaluation["minimum_purge_seconds"] == 310.5
    assert evaluation["sealed_test_may_be_used_once"] is True
    assert evaluation["durable_one_use_ledger_implemented_now"] is True
    assert evaluation["sealed_evaluator_implemented_now"] is True
    assert (
        evaluation[
            "reusable_target_free_candidate_inference_implemented_now"
        ]
        is True
    )
    assert evaluation["reservation_reset_api_available"] is False
    assert evaluation["sealed_test_runs"] == 24
    assert evaluation["sealed_bootstrap_draws"] == (
        ROUND74_SEALED_BOOTSTRAP_DRAWS
    )
    assert (
        evaluation[
            "annualized_roi_sharpe_sortino_or_calmar_reported_without_a_capital_allocation_path"
        ]
        is False
    )
    assert evaluation["profitability_required_by_assertion_or_data_filter"] is False
    assert design["ai_comparison_contract"][
        "ai_may_bypass_data_risk_or_execution_gate"
    ] is False
    ai = design["ai_comparison_contract"]
    assert ai["protocol_implemented_now"] is True
    assert ai["isolated_worker_implemented_now"] is True
    assert ai["fail_closed_parent_runtime_implemented_now"] is True
    assert ai["host_resource_preflight_completed_now"] is True
    assert ai["host_resource_preflight_passed_now"] is False
    assert ai["actual_model_inference_attempted_now"] is False
    assert ai["post_inference_full_gpu_residency_required"] is True
    assert ai["causal_calibrated_model_bridge_implemented_now"] is True
    assert ai["causal_recent_direction_summary_implemented_now"] is True
    assert ai["causal_recent_direction_block_events"] == (
        ROUND74_AI_RECENT_BLOCK_EVENTS
    )
    assert (
        ai["causal_recent_direction_realized_target_or_future_access"] is False
    )
    assert ai["bridge_may_access_realized_targets"] is False
    assert ai["raw_uncalibrated_probability_permitted"] is False
    assert ai["request_binds_probability_calibration_sha256"] is True
    assert ai["actual_multibillion_parameter_inference_completed_now"] is False
    assert ai["supported_review_horizons_seconds"] == [30, 300]
    assert ai["one_and_five_second_ml_paths_wait_for_ai"] is False
    assert ai["ai_receives_only_a_preexisting_target_free_ml_candidate"] is True
    assert ai["ai_may_revive_an_ml_abstention"] is False
    assert ai["paired_development_evaluator_implemented_now"] is True
    assert ai["paired_sealed_evaluator_implemented_now"] is True
    assert ai["durable_one_use_sealed_ledger_implemented_now"] is True
    assert ai["two_model_target_free_review_preparation_implemented_now"] is True
    assert ai["review_progress_emitted_after_every_candidate_model_pair"] is True
    assert (
        ai["paired_development_evaluator_may_select_ai_model_or_promote"]
        is False
    )
    assert ai["missing_review_policy"] == "invalidate entire evaluation"
    assert ai["absolute_date_or_real_symbol_exposed_to_ai"] is False
    assert (
        ai[
            "target_eligibility_or_realized_exit_timing_may_select_ai_review_coverage"
        ]
        is False
    )
    assert ai["ai_veto_may_admit_a_later_overlapping_candidate"] is False
    assert ai[
        "ai_may_create_side_increase_size_set_leverage_or_touch_orders"
    ] is False
    assert ai["deferred_finance_challenger_count"] == 2
    assert ai["deferred_challenger_downloaded_now"] is False
    assert ai["financial_qa_benchmark_implies_trading_edge"] is False
    assert (
        ai[
            "challenger_panel_expansion_requires_predeclared_resource_safe_packet_screen"
        ]
        is True
    )
    authority = design["authority"]
    assert authority["training_only_scaler_implementation"] is True
    assert authority["prospective_target_engine_implementation"] is True
    assert authority["leak_resistant_dataset_implementation"] is True
    assert authority["development_trainer_implementation"] is True
    assert authority["immutable_pretest_policy_implementation"] is True
    assert authority["predeclared_cohort_admission_implementation"] is True
    assert authority["predeclared_cohort_plan"] is True
    assert authority["local_ai_review_protocol_implementation"] is True
    assert authority["tuning_subpartition_implementation"] is True
    assert authority["probability_calibration_implementation"] is True
    assert authority["selective_action_policy_implementation"] is True
    assert (
        authority[
            "paired_ai_uplift_development_evaluator_implementation"
        ]
        is True
    )
    assert authority["causal_calibrated_ai_bridge_implementation"] is True
    assert authority["durable_one_use_sealed_ledger_implementation"] is True
    assert authority["paired_ml_ai_sealed_evaluator_implementation"] is True
    assert authority["target_free_candidate_inference_implementation"] is True
    assert authority["two_model_ai_review_preparation_implementation"] is True
    assert (
        authority["probability_calibration_directml_compute_preflight"]
        is True
    )
    assert authority["local_ai_isolated_worker_implementation"] is True
    assert authority["local_ai_fail_closed_parent_runtime_implementation"] is True
    assert authority["local_ai_host_resource_preflight"] is True
    assert authority["actual_multibillion_parameter_ai_inference"] is False
    for key in (
        "target_generation",
        "model_training",
        "model_selection",
        "financial_edge_tested",
        "profitability_claim",
        "ai_uplift_claim",
        "paper_trading_authority",
        "testnet_trading_authority",
        "live_trading_authority",
    ):
        assert authority[key] is False


def test_round74_event_replay_evidence_is_exact_read_only_and_pre_target() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    replay = _load_hash_bound(REPLAY_PATH, "artifact_sha256")

    binding = design["host_evidence_binding"]
    assert replay["execution_git_commit"] == binding[
        "event_sequence_replay_execution_git_commit"
    ]
    assert replay["artifact_sha256"] == binding[
        "event_sequence_replay_artifact_sha256"
    ]
    assert replay["event_sequence_source_sha256"] == design["source_binding"][
        "event_sequence_sha256"
    ]
    evidence = replay["replay"]
    assert evidence["observation_count"] == 50_444
    assert evidence["token_count"] == evidence["token_limit"] == 50_000
    assert evidence["synchronized_depth_state_count"] == 4_984
    assert sum(evidence["event_counts"].values()) == 50_000
    assert sum(evidence["symbol_counts"].values()) == 50_000
    assert set(evidence["symbol_counts"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert evidence["canonical_length_prefixed_token_sha256"] == (
        "03f52996078dbd514570b83434ae1bb15d713103448a624ddd82edc4be677f02"
    )
    storage = replay["storage_safety"]
    assert storage["database_open_mode"] == "read_only"
    assert storage["database_size_before_bytes"] == (
        storage["database_size_after_bytes"]
    )
    assert storage["wal_size_before_bytes"] == storage["wal_size_after_bytes"] == 0
    assert storage["database_write_detected"] is False
    limitations = replay["observed_sample_limitations"]
    assert limitations["liquidation_event_count"] == 0
    assert limitations["all_observation_stale_depth_event_count"] == 90
    assert limitations["feature_ready_stale_depth_event_count"] == 0
    assert limitations["stale_depth_used_as_fresh_target_state"] is False
    assert limitations["sample_validates_liquidation_path"] is False
    assert limitations["sample_is_financially_representative"] is False
    scaler = replay["diagnostic_scaler"]
    assert scaler["input_event_rows"] == 50_000
    assert scaler["bounded_sample_rows"] == 10_000
    assert scaler["chunk_invariance_verified"] is True
    assert scaler["eligible_for_model_bundle"] is False
    interpretation = replay["interpretation"]
    assert interpretation["targets_constructed"] is False
    assert interpretation["models_evaluated"] is False
    assert interpretation["financial_edge_tested"] is False


def test_round74_directml_evidence_is_amd_accelerated_and_nonfinancial() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    evidence = _load_hash_bound(DIRECTML_PATH, "artifact_sha256")

    binding = design["host_evidence_binding"]
    assert evidence["execution_git_commit"] == binding[
        "event_model_directml_execution_git_commit"
    ]
    assert evidence["artifact_sha256"] == binding[
        "event_model_directml_artifact_sha256"
    ]
    assert evidence["event_model_source_sha256"] == design["source_binding"][
        "event_model_sha256"
    ]
    backend = evidence["backend"]
    assert backend["requested"] == backend["kind"] == "directml"
    assert backend["vendor"] == "AMD Radeon RX 9070 XT"
    assert backend["accelerated"] is True
    verification = evidence["verification"]
    assert verification["forward_completed"] is True
    assert verification["backward_completed"] is True
    assert verification["parameter_update_completed"] is True
    assert verification["warning_count"] == 0
    assert verification["cpu_fallback_warning_count"] == 0
    assert verification["tcn_receptive_field_events"] == (
        ROUND74_EVENT_TCN_RECEPTIVE_FIELD
    )
    assert verification["frozen_sequence_fully_covered"] is True
    assert evidence["input_contract"]["masked_action_targets"] == 1
    for candidate_id, candidate in evidence["candidates"].items():
        assert candidate["parameter_count"] == design["candidate_panel"][
            candidate_id
        ]["parameter_count"]
        assert candidate["parameter_max_abs_change"] > 0.0
    interpretation = evidence["interpretation"]
    assert interpretation["real_market_targets_used"] is False
    assert interpretation["model_fit_performed"] is False
    assert interpretation["financial_edge_tested"] is False
    assert interpretation["profitability_claim"] is False


def test_round74_training_preflight_is_repeated_amd_compute_only() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    evidence = _load_hash_bound(TRAINING_PATH, "artifact_sha256")
    binding = design["host_evidence_binding"]

    assert evidence["execution_git_commit"] == binding[
        "event_training_directml_execution_git_commit"
    ]
    assert evidence["artifact_sha256"] == binding[
        "event_training_directml_artifact_sha256"
    ]
    source = evidence["source_binding"]
    assert source["event_targets_sha256"] == design["source_binding"][
        "event_target_sha256"
    ]
    assert source["event_training_sha256"] == design["source_binding"][
        "event_training_sha256"
    ]
    assert source["event_dataset_sha256"] == design["source_binding"][
        "event_dataset_sha256"
    ]
    assert source["event_model_sha256"] == design["source_binding"][
        "event_model_sha256"
    ]
    assert source["event_cohort_sha256"] == binding[
        "event_training_directml_event_cohort_sha256"
    ]
    assert source["event_cohort_sha256"] == design["source_binding"][
        "event_cohort_sha256"
    ]
    assert "exactly binds" in binding["event_training_directml_reuse_scope"]
    assert source["preflight_runner_sha256"] == _file_sha256(
        source["preflight_runner_path"]
    )
    backend = evidence["backend"]
    assert backend["requested"] == backend["kind"] == "directml"
    assert backend["vendor"] == "AMD Radeon RX 9070 XT"
    assert backend["accelerated"] is True
    assert backend["safetensors_version"] == "0.8.0"
    assert backend["warning_count_per_execution"] == 0
    assert backend["cpu_fallback_warning_count_per_execution"] == 0
    inputs = evidence["input_contract"]
    assert inputs["real_market_events_used"] is False
    assert inputs["real_market_targets_used"] is False
    assert inputs["test_batches_consumed"] == 0
    assert inputs["candidate_ids"] == ["event_pooling_mlp", "causal_event_tcn"]
    assert inputs["seeds"] == list(ROUND74_EVENT_TRAINING_DEFAULT_SEEDS)
    verification = evidence["verification"]
    assert verification["fresh_process_execution_count"] == 2
    assert verification["all_six_peer_updates_completed"] is True
    assert verification["cross_execution_policy_sha256_equal"] is True
    assert verification["cross_execution_model_sha256_equal"] is True
    assert verification["cross_execution_prediction_sha256_equal"] is True
    assert verification["cross_execution_candidate_metrics_equal"] is True
    assert verification["temporary_artifacts_removed_after_each_execution"] is True
    assert (
        verification[
            "classification_peer_probabilities_averaged_before_logit_conversion"
        ]
        is True
    )
    assert (
        verification["post_aggregation_probability_calibration_contract_sealed"]
        is True
    )
    assert verification["tuning_metric_unit"] == "one-hour capture run"
    assert verification["synthetic_tuning_run_count"] == 1
    assert verification["each_tuning_run_has_equal_selection_weight"] is True
    assert verification["mixed_or_repeated_capture_run_policy"] == "fail closed"
    assert verification["checkpoint_selection_metric"] == "run_balanced_loss"
    assert verification["checkpoint_reload_verification_metric"] == (
        "run_balanced_loss"
    )
    assert verification["pooled_loss_used_for_checkpoint_reload_verification"] is False
    assert verification["unequal_tuning_run_size_regression_test_passed"] is True
    assert verification["target_context_panel_schema_version"] == (
        ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION
    )
    assert verification["per_run_target_context_uniformity_required"] is True
    assert verification["cross_run_target_context_change_permitted"] is True
    assert verification["pretest_policy_binds_exact_sorted_target_context_panel"] is True
    assert verification["fully_censored_minibatch_policy"] == (
        "skip before device transfer and record counts"
    )
    assert verification["fully_censored_capture_run_policy"] == "reject"
    assert verification["censored_targets_used_as_negative_labels"] is False
    assert verification["fully_censored_minibatch_regression_test_passed"] is True
    assert verification["tcn_receptive_field_events"] == (
        ROUND74_EVENT_TCN_RECEPTIVE_FIELD
    )
    assert verification["frozen_sequence_fully_covered"] is True
    repeated = evidence["repeated_result"]
    assert repeated["candidate_run_balanced_tuning_proper_loss"] == (
        repeated["candidate_worst_run_tuning_proper_loss"]
    )
    assert repeated["candidate_run_balanced_tuning_proper_loss"] == (
        repeated["candidate_pooled_tuning_proper_loss"]
    )
    assert verification["maximum_entry_state_lateness_ns"] == 250_000_000
    assert verification["maximum_exit_state_lateness_ns"] == 250_000_000
    assert verification["maximum_target_span_ns"] == 310_500_000_000
    assert verification["minimum_partition_purge_ns"] == 310_500_000_000
    assert verification["minimum_partition_embargo_ns"] == 310_500_000_000
    interpretation = evidence["interpretation"]
    assert interpretation["candidate_loss_has_financial_meaning"] is False
    assert interpretation["representative_market_training_performed"] is False
    assert interpretation["sealed_test_evaluated"] is False
    assert interpretation["financial_edge_tested"] is False
    assert interpretation["profitability_claim"] is False


def test_round74_calibration_preflight_is_amd_compute_only() -> None:
    design = _load_hash_bound(DESIGN_PATH, "design_sha256")
    evidence = _load_hash_bound(CALIBRATION_PATH, "artifact_sha256")
    binding = design["host_evidence_binding"]

    assert evidence["execution_git_commit"] == binding[
        "temperature_calibration_directml_execution_git_commit"
    ]
    assert evidence["artifact_sha256"] == binding[
        "temperature_calibration_directml_artifact_sha256"
    ]
    assert evidence["source_binding"]["sha256"] == design[
        "source_binding"
    ]["event_calibration_sha256"]
    backend = evidence["backend"]
    assert backend["requested"] == backend["kind"] == "directml"
    assert backend["vendor"] == "AMD Radeon RX 9070 XT"
    assert backend["accelerated"] is True
    assert backend["warning_count"] == 0
    assert backend["cpu_fallback_warning_count"] == 0
    verification = evidence["verification"]
    assert verification["candidate_temperature_count"] == 257
    assert verification["all_three_temperature_searches_completed"] is True
    assert verification["hidden_cpu_fallback_detected"] is False
    interpretation = evidence["interpretation"]
    assert interpretation["representative_market_calibration_performed"] is False
    assert interpretation["financial_edge_tested"] is False
    assert interpretation["profitability_claim"] is False
