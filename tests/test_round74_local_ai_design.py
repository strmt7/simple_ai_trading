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
    ROUND74_AI_EXECUTION_REPLAY_EVIDENCE_SCHEMA_VERSION,
    ROUND74_AI_UPLIFT_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_execution_replay import (
    ROUND74_AI_EXECUTION_REPLAY_PLAN_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_worker import (
    ROUND74_AI_WORKER_ENVELOPE_SCHEMA_VERSION,
    ROUND74_AI_WORKER_RESULT_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_sealed_evaluation import (
    ROUND74_SEALED_AI_MODEL_COUNT,
    ROUND74_SEALED_BOOTSTRAP_DRAWS,
    ROUND74_SEALED_EVALUATION_SCHEMA_VERSION,
    ROUND74_SEALED_FAMILYWISE_ALPHA,
    ROUND74_SEALED_QUALIFICATION_CONFIGURATION_COUNT,
    ROUND74_TARGET_FREE_INFERENCE_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_sealed_ledger import (
    ROUND74_SEALED_CLAIM_SCHEMA_VERSION,
    ROUND74_SEALED_DATASET_IDENTITY_SCHEMA_VERSION,
    ROUND74_SEALED_LEDGER_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_financial_metrics import (
    ROUND74_REALIZED_METRICS_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_EVIDENCE_SCHEMA_VERSION,
    ROUND74_EVENT_TARGET_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_evidence import (
    ROUND74_BINANCE_CLOCK_PROBE_SCHEMA_VERSION,
    ROUND74_BINANCE_EVIDENCE_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_execution_evidence import (
    ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL,
    ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL_SIDE,
    ROUND74_EXECUTION_CALIBRATION_QUANTILE,
    ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE,
    ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_exchange_info_evidence import (
    ROUND74_EXCHANGE_INFO_EVIDENCE_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_target_assembly import (
    ROUND74_SOURCE_TARGET_ASSEMBLY_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_model import (
    ROUND74_EVENT_MODEL_CANDIDATES,
    ROUND74_EVENT_MODEL_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_training import (
    ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION,
    ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
)
from simple_ai_trading.round74_event_model_operator import (
    ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-local-ai-review-design-v41.json"
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


def _file_sha256(relative_path: str) -> str:
    payload = (REPOSITORY / relative_path).read_bytes()
    canonical = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            assert key not in value
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    assert isinstance(value, dict)
    return value


def test_round74_local_ai_design_is_source_bound_and_fail_closed() -> None:
    artifact = _load_json(ARTIFACT_PATH)
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    assert artifact["file_sha256_normalization"] == (
        "text_bytes_crlf_and_cr_normalized_to_lf_before_sha256"
    )
    source = artifact["source_binding"]
    assert (
        source["file_sha256_normalization"] == (artifact["file_sha256_normalization"])
    )
    for label in (
        "protocol",
        "bridge",
        "calibration",
        "action_policy",
        "targets",
        "target_source_evidence",
        "target_assembly",
        "execution_calibration_evidence",
        "uplift_evaluator",
        "execution_replay",
        "sealed_ledger",
        "sealed_evaluator",
        "financial_metrics",
        "review_preparation",
        "worker",
        "runtime",
        "event_model",
        "event_training",
        "event_model_operator",
    ):
        assert source[f"{label}_sha256"] == _file_sha256(source[f"{label}_path"])
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
    assert source["target_schema_version"] == ROUND74_EVENT_TARGET_SCHEMA_VERSION
    assert source["target_evidence_schema_version"] == (
        ROUND74_EVENT_TARGET_EVIDENCE_SCHEMA_VERSION
    )
    assert source["uplift_evaluator_schema_version"] == (
        ROUND74_AI_UPLIFT_SCHEMA_VERSION
    )
    assert source["execution_replay_plan_schema_version"] == (
        ROUND74_AI_EXECUTION_REPLAY_PLAN_SCHEMA_VERSION
    )
    assert source["execution_replay_evidence_schema_version"] == (
        ROUND74_AI_EXECUTION_REPLAY_EVIDENCE_SCHEMA_VERSION
    )
    assert source["sealed_ledger_schema_version"] == (
        ROUND74_SEALED_LEDGER_SCHEMA_VERSION
    )
    assert source["sealed_dataset_identity_schema_version"] == (
        ROUND74_SEALED_DATASET_IDENTITY_SCHEMA_VERSION
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
    assert (
        source["target_source_evidence_schema_version"]
        == ROUND74_BINANCE_EVIDENCE_SCHEMA_VERSION
    )
    assert (
        source["target_clock_probe_schema_version"]
        == ROUND74_BINANCE_CLOCK_PROBE_SCHEMA_VERSION
    )
    assert source["exchange_info_evidence_sha256"] == _file_sha256(
        source["exchange_info_evidence_path"]
    )
    assert (
        source["exchange_info_evidence_schema_version"]
        == ROUND74_EXCHANGE_INFO_EVIDENCE_SCHEMA_VERSION
    )
    assert (
        source["target_assembly_schema_version"]
        == ROUND74_SOURCE_TARGET_ASSEMBLY_SCHEMA_VERSION
    )
    assert (
        source["execution_calibration_schema_version"]
        == ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION
    )
    assert source["event_model_schema_version"] == ROUND74_EVENT_MODEL_SCHEMA_VERSION
    assert (
        source["event_training_schema_version"] == ROUND74_EVENT_TRAINING_SCHEMA_VERSION
    )
    assert source["pretest_policy_schema_version"] == (
        ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION
    )
    assert (
        source["event_model_operator_schema_version"]
        == ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION
    )
    model_design_path = REPOSITORY / source["event_model_design_path"]
    assert source["event_model_design_file_sha256"] == _file_sha256(
        source["event_model_design_path"]
    )
    model_design = _load_json(model_design_path)
    model_design_claimed = model_design.pop("design_sha256")
    assert model_design_claimed == _canonical_sha256(model_design)
    assert model_design_claimed == source["event_model_design_sha256"]
    market = artifact["market_evidence_binding"]
    assert (
        market["file_sha256_normalization"] == (artifact["file_sha256_normalization"])
    )
    exchange_path = REPOSITORY / market["exchange_info_path"]
    assert market["exchange_info_file_sha256"] == _file_sha256(
        market["exchange_info_path"]
    )
    exchange = _load_json(exchange_path)
    exchange_claimed = exchange.pop("artifact_sha256")
    assert exchange_claimed == _canonical_sha256(exchange)
    assert exchange_claimed == market["exchange_info_artifact_sha256"]
    assert exchange["target_evidence"]["evidence_sha256"] == (
        market["exchange_info_target_evidence_sha256"]
    )
    assert market["exchange_info_raw_payload_persisted"] is False
    funding_path = REPOSITORY / market["funding_path"]
    assert market["funding_file_sha256"] == _file_sha256(market["funding_path"])
    funding = _load_json(funding_path)
    funding_claimed = funding.pop("artifact_sha256")
    assert funding_claimed == _canonical_sha256(funding)
    assert funding_claimed == market["funding_artifact_sha256"]
    assert funding["target_evidence"]["evidence_sha256"] == (
        market["funding_target_evidence_sha256"]
    )
    assert funding["capture_binding"]["run_id"] == (
        market["funding_capture_run_id"]
    )
    assert (
        funding["scope"]["capture_run_may_be_used_for_financial_evaluation"]
        is False
    )
    compute = artifact["model_compute_evidence_binding"]
    assert (
        compute["file_sha256_normalization"] == (artifact["file_sha256_normalization"])
    )
    compute_path = REPOSITORY / compute["path"]
    assert compute["file_sha256"] == _file_sha256(compute["path"])
    compute_evidence = _load_json(compute_path)
    compute_claimed = compute_evidence.pop("artifact_sha256")
    assert compute_claimed == _canonical_sha256(compute_evidence)
    assert compute_claimed == compute["artifact_sha256"]
    assert compute["candidate_ids"] == list(ROUND74_EVENT_MODEL_CANDIDATES)
    assert compute["fresh_process_execution_count"] == 2
    assert compute["peer_update_count"] == 9
    assert compute["accelerated_backend"] == "directml"
    assert compute["warning_count_per_execution"] == 0
    assert compute["cpu_fallback_warning_count_per_execution"] == 0
    assert compute["selected_candidate_id"] == "event_pooling_linear"
    assert compute["observed_paired_capture_run_count"] == 1
    assert compute["required_paired_capture_run_count"] == 24
    assert compute["complete_tuning_panel"] is False
    assert compute["training_capture_run_count"] == 2
    assert compute["optimizer_steps_per_peer"] == 3
    assert compute["run_contributions_per_optimizer_step"] == 2
    assert compute["minimum_eligible_minibatches_per_run"] == 1
    assert compute["maximum_eligible_minibatches_per_run"] == 3
    assert compute["minimum_run_minibatch_contributions"] == 3
    assert compute["maximum_run_minibatch_contributions"] == 3
    assert compute["optimization_population_unit"] == "capture_run"
    assert compute["unequal_training_run_sizes_proven"] is True
    assert compute["statistical_independence_or_significance_claim"] is False
    assert compute["nonlinear_candidate_promoted"] is False
    assert compute["constructed_tensor_compute_only"] is True
    assert compute["real_market_events_used"] is False
    assert compute["candidate_loss_has_financial_meaning"] is False
    assert compute["financial_edge_tested"] is False
    assert compute["profitability_claim"] is False
    assert compute["ai_inference_exercised"] is False
    assert compute["exact_capital_accounting_sources_bound"] is True
    assert compute["market_accounting_or_edge_claim"] is False
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
    assert architecture["ml_candidate_panel_includes_linear_microstructure_control"] is True
    assert architecture["ml_gradient_population_unit"] == "capture_run"
    assert architecture["ml_row_pooled_optimizer_steps_permitted"] is False
    assert architecture["ml_one_minibatch_per_run_per_optimizer_step"] is True
    assert architecture["ml_unequal_run_directml_preflight_completed"] is True
    assert (
        architecture["reference_capital_normalization_uses_actual_walked_entry_quote"]
        is True
    )
    assert (
        architecture["requested_size_multiplier_used_as_realized_notional_proxy"]
        is False
    )
    assert architecture["baseline_and_ai_share_reference_capital_denominator"] is True
    assert (
        architecture[
            "public_sealed_entrypoint_accepts_metadata_identity_not_target_batches"
        ]
        is True
    )
    assert architecture["metadata_identity_reserved_before_test_batch_loader"] is True
    assert (
        architecture["test_batch_identity_reconciled_to_reservation_before_scoring"]
        is True
    )
    assert (
        architecture[
            "sealed_review_provider_receives_target_free_contexts_and_candidates_only"
        ]
        is True
    )
    assert (
        architecture["sealed_replay_provider_invoked_only_after_live_reservation"]
        is True
    )
    assert (
        architecture[
            "precomputed_execution_replay_evidence_accepted_by_public_sealed_entrypoint"
        ]
        is False
    )
    assert (
        architecture[
            "ml_candidate_selection_requires_paired_capture_run_complexity_promotion"
        ]
        is True
    )
    assert (
        architecture["ml_candidate_selection_requires_complete_24_run_tuning_panel"]
        is True
    )
    assert (
        architecture[
            "ml_candidate_selection_permits_any_material_paired_run_degradation"
        ]
        is False
    )
    assert (
        architecture["ml_candidate_selection_makes_dependent_run_significance_claim"]
        is False
    )
    assert architecture["ml_candidate_selection_uses_ai_output"] is False
    assert architecture["ml_candidate_selection_uses_sealed_test"] is False
    assert architecture["ai_receives_only_the_preselected_target_free_ml_candidate"] is True
    assert architecture["ai_may_choose_the_ml_candidate_architecture"] is False
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
    assert architecture["same_entry_latency_eligibility_is_diagnostic_only"] is True
    assert architecture["latency_adjusted_delayed_entry_replay_implemented"] is True
    assert (
        architecture[
            "raw_sample_and_feature_window_hashes_preserved_into_replay_plan"
        ]
        is True
    )
    assert (
        architecture[
            "ai_size_reduction_applied_before_quantity_quantization_and_l2_walk"
        ]
        is True
    )
    assert architecture["baseline_fill_or_payoff_reuse_permitted"] is False
    assert architecture["historical_review_expiration_ceiling_nanoseconds"] == (
        30_000_000_000
    )
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
    assert architecture["action_threshold_quantiles_use_equal_capture_run_weights"] is True
    assert architecture["action_threshold_estimator"] == (
        "median of each active run's within-run linear quantile"
    )
    assert architecture["pooled_total_trade_bps_used_for_action_selection"] is False
    assert architecture["busy_run_row_duplication_regression_tested"] is True
    assert architecture["historical_ai_queue_model_implemented"] is True
    assert architecture["ai_queue_model"] == (
        "one FIFO single-server queue per alternative candidate model"
    )
    assert architecture["review_row_hash_binds_decision_wall_nanoseconds"] is True
    assert (
        architecture[
            "review_hash_binds_service_queue_and_effective_latency_nanoseconds"
        ]
        is True
    )
    assert architecture["same_entry_latency_definition"] == (
        "historical queue delay plus measured parent runtime elapsed"
    )
    assert (
        architecture["inference_service_time_alone_may_establish_same_entry_eligibility"]
        is False
    )
    assert architecture["sealed_familywise_alpha"] == ROUND74_SEALED_FAMILYWISE_ALPHA
    assert architecture["sealed_qualification_configuration_count"] == (
        ROUND74_SEALED_QUALIFICATION_CONFIGURATION_COUNT
    )
    assert architecture["sealed_paired_ai_model_count"] == (
        ROUND74_SEALED_AI_MODEL_COUNT
    )
    assert architecture["unadjusted_interval_may_promote_a_configuration"] is False
    assert (
        architecture["funding_boundary_panel_is_mandatory_and_target_hash_bound"]
        is True
    )
    assert architecture["funding_schedule_source_evidence_digest_is_required"] is True
    assert architecture["silently_empty_or_partial_funding_schedule_permitted"] is False
    assert architecture["funding_crossing_without_a_payment_model_policy"] == (
        "censor target and reject any selected action configuration with incomplete "
        "coverage"
    )
    assert architecture["entry_and_exit_latency_are_separately_bound_per_symbol"] is True
    assert (
        architecture["residual_slippage_is_bound_per_symbol_and_reference_notional"]
        is True
    )
    assert (
        architecture["single_global_latency_or_residual_slippage_assumption_permitted"]
        is False
    )
    assert architecture["structured_target_evidence_records_bind_exact_claims"] is True
    assert (
        architecture[
            "target_evidence_records_bind_environment_source_time_count_query_and_payload"
        ]
        is True
    )
    assert architecture["mixed_target_evidence_environments_permitted"] is False
    assert architecture["quantity_rules_are_parsed_from_exchange_info"] is True
    assert architecture["quantity_rules_evidence_is_required_and_target_hash_bound"] is True
    assert (
        architecture["quantity_rules_use_market_lot_size_and_min_notional_filters"]
        is True
    )
    assert architecture["quantity_precision_may_substitute_for_market_lot_size"] is False
    assert architecture["exchange_info_requires_trading_perpetual_usdt_contracts"] is True
    assert architecture["real_public_exchange_info_evidence_captured"] is True
    assert architecture["source_derived_target_assembly_implemented"] is True
    assert (
        architecture["bounded_post_cohort_model_data_operator_implemented"]
        is True
    )
    assert (
        architecture[
            "source_target_assembly_roundtrip_serialization_implemented"
        ]
        is True
    )
    assert architecture["model_scaler_reads_unique_training_events_only"] is True
    assert architecture["model_operator_reads_capture_database_only"] is True
    assert architecture["model_operator_persists_overlapping_windows"] is False
    assert architecture["development_model_operator_accesses_test_role"] is False
    assert (
        architecture[
            "source_derived_target_assembly_accepts_caller_configured_fees_latency_slippage_or_quantity_rules"
        ]
        is False
    )
    assert (
        architecture[
            "source_derived_target_assembly_hash_binds_spec_and_quantity_claims"
        ]
        is True
    )
    assert (
        architecture["source_derived_target_engine_revalidates_quantity_evidence"]
        is True
    )
    assert (
        architecture[
            "target_payoff_exposes_midpoint_book_walk_explicit_cost_and_total_implementation_shortfall"
        ]
        is True
    )
    assert architecture["fees_plus_residual_slippage_may_be_labeled_total_cost"] is False
    assert architecture["target_accounting_reconciles_in_quote_and_basis_points"] is True
    assert architecture["adverse_selection_uses_midpoint_move_before_execution_friction"] is True
    assert architecture["funding_boundary_is_rechecked_against_actual_exit_timestamp"] is True
    assert (
        architecture["requested_exit_before_funding_cannot_hide_actual_exit_after_funding"]
        is True
    )
    assert architecture["per_symbol_funding_schedule_coverage_is_mandatory"] is True
    assert architecture["targets_outside_verified_funding_coverage_are_censored"] is True
    assert architecture["funding_times_are_bounded_monotonic_uncertainty_intervals"] is True
    assert (
        architecture[
            "positions_overlapping_any_funding_uncertainty_interval_are_censored"
        ]
        is True
    )
    assert architecture["exact_timestamp_false_precision_for_funding_permitted"] is False
    assert (
        architecture[
            "commission_evidence_is_parsed_from_exact_signed_endpoint_responses"
        ]
        is True
    )
    assert architecture["credential_material_may_enter_evidence_or_artifacts"] is False
    assert architecture["funding_history_full_limit_page_is_accepted_as_complete"] is False
    assert (
        architecture[
            "empty_explicitly_bounded_funding_response_is_accepted_as_complete"
        ]
        is True
    )
    assert (
        architecture["missing_symbol_funding_response_is_accepted_as_empty"]
        is False
    )
    assert (
        architecture[
            "funding_evidence_record_count_includes_symbol_responses_and_rows"
        ]
        is True
    )
    assert (
        architecture[
            "funding_clock_probes_are_loaded_only_from_completed_audited_v10_capture"
        ]
        is True
    )
    assert architecture["funding_clock_mapping_interpolates_between_probes"] is False
    assert (
        architecture[
            "funding_boundary_uses_preceding_request_start_and_following_receipt"
        ]
        is True
    )
    assert (
        architecture[
            "only_frames_with_clock_probes_are_decompressed_after_capture_audit"
        ]
        is True
    )
    assert (
        architecture[
            "execution_calibration_requires_flat_before_entry_and_after_reduce_only_exit"
        ]
        is True
    )
    assert architecture["execution_calibration_client_order_ids_are_bot_namespaced"] is True
    assert (
        architecture[
            "execution_calibration_reconciles_terminal_quantity_average_price_and_account_fills"
        ]
        is True
    )
    assert (
        architecture[
            "execution_calibration_expected_price_is_derived_from_fresh_captured_l2_book_walk"
        ]
        is True
    )
    assert architecture["execution_calibration_minimum_completed_pairs_per_symbol"] == (
        ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL
    )
    assert (
        architecture[
            "execution_calibration_minimum_completed_pairs_per_symbol_entry_side"
        ]
        == ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL_SIDE
    )
    assert architecture["execution_tail_quantile"] == (
        ROUND74_EXECUTION_CALIBRATION_QUANTILE
    )
    assert architecture["execution_tail_confidence"] == (
        ROUND74_EXECUTION_CALIBRATION_QUANTILE_CONFIDENCE
    )
    assert "distribution-free" in architecture["execution_tail_estimator"]
    assert (
        architecture[
            "execution_calibration_parser_places_orders_or_grants_trading_authority"
        ]
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
    artifact = _load_json(ARTIFACT_PATH)
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
    assert status["current_three_candidate_ml_compute_preflight_completed"] is True
    assert status["current_complexity_promotion_compute_preflight_completed"] is True
    assert status["equal_run_gradient_optimization_implemented"] is True
    assert status["unequal_run_directml_gradient_schedule_preflight_completed"] is True
    assert status["current_three_candidate_ml_market_training_completed"] is False
    assert status["isolated_worker_implemented"] is True
    assert status["fail_closed_parent_runtime_implemented"] is True
    assert status["causal_calibrated_bridge_implemented"] is True
    assert status["target_free_action_preselection_implemented"] is True
    assert status["paired_uplift_evaluator_implemented"] is True
    assert status["durable_one_use_sealed_ledger_implemented"] is True
    assert status["sealed_ml_ai_evaluator_implemented"] is True
    assert status["target_free_candidate_inference_implemented"] is True
    assert status["two_model_review_preparation_implemented"] is True
    assert status["same_entry_latency_uplift_gate_implemented"] is False
    assert status["same_entry_latency_retained_as_diagnostic"] is True
    assert status["exact_delayed_execution_replay_implemented"] is True
    assert status["sealed_exact_execution_evidence_implemented"] is True
    assert status["exact_reference_capital_accounting_implemented"] is True
    assert status["metadata_only_sealed_reservation_implemented"] is True
    assert status["post_reservation_test_batch_loader_implemented"] is True
    assert status["post_reservation_target_free_ai_review_provider_implemented"] is True
    assert status["post_reservation_exact_replay_provider_implemented"] is True
    assert status["concrete_target_free_ai_review_adapter_implemented"] is True
    assert status["read_only_store_exact_replay_adapter_implemented"] is True
    assert status["operator_cannot_preinspect_test_data_claim"] is False
    assert status["future_censorship_promotion_gate_implemented"] is True
    assert status["equal_run_action_selection_implemented"] is True
    assert status["historical_ai_queue_latency_implemented"] is True
    assert status["sealed_multiple_comparison_control_implemented"] is True
    assert status["mandatory_funding_schedule_binding_implemented"] is True
    assert status["symbol_specific_execution_evidence_implemented"] is True
    assert status["implementation_shortfall_reconciliation_implemented"] is True
    assert status["actual_exit_funding_recheck_implemented"] is True
    assert status["funding_schedule_coverage_gate_implemented"] is True
    assert status["funding_clock_uncertainty_interval_implemented"] is True
    assert status["binance_source_evidence_parser_implemented"] is True
    assert status["audited_clock_probe_loader_implemented"] is True
    assert status["execution_calibration_evidence_parser_implemented"] is True
    assert status["exchange_info_quantity_rules_parser_implemented"] is True
    assert status["quantity_rules_runtime_evidence_match_gate_implemented"] is True
    assert status["source_derived_target_assembly_implemented"] is True
    assert status["bounded_post_cohort_model_data_operator_implemented"] is True
    assert (
        status["source_target_assembly_roundtrip_serialization_implemented"]
        is True
    )
    assert status["complete_empty_bounded_funding_response_implemented"] is True
    assert status["real_public_exchange_info_evidence_captured"] is True
    assert status["real_authenticated_commission_evidence_captured"] is False
    assert status["real_public_funding_evidence_captured"] is True
    assert status["real_testnet_execution_calibration_completed"] is False
    assert artifact["host_preflight"]["actual_model_inference_attempted"] is False
    assert artifact["host_preflight"]["approved_risk_size_bps"] == 0
    assert artifact["host_preflight"]["request_schema_version"] == (
        ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION
    )
    latest = artifact["latest_capability_recheck"]
    assert latest["source"] == "detect_ai_capabilities"
    assert latest["model_name"] == "fino1:8b"
    assert latest["model_parameters_b"] == 8.0
    assert latest["provider_available"] is True
    assert latest["model_available"] is True
    assert latest["model_local"] is True
    assert latest["gpu_vendor"] == "amd"
    assert latest["compute_backend"] == "directml"
    assert latest["compute_device"] == "privateuseone:0"
    assert latest["free_vram_gib"] >= latest["minimum_free_vram_gib"]
    assert latest["free_system_ram_gib"] < latest["minimum_free_system_ram_gib"]
    assert latest["status"] == "blocked_capability"
    assert latest["actual_model_inference_attempted"] is False
    assert latest["host_runtime_preflight_passed"] is False


def test_round74_local_ai_evaluation_cannot_win_by_all_veto() -> None:
    artifact = _load_json(ARTIFACT_PATH)
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
        "retain the validated decision and perform exact delayed-book replay within "
        "the frozen historical latency ceiling"
    )
    assert evaluation["same_entry_latency_eligibility_rate_gate"] is None
    assert evaluation["same_entry_latency_eligibility_is_diagnostic_only"] is True
    assert (
        evaluation["same_entry_latency_eligibility_is_distinct_from_runtime_success"]
        is True
    )
    assert evaluation["latency_adjusted_delayed_entry_replay_implemented"] is True
    assert evaluation["exact_replay_required_for_every_positive_exposure"] is True
    assert evaluation["exact_replay_hash_bound_per_paired_observation"] is True
    assert (
        evaluation["exact_replay_binds_reference_quote_and_actual_entry_quote"] is True
    )
    assert evaluation["actual_deployed_capital_bps_reconciled_per_execution"] is True
    assert (
        evaluation["capital_scaled_payoff_and_mae_recomputed_from_position_bps"] is True
    )
    assert (
        evaluation[
            "requested_quote_fraction_substitution_for_executed_notional_permitted"
        ]
        is False
    )
    assert evaluation["baseline_and_ai_use_same_reference_capital_denominator"] is True
    assert evaluation["test_batch_loader_invoked_only_after_live_reservation"] is True
    assert evaluation["ai_review_provider_invoked_only_after_live_reservation"] is True
    assert (
        evaluation[
            "sealed_review_provider_receives_hash_bound_target_free_inference_only"
        ]
        is True
    )
    assert evaluation["sealed_review_provider_concrete_adapter_implemented"] is True
    assert (
        evaluation["exact_replay_provider_invoked_only_after_live_reservation"] is True
    )
    assert (
        evaluation["sealed_replay_provider_read_only_store_adapter_implemented"] is True
    )
    assert (
        evaluation["sealed_replay_provider_requires_exact_test_run_assembly_panel"]
        is True
    )
    assert (
        evaluation["sealed_replay_provider_restores_global_instruction_order"] is True
    )
    assert (
        evaluation["replay_evidence_reconciled_to_post_reservation_instruction_panel"]
        is True
    )
    assert (
        evaluation[
            "provider_failure_after_reservation_permanently_consumes_test_access"
        ]
        is True
    )
    assert evaluation["cryptographic_or_os_enforced_operator_blinding_claim"] is False
    assert evaluation["baseline_payoff_scaling_without_book_rewalk_permitted"] is False
    assert (
        evaluation["delayed_entry_exit_path_risk_and_adverse_selection_recomputed"]
        is True
    )
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
    assert (
        evaluation["action_threshold_and_objective_use_equal_capture_run_weights"]
        is True
    )
    assert evaluation["pooled_total_trade_bps_is_selection_objective"] is False
    assert evaluation["historical_queue_delay_included_in_same_entry_eligibility"] is True
    assert evaluation["queue_model_is_separate_for_each_alternative_candidate_model"] is True
    assert evaluation["queue_wait_may_be_omitted_from_reported_ai_latency"] is False
    assert evaluation["sealed_profitability_uses_three_configuration_familywise_bound"] is True
    assert evaluation["paired_ai_uplift_uses_two_model_familywise_bound"] is True
    assert evaluation["unadjusted_95_percent_lower_bound_is_diagnostic_only"] is True
    assert evaluation["same_target_cost_and_timing_claim_binds_target_engine_source"] is True
    assert evaluation["funding_schedule_or_source_evidence_may_be_omitted"] is False
    assert (
        evaluation[
            "symbol_specific_latency_fee_slippage_and_funding_evidence_may_be_omitted"
        ]
        is False
    )
    assert evaluation["paired_paths_share_exact_implementation_shortfall_decomposition"] is True
    assert evaluation["paired_target_eligibility_uses_actual_exit_funding_recheck"] is True
    assert evaluation["paired_target_eligibility_requires_verified_funding_coverage"] is True
    assert (
        evaluation[
            "paired_target_eligibility_uses_identical_funding_uncertainty_intervals"
        ]
        is True
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
