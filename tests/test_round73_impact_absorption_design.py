from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-073-impact-absorption-design.json"
)
BASE_CAPTURE_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-073-capture-contract.json"
)
CAPTURE_CONTRACT_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-capture-contract-v2.json"
)
COMPACT_CAPTURE_CONTRACT_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-capture-contract-v3.json"
)
EVENT_TIME_CAPTURE_CONTRACT_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-capture-contract-v4.json"
)
STORAGE_PROFILE_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-storage-profile-2026-07-22.json"
)
V3_PROBE_FAILURE_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-v3-probe-failure-2026-07-22.json"
)
V4_PROBE_EVIDENCE_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-v4-probe-evidence-2026-07-22.json"
)
V4_QUALIFICATION_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-v4-capture-qualification-2026-07-22.json"
)
V4_FEATURE_SOURCE_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-v4-feature-source-diagnostic-2026-07-22.json"
)
V5_CAPTURE_CONTRACT_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-capture-contract-v5.json"
)
V5_PROBE_FAILURE_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-v5-probe-failure-2026-07-22.json"
)
V5_STORAGE_FAILURE_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-v5-storage-probe-failure-2026-07-22.json"
)
V6_CAPTURE_CONTRACT_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-capture-contract-v6.json"
)
V6_TELEMETRY_FAILURE_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-v6-telemetry-failure-2026-07-22.json"
)
V7_CAPTURE_CONTRACT_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-capture-contract-v7.json"
)
V7_TELEMETRY_FAILURE_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-v7-telemetry-failure-2026-07-22.json"
)
V8_CAPTURE_CONTRACT_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-capture-contract-v8.json"
)
V8_TELEMETRY_SUCCESS_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-v8-telemetry-success-2026-07-22.json"
)
V8_CAPTURE_GATE_SUCCESS_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-v8-capture-gate-success-2026-07-22.json"
)
V8_QUALIFICATION_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-v8-capture-qualification-2026-07-22.json"
)
SEGMENTED_CORPUS_CONTRACT_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-segmented-corpus-contract-v1.json"
)
FIRST_CORPUS_MANIFEST_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-first-corpus-manifest-2026-07-22.json"
)
CORRECTION_EVIDENCE_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-feed-contract-correction-evidence-2026-07-22.json"
)
QUALIFICATION_EVIDENCE_PATH = BASE_CAPTURE_CONTRACT_PATH.with_name(
    "round-073-capture-qualification-2026-07-22.json"
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def test_round73_design_is_sealed_and_fail_closed() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    assert design["round"] == 73
    assert design["schema_version"] == "round-073-impact-absorption-design-v2"
    assert design["revision"]["modeling_capture_observed_before_revision"] is False
    assert design["source_contract"]["symbols"] == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]
    assert design["source_contract"]["market_calendar"].startswith("continuous")
    assert design["source_contract"]["listed_etf_or_equity_close_feature"] is False
    assert design["order_book_integrity_contract"]["sequence_gap_policy"].startswith(
        "invalidate"
    )
    assert design["order_book_integrity_contract"]["queue_overflow_policy"].startswith(
        "invalidate"
    )
    assert design["depth_change_semantics"]["exact_cancellation_observable"] is False
    assert (
        design["depth_change_semantics"]["unmatched_removal_is_cancellation_label"]
        is False
    )
    assert design["model_contract"]["temporal_neural_challenger_permitted"] is False
    assert design["model_contract"]["reinforcement_learning_permitted"] is False
    assert design["model_contract"]["ai_veto_permitted"] is False
    assert design["economic_gate_after_predictive_pass"]["unlevered_only"] is True
    assert design["evaluation_contract"]["minimum_symbols_for_portfolio_research"] == 2
    assert design["governance"]["profitability_claim_permitted"] is False
    assert design["governance"]["trading_authority_permitted"] is False


def test_round73_capture_contract_closes_storage_and_calendar_ambiguity() -> None:
    base_contract = json.loads(BASE_CAPTURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    base_claimed = base_contract.pop("capture_contract_sha256")
    assert base_claimed == _canonical_sha256(base_contract)

    contract = json.loads(CAPTURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    claimed = contract.pop("capture_contract_sha256")

    assert claimed == _canonical_sha256(contract)
    assert contract["schema_version"] == "round-073-prospective-capture-contract-v2"
    assert contract["parent_design_sha256"] == (
        "84b5e6c942d03ebd97b7e120951ed576e3fd8161d65755734c359b7261d6b1fe"
    )
    assert contract["inheritance"]["base_contract_sha256"] == base_claimed
    assert contract["scope"]["market_calendar"].startswith("continuous")
    assert "never a crypto market close" in contract["scope"]["utc_day_semantics"]
    assert base_contract["wire_evidence"]["credentials_permitted"] is False
    assert base_contract["writer"]["writer_count"] == 1
    assert base_contract["writer"]["queue_capacity_messages"] == 65_536
    assert base_contract["writer"]["hard_compressed_payload_cap_required"] is True
    assert (
        base_contract["raw_to_typed_link"][
            "typed_event_without_raw_reference_permitted"
        ]
        is False
    )
    assert base_contract["depth_state"]["stored_levels_per_side"] == 20
    assert (
        base_contract["depth_state"]["state_across_integrity_segment_permitted"]
        is False
    )
    assert base_contract["depth_state"]["exact_cancellation_label_permitted"] is False
    assert (
        contract["combined_stream_identity"]["wrapper_must_match_exact_subscription"]
        is True
    )
    assert (
        contract["stream_specific_fields"]["forceOrder"]["stream_type_path"]
        == "data.o.st"
    )
    assert (
        contract["rejected_wire_evidence"]["parse_before_persistence_permitted"]
        is False
    )
    assert contract["qualification"]["failure_authorizes_modeling"] is False


def test_round73_compact_storage_contract_is_hash_bound_and_lossless() -> None:
    profile = json.loads(STORAGE_PROFILE_PATH.read_text(encoding="utf-8"))
    profile_claimed = profile.pop("artifact_sha256")
    assert profile_claimed == _canonical_sha256(profile)
    assert profile["read_only_measurement"] is True
    assert profile["synthetic_market_data_used"] is False
    assert profile["decision"]["start_long_v2_capture"] is False
    assert profile["decision"]["drop_exact_wire_evidence"] is False

    contract = json.loads(COMPACT_CAPTURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    claimed = contract.pop("capture_contract_sha256")
    assert claimed == _canonical_sha256(contract)
    assert contract["measurement_evidence"]["artifact_sha256"] == profile_claimed
    assert contract["inheritance"]["historical_v1_or_v2_evidence_rewritten"] is False
    unchanged = contract["unchanged_logical_evidence"]
    assert unchanged["exact_utf8_wire_receipts"] is True
    assert unchanged["per_message_raw_payload_sha256"] is True
    assert unchanged["per_message_typed_event_sha256"] is True
    link = contract["compact_event_link"]
    assert link["message_id_per_row_stored"] is False
    assert link["primary_key_index"] is False
    assert contract["typed_storage"]["logical_columns_equal_v2"] is True
    assert contract["typed_storage"]["primary_key_indexes"] is False
    isolation = contract["version_isolation"]
    assert isolation["cross_version_frame_pooling_permitted"] is False
    assert isolation["v3_long_capture_before_v3_qualification_permitted"] is False


def test_round73_v4_contract_restores_causal_event_time_after_failed_probe() -> None:
    failure = json.loads(V3_PROBE_FAILURE_PATH.read_text(encoding="utf-8"))
    failure_claimed = failure.pop("artifact_sha256")
    assert failure_claimed == _canonical_sha256(failure)
    assert failure["fresh_process_read_only_replay"]["passed"] is True
    assert failure["post_capture_failure"]["stored_capture_report_present"] is False
    assert failure["decision"]["v3_run_qualifies_capture"] is False
    assert failure["decision"]["v3_run_authorizes_features_or_models"] is False

    contract = json.loads(
        EVENT_TIME_CAPTURE_CONTRACT_PATH.read_text(encoding="utf-8")
    )
    claimed = contract.pop("capture_contract_sha256")
    assert claimed == _canonical_sha256(contract)
    assert contract["failure_evidence"]["artifact_sha256"] == failure_claimed
    link = contract["event_link_v4"]
    assert [column[0] for column in link["columns"]][-2:] == [
        "event_time_ms",
        "typed_event_sha256",
    ]
    assert link["event_time_is_availability_clock"] is False
    terminal = contract["terminal_report_contract"]
    assert terminal["post_capture_materialization_exception_is_startup_error"] is False
    assert terminal["missing_terminal_report_authorizes_qualification"] is False
    assert contract["version_isolation"]["failed_v3_probe_reclassified_as_v4"] is False


def test_round73_v4_probe_authorizes_only_one_hour_requalification() -> None:
    evidence = json.loads(V4_PROBE_EVIDENCE_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")
    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert evidence["smoke_probe"]["fresh_process_audit_passed"] is True
    sustained = evidence["sustained_probe"]
    assert sustained["fresh_process_audit_passed"] is True
    assert sustained["database_size_cap_reached"] is False
    assert sustained["negative_corrected_latency_fraction"] == 0.0
    authorization = evidence["authorization"]
    assert authorization["v4_one_hour_qualification_attempt"] is True
    assert authorization["v4_long_capture"] is False
    assert authorization["round_073_feature_construction"] is False
    assert authorization["round_073_model_evaluation"] is False
    assert authorization["live_trading_authority"] is False


def test_round73_v4_qualification_separates_feed_and_storage_decisions() -> None:
    evidence = json.loads(V4_QUALIFICATION_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")
    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert evidence["attempt_evidence_combined"] is False
    assert evidence["run"]["qualification_passed"] is True
    assert evidence["independent_replay_audit"]["passed"] is True
    assert evidence["gate_reconstruction"]["all_three_symbols_passed"] is True
    storage = evidence["storage_efficiency_observation"]
    assert storage["decision"] == "failed_for_long_capture"
    assert storage["mechanism_status"] == "candidate_not_proven"
    assert storage["physical_growth_bytes"] > 0
    semantics = evidence["market_session_semantics"]
    assert semantics["binance_spot_and_perpetual_formal_daily_close"] is False
    assert semantics["listed_etf_context_included"] is False
    authorization = evidence["authorization"]
    assert authorization["round_073_feature_construction"] is True
    assert authorization["v4_long_capture"] is False
    assert authorization["round_073_model_evaluation"] is False
    assert authorization["profitability_claim"] is False


def test_round73_v4_feature_source_replay_is_hash_bound_and_nonfinancial() -> None:
    evidence = json.loads(V4_FEATURE_SOURCE_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")
    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert evidence["input"]["depth_update_count"] == 104570
    assert evidence["input"]["level_change_count"] == 7432729
    reconciliation = evidence["reconciliation"]
    assert reconciliation["mismatch_count"] == 0
    assert reconciliation["nonfinite_count"] == 0
    assert reconciliation["reconstructed_top_20_states_match_typed_rows"] is True
    assert "not executions" in evidence["feature_semantics"][
        "gross_quote_flow_warning"
    ]
    authority = evidence["authority"]
    assert authority["depth_band_primitives_reconstructed"] is True
    assert authority["all_grid_anchor_features_constructed"] is False
    assert authority["model_evaluated"] is False
    assert authority["profitability_claim"] is False


def test_round73_v5_contract_separates_database_audit_from_wire_replay() -> None:
    contract = json.loads(V5_CAPTURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    claimed = contract.pop("capture_contract_sha256")

    assert claimed == _canonical_sha256(contract)
    assert contract["frozen_before_first_v5_capture"] is True
    frame = contract["frame_contract_v5"]
    assert frame["maximum_messages_per_frame"] == 16_384
    assert frame["flush_message_count"] == 16_384
    assert frame["maximum_uncompressed_frame_bytes"] == 64 * 1024 * 1024
    assert frame["flush_uncompressed_bytes"] == 32 * 1024 * 1024
    assert frame["flush_interval_milliseconds"] == 4_000
    assert frame["queue_capacity_messages"] == 65_536

    bands = contract["depth_band_flow_v5"]
    assert bands["database_audit_binds_stored_band_row_to_typed_event_hash"] is True
    assert (
        bands["database_audit_reconstructs_band_row_from_exact_wire_and_snapshot"]
        is False
    )
    assert bands["independent_exact_wire_feature_source_replay_required"] is True
    assert bands["independent_replay_reconciles_each_stored_band_row"] is True
    assert (
        bands["required_feature_source_diagnostic_schema"]
        == "round-073-feature-source-diagnostic-v2"
    )

    assert contract["duckdb_policy"]["changed_by_v5"] is False
    assert contract["v5_probe_gate"][
        "independent_exact_wire_feature_source_replay_required"
    ] is True
    authorization = contract["authorization"]
    assert authorization["v5_three_minute_probe"] is True
    assert authorization["v5_one_hour_qualification"] is False
    assert authorization["v5_long_capture"] is False
    assert authorization["round_073_model_evaluation"] is False
    assert authorization["profitability_claim"] is False
    assert authorization["live_trading_authority"] is False


def test_round73_v5_failed_probe_is_preserved_without_authority() -> None:
    evidence = json.loads(V5_PROBE_FAILURE_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert evidence["attempt_evidence_combined"] is False
    run = evidence["run"]
    assert run["status"] == "failed"
    assert run["writer_frame_count"] == 0
    assert run["writer_message_count"] == 0
    assert evidence["root_cause"]["frozen_v5_frame_message_limit"] == 16_384
    assert evidence["root_cause"]["encoder_message_limit_before_remediation"] == 1_024
    assert evidence["root_cause"]["frame_format_changed"] is False
    authorization = evidence["authorization"]
    assert authorization["v5_probe_passed"] is False
    assert authorization["v5_one_hour_qualification"] is False
    assert authorization["round_073_model_evaluation"] is False
    assert authorization["profitability_claim"] is False
    assert authorization["live_trading_authority"] is False


def test_round73_v5_storage_failure_rejects_unsupported_checkpoint_change() -> None:
    evidence = json.loads(V5_STORAGE_FAILURE_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert evidence["run"]["qualification_passed"] is False
    assert evidence["fresh_process_read_only_audit"]["passed"] is True
    storage = evidence["storage_observation"]
    assert storage["database_physical_growth_bytes"] == 46_399_488
    assert storage["process_io_write_bytes_per_message"] > 4_096
    assert storage["storage_efficiency_passed"] is False
    benchmark = evidence["bounded_full_path_checkpoint_benchmark"]
    assert benchmark["synthetic_market_data_used"] is False
    assert benchmark["message_count"] == 44_506
    assert benchmark["signatures_equal"] is True
    assert benchmark["candidate_to_default_process_write_ratio"] > 0.99
    decision = evidence["decision"]
    assert decision["root_cause_proven"] is False
    assert decision["checkpoint_change_supported"] is False
    assert decision["feature_source_replay_run"] is False
    authorization = evidence["authorization"]
    assert authorization["v6_telemetry_probe"] is True
    assert authorization["v5_one_hour_qualification"] is False
    assert authorization["round_073_model_evaluation"] is False
    assert authorization["profitability_claim"] is False
    assert authorization["live_trading_authority"] is False


def test_round73_v6_contract_separates_capture_terminal_and_physical_io() -> None:
    contract = json.loads(V6_CAPTURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    claimed = contract.pop("capture_contract_sha256")

    assert claimed == _canonical_sha256(contract)
    assert contract["frozen_before_first_v6_capture"] is True
    assert contract["inheritance"]["wire_frame_format_changed"] is False
    assert contract["inheritance"]["typed_event_hash_changed"] is False
    calendar = contract["market_and_calendar_scope"]
    assert calendar["formal_daily_close"] is False
    assert "actual venue calendar" in calendar["listed_etf_or_security_semantics"]
    capture_io = contract["capture_phase_telemetry"]
    assert capture_io["endpoint_is_sealed_once"] is True
    assert capture_io["physical_ssd_or_nand_wear_claim_permitted"] is False
    assert contract["terminal_phase_telemetry"]["qualification_metric"] is False
    assert contract["duckdb_policy"]["changed_by_v6"] is False
    gate = contract["capture_gate"]
    assert gate["minimum_stream_seconds"] == 180
    assert gate["maximum_capture_phase_process_write_bytes_per_message"] == 4_096
    assert gate["maximum_database_physical_growth_bytes_per_message"] == 1_024
    assert gate["fresh_process_read_only_audit_required"] is True
    assert gate["independent_exact_wire_feature_source_replay_required"] is True
    authorization = contract["authorization"]
    assert authorization["v6_thirty_second_telemetry_diagnostic"] is True
    assert authorization["v6_180_second_capture_gate_attempt"] is False
    assert authorization["v6_one_hour_qualification"] is False
    assert authorization["round_073_model_evaluation"] is False
    assert authorization["profitability_claim"] is False
    assert authorization["live_trading_authority"] is False


def test_round73_v6_telemetry_rejects_capture_phase_write_amplification() -> None:
    evidence = json.loads(V6_TELEMETRY_FAILURE_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert evidence["run"]["capture_gate_passed"] is False
    assert evidence["fresh_process_read_only_audit"]["passed"] is True
    capture = evidence["capture_phase"]
    assert capture["write_bytes_per_message"] > 4_096
    assert capture["database_physical_growth_bytes"] == -262_144
    assert capture["storage_efficiency_passed"] is False
    assert evidence["terminal_phase"]["qualification_metric"] is False
    analysis = evidence["critical_analysis"]
    assert analysis["terminal_io_caused_v5_failure"] is False
    assert analysis["root_cause_proven"] is False
    decision = evidence["decision"]
    assert decision["v6_180_second_capture_gate_authorized"] is False
    assert decision["v7_512MiB_wal_telemetry_diagnostic_authorized"] is True
    assert evidence["authorization"]["round_073_model_evaluation"] is False


def test_round73_v7_contract_changes_only_bounded_checkpoint_policy() -> None:
    failure = json.loads(V6_TELEMETRY_FAILURE_PATH.read_text(encoding="utf-8"))
    failure_claimed = failure.pop("artifact_sha256")
    contract = json.loads(V7_CAPTURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    claimed = contract.pop("capture_contract_sha256")

    assert failure_claimed == _canonical_sha256(failure)
    assert claimed == _canonical_sha256(contract)
    assert contract["frozen_before_first_v7_capture"] is True
    assert contract["failure_evidence"]["v6_telemetry_failure_artifact_sha256"] == (
        failure_claimed
    )
    inheritance = contract["inheritance"]
    assert inheritance["wire_frame_format_changed"] is False
    assert inheritance["typed_event_hash_changed"] is False
    assert inheritance["telemetry_or_gate_threshold_changed"] is False
    policy = contract["duckdb_policy_v7"]
    assert policy["checkpoint_threshold"] == "512MiB"
    assert policy["auto_checkpoint_skip_wal_threshold_bytes"] == 512 * 1024 * 1024
    assert policy["maximum_uncommitted_wall_interval_seconds"] == 4
    gate = contract["unchanged_capture_gate"]
    assert gate["maximum_capture_phase_process_write_bytes_per_message"] == 4_096
    assert gate["maximum_database_physical_growth_bytes_per_message"] == 1_024
    calendar = contract["market_and_calendar_scope"]
    assert calendar["crypto_formal_daily_close"] is False
    assert calendar["listed_products_use_actual_venue_calendars"] is True
    authorization = contract["authorization"]
    assert authorization["v7_thirty_second_telemetry_diagnostic"] is True
    assert authorization["v7_180_second_capture_gate_attempt"] is False
    assert authorization["v7_one_hour_qualification"] is False
    assert authorization["round_073_model_evaluation"] is False
    assert authorization["profitability_claim"] is False
    assert authorization["live_trading_authority"] is False


def test_round73_v7_telemetry_rejects_checkpoint_candidate() -> None:
    evidence = json.loads(V7_TELEMETRY_FAILURE_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert evidence["attempt_evidence_combined"] is False
    capture = evidence["capture_phase"]
    assert capture["checkpoint_threshold"] == "512.0 MiB"
    assert capture["auto_checkpoint_skip_wal_threshold_bytes"] == 512 * 1024 * 1024
    assert capture["write_bytes_per_message"] > 4_096
    assert capture["storage_efficiency_passed"] is False
    assert evidence["fresh_process_read_only_audit"]["passed"] is True
    analysis = evidence["critical_analysis"]
    assert analysis["v7_checkpoint_candidate_passed"] is False
    assert analysis["v7_checkpoint_candidate_promoted"] is False
    assert analysis["root_cause_proven"] is False
    decision = evidence["decision"]
    assert decision["v7_180_second_capture_gate_authorized"] is False
    assert decision["v8_isolated_table_telemetry_diagnostic_authorized"] is True
    assert evidence["authorization"]["round_073_model_evaluation"] is False


def test_round73_v8_contract_isolates_tables_and_reverts_checkpoint_policy() -> None:
    failure = json.loads(V7_TELEMETRY_FAILURE_PATH.read_text(encoding="utf-8"))
    failure_claimed = failure.pop("artifact_sha256")
    contract = json.loads(V8_CAPTURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    claimed = contract.pop("capture_contract_sha256")

    assert failure_claimed == _canonical_sha256(failure)
    assert claimed == _canonical_sha256(contract)
    assert contract["frozen_before_first_v8_capture"] is True
    assert contract["failure_evidence"][
        "v7_telemetry_failure_artifact_sha256"
    ] == failure_claimed
    inheritance = contract["inheritance"]
    assert inheritance["wire_frame_format_changed"] is False
    assert inheritance["event_or_l2_column_shape_changed"] is False
    assert inheritance["typed_event_hash_changed"] is False
    assert inheritance["historical_rows_rewritten"] is False
    storage = contract["storage_schema_v8"]
    assert storage["single_database_file_required"] is True
    table_names = {
        value
        for key, value in storage.items()
        if key.endswith("_table")
    }
    assert len(table_names) == 11
    assert all(name.endswith("_v8") for name in table_names)
    assert storage["v7_or_earlier_rows_migrated_or_reclassified"] is False
    policy = contract["duckdb_policy_v8"]
    assert policy["checkpoint_threshold"] == "16MiB"
    assert policy["auto_checkpoint_skip_wal_threshold_bytes"] == 100_000
    assert policy["v7_candidate_promoted"] is False
    gate = contract["unchanged_capture_contract"]
    assert gate["maximum_capture_phase_process_write_bytes_per_message"] == 4_096
    assert gate["maximum_database_physical_growth_bytes_per_message"] == 1_024
    calendar = contract["market_and_calendar_scope"]
    assert calendar["crypto_formal_daily_close"] is False
    assert calendar["listed_products_use_actual_venue_calendars"] is True
    assert calendar["listed_product_close_creates_crypto_close"] is False
    authorization = contract["authorization"]
    assert authorization["v8_thirty_second_telemetry_diagnostic"] is True
    assert authorization["v8_180_second_capture_gate_attempt"] is False
    assert authorization["v8_one_hour_qualification"] is False
    assert authorization["round_073_model_evaluation"] is False
    assert authorization["profitability_claim"] is False
    assert authorization["live_trading_authority"] is False


def test_round73_v8_telemetry_passes_only_the_frozen_diagnostic() -> None:
    evidence = json.loads(V8_TELEMETRY_SUCCESS_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert evidence["attempt_evidence_combined"] is False
    run = evidence["run"]
    assert run["capture_gate_passed"] is False
    assert run["qualification_passed"] is False
    capture = evidence["capture_phase"]
    assert capture["write_bytes_per_message"] <= 4_096
    assert capture["database_physical_growth_bytes_per_message"] <= 1_024
    assert capture["all_metric_thresholds_passed"] is True
    interpretation = evidence["report_interpretation"]
    assert interpretation["stored_storage_efficiency_passed"] is False
    assert interpretation["elapsed_duration_predicate_passed"] is False
    assert interpretation["all_non_duration_storage_predicates_passed"] is True
    assert interpretation["diagnostic_contract_passed"] is True
    assert evidence["fresh_process_read_only_audit"]["passed"] is True
    comparison = evidence["comparison"]
    assert comparison["v8_to_v7_write_ratio"] < 0.07
    assert comparison["table_isolation_is_the_only_causal_explanation_proven"] is False
    analysis = evidence["critical_analysis"]
    assert analysis["v8_telemetry_diagnostic_passed"] is True
    assert analysis["v8_full_180_second_capture_gate_passed"] is False
    assert analysis["root_cause_fully_proven"] is False
    decision = evidence["decision"]
    assert decision["v8_180_second_capture_gate_authorized"] is True
    assert decision["v8_one_hour_qualification_authorized"] is False
    assert evidence["authorization"]["round_073_model_evaluation"] is False


def test_round73_v8_capture_gate_authorizes_only_one_hour() -> None:
    evidence = json.loads(V8_CAPTURE_GATE_SUCCESS_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert evidence["attempt_evidence_combined"] is False
    run = evidence["run"]
    assert run["capture_gate_passed"] is True
    assert run["qualification_passed"] is False
    assert run["reconnect_count"] == 0
    capture = evidence["capture_phase"]
    assert capture["write_bytes_per_message"] <= 4_096
    assert capture["database_physical_growth_bytes_per_message"] <= 1_024
    assert capture["storage_efficiency_passed"] is True
    audit = evidence["fresh_process_read_only_audit"]
    assert audit["passed"] is True
    assert audit["message_count"] == run["writer_message_count"]
    replay = evidence["feature_source_replay"]
    assert replay["capture_audit_passed"] is True
    assert replay["stored_depth_band_rows_reconciled"] is True
    assert replay["stored_depth_band_row_count"] == replay["depth_update_count"]
    assert replay["future_or_target_data_used"] is False
    assert replay["target_constructed"] is False
    assert replay["model_evaluated"] is False
    analysis = evidence["critical_analysis"]
    assert analysis["v8_180_second_capture_gate_passed"] is True
    assert analysis["v8_one_hour_qualification_passed"] is False
    assert analysis["profitability_evidence"] is False
    decision = evidence["decision"]
    assert decision["v8_one_hour_qualification_authorized"] is True
    assert decision["v8_long_capture_authorized"] is False
    authorization = evidence["authorization"]
    assert authorization["round_073_model_evaluation"] is False
    assert authorization["live_trading_authority"] is False


def test_round73_v8_one_hour_qualification_authorizes_bounded_pipeline() -> None:
    evidence = json.loads(V8_QUALIFICATION_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert evidence["attempt_evidence_combined"] is False
    run = evidence["run"]
    assert run["capture_gate_passed"] is True
    assert run["qualification_passed"] is True
    assert run["elapsed_seconds"] >= 3_600
    assert run["reconnect_count"] == 0
    assert sum(run["event_counts"].values()) == run["writer_message_count"]
    capture = evidence["capture_phase"]
    assert capture["write_bytes_per_message"] <= 4_096
    assert capture["headroom_fraction"] < 0.15
    assert capture["database_physical_growth_bytes_per_message"] <= 1_024
    assert capture["storage_efficiency_passed"] is True
    audit = evidence["fresh_process_read_only_audit"]
    assert audit["passed"] is True
    assert audit["message_count"] == run["writer_message_count"]
    replay = evidence["feature_source_replay"]
    assert replay["stored_depth_band_rows_reconciled"] is True
    assert replay["stored_depth_band_row_count"] == replay["depth_update_count"]
    assert replay["future_or_target_data_used"] is False
    assert replay["model_evaluated"] is False
    analysis = evidence["critical_analysis"]
    assert analysis["v8_one_hour_qualification_passed"] is True
    assert analysis["storage_headroom_is_large"] is False
    assert analysis["profitability_evidence"] is False
    decision = evidence["decision"]
    assert decision["bounded_segmented_corpus_pipeline_authorized"] is True
    assert decision["unbounded_single_run_capture_authorized"] is False
    assert decision["seven_day_capture_authorized_before_rotation_design"] is False
    authorization = evidence["authorization"]
    assert authorization["round_073_feature_construction"] is True
    assert authorization["round_073_model_evaluation"] is False
    assert authorization["live_trading_authority"] is False


def test_round73_segmented_corpus_contract_is_hash_bound_and_fail_closed() -> None:
    contract = json.loads(SEGMENTED_CORPUS_CONTRACT_PATH.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == _canonical_sha256(contract)
    assert contract["frozen_before_first_manifest_write"] is True
    assert contract["qualification_evidence_sha256"] == (
        "5663eea23d71e9a06c4f2d03e6a70ff82d23439942f5c251f93006f4dac9b9fd"
    )
    admission = contract["segment_admission"]
    assert admission["capture_schema_version"] == "round-073-prospective-evidence-v8"
    assert admission["minimum_elapsed_seconds"] == 3_600
    assert admission["independent_feature_source_replay_required"] is True
    assert admission["attempt_evidence_combined"] is False
    assert admission["historical_v1_through_v7_run_admitted"] is False
    resources = contract["resource_admission"]
    assert resources["maximum_process_io_write_bytes_per_message"] == 4_096
    assert resources["maximum_database_physical_growth_bytes_per_message"] == 1_024
    assert resources["maximum_queue_utilization"] == 0.8
    storage = contract["manifest_storage"]
    assert storage["run_manifest_table"] == "impact_corpus_run_manifest_v1"
    assert storage["existing_manifest_mismatch_policy"] == "fail without overwrite"
    assert storage["duplicate_raw_payload_storage_permitted"] is False
    rotation = contract["rotation_policy"]
    assert rotation["segment_duration_seconds"] == 3_600
    assert rotation["maximum_reconnects_per_segment"] == 0
    assert rotation["unbounded_loop_permitted"] is False
    day = contract["day_contract"]
    assert day["crypto_formal_daily_close"] is False
    assert day["minimum_complete_hours"] == 23
    assert day["listed_products_use_actual_venue_calendars"] is True
    assert day["listed_product_close_creates_crypto_close"] is False
    modeling = contract["modeling_gate"]
    assert modeling["minimum_complete_days_for_viability"] == 7
    assert modeling["model_evaluation_authorized"] is False
    assert modeling["profitability_claim"] is False
    assert modeling["live_trading_authority"] is False


def test_round73_first_corpus_manifest_is_real_hash_bound_and_non_predictive() -> None:
    evidence = json.loads(FIRST_CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert evidence["source"]["synthetic_market_data_used"] is False
    manifest = evidence["manifest"]
    assert manifest["run_id"] == "f3e92ba29e1e4d3188c3f309f5c160a2"
    assert manifest["message_count"] == 1_294_128
    assert manifest["coverage_duration_ns"] >= 3_600_000_000_000
    replay = evidence["independent_replay"]
    assert replay["capture_audit_passed"] is True
    assert replay["stored_depth_band_rows_reconciled"] is True
    assert replay["stored_depth_band_row_count"] == replay["depth_update_count"]
    assert replay["future_or_target_data_used"] is False
    assert evidence["post_write_manifest_audit"]["passed"] is True
    storage = evidence["storage_observation"]
    assert storage["physical_growth_bytes"] == 0
    assert storage["ssd_or_nand_wear_inferred"] is False
    day = evidence["utc_partition_diagnostic"]
    assert day["eligible"] is False
    assert day["crypto_formal_daily_close"] is False
    assert day["listed_products_use_actual_venue_calendars"] is True
    analysis = evidence["critical_analysis"]
    assert analysis["complete_day_count"] == 0
    assert analysis["predictive_edge_evidence"] is False
    assert analysis["profitability_evidence"] is False
    authorization = evidence["authorization"]
    assert authorization["round_073_model_evaluation"] is False
    assert authorization["live_trading_authority"] is False


def test_round73_feed_contract_correction_evidence_is_hash_bound() -> None:
    evidence = json.loads(CORRECTION_EVIDENCE_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["credentials_used"] is False
    assert evidence["orders_submitted"] is False
    assert len(evidence["failed_qualification_attempts"]) == 2
    probe = evidence["live_force_order_probe"]
    assert probe["top_level_data_st_present"] is False
    assert probe["observed_paths"]["stream_type"] == "data.o.st"
    assert probe["raw_sha256"] == (
        "972aacab0e6cbd8d026505e9ab4d7721a9a6275346ddd845fb32fa2186b8d805"
    )


def test_round73_capture_qualification_is_hash_bound_and_narrowly_authorized() -> None:
    evidence = json.loads(QUALIFICATION_EVIDENCE_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("artifact_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["schema_version"] == "round-073-capture-qualification-evidence-v2"
    assert evidence["supersedes_artifact_sha256"] == (
        "c815bb7021097230dd1abccbf47af63154c2ca39fefc20a576e0af93f9f6d510"
    )
    assert evidence["governance_correction"]["evidence_values_changed"] is False
    assert evidence["capture"]["qualification_passed"] is True
    assert evidence["attempt_count"] == 1
    assert evidence["attempt_evidence_combined"] is False
    assert evidence["reconnect_count"] == 0
    assert evidence["independent_replay_audit"]["passed"] is True
    assert evidence["gate_reconstruction"]["all_three_symbols_passed"] is True
    assert all(
        symbol["segment_status"] == "valid"
        and symbol["invalid_events"] == 0
        and symbol["sequence_gaps"] == 0
        and symbol["crossed_books"] == 0
        for symbol in evidence["symbols"].values()
    )
    authorization = evidence["authorization"]
    assert authorization["round_073_feature_construction"] is True
    assert authorization["one_hour_feature_pipeline_diagnostic"] is True
    assert authorization["round_073_model_evaluation"] is False
    assert authorization["round_073_viability_model_evaluation"] is False
    assert authorization["round_073_promotion_model_evaluation"] is False
    assert authorization["profitability_claim"] is False
    assert authorization["predictive_edge_claim"] is False
    assert authorization["paper_trading_authority"] is False
    assert authorization["testnet_trading_authority"] is False
    assert authorization["live_trading_authority"] is False
