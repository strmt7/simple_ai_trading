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
