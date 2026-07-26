from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.impact_absorption import ROUND74_CAPTURE_DESIGN_SHA256
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_V10_CONTRACT_SHA256,
    IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION,
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
)


RESEARCH = Path("docs/model-research/action-value")


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


def _linear_quantile(ordered: list[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])


def test_round73_v9_campaign_invalidation_is_hash_bound_and_pre_model() -> None:
    artifact = json.loads(
        (RESEARCH / "round-073-v9-corpus-invalidation-2026-07-25.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    assert artifact["campaign"]["qualified_and_indexed_hours_before_invalidation"] == 18
    assert artifact["campaign"]["completed_but_resource_rejected_hours"] == 2
    assert artifact["campaign"]["target_rows_opened"] is False
    assert len(artifact["accepted_completed_runs"]) == 18
    assert len(artifact["completed_resource_rejections"]) == 2
    assert all(
        row["audit_passed"] is True and row["capture_error"] == ""
        for row in artifact["completed_resource_rejections"]
    )
    decision = artifact["decision"]
    assert decision["all_18_indexed_v9_hours_eligible_for_modeling"] is False
    assert decision["resume_v9_rotation_permitted"] is False
    assert not any(artifact["authority"].values())


def test_round74_design_and_v10_contract_are_hash_bound_and_nonselective() -> None:
    design = json.loads(
        (RESEARCH / "round-074-capture-recovery-design-v1.json").read_text(
            encoding="utf-8"
        )
    )
    claimed_design = design.pop("design_sha256")
    contract = json.loads(
        (RESEARCH / "round-074-capture-contract-v10.json").read_text(encoding="utf-8")
    )
    claimed_contract = contract.pop("capture_contract_sha256")

    assert claimed_design == _canonical_sha256(design) == ROUND74_CAPTURE_DESIGN_SHA256
    assert (
        claimed_contract
        == _canonical_sha256(contract)
        == IMPACT_CAPTURE_V10_CONTRACT_SHA256
    )
    storage = contract["storage_schema_v10"]
    assert storage["run_schema"] == IMPACT_CAPTURE_V10_SCHEMA_VERSION
    assert storage["report_schema"] == IMPACT_CAPTURE_V10_REPORT_SCHEMA_VERSION
    resource = contract["host_resource_safety_v10"]
    assert resource["message_count_used_in_resource_verdict"] is False
    assert resource["bytes_per_message_retained_as_telemetry_only"] is True
    assert (
        resource["resource_failure_campaign_policy"]
        == "halt and require review; never skip the market interval and continue"
    )
    heartbeat = contract["websocket_heartbeat_v10"]
    assert heartbeat["automatic_pong_handling_required"] is True
    assert heartbeat["client_originated_keepalive_ping_interval"] is None
    calendar = contract["market_and_calendar_scope"]
    assert calendar["crypto_formal_daily_close"] is False
    assert calendar["listed_product_close_creates_crypto_close"] is False
    assert (
        calendar["listed_product_calendar_may_grant_crypto_execution_authority"]
        is False
    )
    assert contract["authorization"]["round_074_model_training_or_evaluation"] is False


def test_round74_live_probe_preflight_is_hash_bound_and_narrow() -> None:
    artifact = json.loads(
        (
            RESEARCH / "round-074-v10-live-probe-preflight-2026-07-25.json"
        ).read_text(encoding="utf-8")
    )
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    assert artifact["implementation_git_commit"] == (
        "8457134cadcb859648375c35c8072e54e5ab99d2"
    )
    assert artifact["verification"]["focused_pytest_tests_passed"] == 82
    assert artifact["host_preflight"]["active_python_capture_processes"] == 0
    authorization = artifact["authorization"]
    assert authorization["one_v10_live_probe"] is True
    assert authorization["maximum_duration_seconds"] == 180
    assert authorization["maximum_reconnects"] == 0
    assert authorization["v10_one_hour_qualification"] is False
    assert authorization["round_074_model_training_or_evaluation"] is False
    assert authorization["live_trading_authority"] is False


def test_round74_live_probe_evidence_is_hash_bound_and_authorizes_one_hour() -> None:
    artifact = json.loads(
        (
            RESEARCH / "round-074-v10-live-probe-success-2026-07-25.json"
        ).read_text(encoding="utf-8")
    )
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    assert artifact["execution_git_commit"] == (
        "1e398d97b3f7ca6df3be61d9a09f912a8d9ba601"
    )
    capture = artifact["capture"]
    assert capture["status"] == "completed"
    assert capture["reconnect_count"] == 0
    assert capture["message_count"] == artifact["fresh_process_audit"][
        "message_count"
    ]
    gate = artifact["gate_analysis"]
    assert gate["capture_gate_passed"] is True
    assert gate["data_qualification_passed"] is True
    assert gate["resource_safety_passed"] is True
    assert gate["qualification_passed"] is False
    assert gate["message_count_used_in_resource_verdict"] is False
    authorization = artifact["authorization"]
    assert authorization["one_v10_one_hour_qualification_attempt"] is True
    assert authorization["maximum_qualification_duration_seconds"] == 3600
    assert authorization["maximum_reconnects"] == 0
    assert authorization["v10_rotation"] is False
    assert authorization["round_074_model_training_or_evaluation"] is False
    assert authorization["live_trading_authority"] is False


def test_round74_one_hour_qualification_is_hash_bound_without_hindsight_label() -> None:
    artifact = json.loads(
        (
            RESEARCH
            / "round-074-v10-one-hour-qualification-success-2026-07-25.json"
        ).read_text(encoding="utf-8")
    )
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    assert artifact["execution_git_commit"] == (
        "c2ff589ab74589efae6dfe12d6e2a94d363f135f"
    )
    capture = artifact["capture"]
    audit = artifact["fresh_process_audit"]
    assert capture["status"] == "completed"
    assert capture["reconnect_count"] == 0
    assert capture["elapsed_seconds"] >= 3600.0
    assert capture["message_count"] == audit["message_count"] == 1_001_776
    assert capture["last_frame_sha256"] == audit["last_frame_sha256"]
    assert artifact["gate_analysis"]["qualification_passed"] is True
    activity = artifact["market_activity_interpretation"]
    assert activity["quiet_or_active_classifier_frozen_before_capture"] is False
    assert activity["activity_label"] == "unclassified"
    assert activity["may_retroactively_satisfy_quiet_or_active_qualification"] is False
    authorization = artifact["authorization"]
    assert authorization["round_074_nonselective_period_qualification_design"] is True
    assert authorization["additional_v10_capture_before_period_contract"] is False
    assert authorization["round_074_model_training_or_evaluation"] is False
    assert authorization["live_trading_authority"] is False


def test_round74_activity_stress_contract_freezes_nonselective_thresholds() -> None:
    artifact = json.loads(
        (
            RESEARCH / "round-074-activity-stress-qualification-contract-v1.json"
        ).read_text(encoding="utf-8")
    )
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    calibration = artifact["calibration"]
    rows = calibration["rows"]
    assert len(rows) == 20
    assert calibration["rows_sha256"] == _canonical_sha256(rows)
    rates = [float(row["messages_per_second"]) for row in rows]
    assert rates == sorted(rates)
    assert calibration["quiet_maximum_messages_per_second"] == _linear_quantile(
        rates, 0.25
    )
    assert calibration["active_minimum_messages_per_second"] == _linear_quantile(
        rates, 0.75
    )
    classification = artifact["classification"]
    assert classification["classification_changes_data_qualification"] is False
    assert classification["classification_changes_resource_safety"] is False
    assert classification["classification_can_discard_a_completed_run"] is False
    assert classification["message_count_used_in_resource_verdict"] is False
    hindsight = artifact["hindsight_control"]
    assert hindsight["prior_v10_one_hour_would_meet_new_quiet_threshold"] is True
    assert hindsight["prior_v10_one_hour_may_satisfy_quiet_qualification"] is False
    authorization = artifact["authorization"]
    assert authorization["one_v10_quiet_regime_qualification_attempt"] is True
    assert authorization["v10_active_regime_qualification_attempt"] is False
    assert authorization["round_074_model_training_or_evaluation"] is False


def test_round74_quiet_qualification_is_hash_bound_and_predeclared() -> None:
    artifact = json.loads(
        (
            RESEARCH
            / "round-074-v10-quiet-regime-qualification-success-2026-07-25.json"
        ).read_text(encoding="utf-8")
    )
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    capture = artifact["capture"]
    audit = artifact["fresh_process_audit"]
    assert capture["status"] == "completed"
    assert capture["reconnect_count"] == 0
    assert capture["message_count"] == audit["message_count"] == 881_173
    assert capture["last_frame_sha256"] == audit["last_frame_sha256"]
    assert artifact["gate_analysis"]["qualification_passed"] is True
    activity = artifact["activity_classification"]
    assert activity["threshold_frozen_before_capture"] is True
    assert activity["classification"] == "quiet"
    assert activity["messages_per_second"] <= activity[
        "quiet_maximum_messages_per_second"
    ]
    assert activity["quiet_regime_qualification_passed"] is True
    assert activity["classification_grants_model_authority"] is False
    authority = artifact["authority"]
    assert authority["round_074_active_regime_preflight_design"] is True
    assert authority["v10_active_regime_qualification_attempt"] is False
    assert authority["round_074_model_training_or_evaluation"] is False


def test_round74_active_preflight_is_fixed_and_calendar_safe() -> None:
    artifact = json.loads(
        (
            RESEARCH
            / "round-074-v10-active-regime-qualification-preflight-2026-07-25.json"
        ).read_text(encoding="utf-8")
    )
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    window = artifact["fixed_execution_window"]
    assert window["earliest_start_utc"] == "2026-07-25T12:55:00Z"
    assert window["latest_start_utc"] == "2026-07-25T12:57:00Z"
    assert window["duration_seconds"] == 3600
    assert window["maximum_reconnects"] == 0
    assert window["automatic_retry_permitted"] is False
    assert window["reschedule_after_observing_market_activity_permitted"] is False
    calendar = artifact["market_calendar_semantics"]
    assert calendar["binance_crypto_trading_is_continuous"] is True
    assert calendar["crypto_has_formal_daily_close"] is False
    assert (
        calendar["listed_etf_etp_or_security_venues_have_formal_sessions_and_closes"]
        is True
    )
    assert calendar["scheduled_date_is_saturday"] is True
    assert calendar["listed_venue_weekend_closure_is_context_only"] is True
    assert calendar["listed_product_close_creates_crypto_close"] is False
    assert (
        calendar["listed_product_calendar_may_grant_crypto_execution_authority"]
        is False
    )
    classification = artifact["frozen_classification"]
    assert classification["active_minimum_messages_per_second"] == (
        735.8503256619431
    )
    assert classification["middle_or_quiet_attempt_is_retained"] is True
    assert classification["middle_or_quiet_attempt_may_be_retried"] is False
    assert classification["message_count_used_in_resource_verdict"] is False
    authorization = artifact["authorization"]
    assert authorization[
        "one_v10_active_regime_qualification_attempt_in_fixed_window"
    ] is True
    assert authorization["round_074_model_training_or_evaluation"] is False
    assert authorization["live_trading_authority"] is False
    assert authorization["profitability_or_edge_claim"] is False


def test_round74_missed_active_window_records_zero_market_observation() -> None:
    artifact = json.loads(
        (
            RESEARCH
            / "round-074-v10-active-regime-window-missed-2026-07-25.json"
        ).read_text(encoding="utf-8")
    )
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    evidence = artifact["authoritative_evidence"]
    assert evidence["impact_capture_run_rows_in_window"] == 0
    assert evidence["collector_running_at_audit"] is False
    classification = artifact["classification"]
    assert classification["capture_attempt_started"] is False
    assert classification["market_stream_observed"] is False
    assert classification["messages_per_second_observed"] is False
    assert classification["quiet_middle_or_active_label_permitted"] is False
    assert classification["outcome"] == (
        "operational_window_missed_without_market_observation"
    )
    hindsight = artifact["anti_hindsight_controls"]
    assert hindsight["observed_activity_used_to_choose_later_window"] is False
    assert hindsight["fresh_fixed_preflight_required_before_any_later_attempt"] is True
    assert artifact["authority"]["active_regime_qualified"] is False
    assert artifact["authority"]["round_074_model_training_or_evaluation"] is False


def test_round74_fresh_active_preflight_does_not_select_on_missed_window() -> None:
    artifact = json.loads(
        (
            RESEARCH
            / "round-074-v10-active-regime-qualification-preflight-2026-07-27.json"
        ).read_text(encoding="utf-8")
    )
    claimed = artifact.pop("artifact_sha256")

    assert claimed == _canonical_sha256(artifact)
    basis = artifact["fresh_attempt_basis"]
    assert basis["prior_attempt_started"] is False
    assert basis["prior_market_stream_observed"] is False
    assert basis["prior_activity_classification_observed"] is False
    assert basis["selection_on_prior_market_outcome"] is False
    window = artifact["fixed_execution_window"]
    assert window["earliest_start_utc"] == "2026-07-27T12:55:00Z"
    assert window["latest_start_utc"] == "2026-07-27T12:57:00Z"
    assert window["automatic_retry_permitted"] is False
    calendar = artifact["market_calendar_semantics"]
    assert calendar["binance_crypto_trading_is_continuous"] is True
    assert calendar["listed_venue_status_not_used_to_select_or_admit_capture"] is True
    assert calendar["listed_product_close_creates_crypto_close"] is False
    assert (
        calendar["listed_product_calendar_may_grant_crypto_execution_authority"]
        is False
    )
    classification = artifact["frozen_classification"]
    assert classification["active_minimum_messages_per_second"] == (
        735.8503256619431
    )
    assert classification["middle_or_quiet_attempt_may_be_retried"] is False
    invocation = artifact["capture_invocation"]
    assert "--schema-version" in invocation["arguments"]
    assert "v10" in invocation["arguments"]
    assert invocation["operator_progress_check_interval_maximum_seconds"] == 120
    authorization = artifact["authorization"]
    assert authorization[
        "one_v10_active_regime_qualification_attempt_in_fixed_window"
    ] is True
    assert authorization["round_074_model_training_or_evaluation"] is False
    assert authorization["live_trading_authority"] is False
