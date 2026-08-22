from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_repricing_publication import (
    publish_polymarket_repricing_report,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "polymarket"


def _canonical_artifact_bytes(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".svg"}:
        payload = payload.replace(b"\r\n", b"\n")
    return payload


def test_round_002_publication_is_internally_consistent() -> None:
    report = json.loads(
        (RESEARCH / "round-002-prospective-pipeline-evidence.json").read_text(
            encoding="utf-8"
        )
    )
    with (RESEARCH / "round-002-market-rows.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        markets = list(csv.DictReader(handle))

    assert report["round"] == 2
    assert report["status"] == "pipeline_verified_model_evaluation_blocked"
    assert len(report["recorder"]["report_sha256"]) == 64
    assert len(report["dataset"]["dataset_sha256"]) == 64
    assert len(markets) == report["recorder"]["market_snapshot_count"] == 12
    assert (
        sum(int(row["feature_rows"]) for row in markets)
        == report["dataset"]["row_count"]
    )
    for asset, evidence in report["per_asset"].items():
        asset_rows = [row for row in markets if row["asset"] == asset]
        assert len(asset_rows) == evidence["official_resolutions"] == 4
        assert (
            sum(int(row["feature_rows"]) for row in asset_rows)
            == evidence["feature_rows"]
        )
        assert (
            sum(int(row["feature_rows"]) > 0 for row in asset_rows)
            == report["dataset"]["labeled_market_counts"][asset]
        )

    manifest = {entry["path"]: entry for entry in report["artifact_integrity"]}
    assert set(report["tracked_artifacts"]) - {
        "docs/model-research/polymarket/round-002-prospective-pipeline-evidence.json"
    } == set(manifest)
    for relative_path, expected in manifest.items():
        artifact = ROOT / relative_path
        payload = _canonical_artifact_bytes(artifact)
        assert len(payload) == expected["bytes"]
        assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
    market_manifest = manifest[
        "docs/model-research/polymarket/round-002-market-rows.csv"
    ]
    assert market_manifest["row_count"] == len(markets)
    assert market_manifest["columns"] == list(markets[0])


def test_latest_publication_distinguishes_round25_capture_from_results() -> None:
    latest = RESEARCH / "latest"
    readme = (latest / "README.md").read_text(encoding="utf-8").lower()
    normalized_readme = " ".join(readme.split())
    chart = (
        (latest / "charts" / "optimization-progress.svg")
        .read_text(encoding="utf-8")
        .lower()
    )
    with (latest / "tables" / "optimization-progress.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = {row["round"]: row for row in csv.DictReader(handle)}

    assert "round 20 failed" in readme
    assert "round 25" in readme
    assert "round 25 v1 was retired" in normalized_readme
    assert "round 25 v2 is the source-correct successor" in normalized_readme
    assert "`crypto_prices_twap_thirty`" in normalized_readme
    assert "point-topic frames are rejected" in normalized_readme
    assert "structural probability" in normalized_readme
    assert "settlement hypothesis" in normalized_readme
    assert "minimum condition counts of `2000/400/400`" in normalized_readme
    assert (
        "control fitter is implemented but no control model has been fitted"
        in normalized_readme
    )
    assert (
        "lightgbm operator is implemented but no round 25 tree has been fitted"
        in normalized_readme
    )
    assert (
        "sequence materializer is implemented but no sequence corpus has been materialized"
        in normalized_readme
    )
    assert "no seed model or ensemble has been fitted" in normalized_readme
    assert "the v2 plan has no test role" in normalized_readme
    assert "ai cannot create or enlarge a trade" in normalized_readme
    assert (
        "no v2 campaign database, target, model, ai comparison, economic result"
        in normalized_readme
    )
    assert "round 14 has no execution or pnl claim" in chart
    assert "r14" in chart
    unavailable_metrics = {
        "conditions",
        "held_out_log_loss_skill",
        "held_out_brier_skill",
        "held_out_balanced_accuracy",
    }
    for round_number in ("12", "13"):
        assert all(rows[round_number][field] == "" for field in unavailable_metrics)
        assert rows[round_number]["profitability_claim"] == "False"


def test_current_status_manifest_reconstructs_every_artifact() -> None:
    manifest = json.loads(
        (RESEARCH / "latest" / "publication-integrity.json").read_text(encoding="utf-8")
    )
    claimed = manifest.pop("manifest_sha256")
    canonical = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert manifest["schema_version"] == "polymarket-current-status-publication-v1"
    assert manifest["latest_research_round"] == 29
    assert manifest["latest_evaluated_round"] == 23
    assert manifest["latest_graph_round"] == 23
    assert (
        manifest["status"] == "round29_preregistered_stage1_capture_running_no_result"
    )
    assert manifest["round21_result_available"] is False
    assert manifest["round23_result_available"] is True
    assert manifest["round24_result_available"] is False
    assert manifest["round25_result_available"] is False
    assert manifest["round25_capture_revision"] == 2
    assert manifest["round25_v1_capture_eligible"] is False
    assert manifest["round25_v2_capture_evidence_published"] is True
    assert manifest["round25_twap_native_model_design_frozen"] is True
    assert manifest["round25_twap_wire_schema_corrected"] is True
    assert manifest["round25_twap_wire_schema_correction_revision"] == 2
    assert manifest["round25_feature_contract_implemented"] is True
    assert manifest["round25_joint_feature_materialization_contract_frozen"] is True
    assert manifest["round25_joint_feature_materialization_contract_revision"] == 2
    assert manifest["round25_joint_feature_materializer_implemented"] is True
    assert manifest["round25_joint_feature_store_implemented"] is True
    assert manifest["round25_candidate_ledger_frozen"] is True
    assert manifest["round25_development_dataset_contract_implemented"] is True
    assert manifest["round25_control_fit_contract_frozen"] is True
    assert manifest["round25_control_fit_implemented"] is True
    assert manifest["round25_control_model_fitted"] is False
    assert manifest["round25_lightgbm_fit_contract_frozen"] is True
    assert manifest["round25_lightgbm_operator_implemented"] is True
    assert manifest["round25_lightgbm_model_fitted"] is False
    assert manifest["round25_sequence_materialization_contract_frozen"] is True
    assert manifest["round25_sequence_materializer_implemented"] is True
    assert manifest["round25_sequence_corpus_materialized"] is False
    assert manifest["round25_tcn_fit_contract_frozen"] is True
    assert manifest["round25_tcn_operator_implemented"] is True
    assert manifest["round25_tcn_host_directml_mechanics_verified"] is True
    assert manifest["round25_tcn_seed_model_fitted"] is False
    assert manifest["round25_tcn_ensemble_fitted"] is False
    assert manifest["round25_tcn_model_fitted"] is False
    assert manifest["round25_predictive_evaluation_contract_frozen"] is True
    assert manifest["round25_predictive_evaluator_implemented"] is True
    assert manifest["round25_prediction_panel_frozen"] is False
    assert manifest["round25_predictive_result_available"] is False
    assert manifest["round25_economic_replay_contract_frozen"] is True
    assert manifest["round25_economic_replay_contract_revision"] == 2
    assert manifest["round25_economic_replay_operator_implemented"] is True
    assert manifest["round25_economic_result_available"] is False
    assert manifest["round25_development_economic_gate_passed"] is False
    assert manifest["round25_ai_risk_contract_frozen"] is True
    assert manifest["round25_ai_operator_implemented"] is True
    assert manifest["round25_ai_host_mechanics_verified"] is True
    assert manifest["round25_ai_safety_behavior_mechanics_verified"] is True
    assert manifest["round25_ai_uplift_contract_frozen"] is True
    assert manifest["round25_ai_uplift_contract_revision"] == 2
    assert manifest["round25_ai_uplift_evaluator_implemented"] is True
    assert manifest["round25_ai_uplift_population_available"] is False
    assert manifest["round25_ai_uplift_result_available"] is False
    assert manifest["round25_ai_uplift_verified"] is False
    assert manifest["round25_fin_r1_supervisor_contract_frozen"] is True
    assert manifest["round25_fin_r1_supervisor_behavior_passed"] is False
    assert manifest["round25_fin_r1_supervisor_candidate_rejected"] is True
    assert manifest["round25_fin_r1_supervisor_uplift_evaluator_implemented"] is True
    assert manifest["round25_fin_r1_supervisor_uplift_evaluation_allowed"] is False
    assert manifest["round25_fin_r1_supervisor_called_by_execution"] is False
    assert manifest["round25_qwen3_8b_supervisor_candidate_rejected"] is True
    assert manifest["round25_slow_llm_supervisor_mechanism_rejected"] is True
    assert (
        manifest["round25_slow_llm_supervisor_additional_model_cycling_allowed"]
        is False
    )
    assert manifest["round25_sealed_test_campaign_planned"] is False
    assert manifest["round25_settlement_rule_verified"] is False
    assert manifest["round25_terminal_design_frozen"] is True
    assert manifest["round25_terminal_design_revision"] == 2
    assert manifest["round25_terminal_operator_implemented"] is True
    assert manifest["round25_terminal_receipt_audit_available"] is False
    assert manifest["round25_materialization_result_available"] is False
    assert manifest["round25_resolution_collection_contract_frozen"] is True
    assert manifest["round25_resolution_collection_contract_revision"] == 3
    assert manifest["round25_resolution_public_transport_verified"] is True
    assert manifest["round25_resolution_terminal_schema_verified"] is True
    assert manifest["round25_resolution_store_implemented"] is True
    assert manifest["round25_resolution_authority_available"] is False
    assert manifest["round25_post_capture_coordinator_contract_frozen"] is True
    assert manifest["round25_post_capture_coordinator_contract_revision"] == 4
    assert manifest["round25_post_capture_coordinator_implemented"] is True
    assert manifest["round25_post_capture_coordinator_run"] is False
    assert manifest["round25_post_capture_runner_implemented"] is True
    assert manifest["round25_post_capture_runner_waiting_mode_host_verified"] is True
    assert manifest["round25_post_capture_runner_source_database_opened"] is False
    assert manifest["round25_post_capture_runner_orders_submitted"] == 0
    assert manifest["round27_campaign_admission_gate_implemented"] is True
    assert manifest["round27_campaign_admitted"] is False
    assert manifest["round27_stage1_campaign_contract_frozen"] is True
    assert manifest["round27_stage1_capture_running"] is True
    assert manifest["round27_stage1_result_available"] is False
    assert manifest["round27_target_access_allowed"] is False
    assert manifest["round28_bbo_preregistration_frozen"] is True
    assert manifest["round28_loaded_contract_binding_corrected"] is True
    assert manifest["round28_loaded_contract_binding_revision"] == 2
    assert manifest["round28_model_result_available"] is False
    assert manifest["round28_sealed_evaluation_implementation_source_bound"] is True
    assert manifest["round28_sealed_result_available"] is False
    assert manifest["round28_ai_preregistration_frozen"] is True
    assert manifest["round28_ai_core_implementation_source_bound"] is True
    assert manifest["round28_ai_operator_implementation_source_bound"] is True
    assert manifest["round28_ai_sealed_evaluation_implementation_source_bound"] is True
    assert manifest["round28_ai_sealed_result_available"] is False
    assert manifest["round28_ai_result_available"] is False
    assert manifest["profitability_claim"] is False
    assert manifest["paper_authority"] is False
    assert manifest["live_trading_authority"] is False
    artifact_paths = [entry["path"] for entry in manifest["artifacts"]]
    assert artifact_paths == sorted(set(artifact_paths))
    assert "round-023-lead-lag-results-v1.json" in artifact_paths
    assert (
        "round-028-sealed-evaluation-implementation-amendment-v1.json" in artifact_paths
    )
    assert (
        "round-028-ai-sealed-evaluation-implementation-amendment-v1.json"
        in artifact_paths
    )
    assert "round-023-lead-lag-performance.svg" in artifact_paths
    assert "latest/public-clob-live-probe-2026-08-10.json" in artifact_paths
    assert "round-024-prospective-receipt-lead-lag-spec-v2.json" in artifact_paths
    assert "round-024-prospective-receipt-lead-lag-spec-v3.json" in artifact_paths
    assert "round-024-preregistration-publication-v2-2026-08-10.json" in artifact_paths
    assert "round-025-terminal-receipt-materialization-design-v1.json" in artifact_paths
    assert "round-025-terminal-receipt-materialization-design-v2.json" in artifact_paths
    assert "round-025-twap-wire-schema-correction-v1.json" in artifact_paths
    assert "round-025-twap-wire-schema-correction-v2.json" in artifact_paths
    assert "round-025-joint-feature-materialization-contract-v1.json" in artifact_paths
    assert "round-025-joint-feature-materialization-contract-v2.json" in artifact_paths
    assert "round-025-official-resolution-collection-contract-v1.json" in artifact_paths
    assert "round-025-official-resolution-collection-contract-v2.json" in artifact_paths
    assert "round-025-official-resolution-collection-contract-v3.json" in artifact_paths
    assert (
        "round-025-official-resolution-transport-probe-v1-2026-08-10.json"
        in artifact_paths
    )
    assert "round-025-post-capture-coordinator-contract-v1.json" in artifact_paths
    assert "round-025-post-capture-coordinator-contract-v2.json" in artifact_paths
    assert "round-025-post-capture-coordinator-contract-v3.json" in artifact_paths
    assert "round-025-post-capture-coordinator-contract-v4.json" in artifact_paths
    assert "round-025-economic-replay-contract-v1.json" in artifact_paths
    assert "round-025-economic-replay-contract-v2.json" in artifact_paths
    assert "round-025-control-fit-contract-v1.json" in artifact_paths
    assert "round-025-lightgbm-fit-contract-v1.json" in artifact_paths
    assert "round-025-predictive-evaluation-contract-v1.json" in artifact_paths
    assert "round-025-sequence-materialization-contract-v1.json" in artifact_paths
    assert "round-025-target-free-sequence-inference-contract-v1.json" in artifact_paths
    assert "round-025-model-ledger-contract-v1.json" in artifact_paths
    assert "round-025-tcn-directml-host-probe-2026-08-10.json" in artifact_paths
    assert "round-025-tcn-fit-contract-v1.json" in artifact_paths
    assert (
        "round-025-twap-native-candidate-selection-amendment-v2.json" in artifact_paths
    )
    assert "round-025-twap-native-candidate-selection-design-v1.json" in artifact_paths
    assert "round-025-twap-native-model-design-v1.json" in artifact_paths
    assert "round-025-twap-core-capture-design-v1.json" in artifact_paths
    assert "round-025-twap-core-capture-design-v2.json" in artifact_paths
    assert "round-025-twap-source-qualification-2026-08-10.json" in artifact_paths
    assert "round-025-fin-r1-regime-supervisor-contract-v1.json" in artifact_paths
    assert (
        "round-025-fin-r1-regime-supervisor-host-probe-v2-2026-08-10.json"
        in artifact_paths
    )
    assert (
        "round-025-fin-r1-regime-supervisor-rejection-v1-2026-08-10.json"
        in artifact_paths
    )
    assert (
        "round-025-fin-r1-regime-supervisor-uplift-contract-v1.json" in artifact_paths
    )
    assert (
        "round-025-qwen3-8b-regime-supervisor-host-probe-v1-2026-08-10.json"
        in artifact_paths
    )
    assert (
        "round-025-qwen3-8b-regime-supervisor-rejection-v1-2026-08-10.json"
        in artifact_paths
    )
    assert (
        "round-025-twap-core-campaign-plan-publication-2026-08-10.json"
        in artifact_paths
    )
    assert (
        "round-025-twap-core-campaign-plan-publication-v2-2026-08-10.json"
        in artifact_paths
    )
    assert (
        "round-025-twap-wire-source-qualification-v2-2026-08-10.json" in artifact_paths
    )
    assert (
        "round-027-campaign-admission-gate-correction-amendment-v16.json"
        in artifact_paths
    )
    assert "round-027-effective-source-ledger-v7.json" in artifact_paths
    assert "round-027-static-analysis-remediation-amendment-v17.json" in artifact_paths
    assert "round-027-static-analysis-remediation-amendment-v18.json" in artifact_paths
    assert "round-027-static-analysis-remediation-amendment-v19.json" in artifact_paths
    assert "round-028-loaded-contract-binding-correction-v2.json" in artifact_paths
    assert "round-028-loaded-contract-binding-correction-v3.json" in artifact_paths
    assert "round-028-static-analysis-remediation-amendment-v2.json" in artifact_paths
    assert "round-029-static-analysis-remediation-amendment-v1.json" in artifact_paths
    root = RESEARCH.resolve()
    for entry in manifest["artifacts"]:
        path = (RESEARCH / entry["path"]).resolve()
        assert path.is_relative_to(root)
        payload = _canonical_artifact_bytes(path)
        assert len(payload) == entry["bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]


def test_latest_public_clob_probe_is_truthfully_scoped() -> None:
    probe = json.loads(
        (RESEARCH / "latest" / "public-clob-live-probe-2026-08-10.json").read_text(
            encoding="utf-8"
        )
    )

    assert probe["protocol_version"] == 2
    assert probe["market_count"] == len(probe["markets"]) == 2
    assert probe["scope"] == {
        "asset": "BTC",
        "credentials_accessed": False,
        "market_variant": "fiveminute",
        "orders_submitted": 0,
        "trading_authority": False,
        "wallet_accessed": False,
    }
    assert all(
        market["resolution_source"]
        == "https://data.chain.link/streams/btc-usd-twap-30s-streams"
        and len(market["books"]) == 2
        for market in probe["markets"]
    )
    assert probe["validation"]["all_books_strictly_validated"] is True
    assert probe["validation"]["predictive_edge_tested"] is False
    assert probe["validation"]["profitability_tested"] is False
    assert probe["validation"]["live_readiness_proved"] is False


def test_round25_resolution_transport_probe_is_self_hashed_and_non_authoritative() -> (
    None
):
    probe = json.loads(
        (
            RESEARCH
            / "round-025-official-resolution-transport-probe-v1-2026-08-10.json"
        ).read_text(encoding="utf-8")
    )
    claimed = probe.pop("probe_sha256")
    canonical = json.dumps(
        probe,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert probe["official_sources"]["session_cookie_count_after_requests"] == 0
    assert probe["bounded_sequence"] == {
        "maximum_candidates": 8,
        "candidates_queried": 3,
        "pending_without_inference": 2,
        "jointly_terminal": 1,
        "stopped_after_first_jointly_terminal_candidate": True,
    }
    assert probe["validated_market"]["cross_source_winner_agreement_validated"] is True
    assert not any(bool(value) for value in probe["authority"].values())


def test_round_013_failed_capture_evidence_is_hash_bound_and_ineligible() -> None:
    evidence = json.loads(
        (RESEARCH / "round-013-invalidated-capture-evidence.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = evidence.pop("artifact_sha256")
    canonical = json.dumps(
        evidence,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert evidence["status"] == "failed_capture_ineligible_before_outcome_access"
    assert evidence["frozen_capture_requirement"] == {
        "contract_sha256": (
            "9ace8092d26918b7621aafb7f008106b06c80049c314158e6a26fd5b70dd4325"
        ),
        "duration_shortfall_seconds": 84478.678,
        "observed_duration_seconds": 1921.322,
        "required_duration_seconds": 86400,
        "required_one_shot_capture_completed": False,
        "stream_gap_count": 4,
    }
    assert evidence["outcome_access_evidence"]["performance_labels_opened"] is False
    assert (
        evidence["outcome_access_evidence"][
            "round13_action_and_evaluation_tables_present"
        ]
        is False
    )
    assert all(value is False for value in evidence["authority"].values())


def test_round_008_publication_is_deterministic_and_refuses_tampering(
    tmp_path: Path,
) -> None:
    report = RESEARCH / "round-008-executable-repricing-ceiling-report.json"
    capture = RESEARCH / "round-002-prospective-pipeline-evidence.json"
    local_capture = tmp_path / capture.name
    local_capture.write_bytes(capture.read_bytes())
    first = publish_polymarket_repricing_report(report, local_capture, tmp_path)
    second = publish_polymarket_repricing_report(report, local_capture, tmp_path)

    assert first == second
    manifest = json.loads(
        (tmp_path / "latest" / "publication-integrity.json").read_text(encoding="utf-8")
    )
    assert manifest["claims"]["noncausal_oracle_upper_bound"] is True
    assert manifest["claims"]["profitability_claim"] is False
    assert manifest["claims"]["trading_authority"] is False
    assert manifest["source_report_sha256"] == first.report_sha256

    tampered = tmp_path / "tampered.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["profitability_claim"] = True
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="publication validation"):
        publish_polymarket_repricing_report(tampered, local_capture, tmp_path / "bad")


def test_degraded_capture_and_recorder_benchmark_are_arithmetically_truthful() -> None:
    capture = json.loads(
        (RESEARCH / "round-003-degraded-capture-diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    recorder = capture["capture"]
    gaps = capture["gap_diagnostic"]
    assert sum(recorder["stream_counts"].values()) == recorder["raw_message_count"]
    assert math.isclose(
        recorder["duration_seconds"],
        (recorder["ended_at_ms"] - recorder["started_at_ms"]) / 1_000.0,
    )
    assert sum(capture["resolution"]["asset_outcome_counts"].values()) == 105
    assert capture["resolution"]["finalized_condition_count"] == 105
    assert capture["resolution"]["pending_condition_count"] == 0
    for stream in ("binance_spot", "clob_market", "polymarket_rtds"):
        assert sum(gaps[stream]["reasons"].values()) == gaps[stream]["count"]
    assert (
        sum(
            gaps[stream]["count"]
            for stream in ("binance_spot", "clob_market", "polymarket_rtds")
        )
        == gaps["total"]
        == 54
    )
    assert capture["model_evidence"]["eligible_for_model_fit"] is False
    assert capture["model_evidence"]["profitability_result"] is None

    benchmark = json.loads(
        (RESEARCH / "recorder-v2-liveness-2026-07-15.json").read_text(encoding="utf-8")
    )
    assert benchmark["comparison_limits"]["profitability_evidence"] is False
    before = benchmark["measurements"]["before_coalescing"]
    after = benchmark["measurements"]["coalesced_writer"]
    for measurement in (before, after):
        assert measurement["status"] == "complete"
        assert measurement["stream_gap_count"] == 0
        assert measurement["integrity_error_count"] == 0
        assert math.isclose(
            measurement["average_messages_per_chunk"],
            measurement["raw_message_count"] / measurement["raw_chunk_count"],
        )
        assert math.isclose(
            measurement["compressed_to_uncompressed_ratio"],
            measurement["compressed_bytes"] / measurement["uncompressed_bytes"],
        )
    observed = benchmark["observed_change"]
    before_chunks_per_10k = (
        10_000 * before["raw_chunk_count"] / before["raw_message_count"]
    )
    after_chunks_per_10k = (
        10_000 * after["raw_chunk_count"] / after["raw_message_count"]
    )
    assert math.isclose(
        observed["chunks_per_10000_messages_before"], before_chunks_per_10k
    )
    assert math.isclose(
        observed["chunks_per_10000_messages_after"], after_chunks_per_10k
    )
    assert math.isclose(
        observed["chunks_per_10000_messages_reduction_fraction"],
        1.0 - (after_chunks_per_10k / before_chunks_per_10k),
    )
    assert math.isclose(
        observed["messages_per_chunk_factor"],
        after["average_messages_per_chunk"] / before["average_messages_per_chunk"],
    )

    long_tail = json.loads(
        (RESEARCH / "storage-v3-long-tail-benchmark-2026-07-16.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = long_tail.pop("report_sha256")
    canonical = json.dumps(
        long_tail,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert long_tail["storage_schema_version"] == "polymarket-evidence-storage-v3"
    assert long_tail["persisted_raw_messages"] == long_tail["total_messages"]
    assert long_tail["integrity_errors"] == []
    assert not any(
        constraint[1] in {"PRIMARY KEY", "UNIQUE"}
        for constraint in long_tail["hot_path_constraints"]
    )
    assert (
        min(
            checkpoint["interval_messages_per_second"]
            for checkpoint in long_tail["checkpoints"]
        )
        > 9_700
    )
    assert long_tail["truth_constraints"] == {
        "benchmark_proves_fifteen_hour_capture": False,
        "benchmark_receipt_metadata_is_real": False,
        "financial_edge_claim": False,
        "model_evidence": False,
        "source_payloads_are_real": True,
        "trading_authority": False,
    }


def test_current_ai_risk_evidence_is_truthfully_scoped() -> None:
    latest = RESEARCH / "latest"
    rejected = json.loads(
        (latest / "ai-risk-models-rejected.json").read_text(encoding="utf-8")
    )
    contract = (RESEARCH / "round-003-market-anchored-model-contract.md").read_text(
        encoding="utf-8"
    )

    assert rejected["benchmark_contract"] == "finance-risk-review-adversarial-v7"
    assert rejected["selected_model"] is None
    assert rejected["financial_edge_tested"] is False
    assert rejected["trading_authority"] is False
    assert {item["model"] for item in rejected["results"]} == {
        "qwen3:8b",
        "qwen3.5:9b",
        "fin-r1:8b",
        "fino1:8b",
    }
    assert all(item["passed"] is False for item in rejected["results"])
    assert all(item["valid_json_cases"] == 11 for item in rejected["results"])
    assert "prospective profitability is not established" in contract.lower()
    assert "not market-edge evidence" in contract.lower()
