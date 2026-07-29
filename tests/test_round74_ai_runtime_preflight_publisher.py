from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404
from types import ModuleType


REPOSITORY = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY / "tools/publish_round74_ai_runtime_preflight.py"
EVIDENCE_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-local-ai-runtime-preflight-v5-2026-07-27.json"
)
SESSION_EVIDENCE_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-local-ai-runtime-preflight-v6-2026-07-27.json"
)
SPEC = importlib.util.spec_from_file_location(
    "publish_round74_ai_runtime_preflight",
    TOOL_PATH,
)
assert SPEC is not None and SPEC.loader is not None
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)
assert isinstance(PUBLISHER, ModuleType)


def _source_sha256_at(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        timeout=30,
    )
    payload = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_preflight_schema_tracks_current_ai_contract() -> None:
    assert PUBLISHER.SCHEMA_VERSION == "round-074-local-ai-runtime-preflight-v9"
    assert PUBLISHER.ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION == (
        "round-074-ai-review-request-v7"
    )
    assert PUBLISHER.ROUND74_AI_PROMPT_PAYLOAD_SCHEMA_VERSION == (
        "round-074-ai-prompt-payload-v10"
    )
    assert PUBLISHER.ROUND74_AI_SYSTEM_PROMPT_SCHEMA_VERSION == (
        "round-074-ai-system-prompt-v3"
    )
    assert PUBLISHER.ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION == (
        "round-074-ai-review-panel-v17"
    )


def test_synthetic_request_exercises_profile_and_temporal_path() -> None:
    request = PUBLISHER._synthetic_request()
    request.validate()

    assert request.risk_profile == "conservative"
    assert request.epistemic_peer_count == 3
    assert (
        request.regime_unpredictability_probability_peer_standard_deviation
        == 0.24
    )
    assert request.feature_last[0] == 1.0
    assert request.feature_last[5:8] == (1.0, 0.0, 0.0)
    assert sum(request.feature_mean[:5]) == 1.0
    assert request.feature_mean[5:8] == (1.0, 0.0, 0.0)
    assert request.positive_payoff_probability == 0.70
    assert request.opposing_positive_payoff_probability == 0.10
    assert request.neither_positive_payoff_probability == 0.20
    assert len(request.feature_recent_block_means) == (
        PUBLISHER.ROUND74_AI_TEMPORAL_BLOCK_COUNT
    )
    assert all(
        len(row) == len(PUBLISHER.ROUND74_AI_TEMPORAL_FEATURE_NAMES)
        for row in request.feature_recent_block_means
    )
    spread_path = tuple(row[0] for row in request.feature_recent_block_means)
    volatility_path = tuple(row[10] for row in request.feature_recent_block_means)
    assert spread_path == tuple(sorted(spread_path))
    assert volatility_path == spread_path
    assert len(set(spread_path)) == PUBLISHER.ROUND74_AI_TEMPORAL_BLOCK_COUNT


def test_synthetic_request_is_hash_stable_after_round_trip() -> None:
    request = PUBLISHER._synthetic_request()
    restored = PUBLISHER.Round74AIReviewRequest.from_dict(request.as_dict())

    assert restored == request
    assert restored.request_sha256 == request.request_sha256


def test_persisted_preflight_is_source_bound_isolated_and_nonfinancial() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="ascii"))
    claimed = evidence.pop("artifact_sha256")
    commit = evidence["execution_git_commit"]
    source = evidence["source_binding"]
    inputs = evidence["input_contract"]
    outcomes = evidence["model_outcomes"]
    verification = evidence["verification"]

    assert claimed == PUBLISHER._canonical_sha256(evidence)
    assert evidence["schema_version"] == "round-074-local-ai-runtime-preflight-v5"
    for label in PUBLISHER.SOURCE_PATHS:
        assert source[f"{label}_sha256"] == _source_sha256_at(
            commit,
            source[f"{label}_path"],
        )
    assert inputs["review_request_schema_version"] == ("round-074-ai-review-request-v5")
    assert inputs["prompt_payload_schema_version"] == ("round-074-ai-prompt-payload-v7")
    assert inputs["system_prompt_schema_version"] == ("round-074-ai-system-prompt-v1")
    assert inputs["review_panel_schema_version"] == ("round-074-ai-review-panel-v11")
    assert inputs["risk_profile"] == "conservative"
    assert inputs["temporal_block_count"] == 4
    assert inputs["temporal_feature_count"] == 14
    assert inputs["temporal_order"] == "oldest_to_newest"
    assert not inputs["real_market_events_used"]
    assert not inputs["real_market_targets_used"]
    assert not inputs["test_partition_accessed"]
    assert len(outcomes) == 4
    assert [
        (value["role"], value["model_name"], value["phase"]) for value in outcomes
    ] == [
        ("finance_primary", "fino1:8b", "cold"),
        ("finance_primary", "fino1:8b", "warm"),
        ("general_control", "qwen3:8b", "cold"),
        ("general_control", "qwen3:8b", "warm"),
    ]
    for outcome in outcomes:
        result = outcome["outcome"]
        worker = result["worker_result"]
        decision = worker["decision"]
        capability = result["capability"]
        assert result["status"] == "accepted"
        assert result["approved_risk_size_bps"] <= result["proposed_risk_size_bps"]
        assert worker["prompt_eval_count"] > 0
        assert worker["eval_count"] > 0
        assert worker["residency"]["status"] == "gpu_resident"
        assert worker["residency"]["vram_to_model_ratio"] == 1.0
        assert not decision["may_increase_risk"]
        assert not decision["may_select_side"]
        assert not decision["may_set_leverage"]
        assert not decision["may_submit_or_cancel_orders"]
        if outcome["phase"] == "cold":
            assert capability["free_ram_gb"] >= 16.0
            assert capability["free_vram_gb"] >= 8.0
        else:
            assert capability["pre_inference_exact_model_fully_gpu_resident"]
            assert capability["pre_inference_warm_ram_headroom_passed"]
            assert capability[
                "pre_inference_warm_equivalent_preload_ram_headroom_passed"
            ]
            assert capability["pre_inference_warm_equivalent_preload_ram_gb"] >= 16.0
    for cold, warm in ((outcomes[0], outcomes[1]), (outcomes[2], outcomes[3])):
        cold_worker = cold["outcome"]["worker_result"]
        warm_worker = warm["outcome"]["worker_result"]
        assert cold_worker["decision"] == warm_worker["decision"]
        assert cold_worker["prompt_eval_count"] == warm_worker["prompt_eval_count"]
        assert warm_worker["load_duration_ns"] < cold_worker["load_duration_ns"]
        assert warm["outcome"]["elapsed_ns"] < cold["outcome"]["elapsed_ns"]
    assert verification["model_count"] == 2
    assert verification["request_count"] == 4
    assert verification["all_models_accepted_by_protocol"]
    assert verification["all_models_fully_gpu_resident"]
    assert evidence["runtime_isolation"]["resident_models_before"] == []
    assert evidence["runtime_isolation"]["resident_models_after"] == []
    assert not evidence["interpretation"][
        "representative_market_ai_evaluation_completed"
    ]
    assert not evidence["interpretation"]["ai_uplift_established"]
    assert not evidence["interpretation"]["financial_edge_established"]
    assert not evidence["interpretation"]["profitability_claim"]
    assert not evidence["interpretation"]["paper_trading_authority"]
    assert not evidence["interpretation"]["testnet_trading_authority"]
    assert not evidence["interpretation"]["live_trading_authority"]


def test_persisted_session_preflight_reuses_workers_and_reduces_warm_latency() -> None:
    prior = json.loads(EVIDENCE_PATH.read_text(encoding="ascii"))
    evidence = json.loads(SESSION_EVIDENCE_PATH.read_text(encoding="ascii"))
    claimed = evidence.pop("artifact_sha256")
    commit = evidence["execution_git_commit"]
    source = evidence["source_binding"]
    outcomes = evidence["model_outcomes"]
    isolation = evidence["runtime_isolation"]
    verification = evidence["verification"]

    assert claimed == PUBLISHER._canonical_sha256(evidence)
    assert evidence["schema_version"] == "round-074-local-ai-runtime-preflight-v6"
    for label in PUBLISHER.SOURCE_PATHS:
        assert source[f"{label}_sha256"] == _source_sha256_at(
            commit,
            source[f"{label}_path"],
        )
    assert evidence["input_contract"]["worker_process_mode"] == (
        "persistent_model_batch"
    )
    assert isolation["resident_models_before"] == []
    assert isolation["resident_models_after"] == []
    assert isolation["isolated_worker_process_per_model_batch"]
    assert isolation["isolated_worker_reused_within_model_batch"]
    assert isolation["isolated_worker_restart_count"] == 0
    assert isolation["isolated_worker_processes_closed_after_batches"]
    assert len(outcomes) == len(prior["model_outcomes"]) == 4
    for index, outcome in enumerate(outcomes):
        capability = outcome["outcome"]["capability"]
        prior_outcome = prior["model_outcomes"][index]
        assert outcome["model_name"] == prior_outcome["model_name"]
        assert outcome["phase"] == prior_outcome["phase"]
        assert outcome["outcome"]["status"] == "accepted"
        assert (
            outcome["outcome"]["worker_result"]["decision"]
            == (prior_outcome["outcome"]["worker_result"]["decision"])
        )
        assert capability["worker_process_mode"] == "persistent_model_batch"
        assert capability["worker_session_request_ordinal"] == (
            1 if outcome["phase"] == "cold" else 2
        )
        assert capability["worker_session_restart_count_before_request"] == 0
        assert capability["worker_session_restart_count_after_request"] == 0
    assert (
        outcomes[1]["outcome"]["elapsed_ns"]
        < (prior["model_outcomes"][1]["outcome"]["elapsed_ns"])
    )
    assert (
        outcomes[3]["outcome"]["elapsed_ns"]
        < (prior["model_outcomes"][3]["outcome"]["elapsed_ns"])
    )
    assert (
        outcomes[1]["outcome"]["elapsed_ns"]
        * 10_000
        // (prior["model_outcomes"][1]["outcome"]["elapsed_ns"])
        == 4_082
    )
    assert (
        outcomes[3]["outcome"]["elapsed_ns"]
        * 10_000
        // (prior["model_outcomes"][3]["outcome"]["elapsed_ns"])
        == 5_955
    )
    assert verification["all_model_batches_reused_one_isolated_worker_process"]
    assert verification["all_worker_session_restart_counts_zero"]
    assert not evidence["interpretation"][
        "representative_market_ai_evaluation_completed"
    ]
    assert not evidence["interpretation"]["ai_uplift_established"]
    assert not evidence["interpretation"]["financial_edge_established"]
    assert not evidence["interpretation"]["profitability_claim"]
