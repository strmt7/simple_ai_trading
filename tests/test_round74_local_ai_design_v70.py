from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-local-ai-review-design-v70.json"
)


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_file_sha256_at(commit: str, path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{path}"],  # nosec B607
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def test_round74_ai_design_v70_binds_deferred_oda_registry_integration() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    claimed = artifact.pop("design_sha256")
    commit = artifact["implementation_git_commit"]
    source = artifact["source_binding"]
    candidate = artifact["research_candidate"]
    runtime = candidate["local_quantized_artifact"]
    contract = artifact["registry_contract"]

    assert claimed == _canonical_sha256(artifact)
    assert artifact["schema_version"] == "round-074-local-ai-review-design-v70"
    assert len(artifact["supersedes_artifact_sha256"]) == 64
    for label in ("candidate_registry", "candidate_registry_tests"):
        binding = source[label]
        assert binding["sha256"] == _source_file_sha256_at(
            commit,
            binding["path"],
        )
    for label in ("predecessor_design", "latest_blocked_preload_evidence"):
        binding = source[label]
        assert binding["sha256"] == hashlib.sha256(
            (REPOSITORY / binding["path"]).read_bytes()
        ).hexdigest()

    assert candidate["model_id"] == "OpenDataArena/ODA-Fin-RL-8B"
    assert candidate["upstream_revision"] == (
        "948c22ea48f9bf93e5747f4211657fcad9cb0295"
    )
    assert candidate["parameter_count"] == 8_190_735_360
    assert candidate["license_id"] == "Apache-2.0"
    assert runtime["repository"] == "alexsabaka/ODA-Fin-RL-8B-GGUF"
    assert runtime["repository_revision"] == (
        "fb986718fcec8c8c559f0ebd5d50e6cf9f4ac67f"
    )
    assert runtime["filename"] == "ODA-Fin-RL-8B.Q4_K_M.gguf"
    assert runtime["size_bytes"] == 5_027_784_512
    assert runtime["lfs_sha256"] == (
        "d40d1dd4105be8d85cbb444cb58e92c4882623f0baa4dea5d296745d6bc13861"
    )
    assert runtime["publisher_is_upstream_model_author"] is False
    assert runtime["downloaded"] is False
    assert runtime["digest_verified_locally"] is False
    assert candidate["external_finance_benchmark_claim_is_trading_evidence"] is False
    assert candidate["external_bf16_results_transfer_to_q4_k_m_without_measurement"] is False
    assert candidate["operational_candidate"] is False

    assert contract["registered_in_executable_catalog"] is True
    assert contract["priority_order_is_descending_and_unique"] is True
    assert contract["implicit_default_selection_permitted"] is False
    assert contract["explicit_benchmark_selection_permitted_after_preflight"] is True
    assert contract["ten_case_adversarial_contract_screen_required"] is True
    assert contract["representative_paired_after_cost_ai_vs_ml_uplift_required"] is True
    assert contract["language_model_may_create_or_reverse_trade_side"] is False
    assert contract["language_model_may_increase_risk_or_leverage"] is False
    assert contract["language_model_may_submit_cancel_or_close_orders"] is False
    assert contract["language_model_role"] == "veto_or_reduce_only"

    host = artifact["host_observation"]
    assert host["capture_database_or_wal_opened_for_this_delta"] is False
    assert host["free_system_ram_gib"] < host["minimum_free_system_ram_gib"]
    assert host["oda_download_started"] is False
    assert host["oda_model_loaded"] is False
    assert host["oda_screen_started"] is False
    assert host["fin_r1_preload_failure_is_an_oda_result"] is False

    verification = artifact["verification"]
    assert verification["focused_registry_contract_tests_passed"] == 2
    assert verification["connected_ai_runtime_and_evidence_tests_passed"] == 40
    assert verification["ruff_lint_passed"] is True
    assert verification["model_downloaded"] is False
    assert verification["local_multibillion_parameter_inference_performed"] is False
    assert verification["representative_market_ai_evaluation_performed"] is False
    assert verification["sealed_test_accessed"] is False
    assert verification["live_or_testnet_orders_submitted"] is False
    assert all(value is False for value in artifact["evidence_boundary"].values())
    assert all(value is False for value in artifact["authority"].values())
