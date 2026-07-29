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
    / "round-074-event-sequence-model-design-v125.json"
)


def _canonical_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


def _source_file_sha256_at(commit: str, path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{path}"],  # nosec B607
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def test_round74_event_design_v125_binds_coordinator_replay_evidence() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    claimed = artifact.pop("design_sha256")
    commit = artifact["implementation_git_commit"]
    source = artifact["source_binding"]
    schema = artifact["schema_contract"]
    coordinator = artifact["coordinator_contract"]
    bundle = artifact["bundle_validation_contract"]
    status = artifact["implementation_status"]

    assert claimed == _canonical_sha256(artifact)
    assert artifact["schema_version"] == (
        "round-074-event-sequence-model-design-v125"
    )
    assert artifact["supersedes_artifact_sha256"] == (
        "22c721db57b783d45f9bfee4c548d46c8f2676203b48aafc0f270b8d9cb46606"
    )
    for label in (
        "development_operator",
        "development_operator_test",
        "epistemic_action_policy",
    ):
        binding = source[label]
        assert binding["sha256"] == _source_file_sha256_at(
            commit,
            binding["path"],
        )
    predecessor = source["predecessor_design"]
    assert predecessor["sha256"] == hashlib.sha256(
        (REPOSITORY / predecessor["path"]).read_bytes()
    ).hexdigest()

    assert schema["development_operator_schema_version"] == (
        "round-074-development-policy-operator-v6"
    )
    assert schema["development_bundle_schema_version"] == (
        "round-074-development-policy-bundle-v7"
    )
    assert coordinator["shared_policy_selection_model_outputs_reused"] is True
    assert coordinator["duplicate_model_inference_for_profile_challenges"] is False
    assert coordinator["risk_coverage_policy_challenge_gate_required"] is True
    assert coordinator["accepted_baseline_policy_required_per_profile"] is True
    assert coordinator["baseline_action_policies_remain_authoritative"] is True
    assert coordinator["challenge_eligibility_changes_selected_policy"] is False
    assert coordinator["sealed_test_accessed"] is False
    assert bundle["baseline_policy_selection_digest_must_match"] is True
    assert bundle["execution_panel_digest_must_match_profile"] is True
    assert bundle["source_run_batch_model_output_order_must_match"] is True
    assert status["coordinator_integration_completed"] is True
    assert status["representative_exact_replay_challenge_evaluated"] is False
    assert status["automatic_policy_use_enabled"] is False
    assert status["sealed_test_accessed"] is False
    assert all(value is False for value in artifact["evidence_boundary"].values())
    assert all(value is False for value in artifact["authority"].values())
