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
    / "round-074-event-sequence-model-design-v124.json"
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


def test_round74_event_design_v124_binds_replay_evidence_persistence() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    claimed = artifact.pop("design_sha256")
    commit = artifact["implementation_git_commit"]
    source = artifact["source_binding"]
    contract = artifact["persistence_contract"]
    status = artifact["implementation_status"]

    assert claimed == _canonical_sha256(artifact)
    assert artifact["schema_version"] == (
        "round-074-event-sequence-model-design-v124"
    )
    assert artifact["supersedes_artifact_sha256"] == (
        "762f2defdf84bac99b958b9afb031e1bd937724406c311f54aca502b1ba818b5"
    )
    for label in ("epistemic_action_policy", "action_policy_test"):
        binding = source[label]
        assert binding["sha256"] == _source_file_sha256_at(
            commit,
            binding["path"],
        )
    predecessor = source["predecessor_design"]
    assert predecessor["sha256"] == hashlib.sha256(
        (REPOSITORY / predecessor["path"]).read_bytes()
    ).hexdigest()

    assert contract["canonical_json_sha256_required"] is True
    assert contract["nested_filter_identity_revalidated"] is True
    assert contract["nested_application_identities_revalidated"] is True
    assert contract["application_model_output_order_bound_to_filter"] is True
    assert contract["boolean_as_integer_rejected"] is True
    assert contract["profile_objective_recomputed_from_persisted_metrics"] is True
    assert contract["recomputed_objectives_must_match_persisted_values"] is True
    assert contract["authority_flags_must_be_exact_false_booleans"] is True
    assert status["challenge_roundtrip_implemented"] is True
    assert status["development_coordinator_integration_completed"] is False
    assert status["representative_exact_replay_challenge_evaluated"] is False
    assert status["automatic_policy_use_enabled"] is False
    assert status["sealed_test_accessed"] is False
    assert all(value is False for value in artifact["evidence_boundary"].values())
    assert all(value is False for value in artifact["authority"].values())
