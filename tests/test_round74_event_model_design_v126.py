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
    / "round-074-event-sequence-model-design-v126.json"
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


def test_round74_event_design_v126_binds_final_action_configuration() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    claimed = artifact.pop("design_sha256")
    commit = artifact["implementation_git_commit"]
    source = artifact["source_binding"]
    contract = artifact["configuration_contract"]
    rule = artifact["selection_rule"]
    family = artifact["sealed_qualification_family"]
    status = artifact["implementation_status"]

    assert claimed == _canonical_sha256(artifact)
    assert artifact["schema_version"] == (
        "round-074-event-sequence-model-design-v126"
    )
    assert artifact["supersedes_artifact_sha256"] == (
        "34d61ea06f815c44065b3ca8e0148e7bb8fe1007688bc30f406b15a65d484d0f"
    )
    for label in (
        "final_action_configuration",
        "action_policy_test",
        "development_operator_test",
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

    assert contract["schema_version"] == (
        "round-074-final-action-configuration-v1"
    )
    assert contract["accepted_baseline_action_selection_required"] is True
    assert contract["exact_delayed_l2_execution_panel_required"] is True
    assert contract["canonical_configuration_digest_required"] is True
    assert rule["selection_data_role"] == "tuning_only"
    assert rule["epistemic_filter_selected_only_when_tuning_challenge_eligible"] is True
    assert rule["post_sealed_test_switching_permitted"] is False
    assert family["single_final_ml_configuration"] is True
    assert family["ml_configuration_count"] == 1
    assert family["ai_overlay_configuration_count"] == 2
    assert family["total_configuration_count"] == 3
    assert status["configuration_object_completed"] is True
    assert status["sealed_ledger_configuration_digest_binding_completed"] is False
    assert status["sealed_evaluator_filter_application_completed"] is False
    assert status["sealed_test_accessed"] is False
    assert all(value is False for value in artifact["evidence_boundary"].values())
    assert all(value is False for value in artifact["authority"].values())
