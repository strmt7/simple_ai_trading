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
    / "round-074-event-sequence-model-design-v127.json"
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


def test_round74_event_design_v127_binds_sealed_configuration_governance() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    claimed = artifact.pop("design_sha256")
    commit = artifact["implementation_git_commit"]
    source = artifact["source_binding"]
    schema = artifact["schema_contract"]
    governance = artifact["governance_contract"]
    filtering = artifact["target_free_filter_contract"]
    qualification = artifact["ai_qualification_contract"]
    status = artifact["implementation_status"]

    assert claimed == _canonical_sha256(artifact)
    assert artifact["schema_version"] == (
        "round-074-event-sequence-model-design-v127"
    )
    assert artifact["supersedes_artifact_sha256"] == (
        "6fa1410e9f1be50d5f01139ff5c6e79ef14e55e494934e6f6bf65815002e6ca9"
    )
    for binding in source["implementation_files"] + source["affected_tests"]:
        assert binding["sha256"] == _source_file_sha256_at(
            commit,
            binding["path"],
        )
    predecessor = source["predecessor_design"]
    assert predecessor["sha256"] == hashlib.sha256(
        (REPOSITORY / predecessor["path"]).read_bytes()
    ).hexdigest()

    assert schema["ai_pretest_qualification"] == (
        "round-074-ai-pretest-qualification-v5"
    )
    assert schema["sealed_ledger"] == "round-074-sealed-ledger-v4"
    assert schema["sealed_claim"] == "round-074-sealed-claim-v4"
    assert schema["sealed_evaluation"] == "round-074-sealed-evaluation-v24"
    assert governance["final_configuration_digest_bound_before_target_loading"] is True
    assert governance["ai_prequalification_must_bind_same_configuration_digest"] is True
    assert governance["legacy_unbound_reservation_is_live_evaluation_authority"] is False
    assert governance["post_sealed_test_configuration_switching_permitted"] is False
    assert filtering["filter_applied_before_ai_review"] is True
    assert filtering["realized_target_fields_consumed"] is False
    assert filtering["candidate_set_only_reduced"] is True
    assert qualification["same_final_ml_configuration_as_sealed_evaluation_required"] is True
    assert qualification["ai_role_remains_veto_or_reduce_only"] is True
    assert status["ledger_reservation_configuration_binding_completed"] is True
    assert status["filtered_representative_ai_qualification_run"] is False
    assert status["filtered_representative_sealed_run"] is False
    assert status["sealed_test_accessed"] is False
    assert all(value is False for value in artifact["evidence_boundary"].values())
    assert all(value is False for value in artifact["authority"].values())
