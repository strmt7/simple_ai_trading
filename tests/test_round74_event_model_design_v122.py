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
    / "round-074-event-sequence-model-design-v122.json"
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


def test_round74_event_design_v122_binds_fail_closed_epistemic_filter() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    claimed = artifact.pop("design_sha256")
    commit = artifact["implementation_git_commit"]
    source = artifact["source_binding"]
    contract = artifact["filter_contract"]
    safeguards = artifact["runtime_safeguards"]
    status = artifact["implementation_status"]

    assert claimed == _canonical_sha256(artifact)
    assert artifact["schema_version"] == (
        "round-074-event-sequence-model-design-v122"
    )
    assert artifact["supersedes_artifact_sha256"] == (
        "a2411838831f722370f4a608e974d74881f79956f100d17e23f209c3cc3f4abc"
    )
    for label in (
        "action_policy",
        "epistemic_action_policy",
        "action_policy_test",
        "epistemic_evaluation_test",
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

    assert contract["profiles"]["conservative"]["component_quantile"] == 0.95
    assert contract["profiles"]["regular"]["component_quantile"] == 0.97
    assert contract["profiles"]["aggressive"]["component_quantile"] == 0.99
    assert len(contract["components"]) == 5
    assert contract["tail_allocation"] == (
        "bonferroni_equal_across_five_components"
    )
    assert contract["target_fields_used_for_threshold_fit"] is False
    assert contract["runtime_target_fields_consumed"] is False
    assert contract["future_rejection_rate_guaranteed"] is False
    assert contract["exchangeability_claim"] is False
    assert contract["thresholds_transfer_across_symbols_or_horizons"] is False

    assert safeguards["candidate_set_expansion_permitted"] is False
    assert safeguards["trade_side_creation_permitted"] is False
    assert safeguards["position_size_change_permitted"] is False
    assert safeguards["leverage_change_permitted"] is False
    assert safeguards["order_submission_permitted"] is False
    assert safeguards["position_management_permitted"] is False
    caveat = artifact["execution_path_caveat"]
    assert caveat["candidate_subset_proves_executed_trade_subset"] is False
    assert caveat[
        "automatic_integration_permitted_before_exact_delayed_l2_replay"
    ] is False

    assert status["threshold_fitter_implemented"] is True
    assert status["target_free_runtime_filter_implemented"] is True
    assert status["target_mutation_invariance_tested"] is True
    assert status["development_coordinator_integration_completed"] is False
    assert status["representative_thresholds_fitted"] is False
    assert status["representative_delayed_l2_replay_completed"] is False
    assert status["automatic_policy_use_enabled"] is False
    assert status["sealed_test_accessed"] is False
    campaign = artifact["campaign_observation"]
    assert campaign["terminal_error"] == "writer_shutdown_timeout"
    assert campaign["qualification_passed"] is False
    assert campaign["admitted_to_cohort"] is False
    assert campaign["automatic_retry_permitted"] is False
    assert campaign["credentials_used"] is False
    assert campaign["orders_submitted"] is False
    assert all(value is False for value in artifact["evidence_boundary"].values())
    assert all(value is False for value in artifact["authority"].values())
