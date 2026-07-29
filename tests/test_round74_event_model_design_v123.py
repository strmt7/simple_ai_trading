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
    / "round-074-event-sequence-model-design-v123.json"
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


def test_round74_event_design_v123_binds_exact_replay_challenge() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    claimed = artifact.pop("design_sha256")
    commit = artifact["implementation_git_commit"]
    source = artifact["source_binding"]
    replay = artifact["replay_contract"]
    gate = artifact["tuning_challenge_gate"]
    controls = artifact["synthetic_controls"]
    status = artifact["implementation_status"]

    assert claimed == _canonical_sha256(artifact)
    assert artifact["schema_version"] == (
        "round-074-event-sequence-model-design-v123"
    )
    assert artifact["supersedes_artifact_sha256"] == (
        "c49ae4d5deb11814e50f473f558cb9dceb5e1c9aaefd36c7c92d357419284e49"
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

    assert replay["requires_v121_policy_challenge_eligible"] is True
    assert replay["baseline_quality_threshold_held_fixed"] is True
    assert replay["baseline_trace_recomputed_before_comparison"] is True
    assert replay["same_exact_delayed_l2_execution_panel_used"] is True
    assert replay["retained_removed_and_replacement_trades_recorded"] is True
    assert gate["challenger_objective_strictly_exceeds_baseline"] is True
    assert gate["challenger_total_net_bps_not_below_baseline"] is True
    assert gate["challenger_maximum_drawdown_bps_not_above_baseline"] is True
    assert gate[
        "challenger_maximum_concurrent_adverse_excursion_bps_not_above_baseline"
    ] is True
    assert gate["challenger_adverse_selection_rate_not_above_baseline"] is True
    assert gate["passing_tuning_gate_enables_automatic_policy_use"] is False
    assert gate["passing_tuning_gate_skips_sealed_evaluation"] is False

    assert controls["positive_control"]["removed_trade_count"] == 1
    assert controls["positive_control"]["replacement_trade_count"] == 1
    assert controls["positive_control"]["tuning_challenge_eligible"] is True
    assert controls["negative_control"]["removed_trade_count"] == 1
    assert controls["negative_control"]["replacement_trade_count"] == 1
    assert controls["negative_control"]["tuning_challenge_eligible"] is False
    assert controls["synthetic_controls_are_market_edge_evidence"] is False
    assert status["exact_replay_challenge_evaluator_implemented"] is True
    assert status["development_coordinator_integration_completed"] is False
    assert status["representative_exact_replay_challenge_evaluated"] is False
    assert status["automatic_policy_use_enabled"] is False
    assert status["sealed_test_accessed"] is False
    assert all(value is False for value in artifact["evidence_boundary"].values())
    assert all(value is False for value in artifact["authority"].values())
