from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

from simple_ai_trading.impact_absorption_event_action_policy import (
    ROUND74_ACTION_POLICY_SCHEMA_VERSION,
    ROUND74_ACTION_PROFILES,
)
from simple_ai_trading.round74_delayed_execution_panel import (
    ROUND74_DELAYED_EXECUTION_REPLAY_SCHEMA_VERSION,
)
from simple_ai_trading.round74_event_development_operator import (
    ROUND74_DEVELOPMENT_OPERATOR_SCHEMA_VERSION,
    ROUND74_DEVELOPMENT_POLICY_BUNDLE_SCHEMA_VERSION,
)
from simple_ai_trading.round74_online_decision_latency import (
    ROUND74_ONLINE_DECISION_LATENCY_SCHEMA_VERSION,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v70.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _git_file_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    payload = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_round74_v70_binds_exact_profile_delayed_policy_economics() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v70"
    source = design["source_binding"]
    commit = design["implementation_git_commit"]
    for name in (
        "action_policy",
        "online_decision_latency",
        "delayed_execution",
        "development_operator",
    ):
        assert source[f"{name}_normalized_lf_sha256"] == _git_file_sha256(
            commit,
            source[f"{name}_path"],
        )
    assert source["action_policy_schema_version"] == (
        ROUND74_ACTION_POLICY_SCHEMA_VERSION
    )
    assert source["online_decision_latency_schema_version"] == (
        ROUND74_ONLINE_DECISION_LATENCY_SCHEMA_VERSION
    )
    assert source["delayed_execution_schema_version"] == (
        ROUND74_DELAYED_EXECUTION_REPLAY_SCHEMA_VERSION
    )
    assert source["development_operator_schema_version"] == (
        ROUND74_DEVELOPMENT_OPERATOR_SCHEMA_VERSION
    )
    assert source["development_bundle_schema_version"] == (
        ROUND74_DEVELOPMENT_POLICY_BUNDLE_SCHEMA_VERSION
    )

    replay = design["exact_delayed_execution_contract"]
    assert tuple(replay["profiles"]) == ROUND74_ACTION_PROFILES
    assert replay["database_scans"] == replay["policy_selection_capture_runs"] == 6
    assert replay["in_memory_target_engine_replays"] == 18
    assert replay["naive_threshold_by_profile_database_scans_rejected"] == 72
    assert replay["candidate_identity_remains_target_free"] is True
    assert replay["execution_economics_are_a_separate_hash_bound_panel"] is True
    assert replay["all_three_profile_engines_consume_each_capture_stream_once"] is True
    assert replay["full_outcome_rows_duplicated_in_development_bundle"] is False

    coordinator = design["coordinator_contract"]
    assert coordinator["exact_fitted_scaler_required"] is True
    assert coordinator["read_only_execution_store_required"] is True
    assert coordinator["all_profile_policies_require_execution_panel_sha256"] is True
    assert coordinator["sealed_test_accessed"] is False
    assert all(value is False for value in design["authority"].values())
