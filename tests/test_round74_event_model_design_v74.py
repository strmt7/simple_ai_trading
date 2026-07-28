from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v74.json"
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


def _normalized_file_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _git_file_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    payload = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _git_file_text(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode("ascii")


def test_round74_v74_binds_one_pass_fixed_architecture_comparison() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v74"
    base = design["base_design"]
    assert base["normalized_lf_sha256"] == _normalized_file_sha256(
        REPOSITORY / base["path"]
    )
    source = design["source_binding"]
    for prefix in (
        "event_dataset",
        "model_operator",
        "event_training",
        "development_operator",
        "representation_comparison",
    ):
        assert source[f"{prefix}_normalized_lf_sha256"] == _git_file_sha256(
            design["implementation_git_commit"],
            source[f"{prefix}_path"],
        )

    implementation_commit = design["implementation_git_commit"]
    event_dataset = _git_file_text(
        implementation_commit,
        source["event_dataset_path"],
    )
    model_operator = _git_file_text(
        implementation_commit,
        source["model_operator_path"],
    )
    event_training = _git_file_text(
        implementation_commit,
        source["event_training_path"],
    )
    assert (
        'ROUND74_EVENT_DATASET_SCHEMA_VERSION = "round-074-event-dataset-v10"'
        in event_dataset
    )
    assert (
        'ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION = '
        '"round-074-event-model-operator-v5"'
        in model_operator
    )
    assert (
        'ROUND74_EVENT_TRAINING_SCHEMA_VERSION = "round-074-event-training-v17"'
        in event_training
    )
    assert (
        'ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION = '
        '"round-074-event-pretest-policy-v16"'
        in event_training
    )
    replay = design["one_pass_matched_replay"]
    assert replay["raw_observation_iterator_calls_per_run"] == 1
    assert replay["target_value_or_outcome_used_for_sampling"] is False
    assert replay["test_role_access_permitted"] is False
    selection = design["staged_selection"]
    assert selection["challenger_candidate"] == "baseline_selected_candidate"
    assert selection["sealed_test_used_for_selection"] is False
    assert design["verification"]["prospective_representation_comparison_run"] is False
    assert all(value is False for value in design["authority"].values())
