from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN = (
    REPOSITORY
    / "docs/model-research/action-value"
    / "round-074-event-sequence-model-design-v65.json"
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


def _normalized_lf_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _normalized_lf_sha256_at_commit(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        timeout=30,
    )
    payload = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_round74_v65_binds_guarded_development_runtime() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    runtime = design["development_runtime"]
    commit = subprocess.run(  # nosec B603
        [
            "git",
            "log",
            "-n",
            "1",
            "--format=%H",
            "--",
            str(DESIGN.relative_to(REPOSITORY)),
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        encoding="ascii",
        timeout=30,
    ).stdout.strip()

    assert claimed == _canonical_sha256(design)
    assert design["base_design"]["normalized_lf_sha256"] == _normalized_lf_sha256(
        REPOSITORY / design["base_design"]["path"]
    )
    for label in ("runtime", "tool", "runtime_test", "tool_test"):
        assert runtime[f"{label}_normalized_lf_sha256"] == (
            _normalized_lf_sha256_at_commit(
                commit,
                runtime[f"{label}_path"],
            )
        )


def test_round74_v65_opens_no_database_before_complete_development_panel() -> None:
    runtime = json.loads(DESIGN.read_text(encoding="ascii"))["development_runtime"]

    assert runtime["input_validation_before_database_open"] is True
    assert runtime["required_binding_count"] == 168
    assert runtime["required_development_target_assembly_count"] == 144
    assert runtime["sealed_test_target_assembly_count_read"] == 0
    assert runtime["database_open_mode"] == "read_only"
    assert runtime["capture_process_checks"] == 2
    assert runtime["wal_absence_checks"] == 2
    assert runtime["model_hyperparameters_exposed_by_runner"] is False


def test_round74_v65_preserves_all_claim_and_authority_blocks() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    verification = design["verification"]
    unchanged = design["unchanged_model_contract"]

    assert verification["representative_cohort_available"] is False
    assert verification["development_training_executed"] is False
    assert verification["gpu_model_workload_executed"] is False
    assert unchanged["candidate_panel_changed"] is False
    assert unchanged["execution_evidence_required"] is True
    assert unchanged["default_profile"] == "conservative"
    assert set(design["authority"].values()) == {False}
