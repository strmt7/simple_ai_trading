from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

from simple_ai_trading.round74_representation_comparison import (
    ROUND74_REPRESENTATION_COMPARISON_SCHEMA_VERSION,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v75.json"
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


def test_round74_v75_binds_conservative_paired_run_economics() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v75"
    base = design["base_design"]
    assert base["normalized_lf_sha256"] == _normalized_file_sha256(
        REPOSITORY / base["path"]
    )
    source = design["source_binding"]
    for prefix in ("representation_comparison", "action_policy", "calibration"):
        assert source[f"{prefix}_normalized_lf_sha256"] == _git_file_sha256(
            design["implementation_git_commit"],
            source[f"{prefix}_path"],
        )

    assert ROUND74_REPRESENTATION_COMPARISON_SCHEMA_VERSION == (
        "round-074-representation-comparison-v2"
    )
    gate = design["conservative_run_level_gate"]
    assert gate["policy_selection_run_count"] == 6
    assert gate["every_paired_run_must_be_noninferior"] is True
    assert gate["aggregate_improvement_can_override_a_degraded_run"] is False
    assert gate["all_six_paired_run_deltas_recorded"] is True
    assert gate["derived_summaries_recomputed_on_reload"] is True
    assert gate["sealed_test_used_for_selection"] is False
    boundary = design["active_evidence_boundary"]
    assert boundary["historical_v5_capture_admitted_to_model_cohort"] is False
    assert boundary["prospective_v5_r1_slot_zero_completed"] is False
    assert all(value is False for value in design["authority"].values())
