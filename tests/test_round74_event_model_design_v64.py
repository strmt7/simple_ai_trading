from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v64.json"
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
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _normalized_lf_sha256_at_commit(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        timeout=30,
    )
    raw = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def test_round74_v64_composes_exact_v63_and_committed_sources() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    claimed = design.pop("design_sha256")
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
    base = design["base_design"]
    assert base["normalized_lf_sha256"] == _normalized_lf_sha256(
        REPOSITORY / base["path"]
    )
    for relative, expected in design["normalized_lf_source_binding"].items():
        assert _normalized_lf_sha256_at_commit(commit, relative) == expected


def test_round74_v64_binds_new_cohort_without_model_selection_drift() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    cohort = design["prospective_cohort"]
    delta = design["model_contract_delta"]

    assert cohort["plan_sha256"] == (
        "4373c432bcabb10071a0e60a90bf7ac99299139f223eb2a5afff920e6b78deb4"
    )
    assert cohort["failed_predecessor_data_permitted"] is False
    assert cohort["capture_roles"] == {"training": 120, "tuning": 24, "test": 24}
    assert delta["candidate_panel_changed"] is False
    assert delta["feature_contract_changed"] is False
    assert delta["target_contract_changed"] is False
    assert delta["default_profile"] == "conservative"
    assert delta["model_predictions_are_unlevered"] is True


def test_round74_v64_blocks_claims_without_execution_and_market_evidence() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    execution = design["execution_evidence_delta"]
    authority = design["authority"]
    decisions = design["research_decisions"]

    assert execution["testnet_only"] is True
    assert execution["execution_calibration_complete_now"] is False
    assert execution["missing_execution_evidence_policy"].startswith(
        "block target assembly"
    )
    assert decisions["predictive_accuracy_required"] is True
    assert decisions["predictive_accuracy_sufficient_for_promotion"] is False
    assert decisions["after_cost_tradability_required"] is True
    assert decisions["new_architecture_added_without_representative_evidence"] is False
    assert set(authority.values()) == {False}
