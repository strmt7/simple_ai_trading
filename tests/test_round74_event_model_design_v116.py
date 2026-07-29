from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v116.json"
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


def _normalized_source_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )
    payload = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_round74_v116_binds_ai_uplift_confidence_gate() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    base = design["base_design"]
    base_path = ROOT / base["path"]
    assert base["file_sha256"] == hashlib.sha256(base_path.read_bytes()).hexdigest()
    assert (
        base["design_sha256"]
        == json.loads(base_path.read_text(encoding="ascii"))["design_sha256"]
    )
    implementation_commit = design["implementation_git_commit"]
    for source in design["source_binding"].values():
        if not isinstance(source, dict):
            continue
        assert source["sha256"] == _normalized_source_sha256(
            implementation_commit,
            source["path"],
        )


def test_round74_v116_requires_dependent_run_uplift_without_claiming_edge() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    contract = design["statistical_contract"]
    delta = design["declared_delta"]
    verification = design["verification"]

    assert contract["sampling_unit"] == "whole_independently_adjudicated_capture_run"
    assert contract["bootstrap_samples"] == 10000
    assert contract["familywise_confidence"] == 0.95
    assert contract["pinned_ai_model_comparisons"] == 2
    assert contract["bonferroni_per_model_confidence"] == 0.975
    assert contract["promotion_gate"] == (
        "per_model_mean_ci_lower_strictly_greater_than_zero"
    )
    assert contract["trade_rows_resampled_independently"] is False
    assert delta["bootstrap_evidence_hash_bound_in_report"] is True
    assert delta["bootstrap_evidence_recomputed_on_load"] is True
    assert delta["sparse_single_avoided_loss_rejected"] is True
    assert verification["unique_tests_passed"] == 51
    assert verification["representative_ai_qualification_performed"] is False
    assert verification["real_ai_model_called"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert set(design["authority"].values()) == {False}
