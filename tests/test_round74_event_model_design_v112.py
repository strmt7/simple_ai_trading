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
    / "round-074-event-sequence-model-design-v112.json"
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


def test_round74_v112_binds_ai_subgroup_safeguard_implementation() -> None:
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


def test_round74_v112_rejects_cell_harm_without_claiming_ai_edge() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    verification = design["verification"]

    assert delta["development_report_schema_version"] == (
        "round-074-ai-uplift-development-v14"
    )
    assert delta["sealed_evaluation_schema_version"] == (
        "round-074-sealed-evaluation-v22"
    )
    assert tuple(delta["new_paired_panel_dimensions"]) == (
        "capture_run",
        "symbol",
        "forecast_horizon_seconds",
    )
    assert delta["negative_cell_delta_may_be_offset_by_other_cells"] is False
    assert delta["development_gate_reasons_recomputed_from_evidence"] is True
    assert delta["sealed_gate_reasons_recomputed_from_evidence"] is True
    assert delta["stored_gate_flags_trusted_without_semantic_recalculation"] is False
    assert delta["forward_passes_added"] == 0
    assert delta["backward_passes_added"] == 0
    assert delta["market_data_reads_added"] == 0
    assert verification["unique_tests_passed"] == 164
    assert verification["new_development_masking_counterexample_rejected"] is True
    assert verification["new_sealed_masking_counterexample_rejected"] is True
    assert (
        verification["run_symbol_horizon_gate_is_decisive_in_masking_counterexamples"]
        is True
    )
    assert verification["stored_gate_reason_tampering_rejected"] is True
    assert verification["representative_training_performed"] is False
    assert verification["representative_ai_reviews_executed"] is False
    assert verification["sealed_test_accessed"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert set(design["authority"].values()) == {False}
