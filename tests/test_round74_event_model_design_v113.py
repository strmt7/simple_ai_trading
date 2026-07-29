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
    / "round-074-event-sequence-model-design-v113.json"
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


def test_round74_v113_binds_cell_path_risk_implementation() -> None:
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


def test_round74_v113_rejects_hidden_path_risk_without_edge_claim() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    verification = design["verification"]

    assert delta["development_report_schema_version"] == (
        "round-074-ai-uplift-development-v15"
    )
    assert delta["sealed_evaluation_schema_version"] == (
        "round-074-sealed-evaluation-v23"
    )
    assert delta["new_paired_measure"] == (
        "aggregate_capital_scaled_maximum_adverse_excursion_bps"
    )
    assert delta["adverse_excursion_delta_definition"] == "ai_minus_baseline"
    assert delta["adverse_excursion_noninferiority_threshold_bps"] == 0.0
    assert delta["positive_delta_may_be_offset_by_other_cells"] is False
    assert delta["global_maximum_drawdown_noninferiority_gate_retained"] is True
    assert delta["after_cost_pnl_noninferiority_gates_retained"] is True
    assert delta["development_gate_reasons_recomputed_from_evidence"] is True
    assert delta["sealed_gate_reasons_recomputed_from_evidence"] is True
    assert delta["forward_passes_added"] == 0
    assert delta["backward_passes_added"] == 0
    assert delta["market_data_reads_added"] == 0
    assert verification["unique_tests_passed"] == 184
    assert verification["new_development_path_risk_counterexample_rejected"] is True
    assert verification["new_sealed_path_risk_counterexample_rejected"] is True
    assert verification["fine_grained_adverse_excursion_gate_is_decisive"] is True
    assert verification["stored_cell_evidence_tampering_rejected"] is True
    assert verification["representative_training_performed"] is False
    assert verification["representative_ai_reviews_executed"] is False
    assert verification["sealed_test_accessed"] is False
    assert design["evidence_boundary"]["financial_edge_tested"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert set(design["authority"].values()) == {False}
