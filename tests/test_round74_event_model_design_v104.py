from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v104.json"
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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round74_v104_binds_action_validity_implementation() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")

    assert claimed == _canonical_sha256(design)
    base = design["base_design"]
    base_path = ROOT / base["path"]
    assert base["file_sha256"] == _file_sha256(base_path)
    assert (
        base["design_sha256"]
        == json.loads(base_path.read_text(encoding="ascii"))["design_sha256"]
    )
    for source in design["source_binding"].values():
        if isinstance(source, dict):
            assert source["sha256"] == _file_sha256(ROOT / source["path"])


def test_round74_v104_removes_arbitrary_same_entry_suppression() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    verification = design["verification"]

    assert delta["caller_supplied_ai_latency_budget_permitted"] is False
    assert delta["accepted_ai_decision_zeroed_by_diagnostic_latency"] is False
    assert delta["expired_action_is_retained_as_paired_zero_exposure"] is True
    assert delta["eligible_action_requires_exact_delayed_l2_replay"] is True
    assert delta["action_validity_maximum_ns"] == 30_000_000_000
    assert delta["ai_may_create_side_or_horizon"] is False
    assert verification["focused_tests_passed"] == 145
    assert verification["representative_ai_reviews_executed"] is False
    assert verification["sealed_test_accessed"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert set(design["authority"].values()) == {False}
