from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.impact_absorption_ai_review_preparation import (
    ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_ai_uplift import (
    ROUND74_AI_PRETEST_QUALIFICATION_SCHEMA_VERSION,
    ROUND74_AI_UPLIFT_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_sealed_evaluation import (
    ROUND74_SEALED_EVALUATION_SCHEMA_VERSION,
)
from simple_ai_trading.round74_ai_qualification_operator import (
    ROUND74_AI_QUALIFICATION_OPERATOR_SCHEMA_VERSION,
)
from simple_ai_trading.round74_segmented_development_operator import (
    ROUND74_SEGMENTED_QUALIFIED_DEVELOPMENT_SCHEMA_VERSION,
)
from simple_ai_trading.round74_segmented_development_runtime import (
    ROUND74_SEGMENTED_DEVELOPMENT_RUN_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v105.json"
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


def test_round74_v105_binds_deadline_aware_ai_queue_implementation() -> None:
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
        assert source["sha256"] == _file_sha256(ROOT / source["path"])


def test_round74_v105_rejects_stale_queue_work_without_financial_claims() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    delta = design["declared_delta"]
    schemas = design["schema_contract"]
    verification = design["verification"]

    assert delta["queue_timeout_action"] == "reject_before_model_inference"
    assert delta["expired_request_invokes_injected_model_runner"] is False
    assert delta["expired_request_retained_as_paired_observation"] is True
    assert delta["expired_request_exposure_bps"] == 0
    assert delta["expired_request_counts_against_ai_qualification"] is True
    assert delta["candidate_models_are_independent_overlay_candidates"] is True
    assert delta["candidate_models_are_treated_as_concurrent_ensemble"] is False
    assert schemas == {
        "ai_review_panel": ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION,
        "ai_uplift_development": ROUND74_AI_UPLIFT_SCHEMA_VERSION,
        "ai_pretest_qualification": (ROUND74_AI_PRETEST_QUALIFICATION_SCHEMA_VERSION),
        "ai_qualification_operator": (ROUND74_AI_QUALIFICATION_OPERATOR_SCHEMA_VERSION),
        "sealed_evaluation": ROUND74_SEALED_EVALUATION_SCHEMA_VERSION,
        "segmented_qualified_development": (
            ROUND74_SEGMENTED_QUALIFIED_DEVELOPMENT_SCHEMA_VERSION
        ),
        "segmented_development_run": (ROUND74_SEGMENTED_DEVELOPMENT_RUN_SCHEMA_VERSION),
    }
    assert verification["focused_tests_passed"] == 158
    assert verification["representative_ai_reviews_executed"] is False
    assert verification["local_multibillion_parameter_model_invoked"] is False
    assert verification["sealed_test_accessed"] is False
    assert design["evidence_boundary"]["ai_uplift_claim"] is False
    assert design["evidence_boundary"]["profitability_claim"] is False
    assert set(design["authority"].values()) == {False}
