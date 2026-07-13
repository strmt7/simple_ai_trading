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
    / "round-041-prequential-meta-label-ai-design.json"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_round41_design_is_hash_bound_prequential_and_fail_closed() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    canonical = dict(design)
    claimed = canonical.pop("design_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert design["round"] == 41
    governance = design["governance"]
    assert governance["selection_contaminated"] is True
    assert governance["development_only"] is True
    assert governance["single_prespecified_architecture"] is True
    assert governance["future_opportunity_set_ranking_permitted"] is False
    assert governance["2025_h2_selection_confirmation_access_permitted"] is False
    assert governance["2026_terminal_access_permitted"] is False
    primary = design["prequential_primary_panel"]
    assert len(primary["target_months"]) == 14
    assert primary["primary_models"] == 42
    assert primary["target_month_predictions_strictly_out_of_sample"] is True
    assert primary["prediction_panel_storage"].startswith("single in-memory")
    walk = design["meta_walk_forward"]
    assert len(walk["evaluation_months"]) == 6
    assert walk["all_primary_inputs_strictly_out_of_sample"] is True
    assert walk["example_for_2024_07"]["meta_fit"].startswith("2023-11-01")
    assert design["meta_label_model"]["models"] == 6
    assert design["meta_label_model"]["feature_count"] == 81
    capacity = design["causal_capacity_and_threshold_contract"]
    assert capacity["threshold_cells_total"] == 216
    assert capacity["maximum_entries_per_symbol_per_utc_day"] == 8
    assert design["ai_ablation_contract"]["candidate_model"] == (
        "DianJin/DianJin-R1-7B"
    )
    assert design["ai_ablation_contract"]["entry_gate"] == (
        "aggregate_viability_gate passes in full"
    )
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        assert design[field] is False
