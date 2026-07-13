from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.run_causal_meta_label_capacity_ai import _validate_design


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-040-causal-meta-label-capacity-ai-design.json"
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


def test_round40_design_is_hash_bound_and_fail_closed() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    canonical = dict(design)
    claimed = canonical.pop("design_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert design["round"] == 40
    assert design["governance"]["selection_contaminated"] is True
    assert design["governance"]["development_only"] is True
    assert design["governance"]["future_opportunity_set_ranking_permitted"] is False
    assert design["governance"]["2025_h2_selection_confirmation_access_permitted"] is False
    assert design["governance"]["2026_terminal_access_permitted"] is False
    assert design["governance"]["leverage_permitted"] is False
    assert design["source_contract"]["symbols"] == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]
    assert design["source_contract"]["dataset_rows"] == 1_098_105
    assert design["target_and_accounting_contract"]["round_trip_execution_charge_bps"] == 12.0
    assert design["primary_model"]["models"] == 18
    assert design["meta_label_model"]["models"] == 6
    assert design["meta_label_model"]["in_sample_primary_predictions_permitted"] is False
    capacity = design["causal_capacity_and_threshold_contract"]
    assert capacity["threshold_cells_total"] == 216
    assert capacity["maximum_entries_per_symbol_per_utc_day"] == 8
    assert capacity["future candidate rank_or_daily_maximum_confidence_selection_permitted"] is False
    assert capacity["no_passing_threshold_action"].startswith("route no evaluation actions")
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


def test_round40_runner_accepts_exact_frozen_design() -> None:
    design, claimed = _validate_design(DESIGN)

    assert design["round"] == 40
    assert claimed == design["design_sha256"]
