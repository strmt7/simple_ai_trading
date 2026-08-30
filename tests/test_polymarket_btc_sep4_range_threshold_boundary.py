from __future__ import annotations

import json
from pathlib import Path

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import _canonical_hash


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-btc-sep4-range-threshold-boundary-adjudication-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def test_boundary_adjudication_is_hash_bound_and_fail_closed() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))

    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["mechanical_boundary_proof"]["exact_subset_indicator_equality"] is False
    assert result["adjudication"]["status"] == (
        "current_pair_rejected_before_outcome_sensitive_market_data"
    )
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["venue_http_requests"] == 0
    assert result["adjudication"]["protected_capture_touched"] is False


def test_truth_table_proves_dominance_but_not_equality() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    states = result["mechanical_boundary_proof"]["states"]

    assert [row["threshold_yes_T"] for row in states] == [0, 0, 1]
    assert [row["cumulative_upper_range_T"] for row in states] == [0, 1, 1]
    assert [row["no_threshold_plus_upper_range_payout"] for row in states] == [
        1,
        2,
        1,
    ]


def test_registry_routes_the_corrected_family_without_mutable_global_pins() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="ascii"))
    result = json.loads(RESULT.read_text(encoding="ascii"))

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    rank_31 = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 31
    )
    assert any(
        artifact["result_sha256"] == result["result_sha256"]
        for artifact in rank_31["canonical_artifacts"]
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_Bitcoin_September_4_cross_event_range_threshold_boundary_gate_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
    assert "strict_above_threshold_YES" in terminal["reason"]
