from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = (
    ROOT
    / "docs/model-research/action-value/polymarket-post-observation-maker-window-gate-v1-2026-08-26.json"
)
EXPECTED_HASH = "03dcb88790b96bcaed6a58dc921abff5244e3b2eecd3a39e8f4e82c412f49392"
PROSPECTIVE_PATH = (
    ROOT
    / "docs/model-research/action-value/polymarket-post-observation-prospective-v2-2026-08-26.json"
)
EXPECTED_PROSPECTIVE_HASH = (
    "079925ec06eda0cdfc5851d71d7fc76df96de6f03883bcc70edc0f36da28d421"
)
PROSPECTIVE_V3_PATH = (
    ROOT
    / "docs/model-research/action-value/polymarket-post-observation-prospective-v3-2026-08-26.json"
)
EXPECTED_PROSPECTIVE_V3_HASH = (
    "7b9f21cf3c1a65a709d5e52867877b9d79a9bf17f7a4df448a2fb92a32757e16"
)
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_REGISTRY_HASH = (
    "0d038efb7a32d61b97f7efc0e0643b88fcc3dd73e6c5b702ffd0c516f59bbf9d"
)


def _load() -> dict[str, object]:
    return json.loads(PATH.read_bytes())


def _embedded_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    canonical = json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_gate_is_hash_bound_and_grants_no_authority() -> None:
    artifact = _load()
    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _embedded_hash(artifact) == EXPECTED_HASH
    assert artifact["authority"]["accepted_edge"] is False
    assert artifact["authority"]["orders_placed"] is False
    assert artifact["authority"]["profitability_claim"] is False


def test_public_causal_rows_reconstruct_direction_slices_and_gross() -> None:
    artifact = _load()
    rows = artifact["evidence_rows"]
    summary = artifact["economic_summary"]
    assert len(rows) == summary["complete_conditions"] == 10
    assert all(
        row["oracle_receipt_delay_ms"]
        < row["first_winner_bid_growth_delay_ms"]
        <= row["first_later_winner_sell_fill_delay_ms"]
        for row in rows
    )
    up = [row for row in rows if row["outcome"] == "Up"]
    down = [row for row in rows if row["outcome"] == "Down"]
    assert len(up) == summary["up_condition_count"] == 3
    assert len(down) == summary["down_condition_count"] == 7
    assert sum(Decimal(row["observed_gross_pusd"]) for row in up) == Decimal(
        summary["up_condition_observed_gross_pusd"]
    )
    assert sum(Decimal(row["observed_gross_pusd"]) for row in down) == Decimal(
        summary["down_condition_observed_gross_pusd"]
    )
    assert sum(Decimal(row["observed_gross_pusd"]) for row in rows) == Decimal(
        summary["observed_gross_pusd"]
    )


def test_gate_preserves_the_decisive_execution_unknown() -> None:
    artifact = _load()
    assert (
        artifact["candidate_status"]
        == "high_priority_conditional_execution_lead_not_an_accepted_edge"
    )
    limitations = " ".join(artifact["unresolved_gates"])
    assert "does not" in limitations
    assert "authenticated order" in limitations
    assert "one degraded BTC hour" in limitations


def test_prospective_cross_asset_result_rejects_strong_fill_recurrence() -> None:
    artifact = json.loads(PROSPECTIVE_PATH.read_bytes())
    assert artifact["result_sha256"] == EXPECTED_PROSPECTIVE_HASH
    assert _embedded_hash(artifact) == EXPECTED_PROSPECTIVE_HASH
    summary = artifact["economic_summary"]
    assert summary["complete_conditions"] == 3
    assert summary["conditions_with_post_observation_winner_bid_growth"] == 3
    assert summary["conditions_with_later_winner_sell_fills"] == 1
    assert Decimal(summary["public_observed_gross_pusd"]) == Decimal("0.01022")
    assert artifact["verdict"]["cross_asset_public_recurrence"] is False
    assert artifact["verdict"]["accepted_edge"] is False


def test_multi_interval_result_rejects_the_unchanged_public_mechanism() -> None:
    artifact = json.loads(PROSPECTIVE_V3_PATH.read_bytes())
    assert artifact["result_sha256"] == EXPECTED_PROSPECTIVE_V3_HASH
    assert _embedded_hash(artifact) == EXPECTED_PROSPECTIVE_V3_HASH
    assert artifact["gate_results"] == {
        "all_complete_conditions_have_post_observation_winner_bid_growth": True,
        "capture_status_complete": True,
        "each_asset_has_both_up_and_down_outcomes": True,
        "minimum_complete_conditions_per_asset": False,
        "minimum_later_winner_sell_fill_fraction_per_asset": False,
        "overall_pass": False,
        "positive_public_observed_gross_per_asset": False,
        "stream_error_count_zero": True,
        "stream_gap_count_zero": True,
    }
    assert (
        artifact["per_asset"]["BTC"]["later_winner_sell_fill_condition_fraction"]
        == "0.5"
    )
    assert (
        artifact["per_asset"]["ETH"]["later_winner_sell_fill_condition_fraction"]
        == "0.5"
    )
    assert (
        artifact["per_asset"]["SOL"]["later_winner_sell_fill_condition_fraction"] == "0"
    )
    assert artifact["verdict"]["status"] == "terminal_public_recurrence_failure"


def test_registry_marks_the_public_mechanism_terminal() -> None:
    registry = json.loads(REGISTRY_PATH.read_bytes())
    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_HASH
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 37))
    lead = next(
        row
        for row in hypotheses
        if row["mechanism"] == "post_observation_oracle_to_clob_close_maker_window"
    )
    assert lead["priority_rank"] == 7
    assert lead["market_direction_forecast_required"] is False
    assert lead["canonical_artifacts"][0]["result_sha256"] == EXPECTED_HASH
    assert lead["canonical_artifacts"][1]["result_sha256"] == EXPECTED_PROSPECTIVE_HASH
    assert (
        lead["canonical_artifacts"][2]["result_sha256"] == EXPECTED_PROSPECTIVE_V3_HASH
    )
    assert lead["current_status"].startswith("terminal_public_recurrence_failure")
    assert (
        lead["retry_trigger"]
        == "material_program_term_or_execution_architecture_change"
    )
