from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / (
    "docs/model-research/action-value/"
    "accepted-market-independent-yield-frontier-v1-2026-08-30.json"
)
RESULT_HASH = "bc2db7e81a2e14fee68dc9f57041d226843fefabc0cc47e81db10985e04d84d3"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_frontier_and_all_source_artifacts_reconstruct() -> None:
    frontier = _load(FRONTIER)

    assert frontier["result_sha256"] == RESULT_HASH
    assert _canonical_hash(frontier, "result_sha256") == RESULT_HASH
    for collection in (
        "source_artifacts",
        "supplementary_current_stress_artifacts",
    ):
        for source in frontier[collection]:
            payload = (ROOT / source["path"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == source["file_sha256"]
            assert json.loads(payload)["result_sha256"] == source["result_sha256"]


def test_population_is_complete_without_inflating_acceptance() -> None:
    frontier = _load(FRONTIER)
    population = frontier["population"]
    decision = frontier["portfolio_decision"]

    assert population["registry_accepted_edge_count"] == 29
    assert population["yield_and_capital_efficiency_edges_included"] == 9
    assert (
        population[
            "organic_flow_fee_referral_creator_and_financing_cost_overlays_excluded"
        ]
        == 20
    )
    assert population["population_complete_for_registry_accepted_yield_edges"] is True
    assert decision["new_accepted_edge_count"] == 0
    assert decision["accepted_edge_count_after_frontier"] == 29
    assert population["owned_realized_cash_recurrence_edge_count"] == 1
    assert population["public_after_all_incremental_cost_positive_floor_edge_count"] == 0
    assert population["owned_account_qualified_edge_count"] == 0
    assert decision["current_owned_after_all_incremental_cost_profitable_edge_count"] == 0
    assert decision["deployment_ready_edge_count"] == 0


def test_current_web_checks_are_discovery_only() -> None:
    frontier = _load(FRONTIER)

    assert frontier["authority"]["venue_HTTP_requests"] == 0
    assert frontier["authority"]["source_bound_economic_inputs_refreshed"] == 0
    assert frontier["authority"]["public_web_discovery_batches"] == 3
    assert all(
        row["material_yield_or_structural_profitability_trigger_found"] is False
        for row in frontier["discovery_only_current_source_checks"]
    )


def test_profitability_ladder_separates_scoped_acceptance_from_owned_profit() -> None:
    frontier = _load(FRONTIER)
    rows = frontier["profitability_evidence_ladder"]

    assert [row["rank"] for row in rows] == [1, 2, 3, 4]
    assert sum(row["edge_count"] for row in rows[:2]) == 9
    assert rows[0]["edge_ids"] == ["polymarket_complete_set_holding_yield"]
    assert rows[2]["edge_count"] == 0
    assert rows[3]["edge_count"] == 0
    assert frontier["portfolio_decision"]["immediate_research_spend"].startswith(
        "none_no_exact_public_retry_trigger"
    )


def test_polymarket_leads_on_realized_stability_not_headline_apr() -> None:
    frontier = _load(FRONTIER)
    rows = frontier["evidence_strength_frontier"]
    first = rows[0]
    economics = first["economic_evidence"]

    assert [row["rank"] for row in rows] == list(range(1, 10))
    assert first["edge_id"] == "polymarket_complete_set_holding_yield"
    assert economics["positive_daily_payments"] == economics["possible_daily_payments"]
    assert Decimal(economics["observed_reward_pusd"]) > 0
    assert economics["positive_after_three_percent_alternative_yield"] is True
    assert (
        economics["positive_after_three_point_two_five_percent_alternative_yield"]
        is False
    )
    assert (
        frontier["portfolio_decision"]["strongest_stable_edge"]
        == "polymarket_complete_set_holding_yield"
    )


def test_fixed_bonus_urgency_is_expiry_ordered() -> None:
    frontier = _load(FRONTIER)
    rows = frontier["current_fixed_bonus_urgency_frontier"]

    assert [row["priority"] for row in rows] == [1, 2, 3, 4]
    assert [row["published_end_utc"] for row in rows] == sorted(
        row["published_end_utc"] for row in rows
    )
    usd1 = next(
        row for row in rows if row["edge_id"] == "binance_usd1_simple_earn_fixed_bonus"
    )
    assert "2026_09_02" in usd1["reason"]
    stress = frontier["portfolio_decision"][
        "usd1_current_remaining_horizon_stress"
    ]
    assert stress["stable_profit_proved"] is False
    assert Decimal(stress["margin_before_unproved_other_costs_bips"]) > 0
