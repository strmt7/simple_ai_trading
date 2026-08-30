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
RESULT_HASH = "e3e6941790079587e3a21bf0894e165963cc871b9165821fd7b00600fc3c4dec"


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

    assert population["registry_accepted_edge_count"] == 24
    assert population["yield_and_capital_efficiency_edges_included"] == 9
    assert (
        population[
            "organic_flow_fee_referral_creator_and_financing_cost_overlays_excluded"
        ]
        == 15
    )
    assert population["population_complete_for_registry_accepted_yield_edges"] is True
    assert decision["new_accepted_edge_count"] == 0
    assert decision["accepted_edge_count_after_frontier"] == 24
    assert decision["deployment_ready_edge_count"] == 0


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
