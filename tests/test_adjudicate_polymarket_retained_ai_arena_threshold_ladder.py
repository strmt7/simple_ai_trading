from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.adjudicate_polymarket_retained_ai_arena_threshold_ladder import (
    _preflight_pair,
    _screen,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs/model-research/polymarket/ai-arena-score-retained-threshold-ladder-contract-v1-2026-09-01.json"
)
RESULT = (
    ROOT
    / "docs/model-research/polymarket/ai-arena-score-retained-threshold-ladder-result-v1-2026-09-01.json"
)
FAILURE = (
    ROOT
    / "docs/model-research/polymarket/ai-arena-score-retained-threshold-ladder-preflight-failure-v1-2026-09-01.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
AUDIT = (
    ROOT
    / "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _market(market_id: str, threshold: str, start: str) -> dict[str, object]:
    return {
        "id": market_id,
        "conditionId": f"condition-{market_id}",
        "question": f"Reach {threshold}?",
        "groupItemTitle": threshold,
        "description": "common at least score source rule",
        "startDate": start,
        "outcomes": '["Yes","No"]',
        "clobTokenIds": f'["yes-{market_id}","no-{market_id}"]',
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "bestAsk": "0.4",
        "bestBid": "0.3",
        "feesEnabled": False,
        "feeSchedule": {"rate": "0", "exponent": "1", "takerOnly": True},
        "orderMinSize": 5,
        "orderPriceMinTickSize": 0.001,
    }


def _population(higher_start: str = "2026-01-02T00:00:00Z") -> dict[str, object]:
    markets = [
        _market("1", "10", "2026-01-01T00:00:00Z"),
        _market("2", "20", higher_start),
    ]
    return {
        "events": [
            {
                "id": "event-1",
                "slug": "event",
                "title": "title",
                "markets": markets,
            }
        ]
    }


def _contract() -> dict[str, object]:
    return {
        "event_id": "event-1",
        "event_slug": "event",
        "event_title": "title",
        "cutoff_date": "2026-09-30",
        "required_rule_fragments": ["at least", "source rule"],
        "markets": [
            {
                "market_id": "1",
                "threshold": "10",
                "start_utc": "2026-01-01T00:00:00Z",
                "active": True,
                "closed": False,
                "acceptingOrders": True,
                "enableOrderBook": True,
            },
            {
                "market_id": "2",
                "threshold": "20",
                "start_utc": "2026-01-02T00:00:00Z",
                "active": True,
                "closed": False,
                "acceptingOrders": True,
                "enableOrderBook": True,
            },
        ],
        "pairs": [
            {
                "pair": "10_to_20",
                "lower_market_id": "1",
                "lower_threshold": "10",
                "higher_market_id": "2",
                "higher_threshold": "20",
            }
        ],
    }


def test_preflight_pair_requires_lower_creation_window_to_cover_higher() -> None:
    population = _population()
    event = population["events"][0]
    by_id = {market["id"]: market for market in event["markets"]}
    lower, higher = _preflight_pair(event, _contract()["pairs"][0], by_id)
    assert lower["id"] == "1"
    assert higher["id"] == "2"


def test_preflight_pair_rejects_an_uncovered_lower_creation_gap() -> None:
    population = _population("2025-12-31T00:00:00Z")
    event = population["events"][0]
    by_id = {market["id"]: market for market in event["markets"]}
    with pytest.raises(RuntimeError, match="does not cover"):
        _preflight_pair(event, _contract()["pairs"][0], by_id)


def test_screen_prices_lower_yes_plus_higher_no() -> None:
    rows = _screen(_population(), _contract())
    assert len(rows) == 1
    assert rows[0]["metadata_cost_pUSD_per_share"] == "1.1"
    assert rows[0]["passes_strict_metadata_gate"] is False
    assert [leg["outcome"] for leg in rows[0]["legs"]] == ["Yes", "No"]


def test_frozen_result_is_hash_bound_and_stops_before_books() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    failure = _load(FAILURE)
    assert contract["contract_sha256"] == (
        "8246a5ebc592cdcdd847e1717af26cbf57db1b51d63edd7712b69e1870f617d9"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "cec42898652936f8eeea78a956f8a9f6e55916a57cd1adc56485f1eb9636387f"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert failure["result_sha256"] == (
        "36475331926c94e259959ec71d38015665bc1222389b024c2a9a225a345b4da2"
    )
    assert _canonical_hash(failure, "result_sha256") == failure["result_sha256"]
    screen = result["screen"]
    assert screen["market_count"] == 5
    assert screen["active_market_count"] == 4
    assert screen["excluded_creation_gap_pair_count"] == 2
    assert screen["pair_count"] == 4
    assert screen["side_specific_price_available_count"] == 4
    assert screen["strict_metadata_candidate_count"] == 0
    assert screen["fee_and_one_tick_candidate_count"] == 0
    assert screen["best_pair"]["lower_threshold"] == "1540"
    assert screen["best_pair"]["higher_threshold"] == "1550"
    assert screen["best_pair"]["metadata_cost_pUSD_per_share"] == "1.035"
    assert screen["best_pair"]["after_fee_one_tick_profit_floor_pUSD"] == "-0.19670"
    assert result["authority"]["network_requests"] == 0
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["protected_capture_touched"] is False


def test_registry_and_durability_bind_ai_arena_terminal() -> None:
    registry = _load(REGISTRY)
    audit = _load(AUDIT)
    result = _load(RESULT)
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert _canonical_hash(audit, "result_sha256") == audit["result_sha256"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_AI_Arena_Overall_creation_safe_threshold_ladder_2026_09_01"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
    assert len(registry["terminal_do_not_repeat"]) == 158
    assert (
        audit["source_binding"]["registry_result_sha256"] == registry["result_sha256"]
    )
    assert (
        audit["decision"]["stable_current_account_qualified_after_all_cost_edge_count"]
        == 0
    )
