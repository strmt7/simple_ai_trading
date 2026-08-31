from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _assert_bound(path: Path, field: str) -> dict[str, object]:
    payload = _load(path)
    assert _hash(payload, field) == payload[field]
    return payload


def test_goalscorer_total_graph_is_frozen_complete_and_terminal() -> None:
    contract = _assert_bound(
        ACTION / "polymarket-soccer-goalscorer-total-contract-v1-2026-08-30.json",
        "contract_sha256",
    )
    result = _assert_bound(
        ACTION / "polymarket-soccer-goalscorer-total-result-v1-2026-08-30.json",
        "result_sha256",
    )
    assert hashlib.sha256(
        (ROOT / contract["retained_source"]["path"]).read_bytes()
    ).hexdigest() == contract["retained_source"]["sha256"]
    assert hashlib.sha256(
        (ROOT / contract["implementation"]["path"]).read_bytes()
    ).hexdigest() == contract["implementation"]["sha256"]
    assert result["population"] == {
        "complete_base_count": 5,
        "relation_count": 43,
        "side_specific_price_complete_count": 43,
        "side_specific_price_missing_count": 0,
        "strict_sub_floor_count": 0,
    }
    assert result["best_complete_relation"]["rejection_proxy_sum_pUSD"] == "1.9"
    assert result["strict_sub_floor_relations"] == []
    assert result["adjudication"]["current_book_request_authorized"] is False
    assert result["authority"]["network_requests"] == 0


def test_corner_graph_preserves_failures_and_exhausts_frozen_population() -> None:
    contract_v1 = _assert_bound(
        ACTION / "polymarket-soccer-corner-graph-contract-v1-2026-08-30.json",
        "contract_sha256",
    )
    failure_v1 = _assert_bound(
        ACTION / "polymarket-soccer-corner-graph-failure-v1-2026-08-30.json",
        "failure_sha256",
    )
    contract_v2 = _assert_bound(
        ACTION / "polymarket-soccer-corner-graph-contract-v2-2026-08-30.json",
        "contract_sha256",
    )
    failure_v2 = _assert_bound(
        ACTION / "polymarket-soccer-corner-graph-failure-v2-2026-08-30.json",
        "failure_sha256",
    )
    contract_v3 = _assert_bound(
        ACTION / "polymarket-soccer-corner-graph-contract-v3-2026-08-30.json",
        "contract_sha256",
    )
    result = _assert_bound(
        ACTION / "polymarket-soccer-corner-graph-result-v3-2026-08-30.json",
        "result_sha256",
    )

    assert failure_v1["failure"]["failed_market_id"] == "3842942"
    assert failure_v1["failure"]["result_file_created"] is False
    assert failure_v2["failure"]["error_message"] == (
        "zip() argument 2 is shorter than argument 1"
    )
    assert failure_v2["failure"]["result_file_created"] is False
    for contract in (contract_v1, contract_v2, contract_v3):
        assert hashlib.sha256(
            (ROOT / contract["retained_source"]["path"]).read_bytes()
        ).hexdigest() == contract["retained_source"]["sha256"]
        assert hashlib.sha256(
            (ROOT / contract["implementation"]["path"]).read_bytes()
        ).hexdigest() == contract["implementation"]["sha256"]

    assert result["population"] == {
        "adjacent_full_corner_interval_implies_parity_count": 42,
        "complete_event_count": 7,
        "first_half_corner_total_monotone_count": 21,
        "full_corner_total_monotone_count": 147,
        "half_partition_both_over_implies_full_over_count": 189,
        "half_partition_both_under_implies_full_under_count": 371,
        "relation_count": 1820,
        "second_half_corner_total_monotone_count": 21,
        "side_specific_price_complete_count": 1820,
        "side_specific_price_missing_count": 0,
        "strict_sub_floor_count": 0,
        "team_corner_total_monotone_count": 84,
        "team_partition_both_over_implies_full_over_count": 231,
        "team_partition_both_under_implies_full_under_count": 714,
    }
    assert result["best_complete_relation"]["rejection_proxy_sum_pUSD"] == "1.1"
    assert result["strict_sub_floor_relations"] == []
    assert result["adjudication"]["current_book_request_authorized"] is False
    assert result["authority"]["network_requests"] == 0


def test_corner_partition_rules_are_explicit_in_the_immutable_source() -> None:
    source = _load(
        ROOT
        / "data/polymarket-near-expiry-negrisk-complete-set-catalog-v1/raw/events.json"
    )
    events = [
        event
        for event in source["events"]
        if event["slug"].endswith("-total-corners")
    ]
    assert len(events) == 7
    for event in events:
        for market in event["markets"]:
            market_type = market["sportsMarketType"]
            description = " ".join(market["description"].split())
            if market_type == "total_corners":
                assert "combining corners for both teams" in description
                assert "first 90 minutes of regular play plus stoppage time" in description
            elif market_type == "soccer_first_half_total_corners":
                assert "taken by both teams in the first half" in description
                assert "first 45 minutes of regular play plus first-half stoppage time" in description
                assert "Second-half corners, extra time, and penalty shootouts do not count" in description
            elif market_type == "soccer_second_half_total_corners":
                assert "taken by both teams in the second half" in description
                assert "after the second half begins" in description
                assert "First-half corners, extra time, and penalty shootouts do not count" in description
            elif market_type == "soccer_team_total_corners":
                team = market["groupItemTitle"].split(" Corners: O/U ", 1)[0]
                assert f"based solely on corners taken by {team}" in description
                assert "not the combined number of corners taken by both teams" in description
                assert "first 90 minutes of regular play plus stoppage time" in description
            elif market_type == "soccer_game_corners_odd_even":
                assert "total number of corners taken by both teams" in description
                assert "Zero corners is considered even" in description


def test_goalscorer_and_corner_graphs_are_bound_to_rank_31_and_terminal() -> None:
    goalscorer = _load(
        ACTION / "polymarket-soccer-goalscorer-total-result-v1-2026-08-30.json"
    )
    corners = _load(
        ACTION / "polymarket-soccer-corner-graph-result-v3-2026-08-30.json"
    )
    registry = _assert_bound(REGISTRY, "result_sha256")
    rank_31 = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 31
    )
    artifacts = rank_31["canonical_artifacts"]
    assert {
        "path": "docs/model-research/action-value/polymarket-soccer-goalscorer-total-result-v1-2026-08-30.json",
        "result_sha256": goalscorer["result_sha256"],
    } in artifacts
    assert {
        "path": "docs/model-research/action-value/polymarket-soccer-corner-graph-result-v3-2026-08-30.json",
        "result_sha256": corners["result_sha256"],
    } in artifacts
    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    assert terminal[
        "polymarket_retained_soccer_anytime_goalscorer_to_Over_0_5_graph_2026_08_30"
    ]["canonical_result_sha256"] == goalscorer["result_sha256"]
    assert terminal[
        "polymarket_retained_soccer_corner_monotone_additive_partition_and_parity_graph_2026_08_30"
    ]["canonical_result_sha256"] == corners["result_sha256"]
