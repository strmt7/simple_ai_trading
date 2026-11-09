from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json

import pytest

from tools import screen_future_nfl_pair_ladders as module


MINIMUM, MAXIMUM = "2026-11-10T00:00:00Z", "2026-11-10T23:59:59Z"


def event(count=3, props=0):
    common = {"active": True, "closed": False, "acceptingOrders": True}
    markets = [
        {**common, "id": "1", "sportsMarketType": "moneyline", "outcomes": '["A", "B"]'}
    ]
    for i in range(count):
        threshold = 31 + i
        schedule = "In the upcoming NFL game, scheduled for November 9 at 8:15 PM ET:"
        description = (
            f'{schedule}\nThis market will resolve to "Over" if the A and B '
            f"combine to score {threshold} or more points in this game. "
            f'If the combined total is less than {threshold}, this market will resolve to "Under". '
            "If the game is postponed, this market will remain open until the game has been completed. "
            "If the game is canceled entirely, with no make-up game, this market will resolve 50-50."
        )
        markets.append(
            {
                **common,
                "id": str(i + 2),
                "sportsMarketType": "totals",
                "line": str(Decimal(threshold) - Decimal("0.5")),
                "conditionId": f"0x{i + 2:064x}",
                "clobTokenIds": json.dumps([str(1000 + i * 2), str(1001 + i * 2)]),
                "outcomes": '["Over", "Under"]',
                "description": description,
                "bestAsk": "0.51",
                "bestBid": "0.49",
                "outcomePrices": '["0.01", "0.01"]',
            }
        )
    markets.extend(
        {
            "id": str(10000 + i),
            "sportsMarketType": "player_props",
            "description": "x" * 3000,
        }
        for i in range(props)
    )
    return {
        **common,
        "id": "123",
        "slug": "nfl-aaa-bbb-2026-11-10",
        "startTime": "2026-11-10T01:15:00Z",
        "series": [{"id": "12185"}],
        "markets": markets,
    }


def screen(rows):
    return module.screen({"events": rows}, minimum=MINIMUM, maximum=MAXIMUM)


def test_large_native_event_exhausts_every_pair_without_serialized_size_gate():
    native = event(42, 500)
    with pytest.raises(ValueError, match="resource boundary"):
        module.legacy._event_ladder(native)
    result = screen([native])
    assert result["native_market_count"] == 543
    assert result["eligible_total_count"] == 42
    assert result["relation_count"] == result["price_complete_count"] == 861
    assert result["price_incomplete_count"] == result["strict_sub_floor_count"] == 0
    assert result["strongest_package"]["side_specific_cost_pusd"] == Decimal("1.02")
    assert not result["gross_candidate_only"]


def test_best_relation_can_cross_every_legacy_cardinality_boundary():
    native = event(42)
    native["markets"][1]["bestAsk"] = "0.2"
    native["markets"][-1]["bestBid"] = "0.8"
    result = screen([native])
    assert result["strongest_package"]["side_specific_cost_pusd"] == Decimal("0.4")
    assert result["strongest_package"]["lower_threshold"] == 31
    assert result["strongest_package"]["upper_threshold"] == 72
    assert result["gross_candidate_only"]
    assert not result["fees_depth_clock_and_recurrence_qualified"]


@pytest.mark.parametrize("quote", [None, "NaN", "0"])
def test_invalid_or_zero_selected_quotes_are_not_free_inventory(quote):
    native = event(2)
    native["markets"][1]["bestAsk"] = quote
    result = screen([native])
    assert result["price_incomplete_count"] == 1
    assert result["strongest_package"] is None
    assert not result["gross_candidate_only"]


def test_no_fallback_after_first_native_full_game_fails():
    first = event()
    first["markets"][1]["description"] += " Different cancellation."
    second = deepcopy(event())
    second.update(
        id="456", slug="nfl-ccc-ddd-2026-11-10", startTime="2026-11-10T02:15:00Z"
    )
    with pytest.raises(ValueError, match="complete full-game"):
        screen([second, first])


@pytest.mark.parametrize("change", ["rules", "condition", "token", "line", "id"])
def test_every_pair_keeps_complete_rule_and_identity_guards(change):
    native = event(3)
    first, last = native["markets"][1], native["markets"][-1]
    if change == "rules":
        last["description"] += " Changed resolution."
    else:
        key = {
            "condition": "conditionId",
            "token": "clobTokenIds",
            "line": "line",
            "id": "id",
        }[change]
        last[key] = first[key]
    with pytest.raises(ValueError):
        screen([native])


@pytest.mark.parametrize(
    "payload", [{}, {"events": [], "next_cursor": None}, {"events": [None] * 50}]
)
def test_incomplete_or_unknown_native_populations_fail_before_selection(payload):
    with pytest.raises(ValueError):
        module.screen(payload, minimum=MINIMUM, maximum=MAXIMUM)


def test_empty_population_is_not_a_price_rejection():
    assert screen([])["status"] == "no_deployed_full_game"


@pytest.mark.parametrize(
    "key,value",
    [
        ("active", False),
        ("closed", True),
        ("series", []),
        ("startTime", "2026-11-12T00:00:00Z"),
    ],
)
def test_native_event_scope_cannot_drift(key, value):
    native = event()
    native[key] = value
    with pytest.raises(ValueError):
        screen([native])


def test_pair_budget_is_a_compute_guard_not_an_unrelated_prop_guard():
    with pytest.raises(ValueError, match="outside the frozen scope"):
        screen([event(129)])


def test_non_game_siblings_are_filtered_before_prices():
    prop = deepcopy(event())
    prop.update(id="456", slug="nfl-aaa-bbb-2026-11-10-player-props")
    prop["markets"] = []
    assert screen([prop, event()])["event_id"] == "123"


def test_run_records_capture_failure_without_retry(tmp_path, monkeypatch):
    plan = {
        "contract_sha256": "frozen",
        "screen_outputs": {
            "journal_path": "journal.jsonl",
            "result_path": "result.json",
        },
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(module, "validate", lambda *a: None)
    monkeypatch.setattr(module.legacy, "_root_path", lambda p: tmp_path / p)
    calls = []

    def fail(*args):
        calls.append(1)
        raise OSError("transport unavailable")

    monkeypatch.setattr(module, "capture", fail)
    result = module.run(path)
    assert result["screen"]["status"] == "terminal_unqualified"
    assert result["capture_result_sha256"] is None
    assert not result["accepted_edge"] and not result["book_or_fee_request_authorized"]
    assert len(calls) == 1
    assert len((tmp_path / "journal.jsonl").read_text().splitlines()) == 2


def test_run_preflight_has_no_capture(tmp_path, monkeypatch):
    path = tmp_path / "contract.json"
    path.write_text("{}")
    monkeypatch.setattr(module, "validate", lambda *a: None)
    monkeypatch.setattr(
        module, "capture", lambda *a: pytest.fail("network in preflight")
    )
    assert module.run(path, preflight=True) is None
