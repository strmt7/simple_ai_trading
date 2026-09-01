from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.adjudicate_polymarket_retained_deadline_implications import (
    _preflight_pair,
    _screen,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs/model-research/polymarket/retained-deadline-implications-contract-v1-2026-09-01.json"
)
RESULT = (
    ROOT
    / "docs/model-research/polymarket/retained-deadline-implications-result-v1-2026-09-01.json"
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


def _market(
    market_id: str, label: str, start: str, ask: str, bid: str
) -> dict[str, object]:
    return {
        "id": market_id,
        "conditionId": f"condition-{market_id}",
        "question": f"Event by {label}?",
        "groupItemTitle": label,
        "description": "between market creation and the specified date source rule",
        "startDate": start,
        "outcomes": '["Yes","No"]',
        "clobTokenIds": f'["yes-{market_id}","no-{market_id}"]',
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "bestAsk": ask,
        "bestBid": bid,
        "feesEnabled": False,
        "feeSchedule": {"rate": "0", "exponent": "1", "takerOnly": True},
        "orderMinSize": 5,
        "orderPriceMinTickSize": 0.001,
    }


def _population(later_start: str = "2026-01-01T00:00:00Z") -> dict[str, object]:
    return {
        "events": [
            {
                "id": "event-1",
                "slug": "event",
                "title": "title",
                "markets": [
                    _market("1", "September 15", "2026-01-02T00:00:00Z", "0.6", "0.4"),
                    _market("2", "September 30", later_start, "0.5", "0.5"),
                ],
            }
        ]
    }


def _pair() -> dict[str, object]:
    return {
        "pair": "test",
        "event_id": "event-1",
        "event_slug": "event",
        "event_title": "title",
        "earlier_market_id": "1",
        "earlier_label": "September 15",
        "earlier_deadline": "2026-09-15",
        "earlier_start_utc": "2026-01-02T00:00:00Z",
        "later_market_id": "2",
        "later_label": "September 30",
        "later_deadline": "2026-09-30",
        "later_start_utc": "2026-01-01T00:00:00Z",
        "required_rule_fragments": ["between market creation", "source rule"],
    }


def test_preflight_accepts_only_a_covering_later_creation_window() -> None:
    earlier, later = _preflight_pair(_population(), _pair())
    assert earlier["id"] == "1"
    assert later["id"] == "2"


def test_preflight_rejects_later_market_creation_gap() -> None:
    pair = _pair()
    pair["later_start_utc"] = "2026-01-03T00:00:00Z"
    with pytest.raises(RuntimeError, match="does not cover"):
        _preflight_pair(_population("2026-01-03T00:00:00Z"), pair)


def test_screen_prices_earlier_no_plus_later_yes() -> None:
    rows = _screen(_population(), {"pairs": [_pair()]})
    assert len(rows) == 1
    assert rows[0]["metadata_cost_pUSD_per_share"] == "1.1"
    assert rows[0]["passes_strict_metadata_gate"] is False
    assert [leg["outcome"] for leg in rows[0]["legs"]] == ["No", "Yes"]


def test_frozen_result_is_hash_bound_and_stops_before_books() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    assert contract["contract_sha256"] == (
        "d85ff8d51b720b2b88d41952d25b003b5d5240771499e37d19579204180adde8"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "1b3dcf4167ef6bd5dbec85b1f8d5e23f23b0ad7bcac2cfe4e32b58d65afedb23"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    screen = result["screen"]
    assert screen["pair_count"] == 4
    assert screen["side_specific_price_available_count"] == 2
    assert screen["strict_metadata_candidate_count"] == 0
    assert screen["fee_and_one_tick_candidate_count"] == 0
    assert screen["best_pair"]["metadata_cost_pUSD_per_share"] == "1.19"
    assert screen["best_pair"]["after_fee_one_tick_profit_floor_pUSD"] == "-1.10590"
    assert result["authority"]["network_requests"] == 0
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["protected_capture_touched"] is False


def test_registry_and_durability_bind_deadline_terminal() -> None:
    registry = _load(REGISTRY)
    audit = _load(AUDIT)
    result = _load(RESULT)
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert _canonical_hash(audit, "result_sha256") == audit["result_sha256"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_creation_window_safe_retained_deadline_implications_2026_09_01"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
    assert len(registry["terminal_do_not_repeat"]) == 157
    assert (
        audit["source_binding"]["registry_result_sha256"] == registry["result_sha256"]
    )
    assert (
        audit["decision"]["stable_current_account_qualified_after_all_cost_edge_count"]
        == 0
    )
