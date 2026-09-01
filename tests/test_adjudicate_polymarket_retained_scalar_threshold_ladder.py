from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.adjudicate_polymarket_retained_scalar_threshold_ladder import (
    _normalized_event,
    _preflight_event_schema,
    _select_event,
)
from tools.screen_polymarket_exact_crypto_threshold_ladder_v2 import _screen_event


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs/model-research/polymarket/measles-sep30-retained-threshold-ladder-contract-v1-2026-09-01.json"
)
RESULT = (
    ROOT
    / "docs/model-research/polymarket/measles-sep30-retained-threshold-ladder-result-v1-2026-09-01.json"
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
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _market(threshold: str, ask: str | None, bid: str | None) -> dict[str, object]:
    return {
        "id": threshold,
        "conditionId": f"condition-{threshold}",
        "question": f"At least {threshold}?",
        "groupItemTitle": f"{threshold}+",
        "description": "same source same cutoff same fallback",
        "outcomes": '["Yes","No"]',
        "clobTokenIds": f'["yes-{threshold}","no-{threshold}"]',
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "bestAsk": ask,
        "bestBid": bid,
        "feesEnabled": False,
        "feeSchedule": None,
        "orderMinSize": 5,
        "orderPriceMinTickSize": 0.001,
    }


def _contract() -> dict[str, object]:
    return {
        "event_id": "event-1",
        "event_slug": "event",
        "event_title": "title",
        "expected_thresholds": ["100", "200", "300"],
        "required_rule_fragments": ["same source", "same cutoff", "same fallback"],
    }


def _event() -> dict[str, object]:
    return {
        "id": "event-1",
        "slug": "event",
        "title": "title",
        "markets": [
            _market("100", "0.40", "0.30"),
            _market("200", "0.60", "0.50"),
            _market("300", "0.80", "0.70"),
        ],
    }


def test_retained_selector_and_schema_preflight_do_not_require_economic_values() -> (
    None
):
    event = _event()
    for market in event["markets"]:
        market["bestAsk"] = None
        market["bestBid"] = None
    selected = _select_event({"events": [event]}, "event-1")
    _preflight_event_schema(selected, _contract())


def test_retained_screen_exhausts_all_ordered_threshold_pairs() -> None:
    _, packages = _screen_event(_normalized_event(_event()), _contract())
    assert len(packages) == 3
    assert {(row["lower_threshold"], row["higher_threshold"]) for row in packages} == {
        ("100", "200"),
        ("100", "300"),
        ("200", "300"),
    }
    best = packages[0]
    assert best["displayed_price_sum_pUSD"] == "0.70"
    assert best["passes_strict_displayed_gross_gate"] is True


def test_retained_schema_rejects_population_drift() -> None:
    event = _event()
    event["markets"].pop()
    with pytest.raises(RuntimeError, match="threshold population changed"):
        _preflight_event_schema(event, _contract())


def test_frozen_retained_result_is_hash_bound_and_terminal_before_books() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    assert contract["contract_sha256"] == (
        "feb1a65dcbf3e0a3af2e7b33fea7431fc5eebc3439ca295c4a4bf63058265e5b"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "990207a430a388ed85c74924eceb53b0e68ba7d31c3b3827a6d19a4d04376a50"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    screen = result["screen"]
    assert screen["market_count"] == 4
    assert screen["package_count"] == 6
    assert screen["side_specific_price_available_count"] == 6
    assert screen["strict_displayed_candidate_count"] == 0
    assert screen["best_package"]["displayed_price_sum_pUSD"] == "1.030"
    assert result["authority"]["network_requests"] == 0
    assert result["authority"]["protected_capture_touched"] is False


def test_registry_and_durability_audit_bind_terminal_measles_screen() -> None:
    registry = _load(REGISTRY)
    audit = _load(AUDIT)
    result = _load(RESULT)
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert _canonical_hash(audit, "result_sha256") == audit["result_sha256"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_US_measles_September_30_exact_threshold_ladder_2026_09_01"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
    assert audit["source_binding"]["accepted_edge_count"] == 31
    assert (
        audit["source_binding"]["registry_result_sha256"] == registry["result_sha256"]
    )
    assert (
        audit["decision"]["stable_current_account_qualified_after_all_cost_edge_count"]
        == 0
    )
