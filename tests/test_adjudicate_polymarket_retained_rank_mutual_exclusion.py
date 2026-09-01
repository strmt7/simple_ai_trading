from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.adjudicate_polymarket_retained_rank_mutual_exclusion import (
    _normalized_rank_rules,
    _preflight_family,
    _screen,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs/model-research/polymarket/ai-lab-rank-mutual-exclusion-contract-v1-2026-09-01.json"
)
RESULT = (
    ROOT
    / "docs/model-research/polymarket/ai-lab-rank-mutual-exclusion-result-v1-2026-09-01.json"
)
FAILURE = (
    ROOT
    / "docs/model-research/polymarket/ai-lab-rank-mutual-exclusion-preflight-failure-v1-2026-09-01.json"
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
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _market(label: str, bid: str) -> dict[str, object]:
    return {
        "id": f"market-{label}",
        "conditionId": f"condition-{label}",
        "question": f"Will {label} rank?",
        "groupItemTitle": label,
        "description": "Company occupies second place. tie rule source rule",
        "outcomes": '["Yes","No"]',
        "clobTokenIds": f'["yes-{label}","no-{label}"]',
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "bestAsk": "0.5",
        "bestBid": bid,
        "feesEnabled": False,
        "feeSchedule": {"rate": "0", "exponent": "1", "takerOnly": True},
        "orderMinSize": 5,
        "orderPriceMinTickSize": 0.001,
    }


def _event(event_id: str, role: str, bids: dict[str, str]) -> dict[str, object]:
    markets = []
    for label, bid in bids.items():
        market = _market(label, bid)
        market["description"] = f"Company occupies {role} place. tie rule source rule"
        markets.append(market)
    return {
        "id": event_id,
        "slug": f"{role}-slug",
        "title": f"{role}-title",
        "description": f"Company occupies {role} place. tie rule source rule",
        "markets": markets,
    }


def _family() -> dict[str, object]:
    return {
        "family": "test",
        "second_event_id": "2",
        "second_event_slug": "second-slug",
        "second_event_title": "second-title",
        "third_event_id": "3",
        "third_event_slug": "third-slug",
        "third_event_title": "third-title",
        "expected_all_labels": ["Alpha", "Beta", "Other"],
        "named_identity_labels": ["Alpha", "Beta"],
        "required_rule_fragments": ["tie rule", "source rule"],
    }


def _population() -> dict[str, object]:
    return {
        "events": [
            _event("2", "second", {"Alpha": "0.60", "Beta": "0.20", "Other": "0.20"}),
            _event("3", "third", {"Alpha": "0.50", "Beta": "0.30", "Other": "0.20"}),
        ]
    }


def test_rank_rule_normalization_changes_only_the_ordinal() -> None:
    assert _normalized_rank_rules("Second place and THIRD rank") == (
        "{rank} place and {rank} rank"
    )


def test_preflight_excludes_ambiguous_other_from_named_identity_packages() -> None:
    second, third = _preflight_family(_population(), _family())
    assert second["id"] == "2"
    assert third["id"] == "3"
    assert _family()["named_identity_labels"] == ["Alpha", "Beta"]


def test_preflight_allows_an_excluded_cross_event_spelling_mismatch() -> None:
    population = _population()
    population["events"][1]["markets"][1]["groupItemTitle"] = "BETA"
    family = _family()
    family["third_expected_all_labels"] = ["Alpha", "BETA", "Other"]
    family["named_identity_labels"] = ["Alpha"]
    _preflight_family(population, family)


def test_preflight_allows_inactive_excluded_placeholders_but_not_named_rows() -> None:
    population = _population()
    for event in population["events"]:
        event["markets"][2]["active"] = False
        event["markets"][2]["acceptingOrders"] = False
    _preflight_family(population, _family())
    population["events"][0]["markets"][0]["active"] = False
    with pytest.raises(RuntimeError, match="named rank market is not executable"):
        _preflight_family(population, _family())


def test_screen_uses_two_conservative_no_prices_and_exhausts_named_labels() -> None:
    contract = {"families": [_family()]}
    rows = _screen(_population(), contract)
    assert len(rows) == 2
    assert rows[0]["identity_label"] == "Alpha"
    assert rows[0]["metadata_cost_pUSD_per_share"] == "0.90"
    assert rows[0]["passes_strict_metadata_gate"] is True
    assert rows[0]["passes_fee_and_one_tick_gate"] is True
    assert all(row["identity_label"] != "Other" for row in rows)


def test_frozen_four_family_result_is_hash_bound_and_stops_before_books() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    failure = _load(FAILURE)
    assert contract["contract_sha256"] == (
        "3fd13b0fb651614e1dbce6f9e85f1baf51a5cb6f020a150efa23e0378e8aaa1a"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "268aa6e00d349b164f1727e5d47ae34afe05787627e6552307aa04b59cdbe1e6"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _canonical_hash(failure, "result_sha256") == failure["result_sha256"]
    screen = result["screen"]
    assert screen["family_count"] == 4
    assert screen["package_count"] == 67
    assert screen["side_specific_price_available_count"] == 48
    assert screen["strict_metadata_candidate_count"] == 1
    assert screen["fee_and_one_tick_candidate_count"] == 0
    assert screen["best_package"]["identity_label"] == "Google"
    assert screen["best_package"]["metadata_cost_pUSD_per_share"] == "0.99"
    assert screen["best_package"]["after_fee_one_tick_profit_floor_pUSD"] == "-0.11030"
    assert result["authority"]["network_requests"] == 0
    assert result["authority"]["protected_capture_touched"] is False


def test_registry_and_durability_audit_bind_rank_terminal() -> None:
    registry = _load(REGISTRY)
    audit = _load(AUDIT)
    result = _load(RESULT)
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert _canonical_hash(audit, "result_sha256") == audit["result_sha256"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_AI_Lab_second_third_rank_named_company_mutual_exclusion_2026_09_01"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
    assert len(registry["terminal_do_not_repeat"]) == 156
    assert (
        audit["source_binding"]["registry_result_sha256"] == registry["result_sha256"]
    )
    assert (
        audit["decision"]["stable_current_account_qualified_after_all_cost_edge_count"]
        == 0
    )
