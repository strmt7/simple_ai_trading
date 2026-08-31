from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.screen_polymarket_exact_crypto_threshold_ladder_v2 import _screen_event


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = (
    ACTION_VALUE
    / "polymarket-eth-sep2-threshold-ladder-prefilter-contract-v2-2026-08-31.json"
)
RESULT = (
    ACTION_VALUE
    / "polymarket-eth-sep2-threshold-ladder-prefilter-result-v2-2026-08-31.json"
)
RAW = ROOT / "data/polymarket-eth-sep2-threshold-ladder-prefilter-v2/raw/event.json"
JOURNAL = (
    ROOT
    / "data/polymarket-eth-sep2-threshold-ladder-prefilter-v2/request-journal.jsonl"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


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
        "question": f"Ethereum above ${threshold}?",
        "groupItemTitle": threshold,
        "description": "rule-a rule-b rule-c",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0","0"]',
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


def test_v2_uses_side_specific_fields_and_ignores_outcome_prices() -> None:
    contract = {
        "event_slug": "event",
        "event_title": "title",
        "expected_thresholds": ["100", "200", "300"],
        "required_rule_fragments": ["rule-a", "rule-b", "rule-c"],
    }
    event = {
        "slug": "event",
        "title": "title",
        "markets": [
            _market("100", "0.4", "0.3"),
            _market("200", "0.6", "0.5"),
            _market("300", "0.8", "0.7"),
        ],
    }

    _, packages = _screen_event(event, contract)

    package = next(
        row
        for row in packages
        if row["lower_threshold"] == "100" and row["higher_threshold"] == "200"
    )
    assert package["displayed_price_sum_pUSD"] == "0.9"
    assert package["passes_strict_displayed_gross_gate"] is True


def test_missing_side_specific_price_fails_closed_instead_of_becoming_zero() -> None:
    contract = {
        "event_slug": "event",
        "event_title": "title",
        "expected_thresholds": ["100", "200"],
        "required_rule_fragments": ["rule-a", "rule-b", "rule-c"],
    }
    event = {
        "slug": "event",
        "title": "title",
        "markets": [_market("100", None, "0.3"), _market("200", "0.6", "0.5")],
    }

    _, packages = _screen_event(event, contract)

    assert packages[0]["side_specific_rejection_price_available"] is False
    assert packages[0]["displayed_price_sum_pUSD"] is None
    assert packages[0]["passes_strict_displayed_gross_gate"] is False


def test_frozen_contract_and_one_use_capture_are_hash_bound() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    journal = [json.loads(line) for line in JOURNAL.read_text().splitlines()]

    assert contract["status"] == "frozen_before_one_public_gamma_request"
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "a83c568656fe15000624d0ae872abc75955bd42581714a1a926b114bc4206f33"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    receipt = result["capture"]["receipt"]
    assert receipt["status_code"] == 200
    assert receipt["response_bytes"] == RAW.stat().st_size == 39_864
    assert receipt["response_sha256"] == hashlib.sha256(RAW.read_bytes()).hexdigest()
    assert [entry["phase"] for entry in journal] == ["intent", "completed"]
    assert result["authority"]["public_unauthenticated_read_only_requests"] == 1
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["credentials_used"] is False


def test_all_55_side_specific_packages_fail_before_books() -> None:
    result = _load(RESULT)
    screen = result["screen"]

    assert screen["market_count"] == 11
    assert screen["package_count"] == 55
    assert screen["side_specific_price_available_count"] == 55
    assert screen["strict_displayed_candidate_count"] == 0
    assert screen["best_package"]["displayed_price_sum_pUSD"] == "1.004"
    assert screen["best_package"]["lower_threshold"] == "2000"
    assert screen["best_package"]["higher_threshold"] == "2100"
    assert result["adjudication"] == {
        "accepted_edge": False,
        "deployment_ready": False,
        "profitability_claim": False,
        "status": "rejected_before_books_fees_and_onchain_requests",
    }


def test_rank_31_and_terminal_registry_record_exact_event_without_promotion() -> None:
    registry = _load(REGISTRY)
    result = _load(RESULT)

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "polymarket_cross_market_exact_multi_outcome_subset_equivalence"
    )
    assert hypothesis["priority_rank"] == 31
    assert {
        "path": RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": result["result_sha256"],
    } in hypothesis["canonical_artifacts"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_Ethereum_September_2_exact_monotone_threshold_ladder_2026_08_31"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
