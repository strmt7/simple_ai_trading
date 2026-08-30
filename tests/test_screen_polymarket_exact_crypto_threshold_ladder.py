from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.screen_polymarket_exact_crypto_threshold_ladder import _screen_event


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION / (
    "polymarket-eth-sep4-threshold-ladder-prefilter-contract-v1-2026-08-30.json"
)
RESULT = ACTION / (
    "polymarket-eth-sep4-threshold-ladder-prefilter-result-v1-2026-08-30.json"
)
ADJUDICATION = ACTION / (
    "polymarket-eth-sep4-threshold-ladder-terminal-adjudication-v1-2026-08-30.json"
)
RAW = ROOT / "data/polymarket-eth-sep4-threshold-ladder-prefilter-v1/raw/event.json"
JOURNAL = ROOT / (
    "data/polymarket-eth-sep4-threshold-ladder-prefilter-v1/request-journal.jsonl"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"

CONTRACT_HASH = "0737aa5e76be4151213f1a6174eca525e32ec7c46e7e4347842e6ac41c8a7331"
RESULT_HASH = "42c122e54bd9a7299cc9e739724fabd4cd76716dd3bdfe76c039a1bab8014d2a"
ADJUDICATION_HASH = "25f90c75b9d8657e44b27ebab8dd4c26fb3434a306a6dbd9e35fdc2fdd53419d"
RAW_HASH = "84a0536e067b5f72a5a4c9fc1ac4a215316b51a742a5f7e781ab37e7fbe5b1be"


def _canonical_hash(value: dict[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field)
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _market(*, market_id: int, threshold: str, yes: str, no: str) -> dict[str, object]:
    return {
        "id": market_id,
        "conditionId": f"condition-{market_id}",
        "question": f"Ethereum above ${threshold} on September 4?",
        "groupItemTitle": threshold,
        "description": "higher than the price specified | ETH/USDT 12:00",
        "outcomes": '["Yes","No"]',
        "outcomePrices": f'["{yes}","{no}"]',
        "clobTokenIds": f'["yes-{market_id}","no-{market_id}"]',
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "feesEnabled": True,
        "feeSchedule": {
            "exponent": 1,
            "rate": 0.07,
            "takerOnly": True,
            "rebateRate": 0.2,
        },
        "orderMinSize": 5,
        "orderPriceMinTickSize": 0.001,
    }


def _contract() -> dict[str, object]:
    return {
        "event_slug": "ethereum-above-on-september-4-2026",
        "event_title": "Ethereum above ___ on September 4?",
        "expected_thresholds": ["2000", "2100", "2200"],
        "required_rule_fragments": [
            "higher than the price specified",
            "ETH/USDT 12:00",
        ],
    }


def test_lower_yes_plus_higher_no_has_exact_floor_and_strict_candidate() -> None:
    event = {
        "slug": "ethereum-above-on-september-4-2026",
        "title": "Ethereum above ___ on September 4?",
        "markets": [
            _market(market_id=1, threshold="2,000", yes="0.40", no="0.60"),
            _market(market_id=2, threshold="2,100", yes="0.35", no="0.65"),
            _market(market_id=3, threshold="2,200", yes="0.20", no="0.70"),
        ],
    }

    legs, packages = _screen_event(event, _contract())

    assert [row["threshold"] for row in legs] == ["2000", "2100", "2200"]
    assert len(packages) == 3
    assert packages[0]["lower_threshold"] == "2000"
    assert packages[0]["higher_threshold"] == "2100"
    assert packages[0]["displayed_price_sum_pUSD"] == "1.05"
    assert packages[0]["passes_strict_displayed_gross_gate"] is False

    event["markets"][1]["outcomePrices"] = '["0.55","0.55"]'
    _, packages = _screen_event(event, _contract())
    candidate = next(
        row
        for row in packages
        if row["lower_threshold"] == "2000" and row["higher_threshold"] == "2100"
    )
    assert candidate["displayed_price_sum_pUSD"] == "0.95"
    assert candidate["optimistic_displayed_headroom_pUSD"] == "0.05"
    assert candidate["guaranteed_payout_floor_pUSD_per_share"] == "1"
    assert candidate["passes_strict_displayed_gross_gate"] is True


def test_threshold_population_or_rules_must_match_exactly() -> None:
    event = {
        "slug": "ethereum-above-on-september-4-2026",
        "title": "Ethereum above ___ on September 4?",
        "markets": [
            _market(market_id=1, threshold="2,000", yes="0.4", no="0.6"),
            _market(market_id=2, threshold="2,100", yes="0.3", no="0.7"),
        ],
    }
    contract = _contract()
    contract["expected_thresholds"] = ["2000", "2100"]
    event["markets"][1]["description"] = "different resolution source"

    try:
        _screen_event(event, contract)
    except RuntimeError as exc:
        assert str(exc) == "exact threshold resolution rules changed"
    else:
        raise AssertionError("changed rules must fail closed")


def test_exact_eth_event_is_source_bound_and_terminal_before_books() -> None:
    contract = json.loads(CONTRACT.read_bytes())
    result = json.loads(RESULT.read_bytes())
    adjudication = json.loads(ADJUDICATION.read_bytes())
    registry = json.loads(REGISTRY.read_bytes())

    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    assert _canonical_hash(adjudication, "result_sha256") == ADJUDICATION_HASH
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == RAW_HASH
    journal = [json.loads(line) for line in JOURNAL.read_bytes().splitlines()]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1]["response_sha256"] == RAW_HASH
    assert result["screen"]["market_count"] == 11
    assert result["screen"]["package_count"] == 55
    assert result["screen"]["strict_displayed_candidate_count"] == 0
    assert result["screen"]["best_package"]["lower_threshold"] == "2900"
    assert result["screen"]["best_package"]["higher_threshold"] == "3000"
    assert result["screen"]["best_package"]["displayed_price_sum_pUSD"] == ("1.0015")
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["protected_capture_touched"] is False
    assert registry["result_sha256"] == _canonical_hash(registry, "result_sha256")
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["priority_rank"] == 31
    )
    hashes = {item["result_sha256"] for item in row["canonical_artifacts"]}
    assert {CONTRACT_HASH, RESULT_HASH, ADJUDICATION_HASH}.issubset(hashes)
    terminal = next(
        item
        for item in registry["terminal_do_not_repeat"]
        if item["family"]
        == "polymarket_Ethereum_September_4_exact_monotone_threshold_ladder_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == ADJUDICATION_HASH
    assert (
        adjudication["payoff_adjudication"]["absolute_cross_condition_guarantee_proved"]
        is False
    )
