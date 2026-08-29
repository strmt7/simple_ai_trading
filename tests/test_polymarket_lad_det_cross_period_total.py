from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from tools.screen_polymarket_exact_two_leg_package import _asks


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
DATA = ROOT / "data/polymarket-lad-det-cross-period-total-v1"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
REGISTRY_HASH = "6062ef4cb774983d86d7edd5dad7adcaafa31a8202d37ec777e12fc33028d157"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cross_period_contract_proves_the_retained_gamma_candidate() -> None:
    contract = _load(
        ACTION_VALUE
        / "polymarket-lad-det-cross-period-total-contract-v1-2026-08-29.json"
    )

    assert contract["contract_sha256"] == (
        "fa936c2049de2a05f15215609f735fd1c789237a372671b0f89b324d5f487bff"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    payouts = [
        sum(Decimal(value) for value in state["payouts"].values())
        for state in contract["payoff_proof"]["states"]
    ]
    assert min(payouts) == Decimal("1")
    assert contract["payoff_proof"]["market_direction_forecast_required"] is False
    assert Decimal(contract["gamma_prefilter"]["displayed_price_sum_pUSD"]) == (
        Decimal("0.965")
    )
    assert Decimal(
        contract["gamma_prefilter"][
            "optimistic_profit_floor_per_share_before_execution_costs_pUSD"
        ]
    ) == Decimal("0.035")
    assert contract["execution"]["book_request_count"] == 1
    assert contract["execution"]["maximum_fee_requests"] == 2
    assert contract["authority"]["protected_capture_touched"] is False


def test_exact_depth_rejects_the_cross_period_candidate_before_fee_requests() -> None:
    result = _load(
        ACTION_VALUE
        / "polymarket-lad-det-cross-period-total-result-v1-2026-08-29.json"
    )
    raw = DATA / "raw/books.json"
    journal = [
        json.loads(line)
        for line in (DATA / "request-journal.jsonl").read_text().splitlines()
    ]

    assert result["result_sha256"] == (
        "fc01c54e9c04117067aa3b43ae194649b93efc12a5265fce508e64f082f320b2"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _file_hash(ROOT / result["implementation"]["path"]) == (
        "ba7ebaa1fb5235ba026f7cf4cb21ff73ba19430fee4a4f678fbdea37635eb32f"
    )
    assert _file_hash(raw) == (
        "a337119b156cf484a869735a263e914acdd1b3008d263de2c3820242d0933679"
    )
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[0]["method"] == "POST"
    assert journal[1]["response_sha256"] == _file_hash(raw)
    assert result["capture"]["book_timestamp_skew_ms"] == 1511
    assert result["capture"]["within_frozen_skew_gate"] is True
    assert result["capture"]["fee_receipts"] == {}
    actual = result["economics"]["actual"]
    assert Decimal(actual["cost_pUSD"]) == Decimal("7")
    assert Decimal(actual["optimistic_zero_fee_profit_floor_pUSD"]) == Decimal("-2")
    assert [
        Decimal(fill["fills"][0]["price_pUSD"]) for fill in actual["fills"]
    ] == [Decimal("0.44"), Decimal("0.96")]
    one_second = result["economics"]["delay_1s_sensitivity"]
    assert Decimal(one_second["optimistic_zero_fee_profit_floor_pUSD"]) == Decimal(
        "-2.2"
    )
    assert result["economics"]["delay_3s_sensitivity"] is None
    assert result["adjudication"]["passes_frozen_candidate_gate"] is False
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["deployment_ready"] is False


def test_generic_book_parser_accepts_either_strict_order_and_rejects_mixed() -> None:
    ascending = {"asks": [{"price": "0.4", "size": "5"}, {"price": "0.5", "size": "5"}]}
    descending = {"asks": list(reversed(ascending["asks"]))}
    assert _asks(ascending) == _asks(descending)

    with pytest.raises(RuntimeError, match="strictly monotone"):
        _asks(
            {
                "asks": [
                    {"price": "0.4", "size": "5"},
                    {"price": "0.6", "size": "5"},
                    {"price": "0.5", "size": "5"},
                ]
            }
        )


def test_registry_preserves_the_omission_and_routes_the_cross_period_correction() -> None:
    registry = _load(REGISTRY)
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "polymarket_cross_market_exact_multi_outcome_subset_equivalence"
    )
    hashes = {artifact["result_sha256"] for artifact in row["canonical_artifacts"]}
    assert {
        "5c1de89005404efd8db9a35903df7633f92f9deaaa4c71a639b07d44d8f25e71",
        "fa936c2049de2a05f15215609f735fd1c789237a372671b0f89b324d5f487bff",
        "fc01c54e9c04117067aa3b43ae194649b93efc12a5265fce508e64f082f320b2",
    } <= hashes
    assert registry["accepted_edge_count"] == 19
