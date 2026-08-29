from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
ARTIFACT = ACTION_VALUE / (
    "polymarket-cross-market-dependent-subset-parity-reopen-v1-2026-08-26.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
SPORTS_CONTRACT = ACTION_VALUE / (
    "polymarket-current-sports-monotone-pair-discovery-contract-v1-2026-08-29.json"
)
SPORTS_RESULT = ACTION_VALUE / (
    "polymarket-current-sports-monotone-pair-discovery-result-v1-2026-08-29.json"
)
SPORTS_RAW = ROOT / (
    "data/polymarket-current-sports-monotone-pair-discovery-v1/raw/title-search.json"
)
SPORTS_JOURNAL = ROOT / (
    "data/polymarket-current-sports-monotone-pair-discovery-v1/request-journal.jsonl"
)
EXPECTED_HASH = "0838bea50b70a8d9e102f40146b2ddf041bc06db3039736d312b9f309c72fc6d"
SPORTS_CONTRACT_HASH = (
    "99559dd57d8ba1520fd4f607c4e4e56cea1070a2798536941af10134e4376aed"
)
SPORTS_RESULT_HASH = "e5ce48b6b0521a5ba2fe58ae17316e703ab2155934a126e603eeadf81e219d9c"
REGISTRY_HASH = "0a59b008453a6ff11a5d2402f037acb4fa331f0fb44dac95a60dbd9b6b73c7cf"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_contract_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("contract_sha256")
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_cross_market_subset_family_is_hash_bound_candidate_only() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _canonical_hash(artifact) == EXPECTED_HASH
    assert artifact["adjudication"] == {
        "accepted_edge": False,
        "deployment_ready": False,
        "market_direction_forecast_required": False,
        "profitability_claim": False,
        "status": (
            "materially_reopened_distinct_cross_market_exact_dependent_subset_"
            "parity_candidate_current_executable_after_fee_recurrence_unproved"
        ),
        "trading_authority": False,
    }
    assert artifact["authority"]["market_catalog_or_order_book_requests_sent"] == 0


def test_headline_is_not_misreported_as_cross_market_executable_profit() -> None:
    artifact = _load(ARTIFACT)
    evidence = artifact["historical_primary_evidence"]
    profits = evidence["reported_numeric_cross_market_profit_by_pair_usd"]

    assert (
        sum(map(Decimal, profits.values()))
        == Decimal(evidence["reported_numeric_cross_market_profit_sum_usd"])
        == Decimal("95156.71")
    )
    assert Decimal(
        evidence["headline_all_strategy_estimated_extracted_usd"]
    ) == Decimal("39587585.02")
    assert evidence["reported_cross_market_pairs_with_extraction_sentence_count"] == 5
    assert evidence["reported_cross_market_pairs_with_numeric_profit_count"] == 4
    rejected = " ".join(artifact["methodological_rejections"])
    assert "2.5 hours" in rejected
    assert "New York" in rejected
    assert "five cross-market cases" in rejected


def test_registry_adds_only_exact_truth_table_subset_candidate() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 42)
    )
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "polymarket_cross_market_exact_multi_outcome_subset_equivalence"
    )
    assert row["priority_rank"] == 31
    assert row["mechanism"] == (
        "polymarket_cross_market_exact_multi_outcome_subset_equivalence"
    )
    assert row["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "polymarket-cross-market-dependent-subset-parity-reopen-v1-"
                "2026-08-26.json"
            ),
            "result_sha256": EXPECTED_HASH,
        },
        {
            "path": SPORTS_CONTRACT.relative_to(ROOT).as_posix(),
            "result_sha256": SPORTS_CONTRACT_HASH,
        },
        {
            "path": SPORTS_RESULT.relative_to(ROOT).as_posix(),
            "result_sha256": SPORTS_RESULT_HASH,
        },
    ]
    assert "machine_proved" in row["blocking_evidence"][1]


def test_one_use_current_sports_lead_stops_before_any_book_request() -> None:
    contract = _load(SPORTS_CONTRACT)
    result = _load(SPORTS_RESULT)
    receipt = json.loads(SPORTS_JOURNAL.read_text(encoding="ascii"))

    assert contract["contract_sha256"] == SPORTS_CONTRACT_HASH
    assert _canonical_contract_hash(contract) == SPORTS_CONTRACT_HASH
    assert contract["source_contract"]["maximum_requests"] == 1
    assert contract["source_contract"]["retry_permitted"] is False
    assert contract["source_contract"]["pagination_permitted"] is False
    assert result["result_sha256"] == SPORTS_RESULT_HASH
    assert _canonical_hash(result) == SPORTS_RESULT_HASH
    assert result["capture"]["returned_event_count"] == 0
    assert result["capture"]["population_partial"] is False
    assert result["discovery"]["possible_pair_inventory"] is False
    assert result["authority"]["public_unauthenticated_GET_requests"] == 1
    assert result["authority"]["book_or_price_requests"] == 0
    assert result["authority"]["authenticated_requests"] == 0
    assert (
        receipt["response_sha256"]
        == hashlib.sha256(SPORTS_RAW.read_bytes()).hexdigest()
    )
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["next_action"] == (
        "stop_this_exact_displayed_lead_without_an_adaptive_request"
    )
