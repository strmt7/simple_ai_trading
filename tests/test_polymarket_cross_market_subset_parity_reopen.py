from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "polymarket-cross-market-dependent-subset-parity-reopen-v1-2026-08-26.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "0838bea50b70a8d9e102f40146b2ddf041bc06db3039736d312b9f309c72fc6d"
REGISTRY_HASH = "aca1295421ba83a9a0e97a305baaf0f62371d6d2ea95526f671ec1877a0de035"


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

    assert sum(map(Decimal, profits.values())) == Decimal(
        evidence["reported_numeric_cross_market_profit_sum_usd"]
    ) == Decimal("95156.71")
    assert Decimal(evidence["headline_all_strategy_estimated_extracted_usd"]) == Decimal(
        "39587585.02"
    )
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
        range(1, 41)
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
        }
    ]
    assert "machine_proved" in row["blocking_evidence"][1]
