from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
POLYMARKET = ROOT / "docs/model-research/polymarket"
ARTIFACT = ACTION_VALUE / (
    "polymarket-round27-complete-set-taker-rebate-overlay-v1-2026-08-26.json"
)
REBATE = ACTION_VALUE / ("polymarket-organic-taker-rebate-overlay-v1-2026-08-26.json")
MECHANICS = (
    POLYMARKET / "latest/round-027-mechanics-diagnostic/mechanics-diagnostic.json"
)
CAPTURE = POLYMARKET / ("round-027-stage0-mechanics-capture-result-v1-2026-08-15.json")
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "948f47d9d0c2fb6cbf441da1147ae07006a897f307141dfd6ae25c85e47f13d2"
EXPECTED_REGISTRY_HASH = (
    "2baf1b76070e0ef9081f9eb5fba41f3977b5fd1aa74759ed85034947e9ad1c5a"
)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_bytes())
    assert isinstance(payload, dict)
    return payload


def _embedded_hash(payload: dict[str, object], claim: str) -> str:
    body = dict(payload)
    body.pop(claim)
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _scenarios(artifact: dict[str, object]) -> dict[str, dict[str, object]]:
    return {row["tier"]: row for row in artifact["scenarios"]}


def test_overlay_and_all_predecessors_are_exactly_hash_bound() -> None:
    artifact = _load(ARTIFACT)
    rebate = _load(REBATE)
    mechanics = _load(MECHANICS)
    capture = _load(CAPTURE)
    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _embedded_hash(artifact, "result_sha256") == EXPECTED_HASH
    assert _embedded_hash(rebate, "result_sha256") == rebate["result_sha256"]
    assert (
        _embedded_hash(mechanics, "mechanics_sha256") == mechanics["mechanics_sha256"]
    )
    assert _embedded_hash(capture, "result_sha256") == capture["result_sha256"]
    assert artifact["predecessors"] == {
        "round27_capture_result_sha256": capture["result_sha256"],
        "round27_mechanics_sha256": mechanics["mechanics_sha256"],
        "taker_rebate_result_sha256": rebate["result_sha256"],
    }


def test_zero_tier_exactly_reproduces_the_published_latency_baseline() -> None:
    artifact = _load(ARTIFACT)
    mechanics = _load(MECHANICS)
    baseline = _scenarios(artifact)["None"]
    published = mechanics["complete_set_latency"]
    assert baseline["same_state_episode_count"] == published["same_state_episode_count"]
    assert (
        baseline["venue_delay_survivor_count"]
        == published["venue_delay_survivor_count"]
    )
    assert (
        baseline["minimum_sequential_survivor_count"]
        == published["minimum_sequential_survivor_count"]
    )
    assert baseline["best_minimum_sequential_cost_pusd_per_complete_set"] == min(
        row["best_minimum_sequential_cost"]
        for row in published["segment_benchmarks"]
        if row["best_minimum_sequential_cost"] is not None
    )
    assert baseline["lower_source_cost_first_survivor_count"] == 0
    assert baseline["both_orders_survivor_count"] == 0


def test_gold_ex_post_minimum_is_not_misclassified_as_a_causal_edge() -> None:
    artifact = _load(ARTIFACT)
    gold = _scenarios(artifact)["Gold"]
    assert Decimal(
        gold["best_minimum_sequential_cost_pusd_per_complete_set"]
    ) < Decimal(1)
    assert Decimal(
        gold["best_lower_source_cost_first_cost_pusd_per_complete_set"]
    ) > Decimal(1)
    assert gold["minimum_sequential_survivor_count"] == 1
    assert gold["lower_source_cost_first_survivor_count"] == 0
    assert gold["both_orders_survivor_count"] == 0
    assert (
        artifact["adjudication"]["lowest_historical_causal_ordering_survivor_tier"]
        == "Diamond"
    )


def test_diamond_has_one_causal_historical_episode_but_no_robust_or_atomic_one() -> (
    None
):
    artifact = _load(ARTIFACT)
    diamond = _scenarios(artifact)["Diamond"]
    best_cost = Decimal(
        diamond["best_lower_source_cost_first_cost_pusd_per_complete_set"]
    )
    assert diamond["lower_source_cost_first_survivor_count"] == 1
    assert diamond["venue_delay_survivor_count"] == 0
    assert diamond["both_orders_survivor_count"] == 0
    assert best_cost == Decimal("0.98407728")
    assert Decimal(1) - best_cost == Decimal("0.01592272")
    assert Decimal(5) * (Decimal(1) - best_cost) == Decimal("0.07961360")
    assert Decimal(diamond["best_worst_order_cost_pusd_per_complete_set"]) > Decimal(1)

    adjudication = artifact["adjudication"]
    assert adjudication["candidate_edge"] is True
    assert adjudication["accepted_edge"] is False
    assert adjudication["deployment_ready"] is False
    assert adjudication["profitability_claim"] is False
    assert adjudication["trading_authority"] is False
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["authority"]["new_market_requests"] == 0
    assert artifact["authority"]["orders_or_mutations_submitted"] == 0


def test_candidate_is_registered_without_changing_the_accepted_edge_count() -> None:
    registry = _load(REGISTRY)
    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _embedded_hash(registry, "result_sha256") == EXPECTED_REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 43))
    candidate = next(
        row
        for row in hypotheses
        if row["mechanism"]
        == "polymarket_binary_complete_set_taker_rebate_latency_overlay"
    )
    assert candidate["priority_rank"] == 23
    assert candidate["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "polymarket-round27-complete-set-taker-rebate-overlay-v1-2026-08-26.json"
            ),
            "result_sha256": EXPECTED_HASH,
        }
    ]
