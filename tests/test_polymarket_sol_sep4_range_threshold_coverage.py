from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.screen_polymarket_exact_two_leg_package import _canonical_hash


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
PREFILTER_CONTRACT = ACTION / (
    "polymarket-sol-sep4-range-threshold-coverage-prefilter-contract-v1-2026-08-30.json"
)
PREFILTER_RESULT = ACTION / (
    "polymarket-sol-sep4-range-threshold-coverage-prefilter-result-v1-2026-08-30.json"
)
BOOK_CONTRACT = ACTION / (
    "polymarket-sol-sep4-range-threshold-coverage-books-contract-v1-2026-08-30.json"
)
BOOK_RESULT = ACTION / (
    "polymarket-sol-sep4-range-threshold-coverage-books-result-v1-2026-08-30.json"
)
REWARD_CONTRACT = ACTION / (
    "polymarket-sol-sep4-cross-event-rewards-contract-v1-2026-08-30.json"
)
REWARD_RESULT = ACTION / (
    "polymarket-sol-sep4-cross-event-rewards-result-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_prefilter_retains_one_source_only_candidate() -> None:
    contract = _load(PREFILTER_CONTRACT)
    result = _load(PREFILTER_RESULT)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["contract"]["sha256"] == contract["contract_sha256"]
    assert result["screen"]["package_count"] == 20
    assert result["screen"]["strict_displayed_candidate_count"] == 1
    best = result["screen"]["best_package"]
    assert best["boundary"] == "150"
    assert best["direction"] == "upper_coverage"
    assert best["displayed_price_sum_pUSD"] == "0.9855"
    assert best["range_labels"] == [">150"]
    assert _sha(
        ROOT
        / "data/polymarket-sol-sep4-range-threshold-coverage-prefilter-v1/raw/range-event.json"
    ) == "c6691133d1e66d3e29fc10e862941f81013b0a64f47407676053263c2082ac90"
    assert _sha(
        ROOT
        / "data/polymarket-sol-sep4-range-threshold-coverage-prefilter-v1/raw/threshold-event.json"
    ) == "16dca2cbf5209e59f995dde73402b30dfd44462de7c68c41efcaad85f2731a7c"
    assert _sha(
        ROOT
        / "data/polymarket-sol-sep4-range-threshold-coverage-prefilter-v1/request-journal.jsonl"
    ) == "abd61d25faee6c64b218f4abee3ac10743898300a3e09bf538d8021030e343fa"


def test_exact_depth_rejects_before_fee_requests() -> None:
    contract = _load(BOOK_CONTRACT)
    result = _load(BOOK_RESULT)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["contract"]["sha256"] == contract["contract_sha256"]
    assert result["economics"]["actual"]["cost_pUSD"] == "5.04"
    assert result["economics"]["actual"][
        "optimistic_zero_fee_profit_floor_pUSD"
    ] == "-0.04"
    assert result["economics"]["one_adverse_tick"]["cost_pUSD"] == "5.05"
    assert result["capture"]["fee_receipts"] == {}
    assert result["capture"]["within_frozen_age_gate"] is False
    assert result["capture"]["within_frozen_skew_gate"] is False
    assert result["adjudication"]["passes_frozen_candidate_gate"] is False
    assert result["adjudication"]["accepted_edge"] is False
    assert _sha(
        ROOT
        / "data/polymarket-sol-sep4-range-threshold-coverage-books-v1/raw/books.json"
    ) == "23a27a8ba6bd3e19a3fdea97a13e71464befd601a291d2d83cc461c7a1ccfc5a"
    assert _sha(
        ROOT
        / "data/polymarket-sol-sep4-range-threshold-coverage-books-v1/request-journal.jsonl"
    ) == "4fc71206c5c5767e5a05c2c62ef0f26995c6020451748d9becce9a26fac1bd1d"


def test_registry_routes_exact_terminal_family_without_mutable_global_pins() -> None:
    registry = _load(REGISTRY)
    result = _load(BOOK_RESULT)

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    rank_31 = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 31
    )
    assert rank_31["mechanism"] == (
        "polymarket_cross_market_exact_multi_outcome_subset_equivalence"
    )
    assert any(
        artifact["result_sha256"] == result["result_sha256"]
        for artifact in rank_31["canonical_artifacts"]
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_Solana_September_4_cross_event_range_threshold_coverage_depth_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
    assert "no_fee_request_was_justified" in terminal["reason"]


def test_exact_sponsored_reward_overlay_rejects_two_empty_pools() -> None:
    contract = _load(REWARD_CONTRACT)
    result = _load(REWARD_RESULT)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["contract"]["sha256"] == contract["contract_sha256"]
    assert result["optimistic_rejection_bound"][
        "maximum_one_leg_orphan_loss_pUSD"
    ] == "48.2"
    assert result["optimistic_rejection_bound"][
        "maximum_remaining_reward_pool_pUSD"
    ] == "0"
    assert all(
        row["exact_row_count"] == 0
        for row in result["exact_rewards"].values()
    )
    assert result["authority"]["public_unauthenticated_read_only_requests"] == 2
    assert result["authority"]["orders_or_cancellations"] == 0
    assert result["adjudication"]["accepted_edge"] is False
    for name in ("range_over_150_yes", "threshold_150_no"):
        raw = ROOT / (
            "data/polymarket-sol-sep4-cross-event-rewards-v1/"
            f"reward-{name}.raw"
        )
        assert _sha(raw) == (
            "cb1463591af370d3e3eb39e1dc5821bb1ae64d7dde15ba0748011273b32e9148"
        )


def test_registry_routes_reward_overlay_to_rank_17_and_exact_terminal() -> None:
    registry = _load(REGISTRY)
    result = _load(REWARD_RESULT)

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    rank_17 = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 17
    )
    assert any(
        artifact["result_sha256"] == result["result_sha256"]
        for artifact in rank_17["canonical_artifacts"]
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_Solana_September_4_cross_event_exact_sponsored_reward_overlay_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
    assert "maximum_publicly_proven_remaining_reward_pool_is_zero" in terminal[
        "reason"
    ]
