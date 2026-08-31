from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
REGISTRY_HASH = json.loads(
    (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
        encoding="utf-8"
    )
)["result_sha256"]


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


def test_bounded_catalog_is_complete_one_request_and_retains_every_relation() -> None:
    contract = _load(
        ACTION_VALUE
        / "polymarket-future-nfl-monotone-catalog-contract-v1-2026-08-29.json"
    )
    result = _load(
        ACTION_VALUE
        / "polymarket-future-nfl-monotone-catalog-result-v1-2026-08-29.json"
    )
    raw = ROOT / "data/polymarket-future-nfl-monotone-catalog-v1/raw/events.json"
    journal = [
        json.loads(line)
        for line in (
            ROOT
            / "data/polymarket-future-nfl-monotone-catalog-v1/request-journal.jsonl"
        )
        .read_text()
        .splitlines()
    ]

    assert contract["contract_sha256"] == (
        "3dc5413c76517eaf14c62d23b42fcd040c8f6f9f53b78c6e75f8a9f7e59de608"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "7c4472e0a77cde09f5643a06a1326fbfc2cc1e5ec37641314d875a346e1a7754"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _file_hash(raw) == (
        "54611964aeaa68d133a252ec3ed3476f5072fef9ad33ec9308c7726266d21e3f"
    )
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert result["capture"]["returned_event_count"] == 17
    assert result["capture"]["population_complete_under_frozen_filter"] is True
    screen = result["screen"]
    assert screen["included_event_count"] == 16
    assert screen["excluded_event_count"] == 1
    assert screen["complete_relation_count"] == len(screen["relations"]) == 4621
    assert screen["candidate_count_strictly_below_payout_floor"] == 674
    best = screen["depth_candidate"]
    assert best["event_slug"] == "nfl-was-dal-2026-09-20"
    assert Decimal(best["displayed_price_sum_per_share_pUSD"]) == Decimal("0.785")
    assert result["authority"]["book_requests"] == 0


def test_precommitted_best_catalog_candidate_fails_exact_depth_before_fees() -> None:
    contract = _load(
        ACTION_VALUE
        / "polymarket-commanders-cowboys-total-package-contract-v1-2026-08-29.json"
    )
    result = _load(
        ACTION_VALUE
        / "polymarket-commanders-cowboys-total-package-result-v1-2026-08-29.json"
    )
    books = ROOT / "data/polymarket-commanders-cowboys-total-package-v1/raw/books.json"
    journal = [
        json.loads(line)
        for line in (
            ROOT
            / "data/polymarket-commanders-cowboys-total-package-v1/request-journal.jsonl"
        )
        .read_text()
        .splitlines()
    ]

    assert contract["contract_sha256"] == (
        "0d6fba26dc1656c90e2cf78a0224e215c525364f201b8995c36439d391834292"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "729d482f9a15b60b5345ba6c52ee75941a1f0751db2453e307c30f8872bbac35"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _file_hash(books) == (
        "8da794882e57670204872c40ea9478d9e7ab05fa349eae25ba4ae6c30ec9af9e"
    )
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    actual = result["economics"]["actual"]
    assert Decimal(actual["cost_pUSD"]) == Decimal("7.55")
    assert Decimal(actual["optimistic_zero_fee_profit_floor_pUSD"]) == Decimal("-2.55")
    assert result["capture"]["book_timestamp_skew_ms"] == 25_189_367
    assert result["capture"]["within_frozen_skew_gate"] is False
    assert result["capture"]["fee_receipts"] == {}
    assert result["adjudication"]["passes_frozen_candidate_gate"] is False


def test_tie_collision_is_corrected_offline_without_adaptive_depth() -> None:
    contract = _load(
        ACTION_VALUE
        / "polymarket-cowboys-giants-tie-collision-correction-contract-v1-2026-08-29.json"
    )
    result = _load(
        ACTION_VALUE
        / "polymarket-cowboys-giants-tie-collision-correction-v1-2026-08-29.json"
    )

    assert contract["contract_sha256"] == (
        "d481f24cd43703c4ed094631ebdbae8daa2588d92b7ecca93dc2aee4cd3195f0"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "37f79cc8a4f5f96fa395a729e85a793e12c2127e2124591db693c92b1b459928"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["event"]["slug"] == "nfl-dal-nyg-2026-09-14"
    proof = result["payoff_proof"]
    assert proof["complete_relation_count"] == len(proof["relations"]) == 268
    shared = [row for row in proof["margin_thresholds"] if row["threshold"] == 1]
    assert [row["resolver"] for row in shared] == [
        "moneyline_with_half_half_tie",
        "integer_margin_threshold",
    ]
    assert (
        result["rejection_only_gamma_prefilter"][
            "candidate_count_strictly_below_payout_floor"
        ]
        == 4
    )
    assert result["authority"]["network_requests"] == 0
    assert result["authority"]["book_requests"] == 0


def test_first_nonconsumed_nfl_event_fails_exact_depth_before_fees() -> None:
    metadata = _load(
        ACTION_VALUE
        / "polymarket-patriots-seahawks-exact-event-result-v1-2026-08-31.json"
    )
    prefilter = _load(
        ACTION_VALUE
        / "polymarket-patriots-seahawks-monotone-prefilter-result-v1-2026-08-31.json"
    )
    books = _load(
        ACTION_VALUE
        / "polymarket-patriots-seahawks-total-books-result-v1-2026-08-31.json"
    )

    assert metadata["result_sha256"] == (
        "e62d7b8a93a2a11197058d631e2fea88a408cfa30bd2dda6a9df54998b524d36"
    )
    assert _canonical_hash(metadata, "result_sha256") == metadata["result_sha256"]
    assert metadata["event"]["slug"] == "nfl-ne-sea-2026-09-10"
    assert prefilter["result_sha256"] == (
        "2658f410b44f2a689c528b82f94f71c6459f5f52729824cd5f58fb22e920cc38"
    )
    assert _canonical_hash(prefilter, "result_sha256") == prefilter["result_sha256"]
    assert prefilter["payoff_proof"]["complete_relation_count"] == 291
    gate = prefilter["rejection_only_gamma_prefilter"]
    assert gate["candidate_count_strictly_below_payout_floor"] == 2
    assert Decimal(gate["best_relation"]["displayed_price_sum_per_share_pUSD"]) == (
        Decimal("0.995")
    )
    assert books["result_sha256"] == (
        "2658c04330fdeaa7f39b4a9dc842a079c7e81ef1feeeba75ce74384e328338ae"
    )
    assert _canonical_hash(books, "result_sha256") == books["result_sha256"]
    assert Decimal(books["economics"]["actual"]["cost_pUSD"]) == Decimal("5.55")
    assert Decimal(
        books["economics"]["actual"]["optimistic_zero_fee_profit_floor_pUSD"]
    ) == Decimal("-0.55")
    assert books["capture"]["oldest_book_age_ms"] == 11_031
    assert books["capture"]["fee_receipts"] == {}
    assert books["adjudication"]["passes_frozen_candidate_gate"] is False


def test_retained_team_to_full_total_graph_rejects_without_network() -> None:
    result = _load(
        ACTION_VALUE
        / "polymarket-patriots-seahawks-team-full-total-monotone-v1-2026-08-31.json"
    )

    assert result["result_sha256"] == (
        "51d0f5bda02fe124a7f02eb6f2365c358f8ec6a99d75b8f87bd8de8eec1d669c"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["screen"]["rule_valid_relation_count"] == 9
    assert result["screen"]["strict_displayed_sub_floor_candidate_count"] == 0
    best = result["screen"]["best_relation"]
    assert Decimal(best["displayed_package_sum_pUSD"]) == Decimal("1.465")
    assert Decimal(best["optimistic_headroom_before_execution_costs_pUSD"]) == (
        Decimal("-0.465")
    )
    assert result["authority"]["new_network_requests"] == 0
    assert result["adjudication"]["book_or_fee_request_permitted"] is False


def test_registry_routes_catalog_and_correction_without_acceptance() -> None:
    registry = _load(REGISTRY)
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["priority_rank"] == 30
    )
    hashes = {artifact["result_sha256"] for artifact in row["canonical_artifacts"]}
    assert {
        "7c4472e0a77cde09f5643a06a1326fbfc2cc1e5ec37641314d875a346e1a7754",
        "729d482f9a15b60b5345ba6c52ee75941a1f0751db2453e307c30f8872bbac35",
        "37f79cc8a4f5f96fa395a729e85a793e12c2127e2124591db693c92b1b459928",
        "2658c04330fdeaa7f39b4a9dc842a079c7e81ef1feeeba75ce74384e328338ae",
        "51d0f5bda02fe124a7f02eb6f2365c358f8ec6a99d75b8f87bd8de8eec1d669c",
    } <= hashes
    assert "25189367_ms_skew" in row["current_status"]
    assert "Patriots_Seahawks" in row["current_status"]
