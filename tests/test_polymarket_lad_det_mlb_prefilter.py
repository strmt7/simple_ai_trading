from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import adjudicate


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
DATA = ROOT / "data/polymarket-lad-det-exact-event-prefilter-v1"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
REGISTRY_HASH = "da3ddaf82a2cb0929353460a7e09812b47f940e953a3f1da43b04f72a55c8488"


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


def _metadata() -> dict[str, object]:
    return _load(
        ACTION_VALUE
        / "polymarket-lad-det-exact-event-prefilter-result-v1-2026-08-29.json"
    )


def test_exact_event_capture_is_one_public_request_and_hash_bound() -> None:
    contract = _load(
        ACTION_VALUE
        / "polymarket-lad-det-exact-event-prefilter-contract-v1-2026-08-29.json"
    )
    result = _metadata()
    raw = DATA / "raw/event.json"
    journal = [
        json.loads(line)
        for line in (DATA / "request-journal.jsonl").read_text().splitlines()
    ]

    assert contract["contract_sha256"] == (
        "67402fa5e41f16a44774987654dcbe4ba1dfb8aebea794577de5f766fc785a75"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "7d31545dfb4195b8ecc3fd19e8f2711e4634dd4cc259aa3a8d22f64402852593"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _file_hash(raw) == (
        "144d66b5520a4091ed5bb201043c8cd599537b105e4b406e7c6e4f5d60fac005"
    )
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[0]["method"] == "GET"
    assert journal[1]["response_sha256"] == _file_hash(raw)
    assert result["capture"]["exact_slug_match"] is True
    assert result["capture"]["event_active_and_open"] is True
    assert result["capture"]["active_accepting_market_count"] == 17
    assert result["authority"]["public_unauthenticated_read_only_requests"] == 1
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["fee_requests"] == 0
    assert result["authority"]["protected_capture_touched"] is False


def test_all_exact_ladder_packages_fail_the_rejection_only_gate() -> None:
    artifact = _load(
        ACTION_VALUE
        / "polymarket-lad-det-monotone-prefilter-adjudication-v1-2026-08-29.json"
    )

    assert artifact["result_sha256"] == (
        "5c1de89005404efd8db9a35903df7633f92f9deaaa4c71a639b07d44d8f25e71"
    )
    assert _canonical_hash(artifact, "result_sha256") == artifact["result_sha256"]
    assert _file_hash(ROOT / artifact["implementation"]["path"]) == (
        "958a44d78029b9a1eb69a3771ce18e53ac1ed3c7bc161c8e4318beedce2cf99f"
    )
    proof = artifact["payoff_proof"]
    assert proof["complete_relation_count"] == 24
    counts: dict[str, int] = {}
    for relation in proof["relations"]:
        counts[relation["family"]] = counts.get(relation["family"], 0) + 1
        assert Decimal(relation["minimum_terminal_payout_per_share_pUSD"]) >= 1
        assert relation["passes_strictly_below_payout_gate"] is False
    assert counts == {
        "full_game_margin": 10,
        "first_five_margin": 1,
        "full_game_total": 3,
        "first_five_total": 10,
    }
    gate = artifact["rejection_only_gamma_prefilter"]
    assert gate["candidate_count_strictly_below_payout_floor"] == 0
    assert gate["all_displayed_price_sums_at_or_above_payout_floor"] is True
    assert gate["best_relation"]["family"] == "first_five_margin"
    assert Decimal(gate["best_relation"]["displayed_price_sum_per_share_pUSD"]) == (
        Decimal("1.010")
    )
    assert Decimal(
        gate[
            "best_optimistic_profit_floor_at_five_shares_before_execution_costs_pUSD"
        ]
    ) == Decimal("-0.050")
    assert gate["gamma_can_support_acceptance_or_promotion"] is False
    assert artifact["adjudication"]["status"] == (
        "terminal_exact_event_rejected_before_books_and_fees"
    )
    assert artifact["authority"]["book_requests"] == 0
    assert artifact["authority"]["fee_requests"] == 0


def test_offline_adjudicator_detects_a_strictly_below_floor_sensitivity() -> None:
    metadata = deepcopy(_metadata())
    first_five_spreads = [
        row
        for row in metadata["discovery"]["active_accepting_markets"]
        if row["sportsMarketType"] == "baseball_team_first_five_spread"
    ]
    assert len(first_five_spreads) == 2
    for row in first_five_spreads:
        row["outcomePrices"] = '["0.51", "0.49"]'
    metadata["result_sha256"] = _canonical_hash(metadata, "result_sha256")

    sensitivity = adjudicate(metadata)

    assert (
        sensitivity["rejection_only_gamma_prefilter"][
            "candidate_count_strictly_below_payout_floor"
        ]
        >= 1
    )
    assert sensitivity["adjudication"]["status"] == (
        "candidate_requires_separately_frozen_exact_depth_screen"
    )
    assert sensitivity["authority"]["book_requests"] == 0


def test_offline_adjudicator_rejects_missing_cancellation_semantics() -> None:
    metadata = deepcopy(_metadata())
    market = metadata["discovery"]["active_accepting_markets"][2]
    market["description"] = market["description"].replace("resolve 50-50", "resolve")
    metadata["result_sha256"] = _canonical_hash(metadata, "result_sha256")

    with pytest.raises(RuntimeError, match="50-50 cancellation semantics"):
        adjudicate(metadata)


def test_registry_routes_the_exact_event_to_the_existing_subset_family() -> None:
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
        "67402fa5e41f16a44774987654dcbe4ba1dfb8aebea794577de5f766fc785a75",
        "7d31545dfb4195b8ecc3fd19e8f2711e4634dd4cc259aa3a8d22f64402852593",
        "5c1de89005404efd8db9a35903df7633f92f9deaaa4c71a639b07d44d8f25e71",
    } <= hashes
    assert registry["accepted_edge_count"] == 21
