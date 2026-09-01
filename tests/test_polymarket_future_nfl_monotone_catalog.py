from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools.adjudicate_polymarket_nfl_period_graph import adjudicate
from tools.adjudicate_polymarket_nfl_catalog_side_specific import (
    _side_specific_price,
    adjudicate as adjudicate_catalog_side_specific,
)


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


def test_side_specific_price_requires_only_the_selected_side_field() -> None:
    first_outcome_market = {
        "id": "first",
        "outcomes": ["Over", "Under"],
        "bestAsk": "0.41",
        "bestBid": None,
    }
    second_outcome_market = {
        "id": "second",
        "outcomes": ["Over", "Under"],
        "bestAsk": None,
        "bestBid": "0.37",
    }

    assert _side_specific_price(first_outcome_market, "Over") == (
        Decimal("0.41"),
        "bestAsk",
    )
    assert _side_specific_price(second_outcome_market, "Under") == (
        Decimal("0.63"),
        "1-bestBid",
    )


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


def test_superseded_v1_and_corrected_v2_period_graph_are_preserved() -> None:
    previous = _load(
        ACTION_VALUE
        / "polymarket-patriots-seahawks-team-ladder-additive-cover-v1-2026-08-31.json"
    )
    contract = _load(
        ACTION_VALUE
        / "polymarket-patriots-seahawks-period-graph-contract-v2-2026-08-31.json"
    )
    result = _load(
        ACTION_VALUE
        / "polymarket-patriots-seahawks-period-graph-result-v2-2026-08-31.json"
    )

    assert previous["result_sha256"] == (
        "317ecd24a990456be2e8c64f644f7481309d7c5e663c33274d281270483a848f"
    )
    assert _canonical_hash(previous, "result_sha256") == previous["result_sha256"]
    assert previous["adjudication"]["accepted_edge"] is False
    assert contract["contract_sha256"] == (
        "38df25f415434cb1ae823959e6fd81e0d777f52432bb3f5578122e407ed584f6"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "9c80f1a188c059890c682f16f367cc472667bda595482d5d0db26ebda2d014bb"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert adjudicate(contract) == result
    assert result["supersession"]["previous_valid_additive_subset_count"] == 33
    assert result["family_screens"]["full_team_additive_covers"]["relation_count"] == 325
    assert Decimal(
        result["family_screens"]["full_team_additive_covers"]
        ["best_side_specific_relation"]["side_specific_rejection_sum_pUSD"]
    ) == Decimal("1.78")
    assert result["family_screens"]["full_team_total_ladders"]["relation_count"] == 20
    assert Decimal(
        result["family_screens"]["full_team_total_ladders"]
        ["best_side_specific_relation"]["side_specific_rejection_sum_pUSD"]
    ) == Decimal("1.34")
    assert result["aggregate_screen"]["relation_count"] == 8_853
    assert result["aggregate_screen"]["strict_side_specific_subfloor_count"] == 0
    assert result["aggregate_screen"]["strict_diagnostic_subfloor_count"] == 0
    assert result["authority"]["new_network_requests"] == 0
    assert result["adjudication"]["book_or_fee_request_permitted"] is False


def test_complete_retained_nfl_catalog_is_corrected_before_any_more_books() -> None:
    contract = _load(
        ACTION_VALUE
        / "polymarket-future-nfl-catalog-side-specific-adjudication-contract-v1-2026-08-31.json"
    )
    result = _load(
        ACTION_VALUE
        / "polymarket-future-nfl-catalog-side-specific-adjudication-v1-2026-08-31.json"
    )

    assert contract["contract_sha256"] == (
        "d37c0924ab5c1d66ee95d4ad06956b0cbb24c031db18b5c733033e463d69611b"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "61fb2010d57b3295dd0ca859345c54404372dbd8b0f7ac4bca42d3fb0e40ddfd"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert adjudicate_catalog_side_specific(contract) == result

    screen = result["screen"]
    assert screen["complete_relation_count"] == 4_621
    assert screen["price_complete_relation_count"] == 4_621
    assert screen["source_midpoint_like_strict_subfloor_count"] == 674
    assert screen["strict_side_specific_subfloor_count"] == 0
    assert screen["complete_relations_sha256"] == (
        "4f66205743538c5491fd52246edb40f5a033bc27865195f20ebb5e842339d6fc"
    )
    groups = screen["event_family_summaries"]
    assert len(groups) == 32
    assert sum(row["relation_count"] for row in groups) == 4_621
    assert all(row["strict_side_specific_subfloor_count"] == 0 for row in groups)
    best = screen["best_side_specific_relation"]
    assert best["event_slug"] == "nfl-den-kc-2026-09-15"
    assert Decimal(best["side_specific_rejection_sum_pUSD"]) == Decimal("1.05")
    assert result["authority"]["new_network_requests"] == 0
    assert result["adjudication"]["book_or_fee_request_permitted"] is False


def test_early_september13_catalog_retains_incomplete_prices_and_blocks_depth() -> None:
    catalog_contract = _load(
        ACTION_VALUE
        / "polymarket-nfl-september13-early-monotone-catalog-contract-v1-2026-09-01.json"
    )
    catalog_result = _load(
        ACTION_VALUE
        / "polymarket-nfl-september13-early-monotone-catalog-result-v1-2026-09-01.json"
    )
    failure_v1 = _load(
        ACTION_VALUE
        / "polymarket-nfl-september13-early-side-specific-adjudication-v1-failure-2026-09-01.json"
    )
    contract_v1 = _load(
        ACTION_VALUE
        / "polymarket-nfl-september13-early-side-specific-adjudication-contract-v1-2026-09-01.json"
    )
    failure_v2 = _load(
        ACTION_VALUE
        / "polymarket-nfl-september13-early-side-specific-adjudication-v2-failure-2026-09-01.json"
    )
    contract_v2 = _load(
        ACTION_VALUE
        / "polymarket-nfl-september13-early-side-specific-adjudication-contract-v2-2026-09-01.json"
    )
    contract_v3 = _load(
        ACTION_VALUE
        / "polymarket-nfl-september13-early-side-specific-adjudication-contract-v3-2026-09-01.json"
    )
    result_v3 = _load(
        ACTION_VALUE
        / "polymarket-nfl-september13-early-side-specific-adjudication-v3-2026-09-01.json"
    )
    contract_v4 = _load(
        ACTION_VALUE
        / "polymarket-nfl-september13-early-side-specific-adjudication-contract-v4-2026-09-01.json"
    )
    result_v4 = _load(
        ACTION_VALUE
        / "polymarket-nfl-september13-early-side-specific-adjudication-v4-2026-09-01.json"
    )
    raw = (
        ROOT
        / "data/polymarket-nfl-september13-early-monotone-catalog-v1/raw/events.json"
    )
    journal = [
        json.loads(line)
        for line in (
            ROOT
            / "data/polymarket-nfl-september13-early-monotone-catalog-v1/request-journal.jsonl"
        )
        .read_text()
        .splitlines()
    ]

    for payload, field in (
        (catalog_contract, "contract_sha256"),
        (catalog_result, "result_sha256"),
        (contract_v1, "contract_sha256"),
        (failure_v1, "result_sha256"),
        (contract_v2, "contract_sha256"),
        (failure_v2, "result_sha256"),
        (contract_v3, "contract_sha256"),
        (result_v3, "result_sha256"),
        (contract_v4, "contract_sha256"),
        (result_v4, "result_sha256"),
    ):
        assert _canonical_hash(payload, field) == payload[field]
    assert _file_hash(raw) == (
        "e086bd126636f0d7726d3ce56b58d238d39c11d17c25b7147d63566239cc4ee9"
    )
    assert _file_hash(
        ROOT
        / "data/polymarket-nfl-september13-early-monotone-catalog-v1/request-journal.jsonl"
    ) == "35dfc54dbbe82266f8a05fa7db33a3d3d779115765a7ab1e9b3b19c346fa6911"
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert catalog_result["capture"]["returned_event_count"] == 12
    assert catalog_result["screen"]["included_event_count"] == 10
    assert catalog_result["screen"]["complete_relation_count"] == 2_978
    assert (
        catalog_result["screen"]["candidate_count_strictly_below_payout_floor"]
        == 120
    )
    assert failure_v1["failure"]["error"] == "bestBid is missing"
    assert failure_v1["implementation_correction"]["mistake"].startswith(
        "the side-specific helper validated both"
    )
    assert failure_v2["failure"]["error"] == "bestBid is missing"
    assert "genuinely lacked" in failure_v2["implementation_correction"]["mistake"]
    assert adjudicate_catalog_side_specific(contract_v4) == result_v4

    screen = result_v4["screen"]
    assert screen["source_proved_relation_count"] == 2_978
    assert screen["price_complete_relation_count"] == 2_965
    assert screen["price_incomplete_relation_count"] == 13
    assert screen["strict_side_specific_subfloor_count"] == 0
    assert screen["all_side_specific_sums_at_or_above_payout_floor"] is False
    assert {
        row["event_slug"] for row in screen["price_incomplete_relations"]
    } == {"nfl-atl-pit-2026-09-13"}
    best = screen["best_side_specific_relation"]
    assert best["event_slug"] == "nfl-nyj-ten-2026-09-13"
    assert Decimal(best["side_specific_rejection_sum_pUSD"]) == Decimal("1.04")
    assert (
        result_v4["adjudication"]["status"]
        == "retained_catalog_price_incomplete_no_depth_escalation"
    )
    assert result_v4["adjudication"]["book_or_fee_request_permitted"] is False


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
        "317ecd24a990456be2e8c64f644f7481309d7c5e663c33274d281270483a848f",
        "9c80f1a188c059890c682f16f367cc472667bda595482d5d0db26ebda2d014bb",
        "61fb2010d57b3295dd0ca859345c54404372dbd8b0f7ac4bca42d3fb0e40ddfd",
        "126b1dc61fa379458aaa88a8edef899be437d33fda6cba1a7a29e3536cbe856f",
        "01963e711663f496a296efbbd806a8f6b8271c37b96897b2cd9b34c081edb94f",
        "1965d997ba11fdeb51cf5bac40e9a13569640d724b4f58624a59fa230e9d69f9",
    } <= hashes
    assert "25189367_ms_skew" in row["current_status"]
    assert "Patriots_Seahawks" in row["current_status"]
    assert "zero_strict_sub_floor_candidates" in row["current_status"]
    assert "13_Atlanta_Pittsburgh_relations" in row["current_status"]
    assert "13_price_incomplete_Atlanta_Pittsburgh_relations" in row["next_action"]
    terminal = next(
        item
        for item in registry["terminal_do_not_repeat"]
        if item["family"]
        == "polymarket_future_NFL_2026_09_13_through_2026_09_21_complete_side_specific_monotone_catalog"
    )
    assert terminal["canonical_result_sha256"] == (
        "61fb2010d57b3295dd0ca859345c54404372dbd8b0f7ac4bca42d3fb0e40ddfd"
    )
    early_terminal = next(
        item
        for item in registry["terminal_do_not_repeat"]
        if item["family"]
        == "polymarket_NFL_2026_09_13T00_00_00Z_through_20_25_00Z_monotone_catalog"
    )
    assert early_terminal["canonical_result_sha256"] == (
        "1965d997ba11fdeb51cf5bac40e9a13569640d724b4f58624a59fa230e9d69f9"
    )
