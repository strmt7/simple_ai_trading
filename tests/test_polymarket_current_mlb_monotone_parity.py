from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
EVENT_CONTRACT = ACTION_VALUE / (
    "polymarket-current-mlb-exact-event-inventory-contract-v1-2026-08-29.json"
)
EVENT_RESULT = ACTION_VALUE / (
    "polymarket-current-mlb-exact-event-inventory-result-v1-2026-08-29.json"
)
PARITY_CONTRACT = ACTION_VALUE / (
    "polymarket-current-mlb-monotone-parity-contract-v1-2026-08-29.json"
)
ADJUDICATION = ACTION_VALUE / (
    "polymarket-current-mlb-monotone-parity-failure-adjudication-v1-2026-08-29.json"
)
EVENT_RAW = ROOT / "data/polymarket-current-mlb-exact-event-inventory-v1/raw/event.json"
EVENT_JOURNAL = ROOT / (
    "data/polymarket-current-mlb-exact-event-inventory-v1/request-journal.jsonl"
)
BOOK_RAW = ROOT / "data/polymarket-current-mlb-monotone-parity-v1/raw/books.json"
BOOK_JOURNAL = ROOT / (
    "data/polymarket-current-mlb-monotone-parity-v1/request-journal.jsonl"
)
CAPTURE_TOOL = ROOT / "tools/capture_polymarket_current_mlb_exact_event_inventory.py"
SCREEN_TOOL = ROOT / "tools/screen_polymarket_current_mlb_monotone_parity.py"
ADJUDICATOR = ROOT / (
    "tools/adjudicate_polymarket_current_mlb_monotone_parity_failure.py"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EVENT_CONTRACT_HASH = "6e14586e887cbc8777eaca1c6a0be00fea87c8a87741fc7b846e4bf745b4626e"
EVENT_RESULT_HASH = "e274e3b05227022eb8c021fecdfa1a42e369ba30175ba906b24d6fd8459da80d"
PARITY_CONTRACT_HASH = (
    "231ebf5e9078bc14c8acd3d8274bc98c0200776621b830b64c689a34cdd204b8"
)
ADJUDICATION_HASH = "1e75e049abb116955294d878830f940491fe4044f09c7e3564ad2761c0129178"
REGISTRY_HASH = "6f5b4a277f1c3cea006de5d8a4a1cc6a7ba306b2689a37b2e93f1c44006fb7c2"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_exact_event_inventory_is_active_complete_and_action_free() -> None:
    contract = _load(EVENT_CONTRACT)
    result = _load(EVENT_RESULT)
    receipt = json.loads(EVENT_JOURNAL.read_text(encoding="ascii"))

    assert contract["contract_sha256"] == EVENT_CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == EVENT_CONTRACT_HASH
    assert result["result_sha256"] == EVENT_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == EVENT_RESULT_HASH
    assert result["capture"]["exact_slug_match"] is True
    assert result["capture"]["event_active_and_open"] is True
    assert result["capture"]["embedded_market_count"] == 16
    assert result["capture"]["active_accepting_market_count"] == 16
    assert result["discovery"]["active_sports_market_types"] == [
        "moneyline",
        "nrfi",
        "spreads",
        "totals",
    ]
    assert result["authority"]["public_unauthenticated_GET_requests"] == 1
    assert result["authority"]["book_or_price_requests"] == 0
    assert result["authority"]["authenticated_requests"] == 0
    assert (
        result["implementation"]["sha256"]
        == hashlib.sha256(CAPTURE_TOOL.read_bytes()).hexdigest()
    )
    assert (
        receipt["response_sha256"] == hashlib.sha256(EVENT_RAW.read_bytes()).hexdigest()
    )


def test_frozen_contract_proves_37_guaranteed_payoff_relations_before_books() -> None:
    contract = _load(PARITY_CONTRACT)

    assert contract["contract_sha256"] == PARITY_CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == PARITY_CONTRACT_HASH
    assert contract["source_evidence"]["event_result_sha256"] == EVENT_RESULT_HASH
    assert contract["capture"]["request_count"] == 1
    assert contract["capture"]["retry_permitted"] is False
    assert contract["capture"]["token_count"] == 30
    assert contract["economics"]["quantity_shares_each_leg"] == "5"
    assert contract["economics"]["adverse_ticks_per_leg"] == 2
    relations = contract["payoff_proof"]["relations"]
    assert len(relations) == 37
    assert {row["family"] for row in relations} == {
        "BOS_margin",
        "NYY_margin",
        "total_runs",
    }
    assert {
        row["package"] for row in contract["payoff_proof"]["terminal_states_per_share"]
    } == {
        "1",
        "2",
    }
    assert (
        contract["implementation"]["sha256"]
        == hashlib.sha256(SCREEN_TOOL.read_bytes()).hexdigest()
    )


def test_complete_retained_batch_rejects_every_relation_after_costs() -> None:
    adjudication = _load(ADJUDICATION)
    receipt = json.loads(BOOK_JOURNAL.read_text(encoding="ascii"))
    books = json.loads(BOOK_RAW.read_text(encoding="utf-8"))

    assert adjudication["result_sha256"] == ADJUDICATION_HASH
    assert _canonical_hash(adjudication, "result_sha256") == ADJUDICATION_HASH
    assert (
        receipt["response_sha256"] == hashlib.sha256(BOOK_RAW.read_bytes()).hexdigest()
    )
    assert receipt["request_token_count"] == len(books) == 30
    assert all(
        [Decimal(level["price"]) for level in book["asks"]]
        == sorted([Decimal(level["price"]) for level in book["asks"]], reverse=True)
        for book in books
    )
    assert (
        adjudication["failure_diagnosis"][
            "outcome_aware_sensitivity_can_support_snapshot_promotion"
        ]
        is False
    )
    economics = adjudication["retained_evidence_sensitivity"]["economics"]
    assert (
        economics["relation_count"] == economics["complete_depth_relation_count"] == 37
    )
    assert economics["frozen_candidate_count"] == 0
    best = economics["best_relation"]
    assert best["family"] == "NYY_margin"
    assert best["superset_threshold"] == 4
    assert best["subset_threshold"] == 5
    assert Decimal(best["stressed_after_fee_profit_floor_pUSD"]) == Decimal("-0.53262")
    assert adjudication["adjudication"]["accepted_edge"] is False
    assert (
        adjudication["adjudication"]["candidate_for_fresh_preregistered_recurrence"]
        is False
    )
    assert (
        adjudication["implementation"]["sha256"]
        == hashlib.sha256(ADJUDICATOR.read_bytes()).hexdigest()
    )


def test_registry_terminalizes_only_this_exact_event_snapshot() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "polymarket_cross_market_exact_multi_outcome_subset_equivalence"
    )
    assert [item["result_sha256"] for item in row["canonical_artifacts"][-4:]] == [
        EVENT_CONTRACT_HASH,
        EVENT_RESULT_HASH,
        PARITY_CONTRACT_HASH,
        ADJUDICATION_HASH,
    ]
    terminal = next(
        item
        for item in registry["terminal_do_not_repeat"]
        if item["family"]
        == "polymarket_current_BOS_NYY_moneyline_spread_totals_monotone_parity_snapshot"
    )
    assert terminal["canonical_result_sha256"] == ADJUDICATION_HASH
