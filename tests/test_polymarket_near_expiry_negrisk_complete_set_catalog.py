from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs/model-research/action-value/polymarket-near-expiry-negrisk-complete-set-catalog-contract-v1-2026-08-29.json"
)
RESULT = (
    ROOT
    / "docs/model-research/action-value/polymarket-near-expiry-negrisk-complete-set-catalog-result-v1-2026-08-29.json"
)
RAW = (
    ROOT / "data/polymarket-near-expiry-negrisk-complete-set-catalog-v1/raw/events.json"
)
JOURNAL = (
    ROOT
    / "data/polymarket-near-expiry-negrisk-complete-set-catalog-v1/request-journal.jsonl"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
CONTRACT_HASH = "d5b81adb03fd4fe322d9a54fbacbe15aa8a6a7e55512aa71e9aa361617f2c6e6"
RESULT_HASH = "96610d7cba90a2dc97489bd70c95b7d03568d5b89017ace1e8c92829c70cee14"
RAW_HASH = "1dd21f815a4564cb42711842b80c91e46d6b6a799dcefe2911d0a30e10ab61f2"
SEP6_CONTRACT = (
    ROOT
    / "docs/model-research/action-value/polymarket-sep6-negrisk-complete-set-catalog-contract-v1-2026-08-29.json"
)
SEP6_RESULT = (
    ROOT
    / "docs/model-research/action-value/polymarket-sep6-negrisk-complete-set-catalog-result-v1-2026-08-29.json"
)
SEP6_RAW = ROOT / "data/polymarket-sep6-negrisk-complete-set-catalog-v1/raw/events.json"
SEP6_JOURNAL = (
    ROOT / "data/polymarket-sep6-negrisk-complete-set-catalog-v1/request-journal.jsonl"
)
SEP6_CONTRACT_HASH = "18d513c0b54c6155897ae435cf9f4b8a0ef327f6072d39b122ebd4579b7f0972"
SEP6_RESULT_HASH = "3e3ae8fd8c98c93c3e2194425db5992f06aed412e07d32333525601c2b34bc52"
SEP6_RAW_HASH = "f430490f592cf58297c5f5f118b3fabc32488faa82b5b07b60815279bdc61050"
HOUR00_CONTRACT = (
    ROOT
    / "docs/model-research/action-value/polymarket-sep7-hour00-negrisk-complete-set-catalog-contract-v1-2026-08-29.json"
)
HOUR00_RESULT = (
    ROOT
    / "docs/model-research/action-value/polymarket-sep7-hour00-negrisk-complete-set-catalog-result-v1-2026-08-29.json"
)
HOUR00_RAW = (
    ROOT / "data/polymarket-sep7-hour00-negrisk-complete-set-catalog-v1/raw/events.json"
)
HOUR00_JOURNAL = (
    ROOT
    / "data/polymarket-sep7-hour00-negrisk-complete-set-catalog-v1/request-journal.jsonl"
)
HOUR00_CONTRACT_HASH = (
    "32dfa5e282f43204f73117fa2dba198c69171944bc41a85bf661e05789089439"
)
HOUR00_RESULT_HASH = "92734472ed41bccdc1d88c947b218e05fa35827cad6b1711ec192c06cf60cc64"
HOUR00_RAW_HASH = "4b0322846de9fd229591c460eaf1cb22d4b0ac4c9e6b52ac2f35c8a0bec99442"
SOURCE_SELECTED_CONTRACT = (
    ROOT
    / "docs/model-research/action-value/polymarket-source-selected-crypto-negrisk-series-contract-v1-2026-08-29.json"
)
SOURCE_SELECTED_RESULT = (
    ROOT
    / "docs/model-research/action-value/polymarket-source-selected-crypto-negrisk-series-result-v1-2026-08-29.json"
)
MULTI_STRIKES_CONTRACT = (
    ROOT
    / "docs/model-research/action-value/polymarket-btc-multi-strikes-weekly-negrisk-contract-v1-2026-08-29.json"
)
MULTI_STRIKES_RESULT = (
    ROOT
    / "docs/model-research/action-value/polymarket-btc-multi-strikes-weekly-negrisk-result-v1-2026-08-29.json"
)
REGISTRY_HASH = "0a34d7289331515f8e7b3f09e856fbc331ecbc3a91130fea20542a39ef211f60"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="ascii"))
    assert isinstance(payload, dict)
    return payload


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    raw = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def test_partial_catalog_is_hash_bound_and_stops_before_escalation() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    journal = [
        json.loads(line) for line in JOURNAL.read_text(encoding="ascii").splitlines()
    ]

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert result["result_sha256"] == RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == RAW_HASH
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[1]["status_code"] == 200
    assert journal[1]["response_sha256"] == RAW_HASH

    capture = result["capture"]
    assert capture["returned_event_count"] == 100
    assert capture["next_cursor_present"] is True
    assert capture["population_complete_under_frozen_filter"] is False
    screen = result["screen"]
    assert len(screen["classifications"]) == 100
    assert screen["fixed_negrisk_event_count"] == 49
    assert screen["candidate_count_strictly_below_payout_floor"] == 24
    assert screen["proof_candidate"] is None
    assert result["adjudication"]["status"] == "incomplete_catalog_no_escalation"
    assert result["authority"]["onchain_requests"] == 0
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["fee_requests"] == 0


def test_every_screened_event_reconstructs_its_gamma_sum() -> None:
    result = _load(RESULT)
    events = result["screen"]["events"]
    assert len(events) == 49
    for event in events:
        legs = event["legs"]
        assert len(legs) == event["market_count"]
        assert sum(Decimal(leg["yes_price_pUSD"]) for leg in legs) == Decimal(
            event["displayed_all_yes_sum_pUSD"]
        )
        assert event["gamma_role"] == (
            "rejection_only_never_acceptance_or_promotion_evidence"
        )

    best = result["screen"]["best_candidate"]
    assert best["event_slug"] == "fl1-lyo-hac-2026-08-29-exact-score"
    assert best["market_count"] == 17
    assert Decimal(best["displayed_all_yes_sum_pUSD"]) == Decimal("0.9450")
    assert Decimal(best["optimistic_profit_before_execution_costs_pUSD"]) == Decimal(
        "0.0550"
    )


def test_distinct_daily_window_remains_partial_and_spends_no_proof_requests() -> None:
    contract = _load(SEP6_CONTRACT)
    result = _load(SEP6_RESULT)
    journal = [
        json.loads(line)
        for line in SEP6_JOURNAL.read_text(encoding="ascii").splitlines()
    ]
    assert contract["contract_sha256"] == SEP6_CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == SEP6_CONTRACT_HASH
    assert result["result_sha256"] == SEP6_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == SEP6_RESULT_HASH
    assert hashlib.sha256(SEP6_RAW.read_bytes()).hexdigest() == SEP6_RAW_HASH
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[1]["response_sha256"] == SEP6_RAW_HASH

    capture = result["capture"]
    screen = result["screen"]
    assert capture["returned_event_count"] == 100
    assert capture["next_cursor_present"] is True
    assert capture["population_complete_under_frozen_filter"] is False
    assert screen["fixed_negrisk_event_count"] == 57
    assert screen["candidate_count_strictly_below_payout_floor"] == 2
    assert screen["proof_candidate"] is None
    best = screen["best_candidate"]
    assert best["event_slug"] == "mls-dal-skc-2026-09-05"
    assert best["market_count"] == 3
    assert Decimal(best["displayed_all_yes_sum_pUSD"]) == Decimal("0.985")
    assert result["authority"]["onchain_requests"] == 0
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["fee_requests"] == 0


def test_nonoverlapping_hourly_window_is_complete_and_rejected_before_escalation() -> (
    None
):
    contract = _load(HOUR00_CONTRACT)
    result = _load(HOUR00_RESULT)
    journal = [
        json.loads(line)
        for line in HOUR00_JOURNAL.read_text(encoding="ascii").splitlines()
    ]
    assert contract["contract_sha256"] == HOUR00_CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == HOUR00_CONTRACT_HASH
    assert result["result_sha256"] == HOUR00_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == HOUR00_RESULT_HASH
    assert hashlib.sha256(HOUR00_RAW.read_bytes()).hexdigest() == HOUR00_RAW_HASH
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[1]["response_sha256"] == HOUR00_RAW_HASH

    capture = result["capture"]
    screen = result["screen"]
    assert capture["returned_event_count"] == 6
    assert capture["next_cursor_present"] is False
    assert capture["population_complete_under_frozen_filter"] is True
    assert screen["fixed_negrisk_event_count"] == 5
    assert screen["candidate_count_strictly_below_payout_floor"] == 0
    assert screen["proof_candidate"] is None
    assert screen["best_candidate"] is None
    assert result["adjudication"]["status"] == (
        "complete_window_rejected_before_onchain_books_and_fees"
    )
    assert result["authority"]["onchain_requests"] == 0
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["fee_requests"] == 0


def test_registry_routes_partial_page_without_accepting_an_edge() -> None:
    registry = _load(REGISTRY)
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "polymarket_cross_market_exact_multi_outcome_subset_equivalence"
    )
    artifacts = row["canonical_artifacts"]
    assert artifacts[-10:] == [
        {"path": CONTRACT.relative_to(ROOT).as_posix(), "result_sha256": CONTRACT_HASH},
        {"path": RESULT.relative_to(ROOT).as_posix(), "result_sha256": RESULT_HASH},
        {
            "path": SEP6_CONTRACT.relative_to(ROOT).as_posix(),
            "result_sha256": SEP6_CONTRACT_HASH,
        },
        {
            "path": SEP6_RESULT.relative_to(ROOT).as_posix(),
            "result_sha256": SEP6_RESULT_HASH,
        },
        {
            "path": HOUR00_CONTRACT.relative_to(ROOT).as_posix(),
            "result_sha256": HOUR00_CONTRACT_HASH,
        },
        {
            "path": HOUR00_RESULT.relative_to(ROOT).as_posix(),
            "result_sha256": HOUR00_RESULT_HASH,
        },
        {
            "path": SOURCE_SELECTED_CONTRACT.relative_to(ROOT).as_posix(),
            "result_sha256": "a13f554d2b6b40dce3ea83236bff5cd99993f41aa1c988fe38eb99dc28d43f5e",
        },
        {
            "path": SOURCE_SELECTED_RESULT.relative_to(ROOT).as_posix(),
            "result_sha256": "73ae75ccf391a30ef592f649f63ef535e29865e1234bbff03021c806fd75268b",
        },
        {
            "path": MULTI_STRIKES_CONTRACT.relative_to(ROOT).as_posix(),
            "result_sha256": "ca2f7b556207a0d2d04c006ed502bf35a673a87642e0e5dd443227af8aba82d2",
        },
        {
            "path": MULTI_STRIKES_RESULT.relative_to(ROOT).as_posix(),
            "result_sha256": "f032753b45c82b2e0945d1a8c0e0d5fc01f8fb1727cdad34e73064c7590417ba",
        },
    ]
    assert "complete_series_specific_catalog_was_also_empty" in row["current_status"]
    assert "do_not_repeat_either_empty_forward_series_query" in row["next_action"]
    assert registry["accepted_edge_count"] == 21
