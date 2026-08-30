from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
CATALOG_CONTRACT = ACTION / (
    "polymarket-cfb-september-5-complete-monotone-catalog-contract-v1-2026-08-30.json"
)
SOURCE_RESULT = ACTION / (
    "polymarket-cfb-september-5-complete-monotone-catalog-"
    "source-result-v1-2026-08-30.json"
)
CATALOG_RESULT = ACTION / (
    "polymarket-cfb-september-5-complete-monotone-catalog-result-v1-2026-08-30.json"
)
METADATA_CONTRACT = ACTION / (
    "polymarket-fordham-ndsu-catalog-metadata-contract-v1-2026-08-30.json"
)
METADATA = ACTION / "polymarket-fordham-ndsu-catalog-metadata-v1-2026-08-30.json"
PACKAGE_CONTRACT = ACTION / (
    "polymarket-fordham-ndsu-total-monotone-package-contract-v1-2026-08-30.json"
)
PACKAGE_RESULT = ACTION / (
    "polymarket-fordham-ndsu-total-monotone-package-result-v1-2026-08-30.json"
)
CATALOG_DATA = ROOT / "data/polymarket-cfb-september-5-complete-monotone-catalog-v1"
PACKAGE_DATA = ROOT / "data/polymarket-fordham-ndsu-total-monotone-package-v1"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
SEPT6_CATALOG_CONTRACT = ACTION / (
    "polymarket-cfb-september-6-complete-monotone-catalog-contract-v1-2026-08-30.json"
)
SEPT6_CATALOG_RESULT = ACTION / (
    "polymarket-cfb-september-6-complete-monotone-catalog-result-v1-2026-08-30.json"
)
SEPT6_METADATA_CONTRACT = ACTION / (
    "polymarket-mercyhurst-nmsu-catalog-metadata-contract-v1-2026-08-30.json"
)
SEPT6_METADATA = ACTION / (
    "polymarket-mercyhurst-nmsu-catalog-metadata-v1-2026-08-30.json"
)
SEPT6_PACKAGE_CONTRACT = ACTION / (
    "polymarket-mercyhurst-nmsu-margin-monotone-package-contract-v1-2026-08-30.json"
)
SEPT6_PACKAGE_RESULT = ACTION / (
    "polymarket-mercyhurst-nmsu-margin-monotone-package-result-v1-2026-08-30.json"
)
SEPT6_CATALOG_DATA = ROOT / (
    "data/polymarket-cfb-september-6-complete-monotone-catalog-v1"
)
SEPT6_PACKAGE_DATA = ROOT / (
    "data/polymarket-mercyhurst-nmsu-margin-monotone-package-v1"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return _sha256(_canonical(body))


def test_complete_catalog_is_one_request_hash_bound_and_excludes_consumed_games() -> None:
    contract = _load(CATALOG_CONTRACT)
    source = _load(SOURCE_RESULT)
    result = _load(CATALOG_RESULT)
    raw = (CATALOG_DATA / "raw/events.json").read_bytes()
    journal = [
        json.loads(row)
        for row in (CATALOG_DATA / "request-journal.jsonl").read_text().splitlines()
    ]

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(source, "result_sha256") == source["result_sha256"]
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1]["response_sha256"] == _sha256(raw)
    assert source["source_gate"]["passed"] is True
    screen = result["screen"]
    assert screen["returned_event_count"] == 89
    assert screen["population_complete_under_frozen_filter"] is True
    assert screen["excluded_consumed_event_count"] == 2
    assert screen["included_event_count"] == 58
    assert screen["complete_relation_count"] == len(screen["relations"]) == 88
    assert screen["candidate_count_strictly_below_payout_floor"] == 6
    consumed = {
        "cfb-ballst-ohiost-2026-09-05",
        "cfb-clmsn-lsu-2026-09-05",
    }
    assert {
        row["event_slug"]
        for row in screen["exclusions"]
        if row["reason"] == "excluded_consumed_event"
    } == consumed


def test_precommitted_best_package_fails_depth_and_freshness_before_fees() -> None:
    metadata_contract = _load(METADATA_CONTRACT)
    metadata = _load(METADATA)
    contract = _load(PACKAGE_CONTRACT)
    result = _load(PACKAGE_RESULT)
    books = (PACKAGE_DATA / "raw/books.json").read_bytes()
    journal = [
        json.loads(row)
        for row in (PACKAGE_DATA / "request-journal.jsonl").read_text().splitlines()
    ]

    assert _self_hash(metadata_contract, "contract_sha256") == metadata_contract[
        "contract_sha256"
    ]
    assert _self_hash(metadata, "result_sha256") == metadata["result_sha256"]
    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    assert metadata["selection"]["event_slug"] == "cfb-fordm-ndkst-2026-09-05"
    assert contract["package"]["token_names"] == ["over_56_5", "under_57_5"]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1]["response_sha256"] == _sha256(books)
    assert result["capture"]["fee_receipts"] == {}
    assert result["capture"]["book_timestamp_skew_ms"] == 148059
    assert result["capture"]["oldest_book_age_at_completion_ms"] == 300344
    assert result["capture"]["within_frozen_skew_gate"] is False
    assert result["capture"]["within_frozen_age_gate"] is False
    actual = result["economics"]["actual"]
    assert Decimal(actual["cost_pUSD"]) == Decimal("9.2")
    assert Decimal(actual["optimistic_zero_fee_profit_floor_pUSD"]) == Decimal(
        "-4.2"
    )
    assert result["adjudication"]["passes_frozen_candidate_gate"] is False
    assert result["adjudication"]["accepted_edge"] is False
    assert result["authority"]["credentials_used"] is False
    assert result["authority"]["orders_or_transactions"] == 0


def test_registry_terminalizes_exact_population_without_global_count_pinning() -> None:
    result = _load(PACKAGE_RESULT)
    registry = _load(REGISTRY)

    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 30
    )
    assert {
        "path": PACKAGE_RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": result["result_sha256"],
    } in family["canonical_artifacts"]
    assert "2026_09_06T23_59_59Z" in family["retry_trigger"]
    assert any(
        "other_five_observed_September_5" in shortcut
        for shortcut in family["prohibited_shortcuts"]
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["canonical_result_sha256"] == result["result_sha256"]
    )
    assert "do_not_repeat_paginate_narrow_refetch_or_cherry_pick" in terminal["reason"]
    rules = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "oldest-book-age ceiling" in rules
    assert "freeze one deterministic candidate ordering" in rules


def test_september6_complete_catalog_precommits_one_depth_candidate() -> None:
    contract = _load(SEPT6_CATALOG_CONTRACT)
    result = _load(SEPT6_CATALOG_RESULT)
    raw = (SEPT6_CATALOG_DATA / "raw/events.json").read_bytes()
    journal = [
        json.loads(row)
        for row in (SEPT6_CATALOG_DATA / "request-journal.jsonl")
        .read_text()
        .splitlines()
    ]

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1]["response_sha256"] == _sha256(raw)
    assert result["capture"]["population_complete_under_frozen_filter"] is True
    assert result["capture"]["returned_event_count"] == 24
    screen = result["screen"]
    assert screen["included_event_count"] == 15
    assert screen["complete_relation_count"] == len(screen["relations"]) == 31
    assert screen["candidate_count_strictly_below_payout_floor"] == 2
    assert screen["depth_candidate"]["event_slug"] == (
        "cfb-mhud34ce7-nmxst-2026-09-05"
    )
    assert Decimal(screen["depth_candidate"]["displayed_price_sum_per_share_pUSD"]) == Decimal("0.995")


def test_september6_best_package_fails_depth_before_fee_access() -> None:
    artifacts = (
        (SEPT6_METADATA_CONTRACT, "contract_sha256"),
        (SEPT6_METADATA, "result_sha256"),
        (SEPT6_PACKAGE_CONTRACT, "contract_sha256"),
        (SEPT6_PACKAGE_RESULT, "result_sha256"),
    )
    for path, field in artifacts:
        payload = _load(path)
        assert _self_hash(payload, field) == payload[field]

    metadata = _load(SEPT6_METADATA)
    result = _load(SEPT6_PACKAGE_RESULT)
    books = (SEPT6_PACKAGE_DATA / "raw/books.json").read_bytes()
    journal = [
        json.loads(row)
        for row in (SEPT6_PACKAGE_DATA / "request-journal.jsonl")
        .read_text()
        .splitlines()
    ]
    assert metadata["selection"]["event_id"] == "913009"
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1]["response_sha256"] == _sha256(books)
    assert result["capture"]["fee_receipts"] == {}
    assert result["capture"]["oldest_book_age_at_completion_ms"] == 1763317
    assert result["capture"]["book_timestamp_skew_ms"] == 337138
    assert Decimal(result["economics"]["actual"]["cost_pUSD"]) == Decimal("9.8")
    assert Decimal(
        result["economics"]["actual"]["optimistic_zero_fee_profit_floor_pUSD"]
    ) == Decimal("-4.8")
    assert result["adjudication"]["passes_frozen_candidate_gate"] is False

    registry = _load(REGISTRY)
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 30
    )
    assert {
        "path": SEPT6_PACKAGE_RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": result["result_sha256"],
    } in family["canonical_artifacts"]
    assert any(
        row["canonical_result_sha256"] == result["result_sha256"]
        for row in registry["terminal_do_not_repeat"]
    )
