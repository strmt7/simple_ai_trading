from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.adjudicate_polymarket_mlb_cross_period_catalog import adjudicate


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
DATA = ROOT / "data/polymarket-future-mlb-cross-period-catalog-v1"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
REGISTRY_HASH = "9459be90ad52d85f8d23824b04aca3e39bc397c941b47735aca4342a78f00d82"


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


def _result() -> dict[str, object]:
    return _load(
        ACTION_VALUE
        / "polymarket-future-mlb-cross-period-catalog-result-v1-2026-08-29.json"
    )


def test_one_request_catalog_is_hash_bound_and_explicitly_partial() -> None:
    contract = _load(
        ACTION_VALUE
        / "polymarket-future-mlb-cross-period-catalog-contract-v1-2026-08-29.json"
    )
    result = _result()
    raw = DATA / "raw/events.json"
    journal = [
        json.loads(line)
        for line in (DATA / "request-journal.jsonl").read_text().splitlines()
    ]

    assert contract["contract_sha256"] == (
        "b39b5f66ddb67dc634c76c6244769bb3b01a6c18098f11ca5d6d999a55cc58ba"
    )
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "9b981f0e82c8d26272c1f5f1d7ff580576cae8734e3b697ef3932b4a295a4e14"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _file_hash(raw) == (
        "5a14b8109e232f5d9ff0b6ad9bb933372072848a141ce8555c1f212e4bd597fc"
    )
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[0]["method"] == "GET"
    assert journal[1]["response_sha256"] == _file_hash(raw)
    assert result["capture"]["returned_event_count"] == 100
    assert result["capture"]["limit"] == 100
    assert result["capture"]["population_complete_under_frozen_filter"] is False
    assert result["screen"]["future_event_count_at_completed_request"] == 92
    assert result["screen"]["exact_cross_period_relation_count"] == 5
    assert result["screen"]["candidate_count_strictly_below_payout_floor"] == 0
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["fee_requests"] == 0


def test_offline_adjudication_retains_every_rejected_relation_without_refetch() -> None:
    artifact = _load(
        ACTION_VALUE
        / "polymarket-future-mlb-cross-period-catalog-adjudication-v1-2026-08-29.json"
    )

    assert artifact["result_sha256"] == (
        "d3ba85e995753d781178fdf6144ac0cb7520d2b1830525cd4be1aad1a5b5b598"
    )
    assert _canonical_hash(artifact, "result_sha256") == artifact["result_sha256"]
    assert _file_hash(ROOT / artifact["implementation"]["path"]) == (
        "971fccaf381fd3e0295ff24e12d26361d560e327dda3b3725f957580cf6901bd"
    )
    assert artifact["population"]["partial"] is True
    screen = artifact["complete_retained_page_screen"]
    assert screen["exact_cross_period_relation_count"] == 5
    assert screen["candidate_count_strictly_below_payout_floor"] == 0
    assert [row["gamma_displayed_price_sum_pUSD"] for row in screen["relations"]] == [
        "1.260",
        "1.31",
        "1.370",
        "1.375",
        "1.405",
    ]
    assert screen["best_relation"]["event_slug"] == "mlb-ari-sf-2026-08-29"
    assert artifact["retained_raw"]["refetch_count"] == 0
    assert artifact["adjudication"]["status"] == (
        "retained_partial_page_rejected_before_books_and_fees"
    )
    assert artifact["authority"]["book_requests"] == 0
    assert artifact["authority"]["fee_requests"] == 0


def test_offline_adjudication_rejects_changed_raw_bytes() -> None:
    result = _result()
    raw = (DATA / "raw/events.json").read_bytes() + b" "

    with pytest.raises(RuntimeError, match="raw hash mismatch"):
        adjudicate(
            result=result,
            raw=raw,
            result_path="result.json",
            raw_path="events.json",
        )


def test_registry_routes_partial_catalog_without_complete_population_claim() -> None:
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
        "b39b5f66ddb67dc634c76c6244769bb3b01a6c18098f11ca5d6d999a55cc58ba",
        "9b981f0e82c8d26272c1f5f1d7ff580576cae8734e3b697ef3932b4a295a4e14",
        "d3ba85e995753d781178fdf6144ac0cb7520d2b1830525cd4be1aad1a5b5b598",
    } <= hashes
    assert registry["accepted_edge_count"] == 21
