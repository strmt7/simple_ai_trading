from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.adjudicate_polymarket_crypto_interval_composition_delta import (  # noqa: E402
    _canonical_hash,
    _new_events,
)


ACTION = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION / (
    "polymarket-crypto-interval-composition-delta-contract-v1-2026-08-30.json"
)
SOURCE_RESULT = ACTION / (
    "polymarket-crypto-interval-composition-delta-source-result-v1-2026-08-30.json"
)
RESULT = ACTION / (
    "polymarket-crypto-interval-composition-delta-result-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_one_use_delta_is_source_bound_and_incomplete_before_books() -> None:
    contract = _load(CONTRACT)
    source = _load(SOURCE_RESULT)
    result = _load(RESULT)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(source, "result_sha256") == source["result_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["contract"] == {
        "path": CONTRACT.relative_to(ROOT).as_posix(),
        "sha256": contract["contract_sha256"],
    }
    raw_path = ROOT / contract["outputs"]["raw_path"]
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert raw_hash == source["capture"]["receipt"]["response_sha256"]
    assert raw_hash == result["retained_source"]["raw_sha256"]
    assert result["population"]["returned_event_count"] == 100
    assert result["population"]["new_event_count"] == 100
    assert result["population"]["cursor"] is not None
    assert result["population"]["complete_nonoverlapping_delta"] is False
    assert result["screen"]["package_count"] == 0
    assert result["adjudication"]["book_request_permitted"] is False
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["profitability_claim"] is False


def test_cutoff_is_exact_prior_maximum_created_at() -> None:
    contract = _load(CONTRACT)
    prior_path = ROOT / contract["population"]["prior_maximum_created_at_source"]
    prior = _load(prior_path)
    prior_maximum = max(str(row["createdAt"]) for row in prior["events"])

    assert prior_maximum == contract["population"]["created_after_utc"]


def test_delta_completeness_depends_on_crossing_cutoff_not_cursor() -> None:
    cutoff = datetime(2026, 8, 30, 16, 7, 16, 21321, tzinfo=UTC)
    base = {
        "id": "1",
        "slug": "example",
    }
    crossed = {
        "events": [
            {**base, "id": "2", "slug": "new", "createdAt": "2026-08-30T16:08:00Z"},
            {**base, "createdAt": "2026-08-30T16:07:00Z"},
        ],
        "next_cursor": "older-page",
    }
    not_crossed = {
        "events": [
            {**base, "id": "2", "slug": "newer", "createdAt": "2026-08-30T16:09:00Z"},
            {**base, "createdAt": "2026-08-30T16:08:00Z"},
        ],
        "next_cursor": "older-page",
    }

    rows, complete, cursor = _new_events(crossed, cutoff)
    assert len(rows) == 1
    assert complete is True
    assert cursor == "older-page"
    rows, complete, cursor = _new_events(not_crossed, cutoff)
    assert len(rows) == 2
    assert complete is False
    assert cursor == "older-page"


def test_registry_terminalizes_only_this_consumed_delta() -> None:
    registry = _load(REGISTRY)
    result = _load(RESULT)
    rank_31 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_BTC_ETH_SOL_created_event_interval_composition_delta_after_2026_08_30T16_07_16_021321Z"
    )

    assert any(
        artifact["path"] == RESULT.relative_to(ROOT).as_posix()
        and artifact["result_sha256"] == result["result_sha256"]
        for artifact in rank_31["canonical_artifacts"]
    )
    assert "do_not_paginate_narrow_refresh_alias_or_repeat" in rank_31["next_action"]
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
