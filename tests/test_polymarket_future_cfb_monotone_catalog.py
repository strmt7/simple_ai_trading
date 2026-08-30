from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from tools.screen_polymarket_cfb_monotone_catalog import _validate_contract


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/action-value"
CONTRACT = BASE / ("polymarket-future-cfb-monotone-catalog-contract-v1-2026-08-30.json")
RESULT = BASE / ("polymarket-future-cfb-monotone-catalog-result-v1-2026-08-30.json")
RAW_DIR = ROOT / "data/polymarket-future-cfb-monotone-catalog-v1"
RAW = RAW_DIR / "raw/events.json"
JOURNAL = RAW_DIR / "request-journal.jsonl"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_complete_one_request_catalog_excludes_consumed_events() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    raw = RAW.read_bytes()
    payload = json.loads(raw)
    journal = [json.loads(line) for line in JOURNAL.read_text().splitlines()]

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    for implementation in contract["implementations"]:
        assert (
            hashlib.sha256((ROOT / implementation["path"]).read_bytes()).hexdigest()
            == (implementation["sha256"])
        )
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1]["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["capture"]["returned_event_count"] == len(payload["events"]) == 18
    assert result["capture"]["population_complete_under_frozen_filter"] is True
    assert "next_cursor" not in payload
    slugs = {event["slug"] for event in payload["events"]}
    assert "cfb-ballst-ohiost-2026-09-05" not in slugs
    assert "cfb-clmsn-lsu-2026-09-05" not in slugs


def test_all_relations_are_retained_and_reject_before_books() -> None:
    result = _load(RESULT)
    screen = result["screen"]

    assert screen["included_event_count"] == 13
    assert screen["excluded_event_count"] == 5
    assert screen["complete_relation_count"] == len(screen["relations"]) == 19
    assert screen["candidate_count_strictly_below_payout_floor"] == 0
    best = screen["best_relation"]
    assert best["event_slug"] == "cfb-sjst-emich-2026-09-04"
    assert best["family"] == "full_game_total"
    assert Decimal(best["displayed_price_sum_per_share_pUSD"]) == Decimal("1.02")
    assert screen["depth_candidate"] is None
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["fee_requests"] == 0
    assert result["adjudication"]["accepted_edge"] is False


def test_population_contract_fails_closed_on_series_drift() -> None:
    contract = _load(CONTRACT)
    drifted = deepcopy(contract)
    drifted["capture"]["series_id"] = "different"
    drifted["contract_sha256"] = _self_hash(drifted, "contract_sha256")

    with pytest.raises(RuntimeError, match="population contract changed"):
        _validate_contract(drifted, CONTRACT.resolve())


def test_registry_terminalizes_only_the_exact_catalog_window() -> None:
    result = _load(RESULT)
    registry = _load(REGISTRY)

    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 30
    )
    assert {
        "path": RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": result["result_sha256"],
    } in family["canonical_artifacts"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["canonical_result_sha256"] == result["result_sha256"]
    )
    assert "do_not_paginate_narrow_repeat" in terminal["reason"]
    assert "2026_09_03T00_00_00Z" in family["retry_trigger"]
