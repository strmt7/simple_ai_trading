from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / (
    "docs/model-research/action-value/"
    "binance-stocks-fee-current-source-conflict-v1-2026-08-31.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    expected = str(body.pop("result_sha256"))
    actual = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    assert actual == expected
    return actual


def test_binance_stocks_current_source_conflict_is_fail_closed() -> None:
    result = _load(RESULT)
    _self_hash(result)

    live = result["live_fee_schedule_side"]
    snapshot = ROOT / live["rendered_semantic_snapshot_path"]
    journal = ROOT / live["request_journal_path"]
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == live[
        "rendered_semantic_snapshot_sha256"
    ]
    assert hashlib.sha256(journal.read_bytes()).hexdigest() == live[
        "request_journal_sha256"
    ]
    assert result["source_conflict"] == {
        "announcement_end_utc": "2026-09-30T23:59:00Z",
        "conflict_material": True,
        "live_fee_schedule_end_utc": "2026-08-31T00:00:00Z",
        "resolution": "unresolved_fail_closed",
        "same_fee_amounts": True,
    }
    assert result["adjudication"]["current_discount_credit_permitted"] is False
    assert result["adjudication"]["public_current_incremental_saving_floor_USD"] == "0"
    assert result["adjudication"]["accepted_edge_count_change"] == 0


def test_registry_routes_direct_stocks_to_literal_resolution_trigger() -> None:
    registry = _load(REGISTRY)
    _self_hash(registry)
    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 5
    )
    binding = next(
        item
        for item in family["canonical_artifacts"]
        if item["path"] == RESULT.relative_to(ROOT).as_posix()
    )
    assert binding["result_sha256"] == _load(RESULT)["result_sha256"]
    assert "fail_closed_zero" in family["direct_stocks_current_status"]
    assert "explicit_official_correction" in family["direct_stocks_retry_trigger"]
    assert "do_not_repeat_or_alias_the_consumed_CMS" in family[
        "direct_stocks_next_action"
    ]
