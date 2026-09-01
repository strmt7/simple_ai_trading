from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
CONTRACT = (
    ACTION
    / "polymarket-nfl-september13-period-margin-contract-v1-2026-09-01.json"
)
RESULT = (
    ACTION
    / "polymarket-nfl-september13-period-margin-result-v1-2026-09-01.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
AUDIT = ACTION / "accepted-edge-profitability-durability-audit-v1-2026-08-30.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    canonical = json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_retained_period_margin_cover_is_source_bound_and_fail_closed() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    registry = _load(REGISTRY)
    audit = _load(AUDIT)

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    assert _self_hash(audit, "result_sha256") == audit["result_sha256"]
    assert result["aggregate_screen"] == {
        **result["aggregate_screen"],
        "price_complete_relation_count": 215,
        "price_incomplete_relation_count": 12,
        "relation_count": 227,
        "strict_side_specific_subfloor_count": 0,
    }
    best = result["aggregate_screen"]["best_complete_relation"]
    assert best["side_specific_rejection_sum_pUSD"] == "1.53"
    assert best["tie_payout_pUSD"] == "1.5"
    assert result["adjudication"]["book_or_fee_request_permitted"] is False
    assert result["authority"] == {
        **result["authority"],
        "credentials_used": False,
        "new_network_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
    }
    rank_thirty = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 30
    )
    assert {
        "path": str(RESULT.relative_to(ROOT)).replace("\\", "/"),
        "result_sha256": result["result_sha256"],
    } in rank_thirty["canonical_artifacts"]
    assert len(registry["terminal_do_not_repeat"]) == 144
    assert audit["source_binding"]["registry_result_sha256"] == registry[
        "result_sha256"
    ]
    assert "polymarket_nfl_september13_period_margin_terminal_trigger" in audit[
        "routing"
    ]
