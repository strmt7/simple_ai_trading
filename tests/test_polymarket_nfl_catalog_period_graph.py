from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
CONTRACT = (
    ACTION
    / "polymarket-nfl-september13-catalog-period-graph-contract-v1-2026-09-01.json"
)
RESULT = (
    ACTION
    / "polymarket-nfl-september13-catalog-period-graph-result-v1-2026-09-01.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    canonical = json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_retained_catalog_period_graph_is_source_bound_and_fail_closed() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    registry = _load(REGISTRY)

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    assert result["aggregate_screen"] == {
        **result["aggregate_screen"],
        "price_complete_relation_count": 76607,
        "price_incomplete_relation_count": 7901,
        "relation_count": 84508,
        "strict_side_specific_subfloor_count": 0,
    }
    assert result["aggregate_screen"]["best_complete_relation"][
        "side_specific_rejection_sum_pUSD"
    ] == "1.04"
    assert result["adjudication"]["book_or_fee_request_permitted"] is False
    assert result["authority"]["new_network_requests"] == 0
    rank_thirty = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 30
    )
    assert {
        "path": str(RESULT.relative_to(ROOT)).replace("\\", "/"),
        "result_sha256": result["result_sha256"],
    } in rank_thirty["canonical_artifacts"]
