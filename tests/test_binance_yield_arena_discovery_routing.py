from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-yield-arena-discovery-routing-addendum-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_yield_arena_routes_to_product_families_without_edge_claim() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == _self_hash(artifact)
    assert artifact["routing_contract"]["yield_arena_is_distinct_cash_flow"] is False
    assert artifact["decision"]["new_ranked_hypothesis"] is False
    assert artifact["decision"]["new_accepted_edge"] is False
    assert artifact["decision"]["market_or_price_request_justified"] is False
    assert artifact["authority"]["signed_requests"] == 0

    registry = _load(REGISTRY)
    assert registry["result_sha256"] == _self_hash(registry)
    binding = {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    }
    for rank in (3, 8):
        family = next(
            row
            for row in registry["prioritized_hypotheses"]
            if row["priority_rank"] == rank
        )
        assert binding in family["canonical_artifacts"]
