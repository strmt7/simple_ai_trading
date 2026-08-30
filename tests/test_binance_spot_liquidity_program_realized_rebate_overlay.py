from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-spot-liquidity-program-realized-organic-maker-rebate-overlay-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    return hashlib.sha256(_canonical(body)).hexdigest()


def test_artifact_and_retained_sources_reconstruct() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert _self_hash(artifact) == artifact["result_sha256"]

    for source in artifact["sources"]:
        payload = (ROOT / source["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == source["file_sha256"]
        assert json.loads(payload)["result_sha256"] == source["result_sha256"]


def test_realized_overlay_does_not_promote_or_double_count_base_flow() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    decision = artifact["adjudication"]
    economics = artifact["economic_contract"]
    overlap = artifact["overlap_adjudication"]

    assert decision["accepted_edge"] is True
    assert decision["market_direction_forecast_required"] is False
    assert decision["standalone_market_making_strategy_accepted"] is False
    assert decision["fresh_hypothetical_order_profit_proved"] is False
    assert decision["public_forward_profit_floor_quote_units"] == "0"
    assert economics["rate_assumption"] == "none"
    assert economics["realized_spot_rebate_history_security"] == "USER_DATA"
    assert overlap["accepted_bStocks_zero_maker_fee_overlay_distinct"] is True
    assert overlap["bnb_discount_distinct"] is True
    assert artifact["authority"]["official_venue_API_requests"] == 0
    assert artifact["authority"]["orders_or_cancels"] == 0


def test_registry_accepts_only_the_scoped_overlay_and_binds_artifact() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry) == registry["result_sha256"]
    assert registry["accepted_edge_count"] == 26

    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 15
    )
    assert family["mechanism"] == "spot_market_maker_rebates"
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in family["canonical_artifacts"]
    assert "exact realized positive Binance Spot Liquidity Program" in registry[
        "accepted_edge_scope"
    ]
