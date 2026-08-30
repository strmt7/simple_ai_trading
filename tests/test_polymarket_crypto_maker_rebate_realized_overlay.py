from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-crypto-maker-rebate-realized-organic-flow-overlay-v1-2026-08-30.json"
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


def test_realized_overlay_does_not_promote_standalone_market_making() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    adjudication = artifact["adjudication"]
    recurrence = artifact["public_recurrence_evidence"]

    assert adjudication["accepted_edge"] is True
    assert adjudication["market_direction_forecast_required"] is False
    assert adjudication["standalone_market_making_strategy_accepted"] is False
    assert adjudication["fresh_hypothetical_order_profit_proved"] is False
    assert adjudication["public_forward_profit_floor_pusd"] == "0"
    assert recurrence["public_program_payment_recurrence_proved"] is True
    assert recurrence["fresh_hypothetical_order_payout_floor_pusd"] == "0"
    assert recurrence["wallets_with_positive_receipts"] == 10
    assert recurrence["wallets_with_all_fourteen_utc_dates"] == 8
    assert Decimal(recurrence["scoped_wallet_day_btc_eth_sol_rebate_pusd"]) > 0
    assert artifact["authority"]["orders_or_cancels"] == 0
    assert artifact["authority"]["signed_or_venue_API_requests"] == 0


def test_registry_accepts_only_the_scoped_overlay_and_binds_artifact() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry) == registry["result_sha256"]
    assert registry["accepted_edge_count"] == 25

    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 17
    )
    assert family["mechanism"] == "paired_maker_rebates_and_liquidity_rewards"
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in family["canonical_artifacts"]
    assert "exact realized positive Polymarket crypto maker rebates" in registry[
        "accepted_edge_scope"
    ]
