from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/binance/"
    "spot-stp-transfer-structural-value-preflight-v1-2026-08-30.json"
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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    return _sha256(_canonical(body))


def test_official_sources_and_request_boundary_reconstruct() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    assert _self_hash(artifact) == artifact["result_sha256"]
    for source in (
        artifact["sources"]["official_stp_faq"],
        artifact["sources"]["current_exchange_info"],
        artifact["sources"]["ordinary_internal_transfer_index"],
    ):
        assert _sha256((ROOT / source["path"]).read_bytes()) == source["sha256"]

    intent = json.loads(
        (ROOT / artifact["sources"]["request_intent"]).read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (ROOT / artifact["sources"]["request_receipt"]).read_text(encoding="utf-8")
    )
    assert intent["method"] == receipt["method"] == "GET"
    assert intent["url"] == receipt["url"]
    assert intent["authority"] == "public_unauthenticated_read_only"
    assert receipt["credentials_used"] is False
    assert receipt["status_code"] == 200
    assert (
        receipt["payload_sha256"] == artifact["sources"]["official_stp_faq"]["sha256"]
    )
    comparator = artifact["ordinary_internal_transfer_comparator"]
    assert comparator["official_api_exists"] is True
    assert comparator["exact_fee_latency_limit_and_atomicity_bound"] is False


def test_transfer_example_conserves_aggregate_inventory_and_proves_no_profit() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    identity = artifact["economic_identity"]
    assert (
        Decimal(identity["example_maker_base_change"])
        + Decimal(identity["example_taker_base_change"])
        == Decimal(identity["aggregate_base_change"])
        == 0
    )
    assert (
        Decimal(identity["example_maker_quote_change"])
        + Decimal(identity["example_taker_quote_change"])
        == Decimal(identity["aggregate_quote_change"])
        == 0
    )
    assert Decimal(identity["example_maker_base_change"]) * Decimal(
        identity["example_price"]
    ) == Decimal(identity["example_prevented_notional"])
    assert identity["maker_order_remains_open"] is True
    assert identity["taker_fill_count"] == 0
    assert identity["example_is_live_fee_evidence"] is False

    verdict = artifact["verdict"]
    assert verdict["accepted_edge"] is False
    assert verdict["profitability_claim"] is False
    assert verdict["public_forward_profit_floor_quote_units"] == "0"
    assert verdict["ordinary_internal_transfer_comparator_bound"] is False
    assert artifact["authority"]["orders_or_cancellations"] == 0


def test_current_configuration_and_rank_five_lineage_reconstruct() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    expected_modes = [
        "EXPIRE_TAKER",
        "EXPIRE_MAKER",
        "EXPIRE_BOTH",
        "DECREMENT",
        "TRANSFER",
    ]
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        row = artifact["production_configuration"]["symbols"][symbol]
        assert row == {
            "allowed_modes": expected_modes,
            "default_mode": "EXPIRE_MAKER",
            "status": "TRADING",
        }

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry) == registry["result_sha256"]
    rank_five = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 5
    )
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in rank_five["canonical_artifacts"]
    assert "STP_TRANSFER" in rank_five["retry_trigger"]
    assert registry["accepted_edge_count"] == 21
