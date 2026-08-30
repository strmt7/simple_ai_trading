import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
RESULT = ACTION_VALUE / (
    "binance-convert-limit-order-simple-earn-capital-efficiency-"
    "candidate-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical_hash(document: dict) -> str:
    body = dict(document)
    claimed = body.pop("result_sha256")
    actual = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    assert actual == claimed
    return actual


def test_convert_limit_order_simple_earn_candidate_is_fail_closed() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    result_hash = _canonical_hash(result)
    assert result_hash == (
        "3b33ca4ef8c03a609bef1665ccfc2104a3f6585033770f0bb99ec3c5699949f8"
    )

    for name in ("request_contract", "request_journal", "official_current_cms"):
        binding = result["source_binding"][name]
        payload = (ROOT / binding["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == binding["file_sha256"]

    cms_binding = result["source_binding"]["official_current_cms"]
    cms = json.loads((ROOT / cms_binding["path"]).read_text(encoding="utf-8"))
    assert cms["success"] is True
    assert cms["data"]["code"] == "de7bae694a564005b7ebc5f4693ab865"
    body = cms["data"]["body"]
    assert "generate rewards while waiting for the limit order execution" in body
    assert "your limit order will not be executed" in body
    assert "Daily redemption limits apply" in body

    terms = result["current_primary_terms"]
    assert terms["limit_hit_fill_guaranteed"] is False
    assert (
        terms["public_exact_asset_rate_or_positive_floor_published_in_this_contract"]
        is False
    )
    economics = result["rejection_first_economics"]
    assert economics["public_forward_reward_floor_bps"] == "0"
    maximum = economics["sensitivity_only_reward_bps"][1]
    assert Decimal(maximum["seven_day_reward_bps"]) < Decimal("1")
    assert result["verdict"]["accepted_edge"] is False
    assert result["verdict"]["profitability_proved"] is False

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    _canonical_hash(registry)
    rank_three = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["priority_rank"] == 3
    )
    assert any(
        artifact["result_sha256"] == result_hash
        for artifact in rank_three["canonical_artifacts"]
    )
