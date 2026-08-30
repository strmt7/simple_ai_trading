import hashlib
import importlib.util
import json
from decimal import Decimal
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "capture_binance_block_trade_price_preflight"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME,
    ROOT / "tools" / "capture_binance_block_trade_price_preflight.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = TOOL
SPEC.loader.exec_module(TOOL)
LatestTicker = TOOL.LatestTicker
evaluate_block_trade = TOOL.evaluate_block_trade
ARTIFACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-spot-block-trade-public-price-concession-preflight-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical_hash(document: dict[str, object]) -> str:
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


def _ticker(*, event_ms: int = 1_000) -> LatestTicker:
    return LatestTicker(
        payload={
            "e": "24hrTicker",
            "E": event_ms,
            "s": "BTCUSDT",
            "b": "99.90",
            "B": "10",
            "a": "100.10",
            "A": "10",
        },
        received_at_utc="2026-08-30T00:00:00Z",
        received_monotonic_ns=10,
    )


def _block() -> dict[str, object]:
    return {
        "e": "blockTrade",
        "E": 1_510,
        "s": "BTCUSDT",
        "t": 7,
        "p": "100",
        "q": "2",
        "T": 1_500,
        "m": True,
    }


def test_evaluate_block_trade_uses_only_causal_ticker_and_zero_fee_floor() -> None:
    row = evaluate_block_trade(
        _block(),
        _ticker(),
        received_at_utc="2026-08-30T00:00:01Z",
        received_monotonic_ns=20,
        maximum_ticker_age_ms=2_000,
        block_fee_bps=Decimal("2.5"),
    )

    assert row["analyzable"] is True
    assert row["ticker_age_ms"] == 500
    assert row["buyer_strictly_positive_lower_bound"] is True
    assert row["seller_strictly_positive_lower_bound"] is True


def test_evaluate_block_trade_rejects_future_source_ticker() -> None:
    row = evaluate_block_trade(
        _block(),
        _ticker(event_ms=1_501),
        received_at_utc="2026-08-30T00:00:01Z",
        received_monotonic_ns=20,
        maximum_ticker_age_ms=2_000,
        block_fee_bps=Decimal("2.5"),
    )

    assert row["analyzable"] is False
    assert row["rejection_reason"] == "ticker_event_after_block_trade"


def test_zero_event_preflight_is_source_bound_and_fail_closed() -> None:
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    result_hash = _canonical_hash(result)
    assert result_hash == (
        "b7d60e0d9f3e30b2a62663ff1290be77e6309ac33a7d48776b6f5ea1c8dcfe68"
    )

    for binding in result["source_binding"].values():
        payload = (ROOT / binding["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == binding["file_sha256"]

    capture = result["frozen_capture"]
    assert capture["transport_complete"] is True
    assert capture["ticker_counts"]["total"] == 3368
    assert capture["block_trade_counts"]["total"] == 0
    assert capture["analyzable_block_trade_count"] == 0
    economics = result["economic_adjudication"]
    assert economics["material_follow_up_gate_passed"] is False
    assert economics["accepted_edge"] is False
    assert economics["public_forward_profit_floor_quote"] == "0"

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    _canonical_hash(registry)
    rank_five = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 5
    )
    assert any(
        binding["result_sha256"] == result_hash
        for binding in rank_five["canonical_artifacts"]
    )
