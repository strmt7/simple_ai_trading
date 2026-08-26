from __future__ import annotations

from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "screen_binance_bfusd_redemption_parity.py"
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-bfusd-spot-redemption-parity-v1-2026-08-26.json"
)
EXPECTED_RESULT_HASH = "566be5e515ac14d38377b6a6b42101cc9b8a65585142053791b759efbd77f6bb"
EXPECTED_TOOL_HASH = "8667e3b43138bdcc89be023f3bdc26268157765ba6fde6b29798b46ebb5322ac"
SPEC = importlib.util.spec_from_file_location("bfusd_redemption_parity", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def test_bfusd_artifact_is_source_bound_and_rejects_current_book() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_bytes())
    body = dict(artifact)
    assert body.pop("result_sha256") == EXPECTED_RESULT_HASH
    assert hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest() == (
        EXPECTED_RESULT_HASH
    )
    assert artifact["implementation"]["sha256"] == EXPECTED_TOOL_HASH
    assert hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == EXPECTED_TOOL_HASH
    assert artifact["current_book"]["best_bid"] == "1.00000000"
    assert artifact["current_book"]["best_ask"] == "1.00010000"
    assert artifact["verdict"]["current_direct_positive_path"] is False
    assert artifact["verdict"]["holding_yield_edge_accepted"] is False
    assert artifact["authority"] == {
        "credentials_used": False,
        "funds_used": False,
        "orders_placed": False,
        "trading_authority": False,
    }


def test_parity_math_detects_both_structural_directions() -> None:
    discount = MODULE._screen_quantity(
        quantity=Decimal("100"),
        asks=((Decimal("0.995"), Decimal("100")),),
        bids=((Decimal("0.994"), Decimal("100")),),
        spot_fee_bips=Decimal("10"),
        subscription_fee_bips=Decimal("10"),
        redemption_fee_bips=Decimal("10"),
    )
    assert discount["buy_spot_then_redeem_net_usdt"] == "0.300500"
    assert discount["buy_spot_then_redeem_positive"] is True
    premium = MODULE._screen_quantity(
        quantity=Decimal("100"),
        asks=((Decimal("1.006"), Decimal("100")),),
        bids=((Decimal("1.005"), Decimal("100")),),
        spot_fee_bips=Decimal("10"),
        subscription_fee_bips=Decimal("10"),
        redemption_fee_bips=Decimal("10"),
    )
    assert premium["subscribe_then_sell_spot_net_usdt"] == "0.299500"
    assert premium["subscribe_then_sell_spot_positive"] is True
