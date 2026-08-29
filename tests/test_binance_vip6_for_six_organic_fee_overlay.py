from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "binance-vip6-for-six-organic-fee-overlay-candidate-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
ARTIFACT_HASH = "f638cb6f565c1ee18c9dc065c5f4fc6506442f00833193d23c287bdf9d8ec74d"
REGISTRY_HASH = "ebce99afa23c826f41acec8670dc8259274d62e64d71d255a3645c119f776c95"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _saving_per_million(
    fees: dict[str, dict[str, str]], current_tier: str, role: str
) -> Decimal:
    current_percent = Decimal(fees[current_tier][role])
    vip6_percent = Decimal(fees["VIP_6"][role])
    return (current_percent - vip6_percent) * Decimal("10000")


def test_vip6_candidate_reconstructs_fee_savings_and_fails_closed() -> None:
    artifact = _load(ARTIFACT)
    fees = artifact["current_public_fee_schedule"]
    savings = artifact["fee_saving_sensitivity_USD_per_1000000_USD_organic_notional"]

    assert artifact["result_sha256"] == ARTIFACT_HASH
    assert _canonical_hash(artifact) == ARTIFACT_HASH
    assert artifact["authority"][
        "applications_emails_external_exchange_records_loans_bnb_borrowing_or_trades_accessed_or_changed"
    ] == 0

    cases = (
        ("spot_standard_percent", "spot_standard"),
        (
            "spot_with_BNB_25_percent_discount_percent",
            "spot_with_BNB_25_percent_discount",
        ),
        ("USD_M_USDT_standard_percent", "USD_M_USDT_standard"),
        (
            "USD_M_USDT_with_BNB_10_percent_discount_percent",
            "USD_M_USDT_with_BNB_10_percent_discount",
        ),
    )
    for fee_key, saving_key in cases:
        for tier in range(1, 6):
            source_tier = f"VIP_{tier}"
            route = f"VIP_{tier}_to_VIP_6"
            for role in ("maker", "taker"):
                assert Decimal(savings[saving_key][route][role]) == (
                    _saving_per_million(fees[fee_key], source_tier, role)
                )

    assert artifact["adjudication"]["public_forward_net_saving_floor_USD"] == "0"
    assert artifact["adjudication"]["accepted_edge"] is False
    assert "do not apply" in artifact["prequalification_gate"]["rejection_rule"]

    registry = _load(REGISTRY)
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "binance_spot_fee_minimization_overlays"
    )
    assert {
        "path": "docs/model-research/action-value/binance-vip6-for-six-organic-fee-overlay-candidate-v1-2026-08-27.json",
        "result_sha256": ARTIFACT_HASH,
    } in hypothesis["canonical_artifacts"]
    assert "VIP_6_for_Six" in hypothesis["current_status"]
