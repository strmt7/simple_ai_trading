from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "binance-first-usd-deposit-spcxb-reward-hedge-candidate-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
ARTIFACT_HASH = "e0b6ed9311d2a022abee417a677b952e83cf918fc6b396804f5cba39fd83d4ed"
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


def test_first_usd_deposit_reward_candidate_reconstructs_and_fails_closed() -> None:
    artifact = _load(ARTIFACT)
    economics = artifact["economics"]
    task = economics["task_route_sensitivity"]
    reward = economics["reward_sensitivity"]
    hedge = economics["entitlement_gated_hedge_sensitivity"]

    assert artifact["result_sha256"] == ARTIFACT_HASH
    assert _canonical_hash(artifact) == ARTIFACT_HASH
    assert artifact["authority"][
        "registrations_deposits_trades_hedges_claims_sales_or_withdrawals"
    ] == 0
    assert Decimal(task["USDT_quantity"]) * Decimal(
        task["displayed_ask_USD_per_USDT"]
    ) == Decimal(task["displayed_quote_notional_USD"])
    assert Decimal(task["displayed_quote_notional_USD"]) > Decimal("200")
    theoretical_quantity = Decimal("15") / Decimal("135.2389566")
    assert Decimal(reward["theoretical_SPCXB_quantity_before_voucher_rounding"]) == (
        theoretical_quantity
    )
    assert Decimal(reward["current_gross_liquidation_value_USDT"]) == (
        theoretical_quantity * Decimal("140.77000000")
    )
    assert Decimal(hedge["rounded_down_short_SPCX_quantity"]) == Decimal("0.11")
    assert "do not open any hedge before" in hedge["critical_rule"]
    assert artifact["adjudication"]["public_forward_net_profit_floor"] == "0"
    assert artifact["adjudication"]["accepted_edge"] is False

    registry = _load(REGISTRY)
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "same_account_stable_value_yield_allocation"
    )
    assert {
        "path": "docs/model-research/action-value/binance-first-usd-deposit-spcxb-reward-hedge-candidate-v1-2026-08-27.json",
        "result_sha256": ARTIFACT_HASH,
    } in hypothesis["canonical_artifacts"]
    assert hypothesis["market_direction_forecast_required"] is False
    assert "none_is_deployment_ready" in hypothesis["current_status"]
