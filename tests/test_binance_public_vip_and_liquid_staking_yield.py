from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
VIP_ARTIFACT = (
    ACTION_VALUE
    / "binance-vip-earn-public-btc-eth-sol-comparator-terminal-v1-2026-08-27.json"
)
STAKING_ARTIFACT = (
    ACTION_VALUE
    / "binance-existing-idle-eth-sol-liquid-staking-yield-candidate-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
VIP_HASH = "cd41cad8e0053b9d41ddda64fd4ad8a86a163307ddcc9fabc805c56b9c5028c9"
STAKING_HASH = "b7fc84d0be3968d31afeb801b7a40ee0d382724b11281c28733a8145d12ee035"
REGISTRY_HASH = "5dfe720ff8cb69f5489ef6deb47fffe2d1ae4d036f1c14a13fbb34daf961f14a"


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


def test_public_vip_rows_have_no_positive_in_scope_maximum_uplift() -> None:
    artifact = _load(VIP_ARTIFACT)

    assert artifact["result_sha256"] == VIP_HASH
    assert _canonical_hash(artifact) == VIP_HASH
    assert artifact["authority"]["authenticated_requests"] == 0
    assert artifact["authority"]["subscriptions_redemptions_borrows_transfers_orders_or_quotes"] == 0
    rows = artifact["visible_public_rows"]
    assert {
        asset: Decimal(row["best_VIP_minus_best_visible_non_VIP_bps"])
        for asset, row in rows.items()
    } == {"BTC": Decimal("0"), "ETH": Decimal("-30"), "SOL": Decimal("-15")}
    assert all(
        Decimal(row["best_VIP_minus_best_visible_non_VIP_bps"]) <= 0
        for row in rows.values()
    )
    assert artifact["economic_screen"]["strictly_positive_displayed_maximum_uplift_survivors"] == 0
    assert artifact["economic_screen"]["public_forward_incremental_profit_floor"] == "0"


def test_pre_launch_pdf_is_rejected_as_current_vip_evidence() -> None:
    artifact = _load(VIP_ARTIFACT)
    gate = artifact["source_age_gate"]
    launch = date.fromisoformat(gate["VIP_Earn_launch_date_utc"])
    last_modified = date.fromisoformat(gate["legacy_pdf_last_modified_utc"][:10])

    assert (launch - last_modified).days == gate["days_before_VIP_Earn_launch"] == 441
    assert gate["admissibility"] == "rejected_as_current_VIP_Earn_rate_evidence"


def test_liquid_staking_cost_budgets_reconstruct_from_public_rate_uplift() -> None:
    artifact = _load(STAKING_ARTIFACT)

    assert artifact["result_sha256"] == STAKING_HASH
    assert _canonical_hash(artifact) == STAKING_HASH
    assert artifact["authority"]["authenticated_requests"] == 0
    assert artifact["authority"]["subscriptions_redemptions_conversions_transfers_orders_or_quotes"] == 0
    snapshot = artifact["public_current_snapshot"]
    budgets = artifact["cost_budget_sensitivity"]
    for asset in ("ETH", "SOL"):
        rate = Decimal(snapshot[asset]["displayed_apr_percent"])
        comparator = Decimal(snapshot[asset]["soft_staking_comparator_apr_percent"])
        uplift_bps = (rate - comparator) * Decimal("100")
        assert uplift_bps == Decimal(snapshot[asset]["gross_incremental_apr_bps"])
        for days in (30, 60, 90, 180, 365):
            expected = (uplift_bps * Decimal(days) / Decimal(365)).quantize(
                Decimal("0.00000001")
            )
            assert Decimal(budgets[asset][f"{days}_days"]) == expected
        for cost_bps in (20, 50, 100):
            expected_days = (
                Decimal(365) * Decimal(cost_bps) / uplift_bps
            ).quantize(Decimal("0.00000001"))
            assert Decimal(budgets[asset][f"days_to_recover_{cost_bps}_bps"]) == expected_days
    assert artifact["economic_gate"]["public_forward_net_profit_floor"] == "0"
    assert artifact["adjudication"]["accepted_edge"] is False


def test_registry_routes_public_vip_terminal_and_liquid_staking_candidate() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    hypotheses = registry["prioritized_hypotheses"]
    idle_yield = next(
        row for row in hypotheses if row["mechanism"] == "binance_idle_spot_native_token_yield"
    )
    vip = next(
        row
        for row in hypotheses
        if row["mechanism"]
        == "binance_VIP_Earn_existing_idle_inventory_incremental_yield_overlay"
    )
    assert {
        "path": "docs/model-research/action-value/binance-existing-idle-eth-sol-liquid-staking-yield-candidate-v1-2026-08-27.json",
        "result_sha256": STAKING_HASH,
    } in idle_yield["canonical_artifacts"]
    assert {
        "path": "docs/model-research/action-value/binance-vip-earn-public-btc-eth-sol-comparator-terminal-v1-2026-08-27.json",
        "result_sha256": VIP_HASH,
    } in vip["canonical_artifacts"]
