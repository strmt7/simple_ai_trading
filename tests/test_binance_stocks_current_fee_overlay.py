from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "binance-stocks-current-fee-overlay-edge-v1-2026-08-27.json"
)
TRIGGER_TRIAGE = ROOT / "docs/model-research/action-value" / (
    "binance-aug28-public-structural-trigger-triage-v1-2026-08-29.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
ARTIFACT_HASH = "d4f02be559d9267abbea28ccefb48f4886f375b359ce7274b90b6585b828160a"
TRIGGER_TRIAGE_HASH = "bca11d612042f9a859f53b71e425cd320cca5d4a5d7695cd1f0a0de539b0eea1"
REGISTRY_HASH = "4b3828b49387edf1e26e8ff107221139f1d133c65ab85a8664f0ac08de84e5ad"


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


def test_stocks_fee_overlay_reconstructs_savings_and_fail_closed_boundaries() -> None:
    artifact = _load(ARTIFACT)
    terms = artifact["current_public_terms"]
    savings = artifact["gross_incremental_saving"]

    assert artifact["result_sha256"] == ARTIFACT_HASH
    assert _canonical_hash(artifact) == ARTIFACT_HASH
    assert artifact["authority"]["orders_transfers_conversions_or_account_changes"] == 0
    assert artifact["adjudication"]["accepted_edge"] is True
    assert artifact["adjudication"]["standalone_profitability_claim"] is False
    assert terms["promotion_end_utc"] == "2026-08-31T00:00:00Z"

    above = terms["strictly_above_340_USD"]
    expected_bps = (
        Decimal(above["normal_trading_spread_percent"])
        - Decimal(above["promotional_trading_spread_percent"])
    ) * Decimal("100")
    assert Decimal(savings["strictly_above_340_USD"]["basis_points_of_order_notional"]) == expected_bps
    assert Decimal(savings["strictly_above_340_USD"]["USD_per_1000000_USD_organic_notional"]) == expected_bps * Decimal("100")

    below = terms["strictly_below_340_USD"]
    expected_flat_saving = Decimal(below["normal_trading_spread_USD_per_order"]) - Decimal(
        below["promotional_trading_spread_USD_per_order"]
    )
    assert Decimal(savings["strictly_below_340_USD"]["USD_per_order"]) == expected_flat_saving
    assert terms["exactly_340_USD"]["status"] == "unresolved_due_to_overlapping_inclusive_page_labels"
    assert Decimal(savings["exactly_340_USD"]["precredited_saving"]) == 0

    registry = _load(REGISTRY)
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 20
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "binance_spot_fee_minimization_overlays"
    )
    assert {
        "path": "docs/model-research/action-value/binance-stocks-current-fee-overlay-edge-v1-2026-08-27.json",
        "result_sha256": ARTIFACT_HASH,
    } in hypothesis["canonical_artifacts"]
    assert "Binance_Stocks" in hypothesis["current_status"]


def test_bstocks_zero_maker_extension_is_scoped_and_does_not_reopen_terminal_families() -> None:
    triage = _load(TRIGGER_TRIAGE)

    assert triage["result_sha256"] == TRIGGER_TRIAGE_HASH
    assert _canonical_hash(triage) == TRIGGER_TRIAGE_HASH
    assert triage["authority"]["orders_transfers_conversions_subscriptions_or_account_changes"] == 0
    bstocks = triage["adjudications"]["bstocks_zero_maker_fee_extension"]
    assert bstocks["accepted_edge"] is True
    assert bstocks["standalone_profitability_claim"] is False
    assert bstocks["promotion_end_utc"] == "2026-09-30T23:59:00Z"
    assert bstocks["credited_order_role"] == "maker_only"
    assert "Grid" in " ".join(bstocks["prohibited_shortcuts"])

    mark_change = triage["adjudications"]["tradfi_mark_price_basis_window_change"]
    assert mark_change["accepted_edge"] is False
    assert mark_change["decision"] == "no_structural_edge_retry"
    assert mark_change["effective_at_utc"] == "2026-08-31T08:15:00Z"
    smart = triage["adjudications"]["smart_arbitrage_refresh"]
    assert smart["accepted_edge"] is False
    assert smart["decision"] == "terminal_family_not_reopened"

    registry = _load(REGISTRY)
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "binance_spot_fee_minimization_overlays"
    )
    assert {
        "path": "docs/model-research/action-value/binance-aug28-public-structural-trigger-triage-v1-2026-08-29.json",
        "result_sha256": TRIGGER_TRIAGE_HASH,
    } in hypothesis["canonical_artifacts"]
    assert "2026_09_30T23_59_00Z" in hypothesis["current_status"]
