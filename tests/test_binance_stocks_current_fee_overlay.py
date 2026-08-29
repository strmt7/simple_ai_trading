from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "binance-stocks-current-fee-overlay-edge-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
ARTIFACT_HASH = "d4f02be559d9267abbea28ccefb48f4886f375b359ce7274b90b6585b828160a"
REGISTRY_HASH = "7b03ee420b7874180732f28fc1c59903adbd82e147be268ea54e894f460bbda1"


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
    assert registry["accepted_edge_count"] == 19
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
