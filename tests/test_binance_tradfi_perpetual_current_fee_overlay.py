from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "binance-tradfi-perpetual-current-fee-overlay-edge-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
ARTIFACT_HASH = "705cb3da615c1873623e7f5be31f0d8cf672c3db9635a5ba971407cf6e715b6c"
REGISTRY_HASH = "d9698017a21be49e8f0b5c0021d4c1eeb1dff0a6482bab9badc0a8c76be5df4b"


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


def test_tradfi_fee_overlay_reconstructs_current_savings_and_scope() -> None:
    artifact = _load(ARTIFACT)
    tradfi = artifact["displayed_fee_percent_by_level"]
    standard = artifact["same_snapshot_standard_USD_M_USDT_fee_percent_by_level"]
    savings = artifact[
        "displayed_gross_saving_USD_per_1000000_USD_organic_notional_against_standard_USD_M_USDT_rates"
    ]

    assert artifact["result_sha256"] == ARTIFACT_HASH
    assert _canonical_hash(artifact) == ARTIFACT_HASH
    assert artifact["authority"]["orders_transfers_borrowing_or_account_changes"] == 0
    assert artifact["adjudication"]["accepted_edge"] is True
    assert artifact["adjudication"]["standalone_profitability_claim"] is False
    assert artifact["current_public_terms"]["promotion_end_exposed_on_current_fee_table"] is False

    for level in standard:
        assert Decimal(tradfi[level]["maker"]) == 0
        expected_maker = Decimal(standard[level]["maker"]) * Decimal("10000")
        expected_taker = (
            Decimal(standard[level]["taker"]) - Decimal(tradfi[level]["taker"])
        ) * Decimal("10000")
        expected_bnb_maker = (
            Decimal(standard[level]["maker_with_BNB"]) * Decimal("10000")
        )
        expected_bnb_taker = (
            Decimal(standard[level]["taker_with_BNB"])
            - Decimal(tradfi[level]["taker_with_BNB"])
        ) * Decimal("10000")
        assert Decimal(savings["without_BNB"][level]["maker"]) == expected_maker
        assert Decimal(savings["without_BNB"][level]["taker"]) == expected_taker
        assert Decimal(savings["with_BNB"][level]["maker"]) == expected_bnb_maker
        assert Decimal(savings["with_BNB"][level]["taker"]) == expected_bnb_taker

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
        "path": "docs/model-research/action-value/binance-tradfi-perpetual-current-fee-overlay-edge-v1-2026-08-27.json",
        "result_sha256": ARTIFACT_HASH,
    } in hypothesis["canonical_artifacts"]
    assert "TradFi_perpetual_zero_maker" in hypothesis["current_status"]
