from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-bnb-just-in-time-fee-buffer-gate-v1-2026-08-26.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "b97eed6a93070d5e29b26d1a47757c9be49e0296332c8019a64388ba936c3b6b"
EXPECTED_REGISTRY_HASH = (
    "83dcc86f905b19679198a3dfe7b11d50b1377f7646ad287e647e4dc6d455e3aa"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_bytes())


def _embedded_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_artifact_is_hash_bound_scoped_and_non_authorizing() -> None:
    artifact = _load(ARTIFACT)
    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _embedded_hash(artifact) == EXPECTED_HASH
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["authority"]["orders_placed"] is False
    assert artifact["edge_identity"]["market_direction_forecast_required"] is False
    assert artifact["edge_identity"]["not_a_standalone_trading_strategy"] is True
    assert artifact["verdict"]["accepted_edge"] is True
    assert artifact["verdict"]["deployment_ready"] is False
    assert artifact["sources"]["official_spot_commission_contract"]["facts"][
        "published_example_discount_values"
    ] == ["0.25", "0.75"]


def test_full_and_partial_consumption_economics_reconstruct() -> None:
    artifact = _load(ARTIFACT)
    economics = artifact["economics"]
    discount = Decimal(economics["discount_fraction"])
    buffer = Decimal(economics["minimum_convert_buffer_usdt"])
    with localcontext() as context:
        context.prec = 50
        for row in economics["full_consumption_sensitivity"]:
            cost = Decimal(row["acquisition_cost_bps"]) / Decimal(10_000)
            expected_decline = Decimal(1) - Decimal(1) / (
                (Decimal(1) - cost) * (Decimal(1) / (Decimal(1) - discount))
            )
            assert expected_decline == Decimal(
                row["break_even_bnb_price_decline_fraction"]
            )
            expected_saving = buffer * (
                (Decimal(1) - cost) / (Decimal(1) - discount) - Decimal(1)
            )
            assert expected_saving == Decimal(
                row["minimum_buffer_net_saving_at_zero_price_change_usdt"]
            )

        for row in economics[
            "partial_consumption_diagnostic_not_minimum_buffer_execution_authority"
        ]:
            consumed = Decimal(row["consumed_fraction"])
            cost = Decimal("0.01")
            multiplier = (Decimal(1) - cost) * (
                consumed / (Decimal(1) - discount)
                + (Decimal(1) - consumed) * (Decimal(1) - cost)
            )
            expected_decline = Decimal(1) - Decimal(1) / multiplier
            assert expected_decline == Decimal(
                row["break_even_bnb_price_decline_fraction"]
            )


def test_public_minimums_and_registry_promotion_are_exact() -> None:
    artifact = _load(ARTIFACT)
    economics = artifact["economics"]
    assert Decimal(economics["minimum_convert_buffer_usdt"]) == Decimal("0.01")
    assert Decimal(
        artifact["execution_gate"]["minimum_research_policy"][
            "minimum_expected_consumption_fraction"
        ]
    ) == Decimal(1)
    assert artifact["sources"]["public_reverse_convert_pair_snapshot"]["response"][
        0
    ]["fromAssetMinAmount"] == "0.000014"
    assert Decimal(economics["minimum_spot_fallback_ask_cost_usdt"]) == (
        Decimal(economics["minimum_spot_fallback_quantity_bnb"])
        * Decimal(artifact["sources"]["public_spot_snapshot"]["book"]["ask_price"])
    )
    registry = _load(REGISTRY)
    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 41))
    candidate = next(
        row
        for row in hypotheses
        if row["mechanism"] == "binance_spot_fee_minimization_overlays"
    )
    assert candidate["priority_rank"] == 5
    assert candidate["canonical_artifacts"][0]["result_sha256"] == EXPECTED_HASH
