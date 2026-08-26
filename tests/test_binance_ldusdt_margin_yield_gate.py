from __future__ import annotations

import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = (
    ROOT
    / "docs/model-research/action-value/binance-ldusdt-margin-yield-gate-v1-2026-08-26.json"
)
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "39b66e63c6131d13ff6e6df7f19521745b0de8a715eb3969bf333c485a9ab5f0"
EXPECTED_REGISTRY_HASH = "b5a65a83cdf1abaadf7d2a0215720b80266e06c50c23992f5a3ed3d6c638a1b4"


def _load(path: Path = PATH) -> dict[str, object]:
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


def test_gate_is_hash_bound_and_grants_no_authority() -> None:
    artifact = _load()
    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _embedded_hash(artifact) == EXPECTED_HASH
    assert artifact["created_at_utc"] == "2026-08-26T10:00:01Z"
    assert artifact["authority"]["accepted_edge"] is False
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["authority"]["orders_placed"] is False
    assert artifact["incremental_edge_identity"]["not_a_standalone_trading_strategy"]


def test_normalized_history_reconstructs_the_gross_increment() -> None:
    artifact = _load()
    summary = artifact["economic_summary"]
    first = Decimal(summary["first_normalized_close_ldusdt_per_usdt"])
    last = Decimal(summary["last_normalized_close_ldusdt_per_usdt"])
    with localcontext() as context:
        context.prec = 50
        reconstructed_return = last / first - 1
    assert abs(
        reconstructed_return - Decimal(summary["cumulative_return_fraction"])
    ) < Decimal("1e-28")
    assert summary["aligned_daily_close_count"] == 505
    assert summary["elapsed_days"] == 504
    assert (
        summary["positive_daily_change_count"]
        + summary["zero_daily_change_count"]
        + summary["negative_daily_change_count"]
        == 504
    )
    assert [row["days"] for row in artifact["normalized_windows"]] == [7, 30, 90, 365]
    snapshot = artifact["public_current_snapshot"]
    with localcontext() as context:
        context.prec = 40
        expected_normalized = Decimal(snapshot["ldusdtusd"]["index"]) / Decimal(
            snapshot["usdtusd"]["index"]
        )
    assert expected_normalized == Decimal(snapshot["normalized_current_index_ldusdt_per_usdt"])


def test_gate_preserves_conversion_and_incremental_cost_unknowns() -> None:
    artifact = _load()
    probe = artifact["sources"]["public_futures_convert_probe"]
    assert probe["response"] == []
    limitations = " ".join(artifact["unresolved_gates"])
    assert "conversion" in limitations
    assert "alternative" in limitations
    assert "No existing or proposed futures strategy becomes profitable" in limitations


def test_registry_prioritizes_the_increment_without_accepting_it() -> None:
    registry = _load(REGISTRY_PATH)
    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_HASH
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 15))
    lead = next(
        row
        for row in hypotheses
        if row["mechanism"] == "ldusdt_reward_bearing_futures_margin_increment"
    )
    assert lead["priority_rank"] == 2
    assert lead["market_direction_forecast_required"] is False
    assert lead["canonical_artifacts"][0]["result_sha256"] == EXPECTED_HASH
    assert registry["accepted_edge_count"] == 1
