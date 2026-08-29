from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-bstock-spot-lp-all-symbol-rebate-overlay-candidate-v1-2026-08-27.json"
)
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
EXPECTED_RESULT_SHA256 = (
    "d279f8ab88875c812e6691fa500fdfde741f2e2fbca19ee240b4c0d4a579d607"
)
EXPECTED_REGISTRY_SHA256 = (
    "fc0bddf222a1908db6c12df338dc26963f36514b01e37b5b31fc567760f19aca"
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


def test_public_candidate_hash_authority_and_effective_window() -> None:
    artifact = _load(ARTIFACT_PATH)

    assert artifact["result_sha256"] == EXPECTED_RESULT_SHA256
    assert _embedded_hash(artifact) == EXPECTED_RESULT_SHA256
    assert artifact["authority"] == {
        "account_requests": 0,
        "credentials_used": False,
        "funded_actions": 0,
        "orders_or_applications": 0,
        "public_source_requests": 1,
    }
    assert artifact["current_promotion"][
        "first_performance_review_end_utc"
    ] < artifact["current_promotion"]["first_promotion_tier_effective_start_utc"]
    adjudication = artifact["adjudication"]
    assert adjudication["accepted_edge"] is False
    assert adjudication["stable_edge"] is False
    assert adjudication["profitability_claim"] is False
    assert adjudication["deployment_ready"] is False
    assert adjudication["trading_authority"] is False


def test_bstock_thresholds_and_all_symbol_rebate_ceiling_are_exact() -> None:
    artifact = _load(ARTIFACT_PATH)
    tiers = artifact["program_mechanics"]["tiers"]

    assert [row["tier"] for row in tiers] == [1, 2, 3, 4]
    assert [row["bStocks_weekly_maker_volume_percentage"] for row in tiers] == [
        "0.05",
        "0.10",
        "0.30",
        "0.60",
    ]
    assert [row["maker_rebate_bips"] for row in tiers] == [
        "0",
        "0.4",
        "0.6",
        "0.8",
    ]
    assert artifact["candidate_scope"][
        "maximum_public_rebate_bips_per_maker_notional"
    ] == "0.8"
    assert "all_symbols" in artifact["program_mechanics"]["all_symbol_upgrade"]
    assert artifact["manual_trial"]["economic_rebate_floor_bips"] == "0"


def test_registry_records_overlay_without_increasing_accepted_count() -> None:
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_SHA256
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_SHA256
    assert registry["accepted_edge_count"] == 21
    candidate = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "spot_market_maker_rebates"
    )
    artifacts = {
        row["path"]: row["result_sha256"]
        for row in candidate["canonical_artifacts"]
    }
    assert artifacts[ARTIFACT_PATH.relative_to(ROOT).as_posix()] == (
        EXPECTED_RESULT_SHA256
    )
    assert "first_effective_week_starts_2026_09_01" in candidate["current_status"]
    assert "no_orders_application_or_volume_generation" in candidate["next_action"]
