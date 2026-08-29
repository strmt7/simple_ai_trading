from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-existing-stock-transfer-reward-overlay-candidate-v1-2026-08-27.json"
)
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
EXPECTED_RESULT_SHA256 = (
    "3ecb4f39848719f788b6853bd90120d1809379b8d81b5419da4b1bbc957fec3d"
)
EXPECTED_REGISTRY_SHA256 = (
    "4b3828b49387edf1e26e8ff107221139f1d133c65ab85a8664f0ac08de84e5ad"
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


def test_candidate_reconstructs_and_has_zero_action_authority() -> None:
    artifact = _load(ARTIFACT_PATH)

    assert artifact["result_sha256"] == EXPECTED_RESULT_SHA256
    assert _embedded_hash(artifact) == EXPECTED_RESULT_SHA256
    assert artifact["authority"] == {
        "account_requests": 0,
        "credentials_used": False,
        "external_broker_actions": 0,
        "funded_actions": 0,
        "stock_transfer_requests": 0,
        "public_source_requests": 1,
    }
    adjudication = artifact["adjudication"]
    assert adjudication["accepted_edge"] is False
    assert adjudication["stable_edge"] is False
    assert adjudication["profitability_claim"] is False
    assert adjudication["transfer_authority"] is False
    assert artifact["economics"]["fee_reimbursement_profit_credit_USDC"] == "0"


def test_fixed_bonus_bips_and_hurdle_sensitivities_reconstruct() -> None:
    artifact = _load(ARTIFACT_PATH)
    tiers = artifact["economics"]["reward_tiers"]
    hurdle = Decimal(
        artifact["economics"]["fixed_21_day_annual_hurdle_sensitivities"][1][
            "hurdle_bips_of_transfer_value"
        ]
    )

    assert [row["minimum_first_transfer_value_USD"] for row in tiers] == [
        "2000",
        "10000",
        "30000",
        "150000",
        "400000",
        "1000000",
    ]
    assert [row["bonus_USDC"] for row in tiers] == [
        "50",
        "150",
        "200",
        "600",
        "1000",
        "2000",
    ]
    for row in tiers:
        threshold = Decimal(row["minimum_first_transfer_value_USD"])
        bonus = Decimal(row["bonus_USDC"])
        bonus_bips = bonus / threshold * Decimal(10000)
        assert Decimal(row["bonus_bips_at_threshold"]) == bonus_bips
        assert Decimal(
            row["net_bonus_bips_after_10pct_annual_21_day_hurdle"]
        ) == bonus_bips - hurdle
    assert [
        Decimal(row["net_bonus_bips_after_10pct_annual_21_day_hurdle"]) > 0
        for row in tiers
    ] == [True, True, True, False, False, False]


def test_first_come_pool_and_region_gates_keep_forward_floor_zero() -> None:
    artifact = _load(ARTIFACT_PATH)

    assert artifact["current_program"]["reward_pool_USDC"] == "300000"
    assert "first_come_first_served" in artifact["current_program"][
        "reward_pool_allocation"
    ]
    assert artifact["current_program"][
        "minimum_general_transfer_duration_business_days"
    ] == 14
    assert artifact["current_program"][
        "post_credit_no_transfer_out_or_sell_and_withdraw_days"
    ] == 21
    assert "forward_reward_floor_is_zero" in artifact["adjudication"]["status"]
    assert any(
        "not United States United Kingdom EEA" in row
        for row in artifact["decision_gates"]["account_and_region"]
    )


def test_registry_tracks_high_margin_candidate_without_accepting_it() -> None:
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_SHA256
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_SHA256
    assert registry["accepted_edge_count"] == 20
    candidate = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "bstock_reference_conversion_and_delta_neutral_perpetual_funding"
    )
    artifacts = {
        row["path"]: row["result_sha256"]
        for row in candidate["canonical_artifacts"]
    }
    assert artifacts[ARTIFACT_PATH.relative_to(ROOT).as_posix()] == (
        EXPECTED_RESULT_SHA256
    )
    assert candidate["market_direction_forecast_required"] is False
    assert "not_accepted_stable_or_deployment_ready" in candidate["current_status"]
    assert "explicit_account_read_only_authority" in candidate["next_action"]
