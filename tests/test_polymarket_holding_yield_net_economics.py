from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ARTIFACT_PATH = Path(
    "docs/model-research/polymarket/"
    "complete-set-holding-yield-net-economics-v5-2026-08-26.json"
)
TOOL_PATH = Path("tools/adjudicate_polymarket_holding_yield_net_economics.py")
EXPECTED_RESULT_HASH = (
    "dff80903a20d9bfc8e3402eea01dad8a8f5ee39b0427690514cc30b9fe9dcb85"
)
EXPECTED_TOOL_HASH = "f13389b0a5de846f0cfbd06695bf4e3a0430701d661ab8f44020d0fd5396afa8"
EXPECTED_SOURCE_HASHES = {
    "docs/model-research/polymarket/complete-set-holding-yield-reconciliation-v3-2026-08-26.json": "48e31f3d6021d28946fa1f143f65ff0f6baf9a222424f41e76c2d89875796abe",
    "docs/model-research/polymarket/complete-set-holding-yield-cross-asset-v4-2026-08-26.json": "eda29a314218e1724e39984e2712a4351d9e697503d4583d391c89a060ba53ea",
    "docs/model-research/polymarket/complete-set-holding-reward-readiness-v2.json": "2d3650d65f248294395fcac336c6650e0c6bc332cb490c6f0bac70bc11244e2c",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _load() -> dict[str, object]:
    value = json.loads(ARTIFACT_PATH.read_text())
    assert isinstance(value, dict)
    return value


def test_net_economics_artifact_and_implementation_are_source_bound() -> None:
    artifact = _load()
    body = dict(artifact)
    assert body.pop("result_sha256") == EXPECTED_RESULT_HASH
    assert hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest() == (
        EXPECTED_RESULT_HASH
    )
    assert artifact["implementation"] == {
        "path": TOOL_PATH.as_posix(),
        "sha256": EXPECTED_TOOL_HASH,
    }
    assert hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == EXPECTED_TOOL_HASH
    assert {
        row["path"]: row["result_sha256"] for row in artifact["source_continuity"]
    } == EXPECTED_SOURCE_HASHES


def test_cross_asset_direct_cost_net_reward_reconstructs() -> None:
    artifact = _load()
    expected = {
        "BTC": (Decimal("150"), Decimal("0.1816")),
        "ETH": (Decimal("440"), Decimal("0.5377")),
        "SOL": (Decimal("449"), Decimal("0.5488")),
    }
    assert len(artifact["cases"]) == 3
    for case in artifact["cases"]:
        principal, reward = expected[case["asset"]]
        assert Decimal(case["principal_pusd"]) == principal
        assert Decimal(case["observed_reward_pusd"]) == reward
        assert case["observed_days"] == "14"
        assert case["direct_relayer_split_merge_user_gas_pusd"] == "0"
        assert case["direct_protocol_split_merge_principal_loss_pusd"] == "0"
        assert Decimal(case["net_reward_after_direct_split_merge_cost_pusd"]) == reward

    portfolio = artifact["cross_asset_portfolio"]
    assert Decimal(portfolio["demonstrated_principal_pusd"]) == Decimal("1039")
    assert Decimal(portfolio["observed_reward_pusd"]) == Decimal("1.2681")
    assert portfolio["positive_daily_payout_count"] == 42
    assert portfolio["possible_daily_payout_count"] == 42
    assert Decimal(portfolio["principal_weighted_realized_annualized_rate"]) == Decimal(
        "0.03182019111783308125945277053"
    )


def test_three_percent_alternative_is_positive_but_friction_margin_is_thin() -> None:
    artifact = _load()
    portfolio = artifact["cross_asset_portfolio"]
    three_percent = next(
        row
        for row in portfolio["alternative_yield_sensitivities"]
        if row["alternative_annual_rate"] == "0.03"
    )
    assert Decimal(
        three_percent["net_reward_after_alternative_and_direct_cost_pusd"]
    ) == Decimal("0.072538356164383561643835616")
    assert Decimal(three_percent["annualized_net_spread_bips"]) == Decimal(
        "18.20191117833081259452770530"
    )
    ten_bips = next(
        row
        for row in three_percent["external_friction_break_even"]
        if row["total_external_friction_bips"] == "10"
    )
    assert Decimal(ten_bips["break_even_holding_days"]) == Decimal(
        "200.5283931033180367495703727"
    )
    horizon = next(
        row
        for row in three_percent["maximum_tolerable_total_friction_by_horizon"]
        if row["horizon_days"] == "127"
    )
    assert Decimal(horizon["maximum_total_friction_bips"]) == Decimal(
        "6.333267725063049861657585132"
    )


def test_current_program_rate_is_not_an_after_alternative_yield_edge() -> None:
    artifact = _load()
    portfolio = artifact["cross_asset_portfolio"]
    program_rate = next(
        row
        for row in portfolio["alternative_yield_sensitivities"]
        if row["alternative_annual_rate"] == "0.0325"
    )
    assert Decimal(
        program_rate["net_reward_after_alternative_and_direct_cost_pusd"]
    ) == Decimal("-0.027091780821917808219178082")
    assert Decimal(program_rate["annualized_net_spread_bips"]) == Decimal(
        "-6.798088821669187405472294700"
    )
    assert program_rate["positive"] is False
    assert (
        artifact["adjudication"][
            "positive_after_three_point_two_five_percent_alternative_yield_in_any_case"
        ]
        is False
    )


def test_acceptance_is_narrowly_after_direct_cost_without_deployment_authority() -> (
    None
):
    artifact = _load()
    assert artifact["adjudication"] == {
        "demonstrated_capacity_is_not_a_capacity_ceiling": True,
        "external_alternative_yield_and_friction_fully_proven": False,
        "positive_after_direct_relayer_and_protocol_cost_in_every_case": True,
        "positive_after_three_percent_alternative_yield_in_every_case": True,
        "positive_after_three_point_two_five_percent_alternative_yield_in_any_case": False,
    }
    verdict = artifact["verdict"]
    assert verdict["accepted_edge"] is True
    assert verdict["after_direct_cost"] is True
    assert verdict["after_every_external_cost_and_best_alternative_yield"] is False
    assert verdict["deployment_ready"] is False
    assert verdict["trading_authority"] is False
