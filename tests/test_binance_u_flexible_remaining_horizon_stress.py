from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-u-flexible-remaining-horizon-stress-v1-2026-08-30.json"
)
FRONTIER = ROOT / (
    "docs/model-research/action-value/"
    "accepted-market-independent-yield-frontier-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    return hashlib.sha256(_canonical(body)).hexdigest()


def test_remaining_horizon_artifact_and_sources_reconstruct() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert _self_hash(artifact) == artifact["result_sha256"]

    for source in artifact["sources"]:
        payload = (ROOT / source["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == source["file_sha256"]
        assert json.loads(payload)["result_sha256"] == source["result_sha256"]


def test_fixed_only_day_boundary_reconstructs_without_refitting() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    economics = artifact["economic_contract"]
    with localcontext() as context:
        context.prec = 50
        stress = Decimal(economics["basis_stress_bips"]) + Decimal(
            economics["current_displayed_round_trip_spread_bips"]
        )
        principal = Decimal(economics["principal_cap_u"])

        for row in artifact["day_boundary"]["daily_cases"]:
            u_days = Decimal(row["remaining_u_fixed_bonus_days"])
            usdt_bonus_days = Decimal(row["remaining_usdt_fixed_bonus_days"])
            incremental = (
                (Decimal("0.08") - Decimal("0.03"))
                * u_days
                / Decimal(365)
                * Decimal(10000)
                - (Decimal(500) / principal)
                * Decimal("0.04")
                * usdt_bonus_days
                / Decimal(365)
                * Decimal(10000)
            )
            assert incremental == Decimal(
                row["incremental_fixed_reward_before_stress_bips"]
            )
            assert incremental - stress == Decimal(
                row["margin_after_historical_basis_and_spread_stress_bips"]
            )

    rows = artifact["day_boundary"]["daily_cases"]
    assert Decimal(rows[2]["margin_after_historical_basis_and_spread_stress_bips"]) > 0
    assert Decimal(rows[3]["margin_after_historical_basis_and_spread_stress_bips"]) < 0
    assert artifact["day_boundary"][
        "first_subscription_date_negative_before_every_unproved_other_cost"
    ] == "2026-09-02"


def test_positive_sensitivity_is_not_promoted_and_family_binds_result() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    adjudication = artifact["adjudication"]
    assert adjudication["profitability_claim"] is False
    assert adjudication["stable_acquisition_edge_proved"] is False
    assert adjudication["public_after_all_cost_floor_bips"] == "0"
    assert artifact["authority"]["new_network_requests"] == 0
    assert artifact["authority"][
        "orders_conversions_subscriptions_or_redemptions"
    ] == 0

    expected = {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    }
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry) == registry["result_sha256"]
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 8
    )
    assert family["mechanism"] == "same_account_stable_value_yield_allocation"
    assert expected in family["canonical_artifacts"]

    frontier = json.loads(FRONTIER.read_text(encoding="utf-8"))
    assert _self_hash(frontier) == frontier["result_sha256"]
    supplementary = frontier["supplementary_current_stress_artifacts"]
    row = next(item for item in supplementary if item["edge_id"] == "binance_u_flexible_existing_holding_yield")
    assert row["path"] == expected["path"]
    assert row["result_sha256"] == expected["result_sha256"]
    decision = frontier["portfolio_decision"][
        "u_flexible_current_remaining_horizon_stress"
    ]
    assert decision["stable_acquisition_profit_proved"] is False
    assert decision["public_after_all_cost_floor_bips"] == "0"
