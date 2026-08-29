from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs" / "model-research" / "action-value"
ARTIFACT_PATH = ACTION_VALUE / "binance-bnb-stacked-reward-hedge-evidence-gate-v1.json"
FROZEN_HEDGE_PATH = (
    ACTION_VALUE / "binance-bnb-fee-discount-hedge-recovery-v1-2026-08-25.json"
)
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
EXPECTED_ARTIFACT_HASH = (
    "0bfc615af743f4ba352201ff2f06e2abf0f0c8fec56b548a0e19791faf25f8ed"
)
EXPECTED_HEDGE_HASH = "85d0be66391b53bef87dda33ea73acaf6995d0200e6423de7999d44a8fed3c8f"
EXPECTED_REGISTRY_HASH = (
    "da3ddaf82a2cb0929353460a7e09812b47f940e953a3f1da43b04f72a55c8488"
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


def test_gate_is_hash_bound_get_only_and_grants_no_authority() -> None:
    artifact = _load(ARTIFACT_PATH)

    assert artifact["result_sha256"] == EXPECTED_ARTIFACT_HASH
    assert _embedded_hash(artifact) == EXPECTED_ARTIFACT_HASH
    assert artifact["authority"] == {
        "accepted_edge": False,
        "credentials_used": False,
        "live_trading_authority": False,
        "orders_placed": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "signed_requests_made": 0,
    }
    prequalification = artifact["read_only_account_prequalification"]
    assert prequalification["endpoint_count"] == 7
    assert {row["method"] for row in prequalification["endpoints"]} == {"GET"}
    assert {row["security_type"] for row in prequalification["endpoints"]} == {
        "USER_DATA"
    }
    assert artifact["preflight"]["request_count"] == 0


def test_seven_day_airdrop_hurdle_reconstructs_from_frozen_funding() -> None:
    artifact = _load(ARTIFACT_PATH)
    hedge = _load(FROZEN_HEDGE_PATH)
    rows = hedge["merged_funding_history_payload"]

    rolling = [
        (sum(Decimal(row["fundingRate"]) for row in rows[start : start + 21]), start)
        for start in range(len(rows) - 20)
    ]
    worst_rate, worst_start = min(rolling)
    worst_rows = rows[worst_start : worst_start + 21]
    advertised_base_reward = Decimal("0.0035") * Decimal(7) / Decimal(365)
    shortfall_bips = (-worst_rate - advertised_base_reward) * Decimal(10_000)
    frozen = artifact["frozen_hedge_evidence_reuse"]

    assert hedge["result_sha256"] == EXPECTED_HEDGE_HASH
    assert len(rows) == frozen["funding_row_count"] == 1000
    assert worst_rate * Decimal(10_000) == Decimal(
        frozen["worst_rolling_21_payment_short_funding_bips"]
    )
    assert (
        worst_rows[0]["fundingTime"] == frozen["worst_rolling_21_payment_start_time_ms"]
    )
    assert (
        worst_rows[-1]["fundingTime"] == frozen["worst_rolling_21_payment_end_time_ms"]
    )
    assert shortfall_bips == Decimal(
        frozen["seven_day_fixed_base_reward_shortfall_bips_before_other_costs"]
    )
    assert frozen["new_market_requests_justified"] is False


def test_registry_separates_reward_stack_from_terminal_fee_only_family() -> None:
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_HASH
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 45))
    candidate = next(
        row
        for row in hypotheses
        if row["mechanism"] == "delta_hedged_bnb_simple_earn_and_airdrop_reward_stack"
    )
    assert candidate["priority_rank"] == 9
    assert candidate["market_direction_forecast_required"] is False
    assert candidate["canonical_artifacts"][0]["result_sha256"] == (
        EXPECTED_ARTIFACT_HASH
    )

    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    assert (
        terminal["binance_delta_hedged_bnb_spot_fee_discount_inventory"][
            "canonical_result_sha256"
        ]
        == EXPECTED_HEDGE_HASH
    )
    assert candidate["mechanism"] not in terminal
