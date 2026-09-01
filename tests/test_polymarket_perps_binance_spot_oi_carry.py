import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from tools.adjudicate_polymarket_perps_binance_spot_oi_carry import (
    HOUR_MS,
    _normalize_hour,
    role_economics,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_normalize_hour_enforces_frozen_tolerance() -> None:
    assert _normalize_hour(10 * HOUR_MS + 172, 500) == (10 * HOUR_MS, 172)
    with pytest.raises(ValueError, match="exceeds"):
        _normalize_hour(10 * HOUR_MS + 501, 500)


def test_missing_funding_hours_are_zero_not_dropped_from_capital_time() -> None:
    result = role_economics(
        {HOUR_MS: Decimal("0.001")},
        start_ms=0,
        end_ms=2 * HOUR_MS,
        annual_reward_bips=Decimal("0"),
        annual_opportunity_bips_per_leg=Decimal("0"),
    )

    assert result["expected_funding_hours"] == 2
    assert result["observed_funding_hours"] == 1
    assert result["missing_funding_hours_valued_at_zero"] == 1
    assert result["funding_bips"] == "10.000"


def test_public_crypto_perps_fee_budget_is_source_bound_and_fail_closed() -> None:
    contract = _load(
        "docs/model-research/action-value/"
        "polymarket-perps-crypto-fee-source-contract-v1-2026-09-01.json"
    )
    capture = _load(
        "docs/model-research/action-value/"
        "polymarket-perps-crypto-fee-source-capture-v1-2026-09-01.json"
    )
    adjudication = _load(
        "docs/model-research/action-value/"
        "polymarket-perps-binance-spot-public-fee-budget-adjudication-v1-2026-09-01.json"
    )

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(capture, "result_sha256") == capture["result_sha256"]
    assert (
        _self_hash(adjudication, "result_sha256")
        == adjudication["result_sha256"]
    )

    raw = ROOT / str(adjudication["source_bindings"]["new_fee_source_raw"]["path"])
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == (
        adjudication["source_bindings"]["new_fee_source_raw"]["sha256"]
    )

    tiers = adjudication["polymarket_perps_fee_schedule"]["tiers"]
    assert tiers[0] == {
        "minimum_30_day_volume_usd": "0",
        "taker_percent": "0.0400",
        "maker_percent": "0.0125",
    }
    assert tiers[-1]["maker_percent"] == "-0.0050"

    economics = adjudication["fee_budget_bips"]
    budget = Decimal(
        adjudication["retained_carry_economics_bips"][
            "maximum_total_all_in_friction_budget"
        ]
    )
    assert budget - Decimal("2.50") == Decimal(
        economics[
            "residual_after_polymarket_zero_volume_maker_round_trip_before_every_binance_basis_conversion_transfer_custody_and_failure_cost"
        ]
    )
    assert budget - Decimal("22.50") == Decimal(
        economics[
            "residual_after_combined_standard_maker_fee_sensitivity_before_basis_and_every_external_cost"
        ]
    )
    assert adjudication["adjudication"]["accepted_edge"] is False
    assert adjudication["adjudication"]["book_request_permitted"] is False

    registry = _load(
        "docs/model-research/structural-edge-priority-registry-v1.json"
    )
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["priority_rank"] == 19
    )
    assert any(
        artifact["result_sha256"] == adjudication["result_sha256"]
        for artifact in row["canonical_artifacts"]
    )
    assert "do_not_refresh_public_history_or_books" in row["next_action"]
