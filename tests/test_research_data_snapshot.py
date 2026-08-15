from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_ai_trading.research_data_snapshot import (
    RESEARCH_DATA_CUTOFF_MS,
    RESEARCH_DATA_CUTOFF_NS,
    RESEARCH_DATA_SNAPSHOT_CONTRACT_SHA256,
    RESEARCH_DATA_SNAPSHOT_ID,
    load_research_data_snapshot_contract,
    require_historical_event_time_ms,
    require_historical_event_time_ns,
    require_historical_utc_date,
    validate_prospective_partition,
    validate_research_data_snapshot_contract,
)
from simple_ai_trading.research_data_snapshot_audit import (
    AUDIT_SCHEMA_VERSION,
    audit_research_data_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def test_research_snapshot_contract_is_exact_and_preserves_prospective_work() -> None:
    contract = load_research_data_snapshot_contract(ROOT)

    assert contract["contract_sha256"] == RESEARCH_DATA_SNAPSHOT_CONTRACT_SHA256
    assert contract["snapshot_id"] == RESEARCH_DATA_SNAPSHOT_ID
    assert contract["cutoff"]["exclusive_utc"] == "2026-08-14T00:00:00Z"
    assert contract["governance"]["automatic_latest_extension_allowed"] is False
    assert contract["governance"]["snapshot_cutoff_implies_complete_coverage_through_cutoff"] is False
    assert [
        item["experiment_id"]
        for item in contract["registered_prospective_experiments"]
    ] == [
        "polymarket-round27-stage1",
        "polymarket-round28-binance-bbo",
        "action-value-round75-continuous-capture",
    ]
    assert all(
        item["reusable_historical_training_eligible"] is False
        and item["historical_snapshot_automatically_extended"] is False
        and item["cutoff_filter_applied_to_frozen_campaign"] is False
        for item in contract["registered_prospective_experiments"]
    )


@pytest.mark.parametrize(
    ("validator", "boundary"),
    (
        (require_historical_event_time_ms, RESEARCH_DATA_CUTOFF_MS),
        (require_historical_event_time_ns, RESEARCH_DATA_CUTOFF_NS),
    ),
)
def test_historical_timestamp_cutoff_is_exclusive(validator, boundary: int) -> None:
    assert validator(boundary - 1, name="event time") == boundary - 1
    for invalid in (boundary, boundary + 1, -1, True):
        with pytest.raises(ValueError, match="outside the immutable"):
            validator(invalid, name="event time")


def test_historical_date_cutoff_is_exclusive_and_canonical() -> None:
    assert require_historical_utc_date("2026-08-13", name="period") == "2026-08-13"
    for invalid in ("2026-08-14", "2026-8-13", "not-a-date"):
        with pytest.raises(ValueError):
            require_historical_utc_date(invalid, name="period")


def test_prospective_partition_can_never_be_reusable_historical_training() -> None:
    assert (
        validate_prospective_partition(
            experiment_id="polymarket-round27-stage1",
            reusable_historical_training_eligible=False,
        )
        == "polymarket-round27-stage1"
    )
    with pytest.raises(ValueError, match="partition differs"):
        validate_prospective_partition(
            experiment_id="polymarket-round27-stage1",
            reusable_historical_training_eligible=True,
        )


def test_current_historical_inventory_is_already_below_the_global_cutoff() -> None:
    inventory = json.loads(
        (
            ROOT
            / "docs/model-research/action-value/latest/inventory.json"
        ).read_text(encoding="ascii")
    )
    periods = [
        item["selected_day"] for item in inventory["selected_months"]
    ]

    assert periods
    assert all(
        require_historical_utc_date(period, name="selected day") == period
        for period in periods
    )
    assert max(periods) == "2026-06-22"


def test_snapshot_contract_rejects_in_place_cutoff_tampering() -> None:
    contract = load_research_data_snapshot_contract(ROOT)
    tampered = dict(contract)
    tampered["cutoff"] = {**contract["cutoff"], "exclusive_utc": "2026-08-15T00:00:00Z"}

    with pytest.raises(ValueError, match="contract differs"):
        validate_research_data_snapshot_contract(tampered)


def test_repository_snapshot_audit_binds_history_and_isolates_prospective_data() -> None:
    result = audit_research_data_snapshot(ROOT)

    assert result["schema_version"] == AUDIT_SCHEMA_VERSION
    assert result["status"] == "pass"
    assert len(result["historical_artifacts"]) == 6
    assert len(result["prospective_experiments"]) == 3
    assert max(
        item["maximum_event_date_utc"] for item in result["historical_artifacts"]
    ) == "2026-07-15"
    assert all(
        item["reusable_historical_training_eligible"] is False
        for item in result["prospective_experiments"]
    )
    assert result["authority"] == {
        "credentials_used": False,
        "database_opened": False,
        "network_used": False,
        "orders_submitted": False,
        "profitability_claim": False,
        "trading_authority": False,
    }
