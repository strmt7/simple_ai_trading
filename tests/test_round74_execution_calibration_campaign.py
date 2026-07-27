from __future__ import annotations

from collections import Counter
from copy import deepcopy
from decimal import Decimal

import pytest

from simple_ai_trading.round74_execution_calibration_campaign import (
    Round74ExecutionCampaignPlan,
    build_round74_execution_campaign_plan,
)


def test_default_campaign_is_deterministic_balanced_and_complete() -> None:
    first = build_round74_execution_campaign_plan(
        campaign_id="round74-testnet-calibration",
        target_quote_notional=Decimal("250"),
    )
    second = build_round74_execution_campaign_plan(
        campaign_id="round74-testnet-calibration",
        target_quote_notional=Decimal("250"),
    )

    assert first.plan_sha256 == second.plan_sha256
    assert len(first.slots) == 900
    assert [slot.ordinal for slot in first.slots] == list(range(900))
    symbol_counts = Counter(slot.symbol for slot in first.slots)
    side_counts = Counter((slot.symbol, slot.entry_side) for slot in first.slots)
    assert set(symbol_counts.values()) == {300}
    assert set(side_counts.values()) == {150}
    assert first.as_dict()["authority"] == {
        "testnet_execution_runtime_calibration_only": True,
        "mainnet_execution_evidence": False,
        "mainnet_trading_authority": False,
        "model_training": False,
        "financial_edge_tested": False,
        "profitability_claim": False,
    }


def test_campaign_round_trips_and_selects_first_incomplete_slot() -> None:
    plan = build_round74_execution_campaign_plan(
        campaign_id="round74-testnet-calibration",
        target_quote_notional=Decimal("250"),
    )

    restored = Round74ExecutionCampaignPlan.from_dict(plan.as_dict())
    next_slot = restored.next_slot(
        completed_round_trip_ids=tuple(
            slot.round_trip_id for slot in restored.slots[:4]
        )
    )

    assert restored.plan_sha256 == plan.plan_sha256
    assert next_slot == restored.slots[4]
    assert (
        restored.next_slot(
            completed_round_trip_ids=tuple(
                slot.round_trip_id for slot in restored.slots
            )
        )
        is None
    )


def test_campaign_rejects_tampering_unknown_completion_and_undersampling() -> None:
    plan = build_round74_execution_campaign_plan(
        campaign_id="round74-testnet-calibration",
        target_quote_notional=Decimal("250"),
    )
    tampered = deepcopy(plan.as_dict())
    tampered["slots"][0]["entry_side"] = "SELL"

    with pytest.raises(ValueError, match="digest"):
        Round74ExecutionCampaignPlan.from_dict(tampered)
    with pytest.raises(ValueError, match="completion set"):
        plan.next_slot(completed_round_trip_ids=("unknown-pair",))
    with pytest.raises(ValueError, match="pair count"):
        build_round74_execution_campaign_plan(
            campaign_id="round74-testnet-calibration",
            target_quote_notional=Decimal("250"),
            pairs_per_symbol=299,
        )
