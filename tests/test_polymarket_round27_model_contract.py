from __future__ import annotations

import copy
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round27_model_contract import (
    POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256,
    load_round27_model_contract,
    validate_round27_model_contract,
)
from simple_ai_trading.polymarket_round27_model_amendment import (
    POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD,
    POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256,
)


_ROOT = Path(__file__).resolve().parents[1]


def test_round27_model_contract_is_frozen_and_source_bound() -> None:
    contract = load_round27_model_contract(_ROOT)

    assert contract["contract_sha256"] == POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256
    assert (
        contract[POLYMARKET_ROUND27_MODEL_AMENDMENT_FIELD]
        == POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256
    )
    assert contract["knowledge_at_freeze"] == {
        "ai_assist_economic_metrics_computed": False,
        "campaign_capture_started": False,
        "model_fitted": False,
        "official_outcomes_accessed": False,
        "performance_metrics_computed": False,
        "sealed_partition_accessed": False,
        "selection_partition_accessed": False,
        "stage1_market_states_accessed": False,
    }
    assert contract["ai_assist"]["maximum_authority"] == "veto_or_reduce"
    assert contract["economic_evaluation"]["rate_limit_or_account_authority"] is False


def test_round27_model_contract_rejects_policy_tampering() -> None:
    contract = load_round27_model_contract(_ROOT)
    tampered = copy.deepcopy(contract)
    tampered["ai_assist"]["may_create_or_increase_position"] = True

    with pytest.raises(ValueError, match="identity differs"):
        validate_round27_model_contract(tampered, repository=_ROOT)
