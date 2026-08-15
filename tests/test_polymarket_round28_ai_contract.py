from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round28_ai_contract import (
    POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
    POLYMARKET_ROUND28_AI_MODEL_IDS,
    load_round28_ai_contract,
    validate_round28_ai_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_round28_ai_contract_is_exact_target_free_and_non_authoritative() -> None:
    contract = load_round28_ai_contract(ROOT)

    assert contract["contract_sha256"] == POLYMARKET_ROUND28_AI_CONTRACT_SHA256
    assert tuple(
        candidate["model_id"] for candidate in contract["candidate_program"]
    ) == POLYMARKET_ROUND28_AI_MODEL_IDS
    assert contract["candidate_program"][2]["runtime_digest"] is None
    assert contract["knowledge_at_freeze"] == {
        "official_outcomes_accessed": False,
        "round28_feature_rows_accessed_or_materialized": False,
        "round28_model_fitted": False,
        "sealed_partition_accessed": False,
        "selection_partition_accessed": False,
        "stage1_a_capture_running": True,
    }
    assert set(contract["authority"].values()) == {False}
    assert contract["prompt_contract"]["action"] == "risk_veto_only"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("authority", "orders_submitted", True),
        ("knowledge_at_freeze", "official_outcomes_accessed", True),
        ("case_materialization", "target_accessed", True),
        ("prompt_contract", "free_form_external_text_allowed", True),
        ("inference_runtime", "temperature", 0.1),
        ("evaluation", "maximum_rejected_fraction", 0.75),
    ],
)
def test_round28_ai_contract_rejects_semantic_or_hash_drift(
    section: str,
    field: str,
    value: object,
) -> None:
    original = json.loads(
        (ROOT / "docs/model-research/polymarket/round-028-ai-risk-veto-preregistration-v1.json").read_text(
            encoding="ascii"
        )
    )
    changed = deepcopy(original)
    changed[section][field] = value

    with pytest.raises(ValueError, match="contract differs"):
        validate_round28_ai_contract(changed)


def test_round28_ai_contract_rejects_unqualified_challenger_claim() -> None:
    contract = load_round28_ai_contract(ROOT)
    changed = deepcopy(contract)
    changed["candidate_program"][2]["host_qualification"] = (
        "passed_round27_exact_artifact"
    )
    changed["candidate_program"][2]["runtime_digest"] = "a" * 64

    with pytest.raises(ValueError, match="contract differs"):
        validate_round28_ai_contract(changed)
