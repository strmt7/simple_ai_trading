from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round27_ai_ablation_contract import (
    POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256,
    POLYMARKET_ROUND27_AI_MINIMUM_BASELINE_CANDIDATES,
    load_round27_ai_ablation_contract,
    validate_round27_ai_ablation_contract,
)


def test_round27_ai_ablation_contract_is_frozen_before_capture() -> None:
    repository = Path(__file__).resolve().parents[1]
    contract = load_round27_ai_ablation_contract(repository)

    assert contract["contract_sha256"] == POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
    assert contract["created_at_ms"] < 1_786_784_400_000
    assert contract["evaluation"]["minimum_baseline_candidate_conditions"] == 60
    assert POLYMARKET_ROUND27_AI_MINIMUM_BASELINE_CANDIDATES == 60
    assert contract["reduce_semantics"]["reduce_execution_semantics"] == "abstain"
    assert all(value is False for value in contract["authority"].values())


def test_round27_ai_ablation_contract_rejects_semantic_tampering() -> None:
    repository = Path(__file__).resolve().parents[1]
    contract = load_round27_ai_ablation_contract(repository)
    tampered = copy.deepcopy(contract)
    tampered["prompt_contract"]["target_allowed"] = True

    with pytest.raises(ValueError, match="contract differs"):
        validate_round27_ai_ablation_contract(tampered)


def test_round27_ai_ablation_contract_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps({"schema_version": "first"})[:-1]
        + ',"schema_version":"second"}',
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="not strict JSON"):
        load_round27_ai_ablation_contract(tmp_path, path)
