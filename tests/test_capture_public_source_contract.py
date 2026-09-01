from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import _canonical_hash
from tools.capture_public_source_contract import _validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs/model-research/action-value/"
    "polymarket-us-promotional-credit-usability-contract-v1-2026-09-01.json"
)


def _contract() -> dict[str, object]:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    implementation = ROOT / "tools/capture_public_source_contract.py"
    value["implementations"][0]["sha256"] = hashlib.sha256(
        implementation.read_bytes()
    ).hexdigest()
    value["contract_sha256"] = _canonical_hash(value, "contract_sha256")
    return value


def test_current_public_source_contract_is_valid() -> None:
    _validate_contract(_contract(), CONTRACT)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("request_name"),
        lambda value: value["outputs"].pop("result_path"),
        lambda value: value.update(implementations=[]),
    ],
)
def test_validator_rejects_fields_needed_before_side_effects(
    mutate: Callable[[dict[str, object]], object],
) -> None:
    value = copy.deepcopy(_contract())
    mutate(value)
    value["contract_sha256"] = _canonical_hash(value, "contract_sha256")

    with pytest.raises(RuntimeError):
        _validate_contract(value, CONTRACT)
