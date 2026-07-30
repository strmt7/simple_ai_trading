from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_round20_contract as contract_module


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-020-independent-redundant-corpus-contract-v1.json"
)


def _payload() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _rehash(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("contract_sha256", None)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    body["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def test_round20_is_independent_and_binance_is_optional_read_only() -> None:
    raw = _payload()
    program = contract_module.validate_round20_contract(raw)

    assert program.parent_result_sha256 == (
        "61a7a6fe2cebd3ddc8ba6d4f59c52d6c19b91fe895353fda1bb066e86ecbc5be"
    )
    independence = raw["venue_independence"]
    assert isinstance(independence, dict)
    assert independence["execution_venue"] == "polymarket"
    assert independence["binance_role"] == "optional_read_only_predictor"
    assert independence["binance_credentials_allowed"] is False
    assert independence["binance_order_endpoints_allowed"] is False
    assert independence["binance_unavailability_blocks_polymarket_core"] is False
    assert raw["authority"] == {
        "model_data_eligible": False,
        "model_selected": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }


def test_round20_freezes_redundancy_storage_and_sealed_roles() -> None:
    raw = _payload()
    program = contract_module.validate_round20_contract(raw)

    assert program.capture_unit_seconds == 1200
    assert program.total_capture_units == 2160
    assert program.pairing_window_ms == 2000
    assert program.maximum_joint_unhealthy_ms == 2000
    assert raw["redundant_clob_transport"]["lane_ids"] == ["clob-a", "clob-b"]
    assert raw["storage"]["raw_receipts_are_authority"] is True
    assert raw["storage"]["derived_union_is_rebuildable"] is True
    assert raw["labels_and_roles"]["sealed_test_calendar_days"] == 7
    assert raw["labels_and_roles"]["role_assignment_frozen_before_resolution_access"]


def test_round20_rejects_rehashed_semantic_or_authority_drift() -> None:
    for section, key, value in (
        ("venue_independence", "binance_order_endpoints_allowed", True),
        ("redundant_clob_transport", "pairing_window_ms", 3000),
        ("condition_admission", "maximum_joint_unhealthy_ms", 3000),
        ("authority", "paper_trading_authority", True),
    ):
        changed = _payload()
        nested = changed[section]
        assert isinstance(nested, dict)
        nested[key] = value
        with pytest.raises(ValueError, match="contract differs"):
            contract_module.validate_round20_contract(_rehash(changed))


def test_round20_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"round":20,"round":21}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON keys"):
        contract_module.load_round20_contract(path)
