from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    ROOT / "docs/model-research/action-value/round-062-depth-stress-transition-design.json"
)
INVENTORY_PATH = (
    ROOT / "docs/model-research/action-value/round-062-official-archive-inventory.json"
)


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_round62_design_and_official_inventory_are_hash_bound() -> None:
    design = _read(DESIGN_PATH)
    canonical = dict(design)
    design_sha256 = canonical.pop("design_sha256")

    assert design_sha256 == _canonical_sha256(canonical)
    assert design["source_contract"]["inventory_file_sha256"] == hashlib.sha256(
        INVENTORY_PATH.read_bytes()
    ).hexdigest()
    inventory = _read(INVENTORY_PATH)
    assert inventory["truth_basis"] == "official_binance_data_vision_s3_listing"
    assert inventory["inventory_identity_verified"] is True
    assert inventory["inventory_errors"] == []


def test_round62_uses_full_common_depth_bands_without_cross_symbol_leakage() -> None:
    design = _read(DESIGN_PATH)
    source = design["source_contract"]
    state = design["descriptor_and_state_contract"]
    split = design["rolling_split_contract"]

    assert source["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert source["required_bands_percent"] == [-5.0, -1.0, 1.0, 5.0]
    assert source["optional_fine_bands_must_not_control_eligibility"] == [-0.2, 0.2]
    assert source["missing_rows_may_be_interpolated_or_filled"] is False
    assert state["near_band_percent"] == 1.0
    assert state["far_band_percent"] == 5.0
    assert "separately for each symbol" in state["fit_scope"]
    assert split["symbol_threshold_pooling_permitted"] is False
    assert split["latest_partial_month_permitted"] is False


def test_round62_cannot_claim_profit_or_authorize_trading() -> None:
    design = _read(DESIGN_PATH)
    governance = design["governance"]
    gate = design["evaluation_contract"]
    model = design["model_contract"]

    assert governance["profitability_claim_permitted"] is False
    assert governance["trading_authority_permitted"] is False
    assert governance["testnet_or_live_authority_permitted"] is False
    assert governance["ai_evaluation_permitted"] is False
    assert governance["economic_replay_required_before_risk_gate_integration"] is True
    assert gate["same_frozen_challenger_must_pass_every_symbol_and_horizon"] is True
    assert model["serialized_artifact_sha256_required"] is True
    assert model["artifact_must_encode_no_trading_authority"] is True
    assert model["artifact_must_encode_no_profitability_claim"] is True
