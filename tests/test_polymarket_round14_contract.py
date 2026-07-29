from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round14_contract import (
    POLYMARKET_ROUND14_CAPTURE_DURATION_SECONDS,
    load_round14_contract,
    validate_round14_contract,
)


CONTRACT = (
    Path(__file__).parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-prospective-contract-v1.json"
)


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _rehash(value: dict[str, object]) -> dict[str, object]:
    output = dict(value)
    output.pop("contract_sha256", None)
    return {**output, "contract_sha256": _canonical_sha(output)}


def test_round14_contract_is_btc_only_independent_and_without_authority() -> None:
    program = load_round14_contract(CONTRACT)
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert program.capture_duration_seconds == POLYMARKET_ROUND14_CAPTURE_DURATION_SECONDS
    assert program.minimum_complete_markets == 7000
    assert program.minimum_untouched_test_markets == 1800
    assert program.paper_authority is False
    assert program.live_authority is False
    assert payload["scope"]["asset"] == "BTC"
    assert payload["scope"]["market_variant"] == "fiveminute"
    assert "No Binance credentials" in payload["scope"]["binance_boundary"]
    assert payload["risk_profiles"]["default"] == "conservative"
    assert payload["risk_profiles"]["reinvestment_default"] is False
    assert "sell_owned_fok" in payload["actions"]["choices"]
    assert (
        "buy_complement_fok_when_worst_case_payout_is_locked_after_all_costs"
        in payload["actions"]["choices"]
    )


def test_round14_contract_freezes_models_stress_and_profiles() -> None:
    program = load_round14_contract(CONTRACT)

    assert program.controls == (
        "raw_executable_polymarket_prior",
        "chainlink_structural_endpoint_probability",
        "polymarket_prior_monotone_calibration",
    )
    assert program.endpoint_models == (
        "elastic_net_logistic_residual",
        "shallow_monotone_lightgbm_residual",
        "causal_multiscale_tcn_endpoint",
    )
    assert program.action_models == (
        "distributional_executable_action_value_lightgbm",
        "causal_multitask_tcn_action_value",
    )
    assert tuple(item.name for item in program.scenarios) == (
        "primary",
        "latency_250ms",
        "latency_750ms",
        "latency_1000ms",
        "half_depth",
        "quarter_depth",
        "one_tick_adverse",
        "combined",
    )
    assert tuple(item.name for item in program.risk_profiles) == (
        "conservative",
        "regular",
        "aggressive",
    )


def test_round14_contract_is_relocatable_and_hash_bound(tmp_path: Path) -> None:
    relocated = tmp_path / CONTRACT.name
    relocated.write_bytes(CONTRACT.read_bytes())

    assert (
        load_round14_contract(relocated).contract_sha256
        == load_round14_contract(CONTRACT).contract_sha256
    )
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["status"] = "completed"
    with pytest.raises(ValueError, match="contract hash differs"):
        validate_round14_contract(payload)


def test_round14_semantic_drift_fails_even_when_rehashed() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["prospective_capture"]["duration_seconds"] = 86_400
    with pytest.raises(ValueError, match="prospective capture differs"):
        validate_round14_contract(_rehash(payload))

    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["risk_profiles"]["conservative"][
        "maximum_drawdown_capital_fraction"
    ] = "0.20"
    with pytest.raises(ValueError, match="conservative risk limits differ"):
        validate_round14_contract(_rehash(payload))

    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["ai_review"]["direct_entry_authority"] = True
    with pytest.raises(ValueError, match="AI boundary differs"):
        validate_round14_contract(_rehash(payload))


def test_round14_contract_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(
        '{"schema_version":"x","schema_version":"y"}',
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="duplicate JSON keys"):
        load_round14_contract(path)
