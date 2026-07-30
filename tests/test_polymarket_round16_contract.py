from __future__ import annotations

import hashlib
import json
from pathlib import Path


CONTRACT = (
    Path(__file__).parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-016-btc-15m-horizon-comparison-v2.json"
)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def test_round16_contract_is_hash_bound_and_execution_independent() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="ascii"))
    claimed = payload.pop("contract_sha256")

    assert _canonical_sha256(payload) == claimed
    assert payload["status"] == (
        "preregistered_before_any_round16_identity_or_terminal_target_access"
    )
    assert payload["round16_identity_rows_consulted_before_freeze"] is False
    assert payload["round16_terminal_targets_consulted_before_freeze"] is False
    assert payload["round16_model_scores_consulted_before_freeze"] is False
    scope = payload["scope"]
    assert scope["venue"] == "polymarket"
    assert scope["asset"] == "BTC"
    assert scope["market_variant"] == "fifteenminute"
    independence = scope["independence"]
    assert independence["execution_venue"] == "polymarket_only"
    assert independence["orders_and_fills"] == "polymarket_only"
    assert independence["positions_and_pnl"] == "polymarket_only"
    assert independence["risk_ledger"] == "polymarket_only"
    assert independence["reconciliation_and_stop"] == "polymarket_only"
    assert independence["binance_role"].startswith("optional_public_read_only_")
    assert {
        "credentials",
        "orders",
        "fills",
        "positions",
        "capital",
        "PnL",
        "risk authority",
        "execution authority",
    }.issubset(independence["binance_forbidden"])
    assert payload["source_contract"]["duplicate_binance_storage_allowed"] is False
    assert payload["settlement_manipulation_controls"][
        "new_exposure_in_final_30_seconds"
    ] is False
    assert payload["horizon_comparison_gate"][
        "automatic_live_promotion_allowed"
    ] is False
    assert payload["profitability_claim"] is False
    assert payload["predecessors"]["round16_v1_contract_sha256"] == (
        "a9491716ee8d2d52c20e0b3172ad7ace3d7c3da72c80e3d06cafea1a65ef9903"
    )
    assert payload["candidate_contract"]["controls"] == [
        "constant_training_prevalence",
        "regularized_calendar_logistic",
        "regularized_digital_moneyness_logistic",
    ]
    assert payload["candidate_contract"]["digital_moneyness_control"]["features"] == [
        "spot_event_to_date_log_moneyness",
        "spot_volatility_scaled_digital_moneyness",
        "perpetual_event_to_date_log_moneyness",
        "perpetual_volatility_scaled_digital_moneyness",
    ]
