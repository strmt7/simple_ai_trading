from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION_VALUE / "binance-native-stock-perpetual-parity-contract-v1.json"
RESULT = ACTION_VALUE / "binance-native-stock-perpetual-parity-v1-2026-08-27.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
IMPLEMENTATION = ROOT / "tools/screen_binance_native_stock_perpetual_parity.py"
CONTRACT_HASH = "ec5d4855c69d3afa461838b674530936a07e646394540a1a2b30ae3ddaf77db1"
RESULT_HASH = "2776ff86fddf78e7e87860c6b9500cb237fce5af908a4840d351ae0cc2eff930"
REGISTRY_HASH = "aabfdc0750a619b380929c59546d37c86306686bc2144d85c90d770f5bea6d23"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_contract_freezes_complete_same_usdt_quote_overlap_before_capture() -> None:
    contract = _load(CONTRACT)

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    universe = contract["frozen_universe"]
    assert len(universe) == 14
    assert len({row["ticker"] for row in universe}) == 14
    assert all(row["perpetual_symbol"] == f"{row['ticker']}USDT" for row in universe)
    assert contract["capture"]["maximum_public_GET_requests"] == 28
    assert contract["capture"]["maximum_public_websocket_connections"] == 14
    assert contract["capture"]["retry_permitted"] is False
    assert contract["discovery_boundary"]["excluded_cross_quote"] == {
        "ticker": "SPCX",
        "perpetual_symbol": "SPCXUSD1",
        "reason": (
            "requires_a_distinct_USD1_to_USDC_executable_conversion_path_and_is_"
            "not_part_of_the_frozen_same_USDT_quote_identity"
        ),
    }


def test_capture_is_hash_bound_public_only_and_preserves_incompleteness() -> None:
    result = _load(RESULT)

    assert result["result_sha256"] == RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    assert result["contract"]["sha256"] == CONTRACT_HASH
    assert (
        result["implementation"]["sha256"]
        == hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
    )
    assert result["authority"] == {
        "public_unauthenticated_GET_requests": 26,
        "public_unauthenticated_websocket_connections": 14,
        "authenticated_requests": 0,
        "account_state_accessed": False,
        "orders_quotes_transfers_disclaimer_or_wallet_actions": 0,
        "paper_or_live_trading_authority": False,
    }
    capture = result["capture"]
    assert capture["complete_population"] is False
    assert capture["complete_row_count"] == 13
    assert capture["incomplete_row_count"] == 1
    assert capture["errors"] == [
        {
            "ticker": "KLAC",
            "perpetual_symbol": "KLACUSDT",
            "error_type": "TimeoutError",
            "error": "",
        }
    ]
    assert capture["retained_source_count"] == 39


def test_observed_rows_fail_stress_or_one_share_capacity() -> None:
    result = _load(RESULT)
    economics = result["economics"]

    assert economics["row_count"] == 13
    assert economics["after_labeled_stress_positive_count"] == 0
    assert economics["best_ticker"] == "NBIS"
    assert Decimal(economics["best_gross_entry_headroom_bps"]) < Decimal("30")
    rows = economics["rows"]
    best_capacity_valid = max(
        (row for row in rows if row["all_three_top_level_capacities_pass"]),
        key=lambda row: Decimal(row["gross_entry_headroom_bps"]),
    )
    assert best_capacity_valid["ticker"] == "SNDK"
    assert Decimal(best_capacity_valid["gross_entry_headroom_bps"]) == Decimal(
        "14.64029978610898480329251573"
    )
    for row in rows:
        stock_ask = Decimal(row["native_stock_best_ask_USD"])
        perpetual_bid = Decimal(row["perpetual_best_bid_USDT"])
        fx_ask = Decimal(row["USDCUSDT_best_ask"])
        gross = perpetual_bid / fx_ask - stock_ask
        stress = stock_ask * Decimal("30") / Decimal("10000")
        assert Decimal(row["gross_entry_headroom_USDC"]) == gross
        assert Decimal(row["labeled_30_bps_stress_USDC"]) == stress
        assert Decimal(row["after_labeled_stress_USDC"]) == gross - stress
        assert row["after_labeled_stress_positive"] is False

    adjudication = result["adjudication"]
    assert adjudication["status"] == (
        "incomplete_public_quote_population_no_economic_adjudication"
    )
    assert adjudication["accepted_edge"] is False
    assert adjudication["public_after_cost_profit_floor_USDC"] == "0"


def test_registry_adds_unaccepted_missing_row_recovery_only() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 16
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 41))
    hypothesis = next(
        row
        for row in hypotheses
        if row["mechanism"]
        == "binance_native_stock_USDC_TradFi_perpetual_USDT_parity"
    )
    assert hypothesis["retry_trigger"] == (
        "new_active_KLAC_native_stock_quote_state_for_one_separately_frozen_"
        "missing_row_recovery_only"
    )
    assert "do_not_resample_the_13_observed_rows" in hypothesis["next_action"]
