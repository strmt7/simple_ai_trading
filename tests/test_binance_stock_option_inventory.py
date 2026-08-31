from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION_VALUE / "binance-stock-option-inventory-contract-v1-2026-08-31.json"
RESULT = ACTION_VALUE / "binance-stock-option-inventory-result-v1-2026-08-31.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
TOOL = ROOT / "tools/check_binance_commodity_option_expiry_delta.py"
RAW = ROOT / "data/binance-stock-option-inventory-v1/raw/options-exchange-info.raw"
JOURNAL = ROOT / "data/binance-stock-option-inventory-v1/request-journal.jsonl"


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


def test_contract_freezes_one_public_inventory_request_and_exact_method() -> None:
    contract = _load(CONTRACT)

    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == (
        "08bcd2fd86082e9c4d03b4408c38aabfb5f84248ace9748d6898ebcef3114624"
    )
    assert contract["status"] == "frozen_before_one_public_inventory_request"
    assert contract["population_filter"] == {
        "contract_type": "TRADFI_OPTIONS",
        "status": "TRADING",
        "underlying_type": "EQUITY",
    }
    assert contract["capture"]["maximum_http_requests"] == 1
    assert contract["capture"]["retry_permitted"] is False
    assert (
        contract["method"]["implementation_sha256"]
        == hashlib.sha256(TOOL.read_bytes()).hexdigest()
    )


def test_complete_public_population_is_empty_and_stops_before_economics() -> None:
    result = _load(RESULT)
    receipt = json.loads(JOURNAL.read_text(encoding="ascii"))

    assert result["result_sha256"] == (
        "b72592efad26563aacc4e6d8611f15f3172039ffc220153ab824bf57231afcb3"
    )
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert (
        result["contract"]["sha256"]
        == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    )
    assert result["authority"] == {
        "account_state_accessed": False,
        "authenticated_requests": 0,
        "orders_quotes_transfers_or_wallet_actions": 0,
        "paper_or_live_trading_authority": False,
        "public_unauthenticated_GET_requests": 1,
    }
    assert result["population"] == {
        "active_stock_option_count": 0,
        "expiry_groups": [],
        "sorted_symbol_population_sha256": hashlib.sha256(b"").hexdigest(),
        "underlyings": [],
    }
    assert result["adjudication"]["stock_option_inventory_trigger_satisfied"] is False
    assert result["adjudication"]["accepted_edge"] is False
    assert receipt["method"] == "GET"
    assert receipt["status_code"] == 200
    assert receipt["url"] == "https://eapi.binance.com/eapi/v1/exchangeInfo"
    assert receipt["response_sha256"] == hashlib.sha256(RAW.read_bytes()).hexdigest()
    assert receipt["response_bytes"] == RAW.stat().st_size == 1_290_567


def test_registry_records_literal_reopen_trigger_without_promoting_profit() -> None:
    registry = _load(REGISTRY)

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "binance_long_stock_option_opposite_TradFi_perpetual_terminal_payoff_lower_bound"
    )
    assert hypothesis["priority_rank"] == 46
    assert hypothesis["market_direction_forecast_required"] is False
    assert hypothesis["retry_trigger"] == (
        "new_official_Binance_TradFi_equity_option_listing_or_material_stock_option_settlement_fee_access_contract_unit_or_matching_TradFi_perpetual_architecture_change"
    )
    result_hash = _load(RESULT)["result_sha256"]
    assert hypothesis["canonical_artifacts"][-1] == {
        "path": RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": result_hash,
    }
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"] == "binance_active_TradFi_equity_option_inventory_2026_08_31"
    )
    assert terminal["canonical_result_sha256"] == result_hash
