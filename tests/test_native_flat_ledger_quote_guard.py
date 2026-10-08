"""Native realized cash must retain its quote boundary after the last close."""

from dataclasses import replace

import pytest

from simple_ai_trading.autonomous import (
    AutonomousConfig,
    Decision,
    _apply_close_order,
    _apply_open_order,
    _close_to_trade,
    _entry_gate,
    _loss_budget_guard,
)
from simple_ai_trading.objective import get_objective
from simple_ai_trading.positions import PositionsStore
from simple_ai_trading.types import StrategyConfig
from test_autonomous_spot_cash import position, receipt


def closed_native(store: PositionsStore) -> None:
    opened = _apply_open_order(position(), receipt("BUY", "1", fee="1", asset="USDC"))
    store.record_open(opened)
    trade = _apply_close_order(
        _close_to_trade(opened, 100, "cash-check", clock=lambda: 2),
        receipt("SELL", "1"),
        "sait-c-cash",
    )
    store.record_close(trade)


def guard(store, *, symbol="BTCUSDC", market_type="spot", mark=None):
    return _loss_budget_guard(
        store,
        mark,
        StrategyConfig(),
        AutonomousConfig(starting_reference_cash=100),
        now_ms_value=2000,
        mark_symbol=symbol,
        mark_market_type=market_type,
    )


def test_closed_native_loss_cannot_be_offset_by_foreign_quote_profit(tmp_path):
    store = PositionsStore(tmp_path)
    closed_native(store)
    legacy = replace(position(), id="other", symbol="ETHUSDT")
    store.record_open(legacy)
    store.record_close(_close_to_trade(legacy, 102, "legacy", clock=lambda: 3))
    assert store.load_open() == []
    assert [t.realized_pnl for t in store.load_ledger()] == [-1, 2]
    value = guard(store)
    assert not value.allowed
    assert not value.force_close
    assert value.reason == "cash-valuation-quote-unqualified"


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "BTCBUSD"])
def test_closed_native_quote_must_match_current_admission_quote(tmp_path, symbol):
    store = PositionsStore(tmp_path)
    closed_native(store)
    value = guard(store, symbol=symbol)
    assert not value.allowed
    assert value.reason == "cash-valuation-quote-unqualified"


@pytest.mark.parametrize(
    "symbol,product", [("", "spot"), ("BTCUSDC", ""), ("BTCUSDC", "unknown")]
)
def test_closed_native_cash_still_needs_instrument_context(tmp_path, symbol, product):
    store = PositionsStore(tmp_path)
    closed_native(store)
    value = guard(store, symbol=symbol, market_type=product)
    assert not value.allowed
    assert value.reason == "cash-valuation-instrument-unqualified"


@pytest.mark.parametrize(
    "symbol,product", [("BTCUSDC", "spot"), ("ETHUSDC", "spot"), ("BTCUSDC", "futures")]
)
def test_same_quote_closed_native_cash_needs_no_current_price(
    tmp_path, symbol, product
):
    store = PositionsStore(tmp_path)
    closed_native(store)
    value = guard(store, symbol=symbol, market_type=product)
    assert not value.allowed
    assert value.reason.startswith("daily-loss-lockout:")
    assert value.daily_loss == pytest.approx(0.01)
    assert value.session_loss == pytest.approx(0.01)


def test_flat_legacy_only_history_keeps_existing_behavior(tmp_path):
    store = PositionsStore(tmp_path)
    legacy = replace(position(), id="legacy", symbol="ETHUSDT")
    store.record_open(legacy)
    store.record_close(_close_to_trade(legacy, 102, "legacy", clock=lambda: 3))
    assert guard(store, symbol="", market_type="").allowed


def test_flat_native_quote_boundary_survives_restart_and_reaches_entry_gate(tmp_path):
    store = PositionsStore(tmp_path)
    closed_native(store)
    resumed = PositionsStore(tmp_path)
    value = _entry_gate(
        resumed,
        Decision(side="LONG", confidence=0.9, mark_price=100),
        StrategyConfig(),
        AutonomousConfig(),
        get_objective("default"),
        now_ms_value=2_000_001,
        symbol="BTCUSDT",
        market_type="spot",
    )
    assert not value.allowed
    assert value.reason == "cash-valuation-quote-unqualified"
    assert resumed.load_open() == []


def test_same_quote_native_closed_cash_stays_available_after_day_rollover(tmp_path):
    store = PositionsStore(tmp_path)
    closed_native(store)
    value = _loss_budget_guard(
        store,
        None,
        StrategyConfig(),
        AutonomousConfig(starting_reference_cash=100),
        now_ms_value=86_400_001,
        mark_symbol="ETHUSDC",
        mark_market_type="spot",
    )
    assert value.allowed
    assert value.daily_loss == 0
    assert value.session_loss == pytest.approx(0.01)
