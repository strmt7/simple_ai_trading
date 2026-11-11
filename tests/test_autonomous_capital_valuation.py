"""Capital admission needs comparable finite cash and instrument-bound marks."""

from dataclasses import replace

import pytest

from simple_ai_trading.autonomous import AutonomousConfig, _loss_budget_guard
from simple_ai_trading.positions import ClosedTrade, OpenPosition, PositionsStore
from simple_ai_trading.types import StrategyConfig


def opened(**changes):
    position = OpenPosition(
        id="legacy",
        symbol="BTCUSDC",
        market_type="futures",
        side="LONG",
        qty=1,
        entry_price=100,
        leverage=1,
        opened_at_ms=1,
        notional=100,
    )
    return replace(position, **changes)


def guard(store, mark=100, *, reference=1000, symbol="BTCUSDC", market="futures"):
    return _loss_budget_guard(
        store,
        mark,
        StrategyConfig(),
        AutonomousConfig(starting_reference_cash=reference),
        now_ms_value=3000,
        mark_symbol=symbol,
        mark_market_type=market,
    )


@pytest.mark.parametrize("market", ["spot", "futures"])
def test_legacy_inventory_cannot_use_another_instrument_mark(tmp_path, market):
    store = PositionsStore(tmp_path)
    store.record_open(opened(symbol="ETHUSDC", market_type=market))
    result = guard(store, mark=50_000, market=market)
    assert not result.allowed
    assert not result.force_close
    assert result.reason == "cash-valuation-instrument-unqualified"


@pytest.mark.parametrize("mark", [None, float("nan"), 0])
def test_unmarked_legacy_inventory_cannot_be_treated_as_zero_loss(tmp_path, mark):
    store = PositionsStore(tmp_path)
    store.record_open(opened())
    result = guard(store, mark)
    assert not result.allowed
    assert not result.force_close
    assert result.reason == "cash-valuation-instrument-unqualified"


@pytest.mark.parametrize(
    "reference", [0, -1, float("inf"), float("nan"), True, "unavailable"]
)
def test_invalid_reference_capital_never_admits_new_risk(tmp_path, reference):
    result = guard(PositionsStore(tmp_path), reference=reference)
    assert not result.allowed
    assert not result.force_close
    assert result.reason == "cash-valuation-reference-unqualified"


@pytest.mark.parametrize(
    ("entry_price", "qty", "notional", "mark", "reference"),
    [(1, 1e308, 1e308, 100, 1000), (100, 1, 100, 99, 5e-324)],
    ids=["aggregate-overflow", "loss-fraction-overflow"],
)
def test_nonfinite_valuation_never_admits_new_risk(
    tmp_path, entry_price, qty, notional, mark, reference
):
    store = PositionsStore(tmp_path)
    store.record_open(opened(entry_price=entry_price, qty=qty, notional=notional))
    result = guard(store, mark=mark, reference=reference)
    assert not result.allowed
    assert not result.force_close
    assert result.reason == "cash-valuation-nonfinite"


def test_closed_legacy_cash_cannot_be_implicitly_converted_between_quotes(tmp_path):
    store = PositionsStore(tmp_path)
    store.record_close(
        ClosedTrade(
            id="legacy-usdt",
            symbol="ETHUSDT",
            market_type="futures",
            side="LONG",
            qty=1,
            entry_price=100,
            exit_price=101,
            leverage=1,
            opened_at_ms=1,
            closed_at_ms=2,
            realized_pnl=1,
            realized_pnl_pct=0.01,
        )
    )
    result = guard(store)
    assert not result.allowed
    assert not result.force_close
    assert result.reason == "cash-valuation-quote-unqualified"


def test_valid_fractional_reference_preserves_actual_loss_fraction(tmp_path):
    store = PositionsStore(tmp_path)
    store.record_open(opened(qty=0.001, notional=0.1))
    result = guard(store, mark=99, reference=0.5)
    assert result.allowed
    assert result.daily_loss == pytest.approx(0.002)
    assert result.session_loss == pytest.approx(0.002)


def test_empty_inventory_does_not_require_a_fabricated_mark(tmp_path):
    assert guard(PositionsStore(tmp_path), mark=None).allowed
