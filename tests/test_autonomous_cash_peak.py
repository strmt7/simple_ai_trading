"""Capital limits must retain settled gains when later losses retrace them."""

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from simple_ai_trading.autonomous import (
    AutonomousConfig,
    Decision,
    _entry_gate,
    _loss_budget_guard,
)
from simple_ai_trading.objective import get_objective
from simple_ai_trading.positions import ClosedTrade, OpenPosition, PositionsStore
from simple_ai_trading.types import StrategyConfig
from simple_ai_trading.capital_drawdown import settled_cash_drawdown


def setup(store, pnl=(100.0, -80.0)):
    for index, cash in enumerate(pnl):
        store.record_close(
            ClosedTrade(
                id=f"settled-{index}",
                symbol="BTCUSDC",
                market_type="futures",
                side="LONG",
                qty=1,
                entry_price=100,
                exit_price=100,
                leverage=1,
                opened_at_ms=index + 1,
                closed_at_ms=1000 + index,
                realized_pnl=cash,
                realized_pnl_pct=cash / 1000,
            )
        )
    return replace(
        StrategyConfig(),
        max_trades_per_day=0,
        cooldown_minutes=0,
        max_daily_loss_pct=0,
        max_session_loss_pct=0,
        max_consecutive_losses=0,
        max_drawdown_limit=0.05,
    )


def guard(store, strategy):
    return _loss_budget_guard(
        store,
        100,
        strategy,
        AutonomousConfig(starting_reference_cash=1000),
        now_ms_value=3000,
        mark_symbol="BTCUSDC",
        mark_market_type="futures",
    )


def test_retracement_blocks_while_lifetime_pnl_is_positive(tmp_path):
    store = PositionsStore(tmp_path)
    strategy = setup(store)
    result = guard(store, strategy)
    assert not result.allowed
    assert result.reason.startswith("drawdown-lockout")
    assert result.force_close is True
    assert result.session_loss == 0
    assert result.realized_peak_drawdown == pytest.approx(0.08)


def test_restart_reconstructs_the_same_cash_peak(tmp_path):
    store = PositionsStore(tmp_path)
    strategy = setup(store)
    assert not guard(PositionsStore(tmp_path), strategy).allowed


def test_actual_entry_gate_uses_peak_retracement(tmp_path):
    store = PositionsStore(tmp_path)
    strategy = setup(store)
    result = _entry_gate(
        store,
        Decision(side="LONG", confidence=0.9, mark_price=100, model_entry_ready=True),
        strategy,
        AutonomousConfig(starting_reference_cash=1000),
        get_objective("default"),
        now_ms_value=3000,
        symbol="BTCUSDC",
        market_type="futures",
    )
    assert not result.allowed and result.reason.startswith("drawdown-lockout")
    assert result.drawdown == pytest.approx(0.08)


def test_realized_peak_survives_later_utc_days_without_new_state(tmp_path):
    store = PositionsStore(tmp_path)
    strategy = setup(store)
    result = _loss_budget_guard(
        PositionsStore(tmp_path),
        100,
        strategy,
        AutonomousConfig(starting_reference_cash=1000),
        now_ms_value=2 * 86400000,
        mark_symbol="BTCUSDC",
        mark_market_type="futures",
    )
    assert not result.allowed and result.realized_peak_drawdown == pytest.approx(0.08)
    assert result.daily_loss == 0 and result.session_loss == 0


def test_current_mark_loss_is_counted_once_against_settled_peak(tmp_path):
    store = PositionsStore(tmp_path)
    strategy = setup(store)
    store.record_open(
        OpenPosition(
            id="marked",
            symbol="BTCUSDC",
            market_type="futures",
            side="LONG",
            qty=1,
            entry_price=110,
            leverage=1,
            opened_at_ms=2000,
            notional=110,
        )
    )
    result = guard(store, strategy)
    assert result.realized_peak_drawdown == pytest.approx(0.09)
    assert result.session_loss == 0 and not result.allowed


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_file_order_cannot_reset_settled_peak(tmp_path, order):
    store = PositionsStore(tmp_path)
    setup(store)
    rows = store.load_ledger(strict=True)
    result = settled_cash_drawdown(
        [rows[i] for i in order],
        unrealized_pnl_quote=0,
        reference_cash_quote=1000,
        observed_at_ms=3000,
    )
    assert result.reference_fraction == Fraction(2, 25)


@pytest.mark.parametrize("cash", [(-20, -10), (100, -80), (100, 20)])
def test_equal_time_net_settlements_do_not_invent_an_intermediate_peak(cash):
    for values in [cash, tuple(reversed(cash))]:
        rows = [
            SimpleNamespace(realized_pnl=value, opened_at_ms=0, closed_at_ms=100)
            for value in values
        ]
        result = settled_cash_drawdown(
            rows, unrealized_pnl_quote=0, reference_cash_quote=1000, observed_at_ms=200
        )
        assert result.drawdown_quote == max(0, -sum(cash))
        assert result.reference_fraction == Fraction(max(0, -sum(cash)), 1000)


@pytest.mark.parametrize(
    "mark,expected", [(0, Fraction(2, 25)), (-10, Fraction(9, 100)), (200, Fraction(0))]
)
def test_exact_quote_and_reference_units(mark, expected):
    rows = [
        SimpleNamespace(realized_pnl=cash, opened_at_ms=0, closed_at_ms=stamp)
        for stamp, cash in [(1, 100), (2, -80)]
    ]
    result = settled_cash_drawdown(
        rows, unrealized_pnl_quote=mark, reference_cash_quote=1000, observed_at_ms=2
    )
    assert result.reference_fraction == expected
    assert result.current_pnl_quote == 20 + mark


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "100", None])
def test_unknown_cash_is_not_a_zero_loss_observation(bad):
    rows = [SimpleNamespace(realized_pnl=bad, opened_at_ms=0, closed_at_ms=1)]
    with pytest.raises(ValueError, match="finite"):
        settled_cash_drawdown(
            rows, unrealized_pnl_quote=0, reference_cash_quote=1000, observed_at_ms=2
        )


@pytest.mark.parametrize(
    "opened,closed,now",
    [
        (0, 3, 2),
        (0, True, 2),
        (0, 1.5, 2),
        (-1, 1, 2),
        (2, 1, 3),
        (0, 1, True),
        (0, 1, -1),
    ],
)
def test_ambiguous_or_future_chronology_rejects(opened, closed, now):
    rows = [SimpleNamespace(realized_pnl=1, opened_at_ms=opened, closed_at_ms=closed)]
    with pytest.raises(ValueError):
        settled_cash_drawdown(
            rows, unrealized_pnl_quote=0, reference_cash_quote=1000, observed_at_ms=now
        )


@pytest.mark.parametrize("reference", [0, -1, True, float("inf")])
def test_invalid_reference_is_not_an_admission_basis(reference):
    with pytest.raises(ValueError):
        settled_cash_drawdown(
            [], unrealized_pnl_quote=0, reference_cash_quote=reference, observed_at_ms=1
        )


@pytest.mark.parametrize("body", [b"broken", b"{}", b"[{}]", b"[NaN]"])
def test_malformed_retained_history_cannot_become_an_empty_loss_record(tmp_path, body):
    store = PositionsStore(tmp_path)
    strategy = StrategyConfig()
    store.ledger_path.write_bytes(body)
    result = guard(store, strategy)
    assert not result.allowed and result.reason == "cash-valuation-ledger-unqualified"
    assert result.realized_peak_drawdown is None and not result.force_close


@pytest.mark.parametrize(
    "field", ["max_daily_loss_pct", "max_session_loss_pct", "max_drawdown_limit"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_bad_loss_policy_never_silently_disables_admission(tmp_path, field, value):
    store = PositionsStore(tmp_path)
    strategy = setup(store)
    # Exercise an invalid effective value; constructor normalization is separate.
    setattr(strategy, field, value)
    result = guard(store, strategy)
    assert not result.allowed and result.reason == "risk-policy-loss-limits-unqualified"


@pytest.mark.parametrize("value", [True, -1, 1.5, "2"])
def test_loss_streak_policy_requires_an_exact_nonnegative_count(tmp_path, value):
    store = PositionsStore(tmp_path)
    strategy = setup(store)
    strategy.max_consecutive_losses = value
    result = guard(store, strategy)
    assert not result.allowed and result.reason == "risk-policy-loss-limits-unqualified"
