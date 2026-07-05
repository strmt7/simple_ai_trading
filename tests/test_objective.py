"""Branch-coverage tests for the objective scoring module."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from simple_ai_trading.assets import DEFAULT_AGGRESSIVE_LEVERAGE
from simple_ai_trading.backtest import BacktestResult
from simple_ai_trading import objective as obj


def _result(**overrides) -> BacktestResult:
    base = dict(
        starting_cash=1000.0,
        ending_cash=1100.0,
        realized_pnl=100.0,
        win_rate=0.6,
        trades=10,
        max_drawdown=0.05,
        closed_trades=10,
        gross_exposure=500.0,
        total_fees=1.0,
        stopped_by_drawdown=False,
        max_exposure=500.0,
        trades_per_day_cap_hit=0,
        trade_pnls=(12.5, 12.5, 12.5, -12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5),
        trade_returns=(0.0125, 0.0123, 0.0121, -0.0120, 0.0118, 0.0116, 0.0114, 0.0112, 0.0110, 0.0108),
        gross_profit=112.5,
        gross_loss=12.5,
        profit_factor=9.0,
        expectancy=10.0,
        average_trade_return=0.00927,
        trade_return_stdev=0.0073,
        max_consecutive_losses=1,
        buy_hold_pnl=80.0,
        edge_vs_buy_hold=20.0,
    )
    base.update(overrides)
    return BacktestResult(**base)


def test_available_and_describe():
    names = obj.available_objectives()
    assert set(names) == {"conservative", "regular", "aggressive"}
    described = obj.describe_objectives()
    assert {entry["name"] for entry in described} == {"conservative", "regular", "aggressive"}
    assert all("summary" in entry for entry in described)
    assert obj.RISKY.max_drawdown_rejection <= 0.30
    assert obj.RISKY.training is not None
    assert obj.RISKY.training.max_position_pct <= 0.25
    assert obj.RISKY.training.leverage == pytest.approx(DEFAULT_AGGRESSIVE_LEVERAGE)


def test_get_objective_lookups():
    assert obj.get_objective("DEFAULT").name == "conservative"
    assert obj.get_objective("balanced").name == "regular"
    assert obj.get_objective("risky").name == "aggressive"
    with pytest.raises(ValueError):
        obj.get_objective("")
    with pytest.raises(ValueError):
        obj.get_objective("nonesuch")


def test_scorer_paths_basic():
    result = _result()
    assert obj.CONSERVATIVE.score(result) < obj.RISKY.score(result)
    assert obj.DEFAULT.accepts(result) is True
    assert obj.DEFAULT.name == "conservative"


def test_stopped_by_drawdown_penalty_paths():
    r = _result(stopped_by_drawdown=True, closed_trades=10, max_drawdown=0.10)
    for spec in (obj.CONSERVATIVE, obj.DEFAULT, obj.RISKY):
        # scorer still returns a float; acceptance may differ
        assert isinstance(spec.score(r), float)


def test_conservative_rejects_on_drawdown():
    r = _result(max_drawdown=0.5)
    assert obj.CONSERVATIVE.accepts(r) is False


def test_conservative_rejects_on_stopped_by_drawdown():
    r = _result(stopped_by_drawdown=True)
    assert obj.CONSERVATIVE.accepts(r) is False


def test_min_closed_trades_gate():
    r = _result(closed_trades=1)
    assert obj.CONSERVATIVE.accepts(r) is False


def test_min_realized_pnl_gate():
    r = _result(realized_pnl=-0.01, closed_trades=10)
    assert obj.REGULAR.accepts(r) is False
    ranked = obj.rank_candidates([({"id": "negative"}, r)], obj.REGULAR)
    assert "realized_pnl<=0.0" in ranked[0]["reject_reason"]


def test_positive_pnl_negative_buy_hold_edge_is_rejected():
    r = _result(
        realized_pnl=25.0,
        buy_hold_pnl=80.0,
        edge_vs_buy_hold=-55.0,
        closed_trades=10,
    )

    assert obj.REGULAR.accepts(r) is False
    ranked = obj.rank_candidates([({"id": "underperformer"}, r)], obj.REGULAR)
    assert ranked[0]["accepted"] is False
    assert ranked[0]["score"] == float("-inf")
    assert "edge_vs_buy_hold<0.0" in ranked[0]["reject_reason"]
    assert "market_edge_pct<0.003" in ranked[0]["reject_reason"]
    assert obj.DEFAULT.reject_reason(r) == "edge_vs_buy_hold<0.0; market_edge_pct<0.002"


def test_rejection_reasons_are_stable_machine_labels():
    r = _result(realized_pnl=-1.0, closed_trades=1, edge_vs_buy_hold=-2.0)

    reasons = obj.REGULAR.rejection_reasons(r)

    assert reasons == [
        "closed_trades<3",
        "realized_pnl<=0.0",
        "edge_vs_buy_hold<0.0",
        "market_edge_pct<0.003",
    ]


def test_market_edge_must_clear_material_threshold() -> None:
    barely_profitable = _result(realized_pnl=10.0, buy_hold_pnl=9.5, edge_vs_buy_hold=0.5)

    reasons = obj.REGULAR.rejection_reasons(barely_profitable)

    assert "market_edge_pct<0.003" in reasons
    assert obj.REGULAR.accepts(barely_profitable) is False


def test_path_quality_gates_reject_fragile_profitable_models():
    weak_factor = _result(profit_factor=1.04)
    weak_expectancy = _result(expectancy=0.0)
    long_loss_streak = _result(max_consecutive_losses=6)

    assert "profit_factor<1.1" in obj.CONSERVATIVE.rejection_reasons(weak_factor)
    assert "expectancy<=0.0" in obj.REGULAR.rejection_reasons(weak_expectancy)
    assert "max_consecutive_losses>5" in obj.REGULAR.rejection_reasons(long_loss_streak)


def test_path_quality_gates_skip_legacy_payloads_without_trade_evidence():
    legacy = _result(
        trade_pnls=(),
        trade_returns=(),
        gross_profit=0.0,
        gross_loss=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        max_consecutive_losses=0,
    )

    assert obj.CONSERVATIVE.accepts(legacy) is True


def test_path_quality_gates_can_derive_metrics_from_trade_pnls():
    raw_path = _result(
        closed_trades=3,
        trade_pnls=(10.0, -20.0, -1.0),
        trade_returns=(0.010, -0.020, -0.001),
        gross_profit=0.0,
        gross_loss=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        max_consecutive_losses=0,
    )
    clean_winner = _result(
        closed_trades=4,
        trade_pnls=(6.0, 7.0, 8.0, 9.0),
        trade_returns=(0.006, 0.007, 0.008, 0.009),
        gross_profit=0.0,
        gross_loss=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        max_consecutive_losses=0,
    )

    reasons = obj.REGULAR.rejection_reasons(raw_path)

    assert "profit_factor<1.05" in reasons
    assert "expectancy<=0.0" in reasons
    assert obj.REGULAR.accepts(clean_winner) is True


def test_trade_frequency_gate_allows_risk_gated_inactivity_without_forcing_trades() -> None:
    days = 5
    sparse_safe = _result(
        closed_trades=5,
        equity_curve=(
            {"timestamp": 0, "equity": 1000.0, "drawdown": 0.0, "position_side": 0},
            {"timestamp": days * 86_400_000, "equity": 1100.0, "drawdown": 0.0, "position_side": 0},
        ),
        trade_log=tuple(
            {
                "opened_at": day * 86_400_000,
                "closed_at": day * 86_400_000 + 60_000,
                "net_pnl": 20.0,
            }
            for day in range(days)
        ),
        regime_entry_skips=20,
    )
    sparse_unexplained = _result(
        closed_trades=5,
        equity_curve=sparse_safe.equity_curve,
        trade_log=sparse_safe.trade_log,
        regime_entry_skips=0,
    )

    assert obj.CONSERVATIVE.accepts(sparse_safe) is True
    assert "trades_per_day<2.0" in obj.CONSERVATIVE.rejection_reasons(sparse_unexplained)


def test_safe_and_return_ratio_non_finite():
    assert obj._safe(float("nan")) == 0.0
    assert obj._safe(1.2) == pytest.approx(1.2)
    r = _result(starting_cash=0.0, realized_pnl=10.0)
    assert obj._return_ratio(r) == 0.0
    r2 = _result(realized_pnl=float("nan"))
    # scorer should absorb NaN â†’ 0 via _safe
    assert math.isfinite(obj.CONSERVATIVE.score(r2))


def test_max_drawdown_rejection_one_never_rejects_on_drawdown():
    # Manually craft an objective with rejection=1.0 â€” never rejects on drawdown
    spec = replace(obj.DEFAULT, max_drawdown_rejection=1.0, min_closed_trades=0)
    assert spec.accepts(_result(max_drawdown=0.99)) is True


def test_rank_candidates_orders_accepted_first():
    winning = _result(realized_pnl=500.0, max_drawdown=0.02, closed_trades=10)
    losing = _result(realized_pnl=-100.0, max_drawdown=0.30, closed_trades=1)
    ranked = obj.rank_candidates(
        [({"id": "loser"}, losing), ({"id": "winner"}, winning)],
        obj.CONSERVATIVE,
    )
    assert ranked[0]["params"]["id"] == "winner"
    assert ranked[0]["accepted"] is True
    assert ranked[1]["accepted"] is False
    assert ranked[1]["score"] == float("-inf")
    assert ranked[1]["reject_reason"]


def test_rank_candidates_records_stopped_by_drawdown_reason():
    bad = _result(stopped_by_drawdown=True, max_drawdown=0.3, closed_trades=5)
    ranked = obj.rank_candidates([({"id": "x"}, bad)], obj.CONSERVATIVE)
    assert "stopped_by_drawdown" in ranked[0]["reject_reason"]


def test_rank_candidates_hard_gate_fallback_reason():
    # Build a fake objective where the only rejection is a hard gate we can't
    # describe via the declared reasons â€” forces the "hard-gate-failed" default.
    class _Custom(obj.ObjectiveSpec):
        pass

    spec = replace(obj.DEFAULT, min_closed_trades=0, max_drawdown_rejection=1.0)
    # accepts is overridden implicitly by replace? No â€” but we can use a subclass
    def never(_):
        return False

    spec_obj = replace(spec)  # keep dataclass happy
    # attach an accepts-always-False shim by wrapping
    class _Wrap:
        def __init__(self, inner):
            self._inner = inner
            self.name = inner.name
            self.label = inner.label
            self.summary = inner.summary
            self.long_description = inner.long_description
            self.min_closed_trades = inner.min_closed_trades
            self.min_realized_pnl = inner.min_realized_pnl
            self.max_drawdown_rejection = inner.max_drawdown_rejection
            self.scorer = inner.scorer
            self.training = inner.training

        def score(self, result):
            return 0.0

        def accepts(self, _result):
            return False

    wrapped = _Wrap(spec_obj)
    ranked = obj.rank_candidates([({"id": "x"}, _result())], wrapped)
    assert ranked[0]["reject_reason"] == "hard-gate-failed"


def test_objective_training_dataclass_defaults_present():
    for spec in (obj.CONSERVATIVE, obj.DEFAULT, obj.RISKY):
        assert spec.training is not None
        assert spec.training.epochs > 0
        assert spec.min_realized_pnl == pytest.approx(0.0)
