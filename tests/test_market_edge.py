from __future__ import annotations

from simple_ai_trading.backtest import BacktestResult
from simple_ai_trading.market_edge import build_market_edge_report


def _result(**overrides) -> BacktestResult:
    payload = dict(
        starting_cash=1000.0,
        ending_cash=1040.0,
        realized_pnl=40.0,
        win_rate=1.0,
        trades=5,
        max_drawdown=0.01,
        closed_trades=5,
        gross_exposure=500.0,
        total_fees=2.0,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        buy_hold_pnl=5.0,
        edge_vs_buy_hold=35.0,
        trade_returns=(0.010, 0.012, 0.011, 0.009, 0.013),
        trade_pnls=(10.0, 12.0, 11.0, 9.0, 13.0),
        gross_profit=55.0,
        gross_loss=0.0,
        profit_factor=999.0,
        expectancy=11.0,
    )
    payload.update(overrides)
    return BacktestResult(**payload)


def test_market_edge_report_accepts_benchmark_outperformance_with_samples() -> None:
    report = build_market_edge_report(_result(), "regular")

    assert report.accepted is True
    assert report.benchmark_name == "same_symbol_buy_hold_after_costs"
    assert report.net_edge_pct > 0.03
    assert report.sample_count == 5
    assert report.sign_test_p_value <= report.max_sign_test_p_value
    assert report.downside_return_risk_ratio == 999.0


def test_market_edge_report_rejects_tiny_edge_over_passive_market() -> None:
    report = build_market_edge_report(
        _result(realized_pnl=7.0, ending_cash=1007.0, buy_hold_pnl=5.0, edge_vs_buy_hold=2.0),
        "regular",
    )

    assert report.accepted is False
    assert "net_edge_pct<0.003000" in report.failed_checks


def test_market_edge_report_rejects_liquidation_events() -> None:
    report = build_market_edge_report(
        _result(stopped_by_liquidation=True, liquidation_events=1, liquidation_loss=25.0),
        "regular",
    )

    assert report.accepted is False
    assert report.stopped_by_liquidation is True
    assert report.liquidation_events == 1
    assert report.liquidation_loss == 25.0
    assert "liquidation_events>0" in report.failed_checks


def test_market_edge_report_rejects_missing_trade_level_evidence() -> None:
    report = build_market_edge_report(_result(trade_returns=(), trade_pnls=()), "regular")

    assert report.accepted is False
    assert "sample_count<3" in report.failed_checks
    assert report.evidence_unit == "none"


def test_market_edge_report_rejects_profit_with_bad_downside_tail() -> None:
    trade_returns = tuple([0.006] * 19 + [-0.040])
    report = build_market_edge_report(
        _result(
            realized_pnl=74.0,
            ending_cash=1074.0,
            buy_hold_pnl=5.0,
            edge_vs_buy_hold=69.0,
            closed_trades=20,
            trades=20,
            win_rate=0.95,
            trade_returns=trade_returns,
            trade_pnls=tuple(value * 1000.0 for value in trade_returns),
            gross_profit=114.0,
            gross_loss=40.0,
            profit_factor=2.85,
            expectancy=3.7,
        ),
        "conservative",
    )

    assert report.accepted is False
    assert report.downside_return_risk_ratio < report.min_downside_return_risk_ratio
    assert "downside_return_risk_ratio<0.4500" in report.failed_checks


def test_market_edge_allows_risk_explained_activity_shortfall() -> None:
    report = build_market_edge_report(
        _result(
            closed_trades=3,
            trades=3,
            trade_returns=(0.010, 0.012, 0.011),
            trade_pnls=(10.0, 12.0, 11.0),
            regime_entry_skips=6,
        ),
        "conservative",
    )

    assert report.accepted is True
    assert report.activity_gate_risk_explained is True
    assert report.risk_gate_skip_count == 6
    assert report.min_sample_count == 3
    assert "closed_trades<5" not in report.failed_checks
