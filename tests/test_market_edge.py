from __future__ import annotations

import math

from simple_ai_trading.backtest import BacktestResult
from simple_ai_trading.market_edge import build_market_edge_report


def _sample_stdev(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss > 0.0:
        return gross_profit / gross_loss
    if gross_profit > 0.0:
        return 999.0
    return 0.0


def _max_consecutive_losses(pnls: tuple[float, ...]) -> int:
    longest = 0
    current = 0
    for pnl in pnls:
        if pnl < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _trade_log(pnls: tuple[float, ...], returns: tuple[float, ...]) -> tuple[dict[str, object], ...]:
    trades: list[dict[str, object]] = []
    for index, (net_pnl, return_pct) in enumerate(zip(pnls, returns, strict=True)):
        entry_fee = 0.2
        exit_fee = 0.2
        realized = net_pnl + entry_fee + exit_fee
        trades.append({
            "opened_at": int(index * 120_000),
            "closed_at": int(index * 120_000 + 60_000),
            "side": 1,
            "gross_notional": 500.0,
            "entry_price": 100.0,
            "exit_mark_price": max(0.01, 100.0 + realized),
            "realized_pnl": float(realized),
            "net_pnl": float(net_pnl),
            "return_pct": float(return_pct),
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "exit_reason": "take_profit_close" if net_pnl > 0.0 else "stop_loss_close",
        })
    return tuple(trades)


def _equity_curve(starting_cash: float, pnls: tuple[float, ...]) -> tuple[dict[str, float | int], ...]:
    equity = float(starting_cash)
    peak = equity
    points: list[dict[str, float | int]] = [{
        "timestamp": 0,
        "equity": equity,
        "drawdown": 0.0,
        "position_side": 0,
    }]
    for index, pnl in enumerate(pnls, start=1):
        equity += pnl
        peak = max(peak, equity)
        drawdown = 1.0 if equity <= 0.0 and peak > 0.0 else ((peak - equity) / peak if peak else 0.0)
        points.append({
            "timestamp": int(index * 120_000),
            "equity": float(equity),
            "drawdown": float(drawdown),
            "position_side": 0,
        })
    return tuple(points)


def _scaled_positive_pnls(total: float) -> tuple[float, ...]:
    weights = (10.0, 8.0, 7.0, 6.0, 9.0)
    if total == 0.0:
        return tuple(0.0 for _ in weights)
    scaled = [total * weight / sum(weights) for weight in weights]
    scaled[-1] += total - sum(scaled)
    return tuple(scaled)


def _result(**overrides) -> BacktestResult:
    starting_cash = float(overrides.get("starting_cash", 1000.0))
    requested_realized = float(overrides.get("realized_pnl", 40.0))
    pnls = tuple(float(value) for value in overrides.get("trade_pnls", _scaled_positive_pnls(requested_realized)))
    returns = tuple(float(value) for value in overrides.get("trade_returns", tuple(value / starting_cash for value in pnls)))
    gross_profit = sum(value for value in pnls if value > 0.0)
    gross_loss = abs(sum(value for value in pnls if value < 0.0))
    realized_pnl = float(overrides.get("realized_pnl", sum(pnls)))
    ending_cash = float(overrides.get("ending_cash", starting_cash + realized_pnl))
    buy_hold_pnl = float(overrides.get("buy_hold_pnl", 5.0))
    edge_vs_buy_hold = float(overrides.get("edge_vs_buy_hold", realized_pnl - buy_hold_pnl))
    closed_trades = int(overrides.get("closed_trades", len(pnls)))
    trades = int(overrides.get("trades", closed_trades))
    curve = tuple(overrides.get("equity_curve", _equity_curve(starting_cash, pnls)))
    max_drawdown = max(float(point["drawdown"]) for point in curve) if curve else 0.0
    payload = dict(
        starting_cash=starting_cash,
        ending_cash=ending_cash,
        realized_pnl=realized_pnl,
        win_rate=(sum(1 for value in pnls if value > 0.0) / len(pnls) if pnls else 0.0),
        trades=trades,
        max_drawdown=max_drawdown,
        closed_trades=closed_trades,
        gross_exposure=500.0,
        total_fees=0.4 * len(pnls),
        stopped_by_drawdown=False,
        max_exposure=500.0,
        trades_per_day_cap_hit=0,
        buy_hold_pnl=buy_hold_pnl,
        edge_vs_buy_hold=edge_vs_buy_hold,
        equity_curve=curve,
        trade_returns=returns,
        trade_pnls=pnls,
        trade_log=_trade_log(pnls, returns) if len(pnls) == len(returns) else (),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=_profit_factor(gross_profit, gross_loss),
        expectancy=sum(pnls) / len(pnls) if pnls else 0.0,
        average_trade_return=sum(returns) / len(returns) if returns else 0.0,
        trade_return_stdev=_sample_stdev(returns),
        max_consecutive_losses=_max_consecutive_losses(pnls),
    )
    payload.update(overrides)
    return BacktestResult(**payload)


def test_market_edge_report_accepts_benchmark_outperformance_with_samples() -> None:
    report = build_market_edge_report(_result(), "regular")

    assert report.accepted is True
    assert report.financial_sanity_allowed is True
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
    assert report.financial_sanity_allowed is False
    assert report.stopped_by_liquidation is True
    assert report.liquidation_events == 1
    assert report.liquidation_loss == 25.0
    assert "financial_sanity_failed" in report.failed_checks
    assert "liquidation_events>0" in report.failed_checks


def test_market_edge_report_rejects_missing_trade_level_evidence() -> None:
    report = build_market_edge_report(_result(trade_returns=(), trade_pnls=()), "regular")

    assert report.accepted is False
    assert report.financial_sanity_allowed is True
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


def test_market_edge_report_rejects_failed_backtest_financial_sanity() -> None:
    report = build_market_edge_report(_result(max_exposure=100.0), "regular")

    assert report.accepted is False
    assert report.financial_sanity_allowed is False
    assert report.financial_sanity_block_count >= 1
    assert "financial_sanity_failed" in report.failed_checks
    assert any("gross_exposure" in reason for reason in report.financial_sanity_blocking_reasons)
