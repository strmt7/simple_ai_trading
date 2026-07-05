from __future__ import annotations

from dataclasses import replace

from simple_ai_trading.financial_sanity import (
    build_backtest_financial_sanity_report,
    build_model_financial_sanity_report,
    build_model_lab_financial_sanity_report,
)
from simple_ai_trading.backtest import BacktestResult
from simple_ai_trading.model import TrainedModel


def _backtest_result(**overrides) -> BacktestResult:
    payload = dict(
        starting_cash=1000.0,
        ending_cash=1005.0,
        realized_pnl=5.0,
        win_rate=0.5,
        trades=2,
        max_drawdown=0.02,
        closed_trades=2,
        gross_exposure=500.0,
        total_fees=3.0,
        stopped_by_drawdown=False,
        max_exposure=500.0,
        trades_per_day_cap_hit=0,
        buy_hold_pnl=2.0,
        edge_vs_buy_hold=3.0,
        trade_pnls=(10.0, -5.0),
        trade_returns=(0.01, -0.005),
        trade_log=(
            {
                "realized_pnl": 12.0,
                "net_pnl": 10.0,
                "entry_fee": 1.0,
                "exit_fee": 1.0,
                "exit_reason": "take_profit_close",
            },
            {
                "realized_pnl": -4.0,
                "net_pnl": -5.0,
                "entry_fee": 0.5,
                "exit_fee": 0.5,
                "exit_reason": "stop_loss_close",
            },
        ),
    )
    payload.update(overrides)
    return BacktestResult(**payload)


def test_backtest_financial_sanity_accepts_consistent_accounting() -> None:
    report = build_backtest_financial_sanity_report(_backtest_result())

    assert report.allowed is True
    assert report.block_count == 0
    assert any(check.label == "backtest cash identity" and check.status == "ok" for check in report.checks)
    assert any(check.label == "win rate identity" and check.status == "ok" for check in report.checks)


def test_backtest_financial_sanity_blocks_inconsistent_accounting() -> None:
    bad_trade_log = (
        {
            "realized_pnl": 12.0,
            "net_pnl": 9.0,
            "entry_fee": 1.0,
            "exit_fee": 1.0,
            "exit_reason": "",
        },
    )
    report = build_backtest_financial_sanity_report(
        _backtest_result(
            ending_cash=1007.0,
            win_rate=1.0,
            total_fees=0.0,
            trade_log=bad_trade_log,
            trade_pnls=(10.0,),
            trade_returns=(0.01,),
        )
    )

    labels = {check.label for check in report.checks if check.status == "block"}
    assert report.allowed is False
    assert "backtest cash identity" in labels
    assert "trade log length" in labels
    assert "trade exit reason" in labels
    assert "trade net PnL identity" in labels
    assert "fee identity" in labels


def test_model_financial_sanity_blocks_malformed_parameters() -> None:
    model = TrainedModel(
        weights=[0.0, float("inf")],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        learning_rate=2.0,
        probability_temperature=0.0,
    )

    report = build_model_financial_sanity_report(model)

    assert report.allowed is False
    assert report.block_count >= 3
    assert any(check.path == "weights" and check.status == "block" for check in report.checks)
    assert any(check.path == "learning_rate" and check.status == "block" for check in report.checks)


def test_model_financial_sanity_gates_promoted_probability_calibration() -> None:
    good = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        selection_risk={"passed": True},
        execution_validation={"passed": True},
        probability_calibration_size=128,
        probability_log_loss_before=0.62,
        probability_log_loss_after=0.58,
        probability_brier_before=0.24,
        probability_brier_after=0.22,
        probability_ece_before=0.10,
        probability_ece_after=0.08,
    )
    missing = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        selection_risk={"passed": True},
        execution_validation={"passed": True},
    )
    bad = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        selection_risk={"passed": True},
        execution_validation={"passed": True},
        probability_calibration_size=64,
        probability_log_loss_before=0.61,
        probability_log_loss_after=0.90,
        probability_brier_before=0.22,
        probability_brier_after=0.42,
        probability_ece_before=0.09,
        probability_ece_after=0.24,
    )

    good_report = build_model_financial_sanity_report(good)
    missing_report = build_model_financial_sanity_report(missing)
    bad_report = build_model_financial_sanity_report(bad)
    worse_ece_report = build_model_financial_sanity_report(
        replace(good, probability_ece_before=0.08, probability_ece_after=0.12)
    )

    assert good_report.allowed is True
    assert any(check.path == "probability_brier_after" and check.status == "ok" for check in good_report.checks)
    assert missing_report.allowed is False
    assert any(check.path == "probability_brier_after" and check.status == "block" for check in missing_report.checks)
    assert any(check.path == "probability_ece_after" and check.status == "block" for check in missing_report.checks)
    assert bad_report.allowed is False
    assert any(check.path == "probability_brier_after" and check.status == "block" for check in bad_report.checks)
    assert any(check.path == "probability_ece_after" and check.status == "block" for check in bad_report.checks)
    assert any(check.path == "probability_log_loss_after" and check.status == "block" for check in bad_report.checks)
    assert worse_ece_report.allowed is False
    assert any(check.detail == "calibration increased expected calibration error" for check in worse_ece_report.checks)


def test_model_lab_financial_sanity_blocks_impossible_accepted_report() -> None:
    payload = {
        "accepted_symbols": ["AAAUSDC"],
        "portfolio_risk": {
            "accepted": True,
            "portfolio_cvar_95": 0.01,
            "portfolio_max_drawdown": 0.02,
            "deployed_weight": 0.4,
            "max_pairwise_correlation": 0.1,
            "max_cluster_weight": 0.4,
        },
        "outcomes": [
            {
                "symbol": "AAAUSDC",
                "accepted": True,
                "rows": 0,
                "objective_scores": {"regular": 0.15},
                "data_coverage": {
                    "integrity_status": "ok",
                    "coverage_ratio": 1.0,
                    "gap_count": 0,
                },
                "stress_validation": {"accepted": True, "worst_max_drawdown": 0.01},
                "robustness_validation": {"accepted": True, "worst_max_drawdown": 0.02},
            }
        ],
    }

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any("rows=0" in check.detail for check in report.checks if check.status == "block")


def test_model_lab_financial_sanity_blocks_failed_market_edge_evidence() -> None:
    payload = {
        "accepted_symbols": ["AAAUSDC"],
        "portfolio_risk": {
            "accepted": True,
            "portfolio_cvar_95": 0.01,
            "portfolio_max_drawdown": 0.02,
            "deployed_weight": 0.4,
            "max_pairwise_correlation": 0.1,
            "max_cluster_weight": 0.4,
        },
        "outcomes": [
            {
                "symbol": "AAAUSDC",
                "accepted": True,
                "rows": 500,
                "objective_scores": {"regular": 0.15},
                "data_coverage": {
                    "integrity_status": "ok",
                    "coverage_ratio": 1.0,
                    "gap_count": 0,
                },
                "stress_validation": {
                    "accepted": True,
                    "worst_max_drawdown": 0.01,
                    "objectives": [
                        {
                            "objective": "regular",
                            "accepted": True,
                            "results": [
                                {
                                    "result": {
                                        "market_edge": {
                                            "accepted": False,
                                            "reason": "net_edge_pct<0.003000",
                                            "net_edge_pct": 0.001,
                                            "min_net_edge_pct": 0.003,
                                        }
                                    }
                                }
                            ],
                        }
                    ],
                },
                "robustness_validation": {
                    "accepted": True,
                    "worst_max_drawdown": 0.02,
                    "statistical_edge_accepted": True,
                },
            }
        ],
    }

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(check.label == "market edge" and check.status == "block" for check in report.checks)


def test_model_lab_financial_sanity_blocks_accepted_market_edge_with_bad_downside_risk() -> None:
    payload = {
        "accepted_symbols": ["BTCUSDT"],
        "portfolio_risk": {
            "accepted": True,
            "portfolio_cvar_95": 0.01,
            "portfolio_max_drawdown": 0.02,
            "deployed_weight": 0.4,
            "max_pairwise_correlation": 0.1,
            "max_cluster_weight": 0.4,
        },
        "outcomes": [
            {
                "symbol": "BTCUSDT",
                "accepted": True,
                "rows": 500,
                "objective_scores": {"conservative": 0.15},
                "data_coverage": {
                    "integrity_status": "ok",
                    "coverage_ratio": 1.0,
                    "gap_count": 0,
                },
                "stress_validation": {
                    "accepted": True,
                    "worst_max_drawdown": 0.01,
                    "objectives": [
                        {
                            "objective": "conservative",
                            "accepted": True,
                            "results": [
                                {
                                    "result": {
                                        "market_edge": {
                                            "accepted": True,
                                            "reason": None,
                                            "net_edge_pct": 0.01,
                                            "min_net_edge_pct": 0.002,
                                            "downside_return_risk_ratio": 0.10,
                                            "min_downside_return_risk_ratio": 0.45,
                                        }
                                    }
                                }
                            ],
                        }
                    ],
                },
                "robustness_validation": {
                    "accepted": True,
                    "worst_max_drawdown": 0.02,
                    "statistical_edge_accepted": True,
                },
            }
        ],
    }

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "market edge downside risk" and check.status == "block"
        for check in report.checks
    )
