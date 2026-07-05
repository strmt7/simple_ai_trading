from __future__ import annotations

from dataclasses import replace
import math

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
        gross_profit=10.0,
        gross_loss=5.0,
        profit_factor=2.0,
        expectancy=2.5,
        average_trade_return=0.0025,
        trade_return_stdev=math.sqrt(0.0001125),
        max_consecutive_losses=1,
        trade_pnls=(10.0, -5.0),
        trade_returns=(0.01, -0.005),
        trade_log=(
            {
                "opened_at": 1000,
                "closed_at": 2000,
                "side": 1,
                "gross_notional": 500.0,
                "entry_price": 100.0,
                "exit_mark_price": 102.4,
                "realized_pnl": 12.0,
                "net_pnl": 10.0,
                "return_pct": 0.01,
                "entry_fee": 1.0,
                "exit_fee": 1.0,
                "exit_reason": "take_profit_close",
            },
            {
                "opened_at": 3000,
                "closed_at": 4000,
                "side": -1,
                "gross_notional": 500.0,
                "entry_price": 100.0,
                "exit_mark_price": 100.8,
                "realized_pnl": -4.0,
                "net_pnl": -5.0,
                "return_pct": -0.005,
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


def test_backtest_financial_sanity_blocks_liquidation_evidence() -> None:
    result = _backtest_result(
        ending_cash=995.0,
        realized_pnl=-5.0,
        win_rate=0.0,
        trades=1,
        closed_trades=1,
        total_fees=1.0,
        buy_hold_pnl=-1.0,
        edge_vs_buy_hold=-4.0,
        stopped_by_liquidation=True,
        liquidation_events=1,
        liquidation_loss=20.0,
        gross_profit=0.0,
        gross_loss=5.0,
        profit_factor=0.0,
        expectancy=-5.0,
        average_trade_return=-0.005,
        trade_return_stdev=0.0,
        max_consecutive_losses=1,
        trade_pnls=(-5.0,),
        trade_returns=(-0.005,),
        trade_log=(
            {
                "opened_at": 1000,
                "closed_at": 2000,
                "side": 1,
                "gross_notional": 500.0,
                "entry_price": 100.0,
                "exit_mark_price": 99.0,
                "realized_pnl": -4.0,
                "net_pnl": -5.0,
                "return_pct": -0.005,
                "entry_fee": 0.5,
                "exit_fee": 0.5,
                "exit_reason": "liquidation",
            },
        )
    )
    report = build_backtest_financial_sanity_report(result)

    assert report.allowed is False
    blocked = {(check.label, check.path) for check in report.checks if check.status == "block"}
    assert ("liquidation evidence", "stopped_by_liquidation") in blocked
    assert ("liquidation evidence", "liquidation_events") in blocked
    assert ("liquidation evidence", "liquidation_loss") in blocked
    assert ("liquidation evidence", "trade_log[0].exit_reason") in blocked
    assert not any(check.label == "backtest cash identity" and check.status == "block" for check in report.checks)
    assert not any(check.label == "trade PnL identity" and check.status == "block" for check in report.checks)

    accounting_only = build_backtest_financial_sanity_report(result, reject_liquidation=False)
    assert accounting_only.allowed is True
    assert any(check.label == "liquidation evidence" and check.status == "warn" for check in accounting_only.checks)


def test_backtest_financial_sanity_blocks_inconsistent_liquidation_counters() -> None:
    bad_events = build_backtest_financial_sanity_report(_backtest_result(liquidation_events=-1))
    bad_loss = build_backtest_financial_sanity_report(_backtest_result(liquidation_loss=-1.0))

    assert bad_events.allowed is False
    assert any(check.label == "liquidation events" and check.status == "block" for check in bad_events.checks)
    assert bad_loss.allowed is False
    assert any(check.label == "liquidation loss" and check.status == "block" for check in bad_loss.checks)


def test_backtest_financial_sanity_blocks_path_metric_drift() -> None:
    report = build_backtest_financial_sanity_report(
        _backtest_result(
            gross_profit=999.0,
            gross_loss=0.0,
            profit_factor=999.0,
            expectancy=999.0,
            average_trade_return=999.0,
            trade_return_stdev=999.0,
            max_consecutive_losses=0,
        )
    )

    labels = {check.label for check in report.checks if check.status == "block"}
    assert "gross profit identity" in labels
    assert "gross loss identity" in labels
    assert "profit factor identity" in labels
    assert "expectancy identity" in labels
    assert "average return identity" in labels
    assert "return stdev identity" in labels
    assert "loss streak identity" in labels


def test_backtest_financial_sanity_blocks_equity_curve_drift() -> None:
    report = build_backtest_financial_sanity_report(
        _backtest_result(
            max_drawdown=0.02,
            equity_curve=(
                {"timestamp": 1000, "equity": 1000.0, "drawdown": 0.0, "position_side": 0},
                {"timestamp": 2000, "equity": 1010.0, "drawdown": 0.0, "position_side": 0},
                {"timestamp": 3000, "equity": 1005.0, "drawdown": 0.01, "position_side": 0},
            ),
        )
    )

    labels = {check.label for check in report.checks if check.status == "block"}
    assert "equity curve drawdown identity" in labels
    assert "max drawdown identity" in labels


def test_backtest_financial_sanity_blocks_trade_level_math_omissions() -> None:
    report = build_backtest_financial_sanity_report(
        _backtest_result(
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
    )

    labels = {check.label for check in report.checks if check.status == "block"}
    assert "trade timestamp" in labels
    assert "trade side" in labels
    assert "trade notional/price" in labels
    assert "trade return" in labels


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


def _model_lab_payload_with_symbols(symbols: list[str] | None = None) -> dict[str, object]:
    accepted_symbols = list(symbols or ["AAAUSDC", "BBBUSDC"])
    outcomes = [
        {
            "symbol": symbol,
            "accepted": True,
            "rows": 500,
            "objective_scores": {"regular": 0.15},
            "selection_risk": {"regular": _accepted_selection_risk()},
            "data_coverage": _accepted_data_coverage(symbol),
            "stress_validation": _accepted_stress_validation(),
            "robustness_validation": _accepted_robustness_validation(),
        }
        for symbol in accepted_symbols
    ]
    return {
        "accepted_symbols": accepted_symbols,
        "portfolio_risk": {
            "accepted": True,
            "accepted_symbols": accepted_symbols,
            "effective_symbol_count": float(len(accepted_symbols)),
            "correlation_adjusted_effective_symbol_count": max(1.0, float(len(accepted_symbols)) - 0.25),
            "portfolio_cvar_95": 0.01,
            "portfolio_max_drawdown": 0.02,
            "deployed_weight": min(1.0, 0.20 * len(accepted_symbols)),
            "max_pairwise_correlation": 0.10,
            "max_cluster_weight": 0.40,
        },
        "outcomes": outcomes,
    }


def _accepted_data_coverage(symbol: str = "BTCUSDT") -> dict[str, object]:
    return {
        "symbol": symbol,
        "market_type": "spot",
        "interval": "1m",
        "source_scope": "binance_full_history",
        "expected_interval_ms": 60_000,
        "integrity_status": "ok",
        "integrity_warnings": [],
        "truth_basis": [
            "prices_from_timestamped_closed_candles",
            "coverage_measured_from_candle_close_time",
            "execution_results_are_simulated_not_exchange_fills",
        ],
        "full_history_requested": True,
        "full_available_history_used": True,
        "candles_available": 70080,
        "candles_used": 70080,
        "rows_used": 500,
        "available_start_utc": "2024-01-01T00:00:00Z",
        "available_end_utc": "2024-02-18T16:00:00Z",
        "used_start_utc": "2024-01-01T00:00:00Z",
        "used_end_utc": "2024-02-18T16:00:00Z",
        "used_duration_days": 48.67,
        "used_duration_years": 0.13,
        "gap_count": 0,
        "largest_gap_ms": 60_000,
        "largest_gap_intervals": 1.0,
        "coverage_ratio": 1.0,
        "notes": [],
    }


def _accepted_selection_risk() -> dict[str, object]:
    return {
        "passed": True,
        "reason": None,
        "reasons": [],
        "effective_trials": 24,
        "finite_candidate_scores": 12,
        "selected_score": 0.15,
        "trial_penalty": 0.04,
        "deflated_score": 0.11,
        "overfit_diagnostics": {
            "status": "skipped",
            "reason": "requires_selection_and_validation_scores",
            "passed": True,
        },
    }


def _accepted_stress_validation() -> dict[str, object]:
    return {
        "accepted": True,
        "objective_count": 1,
        "accepted_objectives": 1,
        "scenario_count": 4,
        "worst_realized_pnl": 2.0,
        "worst_max_drawdown": 0.01,
    }


def _accepted_robustness_validation() -> dict[str, object]:
    return {
        "accepted": True,
        "objective_count": 1,
        "accepted_objectives": 1,
        "window_count": 5,
        "accepted_windows": 5,
        "accepted_window_rate": 1.0,
        "worst_realized_pnl": 1.0,
        "worst_max_drawdown": 0.02,
        "statistical_edge_accepted": True,
        "worst_sign_test_p_value": 0.03125,
        "worst_bootstrap_lower_mean_return": 0.002,
    }


def _accepted_ai_uplift() -> dict[str, object]:
    baseline = {
        "realized_pnl": 10.0,
        "roi_pct": 0.10,
        "max_drawdown": 0.04,
        "expectancy": 0.50,
        "profit_factor": 1.45,
        "closed_trades": 12,
        "win_rate": 0.58,
        "liquidation_events": 0,
        "max_consecutive_losses": 2,
        "downside_return_risk_ratio": 1.20,
    }
    ai = {
        "realized_pnl": 13.0,
        "roi_pct": 0.13,
        "max_drawdown": 0.035,
        "expectancy": 0.70,
        "profit_factor": 1.70,
        "closed_trades": 13,
        "win_rate": 0.62,
        "liquidation_events": 0,
        "max_consecutive_losses": 1,
        "downside_return_risk_ratio": 1.45,
    }
    deltas = {key: float(ai[key]) - float(baseline[key]) for key in baseline}
    return {
        "accepted": True,
        "advisory_only": False,
        "model_name": "qwen2.5:7b",
        "model_parameters_b": 7.0,
        "baseline": baseline,
        "ai": ai,
        "deltas": deltas,
        "statistical_evidence": {
            "accepted": True,
            "reasons": [],
            "evidence_unit": "paired_trade_return_delta",
            "sample_count": 12,
            "min_sample_count": 8,
            "positive_delta_count": 10,
            "positive_delta_rate": 10 / 12,
            "min_positive_delta_rate": 0.55,
            "sign_test_p_value": 0.019287109375,
            "max_sign_test_p_value": 0.40,
            "mean_delta": 0.0025,
            "median_delta": 0.002,
            "min_mean_sample_delta": 0.0,
        },
        "reasons": [],
        "policy": {
            "min_model_parameters_b": 2.0,
            "min_paired_samples": 8,
            "min_positive_delta_rate": 0.55,
            "max_sign_test_p_value": 0.40,
            "min_mean_sample_delta": 0.0,
        },
    }


def test_model_lab_financial_sanity_accepts_consistent_portfolio_symbol_evidence() -> None:
    report = build_model_lab_financial_sanity_report(_model_lab_payload_with_symbols())

    assert report.allowed is True
    assert all(check.status != "block" for check in report.checks)


def test_model_lab_financial_sanity_blocks_missing_data_coverage_evidence() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    del payload["outcomes"][0]["data_coverage"]  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "data coverage" and check.path.endswith(".data_coverage")
        for check in report.checks
        if check.status == "block"
    )


def test_model_lab_financial_sanity_blocks_unproven_data_source_scope() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    coverage = _accepted_data_coverage("BTCUSDT")
    coverage["source_scope"] = "synthetic_fixture"
    payload["outcomes"][0]["data_coverage"] = coverage  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "data source" and check.path.endswith(".source_scope")
        for check in report.checks
        if check.status == "block"
    )


def test_model_lab_financial_sanity_blocks_missing_truth_basis() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    coverage = _accepted_data_coverage("BTCUSDT")
    coverage["truth_basis"] = ["prices_from_timestamped_closed_candles"]
    payload["outcomes"][0]["data_coverage"] = coverage  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "data truth basis" and check.path.endswith(".truth_basis")
        for check in report.checks
        if check.status == "block"
    )


def test_model_lab_financial_sanity_blocks_nonpositive_data_row_counts() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    coverage = _accepted_data_coverage("BTCUSDT")
    coverage["candles_used"] = 0
    coverage["rows_used"] = 0
    payload["outcomes"][0]["data_coverage"] = coverage  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    blocked_paths = {check.path for check in report.checks if check.status == "block"}
    assert "outcomes[0].data_coverage.candles_used" in blocked_paths
    assert "outcomes[0].data_coverage.rows_used" in blocked_paths


def test_model_lab_financial_sanity_blocks_missing_selection_risk_evidence() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    del payload["outcomes"][0]["selection_risk"]  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "selection risk" and check.path.endswith(".selection_risk")
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_failed_selection_risk_evidence() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    risk = _accepted_selection_risk()
    risk["passed"] = False
    risk["reason"] = "selection_risk_deflated_score<=0"
    risk["reasons"] = ["selection_risk_deflated_score<=0"]
    risk["deflated_score"] = -0.01
    payload["outcomes"][0]["selection_risk"] = {"regular": risk}  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "selection risk" and check.path.endswith(".deflated_score")
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_failed_selection_overfit_evidence() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    risk = _accepted_selection_risk()
    risk["overfit_diagnostics"] = {
        "status": "available",
        "passed": False,
        "probability_backtest_overfit": 0.75,
        "max_probability_backtest_overfit": 0.50,
    }
    payload["outcomes"][0]["selection_risk"] = {"regular": risk}  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "selection risk" and "PBO" in check.detail
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_accepted_stress_without_scenarios() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    payload["outcomes"][0]["stress_validation"] = {  # type: ignore[index]
        "accepted": True,
        "worst_realized_pnl": 1.0,
        "worst_max_drawdown": 0.01,
    }

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "stress validation" and check.path.endswith(".scenario_count")
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_accepted_temporal_without_statistical_edge() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    robustness = _accepted_robustness_validation()
    del robustness["statistical_edge_accepted"]
    payload["outcomes"][0]["robustness_validation"] = robustness  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "temporal robustness" and check.path.endswith(".statistical_edge_accepted")
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_inconsistent_temporal_window_counts() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    robustness = _accepted_robustness_validation()
    robustness["accepted_windows"] = 6
    robustness["window_count"] = 5
    payload["outcomes"][0]["robustness_validation"] = robustness  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "temporal robustness" and "within window count" in check.detail
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_missing_portfolio_symbol_evidence() -> None:
    payload = _model_lab_payload_with_symbols()
    del payload["portfolio_risk"]["accepted_symbols"]  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.path == "portfolio_risk.accepted_symbols" and check.status == "block"
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_mismatched_accepted_symbol_evidence() -> None:
    payload = _model_lab_payload_with_symbols()
    payload["portfolio_risk"]["accepted_symbols"] = ["AAAUSDC", "CCCUSDC"]  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "portfolio symbols" and "differ" in check.detail and check.status == "block"
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_duplicate_portfolio_symbols() -> None:
    payload = _model_lab_payload_with_symbols()
    payload["accepted_symbols"] = ["AAAUSDC", "AAAUSDC"]
    payload["portfolio_risk"]["accepted_symbols"] = ["AAAUSDC", "AAAUSDC"]  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "portfolio symbols" and "duplicate" in check.detail and check.status == "block"
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_accepted_portfolio_without_accepted_outcomes() -> None:
    payload = _model_lab_payload_with_symbols()
    payload["outcomes"] = []

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "accepted outcomes" and check.status == "block"
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_accepted_outcome_without_portfolio_risk() -> None:
    payload = _model_lab_payload_with_symbols()
    del payload["portfolio_risk"]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "portfolio risk" and check.path == "portfolio_risk" and check.status == "block"
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_accepted_outcome_with_failed_portfolio_risk() -> None:
    payload = _model_lab_payload_with_symbols()
    payload["portfolio_risk"]["accepted"] = False  # type: ignore[index]
    payload["portfolio_risk"]["reason"] = "cvar95>0.0100"  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "portfolio risk" and check.path == "portfolio_risk.accepted" and check.status == "block"
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_top_level_symbols_without_accepted_outcomes() -> None:
    payload = _model_lab_payload_with_symbols()
    payload["portfolio_risk"]["accepted"] = False  # type: ignore[index]
    payload["outcomes"] = []

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "accepted symbols" and "no accepted outcome" in check.detail and check.status == "block"
        for check in report.checks
    )


def test_model_lab_financial_sanity_accepts_complete_ai_uplift_contract() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    payload["outcomes"][0]["ai_uplift"] = _accepted_ai_uplift()  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is True
    assert not any(check.label == "AI uplift evidence" and check.status == "block" for check in report.checks)


def test_model_lab_financial_sanity_blocks_accepted_ai_uplift_missing_contract_metrics() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    uplift = _accepted_ai_uplift()
    del uplift["baseline"]["profit_factor"]  # type: ignore[index]
    del uplift["deltas"]["downside_return_risk_ratio"]  # type: ignore[index]
    payload["outcomes"][0]["ai_uplift"] = uplift  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "AI uplift evidence" and check.path.endswith(".baseline.profit_factor")
        for check in report.checks
    )
    assert any(
        check.label == "AI uplift evidence" and check.path.endswith(".deltas.downside_return_risk_ratio")
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_accepted_ai_uplift_without_model_size_evidence() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    uplift = _accepted_ai_uplift()
    uplift["model_parameters_b"] = 1.0
    payload["outcomes"][0]["ai_uplift"] = uplift  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "AI uplift evidence" and check.path.endswith(".model_parameters_b")
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_accepted_ai_uplift_without_statistical_evidence() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    uplift = _accepted_ai_uplift()
    del uplift["statistical_evidence"]
    payload["outcomes"][0]["ai_uplift"] = uplift  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "AI uplift statistical evidence"
        and check.path.endswith(".statistical_evidence")
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_accepted_ai_uplift_with_weak_sign_test() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    uplift = _accepted_ai_uplift()
    uplift["statistical_evidence"]["positive_delta_rate"] = 0.50  # type: ignore[index]
    uplift["statistical_evidence"]["sign_test_p_value"] = 0.62  # type: ignore[index]
    payload["outcomes"][0]["ai_uplift"] = uplift  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    blocked_paths = {check.path for check in report.checks if check.status == "block"}
    assert "outcomes[0].ai_uplift.statistical_evidence.positive_delta_rate" in blocked_paths
    assert "outcomes[0].ai_uplift.statistical_evidence.sign_test_p_value" in blocked_paths


def test_model_lab_financial_sanity_blocks_accepted_ai_uplift_with_inconsistent_statistics() -> None:
    payload = _model_lab_payload_with_symbols(["BTCUSDT"])
    uplift = _accepted_ai_uplift()
    statistical = uplift["statistical_evidence"]  # type: ignore[index]
    statistical["paired_sample_length_mismatch"] = True  # type: ignore[index]
    statistical["sign_test_p_value"] = 0.001  # type: ignore[index]
    payload["outcomes"][0]["ai_uplift"] = uplift  # type: ignore[index]

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    blocked_paths = {check.path for check in report.checks if check.status == "block"}
    assert (
        "outcomes[0].ai_uplift.statistical_evidence.paired_sample_length_mismatch"
        in blocked_paths
    )
    assert "outcomes[0].ai_uplift.statistical_evidence.sign_test_p_value" in blocked_paths


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
                "selection_risk": {"regular": _accepted_selection_risk()},
                "data_coverage": _accepted_data_coverage("AAAUSDC"),
                "stress_validation": _accepted_stress_validation(),
                "robustness_validation": _accepted_robustness_validation(),
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
                "selection_risk": {"regular": _accepted_selection_risk()},
                "data_coverage": _accepted_data_coverage("AAAUSDC"),
                "stress_validation": {
                    **_accepted_stress_validation(),
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
                "robustness_validation": _accepted_robustness_validation(),
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
                "selection_risk": {"conservative": _accepted_selection_risk()},
                "data_coverage": _accepted_data_coverage("BTCUSDT"),
                "stress_validation": {
                    **_accepted_stress_validation(),
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
                "robustness_validation": _accepted_robustness_validation(),
            }
        ],
    }

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "market edge downside risk" and check.status == "block"
        for check in report.checks
    )


def test_model_lab_financial_sanity_blocks_accepted_ai_uplift_tail_risk() -> None:
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
                "selection_risk": {"conservative": _accepted_selection_risk()},
                "data_coverage": _accepted_data_coverage("BTCUSDT"),
                "ai_uplift": {
                    "accepted": True,
                    "reasons": ["should_not_be_accepted"],
                    "ai": {"liquidation_events": 1},
                    "deltas": {
                        "realized_pnl": 5.0,
                        "profit_factor": -0.2,
                        "win_rate": -0.05,
                        "max_consecutive_losses": 2,
                        "downside_return_risk_ratio": -0.1,
                    },
                },
                "stress_validation": {
                    **_accepted_stress_validation(),
                    "market_edge": {
                        "accepted": True,
                        "reason": None,
                        "net_edge_pct": 0.01,
                        "min_net_edge_pct": 0.002,
                        "downside_return_risk_ratio": 0.60,
                        "min_downside_return_risk_ratio": 0.45,
                    },
                },
                "robustness_validation": {
                    **_accepted_robustness_validation(),
                    "market_edge": {
                        "accepted": True,
                        "reason": None,
                        "net_edge_pct": 0.008,
                        "min_net_edge_pct": 0.002,
                    },
                },
            }
        ],
    }

    report = build_model_lab_financial_sanity_report(payload)
    labels = {check.label for check in report.checks if check.status == "block"}

    assert report.allowed is False
    assert "AI uplift" in labels
    assert "AI uplift tail risk" in labels
    assert "AI uplift liquidation risk" in labels
