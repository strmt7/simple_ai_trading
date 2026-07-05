from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from simple_ai_trading.api import Candle
from simple_ai_trading.model import TrainedModel, load_model, serialize_model
from simple_ai_trading import model_lab
from simple_ai_trading.data_coverage import describe_candle_coverage
from simple_ai_trading.model_lab import run_model_lab
from simple_ai_trading.training_suite import TrainingSuiteRejected
from simple_ai_trading.types import RuntimeConfig, StrategyConfig


class _Client:
    market_type = "spot"

    def get_exchange_info(self):
        return {
            "symbols": [
                {"symbol": "BTCUSDC", "status": "TRADING"},
                {"symbol": "ETHUSDC", "status": "TRADING"},
                {"symbol": "LOWUSDC", "status": "TRADING"},
            ]
        }

    def get_all_tickers_24h(self):
        return [
            {"symbol": "BTCUSDC", "quoteVolume": "1000000", "count": "10000"},
            {"symbol": "ETHUSDC", "quoteVolume": "900000", "count": "9000"},
            {"symbol": "LOWUSDC", "quoteVolume": "1", "count": "1"},
        ]

    def get_all_book_tickers(self):
        return [
            {"symbol": "BTCUSDC", "bidPrice": "99.99", "askPrice": "100.01"},
            {"symbol": "ETHUSDC", "bidPrice": "49.99", "askPrice": "50.01"},
            {"symbol": "LOWUSDC", "bidPrice": "1", "askPrice": "2"},
        ]

    def get_klines(self, symbol: str, interval: str, *, limit: int = 500):
        rows: list[Candle] = []
        for index in range(limit):
            if symbol == "BTCUSDC":
                close = 100.0 + index * 0.01
            else:
                close = 50.0 + index * 0.006 + (((index * 7) % 17) - 8) * 0.002
            rows.append(
                Candle(
                    open_time=index * 60_000,
                    open=close * 0.999,
                    high=close * 1.002,
                    low=close * 0.998,
                    close=close,
                    volume=10.0,
                    close_time=index * 60_000 + 59_000,
                )
            )
        return rows


class _ThreeLiquidClient(_Client):
    def get_exchange_info(self):
        return {
            "symbols": [
                {"symbol": "BTCUSDC", "status": "TRADING"},
                {"symbol": "ETHUSDC", "status": "TRADING"},
                {"symbol": "SOLUSDC", "status": "TRADING"},
            ]
        }

    def get_all_tickers_24h(self):
        return [
            {"symbol": "BTCUSDC", "quoteVolume": "1000000", "count": "10000"},
            {"symbol": "ETHUSDC", "quoteVolume": "900000", "count": "9000"},
            {"symbol": "SOLUSDC", "quoteVolume": "800000", "count": "8000"},
        ]

    def get_all_book_tickers(self):
        return [
            {"symbol": "BTCUSDC", "bidPrice": "99.99", "askPrice": "100.01"},
            {"symbol": "ETHUSDC", "bidPrice": "99.99", "askPrice": "100.01"},
            {"symbol": "SOLUSDC", "bidPrice": "99.99", "askPrice": "100.01"},
        ]


class _PagedHistoryClient(_Client):
    def __init__(self) -> None:
        self.kline_calls: list[int | None] = []

    def get_klines(self, symbol: str, interval: str, *, limit: int = 500, end_time: int | None = None):
        self.kline_calls.append(end_time)
        if end_time is None:
            indexes = [2, 3]
        else:
            indexes = [0, 1]
        return [
            Candle(
                open_time=index * 60_000,
                open=100.0,
                high=105.0,
                low=99.0,
                close=100.0 + index,
                volume=10.0,
                close_time=index * 60_000 + 59_000,
            )
            for index in indexes[:limit]
        ]


class _Stress:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted
        self.worst_realized_pnl = 10.0 if self.accepted else -5.0
        self.worst_max_drawdown = 0.01 if self.accepted else 0.20

    def asdict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "worst_realized_pnl": self.worst_realized_pnl,
            "worst_max_drawdown": self.worst_max_drawdown,
        }


class _Robustness:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted
        self.window_count = 4
        self.accepted_windows = 4 if self.accepted else 2
        self.accepted_window_rate = 1.0 if self.accepted else 0.5
        self.worst_realized_pnl = 8.0 if self.accepted else -3.0
        self.worst_max_drawdown = 0.02 if self.accepted else 0.18
        self.statistical_edge_accepted = accepted
        self.worst_sign_test_p_value = 0.125 if self.accepted else 0.6875
        self.worst_bootstrap_lower_mean_return = 0.003 if self.accepted else -0.006

    def asdict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "window_count": self.window_count,
            "accepted_windows": self.accepted_windows,
            "accepted_window_rate": self.accepted_window_rate,
            "worst_realized_pnl": self.worst_realized_pnl,
            "worst_max_drawdown": self.worst_max_drawdown,
            "statistical_edge_accepted": self.statistical_edge_accepted,
            "worst_sign_test_p_value": self.worst_sign_test_p_value,
            "worst_bootstrap_lower_mean_return": self.worst_bootstrap_lower_mean_return,
            "regime_summary": {
                "window_count": 4,
                "dominant_regime": "trend_up",
                "accepted_regime_count": 1 if self.accepted else 0,
                "concentration_warning": True,
                "by_regime": {"trend_up": {"windows": 4, "accepted_windows": 4 if self.accepted else 2}},
            },
        }


class _WeakAcceptedStress(_Stress):
    def __init__(self) -> None:
        super().__init__(True)
        self.worst_realized_pnl = -1.0


class _WeakAcceptedRobustness(_Robustness):
    def __init__(self) -> None:
        super().__init__(True)
        self.worst_realized_pnl = -0.5


class _ObjectiveReport:
    def __init__(self, *, accepted: bool = True) -> None:
        self.accepted = accepted
        self.scenario_count = 2
        self.accepted_objectives = 1 if accepted else 0
        self.worst_realized_pnl = 4.0 if accepted else -2.0
        self.worst_max_drawdown = 0.02 if accepted else 0.12
        self.window_count = 3
        self.accepted_windows = 3 if accepted else 1
        self.accepted_window_rate = 1.0 if accepted else 1 / 3
        self.statistical_edge_accepted = accepted

    def asdict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "objectives": [
                {"objective": "regular", "accepted": self.accepted, "worst_realized_pnl": self.worst_realized_pnl}
            ],
        }


def test_model_lab_execution_stamp_helper_handles_edges(tmp_path: Path) -> None:
    assert model_lab._objective_report({"objectives": [{"objective": "regular", "accepted": True}]}, "regular") == {
        "objective": "regular",
        "accepted": True,
    }
    assert model_lab._objective_report({"objectives": [{"objective": "regular"}]}, "aggressive") is None

    valid_path = tmp_path / "model_regular.json"
    serialize_model(
        TrainedModel(
            weights=[0.0],
            bias=0.0,
            feature_dim=1,
            epochs=1,
            feature_means=[0.0],
            feature_stds=[1.0],
        ),
        valid_path,
    )
    invalid_path = tmp_path / "bad.json"
    invalid_path.write_text("{}", encoding="utf-8")
    suite = SimpleNamespace(
        outcomes=[
            SimpleNamespace(objective="", model_path=valid_path),
            SimpleNamespace(objective="regular", model_path=tmp_path / "missing.json"),
            SimpleNamespace(objective="regular", model_path=invalid_path),
            SimpleNamespace(objective="regular", model_path=valid_path),
        ]
    )
    liquidity = model_lab.MarketEligibility("BTCUSDC", True, "TRADING", 1_000_000.0, 10_000, 2.0, 0.95, ())
    candles = [
        Candle(
            open_time=index * 60_000,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
            close_time=index * 60_000 + 59_000,
        )
        for index in range(5)
    ]
    data_coverage = describe_candle_coverage(
        symbol="BTCUSDC",
        market_type="spot",
        interval="1m",
        available_candles=candles,
        used_candles=candles,
        rows_used=5,
        source_scope="binance_full_history",
    )

    model_lab._stamp_model_execution_validation(
        suite,
        symbol="BTCUSDC",
        market_type="spot",
        interval="1m",
        liquidity=liquidity,
        stress_profile=object(),
        stress_report=_ObjectiveReport(accepted=True),
        stress_report_path=tmp_path / "stress.json",
        robustness_report=_ObjectiveReport(accepted=True),
        robustness_report_path=tmp_path / "robustness.json",
        portfolio_report=SimpleNamespace(
            accepted=True,
            reason=None,
            accepted_symbols=["BTCUSDC"],
            effective_symbol_count=1.0,
            correlation_adjusted_effective_symbol_count=1.0,
            portfolio_cvar_95=0.01,
            portfolio_max_drawdown=0.02,
            max_pairwise_correlation=0.0,
            max_cluster_weight=0.5,
        ),
        portfolio_report_path=tmp_path / "portfolio.json",
        data_coverage=data_coverage,
    )

    stamped = load_model(valid_path, expected_feature_dim=1)
    assert stamped.execution_validation["passed"] is True
    assert stamped.execution_validation["symbol_execution_profile"] is None
    assert stamped.execution_validation["stress"]["objective"]["objective"] == "regular"
    assert stamped.execution_validation["portfolio"]["accepted"] is True


def test_run_model_lab_ranks_liquid_symbols_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    def fake_suite(candles, strategy, **kwargs):
        assert candles
        assert kwargs["objectives"] == ("regular",)
        assert kwargs["max_candidates"] == 5
        model_path = kwargs["output_dir"] / "model_regular.json"
        serialize_model(
            TrainedModel(
                weights=[0.0],
                bias=0.0,
                feature_dim=1,
                epochs=1,
                feature_means=[0.0],
                feature_stds=[1.0],
                selection_risk={
                    "passed": True,
                    "effective_trials": 24,
                    "selected_score": 0.12,
                    "trial_penalty": 0.01,
                    "deflated_score": 0.11,
                },
            ),
            model_path,
        )
        return SimpleNamespace(
            outcomes=[SimpleNamespace(
                objective="regular",
                model_path=model_path,
                best_score=0.12,
                hybrid_profile="balanced_neighbors",
                selection_risk={
                    "passed": True,
                    "effective_trials": 24,
                    "selected_score": 0.12,
                    "trial_penalty": 0.01,
                    "deflated_score": 0.11,
                },
                hybrid_ablation=[{"removed_expert_kind": "lorentzian_knn", "delta_vs_best": -0.03}],
                feature_ablation=[{"removed_group": "technical_confluence", "delta_vs_selected": -0.04}],
                meta_label_report={"status": "trained", "take_precision": 0.75},
            )],
            total_rows=123,
            objectives_run=["regular"],
            summary_path=kwargs["summary_path"],
        )

    monkeypatch.setattr("simple_ai_trading.model_lab.run_training_suite", fake_suite)
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_under_stress", lambda *_a, **_k: _Stress(True))
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_temporal_robustness", lambda *_a, **_k: _Robustness(True))
    runtime = RuntimeConfig(symbols=("BTCUSDC", "ETHUSDC"), quote_asset="USDC", interval="1m")
    strategy = StrategyConfig(
        min_quote_volume_usdc=1000.0,
        min_trade_count_24h=100,
        max_spread_bps=10.0,
        min_liquidity_score=0.1,
        min_diversified_assets=2,
    )
    report = run_model_lab(
        _Client(),
        runtime,
        strategy,
        objectives=("regular",),
        output_dir=tmp_path,
        starting_cash=1000.0,
        max_symbols=2,
        limit=120,
        compute_backend="cpu",
        max_candidates=5,
    )

    assert report.accepted_symbols == ["BTCUSDC", "ETHUSDC"]
    assert (tmp_path / "model_lab_report.json").exists()
    assert (tmp_path / "BTCUSDC" / "stress_validation.json").exists()
    assert (tmp_path / "BTCUSDC" / "temporal_robustness.json").exists()
    assert (tmp_path / "portfolio_risk.json").exists()
    assert report.portfolio_risk is not None
    assert report.portfolio_risk["accepted"] is True
    assert report.outcomes[0].hybrid_profiles["regular"] == "balanced_neighbors"
    assert report.outcomes[0].selection_risk["regular"]["passed"] is True
    assert report.outcomes[0].hybrid_ablation["regular"][0]["removed_expert_kind"] == "lorentzian_knn"
    assert report.outcomes[0].feature_ablation["regular"][0]["removed_group"] == "technical_confluence"
    assert report.outcomes[0].meta_label_validation["regular"]["status"] == "trained"
    assert report.outcomes[0].stress_validation["accepted"] is True
    assert report.outcomes[0].robustness_validation["accepted"] is True
    assert report.outcomes[0].regime_validation["dominant_regime"] == "trend_up"
    assert report.outcomes[0].data_coverage is not None
    assert report.outcomes[0].data_coverage["symbol"] == "BTCUSDC"
    assert report.outcomes[0].data_coverage["interval"] == "1m"
    assert report.outcomes[0].data_coverage["candles_used"] == 120
    assert "used_duration_years" in report.outcomes[0].data_coverage
    stamped = load_model(tmp_path / "BTCUSDC" / "model_regular.json", expected_feature_dim=1)
    assert stamped.execution_validation["passed"] is True
    assert stamped.execution_validation["symbol"] == "BTCUSDC"
    assert stamped.execution_validation["stress"]["accepted"] is True
    assert stamped.execution_validation["temporal_robustness"]["accepted"] is True


def test_run_model_lab_full_history_paginates_and_labels_data_scope(tmp_path: Path, monkeypatch) -> None:
    def fake_suite(candles, strategy, **kwargs):
        return SimpleNamespace(
            outcomes=[SimpleNamespace(objective="regular", best_score=0.12, hybrid_profile="base_only")],
            total_rows=len(candles),
            objectives_run=["regular"],
            summary_path=kwargs["summary_path"],
        )

    client = _PagedHistoryClient()
    monkeypatch.setattr("simple_ai_trading.model_lab.run_training_suite", fake_suite)
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_under_stress", lambda *_a, **_k: _Stress(True))
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_temporal_robustness", lambda *_a, **_k: _Robustness(True))

    report = run_model_lab(
        client,
        RuntimeConfig(symbols=("BTCUSDC",), quote_asset="USDC", interval="1m"),
        StrategyConfig(
            min_quote_volume_usdc=1000.0,
            min_trade_count_24h=100,
            max_spread_bps=10.0,
            min_liquidity_score=0.1,
            min_diversified_assets=1,
        ),
        objectives=("regular",),
        output_dir=tmp_path,
        starting_cash=1000.0,
        max_symbols=1,
        limit=2,
        compute_backend="cpu",
        full_history=True,
    )

    assert len(client.kline_calls) == 2
    assert report.outcomes[0].data_coverage is not None
    assert report.outcomes[0].data_coverage["source_scope"] == "binance_full_history"
    assert report.outcomes[0].data_coverage["candles_used"] == 4
    assert report.outcomes[0].data_coverage["full_available_history_used"] is True


def test_run_model_lab_preserves_rejected_training_row_count(tmp_path: Path, monkeypatch) -> None:
    def rejected_suite(candles, strategy, **kwargs):
        assert candles
        raise TrainingSuiteRejected(
            "all candidates rejected",
            row_count=77,
            diagnostics={"top_candidates": [{"validation_score": None, "full_sample_score": None}]},
        )

    monkeypatch.setattr("simple_ai_trading.model_lab.run_training_suite", rejected_suite)
    runtime = RuntimeConfig(symbols=("BTCUSDC",), quote_asset="USDC", interval="1m")
    strategy = StrategyConfig(
        min_quote_volume_usdc=1000.0,
        min_trade_count_24h=100,
        max_spread_bps=10.0,
        min_liquidity_score=0.1,
        min_diversified_assets=1,
    )

    report = run_model_lab(
        _Client(),
        runtime,
        strategy,
        objectives=("regular",),
        output_dir=tmp_path,
        starting_cash=1000.0,
        max_symbols=1,
        limit=120,
        compute_backend="cpu",
    )

    assert report.outcomes[0].accepted is False
    assert report.outcomes[0].rows == 77
    assert report.outcomes[0].error == "all candidates rejected"
    assert report.outcomes[0].diagnostics == {"top_candidates": [{"validation_score": None, "full_sample_score": None}]}


def test_run_model_lab_rejects_positive_suite_when_stress_fails(tmp_path: Path, monkeypatch) -> None:
    def fake_suite(candles, strategy, **kwargs):
        return SimpleNamespace(
            outcomes=[SimpleNamespace(objective="regular", best_score=0.12, hybrid_profile="base_only")],
            total_rows=123,
            objectives_run=["regular"],
            summary_path=kwargs["summary_path"],
        )

    monkeypatch.setattr("simple_ai_trading.model_lab.run_training_suite", fake_suite)
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_under_stress", lambda *_a, **_k: _Stress(False))
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_temporal_robustness", lambda *_a, **_k: _Robustness(True))
    runtime = RuntimeConfig(symbols=("BTCUSDC",), quote_asset="USDC", interval="1m")
    strategy = StrategyConfig(
        min_quote_volume_usdc=1000.0,
        min_trade_count_24h=100,
        max_spread_bps=10.0,
        min_liquidity_score=0.1,
        min_diversified_assets=1,
    )

    report = run_model_lab(
        _Client(),
        runtime,
        strategy,
        objectives=("regular",),
        output_dir=tmp_path,
        starting_cash=1000.0,
        max_symbols=1,
        limit=120,
        compute_backend="cpu",
    )

    assert report.accepted_symbols == []
    assert report.outcomes[0].accepted is False
    assert report.outcomes[0].error == "stress_validation_failed"


def test_run_model_lab_rejects_positive_suite_when_selection_risk_fails(tmp_path: Path, monkeypatch) -> None:
    def fake_suite(candles, strategy, **kwargs):
        return SimpleNamespace(
            outcomes=[SimpleNamespace(
                objective="regular",
                best_score=0.12,
                hybrid_profile="base_only",
                selection_risk={
                    "passed": False,
                    "reason": "selection_risk_deflated_score<=0",
                    "effective_trials": 900,
                    "selected_score": 0.12,
                    "trial_penalty": 0.14,
                    "deflated_score": -0.02,
                },
            )],
            total_rows=123,
            objectives_run=["regular"],
            summary_path=kwargs["summary_path"],
        )

    monkeypatch.setattr("simple_ai_trading.model_lab.run_training_suite", fake_suite)
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_under_stress", lambda *_a, **_k: _Stress(True))
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_temporal_robustness", lambda *_a, **_k: _Robustness(True))
    runtime = RuntimeConfig(symbols=("BTCUSDC",), quote_asset="USDC", interval="1m")
    strategy = StrategyConfig(
        min_quote_volume_usdc=1000.0,
        min_trade_count_24h=100,
        max_spread_bps=10.0,
        min_liquidity_score=0.1,
        min_diversified_assets=1,
    )

    report = run_model_lab(
        _Client(),
        runtime,
        strategy,
        objectives=("regular",),
        output_dir=tmp_path,
        starting_cash=1000.0,
        max_symbols=1,
        limit=120,
        compute_backend="cpu",
    )

    assert report.accepted_symbols == []
    assert report.outcomes[0].accepted is False
    assert report.outcomes[0].error == "selection_risk_failed"
    assert report.outcomes[0].selection_risk["regular"]["passed"] is False


def test_run_model_lab_rejects_positive_suite_when_temporal_robustness_fails(tmp_path: Path, monkeypatch) -> None:
    def fake_suite(candles, strategy, **kwargs):
        return SimpleNamespace(
            outcomes=[SimpleNamespace(objective="regular", best_score=0.12, hybrid_profile="base_only")],
            total_rows=123,
            objectives_run=["regular"],
            summary_path=kwargs["summary_path"],
        )

    monkeypatch.setattr("simple_ai_trading.model_lab.run_training_suite", fake_suite)
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_under_stress", lambda *_a, **_k: _Stress(True))
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_temporal_robustness", lambda *_a, **_k: _Robustness(False))
    runtime = RuntimeConfig(symbols=("BTCUSDC",), quote_asset="USDC", interval="1m")
    strategy = StrategyConfig(
        min_quote_volume_usdc=1000.0,
        min_trade_count_24h=100,
        max_spread_bps=10.0,
        min_liquidity_score=0.1,
        min_diversified_assets=1,
    )

    report = run_model_lab(
        _Client(),
        runtime,
        strategy,
        objectives=("regular",),
        output_dir=tmp_path,
        starting_cash=1000.0,
        max_symbols=1,
        limit=120,
        compute_backend="cpu",
    )

    assert report.accepted_symbols == []
    assert report.outcomes[0].accepted is False
    assert report.outcomes[0].error == "temporal_robustness_failed"
    assert report.outcomes[0].robustness_validation["accepted"] is False


def test_run_model_lab_blocks_repeated_symbol_losses_without_recovery_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_suite(candles, strategy, **kwargs):
        model_path = kwargs["output_dir"] / "model_regular.json"
        serialize_model(
            TrainedModel(
                weights=[0.0],
                bias=0.0,
                feature_dim=1,
                epochs=1,
                feature_means=[0.0],
                feature_stds=[1.0],
            ),
            model_path,
        )
        return SimpleNamespace(
            outcomes=[SimpleNamespace(objective="regular", model_path=model_path, best_score=0.12, hybrid_profile="base_only")],
            total_rows=123,
            objectives_run=["regular"],
            summary_path=kwargs["summary_path"],
        )

    feedback_path = tmp_path / "learning_feedback.json"
    feedback_path.write_text(
        json.dumps({
            "generated_at_ms": 123,
            "lookback_trades": 100,
            "closed_trades": 4,
            "wins": 1,
            "losses": 3,
            "net_realized_pnl": -7.0,
            "win_rate": 0.25,
            "max_consecutive_losses": 3,
            "worst_trade_pnl": -4.0,
            "recurring_loss_reasons": {"auto-stop-loss": 3},
            "loss_by_symbol": {"BTCUSDC": 3},
            "loss_by_side": {"LONG": 3},
            "recommendations": [
                "trigger_cooldown_and_replay_recent_loss_streak_before_new_promotion",
                "review_symbol_specific_edge:BTCUSDC",
            ],
            "promotion_safe": False,
            "notes": [],
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr("simple_ai_trading.model_lab.run_training_suite", fake_suite)
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_under_stress", lambda *_a, **_k: _WeakAcceptedStress())
    monkeypatch.setattr(
        "simple_ai_trading.model_lab.validate_suite_temporal_robustness",
        lambda *_a, **_k: _WeakAcceptedRobustness(),
    )
    runtime = RuntimeConfig(symbols=("BTCUSDC",), quote_asset="USDC", interval="1m")
    strategy = StrategyConfig(
        min_quote_volume_usdc=1000.0,
        min_trade_count_24h=100,
        max_spread_bps=10.0,
        min_liquidity_score=0.1,
        min_diversified_assets=1,
    )

    report = run_model_lab(
        _Client(),
        runtime,
        strategy,
        objectives=("regular",),
        output_dir=tmp_path,
        starting_cash=1000.0,
        max_symbols=1,
        limit=120,
        compute_backend="cpu",
        learning_feedback_path=feedback_path,
    )

    assert report.accepted_symbols == []
    assert report.learning_feedback is not None
    assert report.learning_feedback["source_path"] == str(feedback_path)
    assert report.outcomes[0].accepted is False
    assert report.outcomes[0].error == "learning_feedback_failed"
    assert report.outcomes[0].learning_feedback is not None
    assert report.outcomes[0].learning_feedback["blocks_promotion"] is True
    assert report.outcomes[0].diagnostics["learning_feedback_symbol_loss_count"] == 3
    stamped = load_model(tmp_path / "BTCUSDC" / "model_regular.json", expected_feature_dim=1)
    assert stamped.execution_validation["passed"] is False
    assert stamped.execution_validation["learning_feedback"]["blocks_promotion"] is True


def test_run_model_lab_rejects_individual_passes_when_portfolio_gate_fails(tmp_path: Path, monkeypatch) -> None:
    def fake_suite(candles, strategy, **kwargs):
        model_path = kwargs["output_dir"] / "model_regular.json"
        serialize_model(
            TrainedModel(
                weights=[0.0],
                bias=0.0,
                feature_dim=1,
                epochs=1,
                feature_means=[0.0],
                feature_stds=[1.0],
                selection_risk={
                    "passed": True,
                    "effective_trials": 24,
                    "selected_score": 0.12,
                    "trial_penalty": 0.01,
                    "deflated_score": 0.11,
                },
            ),
            model_path,
        )
        return SimpleNamespace(
            outcomes=[SimpleNamespace(
                objective="regular",
                model_path=model_path,
                best_score=0.12,
                hybrid_profile="base_only",
            )],
            total_rows=len(candles),
            objectives_run=["regular"],
            summary_path=kwargs["summary_path"],
        )

    monkeypatch.setattr("simple_ai_trading.model_lab.run_training_suite", fake_suite)
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_under_stress", lambda *_a, **_k: _Stress(True))
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_temporal_robustness", lambda *_a, **_k: _Robustness(True))
    runtime = RuntimeConfig(symbols=("BTCUSDC", "ETHUSDC", "SOLUSDC"), quote_asset="USDC", interval="1m")
    strategy = StrategyConfig(
        min_quote_volume_usdc=1000.0,
        min_trade_count_24h=100,
        max_spread_bps=10.0,
        min_liquidity_score=0.1,
        min_diversified_assets=3,
        max_asset_allocation_pct=0.20,
    )

    report = run_model_lab(
        _ThreeLiquidClient(),
        runtime,
        strategy,
        objectives=("regular",),
        output_dir=tmp_path,
        starting_cash=1000.0,
        max_symbols=3,
        limit=120,
        compute_backend="cpu",
    )

    assert report.accepted_symbols == []
    assert report.portfolio_risk is not None
    assert report.portfolio_risk["accepted"] is False
    assert "corr_adjusted_effective_symbols<" in str(report.portfolio_risk["reason"])
    assert report.portfolio_risk["correlation_adjusted_effective_symbol_count"] < 2.25
    assert {outcome.error for outcome in report.outcomes} == {"portfolio_risk_failed"}
    assert all(outcome.diagnostics and "portfolio_risk_reason" in outcome.diagnostics for outcome in report.outcomes)
    stamped = load_model(tmp_path / "BTCUSDC" / "model_regular.json", expected_feature_dim=1)
    assert stamped.execution_validation["passed"] is False
    assert stamped.execution_validation["stress"]["accepted"] is True
    assert stamped.execution_validation["temporal_robustness"]["accepted"] is True
    assert stamped.execution_validation["portfolio"]["accepted"] is False
    assert "corr_adjusted_effective_symbols<" in str(stamped.execution_validation["portfolio"]["reason"])
