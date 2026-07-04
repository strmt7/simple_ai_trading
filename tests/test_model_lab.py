from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from simple_ai_trading.api import Candle
from simple_ai_trading.model_lab import run_model_lab
from simple_ai_trading.training_suite import TrainingSuiteRejected
from simple_ai_trading.types import RuntimeConfig, StrategyConfig


class _Client:
    market_type = "spot"

    def get_exchange_info(self):
        return {
            "symbols": [
                {"symbol": "AAAUSDC", "status": "TRADING"},
                {"symbol": "BBBUSDC", "status": "TRADING"},
                {"symbol": "LOWUSDC", "status": "TRADING"},
            ]
        }

    def get_all_tickers_24h(self):
        return [
            {"symbol": "AAAUSDC", "quoteVolume": "1000000", "count": "10000"},
            {"symbol": "BBBUSDC", "quoteVolume": "900000", "count": "9000"},
            {"symbol": "LOWUSDC", "quoteVolume": "1", "count": "1"},
        ]

    def get_all_book_tickers(self):
        return [
            {"symbol": "AAAUSDC", "bidPrice": "99.99", "askPrice": "100.01"},
            {"symbol": "BBBUSDC", "bidPrice": "49.99", "askPrice": "50.01"},
            {"symbol": "LOWUSDC", "bidPrice": "1", "askPrice": "2"},
        ]

    def get_klines(self, symbol: str, interval: str, *, limit: int = 500):
        return [
            Candle(
                open_time=index * 60_000,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0 + index * 0.01,
                volume=10.0,
                close_time=index * 60_000 + 59_000,
            )
            for index in range(limit)
        ]


class _ThreeLiquidClient(_Client):
    def get_exchange_info(self):
        return {
            "symbols": [
                {"symbol": "AAAUSDC", "status": "TRADING"},
                {"symbol": "BBBUSDC", "status": "TRADING"},
                {"symbol": "CCCUSDC", "status": "TRADING"},
            ]
        }

    def get_all_tickers_24h(self):
        return [
            {"symbol": "AAAUSDC", "quoteVolume": "1000000", "count": "10000"},
            {"symbol": "BBBUSDC", "quoteVolume": "900000", "count": "9000"},
            {"symbol": "CCCUSDC", "quoteVolume": "800000", "count": "8000"},
        ]

    def get_all_book_tickers(self):
        return [
            {"symbol": "AAAUSDC", "bidPrice": "99.99", "askPrice": "100.01"},
            {"symbol": "BBBUSDC", "bidPrice": "99.99", "askPrice": "100.01"},
            {"symbol": "CCCUSDC", "bidPrice": "99.99", "askPrice": "100.01"},
        ]


class _Stress:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted

    def asdict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "worst_realized_pnl": 10.0 if self.accepted else -5.0,
            "worst_max_drawdown": 0.01 if self.accepted else 0.20,
        }


class _Robustness:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted

    def asdict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "window_count": 4,
            "accepted_windows": 4 if self.accepted else 2,
            "accepted_window_rate": 1.0 if self.accepted else 0.5,
            "worst_realized_pnl": 8.0 if self.accepted else -3.0,
            "worst_max_drawdown": 0.02 if self.accepted else 0.18,
            "statistical_edge_accepted": self.accepted,
            "worst_sign_test_p_value": 0.125 if self.accepted else 0.6875,
            "worst_bootstrap_lower_mean_return": 0.003 if self.accepted else -0.006,
            "regime_summary": {
                "window_count": 4,
                "dominant_regime": "trend_up",
                "accepted_regime_count": 1 if self.accepted else 0,
                "concentration_warning": True,
                "by_regime": {"trend_up": {"windows": 4, "accepted_windows": 4 if self.accepted else 2}},
            },
        }


def test_run_model_lab_ranks_liquid_symbols_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    def fake_suite(candles, strategy, **kwargs):
        assert candles
        assert kwargs["objectives"] == ("regular",)
        assert kwargs["max_candidates"] == 5
        return SimpleNamespace(
            outcomes=[SimpleNamespace(objective="regular", best_score=0.12, hybrid_profile="balanced_neighbors")],
            total_rows=123,
            objectives_run=["regular"],
            summary_path=kwargs["summary_path"],
        )

    monkeypatch.setattr("simple_ai_trading.model_lab.run_training_suite", fake_suite)
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_under_stress", lambda *_a, **_k: _Stress(True))
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_temporal_robustness", lambda *_a, **_k: _Robustness(True))
    runtime = RuntimeConfig(symbols=("AAAUSDC", "BBBUSDC"), quote_asset="USDC", interval="1m")
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

    assert report.accepted_symbols == ["AAAUSDC", "BBBUSDC"]
    assert (tmp_path / "model_lab_report.json").exists()
    assert (tmp_path / "AAAUSDC" / "stress_validation.json").exists()
    assert (tmp_path / "AAAUSDC" / "temporal_robustness.json").exists()
    assert (tmp_path / "portfolio_risk.json").exists()
    assert report.portfolio_risk is not None
    assert report.portfolio_risk["accepted"] is True
    assert report.outcomes[0].hybrid_profiles["regular"] == "balanced_neighbors"
    assert report.outcomes[0].stress_validation["accepted"] is True
    assert report.outcomes[0].robustness_validation["accepted"] is True
    assert report.outcomes[0].regime_validation["dominant_regime"] == "trend_up"


def test_run_model_lab_preserves_rejected_training_row_count(tmp_path: Path, monkeypatch) -> None:
    def rejected_suite(candles, strategy, **kwargs):
        assert candles
        raise TrainingSuiteRejected(
            "all candidates rejected",
            row_count=77,
            diagnostics={"top_candidates": [{"validation_score": None, "full_sample_score": None}]},
        )

    monkeypatch.setattr("simple_ai_trading.model_lab.run_training_suite", rejected_suite)
    runtime = RuntimeConfig(symbols=("AAAUSDC",), quote_asset="USDC", interval="1m")
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
    runtime = RuntimeConfig(symbols=("AAAUSDC",), quote_asset="USDC", interval="1m")
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
    runtime = RuntimeConfig(symbols=("AAAUSDC",), quote_asset="USDC", interval="1m")
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


def test_run_model_lab_rejects_individual_passes_when_portfolio_gate_fails(tmp_path: Path, monkeypatch) -> None:
    def fake_suite(candles, strategy, **kwargs):
        return SimpleNamespace(
            outcomes=[SimpleNamespace(objective="regular", best_score=0.12, hybrid_profile="base_only")],
            total_rows=len(candles),
            objectives_run=["regular"],
            summary_path=kwargs["summary_path"],
        )

    monkeypatch.setattr("simple_ai_trading.model_lab.run_training_suite", fake_suite)
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_under_stress", lambda *_a, **_k: _Stress(True))
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_temporal_robustness", lambda *_a, **_k: _Robustness(True))
    runtime = RuntimeConfig(symbols=("AAAUSDC", "BBBUSDC", "CCCUSDC"), quote_asset="USDC", interval="1m")
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
    assert "cluster_weight>" in str(report.portfolio_risk["reason"])
    assert {outcome.error for outcome in report.outcomes} == {"portfolio_risk_failed"}
    assert all(outcome.diagnostics and "portfolio_risk_reason" in outcome.diagnostics for outcome in report.outcomes)
