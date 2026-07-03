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


class _Stress:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted

    def asdict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "worst_realized_pnl": 10.0 if self.accepted else -5.0,
            "worst_max_drawdown": 0.01 if self.accepted else 0.20,
        }


def test_run_model_lab_ranks_liquid_symbols_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    def fake_suite(candles, strategy, **kwargs):
        assert candles
        assert kwargs["objectives"] == ("regular",)
        return SimpleNamespace(
            outcomes=[SimpleNamespace(objective="regular", best_score=0.12, hybrid_profile="balanced_neighbors")],
            total_rows=123,
            objectives_run=["regular"],
            summary_path=kwargs["summary_path"],
        )

    monkeypatch.setattr("simple_ai_trading.model_lab.run_training_suite", fake_suite)
    monkeypatch.setattr("simple_ai_trading.model_lab.validate_suite_under_stress", lambda *_a, **_k: _Stress(True))
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
    )

    assert report.accepted_symbols == ["AAAUSDC", "BBBUSDC"]
    assert (tmp_path / "model_lab_report.json").exists()
    assert (tmp_path / "AAAUSDC" / "stress_validation.json").exists()
    assert report.outcomes[0].hybrid_profiles["regular"] == "balanced_neighbors"
    assert report.outcomes[0].stress_validation["accepted"] is True


def test_run_model_lab_preserves_rejected_training_row_count(tmp_path: Path, monkeypatch) -> None:
    def rejected_suite(candles, strategy, **kwargs):
        assert candles
        raise TrainingSuiteRejected("all candidates rejected", row_count=77)

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
