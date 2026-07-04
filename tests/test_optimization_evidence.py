from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading.api import Candle
from simple_ai_trading.assets import DEFAULT_REGULAR_LEVERAGE
from simple_ai_trading.backtest import BacktestResult
from simple_ai_trading.features import ModelRow
from simple_ai_trading import optimization_evidence as oe
from simple_ai_trading.market_store import MarketDataStore
from simple_ai_trading.model import TrainedModel
from simple_ai_trading.objective import get_objective
from simple_ai_trading.types import StrategyConfig


def _candle(index: int, *, interval_ms: int = 1000, close: float = 100.0) -> Candle:
    open_time = index * interval_ms
    return Candle(
        open_time=open_time,
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=10.0,
        close_time=open_time + interval_ms - 1,
        quote_volume=close * 10.0,
        trade_count=100,
    )


class _SelectionClient:
    def get_exchange_info(self) -> dict[str, object]:
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                }
            ]
        }

    def get_all_tickers_24h(self) -> list[dict[str, object]]:
        return [
            {
                "symbol": "BTCUSDT",
                "quoteVolume": "2500000000",
                "count": "1200000",
                "lastPrice": "60000",
                "weightedAvgPrice": "59800",
                "highPrice": "61200",
                "lowPrice": "58500",
            }
        ]

    def get_all_book_tickers(self) -> list[dict[str, object]]:
        return [
            {
                "symbol": "BTCUSDT",
                "bidPrice": "60000.00",
                "bidQty": "50",
                "askPrice": "60001.00",
                "askQty": "45",
            }
        ]


class _LowLiquidityClient(_SelectionClient):
    def get_all_tickers_24h(self) -> list[dict[str, object]]:
        return [
            {
                "symbol": "BTCUSDT",
                "quoteVolume": "1000",
                "count": "12",
                "lastPrice": "60000",
                "weightedAvgPrice": "59800",
                "highPrice": "61200",
                "lowPrice": "58500",
            }
        ]

    def get_all_book_tickers(self) -> list[dict[str, object]]:
        return [
            {
                "symbol": "BTCUSDT",
                "bidPrice": "60000.00",
                "bidQty": "1",
                "askPrice": "60150.00",
                "askQty": "1",
            }
        ]


def test_market_data_health_accepts_verified_contiguous_archive(tmp_path: Path) -> None:
    db_path = tmp_path / "market.sqlite"
    archive_url = "https://data.binance.vision/data/spot/daily/klines/ETHUSDT/1s/ETHUSDT-1s-2026-01-01.zip"
    with MarketDataStore(db_path) as store:
        store.upsert_candles("ETHUSDT", "spot", "1s", [_candle(0), _candle(1), _candle(2)], source="binance_archive")
        store.begin_archive_file(
            url=archive_url,
            symbol="ETHUSDT",
            market_type="spot",
            interval="1s",
            period="2026-01-01",
            started_at_ms=1,
        )
        store.complete_archive_file(
            url=archive_url,
            status="complete",
            rows_inserted=3,
            bytes_downloaded=1234,
            sha256="abc",
            checksum_sha256="def",
            checksum_status="verified",
            completed_at_ms=2,
        )

    health = oe.market_data_health_for_symbol(
        db_path=db_path,
        symbol="ethusdt",
        market_type="spot",
        interval="1s",
        min_rows=3,
        min_coverage_ratio=1.0,
        require_verified_checksum=True,
    )

    assert health["status"] == "ok"
    assert health["rows"] == 3
    assert health["gap_count"] == 0
    assert health["coverage_ratio"] == 1.0
    assert health["archive_status_counts"] == {"complete": 1}
    assert health["checksum_status_counts"] == {"verified": 1}
    assert health["reasons"] == []


def test_fetch_full_history_refuses_network_backfill_when_prefill_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_sync(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("network backfill should not run")

    monkeypatch.setattr(oe, "sync_market_data", fail_sync)

    with pytest.raises(ValueError, match="no prefilled candles"):
        oe.fetch_full_history(
            _SelectionClient(),
            "BTCUSDT",
            "1s",
            db_path=tmp_path / "empty.sqlite",
            market_type="spot",
            allow_network_backfill=False,
        )

    assert called is False


def test_train_round_model_uses_selection_slice_not_holdout_for_threshold_and_inversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        ModelRow(
            timestamp=index * 60_000,
            close=100.0 + index,
            features=(1.0,),
            label=1,
        )
        for index in range(100)
    ]
    model = TrainedModel(
        weights=[-1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    observed: dict[str, object] = {"run_lengths": []}
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg: list(rows))

    def fake_train_advanced(train_rows, _feature_cfg, **kwargs):
        observed["train_rows"] = len(train_rows)
        observed["train_validation_rows"] = len(kwargs["validation_rows"])
        return model, SimpleNamespace(row_count=len(train_rows), positive_rate=1.0)

    monkeypatch.setattr(oe, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(
        oe,
        "calibrate_probability_temperature",
        lambda calibration_rows, _model: SimpleNamespace(status="fail", rows=len(calibration_rows)),
    )

    def fake_threshold(selection_rows, _model, _strategy, **_kwargs):
        observed["threshold_rows"] = len(selection_rows)
        return SimpleNamespace(
            accepted=True,
            threshold=0.77,
            score=2.0,
            realized_pnl=12.0,
            closed_trades=7,
        )

    monkeypatch.setattr(oe, "calibrate_threshold_for_backtest", fake_threshold)

    def result_for(realized_pnl: float) -> BacktestResult:
        return BacktestResult(
            starting_cash=1000.0,
            ending_cash=1000.0 + realized_pnl,
            realized_pnl=realized_pnl,
            win_rate=0.75,
            trades=8,
            max_drawdown=0.01,
            closed_trades=8,
            gross_exposure=100.0,
            total_fees=1.0,
            stopped_by_drawdown=False,
            max_exposure=100.0,
            trades_per_day_cap_hit=0,
            buy_hold_pnl=1.0,
            edge_vs_buy_hold=realized_pnl - 1.0,
            profit_factor=1.5,
            expectancy=realized_pnl / 8.0,
            max_consecutive_losses=1,
        )

    def fake_run_backtest(selection_rows, candidate_model, *_args, **_kwargs):
        observed["run_lengths"].append(len(selection_rows))  # type: ignore[index]
        return result_for(20.0 if candidate_model.probability_inverted else 10.0)

    monkeypatch.setattr(oe, "run_backtest", fake_run_backtest)

    selected_model, _report, _all_rows, holdout_rows = oe.train_round_model(
        [_candle(index) for index in range(100)],
        StrategyConfig(),
        get_objective("conservative"),
        market_type="futures",
        starting_cash=1000.0,
        compute_backend="auto",
        batch_size=1024,
    )

    assert observed["train_rows"] == 60
    assert observed["train_validation_rows"] == 15
    assert observed["threshold_rows"] == 15
    assert observed["run_lengths"] == [15, 15]
    assert len(holdout_rows) == 25
    assert selected_model.decision_threshold == pytest.approx(0.77)
    assert selected_model.threshold_source == "round_selection_backtest"
    assert selected_model.probability_inverted is True


def test_build_round_evidence_records_data_health_block_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_training(*_args, **_kwargs):
        raise AssertionError("training should not start when data-health blocks")

    monkeypatch.setattr(oe, "train_round_model", fail_training)
    report = oe.build_round_evidence(
        round_id="round-test-health-gate",
        client=_SelectionClient(),
        strategy=StrategyConfig(),
        quote_asset="USDT",
        symbols=["BTCUSDT"],
        interval="1s",
        market_type="spot",
        objective_name="conservative",
        data_root=tmp_path / "data" / "optimization",
        docs_root=tmp_path / "docs" / "optimization",
        db_path=tmp_path / "market.sqlite",
        require_prefilled_data=True,
        min_data_rows=10,
        require_verified_checksum=True,
    )

    assert report["symbol_count_requested"] == 1
    assert report["symbol_count_completed"] == 1
    assert report["require_prefilled_data"] is True
    assert report["require_verified_checksum"] is True
    assert report["data_health"][0]["status"] == "block"
    assert "rows_below_min:0/10" in report["data_health"][0]["reasons"]
    assert "no_verified_archive_checksum" in report["data_health"][0]["reasons"]
    assert report["metrics"][0]["accepted"] is False
    assert "data_health_failed" in str(report["metrics"][0]["reason"])

    report_path = tmp_path / "docs" / "optimization" / "round-test-health-gate" / "data" / "report.json"
    data_health_path = tmp_path / "docs" / "optimization" / "round-test-health-gate" / "data" / "data-health.json"
    metrics_path = tmp_path / "docs" / "optimization" / "round-test-health-gate" / "data" / "backtest-metrics.csv"
    assert report_path.exists()
    assert data_health_path.exists()
    assert metrics_path.exists()
    assert str(data_health_path).replace("\\", "/") in report["tracked_artifacts"]
    assert json.loads(data_health_path.read_text(encoding="utf-8")) == report["data_health"]


def test_build_round_evidence_blocks_explicit_low_liquidity_symbol_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_training(*_args, **_kwargs):
        raise AssertionError("training should not start for rejected explicit symbols")

    monkeypatch.setattr(oe, "train_round_model", fail_training)
    report = oe.build_round_evidence(
        round_id="round-test-symbol-gate",
        client=_LowLiquidityClient(),
        strategy=StrategyConfig(),
        quote_asset="USDT",
        symbols=["BTCUSDT"],
        interval="1s",
        market_type="spot",
        objective_name="conservative",
        data_root=tmp_path / "data" / "optimization",
        docs_root=tmp_path / "docs" / "optimization",
        db_path=tmp_path / "market.sqlite",
    )

    assert report["metrics"][0]["accepted"] is False
    assert "symbol_selection_failed" in str(report["metrics"][0]["reason"])
    assert "quote_volume_below_default_live_gate" in str(report["metrics"][0]["reason"])
    assert report["data_health"] == []


def test_build_round_evidence_records_objective_strategy_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_training(*_args, **_kwargs):
        raise AssertionError("training should not start when data-health blocks")

    monkeypatch.setattr(oe, "train_round_model", fail_training)
    report = oe.build_round_evidence(
        round_id="round-test-regular-leverage",
        client=_SelectionClient(),
        strategy=StrategyConfig(leverage=1.0),
        quote_asset="USDT",
        symbols=["BTCUSDT"],
        interval="1s",
        market_type="spot",
        objective_name="regular",
        data_root=tmp_path / "data" / "optimization",
        docs_root=tmp_path / "docs" / "optimization",
        db_path=tmp_path / "market.sqlite",
        require_prefilled_data=True,
        min_data_rows=10,
        use_objective_strategy_defaults=True,
    )

    assert report["use_objective_strategy_defaults"] is True
    assert report["strategy"]["risk_level"] == "regular"
    assert report["strategy"]["leverage"] == pytest.approx(DEFAULT_REGULAR_LEVERAGE)
    assert report["effective_leverage"] == pytest.approx(1.0)
    assert report["leverage_applies"] is False
    assert report["metrics"][0]["risk_level"] == "regular"
    assert report["metrics"][0]["leverage"] == pytest.approx(DEFAULT_REGULAR_LEVERAGE)
    assert report["metrics"][0]["effective_leverage"] == pytest.approx(1.0)
    assert report["metrics"][0]["leverage_applies"] is False


def test_build_round_evidence_records_futures_effective_leverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_training(*_args, **_kwargs):
        raise AssertionError("training should not start when data-health blocks")

    monkeypatch.setattr(oe, "train_round_model", fail_training)
    report = oe.build_round_evidence(
        round_id="round-test-futures-leverage",
        client=_SelectionClient(),
        strategy=StrategyConfig(leverage=12.0),
        quote_asset="USDT",
        symbols=["BTCUSDT"],
        interval="1m",
        market_type="futures",
        objective_name="conservative",
        data_root=tmp_path / "data" / "optimization",
        docs_root=tmp_path / "docs" / "optimization",
        db_path=tmp_path / "market.sqlite",
        require_prefilled_data=True,
        min_data_rows=10,
    )

    assert report["market_type"] == "futures"
    assert report["configured_leverage"] == pytest.approx(12.0)
    assert report["effective_leverage"] == pytest.approx(12.0)
    assert report["leverage_applies"] is True
    assert report["metrics"][0]["leverage"] == pytest.approx(12.0)
    assert report["metrics"][0]["effective_leverage"] == pytest.approx(12.0)
    assert report["metrics"][0]["leverage_applies"] is True


def test_build_round_evidence_rejects_unsupported_market_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not supported on futures"):
        oe.build_round_evidence(
            round_id="round-test-bad-interval",
            client=_SelectionClient(),
            strategy=StrategyConfig(),
            quote_asset="USDT",
            symbols=["BTCUSDT"],
            interval="1s",
            market_type="futures",
            objective_name="conservative",
            data_root=tmp_path / "data" / "optimization",
            docs_root=tmp_path / "docs" / "optimization",
            db_path=tmp_path / "market.sqlite",
            require_prefilled_data=True,
        )
