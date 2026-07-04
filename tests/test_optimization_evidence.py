from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_ai_trading.api import Candle
from simple_ai_trading import optimization_evidence as oe
from simple_ai_trading.market_store import MarketDataStore
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
