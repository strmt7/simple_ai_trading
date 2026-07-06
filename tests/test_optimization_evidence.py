from __future__ import annotations

import json
import gzip
import statistics
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading.api import Candle
from simple_ai_trading.assets import DEFAULT_REGULAR_LEVERAGE
from simple_ai_trading.backtest import BacktestResult
from simple_ai_trading.features import FeatureAccelerationError, ModelRow
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


def _evidence(
    symbol: str,
    *,
    accepted: bool,
    roi_pct: float,
    closed_trades: int,
    realized_pnl: float | None = None,
) -> oe.BacktestEvidence:
    pnl = roi_pct * 10.0 if realized_pnl is None else realized_pnl
    return oe.BacktestEvidence(
        round_id="round-test",
        symbol=symbol,
        objective="conservative",
        risk_level="conservative",
        leverage=5.0,
        effective_leverage=5.0,
        leverage_applies=True,
        risk_per_trade=0.005,
        max_position_pct=0.1,
        max_drawdown_limit_pct=10.0,
        accepted=accepted,
        reason=None if accepted else "test rejection",
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-02T00:00:00Z",
        duration_years=1.0 / 365.25,
        candles=1000,
        rows=500,
        training_rows=400,
        training_positive_rate_pct=50.0,
        model_candidate_count=3,
        model_selected_candidate="default",
        model_selection_score=1.0,
        model_training_backend_kind="directml",
        model_training_backend_device="privateuseone:0",
        probability_calibration_backend_kind="directml",
        probability_calibration_backend_device="privateuseone:0",
        threshold_source="round_selection_backtest",
        threshold_calibration_score=1.0,
        threshold_calibration_pnl=pnl,
        threshold_calibration_trades=max(0, closed_trades),
        threshold_diagnostic_best_threshold=0.66,
        threshold_diagnostic_best_score=1.0,
        threshold_diagnostic_best_pnl=pnl,
        threshold_diagnostic_best_trades=max(0, closed_trades),
        decision_threshold=0.66,
        round_selection_gate_passed=accepted,
        round_selection_reject_reason=None if accepted else "test selection rejection",
        model_quality_warnings="",
        meta_label_policy_reason=None,
        starting_cash=1000.0,
        ending_cash=1000.0 + pnl,
        realized_pnl=pnl,
        roi_pct=roi_pct,
        buy_hold_pnl=-50.0,
        buy_hold_roi_pct=-5.0,
        edge_vs_buy_hold=pnl + 50.0,
        market_edge_pct=max(0.0, roi_pct),
        max_drawdown_pct=1.0,
        trades=closed_trades,
        closed_trades=closed_trades,
        win_rate_pct=60.0 if closed_trades else 0.0,
        total_fees=1.0 if closed_trades else 0.0,
        profit_factor=1.4 if accepted else 0.0,
        expectancy=pnl / closed_trades if closed_trades else 0.0,
        avg_trade_return_pct=roi_pct / closed_trades if closed_trades else 0.0,
        max_consecutive_losses=1 if closed_trades else 0,
        low_liquidity_sample_rate_pct=0.0,
        weekend_sample_rate_pct=0.0,
        scoring_backend_kind="directml",
        scoring_backend_device="privateuseone:0",
        chart_path="chart.svg",
        timeline_csv_path="timeline.csv",
    )


def test_futures_one_second_optimization_requires_prefilled_agg_trades_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="futures 1s optimization requires prefilled aggTrades-derived candles"):
        oe.build_round_evidence(
            round_id="round-test",
            client=SimpleNamespace(),
            strategy=StrategyConfig(),
            quote_asset="USDT",
            symbols=["BTCUSDT"],
            interval="1s",
            market_type="futures",
            require_prefilled_data=False,
            data_root=tmp_path / "data",
            docs_root=tmp_path / "docs",
            db_path=tmp_path / "market.sqlite",
        )


def test_critical_round_analysis_rejects_zero_trade_abstention() -> None:
    analysis = oe.critical_round_analysis(
        [
            _evidence("ETHUSDT", accepted=False, roi_pct=0.0, closed_trades=0),
            _evidence("BTCUSDT", accepted=False, roi_pct=0.0, closed_trades=0),
        ]
    )

    assert analysis["verdict"] == "fail"
    assert "all_symbols_zero_closed_trades" in analysis["failures"]
    assert "all_symbols_nonpositive_roi" in analysis["failures"]
    assert "invalid_no_trade_abstention" in str(analysis["interpretation"])
    assert analysis["total_closed_trades"] == 0


def test_critical_round_analysis_accepts_traded_positive_round() -> None:
    analysis = oe.critical_round_analysis(
        [
            _evidence("ETHUSDT", accepted=True, roi_pct=2.5, closed_trades=12),
            _evidence("BTCUSDT", accepted=True, roi_pct=1.0, closed_trades=8),
        ]
    )

    assert analysis["verdict"] == "pass"
    assert analysis["failures"] == []
    assert analysis["accepted_symbol_count"] == 2
    assert analysis["total_closed_trades"] == 20


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


class _MajorSelectionClient(_SelectionClient):
    def get_exchange_info(self) -> dict[str, object]:
        return {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC", "quoteAsset": "USDT"},
                {"symbol": "ETHUSDT", "status": "TRADING", "baseAsset": "ETH", "quoteAsset": "USDT"},
                {"symbol": "SOLUSDT", "status": "TRADING", "baseAsset": "SOL", "quoteAsset": "USDT"},
            ]
        }

    def get_all_tickers_24h(self) -> list[dict[str, object]]:
        return [
            {
                "symbol": symbol,
                "quoteVolume": str(2_500_000_000 // index),
                "count": str(1_200_000 // index),
                "lastPrice": str(price),
                "weightedAvgPrice": str(price),
                "highPrice": str(price * 1.02),
                "lowPrice": str(price * 0.98),
            }
            for index, (symbol, price) in enumerate(
                (("BTCUSDT", 60000.0), ("ETHUSDT", 3000.0), ("SOLUSDT", 150.0)),
                start=1,
            )
        ]

    def get_all_book_tickers(self) -> list[dict[str, object]]:
        return [
            {
                "symbol": symbol,
                "bidPrice": str(price),
                "bidQty": "1000",
                "askPrice": str(price * 1.00001),
                "askQty": "1000",
            }
            for symbol, price in (("BTCUSDT", 60000.0), ("ETHUSDT", 3000.0), ("SOLUSDT", 150.0))
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


class _MixedLiquidityClient(_SelectionClient):
    def get_exchange_info(self) -> dict[str, object]:
        return {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC", "quoteAsset": "USDT"},
                {"symbol": "SMALLUSDT", "status": "TRADING", "baseAsset": "SMALL", "quoteAsset": "USDT"},
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
            },
            {
                "symbol": "SMALLUSDT",
                "quoteVolume": "12000000",
                "count": "12000",
                "lastPrice": "2.0",
                "weightedAvgPrice": "2.0",
                "highPrice": "2.4",
                "lowPrice": "1.8",
            },
        ]

    def get_all_book_tickers(self) -> list[dict[str, object]]:
        return [
            {
                "symbol": "BTCUSDT",
                "bidPrice": "60000.00",
                "bidQty": "50",
                "askPrice": "60001.00",
                "askQty": "45",
            },
            {
                "symbol": "SMALLUSDT",
                "bidPrice": "2.00",
                "bidQty": "50000",
                "askPrice": "2.002",
                "askQty": "50000",
            },
        ]


def test_select_top_liquidity_symbols_defaults_to_strict_live_eligible() -> None:
    strict = oe.select_top_liquidity_symbols(_MixedLiquidityClient(), StrategyConfig(), count=2)
    research = oe.select_top_liquidity_symbols(
        _MixedLiquidityClient(),
        StrategyConfig(),
        count=2,
        strict_only=False,
    )

    assert [item.symbol for item in strict] == ["BTCUSDT"]
    assert all(item.strict_default_eligible for item in strict)
    assert [item.symbol for item in research] == ["BTCUSDT"]


def test_select_named_symbols_rejects_non_major_assets_even_when_liquid() -> None:
    selected = oe.select_named_symbols(
        _MixedLiquidityClient(),
        StrategyConfig(),
        ["BTCUSDT", "SMALLUSDT"],
        quote_asset="USDT",
    )

    by_symbol = {item.symbol: item for item in selected}
    assert by_symbol["BTCUSDT"].strict_default_eligible is True
    assert by_symbol["SMALLUSDT"].strict_default_eligible is False
    assert "unsupported_non_major_asset" in by_symbol["SMALLUSDT"].reasons


def test_build_round_evidence_blocks_strict_liquidity_shortfall_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("fetch should not start when strict liquidity selection is short")

    monkeypatch.setattr(oe, "fetch_full_history", fail_fetch)

    with pytest.raises(ValueError, match="strict_liquidity_selection_shortfall"):
        oe.build_round_evidence(
            round_id="round-test-strict-shortfall",
            client=_MixedLiquidityClient(),
            strategy=StrategyConfig(),
            quote_asset="USDT",
            symbol_count=2,
            interval="1m",
            market_type="futures",
            objective_name="conservative",
            data_root=tmp_path / "data" / "optimization",
            docs_root=tmp_path / "docs" / "optimization",
            db_path=tmp_path / "market.sqlite",
        )


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


def test_market_data_health_warns_when_archive_error_is_superseded(tmp_path: Path) -> None:
    db_path = tmp_path / "market.sqlite"
    good_url = "https://data.binance.vision/data/spot/daily/klines/ETHUSDT/1s/ETHUSDT-1s-2026-01-01.zip"
    bad_url = "https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1s/ETHUSDT-1s-2026-01.zip"
    with MarketDataStore(db_path) as store:
        store.upsert_candles("ETHUSDT", "spot", "1s", [_candle(0), _candle(1), _candle(2)], source="binance_archive")
        store.begin_archive_file(
            url=good_url,
            symbol="ETHUSDT",
            market_type="spot",
            interval="1s",
            period="2026-01-01",
        )
        store.complete_archive_file(
            url=good_url,
            status="complete",
            rows_inserted=3,
            bytes_downloaded=1234,
            sha256="abc",
            checksum_sha256="def",
            checksum_status="verified",
        )
        store.begin_archive_file(
            url=bad_url,
            symbol="ETHUSDT",
            market_type="spot",
            interval="1s",
            period="2026-01",
        )
        store.complete_archive_file(
            url=bad_url,
            status="error",
            rows_inserted=0,
            bytes_downloaded=0,
            sha256="",
            checksum_status="missing",
            error="missing checksum sidecar",
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
    assert health["reasons"] == []
    assert health["warnings"] == ["superseded_archive_errors:1"]


def test_market_data_health_blocks_span_below_min_years(tmp_path: Path) -> None:
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
        )
        store.complete_archive_file(
            url=archive_url,
            status="complete",
            rows_inserted=3,
            bytes_downloaded=1234,
            sha256="abc",
            checksum_sha256="def",
            checksum_status="verified",
        )

    health = oe.market_data_health_for_symbol(
        db_path=db_path,
        symbol="ethusdt",
        market_type="spot",
        interval="1s",
        min_rows=3,
        min_coverage_ratio=1.0,
        require_verified_checksum=True,
        min_span_years=0.001,
    )

    assert health["status"] == "block"
    assert health["span_years"] < 0.001
    assert any(str(reason).startswith("span_years_below_min:") for reason in health["reasons"])


def test_select_data_healthy_top_liquidity_symbols_skips_unhealthy_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        oe.SelectedSymbol(
            rank=index,
            symbol=symbol,
            quote_volume=1_000_000_000.0 / index,
            trade_count=1_000_000,
            spread_bps=1.0,
            liquidity_score=1.0,
            selection_score=10.0 - index,
            strict_default_eligible=True,
            tier="strict-live-eligible-at-selection",
            reasons=(),
        )
        for index, symbol in enumerate(("BADUSDT", "GOOD1USDT", "GOOD2USDT"), start=1)
    ]

    monkeypatch.setattr(oe, "select_top_liquidity_symbols", lambda *_args, **_kwargs: candidates)

    def fake_health(**kwargs):
        symbol = str(kwargs["symbol"])
        if symbol == "BADUSDT":
            return {
                "status": "block",
                "symbol": symbol,
                "rows": 500,
                "coverage_ratio": 0.50,
                "gap_count": 12,
                "reasons": ["rows_below_min:500/1000"],
            }
        return {
            "status": "ok",
            "symbol": symbol,
            "rows": 2000,
            "coverage_ratio": 1.0,
            "gap_count": 0,
            "reasons": [],
        }

    monkeypatch.setattr(oe, "market_data_health_for_symbol", fake_health)

    selected, rejections = oe.select_data_healthy_top_liquidity_symbols(
        _SelectionClient(),
        StrategyConfig(),
        quote_asset="USDT",
        count=2,
        market_type="futures",
        interval="1m",
        db_path=tmp_path / "market.sqlite",
        min_rows=1000,
        require_verified_checksum=True,
    )

    assert [item.symbol for item in selected] == ["GOOD1USDT", "GOOD2USDT"]
    assert [item.rank for item in selected] == [1, 2]
    assert rejections == [
        {
            "selection_rank": 1,
            "symbol": "BADUSDT",
            "tier": "strict-live-eligible-at-selection",
            "rows": 500,
            "span_years": 0.0,
            "coverage_ratio": 0.5,
            "gap_count": 12,
            "reasons": ["rows_below_min:500/1000"],
        }
    ]


def test_render_comparison_svg_decimates_visual_points_without_losing_span() -> None:
    points = [
        oe.EquityPoint(index, 1000.0 + (index % 17) - index * 0.001, (index % 31) / 1000.0, index * 60_000)
        for index in range(20_000)
    ]
    baseline = [
        oe.EquityPoint(index, 1000.0 + index * 0.0005, 0.0, index * 60_000)
        for index in range(20_000)
    ]

    svg = oe.render_comparison_svg(points, baseline, title="Large Round")

    assert len(svg) < 1_500_000
    assert "Rendered" in svg
    assert "full-resolution graph data is in CSV" in svg
    assert "1970-01-01" in svg
    assert "1970-01-14" in svg


def test_rolling_liquidity_flags_filtered_matches_bruteforce() -> None:
    start = 1_704_067_200_000
    candles = []
    for index in range(240):
        close = 100.0 + index * 0.02
        open_time = start + index * 60_000
        volume = 15.0 + (index % 11)
        quote_volume = close * volume
        trades = 120 + (index % 17)
        if index in {111, 137, 201}:
            quote_volume *= 0.1
            trades = 4
        candles.append(
            Candle(
                open_time=open_time,
                open=close,
                high=close + 0.5,
                low=close - 0.5,
                close=close,
                volume=volume,
                close_time=open_time + 59_999,
                quote_volume=quote_volume,
                trade_count=trades,
            )
        )

    def bucket(timestamp_ms: int) -> tuple[int, int, int]:
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        return dt.weekday(), dt.hour, dt.minute // 15

    def brute_force(window: int) -> dict[int, dict[str, float | int | bool | str]]:
        quote_volumes = [max(0.0, float(candle.quote_volume)) for candle in candles]
        trade_counts = [max(0, int(candle.trade_count)) for candle in candles]
        output: dict[int, dict[str, float | int | bool | str]] = {}
        for index, candle in enumerate(candles):
            start_index = max(0, index - window)
            window_volumes = quote_volumes[start_index:index]
            window_trades = trade_counts[start_index:index]
            current_bucket = bucket(int(candle.close_time))
            bucket_volumes = [
                quote_volumes[prior]
                for prior in range(start_index, index)
                if bucket(int(candles[prior].close_time)) == current_bucket
            ]
            bucket_trades = [
                trade_counts[prior]
                for prior in range(start_index, index)
                if bucket(int(candles[prior].close_time)) == current_bucket
            ]
            median_volume = statistics.median(window_volumes) if window_volumes else 0.0
            median_trades = statistics.median(window_trades) if window_trades else 0.0
            bucket_median_volume = statistics.median(bucket_volumes) if len(bucket_volumes) >= 8 else 0.0
            bucket_median_trades = statistics.median(bucket_trades) if len(bucket_trades) >= 8 else 0.0
            close_time = int(candle.close_time)
            dt = datetime.fromtimestamp(close_time / 1000.0, tz=timezone.utc)
            low_volume = bool(median_volume > 0 and quote_volumes[index] < median_volume * 0.35)
            low_trades = bool(median_trades > 0 and trade_counts[index] < median_trades * 0.35)
            low_bucket_volume = bool(bucket_median_volume > 0 and quote_volumes[index] < bucket_median_volume * 0.45)
            low_bucket_trades = bool(bucket_median_trades > 0 and trade_counts[index] < bucket_median_trades * 0.45)
            output[close_time] = {
                "quote_volume": float(quote_volumes[index]),
                "trade_count": int(trade_counts[index]),
                "rolling_quote_volume_median": float(median_volume),
                "rolling_trade_count_median": float(median_trades),
                "clock_bucket": f"{current_bucket[0]}:{current_bucket[1]:02d}:{current_bucket[2]:02d}",
                "clock_bucket_quote_volume_median": float(bucket_median_volume),
                "clock_bucket_trade_count_median": float(bucket_median_trades),
                "data_probed_low_session_flag": bool(low_bucket_volume or low_bucket_trades),
                "low_liquidity_flag": bool(low_volume or low_trades or low_bucket_volume or low_bucket_trades),
                "weekend_flag": bool(dt.weekday() >= 5),
                "utc_hour": int(dt.hour),
                "utc_weekday": int(dt.weekday()),
            }
        return output

    wanted = {candles[index].close_time for index in (32, 111, 137, 201, 239)}
    expected = brute_force(32)
    actual = oe._rolling_liquidity_flags(candles, window=32, timestamps=wanted)

    assert set(actual) == wanted
    assert actual == {timestamp: expected[timestamp] for timestamp in sorted(wanted)}


def test_portfolio_timeline_streaming_aggregate_matches_row_inputs() -> None:
    rows = [
        [
            {
                "timestamp_ms": 1_700_000_000_000,
                "strategy_equity": 1000.0,
                "baseline_equity": 990.0,
                "strategy_drawdown": 0.01,
                "low_liquidity_flag": "false",
            },
            {
                "timestamp_ms": 1_700_000_060_000,
                "strategy_equity": 1010.0,
                "baseline_equity": 991.0,
                "strategy_drawdown": 0.0,
                "low_liquidity_flag": "true",
            },
        ],
        [
            {
                "timestamp_ms": 1_700_000_000_000,
                "strategy_equity": 980.0,
                "baseline_equity": 995.0,
                "strategy_drawdown": 0.02,
                "low_liquidity_flag": "true",
            }
        ],
    ]

    timeline = oe._portfolio_timeline(rows)

    assert timeline[0]["symbols_reporting"] == 2
    assert timeline[0]["mean_strategy_equity"] == pytest.approx(990.0)
    assert timeline[0]["mean_baseline_equity"] == pytest.approx(992.5)
    assert timeline[0]["mean_drawdown"] == pytest.approx(0.015)
    assert timeline[0]["low_liquidity_symbol_count"] == 1
    assert timeline[1]["symbols_reporting"] == 1
    assert timeline[1]["low_liquidity_symbol_count"] == 1


def test_write_csv_supports_compressed_graph_data(tmp_path: Path) -> None:
    output = tmp_path / "timeline.csv.gz"

    oe._write_csv(
        output,
        [{"timestamp_ms": 1, "strategy_equity": 1000.0}],
        ("timestamp_ms", "strategy_equity"),
    )

    with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
        payload = handle.read()
    assert "timestamp_ms,strategy_equity" in payload
    assert "1,1000.0" in payload


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
    observed: dict[str, object] = {"run_lengths": [], "phases": []}
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg, **_kwargs: list(rows))

    def fake_train_advanced(train_rows, _feature_cfg, **kwargs):
        observed["train_rows"] = len(train_rows)
        observed["train_validation_rows"] = len(kwargs["validation_rows"])
        return model, SimpleNamespace(row_count=len(train_rows), positive_rate=1.0)

    monkeypatch.setattr(oe, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(
        oe,
        "calibrate_probability_temperature",
        lambda calibration_rows, _model, **_kwargs: SimpleNamespace(status="fail", rows=len(calibration_rows)),
    )

    def fake_threshold(selection_rows, _model, _strategy, **_kwargs):
        observed["threshold_rows"] = len(selection_rows)
        return SimpleNamespace(
            accepted=True,
            threshold=0.77,
            score=2.0,
            realized_pnl=12.0,
            closed_trades=7,
            scoring_backend_kind="directml",
            scoring_backend_device="privateuseone:0",
            scoring_backend_reason="",
        )

    monkeypatch.setattr(oe, "calibrate_threshold_for_backtest", fake_threshold)

    def result_for(realized_pnl: float) -> BacktestResult:
        closed_trades = 8
        if realized_pnl < 0.0:
            trade_pnls = tuple(realized_pnl / closed_trades for _ in range(closed_trades))
        else:
            trade_pnls = tuple(realized_pnl / closed_trades for _ in range(closed_trades))
        trade_returns = tuple(value / 1000.0 for value in trade_pnls)
        gross_profit = sum(value for value in trade_pnls if value > 0.0)
        gross_loss = abs(sum(value for value in trade_pnls if value < 0.0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0)
        ending_cash = 1000.0 + realized_pnl
        max_drawdown = 0.01 if realized_pnl >= 0.0 else max(0.01, abs(realized_pnl) / 1000.0)
        final_drawdown = 0.0 if ending_cash >= 1000.0 else (1000.0 - ending_cash) / 1000.0
        equity_curve = (
            {"timestamp": 0, "equity": 1000.0, "drawdown": 0.0, "position_side": 0},
            {"timestamp": 60_000, "equity": 1000.0 * (1.0 - max_drawdown), "drawdown": max_drawdown, "position_side": 0},
            {"timestamp": 120_000, "equity": ending_cash, "drawdown": final_drawdown, "position_side": 0},
        )
        trade_log = tuple(
            {
                "opened_at": int(index * 120_000),
                "closed_at": int(index * 120_000 + 60_000),
                "side": 1,
                "gross_notional": 100.0,
                "entry_price": 100.0,
                "exit_mark_price": max(0.01, 100.0 + pnl + 0.1),
                "realized_pnl": float(pnl + 0.1),
                "net_pnl": float(pnl),
                "return_pct": float(ret),
                "entry_fee": 0.05,
                "exit_fee": 0.05,
                "exit_reason": "take_profit_close" if pnl > 0.0 else "stop_loss_close",
            }
            for index, (pnl, ret) in enumerate(zip(trade_pnls, trade_returns, strict=True))
        )
        return BacktestResult(
            starting_cash=1000.0,
            ending_cash=ending_cash,
            realized_pnl=realized_pnl,
            win_rate=sum(1 for value in trade_pnls if value > 0.0) / closed_trades,
            trades=closed_trades,
            max_drawdown=max_drawdown,
            closed_trades=closed_trades,
            gross_exposure=100.0,
            total_fees=0.1 * closed_trades,
            stopped_by_drawdown=False,
            max_exposure=100.0,
            trades_per_day_cap_hit=0,
            buy_hold_pnl=1.0,
            edge_vs_buy_hold=realized_pnl - 1.0,
            equity_curve=equity_curve,
            trade_pnls=trade_pnls,
            trade_returns=trade_returns,
            trade_log=trade_log,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            expectancy=realized_pnl / 8.0,
            average_trade_return=sum(trade_returns) / len(trade_returns),
            trade_return_stdev=0.0,
            max_consecutive_losses=closed_trades if realized_pnl < 0.0 else 0,
        )

    def fake_run_backtest(selection_rows, candidate_model, *_args, **_kwargs):
        observed["run_lengths"].append(len(selection_rows))  # type: ignore[index]
        return result_for(20.0 if candidate_model.probability_inverted else 10.0)

    monkeypatch.setattr(oe, "run_backtest", fake_run_backtest)

    def status_callback(phase, payload):
        observed["phases"].append((phase, dict(payload)))  # type: ignore[index]

    selected_model, _report, _all_rows, holdout_rows = oe.train_round_model(
        [_candle(index) for index in range(100)],
        StrategyConfig(),
        get_objective("conservative"),
        market_type="futures",
        starting_cash=1000.0,
        compute_backend="auto",
        batch_size=1024,
        status_callback=status_callback,
    )

    assert observed["train_rows"] == 60
    assert observed["train_validation_rows"] == 15
    assert observed["threshold_rows"] == 15
    assert observed["run_lengths"] == [15, 15]
    assert len(holdout_rows) == 25
    assert selected_model.decision_threshold == pytest.approx(0.77)
    assert selected_model.threshold_source == "round_selection_backtest"
    assert selected_model.probability_inverted is True
    phases = [item[0] for item in observed["phases"]]  # type: ignore[index]
    assert phases == [
        "feature_generation_started",
        "feature_generation_complete",
        "training_started",
        "training_complete",
        "threshold_calibration_started",
        "threshold_calibration_complete",
        "selection_backtest_complete",
        "inversion_backtest_complete",
    ]


def test_train_round_model_keeps_rejected_selection_diagnostic_holdout_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        ModelRow(timestamp=index * 60_000, close=100.0 + index, features=(1.0,), label=1)
        for index in range(100)
    ]
    model = TrainedModel(
        weights=[1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg, **_kwargs: list(rows))
    monkeypatch.setattr(
        oe,
        "train_advanced",
        lambda train_rows, _feature_cfg, **_kwargs: (
            model,
            SimpleNamespace(row_count=len(train_rows), positive_rate=1.0),
        ),
    )
    monkeypatch.setattr(
        oe,
        "calibrate_probability_temperature",
        lambda calibration_rows, _model, **_kwargs: SimpleNamespace(status="fail", rows=len(calibration_rows)),
    )
    monkeypatch.setattr(
        oe,
        "calibrate_threshold_for_backtest",
        lambda *_args, **_kwargs: SimpleNamespace(
            accepted=False,
            threshold=0.66,
            score=-1.0,
            realized_pnl=-1.0,
            closed_trades=0,
            scoring_backend_kind="directml",
            scoring_backend_device="privateuseone:0",
            scoring_backend_reason="",
        ),
    )
    monkeypatch.setattr(
        oe,
        "run_backtest",
        lambda *_args, **_kwargs: BacktestResult(
            starting_cash=1000.0,
            ending_cash=999.0,
            realized_pnl=-1.0,
            win_rate=0.0,
            trades=1,
            max_drawdown=0.01,
            closed_trades=1,
            gross_exposure=100.0,
            total_fees=1.0,
            stopped_by_drawdown=False,
            max_exposure=100.0,
            trades_per_day_cap_hit=0,
            buy_hold_pnl=1.0,
            edge_vs_buy_hold=-2.0,
            profit_factor=0.0,
            expectancy=-1.0,
            max_consecutive_losses=1,
        ),
    )

    selected_model, _report, _all_rows, holdout_rows = oe.train_round_model(
        [_candle(index) for index in range(100)],
        StrategyConfig(),
        get_objective("conservative"),
        market_type="futures",
        starting_cash=1000.0,
        compute_backend="auto",
        batch_size=1024,
    )

    assert len(holdout_rows) == 25
    assert selected_model.threshold_source == "round_selection_rejected_diagnostic_holdout"
    assert selected_model.round_selection_gate_passed is False
    assert "closed_trades<5" in selected_model.round_selection_reject_reason
    assert "round_selection_gate_failed_diagnostic_holdout_only" in selected_model.quality_warnings
    assert getattr(selected_model, "meta_label_policy", {}) == {}


def test_train_round_model_uses_rejected_best_threshold_for_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        ModelRow(timestamp=index * 60_000, close=100.0 + index, features=(1.0,), label=1)
        for index in range(100)
    ]
    model = TrainedModel(
        weights=[1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg, **_kwargs: list(rows))
    monkeypatch.setattr(
        oe,
        "train_advanced",
        lambda train_rows, _feature_cfg, **_kwargs: (
            model,
            SimpleNamespace(row_count=len(train_rows), positive_rate=1.0),
        ),
    )
    monkeypatch.setattr(
        oe,
        "calibrate_probability_temperature",
        lambda calibration_rows, _model, **_kwargs: SimpleNamespace(status="fail", rows=len(calibration_rows)),
    )
    monkeypatch.setattr(
        oe,
        "calibrate_threshold_for_backtest",
        lambda *_args, **_kwargs: SimpleNamespace(
            accepted=False,
            threshold=0.50,
            score=-50.0,
            realized_pnl=0.0,
            closed_trades=0,
            best_threshold=0.61,
            best_score=-9.0,
            best_realized_pnl=-8.0,
            best_closed_trades=4,
            scoring_backend_kind="directml",
            scoring_backend_device="privateuseone:0",
            scoring_backend_reason="",
        ),
    )

    def fake_run_backtest(_rows, candidate_model, *_args, **_kwargs):
        uses_diagnostic_threshold = abs(float(candidate_model.decision_threshold or 0.0) - 0.61) <= 1e-12
        realized = -8.0 if uses_diagnostic_threshold else 0.0
        closed = 4 if uses_diagnostic_threshold else 0
        return BacktestResult(
            starting_cash=1000.0,
            ending_cash=1000.0 + realized,
            realized_pnl=realized,
            win_rate=0.25 if closed else 0.0,
            trades=closed,
            max_drawdown=0.01,
            closed_trades=closed,
            gross_exposure=100.0 if closed else 0.0,
            total_fees=1.0 if closed else 0.0,
            stopped_by_drawdown=False,
            max_exposure=100.0 if closed else 0.0,
            trades_per_day_cap_hit=0,
            buy_hold_pnl=0.0,
            edge_vs_buy_hold=realized,
            profit_factor=0.5 if closed else 0.0,
            expectancy=realized / closed if closed else 0.0,
            max_consecutive_losses=2 if closed else 0,
        )

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

    assert len(holdout_rows) == 25
    assert selected_model.round_selection_gate_passed is False
    assert selected_model.threshold_source == "round_selection_rejected_best_threshold_diagnostic"
    assert selected_model.decision_threshold == pytest.approx(0.61)
    assert selected_model.threshold_calibration_trades == 4
    assert selected_model.threshold_calibration_pnl == pytest.approx(-8.0)
    assert selected_model.threshold_diagnostic_best_threshold == pytest.approx(0.61)
    assert selected_model.threshold_diagnostic_best_trades == 4
    assert selected_model.threshold_diagnostic_best_pnl == pytest.approx(-8.0)


def test_round_model_candidates_include_risk_gated_signal_diversity() -> None:
    strategy = StrategyConfig(signal_threshold=0.66)
    objective = get_objective("conservative")
    feature_cfg = oe.default_config_for(objective.name, strategy.enabled_features)

    candidates = oe._round_model_candidates(objective, strategy, feature_cfg, requested=10)

    assert [candidate.name for candidate in candidates[:2]] == ["default", "lower_lr_more_l2"]
    assert len(candidates) == 10
    thresholds = [candidate.signal_threshold for candidate in candidates]
    assert min(thresholds) == pytest.approx(0.56)
    assert max(thresholds) == pytest.approx(0.70)
    assert any(candidate.name.startswith("lower_signal") for candidate in candidates)
    assert any(candidate.name == "frequency_probe_forward" for candidate in candidates)
    assert any(candidate.feature_cfg.label_mode == "triple_barrier" for candidate in candidates)


def test_train_round_model_selects_best_scored_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        ModelRow(timestamp=index * 60_000, close=100.0 + index, features=(1.0,), label=index % 2)
        for index in range(100)
    ]
    calls = {"train": 0}
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg, **_kwargs: list(rows))

    def fake_train_advanced(train_rows, _feature_cfg, **_kwargs):
        calls["train"] += 1
        model = TrainedModel(
            weights=[1.0],
            bias=0.0,
            feature_dim=1,
            epochs=1,
            feature_means=[0.0],
            feature_stds=[1.0],
            model_family=f"candidate_{calls['train']}",
        )
        return model, SimpleNamespace(row_count=len(train_rows), positive_rate=0.5)

    monkeypatch.setattr(oe, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(
        oe,
        "calibrate_probability_temperature",
        lambda calibration_rows, _model, **_kwargs: SimpleNamespace(status="fail", rows=len(calibration_rows)),
    )
    monkeypatch.setattr(
        oe,
        "calibrate_threshold_for_backtest",
        lambda *_args, **_kwargs: SimpleNamespace(
            accepted=False,
            threshold=0.66,
            score=-1.0,
            realized_pnl=-1.0,
            closed_trades=0,
            scoring_backend_kind="directml",
            scoring_backend_device="privateuseone:0",
            scoring_backend_reason="",
        ),
    )

    def fake_run_backtest(_rows, candidate_model, *_args, **_kwargs):
        realized = 25.0 if str(candidate_model.model_family).startswith("candidate_2") else 5.0
        return BacktestResult(
            starting_cash=1000.0,
            ending_cash=1000.0 + realized,
            realized_pnl=realized,
            win_rate=0.75,
            trades=8,
            max_drawdown=0.01,
            closed_trades=8,
            gross_exposure=100.0,
            total_fees=1.0,
            stopped_by_drawdown=False,
            max_exposure=100.0,
            trades_per_day_cap_hit=0,
            buy_hold_pnl=0.0,
            edge_vs_buy_hold=realized,
            gross_profit=realized + 1.0,
            gross_loss=1.0,
            profit_factor=2.0,
            expectancy=realized / 8.0,
            max_consecutive_losses=1,
        )

    monkeypatch.setattr(oe, "run_backtest", fake_run_backtest)

    selected_model, report, _all_rows, holdout_rows = oe.train_round_model(
        [_candle(index) for index in range(100)],
        StrategyConfig(),
        get_objective("conservative"),
        market_type="futures",
        starting_cash=1000.0,
        compute_backend="auto",
        batch_size=1024,
        model_candidate_count=2,
    )

    assert calls["train"] == 2
    assert selected_model.model_family.startswith("candidate_2")
    assert selected_model.model_candidate_count == 2
    assert selected_model.model_selected_candidate == "lower_lr_more_l2"
    assert selected_model.model_selection_score > 0.0
    assert "model_selected_candidate" in selected_model.__dataclass_fields__
    assert report.row_count == 60
    assert len(holdout_rows) == 25


def test_train_round_model_require_gpu_rejects_training_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [ModelRow(timestamp=index * 60_000, close=100.0, features=(1.0,), label=1) for index in range(100)]
    model = TrainedModel(
        weights=[1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        training_backend_requested="directml",
        training_backend_kind="cpu",
        training_backend_device="cpu",
        training_backend_reason="DirectML training failed in test",
    )
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg, **_kwargs: list(rows))
    monkeypatch.setattr(
        oe,
        "train_advanced",
        lambda *_args, **_kwargs: (model, SimpleNamespace(row_count=60, positive_rate=1.0)),
    )

    with pytest.raises(RuntimeError, match="gpu_required_but_training_fell_back_to_cpu"):
        oe.train_round_model(
            [_candle(index) for index in range(100)],
            StrategyConfig(),
            get_objective("conservative"),
            market_type="futures",
            starting_cash=1000.0,
            compute_backend="directml",
            batch_size=1024,
            require_gpu=True,
        )


def test_train_round_model_require_gpu_rejects_feature_generation_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fail_feature_generation(_candles, _cfg, **kwargs):
        observed["require_accelerated"] = kwargs.get("require_accelerated")
        raise FeatureAccelerationError("feature_acceleration_required_but_directml_feature_generation_failed")

    monkeypatch.setattr(oe, "make_advanced_rows", fail_feature_generation)

    with pytest.raises(FeatureAccelerationError, match="feature_generation_failed"):
        oe.train_round_model(
            [_candle(index) for index in range(100)],
            StrategyConfig(),
            get_objective("conservative"),
            market_type="futures",
            starting_cash=1000.0,
            compute_backend="directml",
            batch_size=1024,
            require_gpu=True,
        )

    assert observed["require_accelerated"] is True


def test_train_round_model_require_gpu_rejects_threshold_scoring_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [ModelRow(timestamp=index * 60_000, close=100.0, features=(1.0,), label=1) for index in range(100)]
    model = TrainedModel(
        weights=[1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        training_backend_requested="directml",
        training_backend_kind="directml",
        training_backend_device="privateuseone:0",
    )
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg, **_kwargs: list(rows))
    monkeypatch.setattr(
        oe,
        "train_advanced",
        lambda *_args, **_kwargs: (model, SimpleNamespace(row_count=60, positive_rate=1.0)),
    )
    monkeypatch.setattr(
        oe,
        "calibrate_probability_temperature",
        lambda calibration_rows, _model, **_kwargs: SimpleNamespace(status="fail", rows=len(calibration_rows)),
    )
    monkeypatch.setattr(
        oe,
        "calibrate_threshold_for_backtest",
        lambda *_args, **_kwargs: SimpleNamespace(
            accepted=False,
            threshold=0.66,
            score=-1.0,
            realized_pnl=-1.0,
            closed_trades=0,
            scoring_backend_kind="cpu",
            scoring_backend_device="cpu",
            scoring_backend_reason="DirectML scoring failed in test",
        ),
    )

    with pytest.raises(RuntimeError, match="gpu_required_but_threshold_scoring_fell_back_to_cpu"):
        oe.train_round_model(
            [_candle(index) for index in range(100)],
            StrategyConfig(),
            get_objective("conservative"),
            market_type="futures",
            starting_cash=1000.0,
            compute_backend="directml",
            batch_size=1024,
            require_gpu=True,
        )


def test_train_round_model_require_gpu_rejects_probability_calibration_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [ModelRow(timestamp=index * 60_000, close=100.0, features=(1.0,), label=index % 2) for index in range(100)]
    model = TrainedModel(
        weights=[1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        training_backend_requested="directml",
        training_backend_kind="directml",
        training_backend_device="privateuseone:0",
    )
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg, **_kwargs: list(rows))
    monkeypatch.setattr(
        oe,
        "train_advanced",
        lambda *_args, **_kwargs: (model, SimpleNamespace(row_count=60, positive_rate=0.5)),
    )
    monkeypatch.setattr(
        oe,
        "calibrate_probability_temperature",
        lambda calibration_rows, _model, **_kwargs: SimpleNamespace(
            status="ok",
            rows=len(calibration_rows),
            temperature=1.0,
            log_loss_before=0.7,
            log_loss_after=0.7,
            brier_before=0.25,
            brier_after=0.25,
            expected_calibration_error_before=0.1,
            expected_calibration_error_after=0.1,
            calibration_backend_kind="cpu",
            calibration_backend_reason="DirectML calibration failed in test",
        ),
    )

    with pytest.raises(RuntimeError, match="gpu_required_but_probability_calibration_fell_back_to_cpu"):
        oe.train_round_model(
            [_candle(index) for index in range(100)],
            StrategyConfig(),
            get_objective("conservative"),
            market_type="futures",
            starting_cash=1000.0,
            compute_backend="directml",
            batch_size=1024,
            require_gpu=True,
        )


def test_build_round_evidence_require_gpu_rejects_holdout_scoring_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [ModelRow(timestamp=index * 60_000, close=100.0 + index, features=(1.0,), label=1) for index in range(30)]
    model = TrainedModel(
        weights=[1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        training_backend_requested="directml",
        training_backend_kind="directml",
        training_backend_device="privateuseone:0",
    )
    monkeypatch.setattr(
        oe,
        "resolve_backend",
        lambda _requested: SimpleNamespace(
            requested="directml",
            kind="directml",
            device="privateuseone:0",
            vendor="DirectML",
            reason="",
        ),
    )
    monkeypatch.setattr(oe, "fetch_full_history", lambda *_args, **_kwargs: [_candle(index) for index in range(30)])

    def fake_train_round_model(*_args, **kwargs):
        assert kwargs["require_gpu"] is True
        return model, SimpleNamespace(row_count=20), list(rows), list(rows[-10:])

    monkeypatch.setattr(oe, "train_round_model", fake_train_round_model)
    monkeypatch.setattr(
        oe,
        "run_backtest",
        lambda *_args, **_kwargs: BacktestResult(
            starting_cash=1000.0,
            ending_cash=1001.0,
            realized_pnl=1.0,
            win_rate=1.0,
            trades=1,
            max_drawdown=0.0,
            closed_trades=1,
            gross_exposure=100.0,
            total_fees=0.0,
            stopped_by_drawdown=False,
            max_exposure=100.0,
            trades_per_day_cap_hit=0,
            buy_hold_pnl=0.0,
            edge_vs_buy_hold=1.0,
            scoring_backend_requested="directml",
            scoring_backend_kind="cpu",
            scoring_backend_device="cpu",
            scoring_backend_reason="DirectML scoring failed in test",
        ),
    )

    report = oe.build_round_evidence(
        round_id="round-test-require-gpu-holdout",
        client=_SelectionClient(),
        strategy=StrategyConfig(),
        quote_asset="USDT",
        symbols=["BTCUSDT"],
        interval="1m",
        market_type="futures",
        objective_name="conservative",
        data_root=tmp_path / "data" / "optimization",
        docs_root=tmp_path / "docs" / "optimization",
        db_path=tmp_path / "market.sqlite",
        require_gpu=True,
    )

    assert report["metrics"][0]["accepted"] is False
    assert "gpu_required_but_holdout_scoring_fell_back_to_cpu" in str(report["metrics"][0]["reason"])


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
    assert report["metrics"][0]["training_rows"] == 0
    assert report["metrics"][0]["model_training_backend_kind"] == "error"
    assert report["metrics"][0]["probability_calibration_backend_kind"] == "error"
    assert report["metrics"][0]["threshold_source"] is None
    assert report["evidence_verdict"] == "fail"
    assert report["critical_analysis"]["verdict"] == "fail"
    assert "no_closed_trades" in report["critical_analysis"]["failures"]
    assert report["progress"]["critical_verdict"] == "fail"
    assert report["progress"]["total_closed_trades"] == 0

    report_path = tmp_path / "docs" / "optimization" / "round-test-health-gate" / "data" / "report.json"
    data_health_path = tmp_path / "docs" / "optimization" / "round-test-health-gate" / "data" / "data-health.json"
    metrics_path = tmp_path / "docs" / "optimization" / "round-test-health-gate" / "data" / "backtest-metrics.csv"
    assert report_path.exists()
    assert data_health_path.exists()
    assert metrics_path.exists()
    assert str(data_health_path).replace("\\", "/") in report["tracked_artifacts"]
    assert json.loads(data_health_path.read_text(encoding="utf-8")) == report["data_health"]
    integrity = {entry["path"]: entry for entry in report["artifact_integrity"]}
    metrics_key = str(metrics_path).replace("\\", "/")
    data_health_key = str(data_health_path).replace("\\", "/")
    assert metrics_key in integrity
    assert data_health_key in integrity
    assert integrity[metrics_key]["row_count"] == 1
    assert "symbol" in integrity[metrics_key]["columns"]
    assert integrity[metrics_key]["sha256"] == oe._artifact_integrity(metrics_key)["sha256"]
    assert integrity[data_health_key]["bytes"] == data_health_path.stat().st_size
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted_report["artifact_integrity"] == report["artifact_integrity"]


def test_promotion_grade_round_forces_major_scope_and_blocks_unverified_data_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_training(*_args, **_kwargs):
        raise AssertionError("training should not start without promotion-grade data")

    monkeypatch.setattr(oe, "train_round_model", fail_training)

    report = oe.build_round_evidence(
        round_id="round-test-promotion-grade",
        client=_MajorSelectionClient(),
        strategy=StrategyConfig(),
        quote_asset="USDT",
        interval="5m",
        market_type="futures",
        objective_name="conservative",
        data_root=tmp_path / "data" / "optimization",
        docs_root=tmp_path / "docs" / "optimization",
        db_path=tmp_path / "market.sqlite",
        promotion_grade=True,
        min_promotion_data_years=0.0,
    )

    assert report["promotion_grade"] is True
    assert report["interval"] == "1s"
    assert report["market_type"] == "futures"
    assert report["require_prefilled_data"] is True
    assert report["require_verified_checksum"] is True
    assert report["min_coverage_ratio"] == pytest.approx(0.995)
    assert report["max_gap_count"] == 0
    assert report["symbol_count_requested"] == 3
    assert report["explicit_symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert [item["symbol"] for item in report["data_health"]] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert all(item["status"] == "block" for item in report["data_health"])
    assert all(metric["training_rows"] == 0 for metric in report["metrics"])
    contract = report["promotion_grade_contract"]
    assert contract["status"] == "block"
    assert contract["required_symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert "critical_analysis_not_pass" in contract["reasons"]
    assert "data_health_failed:BTCUSDT" in contract["reasons"]
    assert "data_health_missing_verified_checksum:SOLUSDT" in contract["reasons"]


def test_promotion_grade_rejects_incomplete_or_extra_symbol_scope(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="promotion_grade_requires_exact_btc_eth_sol_scope"):
        oe.build_round_evidence(
            round_id="round-test-promotion-scope",
            client=_MajorSelectionClient(),
            strategy=StrategyConfig(),
            quote_asset="USDT",
            symbols=["BTCUSDT", "ETHUSDT"],
            interval="1s",
            market_type="futures",
            objective_name="conservative",
            data_root=tmp_path / "data" / "optimization",
            docs_root=tmp_path / "docs" / "optimization",
            db_path=tmp_path / "market.sqlite",
            promotion_grade=True,
        )


def test_build_round_evidence_blocks_rejected_selection_but_records_diagnostic_trades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [ModelRow(timestamp=index * 60_000, close=100.0 + index, features=(1.0,), label=1) for index in range(40)]
    model = TrainedModel(
        weights=[1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    model.round_selection_gate_passed = False
    model.round_selection_reject_reason = "closed_trades<5; profit_factor<1.1"
    model.threshold_source = "round_selection_rejected_diagnostic_holdout"
    model.quality_warnings = ["round_selection_gate_failed_diagnostic_holdout_only"]

    monkeypatch.setattr(oe, "fetch_full_history", lambda *_args, **_kwargs: [_candle(index) for index in range(80)])
    monkeypatch.setattr(
        oe,
        "train_round_model",
        lambda *_args, **_kwargs: (model, SimpleNamespace(row_count=32, positive_rate=0.5), rows, rows),
    )
    monkeypatch.setattr(
        oe,
        "run_backtest",
        lambda *_args, **_kwargs: BacktestResult(
            starting_cash=1000.0,
            ending_cash=1025.0,
            realized_pnl=25.0,
            win_rate=0.75,
            trades=8,
            max_drawdown=0.02,
            closed_trades=8,
            gross_exposure=100.0,
            total_fees=1.0,
            stopped_by_drawdown=False,
            max_exposure=100.0,
            trades_per_day_cap_hit=0,
            buy_hold_pnl=2.0,
            edge_vs_buy_hold=23.0,
            profit_factor=1.5,
            expectancy=3.125,
            max_consecutive_losses=1,
            scoring_backend_kind="directml",
            scoring_backend_device="privateuseone:0",
        ),
    )

    report = oe.build_round_evidence(
        round_id="round-test-selection-diagnostic",
        client=_SelectionClient(),
        strategy=StrategyConfig(),
        quote_asset="USDT",
        symbols=["BTCUSDT"],
        interval="1m",
        market_type="futures",
        objective_name="conservative",
        data_root=tmp_path / "data" / "optimization",
        docs_root=tmp_path / "docs" / "optimization",
        db_path=tmp_path / "market.sqlite",
    )

    metric = report["metrics"][0]
    assert metric["closed_trades"] == 8
    assert metric["roi_pct"] == pytest.approx(2.5)
    assert metric["accepted"] is False
    assert "selection_gate_failed" in str(metric["reason"])
    assert metric["round_selection_gate_passed"] is False
    assert metric["round_selection_reject_reason"] == "closed_trades<5; profit_factor<1.1"
    assert metric["threshold_source"] == "round_selection_rejected_diagnostic_holdout"
    assert report["critical_analysis"]["total_closed_trades"] == 8
    assert "no_closed_trades" not in report["critical_analysis"]["failures"]
    assert "no_accepted_symbols" in report["critical_analysis"]["failures"]


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


def test_build_round_evidence_can_require_non_cpu_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        oe,
        "resolve_backend",
        lambda _requested: SimpleNamespace(
            requested="directml",
            kind="cpu",
            device="cpu",
            vendor="Python stdlib",
            reason="DirectML unavailable in test",
        ),
    )

    with pytest.raises(ValueError, match="gpu_required_but_unavailable"):
        oe.build_round_evidence(
            round_id="round-test-require-gpu",
            client=_SelectionClient(),
            strategy=StrategyConfig(),
            quote_asset="USDT",
            symbols=["BTCUSDT"],
            interval="1m",
            market_type="futures",
            objective_name="conservative",
            data_root=tmp_path / "data" / "optimization",
            docs_root=tmp_path / "docs" / "optimization",
            db_path=tmp_path / "market.sqlite",
            require_gpu=True,
        )


def test_build_round_evidence_rejects_unsupported_market_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="futures 1s optimization requires prefilled"):
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
            require_prefilled_data=False,
        )
