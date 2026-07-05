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
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg: list(rows))

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


def test_train_round_model_fails_closed_when_selection_rejects_all_variants(
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
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg: list(rows))
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
    assert selected_model.threshold_source == "round_selection_fail_closed"
    assert selected_model.meta_label_policy["enabled"] is True
    assert selected_model.meta_label_policy["mode"] == "take_downsize_skip"
    assert selected_model.meta_label_policy["take_threshold"] == pytest.approx(1_000_000_000.0)
    assert "round_selection_gate_failed_no_final_holdout_entries" in selected_model.quality_warnings


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
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg: list(rows))
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
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg: list(rows))
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
    monkeypatch.setattr(oe, "make_advanced_rows", lambda _candles, _cfg: list(rows))
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
