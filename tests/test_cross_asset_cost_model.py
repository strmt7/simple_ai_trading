from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from simple_ai_trading.ai_trade_veto import _decision, _failed_decision
from simple_ai_trading.cross_asset_cost_data import (
    ArchiveEvidence,
    CrossAssetDataset,
    MinuteSeries,
    SYMBOLS,
    SeriesEvidence,
    SourceEvidence,
    build_cross_asset_dataset,
)
from simple_ai_trading.cross_asset_cost_model import replay_nonoverlapping


def _minute_panel(rows: int = 46_000) -> tuple[dict[str, MinuteSeries], SourceEvidence]:
    start = int(datetime(2021, 12, 1, tzinfo=UTC).timestamp() * 1000)
    timestamps = start + np.arange(rows, dtype=np.int64) * 60_000
    panel: dict[str, MinuteSeries] = {}
    archives = []
    series_evidence = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        trend = 100.0 + symbol_index * 20.0 + np.arange(rows) * 0.0001
        wave = np.sin(np.arange(rows) / 31.0) * 0.01
        close = trend + wave
        open_values = close - 0.001
        high = np.maximum(open_values, close) + 0.01
        low = np.minimum(open_values, close) - 0.01
        volume = 10.0 + np.cos(np.arange(rows) / 17.0)
        quote_volume = volume * close
        trade_count = np.full(rows, 20, dtype=np.int64)
        taker_quote = quote_volume * (0.5 + 0.05 * np.sin(np.arange(rows) / 23.0))
        panel[symbol] = MinuteSeries(
            symbol=symbol,
            open_time_ms=timestamps.copy(),
            open=open_values,
            high=high,
            low=low,
            close=close,
            volume=volume,
            quote_volume=quote_volume,
            trade_count=trade_count,
            taker_buy_base_volume=volume * 0.5,
            taker_buy_quote_volume=taker_quote,
        )
        archives.append(
            ArchiveEvidence(symbol, 1, "2021-12", "2022-01", rows, 0)
        )
        series_evidence.append(
            SeriesEvidence(
                symbol=symbol,
                rows=rows,
                first_open_time_ms=int(timestamps[0]),
                last_open_time_ms=int(timestamps[-1]),
                gap_count=0,
                duplicate_or_regressed_time_count=0,
                nonfinite_numeric_rows=0,
                invalid_ohlc_rows=0,
                invalid_volume_rows=0,
                source_counts={"test": rows},
                stream_sha256="1" * 64,
            )
        )
    evidence = SourceEvidence(
        database_path="synthetic-test-only",
        materialized_start="2021-12-01",
        materialized_end="2022-01-01",
        archive_evidence=tuple(archives),
        series_evidence=tuple(series_evidence),
        panel_stream_sha256="2" * 64,
        selection_confirmation_or_terminal_rows_read=False,
    )
    return panel, evidence


def test_cross_asset_features_are_causal_and_targets_use_delayed_entry() -> None:
    panel, evidence = _minute_panel()
    dataset = build_cross_asset_dataset(panel, evidence)
    assert dataset.rows > 0
    assert dataset.features.dtype == np.float32
    assert np.isfinite(dataset.features).all()

    row = 3
    symbol_index = int(dataset.symbol_index[row])
    symbol = SYMBOLS[symbol_index]
    decision_time = int(dataset.decision_time_ms[row])
    source_index = int(
        np.searchsorted(panel[symbol].open_time_ms, decision_time)
    )
    expected = (
        np.log(panel[symbol].open[source_index + 1 + 15])
        - np.log(panel[symbol].open[source_index + 1])
    ) * 10_000.0
    assert float(dataset.gross_return_bps[15][row]) == np.float32(expected)

    changed = dict(panel)
    future = panel["BTCUSDT"]
    changed_close = future.close.copy()
    changed_close[source_index + 5 :] *= 1.5
    changed["BTCUSDT"] = replace(future, close=changed_close)
    changed_dataset = build_cross_asset_dataset(changed, evidence)
    assert np.array_equal(dataset.features[row], changed_dataset.features[row])


def _replay_dataset() -> CrossAssetDataset:
    start = int(datetime(2024, 10, 1, tzinfo=UTC).timestamp() * 1000)
    times = []
    symbols = []
    actual = []
    for symbol_index in range(3):
        for offset in (0, 5, 20, 40):
            times.append(start + offset * 60_000)
            symbols.append(symbol_index)
            actual.append(25.0 if offset != 5 else -5.0)
    time_array = np.asarray(times, dtype=np.int64)
    symbol_array = np.asarray(symbols, dtype=np.int8)
    role = np.ones(len(times), dtype=bool)
    masks = {
        "training": np.zeros(len(times), dtype=bool),
        "early_stop": np.zeros(len(times), dtype=bool),
        "calibration": role,
        "viability": np.zeros(len(times), dtype=bool),
        "selection_confirmation": np.zeros(len(times), dtype=bool),
        "terminal": np.zeros(len(times), dtype=bool),
    }
    source = SourceEvidence(
        database_path="test",
        materialized_start="2024-10-01",
        materialized_end="2024-10-01",
        archive_evidence=(),
        series_evidence=(),
        panel_stream_sha256="3" * 64,
        selection_confirmation_or_terminal_rows_read=False,
    )
    return CrossAssetDataset(
        feature_names=("x",),
        features=np.ones((len(times), 1), dtype=np.float32),
        decision_time_ms=time_array,
        symbol_index=symbol_array,
        gross_return_bps={15: np.asarray(actual, dtype=np.float32)},
        persistence_prediction_bps={15: np.zeros(len(times), dtype=np.float32)},
        role_masks={15: masks},
        source_evidence=source,
    )


def test_nonoverlap_replay_rejects_overlapping_candidates_and_charges_cost() -> None:
    dataset = _replay_dataset()
    prediction = np.full(dataset.rows, 20.0, dtype=np.float32)
    replay = replay_nonoverlapping(
        dataset,
        prediction,
        horizon=15,
        role="calibration",
        threshold_bps=15.0,
        bootstrap_samples=50,
    )
    assert replay.total_trades == 9
    assert replay.overlap_rejections == 3
    assert replay.trades_by_symbol == {symbol: 3 for symbol in SYMBOLS}
    assert replay.mean_net_bps == 13.0
    assert replay.total_net_bps == 117.0


def test_ai_veto_schema_is_bounded_and_failure_is_zero_risk() -> None:
    valid = _decision(
        {
            "message": {
                "content": (
                    '{"action":"approve","risk_multiplier":0.6,'
                    '"confidence":0.8,"reason_codes":["edge_covers_cost"],'
                    '"summary":"Cost margin and analogs are coherent."}'
                )
            }
        }
    )
    assert valid.valid is True
    assert valid.action == "approve"
    assert valid.risk_multiplier == 0.6

    failed = _failed_decision("timeout")
    assert failed.valid is False
    assert failed.action == "veto"
    assert failed.risk_multiplier == 0.0
