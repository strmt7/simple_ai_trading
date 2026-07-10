from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from simple_ai_trading.foundation_benchmark import (
    BINANCE_ARCHIVE_SOURCE,
    FOUNDATION_SELECTION_END_EXCLUSIVE_MS,
    FOUNDATION_SELECTION_START_MS,
    FOUNDATION_SYMBOLS,
    ForecastObservation,
    FoundationBenchmarkConfig,
    audit_archive_coverage,
    calibrate_forecast_amplitude,
    day_block_bootstrap,
    decision_timestamps,
    parse_utc_ms,
    summarize_metrics,
    write_foundation_benchmark_chart,
)


def _observation(symbol: str, day: int, predicted: float, actual: float) -> ForecastObservation:
    decision_ms = FOUNDATION_SELECTION_START_MS + day * 86_400_000
    return ForecastObservation(
        symbol=symbol,
        decision_ms=decision_ms,
        decision_time_utc=f"2024-07-{day + 1:02d}T00:00:00Z",
        last_close=100.0,
        predicted_average_return=predicted,
        actual_average_return=actual,
        predicted_final_return=predicted,
        actual_final_return=actual,
        absolute_error=abs(predicted - actual),
        random_walk_absolute_error=abs(actual),
        direction_correct=predicted * actual > 0.0,
        inference_batch=1,
    )


def test_config_seals_pretraining_and_terminal_periods() -> None:
    FoundationBenchmarkConfig().validated()
    with pytest.raises(ValueError, match="pretraining cutoff"):
        FoundationBenchmarkConfig(start_ms=FOUNDATION_SELECTION_START_MS - 1).validated()
    with pytest.raises(ValueError, match="sealed terminal"):
        FoundationBenchmarkConfig(
            end_exclusive_ms=FOUNDATION_SELECTION_END_EXCLUSIVE_MS + 1
        ).validated()
    with pytest.raises(ValueError, match="no-fallback"):
        FoundationBenchmarkConfig(lookback_bars=500, prediction_bars=13).validated()
    with pytest.raises(ValueError, match="capped at 3"):
        FoundationBenchmarkConfig(batch_size=4).validated()
    with pytest.raises(ValueError, match="worker_rotation_batches"):
        FoundationBenchmarkConfig(worker_rotation_batches=-1).validated()


def test_decision_sampling_is_aligned_unique_and_deterministic() -> None:
    config = FoundationBenchmarkConfig(
        end_exclusive_ms=FOUNDATION_SELECTION_START_MS + 10 * 86_400_000,
        samples_per_symbol=17,
    )
    first = decision_timestamps(config)
    second = decision_timestamps(config)

    assert first == second
    assert len(first) == 17
    assert len(set(first)) == 17
    assert all(value % 300_000 == 0 for value in first)


def test_metrics_and_day_block_bootstrap_measure_paired_random_walk_uplift() -> None:
    observations = tuple(
        _observation(symbol, day, actual * 0.9, actual)
        for day in range(10)
        for symbol, actual in zip(FOUNDATION_SYMBOLS, (0.01, -0.02, 0.03), strict=True)
    )

    metrics = summarize_metrics(observations)
    first = day_block_bootstrap(observations, samples=200, seed=17)
    second = day_block_bootstrap(observations, samples=200, seed=17)

    assert metrics["overall"]["mae_improvement"] > 0.0
    assert metrics["overall"]["direction_accuracy"] == 1.0
    assert all(metrics["by_symbol"][symbol]["mae_improvement"] > 0.0 for symbol in FOUNDATION_SYMBOLS)
    assert first == second
    assert first["ci_95_low"] > 0.0


def test_amplitude_calibration_uses_earlier_half_and_can_abstain() -> None:
    observations = tuple(
        _observation(
            symbol,
            day,
            (actual * 4.0 if symbol == "ETHUSDT" else -actual),
            actual,
        )
        for day in range(20)
        for symbol, actual in zip(FOUNDATION_SYMBOLS, (0.01, -0.02, 0.03), strict=True)
    )

    report, calibrated = calibrate_forecast_amplitude(
        observations,
        bootstrap_samples=200,
        seed=17,
    )

    assert report["boundary_utc"].startswith("2024-07-11")
    assert report["scales"]["BTCUSDT"] == 0.0
    assert report["scales"]["SOLUSDT"] == 0.0
    assert report["scales"]["ETHUSDT"] == pytest.approx(0.25)
    assert report["eligible_symbols"] == ["ETHUSDT"]
    assert len(calibrated) == 30


def test_archive_audit_requires_exact_minutes_sources_and_verified_months(
    tmp_path: Path,
) -> None:
    database = tmp_path / "market.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE candles (
            symbol TEXT, market_type TEXT, interval TEXT, open_time INTEGER,
            source TEXT, PRIMARY KEY(symbol, market_type, interval, open_time)
        );
        CREATE TABLE archive_files (
            symbol TEXT, market_type TEXT, interval TEXT, period TEXT,
            status TEXT, checksum_status TEXT
        );
        """
    )
    config = FoundationBenchmarkConfig(
        end_exclusive_ms=FOUNDATION_SELECTION_START_MS + 3_600_000,
        samples_per_symbol=2,
        lookback_bars=32,
        prediction_bars=1,
    )
    decisions = decision_timestamps(config)
    coverage_start = decisions[0] - config.lookback_bars * 300_000
    coverage_end = decisions[-1] + config.prediction_bars * 300_000
    rows = [
        (symbol, "futures", "1m", timestamp, BINANCE_ARCHIVE_SOURCE)
        for symbol in FOUNDATION_SYMBOLS
        for timestamp in range(coverage_start, coverage_end, 60_000)
    ]
    connection.executemany("INSERT INTO candles VALUES (?, ?, ?, ?, ?)", rows)
    connection.executemany(
        "INSERT INTO archive_files VALUES (?, 'futures', '1m', ?, 'complete', 'verified')",
        [
            (symbol, period)
            for symbol in FOUNDATION_SYMBOLS
            for period in ("2024-06", "2024-07")
        ],
    )
    connection.commit()

    evidence = audit_archive_coverage(connection, config, decisions)
    assert len(evidence) == 3
    connection.execute(
        "DELETE FROM candles WHERE symbol = ? AND open_time = ?",
        (FOUNDATION_SYMBOLS[0], coverage_start),
    )
    connection.commit()
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        audit_archive_coverage(connection, config, decisions)
    connection.close()


def test_parse_utc_ms_rejects_naive_timestamps() -> None:
    assert parse_utc_ms("2024-07-01T00:00:00Z") == FOUNDATION_SELECTION_START_MS
    with pytest.raises(ValueError, match="timezone"):
        parse_utc_ms("2024-07-01T00:00:00")


def test_foundation_chart_is_deterministic_and_labels_itself_not_pnl(tmp_path: Path) -> None:
    observations = tuple(
        _observation(symbol, day, actual * 0.8, actual)
        for day in range(3)
        for symbol, actual in zip(FOUNDATION_SYMBOLS, (0.01, -0.02, 0.03), strict=True)
    )
    first = tmp_path / "first.svg"
    second = tmp_path / "second.svg"

    first_hash = write_foundation_benchmark_chart(
        first,
        raw_observations=observations,
        calibrated_selection=observations,
        model_label="fixture",
    )
    second_hash = write_foundation_benchmark_chart(
        second,
        raw_observations=observations,
        calibrated_selection=observations,
        model_label="fixture",
    )

    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes()
    text = first.read_text(encoding="utf-8")
    assert "not P&amp;L" in text
    assert "BTCUSDT" in text and "ETHUSDT" in text and "SOLUSDT" in text
