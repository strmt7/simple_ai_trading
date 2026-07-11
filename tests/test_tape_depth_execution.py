from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pytest

from simple_ai_trading.tape_depth_execution import (
    TapeDepthExecutionAssumptions,
    evaluate_tape_depth_taker_execution,
    load_tape_depth_execution_confirmation_design,
)
from simple_ai_trading.tape_depth_model import (
    TapeDepthPredictionBatch,
    TapeDepthSignalPolicy,
)


class _Warehouse:
    def __init__(self) -> None:
        self.connection = duckdb.connect(":memory:")
        self.connection.execute(
            """
            CREATE TABLE current_book_ticker_100ms (
                symbol VARCHAR,
                bucket_ms BIGINT,
                close_bid DOUBLE,
                close_ask DOUBLE,
                close_bid_qty DOUBLE,
                close_ask_qty DOUBLE,
                last_transaction_time_ms BIGINT,
                available_time_ms BIGINT
            )
            """
        )
        self.certificate_kwargs: dict[str, object] | None = None

    def connect(self):
        return self.connection

    def require_corpus_certificate(self, symbol: str, **kwargs: object):
        self.certificate_kwargs = {"symbol": symbol, **kwargs}
        return {
            "status": "pass",
            "verified": True,
            "symbol": symbol,
            "certificate_sha256": "a" * 64,
        }


def _prediction_batch() -> TapeDepthPredictionBatch:
    decisions = np.arange(5, dtype=np.int64) * 4_000 + 1_000
    sides = np.asarray([1, -1, 1, -1, 1], dtype=np.float64)
    return TapeDepthPredictionBatch(
        decision_time_ms=decisions,
        target_entry_time_ms=decisions + 1_000,
        target_exit_time_ms=decisions + 3_000,
        actual_gross_return_bps=sides * 20.0,
        direction_probability=np.where(sides > 0.0, 0.8, 0.2),
        mean_prediction_bps=sides * 2.0,
        lower_prediction_bps=sides * 2.0 - 0.5,
        upper_prediction_bps=sides * 2.0 + 0.5,
        signal_policy=TapeDepthSignalPolicy(
            risk_level="regular",
            magnitude_quantile=0.90,
            direction_confidence_quantile=0.90,
            minimum_direction_probability=0.55,
            interval_width_quantile=0.90,
            signal_threshold_bps=1.0,
            maximum_interval_width_bps=2.0,
            direction_baseline_probability=0.5,
        ),
    )


def test_tape_depth_taker_execution_uses_real_quote_sides_fees_and_participation() -> None:
    warehouse = _Warehouse()
    batch = _prediction_batch()
    rows = []
    sides = batch.action_sides()
    for index, side in enumerate(sides):
        entry_ms = int(batch.target_entry_time_ms[index])
        exit_ms = int(batch.target_exit_time_ms[index])
        if side == 1:
            entry_bid, entry_ask = 99.99, 100.00
            exit_bid, exit_ask = 100.20, 100.21
        else:
            entry_bid, entry_ask = 100.20, 100.21
            exit_bid, exit_ask = 99.99, 100.00
        rows.extend(
            (
                (
                    "BTCUSDT",
                    entry_ms - 200,
                    entry_bid,
                    entry_ask,
                    100.0,
                    100.0,
                    entry_ms - 150,
                    entry_ms - 100,
                ),
                (
                    "BTCUSDT",
                    exit_ms - 200,
                    exit_bid,
                    exit_ask,
                    100.0,
                    100.0,
                    exit_ms - 150,
                    exit_ms - 100,
                ),
            )
        )
    warehouse.connect().executemany(
        "INSERT INTO current_book_ticker_100ms VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    report, evidence_rows = evaluate_tape_depth_taker_execution(
        warehouse,  # type: ignore[arg-type]
        symbol="BTCUSDT",
        predictions=batch,
        assumptions=TapeDepthExecutionAssumptions(
            taker_fee_bps_per_side=5.0,
            reference_order_notional_quote=1_000.0,
            max_l1_participation=0.20,
        ),
    )

    assert report.status == "after_cost_diagnostic_candidate"
    assert report.trading_authority is False
    assert report.execution_claim is False
    assert report.profitability_claim is False
    assert report.metrics.selected_signal_rows == 5
    assert report.metrics.executable_rows == 5
    assert report.metrics.executable_long_rows == 3
    assert report.metrics.executable_short_rows == 2
    assert report.metrics.mean_net_return_bps > 0.0
    assert report.metrics.positive_net_rate == 1.0
    assert all(row.status == "executable" for row in evidence_rows)
    long_row = next(row for row in evidence_rows if row.side == 1)
    short_row = next(row for row in evidence_rows if row.side == -1)
    assert long_row.quote_path_gross_bps == pytest.approx(
        (100.20 / 100.00 - 1.0) * 10_000.0
    )
    assert short_row.quote_path_gross_bps == pytest.approx(
        (1.0 - 100.00 / 100.20) * 10_000.0
    )
    assert long_row.fee_cost_bps == pytest.approx(5.0 * (1.0 + 100.20 / 100.00))
    assert short_row.fee_cost_bps == pytest.approx(5.0 * (1.0 + 100.00 / 100.20))
    assert long_row.net_return_bps == pytest.approx(
        long_row.quote_path_gross_bps - long_row.fee_cost_bps
    )
    assert short_row.net_return_bps == pytest.approx(
        short_row.quote_path_gross_bps - short_row.fee_cost_bps
    )
    assert warehouse.certificate_kwargs is not None
    assert warehouse.certificate_kwargs["required_data_types"] == ("bookTicker",)


def test_tape_depth_taker_execution_rejects_stale_or_oversized_quotes() -> None:
    warehouse = _Warehouse()
    batch = _prediction_batch()
    first_entry = int(batch.target_entry_time_ms[0])
    last_exit = int(batch.target_exit_time_ms[-1])
    warehouse.connect().executemany(
        "INSERT INTO current_book_ticker_100ms VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "BTCUSDT",
                first_entry - 2_100,
                99.99,
                100.00,
                0.001,
                0.001,
                first_entry - 2_000,
                first_entry - 1_900,
            ),
            (
                "BTCUSDT",
                last_exit - 200,
                100.20,
                100.21,
                0.001,
                0.001,
                last_exit - 150,
                last_exit - 100,
            ),
        ],
    )

    report, rows = evaluate_tape_depth_taker_execution(
        warehouse,  # type: ignore[arg-type]
        symbol="BTCUSDT",
        predictions=batch,
    )

    assert report.status == "rejected"
    assert report.metrics.executable_rows == 0
    assert report.metrics.rejected_quote_rows > 0
    assert "quote_path_incomplete_or_stale" in report.rejection_reasons
    assert all(row.status == "rejected" for row in rows)


def test_tape_depth_execution_assumptions_fail_closed() -> None:
    with pytest.raises(ValueError, match="assumptions"):
        TapeDepthExecutionAssumptions(max_l1_participation=0.0)
    with pytest.raises(ValueError, match="assumptions"):
        TapeDepthExecutionAssumptions(suppress_overlapping_positions=False)


def test_tape_depth_execution_confirmation_design_is_hash_and_source_bound(
    tmp_path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    design_path = root / "docs/model-research/tape-depth/confirmation-design.json"
    availability_path = root / "docs/microstructure/availability.json"

    design, digest = load_tape_depth_execution_confirmation_design(
        design_path,
        availability_path=availability_path,
    )

    assert digest == design["design_sha256"]
    assert design["confirmation_periods"] == [
        "2023-07-08",
        "2023-10-21",
        "2024-02-05",
    ]
    design["candidate"]["horizon_seconds"] = 30
    tampered = tmp_path / "tampered-design.json"
    tampered.write_text(json.dumps(design), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable contract"):
        load_tape_depth_execution_confirmation_design(
            tampered,
            availability_path=availability_path,
        )

    changed_availability = tmp_path / "availability.json"
    changed_availability.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="availability evidence differs"):
        load_tape_depth_execution_confirmation_design(
            design_path,
            availability_path=changed_availability,
        )
