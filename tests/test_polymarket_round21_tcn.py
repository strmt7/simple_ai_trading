from __future__ import annotations

import numpy as np

from simple_ai_trading.compute import resolve_backend
import simple_ai_trading.polymarket_round21_tcn as tcn_module
from simple_ai_trading.polymarket_round21_tcn import (
    build_round21_tcn_sequences,
    fit_round21_tcn,
    predict_round21_tcn,
    round21_tcn_parameter_count,
    validate_round21_tcn_payload,
)


def test_round21_tcn_sequence_resets_at_a_cadence_gap() -> None:
    values = np.arange(5, dtype=np.float32).reshape(-1, 1)
    conditions = np.asarray(["a"] * 5, dtype=object)
    decisions = np.asarray([0, 250, 500, 1_000, 1_250], dtype=np.int64)

    sequences = build_round21_tcn_sequences(
        values,
        conditions,
        decisions,
        np.asarray([2, 3, 4], dtype=np.int64),
    )

    assert sequences[0, -3:, 0].tolist() == [0.0, 1.0, 2.0]
    assert sequences[0, -3:, 1].tolist() == [1.0, 1.0, 1.0]
    assert np.count_nonzero(sequences[1, :, 1]) == 1
    assert sequences[1, -1, 0] == 3.0
    assert sequences[2, -2:, 0].tolist() == [3.0, 4.0]


def test_round21_tcn_cpu_fit_round_trips_exact_weights(monkeypatch) -> None:
    monkeypatch.setattr(tcn_module, "ROUND21_TCN_MAXIMUM_EPOCHS", 1)
    condition_count = 32
    rows_per_condition = 2
    condition_numbers = np.repeat(np.arange(condition_count), rows_per_condition)
    conditions = np.asarray(
        [f"condition-{value}" for value in condition_numbers],
        dtype=object,
    )
    decisions = np.concatenate(
        [
            np.asarray([value * 300_000, value * 300_000 + 250])
            for value in range(condition_count)
        ]
    ).astype(np.int64)
    labels = (condition_numbers % 2).astype(np.float32)
    sign = labels * 2.0 - 1.0
    matrix = np.column_stack(
        (
            sign,
            np.sin(condition_numbers * 0.17),
            np.cos(condition_numbers * 0.11),
        )
    ).astype(np.float32)
    structural = np.zeros(len(matrix), dtype=np.float32)
    progress_events: list[str] = []

    fitted = fit_round21_tcn(
        train_matrix=matrix,
        train_labels=labels,
        train_structural_log_odds=structural,
        train_condition_ids=conditions,
        train_decision_time_ms=decisions,
        stop_matrix=matrix,
        stop_labels=labels,
        stop_structural_log_odds=structural,
        stop_condition_ids=conditions,
        stop_decision_time_ms=decisions,
        backend=resolve_backend("cpu", require=True),
        seed=21_021,
        progress=lambda event, _payload: progress_events.append(event),
    )

    assert validate_round21_tcn_payload(
        fitted.payload,
        feature_width=matrix.shape[1],
    )
    assert fitted.payload["parameter_count"] == round21_tcn_parameter_count(3)
    assert "round21_tcn_batch" in progress_events
    assert "round21_tcn_epoch" in progress_events
    predictions = predict_round21_tcn(
        fitted.payload,
        matrix=matrix,
        structural_log_odds=structural,
        condition_ids=conditions,
        decision_time_ms=decisions,
    )
    assert predictions.shape == labels.shape
    assert np.all((predictions > 0.0) & (predictions < 1.0))

    tampered = {**fitted.payload, "unexpected": True}
    assert not validate_round21_tcn_payload(tampered, feature_width=3)
    tampered = {**fitted.payload, "state_sha256": "0" * 64}
    assert not validate_round21_tcn_payload(tampered, feature_width=3)
