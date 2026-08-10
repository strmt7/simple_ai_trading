from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.compute import resolve_backend
import simple_ai_trading.polymarket_round21_tcn as tcn_module
from simple_ai_trading.polymarket_round21_tcn import (
    Round21CompiledTCNPredictor,
    build_round21_tcn_sequences,
    fit_round21_tcn,
    predict_round21_tcn,
    round21_tcn_parameter_count,
    validate_round21_tcn_payload,
)


def test_round21_tcn_partial_batch_does_not_overweight_its_conditions() -> None:
    groups = (
        np.asarray([0], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([3, 4, 5, 6], dtype=np.int64),
    )

    weights = tcn_module._condition_equal_batch_weights(groups)

    condition_weight = 1.0 / tcn_module.ROUND21_TCN_CONDITIONS_PER_BATCH
    assert np.isclose(float(np.sum(weights)), 3.0 * condition_weight)
    assert np.allclose(
        (
            np.sum(weights[:1]),
            np.sum(weights[1:3]),
            np.sum(weights[3:]),
        ),
        (condition_weight,) * 3,
    )


def test_round21_tcn_training_endpoints_rotate_within_fixed_strata() -> None:
    rows_per_condition = 1_200
    conditions = np.asarray(
        ["condition-a"] * rows_per_condition + ["condition-b"] * rows_per_condition,
        dtype=object,
    )

    first = tcn_module._condition_endpoints(conditions, epoch=1, seed=21_021)
    repeated = tcn_module._condition_endpoints(conditions, epoch=1, seed=21_021)
    later = tcn_module._condition_endpoints(conditions, epoch=2, seed=21_021)
    validation = tcn_module._condition_endpoints(conditions)

    assert all(
        np.array_equal(left, right) for left, right in zip(first, repeated, strict=True)
    )
    assert all(
        len(group) == tcn_module.ROUND21_TCN_ENDPOINTS_PER_CONDITION for group in first
    )
    assert all(
        not np.array_equal(left, right)
        for left, right in zip(first, later, strict=True)
    )
    assert all(
        not np.array_equal(left, right)
        for left, right in zip(first, validation, strict=True)
    )
    epochs = tuple(
        tcn_module._condition_endpoints(conditions, epoch=epoch, seed=21_021)[0]
        for epoch in range(1, 33)
    )
    assert all(
        len({int(group[stratum]) for group in epochs}) == 32 for stratum in range(8)
    )
    assert all(
        np.ptp(np.asarray([group[stratum] for group in epochs], dtype=np.int64))
        >= rows_per_condition // 10
        for stratum in range(8)
    )
    for condition_index, group in enumerate(first):
        condition_start = condition_index * rows_per_condition
        relative = group - condition_start
        for stratum, endpoint in enumerate(relative):
            left = (stratum * rows_per_condition) // len(group)
            right = ((stratum + 1) * rows_per_condition) // len(group)
            assert left <= endpoint < right


def test_round21_tcn_training_endpoint_schedule_rejects_invalid_seed_or_epoch() -> None:
    conditions = np.asarray(["condition-a"] * 8, dtype=object)

    with pytest.raises(ValueError, match="training seed"):
        tcn_module._condition_endpoints(conditions, epoch=1, seed=-1)
    with pytest.raises(ValueError, match="training seed"):
        tcn_module._condition_endpoints(conditions, epoch=1, seed=21_021.0)
    with pytest.raises(ValueError, match="sampling epoch"):
        tcn_module._condition_endpoints(conditions, epoch=0, seed=21_021)
    with pytest.raises(ValueError, match="sampling epoch"):
        tcn_module._condition_endpoints(conditions, epoch=1.0, seed=21_021)


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


def test_round21_tcn_prediction_computes_history_once_for_all_batches(
    monkeypatch,
) -> None:
    values = np.arange(15, dtype=np.float32).reshape(5, 3)
    conditions = np.asarray(["a"] * 3 + ["b"] * 2, dtype=object)
    decisions = np.asarray([0, 250, 500, 1_000, 1_250], dtype=np.int64)
    structural = np.zeros(len(values), dtype=np.float32)
    calls = 0
    original = tcn_module._history_starts

    def counted_history_starts(condition_ids, decision_time_ms):
        nonlocal calls
        calls += 1
        return original(condition_ids, decision_time_ms)

    monkeypatch.setattr(tcn_module, "_history_starts", counted_history_starts)
    monkeypatch.setattr(tcn_module, "ROUND21_TCN_PREDICTION_BATCH_SIZE", 2)
    result = tcn_module._predict(
        tcn_module._model(values.shape[1]),
        values,
        conditions,
        decisions,
        structural,
        device="cpu",
        endpoints=np.arange(len(values), dtype=np.int64),
    )

    assert calls == 1
    assert result.shape == (len(values),)
    assert np.all((result > 0.0) & (result < 1.0))


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
    assert fitted.payload["training_seed"] == 21_021
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
    compiled = Round21CompiledTCNPredictor(
        fitted.payload,
        feature_width=matrix.shape[1],
    )
    assert compiled.training_backend_kind == "cpu"
    assert compiled.backend.kind == "cpu"
    assert compiled.backend_substituted is False
    assert compiled.accelerator_fallback is False
    first = compiled.predict(
        matrix=matrix,
        structural_log_odds=structural,
        condition_ids=conditions,
        decision_time_ms=decisions,
    )
    second = compiled.predict(
        matrix=matrix,
        structural_log_odds=structural,
        condition_ids=conditions,
        decision_time_ms=decisions,
    )
    assert np.array_equal(first, predictions)
    assert np.array_equal(second, first)

    cpu_backend = resolve_backend("cpu", require=True)
    monkeypatch.setattr(
        tcn_module,
        "resolve_backend",
        lambda _requested, *, require: (
            cpu_backend if require is False else pytest.fail()
        ),
    )
    fallback = Round21CompiledTCNPredictor(
        {
            **fitted.payload,
            "backend_kind": "directml",
            "backend_device": "privateuseone:0",
        },
        feature_width=matrix.shape[1],
    )
    assert fallback.training_backend_kind == "directml"
    assert fallback.backend.kind == "cpu"
    assert fallback.backend_substituted is True
    assert fallback.accelerator_fallback is True

    tampered = {**fitted.payload, "unexpected": True}
    assert not validate_round21_tcn_payload(tampered, feature_width=3)
    tampered = {**fitted.payload, "state_sha256": "0" * 64}
    assert not validate_round21_tcn_payload(tampered, feature_width=3)
