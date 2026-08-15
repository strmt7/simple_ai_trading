from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np
import pytest

import simple_ai_trading.polymarket_round27_model as subject
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    Round27FeatureRow,
)
from simple_ai_trading.polymarket_round27_model import (
    Round27L2OffsetModel,
    Round27Partition,
    build_round27_model_samples,
    fit_round27_l2_offset,
    fit_round27_lightgbm_offset,
    paired_round27_condition_bootstrap,
    round27_corrected_politis_white_block_length,
    round27_model_from_payload,
    round27_probability_metrics,
    round27_stationary_bootstrap_mean_interval,
    scale_round27_probability_model,
    select_round27_correction_scale,
    select_round27_l2_penalty,
)


_BASE = 1_786_784_400_000


def _rows_and_outcomes(condition_count: int = 40):
    rows: list[Round27FeatureRow] = []
    outcomes: dict[str, int] = {}
    for condition_index in range(condition_count):
        condition_id = "0x" + f"{condition_index + 1:064x}"
        event_start = _BASE + condition_index * 300_000
        target = condition_index % 2
        outcomes[condition_id] = target
        direction = 1.0 if target else -1.0
        for row_index, offset in enumerate((30_000, 120_000, 240_000)):
            values = [0.0] * len(POLYMARKET_ROUND27_FEATURE_NAMES)
            values[0] = direction * (1.0 + row_index * 0.1)
            values[1] = float(300 - offset // 1_000)
            rows.append(
                Round27FeatureRow.create(
                    run_id="run-a",
                    condition_id=condition_id,
                    event_start_ms=event_start,
                    decision_time_ms=event_start + offset,
                    market_prior_probability=0.52 if target else 0.48,
                    values=values,
                    maximum_receipt_wall_ms=event_start + offset - 1,
                    source_chain_sha256="a" * 64,
                )
            )
    return tuple(rows), outcomes


def _samples(condition_count: int = 40):
    rows, outcomes = _rows_and_outcomes(condition_count)
    end = _BASE + condition_count * 300_000
    return build_round27_model_samples(
        rows_by_slot={"stage1-a": rows},
        outcomes_up=outcomes,
        role_intervals=(
            {
                "role": "train",
                "slot_id": "stage1-a",
                "start_ms": _BASE,
                "end_ms": end,
            },
        ),
    )


def test_model_samples_weight_each_condition_once() -> None:
    samples = _samples()

    conditions = sorted({sample.condition_id for sample in samples})
    assert len(conditions) == 40
    for condition in conditions:
        assert (
            math.fsum(
                sample.condition_weight
                for sample in samples
                if sample.condition_id == condition
            )
            == 1.0
        )


def test_l2_offset_model_improves_a_causal_synthetic_signal() -> None:
    partition = Round27Partition.from_samples(_samples(), role="train")

    penalty, scores = select_round27_l2_penalty(partition)
    model = fit_round27_l2_offset(partition, penalty=penalty)
    prediction = model.predict(partition.features, partition.offsets)
    prior = 1.0 / (1.0 + np.exp(-partition.offsets))
    model_metrics = round27_probability_metrics(partition, prediction)
    prior_metrics = round27_probability_metrics(partition, prior)

    assert set(scores) == {"0.01", "0.1", "1.0", "10.0", "100.0"}
    assert model_metrics.log_loss < prior_metrics.log_loss
    assert model_metrics.brier_score < prior_metrics.brier_score
    assert model_metrics.balanced_accuracy == 1.0
    assert np.all((prediction > 0.0) & (prediction < 1.0))

    restored = round27_model_from_payload(model.asdict())
    assert np.array_equal(
        restored.predict(partition.features, partition.offsets),
        prediction,
    )


def test_l2_penalty_folds_are_embargoed_and_strictly_walk_forward() -> None:
    partition = Round27Partition.from_samples(_samples(), role="train")

    folds = subject._walk_forward_condition_folds(partition, fold_count=5)

    assert len(folds) == 5
    validation_conditions: set[str] = set()
    for train, validation in folds:
        train_ids = {str(value) for value in train.conditions}
        held_ids = {str(value) for value in validation.conditions}
        assert train_ids.isdisjoint(held_ids)
        assert validation_conditions.isdisjoint(held_ids)
        validation_conditions.update(held_ids)
        assert (
            max(sample.event_start_ms + 300_000 for sample in train.samples)
            + 600_000
            <= min(sample.event_start_ms for sample in validation.samples)
        )


def test_correction_scale_can_fail_closed_to_the_market_prior() -> None:
    partition = Round27Partition.from_samples(_samples(), role="train")
    feature_count = len(POLYMARKET_ROUND27_FEATURE_NAMES)
    harmful = Round27L2OffsetModel(
        mean=(0.0,) * feature_count,
        scale=(1.0,) * feature_count,
        coefficients=(0.0, -100.0, *((0.0,) * (feature_count - 1))),
        penalty=1.0,
        correction_scale=1.0,
        model_sha256="b" * 64,
    )

    scale, scores = select_round27_correction_scale(harmful, partition)

    assert scale == 0.0
    assert scores["0.0"] < scores["1.0"]


def test_calibration_scale_is_bound_into_l2_model_identity() -> None:
    partition = Round27Partition.from_samples(_samples(), role="train")
    model = fit_round27_l2_offset(partition, penalty=1.0)

    scaled = scale_round27_probability_model(model, 0.5)
    restored = round27_model_from_payload(scaled.asdict())

    assert scaled.correction_scale == 0.5
    assert scaled.model_sha256 != model.model_sha256
    assert restored.asdict() == scaled.asdict()


def test_condition_bootstrap_reports_paired_log_loss_delta() -> None:
    partition = Round27Partition.from_samples(_samples(), role="train")
    prior = 1.0 / (1.0 + np.exp(-partition.offsets))
    candidate = np.where(partition.targets == 1.0, 0.9, 0.1)

    result = paired_round27_condition_bootstrap(
        partition,
        prior,
        candidate,
        draws=1_000,
    )

    assert result["condition_count"] == 40
    assert result["mean_candidate_minus_prior_log_loss"] < 0.0
    assert result["mean_candidate_minus_prior_brier_score"] < 0.0
    assert result["ci95_upper"] < 0.0
    assert result["brier_score"]["ci95_upper"] < 0.0
    assert result["log_loss"]["ci95_upper"] == result["ci95_upper"]
    assert result["method"] == (
        "stationary_bootstrap_block_length_sensitivity_envelope"
    )
    assert set(result["fixed_expected_block_lengths_conditions"]) == {1, 4}
    assert result["automatic_expected_block_length_conditions"] in result[
        "expected_block_lengths_conditions"
    ]
    intervals = result["block_intervals"]
    assert result["ci95_lower"] == min(item["ci95_lower"] for item in intervals)
    assert result["ci95_upper"] == max(item["ci95_upper"] for item in intervals)


def test_stationary_bootstrap_rejects_clustered_synthetic_profit() -> None:
    values = np.asarray([-2.4] * 20 + [2.6] * 40, dtype=np.float64)

    result = round27_stationary_bootstrap_mean_interval(
        values,
        draws=1_000,
        seed=27,
    )

    assert float(np.mean(values)) > 0.0
    assert result["fixed_expected_block_lengths_conditions"] == [1, 4, 12]
    assert result["automatic_expected_block_length_conditions"] == 11
    assert result["expected_block_lengths_conditions"] == [1, 4, 11, 12]
    assert result["block_intervals"][0]["ci95_lower"] > 0.0
    assert result["ci95_lower"] < 0.0


def test_corrected_politis_white_block_length_detects_dependence() -> None:
    independent = np.asarray([float(index % 2) for index in range(80)])
    dependent = np.repeat(np.asarray([-2.0, 2.0], dtype=np.float64), 40)

    assert round27_corrected_politis_white_block_length(independent) >= 1
    assert round27_corrected_politis_white_block_length(dependent) > 1
    assert round27_corrected_politis_white_block_length(np.ones(80)) == 1


def test_shallow_lightgbm_is_bounded_and_cpu_portable() -> None:
    partition = Round27Partition.from_samples(_samples(), role="train")

    model = fit_round27_lightgbm_offset(
        partition,
        compute_backend="cpu",
        seed=27,
    )
    prediction = model.predict(partition.features, partition.offsets)
    raw_correction = np.asarray(
        lgb.Booster(model_str=model.model_text).predict(
            partition.features,
            raw_score=True,
        ),
        dtype=np.float64,
    )
    expected = 1.0 / (1.0 + np.exp(-(partition.offsets + raw_correction)))

    assert model.backend_kind == "cpu"
    assert model.backend_device == "cpu"
    assert len(model.model_sha256) == 64
    assert np.all(np.isfinite(prediction))
    assert np.all((prediction > 0.0) & (prediction < 1.0))
    assert np.allclose(prediction, expected, rtol=0.0, atol=1e-15)

    payload = model.asdict()
    assert payload["model_text"] == model.model_text
    restored = round27_model_from_payload(payload)
    assert np.array_equal(
        restored.predict(partition.features, partition.offsets),
        prediction,
    )

    tampered = dict(payload)
    tampered["model_text"] = f"{payload['model_text']}#"
    with pytest.raises(ValueError, match="persisted LightGBM model differs"):
        round27_model_from_payload(tampered)


def test_calibration_scale_is_bound_into_lightgbm_model_identity() -> None:
    partition = Round27Partition.from_samples(_samples(), role="train")
    model = fit_round27_lightgbm_offset(partition, compute_backend="cpu", seed=27)

    scaled = scale_round27_probability_model(model, 0.25)
    restored = round27_model_from_payload(scaled.asdict())

    assert scaled.correction_scale == 0.25
    assert scaled.model_sha256 != model.model_sha256
    assert restored.asdict() == scaled.asdict()
