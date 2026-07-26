from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simple_ai_trading.impact_absorption_event_scaling import (
    Round74EventFeatureScaler,
    fit_round74_event_feature_scaler,
    fit_round74_event_feature_scaler_stream,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
)


def _training(rows: int = 100) -> np.ndarray:
    generator = np.random.default_rng(7410)
    values = generator.normal(
        size=(rows, len(ROUND74_EVENT_FEATURE_NAMES))
    ).astype(np.float64)
    values[:, :8] = 0.0
    for index in range(rows):
        values[index, index % 5] = 1.0
        values[index, 5 + index % 3] = 1.0
    values[:, ROUND74_EVENT_FEATURE_NAMES.index("depth_update_is_stale")] = 0.0
    return values


def test_round74_scaler_preserves_one_hot_and_zeros_constant_features() -> None:
    training = _training()
    scaler = fit_round74_event_feature_scaler(
        training,
        partition_role="training",
    )

    transformed = scaler.transform(training[:7])

    np.testing.assert_array_equal(transformed[:, :8], training[:7, :8])
    constant_index = ROUND74_EVENT_FEATURE_NAMES.index("depth_update_is_stale")
    assert scaler.constant_mask[constant_index]
    assert np.all(transformed[:, constant_index] == 0.0)
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all()


def test_round74_scaler_clip_statistics_are_training_only() -> None:
    training = _training()
    scaler = fit_round74_event_feature_scaler(
        training,
        partition_role="training",
    )
    feature_index = ROUND74_EVENT_FEATURE_NAMES.index("spread_bps")
    validation = training[:1].copy()
    validation[0, feature_index] = 1e12

    transformed = scaler.transform(validation)

    expected = (
        scaler.upper_clip[feature_index] - scaler.median[feature_index]
    ) / scaler.scale[feature_index]
    assert transformed[0, feature_index] == pytest.approx(expected)
    assert scaler.upper_clip[feature_index] < 1e12


def test_round74_scaler_does_not_erase_rare_training_flow() -> None:
    training = _training()
    feature_index = ROUND74_EVENT_FEATURE_NAMES.index(
        "liquidation_signed_quote_scaled"
    )
    training[:, feature_index] = 0.0
    training[-1, feature_index] = -3.0

    scaler = fit_round74_event_feature_scaler(
        training,
        partition_role="training",
        maximum_fit_rows=10,
    )
    transformed = scaler.transform(training[[-1]])

    assert not scaler.constant_mask[feature_index]
    assert scaler.lower_clip[feature_index] == -3.0
    assert transformed[0, feature_index] < 0.0


def test_round74_scaler_sampling_and_serialization_are_deterministic() -> None:
    training = _training(rows=257)
    first = fit_round74_event_feature_scaler(
        training,
        partition_role="training",
        maximum_fit_rows=31,
    )
    second = fit_round74_event_feature_scaler(
        training.copy(),
        partition_role="training",
        maximum_fit_rows=31,
    )

    assert first.fit_input_rows == 257
    assert first.fit_sample_rows == 31
    assert first.fit_sample_index_sha256 == second.fit_sample_index_sha256
    assert first.scaler_sha256 == second.scaler_sha256
    restored = Round74EventFeatureScaler.from_dict(first.as_dict())
    assert restored.scaler_sha256 == first.scaler_sha256
    np.testing.assert_array_equal(
        restored.transform(training[:5]),
        first.transform(training[:5]),
    )
    chunked = fit_round74_event_feature_scaler_stream(
        (training[:13], training[13:201], training[201:]),
        partition_role="training",
        maximum_fit_rows=31,
    )
    assert chunked.scaler_sha256 == first.scaler_sha256
    with pytest.raises(ValueError, match="sampling contract"):
        replace(first, fit_sampling_seed=first.fit_sampling_seed + 1)


def test_round74_scaler_rejects_nontraining_or_malformed_input() -> None:
    training = _training()
    with pytest.raises(ValueError, match="training partition"):
        fit_round74_event_feature_scaler(
            training,
            partition_role="validation",
        )
    malformed = training.copy()
    malformed[0, :5] = 0.0
    with pytest.raises(ValueError, match="one-hot"):
        fit_round74_event_feature_scaler(
            malformed,
            partition_role="training",
        )
    nonfinite = training.copy()
    nonfinite[0, -1] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        fit_round74_event_feature_scaler(
            nonfinite,
            partition_role="training",
        )
    with pytest.raises(ValueError, match="at least two"):
        fit_round74_event_feature_scaler_stream(
            (),
            partition_role="training",
        )
    with pytest.raises(ValueError, match="maximum fit rows"):
        fit_round74_event_feature_scaler(
            training,
            partition_role="training",
            maximum_fit_rows=2.5,
        )
