from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading import impact_absorption_shallow_model as subject


def _threshold_result(
    threshold: float,
    *,
    lower_expectancy_bps: float,
    completed_trades: int = 25,
) -> subject._ThresholdResult:
    return subject._ThresholdResult(
        threshold=threshold,
        completed_trades=completed_trades,
        attempted_actions=completed_trades,
        pre_entry_aborts=0,
        unresolved_risk_count=0,
        lower_expectancy_bps=lower_expectancy_bps,
        selected_row_indexes=np.arange(completed_trades, dtype=np.int64),
    )


def test_threshold_policy_rejects_nonpositive_tuning_lower_bound(monkeypatch) -> None:
    def evaluate(_dataset, _probability, _predicted_net_bps, *, threshold):
        return _threshold_result(threshold, lower_expectancy_bps=0.0)

    monkeypatch.setattr(subject, "_evaluate_tuning_threshold", evaluate)

    threshold, action_enabled, report = subject._select_threshold(
        object(),
        np.asarray([], dtype=np.float64),
        np.asarray([], dtype=np.float64),
    )

    assert threshold == 0.9
    assert action_enabled is False
    assert all(item["admissible"] is False for item in report)
    assert all(item["positive_lower_expectancy_required"] is True for item in report)


def test_threshold_policy_selects_largest_positive_lower_bound(monkeypatch) -> None:
    lower_by_threshold = {
        threshold: 0.1 for threshold in subject.ROUND73_PROBABILITY_THRESHOLDS
    }
    lower_by_threshold[0.7] = 0.4
    lower_by_threshold[0.8] = 0.4

    def evaluate(_dataset, _probability, _predicted_net_bps, *, threshold):
        return _threshold_result(
            threshold,
            lower_expectancy_bps=lower_by_threshold[threshold],
        )

    monkeypatch.setattr(subject, "_evaluate_tuning_threshold", evaluate)

    threshold, action_enabled, _report = subject._select_threshold(
        object(),
        np.asarray([], dtype=np.float64),
        np.asarray([], dtype=np.float64),
    )

    assert threshold == 0.8
    assert action_enabled is True


def test_tuning_overlap_guard_covers_both_execution_lateness_budgets() -> None:
    first_wall_ns = 10_000_000_000
    second_wall_ns = first_wall_ns + 60_750_000_000
    dataset = SimpleNamespace(
        role_mask=lambda role: np.ones(4, dtype=np.bool_) if role == "tuning" else None,
        outcome_status=np.asarray(
            [
                subject.ROUND73_OBSERVED_STATUS,
                subject.ROUND73_OBSERVED_STATUS,
                subject.ROUND73_OBSERVED_STATUS,
                subject.ROUND73_OBSERVED_STATUS,
            ],
            dtype=np.uint8,
        ),
        anchor_wall_ns=np.asarray(
            [first_wall_ns, first_wall_ns, second_wall_ns, second_wall_ns],
            dtype=np.int64,
        ),
        continuous_target_bps=np.asarray([2.0, -2.0, 2.0, -2.0]),
        run_id_binary=np.asarray([b"a" * 16, b"a" * 16, b"b" * 16, b"b" * 16]),
    )
    probability = np.asarray([0.9, 0.1, 0.9, 0.1])
    predicted_net_bps = np.asarray([1.0, -1.0, 1.0, -1.0])

    result = subject._evaluate_tuning_threshold(
        dataset,
        probability,
        predicted_net_bps,
        threshold=0.5,
    )

    assert subject._TUNING_MAXIMUM_POSITION_NS == 61_000_000_000
    assert result.attempted_actions == 1
    assert result.completed_trades == 1


def test_unresolved_tuning_position_blocks_later_same_symbol_actions() -> None:
    first_wall_ns = 10_000_000_000
    dataset = SimpleNamespace(
        role_mask=lambda role: np.ones(4, dtype=np.bool_) if role == "tuning" else None,
        outcome_status=np.asarray(
            [
                subject.ROUND73_POST_ENTRY_UNRESOLVED_STATUS,
                subject.ROUND73_POST_ENTRY_UNRESOLVED_STATUS,
                subject.ROUND73_OBSERVED_STATUS,
                subject.ROUND73_OBSERVED_STATUS,
            ],
            dtype=np.uint8,
        ),
        anchor_wall_ns=np.asarray(
            [
                first_wall_ns,
                first_wall_ns,
                first_wall_ns + 120_000_000_000,
                first_wall_ns + 120_000_000_000,
            ],
            dtype=np.int64,
        ),
        continuous_target_bps=np.asarray([np.nan, np.nan, 2.0, -2.0]),
        run_id_binary=np.asarray([b"a" * 16, b"a" * 16, b"b" * 16, b"b" * 16]),
    )
    probability = np.asarray([0.9, 0.1, 0.9, 0.1])
    predicted_net_bps = np.asarray([1.0, -1.0, 1.0, -1.0])

    result = subject._evaluate_tuning_threshold(
        dataset,
        probability,
        predicted_net_bps,
        threshold=0.5,
    )

    assert result.attempted_actions == 1
    assert result.unresolved_risk_count == 1
    assert result.completed_trades == 0


def test_linear_heads_fit_finite_signal_and_beat_constant_baselines() -> None:
    generator = np.random.default_rng(20260723)
    features = generator.normal(size=(512, 4))
    median, iqr, retained = subject._fit_linear_preprocessor(features)
    scaled = subject._scale_linear(
        features,
        median=median,
        iqr=iqr,
        retained=retained,
    )

    binary_target = (
        1.4 * scaled[:, 0] - 0.8 * scaled[:, 1] + 0.3 * scaled[:, 2] > 0.0
    ).astype(np.float64)
    logistic = subject._fit_logistic(scaled, binary_target)
    probability = subject._predict_linear(logistic, scaled)
    fitted_log_loss, fitted_brier = subject._binary_losses(binary_target, probability)
    prevalence = np.full(len(binary_target), np.mean(binary_target))
    baseline_log_loss, baseline_brier = subject._binary_losses(
        binary_target, prevalence
    )

    assert logistic["kind"] == "logistic_regression"
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert fitted_log_loss < baseline_log_loss
    assert fitted_brier < baseline_brier

    continuous_target = 2.0 * scaled[:, 0] - 0.7 * scaled[:, 1]
    continuous_target[::37] += 20.0
    huber = subject._fit_huber(scaled, continuous_target)
    prediction = subject._predict_linear(huber, scaled)
    fitted_mse, fitted_mae = subject._continuous_losses(continuous_target, prediction)
    baseline = np.full(len(continuous_target), np.median(continuous_target))
    baseline_mse, baseline_mae = subject._continuous_losses(continuous_target, baseline)

    assert huber["kind"] == "huber_regression"
    assert fitted_mse < baseline_mse
    assert fitted_mae < baseline_mae
    loss, gradient = subject._huber_loss_and_gradient(
        np.asarray([-10.0, -1.0, 0.0, 8.0])
    )
    assert np.all(loss >= 0.0)
    assert gradient.tolist() == [-5.0, -1.0, 0.0, 5.0]

    with pytest.raises(ValueError, match="single class"):
        subject._fit_logistic(scaled, np.zeros(len(scaled)))
    with pytest.raises(ValueError, match="serialized linear model"):
        subject._predict_linear({"kind": "huber_regression"}, scaled)
    with pytest.raises(ValueError, match="binary metric inputs"):
        subject._binary_losses(np.asarray([0.0, 2.0]), np.asarray([0.5, 0.5]))
    with pytest.raises(ValueError, match="continuous metric inputs"):
        subject._continuous_losses(np.asarray([]), np.asarray([]))


def test_lightgbm_heads_preserve_serialized_prediction_identity(monkeypatch) -> None:
    monkeypatch.setattr(subject, "_MAXIMUM_BOOSTING_ITERATIONS", 24)
    monkeypatch.setattr(subject, "_EARLY_STOPPING_ROUNDS", 5)
    monkeypatch.setattr(subject, "_MINIMUM_DATA_IN_LEAF", 10)
    generator = np.random.default_rng(20260724)
    features = generator.normal(size=(640, 4)).astype(np.float32)
    training_features = features[:480]
    tuning_features = features[480:]
    binary_target = (features[:, 0] + 0.4 * features[:, 1] > 0.0).astype(np.float64)
    continuous_target = 3.0 * features[:, 0] - features[:, 2]
    backend = {
        "device_type": "cpu",
        "num_threads": 1,
        "verbosity": -1,
        "seed": subject.ROUND73_MODEL_SEED,
        "deterministic": True,
        "force_col_wise": True,
    }

    binary_model, binary_iteration, train_probability, tune_probability, error = (
        subject._fit_lightgbm_head(
            training_features,
            binary_target[:480],
            tuning_features,
            binary_target[480:],
            feature_names=("a", "b", "c", "d"),
            backend_parameters=backend,
            objective="binary",
        )
    )
    continuous_model, continuous_iteration, train_value, tune_value, value_error = (
        subject._fit_lightgbm_head(
            training_features,
            continuous_target[:480],
            tuning_features,
            continuous_target[480:],
            feature_names=("a", "b", "c", "d"),
            backend_parameters=backend,
            objective="huber",
        )
    )

    assert "Tree=" in binary_model
    assert "Tree=" in continuous_model
    assert 1 <= binary_iteration <= 24
    assert 1 <= continuous_iteration <= 24
    assert train_probability.shape == (480,)
    assert tune_probability.shape == (160,)
    assert train_value.shape == (480,)
    assert tune_value.shape == (160,)
    assert np.all((tune_probability >= 0.0) & (tune_probability <= 1.0))
    assert error <= 1e-12
    assert value_error <= 1e-12


def test_prediction_artifact_round_trip_is_exact_and_tamper_evident() -> None:
    rows = 9
    row_indexes = np.asarray([1, 4, 8], dtype=np.int64)
    predictions = {
        candidate: (
            np.linspace(0.1, 0.9, rows) + candidate_index * 0.001,
            np.linspace(-4.0, 4.0, rows) + candidate_index,
        )
        for candidate_index, candidate in enumerate(subject.ROUND73_SHALLOW_CANDIDATES)
    }
    payload = subject.encode_round73_prediction_artifact(
        symbol="BTCUSDT",
        role="tuning",
        source_rows_sha256="a" * 64,
        row_indexes=row_indexes,
        predictions=predictions,
    )
    decoded = subject.decode_round73_prediction_artifact(payload)

    assert decoded["header"]["row_count"] == len(row_indexes)
    assert np.array_equal(decoded["row_indexes"], row_indexes)
    assert decoded["row_indexes"].flags.writeable is False
    for candidate in subject.ROUND73_SHALLOW_CANDIDATES:
        binary, continuous = decoded["predictions"][candidate]
        assert np.array_equal(binary, predictions[candidate][0][row_indexes])
        assert np.array_equal(continuous, predictions[candidate][1][row_indexes])
        assert binary.flags.writeable is False
        assert continuous.flags.writeable is False

    prepared = subject.Round73PreparedPretestArtifacts(
        model_manifest={"schema_version": "test"},
        artifacts={"predictions.bin": payload},
        symbol_reports=({"symbol": "BTCUSDT"},),
    ).as_dict()
    assert prepared["artifact_count"] == 1
    assert prepared["artifact_bytes"] == len(payload)
    assert prepared["test_target_read"] is False
    assert prepared["trading_authority"] is False

    with pytest.raises(ValueError, match="identity"):
        subject.encode_round73_prediction_artifact(
            symbol="DOGEUSDT",
            role="tuning",
            source_rows_sha256="a" * 64,
            row_indexes=row_indexes,
            predictions=predictions,
        )
    with pytest.raises(ValueError, match="framing"):
        subject.decode_round73_prediction_artifact(b"invalid")
    tampered = bytearray(payload)
    tampered[-1] ^= 1
    with pytest.raises(ValueError, match="decompression|payload"):
        subject.decode_round73_prediction_artifact(bytes(tampered))
