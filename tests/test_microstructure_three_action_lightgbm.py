from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from simple_ai_trading.microstructure_action_features import (
    mirror_microstructure_direction,
)
from simple_ai_trading.microstructure_three_action_lightgbm import (
    THREE_ACTION_LIGHTGBM_SCHEMA_VERSION,
    ThreeActionLightGBMSpec,
    _action_labels,
    _fit_multiclass_calibration,
    _multiclass_log_loss,
    _regret_multiplier,
    _softmax,
    as_selective_action_ensemble,
    ensemble_three_action_predictions,
    load_three_action_lightgbm_model,
    predict_three_action_lightgbm_model,
    save_three_action_lightgbm_model,
    train_three_action_lightgbm_model,
)
from tests.test_microstructure_outcome_lightgbm import _dataset


def _spec(**overrides: object) -> ThreeActionLightGBMSpec:
    values: dict[str, object] = {
        "candidate_id": "test-utility-weighted-three-action",
        "family": "utility_weighted_symmetric_three_action_lightgbm_hurdle",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 32,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.005,
        "lambda_l2": 0.025,
        "max_bin": 63,
        "num_boost_round": 12,
        "early_stopping_rounds": 5,
        "lower_quantile": 0.10,
        "upper_quantile": 0.90,
        "calibration_fraction": 0.50,
        "gpu_use_dp_required": True,
    }
    values.update(overrides)
    return ThreeActionLightGBMSpec(**values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def three_action_bundle():
    dataset, targets = _dataset(rows=8_000)
    phases: list[tuple[str, int, int]] = []
    model = train_three_action_lightgbm_model(
        dataset,
        targets,
        train_endpoints=np.arange(0, 3_500, dtype=np.int64),
        tuning_endpoints=np.arange(4_000, 7_000, dtype=np.int64),
        spec=_spec(),
        compute_backend="cpu",
        seed=33,
        train_sample_weights=np.linspace(0.8, 1.2, 3_500, dtype=np.float32),
        tuning_sample_weights=np.ones(3_000, dtype=np.float32),
        progress=lambda name, step, total: phases.append((name, step, total)),
    )
    return dataset, targets, model, phases


def test_action_class_labels_are_not_side_profit_labels() -> None:
    long_values = np.asarray([2.0, 1.0, 0.0, -1.0])
    short_values = np.asarray([1.0, 2.0, 0.0, -2.0])

    np.testing.assert_array_equal(
        _action_labels(long_values, short_values),
        np.asarray([0, 2, 1, 1], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        long_values > 0.0,
        np.asarray([True, True, False, False]),
    )
    np.testing.assert_array_equal(
        short_values > 0.0,
        np.asarray([True, True, False, False]),
    )


def test_regret_multiplier_is_bounded() -> None:
    span = np.asarray([0.0, 5.0, 10.0, 30.0, 100.0])
    np.testing.assert_array_equal(
        _regret_multiplier(span, 10.0),
        np.asarray([0.5, 0.5, 1.0, 3.0, 3.0]),
    )


def test_projected_multiclass_calibration_is_bounded_and_loss_decreasing() -> None:
    rng = np.random.default_rng(41)
    labels = np.tile(np.asarray([0, 1, 2], dtype=np.int64), 400)
    logits = rng.normal(0.0, 1.0, size=(len(labels), 3))
    logits[np.arange(len(labels)), labels] += 0.35
    uncalibrated_loss = _multiclass_log_loss(_softmax(logits), labels)

    temperature, bias, iterations, gradient, loss, prior_loss = (
        _fit_multiclass_calibration(logits, labels)
    )

    assert np.exp(-4.0) <= temperature <= np.exp(4.0)
    assert -5.0 <= bias <= 5.0
    assert 1 <= iterations <= 100
    assert np.isfinite(gradient) and gradient >= 0.0
    assert loss <= uncalibrated_loss + 1e-12
    assert prior_loss > 0.0


def test_three_action_training_is_purged_and_non_authoritative(
    three_action_bundle,
) -> None:
    dataset, _targets, model, phases = three_action_bundle

    assert model.schema_version == THREE_ACTION_LIGHTGBM_SCHEMA_VERSION
    assert model.backend_kind == "cpu"
    assert model.target_scenario == "stress"
    assert model.training_event_rows == 3_500
    assert model.requested_tuning_event_rows == 3_000
    assert model.early_stop_event_rows == 1_320
    assert model.calibration_event_rows == 1_500
    assert model.internal_purged_event_rows == 180
    assert model.calibration_start_ms == int(dataset.decision_time_ms[5_500])
    assert set(model.action_class_support) == {"train", "early_stop", "calibration"}
    assert set(model.side_profit_class_support) == {
        "train",
        "early_stop",
        "calibration",
    }
    assert sum(model.side_profit_class_support["train"].values()) == 7_000
    assert 0.05 <= model.side_profit_probability_calibration[0] <= 10.0
    assert -10.0 <= model.side_profit_probability_calibration[1] <= 10.0
    assert len(model.best_iterations) == 6
    assert len(model.model_strings) == 6
    assert len(model.model_sha256) == 64
    assert [step for _name, step, _total in phases] == list(range(1, 7))
    assert all(total == 6 for _name, _step, total in phases)
    assert not model.trading_authority
    assert not model.execution_claim
    assert not model.profitability_claim
    assert not model.portfolio_claim
    assert not model.leverage_applied


def test_three_action_probabilities_keep_distinct_semantics(
    three_action_bundle,
) -> None:
    dataset, _targets, model, _phases = three_action_bundle
    endpoints = np.arange(7_100, 7_300, dtype=np.int64)
    prediction = predict_three_action_lightgbm_model(model, dataset, endpoints)

    np.testing.assert_allclose(
        prediction.long_action_probability
        + prediction.abstain_action_probability
        + prediction.short_action_probability,
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        prediction.opportunity_probability,
        prediction.long_action_probability + prediction.short_action_probability,
        rtol=0.0,
        atol=1e-12,
    )
    assert not np.allclose(
        prediction.long_profitable_probability,
        prediction.long_action_probability,
    )
    assert not np.allclose(
        prediction.short_profitable_probability,
        prediction.short_action_probability,
    )
    assert np.all(prediction.long_lower_bps <= prediction.long_upper_bps)
    assert np.all(prediction.short_lower_bps <= prediction.short_upper_bps)
    assert np.all(np.isfinite(prediction.long_mean_bps))
    assert np.all(np.isfinite(prediction.short_mean_bps))


def test_three_action_prediction_is_mirror_equivariant(
    three_action_bundle,
) -> None:
    dataset, _targets, model, _phases = three_action_bundle
    endpoints = np.arange(7_100, 7_160, dtype=np.int64)
    original = predict_three_action_lightgbm_model(model, dataset, endpoints)
    mirrored = predict_three_action_lightgbm_model(
        model,
        replace(
            dataset,
            features=mirror_microstructure_direction(dataset.features),
        ),
        endpoints,
    )

    for original_name, mirrored_name in (
        ("long_action_probability", "short_action_probability"),
        ("short_action_probability", "long_action_probability"),
        ("long_profitable_probability", "short_profitable_probability"),
        ("short_profitable_probability", "long_profitable_probability"),
        ("long_mean_bps", "short_mean_bps"),
        ("short_mean_bps", "long_mean_bps"),
        ("long_lower_bps", "short_lower_bps"),
        ("short_lower_bps", "long_lower_bps"),
        ("long_upper_bps", "short_upper_bps"),
        ("short_upper_bps", "long_upper_bps"),
    ):
        np.testing.assert_allclose(
            getattr(original, original_name),
            getattr(mirrored, mirrored_name),
            rtol=0.0,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        original.abstain_action_probability,
        mirrored.abstain_action_probability,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        original.conditional_long_probability,
        1.0 - mirrored.conditional_long_probability,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        original.action_preference_side,
        -mirrored.action_preference_side,
    )
    np.testing.assert_array_equal(
        original.decision_preference_side,
        -mirrored.decision_preference_side,
    )


def test_three_action_artifact_round_trip_is_exact(
    tmp_path,
    three_action_bundle,
) -> None:
    dataset, _targets, model, _phases = three_action_bundle
    artifact = tmp_path / "three-action.json"
    save_three_action_lightgbm_model(artifact, model)
    loaded = load_three_action_lightgbm_model(artifact)

    assert loaded == model
    endpoints = np.arange(7_100, 7_130, dtype=np.int64)
    expected = predict_three_action_lightgbm_model(model, dataset, endpoints)
    actual = predict_three_action_lightgbm_model(loaded, dataset, endpoints)
    for name in expected.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("tamper", ["authority", "model", "extra_field"])
def test_three_action_artifact_tampering_fails_closed(
    tmp_path,
    three_action_bundle,
    tamper: str,
) -> None:
    _dataset_value, _targets, model, _phases = three_action_bundle
    artifact = tmp_path / f"three-action-tampered-{tamper}.json"
    save_three_action_lightgbm_model(artifact, model)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if tamper == "authority":
        payload["trading_authority"] = True
    elif tamper == "model":
        payload["model_strings"]["side_profit_probability"] += "\n# tampered"
    else:
        payload["unbound"] = "value"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="scalar types|contract|invalid"):
        load_three_action_lightgbm_model(artifact)


def test_three_action_ensemble_preserves_policy_semantics(
    three_action_bundle,
) -> None:
    dataset, _targets, model, _phases = three_action_bundle
    endpoints = np.arange(7_100, 7_130, dtype=np.int64)
    prediction = predict_three_action_lightgbm_model(model, dataset, endpoints)
    ensemble = ensemble_three_action_predictions((prediction, prediction))
    policy_input = as_selective_action_ensemble(ensemble)

    assert policy_input.action_values is ensemble.action_values
    np.testing.assert_array_equal(
        policy_input.opportunity_member_probabilities,
        ensemble.long_action_member_probabilities
        + ensemble.short_action_member_probabilities,
    )
    np.testing.assert_array_equal(
        policy_input.action_values.long_profitable_probability,
        prediction.long_profitable_probability,
    )
    assert not np.allclose(
        policy_input.action_values.long_profitable_probability,
        policy_input.opportunity_probability_mean,
    )


def test_three_action_ensemble_rejects_endpoint_identity_drift(
    three_action_bundle,
) -> None:
    dataset, _targets, model, _phases = three_action_bundle
    first = predict_three_action_lightgbm_model(
        model,
        dataset,
        np.arange(7_100, 7_120, dtype=np.int64),
    )
    second = predict_three_action_lightgbm_model(
        model,
        dataset,
        np.arange(7_101, 7_121, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="endpoint identities"):
        ensemble_three_action_predictions((first, second))
