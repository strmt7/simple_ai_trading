from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from simple_ai_trading.microstructure_action_architecture import (
    ActionValueEnsembleBatch,
)
from simple_ai_trading.microstructure_action_features import (
    mirror_microstructure_direction,
)
from simple_ai_trading.microstructure_action_policy import ActionPolicySpec
from simple_ai_trading.microstructure_selective_action_lightgbm import (
    SELECTIVE_ACTION_LIGHTGBM_SCHEMA_VERSION,
    SelectiveActionEnsembleBatch,
    SelectiveActionLightGBMSpec,
    ensemble_selective_action_predictions,
    load_selective_action_lightgbm_model,
    predict_selective_action_lightgbm_model,
    save_selective_action_lightgbm_model,
    train_selective_action_lightgbm_model,
)
from simple_ai_trading.microstructure_selective_action_policy import (
    SelectiveActionPolicySpec,
    derive_selective_action_scores,
)
from tests.test_microstructure_outcome_lightgbm import _dataset


def _spec(**overrides: object) -> SelectiveActionLightGBMSpec:
    values: dict[str, object] = {
        "candidate_id": "test-factorized-selective-action",
        "family": "factorized_selective_action_lightgbm_hurdle",
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
    return SelectiveActionLightGBMSpec(**values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def selective_action_bundle():
    dataset, targets = _dataset(rows=8_000)
    phases: list[tuple[str, int, int]] = []
    model = train_selective_action_lightgbm_model(
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


def test_selective_action_training_is_purged_and_non_authoritative(
    selective_action_bundle,
) -> None:
    dataset, _targets, model, phases = selective_action_bundle

    assert model.schema_version == SELECTIVE_ACTION_LIGHTGBM_SCHEMA_VERSION
    assert model.backend_kind == "cpu"
    assert model.target_scenario == "stress"
    assert model.training_event_rows == 3_500
    assert model.requested_tuning_event_rows == 3_000
    assert model.early_stop_event_rows == 1_320
    assert model.calibration_event_rows == 1_500
    assert model.internal_purged_event_rows == 180
    assert model.calibration_start_ms == int(dataset.decision_time_ms[5_500])
    assert set(model.class_support) == {
        "opportunity_train",
        "opportunity_early_stop",
        "opportunity_calibration",
        "direction_train",
        "direction_early_stop",
        "direction_calibration",
    }
    assert model.direction_temperature > 0.0
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


def test_selective_action_probabilities_are_finite_and_exhaustive(
    selective_action_bundle,
) -> None:
    dataset, _targets, model, _phases = selective_action_bundle
    endpoints = np.arange(7_100, 7_300, dtype=np.int64)
    prediction = predict_selective_action_lightgbm_model(model, dataset, endpoints)

    np.testing.assert_allclose(
        prediction.long_profitable_probability
        + prediction.short_profitable_probability
        + prediction.abstain_probability,
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.all(prediction.long_lower_bps <= prediction.long_upper_bps)
    assert np.all(prediction.short_lower_bps <= prediction.short_upper_bps)
    assert np.all(np.isfinite(prediction.long_mean_bps))
    assert np.all(np.isfinite(prediction.short_mean_bps))
    assert np.all(
        prediction.side_consensus
        == (
            (prediction.action_preference_side != 0)
            & (
                prediction.action_preference_side
                == prediction.direction_preference_side
            )
        )
    )


def test_selective_action_prediction_is_mirror_equivariant(
    selective_action_bundle,
) -> None:
    dataset, _targets, model, _phases = selective_action_bundle
    endpoints = np.arange(7_100, 7_160, dtype=np.int64)
    original = predict_selective_action_lightgbm_model(model, dataset, endpoints)
    mirrored = predict_selective_action_lightgbm_model(
        model,
        replace(
            dataset,
            features=mirror_microstructure_direction(dataset.features),
        ),
        endpoints,
    )

    np.testing.assert_array_equal(
        original.opportunity_probability,
        mirrored.opportunity_probability,
    )
    np.testing.assert_allclose(
        original.conditional_long_probability,
        1.0 - mirrored.conditional_long_probability,
        rtol=0.0,
        atol=1e-15,
    )
    for long_name, short_name in (
        ("long_mean_bps", "short_mean_bps"),
        ("long_profitable_probability", "short_profitable_probability"),
        ("long_lower_bps", "short_lower_bps"),
        ("long_upper_bps", "short_upper_bps"),
    ):
        np.testing.assert_allclose(
            getattr(original, long_name),
            getattr(mirrored, short_name),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            getattr(original, short_name),
            getattr(mirrored, long_name),
            rtol=0.0,
            atol=1e-12,
        )
    np.testing.assert_array_equal(
        original.action_preference_side,
        -mirrored.action_preference_side,
    )
    np.testing.assert_array_equal(
        original.direction_preference_side,
        -mirrored.direction_preference_side,
    )


def test_selective_action_artifact_round_trip_is_exact(
    tmp_path,
    selective_action_bundle,
) -> None:
    dataset, _targets, model, _phases = selective_action_bundle
    artifact = tmp_path / "selective-action.json"
    save_selective_action_lightgbm_model(artifact, model)
    loaded = load_selective_action_lightgbm_model(artifact)

    assert loaded == model
    endpoints = np.arange(7_100, 7_130, dtype=np.int64)
    expected = predict_selective_action_lightgbm_model(model, dataset, endpoints)
    actual = predict_selective_action_lightgbm_model(loaded, dataset, endpoints)
    for name in expected.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("tamper", ["authority", "model", "extra_field"])
def test_selective_action_artifact_tampering_fails_closed(
    tmp_path,
    selective_action_bundle,
    tamper: str,
) -> None:
    _dataset_value, _targets, model, _phases = selective_action_bundle
    artifact = tmp_path / f"selective-action-tampered-{tamper}.json"
    save_selective_action_lightgbm_model(artifact, model)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if tamper == "authority":
        payload["trading_authority"] = True
    elif tamper == "model":
        payload["model_strings"]["opportunity_probability"] += "\n# tampered"
    else:
        payload["unbound"] = "value"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="authority|contract|invalid"):
        load_selective_action_lightgbm_model(artifact)


def _policy_ensemble() -> SelectiveActionEnsembleBatch:
    endpoints = np.arange(4, dtype=np.int64)
    action = ActionValueEnsembleBatch(
        endpoint_indexes=endpoints,
        long_mean_bps=np.asarray([4.0, -2.0, 4.0, 4.0]),
        short_mean_bps=np.asarray([-2.0, 4.0, -2.0, -2.0]),
        long_epistemic_std_bps=np.full(4, 0.2),
        short_epistemic_std_bps=np.full(4, 0.2),
        long_profitable_probability=np.asarray([0.56, 0.24, 0.38, 0.56]),
        short_profitable_probability=np.asarray([0.24, 0.56, 0.17, 0.24]),
        long_lower_bps=np.asarray([-4.0, -8.0, -4.0, -4.0]),
        short_lower_bps=np.asarray([-8.0, -4.0, -8.0, -8.0]),
        long_upper_bps=np.full(4, 8.0),
        short_upper_bps=np.full(4, 8.0),
        long_positive_member_ratio=np.asarray([1.0, 0.0, 1.0, 1.0]),
        short_positive_member_ratio=np.asarray([0.0, 1.0, 0.0, 0.0]),
        member_count=3,
    )
    opportunity = np.asarray(
        [
            [0.80, 0.80, 0.55, 0.80],
            [0.82, 0.82, 0.54, 0.82],
            [0.79, 0.79, 0.56, 0.79],
        ]
    )
    direction = np.asarray(
        [
            [0.70, 0.30, 0.70, 0.70],
            [0.72, 0.28, 0.72, 0.72],
            [0.69, 0.31, 0.69, 0.69],
        ]
    )
    return SelectiveActionEnsembleBatch(
        action_values=action,
        opportunity_probability_mean=np.mean(opportunity, axis=0),
        opportunity_probability_std=np.std(opportunity, axis=0),
        conditional_long_probability_mean=np.mean(direction, axis=0),
        conditional_long_probability_std=np.std(direction, axis=0),
        opportunity_member_probabilities=opportunity,
        conditional_long_member_probabilities=direction,
        direction_long_member_ratio=np.mean(direction > 0.5, axis=0),
        direction_short_member_ratio=np.mean(direction < 0.5, axis=0),
        side_consensus_member_ratio=np.asarray([1.0, 1.0, 1.0, 1.0 / 3.0]),
        member_count=3,
    )


def test_selective_policy_uses_opportunity_not_joint_side_probability() -> None:
    ensemble = _policy_ensemble()
    score = derive_selective_action_scores(
        ensemble,
        SelectiveActionPolicySpec(
            action_policy=ActionPolicySpec(
                profile="regular",
                epistemic_penalty=0.5,
                minimum_profitable_probability=0.6,
                minimum_member_agreement=2.0 / 3.0,
                maximum_epistemic_std_bps=2.0,
                minimum_lower_bound_bps=-6.0,
            ),
            minimum_conditional_direction_confidence=0.575,
        ),
    )

    np.testing.assert_array_equal(score.side, np.asarray([1, -1, 0, 0]))
    np.testing.assert_array_equal(score.eligible, np.asarray([True, True, False, False]))
    assert ensemble.action_values.long_profitable_probability[0] < 0.6
    assert ensemble.action_values.short_profitable_probability[1] < 0.6


def test_selective_ensemble_rejects_endpoint_identity_drift(
    selective_action_bundle,
) -> None:
    dataset, _targets, model, _phases = selective_action_bundle
    first = predict_selective_action_lightgbm_model(
        model,
        dataset,
        np.arange(7_100, 7_120, dtype=np.int64),
    )
    second = predict_selective_action_lightgbm_model(
        model,
        dataset,
        np.arange(7_101, 7_121, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="endpoint identities"):
        ensemble_selective_action_predictions((first, second))
