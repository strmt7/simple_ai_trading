from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from simple_ai_trading.microstructure_action_features import (
    mirror_microstructure_direction,
)
from simple_ai_trading.microstructure_direction_screen import (
    DIRECTION_SCREEN_MODEL_FAMILY,
    DIRECTION_SCREEN_MODEL_SCHEMA_VERSION,
    DirectionScreenSpec,
    _opportunity_mask,
    load_direction_screen_model,
    predict_direction_screen_model,
    save_direction_screen_model,
    train_direction_screen_model,
    utility_margin_multiplier,
)
from simple_ai_trading.microstructure_features import MICROSTRUCTURE_FEATURE_NAMES
from tests.test_microstructure_outcome_lightgbm import _dataset


def _spec() -> DirectionScreenSpec:
    return DirectionScreenSpec(
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_data_in_leaf=32,
        feature_fraction=0.82,
        bagging_fraction=0.82,
        bagging_freq=1,
        lambda_l1=0.005,
        lambda_l2=0.025,
        max_bin=63,
        num_boost_round=12,
        early_stopping_rounds=5,
        gpu_use_dp_required=True,
    )


@pytest.fixture(scope="module")
def direction_screen_bundle():
    dataset, targets = _dataset(rows=8_000)
    model = train_direction_screen_model(
        dataset,
        targets,
        train_endpoints=np.arange(0, 3_500, dtype=np.int64),
        early_stop_endpoints=np.arange(4_000, 7_000, dtype=np.int64),
        train_sample_weights=np.linspace(0.8, 1.2, 3_500, dtype=np.float32),
        early_stop_sample_weights=np.ones(3_000, dtype=np.float32),
        selected_feature_names=MICROSTRUCTURE_FEATURE_NAMES[:32],
        variant="synthetic_full_uniqueness",
        feature_set="synthetic_first_32",
        weighting="uniqueness",
        spec=_spec(),
        compute_backend="cpu",
        seed=29,
    )
    return dataset, targets, model


def test_direction_screen_target_excludes_nonpositive_opportunities_and_ties() -> None:
    long_values = np.asarray([4.0, -1.0, 2.0, 0.0, 3.0], dtype=np.float64)
    short_values = np.asarray([-2.0, 5.0, 2.0, -1.0, 3.0], dtype=np.float64)

    np.testing.assert_array_equal(
        _opportunity_mask(long_values, short_values),
        np.asarray([True, True, False, False, False]),
    )


def test_direction_screen_utility_margin_multiplier_is_frozen_and_bounded() -> None:
    long_values = np.asarray([0.0, 5.0, 10.0, 30.0, 100.0], dtype=np.float64)
    short_values = np.zeros(5, dtype=np.float64)

    np.testing.assert_array_equal(
        utility_margin_multiplier(long_values, short_values, 10.0),
        np.asarray([0.5, 0.5, 1.0, 3.0, 3.0]),
    )


def test_direction_screen_model_is_hash_bound_and_non_authoritative(
    direction_screen_bundle,
) -> None:
    _dataset_value, _targets, model = direction_screen_bundle

    assert model.schema_version == DIRECTION_SCREEN_MODEL_SCHEMA_VERSION
    assert model.model_family == DIRECTION_SCREEN_MODEL_FAMILY
    assert model.backend_kind == "cpu"
    assert model.weighting == "uniqueness"
    assert model.train_weight_multiplier_mean == 1.0
    assert model.early_stop_weight_multiplier_mean == 1.0
    assert model.train_opportunity_rows < model.train_role_rows
    assert model.early_stop_opportunity_rows < model.early_stop_role_rows
    assert len(model.model_sha256) == 64
    assert model.best_iteration >= 1
    assert not model.promotion_permitted
    assert not model.trading_authority
    assert not model.execution_claim
    assert not model.profitability_claim
    assert not model.portfolio_claim
    assert not model.leverage_applied


def test_direction_screen_prediction_is_exactly_mirror_equivariant(
    direction_screen_bundle,
) -> None:
    dataset, _targets, model = direction_screen_bundle
    endpoints = np.arange(7_100, 7_300, dtype=np.int64)
    original = predict_direction_screen_model(model, dataset, endpoints)
    mirrored = predict_direction_screen_model(
        model,
        replace(
            dataset,
            features=mirror_microstructure_direction(dataset.features),
        ),
        endpoints,
    )

    np.testing.assert_array_equal(
        original.long_superiority_probability,
        mirrored.short_superiority_probability,
    )
    np.testing.assert_array_equal(
        original.short_superiority_probability,
        mirrored.long_superiority_probability,
    )
    np.testing.assert_allclose(
        original.conditional_long_probability,
        1.0 - mirrored.conditional_long_probability,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(original.direction_score, -mirrored.direction_score)
    np.testing.assert_array_equal(original.selected_side, -mirrored.selected_side)


def test_direction_screen_artifact_round_trip_is_exact(
    tmp_path,
    direction_screen_bundle,
) -> None:
    dataset, _targets, model = direction_screen_bundle
    artifact = tmp_path / "direction-screen.json"
    save_direction_screen_model(artifact, model)
    loaded = load_direction_screen_model(artifact)

    assert loaded == model
    endpoints = np.arange(7_100, 7_140, dtype=np.int64)
    expected = predict_direction_screen_model(model, dataset, endpoints)
    actual = predict_direction_screen_model(loaded, dataset, endpoints)
    for name in expected.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_direction_screen_artifact_rejects_claim_tampering(
    tmp_path,
    direction_screen_bundle,
) -> None:
    _dataset_value, _targets, model = direction_screen_bundle
    artifact = tmp_path / "direction-screen.json"
    save_direction_screen_model(artifact, model)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["promotion_permitted"] = True
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="model contract"):
        load_direction_screen_model(artifact)
