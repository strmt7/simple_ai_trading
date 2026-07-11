from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

import tools.run_outcome_mixture_screen as screen
from simple_ai_trading.microstructure_barriers import (
    ADAPTIVE_BARRIER_SCHEMA_VERSION,
    ADAPTIVE_BARRIER_TARGET_MODE,
    AdaptiveBarrierSpec,
    AdaptiveBarrierTargets,
)
from simple_ai_trading.microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
)
from simple_ai_trading.microstructure_outcome_lightgbm import (
    LIGHTGBM_HURDLE_SCHEMA_VERSION,
    LightGBMHurdleSpec,
    load_lightgbm_hurdle_model,
    predict_lightgbm_hurdle_model,
    save_lightgbm_hurdle_model,
    train_lightgbm_hurdle_model,
)


def _spec(**overrides: object) -> LightGBMHurdleSpec:
    values: dict[str, object] = {
        "candidate_id": "test-lightgbm-hurdle",
        "family": "side_specific_lightgbm_hurdle_expected_value",
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
    return LightGBMHurdleSpec(**values)  # type: ignore[arg-type]


def _dataset(rows: int = 4_800) -> tuple[MicrostructureDataset, AdaptiveBarrierTargets]:
    rng = np.random.default_rng(812)
    features = rng.normal(size=(rows, len(MICROSTRUCTURE_FEATURE_NAMES))).astype(
        np.float32
    )
    signal = 7.0 * features[:, 0] + 3.0 * features[:, 1]
    noise = rng.normal(scale=3.0, size=rows)
    long_target = signal - 1.5 + noise
    short_target = -signal - 1.5 - noise
    ones = np.ones(rows, dtype=np.float64)
    times = np.arange(rows, dtype=np.int64) * 5_000 + 10_000
    exits = times + 900_750
    dataset = MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version=MICROSTRUCTURE_FEATURE_VERSION,
        feature_names=MICROSTRUCTURE_FEATURE_NAMES,
        horizon_seconds=900,
        total_latency_ms=750,
        taker_fee_bps=5.0,
        additional_slippage_bps_per_side=1.0,
        reference_order_notional_quote=1_000.0,
        max_l1_participation=1.0,
        max_quote_age_ms=1_000,
        decision_cadence_seconds=5,
        target_mode="fixed_horizon",
        stop_loss_bps=None,
        take_profit_bps=None,
        trigger_execution_slippage_bps=None,
        path_resolution_ms=None,
        decision_time_ms=times,
        long_exit_time_ms=exits,
        short_exit_time_ms=exits,
        features=features,
        long_net_bps=long_target,
        short_net_bps=short_target,
        entry_spread_bps=ones,
        exit_spread_bps=ones,
        entry_quote_age_ms=np.zeros(rows, dtype=np.int64),
        exit_quote_age_ms=np.zeros(rows, dtype=np.int64),
        entry_bid_price=100.0 * ones,
        entry_ask_price=100.1 * ones,
        fixed_exit_bid_price=100.0 * ones,
        fixed_exit_ask_price=100.1 * ones,
        entry_bid_qty=10.0 * ones,
        entry_ask_qty=10.0 * ones,
        fixed_exit_bid_qty=10.0 * ones,
        fixed_exit_ask_qty=10.0 * ones,
        long_l1_participation=0.01 * ones,
        short_l1_participation=0.01 * ones,
        long_liquidity_eligible=np.ones(rows, dtype=bool),
        short_liquidity_eligible=np.ones(rows, dtype=bool),
    )
    outcomes = np.zeros(rows, dtype=np.int8)
    targets = AdaptiveBarrierTargets(
        schema_version=ADAPTIVE_BARRIER_SCHEMA_VERSION,
        target_mode=ADAPTIVE_BARRIER_TARGET_MODE,
        spec=AdaptiveBarrierSpec(
            horizon_seconds=900,
            volatility_feature_name="realized_volatility_300s_bps",
            stop_volatility_multiple=1.0,
            take_volatility_multiple=1.5,
            minimum_stop_bps=18.0,
            maximum_stop_bps=60.0,
            minimum_take_bps=27.0,
            maximum_take_bps=90.0,
            base_protection_delay_ms=250,
            stress_protection_delay_ms=750,
            trigger_execution_slippage_bps=1.0,
        ),
        source_indexes=np.arange(rows, dtype=np.int64),
        valid=np.ones(rows, dtype=bool),
        stop_barrier_bps=np.full(rows, 18.0),
        take_barrier_bps=np.full(rows, 27.0),
        base_long_net_bps=long_target,
        base_short_net_bps=short_target,
        base_long_exit_time_ms=exits,
        base_short_exit_time_ms=exits,
        base_long_outcome=outcomes,
        base_short_outcome=outcomes,
        stress_long_net_bps=long_target - 1.0,
        stress_short_net_bps=short_target - 1.0,
        stress_long_exit_time_ms=exits,
        stress_short_exit_time_ms=exits,
        stress_long_outcome=outcomes,
        stress_short_outcome=outcomes,
    )
    return dataset, targets


@pytest.fixture(scope="module")
def lightgbm_bundle():
    dataset, targets = _dataset()
    phases: list[tuple[str, int, int]] = []
    model = train_lightgbm_hurdle_model(
        dataset,
        targets,
        train_endpoints=np.arange(0, 2_000, dtype=np.int64),
        tuning_endpoints=np.arange(2_400, 4_000, dtype=np.int64),
        spec=_spec(),
        target_scenario="base",
        compute_backend="cpu",
        seed=812,
        train_sample_weights=np.linspace(0.8, 1.2, 2_000, dtype=np.float32),
        tuning_sample_weights=np.ones(1_600, dtype=np.float32),
        progress=lambda name, step, total: phases.append((name, step, total)),
    )
    return dataset, targets, model, phases


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"family": "generic_classifier"}, "specification"),
        ({"lower_quantile": 0.75}, "specification"),
        ({"gpu_use_dp_required": False}, "specification"),
        ({"early_stopping_rounds": 12}, "specification"),
    ],
)
def test_lightgbm_hurdle_spec_rejects_contract_drift(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _spec(**overrides)


def test_lightgbm_hurdle_training_is_purged_complete_and_non_authoritative(
    lightgbm_bundle,
) -> None:
    dataset, _targets, model, phases = lightgbm_bundle

    assert model.schema_version == LIGHTGBM_HURDLE_SCHEMA_VERSION
    assert model.model_family == "side_specific_lightgbm_hurdle_expected_value"
    assert model.backend_kind == "cpu"
    assert model.training_rows == 2_000
    assert model.requested_early_stop_rows == 1_600
    assert model.early_stop_rows == 620
    assert model.calibration_rows == 800
    assert model.internal_purged_rows == 180
    assert model.calibration_start_ms == int(dataset.decision_time_ms[3_200])
    assert len(model.best_iterations) == 10
    assert len(model.model_strings) == 10
    assert len(model.model_sha256) == 64
    assert [step for _name, step, _total in phases] == list(range(1, 11))
    assert all(total == 10 for _name, _step, total in phases)
    assert not model.trading_authority
    assert not model.execution_claim
    assert not model.profitability_claim
    assert not model.portfolio_claim
    assert not model.leverage_applied


def test_lightgbm_hurdle_prediction_is_finite_calibrated_and_ordered(
    lightgbm_bundle,
) -> None:
    dataset, _targets, model, _phases = lightgbm_bundle
    prediction = predict_lightgbm_hurdle_model(
        model,
        dataset,
        np.arange(4_100, 4_160, dtype=np.int64),
    )

    assert prediction.rows == 60
    assert np.all((prediction.long_profitable_probability >= 0.0))
    assert np.all((prediction.long_profitable_probability <= 1.0))
    assert np.all((prediction.short_profitable_probability >= 0.0))
    assert np.all((prediction.short_profitable_probability <= 1.0))
    assert np.all(prediction.long_lower_bps <= prediction.long_upper_bps)
    assert np.all(prediction.short_lower_bps <= prediction.short_upper_bps)
    for values in (
        prediction.long_mean_bps,
        prediction.short_mean_bps,
        prediction.long_lower_bps,
        prediction.short_lower_bps,
        prediction.long_upper_bps,
        prediction.short_upper_bps,
    ):
        assert np.all(np.isfinite(values))


def test_research_runner_ensembles_lightgbm_members_without_router_claims(
    lightgbm_bundle,
) -> None:
    dataset, _targets, model, _phases = lightgbm_bundle
    endpoints = np.arange(4_100, 4_130, dtype=np.int64)

    prediction = screen._ensemble_for_role(
        [model, model, model],
        dataset,
        endpoints,
        compute_backend="cpu",
        batch_size=32,
    )

    assert prediction.member_count == 3
    assert prediction.rows == len(endpoints)
    assert np.all(prediction.long_epistemic_std_bps <= 1e-12)
    assert np.all(prediction.short_epistemic_std_bps <= 1e-12)
    assert prediction.long_router_weights is None
    assert prediction.short_router_weights is None


def test_lightgbm_hurdle_artifact_round_trip_is_exact(
    tmp_path, lightgbm_bundle
) -> None:
    dataset, _targets, model, _phases = lightgbm_bundle
    artifact = tmp_path / "hurdle.json"
    save_lightgbm_hurdle_model(artifact, model)
    loaded = load_lightgbm_hurdle_model(artifact)

    assert loaded == model
    expected = predict_lightgbm_hurdle_model(
        model, dataset, np.arange(4_100, 4_130, dtype=np.int64)
    )
    actual = predict_lightgbm_hurdle_model(
        loaded, dataset, np.arange(4_100, 4_130, dtype=np.int64)
    )
    np.testing.assert_array_equal(actual.endpoint_indexes, expected.endpoint_indexes)
    np.testing.assert_allclose(actual.long_mean_bps, expected.long_mean_bps)
    np.testing.assert_allclose(actual.short_mean_bps, expected.short_mean_bps)
    np.testing.assert_allclose(
        actual.long_profitable_probability,
        expected.long_profitable_probability,
    )


@pytest.mark.parametrize("tamper", ["authority", "model", "extra_field"])
def test_lightgbm_hurdle_artifact_tampering_fails_closed(
    tmp_path, lightgbm_bundle, tamper: str
) -> None:
    _dataset_value, _targets, model, _phases = lightgbm_bundle
    artifact = tmp_path / f"tampered-{tamper}.json"
    save_lightgbm_hurdle_model(artifact, model)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if tamper == "authority":
        payload["trading_authority"] = True
    elif tamper == "model":
        payload["model_strings"]["long_probability"] += "\n# tampered"
    else:
        payload["unbound"] = "value"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="contract|invalid"):
        load_lightgbm_hurdle_model(artifact)


def test_lightgbm_hurdle_prediction_rejects_feature_identity_drift(
    lightgbm_bundle,
) -> None:
    dataset, _targets, model, _phases = lightgbm_bundle
    drifted = replace(dataset, feature_names=tuple(reversed(dataset.feature_names)))

    with pytest.raises(ValueError, match="feature"):
        predict_lightgbm_hurdle_model(
            model,
            drifted,
            np.arange(4_100, 4_130, dtype=np.int64),
        )


def test_lightgbm_hurdle_training_rejects_role_label_overlap() -> None:
    dataset, targets = _dataset()

    with pytest.raises(ValueError, match="overlap"):
        train_lightgbm_hurdle_model(
            dataset,
            targets,
            train_endpoints=np.arange(0, 2_000, dtype=np.int64),
            tuning_endpoints=np.arange(2_050, 3_650, dtype=np.int64),
            spec=_spec(),
            target_scenario="base",
            compute_backend="cpu",
            seed=812,
        )
