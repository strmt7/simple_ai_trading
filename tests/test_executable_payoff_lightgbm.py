from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading.executable_payoff_lightgbm import (
    EXECUTABLE_PAYOFF_MODEL_SCHEMA_VERSION,
    ExecutablePayoffSpec,
    build_executable_payoff_dataset,
    load_executable_payoff_model,
    predict_executable_payoff_model,
    save_executable_payoff_model,
    train_executable_payoff_model,
)
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


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-052-executable-support-hurdle-fincast-design.json"
)


def _spec(architecture: str) -> ExecutablePayoffSpec:
    return ExecutablePayoffSpec(
        candidate_id=f"test-{architecture}",
        family="side_specific_executable_payoff",
        architecture=architecture,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        minimum_leaf_fraction=0.01,
        minimum_leaf_rows=16,
        maximum_leaf_rows=64,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=0.01,
        lambda_l2=0.1,
        max_bin=63,
        num_boost_round=12,
        early_stopping_rounds=5,
    )


def _source(rows: int = 5_200) -> tuple[MicrostructureDataset, AdaptiveBarrierTargets]:
    rng = np.random.default_rng(5201)
    features = rng.normal(size=(rows, len(MICROSTRUCTURE_FEATURE_NAMES))).astype(
        np.float32
    )
    signal = 8.0 * features[:, 0] - 4.0 * features[:, 1]
    noise = rng.normal(scale=4.0, size=rows)
    long_target = signal - 1.0 + noise
    short_target = -signal - 1.0 - noise
    times = 10_000 + np.arange(rows, dtype=np.int64) * 5_000
    exits = times + 300_750
    ones = np.ones(rows, dtype=np.float64)
    long_mask = np.arange(rows) % 5 != 0
    short_mask = np.arange(rows) % 4 != 0
    dataset = MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version=MICROSTRUCTURE_FEATURE_VERSION,
        feature_names=MICROSTRUCTURE_FEATURE_NAMES,
        horizon_seconds=300,
        total_latency_ms=750,
        taker_fee_bps=5.0,
        additional_slippage_bps_per_side=1.0,
        reference_order_notional_quote=1_000.0,
        max_l1_participation=0.1,
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
        long_l1_participation=np.where(long_mask, 0.01, 0.2),
        short_l1_participation=np.where(short_mask, 0.01, 0.2),
        long_liquidity_eligible=long_mask,
        short_liquidity_eligible=short_mask,
    )
    outcomes = np.zeros(rows, dtype=np.int8)
    targets = AdaptiveBarrierTargets(
        schema_version=ADAPTIVE_BARRIER_SCHEMA_VERSION,
        target_mode=ADAPTIVE_BARRIER_TARGET_MODE,
        spec=AdaptiveBarrierSpec(
            horizon_seconds=300,
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
        stress_long_net_bps=long_target - 2.0,
        stress_short_net_bps=short_target - 2.0,
        stress_long_exit_time_ms=exits,
        stress_short_exit_time_ms=exits,
        stress_long_outcome=outcomes,
        stress_short_outcome=outcomes,
    )
    return dataset, targets


def _roles() -> dict[str, np.ndarray]:
    return {
        "train": np.arange(0, 2_000, dtype=np.int64),
        "early_stop": np.arange(2_200, 3_200, dtype=np.int64),
        "probability_calibration": np.arange(3_400, 4_400, dtype=np.int64),
        "prediction": np.arange(4_600, 4_800, dtype=np.int64),
    }


@pytest.fixture(scope="module")
def hurdle_bundle():
    source, targets = _source()
    dataset = build_executable_payoff_dataset(
        source,
        targets,
        target_scenario="base",
    )
    roles = _roles()
    progress: list[tuple[str, str, int, int]] = []
    model = train_executable_payoff_model(
        dataset,
        train_indexes=roles["train"],
        early_stop_indexes=roles["early_stop"],
        probability_calibration_indexes=roles["probability_calibration"],
        probability_calibration_end_exclusive_ms=int(
            dataset.payoff.decision_time_ms[4_500]
        ),
        spec=_spec("sign_magnitude_hurdle"),
        target_scenario="base",
        compute_backend="cpu",
        seed=5201,
        progress=lambda name, side, step, total: progress.append(
            (name, side, step, total)
        ),
    )
    return dataset, model, roles, progress


def test_round52_design_is_canonical_and_fail_closed() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    claimed = design.pop("design_sha256")
    canonical = json.dumps(
        design,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")

    assert hashlib.sha256(canonical).hexdigest() == claimed
    assert design["status"] == "frozen"
    assert design["claims"]["selection_contaminated"] is True
    assert design["claims"]["profitability_claim_permitted"] is False
    assert design["model_contract"]["support_invariant"].startswith("Training")
    assert design["economic_screen"]["leverage"] == 1.0


def test_dataset_binds_side_specific_support_to_exact_source_rows() -> None:
    source, targets = _source()
    dataset = build_executable_payoff_dataset(
        source,
        targets,
        target_scenario="base",
    )

    np.testing.assert_array_equal(
        dataset.long_executable, source.long_liquidity_eligible
    )
    np.testing.assert_array_equal(
        dataset.short_executable, source.short_liquidity_eligible
    )
    assert dataset.dataset_sha256 != dataset.payoff.dataset_sha256
    assert len(dataset.dataset_sha256) == 64


def test_hurdle_training_filters_every_role_by_side_and_is_non_authoritative(
    hurdle_bundle,
) -> None:
    dataset, model, roles, progress = hurdle_bundle

    assert model.schema_version == EXECUTABLE_PAYOFF_MODEL_SCHEMA_VERSION
    assert model.spec.architecture == "sign_magnitude_hurdle"
    assert set(model.model_strings) == {
        "long_probability",
        "long_conditional_gain",
        "long_conditional_loss",
        "short_probability",
        "short_conditional_gain",
        "short_conditional_loss",
    }
    for side, mask in (
        ("long", dataset.long_executable),
        ("short", dataset.short_executable),
    ):
        for role in ("train", "early_stop", "probability_calibration"):
            expected = int(np.sum(mask[roles[role]]))
            assert model.role_rows[side][role] == expected
            assert model.rejected_role_rows[side][role] == len(roles[role]) - expected
            assert sum(model.class_support[side][role].values()) == expected
            assert len(model.role_mask_sha256[side][role]) == 64
    assert [step for _name, _side, step, _total in progress] == list(range(1, 7))
    assert all(total == 6 for _name, _side, _step, total in progress)
    assert not model.trading_authority
    assert not model.execution_claim
    assert not model.profitability_claim
    assert not model.portfolio_claim
    assert not model.leverage_applied


def test_hurdle_prediction_is_finite_calibrated_and_preserves_support(
    hurdle_bundle,
) -> None:
    dataset, model, roles, _progress = hurdle_bundle
    prediction = predict_executable_payoff_model(
        model,
        dataset,
        roles["prediction"],
    )

    assert prediction.rows == len(roles["prediction"])
    np.testing.assert_array_equal(
        prediction.long_executable,
        dataset.long_executable[roles["prediction"]],
    )
    np.testing.assert_array_equal(
        prediction.short_executable,
        dataset.short_executable[roles["prediction"]],
    )
    for values in (
        prediction.long_expected_net_bps,
        prediction.short_expected_net_bps,
        prediction.long_profitable_probability,
        prediction.short_profitable_probability,
        prediction.long_conditional_gain_bps,
        prediction.short_conditional_gain_bps,
        prediction.long_conditional_loss_bps,
        prediction.short_conditional_loss_bps,
    ):
        assert values is not None
        assert np.all(np.isfinite(values))
    assert np.all((prediction.long_profitable_probability >= 0.0))
    assert np.all((prediction.long_profitable_probability <= 1.0))


def test_model_artifact_round_trip_is_exact(tmp_path, hurdle_bundle) -> None:
    dataset, model, roles, _progress = hurdle_bundle
    artifact = tmp_path / "model.json"
    save_executable_payoff_model(artifact, model)
    loaded = load_executable_payoff_model(artifact)
    expected = predict_executable_payoff_model(model, dataset, roles["prediction"])
    actual = predict_executable_payoff_model(loaded, dataset, roles["prediction"])

    assert loaded == model
    np.testing.assert_allclose(
        actual.long_expected_net_bps,
        expected.long_expected_net_bps,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual.short_expected_net_bps,
        expected.short_expected_net_bps,
        rtol=0.0,
        atol=0.0,
    )


def test_direct_model_uses_the_same_support_contract() -> None:
    source, targets = _source()
    dataset = build_executable_payoff_dataset(
        source,
        targets,
        target_scenario="base",
    )
    roles = _roles()
    model = train_executable_payoff_model(
        dataset,
        train_indexes=roles["train"],
        early_stop_indexes=roles["early_stop"],
        probability_calibration_indexes=roles["probability_calibration"],
        probability_calibration_end_exclusive_ms=int(
            dataset.payoff.decision_time_ms[4_500]
        ),
        spec=_spec("direct_mean"),
        target_scenario="base",
        compute_backend="cpu",
        seed=5202,
    )
    prediction = predict_executable_payoff_model(
        model,
        dataset,
        roles["prediction"],
    )

    assert set(model.model_strings) == {"long_mean", "short_mean"}
    assert not model.probability_calibration
    assert prediction.long_profitable_probability is None
    assert prediction.magnitude_floor_count == 0


def test_prediction_rejects_support_identity_drift(hurdle_bundle) -> None:
    dataset, _model, _roles, _progress = hurdle_bundle
    with pytest.raises(ValueError, match="dataset"):
        replace(
            dataset,
            long_executable=np.logical_not(dataset.long_executable),
            dataset_sha256=dataset.dataset_sha256,
        )
