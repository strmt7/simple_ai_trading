"""Tests for causal gross-return architecture research."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import sys

import numpy as np
import pytest

import simple_ai_trading.microstructure_architecture as architecture
from simple_ai_trading.compute import BackendInfo
from simple_ai_trading.microstructure_architecture import (
    GROSS_TARGET_MODE,
    GrossArchitectureSpec,
    GrossPredictionBatch,
    average_label_uniqueness,
    causal_cusum_event_mask,
    evaluate_gross_forecast,
    gross_midpoint_log_returns_bps,
    predict_lightgbm_gross_model,
    predict_torch_gross_model,
    train_lightgbm_gross_baseline,
    train_torch_gross_model,
    valid_sequence_endpoints,
)
from simple_ai_trading.microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
)


def _dataset(rows: int = 1_024) -> tuple[MicrostructureDataset, np.ndarray]:
    rng = np.random.default_rng(31)
    features = rng.normal(
        0.0,
        1.0,
        (rows, len(MICROSTRUCTURE_FEATURE_NAMES)),
    ).astype(np.float32)
    gross = (
        3.0 * features[:, 0]
        - 1.5 * features[:, MICROSTRUCTURE_FEATURE_NAMES.index("l1_imbalance")]
        + rng.normal(0.0, 0.2, rows)
    )
    features[:, MICROSTRUCTURE_FEATURE_NAMES.index("return_5s_bps")] = gross
    features[
        :,
        MICROSTRUCTURE_FEATURE_NAMES.index("realized_volatility_60s_bps"),
    ] = 0.25
    entry_mid = np.full(rows, 100.0)
    exit_mid = entry_mid * np.exp(gross / 10_000.0)
    entry_bid = entry_mid - 0.01
    entry_ask = entry_mid + 0.01
    exit_bid = exit_mid - 0.01
    exit_ask = exit_mid + 0.01
    decisions = np.arange(rows, dtype=np.int64) * 5_000 + 10_000
    dataset = MicrostructureDataset(
        symbol="BTCUSDT",
        feature_version=MICROSTRUCTURE_FEATURE_VERSION,
        feature_names=MICROSTRUCTURE_FEATURE_NAMES,
        horizon_seconds=300,
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
        decision_time_ms=decisions,
        long_exit_time_ms=decisions + 300_750,
        short_exit_time_ms=decisions + 300_750,
        features=features,
        long_net_bps=gross - 12.0,
        short_net_bps=-gross - 12.0,
        entry_spread_bps=np.full(rows, 2.0),
        exit_spread_bps=np.full(rows, 2.0),
        entry_quote_age_ms=np.full(rows, 10, dtype=np.int64),
        exit_quote_age_ms=np.full(rows, 10, dtype=np.int64),
        entry_bid_price=entry_bid,
        entry_ask_price=entry_ask,
        fixed_exit_bid_price=exit_bid,
        fixed_exit_ask_price=exit_ask,
        entry_bid_qty=np.full(rows, 1_000.0),
        entry_ask_qty=np.full(rows, 1_000.0),
        fixed_exit_bid_qty=np.full(rows, 1_000.0),
        fixed_exit_ask_qty=np.full(rows, 1_000.0),
        long_l1_participation=np.full(rows, 0.01),
        short_l1_participation=np.full(rows, 0.01),
        long_liquidity_eligible=np.ones(rows, dtype=bool),
        short_liquidity_eligible=np.ones(rows, dtype=bool),
        source_evidence=None,
    )
    return dataset, gross.astype(np.float64)


@pytest.fixture(scope="module")
def torch_model_bundle():
    pytest.importorskip("torch")
    dataset, actual = _dataset()
    train = np.arange(0, 700, dtype=np.int64)
    tuning = np.arange(700, 1_024, dtype=np.int64)
    spec = GrossArchitectureSpec(
        candidate_id="mlp-smoke",
        family="tabular_mlp",
        sequence_length=1,
        hidden_dim=8,
        residual_blocks=1,
        dropout=0.0,
        gmadl_weight=0.2,
    )
    progress: list[tuple[int, int, float, float]] = []
    model = train_torch_gross_model(
        dataset,
        actual,
        train_endpoints=train,
        tuning_endpoints=tuning,
        spec=spec,
        compute_backend="cpu",
        seed=7,
        batch_size=256,
        max_epochs=1,
        patience=1,
        progress=lambda *values: progress.append(values),
    )
    return dataset, actual, model, progress


@pytest.fixture(scope="module")
def lightgbm_model_bundle():
    dataset, actual = _dataset()
    train = np.arange(0, 600, dtype=np.int64)
    tuning = np.arange(600, 900, dtype=np.int64)
    train_uniqueness = average_label_uniqueness(
        dataset.decision_time_ms,
        dataset.long_exit_time_ms,
        train,
    )
    tuning_uniqueness = average_label_uniqueness(
        dataset.decision_time_ms,
        dataset.long_exit_time_ms,
        tuning,
    )
    model = train_lightgbm_gross_baseline(
        dataset,
        actual,
        train_endpoints=train,
        tuning_endpoints=tuning,
        train_uniqueness=train_uniqueness,
        tuning_uniqueness=tuning_uniqueness,
        compute_backend="cpu",
        seed=11,
    )
    return dataset, actual, model


def test_gross_target_uses_latency_aligned_midpoint_log_return() -> None:
    dataset, expected = _dataset(32)
    np.testing.assert_allclose(
        gross_midpoint_log_returns_bps(dataset),
        expected,
        rtol=0.0,
        atol=1.0e-10,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"candidate_id": ""}, "candidate_id"),
        ({"family": "lstm"}, "family"),
        ({"sequence_length": 2}, "sequence_length=1"),
        (
            {"family": "causal_tcn", "sequence_length": 0},
            "sequence length",
        ),
        ({"hidden_dim": 4}, "hidden dimension"),
        ({"residual_blocks": 0}, "residual block"),
        ({"dropout": float("nan")}, "must be finite"),
        ({"gmadl_weight": 3.0}, "outside bounds"),
    ],
)
def test_gross_architecture_spec_rejects_invalid_contract(changes, message) -> None:
    values = {
        "candidate_id": "candidate",
        "family": "tabular_mlp",
        "sequence_length": 1,
        "hidden_dim": 8,
        "residual_blocks": 1,
        "dropout": 0.0,
        "gmadl_weight": 0.0,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        GrossArchitectureSpec(**values)


def test_gross_target_rejects_path_dependent_dataset() -> None:
    dataset, _actual = _dataset(32)
    with pytest.raises(ValueError, match="fixed-horizon"):
        gross_midpoint_log_returns_bps(
            replace(dataset, target_mode="exchange_trigger_market_exit")
        )
    with pytest.raises(ValueError, match="execution prices must be positive"):
        gross_midpoint_log_returns_bps(
            replace(dataset, entry_bid_price=-dataset.entry_bid_price)
        )


def test_sequence_endpoints_reject_gaps_without_future_lookahead() -> None:
    times = np.asarray([0, 5_000, 10_000, 20_000, 25_000], dtype=np.int64)
    endpoints = np.asarray([2, 3, 4], dtype=np.int64)
    np.testing.assert_array_equal(
        valid_sequence_endpoints(
            times,
            endpoints,
            sequence_length=3,
            cadence_seconds=5,
        ),
        np.asarray([2], dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("times", "endpoints", "length", "cadence", "message"),
    [
        ([0, 0], [1], 1, 5, "timestamps"),
        ([0, 5_000], [1, 1], 1, 5, "endpoints"),
        ([0, 5_000], [1], 0, 5, "contract"),
        ([0, 5_000], [2], 1, 5, "outside"),
    ],
)
def test_sequence_endpoint_contract_rejects_invalid_inputs(
    times,
    endpoints,
    length,
    cadence,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        valid_sequence_endpoints(
            np.asarray(times, dtype=np.int64),
            np.asarray(endpoints, dtype=np.int64),
            sequence_length=length,
            cadence_seconds=cadence,
        )


def test_average_label_uniqueness_downweights_overlap() -> None:
    times = np.arange(8, dtype=np.int64) * 5_000
    exits = times + 15_000
    weights = average_label_uniqueness(
        times,
        exits,
        np.asarray([0, 1, 7], dtype=np.int64),
    )
    assert weights.shape == (3,)
    assert np.mean(weights) == pytest.approx(1.0)
    assert weights[1] < weights[2]


@pytest.mark.parametrize(
    "endpoints",
    [
        np.asarray([], dtype=np.int64),
        np.asarray([-1], dtype=np.int64),
        np.asarray([3], dtype=np.int64),
        np.asarray([1, 1], dtype=np.int64),
    ],
)
def test_average_label_uniqueness_rejects_invalid_endpoints(endpoints) -> None:
    times = np.asarray([0, 5_000, 10_000], dtype=np.int64)
    exits = times + 5_000
    with pytest.raises(ValueError, match="inputs are invalid"):
        average_label_uniqueness(times, exits, endpoints)


def test_cusum_events_reset_at_utc_day_boundary() -> None:
    feature_names = ("return_5s_bps", "realized_volatility_60s_bps")
    dataset = SimpleNamespace(
        feature_names=feature_names,
        features=np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
        decision_time_ms=np.asarray(
            [0, 5_000, 86_399_000, 86_400_000],
            dtype=np.int64,
        ),
        rows=4,
    )
    np.testing.assert_array_equal(
        causal_cusum_event_mask(
            dataset,
            volatility_multiplier=1.0,
            minimum_threshold_bps=2.0,
        ),
        np.asarray([False, True, False, False]),
    )


@pytest.mark.parametrize(
    ("multiplier", "floor", "message"),
    [
        (float("nan"), 1.0, "finite"),
        (0.0, 1.0, "positive"),
    ],
)
def test_cusum_rejects_invalid_thresholds(multiplier, floor, message) -> None:
    dataset, _actual = _dataset(8)
    with pytest.raises(ValueError, match=message):
        causal_cusum_event_mask(
            dataset,
            volatility_multiplier=multiplier,
            minimum_threshold_bps=floor,
        )


def test_cusum_rejects_missing_and_nonfinite_features() -> None:
    missing = SimpleNamespace(
        feature_names=("other",),
        features=np.zeros((2, 1), dtype=np.float32),
        decision_time_ms=np.asarray([0, 5_000]),
        rows=2,
    )
    with pytest.raises(ValueError, match="features are missing"):
        causal_cusum_event_mask(
            missing,
            volatility_multiplier=1.0,
            minimum_threshold_bps=1.0,
        )
    nonfinite = SimpleNamespace(
        feature_names=("return_5s_bps", "realized_volatility_60s_bps"),
        features=np.asarray([[np.nan, 1.0], [1.0, 1.0]], dtype=np.float32),
        decision_time_ms=np.asarray([0, 5_000]),
        rows=2,
    )
    with pytest.raises(ValueError, match="non-finite"):
        causal_cusum_event_mask(
            nonfinite,
            volatility_multiplier=1.0,
            minimum_threshold_bps=1.0,
        )


def test_gross_metrics_keep_top_rows_distinct_from_portfolio_claims() -> None:
    dataset, actual = _dataset(128)
    endpoints = np.arange(128, dtype=np.int64)
    probability = np.where(actual > 0.0, 0.9, 0.1)
    prediction = GrossPredictionBatch(
        endpoint_indexes=endpoints,
        mean_prediction_bps=actual.copy(),
        direction_probability=probability,
        lower_prediction_bps=actual - 1.0,
        upper_prediction_bps=actual + 1.0,
    )
    metrics = evaluate_gross_forecast(dataset, actual, prediction)
    assert metrics.direction_auc == pytest.approx(1.0)
    assert metrics.spearman_information_coefficient == pytest.approx(1.0)
    assert metrics.top_rows[0]["portfolio_claim"] is False
    assert metrics.top_rows[0]["rows"] == 100
    assert metrics.asdict()["rows"] == 128


def test_gross_metrics_reject_invalid_endpoints_arrays_and_top_count() -> None:
    dataset, actual = _dataset(16)
    base = GrossPredictionBatch(
        endpoint_indexes=np.arange(16, dtype=np.int64),
        mean_prediction_bps=actual,
        direction_probability=np.full(16, 0.5),
        lower_prediction_bps=actual - 1.0,
        upper_prediction_bps=actual + 1.0,
    )
    with pytest.raises(ValueError, match="endpoints are invalid"):
        evaluate_gross_forecast(
            dataset,
            actual,
            replace(base, endpoint_indexes=np.asarray([16], dtype=np.int64)),
        )
    with pytest.raises(ValueError, match="arrays are invalid"):
        evaluate_gross_forecast(
            dataset,
            actual,
            replace(base, direction_probability=np.full(16, 2.0)),
        )
    with pytest.raises(ValueError, match="top-row count"):
        evaluate_gross_forecast(dataset, actual, base, requested_top_rows=(0,))


def test_gross_metrics_cover_constant_baselines_and_crossed_intervals() -> None:
    dataset, _actual = _dataset(16)
    actual = np.ones(16, dtype=np.float64)
    prediction = GrossPredictionBatch(
        endpoint_indexes=np.arange(16, dtype=np.int64),
        mean_prediction_bps=np.ones(16),
        direction_probability=np.ones(16),
        lower_prediction_bps=np.full(16, 2.0),
        upper_prediction_bps=np.zeros(16),
    )
    metrics = evaluate_gross_forecast(dataset, actual, prediction)
    assert metrics.direction_auc == 0.5
    assert metrics.pearson_information_coefficient == 0.0
    assert metrics.interval_crossing_rate == 1.0


def test_gross_metrics_exclude_ineligible_model_selected_sides() -> None:
    dataset, actual = _dataset(16)
    dataset = replace(
        dataset,
        long_liquidity_eligible=np.asarray([False] * 8 + [True] * 8),
    )
    prediction = GrossPredictionBatch(
        endpoint_indexes=np.arange(16, dtype=np.int64),
        mean_prediction_bps=np.arange(16, 0, -1, dtype=np.float64),
        direction_probability=np.full(16, 0.75),
        lower_prediction_bps=np.full(16, -1.0),
        upper_prediction_bps=np.full(16, 1.0),
    )
    metrics = evaluate_gross_forecast(
        dataset,
        actual,
        prediction,
        requested_top_rows=(16,),
    )
    assert metrics.exact_after_cost_eligible_rows == 8
    assert metrics.exact_after_cost_eligible_ratio == 0.5
    assert metrics.top_rows[0]["rows"] == 8

    with pytest.raises(ValueError, match="no exact after-cost eligible"):
        evaluate_gross_forecast(
            replace(
                dataset,
                long_liquidity_eligible=np.zeros(16, dtype=bool),
            ),
            actual,
            prediction,
        )


def test_tcn_network_executes_causal_residual_path() -> None:
    torch = pytest.importorskip("torch")
    spec = GrossArchitectureSpec(
        candidate_id="tcn-forward",
        family="causal_tcn",
        sequence_length=8,
        hidden_dim=8,
        residual_blocks=2,
        dropout=0.0,
        gmadl_weight=0.2,
    )
    network = architecture._network(spec, len(MICROSTRUCTURE_FEATURE_NAMES))
    output = network(torch.zeros(4, 8, len(MICROSTRUCTURE_FEATURE_NAMES)))
    assert tuple(output.shape) == (4, 4)


def test_network_does_not_repeat_normalization_on_raw_feature_axis() -> None:
    torch = pytest.importorskip("torch")
    spec = GrossArchitectureSpec(
        candidate_id="directml-portability",
        family="tabular_mlp",
        sequence_length=1,
        hidden_dim=64,
        residual_blocks=1,
        dropout=0.0,
        gmadl_weight=0.0,
    )
    network = architecture._network(spec, len(MICROSTRUCTURE_FEATURE_NAMES))
    normalized_shapes = {
        tuple(module.normalized_shape)
        for module in network.modules()
        if isinstance(module, torch.nn.LayerNorm)
    }
    assert (len(MICROSTRUCTURE_FEATURE_NAMES),) not in normalized_shapes
    assert (spec.hidden_dim,) in normalized_shapes


def test_feature_scaler_handles_constant_columns_and_rejects_nonfinite() -> None:
    features = np.ones((8, 3), dtype=np.float32)
    center, scale = architecture._feature_scaler(features, np.arange(8))
    np.testing.assert_array_equal(center, np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(scale, np.ones(3, dtype=np.float32))
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match="scaler is non-finite"):
        architecture._feature_scaler(features, np.arange(8))


def test_torch_gross_model_trains_and_reloads_on_cpu(torch_model_bundle) -> None:
    dataset, _actual, model, progress = torch_model_bundle
    assert model.trading_authority is False
    assert model.target_mode == GROSS_TARGET_MODE
    assert len(progress) == 1
    prediction = predict_torch_gross_model(
        model,
        dataset,
        np.arange(900, 1_024, dtype=np.int64),
        compute_backend="cpu",
        batch_size=128,
    )
    assert prediction.rows == 124
    assert np.all(np.isfinite(prediction.mean_prediction_bps))


def test_torch_training_and_prediction_reject_invalid_contracts(torch_model_bundle) -> None:
    dataset, actual, model, _progress = torch_model_bundle
    spec = model.spec
    with pytest.raises(ValueError, match="targets are invalid"):
        train_torch_gross_model(
            dataset,
            actual[:-1],
            train_endpoints=np.arange(700),
            tuning_endpoints=np.arange(700, 1_024),
            spec=spec,
            compute_backend="cpu",
            seed=1,
            batch_size=256,
            max_epochs=1,
            patience=1,
        )
    with pytest.raises(ValueError, match="insufficient contiguous rows"):
        train_torch_gross_model(
            dataset,
            actual,
            train_endpoints=np.arange(511),
            tuning_endpoints=np.arange(700, 1_024),
            spec=spec,
            compute_backend="cpu",
            seed=1,
            batch_size=256,
            max_epochs=1,
            patience=1,
        )
    with pytest.raises(ValueError, match="training budget"):
        train_torch_gross_model(
            dataset,
            actual,
            train_endpoints=np.arange(700),
            tuning_endpoints=np.arange(700, 1_024),
            spec=spec,
            compute_backend="cpu",
            seed=1,
            batch_size=1,
            max_epochs=1,
            patience=1,
        )
    with pytest.raises(ValueError, match="sample weights"):
        train_torch_gross_model(
            dataset,
            actual,
            train_endpoints=np.arange(700),
            tuning_endpoints=np.arange(700, 1_024),
            spec=spec,
            compute_backend="cpu",
            seed=1,
            batch_size=256,
            max_epochs=1,
            patience=1,
            train_sample_weights=np.zeros(700, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="model contract"):
        predict_torch_gross_model(
            replace(model, trading_authority=True),
            dataset,
            np.arange(900, 1_024),
            compute_backend="cpu",
            batch_size=128,
        )
    with pytest.raises(ValueError, match="batch size"):
        predict_torch_gross_model(
            model,
            dataset,
            np.arange(900, 1_024),
            compute_backend="cpu",
            batch_size=0,
        )
    with pytest.raises(ValueError, match="no contiguous endpoints"):
        predict_torch_gross_model(
            replace(model, sequence_length=256),
            dataset,
            np.asarray([1], dtype=np.int64),
            compute_backend="cpu",
            batch_size=128,
        )


def test_directml_device_and_seed_paths_are_vendor_agnostic(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    seeded: list[int] = []
    fake_directml = SimpleNamespace(
        device=lambda: "privateuseone:0",
        default_generator=SimpleNamespace(manual_seed=lambda value: seeded.append(value)),
    )
    monkeypatch.setitem(sys.modules, "torch_directml", fake_directml)
    backend = BackendInfo(
        requested="directml",
        kind="directml",
        device="privateuseone:0",
        vendor="DirectML",
        reason="",
    )
    assert architecture._torch_device(backend) == "privateuseone:0"
    architecture._seed_torch(torch, 19, backend)
    assert seeded == [19]


def test_lightgbm_gross_baseline_trains_and_reloads(lightgbm_model_bundle) -> None:
    dataset, _actual, model = lightgbm_model_bundle
    assert model.trading_authority is False
    prediction = predict_lightgbm_gross_model(
        model,
        dataset,
        np.arange(900, 1_024, dtype=np.int64),
    )
    assert prediction.rows == 124
    assert np.all(np.isfinite(prediction.direction_probability))


def test_lightgbm_baseline_rejects_invalid_contracts(lightgbm_model_bundle) -> None:
    dataset, actual, model = lightgbm_model_bundle
    with pytest.raises(ValueError, match="split is invalid"):
        train_lightgbm_gross_baseline(
            dataset,
            actual,
            train_endpoints=np.arange(511),
            tuning_endpoints=np.arange(600, 900),
            train_uniqueness=np.ones(511),
            tuning_uniqueness=np.ones(300),
            compute_backend="cpu",
            seed=1,
        )
    with pytest.raises(ValueError, match="model contract"):
        predict_lightgbm_gross_model(
            replace(model, execution_claim=True),
            dataset,
            np.arange(900, 1_024),
        )
    with pytest.raises(ValueError, match="endpoints are invalid"):
        predict_lightgbm_gross_model(model, dataset, np.asarray([], dtype=np.int64))
    with pytest.raises(ValueError, match="cannot be reloaded"):
        predict_lightgbm_gross_model(
            replace(model, mean_model="not a LightGBM model"),
            dataset,
            np.arange(900, 1_024),
        )
