from __future__ import annotations

from dataclasses import replace
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import simple_ai_trading.microstructure_action_architecture as architecture
import simple_ai_trading.microstructure_outcome_mixture as outcome_mixture
from simple_ai_trading.compute import resolve_backend
from simple_ai_trading.microstructure_action_architecture import (
    ActionValueArchitectureSpec,
    ActionValuePredictionBatch,
    ensemble_action_value_predictions,
    predict_action_value_model,
    train_action_value_model,
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
from simple_ai_trading.microstructure_outcome_mixture import (
    OutcomeMixtureArchitectureSpec,
    load_outcome_mixture_model,
    predict_outcome_mixture_model,
    save_outcome_mixture_model,
    train_outcome_mixture_model,
)


def _spec(**overrides: object) -> ActionValueArchitectureSpec:
    values: dict[str, object] = {
        "candidate_id": "shared-action-value",
        "family": "shared_residual_mlp",
        "sequence_length": 1,
        "hidden_dim": 32,
        "residual_blocks": 1,
        "dropout": 0.0,
        "head_coherence_weight": 0.1,
        "action_utility_weight": 0.1,
        "downside_penalty": 0.25,
        "action_temperature": 0.5,
    }
    values.update(overrides)
    return ActionValueArchitectureSpec(**values)  # type: ignore[arg-type]


def _mixture_spec(**overrides: object) -> OutcomeMixtureArchitectureSpec:
    values: dict[str, object] = {
        "candidate_id": "conditional-outcome-mixture",
        "family": "conditional_outcome_mixture_residual_mlp",
        "sequence_length": 1,
        "hidden_dim": 32,
        "residual_blocks": 1,
        "dropout": 0.0,
        "probability_loss_weight": 0.25,
        "magnitude_loss_weight": 0.35,
        "expected_value_loss_weight": 1.0,
        "quantile_loss_weight": 0.15,
        "ranking_loss_weight": 0.1,
    }
    values.update(overrides)
    return OutcomeMixtureArchitectureSpec(**values)  # type: ignore[arg-type]


def _dataset(rows: int = 1_200) -> tuple[MicrostructureDataset, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    features = rng.normal(size=(rows, len(MICROSTRUCTURE_FEATURE_NAMES))).astype(
        np.float32
    )
    signal = 7.0 * features[:, 0] + 3.0 * features[:, 1]
    noise = rng.normal(scale=2.0, size=rows)
    long_target = signal - 2.0 + noise
    short_target = -signal - 2.0 - noise
    ones = np.ones(rows, dtype=np.float64)
    times = np.arange(rows, dtype=np.int64) * 5_000 + 10_000
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
        long_exit_time_ms=times + 900_750,
        short_exit_time_ms=times + 900_750,
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
    return dataset, long_target, short_target


def _barrier_targets(
    dataset: MicrostructureDataset,
    long_target: np.ndarray,
    short_target: np.ndarray,
) -> AdaptiveBarrierTargets:
    rows = dataset.rows
    exits = dataset.decision_time_ms + dataset.total_latency_ms + 900_000
    outcomes = np.zeros(rows, dtype=np.int8)
    return AdaptiveBarrierTargets(
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


@pytest.fixture(scope="module")
def action_model_bundle():
    dataset, long_target, short_target = _dataset()
    targets = _barrier_targets(dataset, long_target, short_target)
    model = train_action_value_model(
        dataset,
        targets,
        train_endpoints=np.arange(600, dtype=np.int64),
        tuning_endpoints=np.arange(800, 1_100, dtype=np.int64),
        spec=_spec(),
        target_scenario="base",
        compute_backend="cpu",
        seed=17,
        batch_size=128,
        max_epochs=2,
        patience=2,
    )
    return dataset, targets, model


def test_action_value_model_trains_scores_and_binds_after_cost_target(
    action_model_bundle,
) -> None:
    dataset, _targets, model = action_model_bundle

    prediction = predict_action_value_model(
        model,
        dataset,
        np.arange(800, 1_100, dtype=np.int64),
        compute_backend="cpu",
        batch_size=128,
    )

    assert model.target_mode.startswith("exchange_trigger_market_exit_100ms")
    assert len(model.target_contract_sha256) == 64
    assert model.target_scenario == "base"
    assert model.optimizer_kind == "manual_adam_tensor_native_v1"
    assert model.training_data_mode == "device_preloaded"
    assert len(model.model_sha256) == 64
    assert prediction.rows == 300
    assert np.all(np.isfinite(prediction.long_mean_bps))
    assert np.all((prediction.long_profitable_probability > 0.0))
    assert np.all((prediction.long_profitable_probability < 1.0))
    assert np.all(prediction.long_lower_bps <= prediction.long_mean_bps)
    assert np.all(prediction.long_mean_bps <= prediction.long_upper_bps)
    assert np.all(prediction.short_lower_bps <= prediction.short_mean_bps)
    assert np.all(prediction.short_mean_bps <= prediction.short_upper_bps)
    assert model.trading_authority is False


def test_action_value_training_rejects_nonfinite_role_target() -> None:
    dataset, long_target, short_target = _dataset()
    targets = _barrier_targets(dataset, long_target.copy(), short_target.copy())
    targets.base_long_net_bps[900] = np.nan

    with pytest.raises(ValueError, match="validity mask"):
        train_action_value_model(
            dataset,
            targets,
            train_endpoints=np.arange(600, dtype=np.int64),
            tuning_endpoints=np.arange(800, 1_100, dtype=np.int64),
            spec=_spec(),
            target_scenario="stress",
            compute_backend="cpu",
            seed=17,
            batch_size=128,
            max_epochs=1,
            patience=1,
        )


def test_action_value_training_rejects_invalid_role_row() -> None:
    dataset, long_target, short_target = _dataset()
    targets = _barrier_targets(dataset, long_target.copy(), short_target.copy())
    targets.valid[900] = False
    for values in (
        targets.base_long_net_bps,
        targets.base_short_net_bps,
        targets.stress_long_net_bps,
        targets.stress_short_net_bps,
    ):
        values[900] = np.nan
    for values in (
        targets.base_long_exit_time_ms,
        targets.base_short_exit_time_ms,
        targets.stress_long_exit_time_ms,
        targets.stress_short_exit_time_ms,
        targets.base_long_outcome,
        targets.base_short_outcome,
        targets.stress_long_outcome,
        targets.stress_short_outcome,
    ):
        values[900] = -1

    with pytest.raises(ValueError, match="split targets are non-finite"):
        train_action_value_model(
            dataset,
            targets,
            train_endpoints=np.arange(600, dtype=np.int64),
            tuning_endpoints=np.arange(800, 1_100, dtype=np.int64),
            spec=_spec(),
            target_scenario="base",
            compute_backend="cpu",
            seed=17,
            batch_size=128,
            max_epochs=1,
            patience=1,
        )


def test_action_value_training_rejects_unpurged_target_lifecycles() -> None:
    dataset, long_target, short_target = _dataset()

    with pytest.raises(ValueError, match="not purged"):
        train_action_value_model(
            dataset,
            _barrier_targets(dataset, long_target, short_target),
            train_endpoints=np.arange(600, dtype=np.int64),
            tuning_endpoints=np.arange(600, 900, dtype=np.int64),
            spec=_spec(),
            target_scenario="base",
            compute_backend="cpu",
            seed=17,
            batch_size=128,
            max_epochs=1,
            patience=1,
        )


def test_action_value_prediction_rejects_tampered_model_state(
    action_model_bundle,
) -> None:
    dataset, _targets, model = action_model_bundle
    state = {name: value.copy() for name, value in model.state.items()}
    first_name = next(iter(state))
    state[first_name].flat[0] += 1.0

    with pytest.raises(ValueError, match="contract is invalid"):
        predict_action_value_model(
            replace(model, state=state),
            dataset,
            np.arange(800, 1_100, dtype=np.int64),
            compute_backend="cpu",
            batch_size=128,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"target_scenario": "future"}, "scenario is unsupported"),
        ({"train_endpoints": np.arange(500)}, "insufficient contiguous"),
        (
            {
                "train_endpoints": np.arange(100, 700),
                "tuning_endpoints": np.arange(400, 700),
            },
            "not chronological",
        ),
        ({"batch_size": 16}, "budget is invalid"),
        ({"train_sample_weights": np.ones(10)}, "sample weights are invalid"),
    ],
)
def test_action_value_training_rejects_invalid_split_and_budget_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    dataset, long_target, short_target = _dataset()
    parameters: dict[str, object] = {
        "train_endpoints": np.arange(600, dtype=np.int64),
        "tuning_endpoints": np.arange(800, 1_100, dtype=np.int64),
        "spec": _spec(),
        "target_scenario": "base",
        "compute_backend": "cpu",
        "seed": 17,
        "batch_size": 128,
        "max_epochs": 1,
        "patience": 1,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=match):
        train_action_value_model(
            dataset,
            _barrier_targets(dataset, long_target, short_target),
            **parameters,  # type: ignore[arg-type]
        )


def test_action_value_training_streams_stress_targets_and_reports_progress(
    monkeypatch,
) -> None:
    dataset, long_target, short_target = _dataset()
    progress: list[tuple[int, int, float, float]] = []
    monkeypatch.setattr(architecture, "_TRAINING_PRELOAD_LIMIT_BYTES", 0)

    model = train_action_value_model(
        dataset,
        _barrier_targets(dataset, long_target, short_target),
        train_endpoints=np.arange(600, dtype=np.int64),
        tuning_endpoints=np.arange(800, 1_100, dtype=np.int64),
        spec=_spec(),
        target_scenario="stress",
        compute_backend="cpu",
        seed=17,
        batch_size=256,
        max_epochs=1,
        patience=1,
        train_sample_weights=np.linspace(0.5, 1.5, 600, dtype=np.float32),
        tuning_sample_weights=np.linspace(0.5, 1.5, 300, dtype=np.float32),
        progress=lambda *values: progress.append(values),
    )

    assert model.target_scenario == "stress"
    assert model.training_data_mode == "streamed_host_batches"
    assert model.training_preload_bytes == 0
    assert len(progress) == 1


def test_action_value_training_early_stops_and_rejects_nonfinite_loss(
    monkeypatch,
) -> None:
    dataset, long_target, short_target = _dataset()
    targets = _barrier_targets(dataset, long_target, short_target)
    tuning_calls = 0

    def worsening_loss(output, _target, _weight, _class_weight, _spec):
        nonlocal tuning_calls
        value = 1.0
        if output.shape[0] == 300:
            tuning_calls += 1
            value = float(tuning_calls)
        return output.sum() * 0.0 + value

    monkeypatch.setattr(architecture, "_loss", worsening_loss)
    model = train_action_value_model(
        dataset,
        targets,
        train_endpoints=np.arange(600, dtype=np.int64),
        tuning_endpoints=np.arange(800, 1_100, dtype=np.int64),
        spec=_spec(),
        target_scenario="base",
        compute_backend="cpu",
        seed=17,
        batch_size=2_048,
        max_epochs=4,
        patience=2,
    )
    assert model.best_epoch == 1
    assert tuning_calls == 3

    def nonfinite_loss(output, _target, _weight, _class_weight, _spec):
        return output.sum() * float("nan")

    monkeypatch.setattr(architecture, "_loss", nonfinite_loss)
    with pytest.raises(ValueError, match="loss became non-finite"):
        train_action_value_model(
            dataset,
            targets,
            train_endpoints=np.arange(600, dtype=np.int64),
            tuning_endpoints=np.arange(800, 1_100, dtype=np.int64),
            spec=_spec(),
            target_scenario="base",
            compute_backend="cpu",
            seed=17,
            batch_size=2_048,
            max_epochs=1,
            patience=1,
        )


def test_action_value_training_and_inference_run_on_directml_when_available() -> None:
    if resolve_backend("directml").kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    if os.environ.get("SIMPLE_AI_TRADING_DIRECTML_TEST_CHILD") != "1":
        environment = os.environ.copy()
        environment["SIMPLE_AI_TRADING_DIRECTML_TEST_CHILD"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::{test_action_value_training_and_inference_run_on_directml_when_available.__name__}",
            ],
            cwd=os.getcwd(),
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "will fall back to run on the CPU" not in output
        return
    dataset, long_target, short_target = _dataset()
    model = train_action_value_model(
        dataset,
        _barrier_targets(dataset, long_target, short_target),
        train_endpoints=np.arange(600, dtype=np.int64),
        tuning_endpoints=np.arange(800, 1_100, dtype=np.int64),
        spec=_spec(),
        target_scenario="base",
        compute_backend="directml",
        seed=17,
        batch_size=256,
        max_epochs=1,
        patience=1,
    )
    prediction = predict_action_value_model(
        model,
        dataset,
        np.arange(800, 1_100, dtype=np.int64),
        compute_backend="directml",
        batch_size=256,
    )

    assert model.backend_kind == "directml"
    assert model.backend_device == "privateuseone:0"
    assert prediction.rows == 300
    assert np.all(np.isfinite(prediction.long_mean_bps))


def test_outcome_mixture_training_reload_and_inference_run_on_directml_when_available(
    tmp_path,
) -> None:
    if resolve_backend("directml").kind != "directml":
        pytest.skip("DirectML backend is not available on this host")
    if os.environ.get("SIMPLE_AI_TRADING_OUTCOME_DML_TEST_CHILD") != "1":
        environment = os.environ.copy()
        environment["SIMPLE_AI_TRADING_OUTCOME_DML_TEST_CHILD"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::{test_outcome_mixture_training_reload_and_inference_run_on_directml_when_available.__name__}",
            ],
            cwd=os.getcwd(),
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "will fall back to run on the CPU" not in output
        return
    dataset, long_target, short_target = _dataset()
    model = train_outcome_mixture_model(
        dataset,
        _barrier_targets(dataset, long_target, short_target),
        train_endpoints=np.arange(600, dtype=np.int64),
        tuning_endpoints=np.arange(800, 1_100, dtype=np.int64),
        spec=_mixture_spec(
            sequence_length=7,
            pairwise_ranking_loss_weight=0.02,
            temporal_pooling_mode="causal_attention",
            ranking_scope="utc_session",
        ),
        target_scenario="base",
        compute_backend="directml",
        seed=23,
        batch_size=256,
        max_epochs=1,
        patience=1,
    )
    artifact = tmp_path / "directml.safetensors"
    save_outcome_mixture_model(artifact, model)
    reloaded = load_outcome_mixture_model(artifact)
    prediction = predict_outcome_mixture_model(
        reloaded,
        dataset,
        np.arange(800, 1_100, dtype=np.int64),
        compute_backend="directml",
        batch_size=256,
    )

    assert reloaded.backend_kind == "directml"
    assert reloaded.backend_device == "privateuseone:0"
    assert reloaded.spec.ranking_loss_mode == "correlation"
    assert reloaded.spec.pairwise_ranking_loss_weight == 0.02
    assert reloaded.spec.temporal_pooling_mode == "causal_attention"
    assert reloaded.spec.ranking_scope == "utc_session"
    assert reloaded.spec.sequence_length == 7
    assert prediction.rows == 300
    assert np.all(np.isfinite(prediction.long_mean_bps))


def _prediction(offset: float = 0.0) -> ActionValuePredictionBatch:
    endpoints = np.asarray([10, 20], dtype=np.int64)
    return ActionValuePredictionBatch(
        endpoint_indexes=endpoints,
        long_mean_bps=np.asarray([1.0, -1.0]) + offset,
        short_mean_bps=np.asarray([-2.0, 2.0]) - offset,
        long_profitable_probability=np.asarray([0.6, 0.4]),
        short_profitable_probability=np.asarray([0.3, 0.7]),
        long_lower_bps=np.asarray([-1.0, -3.0]),
        short_lower_bps=np.asarray([-4.0, 0.0]),
        long_upper_bps=np.asarray([3.0, 1.0]),
        short_upper_bps=np.asarray([0.0, 4.0]),
    )


def test_action_value_ensemble_exposes_epistemic_dispersion_and_agreement() -> None:
    ensemble = ensemble_action_value_predictions([_prediction(), _prediction(2.0)])

    np.testing.assert_allclose(ensemble.long_mean_bps, [2.0, 0.0])
    np.testing.assert_allclose(ensemble.long_epistemic_std_bps, [1.0, 1.0])
    np.testing.assert_allclose(ensemble.long_positive_member_ratio, [1.0, 0.5])
    assert ensemble.member_count == 2
    assert ensemble.rows == 2
    assert ensemble.trading_authority is False


def test_action_value_ensemble_and_model_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ensemble_action_value_predictions([_prediction()])
    mismatched = replace(_prediction(), endpoint_indexes=np.asarray([10, 21]))
    with pytest.raises(ValueError, match="endpoint identities"):
        ensemble_action_value_predictions([_prediction(), mismatched])
    invalid_probability = replace(
        _prediction(), long_profitable_probability=np.asarray([1.1, 0.4])
    )
    with pytest.raises(ValueError, match="member arrays"):
        ensemble_action_value_predictions([_prediction(), invalid_probability])
    crossed_interval = replace(_prediction(), long_lower_bps=np.asarray([4.0, -3.0]))
    with pytest.raises(ValueError, match="member arrays"):
        ensemble_action_value_predictions([_prediction(), crossed_interval])


def test_action_value_prediction_rejects_invalid_request_contracts(
    action_model_bundle,
) -> None:
    dataset, _targets, model = action_model_bundle
    with pytest.raises(ValueError, match="batch size must be positive"):
        predict_action_value_model(
            model,
            dataset,
            np.arange(800, 1_100, dtype=np.int64),
            compute_backend="cpu",
            batch_size=0,
        )
    with pytest.raises(ValueError, match="no contiguous endpoints"):
        predict_action_value_model(
            model,
            dataset,
            np.asarray([], dtype=np.int64),
            compute_backend="cpu",
            batch_size=128,
        )


def test_action_value_prediction_rejects_nonfinite_network_output(
    action_model_bundle,
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    dataset, _targets, model = action_model_bundle

    class InvalidNetwork:
        def to(self, _device):
            return self

        def load_state_dict(self, _state, *, strict: bool) -> None:
            assert strict is True

        def eval(self) -> None:
            return None

        def __call__(self, values):
            return torch.full(
                (len(values), 2, 4),
                float("nan"),
                dtype=values.dtype,
                device=values.device,
            )

    monkeypatch.setattr(
        architecture, "_network", lambda _spec, _features: InvalidNetwork()
    )
    with pytest.raises(ValueError, match="emitted invalid predictions"):
        predict_action_value_model(
            model,
            dataset,
            np.arange(800, 1_100, dtype=np.int64),
            compute_backend="cpu",
            batch_size=128,
        )


def test_action_value_private_class_weight_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="targets are invalid"):
        architecture._positive_class_weights(np.ones(100))
    with pytest.raises(ValueError, match="class support"):
        architecture._positive_class_weights(np.ones((200, 2)))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"candidate_id": ""}, "cannot be empty"),
        ({"family": "transformer"}, "unsupported"),
        ({"hidden_dim": 8}, "dimensions"),
        ({"dropout": float("nan")}, "must be finite"),
        ({"action_temperature": 0.0}, "outside bounds"),
    ],
)
def test_action_value_spec_rejects_invalid_contracts(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _spec(**overrides)


@pytest.fixture(scope="module")
def outcome_mixture_bundle():
    dataset, long_target, short_target = _dataset()
    targets = _barrier_targets(dataset, long_target, short_target)
    model = train_outcome_mixture_model(
        dataset,
        targets,
        train_endpoints=np.arange(600, dtype=np.int64),
        tuning_endpoints=np.arange(800, 1_100, dtype=np.int64),
        spec=_mixture_spec(),
        target_scenario="base",
        compute_backend="cpu",
        seed=23,
        batch_size=128,
        max_epochs=2,
        patience=2,
    )
    return dataset, targets, model


def test_outcome_mixture_decodes_expected_value_from_conditional_identity() -> None:
    torch = pytest.importorskip("torch")
    raw = torch.tensor(
        [
            [
                [0.4, 1.2, -0.3, 0.1, 0.2],
                [-0.7, -0.2, 0.8, -0.1, 0.4],
            ]
        ],
        dtype=torch.float32,
    )

    probability, positive, negative, expected, lower, upper = (
        outcome_mixture._decoded_heads(raw)
    )

    torch.testing.assert_close(
        expected,
        probability * positive - (1.0 - probability) * negative,
    )
    assert bool(torch.all(positive > 0.0))
    assert bool(torch.all(negative > 0.0))
    assert bool(torch.all(lower <= expected))
    assert bool(torch.all(expected <= upper))


def test_outcome_mixture_pairwise_ranking_prefers_realized_net_return_order() -> None:
    torch = pytest.importorskip("torch")
    target = torch.linspace(-1.5, 1.5, 18, dtype=torch.float32)[:, None].repeat(1, 2)
    sample_weight = torch.linspace(0.5, 1.5, 18, dtype=torch.float32)
    ordered = target * 0.75 + 4.0
    reversed_order = torch.flip(ordered, dims=(0,))

    ordered_loss = outcome_mixture._weighted_pairwise_ranking_loss(
        ordered, target, sample_weight
    )
    reversed_loss = outcome_mixture._weighted_pairwise_ranking_loss(
        reversed_order, target, sample_weight
    )
    tied_loss = outcome_mixture._weighted_pairwise_ranking_loss(
        ordered, torch.zeros_like(target), sample_weight
    )

    assert float(ordered_loss) < float(reversed_loss)
    assert float(tied_loss) == 0.0


def test_outcome_mixture_session_local_ranking_rejects_cross_session_levels() -> None:
    torch = pytest.importorskip("torch")
    prediction = torch.tensor(
        [[1.0, 1.0], [0.0, 0.0], [11.0, 11.0], [10.0, 10.0]]
    )
    target = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]]
    )
    weight = torch.ones(4)
    slices = outcome_mixture._contiguous_group_slices(
        np.asarray([100, 100, 101, 101], dtype=np.int64)
    )

    global_loss = outcome_mixture._scoped_ranking_loss(
        prediction,
        target,
        weight,
        mode="correlation",
        group_slices=None,
    )
    local_loss = outcome_mixture._scoped_ranking_loss(
        prediction,
        target,
        weight,
        mode="correlation",
        group_slices=slices,
    )

    assert slices == ((0, 2), (2, 4))
    assert outcome_mixture._contiguous_group_slices(np.asarray([100])) == ()
    assert float(global_loss) < 0.1
    assert float(local_loss) > 1.9
    with pytest.raises(ValueError, match="not contiguous"):
        outcome_mixture._contiguous_group_slices(
            np.asarray([100, 101, 100, 101], dtype=np.int64)
        )


def test_outcome_mixture_session_training_order_is_seeded_and_group_contiguous() -> (
    None
):
    sessions = np.asarray([100, 100, 100, 101, 101, 102, 102, 102], dtype=np.int64)
    first = outcome_mixture._session_grouped_training_order(
        sessions, np.random.default_rng(73)
    )
    second = outcome_mixture._session_grouped_training_order(
        sessions, np.random.default_rng(73)
    )
    ordered_sessions = sessions[first]

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(len(sessions)))
    assert len(outcome_mixture._contiguous_group_slices(ordered_sessions)) == 3
    with pytest.raises(ValueError, match="sessions are invalid"):
        outcome_mixture._session_grouped_training_order(
            sessions[::-1], np.random.default_rng(73)
        )


def test_outcome_mixture_directml_cpu_fallback_guard_fails_closed() -> None:
    message = "The operator will fall back to run on the CPU."
    with pytest.raises(UserWarning, match="fall back"):
        with outcome_mixture._directml_cpu_fallback_guard("directml"):
            warnings.warn(message, UserWarning, stacklevel=1)
    with pytest.warns(UserWarning, match="fall back"):
        with outcome_mixture._directml_cpu_fallback_guard("cpu"):
            warnings.warn(message, UserWarning, stacklevel=1)


def test_outcome_mixture_independent_towers_are_parameter_matched_and_isolated() -> (
    None
):
    torch = pytest.importorskip("torch")
    feature_count = len(MICROSTRUCTURE_FEATURE_NAMES)
    shared = outcome_mixture._network(
        _mixture_spec(hidden_dim=128, residual_blocks=2), feature_count
    )
    independent = outcome_mixture._network(
        _mixture_spec(
            hidden_dim=88,
            residual_blocks=2,
            side_tower_mode="independent",
        ),
        feature_count,
    )
    shared_parameters = sum(parameter.numel() for parameter in shared.parameters())
    independent_parameters = sum(
        parameter.numel() for parameter in independent.parameters()
    )

    assert shared_parameters == 147_722
    assert independent_parameters == 145_914
    assert abs(independent_parameters / shared_parameters - 1.0) < 0.02
    assert all(name.startswith("towers.") for name in independent.state_dict())

    values = torch.randn(8, 1, feature_count)
    before = independent(values).detach().clone()
    with torch.no_grad():
        independent.towers[0].head.bias.add_(1.0)
    after = independent(values).detach()

    assert not torch.equal(before[:, 0, :], after[:, 0, :])
    torch.testing.assert_close(before[:, 1, :], after[:, 1, :], rtol=0.0, atol=0.0)


def test_outcome_mixture_causal_attention_uses_bounded_history() -> None:
    torch = pytest.importorskip("torch")
    feature_count = len(MICROSTRUCTURE_FEATURE_NAMES)
    network = outcome_mixture._network(
        _mixture_spec(
            sequence_length=7,
            hidden_dim=88,
            residual_blocks=2,
            side_tower_mode="independent",
            temporal_pooling_mode="causal_attention",
        ),
        feature_count,
    )
    network.eval()
    parameters = sum(parameter.numel() for parameter in network.parameters())
    values = torch.randn(8, 7, feature_count)
    changed_history = values.clone()
    changed_history[:, :-1, :] += 0.5

    baseline = network(values).detach()
    changed = network(changed_history).detach()

    assert parameters == 146_090
    assert not torch.equal(baseline, changed)
    assert torch.equal(values[:, -1, :], changed_history[:, -1, :])
    assert (
        sum("temporal_attention.weight" in name for name in network.state_dict()) == 2
    )


def test_outcome_mixture_sample_weights_align_by_sequence_endpoint_identity() -> None:
    requested = np.arange(100, 110, dtype=np.int64)
    selected = np.asarray([102, 105, 109], dtype=np.int64)
    requested_weights = np.linspace(0.5, 1.4, len(requested), dtype=np.float32)

    aligned = outcome_mixture._align_sample_weights(
        requested, selected, requested_weights
    )

    np.testing.assert_array_equal(aligned, requested_weights[[2, 5, 9]])
    with pytest.raises(ValueError, match="sample weights are invalid"):
        outcome_mixture._align_sample_weights(
            requested,
            np.asarray([102, 111], dtype=np.int64),
            requested_weights,
        )


def test_outcome_mixture_independent_tower_artifact_round_trip(tmp_path) -> None:
    dataset, long_target, short_target = _dataset()
    targets = _barrier_targets(dataset, long_target, short_target)
    model = train_outcome_mixture_model(
        dataset,
        targets,
        train_endpoints=np.arange(600, dtype=np.int64),
        tuning_endpoints=np.arange(800, 1_100, dtype=np.int64),
        spec=_mixture_spec(
            sequence_length=7,
            hidden_dim=16,
            side_tower_mode="independent",
            ranking_loss_mode="pairwise_net_return",
            temporal_pooling_mode="causal_attention",
            ranking_scope="utc_session",
        ),
        target_scenario="base",
        compute_backend="cpu",
        seed=29,
        batch_size=256,
        max_epochs=1,
        patience=1,
        train_sample_weights=np.linspace(0.5, 1.5, 600, dtype=np.float32),
        tuning_sample_weights=np.linspace(0.5, 1.5, 300, dtype=np.float32),
    )
    path = tmp_path / "independent-towers.safetensors"
    repeated_path = tmp_path / "independent-towers-repeated.safetensors"

    save_outcome_mixture_model(path, model)
    save_outcome_mixture_model(repeated_path, model)
    loaded = load_outcome_mixture_model(path)

    assert path.read_bytes() == repeated_path.read_bytes()
    assert loaded.spec.side_tower_mode == "independent"
    assert loaded.spec.ranking_loss_mode == "pairwise_net_return"
    assert loaded.spec.temporal_pooling_mode == "causal_attention"
    assert loaded.spec.ranking_scope == "utc_session"
    assert loaded.model_sha256 == model.model_sha256
    assert loaded.state.keys() == model.state.keys()


def test_outcome_mixture_trains_predicts_and_binds_after_cost_target(
    outcome_mixture_bundle,
) -> None:
    dataset, _targets, model = outcome_mixture_bundle

    prediction = predict_outcome_mixture_model(
        model,
        dataset,
        np.arange(800, 1_100, dtype=np.int64),
        compute_backend="cpu",
        batch_size=128,
    )

    assert model.target_mode.startswith("exchange_trigger_market_exit_100ms")
    assert model.spec.family == "conditional_outcome_mixture_residual_mlp"
    assert len(model.target_contract_sha256) == 64
    assert len(model.model_sha256) == 64
    assert model.optimizer_kind == "manual_adam_tensor_native_v1"
    assert model.training_data_mode == "device_preloaded"
    assert all(0.0 < value < 1.0 for value in model.positive_class_prevalence)
    assert prediction.rows == 300
    assert np.all(np.isfinite(prediction.long_mean_bps))
    assert np.all((prediction.long_profitable_probability > 0.0))
    assert np.all((prediction.long_profitable_probability < 1.0))
    assert np.all(prediction.long_lower_bps <= prediction.long_mean_bps)
    assert np.all(prediction.long_mean_bps <= prediction.long_upper_bps)
    assert np.all(prediction.short_lower_bps <= prediction.short_mean_bps)
    assert np.all(prediction.short_mean_bps <= prediction.short_upper_bps)
    assert model.trading_authority is False


def test_outcome_mixture_artifact_round_trip_is_prediction_exact(
    tmp_path,
    outcome_mixture_bundle,
) -> None:
    dataset, _targets, model = outcome_mixture_bundle
    path = tmp_path / "model.safetensors"
    endpoints = np.arange(800, 1_100, dtype=np.int64)
    expected = predict_outcome_mixture_model(
        model,
        dataset,
        endpoints,
        compute_backend="cpu",
        batch_size=128,
    )

    save_outcome_mixture_model(path, model)
    loaded = load_outcome_mixture_model(path)
    actual = predict_outcome_mixture_model(
        loaded,
        dataset,
        endpoints,
        compute_backend="cpu",
        batch_size=128,
    )

    assert loaded.model_sha256 == model.model_sha256
    assert loaded.spec == model.spec
    assert loaded.target_spec == model.target_spec
    assert loaded.positive_class_prevalence == model.positive_class_prevalence
    for name in (
        "long_mean_bps",
        "short_mean_bps",
        "long_profitable_probability",
        "short_profitable_probability",
        "long_lower_bps",
        "short_lower_bps",
        "long_upper_bps",
        "short_upper_bps",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_outcome_mixture_artifact_save_is_atomic_on_failure(
    tmp_path,
    outcome_mixture_bundle,
    monkeypatch,
) -> None:
    _dataset_value, _targets, model = outcome_mixture_bundle
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"existing")

    def fail_save(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(outcome_mixture, "save_safetensors", fail_save)
    with pytest.raises(OSError, match="simulated"):
        save_outcome_mixture_model(path, model)

    assert path.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [path]


def test_outcome_mixture_model_and_artifact_contracts_fail_closed(
    tmp_path,
    outcome_mixture_bundle,
) -> None:
    dataset, _targets, model = outcome_mixture_bundle
    endpoints = np.arange(800, 1_100, dtype=np.int64)
    with pytest.raises(ValueError, match="contract is invalid"):
        predict_outcome_mixture_model(
            replace(model, tuning_loss=model.tuning_loss + 0.01),
            dataset,
            endpoints,
            compute_backend="cpu",
            batch_size=128,
        )

    path = tmp_path / "truncated.safetensors"
    path.write_bytes(b"not-safetensors")
    with pytest.raises(ValueError, match="not valid safetensors"):
        load_outcome_mixture_model(path)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"candidate_id": ""}, "cannot be empty"),
        ({"family": "shared_residual_mlp"}, "unsupported"),
        ({"side_tower_mode": "coupled"}, "unsupported"),
        ({"ranking_loss_mode": "listwise"}, "unsupported"),
        ({"sequence_length": 7}, "unsupported"),
        (
            {"sequence_length": 1, "temporal_pooling_mode": "causal_attention"},
            "unsupported",
        ),
        ({"temporal_pooling_mode": "bidirectional"}, "unsupported"),
        ({"ranking_scope": "rolling_week"}, "unsupported"),
        (
            {
                "ranking_loss_mode": "pairwise_net_return",
                "pairwise_ranking_loss_weight": 0.1,
            },
            "outside bounds",
        ),
        ({"hidden_dim": 8}, "dimensions"),
        ({"dropout": float("nan")}, "must be finite"),
        ({"expected_value_loss_weight": -1.0}, "outside bounds"),
    ],
)
def test_outcome_mixture_spec_rejects_invalid_contracts(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _mixture_spec(**overrides)
