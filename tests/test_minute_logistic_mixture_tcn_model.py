from __future__ import annotations

import numpy as np
import pytest
import torch

from simple_ai_trading.cross_asset_cost_data import SYMBOLS
from simple_ai_trading.minute_logistic_mixture_analysis import (
    MinuteMixtureTrade,
    replay_fixed_trades,
)
from simple_ai_trading.minute_logistic_mixture_tcn_model import (
    FEATURE_COUNT,
    HORIZONS_MINUTES,
    RECEPTIVE_FIELD_STEPS,
    SUPERVISED_STEPS,
    WINDOW_STEPS,
    LogisticMixtureTCN,
    MinuteTemporalDataset,
    logistic_mixture_log_density,
    numpy_hurdle_probabilities,
    numpy_logistic_mixture_log_density,
    pairwise_expected_return_rank_loss,
)


def test_round48_window_contract_matches_large_kernel_receptive_field() -> None:
    assert RECEPTIVE_FIELD_STEPS == 361
    assert WINDOW_STEPS == 576
    assert SUPERVISED_STEPS == 216


def test_logistic_mixture_head_is_ordered_and_probability_coherent() -> None:
    generator = np.random.default_rng(48)
    model = LogisticMixtureTCN(components=3)
    values = torch.from_numpy(
        generator.normal(size=(2, FEATURE_COUNT, 48)).astype(np.float32)
    )
    weights, locations, scales = model(values)

    assert weights.shape == (2, len(HORIZONS_MINUTES), 3, 48)
    torch.testing.assert_close(
        torch.sum(weights, dim=2), torch.ones((2, len(HORIZONS_MINUTES), 48))
    )
    assert torch.all(locations[:, :, 1:] >= locations[:, :, :-1])
    assert torch.all(scales > 0.0)


def test_torch_and_numpy_logistic_density_match() -> None:
    generator = np.random.default_rng(4801)
    model = LogisticMixtureTCN(components=3)
    values = torch.from_numpy(
        generator.normal(size=(2, FEATURE_COUNT, 32)).astype(np.float32)
    )
    targets = torch.from_numpy(
        generator.normal(size=(2, len(HORIZONS_MINUTES), 32)).astype(np.float32)
    )
    weights, locations, scales = model(values)
    torch_density = (
        logistic_mixture_log_density(weights, locations, scales, targets)
        .detach()
        .numpy()
        .transpose(0, 2, 1)
    )
    numpy_weights = weights.detach().numpy().transpose(0, 3, 1, 2)
    numpy_locations = locations.detach().numpy().transpose(0, 3, 1, 2)
    numpy_scales = scales.detach().numpy().transpose(0, 3, 1, 2)
    numpy_targets = targets.numpy().transpose(0, 2, 1)
    numpy_density = numpy_logistic_mixture_log_density(
        numpy_weights,
        numpy_locations,
        numpy_scales,
        numpy_targets,
    )

    np.testing.assert_allclose(torch_density, numpy_density, atol=5e-7, rtol=0.0)
    probabilities = numpy_hurdle_probabilities(
        numpy_weights,
        numpy_locations,
        numpy_scales,
        np.full(len(HORIZONS_MINUTES), -0.5),
        np.full(len(HORIZONS_MINUTES), 0.5),
    )
    np.testing.assert_allclose(
        np.sum(probabilities, axis=-1), 1.0, atol=1e-7, rtol=0.0
    )


def test_pairwise_loss_prefers_the_correct_temporal_order() -> None:
    target = torch.arange(40, dtype=torch.float32).reshape(1, 1, -1)
    correct = target.clone()
    reversed_prediction = torch.flip(target, dims=(-1,))

    assert pairwise_expected_return_rank_loss(
        correct, target, offset=4
    ) < pairwise_expected_return_rank_loss(
        reversed_prediction, target, offset=4
    )


def _replay_dataset() -> MinuteTemporalDataset:
    timestamps = 16
    targets = np.zeros(
        (timestamps, len(SYMBOLS), len(HORIZONS_MINUTES)), dtype=np.float32
    )
    targets[0, 0, 0] = 20.0
    masks = {
        "training": np.zeros(timestamps, dtype=bool),
        "early_stop": np.zeros(timestamps, dtype=bool),
        "calibration": np.zeros(timestamps, dtype=bool),
        "viability": np.ones(timestamps, dtype=bool),
    }
    return MinuteTemporalDataset(
        feature_names=tuple(f"feature_{index}" for index in range(FEATURE_COUNT)),
        timestamps_ms=np.arange(timestamps, dtype=np.int64) * 300_000,
        features=np.zeros(
            (timestamps, len(SYMBOLS), FEATURE_COUNT), dtype=np.float32
        ),
        signed_target_bps=targets,
        role_masks=masks,
        feature_stream_sha256="a" * 64,
        target_stream_sha256="b" * 64,
        dataset_sha256="c" * 64,
        source_evidence={},
    )


def _trade(*, realized_target_bps: float = 20.0) -> MinuteMixtureTrade:
    return MinuteMixtureTrade(
        trade_id="round48-test-trade",
        candidate_id="single_logistic_tcn",
        symbol="BTCUSDT",
        symbol_index=0,
        decision_index=0,
        decision_time_ms=0,
        exit_time_ms=16 * 60_000,
        side=1,
        horizon_minutes=15,
        worst_seed_profit_probability=0.6,
        worst_seed_expected_net_bps=8.0,
        predicted_ensemble_mean_bps=20.0,
        realized_signed_target_bps=realized_target_bps,
    )


def test_replay_reloads_target_and_applies_base_and_stress_costs() -> None:
    dataset = _replay_dataset()
    base = replay_fixed_trades(
        dataset,
        (_trade(),),
        candidate_id="single_logistic_tcn",
        scenario="base",
        execution_charge_bps=12.0,
    )
    stress = replay_fixed_trades(
        dataset,
        (_trade(),),
        candidate_id="single_logistic_tcn",
        scenario="stress",
        execution_charge_bps=16.0,
    )

    assert base.trade_outcomes[0]["realized_net_bps"] == 8.0
    assert stress.trade_outcomes[0]["realized_net_bps"] == 4.0
    with pytest.raises(RuntimeError, match="target or overlap"):
        replay_fixed_trades(
            dataset,
            (_trade(realized_target_bps=21.0),),
            candidate_id="single_logistic_tcn",
            scenario="base",
            execution_charge_bps=12.0,
        )
