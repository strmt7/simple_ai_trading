from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from simple_ai_trading.distributional_tcn_model import DistributionalDataset
from simple_ai_trading.sequential_action_value_model import (
    DEFAULT_SPEC,
    DuelingDistributionalQTCN,
    SequentialQSpec,
    bellman_quantile_targets,
    consensus_policy_actions,
    quantile_huber_loss,
    replay_consensus_actions,
    transition_reward_bps,
)


def test_transition_reward_accounts_for_position_change_units() -> None:
    rewards = transition_reward_bps(np.asarray([10.0]), one_way_cost_bps=2.0)

    assert rewards.shape == (1, 3, 3)
    assert rewards[0, 1, 2] == pytest.approx(8.0)
    assert rewards[0, 2, 2] == pytest.approx(10.0)
    assert rewards[0, 2, 0] == pytest.approx(-14.0)


def test_distributional_q_network_is_monotone_and_position_conditioned() -> None:
    model = DuelingDistributionalQTCN(input_channels=71)
    values = torch.zeros((2, 71, 64), dtype=torch.float32)

    output = model(values).detach().numpy()

    assert output.shape == (2, 3, 3, 5, 64)
    assert np.all(np.diff(output, axis=3) >= 0.0)


def test_quantile_huber_loss_is_zero_for_equal_degenerate_distributions() -> None:
    values = torch.zeros((2, 3, 3, 5, 7), dtype=torch.float32)

    assert float(quantile_huber_loss(values, values)) == pytest.approx(0.0)


def test_bellman_target_uses_next_action_for_current_action_position() -> None:
    returns = torch.zeros((1, 2), dtype=torch.float32)
    online = torch.zeros((1, 3, 3, 5, 2), dtype=torch.float32)
    target = torch.zeros_like(online)
    online[:, :, 2] = 2.0
    target[:, 0, 2] = 3.0
    target[:, 1, 2] = 4.0
    target[:, 2, 2] = 5.0

    result = bellman_quantile_targets(
        returns,
        online,
        target,
        normalized_one_way_cost=1.0,
        discount_factor=0.5,
    )

    assert result.shape == (1, 3, 3, 5, 2)
    assert result[0, 1, 2, 2, 0].item() == pytest.approx(1.5)
    assert result[0, 1, 0, 2, 0].item() == pytest.approx(0.5)


def test_all_seed_consensus_changes_once_then_holds() -> None:
    seed_q = np.zeros((3, 2, 3, 3, 3, 5), dtype=np.float32)
    seed_q[..., 2, :] = 1.0

    actions, diagnostics = consensus_policy_actions(
        seed_q,
        np.asarray([True, True]),
        "median_q_all_seed_consensus",
    )

    assert np.array_equal(actions, np.ones((2, 3), dtype=np.int8))
    assert diagnostics["position_changes"] == 3
    assert diagnostics["unanimous_fraction"] == pytest.approx(1.0)


def test_replay_charges_entry_and_forced_terminal_close() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    timestamps = np.asarray(
        [int((start + timedelta(hours=index)).timestamp() * 1_000) for index in range(3)],
        dtype=np.int64,
    )
    dataset = DistributionalDataset(
        feature_names=tuple(f"feature_{index}" for index in range(71)),
        timestamps_ms=timestamps,
        features=np.zeros((3, 3, 71), dtype=np.float32),
        hourly_return_bps=np.full((3, 3), 100.0, dtype=np.float32),
        forward_return_bps=np.zeros((3, 3, 4), dtype=np.float32),
        dataset_sha256="test",
    )
    actions = np.ones((3, 3), dtype=np.int8)
    spec = SequentialQSpec(bootstrap_samples=20, bootstrap_block_hours=2)

    replay = replay_consensus_actions(
        dataset,
        actions,
        np.ones(3, dtype=bool),
        policy_id="median_q_all_seed_consensus",
        role="test",
        scenario="base",
        one_way_cost_bps=6.0,
        bootstrap_seed=7,
        spec=spec,
    )

    assert np.sum(replay.symbol_net_bps[:, 0]) == pytest.approx(288.0)
    assert replay.metrics["transition_units"] == 6
    assert replay.metrics["closed_trades"] == 3
    assert replay.metrics["symbol_closed_trades"] == {
        "BTCUSDT": 1,
        "ETHUSDT": 1,
        "SOLUSDT": 1,
    }


def test_default_spec_matches_frozen_receptive_field() -> None:
    assert DEFAULT_SPEC.receptive_field == 127
    assert DEFAULT_SPEC.supervised_start == 126
