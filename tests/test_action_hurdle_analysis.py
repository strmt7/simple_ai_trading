from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading.action_hurdle_analysis import (
    replay_fixed_trades,
    select_fixed_policy_trades,
)
from simple_ai_trading.action_hurdle_tcn_model import FEATURE_COUNT
from simple_ai_trading.cross_asset_cost_data import SYMBOLS
from simple_ai_trading.minute_logistic_mixture_tcn_model import MinuteTemporalDataset


def _dataset() -> MinuteTemporalDataset:
    timestamps = 20
    signed = np.zeros((timestamps, len(SYMBOLS), 4), dtype=np.float32)
    signed[:, 0, 0] = 30.0
    signed[:, 1, 0] = -25.0
    training = np.zeros(timestamps, dtype=bool)
    training[:4] = True
    viability = np.zeros(timestamps, dtype=bool)
    viability[4:] = True
    return MinuteTemporalDataset(
        feature_names=tuple(f"feature_{index}" for index in range(FEATURE_COUNT)),
        timestamps_ms=np.arange(timestamps, dtype=np.int64) * 300_000,
        features=np.zeros((timestamps, len(SYMBOLS), FEATURE_COUNT), dtype=np.float32),
        signed_target_bps=signed,
        role_masks={
            "training": training,
            "early_stop": np.arange(timestamps) == 2,
            "calibration": np.arange(timestamps) == 3,
            "viability": viability,
        },
        feature_stream_sha256="a" * 64,
        target_stream_sha256="b" * 64,
        dataset_sha256="c" * 64,
        source_evidence={},
    )


def _bundle(dataset: MinuteTemporalDataset) -> SimpleNamespace:
    probabilities = np.full(
        (3, dataset.timestamps, len(SYMBOLS), 2, 2), 0.6, dtype=np.float32
    )
    action_values = np.full(
        (3, dataset.timestamps, len(SYMBOLS), 2), -1.0, dtype=np.float32
    )
    action_values[:, :, 0, 1] = np.asarray([2.0, 3.0, 4.0])[:, None]
    action_values[:, :, 1, 0] = np.asarray([1.0, 2.0, 3.0])[:, None]
    return SimpleNamespace(
        candidate_id="direct_action_mean_tcn",
        global_indices=np.arange(dataset.timestamps, dtype=np.int64),
        seed_probabilities=probabilities,
        seed_action_values_bps=action_values,
    )


def test_fixed_policy_uses_worst_seed_and_prevents_same_symbol_overlap() -> None:
    dataset = _dataset()
    trades = select_fixed_policy_trades(dataset, _bundle(dataset))

    btc = [trade for trade in trades if trade.symbol == "BTCUSDT"]
    eth = [trade for trade in trades if trade.symbol == "ETHUSDT"]
    assert [trade.decision_index for trade in btc] == [4, 8, 12, 16]
    assert [trade.decision_index for trade in eth] == [4, 8, 12, 16]
    assert all(trade.side == 1 for trade in btc)
    assert all(trade.side == -1 for trade in eth)
    assert all(trade.worst_seed_expected_net_bps == 2.0 for trade in btc)
    assert all(trade.worst_seed_expected_net_bps == 1.0 for trade in eth)
    assert not any(trade.symbol == "SOLUSDT" for trade in trades)
    for symbol in SYMBOLS:
        symbol_trades = [trade for trade in trades if trade.symbol == symbol]
        assert all(
            right.decision_time_ms >= left.exit_time_ms
            for left, right in zip(symbol_trades, symbol_trades[1:])
        )


def test_base_and_stress_replay_use_the_identical_owned_ledger() -> None:
    dataset = _dataset()
    trades = select_fixed_policy_trades(dataset, _bundle(dataset))
    base = replay_fixed_trades(
        dataset,
        trades,
        candidate_id="direct_action_mean_tcn",
        scenario="base",
        execution_charge_bps=12.0,
    )
    stress = replay_fixed_trades(
        dataset,
        trades,
        candidate_id="direct_action_mean_tcn",
        scenario="stress",
        execution_charge_bps=16.0,
    )

    assert [trade.trade_id for trade in base.trades] == [
        trade.trade_id for trade in stress.trades
    ]
    assert base.metrics["trades"] == stress.metrics["trades"] == len(trades)
    assert all(
        stress_outcome["realized_net_bps"] == base_outcome["realized_net_bps"] - 4.0
        for base_outcome, stress_outcome in zip(
            base.trade_outcomes, stress.trade_outcomes, strict=True
        )
    )


def test_replay_rejects_candidate_ownership_tampering() -> None:
    dataset = _dataset()
    trades = select_fixed_policy_trades(dataset, _bundle(dataset))
    tampered = (replace(trades[0], candidate_id="other_candidate"), *trades[1:])

    with pytest.raises(RuntimeError, match="ownership"):
        replay_fixed_trades(
            dataset,
            tampered,
            candidate_id="direct_action_mean_tcn",
            scenario="base",
            execution_charge_bps=12.0,
        )
