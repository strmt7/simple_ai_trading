from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
import torch

from simple_ai_trading.cost_aware_utility_tcn_model import (
    CANDIDATES,
    SEEDS,
    CostAwareUtilityTCN,
    UtilityForecastBundle,
    UtilityScaler,
    action_labels,
    additive_forward_utility_bps,
    binary_logistic_loss,
    fit_utility_scaler,
    multitask_objective,
    pairwise_utility_rank_loss,
    rank_ablation_gate,
    select_utility_trades,
)
from simple_ai_trading.cross_asset_cost_data import SYMBOLS
from simple_ai_trading.distributional_tcn_model import (
    HORIZONS,
    QUANTILES,
    DistributionalDataset,
    FeatureScaler,
    TargetScaler,
    compounded_forward_returns,
)


def _timestamp_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000)


def _dataset(hours: int = 120, hourly_bps: float = 20.0) -> DistributionalDataset:
    hourly = np.full((hours, len(SYMBOLS)), hourly_bps, dtype=np.float32)
    return DistributionalDataset(
        feature_names=tuple(f"feature_{index}" for index in range(71)),
        timestamps_ms=(
            _timestamp_ms("2024-10-01T00:00:00")
            + np.arange(hours, dtype=np.int64) * 3_600_000
        ),
        features=np.zeros((hours, len(SYMBOLS), 71), dtype=np.float32),
        hourly_return_bps=hourly,
        forward_return_bps=compounded_forward_returns(hourly),
        dataset_sha256="a" * 64,
    )


def _bundle(dataset: DistributionalDataset) -> UtilityForecastBundle:
    shape = (len(SEEDS), dataset.timestamps, len(SYMBOLS), len(HORIZONS))
    utility = np.full(shape, 20.0, dtype=np.float32)
    quantiles = np.zeros((*shape, len(QUANTILES)), dtype=np.float32)
    logits = np.zeros((*shape, 2), dtype=np.float32)
    probabilities = np.full((*shape, 2), 0.4, dtype=np.float32)
    probabilities[..., 1] = 0.6
    return UtilityForecastBundle(
        candidate_id=CANDIDATES[0],
        seed_predictions_bps=quantiles,
        ensemble_predictions_bps=np.median(quantiles, axis=0),
        seed_utility_bps=utility,
        ensemble_utility_bps=np.median(utility, axis=0),
        seed_action_logits=logits,
        seed_action_probabilities=probabilities,
        ensemble_action_probabilities=np.median(probabilities, axis=0),
        artifacts=(),
        feature_scaler=FeatureScaler(mean=np.zeros(71), standard_deviation=np.ones(71)),
        target_scaler=TargetScaler(
            center_bps=np.zeros(len(HORIZONS)),
            scale_bps=np.ones(len(HORIZONS)),
        ),
        utility_scaler=UtilityScaler(
            mean_bps=np.zeros(len(HORIZONS)),
            scale_bps=np.ones(len(HORIZONS)),
        ),
        backend_kind="cpu",
        backend_device="cpu",
        training_history=(),
    )


def test_additive_forward_utility_matches_every_exact_window() -> None:
    hourly = np.arange(1, 121, dtype=np.float32).reshape(40, len(SYMBOLS))
    forward = additive_forward_utility_bps(hourly)
    for horizon_index, horizon in enumerate(HORIZONS):
        for start in range(hourly.shape[0] - horizon + 1):
            np.testing.assert_allclose(
                forward[start, :, horizon_index],
                np.sum(hourly[start : start + horizon], axis=0),
                rtol=0.0,
                atol=1e-5,
            )
        assert np.isnan(
            forward[hourly.shape[0] - horizon + 1 :, :, horizon_index]
        ).all()


def test_action_labels_encode_the_exact_twelve_bps_no_trade_region() -> None:
    utility = np.asarray([-13.0, -12.0, 0.0, 12.0, 13.0], dtype=np.float32)
    labels = action_labels(utility)
    np.testing.assert_array_equal(labels[:, 0], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(labels[:, 1], [0, 0, 0, 0, 1])


def test_utility_scaler_uses_training_mean_and_population_standard_deviation() -> None:
    values = np.arange(96, dtype=np.float32).reshape(8, 3, 4)
    mask = np.asarray([True, True, True, True, False, False, False, False])
    scaler = fit_utility_scaler(values, mask)
    selected = values[mask].reshape(-1, len(HORIZONS))
    np.testing.assert_allclose(scaler.mean_bps, np.mean(selected, axis=0))
    np.testing.assert_allclose(scaler.scale_bps, np.std(selected, axis=0))
    normalized = scaler.normalize(values[mask])
    np.testing.assert_allclose(np.mean(normalized, axis=(0, 1)), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(normalized, axis=(0, 1)), 1.0, atol=1e-6)
    np.testing.assert_allclose(scaler.denormalize(normalized), values[mask], atol=1e-5)


def test_cost_aware_tcn_is_causal_and_quantiles_are_monotone() -> None:
    torch.manual_seed(47)
    model = CostAwareUtilityTCN(dropout=0.0).eval()
    left = torch.randn(2, 71, 180)
    right = left.clone()
    right[:, :, 121:] += 100.0
    with torch.no_grad():
        left_outputs = model(left)
        right_outputs = model(right)
    assert left_outputs[0].shape == (2, len(HORIZONS), len(QUANTILES), 180)
    assert left_outputs[1].shape == (2, len(HORIZONS), 180)
    assert left_outputs[2].shape == (2, len(HORIZONS), 2, 180)
    for left_value, right_value in zip(left_outputs, right_outputs, strict=True):
        torch.testing.assert_close(left_value[..., :121], right_value[..., :121])
    assert torch.all(torch.diff(left_outputs[0], dim=2) >= 0.0)


def test_pairwise_rank_loss_rewards_correct_utility_ordering() -> None:
    target = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64)
    correct = target.clone().requires_grad_(True)
    reversed_score = (-target).clone().requires_grad_(True)
    correct_loss = pairwise_utility_rank_loss(correct, target)
    reversed_loss = pairwise_utility_rank_loss(reversed_score, target)
    assert correct_loss.item() < reversed_loss.item()
    correct_loss.backward()
    assert correct.grad is not None
    assert torch.isfinite(correct.grad).all()


def test_directml_native_logistic_loss_matches_pytorch_reference() -> None:
    logits = torch.linspace(-20.0, 20.0, 81)
    labels = (torch.arange(81) % 3 == 0).float()
    observed = binary_logistic_loss(logits, labels)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    assert observed.item() == pytest.approx(expected.item(), rel=1e-7)


def test_multitask_objective_updates_every_output_head() -> None:
    models = []
    outputs = []
    values = torch.randn(4, 71, 160)
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = CostAwareUtilityTCN(dropout=0.0)
        models.append(model)
        outputs.append(model(values))
    quantile_target = torch.randn(4, len(HORIZONS), 160)
    utility_target = torch.randn(4, len(HORIZONS), 160)
    labels = torch.randint(0, 2, (4, len(HORIZONS), 2, 160)).float()
    objective, components = multitask_objective(
        outputs,
        quantile_target,
        utility_target,
        labels,
        rank_weight=0.05,
    )
    objective.backward()
    assert objective.item() > 0.0
    assert components["ranking"].shape == (len(SEEDS),)
    for model in models:
        assert model.quantile_head.weight.grad is not None
        assert model.utility_head.weight.grad is not None
        assert model.action_head.weight.grad is not None
        assert torch.count_nonzero(model.quantile_head.weight.grad).item() > 0
        assert torch.count_nonzero(model.utility_head.weight.grad).item() > 0
        assert torch.count_nonzero(model.action_head.weight.grad).item() > 0


def test_fixed_policy_uses_probability_and_expected_net_consensus() -> None:
    dataset = _dataset()
    utility = additive_forward_utility_bps(dataset.hourly_return_bps)
    trades = select_utility_trades(dataset, utility, _bundle(dataset))
    assert trades
    assert all(trade.side == 1 for trade in trades)
    assert all(trade.horizon_hours == 1 for trade in trades)
    assert all(trade.worst_seed_probability == pytest.approx(0.6) for trade in trades)
    assert all(
        trade.worst_seed_expected_net_bps == pytest.approx(8.0) for trade in trades
    )
    assert all(
        utility[trade.decision_index, trade.symbol_index, 0] - 12.0
        == pytest.approx(8.0)
        for trade in trades
    )


def test_rank_ablation_gate_requires_ordering_gain_without_log_loss_damage() -> None:
    control = {
        "utility_horizons": [{"spearman": 0.01}] * 4,
        "action_side_horizons": [{"log_loss": 0.50}] * 8,
    }
    passed = rank_ablation_gate(
        control,
        {
            "utility_horizons": [{"spearman": 0.02}] * 4,
            "action_side_horizons": [{"log_loss": 0.502}] * 8,
        },
    )
    assert passed["passed"] is True
    failed = rank_ablation_gate(
        control,
        {
            "utility_horizons": [{"spearman": 0.011}] * 4,
            "action_side_horizons": [{"log_loss": 0.52}] * 8,
        },
    )
    assert failed["passed"] is False
    assert len(failed["reasons"]) == 2
