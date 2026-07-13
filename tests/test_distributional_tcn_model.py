from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
import torch

from simple_ai_trading.cross_asset_cost_data import SYMBOLS
from simple_ai_trading.distributional_tcn_model import (
    BASE_ONE_WAY_COST_BPS,
    HORIZONS,
    QUANTILES,
    STRESS_ONE_WAY_COST_BPS,
    DistributionalDataset,
    DistributionalTCN,
    ExplicitAdamW,
    TargetScaler,
    build_distributional_dataset,
    compounded_forward_returns,
    economic_gate,
    fit_feature_scaler,
    fit_target_scaler,
    pinball_loss,
    replay_planned_trades,
    role_mask,
    select_planned_trades,
)
from simple_ai_trading.stateful_turnover_model import StatefulHourlyDataset


def _timestamp_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000)


def _evaluation_dataset(hours: int = 120, hourly_bps: float = 20.0) -> DistributionalDataset:
    timestamps = _timestamp_ms("2024-10-01T00:00:00") + np.arange(
        hours, dtype=np.int64
    ) * 3_600_000
    hourly = np.full((hours, len(SYMBOLS)), hourly_bps, dtype=np.float32)
    forward = compounded_forward_returns(hourly)
    features = np.zeros((hours, len(SYMBOLS), 71), dtype=np.float32)
    return DistributionalDataset(
        feature_names=tuple(f"feature_{index}" for index in range(71)),
        timestamps_ms=timestamps,
        features=features,
        hourly_return_bps=hourly,
        forward_return_bps=forward,
        dataset_sha256="a" * 64,
    )


def test_compounded_forward_returns_are_exact_and_past_independent() -> None:
    hourly = np.asarray(
        [
            [100.0, 0.0, -50.0],
            [200.0, 100.0, 50.0],
            [-100.0, 200.0, 25.0],
            [50.0, -100.0, 10.0],
        ]
    )
    result = compounded_forward_returns(hourly, horizons=(1, 4))
    assert result.shape == (4, 3, 2)
    assert np.array_equal(result[:, :, 0], hourly.astype(np.float32))
    expected = 10_000.0 * (np.prod(1.0 + hourly[:, 0] / 10_000.0) - 1.0)
    assert result[0, 0, 1] == pytest.approx(expected, abs=1e-5)
    assert np.isnan(result[1:, :, 1]).all()


def test_compounded_forward_returns_reject_invalid_wealth_path() -> None:
    hourly = np.zeros((4, len(SYMBOLS)), dtype=np.float64)
    hourly[0, 0] = -10_000.0
    with pytest.raises(ValueError, match="cannot be compounded"):
        compounded_forward_returns(hourly, horizons=(1,))


def test_stateful_source_is_reshaped_without_symbol_or_time_drift() -> None:
    hours = 30
    times = _timestamp_ms("2024-10-01T00:00:00") + np.arange(
        hours, dtype=np.int64
    ) * 3_600_000
    features = np.arange(hours * len(SYMBOLS) * 71, dtype=np.float32).reshape(
        hours * len(SYMBOLS), 71
    )
    target = np.arange(hours * len(SYMBOLS), dtype=np.float32)
    source = StatefulHourlyDataset(
        feature_names=tuple(f"feature_{index}" for index in range(71)),
        baseline_features=features,
        augmented_features=np.column_stack(
            (features, np.zeros((features.shape[0], 6), dtype=np.float32))
        ),
        decision_time_ms=np.repeat(times, len(SYMBOLS)),
        symbol_index=np.tile(np.arange(len(SYMBOLS), dtype=np.int8), hours),
        signed_pre_transition_utility_bps=target,
        funding_cash_flow_bps=np.zeros(target.size, dtype=np.float32),
        source_evidence=None,  # type: ignore[arg-type]
        dataset_sha256="b" * 64,
    )
    dataset = build_distributional_dataset(source)
    assert dataset.features.shape == (hours, len(SYMBOLS), 71)
    assert np.array_equal(dataset.features.reshape(features.shape), features)
    assert np.array_equal(dataset.hourly_return_bps.ravel(), target)
    assert len(dataset.dataset_sha256) == 64


def test_scalers_use_selected_rows_and_preserve_horizon_axis() -> None:
    dataset = _evaluation_dataset()
    dataset.features[:60] = 2.0
    mask = np.zeros(dataset.timestamps, dtype=bool)
    mask[:60] = True
    feature_scaler = fit_feature_scaler(dataset, mask)
    assert np.allclose(feature_scaler.mean, 2.0)
    transformed = feature_scaler.transform(dataset.features)
    assert np.allclose(transformed[:60], 0.0)

    target_scaler = fit_target_scaler(dataset, mask)
    normalized = target_scaler.normalize(dataset.forward_return_bps[:20])
    predictions = np.repeat(normalized[..., None], len(QUANTILES), axis=-1)
    restored = target_scaler.denormalize(predictions)
    assert restored.shape == predictions.shape
    assert np.allclose(
        restored,
        np.repeat(dataset.forward_return_bps[:20, ..., None], len(QUANTILES), axis=-1),
        equal_nan=True,
    )


def test_tcn_is_causal_and_quantiles_cannot_cross() -> None:
    torch.manual_seed(44)
    model = DistributionalTCN(dropout=0.0).eval()
    left = torch.randn(2, 71, 180)
    right = left.clone()
    right[:, :, 121:] += 100.0
    with torch.no_grad():
        left_prediction = model(left)
        right_prediction = model(right)
    assert left_prediction.shape == (2, len(HORIZONS), len(QUANTILES), 180)
    assert torch.equal(left_prediction[..., :121], right_prediction[..., :121])
    assert bool(torch.all(torch.diff(left_prediction, dim=2) >= 0.0))


def test_pinball_loss_is_zero_for_exact_quantiles() -> None:
    targets = torch.zeros(3, len(HORIZONS), 20)
    predictions = torch.zeros(3, len(HORIZONS), len(QUANTILES), 20)
    assert pinball_loss(predictions, targets).item() == 0.0


def test_explicit_adamw_matches_torch_non_foreach_update() -> None:
    left = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    right = torch.nn.Parameter(left.detach().clone())
    explicit = ExplicitAdamW(
        (left,),
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    reference = torch.optim.AdamW(
        (right,),
        lr=1e-3,
        weight_decay=1e-4,
        foreach=False,
    )
    for gradient in (
        torch.tensor([0.25, -0.5]),
        torch.tensor([-0.75, 0.125]),
        torch.tensor([0.5, 0.5]),
    ):
        left.grad = gradient.clone()
        right.grad = gradient.clone()
        explicit.step()
        reference.step()
        explicit.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
    assert torch.allclose(left, right, rtol=1e-6, atol=1e-7)


def test_planned_ledger_is_nonoverlapping_and_stress_only_reprices() -> None:
    dataset = _evaluation_dataset()
    predictions = np.empty(
        (
            dataset.timestamps,
            len(SYMBOLS),
            len(HORIZONS),
            len(QUANTILES),
        ),
        dtype=np.float32,
    )
    predictions[..., 0] = 15.0
    predictions[..., 1] = 20.0
    predictions[..., 2] = 25.0
    predictions[..., 3] = 30.0
    predictions[..., 4] = 40.0
    trades = select_planned_trades(dataset, predictions)
    assert trades
    assert all(trade.side == 1 and trade.horizon_hours == 1 for trade in trades)
    by_symbol: dict[str, list[int]] = {symbol: [] for symbol in SYMBOLS}
    for trade in trades:
        by_symbol[trade.symbol].append(trade.decision_index)
    assert all(len(values) == len(set(values)) for values in by_symbol.values())

    base = replay_planned_trades(
        dataset,
        trades,
        scenario="base",
        one_way_cost_bps=BASE_ONE_WAY_COST_BPS,
    )
    stress = replay_planned_trades(
        dataset,
        trades,
        scenario="stress",
        one_way_cost_bps=STRESS_ONE_WAY_COST_BPS,
    )
    assert [trade.trade_id for trade in base.trades] == [
        trade.trade_id for trade in stress.trades
    ]
    expected_delta = (
        2.0
        * (STRESS_ONE_WAY_COST_BPS - BASE_ONE_WAY_COST_BPS)
        * len(trades)
        * (1.0 / len(SYMBOLS))
    )
    assert np.sum(base.portfolio_return_bps - stress.portfolio_return_bps) == pytest.approx(
        expected_delta
    )
    assert stress.metrics["total_net_return_fraction"] < base.metrics[
        "total_net_return_fraction"
    ]


def test_planner_abstains_when_lower_quartile_cannot_cover_cost() -> None:
    dataset = _evaluation_dataset()
    predictions = np.zeros(
        (
            dataset.timestamps,
            len(SYMBOLS),
            len(HORIZONS),
            len(QUANTILES),
        ),
        dtype=np.float32,
    )
    predictions[..., 0] = -5.0
    predictions[..., 1] = 5.0
    predictions[..., 2] = 6.0
    predictions[..., 3] = 7.0
    predictions[..., 4] = 8.0
    assert select_planned_trades(dataset, predictions) == ()


def test_role_mask_purges_incomplete_twenty_four_hour_targets() -> None:
    dataset = _evaluation_dataset(hours=72)
    mask = role_mask(dataset, "evaluation")
    assert np.count_nonzero(mask) == 49
    assert np.flatnonzero(mask)[-1] == 48


def test_economic_gate_remains_fail_closed_after_forecast_failure() -> None:
    dataset = _evaluation_dataset()
    stress = replay_planned_trades(
        dataset,
        (),
        scenario="stress",
        one_way_cost_bps=STRESS_ONE_WAY_COST_BPS,
    )
    gate = economic_gate(forecast_gate_passed=False, stress=stress)
    assert gate["passed"] is False
    assert gate["promotion_permitted"] is False
    assert "forecast_gate_failed" in gate["reasons"]


def test_target_scaler_rejects_wrong_axes() -> None:
    scaler = TargetScaler(
        center_bps=np.zeros(len(HORIZONS)),
        scale_bps=np.ones(len(HORIZONS)),
    )
    with pytest.raises(ValueError, match="normalization dimensions"):
        scaler.normalize(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="denormalization dimensions"):
        scaler.denormalize(np.zeros((10, len(QUANTILES), len(HORIZONS))))
