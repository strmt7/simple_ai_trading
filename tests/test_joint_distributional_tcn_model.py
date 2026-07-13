from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
import torch

import simple_ai_trading.joint_distributional_tcn_model as joint_module
from simple_ai_trading.cross_asset_cost_data import SYMBOLS
from simple_ai_trading.distributional_tcn_model import (
    BASE_ONE_WAY_COST_BPS,
    HORIZONS,
    QUANTILES,
    STRESS_ONE_WAY_COST_BPS,
    DistributionalDataset,
    ExplicitAdamW,
    FeatureScaler,
    TargetScaler,
    compounded_forward_returns,
)
from simple_ai_trading.joint_distributional_tcn_model import (
    CANDIDATES,
    SEEDS,
    JointDistributionalTCN,
    JointForecastBundle,
    JointTCNArtifact,
    joint_economic_gate,
    joint_forecast_diagnostics,
    joint_pinball_loss,
    optimizer_ablation_gate,
    replay_consensus_trades,
    sam_training_step,
    select_consensus_trades,
)


def _timestamp_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000)


def _evaluation_dataset(
    hours: int = 120,
    hourly_bps: float = 20.0,
) -> DistributionalDataset:
    timestamps = (
        _timestamp_ms("2024-10-01T00:00:00")
        + np.arange(hours, dtype=np.int64) * 3_600_000
    )
    hourly = np.full((hours, len(SYMBOLS)), hourly_bps, dtype=np.float32)
    return DistributionalDataset(
        feature_names=tuple(f"feature_{index}" for index in range(71)),
        timestamps_ms=timestamps,
        features=np.zeros((hours, len(SYMBOLS), 71), dtype=np.float32),
        hourly_return_bps=hourly,
        forward_return_bps=compounded_forward_returns(hourly),
        dataset_sha256="c" * 64,
    )


def _bundle(
    dataset: DistributionalDataset,
    *,
    candidate_id: str = CANDIDATES[0],
    median_bps: float = 20.0,
) -> JointForecastBundle:
    predictions = np.empty(
        (
            len(SEEDS),
            dataset.timestamps,
            len(SYMBOLS),
            len(HORIZONS),
            len(QUANTILES),
        ),
        dtype=np.float32,
    )
    for quantile_index, offset in enumerate((-20.0, -10.0, 0.0, 10.0, 20.0)):
        predictions[..., quantile_index] = median_bps + offset
    return JointForecastBundle(
        candidate_id=candidate_id,
        seed_predictions_bps=predictions,
        ensemble_predictions_bps=np.median(predictions, axis=0),
        artifacts=(),
        feature_scaler=FeatureScaler(mean=np.zeros(71), standard_deviation=np.ones(71)),
        target_scaler=TargetScaler(
            center_bps=np.zeros(len(HORIZONS)),
            scale_bps=np.ones(len(HORIZONS)),
        ),
        backend_kind="cpu",
        backend_device="cpu",
    )


def _artifact(candidate_id: str, seed: int, validation: float) -> JointTCNArtifact:
    return JointTCNArtifact(
        candidate_id=candidate_id,
        seed=seed,
        epochs=4,
        best_epoch=3,
        best_early_stop_pinball=validation,
        parameter_count=1,
        backend_kind="cpu",
        backend_device="cpu",
        path="model.pt",
        bytes=1,
        sha256="d" * 64,
        reload_max_abs_prediction_error=0.0,
        warning_count=0,
    )


def test_joint_tcn_is_causal_and_quantiles_cannot_cross() -> None:
    torch.manual_seed(45)
    model = JointDistributionalTCN(dropout=0.0).eval()
    left = torch.randn(2, 213, 180)
    right = left.clone()
    right[:, :, 121:] += 100.0
    with torch.no_grad():
        left_prediction = model(left)
        right_prediction = model(right)
    assert left_prediction.shape == (
        2,
        len(SYMBOLS),
        len(HORIZONS),
        len(QUANTILES),
        180,
    )
    assert torch.equal(left_prediction[..., :121], right_prediction[..., :121])
    assert bool(torch.all(torch.diff(left_prediction, dim=3) >= 0.0))


def test_joint_pinball_loss_is_zero_for_exact_quantiles() -> None:
    targets = torch.zeros(3, len(SYMBOLS), len(HORIZONS), 20)
    predictions = torch.zeros(3, len(SYMBOLS), len(HORIZONS), len(QUANTILES), 20)
    assert joint_pinball_loss(predictions, targets).item() == 0.0


def test_sam_step_restores_perturbation_before_adamw_update() -> None:
    torch.manual_seed(4501)
    model = JointDistributionalTCN(dropout=0.0)
    optimizer = ExplicitAdamW(
        tuple(model.parameters()), learning_rate=1e-3, weight_decay=1e-4
    )
    before = model.projection.weight.detach().clone()
    values = torch.randn(2, 213, 140)
    targets = torch.randn(2, len(SYMBOLS), len(HORIZONS), 100)
    first_loss, second_loss = sam_training_step(model, optimizer, values, targets)
    assert np.isfinite(first_loss)
    assert np.isfinite(second_loss)
    assert second_loss >= 0.0
    assert not torch.equal(before, model.projection.weight)


def test_consensus_policy_requires_every_seed_to_cover_cost() -> None:
    dataset = _evaluation_dataset()
    bundle = _bundle(dataset)
    bundle.seed_predictions_bps[0, ..., 2] = 11.0
    assert select_consensus_trades(dataset, bundle) == ()


def test_consensus_ledger_is_fixed_under_stress() -> None:
    dataset = _evaluation_dataset()
    bundle = _bundle(dataset)
    trades = select_consensus_trades(dataset, bundle)
    assert trades
    assert all(trade.side == 1 and trade.horizon_hours == 1 for trade in trades)
    base = replay_consensus_trades(
        dataset,
        trades,
        candidate_id=bundle.candidate_id,
        scenario="base",
        one_way_cost_bps=BASE_ONE_WAY_COST_BPS,
    )
    stress = replay_consensus_trades(
        dataset,
        trades,
        candidate_id=bundle.candidate_id,
        scenario="stress",
        one_way_cost_bps=STRESS_ONE_WAY_COST_BPS,
    )
    assert [trade.trade_id for trade in base.trades] == [
        trade.trade_id for trade in stress.trades
    ]
    assert np.array_equal(base.positions, stress.positions)
    expected_delta = (
        2.0
        * (STRESS_ONE_WAY_COST_BPS - BASE_ONE_WAY_COST_BPS)
        * len(trades)
        / len(SYMBOLS)
    )
    assert np.sum(
        base.portfolio_return_bps - stress.portfolio_return_bps
    ) == pytest.approx(expected_delta)
    assert stress.metrics["bootstrap_mean_hourly_portfolio_bps"][
        "lower_quantile"
    ] == pytest.approx(0.0125)


def test_consensus_replay_accepts_a_bound_seed_for_future_candidates() -> None:
    dataset = _evaluation_dataset()
    bundle = _bundle(dataset, candidate_id="wavebound_ema")
    trades = select_consensus_trades(dataset, bundle)
    replay = replay_consensus_trades(
        dataset,
        trades,
        candidate_id=bundle.candidate_id,
        scenario="base",
        one_way_cost_bps=BASE_ONE_WAY_COST_BPS,
        bootstrap_seed=4601,
    )
    assert replay.metrics["candidate_id"] == "wavebound_ema"
    assert replay.metrics["trades"] == len(trades)


def test_forecast_diagnostics_use_bundle_artifact_seed_ids(monkeypatch) -> None:
    dataset = _evaluation_dataset(hours=120)
    dataset.forward_return_bps[:] += np.arange(120, dtype=np.float32)[:, None, None]
    bundle = _bundle(dataset)
    bound_seeds = (4601, 4602, 4603)
    object.__setattr__(
        bundle,
        "artifacts",
        tuple(_artifact(bundle.candidate_id, seed, 0.37) for seed in bound_seeds),
    )

    def selected_role(_dataset, role_name: str) -> np.ndarray:
        indexes = np.arange(dataset.timestamps)
        return indexes < 60 if role_name == "training" else indexes >= 60

    monkeypatch.setattr(joint_module, "role_mask", selected_role)
    _, stability, _ = joint_forecast_diagnostics(dataset, bundle)
    assert {(row["left_seed"], row["right_seed"]) for row in stability} == {
        (4601, 4602),
        (4601, 4603),
        (4602, 4603),
    }


def test_economic_gate_requires_forecast_and_capacity() -> None:
    dataset = _evaluation_dataset()
    stress = replay_consensus_trades(
        dataset,
        (),
        candidate_id=CANDIDATES[0],
        scenario="stress",
        one_way_cost_bps=STRESS_ONE_WAY_COST_BPS,
    )
    gate = joint_economic_gate(forecast_gate_passed=False, stress=stress)
    assert gate["passed"] is False
    assert "forecast_gate_failed" in gate["reasons"]
    assert "fewer_than_one_hundred_eighty_closed_trades" in gate["reasons"]


def test_optimizer_ablation_requires_stability_and_validation_non_degradation() -> None:
    dataset = _evaluation_dataset()
    adam = _bundle(dataset, candidate_id="joint_adamw")
    sam = _bundle(dataset, candidate_id="joint_sam")
    object.__setattr__(
        adam,
        "artifacts",
        tuple(_artifact("joint_adamw", seed, 0.370) for seed in SEEDS),
    )
    object.__setattr__(
        sam,
        "artifacts",
        tuple(_artifact("joint_sam", seed, 0.368) for seed in SEEDS),
    )
    adam_diagnostics = {
        "gate": {
            "passed": True,
            "minimum_pairwise_seed_median_prediction_spearman": 0.51,
        }
    }
    sam_diagnostics = {
        "gate": {
            "passed": True,
            "minimum_pairwise_seed_median_prediction_spearman": 0.55,
        }
    }
    gate = optimizer_ablation_gate(adam, adam_diagnostics, sam, sam_diagnostics)
    assert gate["passed"] is True
    assert gate["sam_seed_stability_delta"] == pytest.approx(0.04)
