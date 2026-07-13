from __future__ import annotations

import numpy as np
import pytest
import torch

from simple_ai_trading.distributional_tcn_model import (
    HORIZONS,
    QUANTILES,
    DistributionalTCN,
    FeatureScaler,
    TargetScaler,
)
from simple_ai_trading.stable_distributional_tcn_model import (
    SEEDS,
    StabilityForecastBundle,
    StabilityTCNArtifact,
    pinball_components,
    stability_mechanism_gate,
    standardized_median_consistency,
    update_ema_target,
    wavebound_pinball_loss,
)


def _artifact(validation: float) -> StabilityTCNArtifact:
    return StabilityTCNArtifact(
        candidate_id="wavebound_ema",
        seed=SEEDS[0],
        epochs=10,
        best_epoch=4,
        best_early_stop_pinball=validation,
        optimizer_updates=380,
        evaluation_model="ema_target",
        parameter_count=1,
        backend_kind="cpu",
        backend_device="cpu",
        path="model.pt",
        bytes=1,
        sha256="a" * 64,
        reload_max_abs_prediction_error=0.0,
        warning_count=0,
    )


def _bundle(validation: float) -> StabilityForecastBundle:
    return StabilityForecastBundle(
        candidate_id="wavebound_ema",
        seed_predictions_bps=np.zeros((3, 1, 3, 4, 5), dtype=np.float32),
        ensemble_predictions_bps=np.zeros((1, 3, 4, 5), dtype=np.float32),
        artifacts=tuple(_artifact(validation) for _ in SEEDS),
        feature_scaler=FeatureScaler(mean=np.zeros(71), standard_deviation=np.ones(71)),
        target_scaler=TargetScaler(
            center_bps=np.zeros(len(HORIZONS)),
            scale_bps=np.ones(len(HORIZONS)),
        ),
        backend_kind="cpu",
        backend_device="cpu",
        training_history=(),
    )


def test_pinball_components_are_zero_for_exact_forecasts() -> None:
    targets = torch.zeros(3, len(HORIZONS), 12)
    predictions = torch.zeros(3, len(HORIZONS), len(QUANTILES), 12)
    components = pinball_components(predictions, targets)
    assert components.shape == (len(HORIZONS), len(QUANTILES))
    assert torch.count_nonzero(components).item() == 0


def test_wavebound_uses_the_frozen_dynamic_lower_bound_equation() -> None:
    generator = torch.Generator().manual_seed(46)
    targets = torch.randn(2, len(HORIZONS), 9, generator=generator)
    source = torch.randn(2, len(HORIZONS), len(QUANTILES), 9, generator=generator)
    target = torch.randn(2, len(HORIZONS), len(QUANTILES), 9, generator=generator)
    objective, source_mean, target_mean = wavebound_pinball_loss(
        source, target, targets, epsilon=0.001
    )
    source_components = pinball_components(source, targets)
    target_components = pinball_components(target, targets)
    expected = (
        torch.abs(source_components - target_components + 0.001)
        + target_components
        - 0.001
    ).mean()
    assert objective.item() == pytest.approx(expected.item())
    assert source_mean.item() == pytest.approx(source_components.mean().item())
    assert target_mean.item() == pytest.approx(target_components.mean().item())


def test_ema_target_update_is_exact() -> None:
    torch.manual_seed(4601)
    source = DistributionalTCN()
    target = DistributionalTCN()
    with torch.no_grad():
        for value in source.parameters():
            value.fill_(4.0)
        for value in target.parameters():
            value.fill_(2.0)
    update_ema_target(target, source, decay=0.75)
    for value in target.parameters():
        assert torch.all(value == 2.5)


def test_standardized_consistency_is_zero_only_for_matching_peers() -> None:
    generator = torch.Generator().manual_seed(4602)
    prediction = torch.randn(5, len(HORIZONS), len(QUANTILES), 11, generator=generator)
    zero, matrix = standardized_median_consistency(
        (prediction, prediction.clone(), prediction.clone())
    )
    assert matrix.shape == (3, len(HORIZONS))
    assert zero.item() == pytest.approx(0.0, abs=1e-8)
    opposite = prediction.clone()
    opposite[:, :, 2, :] *= -1.0
    nonzero, _ = standardized_median_consistency(
        (prediction, prediction.clone(), opposite)
    )
    assert nonzero.item() > 1.0


def test_mechanism_gate_enforces_stability_and_validation_limits() -> None:
    passed = stability_mechanism_gate(
        _bundle(0.37),
        {"gate": {"minimum_pairwise_seed_median_prediction_spearman": 0.51}},
    )
    assert passed["passed"] is True
    failed = stability_mechanism_gate(
        _bundle(0.38),
        {"gate": {"minimum_pairwise_seed_median_prediction_spearman": 0.49}},
    )
    assert failed["passed"] is False
    assert "minimum_seed_stability_below_0_5" in failed["reasons"]
    assert (
        "median_early_stop_pinball_degraded_more_than_two_percent" in failed["reasons"]
    )
