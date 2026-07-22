from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading.lightgbm_backend import lightgbm_backend_parameters
from simple_ai_trading.price_discovery_dataset import PriceDiscoverySymbolDataset
from simple_ai_trading.price_discovery_model import (
    ROUND72_SEED,
    _fit_fold_prediction,
    binary_log_loss,
    build_price_discovery_roles,
    fit_binary_temperature,
    fit_continuous_slope,
    load_price_discovery_folds,
)
from simple_ai_trading.price_discovery_spec import (
    HORIZONS_SECONDS,
    layer_feature_names,
)


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "docs/model-research/action-value/round-072-price-discovery-implementation.json"


def _month_ordinal(year: int, month: int) -> int:
    return year * 12 + month - 1


def _synthetic_dataset(rows_per_month: int = 64) -> PriceDiscoverySymbolDataset:
    from simple_ai_trading.price_discovery_dataset import _dataset_sha256

    rng = np.random.default_rng(20260722)
    anchors: list[np.ndarray] = []
    months: list[np.ndarray] = []
    year, month = 2020, 10
    for _ in range(66):
        start = int(datetime(year, month, 2, 0, 30, 29, tzinfo=UTC).timestamp() * 1_000)
        anchors.append(start + np.arange(rows_per_month, dtype=np.int64) * 30_000)
        months.append(
            np.full(rows_per_month, _month_ordinal(year, month), dtype=np.int32)
        )
        month += 1
        if month == 13:
            year += 1
            month = 1
    anchor = np.concatenate(anchors)
    month_values = np.concatenate(months)
    rows = len(anchor)
    names = layer_feature_names("cross_asset")
    features = rng.normal(size=(rows, len(names))).astype(np.float32)
    signal = rng.normal(size=rows)
    features[:, 0] = signal.astype(np.float32)
    primary = np.column_stack(
        [
            (0.8 + index * 0.2) * signal + rng.normal(scale=0.35, size=rows)
            for index in range(len(HORIZONS_SECONDS))
        ]
    ).astype(np.float64)
    stress = (primary + rng.normal(scale=0.20, size=primary.shape)).astype(np.float64)
    valid = np.ones(primary.shape, dtype=bool)
    provisional = PriceDiscoverySymbolDataset(
        symbol="BTCUSDT",
        feature_names=names,
        anchor_second_ms=anchor,
        available_time_ms=anchor + 1_000,
        month_ordinal=month_values,
        utc_day=(anchor // 86_400_000).astype(np.int32),
        features=features,
        primary_target_bps=primary,
        primary_valid=valid,
        stress_target_bps=stress,
        stress_valid=valid.copy(),
        source_day_count=66,
        candidate_anchors=rows,
        age_eligible_anchors=rows,
        finite_feature_anchors=rows,
        dataset_sha256="",
    )
    dataset = replace(provisional, dataset_sha256=_dataset_sha256(provisional))
    dataset.validate()
    return dataset


def test_round72_folds_and_purged_roles_match_the_frozen_calendar() -> None:
    implementation, folds = load_price_discovery_folds(IMPLEMENTATION)
    dataset = _synthetic_dataset()
    roles = build_price_discovery_roles(dataset, folds[0], horizon_seconds=300)

    assert implementation["implementation_sha256"] == (
        "d8679606e75ec7fa2bf00032b34489218085f7c7f5159419e192f3ee351dfad9"
    )
    assert len(folds) == 6
    assert folds[0].training_end_month == _month_ordinal(2022, 9)
    assert folds[-1].test_end_month == _month_ordinal(2026, 3)
    assert len(roles.training) == 24 * 64
    assert len(roles.tuning) == 6 * 64
    assert len(roles.test) == 6 * 64
    assert roles.training[-1] < roles.tuning[0] < roles.test[0]


def test_round72_calibrators_are_bounded_and_only_retain_improvement() -> None:
    labels = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    overconfident = np.asarray([0.01, 0.02, 0.80, 0.20, 0.98, 0.99])
    temperature, retained, before, after = fit_binary_temperature(
        overconfident, labels
    )
    assert retained is True
    assert 1.0 < temperature <= 4.0
    assert after < before
    assert binary_log_loss(labels, overconfident) == pytest.approx(before)

    unchanged = np.full(labels.shape, 0.5)
    assert fit_binary_temperature(unchanged, labels) == pytest.approx(
        (1.0, False, np.log(2.0), np.log(2.0))
    )

    raw = np.asarray([-4.0, -2.0, 2.0, 4.0])
    slope, slope_retained, raw_mse, calibrated_mse = fit_continuous_slope(
        raw, raw * 0.1
    )
    assert slope_retained is True
    assert slope == pytest.approx(0.1)
    assert calibrated_mse < raw_mse


def test_round72_real_lightgbm_fold_serializes_and_reloads_identically() -> None:
    dataset = _synthetic_dataset()
    _implementation, folds = load_price_discovery_folds(IMPLEMENTATION)
    parameters = json.loads(IMPLEMENTATION.read_text(encoding="utf-8"))[
        "model_contract"
    ]["parameters"]
    backend, kind, device = lightgbm_backend_parameters(
        "cpu", ROUND72_SEED, reproducible=True
    )

    for head in ("binary_direction", "continuous_return_bps"):
        prediction = _fit_fold_prediction(
            dataset,
            folds[0],
            horizon_seconds=30,
            feature_layer="perpetual_only",
            head=head,
            backend_requested="cpu",
            backend_parameters=backend,
            backend_kind=kind,
            backend_device=device,
            model_parameters=parameters,
        )

        prediction.validate()
        assert prediction.test_rows == 6 * 64
        assert prediction.stress_test_rows == 6 * 64
        assert prediction.model_bytes > 0
        assert prediction.reload_max_absolute_prediction_difference <= 1e-12
        assert prediction.profitability_claim is False
        assert prediction.trading_authority is False
