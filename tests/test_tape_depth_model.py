from __future__ import annotations

import json

import numpy as np
import pytest

from simple_ai_trading.tape_depth_features import (
    TAPE_DEPTH_FEATURE_NAMES,
    TAPE_DEPTH_FEATURE_VERSION,
    TAPE_DEPTH_TARGET_MODE,
    TapeDepthForecastDataset,
)
from simple_ai_trading.tape_depth_model import (
    load_tape_depth_model_artifact,
    save_tape_depth_model_artifact,
    train_tape_depth_forecaster,
)


def _predictive_dataset(rows: int = 12_000) -> TapeDepthForecastDataset:
    rng = np.random.default_rng(20260710)
    features = rng.normal(0.0, 1.0, size=(rows, len(TAPE_DEPTH_FEATURE_NAMES))).astype(
        np.float32
    )
    signal = features[:, 0] + 0.5 * features[:, 1]
    targets = (3.0 * signal + rng.normal(0.0, 0.35, size=rows)).astype(np.float64)
    depth_available = TAPE_DEPTH_FEATURE_NAMES.index("depth_available")
    depth_age = TAPE_DEPTH_FEATURE_NAMES.index("depth_age_seconds")
    features[:, depth_available] = 1.0
    features[:, depth_age] = np.abs(features[:, depth_age]) * 30.0
    base_ms = 1_700_000_000_000
    times = base_ms + np.arange(rows, dtype=np.int64) * 5_000
    prices = 100.0 + np.arange(rows, dtype=np.float64) * 0.001
    return TapeDepthForecastDataset(
        symbol="BTCUSDT",
        feature_version=TAPE_DEPTH_FEATURE_VERSION,
        feature_names=TAPE_DEPTH_FEATURE_NAMES,
        target_mode=TAPE_DEPTH_TARGET_MODE,
        horizon_seconds=60,
        total_latency_ms=750,
        decision_cadence_seconds=5,
        maximum_depth_age_ms=60_000,
        decision_time_ms=times,
        target_entry_time_ms=times + 1_000,
        target_exit_time_ms=times + 61_000,
        target_entry_price=prices,
        target_exit_price=prices * (1.0 + targets / 10_000.0),
        gross_return_bps=targets,
        features=features,
        source_evidence={
            "verified": True,
            "schema_version": "binance-usdm-tick-v5",
            "manifest_fingerprint": "a" * 64,
        },
    )


def test_tape_depth_forecaster_is_predictive_but_never_executable(tmp_path) -> None:
    artifact = train_tape_depth_forecaster(
        _predictive_dataset(),
        risk_level="conservative",
        compute_backend="cpu",
        minimum_segment_rows=500,
    )

    assert artifact.status == "research_candidate"
    assert artifact.rejection_reasons == ()
    assert artifact.trading_authority is False
    assert artifact.execution_claim is False
    assert artifact.split.purge_ms == 61_000
    assert artifact.evaluation_metrics.direction_auc > 0.90
    assert (
        artifact.evaluation_metrics.mean_absolute_error_bps
        < artifact.evaluation_metrics.zero_baseline_mae_bps
    )
    assert artifact.evaluation_metrics.top_decile_mean_signed_gross_bps > 0.0
    assert set(artifact.model_strings) == {"direction", "mean", "lower", "upper"}

    path = tmp_path / "tape-depth.json"
    save_tape_depth_model_artifact(artifact, path)
    loaded = load_tape_depth_model_artifact(path)
    assert loaded == artifact


def test_tape_depth_loader_rejects_forged_trading_authority(tmp_path) -> None:
    artifact = train_tape_depth_forecaster(
        _predictive_dataset(6_000),
        compute_backend="cpu",
        minimum_segment_rows=256,
    )
    payload = artifact.asdict()
    payload["trading_authority"] = True
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot authorize trading"):
        load_tape_depth_model_artifact(path)
