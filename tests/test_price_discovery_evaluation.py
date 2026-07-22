from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading.price_discovery_evaluation import (
    binary_predictive_metrics,
    continuous_predictive_metrics,
    day_block_bootstrap_lower,
    evaluate_price_discovery_primary,
)
from simple_ai_trading.price_discovery_model import (
    PRICE_DISCOVERY_MODEL_RUN_SCHEMA,
    PRICE_DISCOVERY_HEADS,
    PRIMARY_FEATURE_LAYERS,
    PriceDiscoveryFoldPrediction,
    PriceDiscoveryPredictionRun,
    _prediction_sha256,
    _run_sha256,
)
from simple_ai_trading.price_discovery_spec import (
    HORIZONS_SECONDS,
    layer_feature_names,
)
from simple_ai_trading.spot_perpetual_flow import FLOW_SYMBOLS


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "docs/model-research/action-value/round-072-price-discovery-implementation.json"


def _certificate() -> dict[str, object]:
    implementation = json.loads(IMPLEMENTATION.read_text(encoding="utf-8"))
    return {
        "schema_version": "spot-perpetual-corpus-certificate-v1",
        "research_round": 72,
        "inventory_sha256": implementation["inventory_sha256"],
        "status": "complete",
        "day_count": 69,
        "source_count": 414,
        "symbol_count": 3,
        "flow_rows": 17_884_800,
        "compressed_bytes": 5_964_131_852,
        "uncompressed_bytes": 10_000_000_000,
        "first_period": "2020-10-19",
        "last_period": "2026-06-01",
        "manifest_fingerprint": "b" * 64,
        "source_fingerprint": "c" * 64,
    }


def _readonly(value: np.ndarray, dtype) -> np.ndarray:
    output = np.asarray(value, dtype=dtype)
    output.setflags(write=False)
    return output


def _block(
    *,
    symbol: str,
    horizon: int,
    layer: str,
    head: str,
    fold: int,
    paired_improves: bool,
) -> PriceDiscoveryFoldPrediction:
    rows_per_day = 20
    day_values = np.asarray([20_000 + fold * 2, 20_001 + fold * 2], dtype=np.int64)
    days = np.repeat(day_values, rows_per_day)
    row_within_day = np.tile(np.arange(rows_per_day, dtype=np.int64), 2)
    anchors = days * 86_400_000 + 29_000 + row_within_day * 30_000
    binary_target = np.tile(np.asarray([0.0, 1.0]), len(anchors) // 2)
    continuous_target = np.tile(
        np.asarray([-2.0, -1.0, 1.0, 2.0]), len(anchors) // 4
    )
    if head == "binary_direction":
        target = binary_target
        perpetual_prediction = np.where(target == 1.0, 0.65, 0.35)
        paired_prediction = np.where(target == 1.0, 0.90, 0.10)
    else:
        target = continuous_target
        perpetual_prediction = target * 0.70
        paired_prediction = target * 0.95
    prediction = (
        paired_prediction
        if layer == "spot_perpetual" and paired_improves
        else perpetual_prediction
    )
    identity = f"{symbol}:{horizon}:{layer}:{head}:{fold}:{paired_improves}"
    provisional = PriceDiscoveryFoldPrediction(
        symbol=symbol,
        horizon_seconds=horizon,
        feature_layer=layer,
        head=head,
        fold=fold,
        feature_count=len(layer_feature_names(layer)),
        training_rows=1_000,
        tuning_rows=200,
        test_rows=len(target),
        stress_test_rows=len(target),
        training_positive_rows=500,
        training_negative_rows=500,
        training_prevalence=0.5,
        training_mean_target_bps=0.0,
        best_iteration=10,
        calibration_value=1.0,
        calibration_retained=False,
        tuning_loss_before_calibration=0.5,
        tuning_loss_after_calibration=0.5,
        model_bytes=100,
        model_sha256=hashlib.sha256(identity.encode("ascii")).hexdigest(),
        reload_max_absolute_prediction_difference=0.0,
        backend_requested="cpu",
        backend_kind="cpu",
        backend_device="cpu",
        lightgbm_version="4.6.0",
        anchor_second_ms=_readonly(anchors, np.int64),
        utc_day=_readonly(days, np.int32),
        primary_target=_readonly(target, np.float64),
        primary_prediction=_readonly(prediction, np.float64),
        stress_anchor_second_ms=_readonly(anchors, np.int64),
        stress_utc_day=_readonly(days, np.int32),
        stress_target=_readonly(target, np.float64),
        stress_prediction=_readonly(prediction, np.float64),
        prediction_sha256="",
    )
    result = replace(
        provisional, prediction_sha256=_prediction_sha256(provisional)
    )
    result.validate()
    return result


def _run(*, paired_improves: bool) -> PriceDiscoveryPredictionRun:
    implementation_sha256 = json.loads(IMPLEMENTATION.read_text(encoding="utf-8"))[
        "implementation_sha256"
    ]
    blocks = tuple(
        _block(
            symbol=symbol,
            horizon=horizon,
            layer=layer,
            head=head,
            fold=fold,
            paired_improves=paired_improves,
        )
        for symbol in FLOW_SYMBOLS
        for horizon in HORIZONS_SECONDS
        for layer in PRIMARY_FEATURE_LAYERS
        for head in PRICE_DISCOVERY_HEADS
        for fold in range(1, 7)
    )
    provisional = PriceDiscoveryPredictionRun(
        schema_version=PRICE_DISCOVERY_MODEL_RUN_SCHEMA,
        implementation_sha256=implementation_sha256,
        dataset_bundle_sha256="a" * 64,
        feature_layers=PRIMARY_FEATURE_LAYERS,
        backend_requested="cpu",
        backend_kind="cpu",
        backend_device="cpu",
        lightgbm_version="4.6.0",
        blocks=blocks,
        run_sha256="",
    )
    result = replace(provisional, run_sha256=_run_sha256(provisional))
    result.validate()
    return result


def test_round72_metric_definitions_handle_ties_and_day_bootstrap() -> None:
    target = np.asarray([0.0, 1.0, 1.0, 0.0])
    probability = np.asarray([0.1, 0.8, 0.8, 0.1])
    binary = binary_predictive_metrics(target, probability)

    assert binary["accuracy"] == 1.0
    assert binary["balanced_accuracy"] == 1.0
    assert binary["MCC"] == 1.0
    assert binary["ROC_AUC"] == 1.0
    assert binary["precision_recall_AUC"] == 1.0

    continuous = continuous_predictive_metrics(
        np.asarray([-2.0, -1.0, 1.0, 2.0]),
        np.asarray([-1.8, -0.9, 0.9, 1.8]),
    )
    assert continuous["Spearman"] == pytest.approx(1.0)
    assert continuous["sign_accuracy"] == 1.0

    bootstrap = day_block_bootstrap_lower(
        np.linspace(0.51, 0.70, 20), draws=1_000, seed=7
    )
    assert bootstrap["finite_days"] == 20
    assert bootstrap["lower_95"] is not None
    assert 0.51 < float(bootstrap["lower_95"]) < 0.70


def test_round72_full_primary_gate_passes_only_real_feature_increment() -> None:
    passing = evaluate_price_discovery_primary(
        _run(paired_improves=True),
        implementation_path=IMPLEMENTATION,
        corpus_certificate=_certificate(),
    )

    assert passing["primary_gate_passed"] is True
    assert passing["feature_increment_gate_passed"] is True
    assert passing["decision"] == "open_frozen_terminal_holdout"
    assert len(passing["feature_comparisons"]) == 36
    assert all(value["passed"] for value in passing["feature_comparisons"])
    assert all(value["passed"] for value in passing["symbol_horizon_components"])
    assert passing["scope"]["terminal_holdout_read"] is False
    assert passing["profitability_claim"] is False

    rejected = evaluate_price_discovery_primary(
        _run(paired_improves=False),
        implementation_path=IMPLEMENTATION,
        corpus_certificate=_certificate(),
    )

    assert rejected["primary_gate_passed"] is False
    assert rejected["feature_increment_gate_passed"] is False
    assert rejected["decision"] == "reject_round_072_price_discovery"
    assert not any(value["passed"] for value in rejected["feature_comparisons"])
