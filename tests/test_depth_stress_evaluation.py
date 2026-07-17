from __future__ import annotations

import hashlib

import numpy as np
import pytest

from simple_ai_trading.depth_stress_evaluation import (
    DEPTH_STRESS_EVALUATION_SCHEMA_VERSION,
    evaluate_depth_stress_symbol,
    finalize_depth_stress_gate,
)
from simple_ai_trading.depth_stress_screen import (
    DEPTH_STRESS_HORIZONS_SECONDS,
    DepthStressPanel,
    build_depth_stress_examples,
)


def _monthly_panel() -> tuple[DepthStressPanel, np.ndarray]:
    timestamp_parts: list[np.ndarray] = []
    descriptor_parts: list[np.ndarray] = []
    for month_offset in range(8):
        month = np.datetime64("2025-01", "M") + np.timedelta64(month_offset, "M")
        month_start = int(month.astype("datetime64[ms]").astype(np.int64))
        for day_offset in range(12):
            rows = 120
            start = month_start + day_offset * 86_400_000 + 3_600_000
            timestamps = start + np.arange(rows, dtype=np.int64) * 30_000
            sequence = np.arange(rows) + day_offset * rows
            phase = (sequence // 10 + month_offset) % 6
            wave = np.sin(sequence / 7.0) * 0.05
            descriptors = np.column_stack(
                (
                    np.where(np.isin(phase, [3, 4, 5]), 10.0, 1.0) + wave,
                    np.where(np.isin(phase, [1, 4, 5]), 11.0, 2.0) - wave,
                    np.where(np.isin(phase, [2, 3, 5]), 12.0, 3.0) + wave,
                )
            )
            timestamp_parts.append(timestamps)
            descriptor_parts.append(descriptors)
    panel = DepthStressPanel(
        symbol="BTCUSDT",
        timestamp_ms=np.concatenate(timestamp_parts),
        descriptors=np.concatenate(descriptor_parts),
        source_fingerprint=hashlib.sha256(b"monthly-panel-source").hexdigest(),
    )
    return panel, np.unique(panel.month_ordinals)


def test_rolling_evaluation_uses_untouched_month_and_both_baselines() -> None:
    panel, months = _monthly_panel()
    examples = {
        horizon: build_depth_stress_examples(panel, horizon_seconds=horizon)
        for horizon in DEPTH_STRESS_HORIZONS_SECONDS
    }
    report = evaluate_depth_stress_symbol(
        panel,
        examples,
        eligible_month_ordinals=months,
        compute_backend="cpu",
        maximum_iterations=32,
        permutation_draws=100,
        seed=17,
    )

    assert report["schema_version"] == DEPTH_STRESS_EVALUATION_SCHEMA_VERSION
    assert report["symbol"] == "BTCUSDT"
    assert report["eligible_months"] == 8
    assert len(report["folds"]) == 1
    assert report["folds"][0]["training_months"] == 6
    assert report["folds"][0]["tuning_month"] == "2025-07"
    assert report["folds"][0]["test_month"] == "2025-08"
    assert len(report["horizons"]) == 2
    assert len(report["comparisons"]) == 8
    assert {value["baseline"] for value in report["comparisons"]} == {
        "marginal",
        "pre_state_transition",
    }
    assert all(
        horizon["model"]["trading_authority"] is False
        and horizon["model"]["model_string_published"] is False
        for horizon in report["folds"][0]["horizons"]
    )


def _gate_report(*, symbol: str, improvement: float, p_value: float) -> dict[str, object]:
    comparisons = []
    for horizon in DEPTH_STRESS_HORIZONS_SECONDS:
        for baseline in ("marginal", "pre_state_transition"):
            for metric in ("negative_log_likelihood", "multiclass_brier"):
                comparisons.append(
                    {
                        "symbol": symbol,
                        "horizon_seconds": horizon,
                        "baseline": baseline,
                        "metric": metric,
                        "relative_improvement": improvement,
                        "one_sided_p_value": p_value,
                    }
                )
    return {
        "schema_version": DEPTH_STRESS_EVALUATION_SCHEMA_VERSION,
        "symbol": symbol,
        "comparisons": comparisons,
        "profitability_claim": False,
        "trading_authority": False,
    }


def test_gate_requires_every_fdr_adjusted_comparison() -> None:
    passing_reports = [
        _gate_report(symbol=symbol, improvement=0.01, p_value=0.001)
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    ]
    passing = finalize_depth_stress_gate(
        passing_reports
    )
    failing_reports = [
        _gate_report(symbol=symbol, improvement=0.01, p_value=0.001)
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    ]
    failing_reports[1]["comparisons"][0]["relative_improvement"] = 0.0
    failing = finalize_depth_stress_gate(failing_reports)

    assert passing["passed"] is True
    assert passing["decision"] == "authorize_separately_frozen_paired_economic_replay"
    assert all(value["q_value"] <= 0.05 for value in passing["comparisons"])
    assert failing["passed"] is False
    assert failing["decision"] == "reject_coarse_depth_stress_layer"
    assert failing["profitability_claim"] is False
    assert failing["trading_authority"] is False


def test_gate_rejects_authority_or_incomplete_comparison_family() -> None:
    reports = [
        _gate_report(symbol=symbol, improvement=0.01, p_value=0.001)
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    ]
    reports[2]["trading_authority"] = True
    with pytest.raises(ValueError, match="contract"):
        finalize_depth_stress_gate(reports)

    reports[2]["trading_authority"] = False
    reports[2]["comparisons"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        finalize_depth_stress_gate(reports)
