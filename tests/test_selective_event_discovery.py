from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.run_action_value_discovery import _canonical_sha256
from tools.run_selective_event_discovery import (
    _artifact_is_reusable,
    _role_calendar_split,
    load_selective_event_design,
)


def _registry(path: Path, window: tuple[str, str]) -> str:
    payload = {
        "schema_version": "action-value-consumed-periods-v1",
        "records": [
            {
                "round": 1,
                "status": "consumed",
                "windows": [
                    {"start_date": window[0], "end_date": window[1]}
                ],
            }
        ],
    }
    payload["registry_sha256"] = _canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(payload["registry_sha256"])


def _design(tmp_path: Path, *, consumed: tuple[str, str]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry_hash = _registry(tmp_path / "registry.json", consumed)
    roles = {
        role: {
            "start_date": f"2024-01-0{index}",
            "end_date": f"2024-01-0{index}",
            "day_count": 1,
        }
        for index, role in enumerate(
            ("train", "early_stop", "calibration", "policy", "selection"),
            start=1,
        )
    }
    profile = {
        "stop_loss_bps": 25.0,
        "take_profit_bps": 40.0,
        "max_l1_participation": 0.05,
        "max_selection_drawdown_bps": 250.0,
    }
    payload = {
        "schema_version": "selective-event-discovery-design-v1",
        "round": 12,
        "status": "precommitted",
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "change_control": {
            "implementation_commit": "a" * 40,
            "implementation_files_sha256": {"placeholder": "b" * 64},
        },
        "data": {
            "provider": "binance",
            "market_type": "futures",
            "symbol": "BTCUSDT",
            "required_data_types": ["bookTicker", "trades"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-05",
            "roles": roles,
            "full_history_inventory_required": True,
            "checksum_verified_partitions_required": True,
            "selection_dates_previously_untouched": True,
            "consumed_registry": "registry.json",
            "consumed_registry_sha256": registry_hash,
        },
        "execution": {
            "total_latency_ms": 750,
            "taker_fee_bps_per_side": 5.0,
            "additional_slippage_bps_per_side": 1.0,
            "trigger_execution_slippage_bps": 1.0,
            "max_quote_age_ms": 1000,
            "reference_order_notional_quote": 1000.0,
            "decision_cadence_seconds": 5,
            "suppress_overlapping_positions": True,
            "maker_fill_claim": False,
            "leverage": 1.0,
        },
        "training": {
            "model_family": "causal_cusum_uniqueness_weighted_distributional_lgbm",
            "feature_version": "l1-tape-causal-v7",
            "cusum_volatility_multiplier": 1.0,
            "cusum_minimum_threshold_bps": 1.0,
            "score_methods": [
                "event_direct_mean",
                "event_upper_quantile",
                "event_distributional_value",
            ],
            "compute_backend": "directml",
            "seed": 20260712,
            "evaluate_terminal": False,
        },
        "risk_profiles": {
            "conservative": profile,
            "regular": {**profile, "max_l1_participation": 0.10},
            "aggressive": {**profile, "max_l1_participation": 0.15},
        },
        "horizon_seconds": [300, 900, 1800],
        "candidate_count": 27,
        "selection": {
            "promotion_allowed": False,
            "minimum_trades_per_selection_day": 5,
            "positive_daily_bootstrap_lower_bound_required": True,
        },
    }
    payload["design_sha256"] = _canonical_sha256(payload)
    path = tmp_path / "design.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_selective_event_design_rejects_consumed_selection_date(tmp_path) -> None:
    valid = _design(tmp_path / "valid", consumed=("2023-12-31", "2023-12-31"))
    assert load_selective_event_design(valid)["candidate_count"] == 27

    invalid = _design(tmp_path / "invalid", consumed=("2024-01-05", "2024-01-05"))
    with pytest.raises(ValueError, match="already consumed"):
        load_selective_event_design(invalid)


def test_role_calendar_split_purges_cross_boundary_labels() -> None:
    per_day = 300
    times = []
    for day in range(5):
        start = day * 86_400_000
        values = start + np.arange(per_day, dtype=np.int64) * 1_000
        values[-1] = start + 86_400_000 - 10_000
        times.append(values)
    decision = np.concatenate(times)
    dataset = SimpleNamespace(
        decision_time_ms=decision,
        long_exit_time_ms=decision + 20_000,
        short_exit_time_ms=decision + 20_000,
    )
    roles = {
        role: {
            "start_date": f"1970-01-0{day + 1}",
            "end_date": f"1970-01-0{day + 1}",
            "day_count": 1,
        }
        for day, role in enumerate(
            ("train", "early_stop", "calibration", "policy", "selection")
        )
    }

    split, evidence = _role_calendar_split(dataset, roles)

    assert all(len(split[role]) == per_day - 1 for role in tuple(roles)[:-1])
    assert len(split["selection"]) == per_day
    assert all(evidence[role]["purged_rows"] == 1 for role in tuple(roles)[:-1])
    assert evidence["selection"]["purged_rows"] == 0


def test_model_fit_resume_requires_canonical_binding(tmp_path) -> None:
    payload = {
        "schema_version": "selective-event-model-fit-v1",
        "design_sha256": "a" * 64,
        "corpus_certificate_sha256": "b" * 64,
        "model_fit_id": "conservative-h300",
        "terminal_holdout_accessed": False,
        "outcomes": [],
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _artifact_is_reusable(
        path,
        design_sha256="a" * 64,
        corpus_sha256="b" * 64,
        model_fit_id="conservative-h300",
    ) is not None

    payload["model_fit_id"] = "regular-h300"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _artifact_is_reusable(
        path,
        design_sha256="a" * 64,
        corpus_sha256="b" * 64,
        model_fit_id="regular-h300",
    ) is None
