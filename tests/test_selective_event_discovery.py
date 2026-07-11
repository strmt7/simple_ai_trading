from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.run_action_value_discovery import _canonical_sha256
from tools.run_selective_event_discovery import (
    _artifact_is_reusable,
    _ensure_causal_feature_bars,
    _file_sha256,
    _implementation_is_current,
    _resolved_runtime_settings,
    _role_calendar_split,
    load_selective_event_design,
)


def _tracked_round_twelve_design(revision: int = 6) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "model-research"
        / "action-value"
        / f"round-012-design-v{revision}.json"
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
        "reserved_terminal": {
            "start_date": "2024-01-06",
            "end_date": "2024-01-06",
            "day_count": 1,
            "included_in_dataset": False,
            "labels_constructed": False,
            "access_allowed_in_round_12": False,
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
        "model_fit_count": 9,
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

    terminal = _design(
        tmp_path / "terminal",
        consumed=("2024-01-06", "2024-01-06"),
    )
    with pytest.raises(ValueError, match="reserved terminal"):
        load_selective_event_design(terminal)


def test_selective_event_design_accepts_bounded_viability_contract(tmp_path) -> None:
    path = _design(tmp_path / "bounded", consumed=("2023-12-31", "2023-12-31"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["purpose"] = "bounded_exact_bbo_model_viability_screen"
    payload["data"]["full_history_inventory_required"] = False
    payload["data"]["inventory_scope"] = "bounded_verified"
    payload["runtime_resources"] = {
        "duckdb_memory_limit": "4GB",
        "warehouse_threads": 8,
        "spill_directory_policy": "warehouse_adjacent",
        "feature_build_chunk_clock": "utc_event_day",
    }
    payload["training"]["predictor_parameter_profile"] = "shared_regularized"
    payload["horizon_seconds"] = [300, 900]
    payload["model_fit_count"] = 6
    payload["candidate_count"] = 18
    payload["selection"] = {
        "promotion_allowed": False,
        "minimum_policy_trades": 20,
        "minimum_selection_trades": 20,
        "activity_is_not_a_trade_quota": True,
        "positive_daily_bootstrap_lower_bound_required": True,
    }
    payload.pop("design_sha256")
    payload["design_sha256"] = _canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_selective_event_design(path)

    assert loaded["data"]["inventory_scope"] == "bounded_verified"
    assert loaded["training"]["predictor_parameter_profile"] == (
        "shared_regularized"
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("data", "full_history_inventory_required", True),
        ("data", "inventory_scope", "full_history"),
        ("training", "predictor_parameter_profile", "risk_specific"),
        ("selection", "minimum_policy_trades", 19),
        ("selection", "minimum_selection_trades", 19),
        ("selection", "activity_is_not_a_trade_quota", False),
    ),
)
def test_selective_event_design_rejects_weakened_bounded_contract(
    tmp_path,
    section: str,
    key: str,
    value: object,
) -> None:
    path = _design(tmp_path, consumed=("2023-12-31", "2023-12-31"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["purpose"] = "bounded_exact_bbo_model_viability_screen"
    payload["data"]["full_history_inventory_required"] = False
    payload["data"]["inventory_scope"] = "bounded_verified"
    payload["runtime_resources"] = {
        "duckdb_memory_limit": "4GB",
        "warehouse_threads": 8,
        "spill_directory_policy": "warehouse_adjacent",
        "feature_build_chunk_clock": "utc_event_day",
    }
    payload["training"]["predictor_parameter_profile"] = "shared_regularized"
    payload["horizon_seconds"] = [300, 900]
    payload["model_fit_count"] = 6
    payload["candidate_count"] = 18
    payload["selection"] = {
        "promotion_allowed": False,
        "minimum_policy_trades": 20,
        "minimum_selection_trades": 20,
        "activity_is_not_a_trade_quota": True,
        "positive_daily_bootstrap_lower_bound_required": True,
    }
    payload[section][key] = value
    payload.pop("design_sha256")
    payload["design_sha256"] = _canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_selective_event_design(path)


def test_bounded_runtime_settings_reject_all_operational_drift(tmp_path) -> None:
    path = _design(tmp_path, consumed=("2023-12-31", "2023-12-31"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["training"]["compute_backend"] = "directml"
    payload["runtime_resources"] = {
        "duckdb_memory_limit": "4GB",
        "warehouse_threads": 8,
        "spill_directory_policy": "warehouse_adjacent",
        "feature_build_chunk_clock": "utc_event_day",
    }

    assert _resolved_runtime_settings(
        payload,
        memory_limit=None,
        threads=None,
        compute_backend=None,
    ) == ("4GB", 8, "directml")
    assert _resolved_runtime_settings(
        payload,
        memory_limit="4gb",
        threads=8,
        compute_backend="directml",
    ) == ("4GB", 8, "directml")
    with pytest.raises(ValueError, match="memory limit override"):
        _resolved_runtime_settings(
            payload,
            memory_limit="8GB",
            threads=8,
            compute_backend="directml",
        )
    with pytest.raises(ValueError, match="thread override"):
        _resolved_runtime_settings(
            payload,
            memory_limit="4GB",
            threads=4,
            compute_backend="directml",
        )
    with pytest.raises(ValueError, match="compute backend override"):
        _resolved_runtime_settings(
            payload,
            memory_limit="4GB",
            threads=8,
            compute_backend="cpu",
        )


def test_implementation_binding_is_launch_directory_agnostic(
    tmp_path,
    monkeypatch,
) -> None:
    relative = Path("tools/run_selective_event_discovery.py")
    repository_file = Path(__file__).resolve().parents[1] / relative
    monkeypatch.chdir(tmp_path)

    _implementation_is_current(
        {"implementation_files_sha256": {str(relative): _file_sha256(repository_file)}}
    )

    with pytest.raises(ValueError, match="repository-relative"):
        _implementation_is_current(
            {"implementation_files_sha256": {str(repository_file): "a" * 64}}
        )


def test_causal_feature_build_reuses_current_evidence() -> None:
    calls: list[object] = []

    class _Warehouse:
        def require_causal_feature_bars(self, symbol):
            calls.append(("require", symbol))
            return {"feature_rows": 42, "verified": True}

        def rebuild_causal_feature_bars(self, symbol, *, progress):
            calls.append(("rebuild", symbol, progress))
            return {"feature_rows": 99, "verified": True}

    evidence = _ensure_causal_feature_bars(
        _Warehouse(),  # type: ignore[arg-type]
        "BTCUSDT",
        progress=lambda phase, done, total: calls.append((phase, done, total)),
    )

    assert evidence["feature_rows"] == 42
    assert calls == [("require", "BTCUSDT"), ("causal-feature-reuse", 42, 42)]


def test_causal_feature_build_rebuilds_stale_evidence() -> None:
    calls: list[object] = []

    class _Warehouse:
        def require_causal_feature_bars(self, symbol):
            calls.append(("require", symbol))
            raise ValueError("stale")

        def rebuild_causal_feature_bars(self, symbol, *, progress):
            calls.append(("rebuild", symbol))
            progress("causal-feature-aggregate", 0, 10)
            return {"feature_rows": 10, "verified": True}

    evidence = _ensure_causal_feature_bars(
        _Warehouse(),  # type: ignore[arg-type]
        "BTCUSDT",
        progress=lambda phase, done, total: calls.append((phase, done, total)),
    )

    assert evidence["feature_rows"] == 10
    assert calls == [
        ("require", "BTCUSDT"),
        ("rebuild", "BTCUSDT"),
        ("causal-feature-aggregate", 0, 10),
    ]


def test_round_twelve_v6_design_binds_resources_roles_and_terminal() -> None:
    design = load_selective_event_design(
        _tracked_round_twelve_design(),
        require_current=True,
    )

    assert design["design_sha256"] == (
        "933a8619248145f4fd2e433952a92cfb8b90db4429846a468a6072f18486587d"
    )
    assert design["change_control"]["implementation_commit"] == (
        "09ae2f2eeba81eacdd147be96075371659e0ba02"
    )
    assert design["design_revision"] == 6
    assert design["runtime_resources"] == {
        "duckdb_memory_limit": "4GB",
        "warehouse_threads": 8,
        "spill_directory_policy": "warehouse_adjacent",
        "feature_build_chunk_clock": "utc_event_day",
    }
    assert design["data"]["roles"]["train"]["day_count"] == 31
    assert design["data"]["roles"]["selection"]["day_count"] == 6
    assert design["reserved_terminal"]["start_date"] == "2023-07-07"
    assert design["supersession"]["model_feature_build_started"] is True
    assert design["supersession"]["model_fit_started"] is False
    assert design["supersession"]["selection_labels_accessed"] is False


def test_round_twelve_v5_design_is_preserved_but_no_longer_current() -> None:
    design = load_selective_event_design(_tracked_round_twelve_design(5))

    assert design["design_sha256"] == (
        "7948bf464c907a0825d62e6ad8208e183d08f6478c49c2c7c07724217c19a49f"
    )
    assert design["change_control"]["implementation_commit"] == (
        "045f51bd986014609f12d531235947175cc0412d"
    )
    assert design["design_revision"] == 5
    assert design["data"]["bounded_corpus_certificate_sha256"] == (
        "975692f08d730de22d628fc1f511ebf8fdcc1965f587eca75659657c2f2c33bd"
    )
    assert design["data"]["roles"]["train"]["day_count"] == 31
    assert design["data"]["roles"]["selection"]["day_count"] == 6
    assert design["training"]["predictor_parameter_profile"] == (
        "shared_regularized"
    )
    assert design["reserved_terminal"] == {
        "start_date": "2023-07-07",
        "end_date": "2023-07-07",
        "day_count": 1,
        "included_in_dataset": False,
        "labels_constructed": False,
        "access_allowed_in_round_12": False,
    }
    with pytest.raises(ValueError, match="implementation changed"):
        load_selective_event_design(
            _tracked_round_twelve_design(5),
            require_current=True,
        )


def test_round_twelve_v4_design_is_preserved_but_no_longer_current() -> None:
    design = load_selective_event_design(_tracked_round_twelve_design(4))

    assert design["design_sha256"] == (
        "f0dcfc57751384120f65b1d26589a1573d6121cdca23cd3006a0c387c41d1e6d"
    )
    assert design["change_control"]["implementation_commit"] == (
        "536a21aeabb461283cd61ee6c6ba00dcc1c39d66"
    )
    assert design["design_revision"] == 4
    assert design["data"]["merge_certificate_required"] is True
    assert design["supersession"]["model_fit_started"] is False
    assert design["supersession"]["reserved_terminal_accessed"] is False
    assert design["data"]["roles"]["train"]["day_count"] == 230
    assert design["data"]["roles"]["selection"]["day_count"] == 21
    assert design["reserved_terminal"] == {
        "start_date": "2024-03-07",
        "end_date": "2024-03-14",
        "day_count": 8,
        "included_in_dataset": False,
        "labels_constructed": False,
        "access_allowed_in_round_12": False,
    }
    with pytest.raises(ValueError, match="implementation changed"):
        load_selective_event_design(
            _tracked_round_twelve_design(4),
            require_current=True,
        )


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
