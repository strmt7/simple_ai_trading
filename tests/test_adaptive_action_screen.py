from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading.microstructure_action_architecture import (
    ActionValueEnsembleBatch,
)
from simple_ai_trading.microstructure_barriers import AdaptiveBarrierSpec
from tools.run_adaptive_action_screen import (
    _DAY_MS,
    _forecast_diagnostics,
    _role_indexes,
    _targets_sha256,
    load_adaptive_action_design,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-016-adaptive-action-design.json"
)


def _day(value: str) -> int:
    parsed = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000 // _DAY_MS)


def _roles() -> dict[str, dict[str, str]]:
    return {
        "train": {"start": "2023-05-16", "end": "2023-06-15"},
        "early_stop": {"start": "2023-06-16", "end": "2023-06-20"},
        "calibration": {"start": "2023-06-21", "end": "2023-06-25"},
        "policy": {"start": "2023-06-26", "end": "2023-06-30"},
        "development_evaluation": {"start": "2023-07-01", "end": "2023-07-06"},
    }


def _target_namespace(rows: int, *, start_index: int = 0):
    source = np.arange(start_index, start_index + rows, dtype=np.int64)
    zeros = np.zeros(rows, dtype=np.int8)
    exits = np.arange(rows, dtype=np.int64) + 1_000
    return SimpleNamespace(
        schema_version="adaptive-bbo-barrier-targets-v1",
        target_mode="exchange_trigger_market_exit_100ms_base_and_adverse_stress_v1",
        spec=AdaptiveBarrierSpec(
            horizon_seconds=900,
            volatility_feature_name="realized_volatility_300s_bps",
            stop_volatility_multiple=1.0,
            take_volatility_multiple=1.5,
            minimum_stop_bps=18.0,
            maximum_stop_bps=60.0,
            minimum_take_bps=27.0,
            maximum_take_bps=90.0,
            base_protection_delay_ms=250,
            stress_protection_delay_ms=750,
            trigger_execution_slippage_bps=1.0,
        ),
        source_indexes=source,
        valid=np.ones(rows, dtype=bool),
        stop_barrier_bps=np.full(rows, 18.0),
        take_barrier_bps=np.full(rows, 27.0),
        base_long_net_bps=np.linspace(-2.0, 3.0, rows),
        base_short_net_bps=np.linspace(2.0, -3.0, rows),
        base_long_exit_time_ms=exits.copy(),
        base_short_exit_time_ms=exits.copy(),
        base_long_outcome=zeros.copy(),
        base_short_outcome=zeros.copy(),
        stress_long_net_bps=np.linspace(-3.0, 2.0, rows),
        stress_short_net_bps=np.linspace(1.0, -4.0, rows),
        stress_long_exit_time_ms=exits.copy(),
        stress_short_exit_time_ms=exits.copy(),
        stress_long_outcome=zeros.copy(),
        stress_short_outcome=zeros.copy(),
        rows=rows,
    )


def test_barrier_target_hash_binds_contract_and_every_array() -> None:
    targets = _target_namespace(8)
    first = _targets_sha256(targets)
    second = _targets_sha256(targets)
    targets.stress_long_net_bps[0] += 1.0

    assert first == second
    assert len(first) == 64
    assert _targets_sha256(targets) != first


def test_role_indexes_are_event_filtered_valid_and_exit_purged() -> None:
    roles = _roles()
    times = np.concatenate(
        [
            role_start * _DAY_MS + 1_000 + np.arange(300, dtype=np.int64) * 5_000
            for role_start in (_day(value["start"]) for value in roles.values())
        ]
    )
    rows = len(times)
    exits = times + 900_750
    dataset = SimpleNamespace(rows=rows, decision_time_ms=times)
    targets = _target_namespace(rows)
    for name in (
        "base_long_exit_time_ms",
        "base_short_exit_time_ms",
        "stress_long_exit_time_ms",
        "stress_short_exit_time_ms",
    ):
        setattr(targets, name, exits.copy())

    output, evidence = _role_indexes(
        dataset,
        targets,
        np.ones(rows, dtype=bool),
        roles,
        _day("2023-07-07"),
    )

    assert {name: len(indexes) for name, indexes in output.items()} == {
        name: 300 for name in roles
    }
    assert all(value["purged"] is True for value in evidence.values())
    assert output["train"][-1] < output["early_stop"][0]


def test_forecast_diagnostics_report_cost_target_quality_and_tail_rows() -> None:
    targets = _target_namespace(8)
    endpoints = np.arange(8, dtype=np.int64)
    ones = np.ones(8, dtype=np.float64)
    prediction = ActionValueEnsembleBatch(
        endpoint_indexes=endpoints,
        long_mean_bps=targets.base_long_net_bps.copy(),
        short_mean_bps=targets.base_short_net_bps.copy(),
        long_epistemic_std_bps=0.1 * ones,
        short_epistemic_std_bps=0.1 * ones,
        long_profitable_probability=np.linspace(0.1, 0.9, 8),
        short_profitable_probability=np.linspace(0.9, 0.1, 8),
        long_lower_bps=targets.base_long_net_bps - 1.0,
        short_lower_bps=targets.base_short_net_bps - 1.0,
        long_upper_bps=targets.base_long_net_bps + 1.0,
        short_upper_bps=targets.base_short_net_bps + 1.0,
        long_positive_member_ratio=ones,
        short_positive_member_ratio=ones,
        member_count=3,
    )

    diagnostics = _forecast_diagnostics(targets, prediction, scenario="base")
    long_metrics = diagnostics["sides"]["long"]

    assert long_metrics["mean_absolute_error_bps"] == 0.0
    assert long_metrics["profitable_auc"] > 0.9
    assert long_metrics["interval_80_coverage"] == 1.0
    assert long_metrics["top_rows"][0]["actual_rows"] == 8
    assert diagnostics["profitability_claim"] is False
    with pytest.raises(ValueError, match="scenario is unsupported"):
        _forecast_diagnostics(targets, prediction, scenario="future")
    with pytest.raises(ValueError, match="differ from barrier targets"):
        _forecast_diagnostics(
            targets,
            replace(prediction, endpoint_indexes=np.arange(1, 9)),
            scenario="base",
        )


def test_design_loader_rejects_unreadable_and_nonobject_payloads(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="unreadable"):
        load_adaptive_action_design(missing)
    source = tmp_path / "design.json"
    source.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_adaptive_action_design(source)


def test_tracked_round16_design_is_hash_bound_current_and_terminal_sealed() -> None:
    design, design_sha256 = load_adaptive_action_design(DESIGN)

    assert (
        design_sha256
        == "15e8702999f7ed2c5acdd5ab27c19535ad87c68e476b92339b7335d98991a639"
    )
    assert design["implementation"]["commit"] == (
        "b3327449ed22ae68007db77084d1d26da0f5cff3"
    )
    assert design["reserved_terminal"] == {
        "date": "2023-07-07",
        "included_in_dataset": False,
        "access_permitted": False,
    }
    assert [value["profile"] for value in design["risk_profiles"]] == [
        "conservative",
        "regular",
        "aggressive",
    ]
    assert design["leverage_applied"] is False


def test_round16_design_rejects_hash_and_contract_tampering(tmp_path: Path) -> None:
    payload = json.loads(DESIGN.read_text(encoding="utf-8"))
    payload["training"]["ensemble_seeds"] = [29, 43, 72]
    source = tmp_path / "tampered.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="design hash is invalid"):
        load_adaptive_action_design(source, require_current=False)
