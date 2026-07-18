from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from simple_ai_trading.polymarket_directional_value import (
    POLYMARKET_ROUND11_CONTRACT_SHA256,
    POLYMARKET_ROUND11_DIRECTION_FEATURE_NAMES,
    POLYMARKET_ROUND11_EXECUTION_FEATURE_NAMES,
    PolymarketRound11Dataset,
    _condition_balanced_weights,
    _fit_scaler,
    _fit_value_head,
    _policy_metrics,
    build_round11_development_split,
)


ROOT = Path(__file__).resolve().parents[1]
POLYMARKET = ROOT / "docs/model-research/polymarket"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def test_round11_frozen_contract_and_outputs_are_hash_bound() -> None:
    contract = json.loads(
        (POLYMARKET / "round-011-single-leg-directional-value-contract.json").read_text(
            encoding="utf-8"
        )
    )
    claimed_contract = contract.pop("contract_sha256")
    assert claimed_contract == POLYMARKET_ROUND11_CONTRACT_SHA256
    assert _canonical_sha256(contract) == claimed_contract

    report = json.loads(
        (POLYMARKET / "round-011-single-leg-directional-value-report.json").read_text(
            encoding="utf-8"
        )
    )
    claimed_report = report.pop("report_sha256")
    assert _canonical_sha256(report) == claimed_report
    assert report["development_passed"] is False
    assert report["profitability_claim"] is False
    assert report["trading_authority"] is False
    assert report["selected_policy"]["gate_passed"] is False
    assert (
        "nonpositive_bootstrap_lower_mean_group_utility"
        in report["selected_policy"]["gate_reasons"]
    )

    artifact = json.loads(
        (POLYMARKET / "round-011-single-leg-directional-value-artifact.json").read_text(
            encoding="utf-8"
        )
    )
    claimed_artifact = artifact.pop("artifact_sha256")
    assert _canonical_sha256(artifact) == claimed_artifact
    assert report["artifact_sha256"] == claimed_artifact
    assert artifact["onnx_exported"] is False


def test_condition_weights_do_not_turn_cadence_rows_into_independent_markets() -> None:
    conditions = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int32)
    mask = np.ones(conditions.size, dtype=np.bool_)
    weights = _condition_balanced_weights(conditions, mask)
    assert np.isclose(np.sum(weights), 1.0)
    condition_mass = [float(np.sum(weights[conditions == value])) for value in range(3)]
    assert np.allclose(condition_mass, np.full(3, 1 / 3))


def test_q90_cost_head_points_upward_on_an_intercept_only_sample() -> None:
    target = np.linspace(0.0, 10.0, 201)
    features = np.zeros((target.size, 2), dtype=np.float64)
    weights = np.full(target.size, 1.0 / target.size)
    scaler = _fit_scaler(features, weights)
    head = _fit_value_head(
        scaler.transform(features),
        target,
        weights,
        name="q90_probe",
        objective_name="pinball_q90",
        l2=0.1,
    )
    prediction = head.predict_value(scaler.transform(features))
    assert np.all(np.isfinite(prediction))
    assert 8.7 <= float(prediction[0]) <= 9.3


def _state_machine_dataset() -> PolymarketRound11Dataset:
    rows = 141
    groups = np.repeat(np.arange(47, dtype=np.int64) * 300_000, 3)
    condition_index = np.arange(rows, dtype=np.int32)
    direction = np.zeros(
        (rows, len(POLYMARKET_ROUND11_DIRECTION_FEATURE_NAMES)), dtype=np.float64
    )
    direction[:, 0] = 200.0
    execution = np.zeros(
        (rows, 2, len(POLYMARKET_ROUND11_EXECUTION_FEATURE_NAMES)),
        dtype=np.float64,
    )
    quantity_index = POLYMARKET_ROUND11_EXECUTION_FEATURE_NAMES.index(
        "minimum_order_quantity"
    )
    execution[:, :, quantity_index] = 5.0
    available = np.ones((rows, 2), dtype=np.bool_)
    observable = np.ones((rows, 2), dtype=np.bool_)
    filled = np.zeros((rows, 2), dtype=np.bool_)
    filled[:, 0] = True
    unknown = np.zeros((rows, 2), dtype=np.bool_)
    entry_cost = np.full((rows, 2), np.nan, dtype=np.float64)
    entry_cost[:, 0] = 2.0
    utility = np.zeros((rows, 2), dtype=np.float64)
    utility[:, 0] = 3.0
    terminal_code = np.zeros((rows, 2), dtype=np.int8)
    terminal_code[:, 0] = 1
    unknown_row = rows - 1
    observable[unknown_row, 0] = False
    filled[unknown_row, 0] = False
    unknown[unknown_row, 0] = True
    entry_cost[unknown_row, 0] = np.nan
    utility[unknown_row, 0] = -5.0
    terminal_code[unknown_row, 0] = 2
    return PolymarketRound11Dataset(
        pipeline_report_sha256="a" * 64,
        dataset_sha256="b" * 64,
        condition_ids=tuple(f"condition-{index}" for index in range(rows)),
        source_feature_sha256=np.asarray([b"c" * 64] * rows, dtype="S64"),
        condition_index=condition_index,
        asset_index=np.tile(np.arange(3, dtype=np.int8), 47),
        event_start_ms=groups,
        decision_monotonic_ns=np.arange(rows, dtype=np.uint64) * 2_000_000_000,
        remaining_seconds=np.full(rows, 200.0),
        official_up=np.zeros(rows, dtype=np.bool_),
        market_prior_up=np.full(rows, 0.5),
        direction_features=direction,
        execution_features=execution,
        available=available,
        observable=observable,
        entry_filled=filled,
        unknown_entry=unknown,
        entry_cost_quote=entry_cost,
        current_top_ask_cost_quote=np.full((rows, 2), 2.0),
        maximum_entry_loss_quote=np.full((rows, 2), 5.0),
        realized_hold_utility_quote=utility,
        terminal_reason_code=terminal_code,
        terminal_reasons=(
            "entry_not_filled",
            "complete_round_trip",
            "missing_entry_execution_book",
        ),
    ).validated()


def test_policy_state_machine_charges_unknown_entry_and_fails_closed() -> None:
    dataset = _state_machine_dataset()
    split = build_round11_development_split(dataset)
    metrics = _policy_metrics(
        dataset,
        split,
        probability_up=np.full(dataset.rows, 0.1),
        observable_probability=np.full((dataset.rows, 2), 0.99),
        fill_probability=np.full((dataset.rows, 2), 0.99),
        upper_entry_cost=np.full((dataset.rows, 2), 2.0),
        margin_quote=0.0,
        minimum_remaining_seconds=120.0,
        include_decisions=True,
    )
    assert metrics["attempts"] == 42
    assert metrics["filled_conditions"] == 41
    assert metrics["wins"] == 41
    assert metrics["unknown_entries"] == 1
    assert metrics["total_utility_quote"] == 118.0
    assert metrics["gate_passed_before_market_prior_comparison"] is False
    assert "selected_unknown_entry" in metrics["gate_reasons"]


def test_latest_round11_csv_reconstructs_reported_equity() -> None:
    report = json.loads(
        (POLYMARKET / "round-011-single-leg-directional-value-report.json").read_text(
            encoding="utf-8"
        )
    )
    with (POLYMARKET / "latest/tables/round11-equity.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 14
    assert np.isclose(
        float(rows[-1]["cumulative_utility_quote"]),
        report["selected_policy"]["total_utility_quote"],
    )
    assert np.isclose(
        max(float(row["drawdown_quote"]) for row in rows),
        report["selected_policy"]["maximum_drawdown_quote"],
    )
