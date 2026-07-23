from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading import impact_absorption_one_use_evaluation as subject
from simple_ai_trading.impact_absorption_model_dataset import (
    ROUND73_OBSERVED_STATUS,
    ROUND73_POST_ENTRY_UNRESOLVED_STATUS,
)


def _event(
    *,
    symbol: str,
    run_byte: int,
    wall_ns: int,
    status: int,
    entry_wall_ns: int,
    exit_wall_ns: int,
    net_bps: float,
    net_quote: float,
    reason_code: int,
) -> subject._AttemptEvent:
    return subject._AttemptEvent(
        symbol=symbol,
        run_id=bytes([run_byte]) * 16,
        anchor_index=run_byte,
        anchor_wall_ns=wall_ns,
        utc_day_ordinal=wall_ns // subject._DAY_NS,
        side_sign=1,
        status=status,
        reason_code=reason_code,
        net_bps=net_bps,
        net_payoff_quote=net_quote,
        actual_entry_wall_ns=entry_wall_ns,
        actual_exit_wall_ns=exit_wall_ns,
        entry_quote_notional=1_000.0,
        exit_quote_notional=1_001.0,
        maximum_adverse_excursion_bps=(
            net_bps - 1.0 if np.isfinite(net_bps) else float("nan")
        ),
        maximum_favorable_excursion_bps=(
            net_bps + 1.0 if np.isfinite(net_bps) else float("nan")
        ),
        maximum_spread_bps=1.0 if np.isfinite(net_bps) else float("nan"),
        minimum_exit_side_capacity_ratio=(
            2.0 if np.isfinite(net_bps) else float("nan")
        ),
    )


def test_fixed_binary_row_preserves_trailing_zero_bytes() -> None:
    values = np.empty(1, dtype="S16")
    expected = b"\x01" * 15 + b"\x00"
    values.view(np.uint8)[:] = np.frombuffer(expected, dtype=np.uint8)

    assert subject._fixed_binary_row(values, 0, 16) == expected


def test_predictive_block_interval_is_deterministic_and_positive() -> None:
    baseline = np.ones(40, dtype=np.float64)
    challenger = np.full(40, 0.8, dtype=np.float64)
    blocks = np.repeat(np.arange(10, dtype=np.int64), 4)

    first = subject._paired_block_bootstrap_interval(
        baseline,
        challenger,
        blocks,
        seed=17,
    )
    second = subject._paired_block_bootstrap_interval(
        baseline,
        challenger,
        blocks,
        seed=17,
    )

    assert first == second
    assert first["relative_improvement_lower"] > 0.0
    assert first["relative_improvement_finite_draws"] == 10_000


def test_run_blocks_follow_first_market_time_occurrence_not_hash_sort_order() -> None:
    run_ids = np.asarray(
        [b"z" * 16, b"z" * 16, b"a" * 16, b"m" * 16, b"a" * 16],
        dtype="S16",
    )

    blocks = subject._chronological_block_ids(run_ids)

    assert blocks.tolist() == [0, 0, 1, 2, 1]
    assert blocks.flags.writeable is False


def test_scenario_allocation_fails_before_exceeding_explicit_memory_budget() -> None:
    required = subject._validate_scenario_memory_budget(
        subject._SCENARIO_ROW_ALLOCATION_BYTES * 2,
        rows=2,
    )

    assert required == subject._SCENARIO_ROW_ALLOCATION_BYTES * 2
    with pytest.raises(MemoryError, match="explicit memory budget"):
        subject._validate_scenario_memory_budget(
            subject._SCENARIO_ROW_ALLOCATION_BYTES * 2 - 1,
            rows=2,
        )


def test_claimed_test_reaudit_replays_every_manifest_before_scoring(
    monkeypatch,
) -> None:
    run_ids = ("1" * 32, "2" * 32)
    manifest_hashes = ("3" * 64, "4" * 64)
    sealed_role_hash = subject._stream_hash(list(manifest_hashes))
    claim = subject.Round73EvaluationAccessClaim(
        study_id="a" * 32,
        pretest_manifest_sha256="b" * 64,
        test_study_manifest_sha256="c" * 64,
        repository_commit_sha="d" * 40,
        repository_tree_sha="e" * 40,
        claimed_at_wall_ns=1,
    )

    class Result:
        def __init__(self, row=None, rows=None):
            self._row = row
            self._rows = rows

        def fetchone(self):
            return self._row

        def fetchall(self):
            return self._rows

    class Connection:
        def execute(self, query, _parameters):
            if subject.ROUND73_TARGET_V3_TEST_STUDY_TABLE in query:
                return Result(
                    row=(
                        2,
                        sealed_role_hash,
                        claim.test_study_manifest_sha256,
                    )
                )
            return Result(rows=list(zip(run_ids, manifest_hashes, strict=True)))

    class Store:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def connect(self):
            return Connection()

    audit_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(subject, "ImpactAbsorptionStore", lambda *_a, **_k: Store())
    monkeypatch.setattr(
        subject,
        "audit_round73_shock_cohort",
        lambda *_a, **_k: SimpleNamespace(passed=True),
    )

    def audit(*_args, **kwargs):
        run_id = kwargs["run_id"]
        audit_calls.append((run_id, kwargs["deep_replay"]))
        index = run_ids.index(run_id)
        return SimpleNamespace(
            passed=True,
            deep_replay_performed=True,
            target_manifest_sha256=manifest_hashes[index],
        )

    monkeypatch.setattr(subject, "audit_round73_role_targets", audit)
    progress: list[str] = []

    count = subject._reaudit_claimed_test_targets(
        "study.duckdb",
        claim=claim,
        memory_limit="2GB",
        threads=2,
        progress_callback=lambda event, _details: progress.append(event),
    )

    assert count == 2
    assert audit_calls == [(run_ids[0], True), (run_ids[1], True)]
    assert progress.count("test_exact_wire_reaudit_started") == 2
    assert progress.count("test_exact_wire_reaudit_completed") == 2

    manifest_hashes = ("3" * 64, "5" * 64)
    with pytest.raises(ValueError, match="role-manifest aggregate"):
        subject._reaudit_claimed_test_targets(
            "study.duckdb",
            claim=claim,
            memory_limit="2GB",
            threads=2,
            progress_callback=None,
        )


def test_unresolved_position_blocks_same_symbol_and_fails_closed() -> None:
    start = 2_000 * subject._DAY_NS
    end = start + 2 * subject._DAY_NS
    events = (
        _event(
            symbol="BTCUSDT",
            run_byte=1,
            wall_ns=start + 1_000_000_000,
            status=ROUND73_POST_ENTRY_UNRESOLVED_STATUS,
            entry_wall_ns=start + 1_500_000_000,
            exit_wall_ns=-1,
            net_bps=float("nan"),
            net_quote=float("nan"),
            reason_code=7,
        ),
        _event(
            symbol="BTCUSDT",
            run_byte=2,
            wall_ns=start + 120_000_000_000,
            status=ROUND73_OBSERVED_STATUS,
            entry_wall_ns=start + 120_500_000_000,
            exit_wall_ns=start + 180_500_000_000,
            net_bps=3.0,
            net_quote=0.3,
            reason_code=0,
        ),
        _event(
            symbol="ETHUSDT",
            run_byte=3,
            wall_ns=start + 2_000_000_000,
            status=ROUND73_OBSERVED_STATUS,
            entry_wall_ns=start + 2_500_000_000,
            exit_wall_ns=start + 62_500_000_000,
            net_bps=2.0,
            net_quote=0.2,
            reason_code=0,
        ),
    )

    report = subject._simulate_portfolio(
        events,
        scenario=next(
            scenario
            for scenario in subject.ROUND73_EVALUATION_SCENARIOS
            if scenario.primary
        ),
        study_start_wall_ns=start,
        study_end_wall_ns=end,
        seed=23,
    )
    combined = report["combined"]

    assert combined["post_entry_unresolved_risk_count"] == 1
    assert combined["same_symbol_open_skips"] == 1
    assert combined["completed_trades"] == 1
    assert combined["complete_transaction_fraction"] == 0.5
    assert combined["operational_gate_passed"] is False
    assert combined["economic_gate_passed"] is False
    assert combined["return_and_risk_metrics_cover_every_deployed_position"] is False
    assert combined["maximum_single_position_adverse_excursion_bps"] == 1.0
    assert combined["maximum_single_position_favorable_excursion_bps"] == 3.0
    assert combined["maximum_observed_spread_bps_during_positions"] == 1.0
    assert combined["minimum_observed_exit_side_capacity_ratio"] == 2.0
    assert combined["intratrade_portfolio_maximum_drawdown_reported"] is False


def _predictive_report(
    *,
    point_failure: bool = False,
    uncertainty_failure: bool = False,
) -> dict[str, object]:
    losses = (
        "log_loss",
        "brier_score",
        "mean_squared_error",
        "mean_absolute_error",
    )
    comparisons = []
    for index, loss in enumerate(losses):
        comparisons.append(
            {
                "stage": "linear_vs_controls",
                "loss": loss,
                "available": True,
                "q_value": 0.01,
                "relative_improvement": (
                    -0.01 if point_failure and index == 0 else 0.01
                ),
                "positive_chronological_folds": 4,
                "block_bootstrap_95_percent_interval": {
                    "relative_improvement_lower": (
                        -0.01 if uncertainty_failure and index == 0 else 0.005
                    )
                },
            }
        )
    return {
        "models": {
            "linear_l1_tape": {"binary": {"available": True, "single_class": False}}
        },
        "comparisons": comparisons,
    }


def test_predictive_gate_requires_primary_and_delay_stress_skill() -> None:
    passed = subject._symbol_predictive_gate(
        _predictive_report(),
        _predictive_report(),
        selected_candidate="linear_l1_tape",
    )
    failed = subject._symbol_predictive_gate(
        _predictive_report(),
        _predictive_report(point_failure=True),
        selected_candidate="linear_l1_tape",
    )
    uncertain = subject._symbol_predictive_gate(
        _predictive_report(uncertainty_failure=True),
        _predictive_report(),
        selected_candidate="linear_l1_tape",
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert any("delay_stress" in reason for reason in failed["reasons"])
    assert uncertain["passed"] is False
    assert any("primary" in reason for reason in uncertain["reasons"])


def test_independent_symbol_gate_requires_predictive_and_economic_passes() -> None:
    predictive = {symbol: {"passed": True} for symbol in subject.IMPACT_CAPTURE_SYMBOLS}
    economics = {
        symbol: {
            "operational_gate_passed": True,
            "economic_gate_passed": symbol != "ETHUSDT",
        }
        for symbol in subject.IMPACT_CAPTURE_SYMBOLS
    }

    gates, count = subject._independent_symbol_viability_gates(
        predictive_gates=predictive,
        primary_economic_by_symbol=economics,
        enabled_symbols=subject.IMPACT_CAPTURE_SYMBOLS,
    )

    assert count == 2
    assert gates["BTCUSDT"]["passed"] is True
    assert gates["ETHUSDT"]["passed"] is False
    assert gates["SOLUSDT"]["passed"] is True

    predictive["SOLUSDT"] = {"passed": False}
    _gates, count = subject._independent_symbol_viability_gates(
        predictive_gates=predictive,
        primary_economic_by_symbol=economics,
        enabled_symbols=subject.IMPACT_CAPTURE_SYMBOLS,
    )
    assert count == 1


def test_multiple_testing_report_discloses_integrity_segment_limit() -> None:
    report = subject._apply_multiple_testing([])

    assert report["resampling_block"] == (
        "one-hour integrity segment identified by run_id"
    )
    assert report["adjacent_segment_dependence_fully_preserved"] is False
    assert report["day_level_or_durable_inference_claimed"] is False


def test_deeper_candidates_require_the_complete_staged_comparison_chain() -> None:
    assert subject._required_stages("linear_l1_tape") == ("linear_vs_controls",)
    assert subject._required_stages("l1_tape") == (
        "linear_vs_controls",
        "l1_tape_vs_linear",
    )
    assert subject._required_stages("l2_state") == (
        "linear_vs_controls",
        "l1_tape_vs_linear",
        "l2_state_vs_l1_tape",
        "l2_state_vs_linear",
    )
    assert subject._required_stages("impact_absorption") == (
        "linear_vs_controls",
        "l1_tape_vs_linear",
        "l2_state_vs_l1_tape",
        "impact_absorption_vs_l2_state",
        "impact_absorption_vs_linear",
    )
