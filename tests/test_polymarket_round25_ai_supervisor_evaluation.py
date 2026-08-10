from __future__ import annotations

from dataclasses import replace
import hashlib

from simple_ai_trading.polymarket_round25_ai_supervisor_evaluation import (
    Round25AISupervisorMatchedCondition,
    create_round25_ai_supervisor_uplift_panel,
    evaluate_round25_ai_supervisor_uplift,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _row(index: int) -> Round25AISupervisorMatchedCondition:
    event_start_ms = 1_200_000 + index * 300_000
    window_start_ms = event_start_ms + 120_000
    return Round25AISupervisorMatchedCondition(
        condition_id=f"0x{index + 1:064x}",
        event_start_ms=event_start_ms,
        decision_time_ms=window_start_ms + 1_000,
        supervisor_window_start_ms=window_start_ms,
        supervisor_packet_sha256=_sha(f"supervisor-packet-{index}"),
        selected_candidate_id="causal-multitask-tcn-residual-v1",
        selected_model_prediction_sha256=_sha(f"prediction-{index}"),
        deterministic_decision_sha256=_sha(f"decision-{index}"),
        matched_execution_scenario_sha256=_sha(f"execution-{index}"),
        resolution_authority_sha256=_sha(f"resolution-{index}"),
        control_trace_sha256=_sha(f"control-{index}"),
        fast_trace_sha256=_sha(f"fast-{index}"),
        slow_trace_sha256=_sha(f"slow-{index}"),
        hierarchical_trace_sha256=_sha(f"hierarchical-{index}"),
        fast_advisory_sha256=_sha(f"fast-advisory-{index}"),
        supervisor_advisory_sha256=_sha(f"supervisor-advisory-{index}"),
        combined_decision_sha256=_sha(f"combined-{index}"),
        control_after_cost_return=-0.002,
        fast_after_cost_return=-0.0005,
        slow_after_cost_return=-0.00025,
        hierarchical_after_cost_return=0.001,
        fast_valid_response=True,
        slow_valid_response=True,
        fast_schema_or_coherence_violation=False,
        slow_schema_or_coherence_violation=False,
        fast_intervened=True,
        slow_intervened=True,
    )


def _panel(rows: tuple[Round25AISupervisorMatchedCondition, ...] | None = None):
    selected = rows or tuple(_row(index) for index in range(500))
    return create_round25_ai_supervisor_uplift_panel(
        selected_candidate_id="causal-multitask-tcn-residual-v1",
        selection_population_end_ms=1_000_000,
        selection_condition_root_sha256=_sha("selection-population"),
        rows=selected,
    )


def test_round25_ai_supervisor_uplift_uses_clustered_four_arm_gate() -> None:
    result = evaluate_round25_ai_supervisor_uplift(_panel())
    pairs = tuple(
        (comparison.baseline_arm, comparison.challenger_arm)
        for comparison in result.comparisons
    )
    assert pairs == (
        ("ml_control", "fast_qwen3_4b"),
        ("ml_control", "slow_fin_r1_8b"),
        ("fast_qwen3_4b", "hierarchical_minimum_risk"),
        ("ml_control", "hierarchical_minimum_risk"),
    )
    assert result.condition_count == 500
    assert result.supervisor_window_count == 500
    assert result.fast_intervention_count == 500
    assert result.slow_intervention_count == 500
    assert result.gate_reasons == ()
    assert result.development_nomination_passed is True
    assert all(comparison.comparison_gate_passed for comparison in result.comparisons)
    assert result.ai_uplift_verified is False
    assert result.profitability_verified is False
    assert result.live_authority is False


def test_round25_ai_supervisor_uplift_is_reproducible_and_hash_bound() -> None:
    panel = _panel()
    first = evaluate_round25_ai_supervisor_uplift(panel)
    second = evaluate_round25_ai_supervisor_uplift(panel)
    assert first.result_sha256 == second.result_sha256
    assert tuple(
        comparison.mean_bootstrap_sha256 for comparison in first.comparisons
    ) == tuple(
        comparison.mean_bootstrap_sha256 for comparison in second.comparisons
    )
    changed = replace(panel.rows[0], control_after_cost_return=-0.003, row_sha256="")
    altered = _panel((changed, *panel.rows[1:]))
    assert altered.panel_sha256 != panel.panel_sha256


def test_round25_ai_supervisor_uplift_rejects_any_schema_violation() -> None:
    rows = list(_panel().rows)
    rows[0] = replace(
        rows[0],
        fast_valid_response=False,
        fast_schema_or_coherence_violation=True,
        row_sha256="",
    )
    result = evaluate_round25_ai_supervisor_uplift(_panel(tuple(rows)))
    assert result.development_nomination_passed is False
    assert "schema_or_coherence_violation_observed" in result.gate_reasons
    assert result.ai_uplift_verified is False
    assert result.paper_authority is False
    assert result.orders_submitted is False
