from __future__ import annotations

import copy
from datetime import date, timedelta

import pytest

from simple_ai_trading.model_experiment import (
    ChoiceDomain,
    FidelityStage,
    FloatDomain,
    IntegerDomain,
    WindowPerformance,
    apply_successive_halving_stage,
    default_fidelity_stages,
    generate_latin_hypercube_design,
    plan_calendar_windows,
    tape_depth_candidate_design,
    validate_experiment_design_payload,
)


def test_latin_hypercube_is_deterministic_and_stratifies_each_continuous_margin() -> None:
    domains = (
        FloatDomain("learning_rate", 0.01, 0.09),
        IntegerDomain("minimum_leaf_rows", 64, 512, scale="log"),
        ChoiceDomain("feature_set", ("core", "cross_asset", "full")),
    )
    first = generate_latin_hypercube_design(domains, sampled_count=8, seed=41)
    repeated = generate_latin_hypercube_design(domains, sampled_count=8, seed=41)
    changed = generate_latin_hypercube_design(domains, sampled_count=8, seed=42)

    assert first == repeated
    assert first.design_sha256 == repeated.design_sha256
    assert first.design_sha256 != changed.design_sha256
    assert first.trial_burden == 8
    values = [candidate.parameter_map()["learning_rate"] for candidate in first.candidates]
    strata = {
        min(7, int((float(value) - 0.01) / (0.09 - 0.01) * 8))
        for value in values
    }
    assert strata == set(range(8))


def test_tape_depth_design_counts_anchors_and_all_sampled_trials() -> None:
    design = tape_depth_candidate_design("conservative", sampled_count=12, seed=7)

    assert design.anchor_count == 3
    assert design.sampled_count == 12
    assert design.trial_burden == 15
    assert len(design.candidates) == 15
    assert len({candidate.candidate_id for candidate in design.candidates}) == 15
    assert all(
        candidate.parameter_map()["risk_level"] == "conservative"
        for candidate in design.candidates
    )


def test_serialized_design_validation_rejects_candidate_and_fingerprint_drift() -> None:
    payload = tape_depth_candidate_design(
        "conservative",
        sampled_count=6,
        seed=7,
    ).asdict()

    validated = validate_experiment_design_payload(payload)
    assert validated["design_sha256"] == payload["design_sha256"]
    assert validated["trial_burden"] == 9

    changed_candidate = copy.deepcopy(payload)
    changed_candidate["candidates"][0]["parameters"]["horizon_seconds"] = 30  # type: ignore[index]
    with pytest.raises(ValueError, match="identity"):
        validate_experiment_design_payload(changed_candidate)

    changed_hash = copy.deepcopy(payload)
    changed_hash["design_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        validate_experiment_design_payload(changed_hash)


def test_design_rejects_invalid_or_duplicate_anchors() -> None:
    domains = (ChoiceDomain("mode", ("a", "b")), FloatDomain("x", 0.1, 1.0))
    duplicate = {"mode": "a", "x": 0.5}
    with pytest.raises(ValueError, match="duplicate"):
        generate_latin_hypercube_design(
            domains,
            sampled_count=2,
            seed=1,
            anchors=(duplicate, duplicate),
        )
    with pytest.raises(ValueError, match="outside"):
        generate_latin_hypercube_design(
            domains,
            sampled_count=2,
            seed=1,
            anchors=({"mode": "invalid", "x": 0.5},),
        )


def test_calendar_window_plan_is_precommitted_spread_and_non_overlapping() -> None:
    first = date(2020, 1, 1)
    periods = [(first + timedelta(days=offset)).isoformat() for offset in range(120)]
    mapping = {symbol: periods for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")}

    windows = plan_calendar_windows(
        mapping,
        window_days=7,
        windows_per_symbol=3,
        seed=11,
    )
    repeated = plan_calendar_windows(
        mapping,
        window_days=7,
        windows_per_symbol=3,
        seed=11,
    )

    assert windows == repeated
    assert len(windows) == 9
    for symbol in mapping:
        selected = [item for item in windows if item.symbol == symbol]
        assert len(selected) == 3
        ranges = [
            (
                date.fromisoformat(item.first_period),
                date.fromisoformat(item.last_period),
            )
            for item in selected
        ]
        assert all((end - start).days == 6 for start, end in ranges)
        assert all(
            left[1] < right[0] or right[1] < left[0]
            for index, left in enumerate(ranges)
            for right in ranges[index + 1 :]
        )


def _stage() -> FidelityStage:
    return FidelityStage(
        name="screen",
        role="viability_screen",
        keep_fraction=0.50,
        minimum_survivors=1,
        minimum_windows=2,
        minimum_symbols=2,
        minimum_closed_trades=10,
        minimum_trades_per_day=1.0,
        minimum_window_pass_rate=0.50,
        minimum_expectancy_bps=0.0,
        minimum_profit_factor=1.0,
        maximum_drawdown_bps=100.0,
        maximum_consecutive_losses=5,
        maximum_side_share=0.80,
    )


def _performance(
    candidate_id: str,
    symbol: str,
    *,
    wins: float,
    losses: float,
    liquidation_events: int = 0,
    cost_coverage: float = 1.0,
    source_verified: bool = True,
) -> WindowPerformance:
    return WindowPerformance(
        candidate_id=candidate_id,
        stage_name="screen",
        window_id=f"{candidate_id}-{symbol}",
        symbol=symbol,
        calendar_days=2.0,
        closed_trades=10,
        winning_net_bps=wins,
        losing_net_bps_abs=losses,
        max_drawdown_bps=20.0,
        max_consecutive_losses=3,
        long_trades=5,
        short_trades=5,
        liquidation_events=liquidation_events,
        cost_model_coverage_ratio=cost_coverage,
        source_verified=source_verified,
    )


def test_successive_halving_applies_hard_risk_gates_and_counts_every_trial() -> None:
    design = generate_latin_hypercube_design(
        (FloatDomain("x", 0.0, 1.0),),
        sampled_count=6,
        seed=3,
    )
    candidates = design.candidates
    results: list[WindowPerformance] = []
    for index, candidate in enumerate(candidates):
        options = {"wins": 30.0 - index, "losses": 10.0}
        if index == 3:
            options["liquidation_events"] = 1
        if index == 4:
            options["cost_coverage"] = 0.99
        if index == 5:
            options.update({"wins": 5.0, "losses": 20.0})
        for symbol in ("BTCUSDT", "ETHUSDT"):
            results.append(_performance(candidate.candidate_id, symbol, **options))

    decision = apply_successive_halving_stage(
        _stage(),
        candidates,
        results,
        prior_trial_burden=7,
    )
    repeated = apply_successive_halving_stage(
        _stage(),
        candidates,
        results,
        prior_trial_burden=7,
    )

    assert decision == repeated
    assert decision.candidate_count == 6
    assert decision.window_evaluation_count == 12
    assert decision.cumulative_trial_burden == 13
    assert len(decision.survivor_ids) == 2
    assert decision.survivor_ids == tuple(
        candidate.candidate_id for candidate in candidates[:2]
    )
    assert decision.authorization == "research_only_no_trading_authority"
    assert decision.terminal_holdout_consumed is False
    assert "liquidation_events>0" in decision.diagnostics[3].reasons
    assert "execution_cost_coverage_incomplete" in decision.diagnostics[4].reasons
    assert decision.diagnostics[5].passed_hard_gates is False


def test_stage_rejects_duplicate_results_and_terminal_role() -> None:
    design = generate_latin_hypercube_design(
        (FloatDomain("x", 0.0, 1.0),),
        sampled_count=1,
        seed=3,
    )
    result = _performance(design.candidates[0].candidate_id, "BTCUSDT", wins=20.0, losses=5.0)
    with pytest.raises(ValueError, match="duplicate"):
        apply_successive_halving_stage(_stage(), design.candidates, [result, result])
    with pytest.raises(ValueError, match="terminal"):
        FidelityStage(
            **{**_stage().__dict__, "role": "terminal"},
        )


def test_default_fidelity_stages_preserve_risk_order_without_authority() -> None:
    conservative = default_fidelity_stages("conservative")
    regular = default_fidelity_stages("regular")
    aggressive = default_fidelity_stages("aggressive")

    assert [stage.role for stage in conservative] == [
        "viability_screen",
        "selection",
        "prequential",
        "full_validation",
    ]
    assert all(
        conservative[index].maximum_drawdown_bps
        < regular[index].maximum_drawdown_bps
        < aggressive[index].maximum_drawdown_bps
        for index in range(4)
    )
