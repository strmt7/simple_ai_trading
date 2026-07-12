from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import simple_ai_trading.frozen_action_confirmation as confirmation
from simple_ai_trading.microstructure_action_policy import BarrierActionTrace
from simple_ai_trading.microstructure_model import TradingMetrics


def _profile() -> dict[str, object]:
    return {
        "profile": "conservative",
        "epistemic_penalty": 1.5,
        "minimum_profitable_probability": 0.65,
        "minimum_member_agreement": 1.0,
        "maximum_epistemic_std_bps": 6.0,
        "minimum_lower_bound_bps": -35.0,
    }


def _trace(*, scenario: str, threshold: float) -> BarrierActionTrace:
    metrics = TradingMetrics(
        trades=int(threshold * 10),
        total_net_bps=threshold * 20.0,
        mean_net_bps=threshold * 2.0,
        median_net_bps=threshold,
        win_rate=0.6,
        profit_factor=1.2,
        max_drawdown_bps=threshold * 2.0,
        worst_trade_bps=-5.0,
        best_trade_bps=10.0,
        long_trades=int(threshold * 5),
        short_trades=int(threshold * 5),
        active_days=5,
        trades_per_active_day=threshold * 2.0,
    )
    return BarrierActionTrace(
        scenario=scenario,
        metrics=metrics,
        net_bps=(threshold,),
        sides=(1,),
        timestamps_ms=(1,),
        exit_times_ms=(2,),
        source_endpoint_indexes=(3,),
    )


def _patch_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        confirmation,
        "derive_action_scores",
        lambda _prediction, _spec: SimpleNamespace(
            eligible=np.asarray([True, False, True], dtype=bool)
        ),
    )

    def simulate(_dataset, _targets, _score, *, scenario, strength_threshold_bps):
        return _trace(scenario=scenario, threshold=float(strength_threshold_bps))

    monkeypatch.setattr(confirmation, "simulate_barrier_action_trace", simulate)
    monkeypatch.setattr(
        confirmation,
        "barrier_trace_gate_reasons",
        lambda trace, **_kwargs: (
            ("maximum_drawdown_bps",)
            if trace.metrics.max_drawdown_bps > 5.0
            else ()
        ),
    )


def test_frozen_confirmation_selects_only_passing_precommitted_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_evaluation(monkeypatch)

    result = confirmation.evaluate_frozen_profile_candidates(
        object(),
        object(),
        object(),
        profile=_profile(),
        candidates=(
            {"quantile": 0.5, "threshold_bps": 1.0},
            {"quantile": 0.7, "threshold_bps": 2.0},
            {"quantile": 0.9, "threshold_bps": 3.0},
        ),
        gates={"minimum_trades": 1},
        expected_days=(1, 2, 3),
        drawdown_penalty=0.25,
        stage="confirmation",
    )

    assert result["eligible_rows"] == 2
    assert result["candidate_count"] == 3
    assert result["passing_candidate_count"] == 2
    assert result["selected_quantile"] == pytest.approx(0.7)
    assert result["selected_threshold_bps"] == pytest.approx(2.0)
    assert result["candidates"][2]["gate_reasons"] == ["maximum_drawdown_bps"]
    assert result["trading_authority"] is False
    assert result["profitability_claim"] is False


def test_later_stage_evaluates_exactly_one_prior_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_evaluation(monkeypatch)

    result = confirmation.evaluate_frozen_profile_stage(
        object(),
        object(),
        object(),
        profile=_profile(),
        quantile=0.7,
        threshold_bps=2.0,
        gates={"minimum_trades": 1},
        expected_days=(1, 2),
        drawdown_penalty=0.25,
        stage="policy",
    )

    assert result["candidate_count"] == 1
    assert result["passed"] is True
    assert result["selected_threshold_bps"] == pytest.approx(2.0)
    assert result["threshold_source"] == "prior_stage_fixed"


@pytest.mark.parametrize(
    ("candidates", "expected_days", "message"),
    [
        (
            (
                {"quantile": 0.5, "threshold_bps": 1.0},
                {"quantile": 0.5, "threshold_bps": 2.0},
            ),
            (1, 2),
            "threshold candidate",
        ),
        (
            ({"quantile": 0.5, "threshold_bps": 1.0},),
            (2, 1),
            "expected days",
        ),
    ],
)
def test_frozen_confirmation_rejects_mutable_or_ambiguous_contracts(
    monkeypatch: pytest.MonkeyPatch,
    candidates: tuple[dict[str, float], ...],
    expected_days: tuple[int, ...],
    message: str,
) -> None:
    _patch_evaluation(monkeypatch)

    with pytest.raises(ValueError, match=message):
        confirmation.evaluate_frozen_profile_candidates(
            object(),
            object(),
            object(),
            profile=_profile(),
            candidates=candidates,
            gates={"minimum_trades": 1},
            expected_days=expected_days,
            drawdown_penalty=0.25,
            stage="confirmation",
        )
