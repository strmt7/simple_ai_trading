from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading import derivatives_ai_veto as subject


FEATURE_NAMES = (
    "target_return_5m_bps",
    "target_return_15m_bps",
    "target_return_60m_bps",
    "target_realized_volatility_60m_bps",
    "target_realized_volatility_240m_bps",
    "target_intrabar_range_bps",
    "target_path_efficiency_60m",
    "target_quote_volume_vs_60m_mean",
    "target_trade_count_vs_60m_mean",
    "target_signed_taker_flow_15m",
    "target_signed_taker_flow_60m",
    "target_return_zscore_240m",
    "target_beta_residual_return_60m_bps",
    "cross_asset_return_dispersion_15m_bps",
    "cross_asset_taker_flow_mean",
    "cross_asset_taker_flow_agreement",
    "target_to_btc_volatility_ratio_60m",
    "target_same_minute_of_week_liquidity_ratio",
    "target_premium_close_bps",
    "target_premium_zscore_240m",
    "target_premium_age_minutes",
    "target_premium_observed_fraction_240m",
    "cross_asset_premium_dispersion_bps",
    "target_last_settled_funding_rate_bps",
    "target_funding_interval_hours",
    "target_minutes_since_funding",
    "target_settled_funding_sum_24h_bps",
    "target_settled_funding_sum_168h_bps",
    "target_funding_event_zscore_30",
    "cross_asset_funding_dispersion_bps",
    "weekend_flag",
)


def _dataset_and_candidate() -> tuple[SimpleNamespace, SimpleNamespace, int]:
    end = datetime(2026, 4, 1, tzinfo=UTC)
    first = end - timedelta(days=89)
    decision_time_ms = np.asarray(
        [
            int((first - timedelta(days=10 - index)).timestamp() * 1000)
            for index in range(3)
        ]
        + [int((first + timedelta(days=index)).timestamp() * 1000) for index in (1, 2)],
        dtype=np.int64,
    )
    features = np.arange(
        len(decision_time_ms) * len(FEATURE_NAMES), dtype=np.float64
    ).reshape(len(decision_time_ms), len(FEATURE_NAMES))
    probabilities = np.asarray(
        [
            [0.10, 0.05, 0.85],
            [0.15, 0.05, 0.80],
            [0.82, 0.05, 0.13],
            [0.08, 0.05, 0.87],
            [0.88, 0.05, 0.07],
        ],
        dtype=np.float64,
    )
    long_net = np.asarray([1.0, 2.0, -1.0, 3.5, -2.0], dtype=np.float64)
    short_net = np.asarray([-1.0, -2.0, 2.5, -3.5, 1.25], dtype=np.float64)
    horizon = 60
    dataset = SimpleNamespace(
        decision_time_ms=decision_time_ms,
        symbol_index=np.zeros(len(decision_time_ms), dtype=np.int64),
        feature_names=FEATURE_NAMES,
        features=features,
        role_masks={
            horizon: {"calibration": np.asarray([True, True, True, False, False])}
        },
        long_net_utility_bps={horizon: long_net},
        short_net_utility_bps={horizon: short_net},
    )
    viability = SimpleNamespace(
        selected_indices=np.asarray([3, 4], dtype=np.int64),
        net_return_bps=np.asarray([3.5, 1.25], dtype=np.float64),
    )
    candidate = SimpleNamespace(
        candidate_id="regularized-causal-hurdle",
        horizon_minutes=horizon,
        architecture="logistic_hurdle",
        feature_set="derivatives_causal",
        probabilities=probabilities,
        maximum_action_probability=0.75,
        direction_probability_margin=0.5,
        viability=viability,
    )
    return dataset, candidate, int(end.timestamp() * 1000)


def test_derivatives_cases_are_causal_deterministic_and_outcome_hidden(
    monkeypatch,
) -> None:
    dataset, candidate, end_exclusive_ms = _dataset_and_candidate()
    monkeypatch.setattr(
        subject,
        "role_by_name",
        lambda role: (
            SimpleNamespace(
                end="2026-04-01T00:00:00",
                end_exclusive_ms=end_exclusive_ms,
            )
            if role == "viability"
            else pytest.fail(f"unexpected role: {role}")
        ),
    )

    cases = subject.build_derivatives_ai_cases(dataset, (candidate,))

    assert len(cases) == 2
    assert [case.direction for case in cases] == ["long", "short"]
    assert [case.relative_day_index for case in cases] == [1, 2]
    assert cases[0].prompt_payload["past_only_nearest_regimes"]["samples"] == 2
    assert cases[1].prompt_payload["past_only_nearest_regimes"]["samples"] == 1
    assert cases[0].prompt_payload["risk_state"]["prior_completed_cases"] == 0
    assert cases[1].prompt_payload["risk_state"]["prior_completed_cases"] == 1
    assert "outcome_net_bps" not in cases[0].identity_payload()
    assert cases[0].evidence_payload()["outcome_net_bps"] == 3.5

    case_set_hash = subject.derivatives_case_set_sha256(cases)
    changed_outcome = (replace(cases[0], outcome_net_bps=-999.0), cases[1])
    assert subject.derivatives_case_set_sha256(changed_outcome) == case_set_hash
    assert subject.build_derivatives_ai_cases(dataset, ()) == ()

    prompt = subject._prompt(cases[0])
    assert "CASE=" in prompt
    assert "outcome_net_bps" not in prompt
    assert "cannot create a trade" in prompt


def test_ai_case_helpers_fail_closed_and_delegate_frozen_prompt(monkeypatch) -> None:
    dataset, candidate, end_exclusive_ms = _dataset_and_candidate()
    monkeypatch.setattr(
        subject,
        "role_by_name",
        lambda _role: SimpleNamespace(
            end="2026-04-01T00:00:00",
            end_exclusive_ms=end_exclusive_ms,
        ),
    )
    case = subject.build_derivatives_ai_cases(dataset, (candidate,))[0]

    routed, direction = subject._routed_action(
        candidate.probabilities,
        probability_threshold=0.75,
        margin_threshold=0.5,
    )
    assert routed.tolist() == [True, True, True, True, True]
    assert direction.tolist() == [1, 1, -1, 1, -1]

    with pytest.raises(ValueError, match="feature contract is incomplete"):
        subject._feature_indices(FEATURE_NAMES[:-1])

    captured: dict[str, object] = {}
    expected = object()

    def fake_benchmark(cases, **kwargs):
        captured["cases"] = cases
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(subject, "benchmark_ai_veto_model", fake_benchmark)
    result = subject.benchmark_derivatives_ai_model(
        (case,),
        model="local-finance-model",
        base_url="http://localhost:11434",
        timeout_seconds=7.0,
    )

    assert result is expected
    assert captured["cases"] == (case,)
    assert captured["model"] == "local-finance-model"
    assert captured["timeout_seconds"] == 7.0
    assert captured["seed"] == 3801
    delegated_prompt = captured["prompt_builder"](case)
    assert delegated_prompt == subject._prompt(case)
