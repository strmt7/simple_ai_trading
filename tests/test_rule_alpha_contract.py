from __future__ import annotations

import math

import pytest

from simple_ai_trading.model import _rule_alpha_score_from_values


_EXPECTED_FAMILY_SCORES = {
    "adaptive_tape_regime": "0x1.8316586c59224p-2",
    "compression_breakout_scalp": "0x1.fd93e8dff5975p-2",
    "directional_regime_rider": "0x1.d728064d0c56ap-2",
    "flow_consensus_breakout": "0x1.3160d038d3990p-1",
    "flow_reversion": "-0x1.489808b09fe34p-4",
    "higher_timeframe_alignment": "0x1.fedb445023b2dp-3",
    "liquidity_absorption_reversal": "-0x1.d92737c697e6bp-2",
    "liquidity_sweep_reversal": "-0x1.16ea785069b24p-2",
    "mean_reversion_vwap": "-0x1.6d61293b79e0fp-3",
    "micro_flow_scalp": "0x1.5240be2199ee8p-1",
    "order_flow_momentum": "0x1.25e19e47b9feep-1",
    "trend_pullback": "0x1.a761461eec70ep-6",
    "unknown_defaults_to_momentum_breakout": "0x1.42801257901ccp-3",
    "volatility_breakout": "0x1.23fee1b67ad64p-2",
    "volume_flow_proxy": "0x1.ba4b672cc0cd3p-3",
    "volume_synchronized_flow": "0x1.0a80be258ae65p-2",
    "vwap_snapback_scalp": "-0x1.00b484e1236fap-2",
}


def _rich_rule_alpha_fixture() -> tuple[list[float], dict[str, int | float]]:
    base = [
        0.0015,
        0.0012,
        0.0009,
        0.0005,
        0.0,
        0.56,
        0.0007,
        0.0002,
        0.0002,
        1.2,
        0.0007,
        0.0001,
        0.2,
    ]
    order_flow = [
        0.72,
        0.54,
        0.45,
        0.12,
        0.10,
        0.05,
        0.0,
        0.25,
        0.24,
        0.30,
        0.28,
        0.30,
        0.05,
    ]
    higher_timeframe = [0.006, 0.003, 0.0002, 0.006, -0.002, 0.005, 0.36, 0.24]
    trade_tape = [0.76, 0.62, 0.55, 0.36, 0.34, 0.22, 0.08, 0.02, 0.10, 0.0, 0.24, 0.30]
    values = (
        base
        + order_flow
        + order_flow
        + higher_timeframe
        + higher_timeframe
        + trade_tape
        + trade_tape
    )
    params: dict[str, int | float] = {
        "deadband": 0.0,
        "order_flow_start": 13,
        "order_flow_width": 13,
        "order_flow_window_count": 2,
        "higher_timeframe_start": 39,
        "higher_timeframe_width": 8,
        "higher_timeframe_window_count": 2,
        "trade_tape_start": 55,
        "trade_tape_width": 12,
        "trade_tape_window_count": 2,
    }
    return values, params


@pytest.mark.parametrize(
    ("family", "expected_hex"), sorted(_EXPECTED_FAMILY_SCORES.items())
)
def test_rule_alpha_family_scores_are_stable(family: str, expected_hex: str) -> None:
    values, params = _rich_rule_alpha_fixture()

    score = _rule_alpha_score_from_values(values, {**params, "family": family})

    assert score == pytest.approx(float.fromhex(expected_hex), rel=1e-14, abs=1e-15)


def test_rule_alpha_higher_timeframe_family_abstains_without_context() -> None:
    values, _params = _rich_rule_alpha_fixture()

    score = _rule_alpha_score_from_values(
        values[:13],
        {"family": "higher_timeframe_alignment", "deadband": 0.0},
    )

    assert score == 0.0


def test_rule_alpha_short_feature_vector_is_zero_padded_and_abstains() -> None:
    assert _rule_alpha_score_from_values([], None) == 0.0


def test_empirical_rule_alpha_preserves_two_feature_and_non_finite_contracts() -> None:
    values, _params = _rich_rule_alpha_fixture()
    empirical = {
        "family": "empirical_feature_edge",
        "feature_index": 15,
        "feature_threshold": 0.0,
        "feature_scale": 1.0,
        "tail_direction": 1.0,
        "second_feature_index": 16,
        "second_feature_threshold": 0.0,
        "second_feature_scale": 1.0,
        "second_tail_direction": 1.0,
        "trade_side": 1.0,
        "edge_confidence": 0.75,
        "edge_slope": 1.0,
        "deadband": 0.0,
    }

    score = _rule_alpha_score_from_values(values, empirical)
    non_finite = _rule_alpha_score_from_values(
        [math.nan, math.inf],
        {
            **empirical,
            "feature_index": 0,
            "feature_threshold": 0.2,
            "second_feature_index": 1,
            "second_feature_threshold": 0.3,
        },
    )
    short_score = _rule_alpha_score_from_values(
        [1.0],
        {
            "family": "empirical_feature_edge",
            "feature_index": 0,
            "feature_threshold": 0.0,
            "feature_scale": 1.0,
            "tail_direction": 1.0,
            "trade_side": -1.0,
            "edge_confidence": 1.0,
            "edge_slope": 1.0,
            "deadband": 0.0,
        },
    )

    assert score == pytest.approx(
        float.fromhex("0x1.6ee173017531fp-4"), rel=1e-14, abs=1e-15
    )
    assert non_finite == 0.0
    assert short_score < 0.0
