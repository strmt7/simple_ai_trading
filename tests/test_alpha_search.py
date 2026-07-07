from __future__ import annotations

from types import SimpleNamespace

from simple_ai_trading.alpha_search import (
    DEFAULT_RULE_ALPHA_MAX_CANDIDATES,
    RuleAlphaCandidate,
    RuleAlphaCandidateResult,
    _diagnostic_rank_key,
    model_for_rule_alpha,
    optimize_rule_alpha_model_zoo,
    rule_alpha_event_study,
    rule_alpha_stop_loss_floor_pct,
    rule_alpha_feature_params,
    rule_alpha_take_profit_floor_pct,
    rule_alpha_candidates,
    summarize_rule_alpha_candidate_distribution,
    summarize_rule_alpha_trade_path,
)
from simple_ai_trading.advanced_model import default_config_for
from simple_ai_trading.backtest import BacktestResult
from simple_ai_trading.features import ModelRow
from simple_ai_trading.features import FEATURE_NAMES
from simple_ai_trading.model import load_model, serialize_model
from simple_ai_trading.types import StrategyConfig


def _rows(*, rising: bool = True, count: int = 36) -> list[ModelRow]:
    rows: list[ModelRow] = []
    for index in range(count):
        close = 100.0 + index * 0.35 if rising else 100.0 - index * 0.35
        direction = 1.0 if rising else -1.0
        rows.append(ModelRow(
            timestamp=index * 60_000,
            close=close,
            features=(
                0.0030 * direction,
                0.0040 * direction,
                0.0050 * direction,
                0.0060 * direction,
                0.0,
                0.55,
                0.0040 * direction,
                0.0020,
                0.0020,
                1.5,
                0.0040 * direction,
                0.0,
                0.3,
            ),
            label=1,
            volume=10_000.0,
        ))
    return rows


def test_rule_alpha_candidates_are_bounded_and_diverse() -> None:
    candidates = rule_alpha_candidates("conservative", max_candidates=15)

    assert len(candidates) == 15
    assert {candidate.family for candidate in candidates} >= {
        "momentum_breakout",
        "flow_consensus_breakout",
        "liquidity_absorption_reversal",
        "micro_flow_scalp",
        "vwap_snapback_scalp",
        "liquidity_sweep_reversal",
        "volume_synchronized_flow",
    }
    assert {candidate.name.split(":")[1] for candidate in candidates} >= {
        "scalp_3s",
        "scalp_8s",
        "scalp_20s",
        "micro",
        "balanced",
        "guarded",
        "held_30s",
        "held_90s",
        "held_180s",
    }
    assert all(0.5 <= candidate.threshold <= 0.95 for candidate in candidates)


def test_rule_alpha_default_search_covers_base_family_profile_matrix() -> None:
    candidates = rule_alpha_candidates("conservative")
    families = {candidate.family for candidate in candidates}
    profiles = {candidate.name.split(":")[1] for candidate in candidates}
    base_pairs = {
        (candidate.family, candidate.name.split(":")[1])
        for candidate in candidates
        if candidate.threshold == 0.54 and candidate.sensitivity == 6.0 and candidate.deadband == 0.02
    }

    assert len(candidates) == DEFAULT_RULE_ALPHA_MAX_CANDIDATES
    assert families == {
        "momentum_breakout",
        "mean_reversion_vwap",
        "trend_pullback",
        "volatility_breakout",
        "volume_flow_proxy",
        "order_flow_momentum",
        "flow_reversion",
        "flow_consensus_breakout",
        "liquidity_absorption_reversal",
        "micro_flow_scalp",
        "vwap_snapback_scalp",
        "liquidity_sweep_reversal",
        "compression_breakout_scalp",
        "volume_synchronized_flow",
        "adaptive_tape_regime",
    }
    assert profiles == {
        "scalp_3s",
        "scalp_8s",
        "scalp_20s",
        "micro",
        "balanced",
        "guarded",
        "held_30s",
        "held_90s",
        "held_180s",
    }
    assert len(base_pairs) == len(families) * len(profiles)


def test_rule_alpha_model_roundtrips_and_affects_probability(tmp_path) -> None:
    rows = _rows()
    candidate = RuleAlphaCandidate(
        name="momentum_breakout:unit",
        family="momentum_breakout",
        threshold=0.58,
        sensitivity=8.0,
        deadband=0.02,
        stop_loss_multiplier=0.14,
        take_profit_multiplier=0.10,
        cooldown_multiplier=0.0,
        min_position_hold_bars=2,
        flat_signal_exit_grace_bars=0,
    )
    model = model_for_rule_alpha(rows, candidate, StrategyConfig(), market_type="futures")

    long_probability = model.predict_proba(rows[0].features)
    short_probability = model.predict_proba(_rows(rising=False)[0].features)

    assert long_probability > 0.5
    assert short_probability < 0.5
    assert model.hybrid_experts[0].params["family"] == "momentum_breakout"

    path = tmp_path / "model.json"
    serialize_model(model, path)
    loaded = load_model(path, expected_feature_dim=13)

    assert loaded.rule_alpha_profile == "momentum_breakout:unit"
    assert loaded.hybrid_experts[0].kind == "rule_alpha"
    assert loaded.hybrid_experts[0].params["deadband"] == 0.02
    assert loaded.predict_proba(rows[0].features) == long_probability


def test_rule_alpha_candidate_summary_roundtrips_with_model(tmp_path) -> None:
    rows = _rows()
    candidate = RuleAlphaCandidate(
        name="micro_flow_scalp:unit",
        family="micro_flow_scalp",
        threshold=0.54,
        sensitivity=8.0,
        deadband=0.02,
        stop_loss_multiplier=0.08,
        take_profit_multiplier=0.07,
        cooldown_multiplier=0.0,
        min_position_hold_bars=2,
        flat_signal_exit_grace_bars=0,
    )
    model = model_for_rule_alpha(rows, candidate, StrategyConfig(), market_type="futures")
    model.rule_alpha_candidate_summary = {
        "evaluated_candidates": 4,
        "active_candidates": 3,
        "profitable_candidates": 1,
        "most_active_candidate": "micro_flow_scalp:unit",
    }
    path = tmp_path / "summary-model.json"

    serialize_model(model, path)
    loaded = load_model(path, expected_feature_dim=13)

    assert loaded.rule_alpha_candidate_summary["evaluated_candidates"] == 4
    assert loaded.rule_alpha_candidate_summary["most_active_candidate"] == "micro_flow_scalp:unit"


def test_rule_alpha_strategy_overrides_respect_execution_cost_floor() -> None:
    rows = _rows()
    strategy = StrategyConfig(
        stop_loss_pct=0.010,
        take_profit_pct=0.022,
        taker_fee_bps=1.0,
        slippage_bps=5.0,
        max_spread_bps=5.0,
        latency_buffer_ms=750,
        testnet_liquidity_haircut=0.50,
    )
    candidate = RuleAlphaCandidate(
        name="micro_flow_scalp:scalp_3s",
        family="micro_flow_scalp",
        threshold=0.54,
        sensitivity=8.0,
        deadband=0.02,
        stop_loss_multiplier=0.06,
        take_profit_multiplier=0.05,
        cooldown_multiplier=0.0,
        min_position_hold_bars=1,
        flat_signal_exit_grace_bars=0,
    )

    model = model_for_rule_alpha(rows, candidate, strategy, market_type="futures")

    assert model.strategy_overrides["stop_loss_pct"] >= rule_alpha_stop_loss_floor_pct(strategy)
    assert model.strategy_overrides["take_profit_pct"] >= rule_alpha_take_profit_floor_pct(strategy)
    assert model.strategy_overrides["take_profit_pct"] > model.strategy_overrides["stop_loss_pct"]
    assert model.strategy_overrides["take_profit_pct"] > strategy.take_profit_pct * candidate.take_profit_multiplier


def test_rule_alpha_can_use_advanced_order_flow_features(tmp_path) -> None:
    candidate = RuleAlphaCandidate(
        name="order_flow_momentum:unit",
        family="order_flow_momentum",
        threshold=0.54,
        sensitivity=8.0,
        deadband=0.01,
        stop_loss_multiplier=0.18,
        take_profit_multiplier=0.16,
        cooldown_multiplier=0.0,
        min_position_hold_bars=30,
        flat_signal_exit_grace_bars=10,
    )
    cfg = default_config_for("conservative", FEATURE_NAMES)
    feature_params = rule_alpha_feature_params(cfg)
    start = int(feature_params["order_flow_start"])
    width = int(feature_params["order_flow_width"])
    assert width == 13
    feature_count = start + (3 * width)
    features = [0.0] * feature_count
    features[0] = 0.001
    features[1] = 0.001
    for group_start in range(start, start + (3 * width), width):
        features[group_start + 0] = 0.70
        features[group_start + 1] = 0.80
        features[group_start + 2] = 0.75
        features[group_start + 3] = 0.45
        features[group_start + 4] = 0.40
        features[group_start + 7] = 0.35
        features[group_start + 8] = 0.30
        features[group_start + 9] = 0.65
        features[group_start + 10] = 0.45
        features[group_start + 11] = 0.30
        features[group_start + 12] = 0.25
    row = ModelRow(timestamp=0, close=100.0, features=tuple(features), label=1, volume=1000.0)
    model = model_for_rule_alpha(
        [row],
        candidate,
        StrategyConfig(),
        market_type="futures",
        feature_params=feature_params,
    )

    assert model.hybrid_experts[0].feature_count == feature_count
    assert model.hybrid_experts[0].params["order_flow_start"] == start
    assert model.predict_proba(row.features) > 0.5

    model.rule_alpha_best_win_rate = 0.5
    model.rule_alpha_best_exit_reason_counts = {"take_profit_close": 2}
    path = tmp_path / "flow-model.json"
    serialize_model(model, path)
    loaded = load_model(path, expected_feature_dim=feature_count)

    assert loaded.hybrid_experts[0].params["order_flow_window_count"] == 3
    assert loaded.rule_alpha_best_exit_reason_counts == {"take_profit_close": 2}
    assert loaded.predict_proba(row.features) == model.predict_proba(row.features)


def test_rule_alpha_new_flow_families_use_expanded_order_flow_state() -> None:
    cfg = default_config_for("conservative", FEATURE_NAMES)
    feature_params = rule_alpha_feature_params(cfg)
    start = int(feature_params["order_flow_start"])
    width = int(feature_params["order_flow_width"])
    features = [0.0] * (start + (3 * width))
    features[0] = 0.0012
    features[1] = 0.0010
    features[2] = 0.0008
    features[10] = 0.0010
    for group_start in range(start, start + (3 * width), width):
        features[group_start + 0] = 0.72
        features[group_start + 1] = 0.70
        features[group_start + 2] = 0.68
        features[group_start + 7] = 0.45
        features[group_start + 8] = 0.32
        features[group_start + 9] = 0.58
        features[group_start + 10] = 0.40
        features[group_start + 11] = 0.35
        features[group_start + 12] = 0.20
    row = ModelRow(timestamp=0, close=100.0, features=tuple(features), label=1, volume=1000.0)
    breakout = model_for_rule_alpha(
        [row],
        RuleAlphaCandidate(
            name="flow_consensus_breakout:unit",
            family="flow_consensus_breakout",
            threshold=0.54,
            sensitivity=8.0,
            deadband=0.01,
            stop_loss_multiplier=0.18,
            take_profit_multiplier=0.16,
            cooldown_multiplier=0.0,
            min_position_hold_bars=30,
            flat_signal_exit_grace_bars=10,
        ),
        StrategyConfig(),
        market_type="futures",
        feature_params=feature_params,
    )
    absorption = model_for_rule_alpha(
        [row],
        RuleAlphaCandidate(
            name="liquidity_absorption_reversal:unit",
            family="liquidity_absorption_reversal",
            threshold=0.54,
            sensitivity=8.0,
            deadband=0.01,
            stop_loss_multiplier=0.18,
            take_profit_multiplier=0.16,
            cooldown_multiplier=0.0,
            min_position_hold_bars=30,
            flat_signal_exit_grace_bars=10,
        ),
        StrategyConfig(),
        market_type="futures",
        feature_params=feature_params,
    )

    assert breakout.predict_proba(row.features) > 0.5
    assert absorption.predict_proba(row.features) < breakout.predict_proba(row.features)


def test_rule_alpha_scalp_families_use_tape_and_price_state() -> None:
    cfg = default_config_for("conservative", FEATURE_NAMES)
    feature_params = rule_alpha_feature_params(cfg)
    start = int(feature_params["order_flow_start"])
    width = int(feature_params["order_flow_width"])
    features = [0.0] * (start + (3 * width))
    features[0] = 0.0018
    features[1] = 0.0014
    features[2] = 0.0008
    features[3] = 0.0005
    features[5] = 0.58
    features[6] = 0.0008
    features[7] = 0.0002
    features[8] = 0.0002
    features[9] = 1.2
    features[10] = 0.0010
    features[11] = 0.0004
    for group_start in range(start, start + (3 * width), width):
        features[group_start + 0] = 0.74
        features[group_start + 1] = 0.66
        features[group_start + 2] = 0.64
        features[group_start + 3] = 0.42
        features[group_start + 4] = 0.35
        features[group_start + 6] = 0.00
        features[group_start + 7] = 0.42
        features[group_start + 8] = 0.30
        features[group_start + 9] = 0.38
        features[group_start + 10] = 0.44
        features[group_start + 11] = 0.35
        features[group_start + 12] = 0.16
    long_row = ModelRow(timestamp=0, close=100.0, features=tuple(features), label=1, volume=1000.0)
    short_features = list(features)
    for index in (0, 1, 2, 3, 6, 10, 11):
        short_features[index] = -short_features[index]
    for group_start in range(start, start + (3 * width), width):
        for offset in (1, 2, 7, 8, 10, 11, 12):
            short_features[group_start + offset] = -short_features[group_start + offset]
        short_features[group_start + 0] = 0.26
    short_row = ModelRow(timestamp=1, close=99.0, features=tuple(short_features), label=0, volume=1000.0)

    candidate = RuleAlphaCandidate(
        name="micro_flow_scalp:unit",
        family="micro_flow_scalp",
        threshold=0.54,
        sensitivity=8.0,
        deadband=0.01,
        stop_loss_multiplier=0.08,
        take_profit_multiplier=0.07,
        cooldown_multiplier=0.0,
        min_position_hold_bars=2,
        flat_signal_exit_grace_bars=0,
    )
    model = model_for_rule_alpha(
        [long_row, short_row],
        candidate,
        StrategyConfig(),
        market_type="futures",
        feature_params=feature_params,
    )

    assert model.predict_proba(long_row.features) > 0.54
    assert model.predict_proba(short_row.features) < 0.46


def test_rule_alpha_volume_synchronized_flow_uses_flow_price_agreement() -> None:
    cfg = default_config_for("conservative", FEATURE_NAMES)
    feature_params = rule_alpha_feature_params(cfg)
    start = int(feature_params["order_flow_start"])
    width = int(feature_params["order_flow_width"])
    features = [0.0] * (start + (3 * width))
    features[0] = 0.0014
    features[1] = 0.0012
    features[2] = 0.0007
    features[6] = 0.0005
    features[9] = 1.3
    features[10] = 0.0008
    for group_start in range(start, start + (3 * width), width):
        features[group_start + 0] = 0.71
        features[group_start + 1] = 0.58
        features[group_start + 2] = 0.55
        features[group_start + 3] = 0.36
        features[group_start + 4] = 0.34
        features[group_start + 5] = 0.22
        features[group_start + 6] = 0.0
        features[group_start + 7] = 0.62
        features[group_start + 8] = 0.24
        features[group_start + 9] = 0.44
        features[group_start + 10] = 0.36
        features[group_start + 11] = 0.28
        features[group_start + 12] = 0.02
    synchronized = ModelRow(timestamp=0, close=100.0, features=tuple(features), label=1, volume=1000.0)
    noisy_features = list(features)
    for group_start in range(start, start + (3 * width), width):
        noisy_features[group_start + 6] = 0.85
        noisy_features[group_start + 7] = -0.60
        noisy_features[group_start + 12] = 0.80
    noisy = ModelRow(timestamp=1, close=100.0, features=tuple(noisy_features), label=0, volume=1000.0)
    candidate = RuleAlphaCandidate(
        name="volume_synchronized_flow:unit",
        family="volume_synchronized_flow",
        threshold=0.54,
        sensitivity=8.0,
        deadband=0.01,
        stop_loss_multiplier=0.18,
        take_profit_multiplier=0.16,
        cooldown_multiplier=0.0,
        min_position_hold_bars=4,
        flat_signal_exit_grace_bars=1,
    )

    model = model_for_rule_alpha(
        [synchronized, noisy],
        candidate,
        StrategyConfig(),
        market_type="futures",
        feature_params=feature_params,
    )

    assert model.predict_proba(synchronized.features) > 0.54
    assert model.predict_proba(noisy.features) < model.predict_proba(synchronized.features)


def test_summarize_rule_alpha_trade_path_counts_exits_and_sides() -> None:
    result = BacktestResult(
        starting_cash=1000.0,
        ending_cash=1001.0,
        realized_pnl=1.0,
        win_rate=0.5,
        trades=2,
        max_drawdown=0.01,
        closed_trades=2,
        gross_exposure=100.0,
        total_fees=0.1,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        profit_factor=1.5,
        average_trade_return=0.001,
        max_consecutive_losses=1,
        trade_log=(
            {"exit_reason": "take_profit_close", "side": 1, "bars_held": 30},
            {"exit_reason": "stop_loss_close", "side": -1, "bars_held": 12},
        ),
    )

    summary = summarize_rule_alpha_trade_path(result)

    assert summary["exit_reason_counts"] == {"stop_loss_close": 1, "take_profit_close": 1}
    assert summary["side_counts"] == {"long": 1, "short": 1}
    assert summary["average_bars_held"] == 21.0


def test_rule_alpha_event_study_records_forward_edge() -> None:
    rows = _rows(count=48)
    candidate = RuleAlphaCandidate(
        name="momentum_breakout:event",
        family="momentum_breakout",
        threshold=0.54,
        sensitivity=8.0,
        deadband=0.01,
        stop_loss_multiplier=0.18,
        take_profit_multiplier=0.16,
        cooldown_multiplier=0.0,
        min_position_hold_bars=3,
        flat_signal_exit_grace_bars=1,
    )
    strategy = StrategyConfig(taker_fee_bps=0.0, slippage_bps=0.0, max_spread_bps=0.0, latency_buffer_ms=0)
    model = model_for_rule_alpha(rows, candidate, strategy, market_type="futures")

    study = rule_alpha_event_study(rows, model, strategy, candidate, market_type="futures")

    assert study["horizon_bars"] == 3
    assert study["signal_count"] > 0
    assert study["long_signal_count"] > 0
    assert study["mean_edge_bps"] > 0.0
    assert study["net_mean_edge_bps"] > 0.0


def test_rejected_rule_alpha_diagnostic_prefers_less_negative_pnl_over_activity_score() -> None:
    candidate = RuleAlphaCandidate(
        name="unit",
        family="momentum_breakout",
        threshold=0.54,
        sensitivity=6.0,
        deadband=0.02,
        stop_loss_multiplier=0.18,
        take_profit_multiplier=0.16,
        cooldown_multiplier=0.0,
        min_position_hold_bars=2,
        flat_signal_exit_grace_bars=0,
    )
    better_loss = BacktestResult(
        starting_cash=1000.0,
        ending_cash=997.0,
        realized_pnl=-3.0,
        win_rate=0.0,
        trades=3,
        max_drawdown=0.003,
        closed_trades=3,
        gross_exposure=100.0,
        total_fees=0.3,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        edge_vs_buy_hold=-1.0,
        profit_factor=0.0,
        max_consecutive_losses=3,
    )
    worse_loss = BacktestResult(
        starting_cash=1000.0,
        ending_cash=990.0,
        realized_pnl=-10.0,
        win_rate=0.0,
        trades=12,
        max_drawdown=0.010,
        closed_trades=12,
        gross_exposure=100.0,
        total_fees=1.2,
        stopped_by_drawdown=False,
        max_exposure=100.0,
        trades_per_day_cap_hit=0,
        edge_vs_buy_hold=-8.0,
        profit_factor=0.0,
        max_consecutive_losses=12,
    )

    better = RuleAlphaCandidateResult(candidate, False, False, float("-inf"), 0.01, "reject", better_loss)
    worse = RuleAlphaCandidateResult(candidate, False, False, float("-inf"), 0.05, "reject", worse_loss)

    assert _diagnostic_rank_key(better) > _diagnostic_rank_key(worse)

    no_trade = BacktestResult(
        starting_cash=1000.0,
        ending_cash=1000.0,
        realized_pnl=0.0,
        win_rate=0.0,
        trades=0,
        max_drawdown=0.0,
        closed_trades=0,
        gross_exposure=0.0,
        total_fees=0.0,
        stopped_by_drawdown=False,
        max_exposure=0.0,
        trades_per_day_cap_hit=0,
        edge_vs_buy_hold=0.0,
    )
    inactive = RuleAlphaCandidateResult(candidate, False, False, float("-inf"), 0.50, "reject", no_trade)

    assert _diagnostic_rank_key(better) > _diagnostic_rank_key(inactive)


def test_rule_alpha_candidate_summary_tracks_active_and_profitable_candidates() -> None:
    candidate = RuleAlphaCandidate(
        name="micro_flow_scalp:scalp_3s:t0.54:s6.0:d0.02",
        family="micro_flow_scalp",
        threshold=0.54,
        sensitivity=6.0,
        deadband=0.02,
        stop_loss_multiplier=0.06,
        take_profit_multiplier=0.05,
        cooldown_multiplier=0.0,
        min_position_hold_bars=1,
        flat_signal_exit_grace_bars=0,
    )
    active_loss = RuleAlphaCandidateResult(
        candidate,
        False,
        False,
        float("-inf"),
        -0.1,
        "loss",
        BacktestResult(
            starting_cash=1000.0,
            ending_cash=996.0,
            realized_pnl=-4.0,
            win_rate=0.0,
            trades=8,
            max_drawdown=0.004,
            closed_trades=8,
            gross_exposure=100.0,
            total_fees=1.0,
            stopped_by_drawdown=False,
            max_exposure=100.0,
            trades_per_day_cap_hit=0,
            edge_vs_buy_hold=-3.0,
            profit_factor=0.0,
        ),
    )
    accepted_profit = RuleAlphaCandidateResult(
        RuleAlphaCandidate(
            name="vwap_snapback_scalp:scalp_8s:t0.54:s6.0:d0.02",
            family="vwap_snapback_scalp",
            threshold=0.54,
            sensitivity=6.0,
            deadband=0.02,
            stop_loss_multiplier=0.08,
            take_profit_multiplier=0.07,
            cooldown_multiplier=0.0,
            min_position_hold_bars=2,
            flat_signal_exit_grace_bars=0,
        ),
        False,
        True,
        0.2,
        0.2,
        None,
        BacktestResult(
            starting_cash=1000.0,
            ending_cash=1001.0,
            realized_pnl=1.0,
            win_rate=0.5,
            trades=2,
            max_drawdown=0.001,
            closed_trades=2,
            gross_exposure=100.0,
            total_fees=0.2,
            stopped_by_drawdown=False,
            max_exposure=100.0,
            trades_per_day_cap_hit=0,
            edge_vs_buy_hold=2.0,
            profit_factor=2.0,
        ),
        {"signal_count": 7, "net_mean_edge_bps": 4.0, "hit_rate": 0.71, "horizon_bars": 8},
    )

    summary = summarize_rule_alpha_candidate_distribution([active_loss, accepted_profit])

    assert summary["evaluated_candidates"] == 2
    assert summary["active_candidates"] == 2
    assert summary["profitable_candidates"] == 1
    assert summary["accepted_candidates"] == 1
    assert summary["event_candidates_with_signals"] == 1
    assert summary["event_positive_candidates"] == 1
    assert summary["event_best_candidate"] == "vwap_snapback_scalp:scalp_8s:t0.54:s6.0:d0.02"
    assert summary["event_best_net_edge_bps"] == 4.0
    assert summary["max_closed_trades"] == 8
    assert summary["most_active_candidate"] == "micro_flow_scalp:scalp_3s:t0.54:s6.0:d0.02"
    assert summary["best_pnl_candidate"] == "vwap_snapback_scalp:scalp_8s:t0.54:s6.0:d0.02"
    assert "micro_flow_scalp" in str(summary["families_with_trades"])


def test_rule_alpha_candidate_summary_does_not_invent_event_winner_without_signals() -> None:
    candidate = RuleAlphaCandidate(
        name="micro_flow_scalp:scalp_3s:t0.54:s6.0:d0.02",
        family="micro_flow_scalp",
        threshold=0.54,
        sensitivity=6.0,
        deadband=0.02,
        stop_loss_multiplier=0.06,
        take_profit_multiplier=0.05,
        cooldown_multiplier=0.0,
        min_position_hold_bars=1,
        flat_signal_exit_grace_bars=0,
    )
    result = RuleAlphaCandidateResult(
        candidate,
        False,
        False,
        float("-inf"),
        -0.1,
        "loss",
        BacktestResult(
            starting_cash=1000.0,
            ending_cash=1000.0,
            realized_pnl=0.0,
            win_rate=0.0,
            trades=0,
            max_drawdown=0.0,
            closed_trades=0,
            gross_exposure=0.0,
            total_fees=0.0,
            stopped_by_drawdown=False,
            max_exposure=0.0,
            trades_per_day_cap_hit=0,
            edge_vs_buy_hold=0.0,
        ),
        {"signal_count": 0, "net_mean_edge_bps": -5.0, "hit_rate": 0.0, "horizon_bars": 1},
    )

    summary = summarize_rule_alpha_candidate_distribution([result])

    assert summary["event_candidates_with_signals"] == 0
    assert summary["event_positive_candidates"] == 0
    assert summary["event_best_candidate"] == ""
    assert summary["event_best_net_edge_bps"] == 0.0


def test_rule_alpha_search_promotes_only_objective_accepted_candidate(monkeypatch) -> None:
    rows = _rows(count=48)
    strategy = StrategyConfig(
        cooldown_minutes=0,
        max_trades_per_day=100,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        liquidity_risk_enabled=False,
        dynamic_liquidity_session_enabled=False,
    )
    objective = SimpleNamespace(
        name="conservative",
        score=lambda result: result.realized_pnl,
        reject_reason=lambda result: None if result.realized_pnl > 0.0 and result.closed_trades > 0 else "no_profit",
    )
    monkeypatch.setattr("simple_ai_trading.alpha_search.get_objective", lambda _name: objective)

    report = optimize_rule_alpha_model_zoo(
        rows,
        strategy,
        objective_name="conservative",
        market_type="futures",
        starting_cash=1000.0,
        compute_backend="cpu",
        max_candidates=6,
    )

    assert report.accepted is True
    assert report.model is not None
    assert report.model.rule_alpha_evaluated_candidates == 12
    assert report.best_result is not None
    assert report.best_result.closed_trades > 0


def test_rule_alpha_search_fails_closed_when_objective_rejects(monkeypatch) -> None:
    rows = _rows(count=24)
    objective = SimpleNamespace(
        name="conservative",
        score=lambda result: result.realized_pnl,
        reject_reason=lambda _result: "forced_reject",
    )
    monkeypatch.setattr("simple_ai_trading.alpha_search.get_objective", lambda _name: objective)

    report = optimize_rule_alpha_model_zoo(
        rows,
        StrategyConfig(taker_fee_bps=0.0, slippage_bps=0.0),
        objective_name="conservative",
        market_type="futures",
        starting_cash=1000.0,
        compute_backend="cpu",
        max_candidates=2,
    )

    assert report.accepted is False
    assert report.model is None
    assert report.evaluated_candidates == 4
    assert report.best_candidate is not None
