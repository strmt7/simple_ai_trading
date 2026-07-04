"""Stress and temporal robustness validation for model-lab acceptance.

This module is deliberately stricter than the ordinary backtest path. A model
that only survives one optimistic execution assumption is not acceptable for
autonomous day trading, so model-lab acceptance requires profitability under a
small matrix of adverse spread, fee, latency, and liquidity assumptions plus
separate chronological windows for the final serialized artifact.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

from .advanced_model import (
    advanced_config_from_signature,
    advanced_feature_dimension,
    advanced_feature_signature,
    default_config_for,
    make_advanced_rows,
)
from .api import Candle
from .backtest import BacktestResult, run_backtest
from .execution_simulation import SymbolExecutionProfile
from .market_edge import build_market_edge_report
from .model import ModelLoadError, TrainedModel, load_model
from .objective import get_objective
from .regime import classify_market_regime, summarize_regime_windows
from .training_suite import SuiteReport
from .types import StrategyConfig


@dataclass(frozen=True)
class StressScenario:
    name: str
    slippage_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    latency_ms: int | None = None
    liquidity_haircut: float | None = None
    fee_multiplier: float = 1.0

    def strategy(self, base: StrategyConfig) -> StrategyConfig:
        spread_floor = max(float(base.max_spread_bps), float(base.slippage_bps))
        return replace(
            base,
            slippage_bps=max(0.0, float(base.slippage_bps) * self.slippage_multiplier),
            max_spread_bps=max(0.0, spread_floor * self.spread_multiplier),
            latency_buffer_ms=max(int(base.latency_buffer_ms), int(self.latency_ms or 0)),
            testnet_liquidity_haircut=max(
                float(base.testnet_liquidity_haircut),
                float(self.liquidity_haircut if self.liquidity_haircut is not None else base.testnet_liquidity_haircut),
            ),
            taker_fee_bps=max(0.0, float(base.taker_fee_bps) * self.fee_multiplier),
        )

    def profile(self, base: SymbolExecutionProfile | None) -> SymbolExecutionProfile | None:
        if base is None:
            return None
        haircut = max(
            float(base.liquidity_haircut),
            float(self.liquidity_haircut if self.liquidity_haircut is not None else base.liquidity_haircut),
        )
        volume_scale = max(0.05, 1.0 - haircut)
        return SymbolExecutionProfile(
            symbol=base.symbol,
            spread_bps=max(0.0, float(base.spread_bps) * self.spread_multiplier),
            quote_volume=max(0.0, float(base.quote_volume) * volume_scale),
            trade_count=max(0, int(round(float(base.trade_count) * volume_scale))),
            liquidity_score=max(0.0, min(1.0, float(base.liquidity_score) * volume_scale)),
            latency_ms=max(int(base.latency_ms), int(self.latency_ms or 0)),
            liquidity_haircut=max(0.0, min(1.0, haircut)),
        )

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StressScenarioResult:
    objective: str
    scenario: str
    accepted: bool
    reject_reason: str | None
    score: float
    result: dict[str, object]
    assumptions: dict[str, object]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveStressReport:
    objective: str
    accepted: bool
    model_path: str
    scenario_count: int
    accepted_scenarios: int
    worst_realized_pnl: float
    worst_max_drawdown: float
    results: list[StressScenarioResult]
    error: str | None = None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteStressReport:
    symbol: str
    accepted: bool
    objective_count: int
    accepted_objectives: int
    scenario_count: int
    worst_realized_pnl: float
    worst_max_drawdown: float
    objectives: list[ObjectiveStressReport]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalRobustnessPolicy:
    objective: str
    target_windows: int
    min_windows: int
    min_accepted_rate: float
    require_latest_window: bool
    min_window_rows: int
    max_sign_test_p_value: float = 0.35
    min_bootstrap_lower_mean_return: float = 0.0
    bootstrap_confidence: float = 0.90
    bootstrap_samples: int = 512

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalWindowResult:
    objective: str
    window_index: int
    accepted: bool
    reject_reason: str | None
    score: float
    start_index: int
    end_index: int
    rows: int
    result: dict[str, object]
    regime: dict[str, object]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveTemporalRobustnessReport:
    objective: str
    accepted: bool
    model_path: str
    policy: dict[str, object]
    reason: str | None
    window_count: int
    min_windows: int
    required_accepted_windows: int
    accepted_windows: int
    accepted_window_rate: float
    latest_window_accepted: bool
    worst_score: float
    worst_realized_pnl: float
    worst_max_drawdown: float
    statistical_edge: dict[str, object]
    regime_summary: dict[str, object]
    windows: list[TemporalWindowResult]
    error: str | None = None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteTemporalRobustnessReport:
    symbol: str
    accepted: bool
    objective_count: int
    accepted_objectives: int
    window_count: int
    accepted_windows: int
    accepted_window_rate: float
    worst_score: float
    worst_realized_pnl: float
    worst_max_drawdown: float
    statistical_edge_accepted: bool
    worst_sign_test_p_value: float
    worst_bootstrap_lower_mean_return: float
    regime_summary: dict[str, object]
    objectives: list[ObjectiveTemporalRobustnessReport]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def default_stress_scenarios() -> tuple[StressScenario, ...]:
    return (
        StressScenario("baseline"),
        StressScenario("wide_spread", slippage_multiplier=1.75, spread_multiplier=1.75, latency_ms=1000),
        StressScenario("latency_spike", slippage_multiplier=1.25, spread_multiplier=1.25, latency_ms=3000, liquidity_haircut=0.65),
        StressScenario(
            "liquidity_crunch",
            slippage_multiplier=2.50,
            spread_multiplier=2.25,
            latency_ms=2500,
            liquidity_haircut=0.85,
            fee_multiplier=1.25,
        ),
    )


def _finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def default_temporal_robustness_policy(objective_name: str) -> TemporalRobustnessPolicy:
    objective = get_objective(objective_name).name
    if objective == "conservative":
        return TemporalRobustnessPolicy(
            objective=objective,
            target_windows=6,
            min_windows=4,
            min_accepted_rate=0.80,
            require_latest_window=True,
            min_window_rows=45,
            max_sign_test_p_value=0.20,
            min_bootstrap_lower_mean_return=0.0,
            bootstrap_confidence=0.90,
        )
    if objective == "aggressive":
        return TemporalRobustnessPolicy(
            objective=objective,
            target_windows=4,
            min_windows=2,
            min_accepted_rate=0.60,
            require_latest_window=False,
            min_window_rows=35,
            max_sign_test_p_value=0.55,
            min_bootstrap_lower_mean_return=-0.005,
            bootstrap_confidence=0.90,
        )
    return TemporalRobustnessPolicy(
        objective=objective,
        target_windows=5,
        min_windows=3,
        min_accepted_rate=0.70,
        require_latest_window=True,
        min_window_rows=40,
        max_sign_test_p_value=0.35,
        min_bootstrap_lower_mean_return=-0.001,
        bootstrap_confidence=0.90,
    )


def _clamped_rate(value: float, *, default: float) -> float:
    if not math.isfinite(float(value)):
        return default
    return max(0.0, min(1.0, float(value)))


def _quantile(values: Sequence[float], q: float) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return float(clean[0])
    q = _clamped_rate(q, default=0.0)
    position = q * (len(clean) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(clean[lower])
    weight = position - lower
    return float(clean[lower] * (1.0 - weight) + clean[upper] * weight)


def _binomial_upper_tail(trials: int, successes: int, p: float = 0.5) -> float:
    n = max(0, int(trials))
    k = max(0, min(n, int(successes)))
    p = _clamped_rate(p, default=0.5)
    if n <= 0:
        return 1.0
    total = 0.0
    for hits in range(k, n + 1):
        total += math.comb(n, hits) * (p ** hits) * ((1.0 - p) ** (n - hits))
    return max(0.0, min(1.0, float(total)))


def _bootstrap_lower_mean_return(
    returns: Sequence[float],
    *,
    confidence: float,
    samples: int,
) -> float:
    clean = [float(value) for value in returns if math.isfinite(float(value))]
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    sample_count = max(64, min(4096, int(samples)))
    means: list[float] = []
    mask = (1 << 32) - 1
    for sample_index in range(sample_count):
        state = (sample_index + 1) & mask
        total = 0.0
        for _ in clean:
            state = (1664525 * state + 1013904223) & mask
            total += clean[state % len(clean)]
        means.append(total / len(clean))
    lower_tail = 1.0 - _clamped_rate(confidence, default=0.90)
    return _quantile(means, lower_tail)


def _chronological_windows(
    rows: Sequence[object],
    *,
    target_windows: int,
    min_window_rows: int,
) -> list[tuple[int, int, list[object]]]:
    row_list = list(rows)
    row_count = len(row_list)
    min_rows = max(1, int(min_window_rows))
    if row_count < min_rows:
        return []
    count = min(max(1, int(target_windows)), max(1, row_count // min_rows))
    window_size = max(min_rows, row_count // count)
    windows: list[tuple[int, int, list[object]]] = []
    start = 0
    for index in range(count):
        end = start + window_size
        if index == count - 1 or end + min_rows > row_count:
            end = row_count
        if end - start >= min_rows:
            windows.append((start, end, row_list[start:end]))
        start = end
        if start >= row_count:
            break
    return windows


def _result_payload(result: BacktestResult, *, objective_name: str | None = None) -> dict[str, object]:
    trade_returns = [
        float(value)
        for value in getattr(result, "trade_returns", ())
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    trade_pnls = [
        float(value)
        for value in getattr(result, "trade_pnls", ())
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    equity_curve = getattr(result, "equity_curve", ())
    payload: dict[str, object] = {
        "starting_cash": float(result.starting_cash),
        "ending_cash": float(result.ending_cash),
        "realized_pnl": float(result.realized_pnl),
        "win_rate": float(result.win_rate),
        "trades": int(result.trades),
        "max_drawdown": float(result.max_drawdown),
        "closed_trades": int(result.closed_trades),
        "gross_exposure": float(result.gross_exposure),
        "total_fees": float(result.total_fees),
        "stopped_by_drawdown": bool(result.stopped_by_drawdown),
        "max_exposure": float(result.max_exposure),
        "trades_per_day_cap_hit": int(result.trades_per_day_cap_hit),
        "buy_hold_pnl": float(result.buy_hold_pnl),
        "edge_vs_buy_hold": float(result.edge_vs_buy_hold),
        "trade_returns": trade_returns,
        "trade_pnls": trade_pnls,
        "trade_return_count": len(trade_returns),
        "equity_curve_points": len(equity_curve) if isinstance(equity_curve, (tuple, list)) else 0,
        "gross_profit": float(getattr(result, "gross_profit", 0.0)),
        "gross_loss": float(getattr(result, "gross_loss", 0.0)),
        "profit_factor": float(getattr(result, "profit_factor", 0.0)),
        "expectancy": float(getattr(result, "expectancy", 0.0)),
        "average_trade_return": float(getattr(result, "average_trade_return", 0.0)),
        "trade_return_stdev": float(getattr(result, "trade_return_stdev", 0.0)),
        "max_consecutive_losses": int(getattr(result, "max_consecutive_losses", 0)),
        "scoring_backend_requested": result.scoring_backend_requested,
        "scoring_backend_kind": result.scoring_backend_kind,
        "scoring_backend_device": result.scoring_backend_device,
        "scoring_backend_reason": result.scoring_backend_reason,
    }
    if objective_name:
        payload["market_edge"] = build_market_edge_report(result, objective_name).asdict()
    return payload


def _reject_reason(result: BacktestResult, *, objective_name: str, accepted: bool) -> str | None:
    if accepted:
        return None
    spec = get_objective(objective_name)
    return spec.reject_reason(result) or "objective_gate_failed"


def validate_model_under_stress(
    rows: Sequence[object],
    model: TrainedModel,
    strategy: StrategyConfig,
    *,
    objective_name: str,
    starting_cash: float,
    market_type: str,
    symbol_profile: SymbolExecutionProfile | None = None,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
    model_path: Path | str | None = None,
    scenarios: Sequence[StressScenario] | None = None,
) -> ObjectiveStressReport:
    row_list = list(rows)
    scenario_list = list(scenarios or default_stress_scenarios())
    spec = get_objective(objective_name)
    results: list[StressScenarioResult] = []
    worst_realized = float("inf")
    worst_drawdown = 0.0
    for scenario in scenario_list:
        scenario_strategy = scenario.strategy(strategy)
        scenario_profile = scenario.profile(symbol_profile)
        result = run_backtest(
            row_list,
            model,
            scenario_strategy,
            starting_cash=starting_cash,
            market_type=market_type,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
            symbol_profile=scenario_profile,
        )
        accepted = spec.accepts(result) and result.realized_pnl > 0.0 and not result.stopped_by_drawdown
        score = spec.score(result) if accepted else float("-inf")
        worst_realized = min(worst_realized, float(result.realized_pnl))
        worst_drawdown = max(worst_drawdown, float(result.max_drawdown))
        results.append(StressScenarioResult(
            objective=spec.name,
            scenario=scenario.name,
            accepted=bool(accepted),
            reject_reason=_reject_reason(result, objective_name=spec.name, accepted=bool(accepted)),
            score=float(score),
            result=_result_payload(result, objective_name=spec.name),
            assumptions={
                "scenario": scenario.asdict(),
                "strategy": {
                    "slippage_bps": scenario_strategy.slippage_bps,
                    "max_spread_bps": scenario_strategy.max_spread_bps,
                    "latency_buffer_ms": scenario_strategy.latency_buffer_ms,
                    "testnet_liquidity_haircut": scenario_strategy.testnet_liquidity_haircut,
                    "taker_fee_bps": scenario_strategy.taker_fee_bps,
                },
                "symbol_profile": scenario_profile.asdict() if scenario_profile is not None else None,
            },
        ))
    accepted_count = sum(1 for item in results if item.accepted)
    if worst_realized == float("inf"):
        worst_realized = 0.0
    return ObjectiveStressReport(
        objective=spec.name,
        accepted=bool(results) and accepted_count == len(results),
        model_path=str(model_path or ""),
        scenario_count=len(results),
        accepted_scenarios=accepted_count,
        worst_realized_pnl=float(worst_realized),
        worst_max_drawdown=float(worst_drawdown),
        results=results,
    )


def _load_objective_model(path: Path, objective_name: str, strategy: StrategyConfig) -> tuple[TrainedModel, object]:
    model = load_model(path, expected_feature_dim=None, expected_feature_signature=None)
    feature_cfg = advanced_config_from_signature(model.feature_signature, strategy.enabled_features)
    if feature_cfg is None:
        feature_cfg = default_config_for(objective_name, strategy.enabled_features)
    feature_dim = advanced_feature_dimension(feature_cfg)
    signature = advanced_feature_signature(feature_cfg)
    if int(model.feature_dim) != feature_dim:
        raise ModelLoadError(f"Feature dimension mismatch: model={model.feature_dim} expected={feature_dim}")
    if str(model.feature_signature or "") != signature:
        raise ModelLoadError(f"Feature signature mismatch: model={model.feature_signature} runtime={signature}")
    return model, feature_cfg


def _temporal_reason(
    *,
    window_count: int,
    min_windows: int,
    accepted_windows: int,
    required_accepted_windows: int,
    latest_window_accepted: bool,
    require_latest_window: bool,
) -> str | None:
    if window_count <= 0:
        return "insufficient_rows_for_temporal_robustness"
    if window_count < min_windows:
        return f"window_count<{min_windows}"
    if accepted_windows < required_accepted_windows:
        return f"accepted_windows<{required_accepted_windows}"
    if require_latest_window and not latest_window_accepted:
        return "latest_window_failed"
    return None


def _statistical_edge_report(
    windows: Sequence[TemporalWindowResult],
    *,
    starting_cash: float,
    policy: TemporalRobustnessPolicy,
) -> dict[str, object]:
    cash = max(1.0, abs(_finite(starting_cash, 1000.0)))
    window_returns = [
        _finite(window.result.get("realized_pnl")) / cash
        for window in windows
        if isinstance(window.result, dict)
    ]
    trade_returns: list[float] = []
    for window in windows:
        if not isinstance(window.result, dict):
            continue
        raw_returns = window.result.get("trade_returns")
        if not isinstance(raw_returns, list):
            continue
        trade_returns.extend(
            float(value)
            for value in raw_returns
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        )
    returns = trade_returns if len(trade_returns) >= max(3, len(window_returns)) else window_returns
    evidence_unit = "trade" if returns is trade_returns else "window"
    positive_windows = sum(1 for value in returns if value > 0.0)
    window_count = len(returns)
    sign_p_value = _binomial_upper_tail(window_count, positive_windows)
    mean_return = sum(returns) / window_count if window_count else 0.0
    median_return = _quantile(returns, 0.5)
    lower_mean = _bootstrap_lower_mean_return(
        returns,
        confidence=policy.bootstrap_confidence,
        samples=policy.bootstrap_samples,
    )
    reason = None
    if window_count <= 0:
        reason = "no_windows_for_statistical_edge"
    elif sign_p_value > policy.max_sign_test_p_value:
        reason = f"sign_test_p_value>{policy.max_sign_test_p_value:.4f}"
    elif lower_mean < policy.min_bootstrap_lower_mean_return:
        reason = f"bootstrap_lower_mean_return<{policy.min_bootstrap_lower_mean_return:.4f}"
    return {
        "accepted": reason is None,
        "reason": reason,
        "evidence_unit": evidence_unit,
        "sample_count": window_count,
        "window_count": len(window_returns),
        "trade_return_count": len(trade_returns),
        "positive_samples": positive_windows,
        "positive_windows": positive_windows,
        "positive_sample_rate": (positive_windows / window_count if window_count else 0.0),
        "positive_window_rate": (positive_windows / window_count if window_count else 0.0),
        "sign_test_p_value": float(sign_p_value),
        "max_sign_test_p_value": float(policy.max_sign_test_p_value),
        "mean_window_return": float(mean_return),
        "median_window_return": float(median_return),
        "bootstrap_confidence": float(policy.bootstrap_confidence),
        "bootstrap_samples": int(policy.bootstrap_samples),
        "bootstrap_lower_mean_return": float(lower_mean),
        "min_bootstrap_lower_mean_return": float(policy.min_bootstrap_lower_mean_return),
    }


def validate_model_temporal_robustness(
    rows: Sequence[object],
    model: TrainedModel,
    strategy: StrategyConfig,
    *,
    objective_name: str,
    starting_cash: float,
    market_type: str,
    symbol_profile: SymbolExecutionProfile | None = None,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
    model_path: Path | str | None = None,
    policy: TemporalRobustnessPolicy | None = None,
) -> ObjectiveTemporalRobustnessReport:
    """Replay the final serialized model across chronological windows."""

    spec = get_objective(objective_name)
    active_policy = policy or default_temporal_robustness_policy(spec.name)
    windows = _chronological_windows(
        rows,
        target_windows=active_policy.target_windows,
        min_window_rows=active_policy.min_window_rows,
    )
    required_accepted = (
        max(1, int(math.ceil(len(windows) * active_policy.min_accepted_rate)))
        if windows
        else max(1, int(math.ceil(active_policy.min_windows * active_policy.min_accepted_rate)))
    )
    window_results: list[TemporalWindowResult] = []
    overall_regime = classify_market_regime(rows).asdict()
    worst_score = float("inf")
    worst_realized = float("inf")
    worst_drawdown = 0.0
    for index, (start, end, window_rows) in enumerate(windows):
        result = run_backtest(
            window_rows,
            model,
            strategy,
            starting_cash=starting_cash,
            market_type=market_type,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
            symbol_profile=symbol_profile,
        )
        accepted = spec.accepts(result) and result.realized_pnl > 0.0 and not result.stopped_by_drawdown
        score = spec.score(result) if accepted else float("-inf")
        worst_score = min(worst_score, score)
        worst_realized = min(worst_realized, float(result.realized_pnl))
        worst_drawdown = max(worst_drawdown, float(result.max_drawdown))
        regime = classify_market_regime(window_rows).asdict()
        window_results.append(TemporalWindowResult(
            objective=spec.name,
            window_index=index,
            accepted=bool(accepted),
            reject_reason=_reject_reason(result, objective_name=spec.name, accepted=bool(accepted)),
            score=float(score),
            start_index=start,
            end_index=end,
            rows=len(window_rows),
            result=_result_payload(result, objective_name=spec.name),
            regime=regime,
        ))
    accepted_windows = sum(1 for item in window_results if item.accepted)
    latest_window_accepted = bool(window_results[-1].accepted) if window_results else False
    reason = _temporal_reason(
        window_count=len(window_results),
        min_windows=active_policy.min_windows,
        accepted_windows=accepted_windows,
        required_accepted_windows=required_accepted,
        latest_window_accepted=latest_window_accepted,
        require_latest_window=active_policy.require_latest_window,
    )
    statistical_edge = _statistical_edge_report(
        window_results,
        starting_cash=starting_cash,
        policy=active_policy,
    )
    if reason is None and not statistical_edge.get("accepted"):
        reason = str(statistical_edge.get("reason") or "statistical_edge_failed")
    regime_summary = summarize_regime_windows(
        [
            {
                "regime": item.regime,
                "result": item.result,
                "accepted": item.accepted,
            }
            for item in window_results
        ],
        overall_regime=overall_regime,
    )
    if worst_score == float("inf"):
        worst_score = float("-inf")
    if worst_realized == float("inf"):
        worst_realized = 0.0
    return ObjectiveTemporalRobustnessReport(
        objective=spec.name,
        accepted=reason is None,
        model_path=str(model_path or ""),
        policy=active_policy.asdict(),
        reason=reason,
        window_count=len(window_results),
        min_windows=active_policy.min_windows,
        required_accepted_windows=required_accepted,
        accepted_windows=accepted_windows,
        accepted_window_rate=(accepted_windows / len(window_results) if window_results else 0.0),
        latest_window_accepted=latest_window_accepted,
        worst_score=float(worst_score),
        worst_realized_pnl=float(worst_realized),
        worst_max_drawdown=float(worst_drawdown),
        statistical_edge=statistical_edge,
        regime_summary=regime_summary,
        windows=window_results,
    )


def validate_suite_temporal_robustness(
    candles: Sequence[Candle],
    strategy: StrategyConfig,
    suite: SuiteReport,
    *,
    symbol: str,
    symbol_profile: SymbolExecutionProfile | None,
    starting_cash: float,
    market_type: str,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
) -> SuiteTemporalRobustnessReport:
    objective_reports: list[ObjectiveTemporalRobustnessReport] = []
    worst_score = float("inf")
    worst_realized = float("inf")
    worst_drawdown = 0.0
    total_windows = 0
    total_accepted_windows = 0
    for outcome in suite.outcomes:
        objective_name = get_objective(outcome.objective).name
        model_path = Path(outcome.model_path)
        try:
            model, feature_cfg = _load_objective_model(model_path, objective_name, strategy)
            rows = make_advanced_rows(candles, feature_cfg)
            if not rows:
                raise ValueError("temporal robustness could not build feature rows")
            report = validate_model_temporal_robustness(
                rows,
                model,
                strategy,
                objective_name=objective_name,
                starting_cash=starting_cash,
                market_type=market_type,
                symbol_profile=symbol_profile,
                compute_backend=compute_backend,
                score_batch_size=score_batch_size,
                model_path=model_path,
            )
        except (ModelLoadError, OSError, ValueError) as exc:
            policy = default_temporal_robustness_policy(objective_name)
            report = ObjectiveTemporalRobustnessReport(
                objective=objective_name,
                accepted=False,
                model_path=str(model_path),
                policy=policy.asdict(),
                reason="temporal_robustness_error",
                window_count=0,
                min_windows=policy.min_windows,
                required_accepted_windows=max(1, int(math.ceil(policy.min_windows * policy.min_accepted_rate))),
                accepted_windows=0,
                accepted_window_rate=0.0,
                latest_window_accepted=False,
                worst_score=float("-inf"),
                worst_realized_pnl=0.0,
                worst_max_drawdown=1.0,
                statistical_edge={
                    "accepted": False,
                    "reason": "temporal_robustness_error",
                    "evidence_unit": "none",
                    "sample_count": 0,
                    "window_count": 0,
                    "trade_return_count": 0,
                    "positive_samples": 0,
                    "positive_windows": 0,
                    "positive_sample_rate": 0.0,
                    "positive_window_rate": 0.0,
                    "sign_test_p_value": 1.0,
                    "max_sign_test_p_value": float(policy.max_sign_test_p_value),
                    "mean_window_return": 0.0,
                    "median_window_return": 0.0,
                    "bootstrap_confidence": float(policy.bootstrap_confidence),
                    "bootstrap_samples": int(policy.bootstrap_samples),
                    "bootstrap_lower_mean_return": 0.0,
                    "min_bootstrap_lower_mean_return": float(policy.min_bootstrap_lower_mean_return),
                },
                regime_summary=summarize_regime_windows([], overall_regime=classify_market_regime([]).asdict()),
                windows=[],
                error=str(exc),
            )
        objective_reports.append(report)
        total_windows += int(report.window_count)
        total_accepted_windows += int(report.accepted_windows)
        worst_score = min(worst_score, _finite(report.worst_score, float("-inf")))
        worst_realized = min(worst_realized, _finite(report.worst_realized_pnl))
        worst_drawdown = max(worst_drawdown, _finite(report.worst_max_drawdown))
    accepted_count = sum(1 for item in objective_reports if item.accepted)
    if worst_score == float("inf"):
        worst_score = float("-inf")
    if worst_realized == float("inf"):
        worst_realized = 0.0
    edge_reports = [
        report.statistical_edge
        for report in objective_reports
        if isinstance(report.statistical_edge, dict)
    ]
    edge_accepted = bool(edge_reports) and all(bool(edge.get("accepted")) for edge in edge_reports)
    worst_sign_p = max(
        (_finite(edge.get("sign_test_p_value"), 1.0) for edge in edge_reports),
        default=1.0,
    )
    worst_bootstrap_lower = min(
        (_finite(edge.get("bootstrap_lower_mean_return")) for edge in edge_reports),
        default=0.0,
    )
    suite_regime_summary = summarize_regime_windows(
        [
            {
                "regime": window.regime,
                "result": window.result,
                "accepted": window.accepted,
            }
            for report in objective_reports
            for window in report.windows
        ],
        overall_regime={
            "dominant_regime": "suite_objective_windows",
            "rows": total_windows,
        },
    )
    return SuiteTemporalRobustnessReport(
        symbol=symbol,
        accepted=bool(objective_reports) and accepted_count == len(objective_reports) and edge_accepted,
        objective_count=len(objective_reports),
        accepted_objectives=accepted_count,
        window_count=total_windows,
        accepted_windows=total_accepted_windows,
        accepted_window_rate=(total_accepted_windows / total_windows if total_windows else 0.0),
        worst_score=float(worst_score),
        worst_realized_pnl=float(worst_realized),
        worst_max_drawdown=float(worst_drawdown),
        statistical_edge_accepted=edge_accepted,
        worst_sign_test_p_value=float(worst_sign_p),
        worst_bootstrap_lower_mean_return=float(worst_bootstrap_lower),
        regime_summary=suite_regime_summary,
        objectives=objective_reports,
    )


def validate_suite_under_stress(
    candles: Sequence[Candle],
    strategy: StrategyConfig,
    suite: SuiteReport,
    *,
    symbol: str,
    symbol_profile: SymbolExecutionProfile | None,
    starting_cash: float,
    market_type: str,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
    scenarios: Sequence[StressScenario] | None = None,
) -> SuiteStressReport:
    objective_reports: list[ObjectiveStressReport] = []
    worst_realized = float("inf")
    worst_drawdown = 0.0
    for outcome in suite.outcomes:
        objective_name = get_objective(outcome.objective).name
        model_path = Path(outcome.model_path)
        try:
            model, feature_cfg = _load_objective_model(model_path, objective_name, strategy)
            rows = make_advanced_rows(candles, feature_cfg)
            if not rows:
                raise ValueError("stress validation could not build feature rows")
            report = validate_model_under_stress(
                rows,
                model,
                strategy,
                objective_name=objective_name,
                starting_cash=starting_cash,
                market_type=market_type,
                symbol_profile=symbol_profile,
                compute_backend=compute_backend,
                score_batch_size=score_batch_size,
                model_path=model_path,
                scenarios=scenarios,
            )
        except (ModelLoadError, OSError, ValueError) as exc:
            report = ObjectiveStressReport(
                objective=objective_name,
                accepted=False,
                model_path=str(model_path),
                scenario_count=0,
                accepted_scenarios=0,
                worst_realized_pnl=0.0,
                worst_max_drawdown=1.0,
                results=[],
                error=str(exc),
            )
        objective_reports.append(report)
        worst_realized = min(worst_realized, _finite(report.worst_realized_pnl))
        worst_drawdown = max(worst_drawdown, _finite(report.worst_max_drawdown))
    accepted_count = sum(1 for item in objective_reports if item.accepted)
    if worst_realized == float("inf"):
        worst_realized = 0.0
    return SuiteStressReport(
        symbol=symbol,
        accepted=bool(objective_reports) and accepted_count == len(objective_reports),
        objective_count=len(objective_reports),
        accepted_objectives=accepted_count,
        scenario_count=sum(item.scenario_count for item in objective_reports),
        worst_realized_pnl=float(worst_realized),
        worst_max_drawdown=float(worst_drawdown),
        objectives=objective_reports,
    )


__all__ = [
    "ObjectiveStressReport",
    "ObjectiveTemporalRobustnessReport",
    "StressScenario",
    "StressScenarioResult",
    "SuiteStressReport",
    "SuiteTemporalRobustnessReport",
    "TemporalRobustnessPolicy",
    "TemporalWindowResult",
    "default_stress_scenarios",
    "default_temporal_robustness_policy",
    "validate_model_under_stress",
    "validate_model_temporal_robustness",
    "validate_suite_under_stress",
    "validate_suite_temporal_robustness",
]
