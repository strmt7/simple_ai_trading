"""Stress validation for model-lab acceptance.

This module is deliberately stricter than the ordinary backtest path. A model
that only survives one optimistic execution assumption is not acceptable for
autonomous day trading, so model-lab acceptance requires profitability under a
small matrix of adverse spread, fee, latency, and liquidity assumptions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

from .advanced_model import advanced_feature_dimension, advanced_feature_signature, default_config_for, make_advanced_rows
from .api import Candle
from .backtest import BacktestResult, run_backtest
from .execution_simulation import SymbolExecutionProfile
from .model import ModelLoadError, TrainedModel, load_model
from .objective import get_objective
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


def _result_payload(result: BacktestResult) -> dict[str, object]:
    return {
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
        "scoring_backend_requested": result.scoring_backend_requested,
        "scoring_backend_kind": result.scoring_backend_kind,
        "scoring_backend_device": result.scoring_backend_device,
        "scoring_backend_reason": result.scoring_backend_reason,
    }


def _reject_reason(result: BacktestResult, *, objective_name: str, accepted: bool) -> str | None:
    if accepted:
        return None
    spec = get_objective(objective_name)
    reasons: list[str] = []
    if result.closed_trades < spec.min_closed_trades:
        reasons.append(f"closed_trades<{spec.min_closed_trades}")
    if spec.min_realized_pnl is not None and result.realized_pnl <= spec.min_realized_pnl:
        reasons.append(f"realized_pnl<={spec.min_realized_pnl}")
    if spec.min_edge_vs_buy_hold is not None and result.edge_vs_buy_hold < spec.min_edge_vs_buy_hold:
        reasons.append(f"edge_vs_buy_hold<{spec.min_edge_vs_buy_hold}")
    if spec.max_drawdown_rejection < 1.0 and result.max_drawdown > spec.max_drawdown_rejection:
        reasons.append(f"max_drawdown>{spec.max_drawdown_rejection}")
    if result.stopped_by_drawdown:
        reasons.append("stopped_by_drawdown")
    return "; ".join(reasons) or "objective_gate_failed"


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
            result=_result_payload(result),
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
    feature_cfg = default_config_for(objective_name, strategy.enabled_features)
    feature_dim = advanced_feature_dimension(feature_cfg)
    signature = advanced_feature_signature(feature_cfg)
    model = load_model(path, expected_feature_dim=feature_dim, expected_feature_signature=signature)
    return model, feature_cfg


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
    "StressScenario",
    "StressScenarioResult",
    "SuiteStressReport",
    "default_stress_scenarios",
    "validate_model_under_stress",
    "validate_suite_under_stress",
]
