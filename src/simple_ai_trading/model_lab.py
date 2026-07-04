"""Cross-symbol model research and profitability-gated optimization workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .api import BinanceAPIError, BinanceClient, Candle
from .market_universe import MarketEligibility, UniverseSelection, rank_high_liquidity_universe
from .portfolio_risk import PortfolioRiskReport, build_portfolio_risk_report
from .robust_validation import (
    SuiteStressReport,
    SuiteTemporalRobustnessReport,
    validate_suite_temporal_robustness,
    validate_suite_under_stress,
)
from .storage import write_json_atomic
from .training_suite import SuiteReport, TrainingSuiteRejected, run_training_suite
from .types import RuntimeConfig, StrategyConfig


@dataclass
class SymbolResearchOutcome:
    symbol: str
    accepted: bool
    rows: int
    objectives: list[str]
    report_path: str | None = None
    error: str | None = None
    liquidity: dict[str, object] | None = None
    objective_scores: dict[str, float] = field(default_factory=dict)
    hybrid_profiles: dict[str, str] = field(default_factory=dict)
    selection_risk: dict[str, object] = field(default_factory=dict)
    hybrid_ablation: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    feature_ablation: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    meta_label_validation: dict[str, object] = field(default_factory=dict)
    stress_validation: dict[str, object] | None = None
    stress_report_path: str | None = None
    robustness_validation: dict[str, object] | None = None
    robustness_report_path: str | None = None
    regime_validation: dict[str, object] | None = None
    diagnostics: dict[str, object] | None = None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ModelLabReport:
    quote_asset: str
    interval: str
    market_type: str
    requested_objectives: list[str]
    universe: dict[str, object]
    outcomes: list[SymbolResearchOutcome]
    output_dir: str
    report_path: str
    portfolio_risk: dict[str, object] | None = None

    @property
    def accepted_symbols(self) -> list[str]:
        return [item.symbol for item in self.outcomes if item.accepted]

    def asdict(self) -> dict[str, object]:
        return {
            "quote_asset": self.quote_asset,
            "interval": self.interval,
            "market_type": self.market_type,
            "requested_objectives": list(self.requested_objectives),
            "universe": self.universe,
            "accepted_symbols": self.accepted_symbols,
            "portfolio_risk": self.portfolio_risk,
            "outcomes": [item.asdict() for item in self.outcomes],
            "output_dir": self.output_dir,
            "report_path": self.report_path,
        }


def _safe_symbol_path(symbol: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in symbol.upper())


def _candles_for_symbol(client: BinanceClient, symbol: str, interval: str, limit: int) -> list[Candle]:
    return client.get_klines(symbol, interval, limit=max(1, int(limit)))


def _outcome_from_suite(
    symbol: str,
    suite: SuiteReport,
    liquidity: MarketEligibility,
    stress_report: SuiteStressReport,
    stress_report_path: Path,
    robustness_report: SuiteTemporalRobustnessReport,
    robustness_report_path: Path,
) -> SymbolResearchOutcome:
    scores = {outcome.objective: float(outcome.best_score) for outcome in suite.outcomes}
    hybrid_profiles = {outcome.objective: str(outcome.hybrid_profile) for outcome in suite.outcomes}
    selection_risk = {
        outcome.objective: dict(getattr(outcome, "selection_risk", {}) or {})
        for outcome in suite.outcomes
        if getattr(outcome, "selection_risk", None)
    }
    hybrid_ablation = {
        outcome.objective: list(getattr(outcome, "hybrid_ablation", []) or [])
        for outcome in suite.outcomes
        if getattr(outcome, "hybrid_ablation", None)
    }
    feature_ablation = {
        outcome.objective: list(getattr(outcome, "feature_ablation", []) or [])
        for outcome in suite.outcomes
        if getattr(outcome, "feature_ablation", None)
    }
    meta_labels = {
        outcome.objective: outcome.meta_label_report
        for outcome in suite.outcomes
        if getattr(outcome, "meta_label_report", None) is not None
    }
    robustness_payload = robustness_report.asdict()
    regime_payload = robustness_payload.get("regime_summary") if isinstance(robustness_payload, dict) else None
    score_accepted = bool(suite.outcomes) and all(score > 0.0 for score in scores.values())
    selection_risk_accepted = all(
        not isinstance(report, dict) or report.get("passed") is not False
        for report in selection_risk.values()
    )
    stress_accepted = bool(stress_report.accepted)
    robustness_accepted = bool(robustness_report.accepted)
    accepted = score_accepted and selection_risk_accepted and stress_accepted and robustness_accepted
    error = None
    if not accepted and score_accepted and not selection_risk_accepted:
        error = "selection_risk_failed"
    elif not accepted and score_accepted and selection_risk_accepted and not stress_accepted:
        error = "stress_validation_failed"
    elif not accepted and score_accepted and selection_risk_accepted and stress_accepted and not robustness_accepted:
        error = "temporal_robustness_failed"
    return SymbolResearchOutcome(
        symbol=symbol,
        accepted=accepted,
        rows=int(suite.total_rows),
        objectives=list(suite.objectives_run),
        report_path=str(suite.summary_path),
        error=error,
        liquidity=liquidity.asdict(),
        objective_scores=scores,
        hybrid_profiles=hybrid_profiles,
        selection_risk=selection_risk,
        hybrid_ablation=hybrid_ablation,
        feature_ablation=feature_ablation,
        meta_label_validation=meta_labels,
        stress_validation=stress_report.asdict(),
        stress_report_path=str(stress_report_path),
        robustness_validation=robustness_payload,
        robustness_report_path=str(robustness_report_path),
        regime_validation=regime_payload if isinstance(regime_payload, dict) else None,
    )


def _apply_portfolio_risk_gate(
    outcomes: list[SymbolResearchOutcome],
    candles_by_symbol: dict[str, list[Candle]],
    strategy: StrategyConfig,
    *,
    min_symbols: int,
) -> PortfolioRiskReport:
    candidates = {
        outcome.symbol: candles_by_symbol[outcome.symbol]
        for outcome in outcomes
        if outcome.accepted and outcome.symbol in candles_by_symbol
    }
    report = build_portfolio_risk_report(candidates, strategy, min_symbols=min_symbols)
    if report.accepted or not candidates:
        return report
    for outcome in outcomes:
        if not outcome.accepted:
            continue
        outcome.accepted = False
        outcome.error = "portfolio_risk_failed"
        details = {
            "portfolio_risk_reason": report.reason,
            "portfolio_cvar_95": report.portfolio_cvar_95,
            "portfolio_max_drawdown": report.portfolio_max_drawdown,
            "effective_symbol_count": report.effective_symbol_count,
            "max_pairwise_correlation": report.max_pairwise_correlation,
            "max_cluster_weight": report.max_cluster_weight,
        }
        outcome.diagnostics = {**(outcome.diagnostics or {}), **details}
    return report


def run_model_lab(
    client: BinanceClient,
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    *,
    objectives: Sequence[str],
    output_dir: Path,
    starting_cash: float,
    max_symbols: int = 6,
    max_scan: int = 250,
    limit: int = 1000,
    compute_backend: str | None = None,
    batch_size: int = 8192,
    score_batch_size: int | None = None,
    max_candidates: int | None = None,
) -> ModelLabReport:
    """Rank liquid symbols, train all risk objectives, and write a lab report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    universe = rank_high_liquidity_universe(
        client,
        strategy,
        quote_asset=runtime.quote_asset,
        max_symbols=max_symbols,
        max_scan=max_scan,
    )
    outcomes: list[SymbolResearchOutcome] = []
    liquidity_by_symbol = {item.symbol: item for item in universe.eligible}
    candles_by_symbol: dict[str, list[Candle]] = {}
    for item in universe.eligible:
        symbol = item.symbol
        symbol_dir = output_dir / _safe_symbol_path(symbol)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        try:
            candles = _candles_for_symbol(client, symbol, runtime.interval, limit)
            candles_by_symbol[symbol] = candles
            suite = run_training_suite(
                candles,
                strategy,
                objectives=objectives,
                market_type=runtime.market_type,
                starting_cash=starting_cash,
                output_dir=symbol_dir,
                summary_path=symbol_dir / "training_suite_summary.json",
                max_workers=1 if compute_backend and compute_backend != "cpu" else None,
                compute_backend=compute_backend,
                batch_size=batch_size,
                score_batch_size=score_batch_size,
                max_candidates=max_candidates,
            )
            liquidity = liquidity_by_symbol[symbol]
            stress_profile = liquidity.execution_profile(
                latency_ms=strategy.latency_buffer_ms,
                liquidity_haircut=strategy.testnet_liquidity_haircut,
            )
            stress_report = validate_suite_under_stress(
                candles,
                strategy,
                suite,
                symbol=symbol,
                symbol_profile=stress_profile,
                starting_cash=starting_cash,
                market_type=runtime.market_type,
                compute_backend=compute_backend,
                score_batch_size=score_batch_size or batch_size,
            )
            stress_report_path = symbol_dir / "stress_validation.json"
            write_json_atomic(stress_report_path, stress_report.asdict(), indent=2, sort_keys=True)
            robustness_report = validate_suite_temporal_robustness(
                candles,
                strategy,
                suite,
                symbol=symbol,
                symbol_profile=stress_profile,
                starting_cash=starting_cash,
                market_type=runtime.market_type,
                compute_backend=compute_backend,
                score_batch_size=score_batch_size or batch_size,
            )
            robustness_report_path = symbol_dir / "temporal_robustness.json"
            write_json_atomic(robustness_report_path, robustness_report.asdict(), indent=2, sort_keys=True)
            outcomes.append(_outcome_from_suite(
                symbol,
                suite,
                liquidity,
                stress_report,
                stress_report_path,
                robustness_report,
                robustness_report_path,
            ))
        except TrainingSuiteRejected as exc:
            outcomes.append(SymbolResearchOutcome(
                symbol=symbol,
                accepted=False,
                rows=int(exc.row_count),
                objectives=list(objectives),
                error=str(exc),
                liquidity=item.asdict(),
                diagnostics=exc.diagnostics,
            ))
        except (BinanceAPIError, ValueError, OSError) as exc:
            outcomes.append(SymbolResearchOutcome(
                symbol=symbol,
                accepted=False,
                rows=0,
                objectives=list(objectives),
                error=str(exc),
                liquidity=item.asdict(),
            ))
    portfolio_report = _apply_portfolio_risk_gate(
        outcomes,
        candles_by_symbol,
        strategy,
        min_symbols=universe.min_required,
    )
    portfolio_report_path = output_dir / "portfolio_risk.json"
    write_json_atomic(portfolio_report_path, portfolio_report.asdict(), indent=2, sort_keys=True)
    report_path = output_dir / "model_lab_report.json"
    report = ModelLabReport(
        quote_asset=runtime.quote_asset,
        interval=runtime.interval,
        market_type=runtime.market_type,
        requested_objectives=list(objectives),
        universe=universe.asdict(),
        outcomes=outcomes,
        output_dir=str(output_dir),
        report_path=str(report_path),
        portfolio_risk={
            **portfolio_report.asdict(),
            "report_path": str(portfolio_report_path),
        },
    )
    write_json_atomic(report_path, report.asdict(), indent=2)
    return report
