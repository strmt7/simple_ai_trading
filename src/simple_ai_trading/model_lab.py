"""Cross-symbol model research and profitability-gated optimization workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Sequence

from .api import BinanceAPIError, BinanceClient, Candle
from .data_coverage import DataCoverageReport, describe_candle_coverage
from .financial_sanity import blocking_reasons, build_model_lab_financial_sanity_report
from .intervals import max_limit
from .market_store import MarketDataStore
from .market_universe import MarketEligibility, UniverseSelection, rank_high_liquidity_universe
from .market_data import clean_candles
from .model import ModelLoadError, load_model, serialize_model
from .portfolio_risk import PortfolioRiskReport, build_portfolio_risk_report
from .positions import LearningFeedbackReport, load_learning_feedback_file
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
    walk_forward_gate: dict[str, object] = field(default_factory=dict)
    selection_risk: dict[str, object] = field(default_factory=dict)
    hybrid_ablation: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    feature_ablation: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    meta_label_validation: dict[str, object] = field(default_factory=dict)
    stress_validation: dict[str, object] | None = None
    stress_report_path: str | None = None
    robustness_validation: dict[str, object] | None = None
    robustness_report_path: str | None = None
    regime_validation: dict[str, object] | None = None
    learning_feedback: dict[str, object] | None = None
    data_coverage: dict[str, object] | None = None
    diagnostics: dict[str, object] | None = None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ModelLabReport:
    quote_asset: str
    interval: str
    market_type: str
    data_source: str
    market_db_path: str | None
    require_db_data: bool
    requested_objectives: list[str]
    universe: dict[str, object]
    outcomes: list[SymbolResearchOutcome]
    output_dir: str
    report_path: str
    portfolio_risk: dict[str, object] | None = None
    learning_feedback: dict[str, object] | None = None
    financial_sanity: dict[str, object] | None = None

    @property
    def accepted_symbols(self) -> list[str]:
        return [item.symbol for item in self.outcomes if item.accepted]

    def asdict(self) -> dict[str, object]:
        return {
            "quote_asset": self.quote_asset,
            "interval": self.interval,
            "market_type": self.market_type,
            "data_source": self.data_source,
            "market_db_path": self.market_db_path,
            "require_db_data": self.require_db_data,
            "requested_objectives": list(self.requested_objectives),
            "universe": self.universe,
            "accepted_symbols": self.accepted_symbols,
            "portfolio_risk": self.portfolio_risk,
            "learning_feedback": self.learning_feedback,
            "financial_sanity": self.financial_sanity,
            "outcomes": [item.asdict() for item in self.outcomes],
            "output_dir": self.output_dir,
            "report_path": self.report_path,
        }


def _safe_symbol_path(symbol: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in symbol.upper())


def _candles_for_symbol(client: BinanceClient, symbol: str, interval: str, limit: int) -> list[Candle]:
    return client.get_klines(symbol, interval, limit=max(1, int(limit)))


def _candles_for_symbol_full_history(
    client: BinanceClient,
    symbol: str,
    interval: str,
    *,
    batch_size: int,
) -> list[Candle]:
    request_limit = max(1, min(max_limit(getattr(client, "market_type", "spot")), int(batch_size)))
    candles_by_open_time: dict[int, Candle] = {}
    end_time = None
    while True:
        kwargs: dict[str, int] = {"limit": request_limit}
        if end_time is not None:
            kwargs["end_time"] = int(end_time)
        chunk = client.get_klines(symbol, interval, **kwargs)
        if not chunk:
            break
        before = len(candles_by_open_time)
        for candle in chunk:
            candles_by_open_time[int(candle.open_time)] = candle
        earliest_open = min(int(candle.open_time) for candle in chunk)
        next_end = earliest_open - 1
        if len(candles_by_open_time) == before or len(chunk) < request_limit or next_end == end_time or next_end < 0:
            break
        end_time = next_end
    return clean_candles(candles_by_open_time.values())


def _candles_for_symbol_from_db(
    symbol: str,
    *,
    market_type: str,
    interval: str,
    db_path: Path,
) -> list[Candle]:
    with MarketDataStore(db_path) as store:
        candles = store.fetch_candles(symbol, market_type, interval)
    cleaned = clean_candles(candles)
    if not cleaned:
        raise ValueError(f"market database has no candles for {symbol} {market_type} {interval}")
    return cleaned


def _load_model_lab_learning_feedback(
    learning_feedback_path: Path | None,
) -> tuple[LearningFeedbackReport | None, Path | None]:
    """Load bounded post-trade feedback for model promotion, if available."""

    if learning_feedback_path is None:
        default_path = Path("data/autonomous/learning_feedback.json")
        if not default_path.exists():
            return None, None
        report = load_learning_feedback_file(default_path)
        if report is None:
            raise ValueError(f"learning feedback file is missing or invalid: {default_path}")
        return report, default_path
    source = Path(learning_feedback_path)
    report = load_learning_feedback_file(source)
    if report is None:
        raise ValueError(f"learning feedback file is missing or invalid: {source}")
    return report, source


def _learning_feedback_summary(
    report: LearningFeedbackReport | None,
    source_path: Path | None,
) -> dict[str, object] | None:
    if report is None:
        return None
    return {
        **report.asdict(),
        "source_path": str(source_path) if source_path is not None else None,
    }


def _report_metric(report: object, field: str) -> float:
    try:
        value = getattr(report, field)
        return float(value)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _walk_forward_gate_passed(report: object) -> bool:
    """Return True only for real, fully accepted purged walk-forward folds."""

    if not isinstance(report, dict) or report.get("passed") is not True:
        return False
    if report.get("reason") not in (None, ""):
        return False
    try:
        fold_count = int(report.get("fold_count", 0) or 0)
        accepted_folds = int(report.get("accepted_folds", 0) or 0)
        worst_score = float(report.get("worst_score", 0.0) or 0.0)
        worst_pnl = float(report.get("worst_realized_pnl", 0.0) or 0.0)
        worst_drawdown = float(report.get("worst_max_drawdown", 1.0) or 1.0)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        fold_count > 0
        and accepted_folds == fold_count
        and worst_score > 0.0
        and worst_pnl > 0.0
        and 0.0 <= worst_drawdown <= 1.0
    )


def _symbol_loss_count(report: LearningFeedbackReport, symbol: str) -> int:
    target = symbol.upper()
    total = 0
    for raw_symbol, count in report.loss_by_symbol.items():
        if str(raw_symbol).upper() == target:
            total += int(count)
    return total


def _symbol_learning_feedback(
    report: LearningFeedbackReport | None,
    symbol: str,
    stress_report: SuiteStressReport,
    robustness_report: SuiteTemporalRobustnessReport,
) -> dict[str, object] | None:
    if report is None:
        return None
    loss_count = _symbol_loss_count(report, symbol)
    review_required = loss_count >= 2
    stress_worst_pnl = _report_metric(stress_report, "worst_realized_pnl")
    robustness_worst_pnl = _report_metric(robustness_report, "worst_realized_pnl")
    recovery_evidence_passed = (
        not review_required
        or (
            bool(getattr(stress_report, "accepted", False))
            and bool(getattr(robustness_report, "accepted", False))
            and stress_worst_pnl > 0.0
            and robustness_worst_pnl > 0.0
        )
    )
    relevant_recommendations = [
        item for item in report.recommendations
        if symbol.upper() in item.upper() or not item.startswith("review_symbol_specific_edge:")
    ][:8]
    blocks_promotion = bool(review_required and not recovery_evidence_passed)
    reason = None
    if blocks_promotion:
        reason = "repeated_symbol_losses_require_positive_stress_and_temporal_recovery_evidence"
    elif review_required:
        reason = "repeated_symbol_losses_recovered_in_current_promotion_evidence"
    return {
        "symbol": symbol.upper(),
        "source": "closed_trade_learning_feedback",
        "symbol_loss_count": loss_count,
        "review_required": review_required,
        "promotion_safe": bool(report.promotion_safe),
        "global_max_consecutive_losses": int(report.max_consecutive_losses),
        "global_net_realized_pnl": float(report.net_realized_pnl),
        "recovery_evidence": {
            "passed": recovery_evidence_passed,
            "stress_accepted": bool(getattr(stress_report, "accepted", False)),
            "stress_worst_realized_pnl": stress_worst_pnl,
            "temporal_robustness_accepted": bool(getattr(robustness_report, "accepted", False)),
            "temporal_worst_realized_pnl": robustness_worst_pnl,
        },
        "blocks_promotion": blocks_promotion,
        "reason": reason,
        "recommendations": relevant_recommendations,
    }


def _data_coverage_blocks_promotion(data_coverage: DataCoverageReport | None) -> bool:
    if data_coverage is None:
        return True
    hard_warnings = {
        "no_candles_used",
        "no_model_rows_used",
        "coverage_gaps_detected",
        "coverage_ratio_below_99_5_percent",
    }
    return any(item in hard_warnings for item in data_coverage.integrity_warnings)


def _outcome_from_suite(
    symbol: str,
    suite: SuiteReport,
    liquidity: MarketEligibility,
    stress_report: SuiteStressReport,
    stress_report_path: Path,
    robustness_report: SuiteTemporalRobustnessReport,
    robustness_report_path: Path,
    learning_feedback: dict[str, object] | None = None,
    data_coverage: DataCoverageReport | None = None,
) -> SymbolResearchOutcome:
    scores = {outcome.objective: float(outcome.best_score) for outcome in suite.outcomes}
    hybrid_profiles = {outcome.objective: str(outcome.hybrid_profile) for outcome in suite.outcomes}
    walk_forward_gate = {
        outcome.objective: dict(getattr(outcome, "walk_forward_gate", {}) or {})
        for outcome in suite.outcomes
        if getattr(outcome, "walk_forward_gate", None)
    }
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
    walk_forward_accepted = all(
        _walk_forward_gate_passed(walk_forward_gate.get(str(outcome.objective)))
        for outcome in suite.outcomes
    )
    selection_risk_accepted = all(
        not isinstance(report, dict) or report.get("passed") is not False
        for report in selection_risk.values()
    )
    stress_accepted = bool(stress_report.accepted)
    robustness_accepted = bool(robustness_report.accepted)
    learning_feedback_accepted = not bool(learning_feedback and learning_feedback.get("blocks_promotion"))
    data_coverage_accepted = not _data_coverage_blocks_promotion(data_coverage)
    accepted = (
        score_accepted
        and walk_forward_accepted
        and selection_risk_accepted
        and stress_accepted
        and robustness_accepted
        and learning_feedback_accepted
        and data_coverage_accepted
    )
    error = None
    if not accepted and score_accepted and not walk_forward_accepted:
        error = "purged_walk_forward_failed"
    elif not accepted and score_accepted and walk_forward_accepted and not selection_risk_accepted:
        error = "selection_risk_failed"
    elif not accepted and score_accepted and walk_forward_accepted and selection_risk_accepted and not stress_accepted:
        error = "stress_validation_failed"
    elif (
        not accepted
        and score_accepted
        and walk_forward_accepted
        and selection_risk_accepted
        and stress_accepted
        and not robustness_accepted
    ):
        error = "temporal_robustness_failed"
    elif (
        not accepted
        and score_accepted
        and walk_forward_accepted
        and selection_risk_accepted
        and stress_accepted
        and robustness_accepted
    ):
        error = "learning_feedback_failed" if not learning_feedback_accepted else "data_coverage_failed"
    diagnostics = None
    if learning_feedback and learning_feedback.get("blocks_promotion"):
        diagnostics = {
            "learning_feedback_reason": learning_feedback.get("reason"),
            "learning_feedback_symbol_loss_count": learning_feedback.get("symbol_loss_count"),
        }
    if data_coverage is not None and data_coverage.integrity_status == "fail":
        diagnostics = {
            **(diagnostics or {}),
            "data_coverage_status": data_coverage.integrity_status,
            "data_coverage_warnings": list(data_coverage.integrity_warnings),
        }
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
        walk_forward_gate=walk_forward_gate,
        selection_risk=selection_risk,
        hybrid_ablation=hybrid_ablation,
        feature_ablation=feature_ablation,
        meta_label_validation=meta_labels,
        stress_validation=stress_report.asdict(),
        stress_report_path=str(stress_report_path),
        robustness_validation=robustness_payload,
        robustness_report_path=str(robustness_report_path),
        regime_validation=regime_payload if isinstance(regime_payload, dict) else None,
        learning_feedback=learning_feedback,
        data_coverage=data_coverage.asdict() if data_coverage is not None else None,
        diagnostics=diagnostics,
    )


def _objective_report(payload: dict[str, object], objective: str) -> dict[str, object] | None:
    reports = payload.get("objectives")
    if not isinstance(reports, list):
        return None
    for item in reports:
        if isinstance(item, dict) and str(item.get("objective") or "") == objective:
            return item
    return None


@dataclass(frozen=True)
class _ExecutionStampContext:
    suite: SuiteReport
    symbol: str
    market_type: str
    interval: str
    liquidity: MarketEligibility
    stress_profile: object
    stress_report: SuiteStressReport
    stress_report_path: Path
    robustness_report: SuiteTemporalRobustnessReport
    robustness_report_path: Path
    learning_feedback: dict[str, object] | None = None
    data_coverage: DataCoverageReport | None = None


def _stamp_model_execution_validation(
    suite: SuiteReport,
    *,
    symbol: str,
    market_type: str,
    interval: str,
    liquidity: MarketEligibility,
    stress_profile: object,
    stress_report: SuiteStressReport,
    stress_report_path: Path,
    robustness_report: SuiteTemporalRobustnessReport,
    robustness_report_path: Path,
    portfolio_report: PortfolioRiskReport,
    portfolio_report_path: Path,
    learning_feedback: dict[str, object] | None = None,
    data_coverage: DataCoverageReport | None = None,
) -> None:
    """Persist symbol-specific execution validation on each model-lab artifact."""

    stress_payload = stress_report.asdict()
    robustness_payload = robustness_report.asdict()
    for outcome in getattr(suite, "outcomes", []):
        model_path = getattr(outcome, "model_path", None)
        objective = str(getattr(outcome, "objective", "") or "")
        if not model_path or not objective:
            continue
        path = Path(model_path)
        if not path.exists():
            continue
        stress_objective = _objective_report(stress_payload, objective)
        robustness_objective = _objective_report(robustness_payload, objective)
        stress_accepted = bool(stress_objective.get("accepted")) if stress_objective else bool(stress_report.accepted)
        robustness_accepted = (
            bool(robustness_objective.get("accepted"))
            if robustness_objective
            else bool(robustness_report.accepted)
        )
        portfolio_accepted = bool(
            portfolio_report.accepted
            and symbol.upper() in {str(item).upper() for item in portfolio_report.accepted_symbols}
        )
        learning_blocked = bool(learning_feedback and learning_feedback.get("blocks_promotion"))
        data_coverage_passed = not _data_coverage_blocks_promotion(data_coverage)
        walk_forward_gate = getattr(outcome, "walk_forward_gate", None)
        walk_forward_payload = dict(walk_forward_gate) if isinstance(walk_forward_gate, dict) else {}
        walk_forward_accepted = _walk_forward_gate_passed(walk_forward_payload)
        try:
            model = load_model(path, expected_feature_version=None, expected_feature_dim=None, expected_feature_signature=None)
            model.execution_validation = {
                "passed": bool(
                    stress_accepted
                    and robustness_accepted
                    and portfolio_accepted
                    and walk_forward_accepted
                    and not learning_blocked
                    and data_coverage_passed
                ),
                "source": "model_lab",
                "symbol": symbol.upper(),
                "market_type": market_type,
                "interval": interval,
                "objective": objective,
                "walk_forward_gate": walk_forward_payload,
                "liquidity": liquidity.asdict(),
                "symbol_execution_profile": (
                    stress_profile.asdict()
                    if hasattr(stress_profile, "asdict")
                    else None
                ),
                "stress": {
                    "accepted": bool(stress_accepted),
                    "suite_accepted": bool(stress_report.accepted),
                    "scenario_count": int(getattr(stress_report, "scenario_count", 0)),
                    "accepted_objectives": int(getattr(stress_report, "accepted_objectives", 0)),
                    "worst_realized_pnl": float(getattr(stress_report, "worst_realized_pnl", 0.0)),
                    "worst_max_drawdown": float(getattr(stress_report, "worst_max_drawdown", 0.0)),
                    "report_path": str(stress_report_path),
                    "objective": stress_objective,
                },
                "temporal_robustness": {
                    "accepted": bool(robustness_accepted),
                    "suite_accepted": bool(robustness_report.accepted),
                    "window_count": int(getattr(robustness_report, "window_count", 0)),
                    "accepted_windows": int(getattr(robustness_report, "accepted_windows", 0)),
                    "accepted_window_rate": float(getattr(robustness_report, "accepted_window_rate", 0.0)),
                    "worst_realized_pnl": float(getattr(robustness_report, "worst_realized_pnl", 0.0)),
                    "worst_max_drawdown": float(getattr(robustness_report, "worst_max_drawdown", 0.0)),
                    "statistical_edge_accepted": bool(getattr(robustness_report, "statistical_edge_accepted", False)),
                    "report_path": str(robustness_report_path),
                    "objective": robustness_objective,
                },
                "portfolio": {
                    "accepted": portfolio_accepted,
                    "suite_accepted": bool(portfolio_report.accepted),
                    "report_path": str(portfolio_report_path),
                    "reason": portfolio_report.reason,
                    "accepted_symbols": list(portfolio_report.accepted_symbols),
                    "effective_symbol_count": float(portfolio_report.effective_symbol_count),
                    "correlation_adjusted_effective_symbol_count": float(
                        portfolio_report.correlation_adjusted_effective_symbol_count
                    ),
                    "portfolio_cvar_95": float(portfolio_report.portfolio_cvar_95),
                    "portfolio_max_drawdown": float(portfolio_report.portfolio_max_drawdown),
                    "max_pairwise_correlation": float(portfolio_report.max_pairwise_correlation),
                    "max_cluster_weight": float(portfolio_report.max_cluster_weight),
                },
                "learning_feedback": learning_feedback,
                "data_coverage": data_coverage.asdict() if data_coverage is not None else None,
            }
            serialize_model(model, path)
        except (ModelLoadError, OSError, ValueError):
            continue


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
            "correlation_adjusted_effective_symbol_count": report.correlation_adjusted_effective_symbol_count,
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
    learning_feedback_path: Path | None = None,
    full_history: bool = False,
    market_db_path: Path | None = None,
    require_db_data: bool = False,
) -> ModelLabReport:
    """Rank liquid symbols, train all risk objectives, and write a lab report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if require_db_data and market_db_path is None:
        market_db_path = Path("data/market_data.sqlite")
    if market_db_path is not None and full_history:
        raise ValueError("--market-db and --full-history are mutually exclusive data sources")
    data_source = "sqlite_market_data" if market_db_path is not None else ("binance_full_history" if full_history else "binance_recent_limit")
    learning_report, learning_source_path = _load_model_lab_learning_feedback(learning_feedback_path)
    learning_summary = _learning_feedback_summary(learning_report, learning_source_path)
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
    execution_stamp_contexts: list[_ExecutionStampContext] = []
    for item in universe.eligible:
        symbol = item.symbol
        symbol_dir = output_dir / _safe_symbol_path(symbol)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        candles: list[Candle] = []
        try:
            candles = (
                _candles_for_symbol_from_db(
                    symbol,
                    market_type=runtime.market_type,
                    interval=runtime.interval,
                    db_path=market_db_path,
                )
                if market_db_path is not None
                else _candles_for_symbol_full_history(
                    client,
                    symbol,
                    runtime.interval,
                    batch_size=max(1, int(limit)),
                )
                if full_history
                else _candles_for_symbol(client, symbol, runtime.interval, limit)
            )
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
            data_coverage = describe_candle_coverage(
                symbol=symbol,
                market_type=runtime.market_type,
                interval=runtime.interval,
                available_candles=candles,
                used_candles=candles,
                rows_used=int(getattr(suite, "total_rows", 0)),
                source_scope=data_source,
            )
            learning_feedback = _symbol_learning_feedback(
                learning_report,
                symbol,
                stress_report,
                robustness_report,
            )
            execution_stamp_contexts.append(_ExecutionStampContext(
                suite=suite,
                symbol=symbol,
                market_type=runtime.market_type,
                interval=runtime.interval,
                liquidity=liquidity,
                stress_profile=stress_profile,
                stress_report=stress_report,
                stress_report_path=stress_report_path,
                robustness_report=robustness_report,
                robustness_report_path=robustness_report_path,
                learning_feedback=learning_feedback,
                data_coverage=data_coverage,
            ))
            outcomes.append(_outcome_from_suite(
                symbol,
                suite,
                liquidity,
                stress_report,
                stress_report_path,
                robustness_report,
                robustness_report_path,
                learning_feedback=learning_feedback,
                data_coverage=data_coverage,
            ))
        except TrainingSuiteRejected as exc:
            data_coverage = (
                describe_candle_coverage(
                    symbol=symbol,
                    market_type=runtime.market_type,
                    interval=runtime.interval,
                    available_candles=candles,
                    used_candles=candles,
                    rows_used=int(exc.row_count),
                    source_scope=data_source,
                ).asdict()
                if candles
                else None
            )
            outcomes.append(SymbolResearchOutcome(
                symbol=symbol,
                accepted=False,
                rows=int(exc.row_count),
                objectives=list(objectives),
                error=str(exc),
                liquidity=item.asdict(),
                data_coverage=data_coverage,
                diagnostics=exc.diagnostics,
            ))
        except (BinanceAPIError, ValueError, OSError) as exc:
            data_coverage = (
                describe_candle_coverage(
                    symbol=symbol,
                    market_type=runtime.market_type,
                    interval=runtime.interval,
                    available_candles=candles,
                    used_candles=candles,
                    rows_used=0,
                    source_scope=data_source,
                ).asdict()
                if candles
                else None
            )
            outcomes.append(SymbolResearchOutcome(
                symbol=symbol,
                accepted=False,
                rows=0,
                objectives=list(objectives),
                error=str(exc),
                liquidity=item.asdict(),
                data_coverage=data_coverage,
            ))
    portfolio_report = _apply_portfolio_risk_gate(
        outcomes,
        candles_by_symbol,
        strategy,
        min_symbols=universe.min_required,
    )
    portfolio_report_path = output_dir / "portfolio_risk.json"
    report_path = output_dir / "model_lab_report.json"
    preliminary_report = ModelLabReport(
        quote_asset=runtime.quote_asset,
        interval=runtime.interval,
        market_type=runtime.market_type,
        data_source=data_source,
        market_db_path=(str(market_db_path) if market_db_path is not None else None),
        require_db_data=bool(require_db_data),
        requested_objectives=list(objectives),
        universe=universe.asdict(),
        outcomes=outcomes,
        output_dir=str(output_dir),
        report_path=str(report_path),
        portfolio_risk={
            **portfolio_report.asdict(),
            "report_path": str(portfolio_report_path),
        },
        learning_feedback=learning_summary,
    )
    sanity_report = build_model_lab_financial_sanity_report(
        preliminary_report.asdict(),
        source="model_lab",
    )
    sanity_blocks = blocking_reasons(sanity_report)
    if sanity_blocks:
        for outcome in outcomes:
            if not outcome.accepted:
                continue
            outcome.accepted = False
            outcome.error = "financial_sanity_failed"
            outcome.diagnostics = {
                **(outcome.diagnostics or {}),
                "financial_sanity_blocking_reasons": sanity_blocks[:8],
            }
        portfolio_report = replace(
            portfolio_report,
            accepted=False,
            reason="financial_sanity_failed",
            accepted_symbols=[],
        )
    write_json_atomic(portfolio_report_path, portfolio_report.asdict(), indent=2, sort_keys=True)
    for context in execution_stamp_contexts:
        _stamp_model_execution_validation(
            context.suite,
            symbol=context.symbol,
            market_type=context.market_type,
            interval=context.interval,
            liquidity=context.liquidity,
            stress_profile=context.stress_profile,
            stress_report=context.stress_report,
            stress_report_path=context.stress_report_path,
            robustness_report=context.robustness_report,
            robustness_report_path=context.robustness_report_path,
            portfolio_report=portfolio_report,
            portfolio_report_path=portfolio_report_path,
            learning_feedback=context.learning_feedback,
            data_coverage=context.data_coverage,
        )
    report = ModelLabReport(
        quote_asset=runtime.quote_asset,
        interval=runtime.interval,
        market_type=runtime.market_type,
        data_source=data_source,
        market_db_path=(str(market_db_path) if market_db_path is not None else None),
        require_db_data=bool(require_db_data),
        requested_objectives=list(objectives),
        universe=universe.asdict(),
        outcomes=outcomes,
        output_dir=str(output_dir),
        report_path=str(report_path),
        portfolio_risk={
            **portfolio_report.asdict(),
            "report_path": str(portfolio_report_path),
        },
        learning_feedback=learning_summary,
        financial_sanity=sanity_report.asdict(),
    )
    write_json_atomic(report_path, report.asdict(), indent=2)
    return report
