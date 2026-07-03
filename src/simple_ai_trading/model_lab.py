"""Cross-symbol model research and profitability-gated optimization workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .api import BinanceAPIError, BinanceClient, Candle
from .market_universe import MarketEligibility, UniverseSelection, rank_high_liquidity_universe
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
) -> SymbolResearchOutcome:
    scores = {outcome.objective: float(outcome.best_score) for outcome in suite.outcomes}
    hybrid_profiles = {outcome.objective: str(outcome.hybrid_profile) for outcome in suite.outcomes}
    accepted = bool(suite.outcomes) and all(score > 0.0 for score in scores.values())
    return SymbolResearchOutcome(
        symbol=symbol,
        accepted=accepted,
        rows=int(suite.total_rows),
        objectives=list(suite.objectives_run),
        report_path=str(suite.summary_path),
        liquidity=liquidity.asdict(),
        objective_scores=scores,
        hybrid_profiles=hybrid_profiles,
    )


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
    for item in universe.eligible:
        symbol = item.symbol
        symbol_dir = output_dir / _safe_symbol_path(symbol)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        try:
            candles = _candles_for_symbol(client, symbol, runtime.interval, limit)
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
            )
            outcomes.append(_outcome_from_suite(symbol, suite, liquidity_by_symbol[symbol]))
        except TrainingSuiteRejected as exc:
            outcomes.append(SymbolResearchOutcome(
                symbol=symbol,
                accepted=False,
                rows=int(exc.row_count),
                objectives=list(objectives),
                error=str(exc),
                liquidity=item.asdict(),
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
    )
    write_json_atomic(report_path, report.asdict(), indent=2)
    return report
