"""Two-pass, source-bound development ablation for Round 21 local AI vetoes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Callable

from .polymarket_ai_veto import (
    POLYMARKET_AI_CASE_SCHEMA_VERSION,
    PolymarketAIVetoCache,
    PolymarketAIVetoCase,
    PolymarketAIVetoConfig,
    PolymarketAIVetoReport,
    benchmark_polymarket_ai_veto,
    polymarket_ai_model_evidence,
    unload_polymarket_ai_model,
)
from .polymarket_round21_ai import (
    POLYMARKET_ROUND21_AI_HISTORICAL_CASE_SCHEMA_VERSION,
    POLYMARKET_ROUND21_AI_HISTORICAL_SCHEDULE_DESIGN_SHA256,
    round21_permissions_from_ai_report,
)
from .polymarket_round21_ai_comparison import (
    POLYMARKET_ROUND21_AI_COMPARISON_SCHEMA_VERSION,
    Round21AIMatchedComparison,
    compare_round21_ai_replay_matrices,
)
from .polymarket_round21_ai_selection import POLYMARKET_ROUND21_AI_CANDIDATES
from .polymarket_round21_ai_selection import (
    Round21AICandidateSelection,
    select_round21_ai_candidate,
)
from .polymarket_round21_dataset import Round21PartitionPolicy
from .polymarket_round21_economic_operator import (
    Round21DevelopmentEconomicResult,
    replay_round21_development_economics,
    replay_round21_development_economics_with_ai_cases,
)
from .polymarket_round21_model import Round21DevelopmentPanel
from .polymarket_round21_replay import (
    Round21EconomicMatrixAccumulator,
    Round21ReplayCondition,
)


POLYMARKET_ROUND21_DEVELOPMENT_AI_REPLAY_SCHEMA_VERSION = (
    "polymarket-round21-development-ai-replay-v1"
)
POLYMARKET_ROUND21_DEVELOPMENT_AI_BENCHMARK_SCHEMA_VERSION = (
    "polymarket-round21-development-ai-benchmark-v1"
)
POLYMARKET_ROUND21_DEVELOPMENT_AI_PROGRAM_SCHEMA_VERSION = (
    "polymarket-round21-development-ai-program-v1"
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ProgressCallback = Callable[[str, Mapping[str, object]], None]
ModelUnloader = Callable[[PolymarketAIVetoConfig], None]
ModelEvidenceReader = Callable[[PolymarketAIVetoConfig], tuple[str, str]]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _digest(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None or selected == _EMPTY_SHA256:
        raise ValueError(f"Round 21 {name} identity is invalid")
    return selected


def _case_set_sha256(cases: Sequence[PolymarketAIVetoCase]) -> str:
    return _canonical_sha256(
        {
            "schema_version": POLYMARKET_AI_CASE_SCHEMA_VERSION,
            "case_sha256": [value.case_sha256 for value in cases],
        }
    )


@dataclass(frozen=True, slots=True)
class Round21DevelopmentAIBenchmarkResult:
    development_economic_result_sha256: str
    development_model_artifact_sha256: str
    selected_population_layer: str
    risk_benchmark_evidence_sha256: str
    historical_case_set_sha256: str
    selection_sha256: str
    cases: tuple[PolymarketAIVetoCase, ...]
    reports: tuple[PolymarketAIVetoReport, ...]
    result_sha256: str
    ai_model_selected: bool = False
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                POLYMARKET_ROUND21_DEVELOPMENT_AI_BENCHMARK_SCHEMA_VERSION
            ),
            "historical_schedule_design_sha256": (
                POLYMARKET_ROUND21_AI_HISTORICAL_SCHEDULE_DESIGN_SHA256
            ),
            "development_economic_result_sha256": (
                self.development_economic_result_sha256
            ),
            "development_model_artifact_sha256": (
                self.development_model_artifact_sha256
            ),
            "selected_population_layer": self.selected_population_layer,
            "risk_benchmark_evidence_sha256": (self.risk_benchmark_evidence_sha256),
            "historical_case_set_sha256": self.historical_case_set_sha256,
            "selection_sha256": self.selection_sha256,
            "case_sha256": [value.case_sha256 for value in self.cases],
            "report_sha256": [value.report_sha256 for value in self.reports],
            "model": [value.config.model for value in self.reports],
            "model_digest": [value.model_digest for value in self.reports],
            "ai_model_selected": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        selected = self.validated()
        return {
            **selected.identity_payload(),
            "cases": [value.asdict() for value in selected.cases],
            "reports": [value.asdict() for value in selected.reports],
            "result_sha256": selected.result_sha256,
        }

    def validated(self) -> Round21DevelopmentAIBenchmarkResult:
        cases = tuple(self.cases)
        reports = tuple(self.reports)
        digests = (
            self.development_economic_result_sha256,
            self.development_model_artifact_sha256,
            self.risk_benchmark_evidence_sha256,
            self.historical_case_set_sha256,
            self.selection_sha256,
        )
        if (
            self.selected_population_layer
            not in {
                "core",
                "core_spot",
                "core_spot_usdm",
            }
            or tuple(value.config.model for value in reports)
            != POLYMARKET_ROUND21_AI_CANDIDATES
            or not cases
            or len({value.condition_id for value in cases}) != len(cases)
            or any(
                value.prompt_payload.get("schema_version")
                != POLYMARKET_ROUND21_AI_HISTORICAL_CASE_SCHEMA_VERSION
                or value.case_sha256 != _canonical_sha256(value.identity_payload())
                for value in cases
            )
            or self.historical_case_set_sha256 != _case_set_sha256(cases)
            or len({value.model_digest for value in reports}) != len(reports)
            or any(
                value.selection_sha256 != self.selection_sha256
                or value.risk_benchmark_evidence_sha256
                != self.risk_benchmark_evidence_sha256
                or value.profitability_claim
                or value.trading_authority
                for value in reports
            )
            or any(
                _report_permissions_differ(cases=cases, report=value)
                for value in reports
            )
            or any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in digests
            )
            or self.ai_model_selected
            or self.profitability_claim
            or self.paper_trading_authority
            or self.live_trading_authority
            or self.result_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 development AI benchmark result differs")
        return self


def _report_permissions_differ(
    *,
    cases: Sequence[PolymarketAIVetoCase],
    report: PolymarketAIVetoReport,
) -> bool:
    try:
        round21_permissions_from_ai_report(cases=cases, report=report)
    except (KeyError, TypeError, ValueError):
        return True
    return False


def round21_development_ai_selection_sha256(
    *,
    economic_result: Round21DevelopmentEconomicResult,
    development_model_artifact_sha256: str,
    selected_population_layer: str,
    risk_benchmark_evidence_sha256: str,
    cases: Sequence[PolymarketAIVetoCase],
) -> str:
    economic = economic_result.validated()
    model_sha = _digest(
        development_model_artifact_sha256,
        name="development model artifact",
    )

    risk_sha = _digest(
        risk_benchmark_evidence_sha256,
        name="AI risk benchmark evidence",
    )
    layer = str(selected_population_layer or "").strip()
    selected_cases = tuple(cases)
    if (
        layer != economic.selected_population_layer
        or not selected_cases
        or any(
            value.prompt_payload.get("schema_version")
            != POLYMARKET_ROUND21_AI_HISTORICAL_CASE_SCHEMA_VERSION
            for value in selected_cases
        )
        or len({value.condition_id for value in selected_cases}) != len(selected_cases)
        or len(selected_cases) != economic.source_condition_count
    ):
        raise ValueError("Round 21 development AI selection population differs")
    return _canonical_sha256(
        {
            "schema_version": (
                POLYMARKET_ROUND21_DEVELOPMENT_AI_BENCHMARK_SCHEMA_VERSION
            ),
            "historical_schedule_design_sha256": (
                POLYMARKET_ROUND21_AI_HISTORICAL_SCHEDULE_DESIGN_SHA256
            ),
            "development_economic_result_sha256": economic.result_sha256,
            "development_model_artifact_sha256": model_sha,
            "selected_population_layer": layer,
            "risk_benchmark_evidence_sha256": risk_sha,
            "historical_case_set_sha256": _case_set_sha256(selected_cases),
            "candidate_models": list(POLYMARKET_ROUND21_AI_CANDIDATES),
            "target_accessed": False,
            "ai_model_selected": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
    )


def preflight_round21_ai_candidate_models(
    *,
    configs: Sequence[PolymarketAIVetoConfig],
    expected_model_digests: Mapping[str, str],
    model_evidence_reader: ModelEvidenceReader = polymarket_ai_model_evidence,
    progress: ProgressCallback | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Verify the finite local model set before either exact source pass."""

    selected_configs = tuple(value.validated() for value in configs)
    models = tuple(value.model for value in selected_configs)
    expected_digests = {
        str(key): _digest(value, name=f"{key} model")
        for key, value in expected_model_digests.items()
    }
    if models != POLYMARKET_ROUND21_AI_CANDIDATES or set(expected_digests) != set(
        POLYMARKET_ROUND21_AI_CANDIDATES
    ):
        raise ValueError("Round 21 development AI preflight candidates differ")
    evidence: list[tuple[str, str, str]] = []
    for config in selected_configs:
        digest, metadata_sha256 = model_evidence_reader(config)
        observed_digest = _digest(digest, name=f"{config.model} observed model")
        metadata_digest = _digest(
            metadata_sha256,
            name=f"{config.model} model metadata",
        )
        if observed_digest != expected_digests[config.model]:
            raise ValueError(
                "Round 21 AI model digest differs from benchmark provenance"
            )
        evidence.append((config.model, observed_digest, metadata_digest))
        if progress is not None:
            progress(
                "ai_model_preflight",
                {
                    "model": config.model,
                    "model_digest": observed_digest,
                    "model_metadata_sha256": metadata_digest,
                },
            )
    return tuple(evidence)


def benchmark_round21_development_ai_candidates(
    *,
    economic_result: Round21DevelopmentEconomicResult,
    development_model_artifact_sha256: str,
    selected_population_layer: str,
    risk_benchmark_evidence_sha256: str,
    cases: Sequence[PolymarketAIVetoCase],
    configs: Sequence[PolymarketAIVetoConfig],
    expected_model_digests: Mapping[str, str],
    cache_store: PolymarketAIVetoCache | None = None,
    progress: ProgressCallback | None = None,
    model_unloader: ModelUnloader = unload_polymarket_ai_model,
) -> Round21DevelopmentAIBenchmarkResult:
    """Run the finite local-model program over one immutable historical case set."""

    economic = economic_result.validated()
    selected_cases = tuple(cases)
    selected_configs = tuple(value.validated() for value in configs)
    models = tuple(value.model for value in selected_configs)
    expected_digests = {
        str(key): _digest(value, name=f"{key} model")
        for key, value in expected_model_digests.items()
    }
    if models != POLYMARKET_ROUND21_AI_CANDIDATES or set(expected_digests) != set(
        POLYMARKET_ROUND21_AI_CANDIDATES
    ):
        raise ValueError("Round 21 development AI benchmark candidates differ")
    model_sha = _digest(
        development_model_artifact_sha256,
        name="development model artifact",
    )
    risk_sha = _digest(
        risk_benchmark_evidence_sha256,
        name="AI risk benchmark evidence",
    )
    selection_sha = round21_development_ai_selection_sha256(
        economic_result=economic,
        development_model_artifact_sha256=model_sha,
        selected_population_layer=selected_population_layer,
        risk_benchmark_evidence_sha256=risk_sha,
        cases=selected_cases,
    )
    reports: list[PolymarketAIVetoReport] = []
    condition_ids = tuple(value.condition_id for value in selected_cases)
    for config in selected_configs:
        if progress is not None:
            progress(
                "ai_candidate_started",
                {"model": config.model, "case_count": len(selected_cases)},
            )

        def candidate_progress(
            event: str,
            payload: Mapping[str, object],
            *,
            model: str = config.model,
        ) -> None:
            if progress is not None:
                progress(event, {"model": model, **payload})

        try:
            report = benchmark_polymarket_ai_veto(
                selected_cases,
                all_condition_ids=condition_ids,
                selection_sha256=selection_sha,
                risk_benchmark_evidence_sha256=risk_sha,
                config=config,
                progress=candidate_progress,
                cache_store=cache_store,
                expected_model_digest=expected_digests[config.model],
            )
        finally:
            model_unloader(config)
        reports.append(report)
        if progress is not None:
            progress("ai_candidate_unloaded", {"model": config.model})
    provisional = Round21DevelopmentAIBenchmarkResult(
        development_economic_result_sha256=economic.result_sha256,
        development_model_artifact_sha256=model_sha,
        selected_population_layer=selected_population_layer,
        risk_benchmark_evidence_sha256=risk_sha,
        historical_case_set_sha256=_case_set_sha256(selected_cases),
        selection_sha256=selection_sha,
        cases=selected_cases,
        reports=tuple(reports),
        result_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        result_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


@dataclass(frozen=True, slots=True)
class Round21DevelopmentAIReplayResult:
    development_economic_result_sha256: str
    baseline_matrix_sha256: str
    terminal_receipt_audit_sha256: str
    matched_population_sha256: str
    comparisons: tuple[Round21AIMatchedComparison, ...]
    result_sha256: str
    ai_model_selected: bool = False
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": (POLYMARKET_ROUND21_DEVELOPMENT_AI_REPLAY_SCHEMA_VERSION),
            "development_economic_result_sha256": (
                self.development_economic_result_sha256
            ),
            "baseline_matrix_sha256": self.baseline_matrix_sha256,
            "terminal_receipt_audit_sha256": self.terminal_receipt_audit_sha256,
            "matched_population_sha256": self.matched_population_sha256,
            "comparison_sha256": [
                value.comparison_sha256 for value in self.comparisons
            ],
            "candidate_count": len(self.comparisons),
            "ai_model_selected": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        selected = self.validated()
        return {
            **selected.identity_payload(),
            "comparisons": [
                {
                    **value.identity_payload(),
                    "deltas": [
                        {
                            **delta.identity_payload(),
                            "delta_sha256": delta.delta_sha256,
                        }
                        for delta in value.deltas
                    ],
                    "comparison_sha256": value.comparison_sha256,
                }
                for value in selected.comparisons
            ],
            "result_sha256": selected.result_sha256,
        }

    def validated(self) -> Round21DevelopmentAIReplayResult:
        comparisons = tuple(value.validated() for value in self.comparisons)
        digests = (
            self.development_economic_result_sha256,
            self.baseline_matrix_sha256,
            self.terminal_receipt_audit_sha256,
            self.matched_population_sha256,
        )
        if (
            not comparisons
            or len(comparisons) > 3
            or len({value.model for value in comparisons}) != len(comparisons)
            or len({value.model_digest for value in comparisons}) != len(comparisons)
            or any(
                value.baseline_matrix_sha256 != self.baseline_matrix_sha256
                or value.matched_population_sha256 != self.matched_population_sha256
                for value in comparisons
            )
            or any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in digests
            )
            or self.ai_model_selected
            or self.profitability_claim
            or self.paper_trading_authority
            or self.live_trading_authority
            or self.result_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 development AI replay result differs")
        return self


@dataclass(frozen=True, slots=True)
class Round21DevelopmentAIProgramResult:
    economic_result: Round21DevelopmentEconomicResult
    benchmark_result: Round21DevelopmentAIBenchmarkResult
    replay_result: Round21DevelopmentAIReplayResult
    candidate_selection: Round21AICandidateSelection
    result_sha256: str
    target_accessed: bool = False
    ai_model_selected: bool = False
    ai_edge_claim: bool = False
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                POLYMARKET_ROUND21_DEVELOPMENT_AI_PROGRAM_SCHEMA_VERSION
            ),
            "economic_result_sha256": self.economic_result.result_sha256,
            "benchmark_result_sha256": self.benchmark_result.result_sha256,
            "replay_result_sha256": self.replay_result.result_sha256,
            "candidate_selection_sha256": (self.candidate_selection.selection_sha256),
            "target_accessed": False,
            "ai_model_selected": False,
            "ai_edge_claim": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        selected = self.validated()
        return {
            **selected.identity_payload(),
            "economic_result": selected.economic_result.asdict(),
            "benchmark_result": selected.benchmark_result.asdict(),
            "replay_result": selected.replay_result.asdict(),
            "candidate_selection": {
                **selected.candidate_selection.identity_payload(),
                "scores": [
                    {
                        **score.identity_payload(),
                        "score_sha256": score.score_sha256,
                    }
                    for score in selected.candidate_selection.scores
                ],
                "selection_sha256": (selected.candidate_selection.selection_sha256),
            },
            "result_sha256": selected.result_sha256,
        }

    def validated(self) -> Round21DevelopmentAIProgramResult:
        economic = self.economic_result.validated()
        benchmark = self.benchmark_result.validated()
        replay = self.replay_result.validated()
        selection = self.candidate_selection.validated()
        report_models = tuple(value.config.model for value in benchmark.reports)
        comparison_models = tuple(value.model for value in replay.comparisons)
        report_digests = tuple(value.model_digest for value in benchmark.reports)
        comparison_digests = tuple(value.model_digest for value in replay.comparisons)
        if (
            benchmark.development_economic_result_sha256 != economic.result_sha256
            or replay.development_economic_result_sha256 != economic.result_sha256
            or report_models != POLYMARKET_ROUND21_AI_CANDIDATES
            or comparison_models != POLYMARKET_ROUND21_AI_CANDIDATES
            or report_digests != comparison_digests
            or selection != select_round21_ai_candidate(replay.comparisons)
            or selection.matched_population_sha256 != replay.matched_population_sha256
            or selection.baseline_matrix_sha256 != replay.baseline_matrix_sha256
            or any(
                (
                    self.target_accessed,
                    self.ai_model_selected,
                    self.ai_edge_claim,
                    self.profitability_claim,
                    self.paper_trading_authority,
                    self.live_trading_authority,
                )
            )
            or self.result_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 development AI program result differs")
        return self


@dataclass(slots=True)
class _CandidateMatrixSink:
    accumulator: Round21EconomicMatrixAccumulator
    matched_condition_sha256: list[str] | None = None

    def __call__(self, condition: Round21ReplayCondition) -> None:
        self.accumulator.observe(condition)
        if self.matched_condition_sha256 is not None:
            self.matched_condition_sha256.append(condition.matched_population_sha256())


def replay_round21_development_ai_ablation(
    *,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    partition_policy: Round21PartitionPolicy,
    development_panels: Sequence[Round21DevelopmentPanel],
    development_model_artifact: Mapping[str, object],
    core_publication_manifest_sha256: str,
    selected_population_layer: str,
    expected_development_economic_result: Round21DevelopmentEconomicResult,
    cases: Sequence[PolymarketAIVetoCase],
    reports: Sequence[PolymarketAIVetoReport],
    initial_capital_quote: Decimal = Decimal("10000"),
    minimum_edge_per_share: Decimal = Decimal("0.02"),
    builder_taker_fee_bps: Decimal = Decimal("0"),
    progress: ProgressCallback | None = None,
) -> Round21DevelopmentAIReplayResult:
    """Replay every finite AI candidate in one second source pass."""

    expected = expected_development_economic_result.validated()
    selected_cases = tuple(cases)
    selected_reports = tuple(reports)
    models = tuple(value.config.model for value in selected_reports)
    if (
        not selected_cases
        or not selected_reports
        or len(selected_reports) > 3
        or len(set(models)) != len(selected_reports)
        or not set(models).issubset(POLYMARKET_ROUND21_AI_CANDIDATES)
    ):
        raise ValueError("Round 21 development AI candidate population differs")
    permissions = tuple(
        round21_permissions_from_ai_report(cases=selected_cases, report=report)
        for report in selected_reports
    )
    if len({value.model_digest for value in selected_reports}) != len(selected_reports):
        raise ValueError("Round 21 development AI model digests are duplicated")
    accumulators = tuple(
        Round21EconomicMatrixAccumulator(
            initial_capital_quote=initial_capital_quote,
            minimum_edge_per_share=minimum_edge_per_share,
            builder_taker_fee_bps=builder_taker_fee_bps,
            directional_permissions=value,
        )
        for value in permissions
    )
    matched_condition_sha256: list[str] = []
    sinks = tuple(
        _CandidateMatrixSink(
            accumulator=value,
            matched_condition_sha256=(matched_condition_sha256 if index == 0 else None),
        )
        for index, value in enumerate(accumulators)
    )
    if progress is not None:
        progress(
            "ai_replay_prepared",
            {
                "candidate_count": len(selected_reports),
                "condition_count": len(selected_cases),
                "ledger_count": len(selected_reports) * 81,
            },
        )
    baseline = replay_round21_development_economics(
        source_database=source_database,
        terminal_transport_manifest=terminal_transport_manifest,
        partition_policy=partition_policy,
        development_panels=development_panels,
        development_model_artifact=development_model_artifact,
        core_publication_manifest_sha256=core_publication_manifest_sha256,
        selected_population_layer=selected_population_layer,
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
        selected_condition_sinks=sinks,
        progress=progress,
    )
    if baseline != expected:
        raise ValueError("Round 21 development AI baseline replay differs")
    matrices = tuple(value.finish() for value in accumulators)
    matched_population_sha256 = _canonical_sha256(
        {
            "schema_version": POLYMARKET_ROUND21_AI_COMPARISON_SCHEMA_VERSION,
            "condition_sha256": matched_condition_sha256,
        }
    )
    comparisons = tuple(
        compare_round21_ai_replay_matrices(
            baseline_matrix=baseline.selected_matrix,
            ai_matrix=matrix,
            cases=selected_cases,
            report=report,
            matched_population_sha256=matched_population_sha256,
        )
        for matrix, report in zip(matrices, selected_reports, strict=True)
    )
    provisional = Round21DevelopmentAIReplayResult(
        development_economic_result_sha256=baseline.result_sha256,
        baseline_matrix_sha256=comparisons[0].baseline_matrix_sha256,
        terminal_receipt_audit_sha256=baseline.terminal_receipt_audit_sha256,
        matched_population_sha256=matched_population_sha256,
        comparisons=comparisons,
        result_sha256=_EMPTY_SHA256,
    )
    result = replace(
        provisional,
        result_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()
    if progress is not None:
        progress(
            "ai_replay_complete",
            {
                "candidate_count": len(comparisons),
                "result_sha256": result.result_sha256,
                "development_qualified_count": sum(
                    value.development_qualified for value in comparisons
                ),
            },
        )
    return result


def complete_round21_development_ai_program(
    *,
    economic_result: Round21DevelopmentEconomicResult,
    benchmark_result: Round21DevelopmentAIBenchmarkResult,
    replay_result: Round21DevelopmentAIReplayResult,
) -> Round21DevelopmentAIProgramResult:
    """Bind the two development passes and nominate at most one challenger."""

    economic = economic_result.validated()
    benchmark = benchmark_result.validated()
    replay = replay_result.validated()
    selection = select_round21_ai_candidate(replay.comparisons)
    provisional = Round21DevelopmentAIProgramResult(
        economic_result=economic,
        benchmark_result=benchmark,
        replay_result=replay,
        candidate_selection=selection,
        result_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        result_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def run_round21_development_ai_program(
    *,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    partition_policy: Round21PartitionPolicy,
    development_panels: Sequence[Round21DevelopmentPanel],
    development_model_artifact: Mapping[str, object],
    core_publication_manifest_sha256: str,
    selected_population_layer: str,
    risk_benchmark_evidence_sha256: str,
    configs: Sequence[PolymarketAIVetoConfig],
    expected_model_digests: Mapping[str, str],
    cache_store: PolymarketAIVetoCache | None = None,
    initial_capital_quote: Decimal = Decimal("10000"),
    minimum_edge_per_share: Decimal = Decimal("0.02"),
    builder_taker_fee_bps: Decimal = Decimal("0"),
    progress: ProgressCallback | None = None,
) -> Round21DevelopmentAIProgramResult:
    """Run one case-building pass, all AI calls, then one matched replay pass."""

    artifact_sha256 = _digest(
        development_model_artifact.get("artifact_sha256"),
        name="development model artifact",
    )
    preflight_round21_ai_candidate_models(
        configs=configs,
        expected_model_digests=expected_model_digests,
        progress=progress,
    )
    economic, cases = replay_round21_development_economics_with_ai_cases(
        source_database=source_database,
        terminal_transport_manifest=terminal_transport_manifest,
        partition_policy=partition_policy,
        development_panels=development_panels,
        development_model_artifact=development_model_artifact,
        core_publication_manifest_sha256=core_publication_manifest_sha256,
        selected_population_layer=selected_population_layer,
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
        progress=progress,
    )
    benchmark = benchmark_round21_development_ai_candidates(
        economic_result=economic,
        development_model_artifact_sha256=artifact_sha256,
        selected_population_layer=selected_population_layer,
        risk_benchmark_evidence_sha256=risk_benchmark_evidence_sha256,
        cases=cases,
        configs=configs,
        expected_model_digests=expected_model_digests,
        cache_store=cache_store,
        progress=progress,
    )
    replay = replay_round21_development_ai_ablation(
        source_database=source_database,
        terminal_transport_manifest=terminal_transport_manifest,
        partition_policy=partition_policy,
        development_panels=development_panels,
        development_model_artifact=development_model_artifact,
        core_publication_manifest_sha256=core_publication_manifest_sha256,
        selected_population_layer=selected_population_layer,
        expected_development_economic_result=economic,
        cases=benchmark.cases,
        reports=benchmark.reports,
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
        progress=progress,
    )
    return complete_round21_development_ai_program(
        economic_result=economic,
        benchmark_result=benchmark,
        replay_result=replay,
    )


credentials_used = False
account_connected = False
binance_execution_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_DEVELOPMENT_AI_BENCHMARK_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_DEVELOPMENT_AI_PROGRAM_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_DEVELOPMENT_AI_REPLAY_SCHEMA_VERSION",
    "Round21DevelopmentAIBenchmarkResult",
    "Round21DevelopmentAIProgramResult",
    "Round21DevelopmentAIReplayResult",
    "benchmark_round21_development_ai_candidates",
    "complete_round21_development_ai_program",
    "preflight_round21_ai_candidate_models",
    "replay_round21_development_ai_ablation",
    "round21_development_ai_selection_sha256",
    "run_round21_development_ai_program",
]
