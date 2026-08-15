"""Concrete one-use terminal evaluator for Polymarket Round 21."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from .polymarket_ai_veto import (
    POLYMARKET_AI_REPORT_SCHEMA_VERSION,
    PolymarketAIVetoCache,
    PolymarketAIVetoReport,
    benchmark_polymarket_ai_veto,
    polymarket_ai_model_evidence,
    unload_polymarket_ai_model,
)
from .polymarket_round21_ai import round21_permissions_from_ai_report
from .polymarket_round21_ai_comparison import (
    POLYMARKET_ROUND21_AI_COMPARISON_SCHEMA_VERSION,
    Round21AIMatchedComparison,
    compare_round21_ai_replay_matrices,
)
from .polymarket_round21_core_features import POLYMARKET_ROUND21_FEATURE_SCHEMA
from .polymarket_round21_corpus_store import (
    load_round21_sealed_core_partition_snapshots,
    validate_round21_core_publication_boundary,
)
from .polymarket_round21_dataset import (
    Round21PartitionPolicy,
    build_round21_sealed_test_panel,
)
from .polymarket_round21_economic_operator import (
    ProgressCallback,
    Round21SealedEconomicReplayEvidence,
    replay_round21_sealed_economics_with_ai_cases,
)
from .polymarket_round21_model import (
    Round21DevelopmentPanel,
    validate_round21_development_artifact,
)
from .polymarket_round21_one_use import (
    Round21OneUseClaim,
    Round21PretestManifest,
    execute_round21_one_use,
)
from .polymarket_round21_operator import (
    apply_round21_optional_binance_features,
    build_round21_core_causal_rows,
    load_round21_official_outcomes,
)
from .polymarket_round21_replay import (
    Round21EconomicMatrixAccumulator,
    Round21ReplayCondition,
)
from .polymarket_round21_sealed import (
    Round21SealedEvaluationResult,
    build_round21_sealed_evaluation_result,
    evaluate_round21_sealed_economics,
    evaluate_round21_sealed_predictions,
)
from .polymarket_round21_sidecar_replay import (
    replay_round21_optional_binance_features,
)
from .polymarket_round21_sidecar_terminal import (
    validate_round21_sidecar_terminal_manifest,
)
from .polymarket_round21_terminal import (
    validate_round21_terminal_transport_manifest,
)


POLYMARKET_ROUND21_SEALED_OPERATOR_SCHEMA_VERSION = (
    "polymarket-round21-terminal-sealed-operator-v1"
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_LAYERS = ("core", "core_spot", "core_spot_usdm")


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _validated_ai_report_identity(
    report: PolymarketAIVetoReport,
) -> PolymarketAIVetoReport:
    payload = report.asdict()
    claimed_sha256 = str(payload.pop("report_sha256", "")).strip().lower()
    validated_config = report.config.validated()
    if (
        report.schema_version != POLYMARKET_AI_REPORT_SCHEMA_VERSION
        or validated_config != report.config
        or len(claimed_sha256) != 64
        or any(value not in "0123456789abcdef" for value in claimed_sha256)
        or claimed_sha256 != report.report_sha256
        or claimed_sha256 != _canonical_sha256(payload)
        or report.advisory_only is not True
        or report.trading_authority
        or report.profitability_claim
    ):
        raise ValueError("Round 21 sealed AI report identity differs")
    return report


@dataclass(frozen=True, slots=True)
class Round21SealedTestAssembly:
    publication_manifest_sha256: str
    sealed_test_population_manifest_sha256: str
    terminal_transport_manifest_sha256: str
    sidecar_terminal_manifest_sha256: str | None
    partition_policy: Round21PartitionPolicy
    test_panel: Round21DevelopmentPanel


@dataclass(frozen=True, slots=True)
class Round21SealedOperatorOutcome:
    result: Round21SealedEvaluationResult
    replay_evidence: Round21SealedEconomicReplayEvidence
    ai_report: PolymarketAIVetoReport | None
    ai_comparison: Round21AIMatchedComparison | None


@dataclass(slots=True)
class _AIMatrixSink:
    accumulator: Round21EconomicMatrixAccumulator
    matched_condition_sha256: list[str]

    def __call__(self, condition: Round21ReplayCondition) -> None:
        self.accumulator.observe(condition)
        self.matched_condition_sha256.append(condition.matched_population_sha256())


def assemble_round21_sealed_test(
    *,
    publication_directory: str | Path,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    one_use_store_path: str | Path,
    claim: Round21OneUseClaim,
    test_access_sha256: str,
    selected_population_layer: str,
    sidecar_database: str | Path | None = None,
    sidecar_terminal_manifest: Mapping[str, object] | None = None,
) -> Round21SealedTestAssembly:
    """Load the physically separate test partition only after access consumption."""

    selected_claim = claim.validated()
    layer = str(selected_population_layer or "").strip()
    if layer not in _LAYERS or layer != selected_claim.selected_population_layer:
        raise ValueError("Round 21 sealed selected layer differs")
    publication = validate_round21_core_publication_boundary(publication_directory)
    transport = validate_round21_terminal_transport_manifest(
        terminal_transport_manifest
    )
    if (
        publication["terminal_transport_manifest_sha256"]
        != transport["manifest_sha256"]
    ):
        raise ValueError("Round 21 sealed publication and transport differ")
    sealed_manifest = publication["sealed_test_partition"]
    if not isinstance(sealed_manifest, Mapping):
        raise ValueError("Round 21 sealed partition manifest is unavailable")
    root = Path(publication_directory).resolve()
    sealed_database = root / str(publication["sealed_test_database"])
    snapshots = load_round21_sealed_core_partition_snapshots(
        sealed_database,
        sealed_manifest,
        one_use_store_path=one_use_store_path,
        claim=selected_claim,
        test_access_sha256=test_access_sha256,
    )
    policy = Round21PartitionPolicy.create(
        campaign_start_ms=int(sealed_manifest["campaign_start_ms"]),
        campaign_end_ms=int(sealed_manifest["campaign_end_ms"]),
    )
    rows = build_round21_core_causal_rows(snapshots)
    sidecar_sha: str | None = None
    optional_required = layer != "core"
    if optional_required != (sidecar_database is not None):
        raise ValueError("Round 21 sealed optional sidecar inputs differ")
    if optional_required != (sidecar_terminal_manifest is not None):
        raise ValueError("Round 21 sealed optional sidecar inputs differ")
    if optional_required:
        sidecar = validate_round21_sidecar_terminal_manifest(
            sidecar_terminal_manifest or {}
        )
        if (
            int(sidecar["campaign_start_ms"]) != policy.campaign_start_ms
            or int(sidecar["campaign_end_ms"]) != policy.campaign_end_ms
        ):
            raise ValueError("Round 21 sealed sidecar campaign differs")
        replay = replay_round21_optional_binance_features(
            source_database=sidecar_database,
            terminal_manifest=sidecar,
            decision_times_ms=tuple(row.decision_time_ms for row in rows),
        )
        rows = apply_round21_optional_binance_features(rows, replay)
        sidecar_sha = replay.terminal_manifest_sha256
    event_starts = {row.condition_id: row.event_start_ms for row in rows}
    outcomes = load_round21_official_outcomes(
        source_database=source_database,
        terminal_transport_manifest=transport,
        condition_event_starts=event_starts,
    )
    panel = build_round21_sealed_test_panel(
        feature_schema=POLYMARKET_ROUND21_FEATURE_SCHEMA,
        partition_policy=policy,
        feature_rows=rows,
        outcomes=outcomes,
        claim_sha256=selected_claim.claim_sha256,
        test_access_sha256=test_access_sha256,
        sealed_test_population_manifest_sha256=(
            selected_claim.sealed_test_population_manifest_sha256
        ),
    )
    return Round21SealedTestAssembly(
        publication_manifest_sha256=str(publication["manifest_sha256"]),
        sealed_test_population_manifest_sha256=str(
            publication["sealed_test_population_manifest_sha256"]
        ),
        terminal_transport_manifest_sha256=str(transport["manifest_sha256"]),
        sidecar_terminal_manifest_sha256=sidecar_sha,
        partition_policy=policy,
        test_panel=panel,
    )


def _sealed_ai_comparison(
    *,
    baseline: Round21SealedEconomicReplayEvidence,
    development_report: PolymarketAIVetoReport,
    development_comparison: Round21AIMatchedComparison,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    assembly: Round21SealedTestAssembly,
    model_artifact: Mapping[str, object],
    claim: Round21OneUseClaim,
    test_access_sha256: str,
    initial_capital_quote: Decimal,
    minimum_edge_per_share: Decimal,
    builder_taker_fee_bps: Decimal,
    cache_store: PolymarketAIVetoCache | None,
    progress: ProgressCallback | None,
) -> tuple[PolymarketAIVetoReport, Round21AIMatchedComparison]:
    selected_claim = claim.validated()
    development = development_comparison.validated()
    _validated_ai_report_identity(development_report)
    if (
        development.model != selected_claim.nominated_ai_model
        or development.model_digest != selected_claim.nominated_ai_model_digest
        or development.ai_report_sha256 != development_report.report_sha256
        or development_report.config.model != development.model
        or development_report.model_digest != development.model_digest
    ):
        raise ValueError("Round 21 sealed AI development nomination differs")
    observed_digest, _metadata = polymarket_ai_model_evidence(development_report.config)
    if observed_digest != development.model_digest:
        raise ValueError("Round 21 sealed AI model digest differs")
    cases = baseline.historical_ai_cases
    selection_sha256 = _canonical_sha256(
        {
            "schema_version": POLYMARKET_ROUND21_SEALED_OPERATOR_SCHEMA_VERSION,
            "claim_sha256": selected_claim.claim_sha256,
            "test_access_sha256": test_access_sha256,
            "development_ai_comparison_sha256": development.comparison_sha256,
            "case_sha256": [value.case_sha256 for value in cases],
        }
    )
    try:
        report = benchmark_polymarket_ai_veto(
            cases,
            all_condition_ids=tuple(value.condition_id for value in cases),
            selection_sha256=selection_sha256,
            risk_benchmark_evidence_sha256=(
                development_report.risk_benchmark_evidence_sha256
            ),
            config=development_report.config,
            cache_store=cache_store,
            expected_model_digest=development.model_digest,
            progress=progress,
        )
    finally:
        unload_polymarket_ai_model(development_report.config)
    permissions = round21_permissions_from_ai_report(cases=cases, report=report)
    accumulator = Round21EconomicMatrixAccumulator(
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
        directional_permissions=permissions,
    )
    condition_sha256: list[str] = []
    repeated = replay_round21_sealed_economics_with_ai_cases(
        source_database=source_database,
        terminal_transport_manifest=terminal_transport_manifest,
        partition_policy=assembly.partition_policy,
        test_panel=assembly.test_panel,
        development_model_artifact=model_artifact,
        core_publication_manifest_sha256=assembly.publication_manifest_sha256,
        claim_sha256=selected_claim.claim_sha256,
        test_access_sha256=test_access_sha256,
        sealed_test_population_manifest_sha256=(
            assembly.sealed_test_population_manifest_sha256
        ),
        selected_population_layer=selected_claim.selected_population_layer,
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
        selected_condition_sinks=(_AIMatrixSink(accumulator, condition_sha256),),
        progress=progress,
    )
    if repeated.replay_result != baseline.replay_result:
        raise ValueError("Round 21 sealed AI baseline replay differs")
    comparison = compare_round21_ai_replay_matrices(
        baseline_matrix=baseline.replay_result.selected_matrix,
        ai_matrix=accumulator.finish(),
        cases=cases,
        report=report,
        matched_population_sha256=_canonical_sha256(
            {
                "schema_version": POLYMARKET_ROUND21_AI_COMPARISON_SCHEMA_VERSION,
                "condition_sha256": condition_sha256,
            }
        ),
    )
    return report, comparison


def evaluate_round21_terminal_sealed_once(
    *,
    store_path: str | Path,
    claim: Round21OneUseClaim,
    pretest: Round21PretestManifest,
    publication_directory: str | Path,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    development_model_artifact: Mapping[str, object],
    development_ai_report: PolymarketAIVetoReport | None = None,
    development_ai_comparison: Round21AIMatchedComparison | None = None,
    sidecar_database: str | Path | None = None,
    sidecar_terminal_manifest: Mapping[str, object] | None = None,
    initial_capital_quote: Decimal = Decimal("10000"),
    minimum_edge_per_share: Decimal = Decimal("0.02"),
    builder_taker_fee_bps: Decimal = Decimal("0"),
    cache_store: PolymarketAIVetoCache | None = None,
    progress: ProgressCallback | None = None,
) -> Round21SealedOperatorOutcome:
    """Consume the test exactly once and produce deterministic plus AI verdicts."""

    selected_claim = claim.validated()
    selected_pretest = pretest.validated()
    artifact = validate_round21_development_artifact(development_model_artifact)
    ai_values = (development_ai_report, development_ai_comparison)
    publication = validate_round21_core_publication_boundary(publication_directory)
    transport = validate_round21_terminal_transport_manifest(
        terminal_transport_manifest
    )
    optional_required = selected_claim.selected_population_layer != "core"
    sidecar_sha256: str | None = None
    if optional_required:
        sidecar = validate_round21_sidecar_terminal_manifest(
            sidecar_terminal_manifest or {}
        )
        sidecar_sha256 = str(sidecar["manifest_sha256"])
    development_comparison = (
        None
        if development_ai_comparison is None
        else development_ai_comparison.validated()
    )
    validated_development_report = (
        None
        if development_ai_report is None
        else _validated_ai_report_identity(development_ai_report)
    )
    if (
        selected_claim.pretest_manifest_sha256 != selected_pretest.manifest_sha256
        or selected_claim.selected_population_layer
        != selected_pretest.selected_population_layer
        or selected_claim.sealed_test_population_manifest_sha256
        != selected_pretest.sealed_test_population_manifest_sha256
        or str(artifact["artifact_sha256"])
        != selected_pretest.development_model_artifact_sha256
        or str(publication["manifest_sha256"])
        != selected_pretest.core_corpus_publication_manifest_sha256
        or str(publication["sealed_test_population_manifest_sha256"])
        != selected_pretest.sealed_test_population_manifest_sha256
        or str(publication["terminal_transport_manifest_sha256"])
        != str(transport["manifest_sha256"])
        or optional_required != (sidecar_database is not None)
        or optional_required != (sidecar_terminal_manifest is not None)
        or sidecar_sha256 != selected_pretest.optional_campaign_terminal_sha256
        or any(value is None for value in ai_values)
        != all(value is None for value in ai_values)
        or (selected_claim.nominated_ai_model is None)
        != all(value is None for value in ai_values)
        or (
            development_comparison is not None
            and (
                development_comparison.comparison_sha256
                != selected_pretest.nominated_ai_comparison_sha256
                or development_comparison.model != selected_pretest.nominated_ai_model
                or development_comparison.model_digest
                != selected_pretest.nominated_ai_model_digest
                or validated_development_report is None
                or development_comparison.ai_report_sha256
                != validated_development_report.report_sha256
            )
        )
    ):
        raise ValueError("Round 21 sealed pretest identity differs")
    for path, name in (
        (Path(source_database), "source database"),
        *(
            ((Path(sidecar_database), "sidecar database"),)
            if sidecar_database is not None
            else ()
        ),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Round 21 sealed {name} is unavailable")
    if validated_development_report is not None and development_comparison is not None:
        observed_digest, _metadata = polymarket_ai_model_evidence(
            validated_development_report.config
        )
        if observed_digest != development_comparison.model_digest:
            raise ValueError(
                "Round 21 sealed AI model digest differs before test access"
            )
    captured: dict[str, object] = {}

    def evaluator(test_access_sha256: str) -> Round21SealedEvaluationResult:
        assembly = assemble_round21_sealed_test(
            publication_directory=publication_directory,
            source_database=source_database,
            terminal_transport_manifest=terminal_transport_manifest,
            one_use_store_path=store_path,
            claim=selected_claim,
            test_access_sha256=test_access_sha256,
            selected_population_layer=selected_claim.selected_population_layer,
            sidecar_database=sidecar_database,
            sidecar_terminal_manifest=sidecar_terminal_manifest,
        )
        predictive = evaluate_round21_sealed_predictions(
            artifact,
            population_layer=selected_claim.selected_population_layer,
            test_panel=assembly.test_panel,
        )
        replay = replay_round21_sealed_economics_with_ai_cases(
            source_database=source_database,
            terminal_transport_manifest=terminal_transport_manifest,
            partition_policy=assembly.partition_policy,
            test_panel=assembly.test_panel,
            development_model_artifact=artifact,
            core_publication_manifest_sha256=assembly.publication_manifest_sha256,
            claim_sha256=selected_claim.claim_sha256,
            test_access_sha256=test_access_sha256,
            sealed_test_population_manifest_sha256=(
                assembly.sealed_test_population_manifest_sha256
            ),
            selected_population_layer=selected_claim.selected_population_layer,
            initial_capital_quote=initial_capital_quote,
            minimum_edge_per_share=minimum_edge_per_share,
            builder_taker_fee_bps=builder_taker_fee_bps,
            progress=progress,
        )
        economic = evaluate_round21_sealed_economics(
            replay.replay_result.selected_matrix,
            test_dataset_sha256=replay.test_dataset_sha256,
            test_target_manifest_sha256=replay.test_target_manifest_sha256,
        )
        ai_report: PolymarketAIVetoReport | None = None
        ai_comparison: Round21AIMatchedComparison | None = None
        if (
            validated_development_report is not None
            and development_comparison is not None
        ):
            ai_report, ai_comparison = _sealed_ai_comparison(
                baseline=replay,
                development_report=validated_development_report,
                development_comparison=development_comparison,
                source_database=source_database,
                terminal_transport_manifest=terminal_transport_manifest,
                assembly=assembly,
                model_artifact=artifact,
                claim=selected_claim,
                test_access_sha256=test_access_sha256,
                initial_capital_quote=initial_capital_quote,
                minimum_edge_per_share=minimum_edge_per_share,
                builder_taker_fee_bps=builder_taker_fee_bps,
                cache_store=cache_store,
                progress=progress,
            )
        result = build_round21_sealed_evaluation_result(
            claim_sha256=selected_claim.claim_sha256,
            test_access_sha256=test_access_sha256,
            selected_population_layer=selected_claim.selected_population_layer,
            sealed_test_population_manifest_sha256=(
                selected_claim.sealed_test_population_manifest_sha256
            ),
            predictive=predictive,
            economic=economic,
            optional_comparison=replay.replay_result.optional_comparison,
            ai_comparison=ai_comparison,
        )
        captured.update(
            replay_evidence=replay,
            ai_report=ai_report,
            ai_comparison=ai_comparison,
        )
        return result

    result = execute_round21_one_use(
        store_path=store_path,
        claim=selected_claim,
        evaluator=evaluator,
    )
    replay_evidence = captured.get("replay_evidence")
    if not isinstance(replay_evidence, Round21SealedEconomicReplayEvidence):
        raise RuntimeError("Round 21 sealed replay evidence is unavailable")
    ai_report = captured.get("ai_report")
    ai_comparison = captured.get("ai_comparison")
    return Round21SealedOperatorOutcome(
        result=result,
        replay_evidence=replay_evidence,
        ai_report=(
            ai_report if isinstance(ai_report, PolymarketAIVetoReport) else None
        ),
        ai_comparison=(
            ai_comparison
            if isinstance(ai_comparison, Round21AIMatchedComparison)
            else None
        ),
    )


credentials_used = False
account_connected = False
binance_execution_connected = False
automatic_promotion = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_SEALED_OPERATOR_SCHEMA_VERSION",
    "Round21SealedOperatorOutcome",
    "Round21SealedTestAssembly",
    "assemble_round21_sealed_test",
    "evaluate_round21_terminal_sealed_once",
]
