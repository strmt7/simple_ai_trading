"""Exact-evidence composition for the no-order Polymarket Round 21 shadow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .polymarket import PolymarketPublicClient
from .polymarket_live import PolymarketLiveBlocked
from .polymarket_round21_model import (
    VerifiedRound21DevelopmentArtifact,
    load_verified_round21_development_artifact,
)
from .polymarket_round21_prospective import Round21ProspectiveScorer
from .polymarket_round21_sealed import (
    VerifiedRound21SealedResultBundle,
    load_verified_round21_sealed_result_bundle,
)
from .polymarket_round21_session import Round21RollingPublicDataService
from .polymarket_round21_shadow import Round21ProspectiveShadowRunner
from .polymarket_round21_shadow_store import Round21ProspectiveShadowStore


POLYMARKET_ROUND21_SHADOW_RUNTIME_SCHEMA_VERSION = (
    "polymarket-round21-prospective-shadow-runtime-v1"
)


@dataclass(frozen=True, slots=True)
class PolymarketRound21ShadowRuntimeStack:
    """Verified public-data shadow stack with no promotion or order authority."""

    public_client: PolymarketPublicClient
    artifact_evidence: VerifiedRound21DevelopmentArtifact
    evaluation_evidence: VerifiedRound21SealedResultBundle
    scorer: Round21ProspectiveScorer
    data_service: Round21RollingPublicDataService
    store: Round21ProspectiveShadowStore
    runner: Round21ProspectiveShadowRunner
    credentials_used: bool = False
    account_connected: bool = False
    binance_execution_connected: bool = False
    grants_execution_authority: bool = False
    trading_authority: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def __post_init__(self) -> None:
        result = self.evaluation_evidence.result
        if (
            not result.candidate_accepted
            or result.predictive.model_artifact_sha256
            != self.artifact_evidence.artifact_sha256
            or self.scorer.source_model_artifact_sha256
            != self.artifact_evidence.artifact_sha256
            or self.scorer.sealed_result.result_sha256 != result.result_sha256
            or self.data_service.scorer is not self.scorer
            or self.runner.data_service is not self.data_service
            or self.runner.store is not self.store
            or any(
                (
                    self.credentials_used,
                    self.account_connected,
                    self.binance_execution_connected,
                    self.grants_execution_authority,
                    self.trading_authority,
                    self.paper_trading_authority,
                    self.live_trading_authority,
                )
            )
        ):
            raise PolymarketLiveBlocked("Round 21 shadow runtime evidence differs")

    def close(self) -> None:
        self.store.close()


def build_polymarket_round21_shadow_runtime_stack(
    *,
    public_client: PolymarketPublicClient,
    model_artifact_path: str | Path,
    expected_model_file_sha256: str,
    evaluation_report_path: str | Path,
    expected_evaluation_file_sha256: str,
    shadow_database_path: str | Path,
    run_id: str | None = None,
    discovery_interval_seconds: float = 1.0,
    poll_interval_seconds: float = 0.05,
    queue_capacity: int = 20_000,
) -> PolymarketRound21ShadowRuntimeStack:
    """Build a restart-auditable target-free shadow from exact verified files."""

    if not isinstance(public_client, PolymarketPublicClient):
        raise TypeError("Round 21 shadow runtime public client differs")
    artifact_evidence = load_verified_round21_development_artifact(
        model_artifact_path,
        expected_file_sha256=expected_model_file_sha256,
    )
    evaluation_evidence = load_verified_round21_sealed_result_bundle(
        evaluation_report_path,
        expected_file_sha256=expected_evaluation_file_sha256,
    )
    result = evaluation_evidence.result
    if (
        not result.candidate_accepted
        or result.predictive.model_artifact_sha256 != artifact_evidence.artifact_sha256
    ):
        raise PolymarketLiveBlocked(
            "Round 21 shadow model differs from accepted sealed evaluation"
        )
    scorer = Round21ProspectiveScorer(
        artifact=artifact_evidence.artifact,
        sealed_result=result,
    )
    data_service = Round21RollingPublicDataService(
        public_client=public_client,
        scorer=scorer,
        discovery_interval_seconds=discovery_interval_seconds,
        queue_capacity=queue_capacity,
    )
    store = Round21ProspectiveShadowStore(shadow_database_path)
    try:
        runner = Round21ProspectiveShadowRunner(
            data_service=data_service,
            store=store,
            run_id=run_id,
            poll_interval_seconds=poll_interval_seconds,
        )
        return PolymarketRound21ShadowRuntimeStack(
            public_client=public_client,
            artifact_evidence=artifact_evidence,
            evaluation_evidence=evaluation_evidence,
            scorer=scorer,
            data_service=data_service,
            store=store,
            runner=runner,
        )
    except Exception:
        store.close()
        raise


credentials_used = False
account_connected = False
binance_execution_connected = False
grants_execution_authority = False
profitability_claim = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_SHADOW_RUNTIME_SCHEMA_VERSION",
    "PolymarketRound21ShadowRuntimeStack",
    "build_polymarket_round21_shadow_runtime_stack",
]
