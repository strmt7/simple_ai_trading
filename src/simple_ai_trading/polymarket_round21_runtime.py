"""Verified runtime composition for independent BTC five-minute decisions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .polymarket import PolymarketPublicClient
from .polymarket_live import PolymarketLiveBlocked
from .polymarket_live_promotion import VerifiedPolymarketLivePromotion
from .polymarket_round21_decision import (
    PolymarketRound21PromotedDecisionProvider,
)
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


POLYMARKET_ROUND21_RUNTIME_SCHEMA_VERSION = "polymarket-round21-runtime-v1"


@dataclass(frozen=True, slots=True)
class PolymarketRound21RuntimeStack:
    """Promotion-bound public prediction stack with no account authority."""

    public_client: PolymarketPublicClient
    promotion: VerifiedPolymarketLivePromotion
    artifact_evidence: VerifiedRound21DevelopmentArtifact
    evaluation_evidence: VerifiedRound21SealedResultBundle
    scorer: Round21ProspectiveScorer
    data_service: Round21RollingPublicDataService
    decision_provider: PolymarketRound21PromotedDecisionProvider
    credentials_used: bool = False
    account_connected: bool = False
    binance_execution_connected: bool = False
    trading_authority: bool = False

    def __post_init__(self) -> None:
        policy = self.promotion.promotion
        result = self.evaluation_evidence.result
        if (
            policy.market_variant != "fiveminute"
            or self.artifact_evidence.path
            != self.promotion.model_artifact_path.resolve()
            or self.artifact_evidence.file_sha256 != policy.model_artifact.sha256
            or self.evaluation_evidence.path
            != self.promotion.evaluation_report_path.resolve()
            or self.evaluation_evidence.file_sha256 != policy.evaluation_report.sha256
            or not result.candidate_accepted
            or result.predictive.model_artifact_sha256
            != self.artifact_evidence.artifact_sha256
            or self.scorer.source_model_artifact_sha256
            != self.artifact_evidence.artifact_sha256
            or self.data_service.scorer is not self.scorer
            or self.decision_provider.data_service is not self.data_service
            or self.decision_provider.artifact_evidence is not self.artifact_evidence
            or any(
                (
                    self.credentials_used,
                    self.account_connected,
                    self.binance_execution_connected,
                    self.trading_authority,
                )
            )
        ):
            raise PolymarketLiveBlocked("Round 21 runtime evidence differs")


def build_polymarket_round21_runtime_stack(
    *,
    public_client: PolymarketPublicClient,
    promotion: VerifiedPolymarketLivePromotion,
    requested_quantity: Decimal,
    risk_level: str = "conservative",
    discovery_interval_seconds: float = 1.0,
    queue_capacity: int = 20_000,
) -> PolymarketRound21RuntimeStack:
    """Build one exact-file-bound, restart-safe five-minute runtime stack."""

    if not isinstance(public_client, PolymarketPublicClient):
        raise TypeError("Round 21 runtime public client differs")
    if not isinstance(promotion, VerifiedPolymarketLivePromotion):
        raise PolymarketLiveBlocked(
            "Round 21 runtime requires verified promotion evidence"
        )
    policy = promotion.promotion
    if policy.market_variant != "fiveminute":
        raise PolymarketLiveBlocked("Round 21 runtime requires a five-minute promotion")
    artifact_evidence = load_verified_round21_development_artifact(
        promotion.model_artifact_path,
        expected_file_sha256=policy.model_artifact.sha256,
    )
    evaluation_evidence = load_verified_round21_sealed_result_bundle(
        promotion.evaluation_report_path,
        expected_file_sha256=policy.evaluation_report.sha256,
    )
    result = evaluation_evidence.result
    if (
        not result.candidate_accepted
        or result.predictive.model_artifact_sha256 != artifact_evidence.artifact_sha256
    ):
        raise PolymarketLiveBlocked(
            "Round 21 model differs from accepted sealed evaluation"
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
    decision_provider = PolymarketRound21PromotedDecisionProvider(
        public_client=public_client,
        data_service=data_service,
        promotion=promotion,
        requested_quantity=requested_quantity,
        risk_level=risk_level,
        artifact_evidence=artifact_evidence,
    )
    return PolymarketRound21RuntimeStack(
        public_client=public_client,
        promotion=promotion,
        artifact_evidence=artifact_evidence,
        evaluation_evidence=evaluation_evidence,
        scorer=scorer,
        data_service=data_service,
        decision_provider=decision_provider,
    )


credentials_used = False
account_connected = False
binance_execution_connected = False
trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_RUNTIME_SCHEMA_VERSION",
    "PolymarketRound21RuntimeStack",
    "build_polymarket_round21_runtime_stack",
]
