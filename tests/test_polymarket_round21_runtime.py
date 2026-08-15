from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import simple_ai_trading.polymarket_round21_runtime as runtime_module

from polymarket_round21_support import sha
from simple_ai_trading.polymarket import PolymarketPublicClient
from simple_ai_trading.polymarket_live import PolymarketLiveBlocked
from simple_ai_trading.polymarket_live_promotion import (
    VerifiedPolymarketLivePromotion,
)
from simple_ai_trading.polymarket_round21_decision import (
    PolymarketRound21PromotedDecisionProvider,
)
from simple_ai_trading.polymarket_round21_model import (
    VerifiedRound21DevelopmentArtifact,
)
from simple_ai_trading.polymarket_round21_prospective import (
    Round21ProspectiveScorer,
)
from simple_ai_trading.polymarket_round21_sealed import (
    VerifiedRound21SealedResultBundle,
)
from simple_ai_trading.polymarket_round21_session import (
    Round21RollingPublicDataService,
)


MODEL_FILE_SHA = sha("round21-runtime-model-file")
MODEL_SHA = sha("round21-runtime-model")
EVALUATION_FILE_SHA = sha("round21-runtime-evaluation-file")


class _PublicClient(PolymarketPublicClient):
    def __init__(self) -> None:
        pass


def _promotion(*, variant: str = "fiveminute"):
    promotion = Mock(spec=VerifiedPolymarketLivePromotion)
    promotion.promotion = SimpleNamespace(
        market_variant=variant,
        model_artifact=SimpleNamespace(sha256=MODEL_FILE_SHA),
        evaluation_report=SimpleNamespace(sha256=EVALUATION_FILE_SHA),
    )
    promotion.model_artifact_path = Path("model.json").resolve()
    promotion.evaluation_report_path = Path("evaluation.json").resolve()
    return promotion


def _evidence():
    artifact = Mock(spec=VerifiedRound21DevelopmentArtifact)
    artifact.path = Path("model.json").resolve()
    artifact.file_sha256 = MODEL_FILE_SHA
    artifact.artifact_sha256 = MODEL_SHA
    artifact.artifact = {"artifact_sha256": MODEL_SHA}
    result = SimpleNamespace(
        candidate_accepted=True,
        predictive=SimpleNamespace(model_artifact_sha256=MODEL_SHA),
    )
    evaluation = Mock(spec=VerifiedRound21SealedResultBundle)
    evaluation.path = Path("evaluation.json").resolve()
    evaluation.file_sha256 = EVALUATION_FILE_SHA
    evaluation.result = result
    return artifact, evaluation, result


def test_round21_runtime_composes_exact_promotion_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_client = _PublicClient()
    promotion = _promotion()
    artifact, evaluation, result = _evidence()
    scorer = Mock(spec=Round21ProspectiveScorer)
    scorer.source_model_artifact_sha256 = MODEL_SHA
    service = Mock(spec=Round21RollingPublicDataService)
    service.scorer = scorer
    provider = Mock(spec=PolymarketRound21PromotedDecisionProvider)
    provider.data_service = service
    provider.artifact_evidence = artifact
    monkeypatch.setattr(
        runtime_module,
        "load_verified_round21_development_artifact",
        lambda path, *, expected_file_sha256: (
            artifact
            if Path(path).resolve() == artifact.path
            and expected_file_sha256 == MODEL_FILE_SHA
            else pytest.fail("unexpected model evidence")
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_verified_round21_sealed_result_bundle",
        lambda path, *, expected_file_sha256: (
            evaluation
            if Path(path).resolve() == evaluation.path
            and expected_file_sha256 == EVALUATION_FILE_SHA
            else pytest.fail("unexpected evaluation evidence")
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "Round21ProspectiveScorer",
        lambda *, artifact, sealed_result: (
            scorer
            if artifact is artifact_evidence.artifact and sealed_result is result
            else pytest.fail("unexpected scorer evidence")
        ),
    )
    artifact_evidence = artifact
    monkeypatch.setattr(
        runtime_module,
        "Round21RollingPublicDataService",
        lambda **kwargs: (
            service
            if kwargs["public_client"] is public_client
            and kwargs["scorer"] is scorer
            and kwargs["discovery_interval_seconds"] == 2.0
            and kwargs["queue_capacity"] == 4_000
            else pytest.fail("unexpected data service arguments")
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "PolymarketRound21PromotedDecisionProvider",
        lambda **kwargs: (
            provider
            if kwargs["public_client"] is public_client
            and kwargs["data_service"] is service
            and kwargs["promotion"] is promotion
            and kwargs["artifact_evidence"] is artifact
            and kwargs["requested_quantity"] == Decimal("3")
            and kwargs["risk_level"] == "regular"
            else pytest.fail("unexpected decision-provider arguments")
        ),
    )

    stack = runtime_module.build_polymarket_round21_runtime_stack(
        public_client=public_client,
        promotion=promotion,
        requested_quantity=Decimal("3"),
        risk_level="regular",
        discovery_interval_seconds=2.0,
        queue_capacity=4_000,
    )

    assert stack.evaluation_evidence.result is result
    assert stack.data_service is service
    assert stack.decision_provider is provider
    assert stack.credentials_used is False
    assert stack.account_connected is False
    assert stack.binance_execution_connected is False
    assert stack.trading_authority is False


def test_round21_runtime_rejects_wrong_variant_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "load_verified_round21_development_artifact",
        lambda *_args, **_kwargs: pytest.fail("model file must remain unopened"),
    )
    with pytest.raises(PolymarketLiveBlocked, match="five-minute promotion"):
        runtime_module.build_polymarket_round21_runtime_stack(
            public_client=_PublicClient(),
            promotion=_promotion(variant="fifteenminute"),
            requested_quantity=Decimal("1"),
        )


def test_round21_runtime_rejects_model_and_sealed_result_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, evaluation, _result = _evidence()
    evaluation.result.predictive.model_artifact_sha256 = sha("different-model")
    monkeypatch.setattr(
        runtime_module,
        "load_verified_round21_development_artifact",
        lambda *_args, **_kwargs: artifact,
    )
    monkeypatch.setattr(
        runtime_module,
        "load_verified_round21_sealed_result_bundle",
        lambda *_args, **_kwargs: evaluation,
    )
    with pytest.raises(PolymarketLiveBlocked, match="accepted sealed evaluation"):
        runtime_module.build_polymarket_round21_runtime_stack(
            public_client=_PublicClient(),
            promotion=_promotion(),
            requested_quantity=Decimal("1"),
        )
