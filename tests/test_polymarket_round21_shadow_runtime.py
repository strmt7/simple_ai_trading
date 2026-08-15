from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import simple_ai_trading.polymarket_round21_shadow_runtime as runtime_module
from simple_ai_trading.polymarket import PolymarketPublicClient
from simple_ai_trading.polymarket_live import PolymarketLiveBlocked
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
from simple_ai_trading.polymarket_round21_shadow import (
    Round21ProspectiveShadowRunner,
)
from simple_ai_trading.polymarket_round21_shadow_store import (
    Round21ProspectiveShadowStore,
)

from polymarket_round21_support import sha


MODEL_FILE_SHA = sha("shadow-runtime-model-file")
MODEL_SHA = sha("shadow-runtime-model")
EVALUATION_FILE_SHA = sha("shadow-runtime-evaluation-file")
RESULT_SHA = sha("shadow-runtime-result")


class _PublicClient(PolymarketPublicClient):
    def __init__(self) -> None:
        pass


def _evidence(*, accepted: bool = True, model_sha: str = MODEL_SHA):
    artifact = Mock(spec=VerifiedRound21DevelopmentArtifact)
    artifact.path = Path("model.json").resolve()
    artifact.file_sha256 = MODEL_FILE_SHA
    artifact.artifact_sha256 = MODEL_SHA
    artifact.artifact = {"artifact_sha256": MODEL_SHA}
    result = SimpleNamespace(
        candidate_accepted=accepted,
        result_sha256=RESULT_SHA,
        predictive=SimpleNamespace(model_artifact_sha256=model_sha),
    )
    evaluation = Mock(spec=VerifiedRound21SealedResultBundle)
    evaluation.path = Path("evaluation.json").resolve()
    evaluation.file_sha256 = EVALUATION_FILE_SHA
    evaluation.result = result
    return artifact, evaluation, result


def test_shadow_runtime_composes_exact_non_promotional_evidence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_client = _PublicClient()
    artifact, evaluation, result = _evidence()
    scorer = Mock(spec=Round21ProspectiveScorer)
    scorer.source_model_artifact_sha256 = MODEL_SHA
    scorer.sealed_result = result
    scorer.population_layer = "core"
    service = Mock(spec=Round21RollingPublicDataService)
    service.scorer = scorer
    runner = Mock(spec=Round21ProspectiveShadowRunner)
    database = tmp_path / "shadow.sqlite3"
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
            else pytest.fail("unexpected service arguments")
        ),
    )

    def build_runner(**kwargs):
        assert kwargs["data_service"] is service
        assert kwargs["run_id"] == "1" * 32
        assert kwargs["poll_interval_seconds"] == 0.1
        runner.data_service = service
        runner.store = kwargs["store"]
        return runner

    monkeypatch.setattr(runtime_module, "Round21ProspectiveShadowRunner", build_runner)
    stack = runtime_module.build_polymarket_round21_shadow_runtime_stack(
        public_client=public_client,
        model_artifact_path=artifact.path,
        expected_model_file_sha256=MODEL_FILE_SHA,
        evaluation_report_path=evaluation.path,
        expected_evaluation_file_sha256=EVALUATION_FILE_SHA,
        shadow_database_path=database,
        run_id="1" * 32,
        discovery_interval_seconds=2.0,
        poll_interval_seconds=0.1,
        queue_capacity=4_000,
    )

    assert stack.runner is runner
    assert stack.store.path == database
    assert stack.evaluation_evidence.result is result
    assert not any(
        (
            stack.credentials_used,
            stack.account_connected,
            stack.binance_execution_connected,
            stack.grants_execution_authority,
            stack.trading_authority,
            stack.paper_trading_authority,
            stack.live_trading_authority,
        )
    )
    with pytest.raises(PolymarketLiveBlocked, match="runtime evidence differs"):
        replace(stack, live_trading_authority=True)
    stack.close()


@pytest.mark.parametrize(
    "accepted,model_sha", [(False, MODEL_SHA), (True, sha("other"))]
)
def test_shadow_runtime_rejects_unaccepted_or_mismatched_sealed_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    accepted: bool,
    model_sha: str,
) -> None:
    artifact, evaluation, _result = _evidence(
        accepted=accepted,
        model_sha=model_sha,
    )
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
        runtime_module.build_polymarket_round21_shadow_runtime_stack(
            public_client=_PublicClient(),
            model_artifact_path="model.json",
            expected_model_file_sha256=MODEL_FILE_SHA,
            evaluation_report_path="evaluation.json",
            expected_evaluation_file_sha256=EVALUATION_FILE_SHA,
            shadow_database_path=tmp_path / "unused.sqlite3",
        )


def test_shadow_runtime_rejects_wrong_public_client_before_file_access(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "load_verified_round21_development_artifact",
        lambda *_args, **_kwargs: pytest.fail("model must remain unopened"),
    )
    with pytest.raises(TypeError, match="public client"):
        runtime_module.build_polymarket_round21_shadow_runtime_stack(
            public_client=object(),
            model_artifact_path="model.json",
            expected_model_file_sha256=MODEL_FILE_SHA,
            evaluation_report_path="evaluation.json",
            expected_evaluation_file_sha256=EVALUATION_FILE_SHA,
            shadow_database_path=tmp_path / "unused.sqlite3",
        )


def test_shadow_runtime_closes_store_when_runner_construction_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, evaluation, result = _evidence()
    scorer = Mock(spec=Round21ProspectiveScorer)
    scorer.source_model_artifact_sha256 = MODEL_SHA
    scorer.sealed_result = result
    scorer.population_layer = "core"
    service = Mock(spec=Round21RollingPublicDataService)
    service.scorer = scorer
    store = Mock(spec=Round21ProspectiveShadowStore)
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
    monkeypatch.setattr(
        runtime_module,
        "Round21ProspectiveScorer",
        lambda **_kwargs: scorer,
    )
    monkeypatch.setattr(
        runtime_module,
        "Round21RollingPublicDataService",
        lambda **_kwargs: service,
    )
    monkeypatch.setattr(
        runtime_module,
        "Round21ProspectiveShadowStore",
        lambda _path: store,
    )
    monkeypatch.setattr(
        runtime_module,
        "Round21ProspectiveShadowRunner",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("runner failed")),
    )
    with pytest.raises(ValueError, match="runner failed"):
        runtime_module.build_polymarket_round21_shadow_runtime_stack(
            public_client=_PublicClient(),
            model_artifact_path="model.json",
            expected_model_file_sha256=MODEL_FILE_SHA,
            evaluation_report_path="evaluation.json",
            expected_evaluation_file_sha256=EVALUATION_FILE_SHA,
            shadow_database_path=tmp_path / "unused.sqlite3",
        )
    store.close.assert_called_once_with()
