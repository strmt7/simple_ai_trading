from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from simple_ai_trading import impact_absorption_evaluation_store as subject
from simple_ai_trading.impact_absorption_model_features import (
    ROUND73_EVALUATION_CONTRACT_SHA256,
)
from simple_ai_trading.impact_absorption_target_store_v3 import (
    ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256,
)


STUDY_ID = "a" * 32
PRETEST_SHA = "b" * 64
TEST_SHA = "c" * 64


def _repository() -> dict[str, object]:
    return {
        "commit_sha": "d" * 40,
        "tree_sha": "e" * 40,
        "clean": True,
        "dirty": False,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _patch_sealed_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _repository()
    monkeypatch.setattr(subject, "_create_v3_tables", lambda _connection: None)
    monkeypatch.setattr(subject, "_assert_v3_table_shapes", lambda _connection: None)
    monkeypatch.setattr(
        subject,
        "_pretest_storage_identity",
        lambda _connection, *, study_id: (
            {"study_id": study_id, "repository": repository},
            PRETEST_SHA,
            "f" * 64,
            0,
            0,
        ),
    )
    monkeypatch.setattr(
        subject,
        "_test_study_identity",
        lambda _connection, *, study_id, pretest_manifest_sha256: (
            {
                "study_id": study_id,
                "pretest_manifest_sha256": pretest_manifest_sha256,
            },
            TEST_SHA,
        ),
    )


def _claim(database: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_sealed_inputs(monkeypatch)
    repository = _repository()
    return subject.claim_round73_evaluation_access(
        database,
        study_id=STUDY_ID,
        pretest_manifest_sha256=PRETEST_SHA,
        repository_root=database.parent,
        repository_state_function=lambda _root: repository,
    )


def _terminal_result(status: str) -> dict[str, object]:
    return {
        "schema_version": subject.ROUND73_EVALUATION_RESULT_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "status": status,
        "staged_holdout_contract_sha256": (ROUND73_STAGED_HOLDOUT_CONTRACT_SHA256),
        "evaluation_contract_sha256": ROUND73_EVALUATION_CONTRACT_SHA256,
        "pretest_manifest_sha256": PRETEST_SHA,
        "test_study_manifest_sha256": TEST_SHA,
        "profitability_claim": False,
        "trading_authority": False,
    }


def test_access_prediction_and_result_are_append_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "round73-evaluation-store.duckdb"
    claim = _claim(database, monkeypatch)

    with pytest.raises(ValueError, match="already consumed"):
        subject.claim_round73_evaluation_access(
            database,
            study_id=STUDY_ID,
            pretest_manifest_sha256=PRETEST_SHA,
            repository_root=tmp_path,
            repository_state_function=lambda _root: _repository(),
        )

    payload = b"test-only-prediction-artifact"
    artifact_hash = subject.persist_round73_test_prediction(
        database,
        study_id=STUDY_ID,
        symbol="BTCUSDT",
        source_rows_sha256="1" * 64,
        payload=payload,
    )
    source_hash, stored_hash, stored_payload = subject.load_round73_test_prediction(
        database,
        study_id=STUDY_ID,
        symbol="BTCUSDT",
    )
    assert source_hash == "1" * 64
    assert stored_hash == artifact_hash
    assert stored_payload == payload
    with pytest.raises(ValueError, match="immutable"):
        subject.persist_round73_test_prediction(
            database,
            study_id=STUDY_ID,
            symbol="BTCUSDT",
            source_rows_sha256="1" * 64,
            payload=payload,
        )

    stored = subject.persist_round73_evaluation_result(
        database,
        claim=claim,
        status="failed",
        result=_terminal_result("failed"),
    )
    assert stored.status == "failed"
    with pytest.raises(ValueError, match="immutable"):
        subject.persist_round73_evaluation_result(
            database,
            claim=claim,
            status="failed",
            result=_terminal_result("failed"),
        )


def test_interrupted_claim_is_closed_without_target_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "round73-interrupted-store.duckdb"
    _claim(database, monkeypatch)

    stored = subject.finalize_interrupted_round73_evaluation(
        database,
        study_id=STUDY_ID,
        reason="test-only simulated process interruption",
    )

    assert stored.status == "interrupted"
    with pytest.raises(ValueError, match="immutable"):
        subject.finalize_interrupted_round73_evaluation(
            database,
            study_id=STUDY_ID,
            reason="second attempt",
        )


def test_terminal_result_cannot_claim_profitability_or_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "round73-result-contract.duckdb"
    claim = _claim(database, monkeypatch)
    invalid = _terminal_result("passed")
    invalid["profitability_claim"] = True

    with pytest.raises(ValueError, match="contract identity differs"):
        subject.persist_round73_evaluation_result(
            database,
            claim=claim,
            status="passed",
            result=invalid,
        )
