from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from simple_ai_trading.foundation_forecast import (
    KRONOS_MAX_CONTEXT,
    KRONOS_MODEL_ARTIFACTS,
    KRONOS_TOKENIZER_ARTIFACT,
    KronosForecastEngine,
    verify_huggingface_artifact_file,
)


def test_kronos_artifact_contracts_are_revision_and_hash_pinned() -> None:
    assert set(KRONOS_MODEL_ARTIFACTS) == {"small", "base"}
    for artifact in (*KRONOS_MODEL_ARTIFACTS.values(), KRONOS_TOKENIZER_ARTIFACT):
        assert len(artifact.revision) == 40
        assert len(artifact.config_sha256) == 64
        assert len(artifact.weights_sha256) == 64
        assert artifact.config_size > 0
        assert artifact.weights_size > 0
        assert artifact.expected_parameters > 0
    assert KRONOS_MAX_CONTEXT == 512


def test_huggingface_file_verifier_rejects_size_and_digest_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"verified payload")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    assert verify_huggingface_artifact_file(
        target,
        expected_size=target.stat().st_size,
        expected_sha256=digest,
        label="fixture",
    ) == target
    with pytest.raises(RuntimeError, match="size mismatch"):
        verify_huggingface_artifact_file(
            target,
            expected_size=1,
            expected_sha256=digest,
            label="fixture",
        )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_huggingface_artifact_file(
            target,
            expected_size=target.stat().st_size,
            expected_sha256="0" * 64,
            label="fixture",
        )


def test_engine_rejects_invalid_batch_contract_without_running_model() -> None:
    engine = KronosForecastEngine(
        predictor=object(),
        report=object(),  # type: ignore[arg-type]
        torch_module=object(),
    )

    with pytest.raises(ValueError, match="non-empty"):
        engine.predict_batch([], [], [], prediction_length=1)
    with pytest.raises(ValueError, match="positive"):
        engine.predict_batch([object()], [object()], [object()], prediction_length=0)
    with pytest.raises(ValueError, match="top_p"):
        engine.predict_batch(
            [object()], [object()], [object()], prediction_length=1, top_p=0.0
        )
