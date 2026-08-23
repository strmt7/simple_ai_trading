from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from simple_ai_trading import ai_start_gate
from simple_ai_trading.ai_model_identity import OllamaModelIdentity
from simple_ai_trading.ai_review import AIReviewReport
from simple_ai_trading.ai_runtime import AICapabilityReport
from simple_ai_trading.types import RuntimeConfig


def _runtime(**changes: object) -> RuntimeConfig:
    return RuntimeConfig(
        symbol="BTCUSDT",
        quote_asset="USDT",
        interval="15m",
        market_type="futures",
        compute_backend="directml",
        ai_model="qwen3:8b",
        **changes,
    )


def _review(**changes: object) -> AIReviewReport:
    values = {
        "status": "ok",
        "approved": True,
        "model": "qwen3:8b",
        "model_digest": "d" * 64,
        "model_metadata_sha256": "e" * 64,
        "source_report": "source.json",
    }
    values.update(changes)
    return cast(AIReviewReport, SimpleNamespace(**values))


def _capability(*, model: str = "qwen3:8b") -> AICapabilityReport:
    return AICapabilityReport(
        ok=True,
        provider="ollama",
        model=model,
        gpu_vendor="amd",
        compute_backend_requested="directml",
        compute_backend_kind="directml",
        compute_backend_device="privateuseone:0",
        compute_backend_reason="",
        free_vram_gb=12.0,
        free_ram_gb=32.0,
        model_parameters_b=8.2,
        messages=(),
        warnings=(),
        provider_available=True,
        model_available=True,
        model_local=True,
    )


def _identity(*, digest: str = "d" * 64) -> OllamaModelIdentity:
    return OllamaModelIdentity(
        canonical_model="qwen3:8b",
        digest=digest,
        metadata_sha256="e" * 64,
        parameter_count=8_200_000_000,
        parameter_size="8.2B",
    ).validated()


def test_disabled_start_gate_skips_all_external_evidence() -> None:
    gate = ai_start_gate.evaluate_ai_start_gate(
        _runtime(ai_enabled=False),
        objective="regular",
        model_artifact=None,
        paper_mode=False,
        review_path=Path("missing-review.json"),
        capability_detector=lambda _config: (_ for _ in ()).throw(
            AssertionError("disabled AI must not probe capabilities")
        ),
        model_identity_resolver=lambda *_args: (_ for _ in ()).throw(
            AssertionError("disabled AI must not resolve a model")
        ),
    )

    assert gate.status == "disabled"
    assert gate.allowed is True
    assert gate.active is False


@pytest.mark.parametrize(
    ("review", "capability", "identity", "runtime", "message"),
    (
        (
            _review(status="review_required", approved=False),
            _capability(),
            _identity(),
            _runtime(),
            "AI review is not approved",
        ),
        (
            _review(),
            _capability(model="other:8b"),
            _identity(),
            _runtime(),
            "AI review model differs",
        ),
        (
            _review(),
            _capability(),
            _identity(digest="f" * 64),
            _runtime(),
            "installed AI model provenance differs",
        ),
        (
            _review(),
            _capability(),
            _identity(),
            _runtime(ai_min_model_parameters_b=9.0),
            "parameter count is below",
        ),
    ),
)
def test_runtime_model_identity_fails_closed_at_each_binding(
    review: AIReviewReport,
    capability: AICapabilityReport,
    identity: OllamaModelIdentity,
    runtime: RuntimeConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ai_start_gate._validated_runtime_model_identity(
            review,
            capability,
            runtime,
            base_url="http://127.0.0.1:11434",
            timeout_seconds=1.0,
            model_identity_resolver=lambda *_args: identity,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            {
                "quote_asset": "USDT",
                "interval": "1m",
                "market_type": "futures",
                "requested_objectives": ["regular"],
            },
            "market contract differs",
        ),
        (
            {
                "quote_asset": "USDT",
                "interval": "15m",
                "market_type": "futures",
                "requested_objectives": [],
            },
            "objective differs",
        ),
    ),
)
def test_reviewed_runtime_outcome_rejects_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(ai_start_gate, "_strict_json_mapping", lambda _path: payload)

    with pytest.raises(ValueError, match=message):
        ai_start_gate._reviewed_runtime_outcome(
            _review(),
            _runtime(),
            objective="regular",
        )


def test_terminal_binding_requires_runtime_model_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ai_start_gate,
        "_expected_terminal_fingerprint",
        lambda *_args, **_kwargs: "a" * 64,
    )

    with pytest.raises(ValueError, match="model artifact is unavailable"):
        ai_start_gate._matching_terminal_model_fingerprint(
            {},
            objective="regular",
            model_artifact=None,
        )
