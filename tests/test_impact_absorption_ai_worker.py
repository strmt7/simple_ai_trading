from __future__ import annotations

from dataclasses import replace
import json

import pytest

from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading.impact_absorption_ai_protocol import (
    ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION,
    Round74AIModelManifest,
    Round74AIReviewRequest,
)
from simple_ai_trading.impact_absorption_ai_worker import (
    ROUND74_AI_WORKER_ENDPOINT,
    Round74AIWorkerEnvelope,
    Round74AIWorkerResult,
    execute_round74_ai_worker,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
)


WALL_NS = 1_800_000_000_000_000_000
MODEL_DIGEST = "d" * 64
METADATA_DIGEST = "e" * 64


def _manifest() -> Round74AIModelManifest:
    return Round74AIModelManifest(
        model_id="TheFinAI/Fino1-8B",
        model_revision="a" * 40,
        model_artifact_sha256=MODEL_DIGEST,
        model_artifact_kind="ollama_manifest",
        parameter_count=8_000_000_000,
        quantization="q6_k",
        runtime_backend="llama.cpp-vulkan",
        runtime_version="0.12.3",
        license_id="llama3.1",
        model_card_url="https://huggingface.co/TheFinAI/Fino1-8B",
        minimum_vram_bytes=8 * 1024**3,
        finance_specialized=True,
    )


def _request() -> Round74AIReviewRequest:
    count = len(ROUND74_EVENT_FEATURE_NAMES)
    return Round74AIReviewRequest(
        pretest_policy_sha256="1" * 64,
        probability_calibration_sha256="4" * 64,
        sample_sha256="2" * 64,
        deterministic_risk_state_sha256="3" * 64,
        risk_profile="conservative",
        asset_slot=0,
        side="long",
        horizon_seconds=30,
        requested_wall_ns=WALL_NS,
        expires_wall_ns=WALL_NS + 20_000_000_000,
        proposed_risk_size_bps=2_500,
        feature_last=tuple(0.0 for _ in range(count)),
        feature_mean=tuple(0.1 for _ in range(count)),
        feature_standard_deviation=tuple(0.2 for _ in range(count)),
        feature_recent_change=tuple(0.0 for _ in range(count)),
        payoff_quantiles_bps=(-5.0, -1.0, 2.0, 4.0, 7.0),
        maximum_adverse_excursion_quantiles_bps=(
            1.0,
            2.0,
            3.0,
            5.0,
            8.0,
        ),
        positive_payoff_probability=0.61,
        adverse_selection_probability=0.27,
        regime_unpredictability_probability=0.18,
    )


def _envelope() -> Round74AIWorkerEnvelope:
    return Round74AIWorkerEnvelope(
        model_name="fino1:8b",
        endpoint=ROUND74_AI_WORKER_ENDPOINT,
        timeout_seconds=10.0,
        model_manifest=_manifest(),
        review_request=_request(),
    )


def _response() -> dict[str, object]:
    content = json.dumps(
        {
            "schema_version": ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION,
            "verdict": "reduce",
            "size_multiplier_bps": 5_000,
            "confidence_bps": 7_500,
            "reason_codes": ["forecast_uncertainty"],
        }
    )
    return {
        "model": "fino1:8b",
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 200,
        "eval_count": 40,
        "total_duration": 1_000,
        "load_duration": 100,
        "prompt_eval_duration": 300,
        "eval_duration": 600,
    }


def _residency(
    _endpoint: str,
    model: str,
    _timeout: float,
    expected_digest: str,
) -> OllamaResidencyReport:
    assert expected_digest == MODEL_DIGEST
    return OllamaResidencyReport(
        requested_model=model,
        status="gpu_resident",
        loaded_model=model,
        digest=MODEL_DIGEST,
        size_bytes=1_000,
        size_vram_bytes=1_000,
        vram_to_model_ratio=1.0,
    )


def test_worker_accepts_only_pinned_fully_gpu_resident_result() -> None:
    posted: list[tuple[str, dict[str, object]]] = []

    def post(
        url: str,
        payload: dict[str, object],
        _timeout: float,
    ) -> object:
        posted.append((url, payload))
        return _response()

    result = execute_round74_ai_worker(
        _envelope(),
        post_json=post,
        provenance_resolver=lambda *_: (
            MODEL_DIGEST,
            METADATA_DIGEST,
        ),
        residency_inspector=_residency,
    )
    restored = Round74AIWorkerResult.from_dict(result.as_dict())

    assert restored == result
    assert result.decision.verdict == "reduce"
    assert result.residency.fully_gpu_resident
    assert posted[0][0] == f"{ROUND74_AI_WORKER_ENDPOINT}/api/chat"
    request = posted[0][1]
    assert request["stream"] is False
    assert request["think"] is False
    assert request["format"]["additionalProperties"] is False
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][1]["role"] == "user"


def test_worker_rejects_artifact_drift_before_generation() -> None:
    generation_called = False

    def post(*_args: object) -> object:
        nonlocal generation_called
        generation_called = True
        return _response()

    with pytest.raises(ValueError, match="model artifact differs"):
        execute_round74_ai_worker(
            _envelope(),
            post_json=post,
            provenance_resolver=lambda *_: (
                "f" * 64,
                METADATA_DIGEST,
            ),
            residency_inspector=_residency,
        )

    assert generation_called is False


def test_worker_rejects_partial_or_cpu_residency() -> None:
    def cpu_residency(
        _endpoint: str,
        model: str,
        _timeout: float,
        _digest: str,
    ) -> OllamaResidencyReport:
        return OllamaResidencyReport(
            requested_model=model,
            status="cpu_only",
            loaded_model=model,
            digest=MODEL_DIGEST,
            size_bytes=1_000,
            size_vram_bytes=0,
            vram_to_model_ratio=0.0,
        )

    with pytest.raises(ValueError, match="full GPU"):
        execute_round74_ai_worker(
            _envelope(),
            post_json=lambda *_: _response(),
            provenance_resolver=lambda *_: (
                MODEL_DIGEST,
                METADATA_DIGEST,
            ),
            residency_inspector=cpu_residency,
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"endpoint": "http://localhost:11434"},
        {"endpoint": "https://example.invalid"},
        {"timeout_seconds": 25.1},
        {"model_name": "fino1:8b bad"},
    ],
)
def test_worker_envelope_rejects_boundary_drift(
    changed: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="Round 74 AI worker"):
        replace(_envelope(), **changed).validate()


def test_worker_rejects_incomplete_or_wrong_model_response() -> None:
    response = _response()
    response["done_reason"] = "length"
    with pytest.raises(ValueError, match="incomplete"):
        execute_round74_ai_worker(
            _envelope(),
            post_json=lambda *_: response,
            provenance_resolver=lambda *_: (
                MODEL_DIGEST,
                METADATA_DIGEST,
            ),
            residency_inspector=_residency,
        )

    response = _response()
    response["model"] = "qwen3:8b"
    with pytest.raises(ValueError, match="model identity"):
        execute_round74_ai_worker(
            _envelope(),
            post_json=lambda *_: response,
            provenance_resolver=lambda *_: (
                MODEL_DIGEST,
                METADATA_DIGEST,
            ),
            residency_inspector=_residency,
        )
