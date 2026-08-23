"""Regression coverage for the Polymarket AI after-fee edge floor."""

from __future__ import annotations

from dataclasses import replace
import json

from simple_ai_trading import polymarket_ai_veto as ai_veto_module
from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading.polymarket_ai_veto import (
    PolymarketAIVetoCase,
    PolymarketAIVetoConfig,
    benchmark_polymarket_ai_veto,
)


def _approve_below_floor(
    url: str,
    _payload: dict[str, object],
    _timeout: float,
    _method: str,
) -> object:
    """Return a typed approval for an economically inadmissible case."""
    if url.endswith("/api/tags"):
        return {"models": [{"name": "qwen3.5:9b", "digest": "f" * 64}]}
    if url.endswith("/api/show"):
        return {"model": "qwen3.5:9b", "parameters": "9B"}
    return {
        "model": "qwen3.5:9b",
        "message": {
            "role": "assistant",
            "content": json.dumps(
                {
                    "action": "approve",
                    "confidence": 0.95,
                    "reason_codes": ["edge_after_fees"],
                    "summary": "Edge is sufficient.",
                }
            ),
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 1,
        "load_duration": 0,
        "prompt_eval_count": 320,
        "prompt_eval_duration": 1,
        "eval_count": 24,
        "eval_duration": 1,
    }


def _gpu_residency(
    _base_url: str,
    model: str,
    _timeout: float,
    *,
    expected_digest: str | None = None,
) -> OllamaResidencyReport:
    """Report the exact benchmark model as fully GPU resident."""
    return OllamaResidencyReport(
        requested_model=model,
        status="gpu_resident",
        loaded_model=model,
        digest=expected_digest or "f" * 64,
        size_bytes=6_000_000_000,
        size_vram_bytes=6_000_000_000,
        vram_to_model_ratio=1.0,
    ).validated()


def test_typed_approval_below_frozen_edge_floor_fails_closed() -> None:
    """Ensure AI approval cannot override the deterministic edge floor."""
    provisional = PolymarketAIVetoCase(
        case_id="1" * 64,
        condition_id="0x" + "2" * 64,
        sample_id="3" * 64,
        asset="BTC",
        event_start_ms=1_800_000_000_000,
        decision_received_wall_ms=1_800_000_120_000,
        decision_received_monotonic_ns=120_000_000_000,
        prompt_payload={
            "expected_edge_per_contract_after_fee": "0.005",
            "minimum_required_edge_per_contract": "0.02",
        },
        case_sha256="",
    )
    case = replace(
        provisional,
        case_sha256=ai_veto_module._canonical_sha256(  # noqa: SLF001
            provisional.identity_payload()
        ),
    )

    report = benchmark_polymarket_ai_veto(
        (case,),
        all_condition_ids=(case.condition_id,),
        selection_sha256="a" * 64,
        risk_benchmark_evidence_sha256="b" * 64,
        config=PolymarketAIVetoConfig(model="qwen3.5:9b"),
        post_json=_approve_below_floor,  # type: ignore[arg-type]
        expected_model_digest="f" * 64,
        residency_inspector=_gpu_residency,
    )

    assert report.provider_failure_count == 1
    assert report.valid_response_count == 0
    assert report.veto_count == 1
    assert report.market_permissions == {case.condition_id: False}
    result = report.results[0]
    assert result.decision.valid is False
    assert result.decision.action == "veto"
    assert result.response_payload["error_type"] == "ValueError"  # type: ignore[index]
    assert result.response_payload["provider_response_received"] is True  # type: ignore[index]
