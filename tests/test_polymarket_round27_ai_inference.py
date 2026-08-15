from __future__ import annotations

import json

from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading.polymarket_round27_ai import (
    POLYMARKET_ROUND27_ODA_HOST_CANDIDATE,
    POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE,
)
from simple_ai_trading.polymarket_round27_ai_cases import (
    materialize_round27_ai_cases,
)
from simple_ai_trading.polymarket_round27_ai_inference import (
    round27_ai_ablation_request,
    run_round27_ai_inference,
)
from simple_ai_trading.polymarket_round27_economics import (
    Round27EconomicBookBatch,
    Round27EconomicConfig,
)
from test_polymarket_round27_ai_cases import _FixedModel, _rows
from test_polymarket_round27_economics import _population


def _panel(count: int):
    markets, partition, _probabilities, books, _outcomes = _population(count)
    batches = tuple(
        Round27EconomicBookBatch(
            condition_ids=tuple(
                market.condition_id for market in markets[start : start + 30]
            ),
            books=tuple(
                book
                for book in books
                if book.market.condition_id
                in {
                    market.condition_id
                    for market in markets[start : start + 30]
                }
            ),
        )
        for start in range(0, count, 30)
    )
    return materialize_round27_ai_cases(
        role="selection",
        rows=_rows(partition),
        selected_model=_FixedModel(),
        model_name=_FixedModel.model_name,
        model_sha256="b" * 64,
        markets=markets,
        book_batches=(batch for batch in batches),
        source_audit_sha256="c" * 64,
        config=Round27EconomicConfig(),
    )


def _raw(model: str, decision: str, reason: str) -> dict[str, object]:
    return {
        "model": model,
        "response": json.dumps(
            {"decision": decision, "reason_codes": [reason]},
            separators=(",", ":"),
        ),
        "done": True,
        "done_reason": "stop",
        "total_duration": 100_000_000,
        "load_duration": 1_000_000,
        "prompt_eval_count": 500,
        "prompt_eval_duration": 50_000_000,
        "eval_count": 12,
        "eval_duration": 40_000_000,
    }


def test_round27_ai_candidates_receive_byte_identical_case_prompts() -> None:
    case = _panel(1).cases[0]
    qwen = round27_ai_ablation_request(
        case,
        POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE,
    )
    oda = round27_ai_ablation_request(
        case,
        POLYMARKET_ROUND27_ODA_HOST_CANDIDATE,
    )

    assert qwen["prompt"] == oda["prompt"]
    assert qwen["format"] == oda["format"]
    assert qwen["options"] == oda["options"]
    assert qwen["model"] != oda["model"]
    assert qwen["think"] is False


def test_round27_ai_inference_measures_latency_and_qualifies_valid_changes() -> None:
    panel = _panel(60)
    candidate = POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE
    case_count = 0

    def post_json(_url, payload, _timeout):
        nonlocal case_count
        prompt = payload.get("prompt")
        if prompt is None:
            return {}
        if str(prompt).startswith("Runtime conformance probe"):
            return _raw(candidate.runtime_model, "reject", "missing_liquidity")
        case_count += 1
        return _raw(
            candidate.runtime_model,
            "reject" if case_count <= 20 else "unchanged",
            "liquidity_thin" if case_count <= 20 else "no_material_risk",
        )

    residency_calls = 0

    def inspect_residency(*_args, **_kwargs):
        nonlocal residency_calls
        residency_calls += 1
        if residency_calls == 1:
            return OllamaResidencyReport(
                requested_model=candidate.runtime_model,
                status="gpu_resident",
                loaded_model=candidate.runtime_model,
                digest=candidate.runtime_digest,
                size_bytes=100,
                size_vram_bytes=100,
                vram_to_model_ratio=1.0,
            ).validated()
        return OllamaResidencyReport(
            requested_model=candidate.runtime_model,
            status="unloaded",
            loaded_model=None,
            digest=None,
            size_bytes=None,
            size_vram_bytes=None,
            vram_to_model_ratio=None,
        ).validated()

    clock = 0

    def monotonic_ns():
        nonlocal clock
        clock += 100_000_000
        return clock

    report = run_round27_ai_inference(
        panel=panel,
        candidate=candidate,
        post_json=post_json,
        residency_inspector=inspect_residency,
        inventory_getter=lambda _url, _timeout: {
            "models": [{"digest": candidate.runtime_digest}]
        },
        monotonic_ns=monotonic_ns,
    )

    assert len(report.responses) == 60
    assert report.status_counts == {"valid": 60}
    assert report.changed_action_count == 20
    assert report.rejected_fraction == 1 / 3
    assert report.candidate_eligible_for_matched_evaluation is True
    assert report.unload_observed is True
    assert {response.wall_latency_ms for response in report.responses} == {100}


def test_round27_ai_invalid_semantics_fail_closed_and_disqualify() -> None:
    panel = _panel(1)
    candidate = POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE

    def post_json(_url, payload, _timeout):
        prompt = payload.get("prompt")
        if prompt is None:
            return {}
        if str(prompt).startswith("Runtime conformance probe"):
            return _raw(candidate.runtime_model, "reject", "missing_liquidity")
        return _raw(candidate.runtime_model, "unchanged", "liquidity_thin")

    residency_calls = 0

    def inspect_residency(*_args, **_kwargs):
        nonlocal residency_calls
        residency_calls += 1
        loaded = residency_calls == 1
        return OllamaResidencyReport(
            requested_model=candidate.runtime_model,
            status="gpu_resident" if loaded else "unloaded",
            loaded_model=candidate.runtime_model if loaded else None,
            digest=candidate.runtime_digest if loaded else None,
            size_bytes=100 if loaded else None,
            size_vram_bytes=100 if loaded else None,
            vram_to_model_ratio=1.0 if loaded else None,
        ).validated()

    clock = 0

    def monotonic_ns():
        nonlocal clock
        clock += 100_000_000
        return clock

    report = run_round27_ai_inference(
        panel=panel,
        candidate=candidate,
        post_json=post_json,
        residency_inspector=inspect_residency,
        inventory_getter=lambda _url, _timeout: {
            "models": [{"digest": candidate.runtime_digest}]
        },
        monotonic_ns=monotonic_ns,
    )

    assert report.responses[0].status == "invalid_response"
    assert report.responses[0].decision == "reject"
    assert report.candidate_eligible_for_matched_evaluation is False
