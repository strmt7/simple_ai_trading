from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading.polymarket_round27_economics import Round27EconomicConfig
from simple_ai_trading.polymarket_round28_ai_cases import (
    materialize_round28_ai_cases,
)
from simple_ai_trading.polymarket_round28_ai_contract import (
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_host import (
    build_round28_ai_artifact_verification,
    probe_round28_ai_candidate_host,
    round28_ai_candidate_from_contract,
)
from simple_ai_trading.polymarket_round28_ai_inference import (
    round28_ai_inference_report_from_mapping,
    round28_ai_inference_request,
    run_round28_ai_inference,
    validate_round28_ai_inference_report,
)
from test_polymarket_round27_economics import _population
from test_polymarket_round28_ai_cases import (
    _FixedAugmentedModel,
    _rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _residency(candidate, *, loaded: bool) -> OllamaResidencyReport:
    return OllamaResidencyReport(
        requested_model=candidate.runtime_model,
        status="gpu_resident" if loaded else "unloaded",
        loaded_model=candidate.runtime_model if loaded else None,
        digest=candidate.runtime_digest if loaded else None,
        size_bytes=1_000 if loaded else None,
        size_vram_bytes=1_000 if loaded else None,
        vram_to_model_ratio=1.0 if loaded else None,
    )


def _raw(candidate, decision: str, reasons: list[str]) -> dict[str, object]:
    return {
        "model": candidate.runtime_model,
        "response": json.dumps(
            {"decision": decision, "reason_codes": reasons},
            separators=(",", ":"),
        ),
        "done": True,
        "done_reason": "stop",
        "total_duration": 90_000_000,
        "load_duration": 10_000_000,
        "prompt_eval_count": 64,
        "eval_count": 8,
    }


def _host_report(contract, candidate):
    calls = 0
    times = iter((0, 100_000_000, 200_000_000, 300_000_000))

    def residency(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _residency(candidate, loaded=calls == 1)

    artifact = build_round28_ai_artifact_verification(
        candidate,
        observed_sha256=candidate.artifact_sha256,
        observed_size_bytes=candidate.artifact_size_bytes,
        verification_method="inherited_round27_exact_artifact",
        source_evidence_sha256="a" * 64,
        observed_at_ms=1,
    )
    return probe_round28_ai_candidate_host(
        candidate,
        artifact_verification=artifact,
        post_json=lambda _url, payload, _timeout: (
            {} if payload.get("keep_alive") == 0 else _raw(candidate, "reject", ["bbo_source_stale_or_gapped"])
        ),
        residency_inspector=residency,
        monotonic_ns=lambda: next(times),
    )


def _panel():
    markets, partition, _probabilities, books, _outcomes = _population(2)
    panel = materialize_round28_ai_cases(
        role="selection",
        rows=_rows(partition),
        selected_model=_FixedAugmentedModel(),
        selection_claim_sha256="e" * 64,
        markets=markets,
        books=books,
        source_audit_sha256="c" * 64,
        config=Round27EconomicConfig(
            minimum_executed_trades=1,
            minimum_profitable_conditions=1,
            bootstrap_draws=1_000,
        ),
    )
    return panel


def test_round28_ai_inference_is_measured_fail_closed_and_restart_safe(
    monkeypatch,
) -> None:
    import simple_ai_trading.polymarket_round28_ai_inference as inference

    contract = load_round28_ai_contract(ROOT)
    candidate = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    panel = _panel()
    host_report = _host_report(contract, candidate)
    monkeypatch.setattr(
        inference,
        "POLYMARKET_ROUND28_AI_MINIMUM_BASELINE_CANDIDATES",
        2,
    )
    monkeypatch.setattr(
        inference,
        "POLYMARKET_ROUND28_AI_MINIMUM_CHANGED_ACTIONS",
        1,
    )
    decisions = iter(
        (
            ("reject", ["liquidity_thin"]),
            ("unchanged", ["no_material_risk"]),
        )
    )
    times = iter(
        (
            0,
            100_000_000,
            200_000_000,
            300_000_000,
            400_000_000,
            500_000_000,
            600_000_000,
            700_000_000,
        )
    )
    residency_calls = 0

    def post_json(_url, payload, _timeout):
        if payload.get("keep_alive") == 0:
            return {}
        if str(payload.get("prompt", "")).startswith("Runtime conformance"):
            return _raw(candidate, "reject", ["bbo_source_stale_or_gapped"])
        decision, reasons = next(decisions)
        return _raw(candidate, decision, reasons)

    def residency(*_args, **_kwargs):
        nonlocal residency_calls
        residency_calls += 1
        return _residency(candidate, loaded=residency_calls == 1)

    report = run_round28_ai_inference(
        panel=panel,
        candidate=candidate,
        contract=contract,
        host_qualification_report=host_report,
        post_json=post_json,
        residency_inspector=residency,
        inventory_getter=lambda _url, _timeout: {
            "models": [{"digest": candidate.runtime_digest}]
        },
        monotonic_ns=lambda: next(times),
    )

    assert report.candidate_eligible_for_matched_evaluation is True
    assert [response.wall_latency_ms for response in report.responses] == [100, 100]
    assert report.changed_action_count == 1
    assert report.rejected_fraction == 0.5
    assert report.unload_observed is True
    assert round28_ai_inference_report_from_mapping(report.asdict()) == report
    assert (
        validate_round28_ai_inference_report(
            report.asdict(),
            contract=contract,
            host_qualification_report=host_report,
            panel=panel,
        )
        == report
    )


def test_round28_ai_invalid_output_becomes_reject_and_disqualifies(
    monkeypatch,
) -> None:
    import simple_ai_trading.polymarket_round28_ai_inference as inference

    contract = load_round28_ai_contract(ROOT)
    candidate = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    panel = _panel()
    host_report = _host_report(contract, candidate)
    monkeypatch.setattr(
        inference,
        "POLYMARKET_ROUND28_AI_MINIMUM_BASELINE_CANDIDATES",
        2,
    )
    times = iter(
        (
            0,
            100_000_000,
            200_000_000,
            300_000_000,
            400_000_000,
            500_000_000,
            600_000_000,
            700_000_000,
        )
    )
    residency_calls = 0

    def post_json(_url, payload, _timeout):
        if payload.get("keep_alive") == 0:
            return {}
        if str(payload.get("prompt", "")).startswith("Runtime conformance"):
            return _raw(candidate, "reject", ["bbo_source_stale_or_gapped"])
        raw = _raw(candidate, "unchanged", ["no_material_risk"])
        raw["response"] = "not-json"
        return raw

    def residency(*_args, **_kwargs):
        nonlocal residency_calls
        residency_calls += 1
        return _residency(candidate, loaded=residency_calls == 1)

    report = run_round28_ai_inference(
        panel=panel,
        candidate=candidate,
        contract=contract,
        host_qualification_report=host_report,
        post_json=post_json,
        residency_inspector=residency,
        inventory_getter=lambda _url, _timeout: {
            "models": [{"digest": candidate.runtime_digest}]
        },
        monotonic_ns=lambda: next(times),
    )

    assert report.status_counts == {"invalid_response": 2}
    assert all(response.decision == "reject" for response in report.responses)
    assert report.candidate_eligible_for_matched_evaluation is False


def test_round28_ai_request_has_strict_bounded_output() -> None:
    contract = load_round28_ai_contract(ROOT)
    candidate = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    request = round28_ai_inference_request(_panel().cases[0], candidate)

    assert request["think"] is False
    assert request["options"] == {
        "temperature": 0,
        "seed": 28,
        "num_ctx": 8192,
        "num_predict": 96,
    }
    assert request["format"]["additionalProperties"] is False
    assert "enum" in request["format"]["properties"]["reason_codes"]["items"]


def test_round28_ai_inference_rejects_nonexclusive_residency() -> None:
    contract = load_round28_ai_contract(ROOT)
    candidate = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    host_report = _host_report(contract, candidate)
    times = iter((0, 100_000_000))

    with pytest.raises(RuntimeError, match="exclusively GPU resident"):
        run_round28_ai_inference(
            panel=_panel(),
            candidate=candidate,
            contract=contract,
            host_qualification_report=host_report,
            post_json=lambda _url, payload, _timeout: (
                {}
                if payload.get("keep_alive") == 0
                else _raw(candidate, "reject", ["bbo_source_stale_or_gapped"])
            ),
            residency_inspector=lambda *_args, **_kwargs: _residency(
                candidate,
                loaded=True,
            ),
            inventory_getter=lambda _url, _timeout: {
                "models": [
                    {"digest": candidate.runtime_digest},
                    {"digest": "f" * 64},
                ]
            },
            monotonic_ns=lambda: next(times),
        )
