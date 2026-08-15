from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading.polymarket_round28_ai_contract import (
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_host import (
    build_round28_ai_artifact_verification,
    probe_round28_ai_candidate_host,
    round28_ai_candidate_from_contract,
    round28_ai_conformance_request,
    validate_round28_ai_host_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _provider_response(model: str) -> dict[str, object]:
    return {
        "model": model,
        "response": json.dumps(
            {
                "decision": "reject",
                "reason_codes": ["bbo_source_stale_or_gapped"],
            },
            separators=(",", ":"),
        ),
        "done": True,
        "done_reason": "stop",
        "total_duration": 90_000_000,
        "load_duration": 10_000_000,
        "prompt_eval_count": 42,
        "eval_count": 8,
    }


def test_round28_ai_candidate_binds_registered_and_pending_digests() -> None:
    contract = load_round28_ai_contract(ROOT)
    control = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    challenger = round28_ai_candidate_from_contract(
        contract,
        model_id="OpenDataArena/ODA-Fin-RL-8B",
        observed_runtime_digest="f" * 64,
    )

    assert control.runtime_digest.startswith("6488c96f")
    assert challenger.runtime_digest == "f" * 64
    assert challenger.artifact_sha256.startswith("d40d1dd4")
    with pytest.raises(ValueError, match="observed runtime digest"):
        round28_ai_candidate_from_contract(
            contract,
            model_id="OpenDataArena/ODA-Fin-RL-8B",
        )


def test_round28_ai_host_probe_requires_exact_gpu_residency_and_unload() -> None:
    contract = load_round28_ai_contract(ROOT)
    candidate = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    times = iter((0, 100_000_000, 200_000_000, 300_000_000))
    residency_calls = 0
    artifact = build_round28_ai_artifact_verification(
        candidate,
        observed_sha256=candidate.artifact_sha256,
        observed_size_bytes=candidate.artifact_size_bytes,
        verification_method="inherited_round27_exact_artifact",
        source_evidence_sha256="a" * 64,
        observed_at_ms=1,
    )

    def post_json(_url, payload, _timeout):
        if payload.get("keep_alive") == 0:
            return {}
        return _provider_response(candidate.runtime_model)

    def residency(_base, model, _timeout, *, expected_digest):
        nonlocal residency_calls
        residency_calls += 1
        if residency_calls == 1:
            return OllamaResidencyReport(
                requested_model=model,
                status="gpu_resident",
                loaded_model=model,
                digest=expected_digest,
                size_bytes=1_000,
                size_vram_bytes=1_000,
                vram_to_model_ratio=1.0,
            )
        return OllamaResidencyReport(
            requested_model=model,
            status="unloaded",
            loaded_model=None,
            digest=None,
            size_bytes=None,
            size_vram_bytes=None,
            vram_to_model_ratio=None,
        )

    report = probe_round28_ai_candidate_host(
        candidate,
        artifact_verification=artifact,
        post_json=post_json,
        residency_inspector=residency,
        monotonic_ns=lambda: next(times),
    )
    validated, restored = validate_round28_ai_host_report(
        report,
        contract=contract,
    )

    assert report["passed"] is True
    assert validated["report_sha256"] == report["report_sha256"]
    assert restored == candidate
    assert report["measurements"][0]["wall_ms"] == 100
    assert report["artifact_verification"]["observed_sha256"] == (
        candidate.artifact_sha256
    )


def test_round28_ai_host_report_rejects_residency_or_hash_drift() -> None:
    contract = load_round28_ai_contract(ROOT)
    candidate = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    request = round28_ai_conformance_request(candidate, keep_alive="30s")

    assert request["think"] is False
    assert request["options"]["temperature"] == 0
    assert request["format"]["additionalProperties"] is False

    with pytest.raises(ValueError, match="keep-alive"):
        round28_ai_conformance_request(candidate, keep_alive="5m")

    # A report cannot be repaired by changing only its semantic fields.
    example = {
        "report_sha256": "a" * 64,
        "candidate": deepcopy(candidate.__dict__) if hasattr(candidate, "__dict__") else {},
    }
    with pytest.raises(ValueError):
        validate_round28_ai_host_report(example, contract=contract)
