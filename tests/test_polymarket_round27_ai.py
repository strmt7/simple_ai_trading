from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading.polymarket_round27_ai import (
    POLYMARKET_ROUND27_ODA_HOST_CANDIDATE,
    POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE,
    Round27AIHostCandidate,
    probe_round27_oda_host,
    probe_round27_qwen_host,
    round27_ai_conformance_request,
)


_ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _provider_response(
    candidate: Round27AIHostCandidate = POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE,
) -> dict[str, object]:
    return {
        "model": candidate.runtime_model,
        "response": json.dumps(
            {"decision": "reject", "reason_codes": ["missing_liquidity"]}
        ),
        "done": True,
        "done_reason": "stop",
        "total_duration": 700_000_000,
        "load_duration": 200_000_000,
        "prompt_eval_count": 35,
        "prompt_eval_duration": 60_000_000,
        "eval_count": 30,
        "eval_duration": 370_000_000,
    }


def _residency(
    candidate: Round27AIHostCandidate,
    *_args: object,
    **kwargs: object,
) -> OllamaResidencyReport:
    assert kwargs["expected_digest"] == candidate.runtime_digest
    return OllamaResidencyReport(
        requested_model=candidate.runtime_model,
        status="gpu_resident",
        loaded_model=candidate.runtime_model,
        digest=candidate.runtime_digest,
        size_bytes=5_400_000_000,
        size_vram_bytes=5_400_000_000,
        vram_to_model_ratio=1.0,
    ).validated()


def _unloaded_residency(
    candidate: Round27AIHostCandidate,
    *_args: object,
    **kwargs: object,
) -> OllamaResidencyReport:
    assert kwargs["expected_digest"] == candidate.runtime_digest
    return OllamaResidencyReport(
        requested_model=candidate.runtime_model,
        status="unloaded",
        loaded_model=None,
        digest=None,
        size_bytes=None,
        size_vram_bytes=None,
        vram_to_model_ratio=None,
    ).validated()


@pytest.mark.parametrize(
    "candidate",
    (
        POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE,
        POLYMARKET_ROUND27_ODA_HOST_CANDIDATE,
    ),
)
def test_round27_ai_request_is_target_free_and_disables_thinking(
    candidate: Round27AIHostCandidate,
) -> None:
    request = round27_ai_conformance_request(candidate, keep_alive="30s")

    assert request["model"] == candidate.runtime_model
    assert request["think"] is False
    assert request["stream"] is False
    assert request["options"] == {
        "temperature": 0,
        "seed": 27,
        "num_ctx": 2048,
        "num_predict": 96,
    }
    serialized = json.dumps(request, sort_keys=True).lower()
    assert "no market data, target, outcome" in serialized
    assert "price" not in serialized
    assert "pnl" not in serialized


@pytest.mark.parametrize(
    ("candidate", "probe"),
    (
        (POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE, probe_round27_qwen_host),
        (POLYMARKET_ROUND27_ODA_HOST_CANDIDATE, probe_round27_oda_host),
    ),
)
def test_round27_ai_host_probe_requires_exact_gpu_residency(
    candidate: Round27AIHostCandidate,
    probe: Callable[..., dict[str, object]],
) -> None:
    requests: list[dict[str, object]] = []
    ticks = iter((0, 900_000_000, 1_000_000_000, 1_700_000_000))
    residencies = iter(
        (
            _residency(candidate, expected_digest=candidate.runtime_digest),
            _unloaded_residency(
                candidate,
                expected_digest=candidate.runtime_digest,
            ),
        )
    )

    def post(
        _url: str,
        payload: dict[str, object],
        _timeout: float,
    ) -> object:
        requests.append(payload)
        if payload.get("keep_alive") == 0:
            return {"done": True}
        return _provider_response(candidate)

    report = probe(
        post_json=post,
        residency_inspector=lambda *_args, **_kwargs: next(residencies),
        monotonic_ns=lambda: next(ticks),
    )

    assert report["passed"] is True
    assert report["claims"] == {
        "host_runtime_qualified": True,
        "offline_matched_ablation_eligible": True,
        "latency_critical_probability_predictor": False,
        "predictive_uplift": False,
        "after_cost_uplift": False,
        "edge": False,
        "profitability": False,
        "live_trading_authority": False,
    }
    assert [request.get("keep_alive") for request in requests] == [0, "30s", "30s", 0]
    assert report["post_unload_residency"]["status"] == "unloaded"


def test_round27_qwen_host_probe_rejects_nonconforming_output() -> None:
    ticks = iter((0, 100_000_000))

    def post(
        _url: str,
        payload: dict[str, object],
        _timeout: float,
    ) -> object:
        if payload.get("keep_alive") == 0:
            return {"done": True}
        response = _provider_response()
        response["response"] = json.dumps(
            {"decision": "unchanged", "reason_codes": []}
        )
        return response

    with pytest.raises(ValueError, match="conformance decision"):
        probe_round27_qwen_host(
            post_json=post,
            residency_inspector=lambda *args, **kwargs: _residency(
                POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE,
                *args,
                **kwargs,
            ),
            monotonic_ns=lambda: next(ticks),
        )


def test_round27_ai_host_publication_is_source_bound_and_nonpromotional() -> None:
    path = (
        _ROOT
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-027-ai-host-qualification-v1-2026-08-15.json"
    )
    report = json.loads(path.read_text(encoding="ascii"))
    claimed = report.pop("evidence_sha256")

    assert claimed == _canonical_sha256(report)
    assert report["qualification"] == {
        "qwen_host_runtime_qualified": True,
        "oda_host_runtime_qualified": True,
        "all_preregistered_candidates_host_qualified": True,
        "matched_after_cost_ai_ablation_complete": False,
        "ai_promoted": False,
        "reason": (
            "Both candidates are eligible only for the later target-free "
            "matched ablation. No Stage 1 outcomes were accessed."
        ),
    }
    assert report["data_authority"] == {
        "market_data_rows": 0,
        "targets": 0,
        "outcomes": 0,
        "resolutions": 0,
        "credentials_used": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    assert report["source_sha256"] == {
        "src/simple_ai_trading/polymarket_round27_ai.py": _file_sha256(
            _ROOT / "src" / "simple_ai_trading" / "polymarket_round27_ai.py"
        ),
        "tools/probe_polymarket_round27_ai_host.py": _file_sha256(
            _ROOT / "tools" / "probe_polymarket_round27_ai_host.py"
        ),
    }
