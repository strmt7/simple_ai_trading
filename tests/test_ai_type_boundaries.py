from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from simple_ai_trading import ai_review, ai_start_gate
from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading.ai_uplift import _model_parameters_b


def _valid_ai_review_report(status: str) -> ai_review.AIReviewReport:
    active = status != "blocked"
    digest = "d" * 64
    residency = OllamaResidencyReport(
        requested_model="qwen3:8b",
        status="gpu_resident",
        loaded_model="qwen3:8b",
        digest=digest,
        size_bytes=100,
        size_vram_bytes=100,
        vram_to_model_ratio=1.0,
    ).validated()
    capability = {
        "ok": True,
        "provider": "ollama",
        "model": "qwen3:8b",
        "compute_backend_kind": "directml",
        "provider_available": True,
        "model_available": True,
        "model_local": True,
        "provider_runtime": residency.asdict(),
    }
    action = {
        "ok": "approve",
        "review_required": "needs_human_review",
        "blocked": "veto",
    }[status]
    report = ai_review.AIReviewReport(
        schema_version=ai_review.AI_REVIEW_REPORT_SCHEMA_VERSION,
        created_at_ms=1,
        status=status,
        approved=status == "ok",
        source_report="source.json",
        source_report_sha256="1" * 64,
        provider="ollama",
        model="qwen3:8b",
        model_digest=digest if active else None,
        model_metadata_sha256="e" * 64 if active else None,
        endpoint="http://127.0.0.1:11434/api/chat",
        latency_ms=1,
        prompt_sha256="a" * 64 if active else ai_review._EMPTY_TEXT_SHA256,
        request_sha256="b" * 64 if active else None,
        response_sha256="c" * 64 if active else None,
        decision=ai_review.AIReviewDecision(action, 0.8, 0.2, "bounded", [], []),
        deterministic_precheck={},
        capability=capability if active else None,
        prompt_chars=1 if active else 0,
        error=None if active else "blocked",
    )
    return ai_review._finalize_report(report)


@pytest.mark.parametrize(
    "changes",
    (
        {"action": "invalid"},
        {"confidence": True},
        {"confidence": float("nan")},
        {"confidence": 1.1},
        {"risk_score": True},
        {"risk_score": -0.1},
        {"rationale": ""},
        {"rationale": "x" * 241},
        {"concerns": "not-a-list"},
        {"concerns": ["x"] * 9},
        {"concerns": [""]},
        {"required_actions": ["x"] * 9},
        {"required_actions": ["x" * 241]},
    ),
)
def test_ai_review_decision_rejects_each_field_family(
    changes: dict[str, object],
) -> None:
    decision = replace(
        ai_review.AIReviewDecision("approve", 0.8, 0.2, "bounded", [], []),
        **changes,
    )

    with pytest.raises(ValueError, match="AI review decision is invalid"):
        decision.validated()


@pytest.mark.parametrize(
    ("inventory", "message"),
    (
        (None, "inventory is malformed"),
        ({"models": {}}, "inventory is malformed"),
        ({"models": [None]}, "inventory entry is malformed"),
        ({"models": []}, "provenance is unavailable"),
        (
            {
                "models": [
                    {"name": "qwen3:latest", "digest": "a" * 64},
                    {"name": "qwen3:latest", "digest": "b" * 64},
                ]
            },
            "provenance is ambiguous",
        ),
        (
            {"models": [{"name": "qwen3:latest", "digest": "bad"}]},
            "digest is invalid",
        ),
    ),
)
def test_ollama_identity_rejects_inventory_before_metadata_request(
    inventory: object,
    message: str,
) -> None:
    metadata_called = False

    def post_json(*_args: object) -> object:
        nonlocal metadata_called
        metadata_called = True
        raise AssertionError("metadata must not be requested")

    with pytest.raises(ValueError, match=message):
        ai_review.resolve_ollama_model_identity(
            "http://127.0.0.1:11434",
            "qwen3",
            1.0,
            get_json=lambda *_args: inventory,
            post_json=post_json,
        )
    assert metadata_called is False


@pytest.mark.parametrize("status", ("ok", "review_required", "blocked"))
def test_ai_review_report_validation_accepts_each_canonical_status(status: str) -> None:
    report = _valid_ai_review_report(status)

    assert report.validated() == report


@pytest.mark.parametrize(
    "changes",
    (
        {"schema_version": "wrong"},
        {"created_at_ms": True},
        {"status": "wrong"},
        {"capability": {}},
        {"source_report_sha256": "bad"},
        {"model_digest": None},
        {"endpoint": ""},
        {"prompt_chars": 0},
        {"deterministic_precheck": []},
        {"report_sha256": "0" * 64},
    ),
)
def test_ai_review_report_validation_rejects_each_field_family(
    changes: dict[str, object],
) -> None:
    report = replace(_valid_ai_review_report("ok"), **changes)
    if "report_sha256" not in changes:
        report = replace(
            report,
            report_sha256=ai_review._canonical_sha256(report.identity_payload()),
        )

    with pytest.raises(ValueError, match="AI review report is invalid"):
        report.validated()


def test_ai_review_report_validation_preserves_decision_error_precedence() -> None:
    report = replace(
        _valid_ai_review_report("ok"),
        decision=ai_review.AIReviewDecision(
            "invalid",
            0.8,
            0.2,
            "bounded",
            [],
            [],
        ),
    )

    with pytest.raises(ValueError, match="AI review decision is invalid"):
        report.validated()


def test_ai_review_transport_and_numeric_boundaries_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> object:
            return {"status": "ok"}

    def fake_post(url: str, **kwargs: object) -> Response:
        observed["url"] = url
        observed.update(kwargs)
        return Response()

    monkeypatch.setattr(ai_review.requests, "post", fake_post)
    payload = {
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": "review"}],
        "stream": False,
    }

    assert ai_review._post_json("http://127.0.0.1/api/chat", payload, 0.0) == {
        "status": "ok"
    }
    assert observed["json"] == payload
    assert observed["timeout"] == pytest.approx(0.1)
    with pytest.raises(TypeError):
        ai_review._post_json(
            "http://127.0.0.1/api/chat",
            {"unsupported": object()},
            1.0,
        )

    assert ai_review._finite("1.25") == pytest.approx(1.25)
    assert ai_review._finite(True, -1.0) == -1.0
    assert ai_review._finite(object(), -1.0) == -1.0
    assert ai_review._optional_finite("2.5") == pytest.approx(2.5)
    assert ai_review._optional_finite(False) is None
    assert ai_review._optional_finite(object()) is None
    assert _model_parameters_b("", True) is None  # type: ignore[arg-type]

    compact = ai_review._compact_model_lab_report(
        {
            "requested_objectives": object(),
            "accepted_symbols": object(),
            "portfolio_risk": {"accepted_symbols": object()},
        }
    )
    assert compact["requested_objectives"] == []
    assert compact["accepted_symbols"] == []
    assert compact["portfolio_risk"]["accepted_symbols"] == []


def test_ai_review_compaction_handles_sparse_and_nested_evidence() -> None:
    assert ai_review._compact_stress_validation(None) is None
    assert ai_review._compact_robustness_validation(None) is None
    assert ai_review._compact_regime_validation(None) is None
    assert ai_review._compact_meta_label_validation(None) == {}
    assert ai_review._compact_portfolio_risk(None) is None

    nested_regime = {"window_count": 2, "dominant_regime": "trend_up"}
    assert (
        ai_review._regime_validation_source({}, {"regime_summary": nested_regime})
        == nested_regime
    )
    assert ai_review._regime_validation_source({}, {}) is None
    assert ai_review._compact_meta_label_validation(
        {
            "ignored": None,
            "regular": {"status": "accepted", "sample_count": 10},
        }
    ) == {
        "regular": {
            "status": "accepted",
            "sample_count": 10,
            "take_count": 0,
            "downsize_count": 0,
            "skip_count": 0,
            "take_precision": 0.0,
            "target_precision": 0.0,
        }
    }


def test_ai_review_compaction_collects_nested_market_edge_reports() -> None:
    accepted = {"accepted": True, "net_edge_pct": 0.3}
    rejected = {"accepted": False, "net_edge_pct": -0.2, "reason": "weak"}
    compact = ai_review._compact_market_edge_validation(
        {
            "market_edge": accepted,
            "objectives": [
                None,
                {
                    "market_edge": accepted,
                    "results": [
                        None,
                        {},
                        {"result": None},
                        {"result": {"market_edge": rejected}},
                    ],
                    "windows": "not-a-list",
                },
            ],
        }
    )

    assert compact == {
        "market_edge_accepted": False,
        "worst_market_edge_pct": -0.2,
        "market_edge_failed_reasons": ["weak"],
    }


def test_ai_review_selection_warnings_stop_at_evidence_bound() -> None:
    outcomes = [
        {
            "symbol": f"SYMBOL{index}",
            "selection_risk": {
                "regular": {
                    "passed": False,
                    "reason": "selection_risk_failed",
                }
            },
        }
        for index in range(ai_review._MAX_CONCERNS)
    ]

    warnings = ai_review._selection_risk_precheck_warnings({"outcomes": outcomes})

    assert len(warnings) == ai_review._MAX_CONCERNS


def test_ai_review_residency_rejects_unloaded_model(tmp_path: Path) -> None:
    context = ai_review._AIReviewRunContext(
        source_path=tmp_path / "source.json",
        output_path=tmp_path / "review.json",
        created_at_ms=1,
        source_report_sha256="a" * 64,
        provider="ollama",
        selected_model="qwen3:8b",
        provider_root="http://127.0.0.1:11434",
        endpoint="http://127.0.0.1:11434/api/chat",
        compact={},
        precheck={},
        timeout_seconds=1.0,
    )

    with pytest.raises(ValueError, match="not resident after inference"):
        ai_review._validated_review_residency(
            context,
            "d" * 64,
            lambda *_args, **_kwargs: OllamaResidencyReport(
                requested_model="qwen3:8b",
                status="unloaded",
                loaded_model=None,
                digest=None,
                size_bytes=None,
                size_vram_bytes=None,
                vram_to_model_ratio=None,
            ).validated(),
        )


def test_ai_json_loaders_map_json_and_encoding_failures(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="source model-lab report is unreadable"):
        ai_start_gate._strict_json_mapping(source)

    source.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="source model-lab report is unreadable"):
        ai_start_gate._strict_json_mapping(source)

    review = tmp_path / "review.json"
    review.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="artifact is unreadable"):
        ai_review.load_ai_review_report(review)
