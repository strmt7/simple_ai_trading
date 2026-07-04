from __future__ import annotations

import json
from pathlib import Path

from simple_ai_trading.ai_review import run_model_lab_ai_review
from simple_ai_trading.ai_runtime import AICapabilityReport
from simple_ai_trading.types import RuntimeConfig


def _capability(ok: bool = True) -> AICapabilityReport:
    return AICapabilityReport(
        ok=ok,
        provider="local-gpu",
        model="qwen2.5:7b",
        gpu_vendor="amd",
        compute_backend_requested="directml",
        compute_backend_kind="directml" if ok else "cpu",
        compute_backend_device="privateuseone:0" if ok else "cpu",
        compute_backend_reason="" if ok else "DirectML unavailable",
        free_vram_gb=12.0 if ok else None,
        free_ram_gb=32.0,
        messages=() if ok else ("AI requires a GPU compute backend",),
        warnings=(),
    )


def _write_report(path: Path, *, accepted: bool = True) -> None:
    payload = {
        "quote_asset": "USDC",
        "interval": "15m",
        "market_type": "futures",
        "requested_objectives": ["regular"],
        "accepted_symbols": ["AAAUSDC", "BBBUSDC"] if accepted else [],
        "portfolio_risk": {
            "accepted": accepted,
            "reason": None if accepted else "symbols<2",
            "effective_symbol_count": 2.0 if accepted else 0.0,
            "max_pairwise_correlation": 0.42,
            "max_cluster_weight": 0.40,
            "portfolio_cvar_95": 0.002,
            "portfolio_max_drawdown": 0.01,
            "deployed_weight": 0.40,
            "accepted_symbols": ["AAAUSDC", "BBBUSDC"] if accepted else [],
        },
        "outcomes": [
            {
                "symbol": "AAAUSDC",
                "accepted": accepted,
                "rows": 500,
                "objective_scores": {"regular": 0.15},
                "hybrid_profiles": {"regular": "balanced_neighbors"},
                "stress_validation": {
                    "accepted": accepted,
                    "scenario_count": 4,
                    "worst_realized_pnl": 4.2,
                    "worst_max_drawdown": 0.01,
                },
                "robustness_validation": {
                    "accepted": accepted,
                    "window_count": 5,
                    "accepted_windows": 5 if accepted else 2,
                    "accepted_window_rate": 1.0 if accepted else 0.4,
                    "worst_realized_pnl": 2.0 if accepted else -3.0,
                    "worst_max_drawdown": 0.015,
                    "statistical_edge_accepted": accepted,
                    "worst_sign_test_p_value": 0.03125 if accepted else 0.8125,
                    "worst_bootstrap_lower_mean_return": 0.002 if accepted else -0.006,
                },
                "regime_validation": {
                    "window_count": 5,
                    "dominant_regime": "trend_up",
                    "dominant_regime_window_share": 0.8,
                    "accepted_regime_count": 2 if accepted else 0,
                    "concentration_warning": True,
                    "notes": ["window_regime_concentration"],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ai_review_uses_structured_ollama_response(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=True)
    observed = {}

    def fake_post(url, payload, timeout):
        observed["url"] = url
        observed["payload"] = payload
        observed["timeout"] = timeout
        return {
            "message": {
                "content": json.dumps({
                    "action": "approve",
                    "confidence": 0.82,
                    "risk_score": 0.21,
                    "rationale": "Deterministic and portfolio gates passed with low tail risk.",
                    "concerns": ["continue paper monitoring"],
                    "required_actions": ["keep stress reports attached"],
                })
            }
        }

    monkeypatch.setattr("simple_ai_trading.ai_review.detect_ai_capabilities", lambda _cfg: _capability(True))

    review = run_model_lab_ai_review(
        report_path,
        RuntimeConfig(compute_backend="directml", ai_model="qwen2.5:7b"),
        base_url="http://127.0.0.1:11434",
        timeout_seconds=7.0,
        post_json=fake_post,
    )

    assert review.approved is True
    assert review.status == "ok"
    assert review.decision.action == "approve"
    assert review.decision.risk_score == 0.21
    assert observed["url"].endswith("/api/chat")
    assert observed["payload"]["format"]["required"] == [
        "action",
        "confidence",
        "risk_score",
        "rationale",
        "concerns",
        "required_actions",
    ]
    prompt = observed["payload"]["messages"][1]["content"]
    assert "regime_validation" in prompt
    assert "trend_up" in prompt
    assert (tmp_path / "ai_risk_review.json").exists()


def test_ai_review_blocks_before_model_call_when_no_accepted_portfolio(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=False)
    called = False

    def fake_post(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("AI provider should not be called")

    monkeypatch.setattr("simple_ai_trading.ai_review.detect_ai_capabilities", lambda _cfg: _capability(True))

    review = run_model_lab_ai_review(report_path, RuntimeConfig(compute_backend="directml"), post_json=fake_post)

    assert called is False
    assert review.approved is False
    assert review.status == "blocked"
    assert review.decision.action == "veto"
    assert "deterministic gates" in review.error


def test_ai_review_fails_closed_on_invalid_ai_payload(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=True)
    monkeypatch.setattr("simple_ai_trading.ai_review.detect_ai_capabilities", lambda _cfg: _capability(True))

    review = run_model_lab_ai_review(
        report_path,
        RuntimeConfig(compute_backend="directml"),
        post_json=lambda *_args, **_kwargs: {"message": {"content": "{\"action\":\"approve\"}"}},
    )

    assert review.approved is False
    assert review.status == "blocked"
    assert "AI review failed" in str(review.error)


def test_ai_review_blocks_on_capability_failure(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=True)
    monkeypatch.setattr("simple_ai_trading.ai_review.detect_ai_capabilities", lambda _cfg: _capability(False))

    review = run_model_lab_ai_review(report_path, RuntimeConfig(compute_backend="cpu"))

    assert review.approved is False
    assert review.status == "blocked"
    assert "GPU compute backend" in str(review.error)
