from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from simple_ai_trading.ai_review import (
    load_ai_review_report,
    run_model_lab_ai_review,
)
from simple_ai_trading.ai_runtime import AICapabilityReport
from simple_ai_trading.terminal_holdout_ledger import terminal_result_fingerprint
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
        model_parameters_b=7.0 if ok else None,
        messages=() if ok else ("AI requires a GPU compute backend",),
        warnings=(),
    )


def _approve_response() -> dict[str, object]:
    return {
        "message": {
            "content": json.dumps({
                "action": "approve",
                "confidence": 0.82,
                "risk_score": 0.21,
                "rationale": "Deterministic and portfolio gates passed.",
                "concerns": ["continue paper monitoring"],
                "required_actions": ["keep stress reports attached"],
            })
        }
    }


def _write_report(
    path: Path,
    *,
    accepted: bool = True,
    harmful_ablation: bool = False,
    harmful_selection_risk: bool = False,
    include_ai_uplift: bool = True,
    failed_ai_uplift: bool = False,
) -> None:
    feature_delta = 0.004 if harmful_ablation else -0.004
    hybrid_delta = 0.003 if harmful_ablation else -0.003
    deflated_score = -0.02 if harmful_selection_risk else 0.11
    ai_uplift = None
    if include_ai_uplift:
        ai_uplift = {
            "schema_version": "ai-uplift-v2",
            "accepted": not failed_ai_uplift,
            "advisory_only": failed_ai_uplift,
            "trading_authority": False,
            "profitability_claim": False,
            "model_name": "qwen2.5:7b",
            "model_parameters_b": 7.0,
            "evidence_binding": {
                "accepted": True,
                "reasons": [],
                "dataset_fingerprint": "d" * 64,
                "baseline_evidence_sha256": "b" * 64,
                "ai_evidence_sha256": "a" * 64,
                "model_artifact_sha256": "c" * 64,
                "paired_samples_sha256": "e" * 64,
            },
            "baseline": {
                "realized_pnl": 6.0,
                "roi_pct": 0.06,
                "max_drawdown": 0.02,
                "expectancy": 0.6,
                "profit_factor": 1.4,
                "closed_trades": 8,
                "win_rate": 0.55,
                "liquidation_events": 0,
                "max_consecutive_losses": 2,
                "downside_return_risk_ratio": 1.2,
            },
            "ai": {
                "realized_pnl": 7.5 if not failed_ai_uplift else 5.0,
                "roi_pct": 0.075 if not failed_ai_uplift else 0.05,
                "max_drawdown": 0.018 if not failed_ai_uplift else 0.04,
                "expectancy": 0.8 if not failed_ai_uplift else 0.4,
                "profit_factor": 1.6 if not failed_ai_uplift else 1.1,
                "closed_trades": 9,
                "win_rate": 0.60 if not failed_ai_uplift else 0.50,
                "liquidation_events": 0,
                "max_consecutive_losses": 1 if not failed_ai_uplift else 3,
                "downside_return_risk_ratio": 1.4 if not failed_ai_uplift else 1.0,
            },
            "deltas": {
                "realized_pnl": 1.5 if not failed_ai_uplift else -1.0,
                "roi_pct": 0.015 if not failed_ai_uplift else -0.01,
                "max_drawdown": -0.002 if not failed_ai_uplift else 0.02,
                "expectancy": 0.2 if not failed_ai_uplift else -0.2,
                "profit_factor": 0.2 if not failed_ai_uplift else -0.3,
                "closed_trades": 1,
                "win_rate": 0.05 if not failed_ai_uplift else -0.05,
                "liquidation_events": 0,
                "max_consecutive_losses": -1 if not failed_ai_uplift else 1,
                "downside_return_risk_ratio": 0.2 if not failed_ai_uplift else -0.2,
            },
            "statistical_evidence": {
                "accepted": not failed_ai_uplift,
                "reasons": [] if not failed_ai_uplift else ["ai_uplift_sign_test_p_value>0.0500"],
                "evidence_unit": "matched_fixed_period_return_delta",
                "scope": "AAAUSDC",
                "sample_count": 30,
                "min_sample_count": 30,
                "positive_delta_count": 30 if not failed_ai_uplift else 12,
                "positive_delta_rate": 1.0 if not failed_ai_uplift else 0.4,
                "min_positive_delta_rate": 0.55,
                "sign_test_p_value": 2**-30 if not failed_ai_uplift else 0.899755,
                "max_sign_test_p_value": 0.05,
                "mean_delta": 0.002 if not failed_ai_uplift else -0.001,
                "median_delta": 0.0018 if not failed_ai_uplift else -0.0005,
                "min_mean_sample_delta": 0.0,
                "paired_sample_length_mismatch": False,
                "period_duration_ms": 3 * 86_400_000,
                "first_period_start_ms": 1_700_000_000_000,
                "last_period_end_ms": 1_707_776_000_000,
                "paired_samples_sha256": "e" * 64,
                "block_bootstrap_samples": 2_000,
                "block_bootstrap_confidence": 0.95,
                "block_length": 5,
                "mean_delta_ci_lower": 0.001 if not failed_ai_uplift else -0.002,
                "mean_delta_ci_upper": 0.003,
                "positive_mean_probability": 1.0 if not failed_ai_uplift else 0.2,
                "min_bootstrap_mean_delta_lower": 0.0,
                "evaluation_span_ms": 90 * 86_400_000,
                "min_evaluation_span_ms": 90 * 86_400_000,
            },
            "reasons": [] if not failed_ai_uplift else ["ai_pnl_not_above_baseline"],
            "policy": {
                "min_model_parameters_b": 2.0,
                "min_paired_samples": 30,
                "min_positive_delta_rate": 0.55,
                "max_sign_test_p_value": 0.05,
                "block_bootstrap_samples": 2_000,
                "block_bootstrap_confidence": 0.95,
                "min_bootstrap_mean_delta_lower": 0.0,
                "min_evaluation_span_days": 90,
            },
        }
    payload = {
        "quote_asset": "USDC",
        "interval": "15m",
        "market_type": "futures",
        "requested_objectives": ["regular"],
        "accepted_symbols": ["AAAUSDC"] if accepted else [],
        "portfolio_risk": {
            "accepted": accepted,
            "reason": None if accepted else "symbols<2",
            "effective_symbol_count": 1.0 if accepted else 0.0,
            "correlation_adjusted_effective_symbol_count": 1.0 if accepted else 0.0,
            "max_pairwise_correlation": 0.42,
            "max_cluster_weight": 0.40,
            "portfolio_cvar_95": 0.002,
            "portfolio_max_drawdown": 0.01,
            "deployed_weight": 0.20,
            "accepted_symbols": ["AAAUSDC"] if accepted else [],
        },
        "outcomes": [
            {
                "symbol": "AAAUSDC",
                "accepted": accepted,
                "rows": 500,
                "data_coverage": {
                    "symbol": "AAAUSDC",
                    "market_type": "futures",
                    "interval": "15m",
                    "source_scope": "binance_full_history",
                    "expected_interval_ms": 900000,
                    "integrity_status": "ok",
                    "integrity_warnings": [],
                    "truth_basis": [
                        "prices_from_timestamped_closed_candles",
                        "coverage_measured_from_candle_close_time",
                        "execution_results_are_simulated_not_exchange_fills",
                    ],
                    "full_history_requested": True,
                    "full_available_history_used": True,
                    "candles_available": 70080,
                    "candles_used": 70080,
                    "rows_used": 500,
                    "requested_start_ms": None,
                    "requested_end_ms": None,
                    "available_start_ms": 1640995200000,
                    "available_end_ms": 1704078900000,
                    "used_start_ms": 1640995200000,
                    "used_end_ms": 1704078900000,
                    "available_start_utc": "2022-01-01T00:00:00Z",
                    "available_end_utc": "2024-01-01T00:15:00Z",
                    "used_start_utc": "2022-01-01T00:00:00Z",
                    "used_end_utc": "2024-01-01T00:15:00Z",
                    "used_duration_days": 730.0104166667,
                    "used_duration_years": 1.9986595939,
                    "gap_count": 0,
                    "largest_gap_ms": 900000,
                    "largest_gap_intervals": 1.0,
                    "coverage_ratio": 1.0,
                    "notes": [],
                },
                "objective_scores": {"regular": 0.15},
                "hybrid_profiles": {"regular": "balanced_neighbors"},
                "walk_forward_gate": {
                    "regular": {
                        "objective": "regular",
                        "passed": True,
                        "reason": None,
                        "fold_count": 3,
                        "accepted_folds": 3,
                        "worst_score": 0.08,
                        "worst_realized_pnl": 1.2,
                        "worst_max_drawdown": 0.025,
                    }
                },
                "selection_risk": {
                    "regular": {
                        "passed": not harmful_selection_risk,
                        "reason": (
                            "selection_risk_deflated_score<=0"
                            if harmful_selection_risk
                            else None
                        ),
                        "reasons": (
                            ["selection_risk_deflated_score<=0"]
                            if harmful_selection_risk
                            else []
                        ),
                        "effective_trials": 900,
                        "finite_candidate_scores": 18,
                        "selected_score": 0.15,
                        "runner_up_score": 0.12,
                        "median_score": 0.02,
                        "score_iqr": 0.04,
                        "trial_penalty": 0.17 if harmful_selection_risk else 0.04,
                        "deflated_score": deflated_score,
                        "score_margin_to_runner_up": 0.03,
                        "terminal_holdout": {
                            "schema_version": "terminal-holdout-v1",
                            "passed": not harmful_selection_risk,
                            "reason": "terminal_holdout_failed" if harmful_selection_risk else None,
                            "evaluation_count": 1,
                            "rows": 100,
                            "start_timestamp": 1_000,
                            "end_timestamp": 2_000,
                            "score": None if harmful_selection_risk else 0.10,
                            "dataset_fingerprint": "a" * 64,
                            "reservation": {
                                "schema_version": "terminal-holdout-reservation-v1",
                                "reservation_id": "1" * 64,
                                "ledger_id": "2" * 64,
                                "symbol": "BTCUSDT",
                                "market_type": "futures",
                                "objective": "regular",
                                "first_timestamp": 1_000,
                                "last_timestamp": 2_000,
                                "rows": 100,
                                "dataset_fingerprint": "a" * 64,
                                "model_fingerprint": "b" * 64,
                                "result_fingerprint": "c" * 64,
                                "status": "complete",
                                "result_status": "accepted" if not harmful_selection_risk else "rejected",
                                "error": "",
                                "reserved_at_ms": 1_000,
                                "completed_at_ms": 2_000,
                            },
                            "result": {
                                "accepted": not harmful_selection_risk,
                                "realized_pnl": -10.0 if harmful_selection_risk else 10.0,
                                "stopped_by_liquidation": False,
                                "liquidation_events": 0,
                            },
                        },
                        "overfit_diagnostics": {
                            "status": "available",
                            "passed": not harmful_selection_risk,
                            "reason": (
                                "selection_risk_pbo>0.50"
                                if harmful_selection_risk
                                else None
                            ),
                            "probability_backtest_overfit": 0.75 if harmful_selection_risk else 0.0,
                            "max_probability_backtest_overfit": 0.50,
                        },
                    }
                },
                "hybrid_ablation": {
                    "regular": [
                        {
                            "removed_expert_kind": "lorentzian_knn",
                            "accepted": True,
                            "score": 0.147 + hybrid_delta,
                            "delta_vs_best": hybrid_delta,
                        }
                    ]
                },
                "feature_ablation": {
                    "regular": [
                        {
                            "removed_group": "technical_confluence",
                            "status": "evaluated",
                            "accepted": True,
                            "score": 0.15 + feature_delta,
                            "delta_vs_selected": feature_delta,
                            "realized_pnl": 3.0,
                            "max_drawdown": 0.01,
                            "closed_trades": 7,
                        }
                    ]
                },
                "ai_uplift": ai_uplift,
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
                "meta_label_validation": {
                    "regular": {
                        "status": "trained",
                        "sample_count": 24,
                        "take_count": 12,
                        "downsize_count": 5,
                        "skip_count": 7,
                        "take_precision": 0.75,
                        "target_precision": 0.60,
                    }
                },
            }
        ],
    }
    terminal = payload["outcomes"][0]["selection_risk"]["regular"]["terminal_holdout"]
    reservation = terminal["reservation"]
    reservation["result_fingerprint"] = terminal_result_fingerprint(terminal)
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
    assert review.validated() == review
    assert review.source_report_sha256 == sha256(report_path.read_bytes()).hexdigest()
    assert len(review.prompt_sha256) == 64
    assert len(review.request_sha256 or "") == 64
    assert len(review.response_sha256 or "") == 64
    assert len(review.report_sha256) == 64
    with pytest.raises(ValueError, match="AI review report is invalid"):
        replace(review, source_report_sha256="0" * 64).validated()
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
    assert "meta_label_validation" in prompt
    assert "selection_risk" in prompt
    assert "walk_forward_gate" in prompt
    assert "hybrid_ablation" in prompt
    assert "feature_ablation" in prompt
    assert "ai_uplift" in prompt
    assert "trend_up" in prompt
    assert (tmp_path / "ai_risk_review.json").exists()
    stored = json.loads((tmp_path / "ai_risk_review.json").read_text(encoding="utf-8"))
    assert stored["report_sha256"] == review.report_sha256
    assert load_ai_review_report(
        tmp_path / "ai_risk_review.json",
        expected_source_report=report_path,
    ) == review
    report_path.write_bytes(report_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source report digest differs"):
        load_ai_review_report(tmp_path / "ai_risk_review.json")


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


def test_ai_review_blocks_before_model_call_on_harmful_ablation(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=True, harmful_ablation=True)
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
    assert review.deterministic_precheck["ablation_warning_count"] == 2
    assert "ablation evidence" in str(review.error)
    assert any("technical_confluence" in item for item in review.deterministic_precheck["ablation_warnings"])


def test_ai_review_blocks_before_model_call_on_failed_selection_risk(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=True, harmful_selection_risk=True)
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
    assert review.deterministic_precheck["selection_risk_warning_count"] == 1
    assert "selection-risk evidence" in str(review.error)
    assert "deflated_score=-0.02" in review.deterministic_precheck["selection_risk_warnings"][0]


def test_ai_review_blocks_before_model_call_without_terminal_holdout(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=True)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    del payload["outcomes"][0]["selection_risk"]["regular"]["terminal_holdout"]
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    called = False

    def fake_post(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("AI provider should not be called")

    monkeypatch.setattr("simple_ai_trading.ai_review.detect_ai_capabilities", lambda _cfg: _capability(True))
    review = run_model_lab_ai_review(
        report_path,
        RuntimeConfig(compute_backend="directml"),
        post_json=fake_post,
    )

    assert called is False
    assert review.approved is False
    assert review.deterministic_precheck["selection_risk_warning_count"] == 1
    assert "terminal_holdout_missing_or_failed" in review.deterministic_precheck["selection_risk_warnings"][0]


def test_ai_review_blocks_before_model_call_when_ai_uplift_missing(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=True, include_ai_uplift=False)
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
    assert review.deterministic_precheck["ai_uplift_warning_count"] == 1
    assert "AI-vs-ML uplift" in str(review.error)
    assert "missing AI-vs-ML uplift evidence" in review.deterministic_precheck["ai_uplift_warnings"][0]


def test_ai_review_blocks_before_model_call_when_ai_uplift_fails(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=True, failed_ai_uplift=True)
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
    assert review.deterministic_precheck["ai_uplift_warning_count"] == 1
    assert "ai_pnl_not_above_baseline" in review.deterministic_precheck["ai_uplift_warnings"][0]


def test_ai_review_blocks_financially_unsound_accepted_report(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=True)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["outcomes"][0]["rows"] = 0
    report_path.write_text(json.dumps(payload), encoding="utf-8")
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
    assert review.deterministic_precheck["financial_sanity_warning_count"] >= 1
    assert "financial sanity" in str(review.error)


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


@pytest.mark.parametrize(
    "content",
    [
        (
            "prefix "
            '{"action":"approve","confidence":0.8,"risk_score":0.2,'
            '"rationale":"ok","concerns":[],"required_actions":[]}'
        ),
        (
            '{"action":"approve","action":"veto","confidence":0.8,'
            '"risk_score":0.2,"rationale":"ok","concerns":[],'
            '"required_actions":[]}'
        ),
        json.dumps(
            {
                "action": "approve",
                "confidence": "0.8",
                "risk_score": 0.2,
                "rationale": "ok",
                "concerns": [],
                "required_actions": [],
            }
        ),
        json.dumps(
            {
                "action": "approve",
                "confidence": 1.2,
                "risk_score": 0.2,
                "rationale": "ok",
                "concerns": [],
                "required_actions": [],
            }
        ),
        json.dumps(
            {
                "action": "approve",
                "confidence": 0.8,
                "risk_score": 0.2,
                "rationale": "ok",
                "concerns": [],
                "required_actions": [],
                "unexpected": True,
            }
        ),
    ],
)
def test_ai_review_rejects_noncanonical_structured_output(
    tmp_path: Path,
    monkeypatch,
    content: str,
) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=True)
    monkeypatch.setattr(
        "simple_ai_trading.ai_review.detect_ai_capabilities",
        lambda _cfg: _capability(True),
    )

    review = run_model_lab_ai_review(
        report_path,
        RuntimeConfig(compute_backend="directml"),
        post_json=lambda *_args, **_kwargs: {"message": {"content": content}},
    )

    assert review.approved is False
    assert review.status == "blocked"
    assert "AI review failed" in str(review.error)


def test_ai_review_rejects_oversized_prompt_without_calling_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=True)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["outcomes"][0]["diagnostics"] = {"unbounded": "x" * 20_000}
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    called = False

    def fake_post(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("oversized evidence must not reach the AI model")

    monkeypatch.setattr(
        "simple_ai_trading.ai_review.detect_ai_capabilities",
        lambda _cfg: _capability(True),
    )
    review = run_model_lab_ai_review(
        report_path,
        RuntimeConfig(compute_backend="directml"),
        post_json=fake_post,
    )

    assert called is False
    assert review.approved is False
    assert "exceeds AI prompt bound" in str(review.error)


def test_ai_review_loader_rejects_ambiguous_json_and_boolean_probability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "model_lab_report.json"
    review_path = tmp_path / "ai_risk_review.json"
    _write_report(report_path, accepted=True)
    monkeypatch.setattr(
        "simple_ai_trading.ai_review.detect_ai_capabilities",
        lambda _cfg: _capability(True),
    )
    review = run_model_lab_ai_review(
        report_path,
        RuntimeConfig(compute_backend="directml"),
        output_path=review_path,
        post_json=lambda *_args, **_kwargs: _approve_response(),
    )

    with pytest.raises(ValueError, match="AI review decision is invalid"):
        replace(
            review,
            decision=replace(review.decision, confidence=True),
        ).validated()

    original = review_path.read_text(encoding="utf-8")
    ambiguous = original.replace(
        '"status": "ok"',
        '"status": "ok", "status": "ok"',
        1,
    )
    assert ambiguous != original
    review_path.write_text(ambiguous, encoding="utf-8")
    with pytest.raises(ValueError, match="artifact is unreadable"):
        load_ai_review_report(
            review_path,
            expected_source_report=report_path,
        )


def test_ai_review_blocks_on_capability_failure(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "model_lab_report.json"
    _write_report(report_path, accepted=True)
    monkeypatch.setattr("simple_ai_trading.ai_review.detect_ai_capabilities", lambda _cfg: _capability(False))

    review = run_model_lab_ai_review(report_path, RuntimeConfig(compute_backend="cpu"))

    assert review.approved is False
    assert review.status == "blocked"
    assert "GPU compute backend" in str(review.error)
