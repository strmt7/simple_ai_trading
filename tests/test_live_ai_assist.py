from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from simple_ai_trading.advanced_model import advanced_feature_signature, default_config_for
from simple_ai_trading.autonomous import (
    AutonomousConfig,
    Decision,
    _close_to_trade,
    _entry_gate,
    _open_position_from_decision,
)
from simple_ai_trading.live_ai_assist import (
    AIAssistedDecisionFunction,
    AsyncLiveAIEntryReviewer,
    LiveAIEntryDecision,
    OllamaLiveAIEntryProvider,
    _parse_provider_decision,
    build_live_ai_entry_case,
)
from simple_ai_trading import live_ai_assist as live_ai_assist_module
from simple_ai_trading.positions import PositionsStore
from simple_ai_trading.types import RuntimeConfig, StrategyConfig
from simple_ai_trading.objective import get_objective


_DIGEST = "a" * 64
_FINGERPRINT = "b" * 64
_FEATURE_SIGNATURE = advanced_feature_signature(
    replace(
        default_config_for("conservative", StrategyConfig().enabled_features),
        label_threshold=0.0015,
        label_stop_threshold=0.0015,
    )
)


def _approval_strategy() -> StrategyConfig:
    return StrategyConfig(
        taker_fee_bps=4.0,
        slippage_bps=2.0,
        max_spread_bps=5.0,
    )


def _approval_evidence() -> dict[str, object]:
    return {
        "signal": {"after_cost_margin_bps": 3.2},
        "cost_model": {
            "configured_round_trip_cost_floor_bps": 13.0,
            "model_gross_label_barrier_bps": 15.0,
        },
        "model_validation": {
            "available": True,
            "probability_calibration": {
                "sample_count": 128,
                "brier_after": 0.20,
                "ece_after": 0.10,
            },
            "selection_risk": {"passed": True, "effective_trials": 24},
            "labeling": {
                "available": True,
                "gross_label_barrier_bps": 15.0,
            },
            "terminal_holdout": {
                "passed": True,
                "accepted": True,
                "liquidation_events": 0,
                "mean_after_cost_sample_return_bps": 2.0,
                "bootstrap_lower_mean_return": 0.0001,
                "market_edge": {
                    "accepted": True,
                    "sample_count": 40,
                    "minimum_sample_count": 6,
                    "financial_sanity_allowed": True,
                },
            },
            "execution_validation": {
                "passed": True,
                "walk_forward_passed": True,
                "stress_passed": True,
                "temporal_passed": True,
                "portfolio_passed": True,
                "microstructure_passed": True,
                "microstructure_seconds": 1_728_000,
                "microstructure_sequence_gaps": 0,
            },
        },
    }


def _validated_model_artifact() -> SimpleNamespace:
    return SimpleNamespace(
        model_family="advanced_logistic",
        model_selected_candidate="candidate-a",
        feature_signature=_FEATURE_SIGNATURE,
        probability_calibration_size=128,
        probability_log_loss_after=0.50,
        probability_brier_after=0.20,
        probability_ece_after=0.10,
        selection_risk={
            "passed": True,
            "effective_trials": 24,
            "deflated_score": 0.12,
            "terminal_holdout": {
                "passed": True,
                "result": {
                    "accepted": True,
                    "closed_trades": 40,
                    "realized_pnl": 12.0,
                    "max_drawdown": 0.03,
                    "total_fees": 5.0,
                    "edge_vs_buy_hold": 8.0,
                    "liquidation_events": 0,
                    "market_edge": {
                        "accepted": True,
                        "sample_count": 40,
                        "min_sample_count": 6,
                        "mean_sample_return": 0.0002,
                        "bootstrap_lower_mean_return": 0.0001,
                        "sign_test_p_value": 0.02,
                        "max_sign_test_p_value": 0.30,
                        "profit_factor": 1.4,
                        "min_profit_factor": 1.0,
                        "downside_return_risk_ratio": 0.8,
                        "min_downside_return_risk_ratio": 0.45,
                        "financial_sanity_allowed": True,
                    },
                },
            },
        },
        execution_validation={
            "passed": True,
            "walk_forward_gate": {
                "passed": True,
                "fold_count": 3,
                "worst_realized_pnl": 1.0,
                "worst_max_drawdown": 0.02,
            },
            "stress": {"accepted": True},
            "temporal_robustness": {"accepted": True},
            "portfolio": {"accepted": True},
            "data_coverage": {
                "used_duration_years": 2.0,
                "coverage_ratio": 1.0,
                "gap_count": 0,
            },
            "microstructure_replay": {
                "passed": True,
                "captured_seconds": 1_728_000,
                "sequence_gap_count": 0,
            },
        },
    )


def _case(*, observed_at_ms: int = 1_000, model_digest: str = _DIGEST):
    return build_live_ai_entry_case(
        symbol="BTCUSDC",
        market_type="futures",
        interval="15m",
        observed_at_ms=observed_at_ms,
        proposed_side="LONG",
        ml_confidence=0.72,
        maximum_risk_multiplier=0.4,
        model_digest=model_digest,
        terminal_model_fingerprint=_FINGERPRINT,
        evidence=_approval_evidence(),
    )


def _approval() -> LiveAIEntryDecision:
    return LiveAIEntryDecision(
        action="approve",
        risk_multiplier=0.4,
        confidence=0.8,
        reason_codes=("edge_after_costs", "liquidity_acceptable"),
        summary="After-cost edge and liquidity evidence are coherent.",
        valid=True,
        response_sha256="c" * 64,
        observed_model_digest=_DIGEST,
        model_residency_status="gpu_resident",
        prompt_tokens=100,
        output_tokens=20,
    )


def _wait_for_review(reviewer: AsyncLiveAIEntryReviewer, case, expected: str):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        review = reviewer.review(case)
        if review.status == expected:
            return review
        time.sleep(0.005)
    pytest.fail(f"AI review did not reach {expected}")


def test_case_identity_is_deterministic_and_tamper_evident() -> None:
    first = _case()
    second = _case()
    assert first.case_id == second.case_id

    tampered = object.__new__(type(first))
    object.__setattr__(tampered, "case_id", first.case_id)
    for name, value in first.__dict__.items():
        if name != "case_id":
            object.__setattr__(tampered, name, value)
    object.__setattr__(tampered, "ml_confidence", 0.99)
    with pytest.raises(ValueError, match="identity mismatch"):
        tampered.validated()


def test_provider_parser_is_exact_and_semantically_fail_closed() -> None:
    content = json.dumps(
        {
            "action": "approve",
            "risk_multiplier": 0.4,
            "confidence": 0.8,
            "reason_codes": ["edge_after_costs"],
            "summary": "Evidence covers modeled cost.",
        }
    )
    parsed = _parse_provider_decision(
        {"model": "qwen3:14b", "done": True, "message": {"content": content}},
        expected_model="qwen3:14b",
    )
    assert parsed.action == "approve"
    assert parsed.risk_multiplier == 0.4

    invalid_contents = (
        content.replace('"action": "approve"', '"action": "APPROVE"'),
        content.replace('"summary":', '"extra": 1, "summary":'),
        content.replace(
            '"action": "approve"',
            '"action": "veto", "action": "approve"',
        ),
        content.replace('"risk_multiplier": 0.4', '"risk_multiplier": 1.1'),
        content.replace('"edge_after_costs"', '"model_uncertainty"'),
    )
    for invalid in invalid_contents:
        with pytest.raises(ValueError):
            _parse_provider_decision(
                {
                    "model": "qwen3:14b",
                    "done": True,
                    "message": {"content": invalid},
                },
                expected_model="qwen3:14b",
            )


def test_approval_requires_bound_after_cost_model_evidence() -> None:
    unsupported = build_live_ai_entry_case(
        symbol="BTCUSDC",
        market_type="futures",
        interval="15m",
        observed_at_ms=1_000,
        proposed_side="LONG",
        ml_confidence=0.72,
        maximum_risk_multiplier=0.4,
        model_digest=_DIGEST,
        terminal_model_fingerprint=_FINGERPRINT,
        evidence={"cost_model": {}},
    )

    with pytest.raises(ValueError, match="bound model evidence"):
        _approval().validated_for(unsupported)


def test_model_validation_summary_is_compact_and_unit_explicit() -> None:
    artifact = _validated_model_artifact()

    evidence = live_ai_assist_module._model_validation_evidence(artifact)

    assert evidence["terminal_holdout"]["mean_after_cost_sample_return_bps"] == 2.0
    assert evidence["execution_validation"]["microstructure_seconds"] == 1_728_000
    assert len(json.dumps(evidence, separators=(",", ":"))) < 2_500


def test_ineligible_model_evidence_never_consumes_provider_tokens(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []

    def provider(case):
        provider_calls.append(case.case_id)
        return _approval()

    reviewer = AsyncLiveAIEntryReviewer(
        provider,
        audit_path=tmp_path / "ai-entry.jsonl",
    )

    def base_decision(*_args):
        return Decision(
            side="LONG",
            confidence=0.72,
            mark_price=100.0,
            observed_at_ms=1_000,
        )

    assisted = AIAssistedDecisionFunction(
        base_decision,
        reviewer,
        model_digest=_DIGEST,
        terminal_model_fingerprint=_FINGERPRINT,
    )

    decision = assisted(
        None,
        RuntimeConfig(symbol="BTCUSDC", market_type="futures", interval="15m"),
        _approval_strategy(),
        None,
    )

    assert decision.ai_assist_status == "shadow_failure"
    assert decision.ai_assist_action == "veto"
    assert "deterministic model evidence" in decision.ai_assist_reason
    assert provider_calls == []
    assert assisted.close(1.0)


def test_coordinator_can_suspend_entry_review_without_affecting_ml_side(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    reviewer = AsyncLiveAIEntryReviewer(
        lambda case: provider_calls.append(case.case_id) or _approval(),
        audit_path=tmp_path / "ai-entry.jsonl",
    )

    def base_decision(*_args):
        return Decision(
            side="SHORT",
            confidence=0.20,
            mark_price=100.0,
            observed_at_ms=1_000,
        )

    base_decision._model_artifact = _validated_model_artifact()
    assisted = AIAssistedDecisionFunction(
        base_decision,
        reviewer,
        model_digest=_DIGEST,
        terminal_model_fingerprint=_FINGERPRINT,
    )
    assisted.set_entry_review_required(False)

    decision = assisted(
        None,
        RuntimeConfig(symbol="BTCUSDC", market_type="futures", interval="15m"),
        _approval_strategy(),
        None,
    )

    assert decision.side == "SHORT"
    assert decision.ai_assist_status == "shadow_idle"
    assert decision.ai_assist_entry_ready is False
    assert provider_calls == []
    assert assisted.close(1.0)


def test_ollama_provider_binds_response_to_digest_gpu_and_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = json.dumps(
        {
            "action": "approve",
            "risk_multiplier": 0.4,
            "confidence": 0.8,
            "reason_codes": ["edge_after_costs"],
            "summary": "Evidence covers modeled cost.",
        }
    )
    response_payload = {
        "model": "qwen3:14b",
        "done": True,
        "message": {"content": content},
        "prompt_eval_count": 321,
        "eval_count": 47,
    }
    requests: list[dict[str, object]] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit: int) -> bytes:
            return json.dumps(response_payload).encode("utf-8")

    def fake_urlopen(request, *, timeout):
        assert timeout == 10.0
        requests.append(json.loads(request.data.decode("utf-8")))
        return _Response()

    monkeypatch.setattr(live_ai_assist_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        live_ai_assist_module,
        "inspect_ollama_model_residency",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="gpu_resident",
            digest=_DIGEST,
        ),
    )
    provider = OllamaLiveAIEntryProvider(
        model="qwen3:14b",
        expected_model_digest=_DIGEST,
        timeout_seconds=10.0,
    )
    decision = provider(_case())
    assert decision.observed_model_digest == _DIGEST
    assert decision.model_residency_status == "gpu_resident"
    assert decision.prompt_tokens == 321
    assert decision.output_tokens == 47
    assert requests[0]["think"] is False
    assert requests[0]["options"]["num_predict"] == 180

    with pytest.raises(ValueError, match="case differs"):
        provider(_case(observed_at_ms=2_000, model_digest="c" * 64))

    monkeypatch.setattr(
        live_ai_assist_module,
        "inspect_ollama_model_residency",
        lambda *_args, **_kwargs: SimpleNamespace(status="unloaded", digest=None),
    )
    with pytest.raises(ValueError, match="approved GPU-resident model"):
        provider(_case(observed_at_ms=3_000))


def test_shadow_reviewer_defers_only_entry_then_preserves_ml_side_and_size(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []

    def provider(case):
        provider_calls.append(case.case_id)
        return _approval()

    reviewer = AsyncLiveAIEntryReviewer(
        provider,
        audit_path=tmp_path / "ai-entry.jsonl",
    )

    def base_decision(*_args):
        return Decision(
            side="LONG",
            confidence=0.72,
            mark_price=100.0,
            size_multiplier=0.4,
            observed_at_ms=1_000,
            ai_evidence={"after_cost_margin_bps": 3.2},
        )

    base_decision._model_artifact = _validated_model_artifact()

    assisted = AIAssistedDecisionFunction(
        base_decision,
        reviewer,
        model_digest=_DIGEST,
        terminal_model_fingerprint=_FINGERPRINT,
    )
    runtime = RuntimeConfig(
        symbol="BTCUSDC",
        market_type="futures",
        interval="15m",
    )
    strategy = _approval_strategy()
    pending = assisted(None, runtime, strategy, None)
    assert pending.side == "LONG"
    assert pending.size_multiplier == 0.4
    assert pending.ai_assist_status == "shadow_pending"
    assert pending.ai_assist_entry_ready is False

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        completed = assisted(None, runtime, strategy, None)
        if completed.ai_assist_status == "shadow_approve":
            break
        time.sleep(0.005)
    else:
        pytest.fail("AI shadow approval did not complete")
    assert completed.side == "LONG"
    assert completed.size_multiplier == 0.4
    assert completed.ai_assist_action == "approve"
    assert completed.ai_assist_entry_ready is True
    assert provider_calls == [completed.ai_assist_case_id]
    assert assisted.close(1.0)

    records = [
        json.loads(line)
        for line in (tmp_path / "ai-entry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["mode"] == "shadow_only"
    assert records[0]["trading_authority"] is False
    assert records[0]["case"]["case_id"] == completed.ai_assist_case_id


def test_provider_failure_is_recorded_without_execution_authority(tmp_path: Path) -> None:
    def failed_provider(_case):
        raise TimeoutError("provider deadline")

    path = tmp_path / "ai-entry.jsonl"
    reviewer = AsyncLiveAIEntryReviewer(failed_provider, audit_path=path)
    case = _case()
    assert reviewer.review(case).status == "shadow_pending"
    failed = _wait_for_review(reviewer, case, "shadow_failure")
    assert failed.decision is not None
    assert failed.decision.action == "veto"
    assert failed.decision.valid is False
    assert reviewer.close(1.0)

    # A clean restart validates the full existing hash chain before accepting work.
    restarted = AsyncLiveAIEntryReviewer(_approval, audit_path=path)
    assert restarted.close(1.0)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decision"]["summary"] = "tampered"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted"):
        AsyncLiveAIEntryReviewer(_approval, audit_path=path)


def test_completed_review_cannot_be_replayed_after_freshness_window(
    tmp_path: Path,
) -> None:
    reviewer = AsyncLiveAIEntryReviewer(
        lambda _case: _approval(),
        audit_path=tmp_path / "ai-entry.jsonl",
        clock=lambda: 2.0,
    )

    def base_decision(*_args):
        return Decision(
            side="LONG",
            confidence=0.72,
            mark_price=100.0,
            size_multiplier=0.4,
            observed_at_ms=1_000,
        )

    base_decision._model_artifact = _validated_model_artifact()

    assisted = AIAssistedDecisionFunction(
        base_decision,
        reviewer,
        model_digest=_DIGEST,
        terminal_model_fingerprint=_FINGERPRINT,
        clock=lambda: 400.0,
    )
    runtime = RuntimeConfig(symbol="BTCUSDC", market_type="futures", interval="15m")
    strategy = _approval_strategy()
    assert assisted(None, runtime, strategy, None).ai_assist_entry_ready is False
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        stale = assisted(None, runtime, strategy, None)
        if stale.ai_assist_status != "shadow_pending":
            break
        time.sleep(0.005)
    else:
        pytest.fail("AI shadow review did not complete")

    assert stale.side == "LONG"
    assert stale.ai_assist_status == "shadow_failure"
    assert stale.ai_assist_entry_ready is False
    assert "stale or future-dated" in stale.ai_assist_reason
    assert assisted.close(1.0)


def test_shadow_contract_failure_blocks_only_entry_not_the_ml_decision(tmp_path: Path) -> None:
    reviewer = AsyncLiveAIEntryReviewer(
        lambda _case: _approval(),
        audit_path=tmp_path / "ai-entry.jsonl",
    )

    def malformed_decision(*_args):
        return Decision(
            side="SHORT",
            confidence=0.2,
            mark_price=100.0,
            size_multiplier=0.3,
            observed_at_ms=0,
        )

    assisted = AIAssistedDecisionFunction(
        malformed_decision,
        reviewer,
        model_digest=_DIGEST,
        terminal_model_fingerprint=_FINGERPRINT,
    )
    decision = assisted(
        None,
        RuntimeConfig(symbol="BTCUSDC", market_type="futures"),
        StrategyConfig(),
        None,
    )
    assert decision.side == "SHORT"
    assert decision.size_multiplier == 0.3
    assert decision.ai_assist_status == "shadow_failure"
    assert decision.ai_assist_entry_ready is False
    assert reviewer.close(1.0)


def test_ai_case_identity_survives_position_and_trade_ledgers(tmp_path: Path) -> None:
    decision = Decision(
        side="LONG",
        confidence=0.7,
        mark_price=100.0,
        observed_at_ms=1_000,
        ai_assist_mode="shadow_only",
        ai_assist_status="shadow_approve",
        ai_assist_case_id="d" * 64,
    )
    position = _open_position_from_decision(
        decision,
        RuntimeConfig(symbol="BTCUSDC", market_type="futures"),
        StrategyConfig(),
        get_objective("conservative"),
        AutonomousConfig(positions_root=tmp_path),
        clock=lambda: 2.0,
    )
    trade = _close_to_trade(position, 101.0, "test", clock=lambda: 3.0)
    assert position.ai_review_case_id == decision.ai_assist_case_id
    assert trade.ai_review_case_id == decision.ai_assist_case_id
    assert trade.ai_review_status == "shadow_approve"


def test_pending_ai_review_cannot_cross_the_entry_boundary(tmp_path: Path) -> None:
    cfg = AutonomousConfig(
        positions_root=tmp_path,
        starting_reference_cash=1_000.0,
    )
    decision = Decision(
        side="LONG",
        confidence=0.9,
        mark_price=100.0,
        ai_assist_mode="shadow_only",
        ai_assist_status="shadow_pending",
        ai_assist_case_id="d" * 64,
        ai_assist_reason="review pending",
        ai_assist_entry_ready=False,
    )
    runtime = RuntimeConfig(symbol="BTCUSDC", market_type="futures")
    strategy = StrategyConfig()
    objective = get_objective("conservative")

    gate = _entry_gate(
        PositionsStore(tmp_path),
        decision,
        strategy,
        cfg,
        objective,
        now_ms_value=2_000,
    )
    assert gate.allowed is False
    assert gate.reason == "review pending"
    with pytest.raises(ValueError, match="completed AI pre-entry review"):
        _open_position_from_decision(
            decision,
            runtime,
            strategy,
            objective,
            cfg,
        )


def test_review_submission_and_shutdown_are_bounded(tmp_path: Path) -> None:
    release = threading.Event()
    entered = threading.Event()

    def blocking_provider(_case):
        entered.set()
        release.wait(2.0)
        return _approval()

    reviewer = AsyncLiveAIEntryReviewer(
        blocking_provider,
        audit_path=tmp_path / "ai-entry.jsonl",
    )
    started = time.perf_counter()
    assert reviewer.review(_case()).status == "shadow_pending"
    assert time.perf_counter() - started < 0.2
    assert entered.wait(1.0)
    try:
        assert reviewer.close(0.001) is False
    finally:
        release.set()
        assert reviewer.close(1.0)


def test_provider_cannot_exceed_ml_risk_bound(tmp_path: Path) -> None:
    reviewer = AsyncLiveAIEntryReviewer(
        lambda _case: LiveAIEntryDecision(
            action="approve",
            risk_multiplier=0.41,
            confidence=0.8,
            reason_codes=("edge_after_costs",),
            summary="Attempts to exceed the ML risk cap.",
            valid=True,
            response_sha256="c" * 64,
            observed_model_digest=_DIGEST,
            model_residency_status="gpu_resident",
            prompt_tokens=100,
            output_tokens=20,
        ),
        audit_path=tmp_path / "ai-entry.jsonl",
    )
    case = _case()
    assert reviewer.review(case).status == "shadow_pending"
    failed = _wait_for_review(reviewer, case, "shadow_failure")
    assert failed.decision is not None
    assert "risk bound" in failed.decision.failure_reason
    assert reviewer.close(1.0)


def test_queue_saturation_is_explicit_and_never_overwrites_pending_case(
    tmp_path: Path,
) -> None:
    release = threading.Event()
    entered = threading.Event()
    provider_calls: list[str] = []

    def blocking_provider(case):
        provider_calls.append(case.case_id)
        if len(provider_calls) == 1:
            entered.set()
            release.wait(2.0)
        return _approval()

    reviewer = AsyncLiveAIEntryReviewer(
        blocking_provider,
        audit_path=tmp_path / "ai-entry.jsonl",
    )
    first = _case(observed_at_ms=1_000)
    second = _case(observed_at_ms=2_000)
    third = _case(observed_at_ms=3_000)
    assert reviewer.review(first).status == "shadow_pending"
    assert entered.wait(1.0)
    assert reviewer.review(second).status == "shadow_pending"
    rejected = reviewer.review(third)
    assert rejected.status == "shadow_failure"
    assert rejected.decision is not None
    assert "queue full" in rejected.decision.failure_reason
    release.set()
    _wait_for_review(reviewer, second, "shadow_approve")
    assert reviewer.close(1.0)
    assert provider_calls == [first.case_id, second.case_id]
