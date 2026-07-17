from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from simple_ai_trading.autonomous import (
    AutonomousConfig,
    Decision,
    _close_to_trade,
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
from simple_ai_trading.types import RuntimeConfig, StrategyConfig
from simple_ai_trading.objective import get_objective


_DIGEST = "a" * 64
_FINGERPRINT = "b" * 64


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
        evidence={"signal": {"after_cost_margin_bps": 3.2}},
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


def test_shadow_reviewer_never_changes_ml_side_or_size(tmp_path: Path) -> None:
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
    strategy = StrategyConfig()
    pending = assisted(None, runtime, strategy, None)
    assert pending.side == "LONG"
    assert pending.size_multiplier == 0.4
    assert pending.ai_assist_status == "shadow_pending"

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


def test_shadow_contract_failure_does_not_change_or_stop_ml_decision(tmp_path: Path) -> None:
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
    assert reviewer.close(1.0)


def test_ai_case_identity_survives_position_and_trade_ledgers(tmp_path: Path) -> None:
    decision = Decision(
        side="LONG",
        confidence=0.7,
        mark_price=100.0,
        observed_at_ms=1_000,
        ai_assist_mode="shadow_only",
        ai_assist_status="shadow_pending",
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
    assert trade.ai_review_status == "shadow_pending"


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
