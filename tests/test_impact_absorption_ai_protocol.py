from __future__ import annotations

from dataclasses import replace
import json

import pytest

from simple_ai_trading.impact_absorption_ai_protocol import (
    ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION,
    ROUND74_AI_REVIEW_MAXIMUM_VALIDITY_NS,
    Round74AIModelManifest,
    Round74AIReviewDecision,
    Round74AIReviewRequest,
    apply_round74_ai_risk_modifier,
    build_round74_ai_review_prompt,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
)


WALL_NS = 1_800_000_000_000_000_000


def _manifest() -> Round74AIModelManifest:
    return Round74AIModelManifest(
        model_id="example/finance-8b",
        model_revision="a" * 40,
        model_artifact_sha256="b" * 64,
        model_artifact_kind="ollama_manifest",
        parameter_count=8_000_000_000,
        quantization="int4",
        runtime_backend="windows-ml",
        runtime_version="1.0.0",
        license_id="Apache-2.0",
        model_card_url="https://example.invalid/finance-8b",
        minimum_vram_bytes=8 * 1024**3,
        finance_specialized=True,
    )


def _request() -> Round74AIReviewRequest:
    count = len(ROUND74_EVENT_FEATURE_NAMES)
    last = [0.0] * count
    last[5] = 1.0
    last[10] = 1.5
    return Round74AIReviewRequest(
        pretest_policy_sha256="1" * 64,
        sample_sha256="2" * 64,
        deterministic_risk_state_sha256="3" * 64,
        asset_slot=0,
        side="long",
        horizon_seconds=30,
        requested_wall_ns=WALL_NS,
        expires_wall_ns=WALL_NS + 10_000_000_000,
        proposed_risk_size_bps=2_500,
        feature_last=tuple(last),
        feature_mean=tuple(0.1 for _ in range(count)),
        feature_standard_deviation=tuple(0.2 for _ in range(count)),
        payoff_quantiles_bps=(-5.0, -1.0, 2.0, 4.0, 7.0),
        maximum_adverse_excursion_quantiles_bps=(
            1.0,
            2.0,
            3.0,
            5.0,
            8.0,
        ),
        positive_payoff_probability=0.61,
        adverse_selection_probability=0.27,
        regime_unpredictability_probability=0.18,
    )


def _decision(
    verdict: str,
    multiplier: int,
    reasons: tuple[str, ...],
) -> Round74AIReviewDecision:
    return Round74AIReviewDecision(
        verdict=verdict,
        size_multiplier_bps=multiplier,
        confidence_bps=7_500,
        reason_codes=reasons,
    )


def test_ai_manifest_is_multibillion_local_and_hash_bound() -> None:
    manifest = _manifest()
    payload = manifest.as_dict()
    restored = Round74AIModelManifest.from_dict(payload)

    assert restored == manifest
    assert restored.manifest_sha256 == payload["manifest_sha256"]
    assert payload["remote_inference_permitted"] is False
    assert payload["execution_authority"] is False
    assert payload["model_size_implies_edge"] is False

    tampered = dict(payload)
    tampered["parameter_count"] = 1_000_000_000
    with pytest.raises(ValueError, match="manifest digest differs"):
        Round74AIModelManifest.from_dict(tampered)


def test_ai_manifest_rejects_unpinned_or_cpu_only_models() -> None:
    with pytest.raises(ValueError, match="manifest differs"):
        replace(_manifest(), model_revision="main").validate()
    with pytest.raises(ValueError, match="manifest differs"):
        replace(_manifest(), runtime_backend="cpu").validate()
    with pytest.raises(ValueError, match="manifest differs"):
        replace(_manifest(), parameter_count=1_999_999_999).validate()
    with pytest.raises(ValueError, match="manifest differs"):
        replace(_manifest(), model_card_url="http://example.invalid").validate()


def test_ai_prompt_is_causal_anonymized_and_schema_constrained() -> None:
    request = _request()
    restored = Round74AIReviewRequest.from_dict(request.as_dict())
    system, user = build_round74_ai_review_prompt(request)
    payload = json.loads(user)

    assert restored == request
    assert len(request.request_sha256) == 64
    assert payload["asset"] == "asset_0"
    assert payload["horizon_seconds"] == 30
    assert "asset_identity_0" in payload["standardized_feature_summary"]
    assert "symbol_is_btcusdt" not in user
    assert "BTCUSDT" not in user
    assert str(WALL_NS) not in user
    assert request.pretest_policy_sha256 not in user
    assert request.sample_sha256 not in user
    assert "future" not in user
    assert "Never infer an identity or date" in system
    assert "increase size" in system
    assert "propose an order" in system


@pytest.mark.parametrize(
    "changed",
    [
        {"horizon_seconds": 5},
        {
            "expires_wall_ns": (
                WALL_NS + ROUND74_AI_REVIEW_MAXIMUM_VALIDITY_NS + 1
            )
        },
        {"proposed_risk_size_bps": 10_001},
        {"positive_payoff_probability": float("nan")},
        {"payoff_quantiles_bps": (-5.0, 0.0, -1.0, 4.0, 7.0)},
        {
            "maximum_adverse_excursion_quantiles_bps": (
                -1.0,
                2.0,
                3.0,
                5.0,
                8.0,
            )
        },
    ],
)
def test_ai_request_rejects_unsafe_or_malformed_context(
    changed: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="Round 74 AI"):
        replace(_request(), **changed).validate()


@pytest.mark.parametrize(
    ("verdict", "multiplier", "reasons"),
    [
        ("allow_unchanged", 10_000, ("none",)),
        ("reduce", 5_000, ("forecast_uncertainty",)),
        ("veto", 0, ("adverse_selection",)),
        ("abstain", 0, ("model_inconsistency",)),
    ],
)
def test_ai_generated_decision_accepts_only_veto_only_schema(
    verdict: str,
    multiplier: int,
    reasons: tuple[str, ...],
) -> None:
    raw = json.dumps(
        {
            "schema_version": ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION,
            "verdict": verdict,
            "size_multiplier_bps": multiplier,
            "confidence_bps": 7_500,
            "reason_codes": list(reasons),
        }
    )
    decision = Round74AIReviewDecision.from_generated_text(raw)

    assert decision.verdict == verdict
    assert decision.size_multiplier_bps == multiplier
    assert len(decision.decision_sha256) == 64
    assert decision.as_dict()["may_select_side"] is False
    assert decision.as_dict()["may_set_leverage"] is False
    assert decision.as_dict()["may_submit_or_cancel_orders"] is False


@pytest.mark.parametrize(
    "raw",
    [
        (
            '{"schema_version":"round-074-ai-review-decision-v1",'
            '"schema_version":"round-074-ai-review-decision-v1",'
            '"verdict":"veto","size_multiplier_bps":0,'
            '"confidence_bps":7000,"reason_codes":["stale_state"]}'
        ),
        (
            '{"schema_version":"round-074-ai-review-decision-v1",'
            '"verdict":"allow_unchanged","size_multiplier_bps":9999,'
            '"confidence_bps":7000,"reason_codes":["none"]}'
        ),
        (
            '{"schema_version":"round-074-ai-review-decision-v1",'
            '"verdict":"reduce","size_multiplier_bps":5000,'
            '"confidence_bps":7000,'
            '"reason_codes":["spread_wide","adverse_selection"]}'
        ),
        (
            '{"schema_version":"round-074-ai-review-decision-v1",'
            '"verdict":"veto","size_multiplier_bps":0,'
            '"confidence_bps":7000,"reason_codes":["stale_state"],'
            '"side":"short"}'
        ),
        (
            "```json\n"
            '{"schema_version":"round-074-ai-review-decision-v1",'
            '"verdict":"veto","size_multiplier_bps":0,'
            '"confidence_bps":7000,"reason_codes":["stale_state"]}'
            "\n```"
        ),
    ],
)
def test_ai_generated_decision_fails_closed_on_output_drift(
    raw: str,
) -> None:
    with pytest.raises((ValueError, json.JSONDecodeError)):
        Round74AIReviewDecision.from_generated_text(raw)


def test_ai_modifier_can_only_preserve_reduce_or_veto() -> None:
    request = _request()
    now = request.requested_wall_ns + 1

    assert apply_round74_ai_risk_modifier(
        request,
        _decision("allow_unchanged", 10_000, ("none",)),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=now,
    ) == 2_500
    assert apply_round74_ai_risk_modifier(
        request,
        _decision("reduce", 5_000, ("forecast_uncertainty",)),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=now,
    ) == 1_250
    assert apply_round74_ai_risk_modifier(
        request,
        _decision("veto", 0, ("adverse_selection",)),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=now,
    ) == 0
    assert apply_round74_ai_risk_modifier(
        request,
        None,
        deterministic_risk_gate_passed=True,
        observed_wall_ns=now,
    ) == 0
    assert apply_round74_ai_risk_modifier(
        request,
        _decision("allow_unchanged", 10_000, ("none",)),
        deterministic_risk_gate_passed=False,
        observed_wall_ns=now,
    ) == 0
    assert apply_round74_ai_risk_modifier(
        request,
        _decision("allow_unchanged", 10_000, ("none",)),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=request.expires_wall_ns + 1,
    ) == 0
