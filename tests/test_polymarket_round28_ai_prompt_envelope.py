from __future__ import annotations

import hashlib

import pytest

import simple_ai_trading.polymarket_round28_ai_prompt_envelope as envelope
from simple_ai_trading.polymarket_round28_ai_cases import round28_ai_case_prompt
from simple_ai_trading.polymarket_round28_ai_contract import (
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_host import (
    round28_ai_candidate_from_contract,
)
from simple_ai_trading.polymarket_round28_ai_inference import (
    run_round28_ai_inference,
)
from test_polymarket_round28_ai_inference import (
    ROOT,
    _host_report,
    _raw,
    _residency,
)
from tools.run_polymarket_round28_ai_inference_enveloped import (
    _parser as _inference_parser,
)
from tools.run_polymarket_round28_ai_sealed_cases_enveloped import (
    _parser as _sealed_parser,
)


def _synthetic_inference(contract, candidate, host_report):
    panel = envelope.build_round28_ai_prompt_envelope_panel()
    times = iter((0, 100_000_000, 200_000_000, 300_000_000))
    residency_calls = 0

    def post_json(_url, payload, _timeout):
        if payload.get("keep_alive") == 0:
            return {}
        if str(payload.get("prompt", "")).startswith("Runtime conformance"):
            return _raw(
                candidate,
                "reject",
                ["bbo_source_stale_or_gapped"],
            )
        return _raw(candidate, "reject", ["volatility_or_jump_risk"])

    def residency(*_args, **_kwargs):
        nonlocal residency_calls
        residency_calls += 1
        return _residency(candidate, loaded=residency_calls == 1)

    return run_round28_ai_inference(
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


def test_prompt_envelope_uses_exact_width_without_market_or_target_data() -> None:
    panel = envelope.build_round28_ai_prompt_envelope_panel()
    case = panel.cases[0]
    prompt = round28_ai_case_prompt(case)

    assert len(case.causal_features) == 278
    assert len(prompt.encode("ascii")) > 15_000
    assert case.token_id == "synthetic-envelope-no-market-data"
    assert panel.identity_payload()["target_accessed"] is False
    assert panel.identity_payload()["outcome_accessed"] is False


def test_prompt_envelope_binds_actual_inference_and_rejects_nested_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_round28_ai_contract(ROOT)
    candidate = round28_ai_candidate_from_contract(
        contract,
        model_id="Qwen/Qwen3.5-9B",
    )
    host_report = _host_report(contract, candidate)
    inference = _synthetic_inference(contract, candidate, host_report)
    monkeypatch.setattr(
        envelope,
        "run_round28_ai_inference",
        lambda **_kwargs: inference,
    )

    report = envelope.evaluate_round28_ai_prompt_envelope(
        contract=contract,
        host_qualification_report=host_report,
    )

    assert report["passed"] is True
    assert set(report["checks"].values()) == {True}
    assert report["inference_report"][
        "candidate_eligible_for_matched_evaluation"
    ] is False
    assert report["edge_claim"] is False
    assert report["profitability_claim"] is False
    assert report["orders_submitted"] is False

    tampered = dict(report)
    nested = dict(tampered["inference_report"])
    responses = [dict(item) for item in nested["responses"]]
    responses[0]["wall_latency_ms"] = 4_999
    response_body = dict(responses[0])
    response_body.pop("response_sha256")
    responses[0]["response_sha256"] = envelope._canonical_sha256(response_body)
    nested["responses"] = responses
    nested_body = dict(nested)
    nested_body.pop("report_sha256")
    nested["report_sha256"] = envelope._canonical_sha256(nested_body)
    tampered["inference_report"] = nested
    report_body = dict(tampered)
    report_body.pop("report_sha256")
    tampered["report_sha256"] = hashlib.sha256(
        envelope._canonical_json(report_body).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match="differs"):
        envelope.validate_round28_ai_prompt_envelope_report(
            tampered,
            contract=contract,
            host_qualification_report=host_report,
        )


def test_enveloped_operator_surfaces_require_prompt_receipts() -> None:
    inference_destinations = {
        action.dest
        for action in _inference_parser()._actions
        if action.dest != "help"
    }
    sealed_destinations = {
        action.dest for action in _sealed_parser()._actions if action.dest != "help"
    }

    assert "prompt_envelope_report" in inference_destinations
    assert "nominated_prompt_envelope_report" in sealed_destinations
    assert all(
        fragment not in destination
        for destination in sealed_destinations
        for fragment in ("target", "outcome", "resolution")
    )
