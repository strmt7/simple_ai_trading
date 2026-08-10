from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Mapping

import pytest

from simple_ai_trading.ai_runtime import OllamaResidencyReport
from simple_ai_trading.polymarket_round25_ai import Round25AIAdvisory
from simple_ai_trading.polymarket_round25_ai_supervisor import (
    POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE,
    POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256,
    POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_CONTRACT_SHA256,
    Round25AISupervisorAdvisory,
    Round25AISupervisorConfig,
    Round25AISupervisorPacket,
    Round25AISupervisorResult,
    Round25AISupervisorTelemetry,
    Round25AISupervisorWorker,
    combine_round25_ai_risk,
    preflight_round25_ai_supervisor,
    review_round25_ai_supervisor_packet,
)


REPOSITORY = Path(__file__).resolve().parents[1]


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


def _packet() -> Round25AISupervisorPacket:
    return Round25AISupervisorPacket(
        condition_id="0x" + "a" * 64,
        window_start_ms=1_000_000,
        observed_at_ms=1_060_000,
        expires_at_ms=1_150_000,
        feature_source_chain_sha256="1" * 64,
        clob_relative_spread_median_60s=0.01,
        clob_relative_spread_p95_60s=0.02,
        clob_top_executable_notional_p10_usd_60s=2_000.0,
        clob_book_receipt_age_p95_ms_60s=80.0,
        reference_receipt_age_p95_ms_60s=40.0,
        realized_volatility_60s=0.003,
        realized_volatility_300s=0.006,
        absolute_log_return_60s=0.002,
        absolute_log_return_300s=0.004,
        absolute_order_flow_imbalance_mean_60s=0.15,
        market_probability_range_60s=0.04,
        round_trip_cost_bps_p95_60s=18.0,
        portfolio_risk_utilization=0.1,
        current_condition_exposure_fraction=0.0,
        deterministic_gate_sha256="2" * 64,
    )


def _config() -> Round25AISupervisorConfig:
    return Round25AISupervisorConfig()


def _show() -> dict[str, object]:
    return {
        "details": {
            "format": "gguf",
            "parameter_size": POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.parameter_size,
            "quantization_level": POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.quantization,
        },
        "model_info": {"general.parameter_count": 7_620_000_000},
        "capabilities": ["completion"],
    }


def _get_json(url: str, _timeout: float) -> object:
    candidate = POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE
    if url.endswith("/api/version"):
        return {"version": "0.32.5"}
    if url.endswith("/api/tags"):
        return {
            "models": [
                {
                    "name": candidate.model,
                    "model": candidate.model,
                    "digest": candidate.digest,
                    "details": {
                        "format": "gguf",
                        "parameter_size": candidate.parameter_size,
                        "quantization_level": candidate.quantization,
                    },
                }
            ]
        }
    raise AssertionError(f"unexpected GET: {url}")


def _provider_response(action: str = "normal") -> dict[str, object]:
    return {
        "model": POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.model,
        "response": json.dumps({"regime_action": action}, separators=(",", ":")),
        "done": True,
        "done_reason": "stop",
        "total_duration": 200_000_000,
        "load_duration": 1_000_000,
        "prompt_eval_count": 500,
        "prompt_eval_duration": 80_000_000,
        "eval_count": 8,
        "eval_duration": 100_000_000,
    }


def _post_json(action: str = "normal"):
    def post_json(url: str, payload: Mapping[str, object], _timeout: float) -> object:
        if url.endswith("/api/show"):
            assert payload == {
                "model": POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.model,
                "verbose": False,
            }
            return _show()
        if url.endswith("/api/generate"):
            assert payload["model"] == POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.model
            return _provider_response(action)
        raise AssertionError(f"unexpected POST: {url}")

    return post_json


def _residency(*, ratio: float = 1.0) -> OllamaResidencyReport:
    candidate = POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE
    size = 6_178_671_164
    return OllamaResidencyReport(
        requested_model=candidate.model,
        status="gpu_resident" if ratio > 0.0 else "cpu_only",
        loaded_model=candidate.model,
        digest=candidate.digest,
        size_bytes=size,
        size_vram_bytes=int(size * ratio),
        vram_to_model_ratio=int(size * ratio) / size,
    ).validated()


def _residency_inspector(*, ratio: float = 1.0):
    def inspect(
        base_url: str,
        model: str,
        _timeout: float,
        *,
        expected_digest: str,
    ) -> OllamaResidencyReport:
        assert base_url == "http://127.0.0.1:11434"
        assert model == POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.model
        assert expected_digest == POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.digest
        return _residency(ratio=ratio)

    return inspect


def _clock(values: tuple[int, ...]):
    selected = iter(values)
    return lambda: next(selected)


def _valid_result(packet: Round25AISupervisorPacket) -> Round25AISupervisorResult:
    advisory = Round25AISupervisorAdvisory(
        candidate_id=POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.candidate_id,
        model=POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.model,
        model_digest=POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.digest,
        packet_sha256=packet.packet_sha256,
        generated_at_ms=1_060_200,
        expires_at_ms=packet.expires_at_ms,
        regime_action="cautious_50",
        maximum_size_multiplier=0.5,
        cooldown_ms=0,
        reason_codes=("regime_size_reduction",),
        valid_model_response=True,
        failure_code=None,
    )
    telemetry = Round25AISupervisorTelemetry(
        packet_sha256=packet.packet_sha256,
        ollama_version="0.32.5",
        show_metadata_sha256="3" * 64,
        prompt_sha256="4" * 64,
        response_sha256="5" * 64,
        measured_latency_seconds=0.2,
        provider_total_duration_ns=200_000_000,
        provider_load_duration_ns=1_000_000,
        provider_prompt_eval_count=500,
        provider_prompt_eval_duration_ns=80_000_000,
        provider_eval_count=8,
        provider_eval_duration_ns=100_000_000,
        residency=_residency(),
    )
    return Round25AISupervisorResult(
        packet_sha256=packet.packet_sha256,
        advisory=advisory,
        telemetry=telemetry,
    )


def test_round25_ai_supervisor_contracts_are_self_hashed_and_claim_free() -> None:
    contract_path = REPOSITORY / "docs/model-research/polymarket/round-025-fin-r1-regime-supervisor-contract-v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")
    assert claimed == _canonical_sha256(contract)
    assert claimed == POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256
    assert contract["truth_state"]["profitability_verified"] is False
    assert contract["authority"]["can_submit_orders"] is False

    uplift_path = REPOSITORY / "docs/model-research/polymarket/round-025-fin-r1-regime-supervisor-uplift-contract-v1.json"
    uplift = json.loads(uplift_path.read_text(encoding="utf-8"))
    uplift_claimed = uplift.pop("contract_sha256")
    assert uplift_claimed == _canonical_sha256(uplift)
    assert uplift_claimed == POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_CONTRACT_SHA256
    assert uplift["claims"]["ai_uplift_verified"] is False
    assert uplift["promotion_gate"]["grants_paper_or_live_authority"] is False

    scenario_path = REPOSITORY / "docs/model-research/polymarket/round-025-fin-r1-regime-supervisor-scenario-contract-v1.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario_claimed = scenario.pop("contract_sha256")
    assert scenario_claimed == _canonical_sha256(scenario)
    assert scenario_claimed == "9828aaf05deafe776c09a26c4e4cc0578762b9c53efa6c75ef83d4c3dc14dac4"
    assert scenario["response_use"]["same_candidate_prompt_tuning_after_failure_allowed"] is False
    assert scenario["claims"]["profitability_verified"] is False

    correction_path = REPOSITORY / "docs/model-research/polymarket/round-025-fin-r1-regime-supervisor-scenario-contract-v2.json"
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    correction_claimed = correction.pop("contract_sha256")
    assert correction_claimed == _canonical_sha256(correction)
    assert correction_claimed == "749f3a8b1e02523b6a11113db1498971d0e89bab034fefb57ec94cc9b98696f5"
    assert correction["correction"]["prompt_changed"] is False
    assert correction["first_attempt_boundary"]["provider_inferences_completed"] == 0

    failure_path = REPOSITORY / "docs/model-research/polymarket/round-025-fin-r1-regime-supervisor-infrastructure-failure-v1-2026-08-10.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    failure_claimed = failure.pop("artifact_sha256")
    assert failure_claimed == _canonical_sha256(failure)
    assert failure_claimed == "034c69c5cbfdd5372ffa373a3568910ca18620a11af29a510fa6547db0651132"
    assert failure["claims"]["model_response_observed"] is False

    probe_path = REPOSITORY / "docs/model-research/polymarket/round-025-fin-r1-regime-supervisor-host-probe-v2-2026-08-10.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe_claimed = probe.pop("evidence_sha256")
    assert probe_claimed == _canonical_sha256(probe)
    assert probe_claimed == "9ba198fbd90f469bd5c0ae7ebe7dd13059e9ce4a51400e2794bce68b94857f19"
    assert probe["status"] == "target_free_supervisor_mechanics_failed"
    assert {row["action"] for row in probe["scenario_results"]} == {"normal"}
    assert probe["checks"]["combined_crisis_at_least_defensive"] is False
    assert probe["claims"]["ai_uplift_verified"] is False

    rejection_path = REPOSITORY / "docs/model-research/polymarket/round-025-fin-r1-regime-supervisor-rejection-v1-2026-08-10.json"
    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    rejection_claimed = rejection.pop("artifact_sha256")
    assert rejection_claimed == _canonical_sha256(rejection)
    assert rejection_claimed == "964d48739608bb6bcc09ec81db54d0c50ae108b61b13dd992fcf64cc4354b87f"
    assert rejection["decision"]["candidate_promoted"] is False
    assert rejection["decision"]["same_candidate_prompt_tuning_allowed"] is False


def test_round25_ai_supervisor_packet_is_target_free_and_hard_gated() -> None:
    packet = _packet()
    assert packet.packet_sha256 == _canonical_sha256(packet.identity_payload())
    prompt = packet.prompt_payload()
    assert not any(
        forbidden in json.dumps(prompt, sort_keys=True)
        for forbidden in ("target", "outcome", "resolution", "credential", "pnl")
    )
    with pytest.raises(ValueError, match="packet hash"):
        replace(packet, market_probability_range_60s=0.2)
    with pytest.raises(ValueError, match="safety state"):
        replace(packet, transport_gap_count_60s=1, packet_sha256="")
    with pytest.raises(ValueError, match="spread quantiles"):
        replace(
            packet,
            clob_relative_spread_median_60s=0.03,
            clob_relative_spread_p95_60s=0.02,
            packet_sha256="",
        )


def test_round25_ai_supervisor_preflight_binds_exact_local_model() -> None:
    version, show_hash = preflight_round25_ai_supervisor(
        _config(),
        get_json=_get_json,
        post_json=_post_json(),
    )
    assert version == "0.32.5"
    assert show_hash == _canonical_sha256(_show())

    def unknown_inventory_quantization(url: str, timeout: float) -> object:
        response = _get_json(url, timeout)
        if url.endswith("/api/tags"):
            assert isinstance(response, dict)
            response["models"][0]["details"]["quantization_level"] = "unknown"
        return response

    assert preflight_round25_ai_supervisor(
        _config(),
        get_json=unknown_inventory_quantization,
        post_json=_post_json(),
    )[0] == "0.32.5"

    def wrong_inventory(url: str, timeout: float) -> object:
        response = _get_json(url, timeout)
        if url.endswith("/api/tags"):
            assert isinstance(response, dict)
            response["models"][0]["digest"] = "f" * 64
        return response

    with pytest.raises(ValueError, match="model digest"):
        preflight_round25_ai_supervisor(
            _config(),
            get_json=wrong_inventory,
            post_json=_post_json(),
        )


@pytest.mark.parametrize(
    ("action", "multiplier", "cooldown_ms"),
    (
        ("normal", 1.0, 0),
        ("cautious_50", 0.5, 0),
        ("defensive_25", 0.25, 0),
        ("halt_300s", 0.0, 300_000),
    ),
)
def test_round25_ai_supervisor_actions_are_trusted_code_mapped(
    action: str,
    multiplier: float,
    cooldown_ms: int,
) -> None:
    result = review_round25_ai_supervisor_packet(
        _packet(),
        get_json=_get_json,
        post_json=_post_json(action),
        residency_inspector=_residency_inspector(),
        wall_clock_ms=_clock((1_060_100, 1_060_300)),
        monotonic_ns=_clock((0, 200_000_000)),
    )
    assert result.advisory.valid_model_response is True
    assert result.advisory.regime_action == action
    assert result.advisory.maximum_size_multiplier == multiplier
    assert result.advisory.cooldown_ms == cooldown_ms
    assert result.telemetry is not None
    assert result.telemetry.residency.fully_gpu_resident is True
    assert result.trading_authority is False


def test_round25_ai_supervisor_stale_invalid_and_nonresident_paths_fail_closed() -> None:
    def provider_must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("stale packet reached provider")

    stale = review_round25_ai_supervisor_packet(
        _packet(),
        get_json=provider_must_not_run,
        post_json=provider_must_not_run,
        wall_clock_ms=lambda: 1_066_000,
    )
    assert stale.advisory.failure_code == "packet_stale"
    assert stale.advisory.maximum_size_multiplier == 0.0

    invalid = review_round25_ai_supervisor_packet(
        _packet(),
        get_json=_get_json,
        post_json=_post_json("unknown"),
        residency_inspector=_residency_inspector(),
        wall_clock_ms=_clock((1_060_100, 1_060_300)),
        monotonic_ns=_clock((0, 200_000_000)),
    )
    assert invalid.advisory.failure_code == "schema_failure"
    assert invalid.telemetry is None

    nonresident = review_round25_ai_supervisor_packet(
        _packet(),
        get_json=_get_json,
        post_json=_post_json(),
        residency_inspector=_residency_inspector(ratio=0.5),
        wall_clock_ms=_clock((1_060_100, 1_060_300)),
        monotonic_ns=_clock((0, 200_000_000)),
    )
    assert nonresident.advisory.failure_code == "residency_failure"
    assert nonresident.advisory.maximum_size_multiplier == 0.0


def test_round25_ai_supervisor_worker_is_capacity_one_and_contains_faults() -> None:
    packet = _packet()
    started = threading.Event()
    release = threading.Event()

    def reviewer(selected: Round25AISupervisorPacket) -> Round25AISupervisorResult:
        started.set()
        assert release.wait(1.0)
        return _valid_result(selected)

    worker = Round25AISupervisorWorker(reviewer)
    assert worker.submit(packet) is True
    assert started.wait(1.0)
    assert worker.submit(packet) is False
    pending = worker.advisory_or_fail_closed(packet, now_ms=1_060_100)
    assert pending.advisory.failure_code == "pending_response"
    assert pending.advisory.maximum_size_multiplier == 0.0
    release.set()
    result = None
    deadline = time.monotonic() + 1.0
    while result is None and time.monotonic() < deadline:
        result = worker.poll(packet.packet_sha256)
        time.sleep(0.005)
    assert result is not None
    assert result.advisory.regime_action == "cautious_50"
    worker.close()


def test_round25_ai_hierarchy_can_only_preserve_or_reduce_risk() -> None:
    fast = Round25AIAdvisory(
        candidate_id="qwen3-4b-risk-advisor-v1",
        model="qwen3:4b",
        model_digest="e55aed6fe643f9368b2f48f8aaa56ec787b75765da69f794c0a0c23bfe7c64b2",
        packet_sha256="6" * 64,
        generated_at_ms=1_060_100,
        expires_at_ms=1_070_000,
        veto_new_entries=False,
        maximum_size_multiplier=0.75,
        cooldown_ms=0,
        reason_codes=("ai_advisory_restriction", "size_reduction_required"),
        summary="AI limited the proposed entry to 75 percent size.",
        valid_model_response=True,
        failure_code=None,
    )
    slow = _valid_result(_packet()).advisory
    combined = combine_round25_ai_risk(fast, slow, now_ms=1_060_300)
    assert combined.maximum_size_multiplier == 0.5
    assert combined.maximum_size_multiplier <= fast.maximum_size_multiplier
    assert combined.maximum_size_multiplier <= slow.maximum_size_multiplier
    assert combined.veto_new_entries is False
    assert combined.trading_authority is False

    stale = combine_round25_ai_risk(fast, slow, now_ms=1_200_000)
    assert stale.maximum_size_multiplier == 0.0
    assert stale.veto_new_entries is True
    assert "fast_advisory_stale" in stale.reason_codes
    assert "supervisor_advisory_stale" in stale.reason_codes


def test_rejected_fin_r1_supervisor_is_not_imported_by_active_source() -> None:
    source_root = REPOSITORY / "src" / "simple_ai_trading"
    research_files = {
        "polymarket_round25_ai_supervisor.py",
        "polymarket_round25_ai_supervisor_evaluation.py",
    }
    active = [
        path
        for path in source_root.glob("*.py")
        if path.name not in research_files
    ]
    assert all(
        "polymarket_round25_ai_supervisor" not in path.read_text(encoding="utf-8")
        and "fin-r1-8b-regime-supervisor-v1" not in path.read_text(encoding="utf-8")
        for path in active
    )
