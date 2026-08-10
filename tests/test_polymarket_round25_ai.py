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
from simple_ai_trading.polymarket_round25_ai import (
    POLYMARKET_ROUND25_AI_PRELOAD_SECONDS,
    POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256,
    POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V1_SHA256,
    POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V2_SHA256,
    POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V3_SHA256,
    POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V4_SHA256,
    Round25AIAdvisoryPacket,
    Round25AIAdvisoryWorker,
    Round25AIConfig,
    preflight_round25_ai_candidate,
    preload_round25_ai_candidate,
    review_round25_ai_packet,
    unload_round25_ai_candidate,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _packet() -> Round25AIAdvisoryPacket:
    return Round25AIAdvisoryPacket(
        condition_id="0x" + "a" * 64,
        event_start_ms=1_000_000,
        decision_time_ms=1_010_000,
        expires_at_ms=1_020_000,
        feature_source_chain_sha256="1" * 64,
        ml_candidate_id="causal-multitask-tcn-residual-v1",
        ml_artifact_sha256="2" * 64,
        ml_prediction_sha256="3" * 64,
        proposed_side="up",
        model_probability_up=0.61,
        market_prior_probability_up=0.55,
        executable_entry_price=0.56,
        conservative_edge_after_cost=0.03,
        epistemic_uncertainty=0.08,
        predicted_adverse_selection_probability=0.18,
        relative_spread=0.015,
        top_executable_notional_usd=2_500.0,
        book_receipt_age_ms=45.0,
        reference_receipt_age_ms=30.0,
        transport_gap_count_60s=0,
        realized_volatility_60s=0.002,
        short_term_log_return_5s=0.0004,
        order_flow_imbalance_5s=0.2,
        current_condition_exposure_fraction=0.0,
        portfolio_risk_utilization=0.1,
        deterministic_gate_sha256="4" * 64,
    )


def _config(candidate_id: str = "qwen3-4b-risk-advisor-v1") -> Round25AIConfig:
    return Round25AIConfig(candidate_id=candidate_id)


def _show(config: Round25AIConfig) -> dict[str, object]:
    return {
        "details": {
            "format": "gguf",
            "parameter_size": config.candidate.parameter_size,
        },
        "model_info": {
            "general.parameter_count": 4_022_468_096,
        },
        "capabilities": ["completion"],
    }


def _get_json(config: Round25AIConfig):
    def get_json(url: str, _timeout: float) -> object:
        if url.endswith("/api/version"):
            return {"version": "0.32.5"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": config.candidate.model,
                        "model": config.candidate.model,
                        "digest": config.candidate.digest,
                        "details": {
                            "format": "gguf",
                            "parameter_size": config.candidate.parameter_size,
                        },
                    }
                ]
            }
        raise AssertionError(f"unexpected GET: {url}")

    return get_json


def _provider_response(config: Round25AIConfig, response: Mapping[str, object] | None = None):
    decision = response or {"risk_action": "allow"}
    return {
        "model": config.candidate.model,
        "created_at": "2026-08-10T16:00:00Z",
        "response": json.dumps(decision, separators=(",", ":")),
        "done": True,
        "done_reason": "stop",
        "total_duration": 200_000_000,
        "load_duration": 1_000_000,
        "prompt_eval_count": 500,
        "prompt_eval_duration": 80_000_000,
        "eval_count": 30,
        "eval_duration": 100_000_000,
    }


def _post_json(config: Round25AIConfig, response: Mapping[str, object] | None = None):
    def post_json(url: str, payload: Mapping[str, object], _timeout: float) -> object:
        if url.endswith("/api/show"):
            assert payload == {"model": config.candidate.model, "verbose": False}
            return _show(config)
        if url.endswith("/api/generate"):
            assert payload["model"] == config.candidate.model
            return _provider_response(config, response)
        raise AssertionError(f"unexpected POST: {url}")

    return post_json


def _residency(
    config: Round25AIConfig,
    *,
    ratio: float = 1.0,
) -> OllamaResidencyReport:
    size = 6_000_000_000
    return OllamaResidencyReport(
        requested_model=config.candidate.model,
        status="gpu_resident" if ratio > 0.0 else "cpu_only",
        loaded_model=config.candidate.model,
        digest=config.candidate.digest,
        size_bytes=size,
        size_vram_bytes=int(size * ratio),
        vram_to_model_ratio=int(size * ratio) / size,
    ).validated()


def _residency_inspector(config: Round25AIConfig, *, ratio: float = 1.0):
    def inspect(
        base_url: str,
        model: str,
        _timeout: float,
        *,
        expected_digest: str,
    ) -> OllamaResidencyReport:
        assert base_url == config.base_url
        assert model == config.candidate.model
        assert expected_digest == config.candidate.digest
        return _residency(config, ratio=ratio)

    return inspect


def _clock(values: tuple[int, ...]):
    selected = iter(values)
    return lambda: next(selected)


def test_round25_ai_contract_is_self_hashed_and_claim_free() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-ai-risk-advisory-contract-v1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V1_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["matched_uplift_evaluation"]["minimum_conditions"] == 500
    assert contract["matched_uplift_evaluation"]["round25_selection_population_reuse_allowed"] is False
    assert contract["truth_state"] == {
        "round25_v2_data_captured": False,
        "predictive_candidate_selected": False,
        "ai_candidate_operator_implemented": False,
        "ai_host_mechanics_verified": False,
        "ai_uplift_population_captured": False,
        "ai_uplift_evaluated": False,
        "ai_uplift_verified": False,
        "predictive_edge_verified": False,
        "profitability_verified": False,
        "paper_authority": False,
        "live_authority": False,
        "orders_submitted": False,
    }


def test_round25_ai_v2_contract_is_self_hashed_and_model_response_blind() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-ai-risk-advisory-contract-v2.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V2_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["observed_failure_boundary"]["model_response_observed"] is False
    assert contract["correction"]["maximum_provider_wall_seconds"] == 10.0
    assert contract["correction"]["hidden_thinking_after"] is False
    assert contract["correction"]["retry_count"] == 0


def test_round25_ai_v3_contract_is_self_hashed_before_candidate_response() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-ai-risk-advisory-contract-v3.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V3_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["active_runtime_candidate"]["ollama_model"] == "qwen3:4b"
    assert contract["candidate_selection_integrity"]["qwen3_4b_model_response_observed_before_freeze"] is False
    assert contract["matched_uplift_evaluation"]["minimum_conditions"] == 500


def test_round25_ai_v4_contract_is_self_hashed_and_preserves_deadline() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-ai-risk-advisory-contract-v4.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V4_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["observed_failure_boundary"]["ollama_generated_tokens"] == 64
    assert contract["correction"]["maximum_output_tokens_after"] == 80
    assert contract["correction"]["maximum_provider_wall_seconds"] == 10.0
    assert contract["correction"]["risk_semantics_changed"] is False


def test_round25_ai_v5_contract_is_self_hashed_and_reduces_model_authority() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-ai-risk-advisory-contract-v5.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["correction"]["model_output_protocol_after"] == "one_enum_risk_action"
    assert contract["correction"]["operator_derives_veto_multiplier_cooldown_reasons_and_summary"] is True
    assert contract["correction"]["maximum_provider_wall_seconds"] == 10.0
    assert contract["inherited_unchanged"]["ai_can_create_entry"] is False


def test_round25_ai_v6_host_probe_is_hash_bound_and_non_authoritative() -> None:
    root = Path(__file__).parents[1]
    path = (
        root
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-ai-risk-advisory-host-probe-v6-2026-08-10.json"
    )
    evidence = json.loads(path.read_text(encoding="ascii"))
    claimed = evidence.pop("evidence_sha256")

    assert claimed == _canonical_sha256(evidence)
    assert evidence["status"] == "runtime_mechanics_verified"
    assert len(evidence["candidates"]) == 1
    candidate = evidence["candidates"][0]
    assert candidate["passed"] is True
    assert all(candidate["checks"].values())
    assert candidate["review_result"]["telemetry"]["residency"]["vram_to_model_ratio"] == 1.0
    assert candidate["review_result"]["telemetry"]["measured_latency_seconds"] <= 10.0
    assert not any(evidence["claims"].values())
    for key, relative in (
        ("contract_file_sha256", evidence["source"]["contract_path"]),
        ("ai_module_sha256", evidence["source"]["ai_module_path"]),
        ("probe_tool_sha256", evidence["source"]["probe_tool_path"]),
    ):
        assert evidence["source"][key] == hashlib.sha256(
            (root / relative).read_bytes()
        ).hexdigest()


def test_round25_ai_packet_is_target_free_hash_bound_and_safety_gated() -> None:
    packet = _packet()

    assert packet.packet_sha256 == _canonical_sha256(packet.identity_payload())
    assert packet.target_accessed is False
    assert packet.outcome_accessed is False
    assert packet.resolution_accessed is False
    assert packet.credential_accessed is False
    assert "target" not in packet.prompt_payload()

    with pytest.raises(ValueError, match="packet hash"):
        replace(packet, relative_spread=0.5)
    with pytest.raises(ValueError, match="safety state"):
        replace(packet, unknown_order_state=True, packet_sha256="")
    with pytest.raises(ValueError, match="safety state"):
        replace(packet, deterministic_entry_allowed=False, packet_sha256="")


def test_round25_ai_preflight_binds_version_digest_and_local_metadata() -> None:
    config = _config()

    version, metadata_sha256 = preflight_round25_ai_candidate(
        config,
        get_json=_get_json(config),
        post_json=_post_json(config),
    )

    assert version == "0.32.5"
    assert metadata_sha256 == _canonical_sha256(_show(config))

    def drifted_get(url: str, timeout: float) -> object:
        value = _get_json(config)(url, timeout)
        if url.endswith("/api/tags"):
            assert isinstance(value, dict)
            value["models"][0]["digest"] = "f" * 64
        return value

    with pytest.raises(ValueError, match="digest"):
        preflight_round25_ai_candidate(
            config,
            get_json=drifted_get,
            post_json=_post_json(config),
        )


def test_round25_ai_review_accepts_only_coherent_fully_gpu_resident_output() -> None:
    config = _config()
    packet = _packet()
    result = review_round25_ai_packet(
        packet,
        config,
        get_json=_get_json(config),
        post_json=_post_json(config),
        residency_inspector=_residency_inspector(config),
        wall_clock_ms=_clock((1_012_000, 1_012_200)),
        monotonic_ns=_clock((0, 200_000_000)),
    )

    assert result.advisory.valid_model_response is True
    assert result.advisory.veto_new_entries is False
    assert result.advisory.maximum_size_multiplier == 1.0
    assert result.advisory.can_create_entry is False
    assert result.advisory.can_increase_risk is False
    assert result.advisory.can_block_exit is False
    assert result.telemetry is not None
    assert result.telemetry.residency.fully_gpu_resident is True
    assert result.telemetry.model_digest == config.candidate.digest
    assert result.ai_uplift_verified is False
    assert result.profitability_verified is False
    assert result.live_authority is False
    assert result.order_submitted is False
    assert result.result_sha256 == _canonical_sha256(result.identity_payload())


@pytest.mark.parametrize(
    ("risk_action", "veto", "multiplier", "cooldown_ms"),
    (
        ("allow", False, 1.0, 0),
        ("reduce_75", False, 0.75, 0),
        ("reduce_50", False, 0.5, 0),
        ("reduce_25", False, 0.25, 0),
        ("veto", True, 0.0, 0),
        ("cooldown_60s", True, 0.0, 60_000),
        ("cooldown_300s", True, 0.0, 300_000),
    ),
)
def test_round25_ai_discrete_actions_have_deterministic_risk_semantics(
    risk_action: str,
    veto: bool,
    multiplier: float,
    cooldown_ms: int,
) -> None:
    config = _config()
    result = review_round25_ai_packet(
        _packet(),
        config,
        get_json=_get_json(config),
        post_json=_post_json(config, {"risk_action": risk_action}),
        residency_inspector=_residency_inspector(config),
        wall_clock_ms=_clock((1_012_000, 1_012_200)),
        monotonic_ns=_clock((0, 200_000_000)),
    )

    assert result.advisory.valid_model_response is True
    assert result.advisory.veto_new_entries is veto
    assert result.advisory.maximum_size_multiplier == multiplier
    assert result.advisory.cooldown_ms == cooldown_ms
    assert result.advisory.can_create_entry is False
    assert result.advisory.can_increase_risk is False
    assert result.advisory.can_block_exit is False


@pytest.mark.parametrize(
    "decision",
    (
        {"risk_action": "reduce_33"},
        {"risk_action": "allow", "unexpected": True},
    ),
)
def test_round25_ai_invalid_discrete_output_fails_closed(
    decision: Mapping[str, object],
) -> None:
    config = _config()
    result = review_round25_ai_packet(
        _packet(),
        config,
        get_json=_get_json(config),
        post_json=_post_json(config, decision),
        residency_inspector=_residency_inspector(config),
        wall_clock_ms=_clock((1_012_000, 1_012_200)),
        monotonic_ns=_clock((0, 200_000_000)),
    )

    assert result.advisory.valid_model_response is False
    assert result.advisory.failure_code == "schema_failure"
    assert result.advisory.veto_new_entries is True
    assert result.advisory.maximum_size_multiplier == 0.0
    assert result.telemetry is None


def test_round25_ai_stale_nonresident_and_slow_paths_fail_closed() -> None:
    config = _config()
    calls = 0

    def no_get(_url: str, _timeout: float) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("stale packet reached provider")

    stale = review_round25_ai_packet(
        _packet(),
        config,
        get_json=no_get,
        wall_clock_ms=lambda: 1_016_000,
    )
    assert stale.advisory.failure_code == "packet_stale"
    assert calls == 0

    nonresident = review_round25_ai_packet(
        _packet(),
        config,
        get_json=_get_json(config),
        post_json=_post_json(config),
        residency_inspector=_residency_inspector(config, ratio=0.5),
        wall_clock_ms=_clock((1_012_000, 1_012_200)),
        monotonic_ns=_clock((0, 200_000_000)),
    )
    assert nonresident.advisory.failure_code == "residency_failure"
    assert nonresident.advisory.veto_new_entries is True

    slow = review_round25_ai_packet(
        _packet(),
        config,
        get_json=_get_json(config),
        post_json=_post_json(config),
        residency_inspector=_residency_inspector(config),
        wall_clock_ms=_clock((1_012_000, 1_019_000)),
        monotonic_ns=_clock((0, 11_000_000_000)),
    )
    assert slow.advisory.failure_code == "latency_failure"
    assert slow.advisory.maximum_size_multiplier == 0.0


def test_round25_ai_preload_and_unload_require_exact_residency_transition() -> None:
    config = _config()
    resident = _residency(config)
    unloaded = OllamaResidencyReport(
        requested_model=config.candidate.model,
        status="unloaded",
        loaded_model=None,
        digest=None,
        size_bytes=None,
        size_vram_bytes=None,
        vram_to_model_ratio=None,
    ).validated()
    residency_values = iter((resident, unloaded))

    def post_json(
        url: str,
        payload: Mapping[str, object],
        _timeout: float,
    ) -> object:
        assert url.endswith("/api/generate")
        if payload.get("keep_alive") == 0:
            return {"done": True, "done_reason": "unload"}
        assert _timeout == POLYMARKET_ROUND25_AI_PRELOAD_SECONDS
        return {"done": True, "done_reason": "load"}

    def inspect(*_args: object, **_kwargs: object) -> OllamaResidencyReport:
        return next(residency_values)

    assert preload_round25_ai_candidate(
        config,
        post_json=post_json,
        residency_inspector=inspect,
    ).fully_gpu_resident
    unload_round25_ai_candidate(
        config,
        post_json=post_json,
        residency_inspector=inspect,
    )


def test_round25_ai_worker_is_nonblocking_daemon_and_contains_reviewer_fault() -> None:
    config = _config()
    packet = _packet()
    release = threading.Event()

    def reviewer(_packet: Round25AIAdvisoryPacket):
        assert release.wait(timeout=1.0)
        raise RuntimeError("contained test failure")

    worker = Round25AIAdvisoryWorker(
        config=config,
        reviewer=reviewer,
        wall_clock_ms=lambda: 1_012_500,
    )
    started = time.perf_counter()
    assert worker.submit(packet) is True
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05
    assert worker.thread_is_daemon is True
    assert worker.submit(packet) is False
    pending = worker.advisory_or_fail_closed(packet)
    assert pending.failure_code == "pending_response"
    assert pending.veto_new_entries is True

    release.set()
    result = None
    deadline = time.monotonic() + 1.0
    while result is None and time.monotonic() < deadline:
        result = worker.poll(packet.packet_sha256)
        time.sleep(0.005)
    worker.close()

    assert result is not None
    assert result.advisory.failure_code == "worker_failure"
    assert result.advisory.veto_new_entries is True
    assert result.advisory.can_block_exit is False
