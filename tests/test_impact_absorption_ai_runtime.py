from __future__ import annotations

import json
import subprocess
from typing import Any, Mapping

import pytest

from simple_ai_trading.ai_runtime import (
    AICapabilityReport,
    AIRuntimeConfig,
    OllamaResidencyReport,
)
from simple_ai_trading.impact_absorption_ai_protocol import (
    Round74AIModelManifest,
    Round74AIReviewDecision,
    Round74AIReviewRequest,
    ROUND74_AI_TEMPORAL_BLOCK_COUNT,
    ROUND74_AI_TEMPORAL_FEATURE_NAMES,
)
from simple_ai_trading.impact_absorption_ai_runtime import (
    Round74AIRuntimeConfig,
    Round74AIWorkerSession,
    preload_round74_ai_model,
    review_round74_ai_candidate,
    unload_round74_ai_model,
)
from simple_ai_trading.impact_absorption_ai_worker import (
    Round74AIWorkerEnvelope,
    Round74AIWorkerResult,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
)


WALL_NS = 1_800_000_000_000_000_000
MODEL_DIGEST = "d" * 64
METADATA_DIGEST = "e" * 64


def _manifest() -> Round74AIModelManifest:
    return Round74AIModelManifest(
        model_id="TheFinAI/Fino1-8B",
        model_revision="a" * 40,
        model_artifact_sha256=MODEL_DIGEST,
        model_artifact_kind="ollama_manifest",
        parameter_count=8_000_000_000,
        quantization="q6_k",
        runtime_backend="llama.cpp-vulkan",
        runtime_version="0.12.3",
        license_id="llama3.1",
        model_card_url="https://huggingface.co/TheFinAI/Fino1-8B",
        minimum_vram_bytes=8 * 1024**3,
        finance_specialized=True,
    )


def _request() -> Round74AIReviewRequest:
    count = len(ROUND74_EVENT_FEATURE_NAMES)
    return Round74AIReviewRequest(
        pretest_policy_sha256="1" * 64,
        probability_calibration_sha256="4" * 64,
        sample_sha256="2" * 64,
        deterministic_risk_state_sha256="3" * 64,
        risk_profile="conservative",
        asset_slot=0,
        side="long",
        horizon_seconds=30,
        requested_wall_ns=WALL_NS,
        expires_wall_ns=WALL_NS + 20_000_000_000,
        proposed_risk_size_bps=2_500,
        feature_last=tuple(0.0 for _ in range(count)),
        feature_mean=tuple(0.1 for _ in range(count)),
        feature_standard_deviation=tuple(0.2 for _ in range(count)),
        feature_recent_change=tuple(0.0 for _ in range(count)),
        feature_recent_block_means=tuple(
            tuple(0.0 for _ in ROUND74_AI_TEMPORAL_FEATURE_NAMES)
            for _ in range(ROUND74_AI_TEMPORAL_BLOCK_COUNT)
        ),
        payoff_quantiles_bps=(-5.0, -1.0, 2.0, 4.0, 7.0),
        maximum_adverse_excursion_quantiles_bps=(
            1.0,
            2.0,
            3.0,
            5.0,
            8.0,
        ),
        positive_payoff_probability=0.61,
        opposing_positive_payoff_probability=0.19,
        neither_positive_payoff_probability=0.20,
        adverse_selection_probability=0.27,
        regime_unpredictability_probability=0.18,
    )


def _capability(
    ok: bool = True,
    *,
    free_vram_gb: float | None = 10.0,
    free_ram_gb: float | None = None,
) -> AICapabilityReport:
    selected_free_ram_gb = (
        free_ram_gb if free_ram_gb is not None else 20.0 if ok else 8.0
    )
    return AICapabilityReport(
        ok=ok,
        provider="ollama",
        model="fino1:8b",
        gpu_vendor="amd",
        compute_backend_requested="directml",
        compute_backend_kind="directml",
        compute_backend_device="AMD Radeon RX 9070 XT",
        compute_backend_reason="",
        free_vram_gb=free_vram_gb,
        free_ram_gb=selected_free_ram_gb,
        model_parameters_b=8.0,
        messages=() if ok else ("free system RAM is below required",),
        warnings=(),
        provider_available=True,
        model_available=True,
        model_local=True,
    )


def _worker_result(envelope: Round74AIWorkerEnvelope) -> Round74AIWorkerResult:
    return Round74AIWorkerResult(
        envelope_sha256=envelope.envelope_sha256,
        manifest_sha256=envelope.model_manifest.manifest_sha256,
        request_sha256=envelope.review_request.request_sha256,
        model_name=envelope.model_name,
        model_digest=MODEL_DIGEST,
        model_metadata_sha256=METADATA_DIGEST,
        system_prompt_sha256="4" * 64,
        user_prompt_sha256="5" * 64,
        raw_response_sha256="6" * 64,
        decision=Round74AIReviewDecision(
            verdict="reduce",
            size_multiplier_bps=5_000,
            confidence_bps=7_500,
            reason_codes=("forecast_uncertainty",),
        ),
        residency=OllamaResidencyReport(
            requested_model=envelope.model_name,
            status="gpu_resident",
            loaded_model=envelope.model_name,
            digest=MODEL_DIGEST,
            size_bytes=1_000,
            size_vram_bytes=1_000,
            vram_to_model_ratio=1.0,
        ),
        prompt_eval_count=200,
        eval_count=40,
        total_duration_ns=1_000,
        load_duration_ns=100,
        prompt_eval_duration_ns=300,
        eval_duration_ns=600,
    )


def _unloaded_residency(
    _endpoint: str,
    model_name: str,
    _timeout_seconds: float,
    **_kwargs: object,
) -> OllamaResidencyReport:
    return OllamaResidencyReport(
        requested_model=model_name,
        status="unloaded",
        loaded_model=None,
        digest=None,
        size_bytes=None,
        size_vram_bytes=None,
        vram_to_model_ratio=None,
    )


class _Process:
    def __init__(
        self,
        output: str | None = None,
        *,
        returncode: int = 0,
        timeout: bool = False,
    ) -> None:
        self.output = output
        self.returncode = returncode
        self.timeout = timeout
        self.terminated = False
        self.killed = False

    def communicate(
        self,
        input: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        if self.timeout and not self.terminated and not self.killed:
            raise subprocess.TimeoutExpired("worker", timeout)
        if self.timeout:
            return "", ""
        if self.output is not None:
            return self.output, ""
        assert input is not None
        envelope = Round74AIWorkerEnvelope.from_dict(json.loads(input))
        return json.dumps(_worker_result(envelope).as_dict()), ""

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _factory(process: _Process) -> Any:
    def create(*_args: object, **_kwargs: object) -> _Process:
        return process

    return create


def _review(
    *,
    process: _Process,
    capability_ok: bool = True,
    deterministic_gate: bool = True,
    wall_times: list[int] | None = None,
):
    selected_times = iter(
        wall_times
        or [
            WALL_NS + 1,
            WALL_NS + 2,
            WALL_NS + 3,
        ]
    )
    return review_round74_ai_candidate(
        Round74AIRuntimeConfig(
            model_name="fino1:8b",
            timeout_seconds=10.0,
        ),
        _manifest(),
        _request(),
        deterministic_risk_gate_passed=deterministic_gate,
        observed_wall_ns=WALL_NS,
        capability_detector=lambda _config: _capability(capability_ok),
        provenance_resolver=lambda *_: (
            MODEL_DIGEST,
            METADATA_DIGEST,
        ),
        residency_inspector=_unloaded_residency,
        popen_factory=_factory(process),
        worker_command=("python", "-m", "isolated-worker"),
        monotonic_ns=lambda: 100,
        wall_time_ns=lambda: next(selected_times),
    )


def test_runtime_accepts_bound_worker_and_only_reduces_risk() -> None:
    outcome = _review(process=_Process())

    assert outcome.status == "accepted"
    assert outcome.approved_risk_size_bps == 1_250
    assert outcome.worker_result is not None
    assert outcome.as_dict()["remote_inference_used"] is False
    assert outcome.as_dict()["execution_authority"] is False


def test_runtime_accepts_hash_bound_persistent_worker_session() -> None:
    session = Round74AIWorkerSession()

    def request(
        envelope: Round74AIWorkerEnvelope,
        *,
        timeout_seconds: float,
    ) -> str:
        assert timeout_seconds > 0.0
        return json.dumps(_worker_result(envelope).as_dict())

    session.request = request  # type: ignore[method-assign]
    outcome = review_round74_ai_candidate(
        Round74AIRuntimeConfig(
            model_name="fino1:8b",
            timeout_seconds=10.0,
        ),
        _manifest(),
        _request(),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=WALL_NS,
        capability_detector=lambda _config: _capability(),
        provenance_resolver=lambda *_: (MODEL_DIGEST, METADATA_DIGEST),
        residency_inspector=_unloaded_residency,
        worker_session=session,
        monotonic_ns=lambda: 100,
        wall_time_ns=lambda: WALL_NS + 1,
    )
    session.close()

    assert outcome.status == "accepted"
    assert outcome.approved_risk_size_bps == 1_250
    assert outcome.capability is not None
    assert outcome.capability["worker_process_mode"] == "persistent_model_batch"
    assert outcome.capability["worker_session_request_ordinal"] == 1
    assert outcome.capability["worker_session_restart_count_before_request"] == 0
    assert outcome.capability["worker_session_restart_count_after_request"] == 0
    assert outcome.as_dict()["protective_exit_path_blocked"] is False


def test_runtime_blocks_before_spawn_when_capability_or_risk_gate_fails() -> None:
    capability_process = _Process()
    outcome = _review(
        process=capability_process,
        capability_ok=False,
    )
    assert outcome.status == "blocked_capability"
    assert outcome.approved_risk_size_bps == 0
    assert capability_process.output is None

    gate_process = _Process()
    outcome = _review(
        process=gate_process,
        deterministic_gate=False,
    )
    assert outcome.status == "blocked_deterministic_gate"
    assert outcome.approved_risk_size_bps == 0


def test_runtime_terminates_timed_out_worker_and_fails_closed() -> None:
    process = _Process(timeout=True)
    outcome = _review(process=process)

    assert outcome.status == "worker_timeout"
    assert outcome.approved_risk_size_bps == 0
    assert process.terminated is True


@pytest.mark.parametrize(
    "output",
    [
        "{}",
        '{"result_sha256":"a","result_sha256":"b"}',
        "not-json",
    ],
)
def test_runtime_rejects_malformed_worker_output(output: str) -> None:
    outcome = _review(process=_Process(output=output))

    assert outcome.status == "worker_output_invalid"
    assert outcome.approved_risk_size_bps == 0
    assert outcome.worker_result is None


def test_runtime_rejects_provenance_drift_without_spawning() -> None:
    spawned = False

    def spawn(*_args: object, **_kwargs: object) -> _Process:
        nonlocal spawned
        spawned = True
        return _Process()

    outcome = review_round74_ai_candidate(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        _request(),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=WALL_NS,
        capability_detector=lambda _config: _capability(),
        provenance_resolver=lambda *_: ("f" * 64, METADATA_DIGEST),
        residency_inspector=_unloaded_residency,
        popen_factory=spawn,
        monotonic_ns=lambda: 100,
        wall_time_ns=lambda: WALL_NS + 1,
    )

    assert outcome.status == "blocked_provenance"
    assert outcome.approved_risk_size_bps == 0
    assert spawned is False


def test_runtime_rechecks_request_expiry_after_worker() -> None:
    outcome = _review(
        process=_Process(),
        wall_times=[
            WALL_NS + 1,
            WALL_NS + 2,
            WALL_NS + 20_000_000_001,
        ],
    )

    assert outcome.status == "blocked_expired"
    assert outcome.approved_risk_size_bps == 0
    assert outcome.worker_result is None


def test_runtime_capability_policy_requires_declared_headroom() -> None:
    captured: list[AIRuntimeConfig] = []

    def detect(config: AIRuntimeConfig) -> AICapabilityReport:
        captured.append(config)
        return _capability(False)

    review_round74_ai_candidate(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        _request(),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=WALL_NS,
        capability_detector=detect,
        residency_inspector=_unloaded_residency,
        monotonic_ns=lambda: 100,
        wall_time_ns=lambda: WALL_NS + 1,
    )

    assert captured[0].require_gpu is False
    assert captured[0].min_free_ram_gb == 16.0
    assert captured[0].min_free_vram_gb == 8.0
    assert captured[0].allow_paper_fallback is False


def test_runtime_accepts_warm_exact_gpu_model_when_cold_load_headroom_is_low() -> None:
    process = _Process()
    inspections: list[tuple[str, str, str]] = []

    def inspect(
        endpoint: str,
        model_name: str,
        _timeout_seconds: float,
        *,
        expected_digest: str,
    ) -> OllamaResidencyReport:
        inspections.append((endpoint, model_name, expected_digest))
        return OllamaResidencyReport(
            requested_model=model_name,
            status="gpu_resident",
            loaded_model=model_name,
            digest=expected_digest,
            size_bytes=1_000,
            size_vram_bytes=1_000,
            vram_to_model_ratio=1.0,
        )

    outcome = review_round74_ai_candidate(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        _request(),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=WALL_NS,
        capability_detector=lambda _config: _capability(free_vram_gb=2.0),
        provenance_resolver=lambda *_: (MODEL_DIGEST, METADATA_DIGEST),
        residency_inspector=inspect,
        popen_factory=_factory(process),
        worker_command=("python", "-m", "isolated-worker"),
        monotonic_ns=lambda: 100,
        wall_time_ns=lambda: WALL_NS + 1,
    )

    assert outcome.status == "accepted"
    assert inspections == [("http://127.0.0.1:11434", "fino1:8b", MODEL_DIGEST)]
    assert outcome.capability is not None
    assert outcome.capability["pre_inference_cold_load_headroom_passed"] is False
    assert outcome.capability["pre_inference_exact_model_fully_gpu_resident"] is True
    assert outcome.capability["pre_inference_warm_ram_headroom_passed"] is True
    assert (
        outcome.capability["pre_inference_warm_equivalent_preload_ram_headroom_passed"]
        is True
    )
    assert outcome.capability["provider_runtime_full_gpu_residency_verified"] is True


def test_runtime_accepts_warm_model_with_state_aware_ram_headroom() -> None:
    detected_minimums: list[float] = []
    detected_free_ram = iter((19.0, 13.5))

    def detect(config: AIRuntimeConfig) -> AICapabilityReport:
        free_ram_gb = next(detected_free_ram)
        detected_minimums.append(config.min_free_ram_gb)
        return _capability(
            ok=free_ram_gb >= config.min_free_ram_gb,
            free_ram_gb=free_ram_gb,
        )

    outcome = review_round74_ai_candidate(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        _request(),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=WALL_NS,
        capability_detector=detect,
        provenance_resolver=lambda *_: (MODEL_DIGEST, METADATA_DIGEST),
        residency_inspector=lambda *_args, **_kwargs: OllamaResidencyReport(
            requested_model="fino1:8b",
            status="gpu_resident",
            loaded_model="fino1:8b",
            digest=MODEL_DIGEST,
            size_bytes=6 * 1024**3,
            size_vram_bytes=6 * 1024**3,
            vram_to_model_ratio=1.0,
        ),
        popen_factory=_factory(_Process()),
        worker_command=("python", "-m", "isolated-worker"),
        monotonic_ns=lambda: 100,
        wall_time_ns=lambda: WALL_NS + 1,
    )

    assert outcome.status == "accepted"
    assert detected_minimums == [16.0, 8.0]
    assert outcome.capability is not None
    assert outcome.capability["pre_inference_warm_ram_headroom_passed"] is True
    assert (
        outcome.capability["pre_inference_warm_equivalent_preload_ram_headroom_passed"]
        is True
    )
    assert outcome.capability["pre_inference_warm_equivalent_preload_ram_gb"] == 19.5


def test_runtime_does_not_relax_ram_headroom_for_a_warm_model() -> None:
    detected_minimums: list[float] = []

    def detect(config: AIRuntimeConfig) -> AICapabilityReport:
        detected_minimums.append(config.min_free_ram_gb)
        return AICapabilityReport(
            **{
                **_capability(free_vram_gb=2.0).asdict(),
                "ok": False,
                "free_ram_gb": 8.0,
                "messages": ("free system RAM is below required",),
            }
        )

    outcome = review_round74_ai_candidate(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        _request(),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=WALL_NS,
        capability_detector=detect,
        provenance_resolver=lambda *_: (MODEL_DIGEST, METADATA_DIGEST),
        residency_inspector=lambda *_args, **_kwargs: OllamaResidencyReport(
            requested_model="fino1:8b",
            status="gpu_resident",
            loaded_model="fino1:8b",
            digest=MODEL_DIGEST,
            size_bytes=1_000,
            size_vram_bytes=1_000,
            vram_to_model_ratio=1.0,
        ),
        popen_factory=_factory(_Process()),
        worker_command=("python", "-m", "isolated-worker"),
        monotonic_ns=lambda: 100,
        wall_time_ns=lambda: WALL_NS + 1,
    )

    assert outcome.status == "blocked_capability"
    assert detected_minimums == [16.0, 8.0]


def test_runtime_blocks_warm_model_without_equivalent_preload_ram_headroom() -> None:
    detections = 0

    def detect(config: AIRuntimeConfig) -> AICapabilityReport:
        nonlocal detections
        detections += 1
        free_ram_gb = 20.0 if detections == 1 else 8.5
        return _capability(
            ok=free_ram_gb >= config.min_free_ram_gb,
            free_ram_gb=free_ram_gb,
        )

    outcome = review_round74_ai_candidate(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        _request(),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=WALL_NS,
        capability_detector=detect,
        provenance_resolver=lambda *_: (MODEL_DIGEST, METADATA_DIGEST),
        residency_inspector=lambda *_args, **_kwargs: OllamaResidencyReport(
            requested_model="fino1:8b",
            status="gpu_resident",
            loaded_model="fino1:8b",
            digest=MODEL_DIGEST,
            size_bytes=5 * 1024**3,
            size_vram_bytes=5 * 1024**3,
            vram_to_model_ratio=1.0,
        ),
        popen_factory=_factory(_Process()),
        worker_command=("python", "-m", "isolated-worker"),
        monotonic_ns=lambda: 100,
        wall_time_ns=lambda: WALL_NS + 1,
    )

    assert outcome.status == "blocked_capability"
    assert outcome.capability is not None
    assert outcome.capability["pre_inference_warm_ram_headroom_passed"] is True
    assert (
        outcome.capability["pre_inference_warm_equivalent_preload_ram_headroom_passed"]
        is False
    )
    assert outcome.capability["pre_inference_warm_equivalent_preload_ram_gb"] == 13.5
    assert "pre-load headroom" in outcome.message


def test_runtime_blocks_low_headroom_when_exact_model_is_not_warm() -> None:
    spawned = False

    def spawn(*_args: object, **_kwargs: object) -> _Process:
        nonlocal spawned
        spawned = True
        return _Process()

    outcome = review_round74_ai_candidate(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        _request(),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=WALL_NS,
        capability_detector=lambda _config: _capability(free_vram_gb=2.0),
        residency_inspector=lambda *_args, **_kwargs: OllamaResidencyReport(
            requested_model="fino1:8b",
            status="unloaded",
            loaded_model=None,
            digest=None,
            size_bytes=None,
            size_vram_bytes=None,
            vram_to_model_ratio=None,
        ),
        popen_factory=spawn,
        monotonic_ns=lambda: 100,
        wall_time_ns=lambda: WALL_NS + 1,
    )

    assert outcome.status == "blocked_capability"
    assert outcome.approved_risk_size_bps == 0
    assert spawned is False
    assert outcome.capability is not None
    assert outcome.capability["pre_inference_exact_model_fully_gpu_resident"] is False


def test_runtime_blocks_nonexact_loaded_model_before_worker() -> None:
    spawned = False

    def spawn(*_args: object, **_kwargs: object) -> _Process:
        nonlocal spawned
        spawned = True
        return _Process()

    outcome = review_round74_ai_candidate(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        _request(),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=WALL_NS,
        capability_detector=lambda _config: _capability(),
        residency_inspector=lambda *_args, **_kwargs: OllamaResidencyReport(
            requested_model="fino1:8b",
            status="gpu_resident",
            loaded_model="unrelated-alias:8b",
            digest=MODEL_DIGEST,
            size_bytes=1_000,
            size_vram_bytes=1_000,
            vram_to_model_ratio=1.0,
        ),
        popen_factory=spawn,
        monotonic_ns=lambda: 100,
        wall_time_ns=lambda: WALL_NS + 1,
    )

    assert outcome.status == "blocked_capability"
    assert spawned is False


def test_runtime_blocks_unreadable_residency_before_worker() -> None:
    spawned = False

    def spawn(*_args: object, **_kwargs: object) -> _Process:
        nonlocal spawned
        spawned = True
        return _Process()

    def inspect(*_args: object, **_kwargs: object) -> OllamaResidencyReport:
        raise ValueError("malformed provider inventory")

    outcome = review_round74_ai_candidate(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        _request(),
        deterministic_risk_gate_passed=True,
        observed_wall_ns=WALL_NS,
        capability_detector=lambda _config: _capability(),
        residency_inspector=inspect,
        popen_factory=spawn,
        monotonic_ns=lambda: 100,
        wall_time_ns=lambda: WALL_NS + 1,
    )

    assert outcome.status == "blocked_capability"
    assert outcome.failure_class == "AICapabilityGate"
    assert spawned is False


def test_runtime_rejects_non_loopback_or_relaxed_resource_policy() -> None:
    with pytest.raises(ValueError, match="configuration differs"):
        Round74AIRuntimeConfig(
            model_name="fino1:8b",
            endpoint="http://localhost:11434",
        ).validate()
    with pytest.raises(ValueError, match="configuration differs"):
        Round74AIRuntimeConfig(
            model_name="fino1:8b",
            minimum_free_ram_gb=15.9,
        ).validate()
    with pytest.raises(ValueError, match="configuration differs"):
        Round74AIRuntimeConfig(
            model_name="fino1:8b",
            minimum_free_vram_gb=7.9,
        ).validate()


def test_declared_model_batch_unload_is_digest_bound_and_verified() -> None:
    loaded = OllamaResidencyReport(
        requested_model="fino1:8b",
        status="gpu_resident",
        loaded_model="fino1:8b",
        digest=MODEL_DIGEST,
        size_bytes=1_000,
        size_vram_bytes=1_000,
        vram_to_model_ratio=1.0,
    )
    reports = iter([loaded, loaded, _unloaded_residency("", "fino1:8b", 1.0)])
    requests: list[tuple[str, dict[str, object], float]] = []
    sleeps: list[float] = []

    def post(
        url: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> object:
        requests.append((url, payload, timeout_seconds))
        return {"done": True}

    unloaded = unload_round74_ai_model(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        residency_inspector=lambda *_args, **_kwargs: next(reports),
        provider_poster=post,
        monotonic=lambda: 0.0,
        sleeper=sleeps.append,
    )

    assert unloaded is True
    assert requests == [
        (
            "http://127.0.0.1:11434/api/generate",
            {"model": "fino1:8b", "keep_alive": 0, "stream": False},
            10.0,
        )
    ]
    assert sleeps == [0.25]


def test_declared_model_batch_preload_is_provenance_bound_and_gpu_verified() -> None:
    unloaded = _unloaded_residency("", "fino1:8b", 1.0)
    loaded = OllamaResidencyReport(
        requested_model="fino1:8b",
        status="gpu_resident",
        loaded_model="fino1:8b",
        digest=MODEL_DIGEST,
        size_bytes=1_000,
        size_vram_bytes=1_000,
        vram_to_model_ratio=1.0,
    )
    reports = iter([unloaded, loaded])
    requests: list[tuple[str, Mapping[str, object], float]] = []

    def post(
        url: str,
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> object:
        requests.append((url, payload, timeout_seconds))
        return {
            "model": "fino1:8b",
            "response": "",
            "done": True,
        }

    result = preload_round74_ai_model(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        capability_detector=lambda _config: _capability(),
        provenance_resolver=lambda *_args: (MODEL_DIGEST, METADATA_DIGEST),
        residency_inspector=lambda *_args, **_kwargs: next(reports),
        provider_poster=post,
    )

    assert result == loaded
    assert requests == [
        (
            "http://127.0.0.1:11434/api/generate",
            {
                "model": "fino1:8b",
                "keep_alive": "30m",
                "options": {
                    "num_ctx": 4096,
                },
                "stream": False,
            },
            120.0,
        )
    ]


def test_declared_model_batch_preload_reuses_exact_gpu_residency() -> None:
    loaded = OllamaResidencyReport(
        requested_model="fino1:8b",
        status="gpu_resident",
        loaded_model="fino1:8b",
        digest=MODEL_DIGEST,
        size_bytes=1_000,
        size_vram_bytes=1_000,
        vram_to_model_ratio=1.0,
    )
    posted = False

    def post(*_args: object, **_kwargs: object) -> object:
        nonlocal posted
        posted = True
        return {}

    result = preload_round74_ai_model(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        capability_detector=lambda _config: _capability(),
        provenance_resolver=lambda *_args: (MODEL_DIGEST, METADATA_DIGEST),
        residency_inspector=lambda *_args, **_kwargs: loaded,
        provider_poster=post,
    )

    assert result == loaded
    assert posted is False


def test_declared_model_batch_preload_rejects_partial_gpu_residency() -> None:
    partial = OllamaResidencyReport(
        requested_model="fino1:8b",
        status="gpu_resident",
        loaded_model="fino1:8b",
        digest=MODEL_DIGEST,
        size_bytes=1_000,
        size_vram_bytes=900,
        vram_to_model_ratio=0.9,
    )

    with pytest.raises(ValueError, match="preload residency differs"):
        preload_round74_ai_model(
            Round74AIRuntimeConfig(model_name="fino1:8b"),
            _manifest(),
            capability_detector=lambda _config: _capability(),
            provenance_resolver=lambda *_args: (MODEL_DIGEST, METADATA_DIGEST),
            residency_inspector=lambda *_args, **_kwargs: partial,
            provider_poster=lambda *_args, **_kwargs: {},
        )


def test_declared_model_batch_unload_does_not_load_an_absent_model() -> None:
    posted = False

    def post(*_args: object, **_kwargs: object) -> object:
        nonlocal posted
        posted = True
        return {"done": True}

    unloaded = unload_round74_ai_model(
        Round74AIRuntimeConfig(model_name="fino1:8b"),
        _manifest(),
        residency_inspector=_unloaded_residency,
        provider_poster=post,
    )

    assert unloaded is False
    assert posted is False


def test_declared_model_batch_unload_refuses_same_digest_alias() -> None:
    posted = False

    def post(*_args: object, **_kwargs: object) -> object:
        nonlocal posted
        posted = True
        return {"done": True}

    with pytest.raises(ValueError, match="model identity differs"):
        unload_round74_ai_model(
            Round74AIRuntimeConfig(model_name="fino1:8b"),
            _manifest(),
            residency_inspector=lambda *_args, **_kwargs: OllamaResidencyReport(
                requested_model="fino1:8b",
                status="gpu_resident",
                loaded_model="unrelated-alias:8b",
                digest=MODEL_DIGEST,
                size_bytes=1_000,
                size_vram_bytes=1_000,
                vram_to_model_ratio=1.0,
            ),
            provider_poster=post,
        )

    assert posted is False
