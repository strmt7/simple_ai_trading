"""Fail-closed parent coordinator for isolated Round 74 AI review."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing
import os
import subprocess  # nosec B404
import sys
import threading
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

from .ai_review import resolve_ollama_model_provenance
from .ai_runtime import (
    AICapabilityReport,
    AIRuntimeConfig,
    OllamaResidencyReport,
    detect_ai_capabilities,
    inspect_ollama_model_residency,
)
from .impact_absorption_ai_protocol import (
    Round74AIModelManifest,
    Round74AIReviewRequest,
    apply_round74_ai_risk_modifier,
)
from .impact_absorption_ai_worker import (
    ROUND74_AI_WORKER_ENDPOINT,
    ROUND74_AI_WORKER_MAXIMUM_INPUT_BYTES,
    ROUND74_AI_WORKER_MAXIMUM_RESPONSE_BYTES,
    ROUND74_AI_WORKER_SESSION_RESPONSE_SCHEMA_VERSION,
    ROUND74_AI_WORKER_MAXIMUM_TIMEOUT_SECONDS,
    Round74AIWorkerEnvelope,
    Round74AIWorkerResult,
    serve_round74_ai_worker_connection,
)


ROUND74_AI_RUNTIME_OUTCOME_SCHEMA_VERSION = "round-074-ai-runtime-outcome-v4"
ROUND74_AI_RUNTIME_MINIMUM_FREE_RAM_GB = 16.0
ROUND74_AI_RUNTIME_MINIMUM_FREE_VRAM_GB = 8.0
ROUND74_AI_RUNTIME_MINIMUM_WARM_FREE_RAM_GB = 8.0
ROUND74_AI_RUNTIME_PRELOAD_TIMEOUT_SECONDS = 120.0
ROUND74_AI_RUNTIME_STATUSES = (
    "accepted",
    "blocked_deterministic_gate",
    "blocked_expired",
    "blocked_capability",
    "blocked_provenance",
    "worker_timeout",
    "worker_failed",
    "worker_output_invalid",
)

CapabilityDetector = Callable[[AIRuntimeConfig], AICapabilityReport]
ProvenanceResolver = Callable[[str, str, float], tuple[str, str]]
ResidencyInspector = Callable[..., OllamaResidencyReport]
ProviderPoster = Callable[[str, Mapping[str, object], float], object]
PopenFactory = Callable[..., subprocess.Popen[str]]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _bounded_message(value: object, limit: int = 240) -> str:
    return " ".join(str(value or "").split())[:limit]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalized_model_reference(value: object) -> str:
    selected = str(value or "").strip().lower()
    return selected if ":" in selected else f"{selected}:latest"


@dataclass(frozen=True)
class Round74AIRuntimeConfig:
    """Resource and process limits for one local, veto-only review."""

    model_name: str
    endpoint: str = ROUND74_AI_WORKER_ENDPOINT
    timeout_seconds: float = 20.0
    minimum_free_ram_gb: float = ROUND74_AI_RUNTIME_MINIMUM_FREE_RAM_GB
    minimum_free_vram_gb: float = ROUND74_AI_RUNTIME_MINIMUM_FREE_VRAM_GB

    def validate(self) -> None:
        values = (
            self.timeout_seconds,
            self.minimum_free_ram_gb,
            self.minimum_free_vram_gb,
        )
        if (
            not self.model_name.strip()
            or len(self.model_name) > 160
            or any(character.isspace() for character in self.model_name)
            or self.endpoint.rstrip("/") != ROUND74_AI_WORKER_ENDPOINT
            or any(
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in values
            )
            or self.timeout_seconds > ROUND74_AI_WORKER_MAXIMUM_TIMEOUT_SECONDS
            or self.minimum_free_ram_gb < ROUND74_AI_RUNTIME_MINIMUM_FREE_RAM_GB
            or self.minimum_free_vram_gb < ROUND74_AI_RUNTIME_MINIMUM_FREE_VRAM_GB
        ):
            raise ValueError("Round 74 AI runtime configuration differs")


@dataclass(frozen=True)
class Round74AIRuntimeOutcome:
    """One auditable parent decision; every non-accepted state is zero risk."""

    status: str
    request_sha256: str
    manifest_sha256: str
    deterministic_risk_gate_passed: bool
    observed_wall_ns: int
    proposed_risk_size_bps: int
    approved_risk_size_bps: int
    capability: Mapping[str, object] | None
    resolved_model_digest: str | None
    resolved_model_metadata_sha256: str | None
    worker_result: Mapping[str, object] | None
    elapsed_ns: int
    failure_class: str | None
    message: str
    schema_version: str = ROUND74_AI_RUNTIME_OUTCOME_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_AI_RUNTIME_OUTCOME_SCHEMA_VERSION
            or self.status not in ROUND74_AI_RUNTIME_STATUSES
            or not isinstance(self.deterministic_risk_gate_passed, bool)
            or isinstance(self.observed_wall_ns, bool)
            or not isinstance(self.observed_wall_ns, int)
            or self.observed_wall_ns <= 0
            or isinstance(self.proposed_risk_size_bps, bool)
            or not isinstance(self.proposed_risk_size_bps, int)
            or not 1 <= self.proposed_risk_size_bps <= 10_000
            or isinstance(self.approved_risk_size_bps, bool)
            or not isinstance(self.approved_risk_size_bps, int)
            or not 0 <= self.approved_risk_size_bps <= self.proposed_risk_size_bps
            or isinstance(self.elapsed_ns, bool)
            or not isinstance(self.elapsed_ns, int)
            or self.elapsed_ns < 0
            or self.message != _bounded_message(self.message)
        ):
            raise ValueError("Round 74 AI runtime outcome differs")
        for value in (self.request_sha256, self.manifest_sha256):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("Round 74 AI runtime outcome digest differs")
        if self.status == "accepted":
            if (
                self.worker_result is None
                or self.resolved_model_digest is None
                or self.resolved_model_metadata_sha256 is None
                or self.failure_class is not None
            ):
                raise ValueError("Round 74 AI accepted outcome evidence differs")
            worker = Round74AIWorkerResult.from_dict(self.worker_result)
            if (
                worker.request_sha256 != self.request_sha256
                or worker.manifest_sha256 != self.manifest_sha256
                or worker.model_digest != self.resolved_model_digest
                or worker.model_metadata_sha256 != self.resolved_model_metadata_sha256
            ):
                raise ValueError("Round 74 AI accepted outcome binding differs")
        elif (
            self.approved_risk_size_bps != 0
            or self.worker_result is not None
            or not self.failure_class
        ):
            raise ValueError("Round 74 AI blocked outcome must fail closed")

    @property
    def outcome_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "status": self.status,
            "request_sha256": self.request_sha256,
            "manifest_sha256": self.manifest_sha256,
            "deterministic_risk_gate_passed": (self.deterministic_risk_gate_passed),
            "observed_wall_ns": self.observed_wall_ns,
            "proposed_risk_size_bps": self.proposed_risk_size_bps,
            "approved_risk_size_bps": self.approved_risk_size_bps,
            "capability": (
                dict(self.capability) if self.capability is not None else None
            ),
            "resolved_model_digest": self.resolved_model_digest,
            "resolved_model_metadata_sha256": (self.resolved_model_metadata_sha256),
            "worker_result": (
                dict(self.worker_result) if self.worker_result is not None else None
            ),
            "elapsed_ns": self.elapsed_ns,
            "failure_class": self.failure_class,
            "message": self.message,
            "remote_inference_used": False,
            "execution_authority": False,
            "protective_exit_path_blocked": False,
        }
        if include_sha256:
            payload["outcome_sha256"] = _canonical_sha256(payload)
        return payload


def _outcome(
    *,
    status: str,
    request: Round74AIReviewRequest,
    manifest: Round74AIModelManifest,
    deterministic_risk_gate_passed: bool,
    observed_wall_ns: int,
    capability: Mapping[str, object] | None,
    resolved_digest: str | None,
    resolved_metadata_sha256: str | None,
    worker_result: Round74AIWorkerResult | None,
    approved_risk_size_bps: int,
    started_ns: int,
    failure_class: str | None,
    message: str,
    monotonic_ns: Callable[[], int],
) -> Round74AIRuntimeOutcome:
    selected = Round74AIRuntimeOutcome(
        status=status,
        request_sha256=request.request_sha256,
        manifest_sha256=manifest.manifest_sha256,
        deterministic_risk_gate_passed=deterministic_risk_gate_passed,
        observed_wall_ns=observed_wall_ns,
        proposed_risk_size_bps=request.proposed_risk_size_bps,
        approved_risk_size_bps=approved_risk_size_bps,
        capability=capability,
        resolved_model_digest=resolved_digest,
        resolved_model_metadata_sha256=resolved_metadata_sha256,
        worker_result=(worker_result.as_dict() if worker_result is not None else None),
        elapsed_ns=max(0, monotonic_ns() - started_ns),
        failure_class=failure_class,
        message=_bounded_message(message),
    )
    selected.validate()
    return selected


def _capability_config(
    config: Round74AIRuntimeConfig,
    manifest: Round74AIModelManifest,
    *,
    exact_model_already_fully_gpu_resident: bool = False,
) -> AIRuntimeConfig:
    return AIRuntimeConfig(
        enabled=True,
        provider="ollama",
        model=config.model_name,
        # Ollama has its own provider runtime. Exact GPU residency is proven
        # after inference; Python's training backend is not provider evidence.
        require_gpu=False,
        min_free_vram_gb=max(
            config.minimum_free_vram_gb,
            manifest.minimum_vram_bytes / 1024**3,
        ),
        min_free_ram_gb=(
            ROUND74_AI_RUNTIME_MINIMUM_WARM_FREE_RAM_GB
            if exact_model_already_fully_gpu_resident
            else config.minimum_free_ram_gb
        ),
        min_model_parameters_b=2.0,
        allow_paper_fallback=False,
    )


def _provider_capability_messages(
    capability: AICapabilityReport,
    *,
    minimum_free_vram_gb: float,
    exact_model_already_fully_gpu_resident: bool,
) -> tuple[str, ...]:
    messages = list(capability.messages)
    if exact_model_already_fully_gpu_resident:
        return tuple(messages)
    if capability.free_vram_gb is None:
        messages.append("free VRAM could not be measured before local AI inference")
    elif capability.free_vram_gb < minimum_free_vram_gb:
        messages.append(
            "free VRAM "
            f"{capability.free_vram_gb:.1f} GiB is below required "
            f"{minimum_free_vram_gb:.1f} GiB"
        )
    return tuple(messages)


def _default_worker_command() -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "simple_ai_trading.impact_absorption_ai_worker",
    )


def _terminate_worker(process: subprocess.Popen[str]) -> None:
    try:
        process.terminate()
        process.communicate(timeout=1.0)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
        process.communicate(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _spawn_worker(
    command: Sequence[str],
    envelope: Round74AIWorkerEnvelope,
    *,
    timeout_seconds: float,
    popen_factory: PopenFactory,
) -> tuple[str, str]:
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("Round 74 AI worker command differs")
    payload = _canonical_json(envelope.as_dict())
    if len(payload.encode("ascii")) > ROUND74_AI_WORKER_MAXIMUM_INPUT_BYTES:
        raise ValueError("Round 74 AI worker input exceeds the limit")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = popen_factory(  # nosec B603 - fixed argv, no shell
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
        shell=False,
        creationflags=creationflags,
    )
    try:
        stdout, stderr = process.communicate(
            payload,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        _terminate_worker(process)
        raise
    if process.returncode != 0:
        detail = _bounded_message(stderr, 120)
        raise RuntimeError(f"Round 74 AI worker exited {process.returncode}: {detail}")
    if len(stdout.encode("utf-8")) > ROUND74_AI_WORKER_MAXIMUM_RESPONSE_BYTES:
        raise ValueError("Round 74 AI worker output exceeds the limit")
    return stdout, stderr


def _strict_worker_result(raw_text: str) -> Round74AIWorkerResult:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("Round 74 AI worker output contains duplicate keys")
            output[key] = value
        return output

    payload = json.loads(
        raw_text,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Round 74 AI worker output contains {value}")
        ),
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Round 74 AI worker output root differs")
    return Round74AIWorkerResult.from_dict(payload)


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
    *,
    label: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{label} contains duplicate keys")
        result[key] = value
    return result


class Round74AIWorkerSession:
    """Supervise one restartable isolated worker process for a model batch."""

    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._connection: object | None = None
        self._process: object | None = None
        self._lock = threading.Lock()
        self._started_once = False
        self._request_count = 0
        self._restart_count = 0

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def restart_count(self) -> int:
        return self._restart_count

    def _discard(self, *, terminate: bool) -> None:
        connection = self._connection
        process = self._process
        self._connection = None
        self._process = None
        if connection is not None:
            try:
                connection.close()
            except (OSError, ValueError):
                pass
        if process is not None:
            try:
                if terminate and process.is_alive():
                    process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
            except (OSError, ValueError):
                pass

    def _start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._discard(terminate=True)
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=serve_round74_ai_worker_connection,
            args=(child,),
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        self._connection = parent
        self._process = process
        if self._started_once:
            self._restart_count += 1
        self._started_once = True

    def request(
        self,
        envelope: Round74AIWorkerEnvelope,
        *,
        timeout_seconds: float,
    ) -> str:
        envelope.validate()
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0.0
        ):
            raise ValueError("Round 74 AI worker session timeout differs")
        raw = _canonical_json(envelope.as_dict()).encode("ascii")
        if len(raw) > ROUND74_AI_WORKER_MAXIMUM_INPUT_BYTES:
            raise ValueError("Round 74 AI worker session input exceeds the limit")
        with self._lock:
            self._start()
            connection = self._connection
            assert connection is not None
            self._request_count += 1
            try:
                connection.send_bytes(raw)
                if not connection.poll(float(timeout_seconds)):
                    raise subprocess.TimeoutExpired(
                        "round74-ai-worker-session",
                        float(timeout_seconds),
                    )
                response_raw = connection.recv_bytes(
                    ROUND74_AI_WORKER_MAXIMUM_RESPONSE_BYTES
                )
            except subprocess.TimeoutExpired:
                self._discard(terminate=True)
                raise
            except (EOFError, OSError, ValueError) as exc:
                self._discard(terminate=True)
                raise RuntimeError(
                    "Round 74 AI worker session transport failed"
                ) from exc
            try:
                response = json.loads(
                    response_raw.decode("ascii"),
                    object_pairs_hook=lambda pairs: _reject_duplicate_pairs(
                        pairs,
                        label="Round 74 AI worker session response",
                    ),
                    parse_constant=lambda item: (_ for _ in ()).throw(
                        ValueError(
                            f"Round 74 AI worker session response contains {item}"
                        )
                    ),
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._discard(terminate=True)
                raise RuntimeError(
                    "Round 74 AI worker session response is invalid"
                ) from exc
            if (
                not isinstance(response, Mapping)
                or set(response)
                != {"schema_version", "status", "result", "error_class"}
                or response.get("schema_version")
                != ROUND74_AI_WORKER_SESSION_RESPONSE_SCHEMA_VERSION
            ):
                self._discard(terminate=True)
                raise RuntimeError(
                    "Round 74 AI worker session response fields differ"
                )
            if response.get("status") == "error":
                error_class = response.get("error_class")
                self._discard(terminate=True)
                raise RuntimeError(
                    "Round 74 AI worker session failed: "
                    f"{error_class if isinstance(error_class, str) else 'unknown'}"
                )
            if (
                response.get("status") != "ok"
                or response.get("error_class") is not None
                or not isinstance(response.get("result"), Mapping)
            ):
                self._discard(terminate=True)
                raise RuntimeError(
                    "Round 74 AI worker session success response differs"
                )
            return _canonical_json(response["result"])

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            if connection is not None:
                try:
                    connection.send_bytes(b"")
                except (EOFError, OSError, ValueError):
                    pass
            self._discard(terminate=False)

    def __enter__(self) -> Round74AIWorkerSession:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _post_provider_json(
    url: str,
    payload: Mapping[str, object],
    timeout_seconds: float,
) -> object:
    if not url.startswith(f"{ROUND74_AI_WORKER_ENDPOINT}/"):
        raise ValueError("Round 74 AI runtime refused a non-loopback request")
    request = urllib_request.Request(
        url,
        data=_canonical_json(payload).encode("ascii"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "simple-ai-trading-round74-ai/0.1",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(  # nosec B310 - exact loopback endpoint
            request,
            timeout=timeout_seconds,
        ) as response:
            raw = response.read(ROUND74_AI_WORKER_MAXIMUM_RESPONSE_BYTES + 1)
    except (OSError, urllib_error.URLError) as exc:
        raise ValueError("Round 74 AI runtime provider request failed") from exc
    if len(raw) > ROUND74_AI_WORKER_MAXIMUM_RESPONSE_BYTES:
        raise ValueError("Round 74 AI runtime provider response is too large")
    try:

        def reject_duplicates(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            output: dict[str, object] = {}
            for key, item in pairs:
                if key in output:
                    raise ValueError(
                        "Round 74 AI runtime provider response has duplicate keys"
                    )
                output[key] = item
            return output

        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"Round 74 AI runtime provider response contains {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 74 AI runtime provider response is invalid") from exc
    return value


def unload_round74_ai_model(
    config: Round74AIRuntimeConfig,
    manifest: Round74AIModelManifest,
    *,
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
    provider_poster: ProviderPoster = _post_provider_json,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    timeout_seconds: float = 10.0,
) -> bool:
    """Unload only the exact declared model and verify that it left residency."""

    config.validate()
    manifest.validate()
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0.0 or timeout > 30.0:
        raise ValueError("Round 74 AI unload timeout differs")
    inspection_timeout = min(timeout, 2.0)
    current = residency_inspector(
        config.endpoint,
        config.model_name,
        inspection_timeout,
        expected_digest=manifest.model_artifact_sha256,
    ).validated()
    if current.status == "unloaded":
        return False
    if current.digest != manifest.model_artifact_sha256 or _normalized_model_reference(
        current.loaded_model
    ) != _normalized_model_reference(config.model_name):
        raise ValueError("Round 74 AI unload model identity differs")
    response = provider_poster(
        f"{config.endpoint}/api/generate",
        {
            "model": config.model_name,
            "keep_alive": 0,
            "stream": False,
        },
        timeout,
    )
    if not isinstance(response, Mapping) or response.get("done") is not True:
        raise ValueError("Round 74 AI unload response differs")
    deadline = monotonic() + timeout
    while True:
        current = residency_inspector(
            config.endpoint,
            config.model_name,
            inspection_timeout,
            expected_digest=manifest.model_artifact_sha256,
        ).validated()
        if current.status == "unloaded":
            return True
        if (
            current.digest != manifest.model_artifact_sha256
            or _normalized_model_reference(current.loaded_model)
            != _normalized_model_reference(config.model_name)
        ):
            raise ValueError("Round 74 AI unload residency identity drifted")
        if monotonic() >= deadline:
            raise TimeoutError("Round 74 AI declared model unload timed out")
        sleeper(0.25)


def preload_round74_ai_model(
    config: Round74AIRuntimeConfig,
    manifest: Round74AIModelManifest,
    *,
    capability_detector: CapabilityDetector = detect_ai_capabilities,
    provenance_resolver: ProvenanceResolver = resolve_ollama_model_provenance,
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
    provider_poster: ProviderPoster = _post_provider_json,
    timeout_seconds: float = ROUND74_AI_RUNTIME_PRELOAD_TIMEOUT_SECONDS,
) -> OllamaResidencyReport:
    """Cold-load one pinned model before reviews and prove full GPU residency."""

    config.validate()
    manifest.validate()
    timeout = float(timeout_seconds)
    if (
        not math.isfinite(timeout)
        or timeout <= 0.0
        or timeout > ROUND74_AI_RUNTIME_PRELOAD_TIMEOUT_SECONDS
    ):
        raise ValueError("Round 74 AI preload timeout differs")
    digest, metadata_sha256 = provenance_resolver(
        config.endpoint,
        config.model_name,
        min(timeout, 3.0),
    )
    if (
        digest != manifest.model_artifact_sha256
        or not _is_sha256(metadata_sha256)
    ):
        raise ValueError("Round 74 AI preload provenance differs")
    inspection_timeout = min(timeout, 2.0)
    current = residency_inspector(
        config.endpoint,
        config.model_name,
        inspection_timeout,
        expected_digest=digest,
    ).validated()
    if current.status != "unloaded":
        if (
            not current.fully_gpu_resident
            or current.digest != digest
            or _normalized_model_reference(current.loaded_model)
            != _normalized_model_reference(config.model_name)
        ):
            raise ValueError("Round 74 AI preload residency differs")
        return current
    capability_config = _capability_config(config, manifest)
    capability = capability_detector(capability_config)
    required_vram_gb = capability_config.min_free_vram_gb
    capability_messages = _provider_capability_messages(
        capability,
        minimum_free_vram_gb=required_vram_gb,
        exact_model_already_fully_gpu_resident=False,
    )
    if not capability.ok or capability_messages:
        raise ValueError("Round 74 AI preload capability gate failed")
    response = provider_poster(
        f"{config.endpoint}/api/generate",
        {
            "model": config.model_name,
            "keep_alive": "30m",
            "stream": False,
        },
        timeout,
    )
    if (
        not isinstance(response, Mapping)
        or response.get("done") is not True
        or _normalized_model_reference(response.get("model"))
        != _normalized_model_reference(config.model_name)
        or response.get("response") not in (None, "")
    ):
        raise ValueError("Round 74 AI preload response differs")
    loaded = residency_inspector(
        config.endpoint,
        config.model_name,
        inspection_timeout,
        expected_digest=digest,
    ).validated()
    if (
        not loaded.fully_gpu_resident
        or loaded.digest != digest
        or _normalized_model_reference(loaded.loaded_model)
        != _normalized_model_reference(config.model_name)
    ):
        raise ValueError("Round 74 AI preload full GPU residency differs")
    return loaded


def review_round74_ai_candidate(
    config: Round74AIRuntimeConfig,
    manifest: Round74AIModelManifest,
    request: Round74AIReviewRequest,
    *,
    deterministic_risk_gate_passed: bool,
    observed_wall_ns: int,
    capability_detector: CapabilityDetector = detect_ai_capabilities,
    provenance_resolver: ProvenanceResolver = (resolve_ollama_model_provenance),
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
    popen_factory: PopenFactory = subprocess.Popen,
    worker_command: Sequence[str] | None = None,
    worker_session: Round74AIWorkerSession | None = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    wall_time_ns: Callable[[], int] = time.time_ns,
) -> Round74AIRuntimeOutcome:
    """Run one review; all uncertainty returns zero approved entry risk."""

    config.validate()
    manifest.validate()
    request.validate()
    if (
        not isinstance(deterministic_risk_gate_passed, bool)
        or isinstance(observed_wall_ns, bool)
        or not isinstance(observed_wall_ns, int)
        or observed_wall_ns < request.requested_wall_ns
        or (
            worker_session is not None
            and (
                not isinstance(worker_session, Round74AIWorkerSession)
                or worker_command is not None
                or popen_factory is not subprocess.Popen
            )
        )
    ):
        raise ValueError("Round 74 AI runtime application context differs")
    started_ns = monotonic_ns()
    base = {
        "request": request,
        "manifest": manifest,
        "deterministic_risk_gate_passed": (deterministic_risk_gate_passed),
        "observed_wall_ns": observed_wall_ns,
        "started_ns": started_ns,
        "monotonic_ns": monotonic_ns,
    }
    if not deterministic_risk_gate_passed:
        return _outcome(
            status="blocked_deterministic_gate",
            capability=None,
            resolved_digest=None,
            resolved_metadata_sha256=None,
            worker_result=None,
            approved_risk_size_bps=0,
            failure_class="DeterministicRiskGate",
            message="The deterministic risk gate denied the candidate.",
            **base,
        )
    if observed_wall_ns > request.expires_wall_ns:
        return _outcome(
            status="blocked_expired",
            capability=None,
            resolved_digest=None,
            resolved_metadata_sha256=None,
            worker_result=None,
            approved_risk_size_bps=0,
            failure_class="ExpiredReviewRequest",
            message="The review request expired before inference.",
            **base,
        )
    cold_capability_config = _capability_config(config, manifest)
    cold_capability = capability_detector(cold_capability_config)
    required_vram_gb = cold_capability_config.min_free_vram_gb
    warm_residency: OllamaResidencyReport | None = None
    warm_residency_error_type: str | None = None
    cold_vram_headroom_passed = (
        cold_capability.free_vram_gb is not None
        and cold_capability.free_vram_gb >= required_vram_gb
    )
    try:
        warm_residency = residency_inspector(
            config.endpoint,
            config.model_name,
            min(config.timeout_seconds, 2.0),
            expected_digest=manifest.model_artifact_sha256,
        ).validated()
    except Exception as exc:  # noqa: BLE001 - gate converts to deny
        warm_residency_error_type = type(exc).__name__
    exact_model_already_fully_gpu_resident = bool(
        warm_residency is not None
        and warm_residency.fully_gpu_resident
        and warm_residency.digest == manifest.model_artifact_sha256
        and _normalized_model_reference(warm_residency.loaded_model)
        == _normalized_model_reference(config.model_name)
    )
    capability_config = _capability_config(
        config,
        manifest,
        exact_model_already_fully_gpu_resident=(exact_model_already_fully_gpu_resident),
    )
    capability = (
        capability_detector(capability_config)
        if exact_model_already_fully_gpu_resident
        else cold_capability
    )
    capability_messages = list(
        _provider_capability_messages(
            capability,
            minimum_free_vram_gb=required_vram_gb,
            exact_model_already_fully_gpu_resident=(
                exact_model_already_fully_gpu_resident
            ),
        )
    )
    warm_equivalent_preload_ram_gb: float | None = None
    warm_equivalent_preload_ram_headroom_passed = False
    if (
        exact_model_already_fully_gpu_resident
        and capability.free_ram_gb is not None
        and warm_residency is not None
        and warm_residency.size_bytes is not None
    ):
        warm_equivalent_preload_ram_gb = (
            capability.free_ram_gb + warm_residency.size_bytes / 1024**3
        )
        warm_equivalent_preload_ram_headroom_passed = (
            warm_equivalent_preload_ram_gb >= config.minimum_free_ram_gb
        )
        if not warm_equivalent_preload_ram_headroom_passed:
            capability_messages.append(
                "warm exact-model free RAM plus bound model size "
                f"{warm_equivalent_preload_ram_gb:.1f} GiB is below required "
                f"pre-load headroom {config.minimum_free_ram_gb:.1f} GiB"
            )
    if warm_residency_error_type is not None:
        capability_messages.append(
            "pre-inference provider residency could not be verified"
        )
    elif (
        warm_residency is not None
        and warm_residency.loaded
        and not (exact_model_already_fully_gpu_resident)
    ):
        capability_messages.append(
            "pre-inference provider residency is not the exact fully GPU-resident "
            "declared model"
        )
    capability_payload = {
        **capability.asdict(),
        "minimum_free_vram_gb": required_vram_gb,
        "minimum_cold_free_ram_gb": config.minimum_free_ram_gb,
        "minimum_warm_free_ram_gb": ROUND74_AI_RUNTIME_MINIMUM_WARM_FREE_RAM_GB,
        "pre_inference_cold_capability_passed": cold_capability.ok,
        "pre_inference_cold_load_headroom_passed": cold_vram_headroom_passed,
        "pre_inference_residency_check_required": True,
        "pre_inference_residency_check_performed": (
            warm_residency is not None and warm_residency_error_type is None
        ),
        "pre_inference_warm_residency_check_required": True,
        "pre_inference_warm_residency": (
            warm_residency.asdict() if warm_residency is not None else None
        ),
        "pre_inference_warm_residency_error_type": warm_residency_error_type,
        "pre_inference_exact_model_fully_gpu_resident": (
            exact_model_already_fully_gpu_resident
        ),
        "pre_inference_warm_ram_headroom_passed": bool(
            exact_model_already_fully_gpu_resident
            and capability.free_ram_gb is not None
            and capability.free_ram_gb >= ROUND74_AI_RUNTIME_MINIMUM_WARM_FREE_RAM_GB
        ),
        "pre_inference_warm_equivalent_preload_ram_gb": (
            warm_equivalent_preload_ram_gb
        ),
        "pre_inference_warm_equivalent_preload_ram_headroom_passed": (
            warm_equivalent_preload_ram_headroom_passed
        ),
        "provider_runtime_full_gpu_residency_required": True,
        "provider_runtime_full_gpu_residency_verified": False,
        "worker_process_mode": (
            "persistent_model_batch" if worker_session is not None else "one_shot"
        ),
        "worker_session_request_ordinal": (
            worker_session.request_count + 1 if worker_session is not None else None
        ),
        "worker_session_restart_count_before_request": (
            worker_session.restart_count if worker_session is not None else None
        ),
        "worker_session_restart_count_after_request": None,
    }
    if not capability.ok or capability_messages:
        return _outcome(
            status="blocked_capability",
            capability=capability_payload,
            resolved_digest=None,
            resolved_metadata_sha256=None,
            worker_result=None,
            approved_risk_size_bps=0,
            failure_class="AICapabilityGate",
            message="; ".join(capability_messages)
            or "The local AI capability gate failed.",
            **base,
        )
    current_wall_ns = max(observed_wall_ns, wall_time_ns())
    if current_wall_ns > request.expires_wall_ns:
        return _outcome(
            status="blocked_expired",
            capability=capability_payload,
            resolved_digest=None,
            resolved_metadata_sha256=None,
            worker_result=None,
            approved_risk_size_bps=0,
            failure_class="ExpiredReviewRequest",
            message="The review request expired during capability checks.",
            **{
                **base,
                "observed_wall_ns": current_wall_ns,
            },
        )
    try:
        digest, metadata_sha256 = provenance_resolver(
            config.endpoint,
            config.model_name,
            min(config.timeout_seconds, 3.0),
        )
        if digest != manifest.model_artifact_sha256 or not _is_sha256(metadata_sha256):
            raise ValueError("Pinned local model provenance differs")
    except Exception as exc:  # noqa: BLE001 - gate converts to deny
        return _outcome(
            status="blocked_provenance",
            capability=capability_payload,
            resolved_digest=None,
            resolved_metadata_sha256=None,
            worker_result=None,
            approved_risk_size_bps=0,
            failure_class=type(exc).__name__,
            message="Pinned local model provenance could not be verified.",
            **base,
        )
    current_wall_ns = max(current_wall_ns, wall_time_ns())
    if current_wall_ns > request.expires_wall_ns:
        return _outcome(
            status="blocked_expired",
            capability=capability_payload,
            resolved_digest=digest,
            resolved_metadata_sha256=metadata_sha256,
            worker_result=None,
            approved_risk_size_bps=0,
            failure_class="ExpiredReviewRequest",
            message="The review request expired during provenance checks.",
            **{
                **base,
                "observed_wall_ns": current_wall_ns,
            },
        )
    envelope = Round74AIWorkerEnvelope(
        model_name=config.model_name,
        endpoint=config.endpoint,
        timeout_seconds=config.timeout_seconds,
        model_manifest=manifest,
        review_request=request,
    )
    try:
        worker_timeout_seconds = min(
            config.timeout_seconds,
            (request.expires_wall_ns - current_wall_ns) / 1_000_000_000,
        )
        if worker_session is None:
            stdout, _ = _spawn_worker(
                worker_command or _default_worker_command(),
                envelope,
                timeout_seconds=worker_timeout_seconds,
                popen_factory=popen_factory,
            )
        else:
            stdout = worker_session.request(
                envelope,
                timeout_seconds=worker_timeout_seconds,
            )
            capability_payload["worker_session_restart_count_after_request"] = (
                worker_session.restart_count
            )
    except subprocess.TimeoutExpired:
        return _outcome(
            status="worker_timeout",
            capability=capability_payload,
            resolved_digest=digest,
            resolved_metadata_sha256=metadata_sha256,
            worker_result=None,
            approved_risk_size_bps=0,
            failure_class="TimeoutExpired",
            message="The isolated AI worker exceeded its deadline.",
            **base,
        )
    except RuntimeError as exc:
        return _outcome(
            status="worker_failed",
            capability=capability_payload,
            resolved_digest=digest,
            resolved_metadata_sha256=metadata_sha256,
            worker_result=None,
            approved_risk_size_bps=0,
            failure_class=type(exc).__name__,
            message=str(exc),
            **base,
        )
    except (OSError, ValueError) as exc:
        return _outcome(
            status="worker_output_invalid",
            capability=capability_payload,
            resolved_digest=digest,
            resolved_metadata_sha256=metadata_sha256,
            worker_result=None,
            approved_risk_size_bps=0,
            failure_class=type(exc).__name__,
            message="The isolated AI worker boundary failed.",
            **base,
        )
    current_wall_ns = max(current_wall_ns, wall_time_ns())
    if current_wall_ns > request.expires_wall_ns:
        return _outcome(
            status="blocked_expired",
            capability=capability_payload,
            resolved_digest=digest,
            resolved_metadata_sha256=metadata_sha256,
            worker_result=None,
            approved_risk_size_bps=0,
            failure_class="ExpiredReviewRequest",
            message="The review request expired before output validation.",
            **{
                **base,
                "observed_wall_ns": current_wall_ns,
            },
        )
    try:
        result = _strict_worker_result(stdout)
        if (
            result.envelope_sha256 != envelope.envelope_sha256
            or result.request_sha256 != request.request_sha256
            or result.manifest_sha256 != manifest.manifest_sha256
            or result.model_digest != digest
            or result.model_metadata_sha256 != metadata_sha256
        ):
            raise ValueError("Round 74 AI worker evidence binding differs")
        approved = apply_round74_ai_risk_modifier(
            request,
            result.decision,
            deterministic_risk_gate_passed=True,
            observed_wall_ns=current_wall_ns,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return _outcome(
            status="worker_output_invalid",
            capability=capability_payload,
            resolved_digest=digest,
            resolved_metadata_sha256=metadata_sha256,
            worker_result=None,
            approved_risk_size_bps=0,
            failure_class=type(exc).__name__,
            message="The isolated AI worker output failed validation.",
            **base,
        )
    return _outcome(
        status="accepted",
        capability={
            **capability_payload,
            "provider_runtime_full_gpu_residency_verified": True,
        },
        resolved_digest=digest,
        resolved_metadata_sha256=metadata_sha256,
        worker_result=result,
        approved_risk_size_bps=approved,
        failure_class=None,
        message="The isolated local AI review passed every gate.",
        **{
            **base,
            "observed_wall_ns": current_wall_ns,
        },
    )


__all__ = [
    "ROUND74_AI_RUNTIME_MINIMUM_FREE_RAM_GB",
    "ROUND74_AI_RUNTIME_MINIMUM_FREE_VRAM_GB",
    "ROUND74_AI_RUNTIME_MINIMUM_WARM_FREE_RAM_GB",
    "ROUND74_AI_RUNTIME_OUTCOME_SCHEMA_VERSION",
    "ROUND74_AI_RUNTIME_PRELOAD_TIMEOUT_SECONDS",
    "ROUND74_AI_RUNTIME_STATUSES",
    "Round74AIRuntimeConfig",
    "Round74AIRuntimeOutcome",
    "Round74AIWorkerSession",
    "preload_round74_ai_model",
    "review_round74_ai_candidate",
    "unload_round74_ai_model",
]
