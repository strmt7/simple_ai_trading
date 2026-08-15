"""Target-free host qualification for the Round 27 AI risk veto."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from .ai_runtime import OllamaResidencyReport, inspect_ollama_model_residency


POLYMARKET_ROUND27_MODEL_CONTRACT_SHA256 = (
    "3e18856b1f526655a514fd524378a92a878c6ec0a1857772d503b9bd7e77d439"
)
POLYMARKET_ROUND27_AI_HOST_PROBE_SCHEMA_VERSION = (
    "polymarket-round27-ai-host-qualification-v1"
)
POLYMARKET_ROUND27_AI_RESPONSE_SCHEMA_VERSION = (
    "polymarket-round27-ai-risk-veto-response-v1"
)
POLYMARKET_ROUND27_AI_BASE_URL = "http://127.0.0.1:11434"
POLYMARKET_ROUND27_AI_COLD_LIMIT_SECONDS = 30.0
POLYMARKET_ROUND27_AI_WARM_LIMIT_SECONDS = 5.0
_MAXIMUM_RESPONSE_BYTES = 2_000_000


JsonPoster = Callable[[str, Mapping[str, object], float], object]
ResidencyInspector = Callable[..., OllamaResidencyReport]


@dataclass(frozen=True, slots=True)
class Round27AIHostCandidate:
    model_id: str
    runtime_model: str
    runtime_digest: str
    upstream_revision: str
    role: str
    quantization: str
    artifact_source: str
    artifact_revision: str
    artifact_sha256: str
    artifact_size_bytes: int

    def __post_init__(self) -> None:
        hashes = (self.runtime_digest, self.artifact_sha256)
        if (
            not self.model_id
            or not self.runtime_model
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
            or len(self.upstream_revision) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.upstream_revision
            )
            or self.role
            not in {
                "general_reasoning_risk_review_control",
                "finance_specialized_risk_review_challenger",
            }
            or not self.quantization
            or not self.artifact_source
            or not self.artifact_revision
            or self.artifact_size_bytes <= 0
        ):
            raise ValueError("Round 27 AI host candidate specification is invalid")


POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE = Round27AIHostCandidate(
    model_id="Qwen/Qwen3.5-9B",
    runtime_model="qwen3.5:9b",
    runtime_digest=(
        "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
    ),
    upstream_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    role="general_reasoning_risk_review_control",
    quantization="Q4_K_M",
    artifact_source="ollama-library/qwen3.5:9b",
    artifact_revision=(
        "sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"
    ),
    artifact_sha256=(
        "dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"
    ),
    artifact_size_bytes=6_594_462_816,
)
POLYMARKET_ROUND27_ODA_HOST_CANDIDATE = Round27AIHostCandidate(
    model_id="OpenDataArena/ODA-Fin-SFT-8B",
    runtime_model="hf.co/mradermacher/ODA-Fin-SFT-8B-GGUF:Q6_K",
    runtime_digest=(
        "7f310e6fa4537b88432260aab5f7be68de819df5a3c94df4ee26d41d0c593a5b"
    ),
    upstream_revision="66940100ebc647846cdba7e7a4e15b94c1ab13ef",
    role="finance_specialized_risk_review_challenger",
    quantization="Q6_K",
    artifact_source="mradermacher/ODA-Fin-SFT-8B-GGUF",
    artifact_revision="b7af28162eb67d771edeeb77067141d3bbdcab04",
    artifact_sha256=(
        "7053bcade78e8698627715df9315e87cc1f880c24e73f01dc09f4d7b7fec4dc3"
    ),
    artifact_size_bytes=6_725_900_384,
)
POLYMARKET_ROUND27_AI_HOST_CANDIDATES = (
    POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE,
    POLYMARKET_ROUND27_ODA_HOST_CANDIDATE,
)


def _strict_json_value(text: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite constant: {value}")
        ),
    )


def _post_json(
    url: str,
    payload: Mapping[str, object],
    timeout_seconds: float,
) -> object:
    if not url.startswith(f"{POLYMARKET_ROUND27_AI_BASE_URL}/"):
        raise ValueError("Round 27 AI provider must be loopback-only")
    body = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(  # nosec B310 - URL is fixed to loopback
            request,
            timeout=timeout_seconds,
        ) as response:
            raw = response.read(_MAXIMUM_RESPONSE_BYTES + 1)
    except (OSError, urllib_error.URLError) as exc:
        raise RuntimeError("Round 27 local AI provider request failed") from exc
    if len(raw) > _MAXIMUM_RESPONSE_BYTES:
        raise ValueError("Round 27 AI provider response exceeds its limit")
    try:
        return _strict_json_value(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 27 AI provider returned invalid JSON") from exc


def round27_ai_response_schema() -> dict[str, object]:
    """Return the immutable risk-reducing output schema."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "reason_codes"],
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["reject", "reduce", "unchanged"],
            },
            "reason_codes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
            },
        },
    }


def round27_ai_conformance_request(
    candidate: Round27AIHostCandidate = POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE,
    *,
    keep_alive: str,
) -> dict[str, object]:
    """Build a target-free request that cannot authorize or enlarge risk."""

    if keep_alive not in {"30s", "0"}:
        raise ValueError("Round 27 AI keep-alive differs")
    if candidate not in POLYMARKET_ROUND27_AI_HOST_CANDIDATES:
        raise ValueError("Round 27 AI host candidate differs")
    return {
        "model": candidate.runtime_model,
        "prompt": (
            "Runtime conformance probe only. No market data, target, outcome, "
            "position, order, credential, or trading recommendation is present. "
            "The candidate lacks a verified liquidity input. Return decision "
            "reject and reason_codes containing missing_liquidity. Do not explain."
        ),
        "stream": False,
        "format": round27_ai_response_schema(),
        "keep_alive": keep_alive,
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 27,
            "num_ctx": 2048,
            "num_predict": 96,
        },
    }


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Round 27 AI {field} is not numeric")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"Round 27 AI {field} is invalid")
    return selected


@dataclass(frozen=True, slots=True)
class Round27AIInferenceMeasurement:
    phase: str
    response: dict[str, object]
    wall_seconds: float
    total_seconds: float
    load_seconds: float
    prompt_tokens: int
    prompt_tokens_per_second: float | None
    output_tokens: int
    output_tokens_per_second: float | None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _parse_measurement(
    raw: object,
    *,
    candidate: Round27AIHostCandidate,
    phase: str,
    wall_seconds: float,
) -> Round27AIInferenceMeasurement:
    if not isinstance(raw, Mapping):
        raise ValueError("Round 27 AI inference response is not an object")
    required = {
        "model",
        "response",
        "done",
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    }
    if not required.issubset(raw):
        raise ValueError("Round 27 AI inference response fields are incomplete")
    if (
        raw["model"] != candidate.runtime_model
        or raw["done"] is not True
        or raw["done_reason"] != "stop"
        or not isinstance(raw["response"], str)
    ):
        raise ValueError("Round 27 AI inference completion differs")
    parsed = _strict_json_value(raw["response"])
    if (
        not isinstance(parsed, Mapping)
        or set(parsed) != {"decision", "reason_codes"}
        or parsed["decision"] != "reject"
        or parsed["reason_codes"] != ["missing_liquidity"]
    ):
        raise ValueError("Round 27 AI conformance decision differs")

    total_ns = _finite_nonnegative(raw["total_duration"], field="total duration")
    load_ns = _finite_nonnegative(raw["load_duration"], field="load duration")
    prompt_ns = _finite_nonnegative(
        raw["prompt_eval_duration"],
        field="prompt duration",
    )
    output_ns = _finite_nonnegative(raw["eval_duration"], field="output duration")
    prompt_count = raw["prompt_eval_count"]
    output_count = raw["eval_count"]
    if (
        isinstance(prompt_count, bool)
        or not isinstance(prompt_count, int)
        or prompt_count <= 0
        or isinstance(output_count, bool)
        or not isinstance(output_count, int)
        or output_count <= 0
        or load_ns > total_ns
    ):
        raise ValueError("Round 27 AI token or duration evidence differs")
    return Round27AIInferenceMeasurement(
        phase=phase,
        response=dict(parsed),
        wall_seconds=wall_seconds,
        total_seconds=total_ns / 1_000_000_000,
        load_seconds=load_ns / 1_000_000_000,
        prompt_tokens=prompt_count,
        prompt_tokens_per_second=(
            None if prompt_ns == 0.0 else prompt_count / (prompt_ns / 1_000_000_000)
        ),
        output_tokens=output_count,
        output_tokens_per_second=(
            None if output_ns == 0.0 else output_count / (output_ns / 1_000_000_000)
        ),
    )


def _unload_request(candidate: Round27AIHostCandidate) -> dict[str, object]:
    return {
        "model": candidate.runtime_model,
        "keep_alive": 0,
        "stream": False,
    }


def probe_round27_ai_candidate_host(
    candidate: Round27AIHostCandidate,
    *,
    post_json: JsonPoster = _post_json,
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> dict[str, object]:
    """Measure exact cold/warm structured inference and GPU residency."""

    if candidate not in POLYMARKET_ROUND27_AI_HOST_CANDIDATES:
        raise ValueError("Round 27 AI host candidate differs")
    endpoint = f"{POLYMARKET_ROUND27_AI_BASE_URL}/api/generate"
    post_json(endpoint, _unload_request(candidate), 5.0)
    measurements: list[Round27AIInferenceMeasurement] = []
    unload_failure: dict[str, str] | None = None
    post_unload_residency: OllamaResidencyReport | None = None
    try:
        for phase in ("cold", "warm"):
            started = monotonic_ns()
            raw = post_json(
                endpoint,
                round27_ai_conformance_request(candidate, keep_alive="30s"),
                POLYMARKET_ROUND27_AI_COLD_LIMIT_SECONDS + 5.0,
            )
            elapsed = (monotonic_ns() - started) / 1_000_000_000
            measurements.append(
                _parse_measurement(
                    raw,
                    candidate=candidate,
                    phase=phase,
                    wall_seconds=elapsed,
                )
            )
        residency = residency_inspector(
            POLYMARKET_ROUND27_AI_BASE_URL,
            candidate.runtime_model,
            2.0,
            expected_digest=candidate.runtime_digest,
        ).validated()
    finally:
        try:
            post_json(endpoint, _unload_request(candidate), 5.0)
            post_unload_residency = residency_inspector(
                POLYMARKET_ROUND27_AI_BASE_URL,
                candidate.runtime_model,
                2.0,
                expected_digest=candidate.runtime_digest,
            ).validated()
            if post_unload_residency.status != "unloaded":
                raise RuntimeError("Round 27 AI model remained loaded after cleanup")
        except Exception as exc:  # noqa: BLE001 - cleanup failure is evidence
            unload_failure = {
                "type": type(exc).__name__,
                "message": str(exc)[:240],
            }
    if len(measurements) != 2:
        raise RuntimeError("Round 27 AI host probe did not complete both phases")
    cold, warm = measurements
    checks = {
        "exact_model_digest": residency.digest == candidate.runtime_digest,
        "full_gpu_residency": residency.fully_gpu_resident,
        "cold_structured_response": cold.response
        == {"decision": "reject", "reason_codes": ["missing_liquidity"]},
        "warm_structured_response": warm.response
        == {"decision": "reject", "reason_codes": ["missing_liquidity"]},
        "cold_within_host_qualification_limit": cold.wall_seconds
        <= POLYMARKET_ROUND27_AI_COLD_LIMIT_SECONDS,
        "warm_within_host_qualification_limit": warm.wall_seconds
        <= POLYMARKET_ROUND27_AI_WARM_LIMIT_SECONDS,
        "think_channel_disabled": True,
        "provider_unloaded_after_probe": bool(
            unload_failure is None
            and post_unload_residency is not None
            and post_unload_residency.status == "unloaded"
        ),
        "target_outcome_and_market_data_absent": True,
        "credentials_and_trading_authority_absent": True,
    }
    return {
        "candidate": {
            **asdict(candidate),
            "maximum_authority": "veto_or_reduce",
            "response_schema_version": (
                POLYMARKET_ROUND27_AI_RESPONSE_SCHEMA_VERSION
            ),
        },
        "measurements": [measurement.asdict() for measurement in measurements],
        "residency": residency.asdict(),
        "checks": checks,
        "passed": all(checks.values()),
        "unload_failure": unload_failure,
        "post_unload_residency": (
            None
            if post_unload_residency is None
            else post_unload_residency.asdict()
        ),
        "claims": {
            "host_runtime_qualified": all(checks.values()),
            "offline_matched_ablation_eligible": all(checks.values()),
            "latency_critical_probability_predictor": False,
            "predictive_uplift": False,
            "after_cost_uplift": False,
            "edge": False,
            "profitability": False,
            "live_trading_authority": False,
        },
    }


def probe_round27_qwen_host(
    **kwargs: Any,
) -> dict[str, object]:
    """Qualify the exact Qwen control artifact."""

    return probe_round27_ai_candidate_host(
        POLYMARKET_ROUND27_QWEN_HOST_CANDIDATE,
        **kwargs,
    )


def probe_round27_oda_host(
    **kwargs: Any,
) -> dict[str, object]:
    """Qualify the exact ODA finance challenger artifact."""

    return probe_round27_ai_candidate_host(
        POLYMARKET_ROUND27_ODA_HOST_CANDIDATE,
        **kwargs,
    )
