"""Exact local-model qualification for the Round 28 AI risk veto."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

from .ai_runtime import OllamaResidencyReport, inspect_ollama_model_residency
from .polymarket_round28_ai_contract import (
    POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
    load_round28_ai_contract,
    validate_round28_ai_contract,
)


POLYMARKET_ROUND28_AI_BASE_URL = "http://127.0.0.1:11434"
POLYMARKET_ROUND28_AI_HOST_REPORT_SCHEMA_VERSION = (
    "polymarket-round28-ai-host-qualification-v1"
)
POLYMARKET_ROUND28_AI_ARTIFACT_REPORT_SCHEMA_VERSION = (
    "polymarket-round28-ai-artifact-verification-v1"
)
POLYMARKET_ROUND28_AI_RESPONSE_SCHEMA_VERSION = (
    "polymarket-round28-ai-risk-veto-response-v1"
)
POLYMARKET_ROUND28_AI_COLD_LIMIT_SECONDS = 30.0
POLYMARKET_ROUND28_AI_WARM_LIMIT_SECONDS = 5.0
_MAXIMUM_RESPONSE_BYTES = 2_000_000
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


JsonPoster = Callable[[str, Mapping[str, object], float], object]
ResidencyInspector = Callable[..., OllamaResidencyReport]


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


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 28 AI {name} SHA-256 differs")
    return selected


def _strict_json_value(text: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate key: {key}")
            output[key] = value
        return output

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
    if not url.startswith(f"{POLYMARKET_ROUND28_AI_BASE_URL}/"):
        raise ValueError("Round 28 AI provider must be loopback-only")
    body = _canonical_json(payload).encode("ascii")
    request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(  # nosec B310 - fixed loopback URL
            request,
            timeout=timeout_seconds,
        ) as response:
            raw = response.read(_MAXIMUM_RESPONSE_BYTES + 1)
    except (OSError, urllib_error.URLError) as exc:
        raise RuntimeError("Round 28 local AI provider request failed") from exc
    if len(raw) > _MAXIMUM_RESPONSE_BYTES:
        raise ValueError("Round 28 AI provider response exceeds its limit")
    try:
        return _strict_json_value(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 28 AI provider returned invalid JSON") from exc


@dataclass(frozen=True, slots=True)
class Round28AIHostCandidate:
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
    contract_host_qualification: str

    def validated(self) -> "Round28AIHostCandidate":
        if (
            not self.model_id
            or not self.runtime_model
            or any(
                len(value) != 64 or set(value) - _SHA256_CHARACTERS
                for value in (self.runtime_digest, self.artifact_sha256)
            )
            or len(self.upstream_revision) != 40
            or set(self.upstream_revision) - _SHA256_CHARACTERS
            or not self.role
            or not self.quantization
            or not self.artifact_source
            or not self.artifact_revision
            or self.artifact_size_bytes <= 0
            or self.contract_host_qualification
            not in {
                "passed_round27_exact_artifact",
                "pending_exact_artifact_download_and_amd_gpu_probe",
            }
        ):
            raise ValueError("Round 28 AI host candidate differs")
        return self


def round28_ai_candidate_from_contract(
    contract: Mapping[str, object],
    *,
    model_id: str,
    observed_runtime_digest: str | None = None,
) -> Round28AIHostCandidate:
    """Bind one preregistered candidate to an exact local runtime digest."""

    selected = validate_round28_ai_contract(contract)
    raw_candidates = selected["candidate_program"]
    matches = tuple(
        item
        for item in raw_candidates
        if isinstance(item, Mapping) and item.get("model_id") == model_id
    )
    if len(matches) != 1:
        raise ValueError("Round 28 AI candidate is not preregistered")
    raw = matches[0]
    registered_digest = raw.get("runtime_digest")
    if registered_digest is None:
        runtime_digest = _sha256(
            observed_runtime_digest,
            name="observed runtime digest",
        )
    else:
        runtime_digest = _sha256(registered_digest, name="runtime digest")
        if observed_runtime_digest is not None and (
            _sha256(observed_runtime_digest, name="observed runtime digest")
            != runtime_digest
        ):
            raise ValueError("Round 28 AI runtime digest differs from contract")
    return Round28AIHostCandidate(
        model_id=str(raw["model_id"]),
        runtime_model=str(raw["runtime_model"]),
        runtime_digest=runtime_digest,
        upstream_revision=str(raw["upstream_revision"]),
        role=str(raw["role"]),
        quantization=str(raw["quantization"]),
        artifact_source=str(raw["artifact_source"]),
        artifact_revision=str(raw["artifact_revision"]),
        artifact_sha256=str(raw["artifact_sha256"]),
        artifact_size_bytes=int(raw["artifact_size_bytes"]),
        contract_host_qualification=str(raw["host_qualification"]),
    ).validated()


def round28_ai_response_schema() -> dict[str, object]:
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
                "minItems": 1,
                "maxItems": 3,
                "uniqueItems": True,
            },
        },
    }


def round28_ai_conformance_request(
    candidate: Round28AIHostCandidate,
    *,
    keep_alive: str,
) -> dict[str, object]:
    selected = candidate.validated()
    if keep_alive not in {"30s", "0"}:
        raise ValueError("Round 28 AI keep-alive differs")
    return {
        "model": selected.runtime_model,
        "prompt": (
            "Runtime conformance probe only. No market, account, position, "
            "credential, or order data is present. Required Binance BBO state "
            "is absent. Return decision reject and reason_codes containing "
            "bbo_source_stale_or_gapped. Return only JSON."
        ),
        "stream": False,
        "format": round28_ai_response_schema(),
        "keep_alive": keep_alive,
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 28,
            "num_ctx": 8192,
            "num_predict": 96,
        },
    }


def round28_ai_unload_request(
    candidate: Round28AIHostCandidate,
) -> dict[str, object]:
    return {
        "model": candidate.validated().runtime_model,
        "keep_alive": 0,
        "stream": False,
    }


def build_round28_ai_artifact_verification(
    candidate: Round28AIHostCandidate,
    *,
    observed_sha256: str,
    observed_size_bytes: int,
    verification_method: str,
    source_evidence_sha256: str,
    observed_at_ms: int,
) -> dict[str, object]:
    """Bind a streamed file/blob verification to its preregistered artifact."""

    selected = candidate.validated()
    observed_sha = _sha256(observed_sha256, name="observed artifact")
    source_evidence = _sha256(source_evidence_sha256, name="source evidence")
    if (
        observed_sha != selected.artifact_sha256
        or type(observed_size_bytes) is not int
        or observed_size_bytes != selected.artifact_size_bytes
        or verification_method
        not in {
            "file_sha256_stream",
            "ollama_blob_sha256_and_size",
            "inherited_round27_exact_artifact",
        }
        or type(observed_at_ms) is not int
        or observed_at_ms <= 0
    ):
        raise ValueError("Round 28 AI artifact verification differs")
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_AI_ARTIFACT_REPORT_SCHEMA_VERSION,
        "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
        "model_id": selected.model_id,
        "artifact_source": selected.artifact_source,
        "artifact_revision": selected.artifact_revision,
        "expected_sha256": selected.artifact_sha256,
        "observed_sha256": observed_sha,
        "expected_size_bytes": selected.artifact_size_bytes,
        "observed_size_bytes": observed_size_bytes,
        "verification_method": verification_method,
        "source_evidence_sha256": source_evidence,
        "observed_at_ms": observed_at_ms,
        "credentials_used": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


def validate_round28_ai_artifact_verification(
    value: Mapping[str, object],
    *,
    candidate: Round28AIHostCandidate,
) -> dict[str, object]:
    selected = candidate.validated()
    report = dict(value)
    claimed = _sha256(report.pop("report_sha256", None), name="artifact report")
    if (
        claimed != _canonical_sha256(report)
        or report.get("schema_version")
        != POLYMARKET_ROUND28_AI_ARTIFACT_REPORT_SCHEMA_VERSION
        or report.get("ai_contract_sha256")
        != POLYMARKET_ROUND28_AI_CONTRACT_SHA256
        or report.get("model_id") != selected.model_id
        or report.get("artifact_source") != selected.artifact_source
        or report.get("artifact_revision") != selected.artifact_revision
        or report.get("expected_sha256") != selected.artifact_sha256
        or report.get("observed_sha256") != selected.artifact_sha256
        or report.get("expected_size_bytes") != selected.artifact_size_bytes
        or report.get("observed_size_bytes") != selected.artifact_size_bytes
        or report.get("verification_method")
        not in {
            "file_sha256_stream",
            "ollama_blob_sha256_and_size",
            "inherited_round27_exact_artifact",
        }
        or type(report.get("observed_at_ms")) is not int
        or int(report["observed_at_ms"]) <= 0
        or _sha256(report.get("source_evidence_sha256"), name="source evidence")
        != report.get("source_evidence_sha256")
        or any(
            report.get(field) is not False
            for field in ("credentials_used", "orders_submitted", "trading_authority")
        )
    ):
        raise ValueError("Round 28 AI artifact report differs")
    return {**report, "report_sha256": claimed}


def _measurement(
    raw: object,
    *,
    candidate: Round28AIHostCandidate,
    phase: str,
    wall_ms: int,
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("Round 28 AI conformance response is not an object")
    required = {
        "model",
        "response",
        "done",
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "eval_count",
    }
    if not required <= set(raw):
        raise ValueError("Round 28 AI conformance response is incomplete")
    parsed = (
        _strict_json_value(raw["response"])
        if isinstance(raw.get("response"), str)
        else None
    )
    if (
        raw.get("model") != candidate.runtime_model
        or raw.get("done") is not True
        or raw.get("done_reason") != "stop"
        or parsed
        != {
            "decision": "reject",
            "reason_codes": ["bbo_source_stale_or_gapped"],
        }
    ):
        raise ValueError("Round 28 AI conformance decision differs")
    values = {
        "provider_total_ms": math.ceil(float(raw["total_duration"]) / 1_000_000),
        "provider_load_ms": math.ceil(float(raw["load_duration"]) / 1_000_000),
        "prompt_tokens": raw["prompt_eval_count"],
        "output_tokens": raw["eval_count"],
    }
    if (
        wall_ms <= 0
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in (
                raw["total_duration"],
                raw["load_duration"],
                raw["prompt_eval_count"],
                raw["eval_count"],
            )
        )
        or values["prompt_tokens"] <= 0
        or values["output_tokens"] <= 0
        or values["provider_load_ms"] > values["provider_total_ms"]
    ):
        raise ValueError("Round 28 AI conformance timing differs")
    return {
        "phase": phase,
        "wall_ms": wall_ms,
        **values,
        "response": parsed,
    }


def probe_round28_ai_candidate_host(
    candidate: Round28AIHostCandidate,
    *,
    artifact_verification: Mapping[str, object],
    post_json: JsonPoster = _post_json,
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> dict[str, object]:
    """Measure cold/warm inference and exact full-GPU residency, then unload."""

    selected = candidate.validated()
    verified_artifact = validate_round28_ai_artifact_verification(
        artifact_verification,
        candidate=selected,
    )
    endpoint = f"{POLYMARKET_ROUND28_AI_BASE_URL}/api/generate"
    post_json(endpoint, round28_ai_unload_request(selected), 5.0)
    measurements: list[dict[str, object]] = []
    residency: OllamaResidencyReport | None = None
    unload_failure: dict[str, str] | None = None
    unloaded = False
    try:
        for phase in ("cold", "warm"):
            started = monotonic_ns()
            raw = post_json(
                endpoint,
                round28_ai_conformance_request(selected, keep_alive="30s"),
                POLYMARKET_ROUND28_AI_COLD_LIMIT_SECONDS + 5.0,
            )
            wall_ms = max(
                1,
                math.ceil((monotonic_ns() - started) / 1_000_000),
            )
            measurements.append(
                _measurement(
                    raw,
                    candidate=selected,
                    phase=phase,
                    wall_ms=wall_ms,
                )
            )
        residency = residency_inspector(
            POLYMARKET_ROUND28_AI_BASE_URL,
            selected.runtime_model,
            2.0,
            expected_digest=selected.runtime_digest,
        ).validated()
    finally:
        try:
            post_json(endpoint, round28_ai_unload_request(selected), 5.0)
            after = residency_inspector(
                POLYMARKET_ROUND28_AI_BASE_URL,
                selected.runtime_model,
                2.0,
                expected_digest=selected.runtime_digest,
            ).validated()
            unloaded = after.status == "unloaded"
            if not unloaded:
                raise RuntimeError("Round 28 AI model remained loaded")
        except Exception as exc:  # noqa: BLE001 - cleanup failure is evidence
            unload_failure = {
                "type": type(exc).__name__,
                "message": str(exc)[:240],
            }
    if len(measurements) != 2 or residency is None:
        raise RuntimeError("Round 28 AI host probe did not complete")
    cold, warm = measurements
    checks = {
        "artifact_verification_bound": True,
        "exact_runtime_digest": residency.digest == selected.runtime_digest,
        "full_gpu_residency": residency.fully_gpu_resident,
        "cold_structured_response": cold["response"]
        == {
            "decision": "reject",
            "reason_codes": ["bbo_source_stale_or_gapped"],
        },
        "warm_structured_response": warm["response"]
        == {
            "decision": "reject",
            "reason_codes": ["bbo_source_stale_or_gapped"],
        },
        "cold_within_limit": int(cold["wall_ms"])
        <= math.ceil(POLYMARKET_ROUND28_AI_COLD_LIMIT_SECONDS * 1_000),
        "warm_within_limit": int(warm["wall_ms"])
        <= math.ceil(POLYMARKET_ROUND28_AI_WARM_LIMIT_SECONDS * 1_000),
        "provider_unloaded_after_probe": unloaded and unload_failure is None,
        "target_outcome_and_market_data_absent": True,
        "credentials_and_trading_authority_absent": True,
    }
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_AI_HOST_REPORT_SCHEMA_VERSION,
        "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
        "candidate": asdict(selected),
        "artifact_verification": verified_artifact,
        "artifact_verification_sha256": verified_artifact["report_sha256"],
        "measurements": measurements,
        "residency": residency.asdict(),
        "checks": checks,
        "passed": all(checks.values()),
        "unload_failure": unload_failure,
        "edge_claim": False,
        "profitability_claim": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


def validate_round28_ai_host_report(
    value: Mapping[str, object],
    *,
    contract: Mapping[str, object],
) -> tuple[dict[str, object], Round28AIHostCandidate]:
    report = dict(value)
    claimed = _sha256(report.pop("report_sha256", None), name="host report")
    raw_candidate = report.get("candidate")
    checks = report.get("checks")
    residency = report.get("residency")
    artifact = report.get("artifact_verification")
    if not isinstance(raw_candidate, Mapping):
        raise ValueError("Round 28 AI host candidate evidence differs")
    candidate = round28_ai_candidate_from_contract(
        contract,
        model_id=str(raw_candidate.get("model_id", "")),
        observed_runtime_digest=str(raw_candidate.get("runtime_digest", "")),
    )
    if (
        claimed != _canonical_sha256(report)
        or report.get("schema_version")
        != POLYMARKET_ROUND28_AI_HOST_REPORT_SCHEMA_VERSION
        or report.get("ai_contract_sha256")
        != POLYMARKET_ROUND28_AI_CONTRACT_SHA256
        or raw_candidate != asdict(candidate)
        or not isinstance(artifact, Mapping)
        or validate_round28_ai_artifact_verification(
            artifact,
            candidate=candidate,
        )["report_sha256"]
        != report.get("artifact_verification_sha256")
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
        or report.get("passed") is not True
        or not isinstance(residency, Mapping)
        or residency.get("digest") != candidate.runtime_digest
        or residency.get("gpu_resident") is not True
        or float(residency.get("vram_to_model_ratio", 0.0)) < 0.999
        or report.get("unload_failure") is not None
        or any(
            report.get(field) is not False
            for field in (
                "edge_claim",
                "profitability_claim",
                "orders_submitted",
                "trading_authority",
            )
        )
    ):
        raise ValueError("Round 28 AI host report differs")
    return {**report, "report_sha256": claimed}, candidate


def load_default_round28_ai_contract(repository: str) -> dict[str, object]:
    """Small convenience used by host-only tools without importing operators."""

    return load_round28_ai_contract(repository)


__all__ = [
    "POLYMARKET_ROUND28_AI_ARTIFACT_REPORT_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_AI_BASE_URL",
    "POLYMARKET_ROUND28_AI_HOST_REPORT_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_AI_RESPONSE_SCHEMA_VERSION",
    "Round28AIHostCandidate",
    "build_round28_ai_artifact_verification",
    "probe_round28_ai_candidate_host",
    "round28_ai_candidate_from_contract",
    "round28_ai_conformance_request",
    "round28_ai_response_schema",
    "round28_ai_unload_request",
    "validate_round28_ai_artifact_verification",
    "validate_round28_ai_host_report",
]
