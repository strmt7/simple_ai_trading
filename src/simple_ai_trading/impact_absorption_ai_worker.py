"""Isolated local-Ollama worker for constrained Round 74 AI review."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from multiprocessing.connection import Connection
import sys
from urllib import error as urllib_error
from urllib import request as urllib_request

from .ai_review import resolve_ollama_model_provenance
from .ai_runtime import (
    OllamaResidencyReport,
    inspect_ollama_model_residency,
    ollama_residency_from_mapping,
)
from .impact_absorption_ai_protocol import (
    ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION,
    ROUND74_AI_REVIEW_REASON_CODES,
    ROUND74_AI_REVIEW_VERDICTS,
    Round74AIModelManifest,
    Round74AIReviewDecision,
    Round74AIReviewRequest,
    build_round74_ai_review_prompt,
)


ROUND74_AI_WORKER_ENVELOPE_SCHEMA_VERSION = "round-074-ai-worker-envelope-v1"
ROUND74_AI_WORKER_RESULT_SCHEMA_VERSION = "round-074-ai-worker-result-v1"
ROUND74_AI_WORKER_SESSION_RESPONSE_SCHEMA_VERSION = (
    "round-074-ai-worker-session-response-v1"
)
ROUND74_AI_WORKER_ENDPOINT = "http://127.0.0.1:11434"
ROUND74_AI_WORKER_CONTEXT_TOKENS = 4096
ROUND74_AI_WORKER_MAXIMUM_INPUT_BYTES = 1_000_000
ROUND74_AI_WORKER_MAXIMUM_RESPONSE_BYTES = 1_000_000
ROUND74_AI_WORKER_MAXIMUM_TIMEOUT_SECONDS = 25.0
ROUND74_AI_WORKER_KEEP_ALIVE = "2m"
ROUND74_AI_WORKER_MAXIMUM_OUTPUT_TOKENS = 256

PostJson = Callable[[str, Mapping[str, object], float], object]
ProvenanceResolver = Callable[[str, str, float], tuple[str, str]]
ResidencyInspector = Callable[
    [str, str, float, str],
    OllamaResidencyReport,
]


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


def _require_sha256(value: object, label: str) -> str:
    selected = str(value)
    if len(selected) != 64 or any(
        character not in "0123456789abcdef" for character in selected
    ):
        raise ValueError(f"Round 74 AI worker {label} digest differs")
    return selected


def _strict_json_object(raw_text: str, *, label: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} contains duplicate JSON keys")
            output[key] = value
        return output

    parsed = json.loads(
        raw_text,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"{label} contains {value}")
        ),
    )
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} root differs")
    return parsed


def _positive_timeout(value: object) -> float:
    selected = float(value)
    if (
        not math.isfinite(selected)
        or selected <= 0.0
        or selected > ROUND74_AI_WORKER_MAXIMUM_TIMEOUT_SECONDS
    ):
        raise ValueError("Round 74 AI worker timeout differs")
    return selected


def _validate_model_name(value: object) -> str:
    selected = str(value).strip()
    if (
        not selected
        or len(selected) > 160
        or any(character.isspace() for character in selected)
        or any(character in selected for character in ("/", "\\", "?"))
    ):
        raise ValueError("Round 74 AI worker model name differs")
    return selected


def _validate_endpoint(value: object) -> str:
    selected = str(value).rstrip("/")
    if selected != ROUND74_AI_WORKER_ENDPOINT:
        raise ValueError("Round 74 AI worker endpoint is not loopback")
    return selected


@dataclass(frozen=True)
class Round74AIWorkerEnvelope:
    """Strict process input binding one request to one local model artifact."""

    model_name: str
    endpoint: str
    timeout_seconds: float
    model_manifest: Round74AIModelManifest
    review_request: Round74AIReviewRequest
    schema_version: str = ROUND74_AI_WORKER_ENVELOPE_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_AI_WORKER_ENVELOPE_SCHEMA_VERSION
            or self.model_manifest.model_artifact_kind != "ollama_manifest"
        ):
            raise ValueError("Round 74 AI worker envelope differs")
        _validate_model_name(self.model_name)
        _validate_endpoint(self.endpoint)
        _positive_timeout(self.timeout_seconds)
        self.model_manifest.validate()
        self.review_request.validate()

    @property
    def envelope_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "endpoint": self.endpoint,
            "timeout_seconds": self.timeout_seconds,
            "model_manifest": self.model_manifest.as_dict(),
            "review_request": self.review_request.as_dict(),
            "remote_inference_permitted": False,
            "execution_authority": False,
        }
        if include_sha256:
            payload["envelope_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74AIWorkerEnvelope:
        payload = dict(value)
        claimed = str(payload.pop("envelope_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 AI worker envelope digest differs")
        try:
            selected = cls(
                model_name=str(payload["model_name"]),
                endpoint=str(payload["endpoint"]),
                timeout_seconds=float(payload["timeout_seconds"]),
                model_manifest=Round74AIModelManifest.from_dict(
                    _mapping(payload["model_manifest"], "model manifest")
                ),
                review_request=Round74AIReviewRequest.from_dict(
                    _mapping(payload["review_request"], "review request")
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 AI worker envelope payload differs") from exc
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 AI worker envelope policy differs")
        selected.validate()
        return selected


@dataclass(frozen=True)
class Round74AIWorkerResult:
    """Hash-bound output proving model identity and full GPU residency."""

    envelope_sha256: str
    manifest_sha256: str
    request_sha256: str
    model_name: str
    model_digest: str
    model_metadata_sha256: str
    system_prompt_sha256: str
    user_prompt_sha256: str
    raw_response_sha256: str
    decision: Round74AIReviewDecision
    residency: OllamaResidencyReport
    prompt_eval_count: int
    eval_count: int
    total_duration_ns: int
    load_duration_ns: int
    prompt_eval_duration_ns: int
    eval_duration_ns: int
    schema_version: str = ROUND74_AI_WORKER_RESULT_SCHEMA_VERSION

    def validate(self) -> None:
        counters = (
            self.prompt_eval_count,
            self.eval_count,
            self.total_duration_ns,
            self.load_duration_ns,
            self.prompt_eval_duration_ns,
            self.eval_duration_ns,
        )
        if (
            self.schema_version != ROUND74_AI_WORKER_RESULT_SCHEMA_VERSION
            or _validate_model_name(self.model_name) != self.model_name
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counters
            )
        ):
            raise ValueError("Round 74 AI worker result differs")
        for label, digest in (
            ("envelope", self.envelope_sha256),
            ("manifest", self.manifest_sha256),
            ("request", self.request_sha256),
            ("model", self.model_digest),
            ("metadata", self.model_metadata_sha256),
            ("system prompt", self.system_prompt_sha256),
            ("user prompt", self.user_prompt_sha256),
            ("raw response", self.raw_response_sha256),
        ):
            _require_sha256(digest, label)
        self.decision.validate()
        self.residency.validated()
        if (
            not self.residency.fully_gpu_resident
            or self.residency.digest != self.model_digest
            or self.residency.requested_model != self.model_name
        ):
            raise ValueError("Round 74 AI worker full GPU residency differs")

    @property
    def result_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "envelope_sha256": self.envelope_sha256,
            "manifest_sha256": self.manifest_sha256,
            "request_sha256": self.request_sha256,
            "model_name": self.model_name,
            "model_digest": self.model_digest,
            "model_metadata_sha256": self.model_metadata_sha256,
            "system_prompt_sha256": self.system_prompt_sha256,
            "user_prompt_sha256": self.user_prompt_sha256,
            "raw_response_sha256": self.raw_response_sha256,
            "decision": self.decision.as_dict(),
            "residency": self.residency.asdict(),
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "total_duration_ns": self.total_duration_ns,
            "load_duration_ns": self.load_duration_ns,
            "prompt_eval_duration_ns": self.prompt_eval_duration_ns,
            "eval_duration_ns": self.eval_duration_ns,
            "remote_inference_used": False,
            "execution_authority": False,
            "full_gpu_residency_verified": True,
        }
        if include_sha256:
            payload["result_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74AIWorkerResult:
        payload = dict(value)
        claimed = str(payload.pop("result_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 AI worker result digest differs")
        try:
            selected = cls(
                envelope_sha256=str(payload["envelope_sha256"]),
                manifest_sha256=str(payload["manifest_sha256"]),
                request_sha256=str(payload["request_sha256"]),
                model_name=str(payload["model_name"]),
                model_digest=str(payload["model_digest"]),
                model_metadata_sha256=str(payload["model_metadata_sha256"]),
                system_prompt_sha256=str(payload["system_prompt_sha256"]),
                user_prompt_sha256=str(payload["user_prompt_sha256"]),
                raw_response_sha256=str(payload["raw_response_sha256"]),
                decision=_decision_from_mapping(payload["decision"]),
                residency=ollama_residency_from_mapping(payload["residency"]),
                prompt_eval_count=_nonnegative_int(
                    payload["prompt_eval_count"],
                    "prompt count",
                ),
                eval_count=_nonnegative_int(
                    payload["eval_count"],
                    "eval count",
                ),
                total_duration_ns=_nonnegative_int(
                    payload["total_duration_ns"],
                    "total duration",
                ),
                load_duration_ns=_nonnegative_int(
                    payload["load_duration_ns"],
                    "load duration",
                ),
                prompt_eval_duration_ns=_nonnegative_int(
                    payload["prompt_eval_duration_ns"],
                    "prompt duration",
                ),
                eval_duration_ns=_nonnegative_int(
                    payload["eval_duration_ns"],
                    "eval duration",
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 AI worker result payload differs") from exc
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 AI worker result policy differs")
        selected.validate()
        return selected


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 74 AI worker {label} differs")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Round 74 AI worker {label} differs")
    return value


def _decision_from_mapping(value: object) -> Round74AIReviewDecision:
    payload = _mapping(value, "decision")
    expected = {
        "schema_version",
        "verdict",
        "size_multiplier_bps",
        "confidence_bps",
        "reason_codes",
        "may_increase_risk",
        "may_select_side",
        "may_set_leverage",
        "may_submit_or_cancel_orders",
        "decision_sha256",
    }
    if set(payload) != expected:
        raise ValueError("Round 74 AI worker decision fields differ")
    raw = _canonical_json(
        {
            key: payload[key]
            for key in (
                "schema_version",
                "verdict",
                "size_multiplier_bps",
                "confidence_bps",
                "reason_codes",
            )
        }
    )
    selected = Round74AIReviewDecision.from_generated_text(raw)
    if selected.as_dict() != dict(payload):
        raise ValueError("Round 74 AI worker decision policy differs")
    return selected


def _post_json(
    url: str,
    payload: Mapping[str, object],
    timeout_seconds: float,
) -> object:
    if not url.startswith(f"{ROUND74_AI_WORKER_ENDPOINT}/"):
        raise ValueError("Round 74 AI worker refused a non-loopback request")
    body = _canonical_json(payload).encode("ascii")
    request = urllib_request.Request(
        url,
        data=body,
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
        raise ValueError("Round 74 AI worker provider request failed") from exc
    if len(raw) > ROUND74_AI_WORKER_MAXIMUM_RESPONSE_BYTES:
        raise ValueError("Round 74 AI worker provider response is too large")
    try:
        return _strict_json_object(
            raw.decode("utf-8"),
            label="Round 74 AI provider response",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 74 AI worker provider response is invalid") from exc


def _real_provenance(
    endpoint: str,
    model_name: str,
    timeout_seconds: float,
) -> tuple[str, str]:
    return resolve_ollama_model_provenance(
        endpoint,
        model_name,
        timeout_seconds,
    )


def _real_residency(
    endpoint: str,
    model_name: str,
    timeout_seconds: float,
    expected_digest: str,
) -> OllamaResidencyReport:
    return inspect_ollama_model_residency(
        endpoint,
        model_name,
        timeout_seconds,
        expected_digest=expected_digest,
    )


def _decision_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "verdict",
            "size_multiplier_bps",
            "confidence_bps",
            "reason_codes",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION,
            },
            "verdict": {
                "type": "string",
                "enum": list(ROUND74_AI_REVIEW_VERDICTS),
            },
            "size_multiplier_bps": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10_000,
            },
            "confidence_bps": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10_000,
            },
            "reason_codes": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": list(ROUND74_AI_REVIEW_REASON_CODES),
                },
                "minItems": 1,
                "maxItems": len(ROUND74_AI_REVIEW_REASON_CODES),
                "uniqueItems": True,
            },
        },
    }


def _ollama_payload(
    envelope: Round74AIWorkerEnvelope,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, object]:
    return {
        "model": envelope.model_name,
        "stream": False,
        "think": False,
        "keep_alive": ROUND74_AI_WORKER_KEEP_ALIVE,
        "format": _decision_json_schema(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "seed": 74,
            "temperature": 0.0,
            "top_p": 1.0,
            "num_ctx": ROUND74_AI_WORKER_CONTEXT_TOKENS,
            "num_predict": ROUND74_AI_WORKER_MAXIMUM_OUTPUT_TOKENS,
        },
    }


def _response_counter(
    payload: Mapping[str, object],
    field: str,
) -> int:
    value = payload.get(field, 0)
    return _nonnegative_int(value, field)


def _response_content(
    payload: Mapping[str, object],
    model_name: str,
) -> str:
    if payload.get("done") is not True or payload.get("done_reason") != "stop":
        raise ValueError("Round 74 AI provider response is incomplete")
    returned_model = payload.get("model")
    if not isinstance(returned_model, str) or returned_model != model_name:
        raise ValueError("Round 74 AI provider model identity differs")
    message = payload.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("Round 74 AI provider message differs")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("Round 74 AI provider content differs")
    if len(content.encode("utf-8")) > ROUND74_AI_WORKER_MAXIMUM_RESPONSE_BYTES:
        raise ValueError("Round 74 AI provider content is too large")
    return content


def execute_round74_ai_worker(
    envelope: Round74AIWorkerEnvelope,
    *,
    post_json: PostJson = _post_json,
    provenance_resolver: ProvenanceResolver = _real_provenance,
    residency_inspector: ResidencyInspector = _real_residency,
) -> Round74AIWorkerResult:
    """Run one local review and prove its exact model and GPU residency."""

    envelope.validate()
    digest, metadata_sha256 = provenance_resolver(
        envelope.endpoint,
        envelope.model_name,
        envelope.timeout_seconds,
    )
    _require_sha256(digest, "resolved model")
    _require_sha256(metadata_sha256, "model metadata")
    if digest != envelope.model_manifest.model_artifact_sha256:
        raise ValueError("Round 74 AI worker model artifact differs")
    system_prompt, user_prompt = build_round74_ai_review_prompt(envelope.review_request)
    response = post_json(
        f"{envelope.endpoint}/api/chat",
        _ollama_payload(envelope, system_prompt, user_prompt),
        envelope.timeout_seconds,
    )
    if not isinstance(response, Mapping):
        raise ValueError("Round 74 AI provider response root differs")
    content = _response_content(response, envelope.model_name)
    decision = Round74AIReviewDecision.from_generated_text(content)
    residency = residency_inspector(
        envelope.endpoint,
        envelope.model_name,
        min(envelope.timeout_seconds, 2.0),
        digest,
    )
    residency.validated()
    if not residency.fully_gpu_resident or residency.digest != digest:
        raise ValueError("Round 74 AI worker requires full GPU model residency")
    result = Round74AIWorkerResult(
        envelope_sha256=envelope.envelope_sha256,
        manifest_sha256=envelope.model_manifest.manifest_sha256,
        request_sha256=envelope.review_request.request_sha256,
        model_name=envelope.model_name,
        model_digest=digest,
        model_metadata_sha256=metadata_sha256,
        system_prompt_sha256=hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        user_prompt_sha256=hashlib.sha256(user_prompt.encode("ascii")).hexdigest(),
        raw_response_sha256=_canonical_sha256(response),
        decision=decision,
        residency=residency,
        prompt_eval_count=_response_counter(
            response,
            "prompt_eval_count",
        ),
        eval_count=_response_counter(response, "eval_count"),
        total_duration_ns=_response_counter(response, "total_duration"),
        load_duration_ns=_response_counter(response, "load_duration"),
        prompt_eval_duration_ns=_response_counter(
            response,
            "prompt_eval_duration",
        ),
        eval_duration_ns=_response_counter(
            response,
            "eval_duration",
        ),
    )
    result.validate()
    return result


def serve_round74_ai_worker_connection(
    connection: Connection,
    *,
    worker_executor: Callable[
        [Round74AIWorkerEnvelope],
        Round74AIWorkerResult,
    ] = execute_round74_ai_worker,
) -> int:
    """Serve strict byte-framed requests for one pinned model identity."""

    bound_identity: tuple[str, str, str] | None = None
    try:
        while True:
            try:
                raw = connection.recv_bytes(ROUND74_AI_WORKER_MAXIMUM_INPUT_BYTES)
            except EOFError:
                return 0
            if not raw:
                return 0
            try:
                payload = _strict_json_object(
                    raw.decode("utf-8"),
                    label="Round 74 AI worker session input",
                )
                envelope = Round74AIWorkerEnvelope.from_dict(payload)
                identity = (
                    envelope.model_name,
                    envelope.model_manifest.manifest_sha256,
                    envelope.endpoint,
                )
                if bound_identity is None:
                    bound_identity = identity
                elif identity != bound_identity:
                    raise ValueError(
                        "Round 74 AI worker session model identity changed"
                    )
                result = worker_executor(envelope)
                result.validate()
                response: dict[str, object] = {
                    "schema_version": (
                        ROUND74_AI_WORKER_SESSION_RESPONSE_SCHEMA_VERSION
                    ),
                    "status": "ok",
                    "result": result.as_dict(),
                    "error_class": None,
                }
            except Exception as exc:  # noqa: BLE001 - process boundary fails closed
                response = {
                    "schema_version": (
                        ROUND74_AI_WORKER_SESSION_RESPONSE_SCHEMA_VERSION
                    ),
                    "status": "error",
                    "result": None,
                    "error_class": type(exc).__name__,
                }
                encoded = _canonical_json(response).encode("ascii")
                connection.send_bytes(encoded)
                return 2
            encoded = _canonical_json(response).encode("ascii")
            if len(encoded) > ROUND74_AI_WORKER_MAXIMUM_RESPONSE_BYTES:
                raise ValueError("Round 74 AI worker session response is too large")
            connection.send_bytes(encoded)
    finally:
        connection.close()


def _read_stdin() -> str:
    raw = sys.stdin.buffer.read(ROUND74_AI_WORKER_MAXIMUM_INPUT_BYTES + 1)
    if len(raw) > ROUND74_AI_WORKER_MAXIMUM_INPUT_BYTES:
        raise ValueError("Round 74 AI worker input is too large")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Round 74 AI worker input is not UTF-8") from exc


def main() -> int:
    """Read one envelope and emit one canonical, hash-bound result."""

    try:
        payload = _strict_json_object(
            _read_stdin(),
            label="Round 74 AI worker input",
        )
        envelope = Round74AIWorkerEnvelope.from_dict(payload)
        result = execute_round74_ai_worker(envelope)
        sys.stdout.write(_canonical_json(result.as_dict()) + "\n")
        return 0
    except Exception as exc:  # noqa: BLE001 - process boundary fails closed
        error_type = type(exc).__name__
        sys.stderr.write(f"Round 74 AI worker failed: {error_type}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ROUND74_AI_WORKER_CONTEXT_TOKENS",
    "ROUND74_AI_WORKER_ENDPOINT",
    "ROUND74_AI_WORKER_ENVELOPE_SCHEMA_VERSION",
    "ROUND74_AI_WORKER_MAXIMUM_INPUT_BYTES",
    "ROUND74_AI_WORKER_MAXIMUM_OUTPUT_TOKENS",
    "ROUND74_AI_WORKER_MAXIMUM_RESPONSE_BYTES",
    "ROUND74_AI_WORKER_MAXIMUM_TIMEOUT_SECONDS",
    "ROUND74_AI_WORKER_RESULT_SCHEMA_VERSION",
    "ROUND74_AI_WORKER_SESSION_RESPONSE_SCHEMA_VERSION",
    "Round74AIWorkerEnvelope",
    "Round74AIWorkerResult",
    "execute_round74_ai_worker",
    "serve_round74_ai_worker_connection",
]
