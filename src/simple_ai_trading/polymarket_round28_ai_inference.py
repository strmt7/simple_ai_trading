"""Measured local inference for the Round 28 augmented-model AI veto."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import time

from . import ai_runtime as _ai_runtime
from . import polymarket_round28_ai_host as _host
from .ai_runtime import (
    OllamaResidencyReport,
    inspect_ollama_model_residency,
    ollama_residency_from_mapping,
)
from .polymarket_round28_ai_cases import (
    Round28AICase,
    Round28AICasePanel,
    round28_ai_case_prompt,
)
from .polymarket_round28_ai_contract import (
    POLYMARKET_ROUND28_AI_CASE_TIMEOUT_SECONDS,
    POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
    POLYMARKET_ROUND28_AI_MAXIMUM_REJECTED_FRACTION,
    POLYMARKET_ROUND28_AI_MINIMUM_BASELINE_CANDIDATES,
    POLYMARKET_ROUND28_AI_MINIMUM_CHANGED_ACTIONS,
    POLYMARKET_ROUND28_AI_REASON_CODES,
    validate_round28_ai_contract,
)
from .polymarket_round28_ai_host import (
    POLYMARKET_ROUND28_AI_BASE_URL,
    Round28AIHostCandidate,
    round28_ai_conformance_request,
    round28_ai_response_schema,
    round28_ai_unload_request,
    validate_round28_ai_host_report,
)


POLYMARKET_ROUND28_AI_INFERENCE_RESPONSE_SCHEMA_VERSION = (
    "polymarket-round28-ai-inference-response-v1"
)
POLYMARKET_ROUND28_AI_INFERENCE_REPORT_SCHEMA_VERSION = (
    "polymarket-round28-ai-inference-report-v1"
)
_VALID_STATUS = "valid"
_FAILURE_STATUSES = frozenset({"invalid_response", "provider_error", "timeout"})
_NON_AUTHORITY_FLAGS = (
    "target_accessed",
    "outcome_accessed",
    "future_books_accessed",
    "pnl_accessed",
    "credentials_used",
    "orders_submitted",
    "trading_authority",
    "edge_claim",
    "profitability_claim",
)


JsonPoster = Callable[[str, Mapping[str, object], float], object]
ResidencyInspector = Callable[..., OllamaResidencyReport]
InventoryGetter = Callable[[str, float], object]
ProgressCallback = Callable[[int, int], None]


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


def round28_ai_inference_request(
    case: Round28AICase,
    candidate: Round28AIHostCandidate,
) -> dict[str, object]:
    selected_case = case.validated()
    selected_candidate = candidate.validated()
    schema = round28_ai_response_schema()
    schema["properties"]["reason_codes"]["items"] = {
        "type": "string",
        "enum": list(POLYMARKET_ROUND28_AI_REASON_CODES),
    }
    return {
        "model": selected_candidate.runtime_model,
        "prompt": round28_ai_case_prompt(selected_case),
        "stream": False,
        "format": schema,
        "keep_alive": "30s",
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 28,
            "num_ctx": 8192,
            "num_predict": 96,
        },
    }


def _candidate_from_mapping(value: Mapping[str, object]) -> Round28AIHostCandidate:
    if set(value) != set(Round28AIHostCandidate.__dataclass_fields__):
        raise ValueError("Round 28 AI persisted candidate differs")
    return Round28AIHostCandidate(
        model_id=str(value["model_id"]),
        runtime_model=str(value["runtime_model"]),
        runtime_digest=str(value["runtime_digest"]),
        upstream_revision=str(value["upstream_revision"]),
        role=str(value["role"]),
        quantization=str(value["quantization"]),
        artifact_source=str(value["artifact_source"]),
        artifact_revision=str(value["artifact_revision"]),
        artifact_sha256=str(value["artifact_sha256"]),
        artifact_size_bytes=int(value["artifact_size_bytes"]),
        contract_host_qualification=str(value["contract_host_qualification"]),
    ).validated()


@dataclass(frozen=True, slots=True)
class Round28AIResponse:
    model_id: str
    runtime_model: str
    runtime_digest: str
    case_sha256: str
    prompt_sha256: str
    status: str
    decision: str
    reason_codes: tuple[str, ...]
    wall_latency_ms: int
    provider_total_ms: int | None
    provider_load_ms: int | None
    prompt_tokens: int | None
    output_tokens: int | None
    error_type: str | None
    error_message: str | None
    response_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND28_AI_INFERENCE_RESPONSE_SCHEMA_VERSION,
            "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
            "model_id": self.model_id,
            "runtime_model": self.runtime_model,
            "runtime_digest": self.runtime_digest,
            "case_sha256": self.case_sha256,
            "prompt_sha256": self.prompt_sha256,
            "status": self.status,
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
            "wall_latency_ms": self.wall_latency_ms,
            "provider_total_ms": self.provider_total_ms,
            "provider_load_ms": self.provider_load_ms,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "error_type": self.error_type,
            "error_message": self.error_message,
            **{field: False for field in _NON_AUTHORITY_FLAGS[:-2]},
        }

    @property
    def valid(self) -> bool:
        return self.status == _VALID_STATUS

    @property
    def abstains(self) -> bool:
        return self.decision in {"reject", "reduce"}

    def validated(
        self,
        candidate: Round28AIHostCandidate,
    ) -> "Round28AIResponse":
        selected = candidate.validated()
        valid_reasons = bool(
            1 <= len(self.reason_codes) <= 3
            and len(set(self.reason_codes)) == len(self.reason_codes)
            and set(self.reason_codes) <= set(POLYMARKET_ROUND28_AI_REASON_CODES)
        )
        unchanged_semantics = (
            self.decision == "unchanged"
            and self.reason_codes == ("no_material_risk",)
        ) or (
            self.decision in {"reject", "reduce"}
            and "no_material_risk" not in self.reason_codes
        )
        failure_semantics = (
            self.status == _VALID_STATUS
            and self.error_type is None
            and self.error_message is None
        ) or (
            self.status in _FAILURE_STATUSES
            and self.decision == "reject"
            and self.reason_codes == ("insufficient_evidence",)
            and bool(self.error_type)
            and bool(self.error_message)
            and all(
                value is None
                for value in (
                    self.provider_total_ms,
                    self.provider_load_ms,
                    self.prompt_tokens,
                    self.output_tokens,
                )
            )
        )
        if (
            (self.model_id, self.runtime_model, self.runtime_digest)
            != (selected.model_id, selected.runtime_model, selected.runtime_digest)
            or len(self.case_sha256) != 64
            or len(self.prompt_sha256) != 64
            or self.decision not in {"reject", "reduce", "unchanged"}
            or not valid_reasons
            or not unchanged_semantics
            or not failure_semantics
            or self.wall_latency_ms <= 0
            or (
                self.status == _VALID_STATUS
                and self.wall_latency_ms
                > math.ceil(POLYMARKET_ROUND28_AI_CASE_TIMEOUT_SECONDS * 1_000)
            )
            or any(
                value is not None
                and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
                for value in (
                    self.provider_total_ms,
                    self.provider_load_ms,
                    self.prompt_tokens,
                    self.output_tokens,
                )
            )
            or self.response_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 28 AI response differs")
        return self

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "response_sha256": self.response_sha256}


def _response_from_raw(
    *,
    raw: object,
    candidate: Round28AIHostCandidate,
    case: Round28AICase,
    prompt_sha256: str,
    wall_latency_ms: int,
) -> Round28AIResponse:
    if not isinstance(raw, Mapping):
        raise ValueError("Round 28 AI provider response is not an object")
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
        raise ValueError("Round 28 AI provider response fields are incomplete")
    if (
        raw["model"] != candidate.runtime_model
        or raw["done"] is not True
        or raw["done_reason"] != "stop"
        or not isinstance(raw["response"], str)
    ):
        raise ValueError("Round 28 AI provider completion differs")
    parsed = _host._strict_json_value(raw["response"])  # noqa: SLF001
    if not isinstance(parsed, Mapping) or set(parsed) != {"decision", "reason_codes"}:
        raise ValueError("Round 28 AI structured response differs")
    decision = parsed["decision"]
    reasons = parsed["reason_codes"]
    numeric = (
        raw["total_duration"],
        raw["load_duration"],
        raw["prompt_eval_count"],
        raw["eval_count"],
    )
    if (
        not isinstance(decision, str)
        or not isinstance(reasons, list)
        or any(not isinstance(value, str) for value in reasons)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in numeric
        )
        or type(raw["prompt_eval_count"]) is not int
        or raw["prompt_eval_count"] <= 0
        or type(raw["eval_count"]) is not int
        or raw["eval_count"] <= 0
    ):
        raise ValueError("Round 28 AI structured response types differ")
    total_ms = math.ceil(float(raw["total_duration"]) / 1_000_000)
    load_ms = math.ceil(float(raw["load_duration"]) / 1_000_000)
    if load_ms > total_ms:
        raise ValueError("Round 28 AI provider timing differs")
    provisional = Round28AIResponse(
        model_id=candidate.model_id,
        runtime_model=candidate.runtime_model,
        runtime_digest=candidate.runtime_digest,
        case_sha256=case.case_sha256,
        prompt_sha256=prompt_sha256,
        status=_VALID_STATUS,
        decision=decision,
        reason_codes=tuple(reasons),
        wall_latency_ms=wall_latency_ms,
        provider_total_ms=total_ms,
        provider_load_ms=load_ms,
        prompt_tokens=int(raw["prompt_eval_count"]),
        output_tokens=int(raw["eval_count"]),
        error_type=None,
        error_message=None,
        response_sha256="",
    )
    return replace(
        provisional,
        response_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated(candidate)


def _failed_response(
    *,
    candidate: Round28AIHostCandidate,
    case: Round28AICase,
    prompt_sha256: str,
    status: str,
    wall_latency_ms: int,
    error: Exception,
) -> Round28AIResponse:
    if status not in _FAILURE_STATUSES:
        raise ValueError("Round 28 AI failure status differs")
    provisional = Round28AIResponse(
        model_id=candidate.model_id,
        runtime_model=candidate.runtime_model,
        runtime_digest=candidate.runtime_digest,
        case_sha256=case.case_sha256,
        prompt_sha256=prompt_sha256,
        status=status,
        decision="reject",
        reason_codes=("insufficient_evidence",),
        wall_latency_ms=max(1, wall_latency_ms),
        provider_total_ms=None,
        provider_load_ms=None,
        prompt_tokens=None,
        output_tokens=None,
        error_type=type(error).__name__,
        error_message=str(error)[:240] or status,
        response_sha256="",
    )
    return replace(
        provisional,
        response_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated(candidate)


def _inventory_has_only_candidate(
    raw: object,
    candidate: Round28AIHostCandidate,
) -> bool:
    if not isinstance(raw, Mapping) or set(raw) != {"models"}:
        return False
    models = raw["models"]
    return bool(
        isinstance(models, list)
        and len(models) == 1
        and isinstance(models[0], Mapping)
        and models[0].get("digest") == candidate.runtime_digest
    )


@dataclass(frozen=True, slots=True)
class Round28AIInferenceReport:
    candidate: dict[str, object]
    host_qualification_report_sha256: str
    case_panel_sha256: str
    prompt_population_sha256: str
    warmup_wall_ms: int
    residency: dict[str, object]
    responses: tuple[Round28AIResponse, ...]
    status_counts: dict[str, int]
    changed_action_count: int
    rejected_fraction: float
    candidate_eligible_for_matched_evaluation: bool
    unload_observed: bool
    unload_failure: dict[str, str] | None
    report_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND28_AI_INFERENCE_REPORT_SCHEMA_VERSION,
            "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
            "candidate": dict(self.candidate),
            "host_qualification_report_sha256": (
                self.host_qualification_report_sha256
            ),
            "case_panel_sha256": self.case_panel_sha256,
            "prompt_population_sha256": self.prompt_population_sha256,
            "warmup_wall_ms": self.warmup_wall_ms,
            "residency": dict(self.residency),
            "response_sha256": [
                response.response_sha256 for response in self.responses
            ],
            "responses": [response.asdict() for response in self.responses],
            "status_counts": dict(self.status_counts),
            "case_count": len(self.responses),
            "changed_action_count": self.changed_action_count,
            "rejected_fraction": self.rejected_fraction,
            "candidate_eligible_for_matched_evaluation": (
                self.candidate_eligible_for_matched_evaluation
            ),
            "unload_observed": self.unload_observed,
            "unload_failure": self.unload_failure,
            **{field: False for field in _NON_AUTHORITY_FLAGS},
        }

    def validated(self) -> "Round28AIInferenceReport":
        candidate = _candidate_from_mapping(self.candidate)
        responses = tuple(response.validated(candidate) for response in self.responses)
        counts = dict(Counter(response.status for response in responses))
        changed = sum(response.abstains for response in responses)
        rejected_fraction = changed / len(responses) if responses else 0.0
        residency = ollama_residency_from_mapping(self.residency)
        eligible = bool(
            len(responses) >= POLYMARKET_ROUND28_AI_MINIMUM_BASELINE_CANDIDATES
            and counts == {_VALID_STATUS: len(responses)}
            and changed >= POLYMARKET_ROUND28_AI_MINIMUM_CHANGED_ACTIONS
            and rejected_fraction <= POLYMARKET_ROUND28_AI_MAXIMUM_REJECTED_FRACTION
            and residency.digest == candidate.runtime_digest
            and residency.fully_gpu_resident
            and self.unload_observed
            and self.unload_failure is None
        )
        if (
            len(self.host_qualification_report_sha256) != 64
            or len(self.case_panel_sha256) != 64
            or len(self.prompt_population_sha256) != 64
            or self.warmup_wall_ms <= 0
            or len({response.case_sha256 for response in responses})
            != len(responses)
            or any(response.model_id != candidate.model_id for response in responses)
            or self.status_counts != counts
            or self.changed_action_count != changed
            or not math.isclose(
                self.rejected_fraction,
                rejected_fraction,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or self.candidate_eligible_for_matched_evaluation is not eligible
            or self.report_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 28 AI inference report differs")
        return self

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "report_sha256": self.report_sha256}


def run_round28_ai_inference(
    *,
    panel: Round28AICasePanel,
    candidate: Round28AIHostCandidate,
    contract: Mapping[str, object],
    host_qualification_report: Mapping[str, object],
    post_json: JsonPoster = _host._post_json,  # noqa: SLF001
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
    inventory_getter: InventoryGetter = _ai_runtime._get_json,  # noqa: SLF001
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    progress: ProgressCallback | None = None,
) -> Round28AIInferenceReport:
    """Run one qualified candidate over one immutable target-free case panel."""

    selected_panel = panel.validated()
    selected_contract = validate_round28_ai_contract(contract)
    host_report, qualified_candidate = validate_round28_ai_host_report(
        host_qualification_report,
        contract=selected_contract,
    )
    selected_candidate = candidate.validated()
    if selected_candidate != qualified_candidate:
        raise ValueError("Round 28 AI inference candidate differs")
    endpoint = f"{POLYMARKET_ROUND28_AI_BASE_URL}/api/generate"
    for raw_candidate in selected_contract["candidate_program"]:
        post_json(
            endpoint,
            {
                "model": raw_candidate["runtime_model"],
                "keep_alive": 0,
                "stream": False,
            },
            5.0,
        )
    warmup_started = monotonic_ns()
    warmup_raw = post_json(
        endpoint,
        round28_ai_conformance_request(selected_candidate, keep_alive="30s"),
        35.0,
    )
    warmup_wall_ms = max(
        1,
        math.ceil((monotonic_ns() - warmup_started) / 1_000_000),
    )
    _host._measurement(  # noqa: SLF001
        warmup_raw,
        candidate=selected_candidate,
        phase="warmup",
        wall_ms=warmup_wall_ms,
    )
    residency = residency_inspector(
        POLYMARKET_ROUND28_AI_BASE_URL,
        selected_candidate.runtime_model,
        2.0,
        expected_digest=selected_candidate.runtime_digest,
    ).validated()
    inventory = inventory_getter(f"{POLYMARKET_ROUND28_AI_BASE_URL}/api/ps", 2.0)
    if not residency.fully_gpu_resident or not _inventory_has_only_candidate(
        inventory,
        selected_candidate,
    ):
        raise RuntimeError("Round 28 AI candidate is not exclusively GPU resident")

    responses: list[Round28AIResponse] = []
    unload_failure: dict[str, str] | None = None
    unload_observed = False
    try:
        for index, case in enumerate(selected_panel.cases, start=1):
            prompt = round28_ai_case_prompt(case)
            prompt_sha256 = hashlib.sha256(prompt.encode("ascii")).hexdigest()
            started = monotonic_ns()
            try:
                raw = post_json(
                    endpoint,
                    round28_ai_inference_request(case, selected_candidate),
                    POLYMARKET_ROUND28_AI_CASE_TIMEOUT_SECONDS,
                )
                latency_ms = max(
                    1,
                    math.ceil((monotonic_ns() - started) / 1_000_000),
                )
                if latency_ms > math.ceil(
                    POLYMARKET_ROUND28_AI_CASE_TIMEOUT_SECONDS * 1_000
                ):
                    raise TimeoutError("Round 28 AI case exceeded its latency limit")
                response = _response_from_raw(
                    raw=raw,
                    candidate=selected_candidate,
                    case=case,
                    prompt_sha256=prompt_sha256,
                    wall_latency_ms=latency_ms,
                )
            except Exception as exc:  # noqa: BLE001 - failure becomes evidence
                latency_ms = max(
                    1,
                    math.ceil((monotonic_ns() - started) / 1_000_000),
                )
                status = (
                    "timeout"
                    if isinstance(exc, TimeoutError)
                    or latency_ms
                    >= math.ceil(POLYMARKET_ROUND28_AI_CASE_TIMEOUT_SECONDS * 1_000)
                    else (
                        "provider_error"
                        if isinstance(exc, (OSError, RuntimeError))
                        else "invalid_response"
                    )
                )
                response = _failed_response(
                    candidate=selected_candidate,
                    case=case,
                    prompt_sha256=prompt_sha256,
                    status=status,
                    wall_latency_ms=latency_ms,
                    error=exc,
                )
            responses.append(response)
            if progress is not None:
                progress(index, len(selected_panel.cases))
    finally:
        try:
            post_json(endpoint, round28_ai_unload_request(selected_candidate), 5.0)
            after = residency_inspector(
                POLYMARKET_ROUND28_AI_BASE_URL,
                selected_candidate.runtime_model,
                2.0,
                expected_digest=selected_candidate.runtime_digest,
            ).validated()
            unload_observed = after.status == "unloaded"
            if not unload_observed:
                raise RuntimeError("Round 28 AI candidate remained loaded")
        except Exception as exc:  # noqa: BLE001 - cleanup failure is evidence
            unload_failure = {
                "type": type(exc).__name__,
                "message": str(exc)[:240],
            }
    counts = dict(sorted(Counter(response.status for response in responses).items()))
    changed = sum(response.abstains for response in responses)
    rejected_fraction = changed / len(responses) if responses else 0.0
    eligible = bool(
        len(responses) >= POLYMARKET_ROUND28_AI_MINIMUM_BASELINE_CANDIDATES
        and counts == {_VALID_STATUS: len(responses)}
        and changed >= POLYMARKET_ROUND28_AI_MINIMUM_CHANGED_ACTIONS
        and rejected_fraction <= POLYMARKET_ROUND28_AI_MAXIMUM_REJECTED_FRACTION
        and unload_observed
        and unload_failure is None
    )
    provisional = Round28AIInferenceReport(
        candidate=asdict(selected_candidate),
        host_qualification_report_sha256=str(host_report["report_sha256"]),
        case_panel_sha256=selected_panel.panel_sha256,
        prompt_population_sha256=str(
            selected_panel.identity_payload()["prompt_population_sha256"]
        ),
        warmup_wall_ms=warmup_wall_ms,
        residency=residency.asdict(),
        responses=tuple(responses),
        status_counts=counts,
        changed_action_count=changed,
        rejected_fraction=rejected_fraction,
        candidate_eligible_for_matched_evaluation=eligible,
        unload_observed=unload_observed,
        unload_failure=unload_failure,
        report_sha256="",
    )
    return replace(
        provisional,
        report_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def round28_ai_response_from_mapping(
    value: Mapping[str, object],
    *,
    candidate: Round28AIHostCandidate,
) -> Round28AIResponse:
    payload = dict(value)
    expected = {
        *Round28AIResponse.__dataclass_fields__,
        "schema_version",
        "ai_contract_sha256",
        *_NON_AUTHORITY_FLAGS[:-2],
    }
    if (
        set(payload) != expected
        or payload.get("schema_version")
        != POLYMARKET_ROUND28_AI_INFERENCE_RESPONSE_SCHEMA_VERSION
        or payload.get("ai_contract_sha256")
        != POLYMARKET_ROUND28_AI_CONTRACT_SHA256
        or any(payload.get(field) is not False for field in _NON_AUTHORITY_FLAGS[:-2])
        or not isinstance(payload.get("reason_codes"), list)
    ):
        raise ValueError("Round 28 persisted AI response differs")
    constructor = {
        key: payload[key]
        for key in Round28AIResponse.__dataclass_fields__
        if key != "reason_codes"
    }
    return Round28AIResponse(
        **constructor,
        reason_codes=tuple(str(item) for item in payload["reason_codes"]),
    ).validated(candidate)


def round28_ai_inference_report_from_mapping(
    value: Mapping[str, object],
) -> Round28AIInferenceReport:
    payload = dict(value)
    expected = {
        *Round28AIInferenceReport.__dataclass_fields__,
        "schema_version",
        "ai_contract_sha256",
        "response_sha256",
        "case_count",
        *_NON_AUTHORITY_FLAGS,
    }
    raw_candidate = payload.get("candidate")
    raw_responses = payload.get("responses")
    if (
        set(payload) != expected
        or payload.get("schema_version")
        != POLYMARKET_ROUND28_AI_INFERENCE_REPORT_SCHEMA_VERSION
        or payload.get("ai_contract_sha256")
        != POLYMARKET_ROUND28_AI_CONTRACT_SHA256
        or any(payload.get(field) is not False for field in _NON_AUTHORITY_FLAGS)
        or not isinstance(raw_candidate, Mapping)
        or not isinstance(raw_responses, list)
        or not isinstance(payload.get("residency"), Mapping)
        or not isinstance(payload.get("status_counts"), Mapping)
        or not isinstance(payload.get("response_sha256"), list)
    ):
        raise ValueError("Round 28 persisted AI inference report differs")
    candidate = _candidate_from_mapping(raw_candidate)
    responses = tuple(
        round28_ai_response_from_mapping(item, candidate=candidate)
        for item in raw_responses
        if isinstance(item, Mapping)
    )
    if (
        len(responses) != len(raw_responses)
        or payload.get("case_count") != len(responses)
        or payload.get("response_sha256")
        != [response.response_sha256 for response in responses]
    ):
        raise ValueError("Round 28 persisted AI response population differs")
    failure = payload.get("unload_failure")
    if failure is not None and (
        not isinstance(failure, Mapping)
        or set(failure) != {"type", "message"}
        or any(not isinstance(failure.get(key), str) for key in ("type", "message"))
    ):
        raise ValueError("Round 28 persisted AI unload failure differs")
    return Round28AIInferenceReport(
        candidate=dict(raw_candidate),
        host_qualification_report_sha256=str(
            payload["host_qualification_report_sha256"]
        ),
        case_panel_sha256=str(payload["case_panel_sha256"]),
        prompt_population_sha256=str(payload["prompt_population_sha256"]),
        warmup_wall_ms=int(payload["warmup_wall_ms"]),
        residency=dict(payload["residency"]),
        responses=responses,
        status_counts={
            str(key): int(count) for key, count in payload["status_counts"].items()
        },
        changed_action_count=int(payload["changed_action_count"]),
        rejected_fraction=float(payload["rejected_fraction"]),
        candidate_eligible_for_matched_evaluation=bool(
            payload["candidate_eligible_for_matched_evaluation"]
        ),
        unload_observed=bool(payload["unload_observed"]),
        unload_failure=None if failure is None else dict(failure),
        report_sha256=str(payload["report_sha256"]),
    ).validated()


def validate_round28_ai_inference_report(
    value: Mapping[str, object],
    *,
    contract: Mapping[str, object],
    host_qualification_report: Mapping[str, object],
    panel: Round28AICasePanel,
) -> Round28AIInferenceReport:
    """Bind a persisted inference receipt to its host and prompt populations."""

    selected_panel = panel.validated()
    selected_contract = validate_round28_ai_contract(contract)
    host_report, candidate = validate_round28_ai_host_report(
        host_qualification_report,
        contract=selected_contract,
    )
    report = round28_ai_inference_report_from_mapping(value)
    restored_candidate = _candidate_from_mapping(report.candidate)
    if (
        restored_candidate != candidate
        or report.host_qualification_report_sha256
        != host_report["report_sha256"]
        or report.case_panel_sha256 != selected_panel.panel_sha256
        or report.prompt_population_sha256
        != selected_panel.identity_payload()["prompt_population_sha256"]
        or [response.case_sha256 for response in report.responses]
        != [case.case_sha256 for case in selected_panel.cases]
    ):
        raise ValueError("Round 28 AI inference evidence lineage differs")
    return report


__all__ = [
    "POLYMARKET_ROUND28_AI_INFERENCE_REPORT_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_AI_INFERENCE_RESPONSE_SCHEMA_VERSION",
    "Round28AIInferenceReport",
    "Round28AIResponse",
    "round28_ai_inference_report_from_mapping",
    "round28_ai_inference_request",
    "round28_ai_response_from_mapping",
    "run_round28_ai_inference",
    "validate_round28_ai_inference_report",
]
