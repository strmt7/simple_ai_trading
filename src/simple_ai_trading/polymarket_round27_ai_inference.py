"""Measured, target-free local inference for the Round 27 matched AI ablation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import time

from . import ai_runtime as _ai_runtime
from . import polymarket_round27_ai as _host
from .ai_runtime import OllamaResidencyReport, inspect_ollama_model_residency
from .polymarket_round27_ai import (
    POLYMARKET_ROUND27_AI_BASE_URL,
    POLYMARKET_ROUND27_AI_HOST_CANDIDATES,
    Round27AIHostCandidate,
    round27_ai_conformance_request,
)
from .polymarket_round27_ai_ablation_contract import (
    POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256,
    POLYMARKET_ROUND27_AI_CASE_TIMEOUT_SECONDS,
    POLYMARKET_ROUND27_AI_MAXIMUM_REJECTED_FRACTION,
    POLYMARKET_ROUND27_AI_MINIMUM_BASELINE_CANDIDATES,
    POLYMARKET_ROUND27_AI_MINIMUM_CHANGED_ACTIONS,
    POLYMARKET_ROUND27_AI_REASON_CODES,
)
from .polymarket_round27_ai_cases import (
    Round27AICase,
    Round27AICasePanel,
    round27_ai_case_prompt,
)


POLYMARKET_ROUND27_AI_RESPONSE_SCHEMA_VERSION = (
    "polymarket-round27-ai-ablation-response-v1"
)
POLYMARKET_ROUND27_AI_INFERENCE_REPORT_SCHEMA_VERSION = (
    "polymarket-round27-ai-inference-report-v1"
)
_VALID_STATUS = "valid"
_FAILURE_STATUSES = frozenset({"invalid_response", "provider_error", "timeout"})


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


def round27_ai_ablation_response_schema() -> dict[str, object]:
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
                "items": {
                    "type": "string",
                    "enum": list(POLYMARKET_ROUND27_AI_REASON_CODES),
                },
                "minItems": 1,
                "maxItems": 3,
                "uniqueItems": True,
            },
        },
    }


def round27_ai_ablation_request(
    case: Round27AICase,
    candidate: Round27AIHostCandidate,
) -> dict[str, object]:
    selected = case.validated()
    if candidate not in POLYMARKET_ROUND27_AI_HOST_CANDIDATES:
        raise ValueError("Round 27 AI inference candidate differs")
    return {
        "model": candidate.runtime_model,
        "prompt": round27_ai_case_prompt(selected),
        "stream": False,
        "format": round27_ai_ablation_response_schema(),
        "keep_alive": "30s",
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 27,
            "num_ctx": 8192,
            "num_predict": 96,
        },
    }


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Round 27 AI {name} differs")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"Round 27 AI {name} differs")
    return selected


@dataclass(frozen=True, slots=True)
class Round27AIResponse:
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
            "schema_version": POLYMARKET_ROUND27_AI_RESPONSE_SCHEMA_VERSION,
            "ablation_contract_sha256": (
                POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
            ),
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
            "target_accessed": False,
            "outcome_accessed": False,
            "future_books_accessed": False,
            "pnl_accessed": False,
            "credentials_used": False,
            "orders_submitted": False,
            "trading_authority": False,
        }

    @property
    def valid(self) -> bool:
        return self.status == _VALID_STATUS

    @property
    def abstains(self) -> bool:
        return self.decision in {"reject", "reduce"}

    def validated(self) -> "Round27AIResponse":
        candidate = next(
            (
                item
                for item in POLYMARKET_ROUND27_AI_HOST_CANDIDATES
                if item.model_id == self.model_id
            ),
            None,
        )
        valid_decision = self.decision in {"reject", "reduce", "unchanged"}
        valid_reasons = (
            1 <= len(self.reason_codes) <= 3
            and len(self.reason_codes) == len(set(self.reason_codes))
            and set(self.reason_codes) <= set(POLYMARKET_ROUND27_AI_REASON_CODES)
        )
        unchanged_semantics = (
            self.decision != "unchanged"
            and "no_material_risk" not in self.reason_codes
        ) or (
            self.decision == "unchanged"
            and self.reason_codes == ("no_material_risk",)
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
            and self.provider_total_ms is None
            and self.provider_load_ms is None
            and self.prompt_tokens is None
            and self.output_tokens is None
        )
        if (
            candidate is None
            or (self.runtime_model, self.runtime_digest)
            != (candidate.runtime_model, candidate.runtime_digest)
            or len(self.case_sha256) != 64
            or len(self.prompt_sha256) != 64
            or not valid_decision
            or not valid_reasons
            or not unchanged_semantics
            or not failure_semantics
            or self.wall_latency_ms <= 0
            or (
                self.status == _VALID_STATUS
                and self.wall_latency_ms
                > math.ceil(POLYMARKET_ROUND27_AI_CASE_TIMEOUT_SECONDS * 1_000)
            )
            or any(
                value is not None and (isinstance(value, bool) or value < 0)
                for value in (
                    self.provider_total_ms,
                    self.provider_load_ms,
                    self.prompt_tokens,
                    self.output_tokens,
                )
            )
            or self.response_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 27 AI response differs")
        return self

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "response_sha256": self.response_sha256}


def _response_from_raw(
    *,
    raw: object,
    candidate: Round27AIHostCandidate,
    case: Round27AICase,
    prompt_sha256: str,
    wall_latency_ms: int,
) -> Round27AIResponse:
    if not isinstance(raw, Mapping):
        raise ValueError("Round 27 AI provider response is not an object")
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
        raise ValueError("Round 27 AI provider response fields are incomplete")
    if (
        raw["model"] != candidate.runtime_model
        or raw["done"] is not True
        or raw["done_reason"] != "stop"
        or not isinstance(raw["response"], str)
    ):
        raise ValueError("Round 27 AI provider completion differs")
    parsed = _host._strict_json_value(raw["response"])  # noqa: SLF001
    if not isinstance(parsed, Mapping) or set(parsed) != {"decision", "reason_codes"}:
        raise ValueError("Round 27 AI structured response differs")
    decision = parsed["decision"]
    reasons = parsed["reason_codes"]
    if not isinstance(decision, str) or not isinstance(reasons, list) or any(
        not isinstance(value, str) for value in reasons
    ):
        raise ValueError("Round 27 AI structured response types differ")
    total_ms = math.ceil(
        _finite_nonnegative(raw["total_duration"], name="total duration")
        / 1_000_000
    )
    load_ms = math.ceil(
        _finite_nonnegative(raw["load_duration"], name="load duration")
        / 1_000_000
    )
    prompt_tokens = raw["prompt_eval_count"]
    output_tokens = raw["eval_count"]
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens <= 0
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens <= 0
        or load_ms > total_ms
    ):
        raise ValueError("Round 27 AI provider token evidence differs")
    provisional = Round27AIResponse(
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
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        error_type=None,
        error_message=None,
        response_sha256="",
    )
    return replace(
        provisional,
        response_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _failed_response(
    *,
    candidate: Round27AIHostCandidate,
    case: Round27AICase,
    prompt_sha256: str,
    status: str,
    wall_latency_ms: int,
    error: Exception,
) -> Round27AIResponse:
    if status not in _FAILURE_STATUSES:
        raise ValueError("Round 27 AI failure status differs")
    provisional = Round27AIResponse(
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
    ).validated()


def _inventory_has_only_candidate(
    raw: object,
    candidate: Round27AIHostCandidate,
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
class Round27AIInferenceReport:
    candidate: dict[str, object]
    case_panel_sha256: str
    prompt_population_sha256: str
    warmup_wall_ms: int
    residency: dict[str, object]
    responses: tuple[Round27AIResponse, ...]
    status_counts: dict[str, int]
    changed_action_count: int
    rejected_fraction: float
    candidate_eligible_for_matched_evaluation: bool
    unload_observed: bool
    unload_failure: dict[str, str] | None
    report_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                POLYMARKET_ROUND27_AI_INFERENCE_REPORT_SCHEMA_VERSION
            ),
            "ablation_contract_sha256": (
                POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
            ),
            "candidate": dict(self.candidate),
            "case_panel_sha256": self.case_panel_sha256,
            "prompt_population_sha256": self.prompt_population_sha256,
            "warmup_wall_ms": self.warmup_wall_ms,
            "residency": dict(self.residency),
            "response_sha256": [
                response.response_sha256 for response in self.responses
            ],
            "status_counts": dict(self.status_counts),
            "case_count": len(self.responses),
            "changed_action_count": self.changed_action_count,
            "rejected_fraction": self.rejected_fraction,
            "candidate_eligible_for_matched_evaluation": (
                self.candidate_eligible_for_matched_evaluation
            ),
            "unload_observed": self.unload_observed,
            "unload_failure": self.unload_failure,
            "target_accessed": False,
            "outcome_accessed": False,
            "future_books_accessed": False,
            "pnl_accessed": False,
            "credentials_used": False,
            "orders_submitted": False,
            "trading_authority": False,
            "edge_claim": False,
            "profitability_claim": False,
        }

    def validated(self) -> "Round27AIInferenceReport":
        responses = tuple(response.validated() for response in self.responses)
        candidate_ids = {
            item.model_id: item for item in POLYMARKET_ROUND27_AI_HOST_CANDIDATES
        }
        candidate = candidate_ids.get(str(self.candidate.get("model_id")))
        counts = dict(Counter(response.status for response in responses))
        changed = sum(response.abstains for response in responses)
        rejected_fraction = changed / len(responses) if responses else 0.0
        eligible = bool(
            candidate is not None
            and len(responses) >= POLYMARKET_ROUND27_AI_MINIMUM_BASELINE_CANDIDATES
            and counts == {_VALID_STATUS: len(responses)}
            and changed >= POLYMARKET_ROUND27_AI_MINIMUM_CHANGED_ACTIONS
            and rejected_fraction <= POLYMARKET_ROUND27_AI_MAXIMUM_REJECTED_FRACTION
            and self.unload_observed
            and self.unload_failure is None
        )
        if (
            candidate is None
            or self.candidate != asdict(candidate)
            or len(self.case_panel_sha256) != 64
            or len(self.prompt_population_sha256) != 64
            or self.warmup_wall_ms <= 0
            or self.residency.get("digest") != candidate.runtime_digest
            or self.residency.get("gpu_resident") is not True
            or self.residency.get("vram_to_model_ratio") != 1.0
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
            raise ValueError("Round 27 AI inference report differs")
        return self

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "responses": [response.asdict() for response in self.responses],
            "report_sha256": self.report_sha256,
        }


def run_round27_ai_inference(
    *,
    panel: Round27AICasePanel,
    candidate: Round27AIHostCandidate,
    post_json: JsonPoster = _host._post_json,  # noqa: SLF001
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
    inventory_getter: InventoryGetter = _ai_runtime._get_json,  # noqa: SLF001
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    progress: ProgressCallback | None = None,
) -> Round27AIInferenceReport:
    """Run one exact candidate over an immutable target-free case panel."""

    selected_panel = panel.validated()
    if candidate not in POLYMARKET_ROUND27_AI_HOST_CANDIDATES:
        raise ValueError("Round 27 AI inference candidate differs")
    endpoint = f"{POLYMARKET_ROUND27_AI_BASE_URL}/api/generate"
    for item in POLYMARKET_ROUND27_AI_HOST_CANDIDATES:
        post_json(endpoint, _host._unload_request(item), 5.0)  # noqa: SLF001
    warmup_started = monotonic_ns()
    warmup_raw = post_json(
        endpoint,
        round27_ai_conformance_request(candidate, keep_alive="30s"),
        35.0,
    )
    warmup_wall_ms = max(1, math.ceil((monotonic_ns() - warmup_started) / 1_000_000))
    _host._parse_measurement(  # noqa: SLF001
        warmup_raw,
        candidate=candidate,
        phase="warmup",
        wall_seconds=warmup_wall_ms / 1_000,
    )
    residency = residency_inspector(
        POLYMARKET_ROUND27_AI_BASE_URL,
        candidate.runtime_model,
        2.0,
        expected_digest=candidate.runtime_digest,
    ).validated()
    inventory = inventory_getter(f"{POLYMARKET_ROUND27_AI_BASE_URL}/api/ps", 2.0)
    if not residency.fully_gpu_resident or not _inventory_has_only_candidate(
        inventory,
        candidate,
    ):
        raise RuntimeError("Round 27 AI candidate is not exclusively GPU resident")

    responses: list[Round27AIResponse] = []
    unload_failure: dict[str, str] | None = None
    unload_observed = False
    try:
        for index, case in enumerate(selected_panel.cases, start=1):
            prompt = round27_ai_case_prompt(case)
            prompt_sha256 = hashlib.sha256(prompt.encode("ascii")).hexdigest()
            started = monotonic_ns()
            try:
                raw = post_json(
                    endpoint,
                    round27_ai_ablation_request(case, candidate),
                    POLYMARKET_ROUND27_AI_CASE_TIMEOUT_SECONDS,
                )
                latency_ms = max(
                    1,
                    math.ceil((monotonic_ns() - started) / 1_000_000),
                )
                if latency_ms > math.ceil(
                    POLYMARKET_ROUND27_AI_CASE_TIMEOUT_SECONDS * 1_000
                ):
                    raise TimeoutError("Round 27 AI case exceeded its latency limit")
                response = _response_from_raw(
                    raw=raw,
                    candidate=candidate,
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
                    >= math.ceil(POLYMARKET_ROUND27_AI_CASE_TIMEOUT_SECONDS * 1_000)
                    else (
                        "provider_error"
                        if isinstance(exc, (OSError, RuntimeError))
                        else "invalid_response"
                    )
                )
                response = _failed_response(
                    candidate=candidate,
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
            post_json(endpoint, _host._unload_request(candidate), 5.0)  # noqa: SLF001
            unloaded = residency_inspector(
                POLYMARKET_ROUND27_AI_BASE_URL,
                candidate.runtime_model,
                2.0,
                expected_digest=candidate.runtime_digest,
            ).validated()
            unload_observed = unloaded.status == "unloaded"
            if not unload_observed:
                raise RuntimeError("Round 27 AI candidate remained loaded")
        except Exception as exc:  # noqa: BLE001 - cleanup failure is evidence
            unload_failure = {
                "type": type(exc).__name__,
                "message": str(exc)[:240],
            }
    counts = dict(sorted(Counter(response.status for response in responses).items()))
    changed = sum(response.abstains for response in responses)
    rejected_fraction = changed / len(responses) if responses else 0.0
    eligible = bool(
        len(responses) >= POLYMARKET_ROUND27_AI_MINIMUM_BASELINE_CANDIDATES
        and counts == {_VALID_STATUS: len(responses)}
        and changed >= POLYMARKET_ROUND27_AI_MINIMUM_CHANGED_ACTIONS
        and rejected_fraction <= POLYMARKET_ROUND27_AI_MAXIMUM_REJECTED_FRACTION
        and unload_observed
        and unload_failure is None
    )
    provisional = Round27AIInferenceReport(
        candidate=asdict(candidate),
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


credentials_used = False
account_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND27_AI_INFERENCE_REPORT_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_AI_RESPONSE_SCHEMA_VERSION",
    "Round27AIInferenceReport",
    "Round27AIResponse",
    "round27_ai_ablation_request",
    "round27_ai_ablation_response_schema",
    "run_round27_ai_inference",
]
