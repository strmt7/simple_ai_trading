"""Target-free, risk-reducing local AI advisories for Round 25."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from queue import Empty, Full, Queue
import re
import threading
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from .ai_runtime import OllamaResidencyReport, inspect_ollama_model_residency
from .polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_IDS,
)
from .polymarket_round25_evaluation import (
    POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256,
)
from .polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_CONDITION_DURATION_MS,
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V1_SHA256 = (
    "10c05777a06511e62aa335b8bab03f895235c85f314b09e338e63caf77cdd5aa"
)
POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V2_SHA256 = (
    "1b6e6c2977e0c2e5bfcc7dd3ddcfb6e3de03cd41b32f17658b8d6cf97fce3f27"
)
POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V3_SHA256 = (
    "baa6e7f65f1bca927d5dc35090f8d9ebd554704c689c022cb7e95749628ff036"
)
POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V4_SHA256 = (
    "c0f14340de46dba9726a6f86d4d7d5ab09d7368454114bb45bedd9cf0a114a31"
)
POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V5_SHA256 = (
    "3d8ea3e7cd77c483bd64b578f8e0893c0cceaa0e7b84adfcfd206d16480e62db"
)
POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256 = (
    "39e862da001f96dbad405c625062918cab65ac2382ef4105b3664e53f6cb2505"
)
POLYMARKET_ROUND25_AI_PACKET_SCHEMA_VERSION = (
    "polymarket-round25-target-free-ai-risk-packet-v1"
)
POLYMARKET_ROUND25_AI_ADVISORY_SCHEMA_VERSION = (
    "polymarket-round25-ai-risk-advisory-v2"
)
POLYMARKET_ROUND25_AI_TELEMETRY_SCHEMA_VERSION = (
    "polymarket-round25-ai-provider-telemetry-v1"
)
POLYMARKET_ROUND25_AI_RESULT_SCHEMA_VERSION = (
    "polymarket-round25-ai-risk-review-result-v2"
)
POLYMARKET_ROUND25_AI_MAXIMUM_PACKET_AGE_MS = 5_000
POLYMARKET_ROUND25_AI_RESPONSE_VALIDITY_MS = 10_000
POLYMARKET_ROUND25_AI_MAXIMUM_COOLDOWN_MS = 300_000
POLYMARKET_ROUND25_AI_MAXIMUM_PROVIDER_SECONDS = 10.0
POLYMARKET_ROUND25_AI_PRELOAD_SECONDS = 60.0
POLYMARKET_ROUND25_AI_MINIMUM_GPU_RESIDENCY_RATIO = 0.99
POLYMARKET_ROUND25_AI_REASON_CODES = (
    "no_additional_restriction",
    "ai_advisory_restriction",
    "weak_after_cost_margin",
    "model_market_disagreement",
    "liquidity_stress",
    "quote_stale",
    "transport_degraded",
    "regime_instability",
    "epistemic_uncertainty",
    "adverse_selection_risk",
    "inventory_risk",
    "risk_budget_tight",
    "size_reduction_required",
    "cooldown_required",
)
POLYMARKET_ROUND25_AI_RISK_ACTIONS = (
    "allow",
    "reduce_75",
    "reduce_50",
    "reduce_25",
    "veto",
    "cooldown_60s",
    "cooldown_300s",
)
POLYMARKET_ROUND25_AI_FAILURE_CODES = (
    "packet_stale",
    "model_identity_failure",
    "provider_failure",
    "schema_failure",
    "telemetry_failure",
    "residency_failure",
    "latency_failure",
    "response_stale",
    "pending_response",
    "queue_full",
    "worker_closed",
    "worker_failure",
)
_MODEL_CANDIDATE_IDS = frozenset(POLYMARKET_ROUND25_CANDIDATE_IDS[1:])
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_OLLAMA_VERSION = re.compile(r"^0\.32\.(\d+)$")
_MAXIMUM_JSON_BYTES = 2_000_000
_PROVIDER_DURATION_TOLERANCE_SECONDS = 1.0
_STOP = object()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


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


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Round 25 AI {name} is not numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"Round 25 AI {name} is not finite")
    return selected


def _bounded_number(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    selected = _finite_number(value, name=name)
    lower_failed = selected < minimum if minimum_inclusive else selected <= minimum
    upper_failed = selected > maximum if maximum_inclusive else selected >= maximum
    if lower_failed or upper_failed:
        raise ValueError(f"Round 25 AI {name} is outside its bound")
    return selected


def _printable_ascii(value: object, *, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise ValueError(f"Round 25 AI {name} length is invalid")
    if any(ord(character) < 32 or ord(character) > 126 for character in value):
        raise ValueError(f"Round 25 AI {name} is not printable ASCII")
    return value


@dataclass(frozen=True, slots=True)
class Round25AICandidateSpec:
    candidate_id: str
    model: str
    digest: str
    parameter_size: str
    quantization: str
    context_length: int
    upstream_revision: str

    def __post_init__(self) -> None:
        if (
            not self.candidate_id
            or not self.model
            or _SHA256.fullmatch(self.digest) is None
            or not self.parameter_size
            or not self.quantization
            or self.context_length < 8192
            or re.fullmatch(r"[0-9a-f]{40}", self.upstream_revision) is None
        ):
            raise ValueError("Round 25 AI candidate specification is invalid")


POLYMARKET_ROUND25_AI_REJECTED_RUNTIME_CANDIDATES = (
    Round25AICandidateSpec(
        candidate_id="qwen35-9b-risk-advisor-v1",
        model="qwen3.5:9b",
        digest="6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
        parameter_size="9.7B",
        quantization="Q4_K_M",
        context_length=262_144,
        upstream_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    ),
    Round25AICandidateSpec(
        candidate_id="fin-r1-8b-risk-advisor-v1",
        model="fin-r1:8b",
        digest="7a02f6045046a36f53f1541e6fe0ceaff202c2ca48a47c1292fc82e055a4a377",
        parameter_size="7.62B",
        quantization="Q6_K",
        context_length=32_768,
        upstream_revision="026768c4a015b591b54b240743edeac1de0970fa",
    ),
)
POLYMARKET_ROUND25_AI_CANDIDATES = (
    Round25AICandidateSpec(
        candidate_id="qwen3-4b-risk-advisor-v1",
        model="qwen3:4b",
        digest="e55aed6fe643f9368b2f48f8aaa56ec787b75765da69f794c0a0c23bfe7c64b2",
        parameter_size="4.0B",
        quantization="Q4_K_M",
        context_length=262_144,
        upstream_revision="1cfa9a7208912126459214e8b04321603b3df60c",
    ),
)


def round25_ai_candidate_spec(candidate_id: str) -> Round25AICandidateSpec:
    matches = tuple(
        candidate
        for candidate in (
            *POLYMARKET_ROUND25_AI_REJECTED_RUNTIME_CANDIDATES,
            *POLYMARKET_ROUND25_AI_CANDIDATES,
        )
        if candidate.candidate_id == candidate_id
    )
    if len(matches) != 1:
        raise ValueError("Round 25 AI candidate is not preregistered")
    return matches[0]


@dataclass(frozen=True, slots=True)
class Round25AIConfig:
    candidate_id: str
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = POLYMARKET_ROUND25_AI_MAXIMUM_PROVIDER_SECONDS
    seed: int = 25_025
    maximum_output_tokens: int = 24
    context_tokens: int = 4096
    keep_alive: str = "5m"

    @property
    def candidate(self) -> Round25AICandidateSpec:
        return round25_ai_candidate_spec(self.candidate_id)

    def validated(self) -> Round25AIConfig:
        candidate = self.candidate
        timeout = _finite_number(self.timeout_seconds, name="timeout")
        if (
            self.base_url != "http://127.0.0.1:11434"
            or candidate not in POLYMARKET_ROUND25_AI_CANDIDATES
            or not 0.1 <= timeout <= POLYMARKET_ROUND25_AI_MAXIMUM_PROVIDER_SECONDS
            or self.seed != 25_025
            or self.maximum_output_tokens != 24
            or self.context_tokens != 4096
            or self.context_tokens > candidate.context_length
            or self.keep_alive != "5m"
        ):
            raise ValueError("Round 25 AI configuration differs from contract")
        return self


@dataclass(frozen=True, slots=True)
class Round25AIAdvisoryPacket:
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    expires_at_ms: int
    feature_source_chain_sha256: str
    ml_candidate_id: str
    ml_artifact_sha256: str
    ml_prediction_sha256: str
    proposed_side: str
    model_probability_up: float
    market_prior_probability_up: float
    executable_entry_price: float
    conservative_edge_after_cost: float
    epistemic_uncertainty: float
    predicted_adverse_selection_probability: float
    relative_spread: float
    top_executable_notional_usd: float
    book_receipt_age_ms: float
    reference_receipt_age_ms: float
    transport_gap_count_60s: int
    realized_volatility_60s: float
    short_term_log_return_5s: float
    order_flow_imbalance_5s: float
    current_condition_exposure_fraction: float
    portfolio_risk_utilization: float
    deterministic_gate_sha256: str
    packet_sha256: str = ""
    schema_version: str = POLYMARKET_ROUND25_AI_PACKET_SCHEMA_VERSION
    model_design_sha256: str = POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
    candidate_design_sha256: str = POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
    candidate_amendment_sha256: str = (
        POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
    )
    evaluation_contract_sha256: str = (
        POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256
    )
    ai_contract_sha256: str = POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256
    target_accessed: bool = False
    outcome_accessed: bool = False
    resolution_accessed: bool = False
    credential_accessed: bool = False
    deterministic_entry_allowed: bool = True
    unknown_order_state: bool = False
    unknown_position_state: bool = False
    recovery_requalification_pending: bool = False
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "packet_sha256"
        }

    def prompt_payload(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "decision_time_ms": self.decision_time_ms,
            "expires_at_ms": self.expires_at_ms,
            "ml_proposal": {
                "candidate_id": self.ml_candidate_id,
                "proposed_side": self.proposed_side,
                "model_probability_up": self.model_probability_up,
                "market_prior_probability_up": self.market_prior_probability_up,
                "executable_entry_price": self.executable_entry_price,
                "conservative_edge_after_cost": self.conservative_edge_after_cost,
                "epistemic_uncertainty": self.epistemic_uncertainty,
                "predicted_adverse_selection_probability": (
                    self.predicted_adverse_selection_probability
                ),
            },
            "causal_market_state": {
                "relative_spread": self.relative_spread,
                "top_executable_notional_usd": self.top_executable_notional_usd,
                "book_receipt_age_ms": self.book_receipt_age_ms,
                "reference_receipt_age_ms": self.reference_receipt_age_ms,
                "transport_gap_count_60s": self.transport_gap_count_60s,
                "realized_volatility_60s": self.realized_volatility_60s,
                "short_term_log_return_5s": self.short_term_log_return_5s,
                "order_flow_imbalance_5s": self.order_flow_imbalance_5s,
            },
            "deterministic_risk_state": {
                "current_condition_exposure_fraction": (
                    self.current_condition_exposure_fraction
                ),
                "portfolio_risk_utilization": self.portfolio_risk_utilization,
                "entry_gate_passed": self.deterministic_entry_allowed,
                "unknown_order_state": self.unknown_order_state,
                "unknown_position_state": self.unknown_position_state,
                "recovery_requalification_pending": (
                    self.recovery_requalification_pending
                ),
            },
            "source_identity": {
                "feature_source_chain_sha256": self.feature_source_chain_sha256,
                "ml_artifact_sha256": self.ml_artifact_sha256,
                "ml_prediction_sha256": self.ml_prediction_sha256,
                "deterministic_gate_sha256": self.deterministic_gate_sha256,
                "packet_sha256": self.packet_sha256,
            },
        }

    def __post_init__(self) -> None:
        integers = (
            self.event_start_ms,
            self.decision_time_ms,
            self.expires_at_ms,
            self.transport_gap_count_60s,
        )
        hashes = (
            self.feature_source_chain_sha256,
            self.ml_artifact_sha256,
            self.ml_prediction_sha256,
            self.deterministic_gate_sha256,
        )
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or any(isinstance(value, bool) or not isinstance(value, int) for value in integers)
            or self.event_start_ms < 0
            or not self.event_start_ms <= self.decision_time_ms
            < self.event_start_ms + POLYMARKET_ROUND25_CONDITION_DURATION_MS
            or self.expires_at_ms
            != self.decision_time_ms + POLYMARKET_ROUND25_AI_RESPONSE_VALIDITY_MS
            or any(_SHA256.fullmatch(value) is None for value in hashes)
            or self.ml_candidate_id not in _MODEL_CANDIDATE_IDS
            or self.proposed_side not in {"up", "down"}
            or self.transport_gap_count_60s < 0
            or self.transport_gap_count_60s > 1000
            or self.schema_version != POLYMARKET_ROUND25_AI_PACKET_SCHEMA_VERSION
            or self.model_design_sha256 != POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
            or self.candidate_design_sha256
            != POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
            or self.candidate_amendment_sha256
            != POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
            or self.evaluation_contract_sha256
            != POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256
            or self.ai_contract_sha256
            != POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256
            or any(
                value is not False
                for value in (
                    self.target_accessed,
                    self.outcome_accessed,
                    self.resolution_accessed,
                    self.credential_accessed,
                    self.unknown_order_state,
                    self.unknown_position_state,
                    self.recovery_requalification_pending,
                    self.trading_authority,
                )
            )
            or self.deterministic_entry_allowed is not True
        ):
            raise ValueError("Round 25 AI packet identity or safety state differs")
        bounds = (
            (self.model_probability_up, "model probability", 0.0, 1.0, False, False),
            (self.market_prior_probability_up, "market prior", 0.0, 1.0, False, False),
            (self.executable_entry_price, "entry price", 0.0, 1.0, False, False),
            (self.conservative_edge_after_cost, "after-cost edge", -1.0, 1.0, True, True),
            (self.epistemic_uncertainty, "epistemic uncertainty", 0.0, 1.0, True, True),
            (
                self.predicted_adverse_selection_probability,
                "adverse-selection probability",
                0.0,
                1.0,
                True,
                True,
            ),
            (self.relative_spread, "relative spread", 0.0, 1.0, True, False),
            (
                self.top_executable_notional_usd,
                "top executable notional",
                0.0,
                1_000_000_000.0,
                True,
                True,
            ),
            (self.book_receipt_age_ms, "book age", 0.0, 5000.0, True, True),
            (
                self.reference_receipt_age_ms,
                "reference age",
                0.0,
                5000.0,
                True,
                True,
            ),
            (
                self.realized_volatility_60s,
                "realized volatility",
                0.0,
                10.0,
                True,
                True,
            ),
            (
                self.short_term_log_return_5s,
                "short-term return",
                -1.0,
                1.0,
                True,
                True,
            ),
            (
                self.order_flow_imbalance_5s,
                "order-flow imbalance",
                -1.0,
                1.0,
                True,
                True,
            ),
            (
                self.current_condition_exposure_fraction,
                "condition exposure",
                0.0,
                1.0,
                True,
                True,
            ),
            (
                self.portfolio_risk_utilization,
                "risk utilization",
                0.0,
                1.0,
                True,
                True,
            ),
        )
        for value, name, minimum, maximum, lower, upper in bounds:
            _bounded_number(
                value,
                name=name,
                minimum=minimum,
                maximum=maximum,
                minimum_inclusive=lower,
                maximum_inclusive=upper,
            )
        expected = _canonical_sha256(self.identity_payload())
        if not self.packet_sha256:
            object.__setattr__(self, "packet_sha256", expected)
        elif self.packet_sha256 != expected:
            raise ValueError("Round 25 AI packet hash differs")

    def validated(self) -> Round25AIAdvisoryPacket:
        self.__post_init__()
        return self


def _coherent_response(
    *,
    veto_new_entries: bool,
    maximum_size_multiplier: float,
    cooldown_ms: int,
    reason_codes: Sequence[str],
) -> bool:
    reasons = set(reason_codes)
    no_restriction = "no_additional_restriction" in reasons
    size_reduction = "size_reduction_required" in reasons
    cooldown = "cooldown_required" in reasons
    if (maximum_size_multiplier == 0.0) is not veto_new_entries:
        return False
    if maximum_size_multiplier == 1.0:
        return (
            not veto_new_entries
            and cooldown_ms == 0
            and tuple(reason_codes) == ("no_additional_restriction",)
        )
    if no_restriction or not size_reduction:
        return False
    if cooldown_ms > 0:
        return veto_new_entries and cooldown and maximum_size_multiplier == 0.0
    return not cooldown


@dataclass(frozen=True, slots=True)
class Round25AIAdvisory:
    candidate_id: str
    model: str
    model_digest: str
    packet_sha256: str
    generated_at_ms: int
    expires_at_ms: int
    veto_new_entries: bool
    maximum_size_multiplier: float
    cooldown_ms: int
    reason_codes: tuple[str, ...]
    summary: str
    valid_model_response: bool
    failure_code: str | None
    advisory_sha256: str = ""
    schema_version: str = POLYMARKET_ROUND25_AI_ADVISORY_SCHEMA_VERSION
    ai_contract_sha256: str = POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256
    can_create_entry: bool = False
    can_increase_risk: bool = False
    can_change_side: bool = False
    can_block_exit: bool = False
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "advisory_sha256"
        }

    def __post_init__(self) -> None:
        candidate = round25_ai_candidate_spec(self.candidate_id)
        if (
            self.model != candidate.model
            or self.model_digest != candidate.digest
            or _SHA256.fullmatch(self.packet_sha256) is None
            or isinstance(self.generated_at_ms, bool)
            or not isinstance(self.generated_at_ms, int)
            or self.generated_at_ms < 0
            or isinstance(self.expires_at_ms, bool)
            or not isinstance(self.expires_at_ms, int)
            or self.expires_at_ms < 0
            or not isinstance(self.veto_new_entries, bool)
            or isinstance(self.cooldown_ms, bool)
            or not isinstance(self.cooldown_ms, int)
            or not 0 <= self.cooldown_ms <= POLYMARKET_ROUND25_AI_MAXIMUM_COOLDOWN_MS
            or not self.reason_codes
            or len(set(self.reason_codes)) != len(self.reason_codes)
            or self.schema_version != POLYMARKET_ROUND25_AI_ADVISORY_SCHEMA_VERSION
            or self.ai_contract_sha256
            != POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256
            or any(
                value is not False
                for value in (
                    self.can_create_entry,
                    self.can_increase_risk,
                    self.can_change_side,
                    self.can_block_exit,
                    self.trading_authority,
                )
            )
        ):
            raise ValueError("Round 25 AI advisory identity differs")
        multiplier = _bounded_number(
            self.maximum_size_multiplier,
            name="size multiplier",
            minimum=0.0,
            maximum=1.0,
        )
        _printable_ascii(self.summary, name="summary", maximum=160)
        if self.valid_model_response:
            if (
                self.failure_code is not None
                or self.generated_at_ms > self.expires_at_ms
                or any(code not in POLYMARKET_ROUND25_AI_REASON_CODES for code in self.reason_codes)
                or tuple(
                    sorted(
                        self.reason_codes,
                        key=POLYMARKET_ROUND25_AI_REASON_CODES.index,
                    )
                )
                != self.reason_codes
                or not _coherent_response(
                    veto_new_entries=self.veto_new_entries,
                    maximum_size_multiplier=multiplier,
                    cooldown_ms=self.cooldown_ms,
                    reason_codes=self.reason_codes,
                )
            ):
                raise ValueError("Round 25 AI model advisory is incoherent")
        elif (
            self.failure_code not in POLYMARKET_ROUND25_AI_FAILURE_CODES
            or self.reason_codes != (self.failure_code,)
            or self.veto_new_entries is not True
            or multiplier != 0.0
            or self.cooldown_ms != 0
        ):
            raise ValueError("Round 25 AI failure advisory is not fail-closed")
        expected = _canonical_sha256(self.identity_payload())
        if not self.advisory_sha256:
            object.__setattr__(self, "advisory_sha256", expected)
        elif self.advisory_sha256 != expected:
            raise ValueError("Round 25 AI advisory hash differs")

    def validated(self) -> Round25AIAdvisory:
        self.__post_init__()
        return self


@dataclass(frozen=True, slots=True)
class Round25AIProviderTelemetry:
    candidate_id: str
    model: str
    model_digest: str
    ollama_version: str
    show_metadata_sha256: str
    prompt_sha256: str
    response_sha256: str
    measured_latency_seconds: float
    provider_total_duration_ns: int
    provider_load_duration_ns: int
    provider_prompt_eval_count: int
    provider_prompt_eval_duration_ns: int
    provider_eval_count: int
    provider_eval_duration_ns: int
    residency: OllamaResidencyReport
    telemetry_sha256: str = ""
    schema_version: str = POLYMARKET_ROUND25_AI_TELEMETRY_SCHEMA_VERSION

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["residency"] = self.residency.asdict()
        payload.pop("telemetry_sha256")
        return payload

    def __post_init__(self) -> None:
        candidate = round25_ai_candidate_spec(self.candidate_id)
        durations = (
            self.provider_total_duration_ns,
            self.provider_load_duration_ns,
            self.provider_prompt_eval_duration_ns,
            self.provider_eval_duration_ns,
        )
        counts = (self.provider_prompt_eval_count, self.provider_eval_count)
        latency = _finite_number(self.measured_latency_seconds, name="latency")
        report = self.residency.validated()
        if (
            self.model != candidate.model
            or self.model_digest != candidate.digest
            or _OLLAMA_VERSION.fullmatch(self.ollama_version) is None
            or int(_OLLAMA_VERSION.fullmatch(self.ollama_version).group(1)) < 4
            or any(_SHA256.fullmatch(value) is None for value in (
                self.show_metadata_sha256,
                self.prompt_sha256,
                self.response_sha256,
            ))
            or not 0.0 <= latency <= POLYMARKET_ROUND25_AI_MAXIMUM_PROVIDER_SECONDS
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in durations)
            or self.provider_total_duration_ns <= 0
            or self.provider_prompt_eval_duration_ns <= 0
            or self.provider_eval_duration_ns <= 0
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts)
            or any(value > self.provider_total_duration_ns for value in durations[1:])
            or self.provider_total_duration_ns / 1_000_000_000.0
            > latency + _PROVIDER_DURATION_TOLERANCE_SECONDS
            or report.requested_model != candidate.model
            or report.digest != candidate.digest
            or not report.fully_gpu_resident
            or report.vram_to_model_ratio is None
            or report.vram_to_model_ratio
            < POLYMARKET_ROUND25_AI_MINIMUM_GPU_RESIDENCY_RATIO
            or self.schema_version != POLYMARKET_ROUND25_AI_TELEMETRY_SCHEMA_VERSION
        ):
            raise ValueError("Round 25 AI provider telemetry differs")
        expected = _canonical_sha256(self.identity_payload())
        if not self.telemetry_sha256:
            object.__setattr__(self, "telemetry_sha256", expected)
        elif self.telemetry_sha256 != expected:
            raise ValueError("Round 25 AI provider telemetry hash differs")

    def validated(self) -> Round25AIProviderTelemetry:
        self.__post_init__()
        return self


@dataclass(frozen=True, slots=True)
class Round25AIReviewResult:
    packet_sha256: str
    advisory: Round25AIAdvisory
    telemetry: Round25AIProviderTelemetry | None
    result_sha256: str = ""
    schema_version: str = POLYMARKET_ROUND25_AI_RESULT_SCHEMA_VERSION
    target_accessed: bool = False
    outcome_accessed: bool = False
    resolution_accessed: bool = False
    edge_verified: bool = False
    profitability_verified: bool = False
    ai_uplift_verified: bool = False
    paper_authority: bool = False
    live_authority: bool = False
    order_submitted: bool = False

    def identity_payload(self) -> dict[str, object]:
        payload = {
            key: value
            for key, value in asdict(self).items()
            if key not in {"advisory", "telemetry", "result_sha256"}
        }
        payload["advisory"] = self.advisory.identity_payload() | {
            "advisory_sha256": self.advisory.advisory_sha256
        }
        payload["telemetry"] = (
            None
            if self.telemetry is None
            else self.telemetry.identity_payload()
            | {"telemetry_sha256": self.telemetry.telemetry_sha256}
        )
        return payload

    def __post_init__(self) -> None:
        advisory = self.advisory.validated()
        telemetry = None if self.telemetry is None else self.telemetry.validated()
        if (
            _SHA256.fullmatch(self.packet_sha256) is None
            or advisory.packet_sha256 != self.packet_sha256
            or (advisory.valid_model_response is not (telemetry is not None))
            or (
                telemetry is not None
                and (
                    telemetry.candidate_id != advisory.candidate_id
                    or telemetry.model != advisory.model
                    or telemetry.model_digest != advisory.model_digest
                )
            )
            or self.schema_version != POLYMARKET_ROUND25_AI_RESULT_SCHEMA_VERSION
            or any(
                value is not False
                for value in (
                    self.target_accessed,
                    self.outcome_accessed,
                    self.resolution_accessed,
                    self.edge_verified,
                    self.profitability_verified,
                    self.ai_uplift_verified,
                    self.paper_authority,
                    self.live_authority,
                    self.order_submitted,
                )
            )
        ):
            raise ValueError("Round 25 AI review result differs")
        expected = _canonical_sha256(self.identity_payload())
        if not self.result_sha256:
            object.__setattr__(self, "result_sha256", expected)
        elif self.result_sha256 != expected:
            raise ValueError("Round 25 AI review result hash differs")

    def validated(self) -> Round25AIReviewResult:
        self.__post_init__()
        return self


_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "risk_action": {
            "type": "string",
            "enum": list(POLYMARKET_ROUND25_AI_RISK_ACTIONS),
        },
    },
    "required": ["risk_action"],
}

_RISK_ACTION_POLICY: dict[
    str, tuple[bool, float, int, tuple[str, ...], str]
] = {
    "allow": (
        False,
        1.0,
        0,
        ("no_additional_restriction",),
        "AI requested no additional entry restriction.",
    ),
    "reduce_75": (
        False,
        0.75,
        0,
        ("ai_advisory_restriction", "size_reduction_required"),
        "AI limited the proposed entry to 75 percent size.",
    ),
    "reduce_50": (
        False,
        0.5,
        0,
        ("ai_advisory_restriction", "size_reduction_required"),
        "AI limited the proposed entry to 50 percent size.",
    ),
    "reduce_25": (
        False,
        0.25,
        0,
        ("ai_advisory_restriction", "size_reduction_required"),
        "AI limited the proposed entry to 25 percent size.",
    ),
    "veto": (
        True,
        0.0,
        0,
        ("ai_advisory_restriction", "size_reduction_required"),
        "AI vetoed this proposed entry.",
    ),
    "cooldown_60s": (
        True,
        0.0,
        60_000,
        (
            "ai_advisory_restriction",
            "size_reduction_required",
            "cooldown_required",
        ),
        "AI vetoed entry and requested a 60 second cooldown.",
    ),
    "cooldown_300s": (
        True,
        0.0,
        300_000,
        (
            "ai_advisory_restriction",
            "size_reduction_required",
            "cooldown_required",
        ),
        "AI vetoed entry and requested a 300 second cooldown.",
    ),
}


JsonGetter = Callable[[str, float], object]
JsonPoster = Callable[[str, Mapping[str, object], float], object]
ResidencyInspector = Callable[..., OllamaResidencyReport]
WallClock = Callable[[], int]
MonotonicClock = Callable[[], int]


def _read_json_response(response: Any) -> object:
    body = response.read(_MAXIMUM_JSON_BYTES + 1)
    if len(body) > _MAXIMUM_JSON_BYTES:
        raise ValueError("Ollama response exceeds the byte limit")
    try:
        return _strict_json_value(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Ollama response is invalid JSON") from exc


def _get_json(url: str, timeout: float) -> object:
    request = urllib_request.Request(url, method="GET")
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:  # nosec B310 - contract fixes loopback URL
            return _read_json_response(response)
    except (OSError, urllib_error.URLError) as exc:
        raise ValueError("Ollama GET request failed") from exc


def _post_json(url: str, payload: Mapping[str, object], timeout: float) -> object:
    body = _canonical_json(payload).encode("ascii")
    request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:  # nosec B310 - contract fixes loopback URL
            return _read_json_response(response)
    except (OSError, urllib_error.URLError) as exc:
        raise ValueError("Ollama POST request failed") from exc


def _normalized_model(value: object) -> str:
    selected = str(value or "").strip().lower()
    return selected if ":" in selected else f"{selected}:latest"


def preflight_round25_ai_candidate(
    config: Round25AIConfig,
    *,
    get_json: JsonGetter = _get_json,
    post_json: JsonPoster = _post_json,
) -> tuple[str, str]:
    cfg = config.validated()
    candidate = cfg.candidate
    version_payload = get_json(f"{cfg.base_url}/api/version", cfg.timeout_seconds)
    if not isinstance(version_payload, Mapping) or set(version_payload) != {"version"}:
        raise ValueError("Round 25 AI Ollama version response differs")
    version = str(version_payload["version"] or "")
    match = _OLLAMA_VERSION.fullmatch(version)
    if match is None or int(match.group(1)) < 4:
        raise ValueError("Round 25 AI Ollama version is outside the frozen line")
    tags = get_json(f"{cfg.base_url}/api/tags", cfg.timeout_seconds)
    if not isinstance(tags, Mapping) or not isinstance(tags.get("models"), list):
        raise ValueError("Round 25 AI model inventory differs")
    matches = []
    for raw in tags["models"]:
        if not isinstance(raw, Mapping):
            raise ValueError("Round 25 AI model inventory entry differs")
        names = {_normalized_model(raw.get("name")), _normalized_model(raw.get("model"))}
        if _normalized_model(candidate.model) in names:
            matches.append(raw)
    if len(matches) != 1 or matches[0].get("digest") != candidate.digest:
        raise ValueError("Round 25 AI local model digest differs")
    details = matches[0].get("details")
    if (
        not isinstance(details, Mapping)
        or details.get("format") != "gguf"
        or details.get("parameter_size") != candidate.parameter_size
    ):
        raise ValueError("Round 25 AI local model inventory metadata differs")
    show = post_json(
        f"{cfg.base_url}/api/show",
        {"model": candidate.model, "verbose": False},
        cfg.timeout_seconds,
    )
    if not isinstance(show, Mapping):
        raise ValueError("Round 25 AI model details response differs")
    show_details = show.get("details")
    model_info = show.get("model_info")
    if (
        not isinstance(show_details, Mapping)
        or show_details.get("format") != "gguf"
        or show_details.get("parameter_size") != candidate.parameter_size
        or not isinstance(model_info, Mapping)
        or int(model_info.get("general.parameter_count") or 0) < 2_000_000_000
    ):
        raise ValueError("Round 25 AI model details metadata differs")
    return version, _canonical_sha256(show)


def _prompt(packet: Round25AIAdvisoryPacket) -> str:
    instructions = (
        "Review one proposed BTC 5m entry using only this target-free packet. Select "
        "one schema risk_action. Never create or increase risk, change side, or affect "
        "exits. Risk increases with spread, receipt age, transport gaps, volatility, "
        "epistemic uncertainty, adverse selection, exposure, and portfolio utilization, "
        "and increases when executable depth falls. Never become less restrictive as "
        "risk worsens. Combined severe risks require veto or cooldown. Use allow only "
        "when the packet is benign. Return only schema JSON. Packet:"
    )
    return f"{instructions}{_canonical_json(packet.prompt_payload())}"


def _parse_model_response(value: object) -> tuple[bool, float, int, tuple[str, ...], str]:
    if not isinstance(value, str) or len(value.encode("utf-8")) > 20_000:
        raise ValueError("Round 25 AI model response text differs")
    parsed = _strict_json_value(value)
    required = {"risk_action"}
    if not isinstance(parsed, Mapping) or set(parsed) != required:
        raise ValueError("Round 25 AI response fields differ")
    action = parsed["risk_action"]
    if not isinstance(action, str) or action not in _RISK_ACTION_POLICY:
        raise ValueError("Round 25 AI risk action differs")
    veto, multiplier, cooldown, reasons, summary = _RISK_ACTION_POLICY[action]
    if not _coherent_response(
        veto_new_entries=veto,
        maximum_size_multiplier=multiplier,
        cooldown_ms=cooldown,
        reason_codes=reasons,
    ):
        raise ValueError("Round 25 AI response is incoherent")
    return veto, multiplier, cooldown, reasons, summary


def _provider_usage(payload: object) -> dict[str, int | str]:
    if not isinstance(payload, Mapping):
        raise ValueError("Round 25 AI provider response differs")
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
    if not required.issubset(payload):
        raise ValueError("Round 25 AI provider telemetry fields differ")
    integers = {
        key: payload[key]
        for key in (
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        )
    }
    if (
        payload["done"] is not True
        or payload["done_reason"] != "stop"
        or any(isinstance(value, bool) or not isinstance(value, int) for value in integers.values())
        or integers["total_duration"] <= 0
        or integers["load_duration"] < 0
        or integers["prompt_eval_count"] <= 0
        or integers["prompt_eval_duration"] <= 0
        or integers["eval_count"] <= 0
        or integers["eval_duration"] <= 0
        or any(
            integers[key] > integers["total_duration"]
            for key in ("load_duration", "prompt_eval_duration", "eval_duration")
        )
    ):
        raise ValueError("Round 25 AI provider usage is incoherent")
    return {"model": str(payload["model"]), **integers}


def _failure_advisory(
    *,
    config: Round25AIConfig,
    packet: Round25AIAdvisoryPacket,
    failure_code: str,
    generated_at_ms: int,
) -> Round25AIAdvisory:
    if failure_code not in POLYMARKET_ROUND25_AI_FAILURE_CODES:
        raise ValueError("Round 25 AI failure code differs")
    candidate = config.candidate
    return Round25AIAdvisory(
        candidate_id=candidate.candidate_id,
        model=candidate.model,
        model_digest=candidate.digest,
        packet_sha256=packet.packet_sha256,
        generated_at_ms=max(0, int(generated_at_ms)),
        expires_at_ms=packet.expires_at_ms,
        veto_new_entries=True,
        maximum_size_multiplier=0.0,
        cooldown_ms=0,
        reason_codes=(failure_code,),
        summary="AI unavailable or invalid; deterministic entry veto remains active.",
        valid_model_response=False,
        failure_code=failure_code,
    )


def _failure_result(
    *,
    config: Round25AIConfig,
    packet: Round25AIAdvisoryPacket,
    failure_code: str,
    generated_at_ms: int,
) -> Round25AIReviewResult:
    return Round25AIReviewResult(
        packet_sha256=packet.packet_sha256,
        advisory=_failure_advisory(
            config=config,
            packet=packet,
            failure_code=failure_code,
            generated_at_ms=generated_at_ms,
        ),
        telemetry=None,
    )


def review_round25_ai_packet(
    packet: Round25AIAdvisoryPacket,
    config: Round25AIConfig,
    *,
    get_json: JsonGetter = _get_json,
    post_json: JsonPoster = _post_json,
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
    wall_clock_ms: WallClock = lambda: time.time_ns() // 1_000_000,
    monotonic_ns: MonotonicClock = time.perf_counter_ns,
) -> Round25AIReviewResult:
    selected = packet.validated()
    cfg = config.validated()
    submitted_at_ms = int(wall_clock_ms())
    if (
        submitted_at_ms < selected.decision_time_ms
        or submitted_at_ms - selected.decision_time_ms
        > POLYMARKET_ROUND25_AI_MAXIMUM_PACKET_AGE_MS
        or submitted_at_ms > selected.expires_at_ms
    ):
        return _failure_result(
            config=cfg,
            packet=selected,
            failure_code="packet_stale",
            generated_at_ms=submitted_at_ms,
        )
    try:
        version, metadata_sha256 = preflight_round25_ai_candidate(
            cfg,
            get_json=get_json,
            post_json=post_json,
        )
    except (OSError, TypeError, ValueError):
        return _failure_result(
            config=cfg,
            packet=selected,
            failure_code="model_identity_failure",
            generated_at_ms=int(wall_clock_ms()),
        )
    prompt = _prompt(selected)
    request_payload = {
        "model": cfg.candidate.model,
        "prompt": prompt,
        "system": (
            "You are a conservative local risk-control reviewer. You may only "
            "veto or reduce a proposed entry. Never create risk or affect exits."
        ),
        "format": _RESPONSE_SCHEMA,
        "stream": False,
        "think": False,
        "keep_alive": cfg.keep_alive,
        "options": {
            "temperature": 0.0,
            "seed": cfg.seed,
            "num_predict": cfg.maximum_output_tokens,
            "num_ctx": cfg.context_tokens,
        },
    }
    started_ns = int(monotonic_ns())
    try:
        provider = post_json(
            f"{cfg.base_url}/api/generate",
            request_payload,
            cfg.timeout_seconds,
        )
    except (OSError, TypeError, ValueError):
        return _failure_result(
            config=cfg,
            packet=selected,
            failure_code="provider_failure",
            generated_at_ms=int(wall_clock_ms()),
        )
    finished_ns = int(monotonic_ns())
    measured_latency = max(0.0, (finished_ns - started_ns) / 1_000_000_000.0)
    generated_at_ms = int(wall_clock_ms())
    try:
        usage = _provider_usage(provider)
        if _normalized_model(usage["model"]) != _normalized_model(cfg.candidate.model):
            raise ValueError("Round 25 AI provider returned another model")
        if measured_latency > POLYMARKET_ROUND25_AI_MAXIMUM_PROVIDER_SECONDS:
            raise TimeoutError("Round 25 AI provider latency exceeded")
        if (
            int(usage["total_duration"]) / 1_000_000_000.0
            > measured_latency + _PROVIDER_DURATION_TOLERANCE_SECONDS
        ):
            raise ValueError("Round 25 AI provider duration exceeds wall clock")
    except TimeoutError:
        return _failure_result(
            config=cfg,
            packet=selected,
            failure_code="latency_failure",
            generated_at_ms=generated_at_ms,
        )
    except (TypeError, ValueError):
        return _failure_result(
            config=cfg,
            packet=selected,
            failure_code="telemetry_failure",
            generated_at_ms=generated_at_ms,
        )
    if generated_at_ms > selected.expires_at_ms:
        return _failure_result(
            config=cfg,
            packet=selected,
            failure_code="response_stale",
            generated_at_ms=generated_at_ms,
        )
    try:
        assert isinstance(provider, Mapping)
        parsed = _parse_model_response(provider.get("response"))
    except (AssertionError, json.JSONDecodeError, TypeError, ValueError):
        return _failure_result(
            config=cfg,
            packet=selected,
            failure_code="schema_failure",
            generated_at_ms=generated_at_ms,
        )
    try:
        residency = residency_inspector(
            cfg.base_url,
            cfg.candidate.model,
            min(2.0, cfg.timeout_seconds),
            expected_digest=cfg.candidate.digest,
        ).validated()
        if (
            residency.digest != cfg.candidate.digest
            or not residency.fully_gpu_resident
            or residency.vram_to_model_ratio is None
            or residency.vram_to_model_ratio
            < POLYMARKET_ROUND25_AI_MINIMUM_GPU_RESIDENCY_RATIO
        ):
            raise ValueError("Round 25 AI model is not fully GPU resident")
    except (OSError, TypeError, ValueError):
        return _failure_result(
            config=cfg,
            packet=selected,
            failure_code="residency_failure",
            generated_at_ms=generated_at_ms,
        )
    veto, multiplier, cooldown, reasons, summary = parsed
    advisory = Round25AIAdvisory(
        candidate_id=cfg.candidate.candidate_id,
        model=cfg.candidate.model,
        model_digest=cfg.candidate.digest,
        packet_sha256=selected.packet_sha256,
        generated_at_ms=generated_at_ms,
        expires_at_ms=selected.expires_at_ms,
        veto_new_entries=veto,
        maximum_size_multiplier=multiplier,
        cooldown_ms=cooldown,
        reason_codes=reasons,
        summary=summary,
        valid_model_response=True,
        failure_code=None,
    )
    telemetry = Round25AIProviderTelemetry(
        candidate_id=cfg.candidate.candidate_id,
        model=cfg.candidate.model,
        model_digest=cfg.candidate.digest,
        ollama_version=version,
        show_metadata_sha256=metadata_sha256,
        prompt_sha256=hashlib.sha256(prompt.encode("ascii")).hexdigest(),
        response_sha256=hashlib.sha256(
            str(provider["response"]).encode("utf-8")
        ).hexdigest(),
        measured_latency_seconds=measured_latency,
        provider_total_duration_ns=int(usage["total_duration"]),
        provider_load_duration_ns=int(usage["load_duration"]),
        provider_prompt_eval_count=int(usage["prompt_eval_count"]),
        provider_prompt_eval_duration_ns=int(usage["prompt_eval_duration"]),
        provider_eval_count=int(usage["eval_count"]),
        provider_eval_duration_ns=int(usage["eval_duration"]),
        residency=residency,
    )
    return Round25AIReviewResult(
        packet_sha256=selected.packet_sha256,
        advisory=advisory,
        telemetry=telemetry,
    )


def preload_round25_ai_candidate(
    config: Round25AIConfig,
    *,
    post_json: JsonPoster = _post_json,
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
) -> OllamaResidencyReport:
    cfg = config.validated()
    response = post_json(
        f"{cfg.base_url}/api/generate",
        {
            "model": cfg.candidate.model,
            "prompt": "",
            "stream": False,
            "keep_alive": cfg.keep_alive,
            "options": {"num_ctx": cfg.context_tokens},
        },
        POLYMARKET_ROUND25_AI_PRELOAD_SECONDS,
    )
    if not isinstance(response, Mapping) or response.get("done") is not True:
        raise ValueError("Round 25 AI model preload failed")
    report = residency_inspector(
        cfg.base_url,
        cfg.candidate.model,
        min(2.0, cfg.timeout_seconds),
        expected_digest=cfg.candidate.digest,
    ).validated()
    if (
        report.digest != cfg.candidate.digest
        or not report.fully_gpu_resident
        or report.vram_to_model_ratio is None
        or report.vram_to_model_ratio
        < POLYMARKET_ROUND25_AI_MINIMUM_GPU_RESIDENCY_RATIO
    ):
        raise ValueError("Round 25 AI preload is not fully GPU resident")
    return report


def unload_round25_ai_candidate(
    config: Round25AIConfig,
    *,
    post_json: JsonPoster = _post_json,
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
) -> None:
    cfg = config.validated()
    response = post_json(
        f"{cfg.base_url}/api/generate",
        {"model": cfg.candidate.model, "keep_alive": 0, "stream": False},
        cfg.timeout_seconds,
    )
    if (
        not isinstance(response, Mapping)
        or response.get("done") is not True
        or response.get("done_reason") != "unload"
    ):
        raise ValueError("Round 25 AI model unload failed")
    report = residency_inspector(
        cfg.base_url,
        cfg.candidate.model,
        min(2.0, cfg.timeout_seconds),
        expected_digest=cfg.candidate.digest,
    ).validated()
    if report.loaded:
        raise ValueError("Round 25 AI model remained loaded")


Reviewer = Callable[[Round25AIAdvisoryPacket], Round25AIReviewResult]


class Round25AIAdvisoryWorker:
    """Capacity-one daemon worker; the caller never waits for model inference."""

    def __init__(
        self,
        *,
        config: Round25AIConfig,
        reviewer: Reviewer,
        wall_clock_ms: WallClock = lambda: time.time_ns() // 1_000_000,
    ) -> None:
        self._config = config.validated()
        self._reviewer = reviewer
        self._wall_clock_ms = wall_clock_ms
        self._queue: Queue[object] = Queue(maxsize=1)
        self._lock = threading.Lock()
        self._closed = False
        self._inflight: str | None = None
        self._results: dict[str, Round25AIReviewResult] = {}
        self._thread = threading.Thread(
            target=self._run,
            name=f"round25-ai-{self._config.candidate_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def thread_is_daemon(self) -> bool:
        return self._thread.daemon

    def submit(self, packet: Round25AIAdvisoryPacket) -> bool:
        selected = packet.validated()
        with self._lock:
            if (
                self._closed
                or self._inflight is not None
                or selected.packet_sha256 in self._results
            ):
                return False
            self._inflight = selected.packet_sha256
        try:
            self._queue.put_nowait(selected)
        except Full:
            with self._lock:
                self._inflight = None
            return False
        return True

    def poll(self, packet_sha256: str) -> Round25AIReviewResult | None:
        if _SHA256.fullmatch(packet_sha256) is None:
            raise ValueError("Round 25 AI poll packet hash differs")
        with self._lock:
            return self._results.pop(packet_sha256, None)

    def advisory_or_fail_closed(
        self,
        packet: Round25AIAdvisoryPacket,
    ) -> Round25AIAdvisory:
        selected = packet.validated()
        result = self.poll(selected.packet_sha256)
        now_ms = int(self._wall_clock_ms())
        if result is not None:
            if now_ms <= selected.expires_at_ms:
                return result.advisory
            return _failure_advisory(
                config=self._config,
                packet=selected,
                failure_code="response_stale",
                generated_at_ms=now_ms,
            )
        with self._lock:
            closed = self._closed
        return _failure_advisory(
            config=self._config,
            packet=selected,
            failure_code="worker_closed" if closed else "pending_response",
            generated_at_ms=now_ms,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._queue.put_nowait(_STOP)
        except Full:
            pass

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.25)
            except Empty:
                with self._lock:
                    if self._closed:
                        return
                continue
            if item is _STOP:
                return
            assert isinstance(item, Round25AIAdvisoryPacket)
            try:
                result = self._reviewer(item).validated()
                if result.packet_sha256 != item.packet_sha256:
                    raise ValueError("Round 25 AI worker result packet differs")
            except Exception:  # noqa: BLE001 - worker must contain every reviewer fault
                result = _failure_result(
                    config=self._config,
                    packet=item,
                    failure_code="worker_failure",
                    generated_at_ms=int(self._wall_clock_ms()),
                )
            with self._lock:
                self._results.clear()
                self._results[item.packet_sha256] = result
                self._inflight = None
                closed = self._closed
            if closed:
                return


__all__ = [
    "POLYMARKET_ROUND25_AI_ADVISORY_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_AI_CANDIDATES",
    "POLYMARKET_ROUND25_AI_FAILURE_CODES",
    "POLYMARKET_ROUND25_AI_PACKET_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_AI_PRELOAD_SECONDS",
    "POLYMARKET_ROUND25_AI_REASON_CODES",
    "POLYMARKET_ROUND25_AI_REJECTED_RUNTIME_CANDIDATES",
    "POLYMARKET_ROUND25_AI_RESPONSE_VALIDITY_MS",
    "POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V1_SHA256",
    "POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V2_SHA256",
    "POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_V3_SHA256",
    "POLYMARKET_ROUND25_AI_RESULT_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_AI_TELEMETRY_SCHEMA_VERSION",
    "Round25AIAdvisory",
    "Round25AIAdvisoryPacket",
    "Round25AIAdvisoryWorker",
    "Round25AICandidateSpec",
    "Round25AIConfig",
    "Round25AIProviderTelemetry",
    "Round25AIReviewResult",
    "preflight_round25_ai_candidate",
    "preload_round25_ai_candidate",
    "review_round25_ai_packet",
    "round25_ai_candidate_spec",
    "unload_round25_ai_candidate",
]
