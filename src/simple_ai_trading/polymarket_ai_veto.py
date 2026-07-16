"""Local multibillion-parameter AI veto ablation for Polymarket proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import time
from typing import Callable, Mapping, Protocol, Sequence
from urllib.request import Request, urlopen

from .ai_runtime import estimate_model_parameters_b
from .polymarket_model import PolymarketModelReport
from .polymarket_model_execution import (
    PolymarketExecutionResearchConfig,
    PolymarketPolicySelection,
)


POLYMARKET_AI_CASE_SCHEMA_VERSION = "polymarket-ai-veto-case-v2"
POLYMARKET_AI_REPORT_SCHEMA_VERSION = "polymarket-ai-veto-report-v2"
POLYMARKET_AI_CACHE_SCHEMA_VERSION = "polymarket-ai-veto-cache-v1"
POLYMARKET_AI_PROMPT_CONTRACT = "polymarket-ai-veto-prompt-v1"
SUPPORTED_POLYMARKET_AI_MODELS = (
    "qwen3:8b",
    "qwen3:14b",
    "qwen3.5:9b",
    "fin-r1:8b",
)
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

_REASON_CODES = (
    "edge_after_fees",
    "weak_probability_uplift",
    "market_disagreement",
    "liquidity_stress",
    "latency_risk",
    "source_staleness",
    "volatile_regime",
    "orderbook_imbalance",
    "model_calibration_risk",
    "insufficient_evidence",
    "cooldown_required",
)
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["approve", "veto", "cooldown"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason_codes": {
            "type": "array",
            "items": {"type": "string", "enum": list(_REASON_CODES)},
            "minItems": 1,
            "maxItems": 4,
        },
        "summary": {"type": "string", "maxLength": 180},
    },
    "required": ["action", "confidence", "reason_codes", "summary"],
    "additionalProperties": False,
}


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


@dataclass(frozen=True)
class PolymarketAIVetoConfig:
    model: str = "qwen3:8b"
    base_url: str = DEFAULT_OLLAMA_URL
    timeout_seconds: float = 30.0
    minimum_approval_confidence: float = 0.65
    maximum_advisory_latency_seconds: float = 15.0
    seed: int = 4701

    def validated(self) -> "PolymarketAIVetoConfig":
        model = str(self.model or "").strip()
        base_url = str(self.base_url or "").strip().rstrip("/")
        parameters = estimate_model_parameters_b(model)
        if (
            model not in SUPPORTED_POLYMARKET_AI_MODELS
            or parameters is None
            or parameters < 2.0
            or not base_url.startswith("http://127.0.0.1:")
            and not base_url.startswith("http://localhost:")
            or not math.isfinite(float(self.timeout_seconds))
            or not 1.0 <= float(self.timeout_seconds) <= 300.0
            or not math.isfinite(float(self.minimum_approval_confidence))
            or not 0.5 <= float(self.minimum_approval_confidence) <= 1.0
            or not math.isfinite(float(self.maximum_advisory_latency_seconds))
            or not 0.1 <= float(self.maximum_advisory_latency_seconds) <= 60.0
            or int(self.seed) < 0
        ):
            raise ValueError("Polymarket AI veto configuration is invalid")
        return replace(self, model=model, base_url=base_url)

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PolymarketAIVetoCase:
    case_id: str
    condition_id: str
    sample_id: str
    asset: str
    event_start_ms: int
    decision_received_wall_ms: int
    prompt_payload: Mapping[str, object]
    case_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_AI_CASE_SCHEMA_VERSION,
            "case_id": self.case_id,
            "condition_id": self.condition_id,
            "sample_id": self.sample_id,
            "asset": self.asset,
            "event_start_ms": self.event_start_ms,
            "decision_received_wall_ms": self.decision_received_wall_ms,
            "prompt_payload": dict(self.prompt_payload),
        }

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "case_sha256": self.case_sha256,
        }


@dataclass(frozen=True)
class PolymarketAIVetoDecision:
    action: str
    confidence: float
    reason_codes: tuple[str, ...]
    summary: str
    valid: bool
    failure_reason: str

    @property
    def permits_entry(self) -> bool:
        return self.valid and self.action == "approve"

    def asdict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "reason_codes": list(self.reason_codes),
            "permits_entry": self.permits_entry,
        }


@dataclass(frozen=True)
class PolymarketAIVetoResult:
    case_id: str
    condition_id: str
    model: str
    latency_seconds: float
    response_sha256: str
    response_payload: object
    decision: PolymarketAIVetoDecision

    def asdict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "condition_id": self.condition_id,
            "model": self.model,
            "latency_seconds": self.latency_seconds,
            "response_sha256": self.response_sha256,
            "response_payload": self.response_payload,
            "decision": self.decision.asdict(),
        }


@dataclass(frozen=True)
class PolymarketAIVetoReport:
    schema_version: str
    config: PolymarketAIVetoConfig
    model_digest: str
    model_metadata_sha256: str
    model_parameters_b: float
    risk_benchmark_evidence_sha256: str
    selection_sha256: str
    case_set_sha256: str
    case_count: int
    valid_response_count: int
    approval_count: int
    veto_count: int
    cooldown_count: int
    provider_failure_count: int
    average_latency_seconds: float
    maximum_latency_seconds: float
    market_permissions: Mapping[str, bool]
    market_permission_sha256: str
    results: tuple[PolymarketAIVetoResult, ...]
    report_sha256: str
    advisory_only: bool = True
    trading_authority: bool = False
    profitability_claim: bool = False

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.asdict(),
            "model_digest": self.model_digest,
            "model_metadata_sha256": self.model_metadata_sha256,
            "model_parameters_b": self.model_parameters_b,
            "risk_benchmark_evidence_sha256": (
                self.risk_benchmark_evidence_sha256
            ),
            "selection_sha256": self.selection_sha256,
            "case_set_sha256": self.case_set_sha256,
            "case_count": self.case_count,
            "valid_response_count": self.valid_response_count,
            "approval_count": self.approval_count,
            "veto_count": self.veto_count,
            "cooldown_count": self.cooldown_count,
            "provider_failure_count": self.provider_failure_count,
            "average_latency_seconds": self.average_latency_seconds,
            "maximum_latency_seconds": self.maximum_latency_seconds,
            "market_permissions": dict(self.market_permissions),
            "market_permission_sha256": self.market_permission_sha256,
            "results": [item.asdict() for item in self.results],
            "report_sha256": self.report_sha256,
            "advisory_only": self.advisory_only,
            "trading_authority": self.trading_authority,
            "profitability_claim": self.profitability_claim,
        }


PostJson = Callable[[str, Mapping[str, object], float, str], object]
ProgressCallback = Callable[[str, Mapping[str, object]], None]


class PolymarketAIVetoCache(Protocol):
    """Integrity-checked cache boundary; implementations retain original latency."""

    def get_polymarket_ai_veto_cache(
        self,
        cache_key_sha256: str,
    ) -> Mapping[str, object] | None: ...

    def put_polymarket_ai_veto_cache(
        self,
        cache_key_sha256: str,
        *,
        identity: Mapping[str, object],
        response_payload: object,
        latency_seconds: float,
    ) -> None: ...


def _cache_identity(
    case: PolymarketAIVetoCase,
    config: PolymarketAIVetoConfig,
    *,
    model_digest: str,
    model_metadata_sha256: str,
    request: Mapping[str, object],
) -> dict[str, object]:
    options = request.get("options")
    if not isinstance(options, Mapping):
        raise ValueError("Polymarket AI request options are invalid")
    return {
        "schema_version": POLYMARKET_AI_CACHE_SCHEMA_VERSION,
        "case_sha256": case.case_sha256,
        "model": config.model,
        "model_digest": model_digest,
        "model_metadata_sha256": model_metadata_sha256,
        "prompt_contract": POLYMARKET_AI_PROMPT_CONTRACT,
        "response_schema_sha256": _canonical_sha256(_RESPONSE_SCHEMA),
        "request_sha256": _canonical_sha256(request),
        "request_options_sha256": _canonical_sha256(dict(options)),
        "endpoint_policy": {
            "base_url": config.base_url,
            "path": "/api/chat",
            "method": "POST",
            "timeout_seconds": float(config.timeout_seconds),
        },
        "decision_policy": {
            "minimum_approval_confidence": float(
                config.minimum_approval_confidence
            ),
            "maximum_advisory_latency_seconds": float(
                config.maximum_advisory_latency_seconds
            ),
        },
    }


def build_polymarket_ai_veto_cases(
    selection: PolymarketPolicySelection,
    probability_report: PolymarketModelReport,
    execution_config: PolymarketExecutionResearchConfig,
) -> tuple[PolymarketAIVetoCase, ...]:
    """Build label-free prompts from the exact frozen pre-execution proposals."""

    if not selection.candidates:
        return ()
    validation_baseline = probability_report.baseline_metrics["validation"]
    validation_model = probability_report.model_metrics["validation"]
    cases: list[PolymarketAIVetoCase] = []
    for candidate in selection.candidates:
        sample = candidate.sample.validated()
        feature = sample.feature_map()
        risk_context = sample.risk_context_map()
        outcome_prior = (
            sample.baseline_up_probability
            if candidate.outcome == "Up"
            else 1.0 - sample.baseline_up_probability
        )
        payload = {
            "schema_version": POLYMARKET_AI_CASE_SCHEMA_VERSION,
            "task": "veto_only_review_of_frozen_ml_proposal",
            "asset": sample.asset,
            "five_minute_market": True,
            "remaining_seconds": sample.horizon_seconds,
            "proposed_outcome": candidate.outcome,
            "model_probability": round(candidate.predicted_probability, 8),
            "market_implied_probability": round(outcome_prior, 8),
            "model_probability_uplift": round(
                candidate.predicted_probability - outcome_prior,
                8,
            ),
            "decision_best_ask": str(candidate.decision_best_ask),
            "protective_limit_price": str(candidate.limit_price),
            "expected_edge_per_contract_after_fee": str(
                candidate.expected_edge_per_contract
            ),
            "minimum_required_edge_per_contract": str(
                execution_config.minimum_expected_edge_per_contract
            ),
            "maximum_loss_fraction_per_market": str(
                execution_config.maximum_loss_fraction_per_market
            ),
            "maximum_loss_fraction_per_time_group": str(
                execution_config.maximum_loss_fraction_per_time_group
            ),
            "assumed_submission_latency_ms": execution_config.submission_latency_ms,
            "microstructure": {
                name: round(float(feature[name]), 8)
                for name in (
                    "direct_distance_from_chainlink_open_bps",
                    "direct_chainlink_basis_bps",
                    "direct_return_100ms_bps",
                    "direct_return_250ms_bps",
                    "direct_return_1000ms_bps",
                    "direct_return_5000ms_bps",
                    "direct_realized_volatility_100ms_bps",
                    "direct_realized_volatility_1000ms_bps",
                    "direct_realized_volatility_5000ms_bps",
                    "direct_diffusion_market_logit_gap",
                    "chainlink_diffusion_market_logit_gap",
                    "direct_trade_imbalance_100ms",
                    "direct_trade_imbalance_250ms",
                    "direct_trade_imbalance_1000ms",
                    "direct_trade_imbalance_5000ms",
                    "direct_top_imbalance",
                    "direct_spread_bps",
                    "up_microprice_deviation_bps",
                    "down_microprice_deviation_bps",
                    "up_top_imbalance",
                    "down_top_imbalance",
                    "outcome_midpoint_sum_error_bps",
                    "executable_ask_pair_premium_bps",
                    "executable_bid_pair_discount_bps",
                )
            },
            "source_freshness_ms": {
                name: round(float(risk_context[name]), 3)
                for name in (
                    "up_book_age_ms",
                    "down_book_age_ms",
                    "direct_binance_age_ms",
                    "chainlink_source_age_ms",
                    "chainlink_arrival_age_ms",
                    "chainlink_anchor_gap_ms",
                )
            },
            "liquidity_context": {
                "proposed_outcome_ask_depth_3_contracts": round(
                    float(
                        risk_context[
                            "up_ask_depth_3_contracts"
                            if candidate.outcome == "Up"
                            else "down_ask_depth_3_contracts"
                        ]
                    ),
                    8,
                ),
                "market_liquidity_quote": round(
                    math.expm1(risk_context["log1p_market_liquidity_quote"]),
                    2,
                ),
                "market_volume_quote": round(
                    math.expm1(risk_context["log1p_market_volume_quote"]),
                    2,
                ),
            },
            "validation_only_model_evidence": {
                "market_baseline_log_loss": round(
                    validation_baseline.weighted_log_loss,
                    10,
                ),
                "residual_model_log_loss": round(
                    validation_model.weighted_log_loss,
                    10,
                ),
                "log_loss_delta": round(
                    probability_report.validation_log_loss_delta,
                    10,
                ),
                "validation_market_count": validation_model.market_count,
            },
            "hard_constraints": {
                "cannot_create_or_reverse_trade": True,
                "cannot_increase_size_or_limit": True,
                "invalid_or_uncertain_response_means_veto": True,
            },
        }
        case_id = _canonical_sha256(
            {
                "selection_sha256": selection.selection_sha256,
                "model_report_sha256": probability_report.report_sha256,
                "sample_id": sample.sample_id,
                "prompt_payload": payload,
            }
        )
        case = PolymarketAIVetoCase(
            case_id=case_id,
            condition_id=sample.condition_id,
            sample_id=sample.sample_id,
            asset=sample.asset,
            event_start_ms=sample.event_start_ms,
            decision_received_wall_ms=sample.decision_received_wall_ms,
            prompt_payload=payload,
            case_sha256="",
        )
        cases.append(
            replace(case, case_sha256=_canonical_sha256(case.identity_payload()))
        )
    cases.sort(
        key=lambda item: (
            item.decision_received_wall_ms,
            item.asset,
            item.condition_id,
        )
    )
    return tuple(cases)


def _request_json(
    url: str,
    payload: Mapping[str, object],
    timeout_seconds: float,
    method: str,
) -> object:
    data = None if method == "GET" else json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": "simple-ai-trading-polymarket-ai/0.1"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def _model_evidence(
    config: PolymarketAIVetoConfig,
    post_json: PostJson,
) -> tuple[str, str]:
    tags = post_json(
        f"{config.base_url}/api/tags",
        {},
        config.timeout_seconds,
        "GET",
    )
    if not isinstance(tags, Mapping) or not isinstance(tags.get("models"), list):
        raise ValueError("Ollama tags response is invalid")
    selected = next(
        (
            item
            for item in tags["models"]
            if isinstance(item, Mapping) and str(item.get("name")) == config.model
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"Ollama model is not installed: {config.model}")
    digest = str(selected.get("digest") or "").lower()
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError("Ollama model digest is invalid")
    show = post_json(
        f"{config.base_url}/api/show",
        {"model": config.model, "verbose": False},
        config.timeout_seconds,
        "POST",
    )
    return digest, _canonical_sha256(show)


def _prompt(case: PolymarketAIVetoCase) -> str:
    return (
        "You are a fail-closed institutional risk reviewer for a five-minute crypto "
        "prediction-market paper strategy. Review only the frozen proposal and causal "
        "evidence. You cannot predict a different side, create a trade, increase size, "
        "raise the limit, waive fees, or assume missing evidence is favorable. Approve "
        "only if the model uplift, direct-market movement, order-book state, volatility, "
        "liquidity, latency, and validation evidence are coherent. Veto contradictions "
        "or weak after-cost margin; use cooldown for unstable regimes. Return JSON only.\n"
        f"CASE={_canonical_json(case.prompt_payload)}"
    )


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("AI response JSON contains duplicate keys")
        parsed[key] = value
    return parsed


def _parse_decision(payload: object) -> PolymarketAIVetoDecision:
    if not isinstance(payload, Mapping):
        raise ValueError("AI response is not an object")
    message = payload.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("AI response message is missing")
    content = str(message["content"])
    if len(content) > 4_096:
        raise ValueError("AI response is too large")
    parsed = json.loads(content, object_pairs_hook=_strict_json_object)
    if not isinstance(parsed, Mapping) or set(parsed) != {
        "action",
        "confidence",
        "reason_codes",
        "summary",
    }:
        raise ValueError("AI response schema is invalid")
    action_raw = parsed["action"]
    confidence_raw = parsed["confidence"]
    codes_raw = parsed["reason_codes"]
    summary_raw = parsed["summary"]
    if (
        not isinstance(action_raw, str)
        or isinstance(confidence_raw, bool)
        or not isinstance(confidence_raw, (int, float))
        or not isinstance(summary_raw, str)
        or not isinstance(codes_raw, list)
        or any(not isinstance(value, str) for value in codes_raw)
    ):
        raise ValueError("AI response values are invalid")
    action = action_raw.strip().lower()
    confidence = float(confidence_raw)
    summary = summary_raw.strip()
    if (
        action not in {"approve", "veto", "cooldown"}
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
        or not 1 <= len(codes_raw) <= 4
        or not summary
        or len(summary) > 180
    ):
        raise ValueError("AI response values are invalid")
    codes = tuple(dict.fromkeys(codes_raw))
    if len(codes) != len(codes_raw) or any(value not in _REASON_CODES for value in codes):
        raise ValueError("AI response reason codes are invalid")
    return PolymarketAIVetoDecision(
        action=action,
        confidence=confidence,
        reason_codes=codes,
        summary=summary,
        valid=True,
        failure_reason="",
    )


def _failed_decision(reason: str) -> PolymarketAIVetoDecision:
    return PolymarketAIVetoDecision(
        action="veto",
        confidence=0.0,
        reason_codes=("insufficient_evidence",),
        summary="Provider, schema, confidence, or latency failure; fail-closed veto.",
        valid=False,
        failure_reason=str(reason)[:240],
    )


def _report_payload(report: PolymarketAIVetoReport) -> dict[str, object]:
    payload = report.asdict()
    payload.pop("report_sha256", None)
    return payload


def benchmark_polymarket_ai_veto(
    cases: Sequence[PolymarketAIVetoCase],
    *,
    all_condition_ids: Sequence[str],
    selection_sha256: str,
    risk_benchmark_evidence_sha256: str,
    config: PolymarketAIVetoConfig | None = None,
    post_json: PostJson = _request_json,
    progress: ProgressCallback | None = None,
    cache_store: PolymarketAIVetoCache | None = None,
    expected_model_digest: str = "",
) -> PolymarketAIVetoReport:
    """Run one local model over immutable label-free cases; failures always veto."""

    cfg = (config or PolymarketAIVetoConfig()).validated()
    selection_digest = str(selection_sha256 or "").strip().lower()
    if len(selection_digest) != 64 or any(
        value not in "0123456789abcdef" for value in selection_digest
    ):
        raise ValueError("Polymarket AI veto selection identity is invalid")
    benchmark_sha256 = str(risk_benchmark_evidence_sha256 or "").strip().lower()
    if len(benchmark_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in benchmark_sha256
    ):
        raise ValueError("Polymarket AI veto risk benchmark identity is invalid")
    conditions = tuple(sorted({str(value) for value in all_condition_ids}))
    if not conditions or any(not value for value in conditions):
        raise ValueError("Polymarket AI veto requires evaluated condition IDs")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Polymarket AI veto cases are duplicated")
    if len({case.condition_id for case in cases}) != len(cases):
        raise ValueError(
            "Polymarket AI veto requires exactly one case per market condition"
        )
    if any(
        case.condition_id not in conditions
        or case.case_sha256 != _canonical_sha256(case.identity_payload())
        for case in cases
    ):
        raise ValueError("Polymarket AI veto case identity is invalid")
    model_digest, metadata_sha256 = _model_evidence(cfg, post_json)
    expected_digest = str(expected_model_digest or "").strip().lower()
    if expected_digest:
        if (
            len(expected_digest) != 64
            or any(value not in "0123456789abcdef" for value in expected_digest)
            or model_digest != expected_digest
        ):
            raise ValueError(
                "Polymarket AI model digest differs from benchmark provenance"
            )
    parameters = estimate_model_parameters_b(cfg.model)
    if parameters is None or parameters < 2.0:
        raise ValueError("Polymarket AI veto model is not multibillion-parameter")
    case_set_sha256 = _canonical_sha256(
        {
            "schema_version": POLYMARKET_AI_CASE_SCHEMA_VERSION,
            "selection_sha256": selection_digest,
            "case_sha256": [case.case_sha256 for case in cases],
        }
    )
    results: list[PolymarketAIVetoResult] = []
    for index, case in enumerate(cases, start=1):
        request = {
            "model": cfg.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON matching the schema. You are veto-only "
                        "and can never increase or create risk."
                    ),
                },
                {"role": "user", "content": _prompt(case)},
            ],
            "stream": False,
            "think": False,
            "format": _RESPONSE_SCHEMA,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 220,
                "seed": cfg.seed,
            },
        }
        cache_identity = _cache_identity(
            case,
            cfg,
            model_digest=model_digest,
            model_metadata_sha256=metadata_sha256,
            request=request,
        )
        cache_key_sha256 = _canonical_sha256(cache_identity)
        cached = (
            cache_store.get_polymarket_ai_veto_cache(cache_key_sha256)
            if cache_store is not None
            else None
        )
        cache_hit = cached is not None
        raw: object = {}
        if cached is not None:
            cached_identity = cached.get("identity")
            if (
                not isinstance(cached_identity, Mapping)
                or dict(cached_identity) != cache_identity
            ):
                raise ValueError("Polymarket AI cache identity is invalid")
            raw = cached.get("response_payload")
            latency = float(cached.get("latency_seconds", math.nan))
            if (
                not math.isfinite(latency)
                or latency < 0.0
                or str(cached.get("response_sha256") or "")
                != _canonical_sha256(raw)
            ):
                raise ValueError("Polymarket AI cache payload is invalid")
            decision = _parse_decision(raw)
        else:
            started = time.perf_counter()
            try:
                raw = post_json(
                    f"{cfg.base_url}/api/chat",
                    request,
                    cfg.timeout_seconds,
                    "POST",
                )
                decision = _parse_decision(raw)
            except Exception as exc:  # noqa: BLE001 - evidence records failures
                decision = _failed_decision(f"{type(exc).__name__}: {exc}")
            latency = time.perf_counter() - started
        if (
            decision.valid
            and decision.action == "approve"
            and decision.confidence < cfg.minimum_approval_confidence
        ):
            decision = _failed_decision("approval confidence below configured floor")
        if latency > cfg.maximum_advisory_latency_seconds:
            decision = _failed_decision("advisory latency exceeded configured ceiling")
        if decision.valid and cache_store is not None and not cache_hit:
            cache_store.put_polymarket_ai_veto_cache(
                cache_key_sha256,
                identity=cache_identity,
                response_payload=raw,
                latency_seconds=latency,
            )
        result = PolymarketAIVetoResult(
            case_id=case.case_id,
            condition_id=case.condition_id,
            model=cfg.model,
            latency_seconds=latency,
            response_sha256=_canonical_sha256(raw),
            response_payload=raw,
            decision=decision,
        )
        results.append(result)
        if progress is not None:
            progress(
                "polymarket_ai_veto",
                {
                    "model": cfg.model,
                    "case": index,
                    "case_count": len(cases),
                    "action": decision.action,
                    "valid": decision.valid,
                    "cache_hit": cache_hit,
                    "latency_seconds": round(latency, 3),
                },
            )
    permissions = {condition: False for condition in conditions}
    for result in results:
        permissions[result.condition_id] = result.decision.permits_entry
    permission_sha256 = _canonical_sha256(
        {
            "schema_version": "polymarket-market-permission-v1",
            "permissions": dict(sorted(permissions.items())),
        }
    )
    latencies = [item.latency_seconds for item in results]
    provisional = PolymarketAIVetoReport(
        schema_version=POLYMARKET_AI_REPORT_SCHEMA_VERSION,
        config=cfg,
        model_digest=model_digest,
        model_metadata_sha256=metadata_sha256,
        model_parameters_b=float(parameters),
        risk_benchmark_evidence_sha256=benchmark_sha256,
        selection_sha256=selection_digest,
        case_set_sha256=case_set_sha256,
        case_count=len(cases),
        valid_response_count=sum(item.decision.valid for item in results),
        approval_count=sum(item.decision.action == "approve" for item in results),
        veto_count=sum(item.decision.action == "veto" for item in results),
        cooldown_count=sum(item.decision.action == "cooldown" for item in results),
        provider_failure_count=sum(not item.decision.valid for item in results),
        average_latency_seconds=(sum(latencies) / len(latencies) if latencies else 0.0),
        maximum_latency_seconds=max(latencies, default=0.0),
        market_permissions=dict(sorted(permissions.items())),
        market_permission_sha256=permission_sha256,
        results=tuple(results),
        report_sha256="",
    )
    return replace(
        provisional,
        report_sha256=_canonical_sha256(_report_payload(provisional)),
    )


__all__ = [
    "DEFAULT_OLLAMA_URL",
    "POLYMARKET_AI_CACHE_SCHEMA_VERSION",
    "POLYMARKET_AI_CASE_SCHEMA_VERSION",
    "POLYMARKET_AI_PROMPT_CONTRACT",
    "POLYMARKET_AI_REPORT_SCHEMA_VERSION",
    "SUPPORTED_POLYMARKET_AI_MODELS",
    "PolymarketAIVetoCase",
    "PolymarketAIVetoCache",
    "PolymarketAIVetoConfig",
    "PolymarketAIVetoDecision",
    "PolymarketAIVetoReport",
    "PolymarketAIVetoResult",
    "benchmark_polymarket_ai_veto",
    "build_polymarket_ai_veto_cases",
]
