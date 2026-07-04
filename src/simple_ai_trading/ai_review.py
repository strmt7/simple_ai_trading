"""Structured local-AI risk review for model-lab artifacts."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import requests

from .ai_runtime import AICapabilityReport, detect_ai_capabilities
from .storage import write_json_atomic
from .types import RuntimeConfig

DEFAULT_AI_REVIEW_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
_MAX_OUTCOMES = 8
_MAX_CONCERNS = 8
_MAX_ACTIONS = 8
_MAX_REASON_CHARS = 240
_MAX_PROMPT_CHARS = 12_000

PostJson = Callable[[str, Mapping[str, object], float], object]

_AI_REVIEW_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["approve", "veto", "needs_human_review"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
        "concerns": {"type": "array", "items": {"type": "string"}, "maxItems": _MAX_CONCERNS},
        "required_actions": {"type": "array", "items": {"type": "string"}, "maxItems": _MAX_ACTIONS},
    },
    "required": ["action", "confidence", "risk_score", "rationale", "concerns", "required_actions"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class AIReviewDecision:
    action: str
    confidence: float
    risk_score: float
    rationale: str
    concerns: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return self.action == "approve"

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AIReviewReport:
    status: str
    approved: bool
    source_report: str
    provider: str
    model: str
    endpoint: str
    latency_ms: int
    decision: AIReviewDecision
    deterministic_precheck: dict[str, object]
    capability: dict[str, object] | None = None
    prompt_chars: int = 0
    output_path: str | None = None
    error: str | None = None

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["decision"] = self.decision.asdict()
        return payload


def _post_json(url: str, payload: Mapping[str, object], timeout: float) -> object:
    response = requests.post(  # nosec B113
        url,
        json=dict(payload),
        timeout=max(0.1, float(timeout)),
        headers={"User-Agent": "simple-ai-trading-ai-review/0.1"},
    )
    response.raise_for_status()
    return response.json()


def _finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _bounded_text(value: object, limit: int = _MAX_REASON_CHARS) -> str:
    return " ".join(str(value or "").split())[:limit]


def _bounded_list(values: object, *, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for value in values[:limit]:
        text = _bounded_text(value)
        if text:
            out.append(text)
    return out


def _json_mapping_from_text(text: str) -> Mapping[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start:end + 1])
    if not isinstance(payload, Mapping):
        raise ValueError("AI response was not a JSON object")
    return payload


def _ollama_response_text(payload: Mapping[str, object]) -> str:
    message = payload.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if content is not None:
            return str(content)
    return str(payload.get("response") or "")


def _decision_from_mapping(payload: Mapping[str, object]) -> AIReviewDecision:
    action = str(payload.get("action") or "").strip().lower()
    if action not in {"approve", "veto", "needs_human_review"}:
        raise ValueError("AI review action is missing or invalid")
    rationale = _bounded_text(payload.get("rationale"))
    if not rationale:
        raise ValueError("AI review rationale is missing")
    return AIReviewDecision(
        action=action,
        confidence=max(0.0, min(1.0, _finite(payload.get("confidence")))),
        risk_score=max(0.0, min(1.0, _finite(payload.get("risk_score")))),
        rationale=rationale,
        concerns=_bounded_list(payload.get("concerns"), limit=_MAX_CONCERNS),
        required_actions=_bounded_list(payload.get("required_actions"), limit=_MAX_ACTIONS),
    )


def _blocked_report(
    *,
    source_report: Path,
    provider: str,
    model: str,
    endpoint: str,
    reason: str,
    deterministic_precheck: dict[str, object],
    capability: AICapabilityReport | None = None,
    output_path: Path | None = None,
) -> AIReviewReport:
    decision = AIReviewDecision(
        action="veto",
        confidence=1.0,
        risk_score=1.0,
        rationale=reason,
        concerns=[reason],
        required_actions=["resolve deterministic model-lab or AI capability gate before autonomous use"],
    )
    return AIReviewReport(
        status="blocked",
        approved=False,
        source_report=str(source_report),
        provider=provider,
        model=model,
        endpoint=endpoint,
        latency_ms=0,
        decision=decision,
        deterministic_precheck=deterministic_precheck,
        capability=capability.asdict() if capability is not None else None,
        output_path=str(output_path) if output_path is not None else None,
        error=reason,
    )


def _compact_model_lab_report(report: Mapping[str, object]) -> dict[str, object]:
    outcomes = report.get("outcomes")
    compact_outcomes: list[dict[str, object]] = []
    if isinstance(outcomes, list):
        for item in outcomes[:_MAX_OUTCOMES]:
            if not isinstance(item, Mapping):
                continue
            stress = item.get("stress_validation")
            stress_summary: dict[str, object] | None = None
            if isinstance(stress, Mapping):
                stress_summary = {
                    "accepted": bool(stress.get("accepted")),
                    "scenario_count": int(_finite(stress.get("scenario_count"))),
                    "worst_realized_pnl": _finite(stress.get("worst_realized_pnl")),
                    "worst_max_drawdown": _finite(stress.get("worst_max_drawdown")),
                }
            robustness = item.get("robustness_validation")
            robustness_summary: dict[str, object] | None = None
            if isinstance(robustness, Mapping):
                robustness_summary = {
                    "accepted": bool(robustness.get("accepted")),
                    "window_count": int(_finite(robustness.get("window_count"))),
                    "accepted_windows": int(_finite(robustness.get("accepted_windows"))),
                    "accepted_window_rate": _finite(robustness.get("accepted_window_rate")),
                    "worst_realized_pnl": _finite(robustness.get("worst_realized_pnl")),
                    "worst_max_drawdown": _finite(robustness.get("worst_max_drawdown")),
                }
            compact_outcomes.append({
                "symbol": str(item.get("symbol") or ""),
                "accepted": bool(item.get("accepted")),
                "rows": int(_finite(item.get("rows"))),
                "error": _bounded_text(item.get("error")),
                "objective_scores": item.get("objective_scores") if isinstance(item.get("objective_scores"), Mapping) else {},
                "hybrid_profiles": item.get("hybrid_profiles") if isinstance(item.get("hybrid_profiles"), Mapping) else {},
                "stress_validation": stress_summary,
                "robustness_validation": robustness_summary,
                "diagnostics": item.get("diagnostics") if isinstance(item.get("diagnostics"), Mapping) else None,
            })
    portfolio = report.get("portfolio_risk")
    portfolio_summary: dict[str, object] | None = None
    if isinstance(portfolio, Mapping):
        portfolio_summary = {
            "accepted": bool(portfolio.get("accepted")),
            "reason": _bounded_text(portfolio.get("reason")),
            "effective_symbol_count": _finite(portfolio.get("effective_symbol_count")),
            "max_pairwise_correlation": _finite(portfolio.get("max_pairwise_correlation")),
            "max_cluster_weight": _finite(portfolio.get("max_cluster_weight")),
            "portfolio_cvar_95": _finite(portfolio.get("portfolio_cvar_95")),
            "portfolio_max_drawdown": _finite(portfolio.get("portfolio_max_drawdown")),
            "deployed_weight": _finite(portfolio.get("deployed_weight")),
            "accepted_symbols": list(portfolio.get("accepted_symbols") or [])[:_MAX_OUTCOMES],
        }
    return {
        "quote_asset": str(report.get("quote_asset") or ""),
        "interval": str(report.get("interval") or ""),
        "market_type": str(report.get("market_type") or ""),
        "requested_objectives": list(report.get("requested_objectives") or []),
        "accepted_symbols": list(report.get("accepted_symbols") or []),
        "portfolio_risk": portfolio_summary,
        "outcomes": compact_outcomes,
    }


def _deterministic_precheck(compact: Mapping[str, object]) -> dict[str, object]:
    accepted_symbols = list(compact.get("accepted_symbols") or [])
    portfolio = compact.get("portfolio_risk")
    portfolio_ok = bool(portfolio.get("accepted")) if isinstance(portfolio, Mapping) else False
    return {
        "accepted_symbol_count": len(accepted_symbols),
        "portfolio_accepted": portfolio_ok,
        "portfolio_reason": _bounded_text(portfolio.get("reason")) if isinstance(portfolio, Mapping) else "missing_portfolio_risk",
        "allowed_for_ai_review": bool(accepted_symbols) and portfolio_ok,
    }


def _prompt(compact: Mapping[str, object]) -> str:
    payload = json.dumps(compact, sort_keys=True, separators=(",", ":"))
    if len(payload) > _MAX_PROMPT_CHARS:
        payload = payload[:_MAX_PROMPT_CHARS] + "...TRUNCATED"
    schema = json.dumps(_AI_REVIEW_SCHEMA, sort_keys=True, separators=(",", ":"))
    return (
        "You are a cautious institutional trading risk reviewer for an autonomous day-trading testnet system. "
        "Review only the provided model-lab artifact. Do not assume missing data is favorable. "
        "Approve only when deterministic gates passed, stress scenarios are coherent, temporal robustness is coherent, "
        "portfolio tail risk is acceptable, and there is no obvious reason to require a human review. "
        "Return JSON matching the schema.\n"
        f"SCHEMA={schema}\n"
        f"MODEL_LAB_REPORT={payload}"
    )


def run_model_lab_ai_review(
    report_path: Path,
    runtime: RuntimeConfig,
    *,
    model: str | None = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = 20.0,
    output_path: Path | None = None,
    post_json: PostJson = _post_json,
) -> AIReviewReport:
    source_path = Path(report_path)
    output_path = output_path or (source_path.parent / "ai_risk_review.json")
    provider = "ollama"
    selected_model = str(model or (runtime.ai_model if runtime.ai_model != "auto" else DEFAULT_AI_REVIEW_MODEL))
    endpoint = f"{str(base_url or DEFAULT_OLLAMA_URL).rstrip('/')}/api/chat"
    report_payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(report_payload, Mapping):
        raise ValueError("model-lab report must be a JSON object")
    compact = _compact_model_lab_report(report_payload)
    precheck = _deterministic_precheck(compact)
    if not precheck["allowed_for_ai_review"]:
        result = _blocked_report(
            source_report=source_path,
            provider=provider,
            model=selected_model,
            endpoint=endpoint,
            reason="deterministic gates did not produce an accepted portfolio for AI review",
            deterministic_precheck=precheck,
            output_path=output_path,
        )
        write_json_atomic(output_path, result.asdict(), indent=2, sort_keys=True)
        return result
    capability = detect_ai_capabilities(runtime.ai_runtime_config())
    if not capability.ok:
        reason = "; ".join(capability.messages) or "AI capability preflight failed"
        result = _blocked_report(
            source_report=source_path,
            provider=provider,
            model=selected_model,
            endpoint=endpoint,
            reason=reason,
            deterministic_precheck=precheck,
            capability=capability,
            output_path=output_path,
        )
        write_json_atomic(output_path, result.asdict(), indent=2, sort_keys=True)
        return result
    prompt = _prompt(compact)
    request = {
        "model": selected_model,
        "messages": [
            {
                "role": "system",
                "content": "Return only valid JSON matching the supplied schema. Be conservative; veto unsafe portfolios.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": _AI_REVIEW_SCHEMA,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 360},
    }
    started = time.perf_counter()
    try:
        response = post_json(endpoint, request, timeout_seconds)
        latency_ms = int((time.perf_counter() - started) * 1000)
        if not isinstance(response, Mapping):
            raise ValueError("AI provider response was not a JSON object")
        decision = _decision_from_mapping(_json_mapping_from_text(_ollama_response_text(response)))
        status = "ok" if decision.action == "approve" else "review_required"
        result = AIReviewReport(
            status=status,
            approved=decision.approved,
            source_report=str(source_path),
            provider=provider,
            model=selected_model,
            endpoint=endpoint,
            latency_ms=latency_ms,
            decision=decision,
            deterministic_precheck=precheck,
            capability=capability.asdict(),
            prompt_chars=len(prompt),
            output_path=str(output_path),
        )
    except Exception as exc:
        result = _blocked_report(
            source_report=source_path,
            provider=provider,
            model=selected_model,
            endpoint=endpoint,
            reason=f"AI review failed: {exc}",
            deterministic_precheck=precheck,
            capability=capability,
            output_path=output_path,
        )
    write_json_atomic(output_path, result.asdict(), indent=2, sort_keys=True)
    return result
