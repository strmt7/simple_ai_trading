"""Structured local-AI risk review for model-lab artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping

import requests

from .ai_runtime import AICapabilityReport, detect_ai_capabilities
from .financial_sanity import blocking_reasons, build_model_lab_financial_sanity_report
from .storage import write_json_atomic
from .terminal_holdout_ledger import (
    reservation_evidence_passed,
    terminal_result_fingerprint,
)
from .types import RuntimeConfig

DEFAULT_AI_REVIEW_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
AI_REVIEW_REPORT_SCHEMA_VERSION = "ai-risk-review-v3"
_EMPTY_TEXT_SHA256 = hashlib.sha256(b"").hexdigest()
_MAX_OUTCOMES = 8
_MAX_CONCERNS = 8
_MAX_ACTIONS = 8
_MAX_REASON_CHARS = 240
_MAX_PROMPT_CHARS = 12_000
_MAX_ABLATION_ITEMS = 6
_MAX_AI_UPLIFT_WARNINGS = 8
_POSITIVE_ABLATION_DELTA_EPS = 1e-9

PostJson = Callable[[str, Mapping[str, object], float], object]
GetJson = Callable[[str, float], object]
ModelProvenance = Callable[[str, str, float], tuple[str, str]]

_AI_REVIEW_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["approve", "veto", "needs_human_review"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
        "concerns": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": _MAX_CONCERNS,
        },
        "required_actions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": _MAX_ACTIONS,
        },
    },
    "required": [
        "action",
        "confidence",
        "risk_score",
        "rationale",
        "concerns",
        "required_actions",
    ],
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

    def validated(self) -> AIReviewDecision:
        if (
            not isinstance(self.action, str)
            or self.action not in {"approve", "veto", "needs_human_review"}
            or isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
            or isinstance(self.risk_score, bool)
            or not isinstance(self.risk_score, (int, float))
            or not math.isfinite(self.risk_score)
            or not 0.0 <= self.risk_score <= 1.0
            or not isinstance(self.rationale, str)
            or not self.rationale
            or len(self.rationale) > _MAX_REASON_CHARS
            or not isinstance(self.concerns, list)
            or not isinstance(self.required_actions, list)
            or len(self.concerns) > _MAX_CONCERNS
            or len(self.required_actions) > _MAX_ACTIONS
            or any(
                not isinstance(value, str)
                or not value
                or len(value) > _MAX_REASON_CHARS
                for value in (*self.concerns, *self.required_actions)
            )
        ):
            raise ValueError("AI review decision is invalid")
        return self


@dataclass(frozen=True)
class AIReviewReport:
    schema_version: str
    created_at_ms: int
    status: str
    approved: bool
    source_report: str
    source_report_sha256: str
    provider: str
    model: str
    model_digest: str | None
    model_metadata_sha256: str | None
    endpoint: str
    latency_ms: int
    prompt_sha256: str
    request_sha256: str | None
    response_sha256: str | None
    decision: AIReviewDecision
    deterministic_precheck: dict[str, object]
    capability: dict[str, object] | None = None
    prompt_chars: int = 0
    output_path: str | None = None
    error: str | None = None
    report_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("report_sha256")
        return payload

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "report_sha256": self.report_sha256}

    def validated(self) -> AIReviewReport:
        self.decision.validated()
        capability_consistent = self.status == "blocked" or (
            isinstance(self.capability, dict)
            and self.capability.get("ok") is True
            and self.capability.get("provider") == self.provider
            and self.capability.get("model") == self.model
            and self.capability.get("compute_backend_kind") != "cpu"
            and self.capability.get("provider_available") is True
            and self.capability.get("model_available") is True
            and self.capability.get("model_local") is True
        )
        status_consistent = (
            (
                self.status == "ok"
                and self.approved
                and self.decision.action == "approve"
                and self.error is None
                and self.prompt_chars > 0
                and self.request_sha256 is not None
                and self.response_sha256 is not None
            )
            or (
                self.status == "review_required"
                and not self.approved
                and self.decision.action in {"veto", "needs_human_review"}
                and self.error is None
                and self.prompt_chars > 0
                and self.request_sha256 is not None
                and self.response_sha256 is not None
            )
            or (
                self.status == "blocked"
                and not self.approved
                and self.decision.action == "veto"
                and bool(self.error)
            )
        )
        if (
            self.schema_version != AI_REVIEW_REPORT_SCHEMA_VERSION
            or isinstance(self.created_at_ms, bool)
            or not isinstance(self.created_at_ms, int)
            or self.created_at_ms <= 0
            or not isinstance(self.status, str)
            or not isinstance(self.approved, bool)
            or not status_consistent
            or not capability_consistent
            or not isinstance(self.source_report, str)
            or not self.source_report
            or not _is_sha256(self.source_report_sha256)
            or self.provider != "ollama"
            or not isinstance(self.model, str)
            or not self.model
            or (self.model_digest is not None and not _is_sha256(self.model_digest))
            or (
                self.model_metadata_sha256 is not None
                and not _is_sha256(self.model_metadata_sha256)
            )
            or (self.model_digest is None) != (self.model_metadata_sha256 is None)
            or (
                self.status in {"ok", "review_required"}
                and (self.model_digest is None or self.model_metadata_sha256 is None)
            )
            or not isinstance(self.endpoint, str)
            or not self.endpoint
            or isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, int)
            or self.latency_ms < 0
            or not _is_sha256(self.prompt_sha256)
            or (self.request_sha256 is not None and not _is_sha256(self.request_sha256))
            or (
                self.response_sha256 is not None
                and not _is_sha256(self.response_sha256)
            )
            or (self.prompt_chars == 0 and self.prompt_sha256 != _EMPTY_TEXT_SHA256)
            or (self.prompt_chars == 0) != (self.request_sha256 is None)
            or isinstance(self.prompt_chars, bool)
            or not isinstance(self.prompt_chars, int)
            or not 0 <= self.prompt_chars <= _MAX_PROMPT_CHARS
            or not isinstance(self.deterministic_precheck, dict)
            or not (self.capability is None or isinstance(self.capability, dict))
            or not (self.output_path is None or isinstance(self.output_path, str))
            or not (self.error is None or isinstance(self.error, str))
            or not _is_sha256(self.report_sha256)
            or self.report_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("AI review report is invalid")
        return self


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    payload = json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))
    if not isinstance(payload, dict):
        raise ValueError("AI review mapping is not JSON-compatible")
    return payload


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finalize_report(report: AIReviewReport) -> AIReviewReport:
    return replace(
        report,
        report_sha256=_canonical_sha256(report.identity_payload()),
    ).validated()


def _strict_json_value(text: str, *, label: str) -> object:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"{label} contains duplicate key: {key}")
            payload[key] = value
        return payload

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite constant: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} was not one exact JSON value") from exc


def load_ai_review_report(
    path: Path,
    *,
    expected_source_report: Path | None = None,
) -> AIReviewReport:
    """Load a review and verify its report and exact source evidence digests."""

    try:
        payload = _strict_json_value(
            Path(path).read_text(encoding="utf-8"),
            label="AI review artifact",
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("AI review artifact is unreadable") from exc
    expected_fields = {
        "schema_version",
        "created_at_ms",
        "status",
        "approved",
        "source_report",
        "source_report_sha256",
        "provider",
        "model",
        "model_digest",
        "model_metadata_sha256",
        "endpoint",
        "latency_ms",
        "prompt_sha256",
        "request_sha256",
        "response_sha256",
        "decision",
        "deterministic_precheck",
        "capability",
        "prompt_chars",
        "output_path",
        "error",
        "report_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("AI review artifact fields are invalid")
    decision_payload = payload["decision"]
    if not isinstance(decision_payload, dict) or set(decision_payload) != {
        "action",
        "confidence",
        "risk_score",
        "rationale",
        "concerns",
        "required_actions",
    }:
        raise ValueError("AI review artifact decision is invalid")
    if not isinstance(decision_payload["concerns"], list) or not isinstance(
        decision_payload["required_actions"],
        list,
    ):
        raise ValueError("AI review artifact decision arrays are invalid")
    try:
        report = AIReviewReport(
            schema_version=payload["schema_version"],
            created_at_ms=payload["created_at_ms"],
            status=payload["status"],
            approved=payload["approved"],
            source_report=payload["source_report"],
            source_report_sha256=payload["source_report_sha256"],
            provider=payload["provider"],
            model=payload["model"],
            model_digest=payload["model_digest"],
            model_metadata_sha256=payload["model_metadata_sha256"],
            endpoint=payload["endpoint"],
            latency_ms=payload["latency_ms"],
            prompt_sha256=payload["prompt_sha256"],
            request_sha256=payload["request_sha256"],
            response_sha256=payload["response_sha256"],
            decision=AIReviewDecision(**decision_payload),
            deterministic_precheck=payload["deterministic_precheck"],
            capability=payload["capability"],
            prompt_chars=payload["prompt_chars"],
            output_path=payload["output_path"],
            error=payload["error"],
            report_sha256=payload["report_sha256"],
        ).validated()
    except (TypeError, ValueError) as exc:
        raise ValueError("AI review artifact is invalid") from exc
    source_path = Path(report.source_report)
    if expected_source_report is not None:
        expected_path = Path(expected_source_report)
        if source_path.resolve() != expected_path.resolve():
            raise ValueError("AI review source report path differs")
        source_path = expected_path
    try:
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("AI review source report is unavailable") from exc
    if source_sha256 != report.source_report_sha256:
        raise ValueError("AI review source report digest differs")
    return report


def _post_json(url: str, payload: Mapping[str, object], timeout: float) -> object:
    response = requests.post(  # nosec B113
        url,
        json=dict(payload),
        timeout=max(0.1, float(timeout)),
        headers={"User-Agent": "simple-ai-trading-ai-review/0.1"},
    )
    response.raise_for_status()
    return response.json()


def _get_json(url: str, timeout: float) -> object:
    response = requests.get(  # nosec B113
        url,
        timeout=max(0.1, float(timeout)),
        headers={"User-Agent": "simple-ai-trading-ai-review/0.1"},
    )
    response.raise_for_status()
    return response.json()


def resolve_ollama_model_provenance(
    base_url: str,
    model: str,
    timeout_seconds: float,
    *,
    get_json: GetJson = _get_json,
    post_json: PostJson = _post_json,
) -> tuple[str, str]:
    """Resolve the immutable Ollama digest and canonical local model metadata."""

    root = str(base_url).rstrip("/")
    tags = get_json(f"{root}/api/tags", timeout_seconds)
    if not isinstance(tags, Mapping) or not isinstance(tags.get("models"), list):
        raise ValueError("Ollama model inventory is malformed")
    candidates = {model} if ":" in model else {model, f"{model}:latest"}
    matches: list[Mapping[str, object]] = []
    for raw in tags["models"]:
        if not isinstance(raw, Mapping):
            raise ValueError("Ollama model inventory entry is malformed")
        names = (raw.get("name"), raw.get("model"))
        if any(isinstance(name, str) and name in candidates for name in names):
            matches.append(raw)
    if not matches:
        raise ValueError(f"Ollama model provenance is unavailable for {model}")
    digests = {raw.get("digest") for raw in matches}
    if len(digests) != 1:
        raise ValueError(f"Ollama model provenance is ambiguous for {model}")
    digest = next(iter(digests))
    if not _is_sha256(digest):
        raise ValueError(f"Ollama model digest is invalid for {model}")
    metadata = post_json(
        f"{root}/api/show",
        {"model": model, "verbose": False},
        timeout_seconds,
    )
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Ollama model metadata is malformed for {model}")
    return digest, _canonical_sha256(metadata)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _optional_finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


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
    payload = _strict_json_value(text, label="AI response")
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
    expected_fields = set(_AI_REVIEW_SCHEMA["required"])
    if set(payload) != expected_fields:
        raise ValueError("AI review response fields are missing or unexpected")
    action_raw = payload["action"]
    if not isinstance(action_raw, str):
        raise ValueError("AI review action must be a string")
    action = action_raw
    if action not in {"approve", "veto", "needs_human_review"}:
        raise ValueError("AI review action is missing or invalid")

    def bounded_string(value: object, *, label: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"AI review {label} must be a string")
        normalized = " ".join(value.split())
        if not normalized or len(normalized) > _MAX_REASON_CHARS:
            raise ValueError(f"AI review {label} is empty or too long")
        return normalized

    def bounded_strings(value: object, *, label: str, limit: int) -> list[str]:
        if not isinstance(value, list) or len(value) > limit:
            raise ValueError(f"AI review {label} must be a bounded array")
        return [bounded_string(item, label=f"{label} item") for item in value]

    def probability(value: object, *, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"AI review {label} must be numeric")
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise ValueError(f"AI review {label} must be finite and within [0,1]")
        return parsed

    return AIReviewDecision(
        action=action,
        confidence=probability(payload["confidence"], label="confidence"),
        risk_score=probability(payload["risk_score"], label="risk score"),
        rationale=bounded_string(payload["rationale"], label="rationale"),
        concerns=bounded_strings(
            payload["concerns"],
            label="concerns",
            limit=_MAX_CONCERNS,
        ),
        required_actions=bounded_strings(
            payload["required_actions"],
            label="required actions",
            limit=_MAX_ACTIONS,
        ),
    )


def _blocked_report(
    *,
    source_report: Path,
    source_report_sha256: str,
    created_at_ms: int,
    provider: str,
    model: str,
    model_digest: str | None = None,
    model_metadata_sha256: str | None = None,
    endpoint: str,
    reason: str,
    deterministic_precheck: dict[str, object],
    capability: AICapabilityReport | None = None,
    output_path: Path | None = None,
    latency_ms: int = 0,
    prompt_sha256: str = _EMPTY_TEXT_SHA256,
    prompt_chars: int = 0,
    request_sha256: str | None = None,
    response_sha256: str | None = None,
) -> AIReviewReport:
    bounded_reason = _bounded_text(reason)
    decision = AIReviewDecision(
        action="veto",
        confidence=1.0,
        risk_score=1.0,
        rationale=bounded_reason,
        concerns=[bounded_reason],
        required_actions=[
            "resolve deterministic model-lab or AI capability gate before autonomous use"
        ],
    )
    return _finalize_report(
        AIReviewReport(
            schema_version=AI_REVIEW_REPORT_SCHEMA_VERSION,
            created_at_ms=created_at_ms,
            status="blocked",
            approved=False,
            source_report=str(source_report),
            source_report_sha256=source_report_sha256,
            provider=provider,
            model=model,
            model_digest=model_digest,
            model_metadata_sha256=model_metadata_sha256,
            endpoint=endpoint,
            latency_ms=latency_ms,
            prompt_sha256=prompt_sha256,
            request_sha256=request_sha256,
            response_sha256=response_sha256,
            decision=decision,
            deterministic_precheck=deterministic_precheck,
            capability=(
                _json_mapping(capability.asdict()) if capability is not None else None
            ),
            prompt_chars=prompt_chars,
            output_path=str(output_path) if output_path is not None else None,
            error=reason,
        )
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
                    **_compact_market_edge_validation(stress),
                }
            robustness = item.get("robustness_validation")
            robustness_summary: dict[str, object] | None = None
            if isinstance(robustness, Mapping):
                robustness_summary = {
                    "accepted": bool(robustness.get("accepted")),
                    "window_count": int(_finite(robustness.get("window_count"))),
                    "accepted_windows": int(
                        _finite(robustness.get("accepted_windows"))
                    ),
                    "accepted_window_rate": _finite(
                        robustness.get("accepted_window_rate")
                    ),
                    "worst_realized_pnl": _finite(robustness.get("worst_realized_pnl")),
                    "worst_max_drawdown": _finite(robustness.get("worst_max_drawdown")),
                    "statistical_edge_accepted": (
                        bool(robustness.get("statistical_edge_accepted"))
                        if "statistical_edge_accepted" in robustness
                        else None
                    ),
                    "worst_sign_test_p_value": _finite(
                        robustness.get("worst_sign_test_p_value")
                    ),
                    "worst_bootstrap_lower_mean_return": _finite(
                        robustness.get("worst_bootstrap_lower_mean_return")
                    ),
                    **_compact_market_edge_validation(robustness),
                }
            regime = item.get("regime_validation")
            if not isinstance(regime, Mapping) and isinstance(robustness, Mapping):
                nested_regime = robustness.get("regime_summary")
                regime = nested_regime if isinstance(nested_regime, Mapping) else None
            regime_summary: dict[str, object] | None = None
            if isinstance(regime, Mapping):
                regime_summary = {
                    "window_count": int(_finite(regime.get("window_count"))),
                    "dominant_regime": _bounded_text(regime.get("dominant_regime")),
                    "dominant_regime_window_share": _finite(
                        regime.get("dominant_regime_window_share")
                    ),
                    "accepted_regime_count": int(
                        _finite(regime.get("accepted_regime_count"))
                    ),
                    "concentration_warning": bool(regime.get("concentration_warning")),
                    "notes": list(regime.get("notes") or [])[:6],
                }
            meta_labels = item.get("meta_label_validation")
            meta_summary: dict[str, object] = {}
            if isinstance(meta_labels, Mapping):
                for objective, raw_meta in list(meta_labels.items())[:4]:
                    if not isinstance(raw_meta, Mapping):
                        continue
                    meta_summary[str(objective)] = {
                        "status": _bounded_text(raw_meta.get("status")),
                        "sample_count": int(_finite(raw_meta.get("sample_count"))),
                        "take_count": int(_finite(raw_meta.get("take_count"))),
                        "downsize_count": int(_finite(raw_meta.get("downsize_count"))),
                        "skip_count": int(_finite(raw_meta.get("skip_count"))),
                        "take_precision": _finite(raw_meta.get("take_precision")),
                        "target_precision": _finite(raw_meta.get("target_precision")),
                    }
            hybrid_ablation = _compact_ablation_map(
                item.get("hybrid_ablation"),
                group_key="removed_expert_kind",
                delta_key="delta_vs_best",
            )
            feature_ablation = _compact_ablation_map(
                item.get("feature_ablation"),
                group_key="removed_group",
                delta_key="delta_vs_selected",
            )
            walk_forward_gate = _compact_walk_forward_map(item.get("walk_forward_gate"))
            selection_risk = _compact_selection_risk_map(item.get("selection_risk"))
            ai_uplift = _compact_ai_uplift(item.get("ai_uplift"))
            learning_feedback = _compact_learning_feedback(
                item.get("learning_feedback")
            )
            data_coverage = _compact_data_coverage(item.get("data_coverage"))
            compact_outcomes.append(
                {
                    "symbol": str(item.get("symbol") or ""),
                    "accepted": bool(item.get("accepted")),
                    "rows": int(_finite(item.get("rows"))),
                    "error": _bounded_text(item.get("error")),
                    "objective_scores": item.get("objective_scores")
                    if isinstance(item.get("objective_scores"), Mapping)
                    else {},
                    "hybrid_profiles": item.get("hybrid_profiles")
                    if isinstance(item.get("hybrid_profiles"), Mapping)
                    else {},
                    "walk_forward_gate": walk_forward_gate,
                    "selection_risk": selection_risk,
                    "stress_validation": stress_summary,
                    "robustness_validation": robustness_summary,
                    "regime_validation": regime_summary,
                    "meta_label_validation": meta_summary,
                    "hybrid_ablation": hybrid_ablation,
                    "feature_ablation": feature_ablation,
                    "ai_uplift": ai_uplift,
                    "learning_feedback": learning_feedback,
                    "data_coverage": data_coverage,
                    "diagnostics": item.get("diagnostics")
                    if isinstance(item.get("diagnostics"), Mapping)
                    else None,
                }
            )
    portfolio = report.get("portfolio_risk")
    portfolio_summary: dict[str, object] | None = None
    if isinstance(portfolio, Mapping):
        portfolio_summary = {
            "accepted": bool(portfolio.get("accepted")),
            "reason": _bounded_text(portfolio.get("reason")),
            "effective_symbol_count": _finite(portfolio.get("effective_symbol_count")),
            "correlation_adjusted_effective_symbol_count": _finite(
                portfolio.get("correlation_adjusted_effective_symbol_count")
            ),
            "max_pairwise_correlation": _finite(
                portfolio.get("max_pairwise_correlation")
            ),
            "max_cluster_weight": _finite(portfolio.get("max_cluster_weight")),
            "portfolio_cvar_95": _finite(portfolio.get("portfolio_cvar_95")),
            "portfolio_max_drawdown": _finite(portfolio.get("portfolio_max_drawdown")),
            "deployed_weight": _finite(portfolio.get("deployed_weight")),
            "accepted_symbols": list(portfolio.get("accepted_symbols") or [])[
                :_MAX_OUTCOMES
            ],
        }
    learning_feedback = _compact_learning_feedback(report.get("learning_feedback"))
    return {
        "quote_asset": str(report.get("quote_asset") or ""),
        "interval": str(report.get("interval") or ""),
        "market_type": str(report.get("market_type") or ""),
        "requested_objectives": list(report.get("requested_objectives") or []),
        "accepted_symbols": list(report.get("accepted_symbols") or []),
        "portfolio_risk": portfolio_summary,
        "learning_feedback": learning_feedback,
        "outcomes": compact_outcomes,
    }


def _compact_market_edge_validation(
    validation: Mapping[str, object],
) -> dict[str, object]:
    reports: list[Mapping[str, object]] = []
    direct = validation.get("market_edge")
    if isinstance(direct, Mapping):
        reports.append(direct)
    objectives = validation.get("objectives")
    if isinstance(objectives, list):
        for objective in objectives[:4]:
            if not isinstance(objective, Mapping):
                continue
            objective_edge = objective.get("market_edge")
            if isinstance(objective_edge, Mapping):
                reports.append(objective_edge)
            for collection_name in ("results", "windows"):
                collection = objective.get(collection_name)
                if not isinstance(collection, list):
                    continue
                for item in collection[:8]:
                    if not isinstance(item, Mapping):
                        continue
                    result = item.get("result")
                    if not isinstance(result, Mapping):
                        continue
                    edge = result.get("market_edge")
                    if isinstance(edge, Mapping):
                        reports.append(edge)
    if not reports:
        if "market_edge_accepted" in validation:
            return {
                "market_edge_accepted": bool(validation.get("market_edge_accepted"))
            }
        return {}
    failed_reasons = [
        _bounded_text(report.get("reason"))
        for report in reports
        if report.get("accepted") is not True and report.get("reason")
    ]
    return {
        "market_edge_accepted": all(
            report.get("accepted") is True for report in reports
        ),
        "worst_market_edge_pct": min(
            (_finite(report.get("net_edge_pct")) for report in reports), default=0.0
        ),
        "market_edge_failed_reasons": failed_reasons[:_MAX_CONCERNS],
    }


def _compact_selection_risk_map(raw_map: object) -> dict[str, dict[str, object]]:
    if not isinstance(raw_map, Mapping):
        return {}
    compact: dict[str, dict[str, object]] = {}
    for objective, raw in list(raw_map.items())[:4]:
        if not isinstance(raw, Mapping):
            continue
        raw_passed = raw.get("passed")
        overfit = raw.get("overfit_diagnostics")
        terminal = raw.get("terminal_holdout")
        compact_overfit = None
        if isinstance(overfit, Mapping):
            compact_overfit = {
                "status": _bounded_text(overfit.get("status")),
                "passed": bool(overfit.get("passed")) if "passed" in overfit else None,
                "reason": _bounded_text(overfit.get("reason")),
                "probability_backtest_overfit": _optional_finite(
                    overfit.get("probability_backtest_overfit")
                ),
                "max_probability_backtest_overfit": _optional_finite(
                    overfit.get("max_probability_backtest_overfit")
                ),
            }
        compact_terminal = None
        if isinstance(terminal, Mapping):
            terminal_result = terminal.get("result")
            reservation = terminal.get("reservation")
            try:
                terminal_result_sha256 = terminal_result_fingerprint(terminal)
            except (TypeError, ValueError, OverflowError):
                terminal_result_sha256 = ""
            compact_reservation = None
            if isinstance(reservation, Mapping):
                compact_reservation = {
                    "schema_version": _bounded_text(reservation.get("schema_version")),
                    "reservation_id": _bounded_text(
                        reservation.get("reservation_id"), limit=64
                    ),
                    "ledger_id": _bounded_text(reservation.get("ledger_id"), limit=64),
                    "symbol": _bounded_text(reservation.get("symbol"), limit=24),
                    "market_type": _bounded_text(
                        reservation.get("market_type"), limit=16
                    ),
                    "objective": _bounded_text(reservation.get("objective"), limit=32),
                    "first_timestamp": int(_finite(reservation.get("first_timestamp"))),
                    "last_timestamp": int(_finite(reservation.get("last_timestamp"))),
                    "rows": int(_finite(reservation.get("rows"))),
                    "dataset_fingerprint": _bounded_text(
                        reservation.get("dataset_fingerprint"),
                        limit=64,
                    ),
                    "model_fingerprint": _bounded_text(
                        reservation.get("model_fingerprint"),
                        limit=64,
                    ),
                    "result_fingerprint": _bounded_text(
                        reservation.get("result_fingerprint"),
                        limit=64,
                    ),
                    "status": _bounded_text(reservation.get("status"), limit=16),
                    "result_status": _bounded_text(
                        reservation.get("result_status"), limit=24
                    ),
                    "reserved_at_ms": int(_finite(reservation.get("reserved_at_ms"))),
                    "completed_at_ms": int(_finite(reservation.get("completed_at_ms"))),
                }
            compact_terminal = {
                "schema_version": _bounded_text(terminal.get("schema_version")),
                "passed": bool(terminal.get("passed"))
                if "passed" in terminal
                else None,
                "reason": _bounded_text(terminal.get("reason")),
                "evaluation_count": int(_finite(terminal.get("evaluation_count"))),
                "rows": int(_finite(terminal.get("rows"))),
                "start_timestamp": int(_finite(terminal.get("start_timestamp"))),
                "end_timestamp": int(_finite(terminal.get("end_timestamp"))),
                "score": _optional_finite(terminal.get("score")),
                "dataset_fingerprint": _bounded_text(
                    terminal.get("dataset_fingerprint"), limit=64
                ),
                "result_fingerprint": terminal_result_sha256,
                "reservation": compact_reservation,
                "result": (
                    {
                        "accepted": bool(terminal_result.get("accepted")),
                        "realized_pnl": _optional_finite(
                            terminal_result.get("realized_pnl")
                        ),
                        "stopped_by_liquidation": bool(
                            terminal_result.get("stopped_by_liquidation")
                        ),
                        "liquidation_events": int(
                            _finite(terminal_result.get("liquidation_events"))
                        ),
                    }
                    if isinstance(terminal_result, Mapping)
                    else None
                ),
            }
        compact[str(objective)] = {
            "passed": raw_passed if isinstance(raw_passed, bool) else None,
            "reason": _bounded_text(raw.get("reason")),
            "reasons": list(raw.get("reasons") or [])[:_MAX_CONCERNS]
            if isinstance(raw.get("reasons"), list)
            else [],
            "effective_trials": int(_finite(raw.get("effective_trials"))),
            "finite_candidate_scores": int(_finite(raw.get("finite_candidate_scores"))),
            "selected_score": _optional_finite(raw.get("selected_score")),
            "runner_up_score": _optional_finite(raw.get("runner_up_score")),
            "median_score": _optional_finite(raw.get("median_score")),
            "score_iqr": _optional_finite(raw.get("score_iqr")),
            "trial_penalty": _optional_finite(raw.get("trial_penalty")),
            "deflated_score": _optional_finite(raw.get("deflated_score")),
            "score_margin_to_runner_up": _optional_finite(
                raw.get("score_margin_to_runner_up")
            ),
            "terminal_holdout": compact_terminal,
            "overfit_diagnostics": compact_overfit,
        }
    return compact


def _compact_walk_forward_map(raw_map: object) -> dict[str, dict[str, object]]:
    if not isinstance(raw_map, Mapping):
        return {}
    compact: dict[str, dict[str, object]] = {}
    for objective, raw in list(raw_map.items())[:4]:
        if not isinstance(raw, Mapping):
            continue
        raw_passed = raw.get("passed")
        compact[str(objective)] = {
            "passed": raw_passed if isinstance(raw_passed, bool) else None,
            "reason": _bounded_text(raw.get("reason")),
            "fold_count": int(_finite(raw.get("fold_count"))),
            "accepted_folds": int(_finite(raw.get("accepted_folds"))),
            "worst_score": _optional_finite(raw.get("worst_score")),
            "worst_realized_pnl": _optional_finite(raw.get("worst_realized_pnl")),
            "worst_max_drawdown": _optional_finite(raw.get("worst_max_drawdown")),
        }
    return compact


def _compact_ablation_map(
    raw_map: object,
    *,
    group_key: str,
    delta_key: str,
) -> dict[str, list[dict[str, object]]]:
    if not isinstance(raw_map, Mapping):
        return {}
    compact: dict[str, list[dict[str, object]]] = {}
    for objective, raw_items in list(raw_map.items())[:4]:
        if not isinstance(raw_items, list):
            continue
        items: list[dict[str, object]] = []
        for item in raw_items[:_MAX_ABLATION_ITEMS]:
            if not isinstance(item, Mapping):
                continue
            items.append(
                {
                    "group": _bounded_text(item.get(group_key)),
                    "accepted": bool(item.get("accepted")),
                    "score": _finite(item.get("score")),
                    "delta": _finite(item.get(delta_key)),
                    "realized_pnl": _finite(item.get("realized_pnl")),
                    "max_drawdown": _finite(item.get("max_drawdown")),
                    "closed_trades": int(_finite(item.get("closed_trades"))),
                    "status": _bounded_text(item.get("status")),
                    "reject_reason": _bounded_text(item.get("reject_reason")),
                }
            )
        if items:
            compact[str(objective)] = items
    return compact


def _compact_ai_uplift(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, Mapping):
        return None
    reasons = raw.get("reasons")
    baseline = raw.get("baseline")
    ai_metrics = raw.get("ai")
    deltas = raw.get("deltas")
    statistical = raw.get("statistical_evidence")
    evidence_binding = raw.get("evidence_binding")
    policy = raw.get("policy")
    return {
        "schema_version": _bounded_text(raw.get("schema_version")),
        "accepted": bool(raw.get("accepted")),
        "advisory_only": bool(raw.get("advisory_only")),
        "trading_authority": raw.get("trading_authority"),
        "profitability_claim": raw.get("profitability_claim"),
        "model_name": _bounded_text(raw.get("model_name")),
        "model_parameters_b": _optional_finite(raw.get("model_parameters_b")),
        "baseline": baseline if isinstance(baseline, Mapping) else {},
        "ai": ai_metrics if isinstance(ai_metrics, Mapping) else {},
        "deltas": deltas if isinstance(deltas, Mapping) else {},
        "statistical_evidence": statistical if isinstance(statistical, Mapping) else {},
        "evidence_binding": evidence_binding
        if isinstance(evidence_binding, Mapping)
        else {},
        "policy": policy if isinstance(policy, Mapping) else {},
        "reasons": _bounded_list(reasons, limit=_MAX_CONCERNS),
    }


def _compact_learning_feedback(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, Mapping):
        return None
    recovery = raw.get("recovery_evidence")
    recovery_summary = None
    if isinstance(recovery, Mapping):
        recovery_summary = {
            "passed": bool(recovery.get("passed")),
            "stress_accepted": bool(recovery.get("stress_accepted")),
            "stress_worst_realized_pnl": _finite(
                recovery.get("stress_worst_realized_pnl")
            ),
            "temporal_robustness_accepted": bool(
                recovery.get("temporal_robustness_accepted")
            ),
            "temporal_worst_realized_pnl": _finite(
                recovery.get("temporal_worst_realized_pnl")
            ),
        }
    loss_by_symbol = raw.get("loss_by_symbol")
    return {
        "source_path": _bounded_text(raw.get("source_path")),
        "source": _bounded_text(raw.get("source")),
        "promotion_safe": (
            bool(raw.get("promotion_safe")) if "promotion_safe" in raw else None
        ),
        "closed_trades": int(_finite(raw.get("closed_trades"))),
        "losses": int(_finite(raw.get("losses"))),
        "net_realized_pnl": _finite(raw.get("net_realized_pnl")),
        "max_consecutive_losses": int(_finite(raw.get("max_consecutive_losses"))),
        "symbol": _bounded_text(raw.get("symbol")),
        "symbol_loss_count": int(_finite(raw.get("symbol_loss_count"))),
        "review_required": bool(raw.get("review_required")),
        "blocks_promotion": bool(raw.get("blocks_promotion")),
        "reason": _bounded_text(raw.get("reason")),
        "recovery_evidence": recovery_summary,
        "loss_by_symbol": (
            {
                str(key): int(_finite(value))
                for key, value in list(loss_by_symbol.items())[:6]
            }
            if isinstance(loss_by_symbol, Mapping)
            else {}
        ),
        "recommendations": _bounded_list(raw.get("recommendations"), limit=6),
    }


def _compact_data_coverage(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, Mapping):
        return None
    return {
        "symbol": _bounded_text(raw.get("symbol")),
        "market_type": _bounded_text(raw.get("market_type")),
        "interval": _bounded_text(raw.get("interval")),
        "source_scope": _bounded_text(raw.get("source_scope")),
        "integrity_status": _bounded_text(raw.get("integrity_status")),
        "integrity_warnings": _bounded_list(raw.get("integrity_warnings"), limit=8),
        "truth_basis": _bounded_list(raw.get("truth_basis"), limit=6),
        "candles_available": int(_finite(raw.get("candles_available"))),
        "candles_used": int(_finite(raw.get("candles_used"))),
        "rows_used": int(_finite(raw.get("rows_used"))),
        "used_start_utc": _bounded_text(raw.get("used_start_utc")),
        "used_end_utc": _bounded_text(raw.get("used_end_utc")),
        "used_duration_years": _finite(raw.get("used_duration_years")),
        "full_available_history_used": bool(raw.get("full_available_history_used")),
        "coverage_ratio": _finite(raw.get("coverage_ratio")),
        "gap_count": int(_finite(raw.get("gap_count"))),
        "largest_gap_intervals": _finite(raw.get("largest_gap_intervals")),
        "notes": _bounded_list(raw.get("notes"), limit=6),
    }


def _ablation_precheck_warnings(compact: Mapping[str, object]) -> list[str]:
    warnings: list[str] = []
    outcomes = compact.get("outcomes")
    if not isinstance(outcomes, list):
        return warnings
    for item in outcomes[:_MAX_OUTCOMES]:
        if not isinstance(item, Mapping):
            continue
        symbol = str(item.get("symbol") or "unknown")
        for evidence_field, label in (
            ("hybrid_ablation", "hybrid"),
            ("feature_ablation", "feature"),
        ):
            raw_map = item.get(evidence_field)
            if not isinstance(raw_map, Mapping):
                continue
            for objective, raw_items in raw_map.items():
                if not isinstance(raw_items, list):
                    continue
                for raw in raw_items:
                    if not isinstance(raw, Mapping):
                        continue
                    delta = _finite(raw.get("delta"))
                    if delta > _POSITIVE_ABLATION_DELTA_EPS:
                        group = _bounded_text(raw.get("group")) or "unknown_group"
                        warnings.append(
                            f"{symbol} {objective} {label} ablation improves score when removing {group}: +{delta:.6g}"
                        )
                        if len(warnings) >= _MAX_CONCERNS:
                            return warnings
    return warnings


def _learning_feedback_precheck_warnings(compact: Mapping[str, object]) -> list[str]:
    warnings: list[str] = []
    outcomes = compact.get("outcomes")
    if not isinstance(outcomes, list):
        return warnings
    for item in outcomes[:_MAX_OUTCOMES]:
        if not isinstance(item, Mapping) or not bool(item.get("accepted")):
            continue
        symbol = str(item.get("symbol") or "unknown")
        raw = item.get("learning_feedback")
        if not isinstance(raw, Mapping):
            continue
        if raw.get("blocks_promotion") is True:
            reason = _bounded_text(raw.get("reason")) or "learning_feedback_failed"
            warnings.append(f"{symbol} learning feedback blocks promotion ({reason})")
        if len(warnings) >= _MAX_CONCERNS:
            return warnings
    return warnings


def _data_coverage_precheck_warnings(compact: Mapping[str, object]) -> list[str]:
    warnings: list[str] = []
    outcomes = compact.get("outcomes")
    if not isinstance(outcomes, list):
        return warnings
    for item in outcomes[:_MAX_OUTCOMES]:
        if not isinstance(item, Mapping) or not bool(item.get("accepted")):
            continue
        symbol = str(item.get("symbol") or "unknown")
        raw = item.get("data_coverage")
        if not isinstance(raw, Mapping):
            warnings.append(f"{symbol} missing data coverage evidence")
        elif raw.get("integrity_status") == "fail":
            reason = ",".join(
                str(value) for value in list(raw.get("integrity_warnings") or [])[:3]
            )
            warnings.append(
                f"{symbol} data coverage failed ({reason or 'integrity_status=fail'})"
            )
        if len(warnings) >= _MAX_CONCERNS:
            return warnings
    return warnings


def _selection_risk_precheck_warnings(compact: Mapping[str, object]) -> list[str]:
    warnings: list[str] = []
    outcomes = compact.get("outcomes")
    if not isinstance(outcomes, list):
        return warnings
    for item in outcomes[:_MAX_OUTCOMES]:
        if not isinstance(item, Mapping):
            continue
        symbol = str(item.get("symbol") or "unknown")
        raw_map = item.get("selection_risk")
        if not isinstance(raw_map, Mapping):
            continue
        for objective, raw in raw_map.items():
            if not isinstance(raw, Mapping):
                continue
            explicit_failed = raw.get("passed") is False
            raw_deflated = raw.get("deflated_score")
            deflated_score = (
                float(raw_deflated) if isinstance(raw_deflated, (int, float)) else None
            )
            terminal = raw.get("terminal_holdout")
            terminal_reservation_passed = bool(
                isinstance(terminal, Mapping)
                and reservation_evidence_passed(
                    terminal.get("reservation"),
                    expected_dataset_fingerprint=str(
                        terminal.get("dataset_fingerprint") or ""
                    ),
                    expected_result_fingerprint=str(
                        terminal.get("result_fingerprint") or ""
                    ),
                    expected_rows=int(_finite(terminal.get("rows"))),
                    expected_first_timestamp=int(
                        _finite(terminal.get("start_timestamp"))
                    ),
                    expected_last_timestamp=int(_finite(terminal.get("end_timestamp"))),
                    expected_objective=str(objective),
                )
            )
            terminal_failed = not (
                isinstance(terminal, Mapping)
                and terminal.get("passed") is True
                and int(_finite(terminal.get("evaluation_count"))) == 1
                and terminal_reservation_passed
            )
            if (
                explicit_failed
                or terminal_failed
                or (deflated_score is not None and deflated_score <= 0.0)
            ):
                reason = _bounded_text(raw.get("reason")) or "selection_risk_failed"
                if terminal_failed and not explicit_failed:
                    reason = "terminal_holdout_missing_or_failed"
                deflated_text = (
                    f"{deflated_score:.6g}" if deflated_score is not None else "missing"
                )
                warnings.append(
                    f"{symbol} {objective} selection risk failed ({reason}); deflated_score={deflated_text}"
                )
                if len(warnings) >= _MAX_CONCERNS:
                    return warnings
    return warnings


def _ai_uplift_precheck_warnings(
    compact: Mapping[str, object],
    *,
    require_ai_uplift: bool,
) -> list[str]:
    if not require_ai_uplift:
        return []
    warnings: list[str] = []
    outcomes = compact.get("outcomes")
    if not isinstance(outcomes, list):
        return ["missing outcomes for AI uplift evidence"]
    for item in outcomes[:_MAX_OUTCOMES]:
        if not isinstance(item, Mapping) or not bool(item.get("accepted")):
            continue
        symbol = str(item.get("symbol") or "unknown")
        raw = item.get("ai_uplift")
        if not isinstance(raw, Mapping):
            warnings.append(f"{symbol} missing AI-vs-ML uplift evidence")
        elif raw.get("accepted") is not True:
            reasons = raw.get("reasons")
            reason_text = (
                ",".join(str(value) for value in reasons[:3])
                if isinstance(reasons, list)
                else "failed"
            )
            warnings.append(f"{symbol} AI-vs-ML uplift failed: {reason_text}")
        if len(warnings) >= _MAX_AI_UPLIFT_WARNINGS:
            return warnings
    return warnings


def _deterministic_precheck(
    compact: Mapping[str, object],
    *,
    require_ai_uplift: bool = False,
    financial_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    accepted_symbols = list(compact.get("accepted_symbols") or [])
    portfolio = compact.get("portfolio_risk")
    portfolio_ok = (
        bool(portfolio.get("accepted")) if isinstance(portfolio, Mapping) else False
    )
    ablation_warnings = _ablation_precheck_warnings(compact)
    data_coverage_warnings = _data_coverage_precheck_warnings(compact)
    learning_feedback_warnings = _learning_feedback_precheck_warnings(compact)
    selection_risk_warnings = _selection_risk_precheck_warnings(compact)
    ai_uplift_warnings = _ai_uplift_precheck_warnings(
        compact,
        require_ai_uplift=require_ai_uplift,
    )
    financial_sanity = build_model_lab_financial_sanity_report(
        financial_report if financial_report is not None else compact,
        source="ai_review_precheck",
    )
    financial_sanity_warnings = blocking_reasons(financial_sanity)[:_MAX_CONCERNS]
    return {
        "accepted_symbol_count": len(accepted_symbols),
        "portfolio_accepted": portfolio_ok,
        "portfolio_reason": _bounded_text(portfolio.get("reason"))
        if isinstance(portfolio, Mapping)
        else "missing_portfolio_risk",
        "ablation_warning_count": len(ablation_warnings),
        "ablation_warnings": ablation_warnings,
        "data_coverage_warning_count": len(data_coverage_warnings),
        "data_coverage_warnings": data_coverage_warnings,
        "learning_feedback_warning_count": len(learning_feedback_warnings),
        "learning_feedback_warnings": learning_feedback_warnings,
        "selection_risk_warning_count": len(selection_risk_warnings),
        "selection_risk_warnings": selection_risk_warnings,
        "ai_uplift_required": bool(require_ai_uplift),
        "ai_uplift_warning_count": len(ai_uplift_warnings),
        "ai_uplift_warnings": ai_uplift_warnings,
        "financial_sanity_warning_count": len(financial_sanity_warnings),
        "financial_sanity_warnings": financial_sanity_warnings,
        "allowed_for_ai_review": (
            bool(accepted_symbols)
            and portfolio_ok
            and not ablation_warnings
            and not data_coverage_warnings
            and not learning_feedback_warnings
            and not selection_risk_warnings
            and not ai_uplift_warnings
            and not financial_sanity_warnings
        ),
    }


def _prompt(compact: Mapping[str, object]) -> str:
    payload = json.dumps(compact, sort_keys=True, separators=(",", ":"))
    schema = json.dumps(_AI_REVIEW_SCHEMA, sort_keys=True, separators=(",", ":"))
    prompt = (
        "You are a cautious institutional trading risk reviewer for an autonomous day-trading testnet system. "
        "Review only the provided model-lab artifact. Do not assume missing data is favorable. "
        "Approve only when deterministic gates passed, stress scenarios are coherent, meta-label evidence does not "
        "show fragile take/skip behavior, temporal robustness and statistical edge evidence are coherent, regime "
        "concentration is not hiding a fragile one-state edge, purged walk-forward evidence used real accepted folds, "
        "selection-risk evidence shows the selected score survived the number of tried models and includes one "
        "positive fingerprinted terminal replay of the exact final model, hybrid and feature ablation evidence does not show "
        "that removing a model component improves the accepted score, any AI-assisted signal has explicit holdout "
        "uplift over the non-AI ML baseline without worse drawdown, learning feedback from closed trades does "
        "not indicate an unresolved repeated-loss pattern, portfolio tail risk is acceptable, and there "
        "is no obvious reason to require a human review. "
        "Return JSON matching the schema.\n"
        f"SCHEMA={schema}\n"
        f"MODEL_LAB_REPORT={payload}"
    )
    if len(prompt) > _MAX_PROMPT_CHARS:
        raise ValueError(
            f"complete prompt exceeds AI prompt bound:{len(prompt)}/{_MAX_PROMPT_CHARS}"
        )
    return prompt


def run_model_lab_ai_review(
    report_path: Path,
    runtime: RuntimeConfig,
    *,
    model: str | None = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = 20.0,
    output_path: Path | None = None,
    post_json: PostJson = _post_json,
    model_provenance: ModelProvenance | None = None,
) -> AIReviewReport:
    source_path = Path(report_path)
    output_path = output_path or (source_path.parent / "ai_risk_review.json")
    created_at_ms = int(time.time() * 1000)
    source_bytes = source_path.read_bytes()
    source_report_sha256 = hashlib.sha256(source_bytes).hexdigest()
    provider = "ollama"
    selected_model = str(
        model
        or (runtime.ai_model if runtime.ai_model != "auto" else DEFAULT_AI_REVIEW_MODEL)
    )
    endpoint = f"{str(base_url or DEFAULT_OLLAMA_URL).rstrip('/')}/api/chat"
    report_payload = json.loads(source_bytes.decode("utf-8"))
    if not isinstance(report_payload, Mapping):
        raise ValueError("model-lab report must be a JSON object")
    compact = _compact_model_lab_report(report_payload)
    precheck = _deterministic_precheck(
        compact,
        require_ai_uplift=bool(runtime.ai_enabled),
        financial_report=report_payload,
    )
    if not precheck["allowed_for_ai_review"]:
        ablation_warnings = precheck.get("ablation_warnings")
        data_coverage_warnings = precheck.get("data_coverage_warnings")
        learning_feedback_warnings = precheck.get("learning_feedback_warnings")
        selection_risk_warnings = precheck.get("selection_risk_warnings")
        ai_uplift_warnings = precheck.get("ai_uplift_warnings")
        financial_sanity_warnings = precheck.get("financial_sanity_warnings")
        if isinstance(ablation_warnings, list) and ablation_warnings:
            reason = "ablation evidence shows accepted model improves when a component is removed"
        elif isinstance(data_coverage_warnings, list) and data_coverage_warnings:
            reason = (
                "data coverage evidence is missing or failed for an accepted symbol"
            )
        elif (
            isinstance(learning_feedback_warnings, list) and learning_feedback_warnings
        ):
            reason = "learning feedback shows unresolved repeated-loss promotion risk"
        elif isinstance(selection_risk_warnings, list) and selection_risk_warnings:
            reason = "selection-risk evidence shows accepted model score does not survive trial burden"
        elif isinstance(ai_uplift_warnings, list) and ai_uplift_warnings:
            reason = (
                "AI-vs-ML uplift evidence is missing or failed for an accepted symbol"
            )
        elif isinstance(financial_sanity_warnings, list) and financial_sanity_warnings:
            reason = "financial sanity checks failed for an accepted model-lab artifact"
        else:
            reason = "deterministic gates did not produce an accepted portfolio for AI review"
        result = _blocked_report(
            source_report=source_path,
            source_report_sha256=source_report_sha256,
            created_at_ms=created_at_ms,
            provider=provider,
            model=selected_model,
            endpoint=endpoint,
            reason=reason,
            deterministic_precheck=precheck,
            output_path=output_path,
        )
        write_json_atomic(output_path, result.asdict(), indent=2, sort_keys=True)
        return result
    capability_config = runtime.ai_runtime_config()
    if capability_config.model == "auto":
        capability_config = replace(capability_config, model=selected_model)
    capability = detect_ai_capabilities(capability_config)
    if not capability.ok:
        reason = "; ".join(capability.messages) or "AI capability preflight failed"
        result = _blocked_report(
            source_report=source_path,
            source_report_sha256=source_report_sha256,
            created_at_ms=created_at_ms,
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
    model_digest: str | None = None
    model_metadata_sha256: str | None = None
    try:
        provenance_fn = model_provenance or resolve_ollama_model_provenance
        model_digest, model_metadata_sha256 = provenance_fn(
            str(base_url or DEFAULT_OLLAMA_URL),
            selected_model,
            timeout_seconds,
        )
        if not _is_sha256(model_digest) or not _is_sha256(model_metadata_sha256):
            raise ValueError("AI model provenance returned invalid digests")
    except Exception as exc:
        result = _blocked_report(
            source_report=source_path,
            source_report_sha256=source_report_sha256,
            created_at_ms=created_at_ms,
            provider=provider,
            model=selected_model,
            model_digest=model_digest,
            model_metadata_sha256=model_metadata_sha256,
            endpoint=endpoint,
            reason=f"AI model provenance failed: {exc}",
            deterministic_precheck=precheck,
            capability=capability,
            output_path=output_path,
        )
        write_json_atomic(output_path, result.asdict(), indent=2, sort_keys=True)
        return result
    try:
        prompt = _prompt(compact)
    except ValueError as exc:
        result = _blocked_report(
            source_report=source_path,
            source_report_sha256=source_report_sha256,
            created_at_ms=created_at_ms,
            provider=provider,
            model=selected_model,
            model_digest=model_digest,
            model_metadata_sha256=model_metadata_sha256,
            endpoint=endpoint,
            reason=f"AI review failed: {exc}",
            deterministic_precheck=precheck,
            capability=capability,
            output_path=output_path,
        )
        write_json_atomic(output_path, result.asdict(), indent=2, sort_keys=True)
        return result
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
    prompt_sha256 = _text_sha256(prompt)
    request_sha256 = _canonical_sha256(request)
    response_sha256: str | None = None
    started = time.perf_counter()
    try:
        response = post_json(endpoint, request, timeout_seconds)
        latency_ms = int((time.perf_counter() - started) * 1000)
        response_sha256 = _canonical_sha256(response)
        if not isinstance(response, Mapping):
            raise ValueError("AI provider response was not a JSON object")
        decision = _decision_from_mapping(
            _json_mapping_from_text(_ollama_response_text(response))
        )
        status = "ok" if decision.action == "approve" else "review_required"
        result = _finalize_report(
            AIReviewReport(
                schema_version=AI_REVIEW_REPORT_SCHEMA_VERSION,
                created_at_ms=created_at_ms,
                status=status,
                approved=decision.approved,
                source_report=str(source_path),
                source_report_sha256=source_report_sha256,
                provider=provider,
                model=selected_model,
                model_digest=model_digest,
                model_metadata_sha256=model_metadata_sha256,
                endpoint=endpoint,
                latency_ms=latency_ms,
                prompt_sha256=prompt_sha256,
                request_sha256=request_sha256,
                response_sha256=response_sha256,
                decision=decision,
                deterministic_precheck=precheck,
                capability=_json_mapping(capability.asdict()),
                prompt_chars=len(prompt),
                output_path=str(output_path),
            )
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        result = _blocked_report(
            source_report=source_path,
            source_report_sha256=source_report_sha256,
            created_at_ms=created_at_ms,
            provider=provider,
            model=selected_model,
            model_digest=model_digest,
            model_metadata_sha256=model_metadata_sha256,
            endpoint=endpoint,
            reason=f"AI review failed: {exc}",
            deterministic_precheck=precheck,
            capability=capability,
            output_path=output_path,
            latency_ms=latency_ms,
            prompt_sha256=prompt_sha256,
            prompt_chars=len(prompt),
            request_sha256=request_sha256,
            response_sha256=response_sha256,
        )
    write_json_atomic(output_path, result.asdict(), indent=2, sort_keys=True)
    return result
