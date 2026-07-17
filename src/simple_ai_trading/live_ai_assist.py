"""Non-blocking, auditable AI shadow review for autonomous entry proposals.

The local language model never owns execution. It reviews only ML proposals,
uses causal structured evidence, and writes an append-only hash chain. New
entries wait asynchronously for exact-case evidence, while exits, monitoring,
and deterministic risk controls never wait for inference.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Callable, Mapping, Protocol, Sequence
from urllib.request import Request, urlopen

from .advanced_model import advanced_config_from_signature
from .ai_runtime import inspect_ollama_model_residency
from .execution_simulation import configured_round_trip_cost_floor_bps


LIVE_AI_ENTRY_CASE_SCHEMA_VERSION = "live-ai-entry-case-v1"
LIVE_AI_ENTRY_AUDIT_SCHEMA_VERSION = "live-ai-entry-audit-v1"
LIVE_AI_ENTRY_PROMPT_CONTRACT = "live-ai-entry-risk-review-v1"
_ZERO_SHA256 = "0" * 64
_MAX_COMPLETED_REVIEWS = 8
_MAX_EVIDENCE_DEPTH = 5
_MAX_EVIDENCE_ITEMS = 64
_MAX_EVIDENCE_JSON_BYTES = 16_384
_MAX_PROVIDER_RESPONSE_BYTES = 65_536
_MAX_AUDIT_RECORD_BYTES = 262_144
_ALLOWED_ACTIONS = frozenset({"approve", "veto", "cooldown"})
_ALLOWED_REASON_CODES = frozenset(
    {
        "edge_after_costs",
        "coherent_regime",
        "liquidity_acceptable",
        "weak_after_cost_edge",
        "unstable_regime",
        "liquidity_risk",
        "drawdown_risk",
        "model_uncertainty",
        "insufficient_evidence",
    }
)
_ADVERSE_REASON_CODES = frozenset(
    {
        "weak_after_cost_edge",
        "unstable_regime",
        "liquidity_risk",
        "drawdown_risk",
        "model_uncertainty",
        "insufficient_evidence",
    }
)

LIVE_AI_ENTRY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(_ALLOWED_ACTIONS)},
        "risk_multiplier": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason_codes": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(_ALLOWED_REASON_CODES)},
            "minItems": 1,
            "maxItems": 4,
            "uniqueItems": True,
        },
        "summary": {"type": "string", "minLength": 1, "maxLength": 180},
    },
    "required": [
        "action",
        "risk_multiplier",
        "confidence",
        "reason_codes",
        "summary",
    ],
    "additionalProperties": False,
}
_RESPONSE_KEYS = frozenset(LIVE_AI_ENTRY_RESPONSE_SCHEMA["required"])


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_object(value: str) -> Mapping[str, object]:
    parsed = json.loads(
        value,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {item}")
        ),
    )
    if not isinstance(parsed, Mapping):
        raise ValueError("JSON payload is not an object")
    return parsed


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _bounded_count(value: object, *, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"AI entry {name} is not an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"AI entry {name} is outside the bounded token budget")
    return value


def _bounded_json_value(value: object, *, depth: int = 0) -> object:
    if depth > _MAX_EVIDENCE_DEPTH:
        raise ValueError("AI evidence nesting is too deep")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if len(value) > 240:
            raise ValueError("AI evidence text is too long")
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("AI evidence contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_EVIDENCE_ITEMS:
            raise ValueError("AI evidence object is too large")
        normalized: dict[str, object] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if not normalized_key or len(normalized_key) > 80:
                raise ValueError("AI evidence key is invalid")
            if normalized_key in normalized:
                raise ValueError("AI evidence keys are ambiguous")
            normalized[normalized_key] = _bounded_json_value(
                item,
                depth=depth + 1,
            )
        return normalized
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_EVIDENCE_ITEMS:
            raise ValueError("AI evidence array is too large")
        return [_bounded_json_value(item, depth=depth + 1) for item in value]
    raise ValueError(f"AI evidence contains unsupported type: {type(value).__name__}")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _nonnegative_int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _model_validation_evidence(model: object) -> dict[str, object]:
    """Extract only compact, unit-explicit, post-cost model evidence."""

    if model is None:
        return {"available": False}
    selection = _mapping(getattr(model, "selection_risk", None))
    terminal = _mapping(selection.get("terminal_holdout"))
    terminal_result = _mapping(terminal.get("result"))
    market_edge = _mapping(terminal_result.get("market_edge"))
    execution = _mapping(getattr(model, "execution_validation", None))
    walk_forward = _mapping(execution.get("walk_forward_gate"))
    coverage = _mapping(execution.get("data_coverage"))
    microstructure = _mapping(execution.get("microstructure_replay"))
    mean_sample_return = _finite_or_none(market_edge.get("mean_sample_return"))
    strategy_overrides = getattr(model, "strategy_overrides", None)
    fallback_features = (
        strategy_overrides.get("enabled_features")
        if isinstance(strategy_overrides, Mapping)
        else None
    )
    feature_config = advanced_config_from_signature(
        getattr(model, "feature_signature", None),
        fallback_features if isinstance(fallback_features, Sequence) else None,
    )
    return {
        "available": True,
        "model_family": str(getattr(model, "model_family", ""))[:80],
        "selected_candidate": str(
            getattr(model, "model_selected_candidate", "")
        )[:80],
        "probability_calibration": {
            "sample_count": _nonnegative_int_or_none(
                getattr(model, "probability_calibration_size", None)
            ),
            "log_loss_after": _finite_or_none(
                getattr(model, "probability_log_loss_after", None)
            ),
            "brier_after": _finite_or_none(
                getattr(model, "probability_brier_after", None)
            ),
            "ece_after": _finite_or_none(
                getattr(model, "probability_ece_after", None)
            ),
        },
        "selection_risk": {
            "passed": selection.get("passed") is True,
            "effective_trials": _nonnegative_int_or_none(
                selection.get("effective_trials")
            ),
            "deflated_score": _finite_or_none(selection.get("deflated_score")),
        },
        "labeling": {
            "available": feature_config is not None,
            "mode": str(feature_config.label_mode) if feature_config is not None else "",
            "lookahead_bars": (
                int(feature_config.label_lookahead)
                if feature_config is not None
                else None
            ),
            "gross_label_barrier_bps": (
                float(feature_config.label_threshold) * 10_000.0
                if feature_config is not None
                else None
            ),
        },
        "terminal_holdout": {
            "passed": terminal.get("passed") is True,
            "accepted": terminal_result.get("accepted") is True,
            "closed_trades": _nonnegative_int_or_none(
                terminal_result.get("closed_trades")
            ),
            "realized_pnl_quote": _finite_or_none(
                terminal_result.get("realized_pnl")
            ),
            "maximum_drawdown_fraction": _finite_or_none(
                terminal_result.get("max_drawdown")
            ),
            "total_fees_quote": _finite_or_none(terminal_result.get("total_fees")),
            "net_edge_vs_buy_hold_quote": _finite_or_none(
                terminal_result.get("edge_vs_buy_hold")
            ),
            "mean_after_cost_sample_return_bps": (
                mean_sample_return * 10_000.0
                if mean_sample_return is not None
                else None
            ),
            "bootstrap_lower_mean_return": _finite_or_none(
                market_edge.get("bootstrap_lower_mean_return")
            ),
            "sign_test_p_value": _finite_or_none(
                market_edge.get("sign_test_p_value")
            ),
            "liquidation_events": _nonnegative_int_or_none(
                terminal_result.get("liquidation_events")
            ),
            "market_edge": {
                "accepted": market_edge.get("accepted") is True,
                "sample_count": _nonnegative_int_or_none(
                    market_edge.get("sample_count")
                ),
                "minimum_sample_count": _nonnegative_int_or_none(
                    market_edge.get("min_sample_count")
                ),
                "profit_factor": _finite_or_none(market_edge.get("profit_factor")),
                "minimum_profit_factor": _finite_or_none(
                    market_edge.get("min_profit_factor")
                ),
                "downside_return_risk_ratio": _finite_or_none(
                    market_edge.get("downside_return_risk_ratio")
                ),
                "minimum_downside_return_risk_ratio": _finite_or_none(
                    market_edge.get("min_downside_return_risk_ratio")
                ),
                "sign_test_p_value": _finite_or_none(
                    market_edge.get("sign_test_p_value")
                ),
                "maximum_sign_test_p_value": _finite_or_none(
                    market_edge.get("max_sign_test_p_value")
                ),
                "financial_sanity_allowed": (
                    market_edge.get("financial_sanity_allowed") is True
                ),
            },
        },
        "execution_validation": {
            "passed": execution.get("passed") is True,
            "walk_forward_passed": walk_forward.get("passed") is True,
            "walk_forward_folds": _nonnegative_int_or_none(
                walk_forward.get("fold_count")
            ),
            "walk_forward_worst_pnl_quote": _finite_or_none(
                walk_forward.get("worst_realized_pnl")
            ),
            "walk_forward_worst_drawdown_fraction": _finite_or_none(
                walk_forward.get("worst_max_drawdown")
            ),
            "stress_passed": _mapping(execution.get("stress")).get("accepted")
            is True,
            "temporal_passed": _mapping(
                execution.get("temporal_robustness")
            ).get("accepted")
            is True,
            "portfolio_passed": _mapping(execution.get("portfolio")).get(
                "accepted"
            )
            is True,
            "coverage_years": _finite_or_none(coverage.get("used_duration_years")),
            "coverage_ratio": _finite_or_none(coverage.get("coverage_ratio")),
            "gap_count": _nonnegative_int_or_none(coverage.get("gap_count")),
            "microstructure_passed": microstructure.get("passed") is True,
            "microstructure_seconds": _nonnegative_int_or_none(
                microstructure.get("captured_seconds")
            ),
            "microstructure_sequence_gaps": _nonnegative_int_or_none(
                microstructure.get("sequence_gap_count")
            ),
        },
    }


def _approval_evidence_is_valid(case: LiveAIEntryCase) -> bool:
    model_validation = _mapping(case.evidence.get("model_validation"))
    calibration = _mapping(model_validation.get("probability_calibration"))
    selection = _mapping(model_validation.get("selection_risk"))
    labeling = _mapping(model_validation.get("labeling"))
    terminal = _mapping(model_validation.get("terminal_holdout"))
    market_edge = _mapping(terminal.get("market_edge"))
    execution = _mapping(model_validation.get("execution_validation"))
    cost_model = _mapping(case.evidence.get("cost_model"))
    meta_label = _mapping(case.evidence.get("meta_label"))
    proposal = _mapping(case.evidence.get("proposal"))
    payoff = _mapping(proposal.get("after_cost_payoff"))
    sample_count = _nonnegative_int_or_none(calibration.get("sample_count"))
    effective_trials = _nonnegative_int_or_none(selection.get("effective_trials"))
    edge_sample_count = _nonnegative_int_or_none(market_edge.get("sample_count"))
    minimum_edge_samples = _nonnegative_int_or_none(
        market_edge.get("minimum_sample_count")
    )
    brier = _finite_or_none(calibration.get("brier_after"))
    ece = _finite_or_none(calibration.get("ece_after"))
    mean_return_bps = _finite_or_none(
        terminal.get("mean_after_cost_sample_return_bps")
    )
    bootstrap_lower = _finite_or_none(
        terminal.get("bootstrap_lower_mean_return")
    )
    cost_floor_bps = _finite_or_none(
        cost_model.get("configured_round_trip_cost_floor_bps")
    )
    evidence_barrier_bps = _finite_or_none(
        cost_model.get("model_gross_label_barrier_bps")
    )
    model_barrier_bps = _finite_or_none(
        labeling.get("gross_label_barrier_bps")
    )
    microstructure_seconds = _nonnegative_int_or_none(
        execution.get("microstructure_seconds")
    )
    microstructure_gaps = _nonnegative_int_or_none(
        execution.get("microstructure_sequence_gaps")
    )
    meta_enabled = meta_label.get("enabled") is True
    meta_minimum_samples = _nonnegative_int_or_none(
        meta_label.get("validation_minimum_sample_count")
    )
    meta_samples = _nonnegative_int_or_none(
        meta_label.get("validation_sample_count")
    )
    meta_minimum_precision = _finite_or_none(
        meta_label.get("validation_minimum_precision")
    )
    meta_precision = _finite_or_none(meta_label.get("validation_precision"))
    meta_expected_return = _finite_or_none(
        meta_label.get("expected_after_cost_return")
    )
    meta_expected_pnl = _finite_or_none(
        meta_label.get("expected_after_cost_pnl_quote")
    )
    payoff_support_count = _nonnegative_int_or_none(
        payoff.get("proposal_support_count")
    )
    payoff_validated_support_count = _nonnegative_int_or_none(
        payoff.get("validated_support_count")
    )
    payoff_validation_rows = _nonnegative_int_or_none(
        payoff.get("minimum_supporting_validation_rows")
    )
    payoff_validation_edge_bps = _finite_or_none(
        payoff.get("minimum_supporting_validation_after_cost_edge_bps")
    )
    payoff_current_edge_bps = _finite_or_none(
        payoff.get("weighted_proposal_expected_after_cost_bps")
    )
    payoff_value_coverage = _finite_or_none(
        payoff.get("proposal_value_weight_coverage")
    )
    meta_contract_valid = bool(
        not meta_enabled
        or (
            meta_label.get("action") in {"take", "downsize"}
            and meta_minimum_samples is not None
            and meta_minimum_samples > 0
            and meta_samples is not None
            and meta_samples >= meta_minimum_samples
            and meta_minimum_precision is not None
            and 0.0 <= meta_minimum_precision <= 1.0
            and meta_precision is not None
            and meta_minimum_precision <= meta_precision <= 1.0
            and meta_expected_return is not None
            and meta_expected_return > 0.0
            and meta_expected_pnl is not None
            and meta_expected_pnl > 0.0
        )
    )
    payoff_contract_valid = bool(
        payoff.get("schema_version") == "active-after-cost-payoff-evidence-v1"
        and payoff.get("available") is True
        and payoff.get("proposal_side") == case.proposed_side
        and payoff_support_count is not None
        and payoff_support_count > 0
        and payoff_validated_support_count is not None
        and payoff_validated_support_count > 0
        and payoff_validation_rows is not None
        and payoff_validation_rows > 0
        and payoff_validation_edge_bps is not None
        and payoff_validation_edge_bps > 0.0
        and payoff_current_edge_bps is not None
        and payoff_current_edge_bps > 0.0
        and payoff_value_coverage is not None
        and 0.0 < payoff_value_coverage <= 1.0
    )
    current_after_cost_edge_valid = bool(
        (meta_enabled and meta_contract_valid) or payoff_contract_valid
    )
    return bool(
        model_validation.get("available") is True
        and sample_count is not None
        and sample_count > 0
        and brier is not None
        and 0.0 <= brier <= 0.35
        and ece is not None
        and 0.0 <= ece <= 0.20
        and selection.get("passed") is True
        and effective_trials is not None
        and effective_trials > 0
        and labeling.get("available") is True
        and terminal.get("passed") is True
        and terminal.get("accepted") is True
        and terminal.get("liquidation_events") == 0
        and market_edge.get("accepted") is True
        and market_edge.get("financial_sanity_allowed") is True
        and edge_sample_count is not None
        and minimum_edge_samples is not None
        and edge_sample_count >= minimum_edge_samples
        and mean_return_bps is not None
        and mean_return_bps > 0.0
        and bootstrap_lower is not None
        and bootstrap_lower > 0.0
        and execution.get("passed") is True
        and execution.get("walk_forward_passed") is True
        and execution.get("stress_passed") is True
        and execution.get("temporal_passed") is True
        and execution.get("portfolio_passed") is True
        and (
            case.market_type != "futures"
            or (
                execution.get("microstructure_passed") is True
                and microstructure_seconds is not None
                and microstructure_seconds > 0
                and microstructure_gaps == 0
            )
        )
        and cost_floor_bps is not None
        and cost_floor_bps >= 0.0
        and evidence_barrier_bps is not None
        and model_barrier_bps is not None
        and math.isclose(evidence_barrier_bps, model_barrier_bps, abs_tol=1e-9)
        and model_barrier_bps > cost_floor_bps
        and meta_contract_valid
        and current_after_cost_edge_valid
    )


@dataclass(frozen=True)
class LiveAIEntryCase:
    """One immutable, label-free ML proposal presented to the AI reviewer."""

    case_id: str
    symbol: str
    market_type: str
    interval: str
    observed_at_ms: int
    proposed_side: str
    ml_confidence: float
    maximum_risk_multiplier: float
    model_digest: str
    terminal_model_fingerprint: str
    evidence: Mapping[str, object]

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": LIVE_AI_ENTRY_CASE_SCHEMA_VERSION,
            "symbol": self.symbol,
            "market_type": self.market_type,
            "interval": self.interval,
            "observed_at_ms": self.observed_at_ms,
            "proposed_side": self.proposed_side,
            "ml_confidence": self.ml_confidence,
            "maximum_risk_multiplier": self.maximum_risk_multiplier,
            "model_digest": self.model_digest,
            "terminal_model_fingerprint": self.terminal_model_fingerprint,
            "evidence": dict(self.evidence),
        }

    def validated(self) -> LiveAIEntryCase:
        if self.proposed_side not in {"LONG", "SHORT"}:
            raise ValueError("AI entry review requires a directional ML proposal")
        if not self.symbol or not self.interval or self.market_type not in {"spot", "futures"}:
            raise ValueError("AI entry review market identity is invalid")
        if int(self.observed_at_ms) <= 0:
            raise ValueError("AI entry review timestamp is invalid")
        if not 0.0 <= float(self.ml_confidence) <= 1.0:
            raise ValueError("AI entry review ML confidence is invalid")
        if not 0.0 <= float(self.maximum_risk_multiplier) <= 1.0:
            raise ValueError("AI entry review risk bound is invalid")
        for name, value in {
            "model_digest": self.model_digest,
            "terminal_model_fingerprint": self.terminal_model_fingerprint,
            "case_id": self.case_id,
        }.items():
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"AI entry review {name} is invalid")
        normalized = _bounded_json_value(self.evidence)
        if not isinstance(normalized, Mapping):
            raise ValueError("AI entry review evidence is invalid")
        if len(_canonical_json(normalized).encode("utf-8")) > _MAX_EVIDENCE_JSON_BYTES:
            raise ValueError("AI entry review evidence exceeds the prompt budget")
        if _canonical_sha256(self.identity_payload()) != self.case_id:
            raise ValueError("AI entry review case identity mismatch")
        return self


def build_live_ai_entry_case(
    *,
    symbol: str,
    market_type: str,
    interval: str,
    observed_at_ms: int,
    proposed_side: str,
    ml_confidence: float,
    maximum_risk_multiplier: float,
    model_digest: str,
    terminal_model_fingerprint: str,
    evidence: Mapping[str, object],
) -> LiveAIEntryCase:
    normalized = _bounded_json_value(evidence)
    if not isinstance(normalized, Mapping):
        raise ValueError("AI entry review evidence is invalid")
    if len(_canonical_json(normalized).encode("utf-8")) > _MAX_EVIDENCE_JSON_BYTES:
        raise ValueError("AI entry review evidence exceeds the prompt budget")
    identity = {
        "schema_version": LIVE_AI_ENTRY_CASE_SCHEMA_VERSION,
        "symbol": str(symbol),
        "market_type": str(market_type),
        "interval": str(interval),
        "observed_at_ms": int(observed_at_ms),
        "proposed_side": str(proposed_side),
        "ml_confidence": float(ml_confidence),
        "maximum_risk_multiplier": float(maximum_risk_multiplier),
        "model_digest": str(model_digest),
        "terminal_model_fingerprint": str(terminal_model_fingerprint),
        "evidence": dict(normalized),
    }
    # ``schema_version`` is hash material, not a dataclass field.
    case = LiveAIEntryCase(
        case_id=_canonical_sha256(identity),
        symbol=str(symbol),
        market_type=str(market_type),
        interval=str(interval),
        observed_at_ms=int(observed_at_ms),
        proposed_side=str(proposed_side),
        ml_confidence=float(ml_confidence),
        maximum_risk_multiplier=float(maximum_risk_multiplier),
        model_digest=str(model_digest),
        terminal_model_fingerprint=str(terminal_model_fingerprint),
        evidence=dict(normalized),
    )
    return case.validated()


@dataclass(frozen=True)
class LiveAIEntryDecision:
    action: str
    risk_multiplier: float
    confidence: float
    reason_codes: tuple[str, ...]
    summary: str
    valid: bool
    failure_reason: str = ""
    response_sha256: str = _ZERO_SHA256
    observed_model_digest: str = _ZERO_SHA256
    model_residency_status: str = "unknown"
    prompt_tokens: int | None = None
    output_tokens: int | None = None

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload

    def validated_for(self, case: LiveAIEntryCase) -> LiveAIEntryDecision:
        case.validated()
        if not self.valid:
            raise ValueError("AI entry decision is not valid provider evidence")
        if self.action not in _ALLOWED_ACTIONS:
            raise ValueError("AI entry decision action is invalid")
        risk_multiplier = _finite_number(
            self.risk_multiplier,
            name="risk_multiplier",
        )
        confidence = _finite_number(self.confidence, name="confidence")
        if (
            not 0.0 <= risk_multiplier <= case.maximum_risk_multiplier
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("AI entry decision exceeds its causal risk bound")
        codes = tuple(self.reason_codes)
        if (
            not 1 <= len(codes) <= 4
            or len(set(codes)) != len(codes)
            or any(code not in _ALLOWED_REASON_CODES for code in codes)
            or not isinstance(self.summary, str)
            or not self.summary.strip()
            or len(self.summary) > 180
        ):
            raise ValueError("AI entry decision evidence is invalid")
        if self.action == "approve" and (
            risk_multiplier <= 0.0 or "edge_after_costs" not in codes
        ):
            raise ValueError("AI approval lacks positive after-cost evidence")
        if self.action == "approve" and not _approval_evidence_is_valid(case):
            raise ValueError("AI approval is not supported by bound model evidence")
        if self.action != "approve" and (
            risk_multiplier != 0.0 or not _ADVERSE_REASON_CODES.intersection(codes)
        ):
            raise ValueError("AI adverse action lacks a zero-risk adverse reason")
        if (
            self.response_sha256 == _ZERO_SHA256
            or len(self.response_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.response_sha256
            )
        ):
            raise ValueError("AI entry decision response identity is invalid")
        if (
            self.observed_model_digest != case.model_digest
            or self.model_residency_status != "gpu_resident"
        ):
            raise ValueError("AI entry decision is not bound to the approved GPU model")
        prompt_tokens = _bounded_count(
            self.prompt_tokens,
            name="prompt token count",
            maximum=4096,
        )
        output_tokens = _bounded_count(
            self.output_tokens,
            name="output token count",
            maximum=180,
        )
        if prompt_tokens <= 0 or output_tokens <= 0:
            raise ValueError("AI entry decision token telemetry is empty")
        return self


@dataclass(frozen=True)
class LiveAIEntryReview:
    case_id: str
    status: str
    decision: LiveAIEntryDecision | None
    latency_seconds: float
    completed_at_ms: int = 0

    @property
    def reason(self) -> str:
        if self.decision is None:
            return "review pending"
        return self.decision.failure_reason or self.decision.summary


class LiveAIEntryProvider(Protocol):
    def __call__(self, case: LiveAIEntryCase) -> LiveAIEntryDecision: ...


def _parse_provider_decision(payload: object, *, expected_model: str) -> LiveAIEntryDecision:
    if not isinstance(payload, Mapping):
        raise ValueError("Ollama response is not an object")
    if payload.get("done") is not True:
        raise ValueError("Ollama response is incomplete")
    if str(payload.get("model") or "") != expected_model:
        raise ValueError("Ollama response model differs from the requested model")
    message = payload.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("Ollama response content is missing")
    parsed = _strict_json_object(str(message["content"]))
    if frozenset(parsed) != _RESPONSE_KEYS:
        raise ValueError("AI entry response fields do not match the frozen schema")
    action = parsed["action"]
    if not isinstance(action, str) or action not in _ALLOWED_ACTIONS:
        raise ValueError("AI entry action is invalid")
    risk_multiplier = _finite_number(parsed["risk_multiplier"], name="risk_multiplier")
    confidence = _finite_number(parsed["confidence"], name="confidence")
    if not 0.0 <= risk_multiplier <= 1.0 or not 0.0 <= confidence <= 1.0:
        raise ValueError("AI entry response bounds are invalid")
    raw_codes = parsed["reason_codes"]
    if (
        not isinstance(raw_codes, list)
        or not 1 <= len(raw_codes) <= 4
        or any(not isinstance(item, str) or item not in _ALLOWED_REASON_CODES for item in raw_codes)
        or len(set(raw_codes)) != len(raw_codes)
    ):
        raise ValueError("AI entry reason codes are invalid")
    codes = tuple(raw_codes)
    summary = parsed["summary"]
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 180:
        raise ValueError("AI entry summary is invalid")
    if action == "approve" and (risk_multiplier <= 0.0 or "edge_after_costs" not in codes):
        raise ValueError("AI approval lacks positive after-cost evidence")
    if action != "approve" and (risk_multiplier != 0.0 or not _ADVERSE_REASON_CODES.intersection(codes)):
        raise ValueError("AI adverse action lacks a zero-risk adverse reason")
    return LiveAIEntryDecision(
        action=action,
        risk_multiplier=risk_multiplier,
        confidence=confidence,
        reason_codes=codes,
        summary=summary.strip(),
        valid=True,
        response_sha256=_canonical_sha256(payload),
    )


class OllamaLiveAIEntryProvider:
    """Strict Ollama structured-output provider for one frozen local model."""

    def __init__(
        self,
        *,
        model: str,
        expected_model_digest: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 30.0,
        seed: int = 3901,
    ) -> None:
        self.model = str(model).strip()
        self.expected_model_digest = str(expected_model_digest).strip().lower()
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.seed = int(seed)
        if (
            not self.model
            or len(self.expected_model_digest) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.expected_model_digest
            )
            or not self.base_url.startswith(("http://", "https://"))
        ):
            raise ValueError("Ollama live AI provider configuration is invalid")

    def __call__(self, case: LiveAIEntryCase) -> LiveAIEntryDecision:
        case.validated()
        if case.model_digest != self.expected_model_digest:
            raise ValueError("AI entry case differs from the approved model digest")
        prompt = (
            "You are a shadow-only risk reviewer for an autonomous crypto day-trading system. "
            "Review the ML proposal using only the causal structured evidence. You cannot create a trade, "
            "change direction, increase risk, infer missing facts, or control execution. Treat costs, "
            "liquidity, regime uncertainty, and drawdown risk conservatively. Return only JSON matching "
            f"{LIVE_AI_ENTRY_PROMPT_CONTRACT}. CASE="
            f"{_canonical_json(case.identity_payload())}"
        )
        request_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only schema-valid JSON. Missing or conflicting evidence requires veto or cooldown.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": LIVE_AI_ENTRY_RESPONSE_SCHEMA,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 180,
                "seed": self.seed,
            },
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=_canonical_json(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            raw_response = response.read(_MAX_PROVIDER_RESPONSE_BYTES + 1)
        if len(raw_response) > _MAX_PROVIDER_RESPONSE_BYTES:
            raise ValueError("Ollama response exceeds the bounded response budget")
        payload = _strict_json_object(raw_response.decode("utf-8"))
        decision = _parse_provider_decision(
            payload,
            expected_model=self.model,
        )
        prompt_tokens = _bounded_count(
            payload.get("prompt_eval_count"),
            name="prompt token count",
            maximum=4096,
        )
        output_tokens = _bounded_count(
            payload.get("eval_count"),
            name="output token count",
            maximum=180,
        )
        residency = inspect_ollama_model_residency(
            self.base_url,
            self.model,
            timeout_seconds=min(2.0, self.timeout_seconds),
            expected_digest=self.expected_model_digest,
        )
        if (
            residency.status != "gpu_resident"
            or residency.digest != self.expected_model_digest
        ):
            raise ValueError(
                "Ollama response is not bound to the approved GPU-resident model"
            )
        return replace(
            decision,
            response_sha256=hashlib.sha256(raw_response).hexdigest(),
            observed_model_digest=residency.digest,
            model_residency_status=residency.status,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        ).validated_for(case)


def _failed_decision(reason: str) -> LiveAIEntryDecision:
    return LiveAIEntryDecision(
        action="veto",
        risk_multiplier=0.0,
        confidence=0.0,
        reason_codes=("insufficient_evidence",),
        summary="Provider, schema, or audit failure; shadow decision is fail-closed.",
        valid=False,
        failure_reason=str(reason)[:240],
    )


def _audit_case_and_decision(
    record: Mapping[str, object],
    *,
    line_number: int,
) -> tuple[LiveAIEntryCase, LiveAIEntryDecision]:
    if (
        record.get("schema_version") != LIVE_AI_ENTRY_AUDIT_SCHEMA_VERSION
        or record.get("mode") != "shadow_only"
        or record.get("trading_authority") is not False
    ):
        raise ValueError(f"AI entry audit contract is invalid at line {line_number}")
    raw_case = record.get("case")
    raw_decision = record.get("decision")
    if not isinstance(raw_case, Mapping) or not isinstance(raw_decision, Mapping):
        raise ValueError(f"AI entry audit evidence is missing at line {line_number}")
    if raw_case.get("schema_version") != LIVE_AI_ENTRY_CASE_SCHEMA_VERSION:
        raise ValueError(f"AI entry audit case schema is invalid at line {line_number}")
    try:
        case = LiveAIEntryCase(
            case_id=str(raw_case["case_id"]),
            symbol=str(raw_case["symbol"]),
            market_type=str(raw_case["market_type"]),
            interval=str(raw_case["interval"]),
            observed_at_ms=int(raw_case["observed_at_ms"]),
            proposed_side=str(raw_case["proposed_side"]),
            ml_confidence=float(raw_case["ml_confidence"]),
            maximum_risk_multiplier=float(raw_case["maximum_risk_multiplier"]),
            model_digest=str(raw_case["model_digest"]),
            terminal_model_fingerprint=str(raw_case["terminal_model_fingerprint"]),
            evidence=dict(raw_case["evidence"]),
        ).validated()
        decision = LiveAIEntryDecision(
            action=str(raw_decision["action"]),
            risk_multiplier=float(raw_decision["risk_multiplier"]),
            confidence=float(raw_decision["confidence"]),
            reason_codes=tuple(raw_decision["reason_codes"]),
            summary=str(raw_decision["summary"]),
            valid=bool(raw_decision["valid"]),
            failure_reason=str(raw_decision.get("failure_reason", "")),
            response_sha256=str(raw_decision.get("response_sha256", _ZERO_SHA256)),
            observed_model_digest=str(
                raw_decision.get("observed_model_digest", _ZERO_SHA256)
            ),
            model_residency_status=str(
                raw_decision.get("model_residency_status", "unknown")
            ),
            prompt_tokens=(
                int(raw_decision["prompt_tokens"])
                if raw_decision.get("prompt_tokens") is not None
                else None
            ),
            output_tokens=(
                int(raw_decision["output_tokens"])
                if raw_decision.get("output_tokens") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"AI entry audit typed evidence is invalid at line {line_number}"
        ) from exc
    if decision.valid:
        decision.validated_for(case)
    elif (
        decision.action != "veto"
        or decision.risk_multiplier != 0.0
        or not decision.failure_reason
        or decision.reason_codes != ("insufficient_evidence",)
    ):
        raise ValueError(f"AI entry audit failure evidence is invalid at line {line_number}")
    try:
        completed_at_ms = int(record["completed_at_ms"])
        latency_seconds = float(record["latency_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"AI entry audit timing is invalid at line {line_number}") from exc
    if (
        completed_at_ms <= 0
        or completed_at_ms < case.observed_at_ms
        or not math.isfinite(latency_seconds)
        or latency_seconds < 0.0
    ):
        raise ValueError(f"AI entry audit timing is invalid at line {line_number}")
    return case, decision


def validate_live_ai_entry_audit_records(
    values: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Verify a complete ordered audit chain supplied by any storage adapter."""

    previous_sha256 = _ZERO_SHA256
    case_ids: set[str] = set()
    records: list[dict[str, object]] = []
    for line_number, value in enumerate(values, start=1):
        if not isinstance(value, Mapping):
            raise ValueError(f"AI entry audit record is invalid at line {line_number}")
        record = dict(value)
        record_sha = record.get("record_sha256")
        if not isinstance(record_sha, str) or len(record_sha) != 64:
            raise ValueError(f"AI entry audit hash is invalid at line {line_number}")
        unsigned = dict(record)
        unsigned.pop("record_sha256", None)
        if unsigned.get("previous_record_sha256") != previous_sha256:
            raise ValueError(f"AI entry audit chain is broken at line {line_number}")
        if _canonical_sha256(unsigned) != record_sha:
            raise ValueError(f"AI entry audit record is corrupted at line {line_number}")
        case, _decision = _audit_case_and_decision(record, line_number=line_number)
        if case.case_id in case_ids:
            raise ValueError(f"AI entry audit repeats a case at line {line_number}")
        case_ids.add(case.case_id)
        records.append(record)
        previous_sha256 = record_sha
    return tuple(records)


def load_live_ai_entry_audit(path: Path) -> tuple[dict[str, object], ...]:
    """Load and semantically verify every append-only AI shadow record."""

    audit_path = Path(path)
    if not audit_path.exists():
        return ()
    raw_records: list[dict[str, object]] = []
    with audit_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"AI entry audit has a blank record at line {line_number}")
            if len(line.encode("utf-8")) > _MAX_AUDIT_RECORD_BYTES:
                raise ValueError(f"AI entry audit record is too large at line {line_number}")
            raw_records.append(dict(_strict_json_object(line)))
    return validate_live_ai_entry_audit_records(raw_records)


class _HashChainedReviewLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.previous_sha256 = _ZERO_SHA256
        records = load_live_ai_entry_audit(self.path)
        if records:
            self.previous_sha256 = str(records[-1]["record_sha256"])

    def append(self, *, case: LiveAIEntryCase, review: LiveAIEntryReview, completed_at_ms: int) -> None:
        if review.decision is None:
            raise ValueError("pending AI reviews cannot be audited as completed")
        unsigned = {
            "schema_version": LIVE_AI_ENTRY_AUDIT_SCHEMA_VERSION,
            "previous_record_sha256": self.previous_sha256,
            "completed_at_ms": int(completed_at_ms),
            "latency_seconds": float(review.latency_seconds),
            "case": case.identity_payload() | {"case_id": case.case_id},
            "decision": review.decision.asdict(),
            "mode": "shadow_only",
            "trading_authority": False,
        }
        record_sha = _canonical_sha256(unsigned)
        record = unsigned | {"record_sha256": record_sha}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.previous_sha256 = record_sha


class AsyncLiveAIEntryReviewer:
    """Bounded single-worker reviewer that never blocks the trading loop."""

    def __init__(
        self,
        provider: LiveAIEntryProvider,
        *,
        audit_path: Path = Path("data/autonomous/ai-entry-reviews.jsonl"),
        clock: Callable[[], float] = time.time,
        perf_counter: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._provider = provider
        self._clock = clock
        self._perf_counter = perf_counter
        self._audit = _HashChainedReviewLog(Path(audit_path))
        self._condition = threading.Condition()
        self._next_case: LiveAIEntryCase | None = None
        self._active_case_id = ""
        self._completed: OrderedDict[str, LiveAIEntryReview] = OrderedDict()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._fatal_reason = ""

    def _ensure_worker_locked(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._worker,
            name="live-ai-entry-shadow-review",
            daemon=True,
        )
        self._thread.start()

    def review(self, case: LiveAIEntryCase) -> LiveAIEntryReview:
        case.validated()
        with self._condition:
            if self._closed:
                return LiveAIEntryReview(case.case_id, "shadow_failure", _failed_decision("reviewer closed"), 0.0)
            if self._fatal_reason:
                return LiveAIEntryReview(
                    case.case_id,
                    "shadow_failure",
                    _failed_decision(self._fatal_reason),
                    0.0,
                )
            completed = self._completed.get(case.case_id)
            if completed is not None:
                return completed
            if self._active_case_id == case.case_id or (
                self._next_case is not None and self._next_case.case_id == case.case_id
            ):
                return LiveAIEntryReview(case.case_id, "shadow_pending", None, 0.0)
            if self._next_case is not None:
                return LiveAIEntryReview(
                    case.case_id,
                    "shadow_failure",
                    _failed_decision("review queue full; case was not submitted"),
                    0.0,
                )
            self._next_case = case
            self._ensure_worker_locked()
            self._condition.notify_all()
            return LiveAIEntryReview(case.case_id, "shadow_pending", None, 0.0)

    def _worker(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._next_case is not None)
                if self._closed:
                    return
                case = self._next_case
                self._next_case = None
                if case is None:
                    continue
                self._active_case_id = case.case_id
            started = self._perf_counter()
            try:
                decision = self._provider(case)
                if not isinstance(decision, LiveAIEntryDecision):
                    raise TypeError("AI entry provider returned an invalid decision type")
                decision.validated_for(case)
            except Exception as exc:  # noqa: BLE001 - provider failures are fail-closed evidence
                decision = _failed_decision(f"{type(exc).__name__}: {exc}")
            latency = max(0.0, float(self._perf_counter() - started))
            status = (
                "shadow_failure"
                if not decision.valid
                else {
                    "approve": "shadow_approve",
                    "veto": "shadow_veto",
                    "cooldown": "shadow_cooldown",
                }[decision.action]
            )
            completed_at_ms = int(self._clock() * 1000)
            review = LiveAIEntryReview(
                case.case_id,
                status,
                decision,
                latency,
                completed_at_ms,
            )
            try:
                self._audit.append(
                    case=case,
                    review=review,
                    completed_at_ms=completed_at_ms,
                )
            except Exception as exc:  # noqa: BLE001 - corrupt audit blocks further AI evidence
                review = LiveAIEntryReview(
                    case.case_id,
                    "shadow_failure",
                    _failed_decision(f"audit:{type(exc).__name__}: {exc}"),
                    latency,
                    completed_at_ms,
                )
                fatal_reason = review.reason
            else:
                fatal_reason = ""
            with self._condition:
                self._active_case_id = ""
                self._completed[case.case_id] = review
                self._completed.move_to_end(case.case_id)
                while len(self._completed) > _MAX_COMPLETED_REVIEWS:
                    self._completed.popitem(last=False)
                if fatal_reason:
                    self._fatal_reason = fatal_reason
                self._condition.notify_all()

    def close(self, timeout_seconds: float = 0.25) -> bool:
        with self._condition:
            self._closed = True
            self._next_case = None
            self._condition.notify_all()
            thread = self._thread
        if thread is None:
            return True
        thread.join(max(0.0, float(timeout_seconds)))
        return not thread.is_alive()


class AIAssistedDecisionFunction:
    """Decorate the ML decision function with shadow-only AI observations."""

    def __init__(
        self,
        base_decision_fn: Callable[..., object],
        reviewer: AsyncLiveAIEntryReviewer,
        *,
        model_digest: str,
        terminal_model_fingerprint: str,
        maximum_review_age_seconds: int = 300,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._base_decision_fn = base_decision_fn
        self._reviewer = reviewer
        self._model_digest = model_digest
        self._terminal_model_fingerprint = terminal_model_fingerprint
        self._maximum_review_age_ms = int(maximum_review_age_seconds) * 1_000
        self._clock = clock
        self._entry_review_required = True
        self._model_artifact = getattr(base_decision_fn, "_model_artifact", None)
        self._model_validation_evidence = _model_validation_evidence(
            self._model_artifact
        )
        if not 1_000 <= self._maximum_review_age_ms <= 300_000:
            raise ValueError("live AI maximum review age is invalid")
        for name, value in {
            "model_digest": self._model_digest,
            "terminal_model_fingerprint": self._terminal_model_fingerprint,
        }.items():
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"live AI {name} is invalid")
        for attribute in ("_effective_strategy", "_model_artifact"):
            if hasattr(base_decision_fn, attribute):
                setattr(self, attribute, getattr(base_decision_fn, attribute))

    def set_entry_review_required(self, required: bool) -> None:
        """Tell the wrapper whether this iteration can reach a new entry."""

        self._entry_review_required = bool(required)

    def __call__(self, client: object, runtime: object, strategy: object, objective: object) -> object:
        decision = self._base_decision_fn(client, runtime, strategy, objective)
        side = str(getattr(decision, "side", ""))
        if side not in {"LONG", "SHORT"}:
            return replace(
                decision,
                ai_assist_mode="shadow_only",
                ai_assist_status="shadow_idle",
                ai_assist_reason="no directional ML proposal",
            )
        if not self._entry_review_required:
            return replace(
                decision,
                ai_assist_mode="shadow_only",
                ai_assist_status="shadow_idle",
                ai_assist_action="",
                ai_assist_case_id="",
                ai_assist_reason="coordinator reports no new-entry boundary",
                ai_assist_entry_ready=False,
            )
        proposed_size_multiplier = _finite_or_none(
            getattr(decision, "size_multiplier", 1.0)
        )
        if proposed_size_multiplier is None:
            return replace(
                decision,
                ai_assist_mode="shadow_only",
                ai_assist_status="shadow_failure",
                ai_assist_action="veto",
                ai_assist_reason="deterministic model proposed an invalid size",
                ai_assist_entry_ready=False,
            )
        if proposed_size_multiplier <= 0.0:
            return replace(
                decision,
                ai_assist_mode="shadow_only",
                ai_assist_status="shadow_idle",
                ai_assist_action="",
                ai_assist_case_id="",
                ai_assist_reason="deterministic model suppresses this entry",
                ai_assist_entry_ready=False,
            )
        proposal_evidence = dict(
            _mapping(getattr(decision, "ai_evidence", {}))
        )
        provided_payoff_evidence = proposal_evidence.pop(
            "after_cost_payoff",
            None,
        )
        predict_payoff_evidence = getattr(
            self._model_artifact,
            "predict_payoff_evidence",
            None,
        )
        if callable(predict_payoff_evidence):
            raw_features = getattr(decision, "model_features", ())
            try:
                model_features = tuple(float(value) for value in raw_features)
            except (TypeError, ValueError, OverflowError):
                model_features = ()
            if not model_features or any(
                not math.isfinite(value) for value in model_features
            ):
                return replace(
                    decision,
                    ai_assist_mode="shadow_only",
                    ai_assist_status="shadow_failure",
                    ai_assist_action="veto",
                    ai_assist_reason="model-bound AI payoff features are invalid",
                    ai_assist_entry_ready=False,
                )
            try:
                model_payoff_evidence = predict_payoff_evidence(
                    model_features,
                    proposed_side=side,
                )
            except (TypeError, ValueError, OverflowError):
                return replace(
                    decision,
                    ai_assist_mode="shadow_only",
                    ai_assist_status="shadow_failure",
                    ai_assist_action="veto",
                    ai_assist_reason="model-bound AI payoff evidence is invalid",
                    ai_assist_entry_ready=False,
                )
            if not isinstance(model_payoff_evidence, dict):
                return replace(
                    decision,
                    ai_assist_mode="shadow_only",
                    ai_assist_status="shadow_failure",
                    ai_assist_action="veto",
                    ai_assist_reason="model-bound AI payoff evidence is invalid",
                    ai_assist_entry_ready=False,
                )
            payoff_evidence_mismatch = False
            if provided_payoff_evidence is not None:
                try:
                    payoff_evidence_mismatch = _canonical_json(
                        provided_payoff_evidence
                    ) != _canonical_json(model_payoff_evidence)
                except (TypeError, ValueError, OverflowError):
                    payoff_evidence_mismatch = True
            if payoff_evidence_mismatch:
                return replace(
                    decision,
                    ai_assist_mode="shadow_only",
                    ai_assist_status="shadow_failure",
                    ai_assist_action="veto",
                    ai_assist_reason="supplied payoff evidence differs from the model",
                    ai_assist_entry_ready=False,
                )
            proposal_evidence["after_cost_payoff"] = model_payoff_evidence
        labeling = _mapping(self._model_validation_evidence.get("labeling"))
        model_barrier_bps = _finite_or_none(labeling.get("gross_label_barrier_bps"))
        round_trip_cost_floor_bps = configured_round_trip_cost_floor_bps(strategy)
        evidence = {
            "proposal": proposal_evidence,
            "risk_contract": {
                "risk_level": str(getattr(strategy, "risk_level", "")),
                "leverage": float(getattr(strategy, "leverage", 1.0)),
                "risk_per_trade": float(getattr(strategy, "risk_per_trade", 0.0)),
                "max_position_pct": float(getattr(strategy, "max_position_pct", 0.0)),
                "max_drawdown_limit": float(getattr(strategy, "max_drawdown_limit", 0.0)),
            },
            "cost_model": {
                "one_way_taker_fee_bps": float(
                    getattr(strategy, "taker_fee_bps", 0.0)
                ),
                "configured_spread_floor_bps": max(
                    float(getattr(strategy, "slippage_bps", 0.0)),
                    float(getattr(strategy, "max_spread_bps", 0.0)),
                ),
                "configured_round_trip_cost_floor_bps": round_trip_cost_floor_bps,
                "model_gross_label_barrier_bps": model_barrier_bps,
                "minimum_model_target_net_edge_bps": (
                    model_barrier_bps - round_trip_cost_floor_bps
                    if model_barrier_bps is not None
                    else None
                ),
            },
            "regime": {
                "name": str(getattr(decision, "regime", "")),
                "confidence": float(getattr(decision, "regime_confidence", 0.0)),
                "unpredictability_score": getattr(decision, "regime_unpredictability_score", None),
                "notes": list(getattr(decision, "regime_notes", ())[:8]),
            },
            "meta_label": {
                "enabled": bool(getattr(decision, "meta_label_enabled", False)),
                "action": str(getattr(decision, "meta_label_action", "")),
                "reason": str(getattr(decision, "meta_label_reason", ""))[:180],
                "size_multiplier": float(getattr(decision, "size_multiplier", 1.0)),
                "signal_strength": float(
                    getattr(decision, "meta_label_signal_strength", 0.0)
                ),
                "validation_minimum_sample_count": int(
                    getattr(
                        decision,
                        "meta_label_validation_minimum_sample_count",
                        0,
                    )
                ),
                "validation_minimum_precision": float(
                    getattr(
                        decision,
                        "meta_label_validation_minimum_precision",
                        0.0,
                    )
                ),
                "validation_sample_count": int(
                    getattr(decision, "meta_label_validation_sample_count", 0)
                ),
                "validation_precision": float(
                    getattr(decision, "meta_label_validation_precision", 0.0)
                ),
                "expected_after_cost_return": float(
                    getattr(decision, "meta_label_expected_after_cost_return", 0.0)
                ),
                "expected_after_cost_pnl_quote": float(
                    getattr(decision, "meta_label_expected_after_cost_pnl", 0.0)
                ),
            },
            "model_validation": self._model_validation_evidence,
        }
        try:
            case = build_live_ai_entry_case(
                symbol=str(getattr(runtime, "symbol", "")),
                market_type=str(getattr(runtime, "market_type", "")),
                interval=str(getattr(runtime, "interval", "")),
                observed_at_ms=int(getattr(decision, "observed_at_ms", 0)),
                proposed_side=side,
                ml_confidence=float(getattr(decision, "confidence", 0.0)),
                maximum_risk_multiplier=min(
                    1.0,
                    max(0.0, proposed_size_multiplier),
                ),
                model_digest=self._model_digest,
                terminal_model_fingerprint=self._terminal_model_fingerprint,
                evidence=evidence,
            )
            if not _approval_evidence_is_valid(case):
                return replace(
                    decision,
                    ai_assist_mode="shadow_only",
                    ai_assist_status="shadow_failure",
                    ai_assist_action="veto",
                    ai_assist_case_id=case.case_id,
                    ai_assist_reason=(
                        "deterministic model evidence cannot support AI approval"
                    ),
                    ai_assist_entry_ready=False,
                )
            review = self._reviewer.review(case)
        except Exception as exc:  # noqa: BLE001 - shadow AI cannot stop deterministic execution
            return replace(
                decision,
                ai_assist_mode="shadow_only",
                ai_assist_status="shadow_failure",
                ai_assist_reason=f"{type(exc).__name__}: {exc}"[:240],
                ai_assist_entry_ready=False,
            )
        reviewed_action = review.decision.action if review.decision is not None else "pending"
        reviewed_risk = review.decision.risk_multiplier if review.decision is not None else 0.0
        now_ms = int(self._clock() * 1_000)
        review_age_ms = now_ms - int(review.completed_at_ms)
        entry_ready = bool(
            review.decision is not None
            and review.decision.valid
            and 0 <= review_age_ms <= self._maximum_review_age_ms
        )
        review_reason = review.reason
        review_status = review.status
        if review.decision is not None and review.decision.valid and not entry_ready:
            review_status = "shadow_failure"
            review_reason = "completed AI pre-entry review is stale or future-dated"
        return replace(
            decision,
            ai_assist_mode="shadow_only",
            ai_assist_status=review_status,
            ai_assist_case_id=case.case_id,
            ai_assist_action=reviewed_action,
            ai_assist_risk_multiplier=reviewed_risk,
            ai_assist_reason=review_reason[:240],
            ai_assist_entry_ready=entry_ready,
        )

    def close(self, timeout_seconds: float = 0.25) -> bool:
        return self._reviewer.close(timeout_seconds)


__all__ = [
    "AIAssistedDecisionFunction",
    "AsyncLiveAIEntryReviewer",
    "LIVE_AI_ENTRY_RESPONSE_SCHEMA",
    "LiveAIEntryCase",
    "LiveAIEntryDecision",
    "LiveAIEntryReview",
    "OllamaLiveAIEntryProvider",
    "build_live_ai_entry_case",
    "load_live_ai_entry_audit",
    "validate_live_ai_entry_audit_records",
]
