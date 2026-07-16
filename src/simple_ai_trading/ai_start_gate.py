"""Hash-bound AI governance gate for autonomous startup."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

from .ai_review import (
    AIReviewReport,
    load_ai_review_report,
    resolve_ollama_model_provenance,
)
from .ai_runtime import AIRuntimeConfig, AICapabilityReport, detect_ai_capabilities
from .terminal_holdout_ledger import (
    reservation_evidence_passed,
    terminal_model_fingerprint,
)
from .types import RuntimeConfig

DEFAULT_AI_REVIEW_PATH = Path("data/model_lab/ai_risk_review.json")

CapabilityDetector = Callable[[AIRuntimeConfig], AICapabilityReport]
ModelProvenance = Callable[[str, str, float], tuple[str, str]]


@dataclass(frozen=True)
class AIStartGateReport:
    status: str
    allowed: bool
    active: bool
    reason: str
    review_path: str
    review_sha256: str | None = None
    source_report_sha256: str | None = None
    model: str | None = None
    model_digest: str | None = None
    terminal_model_fingerprint: str | None = None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _strict_json_mapping(path: Path) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("AI review source model-lab report is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("AI review source model-lab report is not an object")
    return payload


def _fallback_or_block(
    *,
    runtime: RuntimeConfig,
    paper_mode: bool,
    review_path: Path,
    reason: str,
    review: AIReviewReport | None = None,
) -> AIStartGateReport:
    fallback = bool(paper_mode and runtime.ai_allow_paper_fallback)
    return AIStartGateReport(
        status="paper_fallback" if fallback else "blocked",
        allowed=fallback,
        active=False,
        reason=reason,
        review_path=str(review_path),
        review_sha256=review.report_sha256 if review is not None else None,
        source_report_sha256=(
            review.source_report_sha256 if review is not None else None
        ),
        model=review.model if review is not None else None,
        model_digest=review.model_digest if review is not None else None,
    )


def _accepted_outcome(
    source: Mapping[str, object],
    *,
    symbol: str,
    objective: str,
) -> Mapping[str, object]:
    accepted_symbols = source.get("accepted_symbols")
    if (
        not isinstance(accepted_symbols, list)
        or any(not isinstance(item, str) for item in accepted_symbols)
        or symbol not in accepted_symbols
    ):
        raise ValueError("runtime symbol is not accepted by the reviewed model lab")
    outcomes = source.get("outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("reviewed model-lab outcomes are missing")
    matches = [
        item
        for item in outcomes
        if isinstance(item, Mapping) and item.get("symbol") == symbol
    ]
    if len(matches) != 1 or matches[0].get("accepted") is not True:
        raise ValueError("reviewed runtime symbol outcome is missing or not accepted")
    scores = matches[0].get("objective_scores")
    if not isinstance(scores, Mapping) or objective not in scores:
        raise ValueError("reviewed runtime objective score is missing")
    return matches[0]


def _expected_terminal_fingerprint(
    outcome: Mapping[str, object],
    *,
    objective: str,
) -> str:
    selection = outcome.get("selection_risk")
    if not isinstance(selection, Mapping):
        raise ValueError("reviewed selection-risk evidence is missing")
    objective_selection = selection.get(objective)
    if (
        not isinstance(objective_selection, Mapping)
        or objective_selection.get("passed") is not True
    ):
        raise ValueError("reviewed objective did not pass selection-risk evidence")
    terminal = objective_selection.get("terminal_holdout")
    if not isinstance(terminal, Mapping) or terminal.get("passed") is not True:
        raise ValueError("reviewed terminal holdout is missing or failed")
    reservation = terminal.get("reservation")
    if not isinstance(reservation, Mapping) or not reservation_evidence_passed(
        reservation
    ):
        raise ValueError("reviewed terminal reservation evidence is invalid")
    fingerprint = reservation.get("model_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("reviewed terminal model fingerprint is invalid")
    return fingerprint


def evaluate_ai_start_gate(
    runtime: RuntimeConfig,
    *,
    objective: str,
    model_artifact: object | None,
    paper_mode: bool,
    review_path: Path = DEFAULT_AI_REVIEW_PATH,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 10.0,
    capability_detector: CapabilityDetector = detect_ai_capabilities,
    model_provenance: ModelProvenance = resolve_ollama_model_provenance,
) -> AIStartGateReport:
    """Require exact reviewed evidence before AI is active for a bot run."""

    path = Path(review_path)
    if not runtime.ai_enabled:
        return AIStartGateReport(
            status="disabled",
            allowed=True,
            active=False,
            reason="AI features are disabled",
            review_path=str(path),
        )
    try:
        review = load_ai_review_report(path)
    except (OSError, ValueError) as exc:
        return _fallback_or_block(
            runtime=runtime,
            paper_mode=paper_mode,
            review_path=path,
            reason=f"AI review is unavailable: {exc}",
        )
    try:
        capability = capability_detector(runtime.ai_runtime_config())
    except Exception as exc:
        return _fallback_or_block(
            runtime=runtime,
            paper_mode=paper_mode,
            review_path=path,
            reason=f"AI capability check failed: {exc}",
            review=review,
        )
    if not capability.ok:
        reason = "; ".join(capability.messages) or "AI capability check failed"
        return _fallback_or_block(
            runtime=runtime,
            paper_mode=paper_mode,
            review_path=path,
            reason=reason,
            review=review,
        )
    try:
        if review.status != "ok" or not review.approved:
            raise ValueError("AI review is not approved")
        if review.model != capability.model:
            raise ValueError("AI review model differs from the current runtime model")
        current_digest, current_metadata_sha256 = model_provenance(
            base_url,
            review.model,
            timeout_seconds,
        )
        if (
            current_digest != review.model_digest
            or current_metadata_sha256 != review.model_metadata_sha256
        ):
            raise ValueError("installed AI model provenance differs from the review")
        source_path = Path(review.source_report)
        source = _strict_json_mapping(source_path)
        expected_quote = str(source.get("quote_asset") or "")
        if (
            source.get("interval") != runtime.interval
            or source.get("market_type") != runtime.market_type
            or not expected_quote
            or not runtime.symbol.endswith(expected_quote)
        ):
            raise ValueError("AI review market contract differs from runtime")
        requested = source.get("requested_objectives")
        if (
            not isinstance(requested, list)
            or any(not isinstance(item, str) for item in requested)
            or objective not in requested
        ):
            raise ValueError("AI review objective differs from runtime")
        outcome = _accepted_outcome(
            source,
            symbol=runtime.symbol,
            objective=objective,
        )
        expected_fingerprint = _expected_terminal_fingerprint(
            outcome,
            objective=objective,
        )
        if model_artifact is None:
            raise ValueError("runtime model artifact is unavailable for AI binding")
        actual_fingerprint = terminal_model_fingerprint(model_artifact)
        if actual_fingerprint != expected_fingerprint:
            raise ValueError("runtime model fingerprint differs from the AI review")
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        return _fallback_or_block(
            runtime=runtime,
            paper_mode=paper_mode,
            review_path=path,
            reason=str(exc),
            review=review,
        )
    return AIStartGateReport(
        status="active",
        allowed=True,
        active=True,
        reason="AI review and runtime model provenance are exact matches",
        review_path=str(path),
        review_sha256=review.report_sha256,
        source_report_sha256=review.source_report_sha256,
        model=review.model,
        model_digest=review.model_digest,
        terminal_model_fingerprint=actual_fingerprint,
    )


__all__ = [
    "AIStartGateReport",
    "DEFAULT_AI_REVIEW_PATH",
    "evaluate_ai_start_gate",
]
