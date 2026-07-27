"""Deterministic adjudication of the retained Round 74 active result."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
import re

from .round74_active_qualification import (
    ROUND74_ACTIVE_PREFLIGHT_SHA256,
    ROUND74_ACTIVE_RESULT_SCHEMA_VERSION,
    _canonical_sha256,
    _durable_json_replace,
    _evaluate_operator_result,
    _sha256_file,
    _strict_json_object,
    load_round74_active_preflight,
)


ROUND74_ACTIVE_RESULT_EVIDENCE_RELATIVE_PATH = Path(
    "docs/model-research/action-value/"
    "round-074-v10-active-regime-qualification-result-2026-07-27.json"
)
ROUND74_ACTIVE_RESULT_FILE_SHA256 = (
    "c71a3c6be74f36c27b21b39b9b606fb7c77242cd592f5f1831319053f10ccccd"
)
ROUND74_ACTIVE_RESULT_SHA256 = (
    "baa31064a82fa6bb742f5de288661b7a8b19e3d88bddc6accbb8adaee84d20ab"
)
ROUND74_ACTIVE_ADJUDICATION_SCHEMA_VERSION = (
    "round-074-active-qualification-adjudication-v1"
)
ROUND74_ACTIVE_ADJUDICATION_RELATIVE_PATH = Path(
    "docs/model-research/action-value/"
    "round-074-v10-active-regime-qualification-adjudication-2026-07-27.json"
)
_ORIGINAL_IDENTITY_ERROR = "fresh_audit_or_report_identity_failed"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _load_source_result(repository: Path) -> dict[str, object]:
    path = repository.resolve() / ROUND74_ACTIVE_RESULT_EVIDENCE_RELATIVE_PATH
    if _sha256_file(path) != ROUND74_ACTIVE_RESULT_FILE_SHA256:
        raise ValueError("Round 74 active result file identity differs")
    result = _strict_json_object(
        path.read_text(encoding="utf-8"),
        "Round 74 active result evidence",
    )
    claimed = str(result.get("result_sha256", ""))
    canonical = dict(result)
    canonical.pop("result_sha256", None)
    verdict = result.get("verdict")
    if (
        claimed != ROUND74_ACTIVE_RESULT_SHA256
        or claimed != _canonical_sha256(canonical)
        or result.get("schema_version") != ROUND74_ACTIVE_RESULT_SCHEMA_VERSION
        or result.get("preflight_sha256") != ROUND74_ACTIVE_PREFLIGHT_SHA256
        or result.get("capture_return_code") != 0
        or result.get("watchdog_breaches") != []
        or result.get("credentials_used") is not False
        or result.get("orders_submitted") is not False
        or result.get("automatic_retry_permitted") is not False
        or not isinstance(verdict, Mapping)
        or verdict.get("outcome") != "failed"
        or verdict.get("active_qualified") is not False
        or verdict.get("capture_data_passed") is not True
        or verdict.get("activity_label") != "active"
        or verdict.get("errors") != [_ORIGINAL_IDENTITY_ERROR]
    ):
        raise ValueError("Round 74 active source result contract differs")
    return result


def build_round74_active_adjudication(
    repository: Path,
    *,
    adjudicated_at_utc: datetime | None = None,
) -> dict[str, object]:
    """Re-evaluate the immutable result against the actual v10 report schema."""

    root = repository.resolve()
    source = _load_source_result(root)
    supervisor = source.get("supervisor_report")
    audits = source.get("fresh_process_audits")
    if not isinstance(supervisor, Mapping) or not isinstance(audits, list):
        raise ValueError("Round 74 active source evidence sections differ")
    attempts = supervisor.get("attempts")
    if (
        not isinstance(attempts, list)
        or len(attempts) != 1
        or not isinstance(attempts[0], Mapping)
        or len(audits) != 1
        or not isinstance(audits[0], Mapping)
    ):
        raise ValueError("Round 74 active source evidence cardinality differs")
    attempt = attempts[0]
    audit = audits[0].get("audit")
    if (
        not isinstance(audit, Mapping)
        or "last_frame_sha256" in attempt
        or _SHA256_PATTERN.fullmatch(str(audit.get("last_frame_sha256", ""))) is None
    ):
        raise ValueError("Round 74 active v10 frame identity shape differs")

    corrected = _evaluate_operator_result(
        load_round74_active_preflight(root),
        capture_return_code=int(source["capture_return_code"]),
        supervisor=supervisor,
        supervisor_parse_error=str(source.get("supervisor_parse_error", "")),
        audits=audits,
        watchdog_breaches=source["watchdog_breaches"],
    )
    if (
        corrected.get("outcome") != "active_qualified"
        or corrected.get("active_qualified") is not True
        or corrected.get("capture_data_passed") is not True
        or corrected.get("activity_label") != "active"
        or corrected.get("errors") != []
    ):
        raise ValueError("Round 74 corrected active verdict did not pass exactly")

    observed = adjudicated_at_utc or datetime.now(timezone.utc)
    payload: dict[str, object] = {
        "schema_version": ROUND74_ACTIVE_ADJUDICATION_SCHEMA_VERSION,
        "adjudicated_at_utc": observed.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_result": {
            "path": str(ROUND74_ACTIVE_RESULT_EVIDENCE_RELATIVE_PATH).replace(
                "\\", "/"
            ),
            "file_sha256": ROUND74_ACTIVE_RESULT_FILE_SHA256,
            "result_sha256": ROUND74_ACTIVE_RESULT_SHA256,
        },
        "defect": {
            "identifier": "v10_report_frame_identity_schema_mismatch",
            "original_error": _ORIGINAL_IDENTITY_ERROR,
            "v10_report_contains_last_frame_sha256": False,
            "fresh_audit_contains_last_frame_sha256": True,
            "fresh_audit_stored_report_sha256_matches": (
                audit.get("stored_report_sha256") == corrected["capture_report_sha256"]
            ),
            "correction": (
                "validate the audited frame-chain hash shape and compare the "
                "fresh audit to v10 report fields that actually exist"
            ),
        },
        "adjudication": {
            "corrected_verdict": corrected,
            "capture_retried": False,
            "new_market_data_used": False,
            "capture_or_audit_data_mutated": False,
            "original_result_mutated": False,
        },
        "authority": {
            "orders_submitted": False,
            "credentials_used": False,
            "trading_or_model_promotion_authority": False,
            "profitability_or_edge_claim": False,
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def write_round74_active_adjudication(repository: Path) -> Path:
    """Write the adjudication exactly once."""

    root = repository.resolve()
    path = root / ROUND74_ACTIVE_ADJUDICATION_RELATIVE_PATH
    if path.exists():
        raise FileExistsError(path)
    payload = build_round74_active_adjudication(root)
    _durable_json_replace(path, payload)
    return path


__all__ = [
    "ROUND74_ACTIVE_ADJUDICATION_RELATIVE_PATH",
    "ROUND74_ACTIVE_ADJUDICATION_SCHEMA_VERSION",
    "ROUND74_ACTIVE_RESULT_EVIDENCE_RELATIVE_PATH",
    "ROUND74_ACTIVE_RESULT_FILE_SHA256",
    "ROUND74_ACTIVE_RESULT_SHA256",
    "build_round74_active_adjudication",
    "write_round74_active_adjudication",
]
