"""Durable one-shot authorization for preregistered local-AI benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re
import time
from typing import Mapping
from urllib.parse import urlparse

from .ai_model_benchmark import (
    AI_MODEL_BENCHMARK_CONTRACT,
    AIModelBenchmarkReport,
    default_finance_ai_test_cases,
    rescore_finance_ai_benchmark_payload,
)
from .ai_runtime import OllamaResidencyReport, ollama_residency_from_mapping
from .polymarket_continuity import (
    POLYMARKET_ACTION_CONTRACT_COMMITTED_AT_MS,
    POLYMARKET_CONTINUITY_ELIGIBILITY_SCHEMA_VERSION,
    evaluate_polymarket_continuity_eligibility,
)
from .polymarket_recorder import (
    POLYMARKET_STORAGE_SCHEMA_VERSION,
    PolymarketEvidenceStore,
)
from .storage import write_json_atomic


AI_BENCHMARK_CLAIM_SCHEMA_VERSION = "preregistered-ai-benchmark-claim-v2"
AI_BENCHMARK_PREREGISTRATION_SCHEMA_VERSION = (
    "finance-risk-review-candidate-preregistration-v3"
)
AI_BENCHMARK_RUNTIME_EVIDENCE_SCHEMA_VERSION = "preregistered-ai-benchmark-runtime-v2"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_JSON_BYTES = 32 * 1024 * 1024
_ADMISSIBLE_CONFIRMATION_STATUSES = ("complete", "degraded")
_MINIMUM_CONFIRMATION_DURATION_SECONDS = 54_000
_MINIMUM_CONTINUITY_GROUPS = 30
_APPROVED_PREREGISTRATION_SHA256 = {
    "qwen3:14b": "1d4293fcb7e818ade3567e960e95d2f184263158f101beadf1afb07ab33f3ced",
}


@dataclass(frozen=True)
class PreregisteredAIBenchmarkClaim:
    claim_sha256: str
    status: str
    model: str
    confirmation_run_id: str
    confirmation_report_sha256: str
    confirmation_recorder_status: str
    confirmation_storage_schema_version: str
    confirmation_started_at_ms: int
    confirmation_ended_at_ms: int
    confirmation_continuity_report_sha256: str
    confirmation_eligible_group_count: int
    preregistration_sha256: str
    output_path: str
    report_file_sha256: str = ""
    benchmark_passed: bool | None = None


def requires_preregistered_ai_benchmark(model: str) -> bool:
    """Return whether a model is protected by a frozen one-shot benchmark."""

    return str(model or "").strip().lower() in _APPROVED_PREREGISTRATION_SHA256


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> tuple[dict[str, object], bytes]:
    if not path.is_file() or not 1 <= path.stat().st_size <= _MAX_JSON_BYTES:
        raise ValueError(f"AI benchmark JSON has an invalid size: {path.name}")
    payload = path.read_bytes()
    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"AI benchmark JSON is invalid: {path.name}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"AI benchmark JSON root is not an object: {path.name}")
    return parsed, payload


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is not an object")
    return value


def _local_ollama_base_url(value: object) -> str:
    base_url = str(value or "").rstrip("/")
    parsed = urlparse(base_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("AI benchmark Ollama endpoint is invalid") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise ValueError("AI benchmark requires a local Ollama HTTP endpoint")
    return base_url


def _runtime_model_provenance(value: object, name: str) -> Mapping[str, object]:
    provenance = _mapping(value, name)
    if set(provenance) != {"model_digest", "model_metadata_sha256"} or any(
        _SHA256.fullmatch(str(provenance.get(field) or "")) is None
        for field in provenance
    ):
        raise ValueError(f"{name} is invalid")
    return provenance


def validate_preregistered_ai_runtime_evidence(
    value: object,
    *,
    model: str,
    case_count: int,
    base_url: str,
    claim: PreregisteredAIBenchmarkClaim | None = None,
) -> dict[str, object]:
    """Validate exact inference weights, endpoint, claim, and GPU residency."""

    evidence = _mapping(value, "AI benchmark inference runtime evidence")
    expected_fields = {
        "schema_version",
        "provider",
        "model",
        "benchmark_contract",
        "base_url",
        "case_count",
        "claim",
        "pre_inference",
        "post_inference",
        "residency",
    }
    if (
        set(evidence) != expected_fields
        or evidence.get("schema_version")
        != AI_BENCHMARK_RUNTIME_EVIDENCE_SCHEMA_VERSION
        or evidence.get("provider") != "ollama"
        or evidence.get("model") != model
        or evidence.get("benchmark_contract") != AI_MODEL_BENCHMARK_CONTRACT
        or evidence.get("base_url") != _local_ollama_base_url(base_url)
        or isinstance(evidence.get("case_count"), bool)
        or evidence.get("case_count") != case_count
    ):
        raise ValueError("AI benchmark inference runtime evidence is inconsistent")
    claim_evidence = _mapping(evidence.get("claim"), "AI benchmark claim evidence")
    claim_fields = {
        "claim_sha256",
        "confirmation_run_id",
        "confirmation_report_sha256",
        "confirmation_recorder_status",
        "confirmation_storage_schema_version",
        "confirmation_started_at_ms",
        "confirmation_ended_at_ms",
        "confirmation_continuity_report_sha256",
        "confirmation_eligible_group_count",
        "preregistration_sha256",
    }
    if (
        set(claim_evidence) != claim_fields
        or _SHA256.fullmatch(str(claim_evidence.get("claim_sha256") or "")) is None
        or not str(claim_evidence.get("confirmation_run_id") or "")
        or _SHA256.fullmatch(
            str(claim_evidence.get("confirmation_report_sha256") or "")
        )
        is None
        or claim_evidence.get("confirmation_recorder_status")
        not in _ADMISSIBLE_CONFIRMATION_STATUSES
        or claim_evidence.get("confirmation_storage_schema_version")
        != POLYMARKET_STORAGE_SCHEMA_VERSION
        or isinstance(claim_evidence.get("confirmation_started_at_ms"), bool)
        or not isinstance(claim_evidence.get("confirmation_started_at_ms"), int)
        or isinstance(claim_evidence.get("confirmation_ended_at_ms"), bool)
        or not isinstance(claim_evidence.get("confirmation_ended_at_ms"), int)
        or int(claim_evidence.get("confirmation_started_at_ms", 0))
        < POLYMARKET_ACTION_CONTRACT_COMMITTED_AT_MS
        or int(claim_evidence.get("confirmation_ended_at_ms", 0))
        - int(claim_evidence.get("confirmation_started_at_ms", 0))
        < _MINIMUM_CONFIRMATION_DURATION_SECONDS * 1_000
        or _SHA256.fullmatch(
            str(claim_evidence.get("confirmation_continuity_report_sha256") or "")
        )
        is None
        or isinstance(claim_evidence.get("confirmation_eligible_group_count"), bool)
        or not isinstance(claim_evidence.get("confirmation_eligible_group_count"), int)
        or int(claim_evidence.get("confirmation_eligible_group_count", 0))
        < _MINIMUM_CONTINUITY_GROUPS
        or _SHA256.fullmatch(str(claim_evidence.get("preregistration_sha256") or ""))
        is None
    ):
        raise ValueError("AI benchmark claim evidence is invalid")
    if claim is not None and dict(claim_evidence) != {
        "claim_sha256": claim.claim_sha256,
        "confirmation_run_id": claim.confirmation_run_id,
        "confirmation_report_sha256": claim.confirmation_report_sha256,
        "confirmation_recorder_status": claim.confirmation_recorder_status,
        "confirmation_storage_schema_version": (
            claim.confirmation_storage_schema_version
        ),
        "confirmation_started_at_ms": claim.confirmation_started_at_ms,
        "confirmation_ended_at_ms": claim.confirmation_ended_at_ms,
        "confirmation_continuity_report_sha256": (
            claim.confirmation_continuity_report_sha256
        ),
        "confirmation_eligible_group_count": (claim.confirmation_eligible_group_count),
        "preregistration_sha256": claim.preregistration_sha256,
    }:
        raise ValueError("AI benchmark runtime evidence differs from its claim")
    pre = _runtime_model_provenance(
        evidence.get("pre_inference"), "AI benchmark pre-inference provenance"
    )
    post = _runtime_model_provenance(
        evidence.get("post_inference"), "AI benchmark post-inference provenance"
    )
    if dict(pre) != dict(post):
        raise ValueError("Ollama model identity changed during AI benchmark inference")
    residency = ollama_residency_from_mapping(evidence.get("residency"))
    if (
        residency.requested_model != model
        or residency.status != "gpu_resident"
        or residency.digest != pre["model_digest"]
        or not residency.gpu_resident
        or residency.size_vram_bytes is None
        or residency.size_vram_bytes <= 0
    ):
        raise ValueError(
            "AI benchmark inference was not bound to exact GPU-resident weights"
        )
    return dict(evidence)


def write_preregistered_ai_benchmark_output(
    report: AIModelBenchmarkReport,
    output_path: str | Path,
    *,
    claim: PreregisteredAIBenchmarkClaim,
    pre_model_digest: str,
    pre_model_metadata_sha256: str,
    post_model_digest: str,
    post_model_metadata_sha256: str,
    residency: OllamaResidencyReport,
) -> Path:
    """Atomically persist a one-shot report with exact inference-time evidence."""

    payload = report.asdict()
    evidence: dict[str, object] = {
        "schema_version": AI_BENCHMARK_RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "provider": "ollama",
        "model": claim.model,
        "benchmark_contract": AI_MODEL_BENCHMARK_CONTRACT,
        "base_url": _local_ollama_base_url(report.base_url),
        "case_count": len(report.tests),
        "claim": {
            "claim_sha256": claim.claim_sha256,
            "confirmation_run_id": claim.confirmation_run_id,
            "confirmation_report_sha256": claim.confirmation_report_sha256,
            "confirmation_recorder_status": claim.confirmation_recorder_status,
            "confirmation_storage_schema_version": (
                claim.confirmation_storage_schema_version
            ),
            "confirmation_started_at_ms": claim.confirmation_started_at_ms,
            "confirmation_ended_at_ms": claim.confirmation_ended_at_ms,
            "confirmation_continuity_report_sha256": (
                claim.confirmation_continuity_report_sha256
            ),
            "confirmation_eligible_group_count": (
                claim.confirmation_eligible_group_count
            ),
            "preregistration_sha256": claim.preregistration_sha256,
        },
        "pre_inference": {
            "model_digest": pre_model_digest,
            "model_metadata_sha256": pre_model_metadata_sha256,
        },
        "post_inference": {
            "model_digest": post_model_digest,
            "model_metadata_sha256": post_model_metadata_sha256,
        },
        "residency": residency.asdict(),
    }
    payload["inference_runtime_evidence"] = evidence
    rescored = rescore_finance_ai_benchmark_payload(payload)
    if (
        rescored.benchmark_contract != AI_MODEL_BENCHMARK_CONTRACT
        or len(rescored.results) != 1
        or rescored.results[0].model != claim.model
        or not rescored.results[0].installed
        or payload.get("selected_model") != rescored.selected_model
        or payload.get("passed") is not rescored.passed
    ):
        raise ValueError("preregistered AI benchmark report is inconsistent")
    validate_preregistered_ai_runtime_evidence(
        evidence,
        model=claim.model,
        case_count=len(report.tests),
        base_url=report.base_url,
        claim=claim,
    )
    path = Path(output_path)
    write_json_atomic(path, payload, indent=2, sort_keys=True)
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_preregistration(
    path: Path,
    *,
    model: str,
    timeout_seconds: float,
    minimum_score: float,
) -> tuple[dict[str, object], str]:
    payload, encoded = _read_json(path)
    preregistration_sha256 = hashlib.sha256(encoded).hexdigest()
    candidate = _mapping(payload.get("candidate"), "AI benchmark candidate")
    frozen = _mapping(payload.get("frozen_run"), "AI benchmark frozen run")
    admission = _mapping(payload.get("admission"), "AI benchmark admission")
    suite = [asdict(case) for case in default_finance_ai_test_cases()]
    suite_sha256 = _sha256(suite)
    source_sha256 = _file_sha256(Path(__file__).with_name("ai_model_benchmark.py"))
    try:
        frozen_timeout = float(frozen.get("timeout_seconds"))
        frozen_score = float(frozen.get("minimum_score"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("AI benchmark frozen numeric settings are invalid") from exc
    if (
        _APPROVED_PREREGISTRATION_SHA256.get(model) != preregistration_sha256
        or payload.get("schema_version") != AI_BENCHMARK_PREREGISTRATION_SCHEMA_VERSION
        or payload.get("benchmark_contract") != AI_MODEL_BENCHMARK_CONTRACT
        or candidate.get("model") != model
        or payload.get("benchmark_source_sha256") != source_sha256
        or payload.get("test_suite_sha256") != suite_sha256
        or frozen.get("case_count") != len(suite)
        or frozen.get("run_count") != 1
        or frozen.get("run_after_valid_confirmation_recorder_finalization") is not True
        or frozen.get("admissible_recorder_statuses")
        != list(_ADMISSIBLE_CONFIRMATION_STATUSES)
        or frozen.get("minimum_capture_duration_seconds")
        != _MINIMUM_CONFIRMATION_DURATION_SECONDS
        or frozen.get("minimum_capture_started_at_ms")
        != POLYMARKET_ACTION_CONTRACT_COMMITTED_AT_MS
        or frozen.get("required_storage_schema_version")
        != POLYMARKET_STORAGE_SCHEMA_VERSION
        or frozen.get("required_continuity_schema_version")
        != POLYMARKET_CONTINUITY_ELIGIBILITY_SCHEMA_VERSION
        or frozen.get("minimum_eligible_synchronized_groups")
        != _MINIMUM_CONTINUITY_GROUPS
        or frozen.get("global_gap_free_required") is not False
        or frozen.get("gaps_inside_eligible_windows_allowed") is not False
        or frozen.get("prompt_or_case_changes_allowed") is not False
        or frozen.get("temperature") != 0
        or frozen.get("thinking") is not False
        or frozen_timeout != float(timeout_seconds)
        or frozen_score != float(minimum_score)
        or admission.get("minimum_score_required") != float(minimum_score)
        or not all(
            admission.get(name) is True
            for name in (
                "top_level_json_object_only",
                "duplicate_keys_forbidden",
                "exact_required_fields_and_types",
                "finite_numeric_ranges_required",
                "all_actions_exact",
                "all_risk_ranges_exact",
                "all_required_terms_present",
            )
        )
    ):
        raise ValueError("AI benchmark preregistration differs from frozen code or run")
    return payload, preregistration_sha256


def _validate_prior_comparison(
    preregistration_path: Path,
    preregistration: Mapping[str, object],
) -> None:
    prior = preregistration_path.parent / "latest" / "comparison.json"
    expected = str(preregistration.get("prior_comparison_sha256") or "")
    if _SHA256.fullmatch(expected) is None or _file_sha256(prior) != expected:
        raise ValueError("AI benchmark prior comparison differs from preregistration")


def _ensure_table(store: PolymarketEvidenceStore) -> None:
    store.connect().execute(
        """
        CREATE TABLE IF NOT EXISTS preregistered_ai_benchmark_claim (
            claim_sha256 VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            preregistration_sha256 VARCHAR NOT NULL,
            confirmation_report_sha256 VARCHAR NOT NULL,
            identity_json VARCHAR NOT NULL,
            state VARCHAR NOT NULL CHECK(state IN ('started', 'failed', 'completed')),
            report_file_sha256 VARCHAR,
            benchmark_passed BOOLEAN,
            failure_sha256 VARCHAR,
            started_at_ms UBIGINT NOT NULL,
            completed_at_ms UBIGINT,
            UNIQUE(preregistration_sha256, confirmation_report_sha256)
        )
        """
    )


def _claim_from_row(
    identity: Mapping[str, object],
    row: tuple[object, ...],
    *,
    status: str,
) -> PreregisteredAIBenchmarkClaim:
    return PreregisteredAIBenchmarkClaim(
        claim_sha256=str(row[0]),
        status=status,
        model=str(identity["model"]),
        confirmation_run_id=str(identity["confirmation_run_id"]),
        confirmation_report_sha256=str(identity["confirmation_report_sha256"]),
        confirmation_recorder_status=str(identity["confirmation_recorder_status"]),
        confirmation_storage_schema_version=str(
            identity["confirmation_storage_schema_version"]
        ),
        confirmation_started_at_ms=int(identity["confirmation_started_at_ms"]),
        confirmation_ended_at_ms=int(identity["confirmation_ended_at_ms"]),
        confirmation_continuity_report_sha256=str(
            identity["confirmation_continuity_report_sha256"]
        ),
        confirmation_eligible_group_count=int(
            identity["confirmation_eligible_group_count"]
        ),
        preregistration_sha256=str(identity["preregistration_sha256"]),
        output_path=str(identity["output_path"]),
        report_file_sha256=str(row[3] or ""),
        benchmark_passed=(None if row[4] is None else bool(row[4])),
    )


def load_claimed_ai_benchmark_output(
    claim: PreregisteredAIBenchmarkClaim,
) -> dict[str, object]:
    """Load and fully rescore the immutable output bound to a completed claim."""

    path = Path(claim.output_path)
    payload, encoded = _read_json(path)
    digest = hashlib.sha256(encoded).hexdigest()
    if claim.report_file_sha256 and digest != claim.report_file_sha256:
        raise ValueError("claimed AI benchmark output digest differs")
    report = rescore_finance_ai_benchmark_payload(payload)
    if (
        report.benchmark_contract != AI_MODEL_BENCHMARK_CONTRACT
        or len(report.results) != 1
        or report.results[0].model != claim.model
        or not report.results[0].installed
        or payload.get("selected_model") != report.selected_model
        or payload.get("passed") is not report.passed
        or payload.get("financial_edge_tested") is not False
        or payload.get("trading_authority") is not False
        or (
            claim.benchmark_passed is not None
            and report.passed is not claim.benchmark_passed
        )
    ):
        raise ValueError("claimed AI benchmark output is inconsistent")
    validate_preregistered_ai_runtime_evidence(
        payload.get("inference_runtime_evidence"),
        model=claim.model,
        case_count=len(report.tests),
        base_url=report.base_url,
        claim=claim,
    )
    return payload


def begin_preregistered_ai_benchmark_claim(
    store: PolymarketEvidenceStore,
    *,
    preregistration_path: Path,
    confirmation_run_id: str,
    model: str,
    timeout_seconds: float,
    minimum_score: float,
    output_path: Path,
) -> PreregisteredAIBenchmarkClaim:
    """Audit confirmation evidence and durably consume a one-shot benchmark."""

    selected_run = str(confirmation_run_id or "").strip()
    selected_model = str(model or "").strip()
    if not selected_run or not selected_model:
        raise ValueError("preregistered AI benchmark identity is incomplete")
    preregistration, preregistration_sha256 = _validated_preregistration(
        preregistration_path,
        model=selected_model,
        timeout_seconds=timeout_seconds,
        minimum_score=minimum_score,
    )
    run = (
        store.connect()
        .execute(
            """
        SELECT status, error, report_sha256, storage_schema_version,
               started_at_ms, ended_at_ms
        FROM polymarket_recorder_run WHERE run_id = ?
        """,
            [selected_run],
        )
        .fetchone()
    )
    if run is None:
        raise ValueError("AI benchmark confirmation recorder is unknown")
    recorder_status = str(run[0])
    report_sha256 = str(run[2] or "")
    storage_schema_version = str(run[3] or "")
    try:
        started_at_ms = int(run[4])
        ended_at_ms = int(run[5])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("AI benchmark confirmation timing is invalid") from exc
    if (
        recorder_status not in _ADMISSIBLE_CONFIRMATION_STATUSES
        or str(run[1] or "").strip()
        or _SHA256.fullmatch(report_sha256) is None
        or storage_schema_version != POLYMARKET_STORAGE_SCHEMA_VERSION
        or started_at_ms < POLYMARKET_ACTION_CONTRACT_COMMITTED_AT_MS
        or ended_at_ms - started_at_ms < _MINIMUM_CONFIRMATION_DURATION_SECONDS * 1_000
    ):
        raise ValueError(
            "AI benchmark confirmation recorder does not satisfy the frozen "
            "storage, timing, or terminal-status contract"
        )
    output = str(output_path.resolve())
    base_identity = {
        "schema_version": AI_BENCHMARK_CLAIM_SCHEMA_VERSION,
        "benchmark_contract": AI_MODEL_BENCHMARK_CONTRACT,
        "preregistration_sha256": preregistration_sha256,
        "benchmark_source_sha256": preregistration["benchmark_source_sha256"],
        "test_suite_sha256": preregistration["test_suite_sha256"],
        "confirmation_run_id": selected_run,
        "confirmation_report_sha256": report_sha256,
        "confirmation_recorder_status": recorder_status,
        "confirmation_storage_schema_version": storage_schema_version,
        "confirmation_started_at_ms": started_at_ms,
        "confirmation_ended_at_ms": ended_at_ms,
        "model": selected_model,
        "timeout_seconds": float(timeout_seconds),
        "minimum_score": float(minimum_score),
        "output_path": output,
    }
    _ensure_table(store)
    connection = store.connect()
    query = """
        SELECT claim_sha256, identity_json, state, report_file_sha256,
               benchmark_passed
        FROM preregistered_ai_benchmark_claim
        WHERE preregistration_sha256 = ?
    """
    parameters = [preregistration_sha256]

    def existing_claim(
        row: tuple[object, ...],
        *,
        expected_identity: Mapping[str, object] | None = None,
    ) -> PreregisteredAIBenchmarkClaim:
        try:
            stored_identity = json.loads(
                str(row[1]),
                object_pairs_hook=_strict_object,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "preregistered AI benchmark claim identity is invalid"
            ) from exc
        if not isinstance(stored_identity, dict):
            raise ValueError("preregistered AI benchmark claim identity is invalid")
        continuity_sha256 = str(
            stored_identity.get("confirmation_continuity_report_sha256") or ""
        )
        eligible_group_count = stored_identity.get("confirmation_eligible_group_count")
        if (
            set(base_identity).difference(stored_identity)
            or any(
                stored_identity.get(key) != value
                for key, value in base_identity.items()
            )
            or set(stored_identity)
            != {
                *base_identity,
                "confirmation_continuity_report_sha256",
                "confirmation_eligible_group_count",
            }
            or _SHA256.fullmatch(continuity_sha256) is None
            or isinstance(eligible_group_count, bool)
            or not isinstance(eligible_group_count, int)
            or eligible_group_count < _MINIMUM_CONTINUITY_GROUPS
            or str(row[1]) != _canonical_json(stored_identity)
            or str(row[0]) != _sha256(stored_identity)
            or (
                expected_identity is not None
                and dict(stored_identity) != dict(expected_identity)
            )
        ):
            raise ValueError("preregistered AI benchmark claim identity differs")
        if str(row[2]) != "completed":
            raise ValueError(
                f"preregistered AI benchmark is already claimed:state={row[2]}"
            )
        claim = _claim_from_row(stored_identity, row, status="existing")
        load_claimed_ai_benchmark_output(claim)
        return claim

    row = connection.execute(query, parameters).fetchone()
    if row is not None:
        return existing_claim(row)
    integrity = store.integrity_errors(selected_run)
    if integrity:
        raise ValueError(
            "AI benchmark confirmation integrity failed: " + "; ".join(integrity)
        )
    continuity = evaluate_polymarket_continuity_eligibility(
        store,
        run_id=selected_run,
    )
    if (
        continuity.run_id != selected_run
        or not continuity.confirmation_eligible
        or continuity.eligible_group_count < _MINIMUM_CONTINUITY_GROUPS
        or _SHA256.fullmatch(continuity.report_sha256) is None
    ):
        raise ValueError(
            "AI benchmark confirmation lacks enough hash-bound, label-free "
            "continuity-eligible synchronized groups"
        )
    identity = {
        **base_identity,
        "confirmation_continuity_report_sha256": continuity.report_sha256,
        "confirmation_eligible_group_count": continuity.eligible_group_count,
    }
    identity_json = _canonical_json(identity)
    claim_sha256 = hashlib.sha256(identity_json.encode("ascii")).hexdigest()
    if Path(output).exists():
        raise ValueError("preregistered AI benchmark output already exists")
    _validate_prior_comparison(preregistration_path, preregistration)
    connection.execute("BEGIN TRANSACTION")
    try:
        row = connection.execute(query, parameters).fetchone()
        if row is not None:
            claim = existing_claim(row, expected_identity=identity)
            connection.execute("COMMIT")
            return claim
        if Path(output).exists():
            raise ValueError("preregistered AI benchmark output already exists")
        now_ms = time.time_ns() // 1_000_000
        connection.execute(
            """
            INSERT INTO preregistered_ai_benchmark_claim VALUES (
                ?, ?, ?, ?, ?, 'started', NULL, NULL, NULL, ?, NULL
            )
            """,
            [
                claim_sha256,
                AI_BENCHMARK_CLAIM_SCHEMA_VERSION,
                preregistration_sha256,
                report_sha256,
                identity_json,
                now_ms,
            ],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return PreregisteredAIBenchmarkClaim(
        claim_sha256=claim_sha256,
        status="claimed",
        model=selected_model,
        confirmation_run_id=selected_run,
        confirmation_report_sha256=report_sha256,
        confirmation_recorder_status=recorder_status,
        confirmation_storage_schema_version=storage_schema_version,
        confirmation_started_at_ms=started_at_ms,
        confirmation_ended_at_ms=ended_at_ms,
        confirmation_continuity_report_sha256=continuity.report_sha256,
        confirmation_eligible_group_count=continuity.eligible_group_count,
        preregistration_sha256=preregistration_sha256,
        output_path=output,
    )


def complete_preregistered_ai_benchmark_claim(
    store: PolymarketEvidenceStore,
    claim: PreregisteredAIBenchmarkClaim,
) -> PreregisteredAIBenchmarkClaim:
    """Bind a strict benchmark output to its started one-shot claim."""

    payload, encoded = _read_json(Path(claim.output_path))
    report_file_sha256 = hashlib.sha256(encoded).hexdigest()
    provisional = replace(
        claim,
        report_file_sha256=report_file_sha256,
        benchmark_passed=bool(payload.get("passed")),
    )
    load_claimed_ai_benchmark_output(provisional)
    connection = store.connect()
    row = connection.execute(
        """
        SELECT state FROM preregistered_ai_benchmark_claim
        WHERE claim_sha256 = ?
        """,
        [claim.claim_sha256],
    ).fetchone()
    if row is None or str(row[0]) != "started":
        raise ValueError("preregistered AI benchmark completion claim is invalid")
    connection.execute(
        """
        UPDATE preregistered_ai_benchmark_claim
        SET state = 'completed', report_file_sha256 = ?, benchmark_passed = ?,
            failure_sha256 = NULL, completed_at_ms = ?
        WHERE claim_sha256 = ? AND state = 'started'
        """,
        [
            report_file_sha256,
            provisional.benchmark_passed,
            time.time_ns() // 1_000_000,
            claim.claim_sha256,
        ],
    )
    return replace(provisional, status="completed")


def fail_preregistered_ai_benchmark_claim(
    store: PolymarketEvidenceStore,
    claim: PreregisteredAIBenchmarkClaim,
    error: BaseException,
) -> None:
    """Persist failure so test cases cannot be silently reopened."""

    failure_sha256 = _sha256(
        {"error_type": type(error).__name__, "error_message": str(error)}
    )
    store.connect().execute(
        """
        UPDATE preregistered_ai_benchmark_claim
        SET state = 'failed', failure_sha256 = ?, completed_at_ms = ?
        WHERE claim_sha256 = ? AND state = 'started'
        """,
        [
            failure_sha256,
            time.time_ns() // 1_000_000,
            claim.claim_sha256,
        ],
    )


__all__ = [
    "AI_BENCHMARK_CLAIM_SCHEMA_VERSION",
    "AI_BENCHMARK_RUNTIME_EVIDENCE_SCHEMA_VERSION",
    "PreregisteredAIBenchmarkClaim",
    "begin_preregistered_ai_benchmark_claim",
    "complete_preregistered_ai_benchmark_claim",
    "fail_preregistered_ai_benchmark_claim",
    "load_claimed_ai_benchmark_output",
    "requires_preregistered_ai_benchmark",
    "validate_preregistered_ai_runtime_evidence",
    "write_preregistered_ai_benchmark_output",
]
