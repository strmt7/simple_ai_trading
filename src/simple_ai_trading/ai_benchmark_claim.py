"""Durable one-shot authorization for preregistered local-AI benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Mapping

from .ai_model_benchmark import (
    AI_MODEL_BENCHMARK_CONTRACT,
    default_finance_ai_test_cases,
    rescore_finance_ai_benchmark_payload,
)
from .polymarket_recorder import PolymarketEvidenceStore


AI_BENCHMARK_CLAIM_SCHEMA_VERSION = "preregistered-ai-benchmark-claim-v1"
AI_BENCHMARK_PREREGISTRATION_SCHEMA_VERSION = (
    "finance-risk-review-candidate-preregistration-v2"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_JSON_BYTES = 32 * 1024 * 1024
_APPROVED_PREREGISTRATION_SHA256 = {
    "qwen3:14b": "7f872babbe9588c8bfe45a65e146ecdb5e0f0a8e78977500ca5afff92aa87e75",
}


@dataclass(frozen=True)
class PreregisteredAIBenchmarkClaim:
    claim_sha256: str
    status: str
    model: str
    confirmation_run_id: str
    confirmation_report_sha256: str
    preregistration_sha256: str
    output_path: str
    report_file_sha256: str = ""
    benchmark_passed: bool | None = None


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
        or frozen.get("required_recorder_status") != "complete"
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
        SELECT status, error, report_sha256
        FROM polymarket_recorder_run WHERE run_id = ?
        """,
            [selected_run],
        )
        .fetchone()
    )
    if (
        run is None
        or str(run[0]) != "complete"
        or str(run[1] or "").strip()
        or _SHA256.fullmatch(str(run[2] or "")) is None
    ):
        raise ValueError("AI benchmark confirmation recorder is not complete")
    output = str(output_path.resolve())
    identity = {
        "schema_version": AI_BENCHMARK_CLAIM_SCHEMA_VERSION,
        "benchmark_contract": AI_MODEL_BENCHMARK_CONTRACT,
        "preregistration_sha256": preregistration_sha256,
        "benchmark_source_sha256": preregistration["benchmark_source_sha256"],
        "test_suite_sha256": preregistration["test_suite_sha256"],
        "confirmation_run_id": selected_run,
        "confirmation_report_sha256": str(run[2]),
        "model": selected_model,
        "timeout_seconds": float(timeout_seconds),
        "minimum_score": float(minimum_score),
        "output_path": output,
    }
    identity_json = _canonical_json(identity)
    claim_sha256 = hashlib.sha256(identity_json.encode("ascii")).hexdigest()
    _ensure_table(store)
    connection = store.connect()
    query = """
        SELECT claim_sha256, identity_json, state, report_file_sha256,
               benchmark_passed
        FROM preregistered_ai_benchmark_claim
        WHERE preregistration_sha256 = ?
    """
    parameters = [preregistration_sha256]

    def existing_claim(row: tuple[object, ...]) -> PreregisteredAIBenchmarkClaim:
        if str(row[0]) != claim_sha256 or str(row[1]) != identity_json:
            raise ValueError("preregistered AI benchmark claim identity differs")
        if str(row[2]) != "completed":
            raise ValueError(
                f"preregistered AI benchmark is already claimed:state={row[2]}"
            )
        claim = _claim_from_row(identity, row, status="existing")
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
    if Path(output).exists():
        raise ValueError("preregistered AI benchmark output already exists")
    _validate_prior_comparison(preregistration_path, preregistration)
    connection.execute("BEGIN TRANSACTION")
    try:
        row = connection.execute(query, parameters).fetchone()
        if row is not None:
            claim = existing_claim(row)
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
                str(run[2]),
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
        confirmation_report_sha256=str(run[2]),
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
    "PreregisteredAIBenchmarkClaim",
    "begin_preregistered_ai_benchmark_claim",
    "complete_preregistered_ai_benchmark_claim",
    "fail_preregistered_ai_benchmark_claim",
    "load_claimed_ai_benchmark_output",
]
