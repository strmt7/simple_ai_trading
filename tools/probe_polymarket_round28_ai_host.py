#!/usr/bin/env python3
"""Verify one preregistered AI artifact and qualify its local GPU runtime."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time

from simple_ai_trading.polymarket_round27_operator import artifact_writer
from simple_ai_trading.polymarket_round28_ai_contract import (
    load_round28_ai_contract,
)
from simple_ai_trading.polymarket_round28_ai_host import (
    build_round28_ai_artifact_verification,
    probe_round28_ai_candidate_host,
    round28_ai_candidate_from_contract,
    validate_round28_ai_host_report,
)
from simple_ai_trading.polymarket_round28_ai_selection import (
    build_round28_ai_host_failure,
    validate_round28_ai_host_failure,
)


_CHUNK_BYTES = 8 * 1024 * 1024
_PROGRESS_BYTES = 512 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--runtime-digest")
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--host-report", type=Path, required=True)
    parser.add_argument("--failure-report", type=Path, required=True)
    return parser


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _progress(phase: str, detail: Mapping[str, object]) -> None:
    print(
        json.dumps(
            {
                "observed_at_ms": time.time_ns() // 1_000_000,
                "phase": phase,
                **detail,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _stream_sha256(path: Path, expected_size: int) -> str:
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise ValueError("AI artifact size differs from preregistration")
    digest = hashlib.sha256()
    read_bytes = 0
    next_progress = _PROGRESS_BYTES
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
            read_bytes += len(chunk)
            if read_bytes >= next_progress:
                _progress(
                    "artifact-hash-progress",
                    {
                        "bytes_read": read_bytes,
                        "total_bytes": expected_size,
                    },
                )
                next_progress += _PROGRESS_BYTES
    if read_bytes != expected_size:
        raise ValueError("AI artifact changed while hashing")
    return digest.hexdigest()


def _resolve(root: Path, path: Path, *, strict: bool) -> Path:
    selected = path if path.is_absolute() else root / path
    return selected.resolve(strict=strict)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository.resolve(strict=True)
    lexical_artifact_path = (
        args.artifact_path
        if args.artifact_path.is_absolute()
        else repository / args.artifact_path
    )
    if lexical_artifact_path.is_symlink():
        raise ValueError("Round 28 AI artifact path must not be a symlink")
    artifact_path = _resolve(repository, args.artifact_path, strict=False)
    host_path = _resolve(repository, args.host_report, strict=False)
    failure_path = _resolve(repository, args.failure_report, strict=False)
    if (
        host_path == failure_path
        or host_path.is_symlink()
        or failure_path.is_symlink()
        or (host_path.exists() and failure_path.exists())
    ):
        raise ValueError("Round 28 AI host-probe paths differ")
    contract = load_round28_ai_contract(repository)
    candidate = round28_ai_candidate_from_contract(
        contract,
        model_id=args.model_id,
        observed_runtime_digest=args.runtime_digest,
    )
    if host_path.exists():
        report = json.loads(host_path.read_text(encoding="ascii"))
        validated, restored = validate_round28_ai_host_report(
            report,
            contract=contract,
        )
        if restored != candidate:
            raise ValueError("Round 28 existing host report candidate differs")
        print(json.dumps(validated, allow_nan=False, indent=2, sort_keys=True))
        return 0
    if failure_path.exists():
        failure = validate_round28_ai_host_failure(
            json.loads(failure_path.read_text(encoding="ascii")),
            contract=contract,
        )
        if failure["model_id"] != candidate.model_id:
            raise ValueError("Round 28 existing host failure candidate differs")
        print(json.dumps(failure, allow_nan=False, indent=2, sort_keys=True))
        return 2
    phase = "artifact_verification"
    try:
        if not artifact_path.is_file():
            phase = "artifact_download"
            raise FileNotFoundError("AI artifact is unavailable")
        _progress(
            "artifact-hash-started",
            {
                "model_id": candidate.model_id,
                "total_bytes": candidate.artifact_size_bytes,
            },
        )
        observed_sha256 = _stream_sha256(
            artifact_path,
            candidate.artifact_size_bytes,
        )
        if observed_sha256 != candidate.artifact_sha256:
            raise ValueError("AI artifact SHA-256 differs from preregistration")
        observed_at_ms = time.time_ns() // 1_000_000
        source_evidence = _canonical_sha256(
            {
                "model_id": candidate.model_id,
                "artifact_sha256": observed_sha256,
                "artifact_size_bytes": candidate.artifact_size_bytes,
                "observed_at_ms": observed_at_ms,
            }
        )
        artifact = build_round28_ai_artifact_verification(
            candidate,
            observed_sha256=observed_sha256,
            observed_size_bytes=candidate.artifact_size_bytes,
            verification_method="file_sha256_stream",
            source_evidence_sha256=source_evidence,
            observed_at_ms=observed_at_ms,
        )
        phase = "cold_conformance"
        report = probe_round28_ai_candidate_host(
            candidate,
            artifact_verification=artifact,
        )
        validate_round28_ai_host_report(report, contract=contract)
        artifact_writer(host_path, "report_sha256")(report)
        _progress(
            "host-qualified",
            {
                "model_id": candidate.model_id,
                "host_report_sha256": report["report_sha256"],
                "passed": True,
            },
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - sanitized terminal evidence
        error_text = f"{type(exc).__name__}:{exc}"
        if phase == "artifact_download":
            error_code = "artifact_unavailable"
        elif "artifact" in error_text.lower():
            error_code = "artifact_hash_or_size_mismatch"
        elif "digest" in error_text.lower():
            error_code = "runtime_digest_mismatch"
        elif "resident" in error_text.lower() or "gpu" in error_text.lower():
            error_code = "gpu_residency_incomplete"
            phase = "gpu_residency"
        elif "unload" in error_text.lower():
            error_code = "unload_not_observed"
            phase = "mandatory_unload"
        elif "conformance" in error_text.lower():
            error_code = "conformance_response_invalid"
        else:
            error_code = "provider_unavailable"
        failure = build_round28_ai_host_failure(
            contract=contract,
            model_id=args.model_id,
            phase=phase,
            error_code=error_code,
            private_detail_sha256=hashlib.sha256(
                error_text.encode("utf-8", errors="replace")
            ).hexdigest(),
            observed_at_ms=time.time_ns() // 1_000_000,
        )
        artifact_writer(failure_path, "report_sha256")(failure)
        _progress(
            "host-rejected",
            {
                "model_id": args.model_id,
                "phase": phase,
                "error_code": error_code,
                "failure_report_sha256": failure["report_sha256"],
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
