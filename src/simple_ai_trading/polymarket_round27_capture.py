"""Prospective Round 27 capture gate for documented public predictor sources."""

from __future__ import annotations

import copy
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import BinaryIO, Callable, Mapping

from .polymarket import (
    POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE,
    PolymarketPublicClient,
)
from .polymarket_recorder import (
    POLYMARKET_RTDS_CHAINLINK_TWAP_60_TOPIC,
    PolymarketEvidenceStore,
    PolymarketPublicRecorder,
    RecorderReport,
)
from .polymarket_source_quality import audit_binance_trade_quality
from .storage import write_bytes_atomic


ROUND27_CAPTURE_CONTRACT_SCHEMA_VERSION = (
    "polymarket-round27-documented-source-smoke-contract-v1"
)
ROUND27_CAPTURE_RESULT_SCHEMA_VERSION = (
    "polymarket-round27-documented-source-smoke-result-v1"
)
ROUND27_CAPTURE_DURATION_SECONDS = 600
ROUND27_DATABASE_CAP_BYTES = 2 * 1024**3
ROUND27_DATABASE_STOP_RESERVE_BYTES = 128 * 1024**2
ROUND27_MINIMUM_FREE_BYTES = 32 * 1024**3
ROUND27_PROGRESS_INTERVAL_SECONDS = 30
ROUND27_QUEUE_CAPACITY = 100_000
ROUND27_SOURCE_QUALIFICATION_RELATIVE = Path(
    "docs/model-research/polymarket/"
    "binance-usdm-aggregate-trade-source-qualification-v1-2026-08-15.json"
)
ROUND27_HYPOTHESIS_PREREGISTRATION_RELATIVE = Path(
    "docs/model-research/polymarket/"
    "round-027-execution-hypothesis-preregistration-v3.json"
)
ROUND27_SOURCE_RELATIVES = (
    Path("src/simple_ai_trading/polymarket.py"),
    Path("src/simple_ai_trading/polymarket_recorder.py"),
    Path("src/simple_ai_trading/polymarket_source_quality.py"),
    Path("src/simple_ai_trading/polymarket_round27_capture.py"),
    Path("tools/run_polymarket_round27_capture.py"),
)
_AUTHORITY = {
    "credentials_used": False,
    "execution_connected": False,
    "orders_submitted": False,
    "model_data_eligible": False,
    "edge_claim": False,
    "profitability_claim": False,
    "paper_trading_authority": False,
    "live_trading_authority": False,
}
_CAPTURE_SCOPE = {
    "asset": "BTC",
    "market": "Polymarket BTC five-minute Up/Down",
    "resolution_source": POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE,
    "binance_futures_profile": "documented_aggregate_trades",
    "binance_futures_event_type": "aggTrade",
    "required_streams": [
        "binance_futures",
        "binance_spot",
        "clob_market",
        "polymarket_rtds",
    ],
    "required_clob_lanes": ["clob"],
    "required_rtds_topics": [
        "crypto_prices",
        POLYMARKET_RTDS_CHAINLINK_TWAP_60_TOPIC,
    ],
}
_RESOURCE_POLICY = {
    "database_cap_bytes": ROUND27_DATABASE_CAP_BYTES,
    "database_stop_reserve_bytes": ROUND27_DATABASE_STOP_RESERVE_BYTES,
    "minimum_free_bytes": ROUND27_MINIMUM_FREE_BYTES,
    "memory_limit": "1GB",
    "database_threads": 2,
    "queue_capacity": ROUND27_QUEUE_CAPACITY,
    "progress_interval_seconds": ROUND27_PROGRESS_INTERVAL_SECONDS,
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


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _verified_artifact_claim(path: Path, *, field: str, label: str) -> str:
    payload = _read_json(path, label=label)
    claim = str(payload.pop(field, "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", claim):
        raise ValueError(f"{label} claim is invalid")
    if not hmac.compare_digest(claim, _canonical_sha256(payload)):
        raise ValueError(f"{label} claim differs")
    return claim


def _repository_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("repository HEAD is invalid")
    return commit


def create_round27_capture_contract(
    repository: str | Path,
    *,
    created_at_ms: int,
) -> dict[str, object]:
    root = Path(repository).resolve()
    created = int(created_at_ms)
    if not root.is_dir() or created <= 0:
        raise ValueError("Round 27 capture contract inputs are invalid")
    source_hashes = {
        relative.as_posix(): _text_sha256(root / relative)
        for relative in ROUND27_SOURCE_RELATIVES
    }
    payload: dict[str, object] = {
        "schema_version": ROUND27_CAPTURE_CONTRACT_SCHEMA_VERSION,
        "created_at_ms": created,
        "phase": "documented_source_smoke",
        "capture_duration_seconds": ROUND27_CAPTURE_DURATION_SECONDS,
        "capture_scope": copy.deepcopy(_CAPTURE_SCOPE),
        "resource_policy": copy.deepcopy(_RESOURCE_POLICY),
        "repository_commit": _repository_head(root),
        "source_text_sha256": source_hashes,
        "source_qualification_sha256": _verified_artifact_claim(
            root / ROUND27_SOURCE_QUALIFICATION_RELATIVE,
            field="qualification_sha256",
            label="Binance USD-M aggregate-trade source qualification",
        ),
        "hypothesis_preregistration_sha256": _verified_artifact_claim(
            root / ROUND27_HYPOTHESIS_PREREGISTRATION_RELATIVE,
            field="preregistration_sha256",
            label="Round 27 execution hypothesis preregistration",
        ),
        "success_gate": {
            "terminal_recorder_status": "complete",
            "required_stream_message_count_positive": True,
            "stream_gap_count": 0,
            "recorder_error_count": 0,
            "integrity_error_count": 0,
            "documented_spot_and_futures_trade_quality_passed": True,
            "database_footprint_at_or_below_cap": True,
        },
        "authority": copy.deepcopy(_AUTHORITY),
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class Round27CaptureContract:
    contract_sha256: str
    created_at_ms: int
    capture_duration_seconds: int
    repository_commit: str
    source_text_sha256: dict[str, str]
    source_qualification_sha256: str
    hypothesis_preregistration_sha256: str


def validate_round27_capture_contract(
    payload: Mapping[str, object],
    *,
    repository: str | Path,
) -> Round27CaptureContract:
    root = Path(repository).resolve()
    raw = dict(payload)
    claim = str(raw.pop("contract_sha256", "")).lower()
    source_hashes = raw.get("source_text_sha256")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", claim)
        or not hmac.compare_digest(claim, _canonical_sha256(raw))
        or raw.get("schema_version") != ROUND27_CAPTURE_CONTRACT_SCHEMA_VERSION
        or raw.get("phase") != "documented_source_smoke"
        or raw.get("capture_duration_seconds") != ROUND27_CAPTURE_DURATION_SECONDS
        or raw.get("capture_scope") != _CAPTURE_SCOPE
        or raw.get("resource_policy") != _RESOURCE_POLICY
        or raw.get("success_gate")
        != {
            "terminal_recorder_status": "complete",
            "required_stream_message_count_positive": True,
            "stream_gap_count": 0,
            "recorder_error_count": 0,
            "integrity_error_count": 0,
            "documented_spot_and_futures_trade_quality_passed": True,
            "database_footprint_at_or_below_cap": True,
        }
        or raw.get("authority") != _AUTHORITY
        or type(raw.get("created_at_ms")) is not int
        or int(raw["created_at_ms"]) <= 0
        or not re.fullmatch(r"[0-9a-f]{40}", str(raw.get("repository_commit", "")))
        or not isinstance(source_hashes, Mapping)
        or set(source_hashes) != {path.as_posix() for path in ROUND27_SOURCE_RELATIVES}
    ):
        raise ValueError("Round 27 capture contract differs")
    normalized_hashes: dict[str, str] = {}
    for relative, expected_value in source_hashes.items():
        expected = str(expected_value).lower()
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected)
            or not hmac.compare_digest(_text_sha256(root / str(relative)), expected)
        ):
            raise ValueError(f"Round 27 capture source differs: {relative}")
        normalized_hashes[str(relative)] = expected
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            str(raw["repository_commit"]),
            "HEAD",
        ],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("Round 27 capture source commit is not an ancestor of HEAD")
    source_claim = _verified_artifact_claim(
        root / ROUND27_SOURCE_QUALIFICATION_RELATIVE,
        field="qualification_sha256",
        label="Binance USD-M aggregate-trade source qualification",
    )
    hypothesis_claim = _verified_artifact_claim(
        root / ROUND27_HYPOTHESIS_PREREGISTRATION_RELATIVE,
        field="preregistration_sha256",
        label="Round 27 execution hypothesis preregistration",
    )
    if (
        raw.get("source_qualification_sha256") != source_claim
        or raw.get("hypothesis_preregistration_sha256") != hypothesis_claim
    ):
        raise ValueError("Round 27 capture evidence binding differs")
    return Round27CaptureContract(
        contract_sha256=claim,
        created_at_ms=int(raw["created_at_ms"]),
        capture_duration_seconds=ROUND27_CAPTURE_DURATION_SECONDS,
        repository_commit=str(raw["repository_commit"]),
        source_text_sha256=normalized_hashes,
        source_qualification_sha256=source_claim,
        hypothesis_preregistration_sha256=hypothesis_claim,
    )


def load_round27_capture_contract(
    path: str | Path,
    *,
    repository: str | Path,
) -> Round27CaptureContract:
    return validate_round27_capture_contract(
        _read_json(Path(path), label="Round 27 capture contract"),
        repository=repository,
    )


def write_round27_capture_contract(path: str | Path, value: Mapping[str, object]) -> None:
    write_bytes_atomic(
        Path(path),
        (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("ascii"),
    )


@dataclass(frozen=True, slots=True)
class Round27CaptureConfig:
    repository: Path
    contract_path: Path
    database_path: Path
    result_path: Path
    lock_path: Path

    def validated(self) -> Round27CaptureConfig:
        root = self.repository.resolve()
        selected = tuple(
            path.resolve()
            for path in (
                self.contract_path,
                self.database_path,
                self.result_path,
                self.lock_path,
            )
        )
        if (
            not root.is_dir()
            or not selected[0].is_file()
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("Round 27 capture configuration differs")
        return Round27CaptureConfig(root, *selected)


class _CaptureFileLock(AbstractContextManager["_CaptureFileLock"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: BinaryIO | None = None

    def __enter__(self) -> _CaptureFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("Round 27 capture is already running") from exc
        self.handle = handle
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self.handle is None:
            return
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def _database_footprint_bytes(database: Path) -> int:
    total = 0
    for candidate in (
        database,
        Path(f"{database}.wal"),
        Path(f"{database}.tmp"),
    ):
        if candidate.is_file():
            total += candidate.stat().st_size
        elif candidate.is_dir():
            total += sum(path.stat().st_size for path in candidate.rglob("*") if path.is_file())
    return total


def _resource_block(database: Path) -> str | None:
    footprint = _database_footprint_bytes(database)
    if footprint >= ROUND27_DATABASE_CAP_BYTES - ROUND27_DATABASE_STOP_RESERVE_BYTES:
        return "database_cap_reserve_reached"
    database.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(database.parent).free < ROUND27_MINIMUM_FREE_BYTES:
        return "minimum_free_space_breached"
    return None


def _require_fresh_capture_paths(database: Path, result: Path) -> None:
    occupied = [
        candidate
        for candidate in (
            database,
            Path(f"{database}.wal"),
            Path(f"{database}.tmp"),
            result,
        )
        if candidate.exists()
    ]
    if occupied:
        raise RuntimeError(
            "Round 27 capture requires fresh database and result paths: "
            + ", ".join(str(path) for path in occupied)
        )


def _create_recorder(database: Path) -> PolymarketPublicRecorder:
    return PolymarketPublicRecorder(
        database,
        client=PolymarketPublicClient(
            required_five_minute_resolution_sources={
                "BTC": POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE
            }
        ),
        queue_capacity=ROUND27_QUEUE_CAPACITY,
        discovery_interval_seconds=30,
        memory_limit="1GB",
        database_threads=2,
        assets=("BTC",),
        include_binance_futures=True,
        include_binance_spot=True,
        include_rtds_binance=True,
        chainlink_price_mode="twap_60s",
        clob_lane_ids=("clob",),
        binance_futures_aggregate_trades=True,
    )


def _manifest(
    contract: Round27CaptureContract,
    *,
    run_id: str,
    started_at_ms: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": ROUND27_CAPTURE_CONTRACT_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "run_id": run_id,
        "created_at_ms": started_at_ms,
        "capture_duration_seconds": contract.capture_duration_seconds,
        "phase": "documented_source_smoke",
        "required_assets": ["BTC"],
        "required_streams": list(_CAPTURE_SCOPE["required_streams"]),
        "required_clob_lanes": list(_CAPTURE_SCOPE["required_clob_lanes"]),
        "required_rtds_topics": list(_CAPTURE_SCOPE["required_rtds_topics"]),
        "binance_futures_profile": "documented_aggregate_trades",
        "source_qualification_sha256": contract.source_qualification_sha256,
        "hypothesis_preregistration_sha256": (
            contract.hypothesis_preregistration_sha256
        ),
        **_AUTHORITY,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def _audit_sources(database: Path, run_id: str) -> Mapping[str, object]:
    with PolymarketEvidenceStore(
        database,
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        return audit_binance_trade_quality(store, run_id=run_id)


def _result_payload(
    contract: Round27CaptureContract,
    report: RecorderReport,
    source_quality: Mapping[str, object],
    *,
    database_footprint_bytes: int,
    resource_stop_reason: str,
) -> dict[str, object]:
    required_counts_positive = all(
        int(report.stream_counts.get(stream, 0)) > 0
        for stream in _CAPTURE_SCOPE["required_streams"]
    )
    checks = {
        "terminal_recorder_status_complete": report.status == "complete",
        "required_stream_message_count_positive": required_counts_positive,
        "stream_gap_count_zero": report.stream_gap_count == 0,
        "recorder_error_count_zero": not report.errors,
        "integrity_error_count_zero": not report.integrity_errors,
        "documented_spot_and_futures_trade_quality_passed": (
            source_quality.get("passed") is True
        ),
        "database_footprint_at_or_below_cap": (
            database_footprint_bytes <= ROUND27_DATABASE_CAP_BYTES
        ),
        "resource_stop_not_triggered": not resource_stop_reason,
    }
    failure_reasons = [name for name, passed in checks.items() if not passed]
    payload: dict[str, object] = {
        "schema_version": ROUND27_CAPTURE_RESULT_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "status": "passed" if not failure_reasons else "failed",
        "run_id": report.run_id,
        "capture_report": asdict(report),
        "source_quality": dict(source_quality),
        "gate_checks": checks,
        "failure_reasons": failure_reasons,
        "database_footprint_bytes": database_footprint_bytes,
        "database_cap_bytes": ROUND27_DATABASE_CAP_BYTES,
        "resource_stop_reason": resource_stop_reason,
        "conclusion": (
            "documented public source smoke passed; no model or economic claim"
            if not failure_reasons
            else "documented public source smoke failed; larger capture remains blocked"
        ),
        "authority": _AUTHORITY,
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


async def run_round27_capture(
    config: Round27CaptureConfig,
    *,
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
    recorder_factory: Callable[[Path], PolymarketPublicRecorder] = _create_recorder,
    source_audit: Callable[[Path, str], Mapping[str, object]] = _audit_sources,
) -> dict[str, object]:
    selected = config.validated()
    contract = load_round27_capture_contract(
        selected.contract_path,
        repository=selected.repository,
    )
    selected.database_path.parent.mkdir(parents=True, exist_ok=True)
    selected.result_path.parent.mkdir(parents=True, exist_ok=True)
    resource_stop_reason = ""
    with _CaptureFileLock(selected.lock_path):
        if time.time_ns() // 1_000_000 < contract.created_at_ms:
            raise RuntimeError("Round 27 capture contract is dated in the future")
        _require_fresh_capture_paths(selected.database_path, selected.result_path)
        initial_block = _resource_block(selected.database_path)
        if initial_block:
            raise RuntimeError(f"Round 27 capture resource block: {initial_block}")

        def stop_requested() -> str | None:
            nonlocal resource_stop_reason
            reason = _resource_block(selected.database_path)
            if reason:
                resource_stop_reason = reason
            return reason

        def manifest_factory(run_id: str, started_at_ms: int) -> Mapping[str, object]:
            return _manifest(contract, run_id=run_id, started_at_ms=started_at_ms)

        report = await recorder_factory(selected.database_path).run(
            duration_seconds=contract.capture_duration_seconds,
            progress=progress,
            progress_interval_seconds=ROUND27_PROGRESS_INTERVAL_SECONDS,
            stop_requested=stop_requested,
            preregistration_manifest_factory=manifest_factory,
        )
        if progress is not None:
            progress(
                "source-audit-started",
                {"run_id": report.run_id, "observed_at_ms": time.time_ns() // 1_000_000},
            )
        try:
            quality = dict(source_audit(selected.database_path, report.run_id))
        except Exception as exc:
            quality = {
                "passed": False,
                "error": f"{exc.__class__.__name__}:{exc}",
            }
        footprint = _database_footprint_bytes(selected.database_path)
        result = _result_payload(
            contract,
            report,
            quality,
            database_footprint_bytes=footprint,
            resource_stop_reason=resource_stop_reason,
        )
        write_bytes_atomic(
            selected.result_path,
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("ascii"),
        )
        if progress is not None:
            progress(
                "source-audit-finalized",
                {
                    "run_id": report.run_id,
                    "observed_at_ms": time.time_ns() // 1_000_000,
                    "status": result["status"],
                    "result_sha256": result["result_sha256"],
                },
            )
        return result


__all__ = [
    "ROUND27_CAPTURE_DURATION_SECONDS",
    "ROUND27_CAPTURE_RESULT_SCHEMA_VERSION",
    "Round27CaptureConfig",
    "Round27CaptureContract",
    "create_round27_capture_contract",
    "load_round27_capture_contract",
    "run_round27_capture",
    "validate_round27_capture_contract",
    "write_round27_capture_contract",
]
