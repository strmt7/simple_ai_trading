"""Target-blind one-pass materialization of qualified Round 25 failure evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
import re
import time

from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_round25_joint_materialization import (
    Round25JointMaterializationObserver,
    Round25JointReceiptCondition,
    round25_joint_condition_from_snapshot,
)
from .polymarket_round25_joint_store import Round25ForensicJointStoreWriter


POLYMARKET_ROUND25_FORENSIC_AUDIT_SHA256 = (
    "8ee546844fada87ab4a542f6620bc5e83654b635b6a72145a338c02431c41276"
)
POLYMARKET_ROUND25_SALVAGE_CONTRACT_SHA256 = (
    "f46c9c629427ab5e2ce5582bdec9be7f6e67bc8f69831fc27ebde1b1f13eafcb"
)
POLYMARKET_ROUND25_FORENSIC_SCAN_SCHEMA_VERSION = (
    "polymarket-round25-forensic-feature-scan-v1"
)
_AUDIT_SCHEMA_VERSION = "polymarket-round25-v2-transport-failure-forensic-audit-v1"
_CONTRACT_SCHEMA_VERSION = "polymarket-round25-v2-condition-salvage-contract-v1"
_MODEL_STREAMS = ("clob_market", "polymarket_rtds")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _validate_self_hashed(
    value: Mapping[str, object],
    *,
    hash_field: str,
    expected_sha256: str,
    schema_version: str,
) -> dict[str, object]:
    selected = dict(value)
    claimed = str(selected.pop(hash_field, ""))
    if (
        _SHA256.fullmatch(expected_sha256) is None
        or claimed != expected_sha256
        or claimed != _canonical_sha256(selected)
        or selected.get("schema_version") != schema_version
    ):
        raise ValueError("Round 25 forensic source artifact differs")
    return {**selected, hash_field: claimed}


def validate_round25_forensic_sources(
    *,
    forensic_audit: Mapping[str, object],
    salvage_contract: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    audit = _validate_self_hashed(
        forensic_audit,
        hash_field="artifact_sha256",
        expected_sha256=POLYMARKET_ROUND25_FORENSIC_AUDIT_SHA256,
        schema_version=_AUDIT_SCHEMA_VERSION,
    )
    contract = _validate_self_hashed(
        salvage_contract,
        hash_field="contract_sha256",
        expected_sha256=POLYMARKET_ROUND25_SALVAGE_CONTRACT_SHA256,
        schema_version=_CONTRACT_SCHEMA_VERSION,
    )
    source = audit.get("source_bindings")
    admissible = contract.get("admissible_source")
    parents = contract.get("parents")
    boundary = contract.get("claims_and_authority")
    truth = contract.get("truth_state")
    if (
        audit.get("status") != "forensic_audit_passed_capture_remains_globally_failed"
        or not isinstance(source, Mapping)
        or not isinstance(admissible, Mapping)
        or not isinstance(parents, Mapping)
        or parents.get("transport_failure_forensic_audit_artifact_sha256")
        != audit["artifact_sha256"]
        or admissible.get("run_id") != source.get("run_id")
        or admissible.get("recorder_report_sha256")
        != source.get("recorder_report_sha256")
        or admissible.get("evidence_manifest_sha256")
        != source.get("evidence_manifest_sha256")
        or admissible.get("recorder_status_must_remain") != "failed"
        or admissible.get("required_streams") != list(_MODEL_STREAMS)
        or admissible.get("source_database_read_only") is not True
        or admissible.get("source_database_wal_allowed") is not False
        or admissible.get("receipt_scan_count") != 1
        or not isinstance(boundary, Mapping)
        or any(
            boundary.get(field) is not False
            for field in (
                "predictive_edge_claim_allowed",
                "profitability_claim_allowed",
                "ai_uplift_claim_allowed",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
        or not isinstance(truth, Mapping)
        or truth.get("targets_or_resolutions_accessed") is not False
    ):
        raise ValueError("Round 25 forensic source boundary differs")
    return audit, contract


def _source_path(database: str | Path) -> Path:
    selected = Path(database)
    if selected.is_symlink():
        raise ValueError("Round 25 forensic source database cannot be a symlink")
    try:
        path = selected.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Round 25 forensic source database is unavailable") from exc
    if not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError("Round 25 forensic source must be inactive and WAL-free")
    return path


def _snapshot_rows(
    store: PolymarketEvidenceStore,
    run_id: str,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        store.connect()
        .execute(
            """
            SELECT snapshot_id, run_id, observed_wall_ms,
                   observed_monotonic_ns, asset, condition_id,
                   event_start_ms, end_ms, up_token_id, down_token_id,
                   resolution_source, gamma_payload_json,
                   gamma_payload_sha256, clob_info_sha256,
                   up_fee_rate_sha256, down_fee_rate_sha256,
                   maker_base_fee, taker_base_fee,
                   taker_order_delay_enabled, minimum_order_age_seconds,
                   snapshot_payload_json, snapshot_sha256
            FROM polymarket_market_snapshot
            WHERE run_id = ?
            ORDER BY event_start_ms, condition_id, observed_wall_ms
            """,
            [run_id],
        )
        .fetchall()
    )


def load_round25_forensic_joint_receipt_conditions(
    *,
    database: str | Path,
    forensic_audit: Mapping[str, object],
    salvage_contract: Mapping[str, object],
) -> tuple[
    tuple[Round25JointReceiptCondition, ...],
    dict[str, int],
    dict[str, object],
    dict[str, object],
]:
    """Load condition identities without reading receipts, targets, or outcomes."""

    audit, contract = validate_round25_forensic_sources(
        forensic_audit=forensic_audit,
        salvage_contract=salvage_contract,
    )
    path = _source_path(database)
    source = audit["source_bindings"]
    run_id = str(source["run_id"])
    with PolymarketEvidenceStore(
        path,
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        report = store.inspect_transport_failed_capture(run_id)
        capture = audit["capture_facts"]
        if (
            report.get("report_sha256") != source["recorder_report_sha256"]
            or report.get("evidence_manifest_sha256")
            != source["evidence_manifest_sha256"]
            or report.get("started_at_ms") != capture["started_at_ms"]
            or report.get("ended_at_ms") != capture["ended_at_ms"]
            or report.get("raw_message_count") != capture["raw_message_count"]
            or report.get("stream_counts") != capture["stream_counts"]
            or report.get("stream_gap_count") != capture["stream_gap_count"]
            or len(report.get("conditions", ())) != capture["condition_snapshot_count"]
        ):
            raise ValueError("Round 25 forensic recorder evidence differs")
        segment: dict[str, object] = {
            "ended_at_ms": int(report["ended_at_ms"]),
            "run_id": run_id,
            "segment_index": int(source["segment_index"]),
            "started_at_ms": int(report["started_at_ms"]),
        }
        conditions: list[Round25JointReceiptCondition] = []
        purged_count = 0
        rows = _snapshot_rows(store, run_id)
        for row in rows:
            condition = round25_joint_condition_from_snapshot(row, segment=segment)
            if condition is None:
                purged_count += 1
            else:
                conditions.append(condition)

    identities = [condition.condition_id for condition in conditions]
    if not conditions or len(set(identities)) != len(identities):
        raise ValueError("Round 25 forensic condition population differs")
    ordered = tuple(
        sorted(
            conditions,
            key=lambda item: (item.event_start_ms, item.condition_id),
        )
    )
    roles = Counter(item.role for item in ordered)
    source_counts = {
        "admitted_condition_count": len(ordered),
        "calibration_condition_count": roles["calibration"],
        "purged_condition_count": purged_count,
        "selection_condition_count": roles["selection"],
        "source_snapshot_count": len(rows),
        "train_condition_count": roles["train"],
    }
    return ordered, source_counts, segment, report


def materialize_round25_forensic_joint_feature_store(
    *,
    source_database: str | Path,
    destination_database: str | Path,
    forensic_audit: Mapping[str, object],
    salvage_contract: Mapping[str, object],
    observed_at_ms: int | None = None,
    progress: Callable[[int, int, float], None] | None = None,
    progress_interval: int = 100_000,
) -> tuple[dict[str, object], dict[str, object]]:
    """Create one compact target-free store from one qualified failed run scan."""

    if progress is not None and not callable(progress):
        raise TypeError("Round 25 forensic progress callback differs")
    if type(progress_interval) is not int or progress_interval <= 0:
        raise ValueError("Round 25 forensic progress interval differs")
    audit, contract = validate_round25_forensic_sources(
        forensic_audit=forensic_audit,
        salvage_contract=salvage_contract,
    )
    source = _source_path(source_database)
    before = source.stat()
    conditions, source_counts, segment, report = (
        load_round25_forensic_joint_receipt_conditions(
            database=source,
            forensic_audit=audit,
            salvage_contract=contract,
        )
    )
    bindings = audit["source_bindings"]
    writer = Round25ForensicJointStoreWriter(
        destination_database,
        forensic_audit_sha256=audit["artifact_sha256"],
        salvage_contract_sha256=contract["contract_sha256"],
        source_report_sha256=str(bindings["recorder_report_sha256"]),
        source_evidence_manifest_sha256=str(bindings["evidence_manifest_sha256"]),
        source_run_id=str(bindings["run_id"]),
    )
    observer = Round25JointMaterializationObserver(conditions, sink=writer.add)
    processed = 0
    started = time.monotonic()
    expected = sum(int(report["stream_counts"].get(name, 0)) for name in _MODEL_STREAMS)
    try:
        with PolymarketEvidenceStore(
            source,
            read_only=True,
            memory_limit="1GB",
            threads=2,
        ) as store:
            gaps = tuple(
                gap
                for gap in store.iter_transport_failed_stream_gaps(
                    str(bindings["run_id"])
                )
                if gap.stream in _MODEL_STREAMS
            )
            observer.start_run(segment, gaps)
            for message in store.iter_transport_failed_capture_messages(
                str(bindings["run_id"]),
                streams=_MODEL_STREAMS,
            ):
                observer.observe_message(segment, message)
                processed += 1
                if progress is not None and processed % progress_interval == 0:
                    progress(processed, expected, time.monotonic() - started)
            observer.finish_run(segment)
        after = source.stat()
        if (
            processed != expected
            or observer.condition_count != len(conditions)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or Path(f"{source}.wal").exists()
        ):
            raise RuntimeError(
                "Round 25 forensic source changed during materialization"
            )
        created_at_ms = (
            int(observed_at_ms)
            if observed_at_ms is not None
            else int(time.time() * 1_000)
        )
        manifest = writer.finalize_forensic(
            created_at_ms=created_at_ms,
            source_counts=source_counts,
        )
    except Exception:
        writer.abort()
        raise

    scan_body: dict[str, object] = {
        "admitted_condition_count": observer.admitted_condition_count,
        "condition_count": observer.condition_count,
        "created_at_ms": created_at_ms,
        "destination_manifest_sha256": manifest["manifest_sha256"],
        "feature_row_count": observer.persisted_snapshot_count,
        "forensic_audit_sha256": audit["artifact_sha256"],
        "live_trading_authority": False,
        "model_scores_consulted": False,
        "outcomes_consulted": False,
        "paper_trading_authority": False,
        "processed_model_receipt_count": processed,
        "profitability_claim": False,
        "rejected_condition_count": (
            observer.condition_count - observer.admitted_condition_count
        ),
        "rejection_reason_counts": dict(sorted(observer.rejection_counts.items())),
        "salvage_contract_sha256": contract["contract_sha256"],
        "schema_version": POLYMARKET_ROUND25_FORENSIC_SCAN_SCHEMA_VERSION,
        "source_database_mutated": False,
        "source_evidence_manifest_sha256": bindings["evidence_manifest_sha256"],
        "source_report_sha256": bindings["recorder_report_sha256"],
        "source_run_id": bindings["run_id"],
        "target_accessed": False,
        "unavailable_reason_counts": dict(
            sorted(observer.unavailable_reason_counts.items())
        ),
    }
    return manifest, {
        **scan_body,
        "scan_sha256": _canonical_sha256(scan_body),
    }


__all__ = [
    "POLYMARKET_ROUND25_FORENSIC_AUDIT_SHA256",
    "POLYMARKET_ROUND25_FORENSIC_SCAN_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_SALVAGE_CONTRACT_SHA256",
    "load_round25_forensic_joint_receipt_conditions",
    "materialize_round25_forensic_joint_feature_store",
    "validate_round25_forensic_sources",
]
