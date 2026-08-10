from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import duckdb
import pytest

from simple_ai_trading import polymarket_round25_joint_store as joint_store
from simple_ai_trading.polymarket_round25_active_campaign import (
    POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256,
    POLYMARKET_ROUND25_ACTIVE_SOURCE_QUALIFICATION_SHA256,
    POLYMARKET_ROUND25_RESOLUTION_SOURCE,
)
from simple_ai_trading.polymarket_recorder import StreamGap
from simple_ai_trading.polymarket_round25_dataset import (
    POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    POLYMARKET_ROUND25_CAPTURE_PLAN_SHA256,
    POLYMARKET_ROUND25_SELECTION_END_MS,
    POLYMARKET_ROUND25_TRAIN_END_MS,
)
from simple_ai_trading.polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
    Round25JointFeatureSnapshot,
)
from simple_ai_trading.polymarket_round25_joint_materialization import (
    POLYMARKET_ROUND25_EXPECTED_DECISIONS,
    Round25ConditionFeatureMaterialization,
    Round25JointReceiptCondition,
    reject_round25_joint_condition,
    round25_joint_snapshot_sha256,
)
from simple_ai_trading.polymarket_round25_joint_store import (
    POLYMARKET_ROUND25_JOINT_CHUNK_CODEC,
    Round25JointStoreWriter,
    audit_round25_joint_store,
    load_round25_joint_condition_batch,
    load_round25_joint_condition,
    load_round25_joint_endpoint_inputs,
    load_round25_joint_store_manifest,
)
from simple_ai_trading.polymarket_round25_terminal import (
    POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
    POLYMARKET_ROUND25_TERMINAL_RECEIPT_AUDIT_SCHEMA_VERSION,
    POLYMARKET_ROUND25_TERMINAL_TRANSPORT_SCHEMA_VERSION,
)
from simple_ai_trading.polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


RUN_ID = "3" * 32
CONDITION_ID = "0x" + "1" * 64
EVENT_START_MS = POLYMARKET_ROUND25_CAMPAIGN_START_MS + 300_000
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _transport() -> dict[str, object]:
    duration_ms = (
        POLYMARKET_ROUND25_SELECTION_END_MS
        - POLYMARKET_ROUND25_CAMPAIGN_START_MS
    )
    segment = {
        "condition_count": 2,
        "duration_seconds": duration_ms / 1_000.0,
        "eligible_for_condition_rebuild": True,
        "ended_at_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
        "errors": [],
        "exclusion_reasons": [],
        "integrity_errors": [],
        "observed_at_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
        "raw_message_count": 2,
        "report_sha256": "9" * 64,
        "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        "run_id": RUN_ID,
        "segment_index": 0,
        "source_manifest_sha256": "8" * 64,
        "source_result_sha256": "7" * 64,
        "started_at_ms": POLYMARKET_ROUND25_CAMPAIGN_START_MS,
        "status": "complete",
        "stream_counts": {"clob_market": 1, "polymarket_rtds": 1},
        "stream_gap_count": 0,
    }
    interval = {
        "duration_ms": duration_ms,
        "end_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
        "segment_index": 0,
        "start_ms": POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    }
    body: dict[str, object] = {
        "all_scheduled_transport_interval_covered": True,
        "campaign_end_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
        "campaign_start_ms": POLYMARKET_ROUND25_CAMPAIGN_START_MS,
        "campaign_state_artifact_sha256": "6" * 64,
        "campaign_status": "campaign_window_ended",
        "condition_admission_pending": True,
        "created_at_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
        "eligible_run_ids": [RUN_ID],
        "known_ineligible_or_unobserved_intervals": [],
        "live_trading_authority": False,
        "model_data_eligible": False,
        "model_scores_consulted": False,
        "outcomes_consulted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "provisional_eligible_transport_intervals": [interval],
        "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        "schema_version": POLYMARKET_ROUND25_TERMINAL_TRANSPORT_SCHEMA_VERSION,
        "segments": [segment],
        "source_capture_design_sha256": POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256,
        "source_plan_sha256": POLYMARKET_ROUND25_CAPTURE_PLAN_SHA256,
        "source_qualification_sha256": (
            POLYMARKET_ROUND25_ACTIVE_SOURCE_QUALIFICATION_SHA256
        ),
        "terminal_design_sha256": POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


def _receipt_audit(transport: dict[str, object]) -> dict[str, object]:
    eligible_run = {
        "first_gap_opened_at_ms": None,
        "first_receipt_wall_ms": POLYMARKET_ROUND25_CAMPAIGN_START_MS,
        "gap_chain_sha256": EMPTY_SHA256,
        "gap_count": 0,
        "last_gap_opened_at_ms": None,
        "last_receipt_wall_ms": POLYMARKET_ROUND25_CAMPAIGN_START_MS + 1,
        "preregistration_manifest_sha256": "8" * 64,
        "receipt_chain_sha256": "a" * 64,
        "receipt_count": 2,
        "report_sha256": "9" * 64,
        "run_id": RUN_ID,
        "segment_index": 0,
        "status": "complete",
        "stream_counts": {"clob_market": 1, "polymarket_rtds": 1},
    }
    body: dict[str, object] = {
        "condition_admission_pending": True,
        "created_at_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
        "database_run_count": 1,
        "eligible_runs": [eligible_run],
        "ineligible_runs": [],
        "live_trading_authority": False,
        "model_data_eligible": False,
        "model_scores_consulted": False,
        "outcomes_consulted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "receipt_replay_complete": True,
        "schema_version": POLYMARKET_ROUND25_TERMINAL_RECEIPT_AUDIT_SCHEMA_VERSION,
        "terminal_design_sha256": POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
        "terminal_transport_manifest_sha256": transport["manifest_sha256"],
    }
    return {**body, "audit_sha256": _canonical_sha256(body)}


def _snapshot(decision_time_ms: int) -> Round25JointFeatureSnapshot:
    twap = hashlib.sha256(f"twap:{decision_time_ms}".encode("ascii")).hexdigest()
    clob = hashlib.sha256(f"clob:{decision_time_ms}".encode("ascii")).hexdigest()
    source = _canonical_sha256(
        {
            "clob_source_chain_sha256": clob,
            "condition_id": CONDITION_ID,
            "decision_time_ms": decision_time_ms,
            "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
            "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
            "twap_source_chain_sha256": twap,
        }
    )
    return Round25JointFeatureSnapshot(
        condition_id=CONDITION_ID,
        event_start_ms=EVENT_START_MS,
        decision_time_ms=decision_time_ms,
        available=True,
        reasons=(),
        market_prior_probability=0.5,
        values=(0.0,) * len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES),
        source_chain_sha256=source,
        twap_source_chain_sha256=twap,
        clob_source_chain_sha256=clob,
        maximum_receipt_ms=decision_time_ms,
    )


def _admitted_materialization() -> Round25ConditionFeatureMaterialization:
    rows = tuple(
        _snapshot(EVENT_START_MS + phase * 75_000 + offset * 250)
        for phase in range(4)
        for offset in range(4)
    )
    decisions = tuple(row.decision_time_ms for row in rows)
    provisional = Round25ConditionFeatureMaterialization(
        run_id=RUN_ID,
        segment_index=0,
        source_snapshot_sha256="4" * 64,
        source_snapshot_observed_wall_ms=EVENT_START_MS - 10_000,
        market_id="12345",
        condition_id=CONDITION_ID,
        slug=f"btc-updown-5m-{EVENT_START_MS // 1_000}",
        event_start_ms=EVENT_START_MS,
        event_end_ms=EVENT_START_MS + 300_000,
        up_token_id="1" * 40,
        down_token_id="2" * 40,
        resolution_source=POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        role="train",
        source_record_count=42,
        decision_count=POLYMARKET_ROUND25_EXPECTED_DECISIONS,
        available_decision_count=len(rows),
        admitted=True,
        rejection_reasons=(),
        selected_endpoint_decision_time_ms=decisions,
        persisted_snapshots=rows,
        persisted_snapshot_sha256=tuple(
            round25_joint_snapshot_sha256(row) for row in rows
        ),
        unavailable_reason_counts=(),
        materialization_sha256="0" * 64,
    )
    return replace(
        provisional,
        materialization_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _rejected_materialization() -> Round25ConditionFeatureMaterialization:
    event_start = POLYMARKET_ROUND25_TRAIN_END_MS + 300_000
    return reject_round25_joint_condition(
        condition=Round25JointReceiptCondition(
            run_id=RUN_ID,
            segment_index=0,
            snapshot_sha256="4" * 64,
            snapshot_observed_wall_ms=event_start - 10_000,
            market_id="12346",
            condition_id="0x" + "2" * 64,
            slug=f"btc-updown-5m-{event_start // 1_000}",
            event_start_ms=event_start,
            event_end_ms=event_start + 300_000,
            up_token_id="1" * 40,
            down_token_id="2" * 40,
            resolution_source=POLYMARKET_ROUND25_RESOLUTION_SOURCE,
            role="calibration",
        ),
        source_record_count=0,
        rejection_reasons=("stream_gap:clob_market",),
    )


def _source_counts() -> dict[str, int]:
    return {
        "admitted_condition_count": 2,
        "calibration_condition_count": 1,
        "purged_condition_count": 0,
        "selection_condition_count": 0,
        "source_snapshot_count": 2,
        "train_condition_count": 1,
    }


def test_joint_store_atomically_round_trips_admitted_and_rejected_rows(
    tmp_path: Path,
) -> None:
    transport = _transport()
    destination = tmp_path / "round25-joint.duckdb"
    writer = Round25JointStoreWriter(
        destination,
        terminal_transport_manifest=transport,
    )
    admitted = _admitted_materialization()
    rejected = _rejected_materialization()
    writer.add(admitted)
    writer.add(rejected)
    manifest = writer.finalize(
        terminal_receipt_audit=_receipt_audit(transport),
        source_counts=_source_counts(),
    )

    assert destination.is_file()
    assert not (tmp_path / ".round25-joint.duckdb.partial").exists()
    assert not Path(f"{destination}.wal").exists()
    assert manifest["chunk_codec"] == POLYMARKET_ROUND25_JOINT_CHUNK_CODEC
    assert manifest["condition_count"] == 2
    assert manifest["admitted_condition_count"] == 1
    assert manifest["rejected_condition_count"] == 1
    assert manifest["feature_row_count"] == 16
    assert load_round25_joint_store_manifest(destination) == manifest
    assert audit_round25_joint_store(destination) == manifest
    assert load_round25_joint_condition(destination, admitted.condition_id) == admitted
    assert load_round25_joint_condition(destination, rejected.condition_id) == rejected
    endpoint_manifest, endpoints = load_round25_joint_endpoint_inputs(destination)
    assert endpoint_manifest == manifest
    assert len(endpoints["train"]) == 16
    assert endpoints["calibration"] == ()
    assert endpoints["selection"] == ()
    assert load_round25_joint_condition_batch(
        destination,
        (admitted.condition_id,),
        expected_manifest_sha256=manifest["manifest_sha256"],
    ) == (admitted,)
    with pytest.raises(ValueError, match="batch manifest differs"):
        load_round25_joint_condition_batch(
            destination,
            (admitted.condition_id,),
            expected_manifest_sha256="0" * 64,
        )


def test_joint_store_rejects_compressed_blob_tampering(tmp_path: Path) -> None:
    transport = _transport()
    destination = tmp_path / "round25-joint.duckdb"
    writer = Round25JointStoreWriter(
        destination,
        terminal_transport_manifest=transport,
    )
    admitted = _admitted_materialization()
    writer.add(admitted)
    writer.add(_rejected_materialization())
    writer.finalize(
        terminal_receipt_audit=_receipt_audit(transport),
        source_counts=_source_counts(),
    )
    with duckdb.connect(str(destination)) as connection:
        connection.execute(
            """
            UPDATE round25_joint_feature_chunk SET compressed_payload = ?
            WHERE condition_id = ?
            """,
            [b"tampered", admitted.condition_id],
        )
        connection.execute("CHECKPOINT")

    with pytest.raises(ValueError, match="chunk envelope differs"):
        audit_round25_joint_store(destination)


def test_joint_store_abort_removes_only_its_partial_file(tmp_path: Path) -> None:
    transport = _transport()
    destination = tmp_path / "round25-joint.duckdb"
    writer = Round25JointStoreWriter(
        destination,
        terminal_transport_manifest=transport,
    )
    writer.add(_admitted_materialization())
    writer.abort()

    assert not destination.exists()
    assert not (tmp_path / ".round25-joint.duckdb.partial").exists()
    assert not (tmp_path / ".round25-joint.duckdb.partial.wal").exists()


def test_joint_store_rejects_source_count_drift_without_publication(
    tmp_path: Path,
) -> None:
    transport = _transport()
    destination = tmp_path / "round25-joint.duckdb"
    writer = Round25JointStoreWriter(
        destination,
        terminal_transport_manifest=transport,
    )
    writer.add(_admitted_materialization())
    writer.add(_rejected_materialization())
    counts = _source_counts()
    counts["source_snapshot_count"] += 1

    with pytest.raises(ValueError, match="final accounting differs"):
        writer.finalize(
            terminal_receipt_audit=_receipt_audit(transport),
            source_counts=counts,
        )
    writer.abort()
    assert not destination.exists()


def test_joint_store_orchestrator_uses_one_terminal_audit_and_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "terminal-source.duckdb"
    source.write_bytes(b"terminal-source-fixture")
    destination = tmp_path / "round25-joint.duckdb"
    transport = _transport()
    event_start = EVENT_START_MS
    condition = Round25JointReceiptCondition(
        run_id=RUN_ID,
        segment_index=0,
        snapshot_sha256="4" * 64,
        snapshot_observed_wall_ms=event_start - 10_000,
        market_id="12345",
        condition_id=CONDITION_ID,
        slug=f"btc-updown-5m-{event_start // 1_000}",
        event_start_ms=event_start,
        event_end_ms=event_start + 300_000,
        up_token_id="1" * 40,
        down_token_id="2" * 40,
        resolution_source=POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        role="train",
    ).validated()
    calls = []

    def fake_load(**_kwargs: object) -> tuple[tuple[object, ...], dict[str, int]]:
        return (condition,), {
            "admitted_condition_count": 1,
            "calibration_condition_count": 0,
            "purged_condition_count": 0,
            "selection_condition_count": 0,
            "source_snapshot_count": 1,
            "train_condition_count": 1,
        }

    def fake_audit(**kwargs: object) -> dict[str, object]:
        calls.append("terminal-audit")
        observer = kwargs["observer"]
        segment = {
            "ended_at_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
            "run_id": RUN_ID,
            "started_at_ms": POLYMARKET_ROUND25_CAMPAIGN_START_MS,
        }
        observer.start_run(
            segment,
            (
                StreamGap(
                    stream="clob_market",
                    connection_id="clob:" + "a" * 32,
                    opened_at_ms=event_start + 1_000,
                    reason="fixture_gap",
                    last_sequence_number=1,
                ),
            ),
        )
        observer.finish_run(segment)
        return _receipt_audit(transport)

    monkeypatch.setattr(joint_store, "load_round25_joint_receipt_conditions", fake_load)
    monkeypatch.setattr(joint_store, "audit_round25_terminal_receipts", fake_audit)

    manifest, receipt = joint_store.materialize_round25_joint_feature_store(
        source_database=source,
        destination_database=destination,
        terminal_transport_manifest=transport,
    )

    assert calls == ["terminal-audit"]
    assert receipt == _receipt_audit(transport)
    assert manifest["condition_count"] == 1
    assert manifest["admitted_condition_count"] == 0
    assert manifest["receipt_scan_count"] == 1
    assert audit_round25_joint_store(destination) == manifest
