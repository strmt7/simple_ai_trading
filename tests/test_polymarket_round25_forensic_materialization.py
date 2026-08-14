from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_round25_forensic_materialization as forensic
from simple_ai_trading.polymarket_recorder import RawStreamMessage, StreamGap
from simple_ai_trading.polymarket_round21_core_features import (
    build_round21_execution_books,
)
from simple_ai_trading.polymarket_round25_clob_features import (
    Round25ClobFeatureEngine,
)
from simple_ai_trading.polymarket_round25_campaign import (
    POLYMARKET_ROUND25_RESOLUTION_SOURCE,
)
from simple_ai_trading.polymarket_round25_forensic_materialization import (
    POLYMARKET_ROUND25_FORENSIC_AUDIT_SHA256,
    POLYMARKET_ROUND25_SALVAGE_CONTRACT_SHA256,
    materialize_round25_forensic_joint_feature_store,
    validate_round25_forensic_sources,
)
from simple_ai_trading.polymarket_round25_forensic_partition import (
    POLYMARKET_ROUND25_FORENSIC_PARTITION_SCHEMA_VERSION,
    partition_round25_forensic_conditions,
    validate_round25_forensic_partition_manifest,
)
from simple_ai_trading.polymarket_round25_forensic_venue import (
    POLYMARKET_ROUND25_FORENSIC_VENUE_PARAMETER_AUDIT_SHA256,
)
from simple_ai_trading.polymarket_round25_joint_materialization import (
    Round25JointReceiptCondition,
    Round25SingleLaneClobDecoder,
)
from simple_ai_trading.polymarket_round25_joint_store import (
    POLYMARKET_ROUND25_FORENSIC_JOINT_STORE_MANIFEST_SCHEMA_VERSION,
    audit_round25_joint_store,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "polymarket"
AUDIT = RESEARCH / "round-025-v2-transport-failure-forensic-audit-2026-08-14.json"
CONTRACT = RESEARCH / "round-025-v2-condition-salvage-contract-v1.json"
RUN_ID = "f96a24bdaa2d4f5f8cdad3f06193a0ce"
EVENT_START_MS = 1_786_515_300_000
CONDITION_ID = "0x" + "1" * 64


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sources() -> tuple[dict[str, object], dict[str, object]]:
    return (
        json.loads(AUDIT.read_text(encoding="ascii")),
        json.loads(CONTRACT.read_text(encoding="ascii")),
    )


def _condition() -> Round25JointReceiptCondition:
    return Round25JointReceiptCondition(
        run_id=RUN_ID,
        segment_index=2,
        snapshot_sha256="4" * 64,
        snapshot_observed_wall_ms=EVENT_START_MS - 10_000,
        market_id="12345",
        condition_id=CONDITION_ID,
        slug=f"btc-updown-5m-{EVENT_START_MS // 1_000}",
        event_start_ms=EVENT_START_MS,
        event_end_ms=EVENT_START_MS + 300_000,
        up_token_id="1" * 40,
        down_token_id="2" * 40,
        resolution_source=POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        role="train",
    ).validated()


def test_forensic_source_contracts_are_self_hashed_and_fail_closed() -> None:
    audit, contract = _sources()
    selected_audit, selected_contract = validate_round25_forensic_sources(
        forensic_audit=audit,
        salvage_contract=contract,
    )

    assert selected_audit["artifact_sha256"] == (
        POLYMARKET_ROUND25_FORENSIC_AUDIT_SHA256
    )
    assert selected_contract["contract_sha256"] == (
        POLYMARKET_ROUND25_SALVAGE_CONTRACT_SHA256
    )
    tampered = dict(contract)
    tampered["status"] = "tampered"
    with pytest.raises(ValueError, match="source artifact differs"):
        validate_round25_forensic_sources(
            forensic_audit=audit,
            salvage_contract=tampered,
        )


def test_forensic_partition_is_chronological_purged_and_target_blind() -> None:
    population = tuple(
        (f"0x{index + 1:064x}", EVENT_START_MS + index * 300_000)
        for index in range(50)
    )

    partition = partition_round25_forensic_conditions(population)
    roles = [role for _, _, role in partition]

    assert roles.count("train") == 29
    assert roles.count("calibration") == 8
    assert roles.count("selection") == 9
    assert roles.count("purged") == 4
    assert [index for index, role in enumerate(roles) if role == "purged"] == [
        29,
        30,
        39,
        40,
    ]
    with pytest.raises(ValueError, match="population differs"):
        partition_round25_forensic_conditions(tuple(reversed(population)))


def test_forensic_partition_manifest_recomputes_roles_and_hash() -> None:
    population = tuple(
        (f"0x{index + 1:064x}", EVENT_START_MS + index * 300_000)
        for index in range(50)
    )
    rows = [
        {"condition_id": condition_id, "event_start_ms": event_start_ms, "role": role}
        for condition_id, event_start_ms, role in partition_round25_forensic_conditions(
            population
        )
    ]
    body = {
        "condition_count": 50,
        "conditions": rows,
        "created_at_ms": population[-1][1] + 300_000,
        "feature_store_manifest_sha256": "a" * 64,
        "live_trading_authority": False,
        "model_scores_consulted": False,
        "outcomes_consulted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "role_counts": {
            "train": 29,
            "calibration": 8,
            "selection": 9,
            "purged": 4,
        },
        "salvage_contract_sha256": POLYMARKET_ROUND25_SALVAGE_CONTRACT_SHA256,
        "schema_version": POLYMARKET_ROUND25_FORENSIC_PARTITION_SCHEMA_VERSION,
        "selection_accessed": False,
        "target_accessed": False,
        "venue_parameter_audit_sha256": (
            POLYMARKET_ROUND25_FORENSIC_VENUE_PARAMETER_AUDIT_SHA256
        ),
    }
    manifest = {**body, "partition_sha256": _canonical_sha256(body)}

    assert validate_round25_forensic_partition_manifest(
        manifest,
        expected_feature_store_manifest_sha256="a" * 64,
    ) == manifest

    tampered = json.loads(json.dumps(manifest))
    tampered["conditions"][0]["role"] = "selection"
    tampered_body = dict(tampered)
    tampered_body.pop("partition_sha256")
    tampered["partition_sha256"] = _canonical_sha256(tampered_body)
    with pytest.raises(ValueError, match="binding differs"):
        validate_round25_forensic_partition_manifest(tampered)


def test_single_clob_frame_can_carry_multiple_book_events_for_one_token() -> None:
    token = "1" * 40
    other = "2" * 40

    def book(timestamp: int, bid: str, ask: str) -> dict[str, object]:
        return {
            "asks": [{"price": ask, "size": "9"}],
            "asset_id": token,
            "bids": [{"price": bid, "size": "8"}],
            "event_type": "book",
            "hash": str(timestamp),
            "market": CONDITION_ID,
            "timestamp": str(timestamp),
        }

    message = RawStreamMessage(
        stream="clob_market",
        connection_id="clob:" + "b" * 32,
        sequence_number=1,
        received_wall_ms=EVENT_START_MS + 1_000,
        received_monotonic_ns=(EVENT_START_MS + 1_000) * 1_000_000,
        raw_text=json.dumps(
            [
                book(EVENT_START_MS + 900, "0.49", "0.51"),
                book(EVENT_START_MS + 950, "0.50", "0.52"),
            ],
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
    decoder = Round25SingleLaneClobDecoder()
    events = tuple(event for event, _ in decoder.add(message))

    books = build_round21_execution_books(
        condition_id=CONDITION_ID,
        up_token_id=token,
        down_token_id=other,
        union_events=events,
        admitted_gap_free=True,
    )

    assert len(books) == 2
    assert books[0].received_wall_ms == books[1].received_wall_ms
    assert books[0].source_payload_sha256 != books[1].source_payload_sha256
    assert [book.source_time_ms for book in books] == [
        EVENT_START_MS + 900,
        EVENT_START_MS + 950,
    ]
    engine = Round25ClobFeatureEngine(
        condition_id=CONDITION_ID,
        up_token_id=token,
        down_token_id=other,
        event_start_ms=EVENT_START_MS,
    )
    engine.ingest(books[0])
    engine.ingest(books[1])
    with pytest.raises(ValueError, match="chronology"):
        engine.ingest(books[1])


def test_forensic_materializer_publishes_atomic_non_authorizing_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, contract = _sources()
    source = tmp_path / "inactive-source.duckdb"
    source.write_bytes(b"immutable-source-fixture")
    destination = tmp_path / "forensic-joint.duckdb"
    segment = {
        "ended_at_ms": EVENT_START_MS + 600_000,
        "run_id": RUN_ID,
        "segment_index": 2,
        "started_at_ms": EVENT_START_MS - 130_000,
    }
    report = {"stream_counts": {"clob_market": 0, "polymarket_rtds": 0}}
    source_counts = {
        "admitted_condition_count": 1,
        "calibration_condition_count": 0,
        "purged_condition_count": 0,
        "selection_condition_count": 0,
        "source_snapshot_count": 1,
        "train_condition_count": 1,
    }

    monkeypatch.setattr(
        forensic,
        "load_round25_forensic_joint_receipt_conditions",
        lambda **_kwargs: ((_condition(),), source_counts, segment, report),
    )

    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeStore:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def iter_transport_failed_stream_gaps(
            self, _run_id: str
        ) -> tuple[StreamGap, ...]:
            return (
                StreamGap(
                    stream="clob_market",
                    connection_id="clob:" + "a" * 32,
                    opened_at_ms=EVENT_START_MS + 1_000,
                    reason="fixture_gap",
                    last_sequence_number=1,
                ),
            )

        def iter_transport_failed_capture_messages(
            self,
            _run_id: str,
            *,
            streams: tuple[str, ...],
        ) -> tuple[object, ...]:
            assert streams == ("clob_market", "polymarket_rtds")
            return ()

    monkeypatch.setattr(forensic, "PolymarketEvidenceStore", FakeStore)
    before = source.stat()
    manifest, scan = materialize_round25_forensic_joint_feature_store(
        source_database=source,
        destination_database=destination,
        forensic_audit=audit,
        salvage_contract=contract,
        observed_at_ms=EVENT_START_MS + 700_000,
    )
    after = source.stat()

    assert before.st_size == after.st_size
    assert before.st_mtime_ns == after.st_mtime_ns
    assert manifest["schema_version"] == (
        POLYMARKET_ROUND25_FORENSIC_JOINT_STORE_MANIFEST_SCHEMA_VERSION
    )
    assert manifest["source_recorder_status"] == "failed"
    assert manifest["diagnostic_only"] is True
    assert manifest["admitted_condition_count"] == 0
    assert manifest["rejected_condition_count"] == 1
    assert manifest["target_accessed"] is False
    assert manifest["profitability_claim"] is False
    assert scan["processed_model_receipt_count"] == 0
    assert scan["rejection_reason_counts"] == {"stream_gap:clob_market": 1}
    assert audit_round25_joint_store(destination) == manifest
