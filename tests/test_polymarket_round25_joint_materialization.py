from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_capture_frame import CaptureFrameRecord
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_recorder import (
    MarketEvidence,
    PolymarketEvidenceStore,
    RawStreamMessage,
    StreamGap,
)
from simple_ai_trading.polymarket_round25_campaign import (
    POLYMARKET_ROUND25_DESIGN_SHA256,
    POLYMARKET_ROUND25_RESOLUTION_SOURCE,
)
from simple_ai_trading.polymarket_round25_dataset import (
    POLYMARKET_ROUND25_CALIBRATION_END_MS,
    POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    POLYMARKET_ROUND25_CAPTURE_PLAN_SHA256,
    POLYMARKET_ROUND25_SELECTION_END_MS,
    POLYMARKET_ROUND25_TRAIN_END_MS,
)
from simple_ai_trading.polymarket_round25_joint_materialization import (
    POLYMARKET_ROUND25_EXPECTED_DECISIONS,
    POLYMARKET_ROUND25_JOINT_MATERIALIZATION_CONTRACT_SHA256,
    POLYMARKET_ROUND25_SEQUENCE_CONTEXT_STEPS,
    Round25JointReceiptCondition,
    Round25JointMaterializationObserver,
    Round25SingleLaneClobDecoder,
    decode_round25_twap_record,
    load_round25_joint_receipt_conditions,
    materialize_round25_joint_condition,
)
from simple_ai_trading.polymarket_round25_terminal import (
    POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
    POLYMARKET_ROUND25_TERMINAL_TRANSPORT_SCHEMA_VERSION,
)
from simple_ai_trading.polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_TOPIC,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-joint-feature-materialization-contract-v1.json"
)
EVENT_START_MS = 1_800_000_000_000
CONDITION_ID = "0x" + "1" * 64
UP_TOKEN = "1" * 40
DOWN_TOKEN = "2" * 40
RUN_ID = "3" * 32
E18 = 1_000_000_000_000_000_000


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


def _condition(*, role: str = "train") -> Round25JointReceiptCondition:
    return Round25JointReceiptCondition(
        run_id=RUN_ID,
        segment_index=0,
        snapshot_sha256="4" * 64,
        snapshot_observed_wall_ms=EVENT_START_MS - 10_000,
        market_id="12345",
        condition_id=CONDITION_ID,
        slug=f"btc-updown-5m-{EVENT_START_MS // 1_000}",
        event_start_ms=EVENT_START_MS,
        event_end_ms=EVENT_START_MS + 300_000,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        resolution_source=POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        role=role,
    ).validated()


def _twap_frame(offset_ms: int, *, ordinal: int) -> CaptureFrameRecord:
    source = EVENT_START_MS + offset_ms
    publisher = source + 50
    receipt = source + 100
    exact = 65_000 * E18 + (offset_ms // 1_000) * 10**14
    return CaptureFrameRecord(
        stream="polymarket_rtds",
        connection_id="rtds:chainlink:" + "a" * 32,
        sequence_number=ordinal,
        received_wall_ms=receipt,
        received_monotonic_ns=receipt * 1_000_000 + ordinal,
        raw_text=json.dumps(
            {
                "connection_id": "public-wire-fixture",
                "payload": {
                    "full_accuracy_value": str(exact),
                    "symbol": "btc/usd",
                    "timestamp": source,
                    "value": exact / E18,
                },
                "timestamp": publisher,
                "topic": POLYMARKET_ROUND25_TWAP_TOPIC,
                "type": "update",
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _book_payload(token: str, *, offset_ms: int, up: bool) -> dict[str, object]:
    bid = 0.49 if up else 0.48
    ask = 0.51 if up else 0.52
    return {
        "asks": [
            {"price": format(ask, ".2f"), "size": "9"},
            {"price": format(ask + 0.01, ".2f"), "size": "6"},
        ],
        "asset_id": token,
        "bids": [
            {"price": format(bid, ".2f"), "size": "8"},
            {"price": format(bid - 0.01, ".2f"), "size": "7"},
        ],
        "event_type": "book",
        "hash": hashlib.sha256(
            f"{token}:{offset_ms}".encode("ascii")
        ).hexdigest(),
        "market": CONDITION_ID,
        "timestamp": str(EVENT_START_MS + offset_ms),
    }


def _sources(seconds: int = 300) -> tuple[object, ...]:
    pending: list[tuple[int, str, object]] = []
    for second in range(seconds):
        offset = second * 1_000
        pending.append((EVENT_START_MS + offset + 100, "twap", second + 1))
        pending.append((EVENT_START_MS + offset + 120, "clob", (UP_TOKEN, True)))
        pending.append((EVENT_START_MS + offset + 121, "clob", (DOWN_TOKEN, False)))
    decoder = Round25SingleLaneClobDecoder()
    clob_sequence = 0
    output: list[object] = []
    for ordinal, (receipt, kind, value) in enumerate(sorted(pending), start=1):
        if kind == "twap":
            output.append(
                _twap_frame(
                    receipt - EVENT_START_MS - 100,
                    ordinal=int(value),
                )
            )
            continue
        clob_sequence += 1
        token, up = value  # type: ignore[misc]
        offset = receipt - EVENT_START_MS - (120 if up else 121)
        message = RawStreamMessage(
            stream="clob_market",
            connection_id="clob:" + "b" * 32,
            sequence_number=clob_sequence,
            received_wall_ms=receipt,
            received_monotonic_ns=receipt * 1_000_000 + ordinal,
            raw_text=json.dumps(
                _book_payload(str(token), offset_ms=offset, up=bool(up)),
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        output.extend(event for event, _condition_id in decoder.add(message))
    return tuple(output)


def _raw_messages(seconds: int = 300) -> tuple[RawStreamMessage, ...]:
    pending: list[tuple[int, str, dict[str, object]]] = []
    for second in range(seconds):
        offset = second * 1_000
        pending.append(
            (
                EVENT_START_MS + offset + 100,
                "polymarket_rtds",
                json.loads(_twap_frame(offset, ordinal=second + 1).raw_text),
            )
        )
        pending.append(
            (
                EVENT_START_MS + offset + 120,
                "clob_market",
                _book_payload(UP_TOKEN, offset_ms=offset, up=True),
            )
        )
        pending.append(
            (
                EVENT_START_MS + offset + 121,
                "clob_market",
                _book_payload(DOWN_TOKEN, offset_ms=offset, up=False),
            )
        )
    sequence = {"clob_market": 0, "polymarket_rtds": 0}
    output: list[RawStreamMessage] = []
    for ordinal, (receipt, stream, payload) in enumerate(sorted(pending), start=1):
        sequence[stream] += 1
        output.append(
            RawStreamMessage(
                stream=stream,
                connection_id=(
                    "clob:" + "b" * 32
                    if stream == "clob_market"
                    else "rtds:chainlink:" + "a" * 32
                ),
                sequence_number=sequence[stream],
                received_wall_ms=receipt,
                received_monotonic_ns=receipt * 1_000_000 + ordinal,
                raw_text=json.dumps(payload, separators=(",", ":"), sort_keys=True),
            )
        )
    return tuple(output)


def _market_evidence(
    *,
    event_start_ms: int,
    identity_digit: str,
) -> MarketEvidence:
    condition_id = "0x" + identity_digit * 64
    up_token = identity_digit * 40
    down_digit = str(int(identity_digit) + 5)
    down_token = down_digit * 40
    start = datetime.fromtimestamp(event_start_ms / 1_000, tz=UTC)
    end = datetime.fromtimestamp((event_start_ms + 300_000) / 1_000, tz=UTC)
    payload: dict[str, object] = {
        "acceptingOrders": True,
        "active": True,
        "clobTokenIds": json.dumps([up_token, down_token]),
        "closed": False,
        "conditionId": condition_id,
        "cryptoMarketConfig": {
            "asset": "btc",
            "duration": "5m",
            "id": "btc-5m-twap-30",
            "twapEnabled": True,
            "twapLookbackSeconds": 30,
        },
        "cryptoMarketConfigId": "btc-5m-twap-30",
        "enableOrderBook": True,
        "endDate": end.isoformat().replace("+00:00", "Z"),
        "eventStartTime": start.isoformat().replace("+00:00", "Z"),
        "feeSchedule": {
            "exponent": 1,
            "rate": 0.07,
            "rebateRate": 0.2,
            "takerOnly": True,
        },
        "feesEnabled": True,
        "id": f"round25-joint-{identity_digit}",
        "liquidityNum": 20_000.5,
        "orderMinSize": 5,
        "orderPriceMinTickSize": 0.01,
        "outcomes": '["Up", "Down"]',
        "question": "Bitcoin Up or Down",
        "resolutionSource": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        "slug": f"btc-updown-5m-{event_start_ms // 1_000}",
        "volumeNum": 50_000.25,
    }
    market = parse_polymarket_five_minute_market(payload)
    clob = json.dumps(
        {"condition": market.condition_id, "tokens": list(market.token_ids)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    fee = '{"base_fee":1000}'
    observed = max(POLYMARKET_ROUND25_CAMPAIGN_START_MS, event_start_ms - 10_000)
    return MarketEvidence(
        market=market,
        observed_wall_ms=observed,
        observed_monotonic_ns=observed * 1_000_000 + int(identity_digit, 16),
        clob_info_json=clob,
        clob_info_sha256=hashlib.sha256(clob.encode("ascii")).hexdigest(),
        up_fee_rate_json=fee,
        up_fee_rate_sha256=hashlib.sha256(fee.encode("ascii")).hexdigest(),
        down_fee_rate_json=fee,
        down_fee_rate_sha256=hashlib.sha256(fee.encode("ascii")).hexdigest(),
        maker_base_fee=1_000,
        taker_base_fee=1_000,
        taker_order_delay_enabled=True,
        minimum_order_age_seconds=0,
    )


def _metadata_database(path: Path, event_starts: tuple[int, ...]) -> None:
    with PolymarketEvidenceStore(path) as store:
        store.start_run(RUN_ID, POLYMARKET_ROUND25_CAMPAIGN_START_MS)
        for index, event_start_ms in enumerate(event_starts, start=1):
            store.record_market_evidence(
                RUN_ID,
                _market_evidence(
                    event_start_ms=event_start_ms,
                    identity_digit=format(index, "x"),
                ),
            )
        store.connect().execute(
            "UPDATE polymarket_recorder_run SET status = 'complete' WHERE run_id = ?",
            [RUN_ID],
        )


def _terminal_transport_manifest(
    *,
    condition_count: int,
    source_plan_sha256: str = POLYMARKET_ROUND25_CAPTURE_PLAN_SHA256,
) -> dict[str, object]:
    duration_ms = (
        POLYMARKET_ROUND25_SELECTION_END_MS
        - POLYMARKET_ROUND25_CAMPAIGN_START_MS
    )
    segment = {
        "condition_count": condition_count,
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
        "source_capture_design_sha256": POLYMARKET_ROUND25_DESIGN_SHA256,
        "source_plan_sha256": source_plan_sha256,
        "source_qualification_sha256": "5" * 64,
        "terminal_design_sha256": POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


def test_joint_materialization_contract_is_self_hashed_and_target_free() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    claimed = payload.pop("contract_sha256")

    assert claimed == _canonical_sha256(payload)
    assert claimed == POLYMARKET_ROUND25_JOINT_MATERIALIZATION_CONTRACT_SHA256
    assert payload["source_boundary"]["target_accessed"] is False
    assert payload["bounded_population"]["sequence_steps_per_endpoint"] == 64
    assert not any(payload["truth_state"].values())


def test_joint_condition_loader_derives_v2_roles_and_excludes_purge(
    tmp_path: Path,
) -> None:
    event_starts = (
        POLYMARKET_ROUND25_CAMPAIGN_START_MS,
        POLYMARKET_ROUND25_TRAIN_END_MS - 300_000,
        POLYMARKET_ROUND25_TRAIN_END_MS + 300_000,
        POLYMARKET_ROUND25_CALIBRATION_END_MS + 300_000,
    )
    database = tmp_path / "round25-source.duckdb"
    _metadata_database(database, event_starts)

    conditions, counts = load_round25_joint_receipt_conditions(
        database=database,
        terminal_transport_manifest=_terminal_transport_manifest(
            condition_count=len(event_starts)
        ),
    )

    assert [item.role for item in conditions] == [
        "train",
        "calibration",
        "selection",
    ]
    assert [item.event_start_ms for item in conditions] == [
        event_starts[0],
        event_starts[2],
        event_starts[3],
    ]
    assert counts == {
        "admitted_condition_count": 3,
        "calibration_condition_count": 1,
        "purged_condition_count": 1,
        "selection_condition_count": 1,
        "source_snapshot_count": 4,
        "train_condition_count": 1,
    }
    assert all(not item.role.startswith("tune_") for item in conditions)


def test_joint_condition_loader_rejects_capture_plan_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="capture plan differs"):
        load_round25_joint_receipt_conditions(
            database=tmp_path / "absent.duckdb",
            terminal_transport_manifest=_terminal_transport_manifest(
                condition_count=1,
                source_plan_sha256="a" * 64,
            ),
        )


def test_joint_condition_loader_rejects_active_wal(tmp_path: Path) -> None:
    database = tmp_path / "round25-source.duckdb"
    _metadata_database(database, (POLYMARKET_ROUND25_CAMPAIGN_START_MS,))
    Path(f"{database}.wal").write_bytes(b"active")

    with pytest.raises(ValueError, match="terminal WAL-free database"):
        load_round25_joint_receipt_conditions(
            database=database,
            terminal_transport_manifest=_terminal_transport_manifest(
                condition_count=1
            ),
        )


def test_joint_condition_loader_rejects_recorder_status_drift(tmp_path: Path) -> None:
    database = tmp_path / "round25-source.duckdb"
    _metadata_database(database, (POLYMARKET_ROUND25_CAMPAIGN_START_MS,))
    with PolymarketEvidenceStore(database) as store:
        store.connect().execute(
            "UPDATE polymarket_recorder_run SET status = 'failed' WHERE run_id = ?",
            [RUN_ID],
        )

    with pytest.raises(ValueError, match="database status differs"):
        load_round25_joint_receipt_conditions(
            database=database,
            terminal_transport_manifest=_terminal_transport_manifest(
                condition_count=1
            ),
        )


def test_joint_observer_streams_one_condition_to_sink() -> None:
    results = []
    observer = Round25JointMaterializationObserver(
        (_condition(),),
        sink=results.append,
    )
    segment = {
        "ended_at_ms": EVENT_START_MS + 305_000,
        "run_id": RUN_ID,
        "started_at_ms": EVENT_START_MS - 120_000,
    }
    observer.start_run(segment, ())
    for message in _raw_messages():
        observer.observe_message(segment, message)
    observer.finish_run(segment)

    assert len(results) == 1
    assert results[0].admitted is True
    assert results[0].available_decision_count == 592
    assert observer.condition_count == 1
    assert observer.admitted_condition_count == 1
    assert observer.persisted_snapshot_count == len(results[0].persisted_snapshots)
    assert observer.rejection_counts == {}


def test_joint_observer_rejects_gap_without_buffering_partial_rows() -> None:
    results = []
    observer = Round25JointMaterializationObserver(
        (_condition(role="selection"),),
        sink=results.append,
    )
    segment = {
        "ended_at_ms": EVENT_START_MS + 305_000,
        "run_id": RUN_ID,
        "started_at_ms": EVENT_START_MS - 120_000,
    }
    observer.start_run(
        segment,
        (
            StreamGap(
                stream="clob_market",
                connection_id="clob:" + "b" * 32,
                opened_at_ms=EVENT_START_MS + 1_000,
                reason="fixture_disconnect",
                last_sequence_number=1,
            ),
        ),
    )
    for message in _raw_messages(seconds=4):
        observer.observe_message(segment, message)
    observer.finish_run(segment)

    assert len(results) == 1
    assert results[0].admitted is False
    assert results[0].rejection_reasons == ("stream_gap:clob_market",)
    assert results[0].source_record_count == 0
    assert results[0].persisted_snapshots == ()
    assert observer.rejection_counts == {"stream_gap:clob_market": 1}


def test_exact_five_minute_replay_retains_only_endpoint_contexts() -> None:
    result = materialize_round25_joint_condition(
        condition=_condition(),
        sources=_sources(),
    )

    assert result.admitted is True
    assert result.decision_count == POLYMARKET_ROUND25_EXPECTED_DECISIONS
    assert result.available_decision_count == 592
    assert dict(result.unavailable_reason_counts)[
        "clob:up_book_receipt_stale"
    ] == 599
    assert dict(result.unavailable_reason_counts)[
        "clob:down_book_receipt_stale"
    ] == 599
    assert len(result.selected_endpoint_decision_time_ms) == 16
    assert len(result.persisted_snapshots) <= (
        16 * POLYMARKET_ROUND25_SEQUENCE_CONTEXT_STEPS
    )
    assert len(result.persisted_snapshots) < result.available_decision_count
    assert all(
        row.maximum_receipt_ms <= row.decision_time_ms
        for row in result.persisted_snapshots
    )
    assert result.target_accessed is False
    assert result.trading_authority is False
    assert result.validated() is result


def test_underfilled_condition_is_rejected_without_partial_rows() -> None:
    result = materialize_round25_joint_condition(
        condition=_condition(role="selection"),
        sources=_sources(seconds=4),
    )

    assert result.admitted is False
    assert result.rejection_reasons
    assert result.persisted_snapshots == ()
    assert result.selected_endpoint_decision_time_ms == ()
    assert result.target_accessed is False


def test_twap_decoder_accepts_empty_control_and_rejects_schema_drift() -> None:
    control = replace(_twap_frame(0, ordinal=1), raw_text="")
    assert decode_round25_twap_record(control) is None

    payload = json.loads(_twap_frame(0, ordinal=1).raw_text)
    payload["payload"]["window_s"] = 30
    drifted = replace(
        _twap_frame(0, ordinal=1),
        raw_text=json.dumps(payload, separators=(",", ":")),
    )
    with pytest.raises(ValueError, match="identity differs"):
        decode_round25_twap_record(drifted)


def test_joint_materialization_rejects_source_chronology_regression() -> None:
    sources = list(_sources(seconds=4))
    sources[0], sources[1] = sources[1], sources[0]

    with pytest.raises(ValueError, match="chronology regressed"):
        materialize_round25_joint_condition(
            condition=_condition(),
            sources=sources,
        )


def test_joint_materialization_self_hash_detects_tampering() -> None:
    result = materialize_round25_joint_condition(
        condition=_condition(role="calibration"),
        sources=_sources(),
    )

    with pytest.raises(ValueError, match="materialization differs"):
        replace(result, source_record_count=result.source_record_count + 1).validated()
