from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_capture_frame import CaptureFrameRecord
from simple_ai_trading.polymarket_recorder import (
    MarketEvidence,
    PolymarketEvidenceStore,
    RawStreamMessage,
    StreamGap,
)
from simple_ai_trading import polymarket_round21_corpus as corpus_module
from simple_ai_trading.polymarket_redundant_union import (
    PolymarketClobLaneReceipt,
    PolymarketRedundantUnionBuilder,
)
from simple_ai_trading.polymarket_round21_corpus import (
    POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256,
    Round21ConditionSource,
    Round21CoreCondition,
    Round21CoreCorpusObserver,
    build_round21_core_condition_materialization,
    load_round21_core_corpus_design,
    load_round21_core_conditions,
    validate_round21_condition_admission,
)
from simple_ai_trading.polymarket_round21_dataset import Round21PartitionPolicy


ROOT = Path(__file__).resolve().parents[1]
EVENT_START_MS = 1_800_001_200_000
EVENT_END_MS = EVENT_START_MS + 300_000
CONDITION_ID = "0x" + "1" * 64
UP_TOKEN = "1" * 40
DOWN_TOKEN = "2" * 40
RUN_ID = "3" * 32
CONNECTION_ID = "rtds:chainlink:btc:" + "c" * 32


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _rehash(value: dict[str, object], name: str) -> dict[str, object]:
    body = dict(value)
    body.pop(name, None)
    body[name] = hashlib.sha256(_canonical(body).encode("ascii")).hexdigest()
    return body


def _condition(*, event_start_ms: int = EVENT_START_MS) -> Round21CoreCondition:
    return Round21CoreCondition(
        run_id=RUN_ID,
        segment_index=1,
        snapshot_sha256="a" * 64,
        snapshot_observed_wall_ms=event_start_ms - 30_000,
        condition_id=CONDITION_ID,
        event_start_ms=event_start_ms,
        event_end_ms=event_start_ms + 300_000,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
    )


def _book(token: str, offset_ms: int) -> dict[str, object]:
    up = token == UP_TOKEN
    bid = "0.49" if up else "0.48"
    ask = "0.51" if up else "0.52"
    return {
        "event_type": "book",
        "market": CONDITION_ID,
        "asset_id": token,
        "timestamp": str(EVENT_START_MS + offset_ms),
        "hash": hashlib.sha256(f"{token}:{offset_ms}".encode("ascii")).hexdigest(),
        "bids": [
            {"price": bid, "size": "8"},
            {"price": format(float(bid) - 0.01, ".2f"), "size": "7"},
        ],
        "asks": [
            {"price": ask, "size": "9"},
            {"price": format(float(ask) + 0.01, ".2f"), "size": "6"},
        ],
    }


def _union_events(*, both_lanes: bool = True) -> tuple:
    builder = PolymarketRedundantUnionBuilder(pairing_window_ms=2_000)
    sequences = {"clob-a": 0, "clob-b": 0}
    output = []
    lanes = ("clob-a", "clob-b") if both_lanes else ("clob-a",)
    for offset_ms in range(0, 300_000, 1_000):
        for token, token_delay_ms in ((UP_TOKEN, 0), (DOWN_TOKEN, 20)):
            event_offset = offset_ms + token_delay_ms
            raw = _canonical(_book(token, event_offset))
            for lane_index, lane in enumerate(lanes):
                sequences[lane] += 1
                output.extend(
                    builder.add(
                        PolymarketClobLaneReceipt(
                            lane_id=lane,
                            connection_id=f"{lane}:" + lane[-1] * 32,
                            sequence_number=sequences[lane],
                            received_wall_ms=(
                                EVENT_START_MS + event_offset + 10 + lane_index * 5
                            ),
                            received_monotonic_ns=(
                                (EVENT_START_MS + event_offset + 10) * 1_000_000
                                + lane_index * 5_000_000
                            ),
                            raw_text=raw,
                        )
                    )
                )
    trailing, _audit = builder.finish()
    output.extend(trailing)
    return tuple(output)


def _chainlink_records(
    *,
    include_close: bool = True,
    reconnect_at_sequence: int | None = None,
    contradictory_close: bool = False,
) -> tuple[CaptureFrameRecord, ...]:
    records: list[CaptureFrameRecord] = []
    first_sequence = 417
    offsets = range(0, 300_001 if include_close else 300_000, 1_000)
    for index, offset_ms in enumerate(offsets):
        sequence = first_sequence + index
        connection = (
            "rtds:chainlink:btc:" + "d" * 32
            if reconnect_at_sequence is not None and sequence >= reconnect_at_sequence
            else CONNECTION_ID
        )
        payload = {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "timestamp": EVENT_START_MS + offset_ms,
            "payload": {
                "symbol": "btc/usd",
                "timestamp": EVENT_START_MS + offset_ms,
                "value": format(60_000.0 + index * 0.5, ".8f"),
            },
        }
        records.append(
            CaptureFrameRecord(
                stream="polymarket_rtds",
                connection_id=connection,
                sequence_number=sequence,
                received_wall_ms=EVENT_START_MS + offset_ms + 10,
                received_monotonic_ns=(EVENT_START_MS + offset_ms + 10) * 1_000_000,
                raw_text=_canonical(payload),
            )
        )
    if contradictory_close:
        last = records[-1]
        payload = json.loads(last.raw_text)
        payload["payload"]["value"] = "70000.00000000"
        records.append(
            replace(
                last,
                sequence_number=last.sequence_number + 1,
                received_wall_ms=last.received_wall_ms + 1,
                received_monotonic_ns=last.received_monotonic_ns + 1_000_000,
                raw_text=_canonical(payload),
            )
        )
    return tuple(records)


def _raw_messages() -> tuple[RawStreamMessage, ...]:
    messages: list[RawStreamMessage] = []
    sequences = {"clob-a": 0, "clob-b": 0}
    for offset_ms in range(0, 300_000, 1_000):
        for token, token_delay_ms in ((UP_TOKEN, 0), (DOWN_TOKEN, 20)):
            event_offset = offset_ms + token_delay_ms
            raw = _canonical(_book(token, event_offset))
            for lane_index, lane in enumerate(("clob-a", "clob-b")):
                sequences[lane] += 1
                messages.append(
                    RawStreamMessage(
                        stream="clob_market",
                        connection_id=f"{lane}:" + lane[-1] * 32,
                        sequence_number=sequences[lane],
                        received_wall_ms=(
                            EVENT_START_MS + event_offset + 10 + lane_index * 5
                        ),
                        received_monotonic_ns=(
                            (EVENT_START_MS + event_offset + 10) * 1_000_000
                            + lane_index * 5_000_000
                        ),
                        raw_text=raw,
                    )
                )
    messages.extend(
        RawStreamMessage(**record.__dict__) for record in _chainlink_records()
    )
    return tuple(
        sorted(
            messages,
            key=lambda value: (
                value.received_monotonic_ns,
                value.received_wall_ms,
                value.stream,
                value.connection_id,
            ),
        )
    )


def _source(
    *,
    both_lanes: bool = True,
    include_close: bool = True,
    reconnect_at_sequence: int | None = None,
    contradictory_close: bool = False,
    gaps: tuple[StreamGap, ...] = (),
    lane_pause: tuple[int, int] | None = None,
) -> Round21ConditionSource:
    wall_times = tuple(EVENT_START_MS + offset + 10 for offset in range(0, 300_000, 1_000))
    if lane_pause is not None:
        pause_start, pause_end = lane_pause
        wall_times = tuple(
            value
            for value in wall_times
            if not EVENT_START_MS + pause_start <= value <= EVENT_START_MS + pause_end
        )
    return Round21ConditionSource(
        union_events=_union_events(both_lanes=both_lanes),
        chainlink_records=_chainlink_records(
            include_close=include_close,
            reconnect_at_sequence=reconnect_at_sequence,
            contradictory_close=contradictory_close,
        ),
        lane_event_wall_ms={
            "clob-a": wall_times,
            "clob-b": wall_times if both_lanes else (),
        },
        stream_gaps=gaps,
    )


def _policy() -> Round21PartitionPolicy:
    return Round21PartitionPolicy.create(
        campaign_start_ms=EVENT_START_MS,
        campaign_end_ms=EVENT_START_MS + 30 * 86_400_000,
    )


def test_round21_core_corpus_design_is_hash_bound_and_target_blind() -> None:
    design = load_round21_core_corpus_design(ROOT)

    assert design["design_sha256"] == POLYMARKET_ROUND21_CORE_CORPUS_DESIGN_SHA256
    assert design["source_boundary"]["outcomes_consulted"] is False
    assert design["source_boundary"]["optional_binance_consulted"] is False
    assert design["continuous_chainlink_slice"][
        "first_observed_sequence_number_is_bound_not_renumbered"
    ] is True


def test_round21_condition_materializes_full_causal_five_minute_grid() -> None:
    result = build_round21_core_condition_materialization(
        condition=_condition(),
        source=_source(),
        partition_policy=_policy(),
    )

    assert result.admission["admitted"] is True
    assert result.admission["role"] == "train"
    assert result.admission["union_event_count"] == 600
    assert result.admission["lane_coverage_fraction"] == {
        "clob-a": 1.0,
        "clob-b": 1.0,
    }
    assert result.admission["shared_fraction"] == 1.0
    assert result.admission["exact_chainlink_open_receipt_count"] == 1
    assert result.admission["exact_chainlink_close_receipt_count"] == 1
    assert result.admission["available_feature_row_count"] > 1_000
    assert (
        result.admission["available_feature_row_count"]
        + result.admission["unavailable_feature_row_count"]
        == 1_200
    )
    assert result.available_features[0].decision_time_ms >= EVENT_START_MS + 30_000
    assert all(
        row.maximum_receipt_ms <= row.decision_time_ms
        for row in result.available_features
    )
    assert "chainlink_connection_gap" not in result.unavailable_reason_counts
    assert result.admission["outcomes_consulted"] is False
    assert result.admission["optional_binance_consulted"] is False
    assert result.admission["model_data_eligible"] is False


@pytest.mark.parametrize(
    ("source", "reason"),
    (
        (_source(both_lanes=False), "minimum_lane_coverage_not_met"),
        (_source(include_close=False), "missing_exact_chainlink_close"),
        (
            _source(reconnect_at_sequence=500),
            "chainlink_connection_changed",
        ),
        (
            _source(contradictory_close=True),
            "contradictory_exact_chainlink_close",
        ),
    ),
)
def test_round21_condition_rejects_frozen_transport_failures(
    source: Round21ConditionSource,
    reason: str,
) -> None:
    result = build_round21_core_condition_materialization(
        condition=_condition(),
        source=source,
        partition_policy=_policy(),
    )

    assert result.admission["admitted"] is False
    assert reason in result.admission["rejection_reasons"]
    assert result.available_features == ()
    assert result.unavailable_feature_row_count == 0


def test_round21_joint_gap_rejects_only_affected_condition() -> None:
    gap_time = EVENT_START_MS + 100_000
    gaps = tuple(
        StreamGap(
            stream="clob_market",
            connection_id=f"{lane}:" + lane[-1] * 32,
            opened_at_ms=gap_time,
            reason="fixture_disconnect",
            last_sequence_number=100,
        )
        for lane in ("clob-a", "clob-b")
    )
    result = build_round21_core_condition_materialization(
        condition=_condition(),
        source=_source(gaps=gaps, lane_pause=(100_000, 115_000)),
        partition_policy=_policy(),
    )

    assert result.admission["admitted"] is False
    assert result.admission["joint_unhealthy_ms"] > 2_000
    assert "joint_clob_unhealthy_limit_exceeded" in result.admission[
        "rejection_reasons"
    ]
    assert result.admission["lane_gap_counts"] == {"clob-a": 1, "clob-b": 1}


def test_round21_purge_condition_is_audited_but_not_materialized() -> None:
    campaign_start = EVENT_START_MS - 18 * 86_400_000
    policy = Round21PartitionPolicy.create(
        campaign_start_ms=campaign_start,
        campaign_end_ms=campaign_start + 30 * 86_400_000,
    )
    result = build_round21_core_condition_materialization(
        condition=_condition(),
        source=_source(),
        partition_policy=policy,
    )

    assert result.admission["role"] == "purge_train_to_tune"
    assert result.available_features == ()


def test_round21_condition_admission_is_tamper_evident() -> None:
    result = build_round21_core_condition_materialization(
        condition=_condition(),
        source=_source(include_close=False),
        partition_policy=_policy(),
    )
    changed = dict(result.admission)
    changed["admitted"] = True

    with pytest.raises(ValueError, match="condition admission differs"):
        validate_round21_condition_admission(_rehash(changed, "admission_sha256"))


def test_round21_terminal_observer_materializes_during_one_receipt_pass() -> None:
    results = []
    observer = Round21CoreCorpusObserver(
        conditions=(_condition(),),
        partition_policy=_policy(),
        sink=results.append,
    )
    segment = {"run_id": RUN_ID}

    observer.start_run(segment, ())
    for message in _raw_messages():
        observer.observe_message(segment, message)
    observer.finish_run(segment)

    assert observer.materialized_condition_count == 1
    assert len(results) == 1
    assert results[0].admission["admitted"] is True
    assert results[0].admission["available_feature_row_count"] > 1_000


def test_round21_terminal_observer_rejects_clock_regression() -> None:
    observer = Round21CoreCorpusObserver(
        conditions=(_condition(),),
        partition_policy=_policy(),
        sink=lambda _result: None,
    )
    segment = {"run_id": RUN_ID}
    messages = _raw_messages()
    observer.start_run(segment, ())
    observer.observe_message(segment, messages[0])
    regressed = replace(
        messages[1],
        received_wall_ms=messages[0].received_wall_ms - 1,
    )

    with pytest.raises(ValueError, match="clock regressed"):
        observer.observe_message(segment, regressed)


def test_round21_terminal_observer_flushes_union_before_rtds_finalization() -> None:
    results = []
    observer = Round21CoreCorpusObserver(
        conditions=(_condition(),),
        partition_policy=_policy(),
        sink=results.append,
    )
    segment = {"run_id": RUN_ID}
    messages = _raw_messages()
    terminal_rtds = next(
        message
        for message in messages
        if message.stream == "polymarket_rtds"
        and message.received_wall_ms == EVENT_END_MS + 10
    )
    delayed = replace(
        terminal_rtds,
        received_wall_ms=EVENT_END_MS + 5_001,
        received_monotonic_ns=(EVENT_END_MS + 5_001) * 1_000_000,
    )

    observer.start_run(segment, ())
    for message in messages:
        if message is not terminal_rtds:
            observer.observe_message(segment, message)
    observer.observe_message(segment, delayed)
    observer.finish_run(segment)

    assert len(results) == 1
    assert results[0].admission["union_event_count"] == 600


def test_round21_snapshot_loader_reconciles_actual_duckdb_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime.fromtimestamp(EVENT_START_MS / 1_000, tz=UTC)
    end = datetime.fromtimestamp(EVENT_END_MS / 1_000, tz=UTC)
    payload = {
        "id": "btc-five-minute-fixture",
        "question": "Bitcoin Up or Down",
        "conditionId": CONDITION_ID,
        "slug": f"btc-updown-5m-{EVENT_START_MS // 1_000}",
        "eventStartTime": start.isoformat().replace("+00:00", "Z"),
        "endDate": end.isoformat().replace("+00:00", "Z"),
        "active": True,
        "closed": False,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "clobTokenIds": json.dumps([UP_TOKEN, DOWN_TOKEN]),
        "outcomes": '["Up","Down"]',
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5,
        "feesEnabled": True,
        "feeSchedule": {
            "exponent": 1,
            "rate": 0.07,
            "takerOnly": True,
            "rebateRate": 0.2,
        },
        "liquidityNum": 20_000,
        "volumeNum": 50_000,
        "resolutionSource": "https://data.chain.link/streams/btc-usd",
    }
    market = parse_polymarket_five_minute_market(payload)
    clob_json = _canonical({"condition_id": CONDITION_ID})
    fee_json = _canonical({"base_fee": 1000})
    database = tmp_path / "snapshots.duckdb"
    with PolymarketEvidenceStore(database) as store:
        store.start_run(RUN_ID, EVENT_START_MS - 60_000)
        store.record_market_evidence(
            RUN_ID,
            MarketEvidence(
                market=market,
                observed_wall_ms=EVENT_START_MS - 30_000,
                observed_monotonic_ns=(EVENT_START_MS - 30_000) * 1_000_000,
                clob_info_json=clob_json,
                clob_info_sha256=hashlib.sha256(
                    clob_json.encode("ascii")
                ).hexdigest(),
                up_fee_rate_json=fee_json,
                up_fee_rate_sha256=hashlib.sha256(
                    fee_json.encode("ascii")
                ).hexdigest(),
                down_fee_rate_json=fee_json,
                down_fee_rate_sha256=hashlib.sha256(
                    fee_json.encode("ascii")
                ).hexdigest(),
                maker_base_fee=0,
                taker_base_fee=1000,
                taker_order_delay_enabled=True,
                minimum_order_age_seconds=0,
            ),
        )
    transport = {
        "segments": [
            {
                "run_id": RUN_ID,
                "segment_index": 7,
                "eligible_for_condition_rebuild": True,
            }
        ]
    }
    monkeypatch.setattr(
        corpus_module,
        "validate_round21_terminal_transport_manifest",
        lambda value: value,
    )

    conditions = load_round21_core_conditions(
        database=database,
        terminal_transport_manifest=transport,
    )

    assert len(conditions) == 1
    assert conditions[0].segment_index == 7
    assert conditions[0].condition_id == CONDITION_ID
    assert conditions[0].event_start_ms == EVENT_START_MS
    assert conditions[0].up_token_id == UP_TOKEN
