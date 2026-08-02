from __future__ import annotations

import json

import pytest

from simple_ai_trading.polymarket_redundant_union import (
    PolymarketClobLaneReceipt,
    PolymarketRedundantUnionBuilder,
)


def _receipt(
    lane: str,
    sequence: int,
    monotonic_ns: int,
    payload: object | str,
    *,
    connection: str | None = None,
) -> PolymarketClobLaneReceipt:
    return PolymarketClobLaneReceipt(
        lane_id=lane,
        connection_id=connection or f"{lane}:connection",
        sequence_number=sequence,
        received_wall_ms=1_800_000_000_000 + monotonic_ns // 1_000_000,
        received_monotonic_ns=monotonic_ns,
        raw_text=(
            payload
            if isinstance(payload, str)
            else json.dumps(payload, separators=(",", ":"), sort_keys=True)
        ),
    )


def test_union_pairs_redundant_events_and_preserves_both_receipts() -> None:
    event = {
        "event_type": "price_change",
        "market": "0xabc",
        "timestamp": "1800000000000",
        "price_changes": [{"asset_id": "7", "price": "0.5", "size": "10"}],
    }
    builder = PolymarketRedundantUnionBuilder(pairing_window_ms=2_000)

    assert builder.add(_receipt("clob-a", 1, 1_000_000_000, event)) == ()
    assert builder.add(_receipt("clob-b", 1, 1_025_000_000, event)) == ()
    trailing, audit = builder.finish()

    assert len(trailing) == 1
    assert trailing[0].selected_lane_id == "clob-a"
    assert len(trailing[0].lane_receipts) == 2
    assert trailing[0].source_time_ms == 1_800_000_000_000
    assert audit.union_event_count == 1
    assert audit.shared_event_count == 1
    assert audit.single_lane_event_count == 0
    assert audit.shared_fraction == 1.0
    assert audit.lane_coverage_fraction == {"clob-a": 1.0, "clob-b": 1.0}
    assert audit.receipt_difference_ms["maximum"] == 25.0
    assert len(audit.audit_sha256) == 64


def test_union_retains_unmatched_event_after_bounded_pairing_window() -> None:
    builder = PolymarketRedundantUnionBuilder(pairing_window_ms=100)
    first = {"event_type": "book", "asset_id": "7", "timestamp": "1"}
    second = {"event_type": "book", "asset_id": "8", "timestamp": "2"}

    assert builder.add(_receipt("clob-a", 1, 1_000_000_000, first)) == ()
    ready = builder.add(_receipt("clob-b", 1, 1_200_000_000, second))
    trailing, audit = builder.finish()

    assert [event.selected_lane_id for event in (*ready, *trailing)] == [
        "clob-a",
        "clob-b",
    ]
    assert audit.union_event_count == 2
    assert audit.shared_event_count == 0
    assert audit.single_lane_event_count == 2
    assert audit.lane_coverage_fraction == {"clob-a": 0.5, "clob-b": 0.5}


def test_union_external_watermark_flushes_without_losing_causal_order() -> None:
    builder = PolymarketRedundantUnionBuilder(pairing_window_ms=100)
    event = {"event_type": "book", "asset_id": "7", "timestamp": "1"}

    assert builder.add(_receipt("clob-a", 1, 1_000_000_000, event)) == ()
    ready = builder.advance(1_100_000_001)

    assert len(ready) == 1
    assert ready[0].selected_lane_id == "clob-a"
    with pytest.raises(ValueError, match="precedes the union watermark"):
        builder.add(_receipt("clob-b", 1, 1_050_000_000, event))
    with pytest.raises(ValueError, match="watermark regressed"):
        builder.advance(1_000_000_000)


def test_union_pairs_repeated_identical_events_fifo_without_dropping_occurrences() -> None:
    event = {"event_type": "best_bid_ask", "asset_id": "7", "timestamp": "3"}
    builder = PolymarketRedundantUnionBuilder(pairing_window_ms=1_000)
    receipts = (
        _receipt("clob-a", 1, 1_000_000_000, [event, event]),
        _receipt("clob-b", 1, 1_010_000_000, [event, event]),
    )
    for receipt in receipts:
        builder.add(receipt)
    events, audit = builder.finish()

    assert len(events) == 2
    assert [event.semantic_occurrence_index for event in events] == [1, 2]
    assert all(len(event.lane_receipts) == 2 for event in events)
    assert audit.lane_event_counts == {"clob-a": 2, "clob-b": 2}
    assert audit.shared_event_count == 2


def test_union_rejects_local_sequence_loss_clock_regression_and_ambiguous_json() -> None:
    event = {"event_type": "book", "asset_id": "7"}
    sequence_builder = PolymarketRedundantUnionBuilder()
    sequence_builder.add(_receipt("clob-a", 1, 2_000_000_000, event))
    with pytest.raises(ValueError, match="sequence is not contiguous"):
        sequence_builder.add(_receipt("clob-a", 3, 2_100_000_000, event))

    clock_builder = PolymarketRedundantUnionBuilder()
    clock_builder.add(_receipt("clob-b", 1, 2_000_000_000, event))
    with pytest.raises(ValueError, match="merged receipt order regressed"):
        clock_builder.add(_receipt("clob-a", 1, 1_999_999_999, event))

    json_builder = PolymarketRedundantUnionBuilder()
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        json_builder.add(
            _receipt(
                "clob-a",
                1,
                1_000_000_000,
                '{"event_type":"book","event_type":"price_change"}',
            )
        )


def test_union_requires_sequence_one_for_each_new_connection_and_finishes_once() -> None:
    event = {"event_type": "book", "asset_id": "7"}
    builder = PolymarketRedundantUnionBuilder()
    with pytest.raises(ValueError, match="start at sequence one"):
        builder.add(
            _receipt(
                "clob-a",
                2,
                1_000_000_000,
                event,
                connection="clob-a:new",
            )
        )

    valid = PolymarketRedundantUnionBuilder()
    valid.add(_receipt("clob-a", 1, 1_000_000_000, event))
    _events, audit = valid.finish()
    assert audit.terminal_pending_event_count == 0
    with pytest.raises(RuntimeError, match="already finished"):
        valid.finish()
