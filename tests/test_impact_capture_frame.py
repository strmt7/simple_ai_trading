from __future__ import annotations

import pytest

from simple_ai_trading.impact_capture_frame import (
    IMPACT_CAPTURE_FRAME_MAGIC,
    IMPACT_CAPTURE_FRAME_MAX_MESSAGES,
    ImpactCaptureFrameRecord,
    decode_impact_capture_frame,
    encode_impact_capture_frame,
    impact_capture_frame_record_size,
    scan_impact_capture_frame_receipts,
)


def _record(
    *,
    sequence: int = 1,
    stream: str = "binance_futures_public",
    raw_text: str = '{"stream":"btcusdt@depth@100ms","data":{"value":"\u20ac"}}',
) -> ImpactCaptureFrameRecord:
    return ImpactCaptureFrameRecord(
        stream=stream,
        connection_id=f"{stream}:connection",
        sequence_number=sequence,
        received_wall_ns=1_784_058_600_000_000_000 + sequence,
        received_monotonic_ns=456_000_000 + sequence,
        raw_text=raw_text,
    )


def test_impact_frame_round_trip_preserves_exact_wire_text_and_nanosecond_clocks() -> (
    None
):
    records = (
        _record(),
        _record(
            sequence=2,
            stream="binance_futures_market",
            raw_text='{ "data": {"e":"aggTrade"} }',
        ),
    )

    frame, located = encode_impact_capture_frame(records)
    decoded = decode_impact_capture_frame(frame, expected_message_count=2)
    receipts = scan_impact_capture_frame_receipts(frame, expected_message_count=2)

    assert frame.startswith(IMPACT_CAPTURE_FRAME_MAGIC)
    assert tuple(item.record for item in decoded) == records
    assert decoded == located
    assert receipts.minimum_received_wall_ns == records[0].received_wall_ns
    assert receipts.maximum_received_monotonic_ns == records[1].received_monotonic_ns
    assert receipts.stream_counts == {
        "binance_futures_market": 1,
        "binance_futures_public": 1,
    }
    assert len(frame) == len(IMPACT_CAPTURE_FRAME_MAGIC) + sum(
        impact_capture_frame_record_size(record) for record in records
    )
    for item in decoded:
        raw = item.record.raw_text.encode("utf-8")
        assert frame[item.raw_offset : item.raw_offset + item.raw_size] == raw


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda frame: b"BROKEN!!" + frame[8:], "envelope"),
        (lambda frame: frame[:-1], "boundary"),
        (lambda frame: frame + b"x", "trailing"),
    ],
)
def test_impact_frame_corruption_fails_closed(mutation, error: str) -> None:
    frame, _located = encode_impact_capture_frame((_record(),))

    with pytest.raises(ValueError, match=error):
        decode_impact_capture_frame(mutation(frame), expected_message_count=1)


def test_impact_frame_rejects_unknown_sources_empty_payloads_and_bad_counts() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        encode_impact_capture_frame(
            (ImpactCaptureFrameRecord(**{**_record().__dict__, "stream": "unknown"}),)
        )
    with pytest.raises(ValueError, match="empty or oversized"):
        encode_impact_capture_frame(
            (ImpactCaptureFrameRecord(**{**_record().__dict__, "raw_text": ""}),)
        )
    with pytest.raises(ValueError, match="message count"):
        encode_impact_capture_frame(())

    frame, _located = encode_impact_capture_frame((_record(),))
    with pytest.raises(ValueError, match="truncated"):
        decode_impact_capture_frame(frame, expected_message_count=2)


def test_impact_frame_accepts_frozen_v5_message_bound_and_rejects_overflow() -> None:
    records = tuple(
        _record(sequence=index + 1)
        for index in range(IMPACT_CAPTURE_FRAME_MAX_MESSAGES)
    )

    frame, located = encode_impact_capture_frame(records)
    receipts = scan_impact_capture_frame_receipts(
        frame,
        expected_message_count=IMPACT_CAPTURE_FRAME_MAX_MESSAGES,
    )

    assert len(located) == IMPACT_CAPTURE_FRAME_MAX_MESSAGES
    assert receipts.message_count == IMPACT_CAPTURE_FRAME_MAX_MESSAGES
    with pytest.raises(ValueError, match="message count"):
        encode_impact_capture_frame(records + (_record(sequence=20_000),))
