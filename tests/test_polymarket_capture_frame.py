from __future__ import annotations

import pytest

from simple_ai_trading.polymarket_capture_frame import (
    CAPTURE_FRAME_MAGIC,
    CaptureFrameRecord,
    capture_frame_record_size,
    decode_capture_frame,
    encode_capture_frame,
)


def _record(
    *, sequence: int = 1, raw_text: str = '{"value":"\u20ac"}'
) -> CaptureFrameRecord:
    return CaptureFrameRecord(
        stream="polymarket_rtds",
        connection_id="rtds:btc:connection",
        sequence_number=sequence,
        received_wall_ms=1_784_058_600_000 + sequence,
        received_monotonic_ns=456_000 + sequence,
        raw_text=raw_text,
    )


def test_capture_frame_round_trip_preserves_exact_metadata_and_utf8() -> None:
    records = (_record(), _record(sequence=2, raw_text="PING"))

    frame, located = encode_capture_frame(records)
    decoded = decode_capture_frame(frame, expected_message_count=2)

    assert frame.startswith(CAPTURE_FRAME_MAGIC)
    assert tuple(item.record for item in decoded) == records
    assert decoded == located
    assert len(frame) == len(CAPTURE_FRAME_MAGIC) + sum(
        capture_frame_record_size(record) for record in records
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
def test_capture_frame_corruption_fails_closed(mutation, error: str) -> None:
    frame, _located = encode_capture_frame((_record(),))

    with pytest.raises(ValueError, match=error):
        decode_capture_frame(mutation(frame), expected_message_count=1)


def test_capture_frame_rejects_invalid_stream_and_count() -> None:
    invalid = CaptureFrameRecord(**{**_record().__dict__, "stream": "unknown"})
    with pytest.raises(ValueError, match="unsupported capture-frame stream"):
        encode_capture_frame((invalid,))
    with pytest.raises(ValueError, match="invalid message count"):
        encode_capture_frame(())

    frame, _located = encode_capture_frame((_record(),))
    with pytest.raises(ValueError, match="truncated"):
        decode_capture_frame(frame, expected_message_count=2)
