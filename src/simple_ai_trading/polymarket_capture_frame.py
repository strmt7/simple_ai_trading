"""Bounded binary frames for exact Polymarket public-feed capture."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Sequence


CAPTURE_FRAME_FORMAT = "receipt-metadata-length-prefixed-utf8-v1"
CAPTURE_FRAME_MAGIC = b"SATPMF4\x00"
CAPTURE_FRAME_MAX_BYTES = 64 * 1024 * 1024
CAPTURE_FRAME_MAX_MESSAGES = 1_024
CAPTURE_FRAME_MAX_RAW_BYTES = 8 * 1024 * 1024
CAPTURE_FRAME_MAX_CONNECTION_BYTES = 160

_RECORD_HEADER = struct.Struct("<B H Q q Q")
_RAW_SIZE = struct.Struct("<I")
_STREAM_TO_CODE = {
    "clob_market": 1,
    "polymarket_rtds": 2,
    "binance_spot": 3,
    "clob_rest_book": 4,
}
_CODE_TO_STREAM = {code: stream for stream, code in _STREAM_TO_CODE.items()}
_MAX_SIGNED_64 = (1 << 63) - 1
_MAX_UNSIGNED_64 = (1 << 64) - 1


@dataclass(frozen=True)
class CaptureFrameRecord:
    """One exact public-feed message plus local receipt metadata."""

    stream: str
    connection_id: str
    sequence_number: int
    received_wall_ms: int
    received_monotonic_ns: int
    raw_text: str


@dataclass(frozen=True)
class LocatedCaptureFrameRecord:
    """Decoded record with its raw payload location in the uncompressed frame."""

    record: CaptureFrameRecord
    message_index: int
    raw_offset: int
    raw_size: int


def _encoded_parts(record: CaptureFrameRecord) -> tuple[int, bytes, bytes]:
    stream = str(record.stream)
    try:
        stream_code = _STREAM_TO_CODE[stream]
    except KeyError as exc:
        raise ValueError(f"unsupported capture-frame stream: {stream}") from exc
    connection = str(record.connection_id).encode("utf-8", errors="strict")
    raw = str(record.raw_text).encode("utf-8", errors="strict")
    sequence = int(record.sequence_number)
    wall = int(record.received_wall_ms)
    monotonic = int(record.received_monotonic_ns)
    if not 1 <= len(connection) <= CAPTURE_FRAME_MAX_CONNECTION_BYTES:
        raise ValueError("capture-frame connection ID is invalid")
    if len(raw) > CAPTURE_FRAME_MAX_RAW_BYTES:
        raise ValueError("capture-frame raw payload is oversized")
    if not 0 <= sequence <= _MAX_UNSIGNED_64:
        raise ValueError("capture-frame sequence is outside unsigned 64-bit range")
    if not 0 <= wall <= _MAX_SIGNED_64:
        raise ValueError("capture-frame wall clock is outside signed 64-bit range")
    if not 0 <= monotonic <= _MAX_UNSIGNED_64:
        raise ValueError(
            "capture-frame monotonic clock is outside unsigned 64-bit range"
        )
    return stream_code, connection, raw


def capture_frame_record_size(record: CaptureFrameRecord) -> int:
    """Return the exact encoded byte count for one validated record."""

    _stream_code, connection, raw = _encoded_parts(record)
    return _RECORD_HEADER.size + len(connection) + _RAW_SIZE.size + len(raw)


def encode_capture_frame(
    records: Sequence[CaptureFrameRecord],
) -> tuple[bytes, tuple[LocatedCaptureFrameRecord, ...]]:
    """Encode one bounded frame and return exact raw-payload locations."""

    if not 1 <= len(records) <= CAPTURE_FRAME_MAX_MESSAGES:
        raise ValueError("capture frame has an invalid message count")
    frame = bytearray(CAPTURE_FRAME_MAGIC)
    located: list[LocatedCaptureFrameRecord] = []
    for message_index, record in enumerate(records):
        stream_code, connection, raw = _encoded_parts(record)
        frame.extend(
            _RECORD_HEADER.pack(
                stream_code,
                len(connection),
                int(record.sequence_number),
                int(record.received_wall_ms),
                int(record.received_monotonic_ns),
            )
        )
        frame.extend(connection)
        frame.extend(_RAW_SIZE.pack(len(raw)))
        raw_offset = len(frame)
        frame.extend(raw)
        located.append(
            LocatedCaptureFrameRecord(
                record=record,
                message_index=message_index,
                raw_offset=raw_offset,
                raw_size=len(raw),
            )
        )
        if len(frame) > CAPTURE_FRAME_MAX_BYTES:
            raise ValueError("capture frame exceeded its bounded size")
    return bytes(frame), tuple(located)


def decode_capture_frame(
    frame: bytes,
    *,
    expected_message_count: int,
) -> tuple[LocatedCaptureFrameRecord, ...]:
    """Decode and fully boundary-check one untrusted uncompressed frame."""

    payload = bytes(frame)
    expected = int(expected_message_count)
    if (
        not 1 <= expected <= CAPTURE_FRAME_MAX_MESSAGES
        or not len(CAPTURE_FRAME_MAGIC) < len(payload) <= CAPTURE_FRAME_MAX_BYTES
        or not payload.startswith(CAPTURE_FRAME_MAGIC)
    ):
        raise ValueError("capture frame envelope is invalid")
    offset = len(CAPTURE_FRAME_MAGIC)
    decoded: list[LocatedCaptureFrameRecord] = []
    for message_index in range(expected):
        header_end = offset + _RECORD_HEADER.size
        if header_end > len(payload):
            raise ValueError("capture frame record header is truncated")
        stream_code, connection_size, sequence, wall, monotonic = (
            _RECORD_HEADER.unpack_from(payload, offset)
        )
        offset = header_end
        stream = _CODE_TO_STREAM.get(stream_code)
        connection_end = offset + int(connection_size)
        if (
            stream is None
            or not 1 <= int(connection_size) <= CAPTURE_FRAME_MAX_CONNECTION_BYTES
            or connection_end + _RAW_SIZE.size > len(payload)
        ):
            raise ValueError("capture frame record metadata is invalid")
        try:
            connection_id = payload[offset:connection_end].decode(
                "utf-8", errors="strict"
            )
        except UnicodeDecodeError as exc:
            raise ValueError("capture frame connection ID is not UTF-8") from exc
        offset = connection_end
        raw_size = int(_RAW_SIZE.unpack_from(payload, offset)[0])
        offset += _RAW_SIZE.size
        raw_offset = offset
        raw_end = raw_offset + raw_size
        if raw_size > CAPTURE_FRAME_MAX_RAW_BYTES or raw_end > len(payload):
            raise ValueError("capture frame raw payload boundary is invalid")
        try:
            raw_text = payload[raw_offset:raw_end].decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("capture frame raw payload is not UTF-8") from exc
        offset = raw_end
        decoded.append(
            LocatedCaptureFrameRecord(
                record=CaptureFrameRecord(
                    stream=stream,
                    connection_id=connection_id,
                    sequence_number=int(sequence),
                    received_wall_ms=int(wall),
                    received_monotonic_ns=int(monotonic),
                    raw_text=raw_text,
                ),
                message_index=message_index,
                raw_offset=raw_offset,
                raw_size=raw_size,
            )
        )
    if offset != len(payload):
        raise ValueError("capture frame contains trailing or uncounted bytes")
    return tuple(decoded)


__all__ = [
    "CAPTURE_FRAME_FORMAT",
    "CAPTURE_FRAME_MAGIC",
    "CAPTURE_FRAME_MAX_BYTES",
    "CAPTURE_FRAME_MAX_MESSAGES",
    "CaptureFrameRecord",
    "LocatedCaptureFrameRecord",
    "capture_frame_record_size",
    "decode_capture_frame",
    "encode_capture_frame",
]
