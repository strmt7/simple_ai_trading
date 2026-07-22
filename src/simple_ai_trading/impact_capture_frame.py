"""Bounded exact-wire frames for Round 73 prospective evidence."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Sequence


IMPACT_CAPTURE_FRAME_FORMAT = "round-073-length-prefixed-utf8-receipts-v1"
IMPACT_CAPTURE_FRAME_MAGIC = b"SATIAF1\x00"
IMPACT_CAPTURE_FRAME_MAX_BYTES = 64 * 1024 * 1024
IMPACT_CAPTURE_FRAME_MAX_MESSAGES = 16_384
IMPACT_CAPTURE_FRAME_MAX_RAW_BYTES = 8 * 1024 * 1024
IMPACT_CAPTURE_FRAME_MAX_CONNECTION_BYTES = 160

_RECORD_HEADER = struct.Struct("<B H Q Q Q")
_RAW_SIZE = struct.Struct("<I")
_STREAM_TO_CODE = {
    "binance_futures_public": 1,
    "binance_futures_market": 2,
    "binance_futures_rest": 3,
}
_CODE_TO_STREAM = {code: stream for stream, code in _STREAM_TO_CODE.items()}
_MAX_UNSIGNED_64 = (1 << 64) - 1


@dataclass(frozen=True)
class ImpactCaptureFrameRecord:
    """One exact public payload and its local receipt metadata."""

    stream: str
    connection_id: str
    sequence_number: int
    received_wall_ns: int
    received_monotonic_ns: int
    raw_text: str


@dataclass(frozen=True)
class LocatedImpactCaptureFrameRecord:
    """A decoded record and the exact raw-text location in its frame."""

    record: ImpactCaptureFrameRecord
    message_index: int
    raw_offset: int
    raw_size: int


@dataclass(frozen=True)
class ImpactCaptureFrameReceiptSummary:
    """Validated receipt-clock bounds and source counts without JSON parsing."""

    message_count: int
    minimum_received_wall_ns: int
    maximum_received_wall_ns: int
    minimum_received_monotonic_ns: int
    maximum_received_monotonic_ns: int
    stream_counts: dict[str, int]


def _encoded_parts(record: ImpactCaptureFrameRecord) -> tuple[int, bytes, bytes]:
    try:
        stream_code = _STREAM_TO_CODE[str(record.stream)]
    except KeyError as exc:
        raise ValueError(f"unsupported impact-capture stream: {record.stream}") from exc
    connection = str(record.connection_id).encode("utf-8", errors="strict")
    raw = str(record.raw_text).encode("utf-8", errors="strict")
    sequence = int(record.sequence_number)
    wall = int(record.received_wall_ns)
    monotonic = int(record.received_monotonic_ns)
    if not 1 <= len(connection) <= IMPACT_CAPTURE_FRAME_MAX_CONNECTION_BYTES:
        raise ValueError("impact-capture connection ID is invalid")
    if not 1 <= len(raw) <= IMPACT_CAPTURE_FRAME_MAX_RAW_BYTES:
        raise ValueError("impact-capture raw payload is empty or oversized")
    if not 0 <= sequence <= _MAX_UNSIGNED_64:
        raise ValueError("impact-capture sequence is outside unsigned 64-bit range")
    if not 1 <= wall <= _MAX_UNSIGNED_64:
        raise ValueError("impact-capture wall clock is outside unsigned 64-bit range")
    if not 1 <= monotonic <= _MAX_UNSIGNED_64:
        raise ValueError(
            "impact-capture monotonic clock is outside unsigned 64-bit range"
        )
    return stream_code, connection, raw


def impact_capture_frame_record_size(record: ImpactCaptureFrameRecord) -> int:
    """Return the exact encoded size of one validated record."""

    _stream_code, connection, raw = _encoded_parts(record)
    return _RECORD_HEADER.size + len(connection) + _RAW_SIZE.size + len(raw)


def encode_impact_capture_frame(
    records: Sequence[ImpactCaptureFrameRecord],
) -> tuple[bytes, tuple[LocatedImpactCaptureFrameRecord, ...]]:
    """Encode one bounded frame and retain exact payload offsets."""

    if not 1 <= len(records) <= IMPACT_CAPTURE_FRAME_MAX_MESSAGES:
        raise ValueError("impact capture frame has an invalid message count")
    frame = bytearray(IMPACT_CAPTURE_FRAME_MAGIC)
    located: list[LocatedImpactCaptureFrameRecord] = []
    for message_index, record in enumerate(records):
        stream_code, connection, raw = _encoded_parts(record)
        frame.extend(
            _RECORD_HEADER.pack(
                stream_code,
                len(connection),
                int(record.sequence_number),
                int(record.received_wall_ns),
                int(record.received_monotonic_ns),
            )
        )
        frame.extend(connection)
        frame.extend(_RAW_SIZE.pack(len(raw)))
        raw_offset = len(frame)
        frame.extend(raw)
        if len(frame) > IMPACT_CAPTURE_FRAME_MAX_BYTES:
            raise ValueError("impact capture frame exceeded its bounded size")
        located.append(
            LocatedImpactCaptureFrameRecord(
                record=record,
                message_index=message_index,
                raw_offset=raw_offset,
                raw_size=len(raw),
            )
        )
    return bytes(frame), tuple(located)


def _validated_envelope(frame: bytes, expected_message_count: int) -> tuple[bytes, int]:
    payload = bytes(frame)
    expected = int(expected_message_count)
    if (
        not 1 <= expected <= IMPACT_CAPTURE_FRAME_MAX_MESSAGES
        or not len(IMPACT_CAPTURE_FRAME_MAGIC)
        < len(payload)
        <= IMPACT_CAPTURE_FRAME_MAX_BYTES
        or not payload.startswith(IMPACT_CAPTURE_FRAME_MAGIC)
    ):
        raise ValueError("impact capture frame envelope is invalid")
    return payload, expected


def decode_impact_capture_frame(
    frame: bytes,
    *,
    expected_message_count: int,
) -> tuple[LocatedImpactCaptureFrameRecord, ...]:
    """Fully boundary-check and decode an untrusted uncompressed frame."""

    payload, expected = _validated_envelope(frame, expected_message_count)
    offset = len(IMPACT_CAPTURE_FRAME_MAGIC)
    decoded: list[LocatedImpactCaptureFrameRecord] = []
    for message_index in range(expected):
        header_end = offset + _RECORD_HEADER.size
        if header_end > len(payload):
            raise ValueError("impact capture frame record header is truncated")
        stream_code, connection_size, sequence, wall, monotonic = (
            _RECORD_HEADER.unpack_from(payload, offset)
        )
        stream = _CODE_TO_STREAM.get(stream_code)
        connection_end = header_end + int(connection_size)
        if (
            stream is None
            or not 1
            <= int(connection_size)
            <= IMPACT_CAPTURE_FRAME_MAX_CONNECTION_BYTES
            or int(wall) < 1
            or int(monotonic) < 1
            or connection_end + _RAW_SIZE.size > len(payload)
        ):
            raise ValueError("impact capture frame record metadata is invalid")
        try:
            connection_id = payload[header_end:connection_end].decode(
                "utf-8", errors="strict"
            )
        except UnicodeDecodeError as exc:
            raise ValueError("impact capture frame connection ID is not UTF-8") from exc
        raw_size = int(_RAW_SIZE.unpack_from(payload, connection_end)[0])
        raw_offset = connection_end + _RAW_SIZE.size
        raw_end = raw_offset + raw_size
        if not 1 <= raw_size <= IMPACT_CAPTURE_FRAME_MAX_RAW_BYTES or raw_end > len(
            payload
        ):
            raise ValueError("impact capture frame raw payload boundary is invalid")
        try:
            raw_text = payload[raw_offset:raw_end].decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("impact capture frame raw payload is not UTF-8") from exc
        decoded.append(
            LocatedImpactCaptureFrameRecord(
                record=ImpactCaptureFrameRecord(
                    stream=stream,
                    connection_id=connection_id,
                    sequence_number=int(sequence),
                    received_wall_ns=int(wall),
                    received_monotonic_ns=int(monotonic),
                    raw_text=raw_text,
                ),
                message_index=message_index,
                raw_offset=raw_offset,
                raw_size=raw_size,
            )
        )
        offset = raw_end
    if offset != len(payload):
        raise ValueError("impact capture frame contains trailing or uncounted bytes")
    return tuple(decoded)


def scan_impact_capture_frame_receipts(
    frame: bytes,
    *,
    expected_message_count: int,
) -> ImpactCaptureFrameReceiptSummary:
    """Validate frame boundaries and scan receipt metadata without decoding JSON."""

    payload, expected = _validated_envelope(frame, expected_message_count)
    offset = len(IMPACT_CAPTURE_FRAME_MAGIC)
    minimum_wall = _MAX_UNSIGNED_64
    maximum_wall = 0
    minimum_monotonic = _MAX_UNSIGNED_64
    maximum_monotonic = 0
    stream_counts: dict[str, int] = {}
    for _message_index in range(expected):
        header_end = offset + _RECORD_HEADER.size
        if header_end > len(payload):
            raise ValueError("impact capture frame record header is truncated")
        stream_code, connection_size, _sequence, wall, monotonic = (
            _RECORD_HEADER.unpack_from(payload, offset)
        )
        stream = _CODE_TO_STREAM.get(stream_code)
        connection_end = header_end + int(connection_size)
        if (
            stream is None
            or not 1
            <= int(connection_size)
            <= IMPACT_CAPTURE_FRAME_MAX_CONNECTION_BYTES
            or int(wall) < 1
            or int(monotonic) < 1
            or connection_end + _RAW_SIZE.size > len(payload)
        ):
            raise ValueError("impact capture frame record metadata is invalid")
        raw_size = int(_RAW_SIZE.unpack_from(payload, connection_end)[0])
        raw_end = connection_end + _RAW_SIZE.size + raw_size
        if not 1 <= raw_size <= IMPACT_CAPTURE_FRAME_MAX_RAW_BYTES or raw_end > len(
            payload
        ):
            raise ValueError("impact capture frame raw payload boundary is invalid")
        offset = raw_end
        minimum_wall = min(minimum_wall, int(wall))
        maximum_wall = max(maximum_wall, int(wall))
        minimum_monotonic = min(minimum_monotonic, int(monotonic))
        maximum_monotonic = max(maximum_monotonic, int(monotonic))
        stream_counts[stream] = stream_counts.get(stream, 0) + 1
    if offset != len(payload):
        raise ValueError("impact capture frame contains trailing or uncounted bytes")
    return ImpactCaptureFrameReceiptSummary(
        message_count=expected,
        minimum_received_wall_ns=minimum_wall,
        maximum_received_wall_ns=maximum_wall,
        minimum_received_monotonic_ns=minimum_monotonic,
        maximum_received_monotonic_ns=maximum_monotonic,
        stream_counts=dict(sorted(stream_counts.items())),
    )


__all__ = [
    "IMPACT_CAPTURE_FRAME_FORMAT",
    "IMPACT_CAPTURE_FRAME_MAGIC",
    "IMPACT_CAPTURE_FRAME_MAX_BYTES",
    "IMPACT_CAPTURE_FRAME_MAX_MESSAGES",
    "ImpactCaptureFrameReceiptSummary",
    "ImpactCaptureFrameRecord",
    "LocatedImpactCaptureFrameRecord",
    "decode_impact_capture_frame",
    "encode_impact_capture_frame",
    "impact_capture_frame_record_size",
    "scan_impact_capture_frame_receipts",
]
