"""Snapshot-gated HftBacktest input; physical receipts remain in the raw source."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from fractions import Fraction
import gzip
from importlib.metadata import version
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

from .microstructure_validation import (
    depth_update_ids,
    nonnegative_native_integer,
    positive_native_integer,
    validate_price_levels,
)


NORMALIZATION_SCHEMA_VERSION = "hftbacktest-timed-snapshot-v1"
_MAX_NS = (1 << 63) - 1


@dataclass(frozen=True)
class TimedNormalization:
    schema_version: str
    engine_version: str
    snapshot_last_update_id: int
    snapshot_transaction_time_ms: int
    snapshot_received_at_ns: int
    clock_offset_ns: int
    snapshot_available_at_ns: int
    source_messages: int
    buffered_messages: int
    input_event_rows: int
    normalized_rows: int = 0
    local_timestamp_semantics: str = (
        "snapshot-gated receipt plus declared clock-offset estimate"
    )
    clock_qualification: str = (
        "declared input units; not independent clock verification"
    )
    initial_snapshot_rows: int = 0

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _clock_ns(value: int, field: str) -> int:
    if not 0 < value <= _MAX_NS:
        raise ValueError(f"{field} is outside positive int64 nanoseconds")
    return value


def prepare_timed_input(
    source_path: Path,
    output_path: Path,
    snapshot: Mapping[str, object],
    *,
    symbol: str,
    snapshot_received_at_ns: int,
    clock_offset_ms: float,
) -> TimedNormalization:
    """Seed the stream at receipt, then release buffered deltas in source order."""
    snapshot_t = positive_native_integer(snapshot.get("T"), "snapshot.T")
    snapshot_id = positive_native_integer(snapshot.get("lastUpdateId"), "lastUpdateId")
    received = positive_native_integer(snapshot_received_at_ns, "snapshot receipt")
    if type(clock_offset_ms) not in (int, float) or not math.isfinite(clock_offset_ms):
        raise ValueError("clock offset must be a finite declared estimate")
    offset_ns = math.ceil(Fraction(str(clock_offset_ms)) * 1_000_000)
    available = _clock_ns(received + offset_ns, "snapshot availability")
    snapshot_exchange = _clock_ns(snapshot_t * 1_000_000, "snapshot transaction time")
    if available < snapshot_exchange:
        raise ValueError("snapshot receipt precedes its native transaction clock")
    if not 10**18 <= available < 10**19:
        raise ValueError("HftBacktest input requires a 19-digit nanosecond timestamp")
    bids, asks = snapshot.get("bids"), snapshot.get("asks")
    bid_rows, ask_rows = validate_price_levels(bids), validate_price_levels(asks)
    if not bid_rows or not ask_rows:
        raise ValueError("timed snapshot requires both sides")
    if max(float(level[0]) for level in bids) >= min(float(level[0]) for level in asks):
        raise ValueError("crossed initial snapshot")
    messages = buffered = 0
    rows = bid_rows + ask_rows + 2
    previous_id = None
    previous_depth_t = snapshot_t
    previous_receipt = None
    with (
        gzip.open(source_path, "rt", encoding="utf-8") as source,
        gzip.open(output_path, "xt", encoding="utf-8", newline="\n") as target,
    ):
        # HftBacktest accepts an unwrapped snapshot with explicit transaction T.
        seed = {"T": snapshot_t, "bids": bids, "asks": asks}
        target.write(f"{available} {json.dumps(seed, separators=(',', ':'))}\n")
        for line in source:
            raw_received, raw_json = line.rstrip("\n").split(" ", 1)
            if not raw_received.isascii() or not raw_received.isdecimal():
                raise ValueError("raw receipt must be exact decimal nanoseconds")
            local = _clock_ns(int(raw_received) + offset_ns, "event receipt")
            if previous_receipt is not None and local < previous_receipt:
                raise ValueError(
                    "physical receipt clock regressed; no source reordering"
                )
            previous_receipt = local
            envelope = json.loads(raw_json)
            if not isinstance(envelope, dict):
                raise ValueError("invalid synchronized message envelope")
            data = envelope.get("data", envelope)
            if not isinstance(data, dict) or data.get("s") != symbol:
                raise ValueError("synchronized message symbol differs")
            stream = envelope.get("stream")
            if stream is not None and not str(stream).startswith(symbol.lower() + "@"):
                raise ValueError("stream envelope symbol differs")
            event = data.get("e")
            transaction = positive_native_integer(data.get("T"), "event.T")
            exchange = _clock_ns(transaction * 1_000_000, "event transaction time")
            if local < exchange:
                raise ValueError(
                    "raw receipt precedes its native clock; no latency repair"
                )
            release = max(local, available)
            if not 10**18 <= release < 10**19:
                raise ValueError(
                    "HftBacktest input requires a 19-digit nanosecond timestamp"
                )
            if event == "depthUpdate":
                first, final, prior = depth_update_ids(data)
                if previous_id is None:
                    if final <= snapshot_id or not (
                        first <= snapshot_id <= final or prior == snapshot_id
                    ):
                        raise ValueError("timed snapshot bridge differs")
                elif prior != previous_id:
                    raise ValueError("depth sequence gap during normalization")
                elif final <= previous_id:
                    raise ValueError("depth update ID did not advance")
                if transaction < previous_depth_t:
                    raise ValueError("native depth clock regressed after the snapshot")
                previous_id, previous_depth_t = final, transaction
                rows += validate_price_levels(data.get("b")) + validate_price_levels(
                    data.get("a")
                )
            elif previous_id is None:
                raise ValueError("message precedes the snapshot bridge")
            elif event in {"trade", "aggTrade"}:
                identity = nonnegative_native_integer(
                    data.get("a" if event == "aggTrade" else "t"), "trade ID"
                )
                if isinstance(data.get("p"), bool) or isinstance(data.get("q"), bool):
                    raise ValueError("trade economics must not be booleans")
                price, quantity = float(data.get("p", 0)), float(data.get("q", 0))
                if (
                    not all(math.isfinite(v) and v > 0 for v in (price, quantity))
                    or type(data.get("m")) is not bool
                ):
                    raise ValueError("invalid native trade economics or maker flag")
                if event == "aggTrade":
                    data = {**data, "e": "trade", "t": identity, "X": "MARKET"}
                    if stream is not None:
                        envelope["stream"] = (
                            str(stream).removesuffix("@aggTrade") + "@trade"
                        )
                if data.get("X") == "MARKET":
                    rows += 1
            elif event == "bookTicker":
                validate_price_levels(
                    [[data.get("b"), data.get("B")], [data.get("a"), data.get("A")]]
                )
                if float(data["b"]) >= float(data["a"]):
                    raise ValueError("crossed native book ticker")
                rows += 2
            else:
                raise ValueError("unsupported synchronized event")
            if "data" in envelope:
                envelope["data"] = data
            else:
                envelope = {"stream": f"{symbol.lower()}@{data['e']}", "data": data}
            target.write(f"{release} {json.dumps(envelope, separators=(',', ':'))}\n")
            buffered += local < available
            messages += 1
    if previous_id is None:
        raise ValueError("timed normalization requires a snapshot bridge")
    return TimedNormalization(
        NORMALIZATION_SCHEMA_VERSION,
        version("hftbacktest"),
        snapshot_id,
        snapshot_t,
        received,
        offset_ns,
        available,
        messages,
        buffered,
        rows,
    )


def normalize_capture(
    source_path: Path,
    normalized_path: Path,
    initial_snapshot_path: Path,
    snapshot: Mapping[str, object],
    *,
    symbol: str,
    snapshot_received_at_ns: int,
    clock_offset_ms: float,
) -> TimedNormalization:
    """Export timed data without a future time-zero book or silent clock shifts."""
    import numpy as np
    from hftbacktest import event_dtype
    from hftbacktest.data.utils import binancefutures

    if normalized_path.resolve() == initial_snapshot_path.resolve():
        raise ValueError("normalization output paths must differ")
    log = normalized_path.with_suffix(".conversion.log")
    if normalized_path.exists() or initial_snapshot_path.exists() or log.exists():
        raise FileExistsError("normalization outputs must be new")
    with TemporaryDirectory(
        prefix="hft-timed-", dir=normalized_path.parent
    ) as temporary:
        prepared = Path(temporary) / "input.jsonl.gz"
        report = prepare_timed_input(
            source_path,
            prepared,
            snapshot,
            symbol=symbol,
            snapshot_received_at_ns=snapshot_received_at_ns,
            clock_offset_ms=clock_offset_ms,
        )
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                data = binancefutures.convert(
                    str(prepared),
                    opt="t",
                    combined_stream=True,
                    buffer_size=max(10_000, report.input_event_rows + 1),
                )
        finally:
            with log.open("x", encoding="utf-8") as target:
                target.write(output.getvalue())
        if not len(data) or np.any(data["local_ts"] < report.snapshot_available_at_ns):
            raise ValueError("normalized data violates snapshot availability")
        # The standalone artifact is intentionally empty. All book state is timed
        # in the stream; no BacktestAsset.initial_snapshot call may load it early.
        with initial_snapshot_path.open("xb") as target:
            np.savez_compressed(target, data=np.empty(0, dtype=event_dtype))
        with normalized_path.open("xb") as target:
            np.savez_compressed(target, data=data)
    return TimedNormalization(**{**report.asdict(), "normalized_rows": len(data)})
