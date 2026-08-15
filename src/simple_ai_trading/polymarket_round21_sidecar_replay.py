"""Causal replay for the independent Round 21 public Binance sidecar."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from .polymarket_capture_frame import CaptureFrameRecord
from .polymarket_recorder import PolymarketEvidenceStore, RawStreamMessage, StreamGap
from .polymarket_round21_binance_features import (
    Round21IndependentBinanceFeatureEngine,
    Round21OptionalBinanceFeatures,
)
from .polymarket_round21_sidecar_terminal import (
    validate_round21_sidecar_terminal_manifest,
)


POLYMARKET_ROUND21_SIDECAR_REPLAY_SCHEMA_VERSION = (
    "polymarket-round21-binance-sidecar-replay-v1"
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STREAM_MARKET = {
    "binance_spot": "spot",
    "binance_futures": "usdm",
}


def _canonical_json(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _receipt_chain(previous: str, message: RawStreamMessage) -> str:
    identity = {
        "stream": message.stream,
        "connection_id": message.connection_id,
        "sequence_number": message.sequence_number,
        "received_wall_ms": message.received_wall_ms,
        "received_monotonic_ns": message.received_monotonic_ns,
        "raw_sha256": hashlib.sha256(message.raw_text.encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(
        f"{previous}:{_canonical_json(identity)}".encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Round21SidecarReplay:
    terminal_manifest_sha256: str
    eligible_run_ids: tuple[str, ...]
    decision_times_ms: tuple[int, ...]
    features: tuple[Round21OptionalBinanceFeatures, ...]
    raw_message_count: int
    stream_counts: Mapping[str, int]
    stream_gap_count: int
    receipt_chain_sha256: str
    development_feature_eligible: bool = True
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def validated(self) -> Round21SidecarReplay:
        decisions = tuple(int(value) for value in self.decision_times_ms)
        features = tuple(self.features)
        counts = dict(self.stream_counts)
        if (
            _SHA256.fullmatch(self.terminal_manifest_sha256) is None
            or not self.eligible_run_ids
            or len(set(self.eligible_run_ids)) != len(self.eligible_run_ids)
            or not decisions
            or tuple(sorted(set(decisions))) != decisions
            or len(features) != len(decisions)
            or tuple(item.decision_time_ms for item in features) != decisions
            or self.raw_message_count <= 0
            or set(counts) != set(_STREAM_MARKET)
            or any(type(count) is not int or count <= 0 for count in counts.values())
            or sum(counts.values()) != self.raw_message_count
            or type(self.stream_gap_count) is not int
            or self.stream_gap_count < 0
            or _SHA256.fullmatch(self.receipt_chain_sha256) is None
            or self.receipt_chain_sha256 == _EMPTY_SHA256
            or self.development_feature_eligible is not True
            or any(
                value
                for value in (
                    self.profitability_claim,
                    self.paper_trading_authority,
                    self.live_trading_authority,
                )
            )
        ):
            raise ValueError("Round 21 sidecar replay differs")
        return self


class _CausalReplayObserver:
    def __init__(self, decision_times_ms: Sequence[int]) -> None:
        self.decisions = tuple(int(value) for value in decision_times_ms)
        self.engine = Round21IndependentBinanceFeatureEngine()
        self.features: list[Round21OptionalBinanceFeatures] = []
        self._decision_index = 0
        self._connections: dict[str, str] = {}
        self._gaps: tuple[StreamGap, ...] = ()
        self._gap_index = 0
        self._run_id = ""
        self._segment_start_ms = 0
        self._segment_end_ms = 0
        self._last_wall_ms = 0
        self._last_monotonic_ns = 0

    def _invalidate(self, stream: str) -> None:
        market = _STREAM_MARKET.get(stream)
        if market is None:
            raise ValueError("Round 21 sidecar gap stream differs")
        self.engine.invalidate_market(market)
        self._connections.pop(stream, None)

    def _invalidate_all(self) -> None:
        for stream in _STREAM_MARKET:
            self._invalidate(stream)

    def _emit_before(self, wall_ms: int) -> None:
        while (
            self._decision_index < len(self.decisions)
            and self.decisions[self._decision_index] < wall_ms
        ):
            decision = self.decisions[self._decision_index]
            self.features.append(self.engine.build(decision))
            self._decision_index += 1

    def _emit_through(self, wall_ms: int) -> None:
        self._emit_before(int(wall_ms) + 1)

    def _apply_gap(self, gap: StreamGap) -> None:
        selected = gap.validated()
        if (
            selected.opened_at_ms < self._last_wall_ms
            or not self._segment_start_ms
            <= selected.opened_at_ms
            <= self._segment_end_ms
        ):
            raise ValueError("Round 21 sidecar gap chronology differs")
        self._emit_before(selected.opened_at_ms)
        self._invalidate(selected.stream)

    def start_run(
        self,
        segment: Mapping[str, object],
        gaps: Sequence[StreamGap],
    ) -> None:
        if self._run_id:
            raise RuntimeError("Round 21 sidecar replay run is already open")
        self._run_id = str(segment["run_id"])
        self._segment_start_ms = int(segment["started_at_ms"])
        self._segment_end_ms = int(segment["ended_at_ms"])
        self._last_wall_ms = self._segment_start_ms
        self._last_monotonic_ns = 0
        self._gaps = tuple(
            sorted(
                (gap.validated() for gap in gaps),
                key=lambda item: (item.opened_at_ms, item.stream, item.connection_id),
            )
        )
        self._gap_index = 0
        self._invalidate_all()
        self._emit_before(self._segment_start_ms)

    def observe_message(self, message: RawStreamMessage) -> RawStreamMessage:
        if not self._run_id:
            raise RuntimeError("Round 21 sidecar replay run is unavailable")
        selected = message.validated()
        if (
            selected.stream not in _STREAM_MARKET
            or not self._segment_start_ms
            <= selected.received_wall_ms
            <= self._segment_end_ms
            or selected.received_wall_ms < self._last_wall_ms
            or selected.received_monotonic_ns < self._last_monotonic_ns
        ):
            raise ValueError("Round 21 sidecar receipt chronology differs")
        while (
            self._gap_index < len(self._gaps)
            and self._gaps[self._gap_index].opened_at_ms <= selected.received_wall_ms
        ):
            self._apply_gap(self._gaps[self._gap_index])
            self._gap_index += 1
        self._emit_before(selected.received_wall_ms)
        connection = selected.connection_id.strip().lower()
        if self._connections.get(selected.stream) != connection:
            market = _STREAM_MARKET[selected.stream]
            self.engine.reset_market(market, connection)
            self._connections[selected.stream] = connection
        self.engine.ingest_record(
            CaptureFrameRecord(
                stream=selected.stream,
                connection_id=selected.connection_id,
                sequence_number=selected.sequence_number,
                received_wall_ms=selected.received_wall_ms,
                received_monotonic_ns=selected.received_monotonic_ns,
                raw_text=selected.raw_text,
            )
        )
        self._last_wall_ms = selected.received_wall_ms
        self._last_monotonic_ns = selected.received_monotonic_ns
        return selected

    def finish_run(self) -> None:
        if not self._run_id:
            raise RuntimeError("Round 21 sidecar replay run is unavailable")
        while self._gap_index < len(self._gaps):
            self._apply_gap(self._gaps[self._gap_index])
            self._gap_index += 1
        if self._last_wall_ms > self._segment_end_ms:
            raise ValueError("Round 21 sidecar segment end differs")
        self._emit_before(self._segment_end_ms)
        self._invalidate_all()
        self._emit_through(self._segment_end_ms)
        self._gaps = ()
        self._gap_index = 0
        self._run_id = ""
        self._segment_start_ms = 0
        self._segment_end_ms = 0
        self._last_wall_ms = 0
        self._last_monotonic_ns = 0

    def finish_campaign(
        self, campaign_end_ms: int
    ) -> tuple[Round21OptionalBinanceFeatures, ...]:
        if self._run_id:
            raise RuntimeError("Round 21 sidecar replay run remains open")
        self._invalidate_all()
        self._emit_before(campaign_end_ms)
        if self._decision_index != len(self.decisions):
            raise ValueError("Round 21 sidecar replay decisions exceed the campaign")
        return tuple(self.features)


def _validate_database_run(
    store: PolymarketEvidenceStore,
    segment: Mapping[str, object],
) -> None:
    run_id = str(segment["run_id"])
    row = (
        store.connect()
        .execute(
            """
        SELECT status, started_at_ms, ended_at_ms, report_sha256
        FROM polymarket_recorder_run WHERE run_id = ?
        """,
            [run_id],
        )
        .fetchone()
    )
    manifest_row = (
        store.connect()
        .execute(
            """
        SELECT manifest_sha256 FROM polymarket_preregistration_manifest
        WHERE run_id = ?
        """,
            [run_id],
        )
        .fetchone()
    )
    if (
        row is None
        or manifest_row is None
        or tuple(row)
        != (
            segment["status"],
            segment["started_at_ms"],
            segment["ended_at_ms"],
            segment["recorder_report_sha256"],
        )
        or manifest_row[0] != segment["preregistration_manifest_sha256"]
    ):
        raise ValueError("Round 21 sidecar terminal database identity differs")


def replay_round21_optional_binance_features(
    *,
    source_database: str | Path,
    terminal_manifest: Mapping[str, object],
    decision_times_ms: Sequence[int],
) -> Round21SidecarReplay:
    """Replay public Binance receipts once into target-blind decision features."""

    terminal = validate_round21_sidecar_terminal_manifest(terminal_manifest)
    decisions = tuple(int(value) for value in decision_times_ms)
    if (
        not decisions
        or tuple(sorted(set(decisions))) != decisions
        or decisions[0] < int(terminal["campaign_start_ms"])
        or decisions[-1] >= int(terminal["campaign_end_ms"])
    ):
        raise ValueError("Round 21 sidecar replay decision times differ")
    database = Path(source_database).resolve()
    wal = Path(f"{database}.wal")
    if database.is_symlink() or not database.is_file() or wal.exists():
        raise RuntimeError(
            "Round 21 sidecar replay requires a terminal WAL-free database"
        )

    observer = _CausalReplayObserver(decisions)
    counts: defaultdict[str, int] = defaultdict(int)
    gap_count = 0
    receipt_count = 0
    receipt_chain = _EMPTY_SHA256
    eligible = tuple(
        segment
        for segment in terminal["segments"]
        if segment["eligible_for_optional_feature_replay"]
    )
    with PolymarketEvidenceStore(
        database,
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        for segment in eligible:
            _validate_database_run(store, segment)
            run_id = str(segment["run_id"])
            gaps = tuple(store.iter_terminal_stream_gaps(run_id))
            if len(gaps) != segment["stream_gap_count"]:
                raise ValueError("Round 21 sidecar terminal gap accounting differs")
            observer.start_run(segment, gaps)
            run_counts: defaultdict[str, int] = defaultdict(int)
            run_receipts = 0
            for message in store.iter_terminal_capture_messages(
                run_id,
                streams=("binance_spot", "binance_futures"),
            ):
                selected = observer.observe_message(message)
                receipt_chain = _receipt_chain(receipt_chain, selected)
                run_counts[selected.stream] += 1
                counts[selected.stream] += 1
                run_receipts += 1
                receipt_count += 1
            observer.finish_run()
            if (
                run_receipts != segment["raw_message_count"]
                or dict(sorted(run_counts.items())) != segment["stream_counts"]
            ):
                raise ValueError("Round 21 sidecar terminal receipt accounting differs")
            gap_count += len(gaps)
    features = observer.finish_campaign(int(terminal["campaign_end_ms"]))
    result = Round21SidecarReplay(
        terminal_manifest_sha256=str(terminal["manifest_sha256"]),
        eligible_run_ids=tuple(str(segment["run_id"]) for segment in eligible),
        decision_times_ms=decisions,
        features=features,
        raw_message_count=receipt_count,
        stream_counts=dict(sorted(counts.items())),
        stream_gap_count=gap_count,
        receipt_chain_sha256=receipt_chain,
    )
    return result.validated()


__all__ = [
    "POLYMARKET_ROUND21_SIDECAR_REPLAY_SCHEMA_VERSION",
    "Round21SidecarReplay",
    "replay_round21_optional_binance_features",
]
