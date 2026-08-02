"""In-memory public-feed coordination for independent Round 21 inference."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from threading import Lock

from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_capture_frame import CaptureFrameRecord
from .polymarket_redundant_union import (
    PolymarketClobLaneReceipt,
    PolymarketRedundantUnionBuilder,
    PolymarketUnionEvent,
)
from .polymarket_round21_binance_features import (
    Round21IndependentBinanceFeatureEngine,
)
from .polymarket_round21_core_features import (
    Round21CoreFeatureEngine,
    join_round21_causal_features,
)
from .polymarket_round21_dataset import (
    POLYMARKET_ROUND21_DECISION_CADENCE_MS,
)
from .polymarket_round21_prospective import (
    Round21ProspectivePrediction,
    Round21ProspectiveScorer,
)


POLYMARKET_ROUND21_LIVE_FEATURE_SCHEMA_VERSION = (
    "polymarket-round21-live-feature-coordinator-v1"
)
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONNECTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,159}$")
_STREAMS = frozenset(
    ("clob_market", "polymarket_rtds", "binance_spot", "binance_futures")
)
_CORE_STREAMS = frozenset(("clob_market", "polymarket_rtds"))
_OPTIONAL_MARKETS = {
    "binance_spot": "spot",
    "binance_futures": "usdm",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class Round21PublicSourceGap:
    stream: str
    connection_id: str
    observed_wall_ms: int
    observed_monotonic_ns: int
    last_sequence_number: int
    reason: str
    gap_sha256: str = ""

    def __post_init__(self) -> None:
        stream = str(self.stream or "").strip()
        connection = str(self.connection_id or "").strip().lower()
        wall = int(self.observed_wall_ms)
        monotonic = int(self.observed_monotonic_ns)
        sequence = int(self.last_sequence_number)
        reason = str(self.reason or "").strip()
        if (
            stream not in _STREAMS
            or not connection
            or len(connection) > 160
            or min(wall, monotonic) <= 0
            or sequence < 0
            or not reason
            or len(reason) > 240
        ):
            raise ValueError("Round 21 public source gap is invalid")
        body = {
            "schema_version": POLYMARKET_ROUND21_LIVE_FEATURE_SCHEMA_VERSION,
            "stream": stream,
            "connection_id": connection,
            "observed_wall_ms": wall,
            "observed_monotonic_ns": monotonic,
            "last_sequence_number": sequence,
            "reason": reason,
            "credentials_used": False,
            "trading_authority": False,
        }
        actual = _canonical_sha256(body)
        claimed = str(self.gap_sha256 or "").strip().lower()
        if claimed and claimed != actual:
            raise ValueError("Round 21 public source gap hash differs")
        object.__setattr__(self, "stream", stream)
        object.__setattr__(self, "connection_id", connection)
        object.__setattr__(self, "observed_wall_ms", wall)
        object.__setattr__(self, "observed_monotonic_ns", monotonic)
        object.__setattr__(self, "last_sequence_number", sequence)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "gap_sha256", actual)


@dataclass(frozen=True, slots=True)
class Round21CoordinatedPrediction:
    status: str
    reasons: tuple[str, ...]
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    observed_at_ms: int
    core_source_healthy: bool
    optional_source_healthy: bool
    core_gap_sha256: tuple[str, ...]
    prediction: Round21ProspectivePrediction | None
    decision_sha256: str
    target_accessed: bool = False
    credentials_used: bool = False
    account_connected: bool = False
    binance_execution_connected: bool = False
    grants_execution_authority: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    @classmethod
    def create(
        cls,
        *,
        status: str,
        reasons: tuple[str, ...],
        market: PolymarketFiveMinuteMarket,
        decision_time_ms: int,
        observed_at_ms: int,
        core_source_healthy: bool,
        optional_source_healthy: bool,
        core_gap_sha256: tuple[str, ...],
        prediction: Round21ProspectivePrediction | None,
    ) -> Round21CoordinatedPrediction:
        selected_status = str(status or "").strip()
        selected_reasons = tuple(str(value or "").strip() for value in reasons)
        decision = int(decision_time_ms)
        observed = int(observed_at_ms)
        gaps = tuple(str(value or "").strip().lower() for value in core_gap_sha256)
        if prediction is not None and not isinstance(
            prediction,
            Round21ProspectivePrediction,
        ):
            raise TypeError("Round 21 coordinated prediction type differs")
        selected_prediction = (
            None if prediction is None else prediction.validated()
        )
        if (
            selected_status not in {"observed", "abstain"}
            or any(not value or len(value) > 160 for value in selected_reasons)
            or len(set(selected_reasons)) != len(selected_reasons)
            or _CONDITION_ID.fullmatch(market.condition_id) is None
            or market.asset != "BTC"
            or market.horizon_minutes != 5
            or not market.event_start_ms <= decision < market.end_ms
            or (decision - market.event_start_ms)
            % POLYMARKET_ROUND21_DECISION_CADENCE_MS
            or observed < decision
            or type(core_source_healthy) is not bool
            or type(optional_source_healthy) is not bool
            or any(_SHA256.fullmatch(value) is None for value in gaps)
            or len(set(gaps)) != len(gaps)
            or (bool(gaps) and core_source_healthy)
            or (selected_status == "observed" and not core_source_healthy)
            or (selected_status == "observed")
            != (not selected_reasons and selected_prediction is not None)
            or (selected_status == "abstain") != bool(selected_reasons)
            or (
                selected_prediction is not None
                and (
                    selected_prediction.condition_id != market.condition_id
                    or selected_prediction.event_start_ms != market.event_start_ms
                    or selected_prediction.decision_time_ms != decision
                    or selected_prediction.status != selected_status
                    or (
                        selected_prediction.status == "abstain"
                        and selected_reasons != (selected_prediction.reason,)
                    )
                )
            )
        ):
            raise ValueError("Round 21 coordinated prediction is invalid")
        provisional = cls(
            status=selected_status,
            reasons=selected_reasons,
            condition_id=market.condition_id,
            event_start_ms=market.event_start_ms,
            decision_time_ms=decision,
            observed_at_ms=observed,
            core_source_healthy=core_source_healthy,
            optional_source_healthy=optional_source_healthy,
            core_gap_sha256=gaps,
            prediction=selected_prediction,
            decision_sha256="",
        )
        return replace(
            provisional,
            decision_sha256=_canonical_sha256(provisional.identity_payload()),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_LIVE_FEATURE_SCHEMA_VERSION,
            "status": self.status,
            "reasons": list(self.reasons),
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "decision_time_ms": self.decision_time_ms,
            "observed_at_ms": self.observed_at_ms,
            "core_source_healthy": self.core_source_healthy,
            "optional_source_healthy": self.optional_source_healthy,
            "core_gap_sha256": list(self.core_gap_sha256),
            "prospective_prediction_sha256": (
                None if self.prediction is None else self.prediction.prediction_sha256
            ),
            "target_accessed": False,
            "credentials_used": False,
            "account_connected": False,
            "binance_execution_connected": False,
            "grants_execution_authority": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "prediction": (
                None if self.prediction is None else self.prediction.asdict()
            ),
            "decision_sha256": self.decision_sha256,
        }

    def validated(self) -> Round21CoordinatedPrediction:
        false_fields = (
            self.target_accessed,
            self.credentials_used,
            self.account_connected,
            self.binance_execution_connected,
            self.grants_execution_authority,
            self.paper_trading_authority,
            self.live_trading_authority,
        )
        if (
            self.status not in {"observed", "abstain"}
            or any(not value or len(value) > 160 for value in self.reasons)
            or len(set(self.reasons)) != len(self.reasons)
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or type(self.event_start_ms) is not int
            or type(self.decision_time_ms) is not int
            or type(self.observed_at_ms) is not int
            or self.event_start_ms <= 0
            or self.event_start_ms % 300_000
            or not self.event_start_ms
            <= self.decision_time_ms
            < self.event_start_ms + 300_000
            or (self.decision_time_ms - self.event_start_ms)
            % POLYMARKET_ROUND21_DECISION_CADENCE_MS
            or self.observed_at_ms < self.decision_time_ms
            or type(self.core_source_healthy) is not bool
            or type(self.optional_source_healthy) is not bool
            or any(_SHA256.fullmatch(value) is None for value in self.core_gap_sha256)
            or len(set(self.core_gap_sha256)) != len(self.core_gap_sha256)
            or (bool(self.core_gap_sha256) and self.core_source_healthy)
            or (self.status == "observed" and not self.core_source_healthy)
            or _SHA256.fullmatch(self.decision_sha256) is None
            or any(type(value) is not bool for value in false_fields)
            or any(false_fields)
            or (self.status == "observed")
            != (not self.reasons and self.prediction is not None)
            or (self.status == "abstain") != bool(self.reasons)
            or (
                self.prediction is not None
                and (
                    self.prediction.validated() != self.prediction
                    or self.prediction.condition_id != self.condition_id
                    or self.prediction.event_start_ms != self.event_start_ms
                    or self.prediction.decision_time_ms != self.decision_time_ms
                    or self.prediction.status != self.status
                    or (
                        self.prediction.status == "abstain"
                        and self.reasons != (self.prediction.reason,)
                    )
                )
            )
            or self.decision_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 coordinated prediction differs")
        return self


class Round21LiveFeatureCoordinator:
    """Serialize independent public feeds into one fail-closed causal scorer."""

    credentials_used = False
    account_connected = False
    binance_execution_connected = False
    grants_execution_authority = False
    trading_authority = False

    def __init__(
        self,
        *,
        market: PolymarketFiveMinuteMarket,
        scorer: Round21ProspectiveScorer,
        pairing_window_ms: int = 2_000,
    ) -> None:
        if not isinstance(market, PolymarketFiveMinuteMarket):
            raise TypeError("Round 21 live market type differs")
        if market.asset != "BTC" or market.horizon_minutes != 5:
            raise ValueError("Round 21 live market must be BTC five-minute")
        if not isinstance(scorer, Round21ProspectiveScorer):
            raise TypeError("Round 21 live scorer type differs")
        self.market = market
        self.scorer = scorer
        self._core = Round21CoreFeatureEngine(
            condition_id=market.condition_id,
            up_token_id=market.up_token_id,
            down_token_id=market.down_token_id,
            event_start_ms=market.event_start_ms,
        )
        self._optional = Round21IndependentBinanceFeatureEngine()
        self._union = PolymarketRedundantUnionBuilder(
            pairing_window_ms=pairing_window_ms
        )
        self._chainlink_connection = ""
        self._optional_connections: dict[str, str] = {}
        self._core_gaps: list[Round21PublicSourceGap] = []
        self._last_core_monotonic_ns = 0
        self._last_decision_time_ms = 0
        self._last_decision: Round21CoordinatedPrediction | None = None
        self._lock = Lock()

    def _assert_core_order(self, received_monotonic_ns: int) -> None:
        received = int(received_monotonic_ns)
        if received <= 0 or received < self._last_core_monotonic_ns:
            raise ValueError("Round 21 public-feed receipt order regressed")
        self._last_core_monotonic_ns = received

    def _route_union(self, events: tuple[PolymarketUnionEvent, ...]) -> None:
        for event in events:
            self._core.ingest_union_event(event)

    def ingest_clob_receipt(self, receipt: PolymarketClobLaneReceipt) -> None:
        selected = receipt.validated()
        if selected.connection_id.partition(":")[0] != selected.lane_id:
            raise ValueError("Round 21 CLOB lane and connection differ")
        with self._lock:
            self._assert_core_order(selected.received_monotonic_ns)
            self._route_union(self._union.add(selected))

    def ingest_chainlink_record(self, record: CaptureFrameRecord) -> None:
        if not isinstance(record, CaptureFrameRecord):
            raise TypeError("Round 21 Chainlink record type differs")
        connection = str(record.connection_id or "").strip().lower()
        if (
            record.stream != "polymarket_rtds"
            or _CONNECTION_ID.fullmatch(connection) is None
            or int(record.sequence_number) <= 0
            or int(record.received_wall_ms) <= 0
            or int(record.received_monotonic_ns) <= 0
        ):
            raise ValueError("Round 21 Chainlink record metadata differs")
        with self._lock:
            self._assert_core_order(record.received_monotonic_ns)
            self._route_union(self._union.advance(record.received_monotonic_ns))
            if connection != self._chainlink_connection:
                self._core.start_chainlink_epoch(
                    connection,
                    first_sequence_number=record.sequence_number,
                )
                self._chainlink_connection = connection
            self._core.ingest_chainlink_record(record)

    def ingest_optional_binance_record(self, record: CaptureFrameRecord) -> None:
        if not isinstance(record, CaptureFrameRecord):
            raise TypeError("Round 21 optional Binance record type differs")
        market = _OPTIONAL_MARKETS.get(record.stream)
        if market is None:
            raise ValueError("Round 21 optional Binance stream differs")
        connection = str(record.connection_id or "").strip().lower()
        with self._lock:
            if self._optional_connections.get(market) != connection:
                self._optional.reset_market(market, connection)
                self._optional_connections[market] = connection
            self._optional.ingest_record(record)

    def record_gap(self, gap: Round21PublicSourceGap) -> None:
        if not isinstance(gap, Round21PublicSourceGap):
            raise TypeError("Round 21 public source gap type differs")
        with self._lock:
            if gap.stream in _CORE_STREAMS:
                self._assert_core_order(gap.observed_monotonic_ns)
                if gap.gap_sha256 not in {
                    item.gap_sha256 for item in self._core_gaps
                }:
                    self._core_gaps.append(gap)
            else:
                self._optional = Round21IndependentBinanceFeatureEngine()
                self._optional_connections.clear()

    def evaluate(
        self,
        *,
        decision_time_ms: int,
        observed_at_ms: int,
        observed_monotonic_ns: int,
    ) -> Round21CoordinatedPrediction:
        decision = int(decision_time_ms)
        observed = int(observed_at_ms)
        monotonic = int(observed_monotonic_ns)
        if (
            not self.market.event_start_ms <= decision < self.market.end_ms
            or (decision - self.market.event_start_ms)
            % POLYMARKET_ROUND21_DECISION_CADENCE_MS
            or observed < decision
        ):
            raise ValueError("Round 21 coordinated decision chronology is invalid")
        with self._lock:
            if decision == self._last_decision_time_ms:
                if self._last_decision is None:
                    raise RuntimeError("Round 21 coordinated decision cache is missing")
                return self._last_decision
            if decision < self._last_decision_time_ms:
                raise ValueError("Round 21 coordinated decision time regressed")
            self._assert_core_order(monotonic)
            self._route_union(self._union.advance(monotonic))
            prediction: Round21ProspectivePrediction | None = None
            optional = self._optional.build(decision)
            if self._core_gaps:
                reasons = tuple(
                    dict.fromkeys(f"core_source_gap:{gap.stream}" for gap in self._core_gaps)
                )
                status = "abstain"
                core_healthy = False
            else:
                core = self._core.build(decision)
                core_healthy = core.available
                if not core.available:
                    reasons = tuple(dict.fromkeys(core.reasons))
                    status = "abstain"
                else:
                    row = join_round21_causal_features(core, optional)
                    prediction = self.scorer.evaluate(row, observed_at_ms=observed)
                    status = prediction.status
                    reasons = (() if status == "observed" else (prediction.reason,))
            optional_healthy = (
                optional.spot_available
                if self.scorer.population_layer == "core_spot"
                else (
                    optional.usdm_available
                    if self.scorer.population_layer == "core_spot_usdm"
                    else True
                )
            )
            result = Round21CoordinatedPrediction.create(
                status=status,
                reasons=reasons,
                market=self.market,
                decision_time_ms=decision,
                observed_at_ms=observed,
                core_source_healthy=core_healthy,
                optional_source_healthy=optional_healthy,
                core_gap_sha256=tuple(gap.gap_sha256 for gap in self._core_gaps),
                prediction=prediction,
            ).validated()
            self._last_decision_time_ms = decision
            self._last_decision = result
            return result


credentials_used = False
account_connected = False
binance_execution_connected = False
grants_execution_authority = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_LIVE_FEATURE_SCHEMA_VERSION",
    "Round21CoordinatedPrediction",
    "Round21LiveFeatureCoordinator",
    "Round21PublicSourceGap",
]
