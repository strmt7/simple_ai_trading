"""Target-free Round 17 rows from one admitted, terminal capture condition."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Mapping, Sequence

from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_btc_reference import (
    PolymarketChainlinkBtcTick,
    parse_polymarket_chainlink_btc_tick,
)
from .polymarket_recorder import DecodedPublicEvent
from .polymarket_replay import PolymarketRecordedBook
from .polymarket_round14_dataset import PolymarketRound14ConditionAdmission
from .polymarket_round14_features import PolymarketRound14FeatureRow
from .polymarket_round17_features import (
    POLYMARKET_ROUND17_CONTRACT_SHA256,
    POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
    PolymarketRound17BinanceTrade,
    PolymarketRound17FeatureAccumulator,
    PolymarketRound17FeatureRow,
)


POLYMARKET_ROUND17_CONDITION_DATASET_SCHEMA_VERSION = (
    "polymarket-round17-condition-dataset-v1"
)
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONNECTION_ID = re.compile(r"^[A-Za-z0-9:._-]{1,160}$")
_MAXIMUM_LOOKBACK_MS = 120_000
_BINANCE_STREAM_MARKET = {
    "binance_spot": ("spot", "BINANCE_SPOT"),
    "binance_futures": ("perpetual", "BINANCE_USD_M_FUTURES"),
}
_BINANCE_REQUIRED_TRADE_KEYS = frozenset({"e", "E", "s", "t", "p", "q", "T", "m"})
_BINANCE_OPTIONAL_TRADE_KEYS = frozenset({"M", "a", "b", "X", "st"})


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _positive_decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive decimal")
    return parsed


def _verified_event(event: DecodedPublicEvent) -> DecodedPublicEvent:
    if not isinstance(event, DecodedPublicEvent):
        raise TypeError("Round 17 input is not a decoded public event")
    if (
        not event.run_id
        or not event.event_id
        or not event.message_id
        or _CONNECTION_ID.fullmatch(str(event.connection_id)) is None
        or _SHA256.fullmatch(str(event.event_sha256)) is None
        or _canonical_sha256(event.event) != event.event_sha256
        or int(event.sequence_number) < 0
        or int(event.sub_index) < 0
        or int(event.received_wall_ms) <= 0
        or int(event.received_monotonic_ns) <= 0
    ):
        raise ValueError("Round 17 decoded event integrity differs")
    return event


@dataclass(frozen=True, slots=True)
class PolymarketRound17ChainlinkObservation:
    run_id: str
    connection_id: str
    event_id: str
    event_sha256: str
    received_monotonic_ns: int
    tick: PolymarketChainlinkBtcTick
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if (
            not self.run_id
            or _CONNECTION_ID.fullmatch(str(self.connection_id)) is None
            or not self.event_id
            or _SHA256.fullmatch(str(self.event_sha256)) is None
            or int(self.received_monotonic_ns) <= 0
            or not isinstance(self.tick, PolymarketChainlinkBtcTick)
            or self.tick.source_payload_sha256 != self.event_sha256
            or self.trading_authority
        ):
            raise ValueError("Round 17 Chainlink observation is invalid")


def parse_round17_chainlink_event(
    event: DecodedPublicEvent,
) -> PolymarketRound17ChainlinkObservation:
    """Parse one exact Chainlink update without exposing settlement data."""

    selected = _verified_event(event)
    if (
        selected.stream != "polymarket_rtds"
        or selected.event_type != "crypto_prices_chainlink:update"
        or selected.condition_id
        or selected.asset_id
    ):
        raise ValueError("Round 17 Chainlink event identity differs")
    tick = parse_polymarket_chainlink_btc_tick(
        selected.event,
        received_at_ms=selected.received_wall_ms,
    )
    return PolymarketRound17ChainlinkObservation(
        run_id=selected.run_id,
        connection_id=selected.connection_id,
        event_id=selected.event_id,
        event_sha256=selected.event_sha256,
        received_monotonic_ns=selected.received_monotonic_ns,
        tick=tick,
    )


def parse_round17_binance_trade_event(
    event: DecodedPublicEvent,
) -> PolymarketRound17BinanceTrade | None:
    """Parse one public raw trade; ignore only the exact futures zero sentinel."""

    selected = _verified_event(event)
    market_identity = _BINANCE_STREAM_MARKET.get(selected.stream)
    if (
        market_identity is None
        or selected.event_type != "trade"
        or selected.condition_id
        or selected.asset_id
    ):
        raise ValueError("Round 17 Binance trade event identity differs")
    envelope = selected.event
    if set(envelope) != {"stream", "data"}:
        raise ValueError("Round 17 Binance combined-stream envelope drifted")
    if str(envelope["stream"] or "").strip().lower() != "btcusdt@trade":
        raise ValueError("Round 17 Binance raw-trade stream differs")
    body = envelope["data"]
    if not isinstance(body, Mapping):
        raise ValueError("Round 17 Binance raw-trade payload is not an object")
    keys = frozenset(body)
    if (
        not _BINANCE_REQUIRED_TRADE_KEYS.issubset(keys)
        or keys - _BINANCE_REQUIRED_TRADE_KEYS - _BINANCE_OPTIONAL_TRADE_KEYS
    ):
        raise ValueError("Round 17 Binance raw-trade schema drifted")
    if body["e"] != "trade" or str(body["s"] or "").strip().upper() != "BTCUSDT":
        raise ValueError("Round 17 Binance raw-trade symbol differs")
    event_time = int(body["T"])
    publisher_time = int(body["E"])
    if (
        selected.source_time_ms != event_time
        or selected.publisher_time_ms != publisher_time
    ):
        raise ValueError("Round 17 Binance indexed timestamps differ")
    price = Decimal(str(body["p"]))
    quantity = Decimal(str(body["q"]))
    if (
        selected.stream == "binance_futures"
        and price == 0
        and quantity == 0
        and body.get("X") == "NA"
        and body.get("st") == 1
    ):
        return None
    parsed_price = _positive_decimal(price, name="Binance raw-trade price")
    parsed_quantity = _positive_decimal(quantity, name="Binance raw-trade quantity")
    if type(body["m"]) is not bool:
        raise ValueError("Round 17 Binance maker flag is not boolean")
    market, source = market_identity
    return PolymarketRound17BinanceTrade(
        market=market,
        source=source,
        symbol="BTCUSDT",
        connection_id=selected.connection_id,
        event_time_ms=event_time,
        received_at_ms=selected.received_wall_ms,
        trade_id=int(body["t"]),
        price=float(parsed_price),
        quantity=float(parsed_quantity),
        buyer_is_maker=body["m"],
        source_event_sha256=selected.event_sha256,
    )


def _event_order(event: DecodedPublicEvent) -> tuple[int, int, str, int, int]:
    return (
        int(event.received_wall_ms),
        int(event.received_monotonic_ns),
        str(event.connection_id),
        int(event.sequence_number),
        int(event.sub_index),
    )


def _book_order(book: PolymarketRecordedBook) -> tuple[int, int, str, int, int]:
    return (
        int(book.received_wall_ms),
        int(book.received_monotonic_ns),
        str(book.connection_id),
        int(book.sequence_number),
        int(book.sub_index),
    )


def _single_identity(values: Sequence[str], *, name: str) -> str:
    identities = tuple(sorted(set(values)))
    if len(identities) != 1 or not identities[0]:
        raise ValueError(f"Round 17 {name} crossed a connection epoch")
    return identities[0]


@dataclass(frozen=True, slots=True)
class PolymarketRound17ConditionDataset:
    run_id: str
    condition_id: str
    event_start_ms: int
    event_end_ms: int
    admission_sha256: str
    causal_segment_sha256: str
    feature_names_sha256: str
    base_row_count: int
    chainlink_event_count: int
    spot_trade_count: int
    perpetual_trade_count: int
    up_book_count: int
    down_book_count: int
    binance_layer_eligible: bool
    rows: tuple[PolymarketRound17FeatureRow, ...]
    dataset_sha256: str
    training_authority: bool = False
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND17_CONDITION_DATASET_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "event_end_ms": self.event_end_ms,
            "admission_sha256": self.admission_sha256,
            "causal_segment_sha256": self.causal_segment_sha256,
            "feature_names_sha256": self.feature_names_sha256,
            "base_row_count": self.base_row_count,
            "chainlink_event_count": self.chainlink_event_count,
            "spot_trade_count": self.spot_trade_count,
            "perpetual_trade_count": self.perpetual_trade_count,
            "up_book_count": self.up_book_count,
            "down_book_count": self.down_book_count,
            "binance_layer_eligible": self.binance_layer_eligible,
            "row_input_sha256s": [row.input_sha256 for row in self.rows],
            "row_values_sha256s": [row.values_sha256 for row in self.rows],
            "labels_consulted": False,
            "outcomes_consulted": False,
            "resolutions_consulted": False,
            "model_scores_consulted": False,
            "training_authority": False,
            "trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "rows": [row.asdict() for row in self.rows],
            "dataset_sha256": self.dataset_sha256,
        }

    def validated(self) -> PolymarketRound17ConditionDataset:
        counts = (
            self.base_row_count,
            self.chainlink_event_count,
            self.spot_trade_count,
            self.perpetual_trade_count,
            self.up_book_count,
            self.down_book_count,
        )
        if (
            not self.run_id
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or self.event_start_ms <= 0
            or self.event_start_ms % 300_000
            or self.event_end_ms - self.event_start_ms != 300_000
            or _SHA256.fullmatch(self.admission_sha256) is None
            or _SHA256.fullmatch(self.causal_segment_sha256) is None
            or self.feature_names_sha256 != POLYMARKET_ROUND17_FEATURE_NAMES_SHA256
            or min(counts) < 0
            or self.base_row_count != len(self.rows)
            or not self.rows
            or any(
                row.condition_id != self.condition_id
                or row.admission_sha256 != self.admission_sha256
                or row.causal_segment_sha256 != self.causal_segment_sha256
                for row in self.rows
            )
            or tuple(
                sorted(
                    self.rows,
                    key=lambda row: (row.decision_time_ms, row.input_sha256),
                )
            )
            != self.rows
            or (
                self.binance_layer_eligible
                and min(self.spot_trade_count, self.perpetual_trade_count) <= 0
            )
            or self.training_authority
            or self.trading_authority
            or _SHA256.fullmatch(self.dataset_sha256) is None
            or self.dataset_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 condition dataset is invalid")
        return self


def _validate_market_and_admission(
    market: PolymarketFiveMinuteMarket,
    admission: PolymarketRound14ConditionAdmission,
) -> tuple[PolymarketFiveMinuteMarket, PolymarketRound14ConditionAdmission]:
    if not isinstance(market, PolymarketFiveMinuteMarket):
        raise TypeError("Round 17 market type differs")
    if not isinstance(admission, PolymarketRound14ConditionAdmission):
        raise TypeError("Round 17 admission type differs")
    selected_admission = admission.validated()
    if (
        market.asset != "BTC"
        or market.condition_id != selected_admission.condition_id
        or market.event_start_ms != selected_admission.event_start_ms
        or market.end_ms != selected_admission.event_end_ms
        or not selected_admission.core_eligible
    ):
        raise ValueError("Round 17 market and admission identities differ")
    return market, selected_admission


def materialize_round17_condition_rows(
    *,
    market: PolymarketFiveMinuteMarket,
    admission: PolymarketRound14ConditionAdmission,
    base_rows: Sequence[PolymarketRound14FeatureRow],
    events: Sequence[DecodedPublicEvent],
    books: Sequence[PolymarketRecordedBook],
) -> PolymarketRound17ConditionDataset:
    """Build causal rows without reading labels, resolutions, or model scores."""

    selected_market, selected_admission = _validate_market_and_admission(
        market,
        admission,
    )
    rows = tuple(base_rows)
    if (
        not rows
        or tuple(sorted(rows, key=lambda row: row.decision_time_ms)) != rows
        or len({row.decision_time_ms for row in rows}) != len(rows)
        or any(
            row.condition_id != selected_market.condition_id
            or not selected_market.event_start_ms
            <= row.decision_time_ms
            < selected_market.end_ms
            for row in rows
        )
    ):
        raise ValueError("Round 17 base rows are invalid or unordered")
    event_rows = tuple(events)
    if tuple(sorted(event_rows, key=_event_order)) != event_rows:
        raise ValueError("Round 17 public events are not in receipt order")
    book_rows = tuple(books)
    if tuple(sorted(book_rows, key=_book_order)) != book_rows:
        raise ValueError("Round 17 books are not in receipt order")
    minimum_receipt = selected_market.event_start_ms - _MAXIMUM_LOOKBACK_MS
    maximum_receipt = rows[-1].decision_time_ms

    chainlink: list[PolymarketRound17ChainlinkObservation] = []
    binance: list[PolymarketRound17BinanceTrade] = []
    for event in event_rows:
        if event.run_id != selected_admission.run_id:
            raise ValueError("Round 17 public event belongs to another run")
        if not minimum_receipt <= event.received_wall_ms <= maximum_receipt:
            continue
        if (
            event.stream == "polymarket_rtds"
            and event.event_type == "crypto_prices_chainlink:update"
        ):
            chainlink.append(parse_round17_chainlink_event(event))
        elif event.stream in _BINANCE_STREAM_MARKET and event.event_type == "trade":
            trade = parse_round17_binance_trade_event(event)
            if trade is not None and selected_admission.binance_layer_eligible:
                binance.append(trade)
    selected_books = tuple(
        book
        for book in book_rows
        if minimum_receipt <= book.received_wall_ms <= maximum_receipt
    )
    if any(
        book.run_id != selected_admission.run_id
        or book.market.condition_id != selected_market.condition_id
        or book.market.event_start_ms != selected_market.event_start_ms
        or book.market.end_ms != selected_market.end_ms
        or book.token_id not in selected_market.token_ids
        or not book.snapshot.connected
        or not book.snapshot.gap_free
        for book in selected_books
    ):
        raise ValueError("Round 17 book source identity or gap state differs")
    up_books = tuple(
        book for book in selected_books if book.token_id == selected_market.up_token_id
    )
    down_books = tuple(
        book
        for book in selected_books
        if book.token_id == selected_market.down_token_id
    )
    if not chainlink or not up_books or not down_books:
        raise ValueError("Round 17 condition lacks required causal source data")

    segment_payload: dict[str, object] = {
        "schema_version": "polymarket-round17-causal-segment-v1",
        "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
        "run_id": selected_admission.run_id,
        "condition_id": selected_market.condition_id,
        "admission_sha256": selected_admission.admission_sha256,
        "receipt_start_ms": minimum_receipt,
        "receipt_end_ms": maximum_receipt,
        "chainlink_connection_id": _single_identity(
            [item.connection_id for item in chainlink],
            name="Chainlink source",
        ),
        "clob_connection_id": _single_identity(
            [item.connection_id for item in selected_books],
            name="CLOB source",
        ),
        "clob_segment_id": _single_identity(
            [item.segment_id for item in selected_books],
            name="CLOB segment",
        ),
        "binance_layer_eligible": selected_admission.binance_layer_eligible,
    }
    if selected_admission.binance_layer_eligible:
        spot = [item for item in binance if item.market == "spot"]
        perpetual = [item for item in binance if item.market == "perpetual"]
        if not spot or not perpetual:
            raise ValueError("Round 17 condition lacks admitted Binance trades")
        segment_payload["binance_spot_connection_id"] = _single_identity(
            [item.connection_id for item in spot],
            name="Binance spot source",
        )
        segment_payload["binance_perpetual_connection_id"] = _single_identity(
            [item.connection_id for item in perpetual],
            name="Binance perpetual source",
        )
    causal_segment_sha256 = _canonical_sha256(segment_payload)
    accumulator = PolymarketRound17FeatureAccumulator(
        condition_id=selected_market.condition_id,
        market_id=selected_market.condition_id,
        up_token_id=selected_market.up_token_id,
        down_token_id=selected_market.down_token_id,
        event_start_ms=selected_market.event_start_ms,
        event_end_ms=selected_market.end_ms,
        admission=selected_admission,
        causal_segment_sha256=causal_segment_sha256,
    )

    chainlink_index = 0
    binance_index = 0
    book_index = 0
    output_rows: list[PolymarketRound17FeatureRow] = []
    for base in rows:
        while (
            chainlink_index < len(chainlink)
            and chainlink[chainlink_index].tick.received_at_ms <= base.decision_time_ms
        ):
            accumulator.ingest_chainlink(
                chainlink[chainlink_index].tick,
                causal_segment_sha256=causal_segment_sha256,
            )
            chainlink_index += 1
        while (
            binance_index < len(binance)
            and binance[binance_index].received_at_ms <= base.decision_time_ms
        ):
            accumulator.ingest_binance(
                binance[binance_index],
                causal_segment_sha256=causal_segment_sha256,
            )
            binance_index += 1
        while (
            book_index < len(selected_books)
            and selected_books[book_index].received_wall_ms <= base.decision_time_ms
        ):
            book = selected_books[book_index]
            accumulator.ingest_book(
                "up" if book.token_id == selected_market.up_token_id else "down",
                book.snapshot,
                causal_segment_sha256=causal_segment_sha256,
            )
            book_index += 1
        output_rows.append(accumulator.build(base))

    provisional = PolymarketRound17ConditionDataset(
        run_id=selected_admission.run_id,
        condition_id=selected_market.condition_id,
        event_start_ms=selected_market.event_start_ms,
        event_end_ms=selected_market.end_ms,
        admission_sha256=selected_admission.admission_sha256,
        causal_segment_sha256=causal_segment_sha256,
        feature_names_sha256=POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
        base_row_count=len(rows),
        chainlink_event_count=len(chainlink),
        spot_trade_count=sum(item.market == "spot" for item in binance),
        perpetual_trade_count=sum(item.market == "perpetual" for item in binance),
        up_book_count=len(up_books),
        down_book_count=len(down_books),
        binance_layer_eligible=selected_admission.binance_layer_eligible,
        rows=tuple(output_rows),
        dataset_sha256="0" * 64,
    )
    return replace(
        provisional,
        dataset_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


__all__ = [
    "POLYMARKET_ROUND17_CONDITION_DATASET_SCHEMA_VERSION",
    "PolymarketRound17ChainlinkObservation",
    "PolymarketRound17ConditionDataset",
    "materialize_round17_condition_rows",
    "parse_round17_binance_trade_event",
    "parse_round17_chainlink_event",
]
