"""Development-only executable lead-lag analysis for the Round 26 pilot."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_recorder import DecodedPublicEvent, PolymarketEvidenceStore
from .polymarket_replay import (
    PolymarketEvidenceReplay,
    PolymarketRecordedBook,
    PolymarketResolutionEvidence,
)
from .polymarket_round26_pilot import (
    Round26PilotContract,
    load_round26_pilot_contract,
)
from .storage import write_bytes_atomic


ROUND26_ANALYSIS_SCHEMA_VERSION = "polymarket-round26-twap60-analysis-v2"
ROUND26_DECISION_STEP_MS = 100
ROUND26_BOOK_SAMPLE_INTERVAL_MS = 50
ROUND26_MAXIMUM_BOOK_OBSERVATION_DELAY_MS = 250
ROUND26_QUANTITY = Decimal("5")
ROUND26_LOOKBACK_MS = (100, 250, 500, 1000, 2000)
ROUND26_THRESHOLDS_BPS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
ROUND26_TAKER_DELAYS_MS = (250, 500, 1000, 2000)
ROUND26_HOLD_MS = (250, 500, 1000, 2000)
ROUND26_SIGNAL_MODES = ("momentum", "reversion")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


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


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _verify_self_hash(
    value: Mapping[str, object],
    *,
    field: str,
    label: str,
) -> str:
    payload = dict(value)
    claimed = str(payload.pop(field, "")).lower()
    if len(claimed) != 64 or claimed != _canonical_sha256(payload):
        raise ValueError(f"{label} hash differs")
    return claimed


@dataclass(frozen=True, slots=True)
class _PricePoint:
    received_monotonic_ns: int
    received_wall_ms: int
    price: float


@dataclass(frozen=True, slots=True)
class _TwapPoint:
    source_time_ms: int
    publisher_time_ms: int
    received_wall_ms: int
    received_monotonic_ns: int
    exact_e18: int


@dataclass(frozen=True, slots=True)
class _Signal:
    received_monotonic_ns: int
    received_wall_ms: int
    score_bps: float


@dataclass(frozen=True, slots=True)
class _BookSeries:
    times_ns: tuple[int, ...]
    books: tuple[PolymarketRecordedBook, ...]

    def first_at_or_after(
        self,
        target_ns: int,
        *,
        maximum_delay_ms: int,
    ) -> PolymarketRecordedBook | None:
        index = bisect_left(self.times_ns, int(target_ns))
        if index >= len(self.books):
            return None
        book = self.books[index]
        if book.received_monotonic_ns - target_ns > maximum_delay_ms * 1_000_000:
            return None
        return book


@dataclass(frozen=True, slots=True)
class _Trade:
    condition_id: str
    outcome: str
    decision_wall_ms: int
    decision_monotonic_ns: int
    entry_wall_ms: int
    entry_monotonic_ns: int
    exit_wall_ms: int
    exit_monotonic_ns: int
    entry_price: Decimal
    exit_price: Decimal
    gross_pnl: Decimal
    fees: Decimal
    net_pnl: Decimal


def _twap_point(event: DecodedPublicEvent) -> _TwapPoint | None:
    if event.event_type != "crypto_prices_twap_sixty:update":
        return None
    message = event.event
    payload = message.get("payload")
    if (
        message.get("topic") != "crypto_prices_twap_sixty"
        or message.get("type") != "update"
        or not isinstance(payload, Mapping)
    ):
        raise ValueError("Round 26 TWAP60 event identity differs")
    source_time = payload.get("timestamp")
    publisher_time = message.get("timestamp")
    exact_text = payload.get("full_accuracy_value")
    if (
        isinstance(source_time, bool)
        or not isinstance(source_time, int)
        or source_time <= 0
        or source_time % 1_000 != 0
        or isinstance(publisher_time, bool)
        or not isinstance(publisher_time, int)
        or publisher_time < source_time
        or payload.get("symbol") != "btc/usd"
        or payload.get("window_s") != 60
        or not isinstance(exact_text, str)
        or not exact_text.isascii()
        or not exact_text.isdigit()
        or int(exact_text) <= 0
    ):
        raise ValueError("Round 26 TWAP60 payload differs")
    return _TwapPoint(
        source_time_ms=source_time,
        publisher_time_ms=publisher_time,
        received_wall_ms=event.received_wall_ms,
        received_monotonic_ns=event.received_monotonic_ns,
        exact_e18=int(exact_text),
    )


def _source_points(
    store: PolymarketEvidenceStore,
    run_id: str,
) -> tuple[
    tuple[_PricePoint, ...],
    tuple[_PricePoint, ...],
    tuple[_TwapPoint, ...],
    int,
]:
    """Load CEX trades and TWAP60 after validating the complete replay.

    ``verified_source=True`` intentionally requires a gap-free terminal audit.
    Round 26 admits segmented CLOB gaps for diagnostics, so the enclosing replay
    first verifies manifests, hashes, recorder errors, and gap structure; this
    iterator then reconstructs the independent Binance streams from that store.
    """

    by_stream: dict[str, list[_PricePoint]] = {
        "binance_spot": [],
        "binance_futures": [],
    }
    twap: list[_TwapPoint] = []
    event_count = 0
    for event in store.iter_public_events(
        run_id,
        streams=("binance_spot", "binance_futures", "polymarket_rtds"),
        ordered=True,
        verified_source=False,
    ):
        event_count += 1
        if event.stream == "polymarket_rtds":
            point = _twap_point(event)
            if point is not None:
                twap.append(point)
            continue
        if event.event_type != "trade" or event.symbol != "BTC":
            continue
        payload = event.event.get("data")
        if not isinstance(payload, Mapping):
            raise ValueError("Round 26 Binance trade payload differs")
        try:
            price = float(payload.get("p"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Round 26 Binance trade price differs") from exc
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Round 26 Binance trade price differs")
        points = by_stream[event.stream]
        point = _PricePoint(
            received_monotonic_ns=event.received_monotonic_ns,
            received_wall_ms=event.received_wall_ms,
            price=price,
        )
        if points and point.received_monotonic_ns < points[-1].received_monotonic_ns:
            raise ValueError("Round 26 Binance receipt clock regressed")
        points.append(point)
    return (
        tuple(by_stream["binance_spot"]),
        tuple(by_stream["binance_futures"]),
        tuple(twap),
        event_count,
    )


def _settlement_mechanism_audit(
    markets: Sequence[PolymarketFiveMinuteMarket],
    resolutions: Sequence[PolymarketResolutionEvidence],
    twap_points: Sequence[_TwapPoint],
) -> dict[str, object]:
    by_source_time: dict[int, _TwapPoint] = {}
    duplicate_count = 0
    for point in twap_points:
        existing = by_source_time.get(point.source_time_ms)
        if existing is None:
            by_source_time[point.source_time_ms] = point
            continue
        duplicate_count += 1
        if existing.exact_e18 != point.exact_e18:
            raise ValueError("Round 26 TWAP60 duplicate source values conflict")
        if point.received_monotonic_ns < existing.received_monotonic_ns:
            by_source_time[point.source_time_ms] = point
    resolution_by_condition: dict[str, PolymarketResolutionEvidence] = {}
    for resolution in resolutions:
        existing = resolution_by_condition.get(resolution.condition_id)
        if existing is not None and (
            existing.winning_asset_id != resolution.winning_asset_id
            or existing.winning_outcome != resolution.winning_outcome
        ):
            raise ValueError("Round 26 official resolution evidence conflicts")
        resolution_by_condition[resolution.condition_id] = resolution
    rows: list[dict[str, object]] = []
    missing_boundary_count = 0
    tie_count = 0
    for market in sorted(markets, key=lambda item: item.event_start_ms):
        resolution = resolution_by_condition.get(market.condition_id)
        if resolution is None:
            continue
        if resolution.winning_outcome not in {"Up", "Down"}:
            raise ValueError("Round 26 official winning outcome differs")
        start = by_source_time.get(market.event_start_ms)
        end = by_source_time.get(market.end_ms)
        if start is None or end is None:
            missing_boundary_count += 1
            continue
        if end.exact_e18 == start.exact_e18:
            tie_count += 1
            continue
        predicted = "Up" if end.exact_e18 > start.exact_e18 else "Down"
        rows.append(
            {
                "condition_id": market.condition_id,
                "event_start_ms": market.event_start_ms,
                "end_ms": market.end_ms,
                "start_twap_e18": str(start.exact_e18),
                "end_twap_e18": str(end.exact_e18),
                "start_receipt_delay_ms": start.received_wall_ms - start.source_time_ms,
                "end_receipt_delay_ms": end.received_wall_ms - end.source_time_ms,
                "predicted_outcome": predicted,
                "official_outcome": resolution.winning_outcome,
                "agrees": predicted == resolution.winning_outcome,
            }
        )
    agreement_count = sum(bool(row["agrees"]) for row in rows)
    minimum_sample_met = len(rows) >= 8
    exact_boundary_rule_supported = minimum_sample_met and agreement_count == len(rows)
    return {
        "twap_point_count": len(twap_points),
        "unique_source_time_count": len(by_source_time),
        "identical_duplicate_count": duplicate_count,
        "official_resolution_count": len(resolution_by_condition),
        "evaluated_market_count": len(rows),
        "missing_exact_boundary_count": missing_boundary_count,
        "tie_rule_unverified_count": tie_count,
        "agreement_count": agreement_count,
        "agreement_rate": agreement_count / len(rows) if rows else 0.0,
        "minimum_eight_markets_met": minimum_sample_met,
        "exact_boundary_rule_supported_in_pilot": exact_boundary_rule_supported,
        "market_results": rows,
        "qualification_claim": False,
        "edge_claim": False,
        "profitability_claim": False,
    }


def _latest_point(
    points: Sequence[_PricePoint],
    times: Sequence[int],
    target_ns: int,
) -> _PricePoint | None:
    index = bisect_right(times, target_ns) - 1
    return None if index < 0 else points[index]


def _return_bps(
    points: Sequence[_PricePoint],
    times: Sequence[int],
    *,
    now_ns: int,
    lookback_ms: int,
    maximum_staleness_ms: int = 250,
) -> float | None:
    current = _latest_point(points, times, now_ns)
    previous_target = now_ns - lookback_ms * 1_000_000
    previous = _latest_point(points, times, previous_target)
    if current is None or previous is None:
        return None
    if (
        now_ns - current.received_monotonic_ns > maximum_staleness_ms * 1_000_000
        or previous_target - previous.received_monotonic_ns
        > maximum_staleness_ms * 1_000_000
    ):
        return None
    return math.log(current.price / previous.price) * 10_000.0


def _signals(
    spot: Sequence[_PricePoint],
    futures: Sequence[_PricePoint],
    *,
    lookback_ms: int,
) -> tuple[_Signal, ...]:
    if not spot or not futures:
        return ()
    spot_times = tuple(point.received_monotonic_ns for point in spot)
    futures_times = tuple(point.received_monotonic_ns for point in futures)
    output: list[_Signal] = []
    last_bucket = -1
    for point in spot:
        bucket = point.received_monotonic_ns // (ROUND26_DECISION_STEP_MS * 1_000_000)
        if bucket == last_bucket:
            continue
        last_bucket = bucket
        spot_return = _return_bps(
            spot,
            spot_times,
            now_ns=point.received_monotonic_ns,
            lookback_ms=lookback_ms,
        )
        futures_return = _return_bps(
            futures,
            futures_times,
            now_ns=point.received_monotonic_ns,
            lookback_ms=lookback_ms,
        )
        if spot_return is None or futures_return is None:
            continue
        output.append(
            _Signal(
                received_monotonic_ns=point.received_monotonic_ns,
                received_wall_ms=point.received_wall_ms,
                score_bps=(spot_return + futures_return) / 2.0,
            )
        )
    return tuple(output)


def _market_for_time(
    markets: Sequence[PolymarketFiveMinuteMarket],
    starts: Sequence[int],
    wall_ms: int,
) -> PolymarketFiveMinuteMarket | None:
    index = bisect_right(starts, wall_ms) - 1
    if index < 0:
        return None
    market = markets[index]
    return market if market.event_start_ms <= wall_ms < market.end_ms else None


def _book_series(
    replay: PolymarketEvidenceReplay,
) -> dict[str, _BookSeries]:
    grouped: dict[str, list[PolymarketRecordedBook]] = {}
    for book in replay.books:
        grouped.setdefault(book.token_id, []).append(book)
    output: dict[str, _BookSeries] = {}
    for token, books in grouped.items():
        ordered = tuple(sorted(books, key=lambda item: item.received_monotonic_ns))
        output[token] = _BookSeries(
            times_ns=tuple(item.received_monotonic_ns for item in ordered),
            books=ordered,
        )
    return output


def _execute_taker_trade(
    signal: _Signal,
    market: PolymarketFiveMinuteMarket,
    series: _BookSeries,
    *,
    mode: str,
    delay_ms: int,
    hold_ms: int,
) -> _Trade | None:
    direction = 1 if signal.score_bps > 0 else -1
    if mode == "reversion":
        direction *= -1
    outcome = "Up" if direction > 0 else "Down"
    token = market.up_token_id if outcome == "Up" else market.down_token_id
    if not series.books or series.books[0].token_id != token:
        raise ValueError("Round 26 book series token differs")
    entry_target = signal.received_monotonic_ns + delay_ms * 1_000_000
    entry = series.first_at_or_after(
        entry_target,
        maximum_delay_ms=ROUND26_MAXIMUM_BOOK_OBSERVATION_DELAY_MS,
    )
    if entry is None or not entry.snapshot.asks:
        return None
    # A marketable close is a second taker order. It incurs the same conservative
    # venue/submission delay after the requested holding interval as the entry.
    exit_target = entry.received_monotonic_ns + (hold_ms + delay_ms) * 1_000_000
    exit_book = series.first_at_or_after(
        exit_target,
        maximum_delay_ms=ROUND26_MAXIMUM_BOOK_OBSERVATION_DELAY_MS,
    )
    if (
        exit_book is None
        or not exit_book.snapshot.bids
        or entry.segment_id != exit_book.segment_id
        or exit_book.market.condition_id != market.condition_id
        or exit_book.received_wall_ms >= market.end_ms
    ):
        return None
    ask = entry.snapshot.asks[0]
    bid = exit_book.snapshot.bids[0]
    if ask.quantity < ROUND26_QUANTITY or bid.quantity < ROUND26_QUANTITY:
        return None
    fee_model = market.fee_schedule.fee_model()
    entry_fee = fee_model(ask.price, ROUND26_QUANTITY, "taker")
    exit_fee = fee_model(bid.price, ROUND26_QUANTITY, "taker")
    gross = ROUND26_QUANTITY * (bid.price - ask.price)
    fees = entry_fee + exit_fee
    return _Trade(
        condition_id=market.condition_id,
        outcome=outcome,
        decision_wall_ms=signal.received_wall_ms,
        decision_monotonic_ns=signal.received_monotonic_ns,
        entry_wall_ms=entry.received_wall_ms,
        entry_monotonic_ns=entry.received_monotonic_ns,
        exit_wall_ms=exit_book.received_wall_ms,
        exit_monotonic_ns=exit_book.received_monotonic_ns,
        entry_price=ask.price,
        exit_price=bid.price,
        gross_pnl=gross,
        fees=fees,
        net_pnl=gross - fees,
    )


def _maximum_drawdown(pnls: Sequence[Decimal]) -> Decimal:
    equity = Decimal("0")
    peak = Decimal("0")
    drawdown = Decimal("0")
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def _configuration_result(
    signals: Sequence[_Signal],
    markets: Sequence[PolymarketFiveMinuteMarket],
    starts: Sequence[int],
    series_by_token: Mapping[str, _BookSeries],
    *,
    lookback_ms: int,
    threshold_bps: float,
    mode: str,
    delay_ms: int,
    hold_ms: int,
) -> tuple[dict[str, object], tuple[_Trade, ...]]:
    trades: list[_Trade] = []
    eligible_signal_count = 0
    last_exit_ns = -1
    for signal in signals:
        if abs(signal.score_bps) < threshold_bps or signal.received_monotonic_ns < last_exit_ns:
            continue
        market = _market_for_time(markets, starts, signal.received_wall_ms)
        if market is None:
            continue
        eligible_signal_count += 1
        direction = 1 if signal.score_bps > 0 else -1
        if mode == "reversion":
            direction *= -1
        token = market.up_token_id if direction > 0 else market.down_token_id
        series = series_by_token.get(token)
        if series is None:
            continue
        trade = _execute_taker_trade(
            signal,
            market,
            series,
            mode=mode,
            delay_ms=delay_ms,
            hold_ms=hold_ms,
        )
        if trade is None:
            continue
        trades.append(trade)
        last_exit_ns = trade.exit_monotonic_ns
    gross = sum((trade.gross_pnl for trade in trades), Decimal("0"))
    fees = sum((trade.fees for trade in trades), Decimal("0"))
    net = sum((trade.net_pnl for trade in trades), Decimal("0"))
    wins = sum(trade.net_pnl > 0 for trade in trades)
    result = {
        "lookback_ms": lookback_ms,
        "threshold_bps": threshold_bps,
        "signal_mode": mode,
        "taker_delay_ms": delay_ms,
        "hold_ms": hold_ms,
        "eligible_signal_count": eligible_signal_count,
        "trade_count": len(trades),
        "unique_condition_count": len({trade.condition_id for trade in trades}),
        "gross_pnl_quote": float(gross),
        "fees_quote": float(fees),
        "net_pnl_quote": float(net),
        "mean_net_pnl_quote": float(net / len(trades)) if trades else 0.0,
        "win_rate": wins / len(trades) if trades else 0.0,
        "maximum_drawdown_quote": float(
            _maximum_drawdown(tuple(trade.net_pnl for trade in trades))
        ),
    }
    return result, tuple(trades)


def _trade_payload(trade: _Trade) -> dict[str, object]:
    return {
        "condition_id": trade.condition_id,
        "outcome": trade.outcome,
        "decision_wall_ms": trade.decision_wall_ms,
        "decision_monotonic_ns": trade.decision_monotonic_ns,
        "entry_wall_ms": trade.entry_wall_ms,
        "entry_monotonic_ns": trade.entry_monotonic_ns,
        "exit_wall_ms": trade.exit_wall_ms,
        "exit_monotonic_ns": trade.exit_monotonic_ns,
        "entry_price": str(trade.entry_price),
        "exit_price": str(trade.exit_price),
        "gross_pnl_quote": str(trade.gross_pnl),
        "fees_quote": str(trade.fees),
        "net_pnl_quote": str(trade.net_pnl),
    }


def _load_capture_result(
    path: Path,
    contract: Round26PilotContract,
) -> dict[str, object]:
    result = _read_json(path, label="Round 26 pilot result")
    _verify_self_hash(result, field="result_sha256", label="Round 26 pilot result")
    if (
        result.get("contract_sha256") != contract.contract_sha256
        or result.get("status") not in {"complete", "degraded"}
        or result.get("model_data_eligible") is not False
        or result.get("edge_claim") is not False
        or result.get("profitability_claim") is not False
        or not isinstance(result.get("integrity_errors"), list)
    ):
        raise ValueError("Round 26 pilot result differs")
    return result


def run_round26_pilot_analysis(
    repository: str | Path,
    *,
    contract_path: str | Path,
    database_path: str | Path,
    capture_result_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    root = Path(repository).resolve()
    contract = load_round26_pilot_contract(contract_path, repository=root)
    capture = _load_capture_result(Path(capture_result_path), contract)
    if capture["integrity_errors"]:
        raise ValueError("Round 26 pilot has terminal integrity errors")
    run_id = str(capture.get("run_id") or "")
    with PolymarketEvidenceStore(
        database_path,
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id=run_id,
            allow_segmented_gaps=True,
            include_resolutions=True,
            book_sample_interval_ms=ROUND26_BOOK_SAMPLE_INTERVAL_MS,
            materialized_minimum_depth_levels=1,
            cap_materialized_depth_to_minimum_order_size=True,
        )
        spot, futures, twap_points, source_event_count = _source_points(store, run_id)
    markets = tuple(sorted(replay.markets, key=lambda market: market.event_start_ms))
    starts = tuple(market.event_start_ms for market in markets)
    series_by_token = _book_series(replay)
    settlement_audit = _settlement_mechanism_audit(
        markets,
        replay.resolutions,
        twap_points,
    )
    configurations: list[dict[str, object]] = []
    trades_by_key: dict[tuple[object, ...], tuple[_Trade, ...]] = {}
    signal_count_by_lookback: dict[str, int] = {}
    for lookback_ms in ROUND26_LOOKBACK_MS:
        signals = _signals(spot, futures, lookback_ms=lookback_ms)
        signal_count_by_lookback[str(lookback_ms)] = len(signals)
        for threshold_bps in ROUND26_THRESHOLDS_BPS:
            for mode in ROUND26_SIGNAL_MODES:
                for delay_ms in ROUND26_TAKER_DELAYS_MS:
                    for hold_ms in ROUND26_HOLD_MS:
                        result, trades = _configuration_result(
                            signals,
                            markets,
                            starts,
                            series_by_token,
                            lookback_ms=lookback_ms,
                            threshold_bps=threshold_bps,
                            mode=mode,
                            delay_ms=delay_ms,
                            hold_ms=hold_ms,
                        )
                        configurations.append(result)
                        key = (
                            lookback_ms,
                            threshold_bps,
                            mode,
                            delay_ms,
                            hold_ms,
                        )
                        trades_by_key[key] = trades
    ranked = sorted(
        configurations,
        key=lambda item: (
            int(item["trade_count"]) >= 20,
            float(item["net_pnl_quote"]),
            float(item["mean_net_pnl_quote"]),
            int(item["trade_count"]),
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    best_trades: tuple[_Trade, ...] = ()
    if best is not None:
        best_key = (
            best["lookback_ms"],
            best["threshold_bps"],
            best["signal_mode"],
            best["taker_delay_ms"],
            best["hold_ms"],
        )
        best_trades = trades_by_key[best_key]
    continuity_clean = int(capture.get("stream_gap_count", 0)) == 0
    pilot_conditions = {
        "minimum_action_count_met": bool(best and int(best["trade_count"]) >= 20),
        "positive_after_cost_pnl": bool(best and float(best["net_pnl_quote"]) > 0),
        "positive_mean_markout": bool(best and float(best["gross_pnl_quote"]) > 0),
        "no_stream_gaps": continuity_clean,
    }
    payload: dict[str, object] = {
        "schema_version": ROUND26_ANALYSIS_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "capture_result_sha256": capture["result_sha256"],
        "run_id": run_id,
        "status": "development_analysis_complete",
        "data_role": "development_only",
        "capture_status": capture["status"],
        "capture_stream_gap_count": int(capture.get("stream_gap_count", 0)),
        "replay_diagnostics": replay.diagnostics.asdict(),
        "market_count": len(markets),
        "resolved_market_count": len(replay.resolutions),
        "materialized_book_count": len(replay.books),
        "source_event_count": source_event_count,
        "binance_spot_trade_count": len(spot),
        "binance_futures_trade_count": len(futures),
        "settlement_mechanism_audit": settlement_audit,
        "signal_count_by_lookback": signal_count_by_lookback,
        "configuration_count": len(configurations),
        "fixed_grid": {
            "lookback_ms": list(ROUND26_LOOKBACK_MS),
            "threshold_bps": list(ROUND26_THRESHOLDS_BPS),
            "signal_modes": list(ROUND26_SIGNAL_MODES),
            "taker_delay_ms": list(ROUND26_TAKER_DELAYS_MS),
            "hold_ms": list(ROUND26_HOLD_MS),
            "quantity": str(ROUND26_QUANTITY),
            "book_sample_interval_ms": ROUND26_BOOK_SAMPLE_INTERVAL_MS,
            "maximum_book_observation_delay_ms": (
                ROUND26_MAXIMUM_BOOK_OBSERVATION_DELAY_MS
            ),
        },
        "taker_results": ranked,
        "best_in_sample_taker_configuration": best,
        "best_in_sample_trades": [_trade_payload(trade) for trade in best_trades],
        "maker_evaluation": {
            "status": "not_economically_identifiable_from_public_feed",
            "reason": (
                "public last_trade_price direction and depth depletion cannot prove "
                "the bot's queue position or fill; maker PnL is not scored"
            ),
            "profitability_claim": False,
        },
        "pilot_pass_conditions": pilot_conditions,
        "pilot_passed": all(pilot_conditions.values()),
        "selection_bias_warning": (
            "The best configuration was selected on the same one-hour development "
            "sample; it is hypothesis generation, not edge evidence."
        ),
        "sealed_selection_eligible": False,
        "edge_claim": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["analysis_sha256"] = _canonical_sha256(payload)
    write_bytes_atomic(
        Path(output_path),
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    return payload


__all__ = [
    "ROUND26_ANALYSIS_SCHEMA_VERSION",
    "run_round26_pilot_analysis",
]
