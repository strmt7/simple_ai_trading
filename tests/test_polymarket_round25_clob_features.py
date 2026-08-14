from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import math

import pytest

from simple_ai_trading import polymarket_round25_clob_features as clob_features
from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket_round25_clob_features import (
    POLYMARKET_ROUND25_CLOB_FEATURE_NAMES,
    POLYMARKET_ROUND25_CLOB_FEATURE_SCHEMA_VERSION,
    POLYMARKET_ROUND25_CLOB_WINDOWS_MS,
    Round25ClobFeatureEngine,
)


START_MS = 1_800_000_000_000
CONDITION_ID = "0x" + "a" * 64
UP_TOKEN = "1" * 77
DOWN_TOKEN = "2" * 77


def _levels(
    best: str,
    *,
    descending: bool,
    quantity: str = "10",
    count: int = 3,
) -> tuple[BookLevel, ...]:
    start = Decimal(best)
    step = Decimal("0.01") * (-1 if descending else 1)
    return tuple(
        BookLevel(price=start + step * index, quantity=Decimal(quantity) + index)
        for index in range(count)
    )


def _book(
    token: str,
    *,
    offset_ms: int = 100,
    monotonic_ns: int = 1_000,
    best_bid: str | None = None,
    best_ask: str | None = None,
    quantity: str = "10",
    bid_quantity: str | None = None,
    ask_quantity: str | None = None,
    source_offset_ms: int | None = None,
    received_offset_ms: int | None = None,
) -> PaperBookSnapshot:
    up = token == UP_TOKEN
    bid = best_bid or ("0.49" if up else "0.48")
    ask = best_ask or ("0.51" if up else "0.50")
    source_offset = offset_ms if source_offset_ms is None else source_offset_ms
    received_offset = offset_ms if received_offset_ms is None else received_offset_ms
    bid_size = quantity if bid_quantity is None else bid_quantity
    ask_size = quantity if ask_quantity is None else ask_quantity
    identity = f"{token}:{offset_ms}:{monotonic_ns}:{bid}:{ask}:{bid_size}:{ask_size}"
    return PaperBookSnapshot(
        venue="polymarket",
        market_id=CONDITION_ID,
        asset_id=token,
        bids=_levels(bid, descending=True, quantity=bid_size),
        asks=_levels(ask, descending=False, quantity=ask_size),
        source_time_ms=START_MS + source_offset,
        received_wall_ms=START_MS + received_offset,
        received_monotonic_ns=monotonic_ns,
        source_payload_sha256=hashlib.sha256(identity.encode("ascii")).hexdigest(),
        connected=True,
        gap_free=True,
    )


def _engine() -> Round25ClobFeatureEngine:
    return Round25ClobFeatureEngine(
        condition_id=CONDITION_ID,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        event_start_ms=START_MS,
    )


def _ready_engine() -> Round25ClobFeatureEngine:
    engine = _engine()
    engine.ingest(_book(UP_TOKEN, monotonic_ns=1_000))
    engine.ingest(_book(DOWN_TOKEN, monotonic_ns=1_000))
    engine.ingest(
        _book(
            UP_TOKEN,
            offset_ms=200,
            monotonic_ns=2_000,
            bid_quantity="14",
        )
    )
    engine.ingest(
        _book(
            DOWN_TOKEN,
            offset_ms=200,
            monotonic_ns=2_000,
            bid_quantity="8",
        )
    )
    return engine


def _feature(snapshot: object, name: str) -> float:
    index = POLYMARKET_ROUND25_CLOB_FEATURE_NAMES.index(name)
    return snapshot.values[index]  # type: ignore[attr-defined,no-any-return]


def test_schema_is_unique_target_blind_and_bound_to_fixed_windows() -> None:
    assert POLYMARKET_ROUND25_CLOB_FEATURE_SCHEMA_VERSION.endswith("v1")
    assert len(POLYMARKET_ROUND25_CLOB_FEATURE_NAMES) == 111
    assert len(set(POLYMARKET_ROUND25_CLOB_FEATURE_NAMES)) == 111
    assert POLYMARKET_ROUND25_CLOB_WINDOWS_MS == (
        250,
        1_000,
        5_000,
        15_000,
        30_000,
        60_000,
    )
    assert all(
        forbidden not in name
        for name in POLYMARKET_ROUND25_CLOB_FEATURE_NAMES
        for forbidden in ("structural_probability", "target", "resolution", "pnl")
    )


def test_available_snapshot_has_exact_market_prior_and_no_authority() -> None:
    first = _ready_engine().build(START_MS + 250)
    second = _ready_engine().build(START_MS + 250)
    expected_prior = Decimal("0.50") / (Decimal("0.50") + Decimal("0.49"))

    assert first == second
    assert first.available is True
    assert first.reasons == ()
    assert first.trading_authority is False
    assert first.market_prior_probability == pytest.approx(float(expected_prior))
    assert _feature(first, "clob.normalized_market_prior_up") == pytest.approx(
        float(expected_prior)
    )
    assert _feature(first, "clob.complement_buy_overround") == pytest.approx(0.01)
    assert _feature(first, "clob.complement_sell_underround") == pytest.approx(0.03)
    assert _feature(first, "clob.complement_midpoint_error") == pytest.approx(-0.01)
    assert first.maximum_receipt_ms == START_MS + 200
    assert first.source_chain_sha256 != hashlib.sha256(b"").hexdigest()
    assert all(math.isfinite(value) for value in first.values)


def test_one_batch_may_update_both_tokens_at_same_monotonic_receipt() -> None:
    engine = _engine()

    engine.ingest(_book(UP_TOKEN, monotonic_ns=5_000))
    engine.ingest(_book(DOWN_TOKEN, monotonic_ns=5_000))

    assert engine.build(START_MS + 250).available is True


def test_exact_duplicate_or_global_regression_is_rejected() -> None:
    engine = _engine()
    first = _book(UP_TOKEN, monotonic_ns=5_000)
    engine.ingest(first)

    with pytest.raises(ValueError, match="chronology"):
        engine.ingest(first)
    engine.ingest(_book(UP_TOKEN, offset_ms=150, monotonic_ns=5_000))
    with pytest.raises(ValueError, match="chronology"):
        engine.ingest(_book(DOWN_TOKEN, offset_ms=150, monotonic_ns=4_999))


def test_flow_windows_capture_direction_depth_and_update_count() -> None:
    snapshot = _ready_engine().build(START_MS + 250)

    assert _feature(snapshot, "clob.up_book_update_count_250ms") == 2.0
    assert _feature(snapshot, "clob.down_book_update_count_250ms") == 2.0
    assert _feature(snapshot, "clob.up_top_order_flow_imbalance_250ms") > 0.0
    assert _feature(snapshot, "clob.down_top_order_flow_imbalance_250ms") < 0.0
    assert _feature(snapshot, "clob.up_log1p_gross_level_quantity_change_250ms") > 0.0
    assert _feature(snapshot, "clob.down_log1p_gross_level_quantity_change_250ms") > 0.0


def test_flow_window_recomputes_a_numerically_inconsistent_prefix_delta() -> None:
    series = clob_features._RollingBookSeries()
    series.append(clob_features._book_point(_book(UP_TOKEN, monotonic_ns=1_000)))
    series.append(
        clob_features._book_point(
            _book(
                UP_TOKEN,
                offset_ms=200,
                monotonic_ns=2_000,
                bid_quantity="14",
            )
        )
    )
    series.prefixes[0][-1] = series.prefixes[1][-1] + 1.0

    window = series.window(START_MS + 250, 250)

    assert abs(window.top_ofi) <= 1.0
    assert window.gross_level_change >= 0.0


@pytest.mark.parametrize(
    ("prepare", "decision_offset", "reason"),
    [
        (
            lambda engine: engine.ingest(_book(UP_TOKEN, monotonic_ns=1_000)),
            250,
            "down_book_unavailable",
        ),
        (lambda engine: None, 250, "up_book_unavailable"),
    ],
)
def test_missing_books_are_fail_closed(
    prepare: object,
    decision_offset: int,
    reason: str,
) -> None:
    engine = _engine()
    prepare(engine)  # type: ignore[operator]

    snapshot = engine.build(START_MS + decision_offset)

    assert snapshot.available is False
    assert reason in snapshot.reasons
    assert not any(snapshot.values)
    assert snapshot.trading_authority is False


def test_stale_receipts_and_sources_are_rejected_separately() -> None:
    receipt_stale = _engine()
    receipt_stale.ingest(_book(UP_TOKEN, monotonic_ns=1_000))
    receipt_stale.ingest(_book(DOWN_TOKEN, monotonic_ns=1_000))
    receipt_snapshot = receipt_stale.build(START_MS + 750)

    source_stale = _engine()
    source_stale.ingest(
        _book(
            UP_TOKEN,
            offset_ms=5_500,
            monotonic_ns=1_000,
            source_offset_ms=100,
            received_offset_ms=5_500,
        )
    )
    source_stale.ingest(
        _book(
            DOWN_TOKEN,
            offset_ms=5_500,
            monotonic_ns=1_000,
            source_offset_ms=100,
            received_offset_ms=5_500,
        )
    )
    source_snapshot = source_stale.build(START_MS + 5_750)

    assert "up_book_receipt_stale" in receipt_snapshot.reasons
    assert "down_book_receipt_stale" in receipt_snapshot.reasons
    assert "up_book_source_stale" in source_snapshot.reasons
    assert "down_book_source_stale" in source_snapshot.reasons


def test_gap_future_source_and_future_receipt_are_fail_closed() -> None:
    gap = _ready_engine()
    gap.mark_stream_gap()
    assert gap.build(START_MS + 250).reasons == ("clob_stream_gap_detected",)

    future_source = _engine()
    future_source.ingest(
        _book(
            UP_TOKEN,
            monotonic_ns=1_000,
            source_offset_ms=300,
            received_offset_ms=200,
        )
    )
    future_source.ingest(
        _book(
            DOWN_TOKEN,
            monotonic_ns=1_000,
            source_offset_ms=300,
            received_offset_ms=200,
        )
    )
    source_snapshot = future_source.build(START_MS + 250)
    assert "future_up_book_source_timestamp" in source_snapshot.reasons
    assert "future_down_book_source_timestamp" in source_snapshot.reasons

    future_receipt = _engine()
    future_receipt.ingest(
        _book(UP_TOKEN, monotonic_ns=1_000, received_offset_ms=300)
    )
    future_receipt.ingest(
        _book(DOWN_TOKEN, monotonic_ns=1_000, received_offset_ms=300)
    )
    with pytest.raises(ValueError, match="future receipts"):
        future_receipt.build(START_MS + 250)


@pytest.mark.parametrize(
    "mutation",
    [
        {"venue": "binance"},
        {"connected": False},
        {"gap_free": False},
        {"source_time_ms": True},
        {"received_wall_ms": True},
        {"received_monotonic_ns": True},
    ],
)
def test_snapshot_contract_rejects_wrong_venue_state_or_types(
    mutation: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="contract"):
        _engine().ingest(replace(_book(UP_TOKEN), **mutation))


def test_snapshot_contract_rejects_more_than_top_twenty_levels() -> None:
    snapshot = replace(
        _book(UP_TOKEN),
        bids=_levels("0.49", descending=True, count=21),
        asks=_levels("0.51", descending=False, count=21),
    )

    with pytest.raises(ValueError, match="contract"):
        _engine().ingest(snapshot)


def test_engine_rejects_wrong_market_token_and_outcome_identity() -> None:
    wrong_market = replace(_book(UP_TOKEN), market_id="0x" + "b" * 64)
    wrong_token = replace(_book(UP_TOKEN), asset_id="3" * 77)

    with pytest.raises(ValueError, match="identity"):
        _engine().ingest(wrong_market)
    with pytest.raises(ValueError, match="identity"):
        _engine().ingest(wrong_token)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"condition_id": "bad"},
        {"up_token_id": "1"},
        {"down_token_id": UP_TOKEN},
        {"event_start_ms": True},
        {"event_start_ms": START_MS + 1},
    ],
)
def test_engine_rejects_invalid_identity(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "condition_id": CONDITION_ID,
        "up_token_id": UP_TOKEN,
        "down_token_id": DOWN_TOKEN,
        "event_start_ms": START_MS,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        Round25ClobFeatureEngine(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("decision", [True, START_MS + 1, START_MS + 300_000])
def test_engine_rejects_invalid_decision(decision: object) -> None:
    with pytest.raises(ValueError):
        _ready_engine().build(decision)  # type: ignore[arg-type]
