from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket_btc_reference import (
    PolymarketBtcEndpointEstimate,
    PolymarketBtcReferenceWindow,
    PolymarketChainlinkBtcTick,
)
from simple_ai_trading.polymarket_external_signal import (
    PolymarketBtcReferenceFeatures,
)
from simple_ai_trading.polymarket_round14_features import (
    POLYMARKET_ROUND14_FEATURE_NAMES,
    POLYMARKET_ROUND14_FEATURE_NAMES_SHA256,
    PolymarketRound14Snapshot,
    build_round14_snapshot_features,
)


EVENT_START_MS = 1_800_000_000_000
EVENT_END_MS = EVENT_START_MS + 300_000
DECISION_MS = EVENT_START_MS + 60_000
CONDITION_ID = "0x" + "1" * 64
UP_TOKEN_ID = "1" * 40
DOWN_TOKEN_ID = "2" * 40


def _book(
    token_id: str,
    *,
    bids: tuple[tuple[str, str], ...],
    asks: tuple[tuple[str, str], ...],
    received_wall_ms: int = DECISION_MS - 5,
) -> PaperBookSnapshot:
    return PaperBookSnapshot(
        venue="polymarket",
        market_id=CONDITION_ID,
        asset_id=token_id,
        bids=tuple(
            BookLevel(price=Decimal(price), quantity=Decimal(quantity))
            for price, quantity in bids
        ),
        asks=tuple(
            BookLevel(price=Decimal(price), quantity=Decimal(quantity))
            for price, quantity in asks
        ),
        source_time_ms=received_wall_ms - 2,
        received_wall_ms=received_wall_ms,
        received_monotonic_ns=9_000_000_000,
        source_payload_sha256="a" * 64 if token_id == UP_TOKEN_ID else "b" * 64,
    )


def _snapshot(
    *,
    reference: PolymarketBtcReferenceWindow | None = None,
    up_book: PaperBookSnapshot | None = None,
    external: PolymarketBtcReferenceFeatures | None | object = ...,
) -> PolymarketRound14Snapshot:
    reference = reference or PolymarketBtcReferenceWindow(
        event_start_ms=EVENT_START_MS,
        end_ms=EVENT_END_MS,
        open_price=Decimal("64000"),
        close_price=None,
        observed_at_ms=EVENT_START_MS + 1,
        completed=False,
        incomplete=True,
        cached=False,
        source_payload_sha256="c" * 64,
    )
    tick = PolymarketChainlinkBtcTick(
        source_time_ms=DECISION_MS - 10,
        publisher_time_ms=DECISION_MS - 8,
        received_at_ms=DECISION_MS - 5,
        price=Decimal("64100"),
        source_payload_sha256="d" * 64,
    )
    estimate = PolymarketBtcEndpointEstimate(
        available=True,
        probability_up=0.65,
        variance_rate_per_second=1e-8,
        log_distance_from_open=0.001561281,
        remaining_seconds=240.0,
        tick_count=100,
        coverage_seconds=59.0,
        reasons=(),
    )
    up = up_book or _book(
        UP_TOKEN_ID,
        bids=(("0.50", "10"), ("0.49", "20")),
        asks=(("0.52", "5"), ("0.53", "15")),
    )
    down = _book(
        DOWN_TOKEN_ID,
        bids=(("0.47", "8"), ("0.46", "12")),
        asks=(("0.49", "6"), ("0.50", "14")),
    )
    default_external = PolymarketBtcReferenceFeatures(
        observed_at_ms=DECISION_MS - 5,
        spot_mid=Decimal("64099"),
        futures_mid=Decimal("64101"),
        spot_spread_bps=Decimal("0.4"),
        futures_spread_bps=Decimal("0.5"),
        futures_basis_bps=Decimal("0.312"),
        spot_log_return=0.0001,
        futures_log_return=0.00012,
        event_time_skew_ms=2,
        receive_time_skew_ms=3,
    )
    selected_external = default_external if external is ... else external
    return PolymarketRound14Snapshot(
        condition_id=CONDITION_ID,
        up_token_id=UP_TOKEN_ID,
        down_token_id=DOWN_TOKEN_ID,
        event_start_ms=EVENT_START_MS,
        event_end_ms=EVENT_END_MS,
        decision_time_ms=DECISION_MS,
        reference=reference,
        chainlink_tick=tick,
        structural_estimate=estimate,
        up_book=up,
        down_book=down,
        external_features=selected_external,  # type: ignore[arg-type]
    )


def test_round14_snapshot_features_are_target_free_and_reconciled() -> None:
    row = build_round14_snapshot_features(_snapshot())
    values = row.value_map()

    assert len(row.values) == len(POLYMARKET_ROUND14_FEATURE_NAMES) == 50
    assert row.feature_names_sha256 == POLYMARKET_ROUND14_FEATURE_NAMES_SHA256
    assert values["elapsed_fraction"] == pytest.approx(0.2)
    assert values["remaining_seconds"] == 240.0
    assert values["structural_probability_up"] == 0.65
    assert values["ask_pair_cost"] == pytest.approx(1.01)
    assert values["bid_pair_value"] == pytest.approx(0.97)
    assert values["executable_parity_gap"] == pytest.approx(-0.01)
    assert values["external_available"] == 1.0
    assert "official_up" not in values
    assert "close_price" not in values


def test_round14_snapshot_hash_is_deterministic_and_binds_external_state() -> None:
    with_external = build_round14_snapshot_features(_snapshot())
    repeated = build_round14_snapshot_features(_snapshot())
    without_external = build_round14_snapshot_features(_snapshot(external=None))

    assert with_external.input_sha256 == repeated.input_sha256
    assert with_external.input_sha256 != without_external.input_sha256
    assert without_external.value_map()["external_available"] == 0.0


def test_round14_snapshot_rejects_target_or_future_book() -> None:
    completed = PolymarketBtcReferenceWindow(
        event_start_ms=EVENT_START_MS,
        end_ms=EVENT_END_MS,
        open_price=Decimal("64000"),
        close_price=Decimal("64100"),
        observed_at_ms=EVENT_END_MS + 1,
        completed=True,
        incomplete=False,
        cached=True,
        source_payload_sha256="e" * 64,
    )
    with pytest.raises(ValueError, match="contains a target"):
        build_round14_snapshot_features(_snapshot(reference=completed))

    future = _book(
        UP_TOKEN_ID,
        bids=(("0.50", "10"),),
        asks=(("0.52", "5"),),
        received_wall_ms=DECISION_MS + 1,
    )
    with pytest.raises(ValueError, match="identity or chronology"):
        build_round14_snapshot_features(_snapshot(up_book=future))


def test_round14_snapshot_rejects_future_observed_reference() -> None:
    reference = replace(
        _snapshot().reference,
        observed_at_ms=DECISION_MS + 1,
    )

    with pytest.raises(ValueError, match="future"):
        build_round14_snapshot_features(_snapshot(reference=reference))


def test_round14_snapshot_rejects_stale_external_features() -> None:
    stale = PolymarketBtcReferenceFeatures(
        observed_at_ms=DECISION_MS - 1_501,
        spot_mid=Decimal("64099"),
        futures_mid=Decimal("64101"),
        spot_spread_bps=Decimal("0.4"),
        futures_spread_bps=Decimal("0.5"),
        futures_basis_bps=Decimal("0.312"),
        spot_log_return=0.0,
        futures_log_return=0.0,
        event_time_skew_ms=2,
        receive_time_skew_ms=3,
    )

    with pytest.raises(ValueError, match="external features are stale"):
        build_round14_snapshot_features(_snapshot(external=stale))


def test_round14_feature_row_detects_vector_tampering() -> None:
    row = build_round14_snapshot_features(_snapshot())

    with pytest.raises(ValueError, match="feature vector is invalid"):
        replace(row, values=row.values[:-1])
