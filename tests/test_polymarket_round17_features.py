from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import pytest

from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket_btc_reference import PolymarketChainlinkBtcTick
from simple_ai_trading.polymarket_round14_dataset import (
    PolymarketRound14ConditionAdmission,
)
from simple_ai_trading.polymarket_round14_features import (
    POLYMARKET_ROUND14_FEATURE_NAMES,
    POLYMARKET_ROUND14_FEATURE_NAMES_SHA256,
    PolymarketRound14FeatureRow,
)
from simple_ai_trading.polymarket_round17_features import (
    POLYMARKET_ROUND17_CONTRACT_SHA256,
    POLYMARKET_ROUND17_FEATURE_NAMES,
    POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
    PolymarketRound17BinanceTrade,
    PolymarketRound17FeatureAccumulator,
    PolymarketRound17FeatureRow,
)


EVENT_START_MS = 1_800_000_000_000
EVENT_END_MS = EVENT_START_MS + 300_000
DECISION_MS = EVENT_START_MS + 60_000
CONDITION_ID = "0x" + "1" * 64
UP_TOKEN_ID = "1" * 40
DOWN_TOKEN_ID = "2" * 40
CAUSAL_SEGMENT_SHA256 = "f" * 64
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _admission(
    *,
    binance_layer_eligible: bool = True,
) -> PolymarketRound14ConditionAdmission:
    binance_reasons = () if binance_layer_eligible else ("binance_gap",)
    provisional = PolymarketRound14ConditionAdmission(
        run_id="round17-test-run",
        condition_id=CONDITION_ID,
        event_start_ms=EVENT_START_MS,
        event_end_ms=EVENT_END_MS,
        candidate_row_count=1_120,
        materialized_row_count=1_120,
        external_row_count=1_120 if binance_layer_eligible else 0,
        row_coverage_fraction=1.0,
        external_coverage_fraction=1.0 if binance_layer_eligible else 0.0,
        maximum_consecutive_missing_ms=0,
        external_maximum_consecutive_missing_ms=(
            0 if binance_layer_eligible else 300_000
        ),
        chainlink_tick_count=300,
        spot_bbo_count=1_000 if binance_layer_eligible else 0,
        futures_bbo_count=1_000 if binance_layer_eligible else 0,
        spot_trade_count=1_000 if binance_layer_eligible else 0,
        futures_trade_count=1_000 if binance_layer_eligible else 0,
        ignored_futures_zero_trade_count=0,
        exact_chainlink_open_event_sha256="a" * 64,
        row_manifest_sha256="b" * 64,
        core_eligible=True,
        binance_layer_eligible=binance_layer_eligible,
        reasons=(),
        binance_reasons=binance_reasons,
        admission_sha256="0" * 64,
    )
    return replace(
        provisional,
        admission_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def _accumulator(
    *,
    binance_layer_eligible: bool = True,
) -> PolymarketRound17FeatureAccumulator:
    return PolymarketRound17FeatureAccumulator(
        condition_id=CONDITION_ID,
        market_id=CONDITION_ID,
        up_token_id=UP_TOKEN_ID,
        down_token_id=DOWN_TOKEN_ID,
        event_start_ms=EVENT_START_MS,
        event_end_ms=EVENT_END_MS,
        admission=_admission(binance_layer_eligible=binance_layer_eligible),
        causal_segment_sha256=CAUSAL_SEGMENT_SHA256,
    )


def _base(*, decision_time_ms: int = DECISION_MS) -> PolymarketRound14FeatureRow:
    return PolymarketRound14FeatureRow(
        condition_id=CONDITION_ID,
        decision_time_ms=decision_time_ms,
        feature_names_sha256=POLYMARKET_ROUND14_FEATURE_NAMES_SHA256,
        input_sha256="e" * 64,
        values=(0.0,) * len(POLYMARKET_ROUND14_FEATURE_NAMES),
    )


def _chainlink(
    *,
    received_at_ms: int,
    price: str,
    identity: str,
) -> PolymarketChainlinkBtcTick:
    return PolymarketChainlinkBtcTick(
        source_time_ms=received_at_ms - 2,
        publisher_time_ms=received_at_ms - 1,
        received_at_ms=received_at_ms,
        price=Decimal(price),
        source_payload_sha256=_sha256(identity),
    )


def _trade(
    *,
    market: str,
    received_at_ms: int,
    aggregate_trade_id: int,
    price: float,
    quantity: float,
    buyer_is_maker: bool,
) -> PolymarketRound17BinanceTrade:
    source = "BINANCE_SPOT" if market == "spot" else "BINANCE_USD_M_FUTURES"
    return PolymarketRound17BinanceTrade(
        market=market,
        source=source,
        symbol="BTCUSDT",
        connection_id=f"round17:{market}:" + "a" * 32,
        event_time_ms=received_at_ms - 1,
        received_at_ms=received_at_ms,
        trade_id=aggregate_trade_id,
        price=price,
        quantity=quantity,
        buyer_is_maker=buyer_is_maker,
        source_event_sha256=_sha256([market, received_at_ms, aggregate_trade_id]),
    )


def _book(
    token_id: str,
    *,
    received_at_ms: int,
    identity: str,
    bids: tuple[tuple[str, str], ...],
    asks: tuple[tuple[str, str], ...],
    connected: bool = True,
    gap_free: bool = True,
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
        source_time_ms=received_at_ms - 2,
        received_wall_ms=received_at_ms,
        received_monotonic_ns=received_at_ms * 1_000_000,
        source_payload_sha256=_sha256(identity),
        connected=connected,
        gap_free=gap_free,
    )


def _ingest_price_inputs(
    accumulator: PolymarketRound17FeatureAccumulator,
) -> None:
    for index, (offset_ms, price) in enumerate(
        ((-1_000, "100"), (-500, "101"), (0, "102")),
        start=1,
    ):
        accumulator.ingest_chainlink(
            _chainlink(
                received_at_ms=DECISION_MS + offset_ms,
                price=price,
                identity=f"chainlink-{index}",
            ),
            causal_segment_sha256=CAUSAL_SEGMENT_SHA256,
        )
    for market in ("spot", "perpetual"):
        for trade in (
            _trade(
                market=market,
                received_at_ms=DECISION_MS - 1_000,
                aggregate_trade_id=1,
                price=100.0,
                quantity=1.0,
                buyer_is_maker=False,
            ),
            _trade(
                market=market,
                received_at_ms=DECISION_MS - 500,
                aggregate_trade_id=2,
                price=101.0 if market == "spot" else 100.5,
                quantity=2.0,
                buyer_is_maker=False,
            ),
            _trade(
                market=market,
                received_at_ms=DECISION_MS - 100,
                aggregate_trade_id=3,
                price=102.0 if market == "spot" else 101.0,
                quantity=1.0,
                buyer_is_maker=True,
            ),
        ):
            accumulator.ingest_binance(
                trade,
                causal_segment_sha256=CAUSAL_SEGMENT_SHA256,
            )


def _ingest_book_inputs(
    accumulator: PolymarketRound17FeatureAccumulator,
) -> None:
    up_initial = _book(
        UP_TOKEN_ID,
        received_at_ms=DECISION_MS - 1_000,
        identity="up-initial",
        bids=(("0.49", "10"),),
        asks=(("0.51", "10"),),
    )
    accumulator.ingest_book(
        "up",
        up_initial,
        causal_segment_sha256=CAUSAL_SEGMENT_SHA256,
    )
    accumulator.ingest_book(
        "up",
        up_initial,
        causal_segment_sha256=CAUSAL_SEGMENT_SHA256,
    )
    accumulator.ingest_book(
        "up",
        _book(
            UP_TOKEN_ID,
            received_at_ms=DECISION_MS - 500,
            identity="up-second",
            bids=(("0.50", "12"),),
            asks=(("0.52", "8"),),
        ),
        causal_segment_sha256=CAUSAL_SEGMENT_SHA256,
    )
    for index, received_at_ms in enumerate(
        (DECISION_MS - 1_000, DECISION_MS - 500),
        start=1,
    ):
        accumulator.ingest_book(
            "down",
            _book(
                DOWN_TOKEN_ID,
                received_at_ms=received_at_ms,
                identity=f"down-{index}",
                bids=(("0.48", "10"),),
                asks=(("0.50", "10"),),
            ),
            causal_segment_sha256=CAUSAL_SEGMENT_SHA256,
        )


def test_round17_v2_contract_is_self_hashed_and_binds_raw_trade_parity() -> None:
    path = (
        REPOSITORY_ROOT
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-017-btc-5m-causal-flow-model-v2.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND17_CONTRACT_SHA256 == _sha256(payload)
    feature_contract = payload["feature_contract"]
    assert feature_contract["binance_trade_input"] == {
        "campaign_stream": "btcusdt@trade",
        "event_type": "trade",
        "identity": (
            "venue plus BTCUSDT plus connection ID plus trade ID plus immutable "
            "source-event SHA-256"
        ),
        "direction": (
            "buyer_is_maker maps quote notional negative; otherwise positive"
        ),
        "aggregate_trade_substitution_permitted": False,
        "credentials_or_private_state_permitted": False,
        "live_parity": (
            "The Round 17 live predictor must consume the same public raw-trade "
            "schema. An aggregate-trade feed cannot silently replace it."
        ),
    }
    assert "trade_count" in feature_contract["binance_features_per_market_per_window"]
    assert (
        "aggregate_trade_count"
        not in feature_contract["binance_features_per_market_per_window"]
    )


def test_round17_features_are_deterministic_target_free_and_non_authoritative() -> None:
    first = _accumulator()
    second = _accumulator()
    for accumulator in (first, second):
        _ingest_price_inputs(accumulator)
        _ingest_book_inputs(accumulator)

    row = first.build(_base())
    repeated = second.build(_base())

    assert len(row.values) == len(POLYMARKET_ROUND17_FEATURE_NAMES) == 188
    assert row.feature_names_sha256 == POLYMARKET_ROUND17_FEATURE_NAMES_SHA256
    assert row.input_sha256 == repeated.input_sha256
    assert row.values_sha256 == repeated.values_sha256
    assert row.values == repeated.values
    assert row.trading_authority is False
    assert row.asdict()["trading_authority"] is False
    assert not {
        "close_price",
        "official_up",
        "profit",
        "pnl",
    }.intersection(POLYMARKET_ROUND17_FEATURE_NAMES)


def test_round17_price_and_book_statistics_match_analytical_values() -> None:
    accumulator = _accumulator()
    _ingest_price_inputs(accumulator)
    _ingest_book_inputs(accumulator)

    values = accumulator.build(_base()).value_map()
    first_return = math.log(101.0 / 100.0)
    second_return = math.log(102.0 / 101.0)

    assert values["chainlink_log_return_1000ms"] == pytest.approx(math.log(1.02))
    assert values["chainlink_realized_variance_1000ms"] == pytest.approx(
        first_return**2 + second_return**2
    )
    assert values["chainlink_bipower_variation_1000ms"] == pytest.approx(
        (math.pi / 2.0) * abs(first_return * second_return)
    )
    assert values["chainlink_tick_count_1000ms"] == 3.0
    assert values["binance_spot_signed_quote_imbalance_1000ms"] == pytest.approx(
        (100.0 + 202.0 - 102.0) / (100.0 + 202.0 + 102.0)
    )
    assert values["binance_spot_trade_count_1000ms"] == 3.0
    assert values[
        "polymarket_up_top_of_book_order_flow_imbalance_1000ms"
    ] == pytest.approx(1.0)
    assert values["polymarket_up_level_quantity_flow_pressure_1000ms"] == pytest.approx(
        0.1
    )
    assert values["polymarket_up_book_update_count_1000ms"] == 2.0
    assert values["binance_spot_minus_perpetual_log_return_1000ms"] == pytest.approx(
        math.log(102.0 / 100.0) - math.log(101.0 / 100.0)
    )


def test_round17_rejects_future_cross_segment_gap_and_conflicting_replay() -> None:
    accumulator = _accumulator()
    with pytest.raises(ValueError, match="causal stream segment"):
        accumulator.ingest_chainlink(
            _chainlink(
                received_at_ms=DECISION_MS - 10,
                price="100",
                identity="wrong-segment",
            ),
            causal_segment_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="identity or gap"):
        accumulator.ingest_book(
            "up",
            _book(
                UP_TOKEN_ID,
                received_at_ms=DECISION_MS - 10,
                identity="gap-book",
                bids=(("0.49", "10"),),
                asks=(("0.51", "10"),),
                gap_free=False,
            ),
            causal_segment_sha256=CAUSAL_SEGMENT_SHA256,
        )

    future = _chainlink(
        received_at_ms=DECISION_MS + 1,
        price="100",
        identity="future",
    )
    accumulator.ingest_chainlink(
        future,
        causal_segment_sha256=CAUSAL_SEGMENT_SHA256,
    )
    with pytest.raises(ValueError, match="future receipt evidence"):
        accumulator.build(_base())

    replay = _accumulator()
    original = _book(
        UP_TOKEN_ID,
        received_at_ms=DECISION_MS - 10,
        identity="same-wire-message",
        bids=(("0.49", "10"),),
        asks=(("0.51", "10"),),
    )
    replay.ingest_book(
        "up",
        original,
        causal_segment_sha256=CAUSAL_SEGMENT_SHA256,
    )
    with pytest.raises(ValueError, match="duplicate book observation conflicts"):
        replay.ingest_book(
            "up",
            replace(
                original,
                bids=(BookLevel(price=Decimal("0.49"), quantity=Decimal("11")),),
            ),
            causal_segment_sha256=CAUSAL_SEGMENT_SHA256,
        )


def test_round17_enforces_admission_cadence_and_feature_vector_integrity() -> None:
    without_binance = _accumulator(binance_layer_eligible=False)
    with pytest.raises(ValueError, match="lacks condition admission"):
        without_binance.ingest_binance(
            _trade(
                market="spot",
                received_at_ms=DECISION_MS - 10,
                aggregate_trade_id=1,
                price=100.0,
                quantity=1.0,
                buyer_is_maker=False,
            ),
            causal_segment_sha256=CAUSAL_SEGMENT_SHA256,
        )
    with pytest.raises(ValueError, match="identity or chronology"):
        without_binance.build(_base(decision_time_ms=DECISION_MS + 1))

    row = without_binance.build(_base())
    with pytest.raises(ValueError, match="feature row is invalid"):
        PolymarketRound17FeatureRow(
            condition_id=row.condition_id,
            decision_time_ms=row.decision_time_ms,
            admission_sha256=row.admission_sha256,
            causal_segment_sha256=row.causal_segment_sha256,
            feature_names_sha256=row.feature_names_sha256,
            input_sha256=row.input_sha256,
            values_sha256=row.values_sha256,
            values=(*row.values[:-1], row.values[-1] + 1.0),
        )
