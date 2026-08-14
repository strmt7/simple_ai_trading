from decimal import Decimal
from types import SimpleNamespace

from simple_ai_trading.paper_execution import BookLevel
from simple_ai_trading.polymarket_fees import PolymarketFeeModel
from simple_ai_trading.polymarket_round27_mechanics import (
    _PairedQuote,
    _latency_benchmarks,
    _paired_quotes,
)


def _quote(at_ms: int, cost: str) -> _PairedQuote:
    half = Decimal(cost) / 2
    return _PairedQuote(
        condition_id="condition",
        slug="btc-updown-5m-1",
        segment_id="segment",
        received_monotonic_ns=at_ms * 1_000_000,
        received_wall_ms=1_000 + at_ms,
        interval_end_ms=10_000,
        market_end_ms=20_000,
        taker_delay_ms=250,
        up_best_ask=half,
        down_best_ask=half,
        up_buy_cost=half,
        down_buy_cost=half,
        up_sell_value=Decimal("0.4"),
        down_sell_value=Decimal("0.4"),
    )


def test_complete_set_quote_must_survive_delay_and_sequential_legs() -> None:
    result = _latency_benchmarks(
        (
            _quote(0, "0.98"),
            _quote(250, "1.01"),
            _quote(500, "1.02"),
        )
    )

    assert result == {
        "same_state_episode_count": 1,
        "venue_delay_survivor_count": 0,
        "minimum_sequential_survivor_count": 0,
        "best_same_state_cost": "0.98",
        "best_venue_delay_cost": "1.010",
        "best_minimum_sequential_cost": "1.015",
    }


def test_complete_set_quote_counts_a_surviving_minimum_sequence() -> None:
    result = _latency_benchmarks(
        (
            _quote(0, "0.97"),
            _quote(250, "0.98"),
            _quote(500, "0.99"),
            _quote(750, "1.01"),
        )
    )

    assert result["venue_delay_survivor_count"] == 1
    assert result["minimum_sequential_survivor_count"] == 1
    assert result["best_minimum_sequential_cost"] == "0.985"


def test_pair_evaluation_applies_every_token_update_in_one_message_first() -> None:
    fee = PolymarketFeeModel(False, Decimal("0"), 1, True)
    market = SimpleNamespace(
        condition_id="condition",
        slug="btc-updown-5m-1",
        minimum_order_size=Decimal("5"),
        end_ms=20_000,
        fee_schedule=SimpleNamespace(fee_model=lambda: fee),
    )

    def book(outcome: str, price: str, sequence: int, at_ms: int):
        snapshot = SimpleNamespace(
            asks=(BookLevel(Decimal(price), Decimal("5")),),
            bids=(BookLevel(Decimal(price) - Decimal("0.01"), Decimal("5")),),
        )
        return SimpleNamespace(
            market=market,
            outcome=outcome,
            segment_id="segment",
            connection_id="connection",
            sequence_number=sequence,
            received_monotonic_ns=at_ms * 1_000_000,
            received_wall_ms=1_000 + at_ms,
            snapshot=snapshot,
        )

    replay = SimpleNamespace(
        markets=(market,),
        market_execution_evidence=(
            SimpleNamespace(condition_id="condition", taker_order_delay_ms=250),
        ),
        books=(
            book("Up", "0.60", 1, 0),
            book("Down", "0.50", 1, 0),
            book("Up", "0.40", 2, 50),
            book("Down", "0.60", 2, 50),
        ),
    )

    quotes = _paired_quotes(
        replay,
        intervals={("condition", "segment"): (0, 10_000)},
    )

    assert [quote.complete_set_cost for quote in quotes] == [
        Decimal("1.10"),
        Decimal("1.00"),
    ]
