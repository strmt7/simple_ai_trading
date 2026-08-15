from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib

import numpy as np
import pytest

from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_replay import PolymarketRecordedBook
from simple_ai_trading.polymarket_round27_economics import (
    Round27EconomicBookBatch,
    Round27EconomicConfig,
    evaluate_round27_economic_scenarios,
)
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round27_model import (
    Round27ModelSample,
    Round27Partition,
)


_START_MS = 1_786_784_400_000
_HASH = "a" * 64


def _market(index: int) -> PolymarketFiveMinuteMarket:
    start = _START_MS + index * 300_000
    condition = "0x" + format(index + 1, "064x")
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id=f"market-{index}",
        condition_id=condition,
        slug=f"btc-updown-5m-{start // 1000}",
        question="BTC Up or Down",
        event_start_ms=start,
        end_ms=start + 300_000,
        up_token_id=f"up-{index}",
        down_token_id=f"down-{index}",
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("5"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=1,
            taker_only=True,
            rebate_rate=Decimal("0.20"),
        ),
        liquidity_quote=Decimal("20000"),
        volume_quote=Decimal("50000"),
        resolution_source="chainlink-btc-usd-twap-60s",
        gamma_payload_sha256=_HASH,
        gamma_payload_json="{}",
    )


def _book(
    market: PolymarketFiveMinuteMarket,
    *,
    outcome: str,
    offset_ms: int,
    segment: str = "segment-1",
    ask_quantity: Decimal = Decimal("10"),
    ask: Decimal | None = None,
    tick_size: Decimal | None = None,
) -> PolymarketRecordedBook:
    selected_ask = ask or (Decimal("0.46") if outcome == "Up" else Decimal("0.54"))
    bid = selected_ask - Decimal("0.01")
    token = market.up_token_id if outcome == "Up" else market.down_token_id
    wall_ms = market.event_start_ms + offset_ms
    payload_sha = hashlib.sha256(
        f"{market.condition_id}:{outcome}:{offset_ms}:{segment}:{ask_quantity}".encode(
            "ascii"
        )
    ).hexdigest()
    snapshot = PaperBookSnapshot(
        venue="polymarket",
        market_id=market.condition_id,
        asset_id=token,
        bids=(BookLevel(bid, Decimal("20")),),
        asks=(
            BookLevel(selected_ask, min(Decimal("2"), ask_quantity)),
            *(
                (BookLevel(selected_ask + Decimal("0.01"), ask_quantity - 2),)
                if ask_quantity > 2
                else ()
            ),
        ),
        source_time_ms=wall_ms,
        received_wall_ms=wall_ms,
        received_monotonic_ns=(market.event_start_ms + offset_ms) * 1_000_000,
        source_payload_sha256=payload_sha,
    ).validated()
    return PolymarketRecordedBook(
        run_id="round27-stage1-test",
        event_id=f"event-{market.market_id}-{outcome}-{offset_ms}-{segment}",
        event_type="book",
        connection_id=f"connection-{segment}",
        segment_id=segment,
        sequence_number=offset_ms,
        sub_index=0,
        market=market,
        outcome=outcome,
        tick_size=tick_size or market.tick_size,
        snapshot=snapshot,
    )


def _population(
    count: int,
) -> tuple[
    tuple[PolymarketFiveMinuteMarket, ...],
    Round27Partition,
    np.ndarray,
    tuple[PolymarketRecordedBook, ...],
    dict[str, int],
]:
    markets = tuple(_market(index) for index in range(count))
    books: list[PolymarketRecordedBook] = []
    samples: list[Round27ModelSample] = []
    outcomes: dict[str, int] = {}
    execution_offsets = (30_251, 30_501, 31_001, 31_251, 31_501, 32_001, 32_501)
    for index, market in enumerate(markets):
        decision = market.event_start_ms + 30_000
        books.extend(
            (
                _book(market, outcome="Up", offset_ms=29_999),
                _book(market, outcome="Down", offset_ms=29_999),
            )
        )
        books.extend(
            _book(market, outcome="Up", offset_ms=offset)
            for offset in execution_offsets
        )
        prior = float(
            Decimal("0.455") / (Decimal("0.455") + Decimal("0.535"))
        )
        samples.append(
            Round27ModelSample(
                slot_id="stage1-b",
                role="selection",
                condition_id=market.condition_id,
                event_start_ms=market.event_start_ms,
                decision_time_ms=decision,
                market_prior_probability=prior,
                values=(0.0,) * len(POLYMARKET_ROUND27_FEATURE_NAMES),
                target_up=1,
                condition_weight=1.0,
                feature_row_sha256=hashlib.sha256(
                    f"feature-{index}".encode("ascii")
                ).hexdigest(),
            ).validated()
        )
        outcomes[market.condition_id] = 1
    partition = Round27Partition.from_samples(samples, role="selection")
    return markets, partition, np.full(count, 0.80), tuple(books), outcomes


def _evaluate(
    markets: tuple[PolymarketFiveMinuteMarket, ...],
    partition: Round27Partition,
    probabilities: np.ndarray,
    books: tuple[PolymarketRecordedBook, ...],
    outcomes: dict[str, int],
    *,
    config: Round27EconomicConfig | None = None,
) -> dict[str, object]:
    return evaluate_round27_economic_scenarios(
        partition=partition,
        predictions=probabilities,
        markets=markets,
        books=books,
        outcomes_up=outcomes,
        model_name="test-model",
        model_sha256="b" * 64,
        source_audit_sha256="c" * 64,
        resolution_evidence_sha256="d" * 64,
        config=config,
    )


def test_round27_economics_walks_depth_and_passes_every_fixed_delay() -> None:
    markets, partition, probabilities, books, outcomes = _population(20)
    report = _evaluate(
        markets,
        partition,
        probabilities,
        books,
        outcomes,
        config=Round27EconomicConfig(
            minimum_executed_trades=20,
            minimum_profitable_conditions=20,
            bootstrap_draws=1_000,
        ),
    )

    assert report["candidate_condition_count"] == 20
    assert report["economic_edge_gate_passed"] is True
    assert report["edge_claim"] is False
    assert report["orders_submitted"] is False
    for scenario in report["scenarios"]:
        assert scenario["filled_order_count"] == 20
        assert scenario["scenario_edge_gate_passed"] is True
        bootstrap = scenario["condition_bootstrap"]
        assert bootstrap["method"] == (
            "stationary_bootstrap_block_length_sensitivity_envelope"
        )
        assert set(bootstrap["fixed_expected_block_lengths_conditions"]) == {1, 4}
        assert bootstrap["automatic_expected_block_length_conditions"] in bootstrap[
            "expected_block_lengths_conditions"
        ]
        assert bootstrap["ci95_lower"] == min(
            item["ci95_lower"] for item in bootstrap["block_intervals"]
        )
        trade = scenario["trades"][0]
        assert trade["average_fill_price"] == "0.466"
        assert Decimal(trade["fee_quote"]) > 0
        assert trade["trade_sha256"] != trade["source_payload_sha256"]


def test_round27_economics_blocks_new_entries_in_settlement_hazard_window() -> None:
    markets, partition, probabilities, _books, outcomes = _population(1)
    market = markets[0]
    late_sample = replace(
        partition.samples[0],
        decision_time_ms=market.end_ms - 50_000,
    ).validated()
    late_partition = Round27Partition.from_samples(
        (late_sample,),
        role="selection",
    )
    books = (
        _book(market, outcome="Up", offset_ms=249_999),
        _book(market, outcome="Down", offset_ms=249_999),
    )

    report = _evaluate(
        markets,
        late_partition,
        probabilities,
        books,
        outcomes,
    )

    assert report["config"]["minimum_new_entry_time_to_settlement_ms"] == 60_000
    assert report["candidate_condition_count"] == 0
    for scenario in report["scenarios"]:
        assert scenario["attempted_order_count"] == 0
        assert scenario["reason_counts"] == {
            "no_positive_after_cost_candidate": 1,
            "settlement_manipulation_hazard_window": 1,
        }


def test_round27_economics_fok_refuses_insufficient_displayed_depth() -> None:
    markets, partition, probabilities, books, outcomes = _population(1)
    target = markets[0]
    reduced = tuple(
        _book(
            target,
            outcome="Up",
            offset_ms=book.received_wall_ms - target.event_start_ms,
            ask_quantity=Decimal("4"),
        )
        if book.outcome == "Up" and book.received_wall_ms > target.event_start_ms + 30_000
        else book
        for book in books
    )
    report = _evaluate(markets, partition, probabilities, reduced, outcomes)

    primary = next(item for item in report["scenarios"] if item["delay_ms"] == 500)
    assert primary["filled_order_count"] == 0
    assert primary["trades"][0]["execution_state"] == "CANCELLED"
    assert (
        primary["trades"][0]["execution_reason"]
        == "insufficient_displayed_depth_for_fok"
    )


def test_round27_economics_excludes_ambiguous_same_millisecond_books() -> None:
    markets, partition, probabilities, books, outcomes = _population(1)
    market = markets[0]
    ambiguous = (
        *books,
        _book(market, outcome="Up", offset_ms=30_000, ask=Decimal("0.80")),
        _book(market, outcome="Down", offset_ms=30_000, ask=Decimal("0.20")),
        _book(market, outcome="Up", offset_ms=30_500, ask=Decimal("0.80")),
    )

    report = _evaluate(markets, partition, probabilities, ambiguous, outcomes)
    primary = next(item for item in report["scenarios"] if item["delay_ms"] == 500)
    trade = primary["trades"][0]

    assert trade["execution_state"] == "FILLED"
    assert trade["effective_latency_ms"] == 501
    assert trade["average_fill_price"] == "0.466"


def test_round27_economics_bootstraps_every_evaluated_condition() -> None:
    markets, partition, probabilities, books, outcomes = _population(20)
    probabilities[10:] = np.asarray(
        [sample.market_prior_probability for sample in partition.samples[10:]],
        dtype=np.float64,
    )

    report = _evaluate(
        markets,
        partition,
        probabilities,
        books,
        outcomes,
        config=Round27EconomicConfig(
            minimum_executed_trades=10,
            minimum_profitable_conditions=10,
            bootstrap_draws=1_000,
        ),
    )
    primary = next(item for item in report["scenarios"] if item["delay_ms"] == 500)

    assert primary["filled_order_count"] == 10
    assert primary["evaluated_condition_count"] == 20
    assert primary["condition_bootstrap"]["condition_count"] == 20
    assert primary["condition_bootstrap_population"] == (
        "all_evaluated_conditions_zero_pnl_for_no_fill"
    )
    assert primary["condition_bootstrap"]["mean_net_pnl_quote"] == pytest.approx(
        float(Decimal(primary["net_pnl_quote"]) / 20)
    )


def test_round27_economics_fails_closed_across_connection_segment() -> None:
    markets, partition, probabilities, books, outcomes = _population(1)
    target = markets[0]
    changed = tuple(
        _book(
            target,
            outcome="Up",
            offset_ms=book.received_wall_ms - target.event_start_ms,
            segment="segment-2",
        )
        if book.outcome == "Up" and book.received_wall_ms > target.event_start_ms + 30_000
        else book
        for book in books
    )
    report = _evaluate(markets, partition, probabilities, changed, outcomes)

    primary = next(item for item in report["scenarios"] if item["delay_ms"] == 500)
    assert primary["unknown_order_count"] == 1
    assert primary["trades"][0]["execution_state"] == "UNKNOWN"
    assert primary["trades"][0]["source_payload_sha256"] == next(
        book.snapshot.source_payload_sha256
        for book in books
        if book.outcome == "Up"
        and book.received_wall_ms == target.event_start_ms + 29_999
    )


def test_round27_economics_uses_the_active_decision_tick_lattice() -> None:
    markets, partition, probabilities, books, outcomes = _population(1)
    market = markets[0]
    revised_books = tuple(
        _book(
            market,
            outcome=book.outcome,
            offset_ms=book.received_wall_ms - market.event_start_ms,
            ask=Decimal("0.461") if book.outcome == "Up" else Decimal("0.539"),
            tick_size=Decimal("0.001"),
        )
        for book in books
    )
    prior = float(Decimal("0.456") / (Decimal("0.456") + Decimal("0.534")))
    revised_sample = replace(
        partition.samples[0],
        market_prior_probability=prior,
    )
    revised_partition = Round27Partition.from_samples(
        (revised_sample,),
        role="selection",
    )

    report = _evaluate(
        markets,
        revised_partition,
        probabilities,
        revised_books,
        outcomes,
    )
    primary = next(item for item in report["scenarios"] if item["delay_ms"] == 500)
    trade = primary["trades"][0]

    assert trade["execution_state"] == "FILLED"
    assert Decimal(trade["limit_price"]) % Decimal("0.001") == 0
    assert Decimal(trade["limit_price"]) % Decimal("0.01") != 0
    assert trade["decision_tick_size"] == "0.001"
    assert trade["execution_tick_size"] == "0.001"


def test_round27_economics_rejects_an_execution_tick_lattice_change() -> None:
    markets, partition, probabilities, books, outcomes = _population(1)
    market = markets[0]
    revised_books = []
    for book in books:
        offset = book.received_wall_ms - market.event_start_ms
        if book.outcome == "Up" and offset == 29_999:
            revised_books.append(
                _book(
                    market,
                    outcome="Up",
                    offset_ms=offset,
                    ask=Decimal("0.461"),
                    tick_size=Decimal("0.001"),
                )
            )
        elif book.outcome == "Down" and offset == 29_999:
            revised_books.append(
                _book(
                    market,
                    outcome="Down",
                    offset_ms=offset,
                    ask=Decimal("0.539"),
                    tick_size=Decimal("0.001"),
                )
            )
        else:
            revised_books.append(book)
    prior = float(Decimal("0.456") / (Decimal("0.456") + Decimal("0.534")))
    revised_partition = Round27Partition.from_samples(
        (
            replace(
                partition.samples[0],
                market_prior_probability=prior,
            ),
        ),
        role="selection",
    )

    report = _evaluate(
        markets,
        revised_partition,
        probabilities,
        tuple(revised_books),
        outcomes,
    )
    primary = next(item for item in report["scenarios"] if item["delay_ms"] == 500)
    trade = primary["trades"][0]

    assert trade["execution_state"] == "REJECTED"
    assert trade["execution_reason"] == (
        "limit_price_not_aligned_to_execution_tick_size"
    )
    assert trade["decision_tick_size"] == "0.001"
    assert trade["execution_tick_size"] == "0.01"


def test_round27_economics_rejects_a_book_off_its_active_tick_lattice() -> None:
    markets, partition, probabilities, books, outcomes = _population(1)
    market = markets[0]
    malformed = tuple(
        _book(
            market,
            outcome=book.outcome,
            offset_ms=book.received_wall_ms - market.event_start_ms,
            ask=Decimal("0.461") if book.outcome == "Up" else Decimal("0.539"),
            tick_size=Decimal("0.01"),
        )
        if book.received_wall_ms == market.event_start_ms + 29_999
        else book
        for book in books
    )

    with pytest.raises(ValueError, match="active book tick lattice"):
        _evaluate(markets, partition, probabilities, malformed, outcomes)


def test_round27_economics_rejects_feature_book_prior_drift() -> None:
    markets, partition, probabilities, books, outcomes = _population(1)
    bad_sample = replace(
        partition.samples[0],
        market_prior_probability=partition.samples[0].market_prior_probability + 0.01,
    )
    drifted = Round27Partition.from_samples((bad_sample,), role="selection")

    with pytest.raises(ValueError, match="feature prior and decision books"):
        _evaluate(markets, drifted, probabilities, books, outcomes)


def test_round27_economics_requires_exact_outcome_population() -> None:
    markets, partition, probabilities, books, outcomes = _population(1)
    outcomes["0x" + "f" * 64] = 1

    with pytest.raises(ValueError, match="outcome population"):
        _evaluate(markets, partition, probabilities, books, outcomes)


def test_round27_economics_condition_batches_match_single_resident_replay() -> None:
    markets, partition, probabilities, books, outcomes = _population(20)
    config = Round27EconomicConfig(
        minimum_executed_trades=20,
        minimum_profitable_conditions=20,
        bootstrap_draws=1_000,
    )
    direct = _evaluate(
        markets,
        partition,
        probabilities,
        books,
        outcomes,
        config=config,
    )
    condition_ids = [market.condition_id for market in markets]
    batches = tuple(
        Round27EconomicBookBatch(
            condition_ids=tuple(condition_ids[start : start + 3]),
            books=tuple(
                book
                for book in books
                if book.market.condition_id in condition_ids[start : start + 3]
            ),
        )
        for start in range(0, len(condition_ids), 3)
    )
    batched = evaluate_round27_economic_scenarios(
        partition=partition,
        predictions=probabilities,
        markets=markets,
        outcomes_up=outcomes,
        model_name="test-model",
        model_sha256="b" * 64,
        source_audit_sha256="c" * 64,
        resolution_evidence_sha256="d" * 64,
        config=config,
        book_batches=(batch for batch in batches),
    )

    assert batched == direct


def test_round27_economics_rejects_incomplete_book_batch_coverage() -> None:
    markets, partition, probabilities, books, outcomes = _population(2)
    first_condition = markets[0].condition_id

    with pytest.raises(ValueError, match="do not cover the role"):
        evaluate_round27_economic_scenarios(
            partition=partition,
            predictions=probabilities,
            markets=markets,
            outcomes_up=outcomes,
            model_name="test-model",
            model_sha256="b" * 64,
            source_audit_sha256="c" * 64,
            resolution_evidence_sha256="d" * 64,
            book_batches=(
                Round27EconomicBookBatch(
                    condition_ids=(first_condition,),
                    books=tuple(
                        book
                        for book in books
                        if book.market.condition_id == first_condition
                    ),
                ),
            ),
        )


def test_round27_economics_requires_batching_above_residency_ceiling() -> None:
    markets, partition, probabilities, books, outcomes = _population(33)

    with pytest.raises(ValueError, match="book batch scope differs"):
        _evaluate(markets, partition, probabilities, books, outcomes)
