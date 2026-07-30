from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_replay import (
    PolymarketRecordedBook,
    PolymarketResolutionEvidence,
)
from simple_ai_trading.polymarket_round14_contract import load_round14_contract
from simple_ai_trading.polymarket_round17_dataset import (
    PolymarketRound17ConditionDataset,
)
from simple_ai_trading.polymarket_round17_economic import (
    POLYMARKET_ROUND17_ECONOMIC_PATHS,
)
from simple_ai_trading.polymarket_round17_features import (
    POLYMARKET_ROUND17_FEATURE_NAMES,
    POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
    PolymarketRound17FeatureRow,
)
from simple_ai_trading.polymarket_round17_outcomes import (
    build_round17_decision_probability,
    materialize_round17_condition_economic_outcomes,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-prospective-contract-v1.json"
)
START_MS = 1_800_057_600_000
DECISION_ONE_MS = START_MS + 60_000
DECISION_TWO_MS = START_MS + 62_000
CONDITION_ID = "0x" + "1" * 64
UP_TOKEN_ID = "1" * 40
DOWN_TOKEN_ID = "2" * 40
RUN_ID = "run-round17-economic-fixture"
SEGMENT_ID = "segment-" + "3" * 32


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _market() -> PolymarketFiveMinuteMarket:
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id="12345",
        condition_id=CONDITION_ID,
        slug=f"btc-updown-5m-{START_MS // 1000}",
        question="Bitcoin Up or Down",
        event_start_ms=START_MS,
        end_ms=START_MS + 300_000,
        up_token_id=UP_TOKEN_ID,
        down_token_id=DOWN_TOKEN_ID,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("1"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=False,
            rate=Decimal("0"),
            exponent=2,
            taker_only=True,
            rebate_rate=Decimal("0"),
        ),
        liquidity_quote=Decimal("100000"),
        volume_quote=Decimal("100000"),
        resolution_source="chainlink",
        gamma_payload_sha256=_sha256("market"),
        gamma_payload_json="{}",
    )


def _feature_row(decision_time_ms: int) -> PolymarketRound17FeatureRow:
    values = tuple(0.0 for _ in POLYMARKET_ROUND17_FEATURE_NAMES)
    return PolymarketRound17FeatureRow(
        condition_id=CONDITION_ID,
        decision_time_ms=decision_time_ms,
        admission_sha256=_sha256("admission"),
        causal_segment_sha256=_sha256("segment"),
        feature_names_sha256=POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
        input_sha256=_sha256(["input", decision_time_ms]),
        values_sha256=_sha256(list(values)),
        values=values,
    )


def _dataset() -> PolymarketRound17ConditionDataset:
    rows = (_feature_row(DECISION_ONE_MS), _feature_row(DECISION_TWO_MS))
    provisional = PolymarketRound17ConditionDataset(
        run_id=RUN_ID,
        condition_id=CONDITION_ID,
        event_start_ms=START_MS,
        event_end_ms=START_MS + 300_000,
        admission_sha256=_sha256("admission"),
        causal_segment_sha256=_sha256("segment"),
        feature_names_sha256=POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
        base_row_count=len(rows),
        chainlink_event_count=2,
        spot_trade_count=0,
        perpetual_trade_count=0,
        up_book_count=12,
        down_book_count=12,
        binance_layer_eligible=False,
        rows=rows,
        dataset_sha256="0" * 64,
    )
    return replace(
        provisional,
        dataset_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def _book(
    outcome: str,
    received_at_ms: int,
    *,
    bid: str,
    ask: str,
    sequence: int,
) -> PolymarketRecordedBook:
    market = _market()
    token_id = UP_TOKEN_ID if outcome == "Up" else DOWN_TOKEN_ID
    snapshot = PaperBookSnapshot(
        venue="polymarket",
        market_id=CONDITION_ID,
        asset_id=token_id,
        bids=(BookLevel(Decimal(bid), Decimal("100")),),
        asks=(BookLevel(Decimal(ask), Decimal("100")),),
        source_time_ms=received_at_ms - 1,
        received_wall_ms=received_at_ms,
        received_monotonic_ns=received_at_ms * 1_000_000,
        source_payload_sha256=_sha256([outcome, received_at_ms, bid, ask, sequence]),
    ).validated()
    return PolymarketRecordedBook(
        run_id=RUN_ID,
        event_id=_sha256(["book", sequence]),
        event_type="book",
        connection_id="clob:" + "4" * 32,
        segment_id=SEGMENT_ID,
        sequence_number=sequence,
        sub_index=0,
        market=market,
        outcome=outcome,
        tick_size=Decimal("0.01"),
        snapshot=snapshot,
    )


def _books() -> tuple[PolymarketRecordedBook, ...]:
    values: list[PolymarketRecordedBook] = []
    sequence = 1
    for decision, prices in (
        (DECISION_ONE_MS, {"Up": ("0.39", "0.40"), "Down": ("0.59", "0.60")}),
        (DECISION_TWO_MS, {"Up": ("0.60", "0.61"), "Down": ("0.29", "0.30")}),
    ):
        for outcome, (bid, ask) in prices.items():
            values.append(
                _book(
                    outcome,
                    decision - 100,
                    bid=bid,
                    ask=ask,
                    sequence=sequence,
                )
            )
            sequence += 1
        for latency in (250, 500, 750, 1_000):
            for outcome, (bid, ask) in prices.items():
                values.append(
                    _book(
                        outcome,
                        decision + latency,
                        bid=bid,
                        ask=ask,
                        sequence=sequence,
                    )
                )
                sequence += 1
    return tuple(
        sorted(
            values,
            key=lambda item: (
                item.received_wall_ms,
                item.received_monotonic_ns,
                item.event_id,
            ),
        )
    )


def _resolution() -> PolymarketResolutionEvidence:
    return PolymarketResolutionEvidence(
        run_id=RUN_ID,
        event_id=_sha256("resolution-id"),
        condition_id=CONDITION_ID,
        winning_asset_id=UP_TOKEN_ID,
        winning_outcome="Up",
        resolved_at_ms=START_MS + 300_001,
        received_wall_ms=START_MS + 300_002,
        received_monotonic_ns=(START_MS + 300_002) * 1_000_000,
        event_sha256=_sha256("resolution"),
        source="clob_gamma_crosscheck",
    )


def _predictions(dataset: PolymarketRound17ConditionDataset):
    return (
        build_round17_decision_probability(
            dataset.rows[0],
            probability_up=Decimal("0.90"),
            lower_up=Decimal("0.85"),
            upper_up=Decimal("0.95"),
            model_pretest_sha256="a" * 64,
        ),
        build_round17_decision_probability(
            dataset.rows[1],
            probability_up=Decimal("0.20"),
            lower_up=Decimal("0.15"),
            upper_up=Decimal("0.25"),
            model_pretest_sha256="a" * 64,
        ),
    )


def test_round17_materializes_complete_causal_condition_economic_grid() -> None:
    dataset = _dataset()
    predictions = _predictions(dataset)

    outcomes = materialize_round17_condition_economic_outcomes(
        market=_market(),
        dataset=dataset,
        predictions=predictions,
        books=_books(),
        resolution=_resolution(),
        program=load_round14_contract(CONTRACT_PATH),
        risk_capital_quote=Decimal("1000"),
    )

    assert len(outcomes) == 360
    assert {item.path for item in outcomes} == set(POLYMARKET_ROUND17_ECONOMIC_PATHS)
    assert {item.risk_profile for item in outcomes} == {
        "conservative",
        "regular",
        "aggressive",
    }
    assert {item.scenario for item in outcomes} == {
        "primary",
        "latency_250ms",
        "latency_750ms",
        "latency_1000ms",
        "half_depth",
        "quarter_depth",
        "one_tick_adverse",
        "combined",
    }
    assert all(item.entry_executed for item in outcomes)
    assert all(not item.ownership_violation for item in outcomes)
    assert all(item.realized_net_quote >= 0 for item in outcomes)


def test_round17_economic_materializer_rejects_cross_run_books() -> None:
    dataset = _dataset()
    books = list(_books())
    books[0] = replace(books[0], run_id="foreign-run")

    with pytest.raises(ValueError, match="replay book identity"):
        materialize_round17_condition_economic_outcomes(
            market=_market(),
            dataset=dataset,
            predictions=_predictions(dataset),
            books=books,
            resolution=_resolution(),
            program=load_round14_contract(CONTRACT_PATH),
            risk_capital_quote=Decimal("1000"),
        )


def test_round17_decision_probability_binds_model_feature_and_envelope() -> None:
    row = _dataset().rows[0]
    prediction = build_round17_decision_probability(
        row,
        probability_up=Decimal("0.8"),
        lower_up=Decimal("0.7"),
        upper_up=Decimal("0.9"),
        model_pretest_sha256="b" * 64,
    )

    assert prediction.envelope.evidence_sha256 == prediction.prediction_sha256
    assert prediction.feature_input_sha256 == row.input_sha256
    assert prediction.feature_values_sha256 == row.values_sha256
