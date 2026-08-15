from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading.paper_execution import BookLevel
from simple_ai_trading.polymarket_fees import PolymarketFeeModel
from simple_ai_trading.polymarket_round27_mechanics import (
    _PairedQuote,
    _load_claim,
    _latency_benchmarks,
    _paired_quotes,
    _validate_stage0_lineage,
)
from tools.publish_polymarket_round27_mechanics import _svg


ROOT = Path(__file__).resolve().parents[1]


def _artifact(relative: str, claim: str) -> dict[str, object]:
    return _load_claim(ROOT / relative, claim=claim, label=relative)


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


def test_stage0_mechanics_lineage_binds_capture_result_and_condition_audit() -> None:
    audit = _artifact(
        "docs/model-research/polymarket/latest/round-027-stage0-condition-audit/"
        "condition-replay-audit.json",
        "audit_sha256",
    )
    preregistration = _artifact(
        "docs/model-research/polymarket/"
        "round-027-execution-hypothesis-preregistration-v3.json",
        "preregistration_sha256",
    )
    capture_contract = _artifact(
        "docs/model-research/polymarket/"
        "round-027-stage0-mechanics-capture-v1.json",
        "contract_sha256",
    )
    capture_result = _artifact(
        "docs/model-research/polymarket/"
        "round-027-stage0-mechanics-capture-result-v1-2026-08-15.json",
        "result_sha256",
    )

    lineage = _validate_stage0_lineage(
        audit=audit,
        preregistration=preregistration,
        capture_contract=capture_contract,
        capture_result=capture_result,
    )

    assert lineage["cohort_role"] == "preregistered_stage0_mechanics"
    assert lineage["preregistered_stage_0"] is True
    tampered_result = dict(capture_result)
    tampered_result["capture_report"] = {
        **capture_result["capture_report"],
        "report_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="lineage differs"):
        _validate_stage0_lineage(
            audit=audit,
            preregistration=preregistration,
            capture_contract=capture_contract,
            capture_result=tampered_result,
        )


def test_mechanics_graph_uses_result_coverage_instead_of_hard_coded_cohort() -> None:
    result = {
        "candidate_counts": {
            name: {"state_count": count, "market_count": 1}
            for name, count in (
                ("extreme_settlement_value", 10),
                ("late_strong_favorite", 5),
                ("complete_set_after_fee", 1),
                ("split_sell_after_fee", 0),
            )
        },
        "coverage": {
            "eligible_market_count": 53,
            "paired_quote_state_count": 123_456,
        },
        "complete_set_latency": {
            "same_state_episode_count": 1,
            "venue_delay_survivor_count": 0,
            "minimum_sequential_survivor_count": 0,
            "segment_benchmarks": [],
        },
    }

    svg = _svg(result).decode("ascii")

    assert "53 BTC five-minute markets | 123,456 paired states" in svg
    assert "11 BTC five-minute markets | 54,983 paired states" not in svg
